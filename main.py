import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd

from gan import Generator, Discriminator

from tqdm import tqdm

# 데이터 로드 및 전처리
data = pd.read_csv('bitdata.csv')
data = data['Close'].values  # numpy 배열로 변환

# 데이터셋 분할
train_data = data[:int(len(data)*0.7)]
val_data = data[int(len(data)*0.7):int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# 시퀀스 데이터셋 생성
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len]  # (seq_len, 1)

seq_len = 25
batch_size = 64

train_dataset = SequenceDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.MSELoss()  
generate_lr = 2e-4
discriminator_lr = 2e-4

g_optimizer = torch.optim.Adam(generator.parameters(), lr=generate_lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr)

num_epochs = 20

for epoch in range(num_epochs):
    for real_data in train_loader:
        real_data = real_data.to(device)  # (batch_size, seq_len, 1)
        
        batch_size = real_data.size(0)
        breakpoint()
        
        # Discriminator 학습
        d_optimizer.zero_grad()
        
        # 진짜 데이터에 대한 판별자 출력
        real_labels = torch.ones(batch_size, 1).to(device)
        real_outputs = discriminator(real_data)
        d_real_loss = criterion(real_outputs, real_labels)
        
        # 가짜 데이터 생성 및 판별자 출력
        noise = torch.randn(batch_size, seq_len, 1).to(device)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_outputs = discriminator(fake_data.detach())
        d_fake_loss = criterion(fake_outputs, fake_labels)
        
        # Discriminator 총 손실 및 역전파
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()
        
        # Generator 학습
        g_optimizer.zero_grad()
        
        # 생성된 가짜 데이터에 대한 판별자 출력
        fake_outputs = discriminator(fake_data)
        g_loss = criterion(fake_outputs, real_labels)  # Generator는 판별자를 속이려고 함
        
        g_loss.backward()
        g_optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')
