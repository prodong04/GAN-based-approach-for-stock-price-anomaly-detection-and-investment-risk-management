import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from gan import Generator, Discriminator

# 데이터 로드 및 전처리
data = pd.read_csv('bitdata.csv')
data = data['Close'].values  # numpy 배열로 변환

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

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
        sequence = self.data[idx:idx+self.seq_len]
        return sequence.unsqueeze(0)  # (1, seq_len)

seq_len = 25
batch_size = 64
noise_dim = 100
lamda = 0.9  # Discriminator loss에서 lambda 파라미터

train_dataset = SequenceDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
generator = Generator(seq_len, noise_dim).to(device)
discriminator = Discriminator(seq_len).to(device)

try:
    generator.load_state_dict(torch.load('best_g.pth'))
    discriminator.load_state_dict(torch.load('best_d.pth'))
    print('Model loaded')
except FileNotFoundError:
    print('Starting fresh training.')

# 손실 함수 및 최적화 기법 설정
criterion = nn.BCEWithLogitsLoss()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1.5e-4)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

num_epochs = 1

best_d_loss = float('inf')
best_g_loss = float('inf')

for epoch in tqdm(range(num_epochs)):
    generator.train()
    discriminator.train()

    for real_data in train_loader:
        real_data = real_data.to(device)  # (batch_size, 1, seq_len)
        batch_size = real_data.size(0)

        # 진짜와 가짜 레이블 생성
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminator 학습
        d_optimizer.zero_grad()

        # 진짜 데이터에 대한 손실 계산
        real_outputs = discriminator(real_data)
        d_loss_real = criterion(real_outputs, real_labels)

        # 가짜 데이터 생성 및 손실 계산
        noise = torch.randn(batch_size, noise_dim).to(device)
        noise = noise.unsqueeze(1)  # (batch_size, 1, noise_dim)
        fake_data = generator(noise)
        fake_outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)

        # Discriminator의 총 손실 및 역전파
        d_loss = (1 - lamda) * d_loss_real + lamda * d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # Generator 학습
        g_optimizer.zero_grad()
        noise = torch.randn(batch_size, noise_dim).to(device)
        noise = noise.unsqueeze(1)
        fake_data = generator(noise)
        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)  # Generator는 판별자를 속이려고 함

        g_loss.backward()
        g_optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 모델 저장
    if d_loss < best_d_loss and g_loss < best_g_loss:
        best_d_loss = d_loss
        best_g_loss = g_loss
        torch.save(discriminator.state_dict(), 'best_d.pth')
        torch.save(generator.state_dict(), 'best_g.pth')
        print('Best model saved!')

# 학습 후 anomaly score 계산을 위한 함수 정의
def compute_anomaly_score(generator, discriminator, real_data, noise_dim, lamda=0.9, n_iter=3):
    generator.eval()
    discriminator.eval()
    
    # 초기 latent vector z 설정 (노이즈에서 샘플링)
    z = torch.randn(batch_size, noise_dim).to(device)
    z = z.unsqueeze(1) 
    z_optimizer = torch.optim.Adam([z], lr=1e-3)

    real_data = real_data.unsqueeze(0).to(device)
    real_data = real_data.unsqueeze(0).to(device)

    
    #breakpoint()

    for _ in range(n_iter):
        z_optimizer.zero_grad()

        # Generator를 통해 가짜 데이터 생성
        generated_data = generator(z)
        
        # Residual Loss 계산 (L1 노름 사용)
        residual_loss = torch.mean(torch.abs(real_data - generated_data))

        # Discriminator Loss 계산
        real_disc_out = discriminator(real_data)
        fake_disc_out = discriminator(generated_data)
        discriminator_loss = torch.mean(torch.abs(real_disc_out - fake_disc_out))

        # Total Anomaly Score 계산
        anomaly_score = (1 - lamda) * residual_loss + lamda * discriminator_loss

        anomaly_score.backward()
        z_optimizer.step()

    return anomaly_score.item()

# 테스트 데이터에서 anomaly score 계산
test_dataset = SequenceDataset(test_data, seq_len)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
anomaly_lst = []
for test_sample in tqdm(test_loader):
    anomaly_score = compute_anomaly_score(generator, discriminator, test_sample.squeeze(), noise_dim, lamda=0.9)
    print(f'Anomaly Score: {anomaly_score}')
    anomaly_lst.append(anomaly_score)
    
anomaly_frame = pd.DataFrame(anomaly_lst)
anomaly_frame.to_csv('anomaly_score.csv')


