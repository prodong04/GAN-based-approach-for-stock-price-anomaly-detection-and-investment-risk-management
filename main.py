import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from gan import Generator, Discriminator

from tqdm import tqdm

data = pd.read_csv('bitdata.csv')
data = data['Close']


train_data = data[:int(len(data)*0.7)]
val_data = data[int(len(data)*0.7):int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

train_data = torch.tensor(train_data.values).float().view(-1, 1, 1)
val_data = torch.tensor(val_data.values).float().view(-1, 1, 1)
test_data = torch.tensor(test_data.values).float().view(-1, 1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.MSELoss()
lr = 0.001
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

num_epochs = 20
batch_size = 64
seq_len = 25

for epoch in tqdm(range(num_epochs)):
    for i in range(0, len(train_data) - seq_len, batch_size):
        x = train_data[i:i+seq_len].to(device)
        z = torch.randn(seq_len, 1, 100).to(device)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        fake_data = generator(z)
        fake_data = fake_data.unsqueeze(1)
        d_fake = discriminator(fake_data)
        d_real = discriminator(x)
        
        d_loss = criterion(d_fake, torch.zeros_like(d_fake)) + criterion(d_real, torch.ones_like(d_real))
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        fake_data = generator(z)
        fake_data = fake_data.unsqueeze(1)
        d_fake = discriminator(fake_data)
        
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        g_loss.backward()
        g_optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_data) - seq_len}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

