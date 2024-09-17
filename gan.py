import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, seq_len, noise_dim=100):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
                
        self.fc1 = nn.Linear(in_features=1344, out_features=seq_len)



    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = x.view(x.size(0), 1, -1)
        return x

class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Convolution 레이어를 거친 후의 시퀀스 길이 계산
        
        self.fc1 = nn.Linear(in_features=1472 , out_features=1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

        
'''


# Generator 정의
class Generator(nn.Module):
    def __init__(self, seq_len, noise_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.fc2 = nn.Linear(128, seq_len)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()  # 출력 값을 -1과 1 사이로 조정

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = x.view(x.size(0), 1, -1)
        return x

# Discriminator 정의
class Discriminator(nn.Module):
    def __init__(self, seq_len):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(seq_len, 128)
        self.fc2 = nn.Linear(128, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 시퀀스를 평탄화
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
        '''