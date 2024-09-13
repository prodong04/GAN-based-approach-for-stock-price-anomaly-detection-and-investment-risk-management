import torch
import torch.nn as nn
# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, stride=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=2, stride=1)

        self.fc = nn.Linear(160, 25)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(1472, 25)    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x