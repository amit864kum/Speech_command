import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleAudioClassifier, self).__init__()
        # Input shape for Conv1d: [batch_size, channels, length]
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
