import torch
import torch.nn as nn
import torch.nn.functional as F

class GKWS_CNN(nn.Module):
    """
    A simple Convolutional Neural Network for Gated Keyword Spotting (GKWS).
    It processes 2D Mel-spectrograms.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        :param input_dim: The number of Mel-spectrogram features (n_mels).
        :param output_dim: The number of output classes (keywords).
        """
        super(GKWS_CNN, self).__init__()
        
        # We assume input tensors of shape [batch_size, channels, freq, time]
        # Here, channels=1 (monoaural), freq=input_dim
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # An approximation of the output size after convolutions and pooling.
        # This will depend on the time dimension of the spectrogram, which can vary.
        # A typical time dimension is ~101 for 1-second audio.
        # After convs and a single pooling layer: `input_dim / 2 * time_dim / 2`
        # We will use a flexible approach with adaptive pooling.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: A Mel-spectrogram tensor of shape [batch, 1, freq, time]
        :return: A tensor of class probabilities.
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten for the fully connected layers
        # Use adaptive pooling for a consistent size regardless of input time dimension
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except the batch dimension

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x