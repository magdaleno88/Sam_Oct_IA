"""Simple CNN model for diabetic retinopathy detection."""

from typing import Tuple

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network for image classification.
    
    Architecture:
    - 3 convolutional blocks with max pooling
    - 2 fully connected layers with dropout
    - Output layer with num_classes neurons
    
    This is a demonstration model for the registry system.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 512, 512),  # (channels, height, width)
        num_classes: int = 5,
    ) -> None:
        """
        Initialize the SimpleCNN model.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 512x512 -> 256x256
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x256 -> 128x128
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x128 -> 64x64
        )
        
        # Calculate flattened size: 128 channels * 64 * 64 = 524,288
        self.flattened_size = 128 * 64 * 64
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Output layer
        self.output = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, flattened_size)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        
        # Output layer
        x = self.output(x)
        
        return x
