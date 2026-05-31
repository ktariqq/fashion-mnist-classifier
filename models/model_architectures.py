"""
Model architectures for FashionMNIST classification.
Contains MLP, Basic CNN, Advanced CNN, and ResNet implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with 3 hidden layers.
    Simple baseline model using only fully connected layers.
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super(SimpleMLP, self).__init__()
        
        # Layer 1: Input to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])  # Normalizes activations
        self.dropout1 = nn.Dropout(0.3)  # 30% dropout to prevent overfitting
        
        # Layer 2: First to second hidden layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.dropout2 = nn.Dropout(0.3)
        
        # Layer 3: Second to third hidden layer
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2])
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_sizes[2], num_classes)
        
    def forward(self, x):
        # Flatten image from 28x28 to 784
        x = x.view(x.size(0), -1)
        
        # Pass through layers with ReLU activation
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x


class BasicCNN(nn.Module):
    """
    Basic Convolutional Neural Network.
    Uses 2 convolutional layers followed by 2 fully connected layers.
    """
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        
        # Convolutional Layer 1: 1 input channel (grayscale), 32 output channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduces size from 28x28 to 14x14
        
        # Convolutional Layer 2: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduces size from 14x14 to 7x7
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AdvancedCNN(nn.Module):
    """
    Advanced CNN with 4 convolutional layers and more sophisticated architecture.
    Includes batch normalization and dropout for better generalization.
    """
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)  # 2D dropout for conv layers
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet architecture.
    Adds skip connections to help gradient flow in deep networks.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection: if dimensions change, use 1x1 conv to match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class SimplifiedResNet(nn.Module):
    """
    Simplified ResNet architecture for FashionMNIST.
    Uses residual blocks to enable deeper network training.
    """
    def __init__(self, num_classes=10):
        super(SimplifiedResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        
        # Global average pooling and final FC layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper function to create a sequence of residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
