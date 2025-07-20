import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple, deque

class DeeQModel(torch.nn.Module):
    """Optimized deep Q-learning model with modern architecture"""
    
    def __init__(self, grid_size, n_state, ac=5):
        super(DeeQModel, self).__init__()
        
        # More efficient convolutional layers with residual connections
        self.conv1 = nn.Conv2d(n_state, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Stride 2 for efficiency
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # Stride 2 for efficiency
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global average pooling instead of adaptive pooling (MPS compatible)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # More efficient fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, ac)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Efficient forward pass with ReLU inplace operations
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        
        return x


class DuelingDQN(nn.Module):
    """Dueling DQN architecture for improved learning"""
    
    def __init__(self, grid_size, n_state, ac=5):
        super(DuelingDQN, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(n_state, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_size = 256
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, ac)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
