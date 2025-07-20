import torch
from torch import nn
from collections import namedtuple, deque

class DeeQModel(torch.nn.Module):
    """A simple deep Q-learning model"""
    
    def __init__(self, grid_size, n_state, ac=5):
        super(DeeQModel, self).__init__()
        self.seq = nn.Sequential(
            # First conv block
            nn.Conv2d(n_state, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second conv block  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(128 * (grid_size // 2) * (grid_size // 2), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, ac)
        )
        
    def forward(self, x):
        return self.seq(x)
