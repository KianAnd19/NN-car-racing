import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_channels=3, action_dim=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        self.fc_input_dim = self.calculate_conv_output_dim()
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def calculate_conv_output_dim(self):
        input_dim = torch.zeros(1, 3, 96, 96)
        output_dim = self.conv1(input_dim)
        output_dim = self.conv2(output_dim)
        output_dim = self.conv3(output_dim)
        return int(np.prod(output_dim.shape))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))  # Output in range [-1, 1]