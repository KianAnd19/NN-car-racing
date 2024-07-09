import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self, input_channels=1, action_dim=5):
        super(cnn, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=4, stride=4)
        # self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 24, kernel_size=4, stride=1)
        # self.bn2 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Calculate the size of the flattened features
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 96, 96)
            x = self.pool(F.relu(self.conv2(F.relu(self.conv1(sample_input)))))
            self.fc_input_dim = x.numel() // x.size(0)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # x[:, 0] = torch.tanh(x[:, 0])  # Scale first value to [-1, 1] for steering
        # x[:, 1:] = torch.sigmoid(x[:, 1:])  # Scale second and third values to [0, 1] for acceleration and brake
        return x
