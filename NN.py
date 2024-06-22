import torch
import torch.nn as nn
import gym
import time

rewards = []

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.fc(x)
    

    
def train():
    pass


env = gym.make('CartPole-v1', render_mode='human')
observation, info = env.reset(seed=0)

for range in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    # if terminated or truncated:
    #     observation, info = env.reset()
    
env.close()