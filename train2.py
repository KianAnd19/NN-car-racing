import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network as QNetwork
from collections import deque
import random

# Create the CartPole environment
env = gym.make('CartPole-v1')

class replayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
epochs = 1000
batch_size = 64
buffer_size = 10000

# Initialize the Q-network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
q_network = QNetwork.DQN(input_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

replayBuffer = replayBuffer(buffer_size)

best = 0

# Training loop
for epoch in range(epochs):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(state_tensor).argmax().item()

        # Take action and observe next state and reward
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        replayBuffer.push(state, action, reward, next_state, done)
        
        if len(replayBuffer) >= batch_size:
            batch = replayBuffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)
            
            #update q_network
            q_values = q_network(states).gather(1, actions)
            next_q_values = q_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            loss = nn.MSELoss()(q_values, target_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        done = done or truncated

    if (total_reward >= best):
        best = total_reward
        torch.save(q_network, 'model2_best.pth')

    print(f"epoch {epoch + 1}, Total Reward: {total_reward}")

torch.save(q_network, 'model2_last.pth')
env.close()