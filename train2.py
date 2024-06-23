import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import simple_network as QNetwork

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define a simple neural network


# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
episodes = 2000

# Initialize the Q-network
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
q_network = QNetwork.QNetwork(input_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

best = 0

# Training loop
for episode in range(episodes):
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
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Compute TD target and loss
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        target = reward + gamma * q_network(next_state_tensor).max(1)[0] * (1 - done)
        current_q = q_network(state_tensor)[:, action]

        loss = nn.MSELoss()(current_q, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state

    if (total_reward > best):
        best = total_reward
        torch.save(q_network, 'model2.pth')

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

print(f"Best: {best}")
# torch.save(q_network, 'model2.pth')
env.close()