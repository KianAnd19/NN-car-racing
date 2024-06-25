import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network as QNetwork
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib

# Create the CarRacing environment
env = gym.make('CarRacing-v2')

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = preprocess_state(state)
        next_state = preprocess_state(next_state)
        self.buffer.append((state, action, reward, next_state, done))
                
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
def preprocess_state(state):
    if isinstance(state, torch.Tensor):
        if state.shape == (3, 96, 96):
            return state  # Already preprocessed
        state = state.numpy()
    
    if state.shape == (96, 96, 3):
        # Normalize pixel values
        state = state / 255.0
        # Transpose from (height, width, channels) to (channels, height, width)
        state = np.transpose(state, (2, 0, 1))
    elif state.shape != (3, 96, 96):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    
    # Convert to torch tensor
    return torch.FloatTensor(state)

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
epochs = 1000
batch_size = 64
buffer_size = 10000
target_update = 10

# Initialize the Q-network
output_size = env.action_space.shape[0]
input_channels = env.observation_space.shape[2]  # Should be 3 for RGB
q_network = QNetwork.DQN(input_channels=input_channels, action_dim=output_size)
target_network = QNetwork.DQN(input_channels=input_channels, action_dim=output_size)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

replay_buffer = ReplayBuffer(buffer_size)

best_reward = float('-inf')
total_rewards = []
epsilon = epsilon_start

# Set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# Training loop
for epoch in range(epochs):
    state = env.reset()[0]
    state = preprocess_state(state)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(state.unsqueeze(0)).squeeze().numpy()
                # Clip the action to be within the valid range
                action = np.clip(action, -1, 1)
        
        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states)
            actions = torch.FloatTensor(np.array(actions))
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.stack(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            current_q_values = q_network(states)
            next_q_values = target_network(next_states)
            
            # Calculate the target Q-values
            target_q_values = current_q_values.clone()
            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i] = rewards[i]
                else:
                    target_q_values[i] = rewards[i] + gamma * next_q_values[i].max()
            
            loss = nn.MSELoss()(current_q_values, target_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = preprocess_state(next_state)
        done = done or truncated
        
        if done:
            print(f"Epoch {epoch + 1}, Total Reward: {total_reward}")
            break
    
    if epoch % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

torch.save(q_network.state_dict(), 'model_continuous.pth')
env.close()

# Plot final results
plt.figure(2)
plt.title('Training Results')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(total_rewards)
plt.savefig('training_result.png')
plt.show()