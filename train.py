import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import cnn
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib
import cv2

# Create the CarRacing environment
env = gym.make('CarRacing-v2', continuous=False)

device = torch.device("cuda")

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
        if state.shape == (1, 96, 96):
            return state  # Already preprocessed
        state = state.numpy()
    
    if state.shape == (96, 96, 3):
        # Convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Normalize pixel values
        state = state / 255.0
        # Add channel dimension
        state = np.expand_dims(state, axis=0)
    elif state.shape != (1, 96, 96):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    
    # Convert to torch tensor
    return torch.FloatTensor(state).to(device)

# Hyperparameters
learning_rate = 1e-4
gamma = 0.99
epsilon_start = 0.2
epsilon_end = 0.01
epsilon_decay = 0.997
epochs = 20000
batch_size = 64
buffer_size = 10000
target_update = 10

# Initialize the Q-network
output_size = env.action_space.n
q_network = cnn().to(device)
target_network = cnn().to(device)
# target_network.load_state_dict(torch.load('runs/model3.pth', map_location=device))  
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
    plt.ylabel('Reward')
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

best = -np.inf
# Training loop
for epoch in range(epochs):
    state = env.reset()[0]
    state = preprocess_state(state).to(device)
    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            if np.random.random() < 0.5:
                action = 3
            else:
                action = env.action_space.sample()  # Random action
        else:
            with torch.no_grad():
                q_values = q_network(state.unsqueeze(0)).squeeze()
                action = q_values.argmax().item()        
                        
        next_state, reward, done, truncated, _ = env.step(action)
        
        if action == 0:
            reward -= 0.05  # Penalty for doing nothing
        elif action == 4:
            reward -= 0.15  # Penalty for braking
        elif action == 3:
            reward += 0.1  # Reward for accelerating
        
        total_reward += reward
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
            
            current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
            
            loss = nn.MSELoss()(current_q_values, target_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = preprocess_state(next_state)
        done = done or truncated
        
        if done:
            episode_durations.append(total_reward)
            plot_durations()
            print(f"Epoch {epoch + 1}, Total Reward: {total_reward}, Epsilon: {epsilon}, Best: {best}")
            
            if best < total_reward:
                best = total_reward
                torch.save(q_network.state_dict(), 'model_best.pth')
            
            torch.save(q_network.state_dict(), 'model_last.pth')
            break
        
    if epoch % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())
        
    plot_durations()
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

torch.save(q_network.state_dict(), 'model_last.pth')
env.close()

# Plot final results
plt.figure(2)
plt.title('Training Results')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(episode_durations)
plt.savefig('training_result.png')
plt.show()