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
from torch.optim.lr_scheduler import StepLR

# Create the CarRacing environment
env = gym.make('CarRacing-v2')

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
learning_rate = 1e-3
gamma = 0.99
epsilon_start = 0.5
epsilon_end = 0.01
epsilon_decay = 0.997
epochs = 1000
batch_size = 64
buffer_size = 10000
target_update = 10

# Initialize the Q-network
q_network = cnn().to(device)
target_network = cnn().to(device)
# target_network.load_state_dict(torch.load('runs/model3.pth', map_location=device))  
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

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
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state.unsqueeze(0)).squeeze()
                action = q_values.cpu().numpy()
        
        next_state, reward, done, truncated, _ = env.step(action)
        
        reward += action[1] * 0.1  # Acceleration reward
        reward -= action[2] * 0.05  # Brake penalty
        
        total_reward += reward
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).to(device)
            actions = torch.FloatTensor(np.array(actions)).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.stack(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
            
            current_q_values = q_network(states)
            
            # Convert actions to indices or use a different approach for continuous actions
            # Here we assume actions are already continuous and not indices
            
            next_q_values = target_network(next_states)
            target_q_values = rewards + (1 - dones) * gamma * next_q_values
            
            # Compute the loss between the current Q-values and target Q-values
            loss = nn.MSELoss()(current_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
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
