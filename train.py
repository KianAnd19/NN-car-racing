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
import matplotlib.pyplot as plt

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
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
q_network = QNetwork.DQN(input_size, output_size)
target_network = QNetwork.DQN(input_size, output_size)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

replayBuffer = replayBuffer(buffer_size)

best = 0
total_rewards = []
epsilon = epsilon_start

# set up matplotlib
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
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



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
            
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones).unsqueeze(1)
            
            #update q_network
            q_values = q_network(states).gather(1, actions)
            next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            loss = nn.SmoothL1Loss()(q_values, target_q_values.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        state = next_state
        done = done or truncated
        
        if done:
            episode_durations.append(total_reward)
            plot_durations()
            break
        
    if epoch % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

    if (total_reward >= best):
        best = total_reward
        torch.save(q_network, 'model2_best.pth')

    total_rewards.append(total_reward)
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # print(f"epoch {epoch + 1}, Total Reward: {total_reward}")

torch.save(q_network, 'model2_last.pth')
env.close()

#plot rewards
plt.savefig('training_result.png')
plt.show()