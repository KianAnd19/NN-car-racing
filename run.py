import gymnasium as gym
import torch
import numpy as np
from network import DQN  # Make sure this import matches your network file name

# Create the environment
env = gym.make('CarRacing-v2', continuous=False, render_mode="human")

# Initialize the model
input_channels = env.observation_space.shape[2]  # Should be 3 for RGB
output_size = env.action_space.n
model = DQN(input_channels=input_channels, output_size=output_size)

# Load the saved state dictionary
model.load_state_dict(torch.load('model_best.pth'))
model.eval()

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

def select_action_run(state):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return model(state).max(1)[1].view(1, 1)

# Run the environment
state = env.reset()[0]
state = preprocess_state(state)

for _ in range(1000):  # Run for 1000 steps
    action = select_action_run(state)
    next_state, reward, done, truncated, _ = env.step(action.item())
    state = preprocess_state(next_state)
    if done or truncated:
        break

env.close()