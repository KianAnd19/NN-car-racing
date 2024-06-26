import gymnasium as gym
import torch
import numpy as np
import cv2
import sys
from network import cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('CarRacing-v2', render_mode="human", continuous=False)

# Initialize the model
input_channels = 1  # Grayscale input
output_size = env.action_space.n
model = cnn().to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load('model_best.pth', map_location=device))    
if len(sys.argv) > 1:
    if sys.argv[1] == "1":
        model.load_state_dict(torch.load('model_last.pth', map_location=device))
    elif sys.argv[1] == "2":
        model.load_state_dict(torch.load('model1.pth', map_location=device))
model.eval()

def preprocess_state(state):
    if isinstance(state, np.ndarray) and state.shape == (96, 96, 3):
        # Convert to grayscale
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Normalize pixel values
        state = state / 255.0
        # Add channel dimension
        state = np.expand_dims(state, axis=0)
    elif not isinstance(state, torch.Tensor) or state.shape != (1, 96, 96):
        raise ValueError(f"Unexpected state shape: {state.shape}")
    
    # Convert to torch tensor if it's not already
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state)
    
    return state.to(device)

def select_action_run(state):
    with torch.no_grad():
        q_values = model(state.unsqueeze(0)).squeeze()
        action = q_values.argmax().item()
        return action

try:
    # Run the environment
    state, _ = env.reset()
    state = preprocess_state(state)
    total_reward = 0

    for step in range(1000):  # Run for 1000 steps
        action = select_action_run(state)
        print(f"Step {step}, Action: {action}")
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = preprocess_state(next_state)
        print(f"Reward: {reward}, Total Reward: {total_reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            print("Episode finished")
            break

    print(f"Run completed. Total Reward: {total_reward}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    env.close()