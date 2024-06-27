import gymnasium as gym
import torch
import numpy as np
import cv2
from network import cnn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create the environment
env = gym.make('CarRacing-v2', continuous=False)

# Initialize the model
input_channels = 1  # Grayscale input
output_size = env.action_space.n
model = cnn().to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load('model_last.pth', map_location=device))

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

results = []
for trial in range(100):
    print(f"Trial {trial + 1}")
    try:
        # Run the environment
        state, _ = env.reset()
        state = preprocess_state(state)
        total_reward = 0

        for step in range(1000):  # Run for 1000 steps
            action = select_action_run(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = preprocess_state(next_state)
            if terminated or truncated:
                break
            
        results.append(total_reward)
        print(f"Run completed. Total Reward: {total_reward}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        env.close()
        
print(f"Average reward: {np.mean(results)}")
print(f"Standard deviation: {np.std(results)}")
print(f"Minimum reward: {np.min(results)}")
print(f"Maximum reward: {np.max(results)}")

#save plot of results
plt.title('Training Results')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(results)
plt.savefig('result.png')
plt.show()