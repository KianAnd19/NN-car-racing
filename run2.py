import gymnasium as gym
import torch
import torch.nn.functional as F
# import network

device = torch.device("cuda")
loaded_policy_net = torch.load('model2_best.pth', map_location=device)

def select_action_run(state):
    with torch.no_grad():
        return loaded_policy_net(state).max(1).indices.view(1, 1)


# Create a new environment for visualization
env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False

total_reward = 0
steps = 0

for i in range(10000):
    action = select_action_run(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
    
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    state = next_state
    
    total_reward += reward.item()
    steps += 1

env.close()

print(f"Model performance: Total steps: {steps}, Total reward: {total_reward}")