import sys
import torch
import numpy as np
from Game2048 import Game2048
from env import Game2048Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
loaded_policy_net = torch.load('model2_best.pth', map_location=device)
if len(sys.argv) > 1:
    if sys.argv[1] == '1':
        loaded_policy_net = torch.load('model2_last.pth', map_location=device)

def select_action_run(state):
    with torch.no_grad():
        return loaded_policy_net(state).max(1).indices.view(1, 1)

# Create a new environment for visualization
env = Game2048Env()

state = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False
total_reward = 0
steps = 0
stuck_counter = 0

while not done:
    action = select_action_run(state)
    observation, reward, done = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    env.render()  # This will print the current state of the board
    print(f"Step: {steps}, Action: {action.item()}, Reward: {reward.item()}")
    
    state = next_state
    total_reward += reward.item()
    steps += 1

    if steps % 100 == 0:
        print(f"Step {steps}, Current max tile: {env.game.get_highest_tile()}, Current score: {env.game.get_score()}")

print(f"Game Over! Total steps: {steps}, Max tile: {env.game.get_highest_tile()}, Total score: {env.game.get_score()}")

# Print model's action preferences for the final state
final_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    action_values = loaded_policy_net(final_state)
print("Final state action values:")
print("Up:", action_values[0][0].item())
print("Down:", action_values[0][1].item())
print("Left:", action_values[0][2].item())
print("Right:", action_values[0][3].item())