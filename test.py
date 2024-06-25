import gymnasium as gym

env = gym.make('CarRacing-v2', render_mode='human', continuous=False)
state = env.reset()

print(env.action_space)
print(env.observation_space)

for i in range(1000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        env.reset()