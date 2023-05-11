import gymnasium as gym
from Deep_q_CartPole import DQN
import torch

env = gym.make("CartPole-v1", render_mode="human")
 # Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

observation, info = env.reset(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load("deep_q_target_net_middle.pth"))
model.eval()

for _ in range(600):
    # action = env.action_space.sample()  
    ob = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = model(ob).max(1)[1].view(1, 1)
    observation, reward, terminated, truncated, info = env.step(action.item())
    # print("Rewards = ", reward)
    # print("Observation =", observation)
    if terminated or truncated:
        observation, info = env.reset()

env.close()