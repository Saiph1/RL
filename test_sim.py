import gymnasium as gym
from Deep_q_CartPole import DQN
import torch
# from gymnasium.wrappers import RescaleAction
from custom_sim import LunarLander

# env = gym.make(
#     "LunarLander-v2", render_mode="human")
env = LunarLander(render_mode="human")
 # Get number of actions from gym action space
n_actions = env.action_space.n
print("n_action", n_actions)
# Get the number of state observations
state, info = env.reset()
# print("state:", state)
n_observations = len(state)
# print("n_observation:", n_observations)
observation, info = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load("minimize_loss_withguide_y100view10000dis10000_andgledevpenalty20000_velocitypenalty1000_endRefine_reward_15_deep_q_policy_net_custom_1000.pth"))
model.eval()

for _ in range(1000):
    # action = env.action_space.sample()  
    ob = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = model(ob).max(1)[1].view(1, 1)
    observation, reward, terminated, truncated, info = env.step(action.item())
    # print("Rewards = ", reward)
    # print("Observation =", observation[2])
    # print("ob2", observation[3])
    if terminated or truncated:
        observation, info = env.reset()

env.close()