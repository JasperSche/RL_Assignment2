import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from itertools import count

from Q_net import DQN
from utils import ExperienceReplay, optimize_model
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = gym.make("CartPole-v1", render_mode="human")
eval_env = gym.make("CartPole-v1", render_mode="human")
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
# Probably impractical, with 1 million timesteps evaluation takes too long
def evaluate(eval_env,n_eval_episodes=30, max_episode_length=500):
    returns = []  # list to store the reward per episode
    for i in range(n_eval_episodes):
        state, info = eval_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        R_ep = 0
        for t in range(max_episode_length):
            action = select_action(state)
            observation, reward, terminated, truncated, _ = eval_env.step(action.item())
            reward = torch.tensor([reward], device=device)
            R_ep += reward
            done = terminated or truncated
            if done:
                break
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                state = next_state
                    
        returns.append(R_ep)
    mean_return = torch.mean(torch.tensor(returns, device=device))
    return mean_return
    
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ExperienceReplay(10000)

episode_durations = []

#if torch.cuda.is_available() or torch.backends.mps.is_available():
#    num_episodes = 600
#else:
#    num_episodes = 50
counter = 0
budget = 1000000
#Not implemented yet
n_steps = 1
#Based on the baseline
eval_interval = 250
max_episode_length = 500

ER = True
TN = True
returns = []
eval_timesteps = []
eval_returns = []
#Depends on evaluation being viable or not
#if counter%eval_interval == 0:
#    eval_timesteps.append(counter)
#    eval_returns.append(evaluate(eval_env = eval_env))
while counter < budget:
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    finished_episode = False
    episode_length = 0
    episode_return = 0
    while counter < budget and not finished_episode and episode_length < max_episode_length:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        episode_return = episode_return + 1
        done = terminated or truncated
        counter = counter + 1
        episode_length = episode_length + 1
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        if not ER:
            memory.clear()
        memory.push(state, action, next_state, reward)

        state = next_state

        policy_net, optimizer = optimize_model(
            policy_net=policy_net,
            target_net=target_net,
            memory=memory,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            device=device,
            optim=optimizer
        )

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        #Not 100% sure about this one but it makes sense to me
        if TN:
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        else:
            target_net_state_dict = policy_net.state_dict()
        target_net.load_state_dict(target_net_state_dict)    
        #Depends on evaluation being viable or not
        #if counter%eval_interval == 0:
        #    eval_timesteps.append(counter)
        #    eval_returns.append(evaluate(eval_env = eval_env))
        if done:
            episode_durations.append(episode_length + 1)
            finished_episode = True
            # plot_durations()
            break
    if counter >= 12500:
        eval_timesteps.append(counter)
        eval_returns.append(episode_return)
df = pd.DataFrame({"eval_timesteps": eval_timesteps, "eval_returns":eval_returns})
df.to_csv('Data.csv')
print('done')