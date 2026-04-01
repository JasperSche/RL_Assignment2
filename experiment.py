import gymnasium as gym
import numpy as np
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

env = gym.make("CartPole-v1")




    
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


#Not implemented yet
#Based on the baseline
eval_interval = 250
## unnessisary because thh 500th state is terminal per definition


def epsilon_greedy(model, state, epsilon):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def update_target_net(target_net, policy_net,tau,TN):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    if TN:
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
    else:
        target_net_state_dict = policy_net.state_dict()
    target_net.load_state_dict(target_net_state_dict)    

def evla_policy(policy_net, eval_iterations = 10):
    eval_env = gym.make("CartPole-v1")
    episode_returns = []
    for _ in range(eval_iterations):
        episode_return = 0
        s,_ = eval_env.reset()
        s = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        while not terminated:
            #Just do greedy for evaluation
            with torch.no_grad():
                a =  policy_net(s).max(1).indices.view(1, 1)
            sp, r, terminated, truncated, _ = eval_env.step(a.item())
            terminated = terminated or truncated
            episode_return += r
            sp = torch.tensor(sp, dtype=torch.float32, device=device).unsqueeze(0)
            s = sp
        episode_returns.append(episode_return)
    eval_env.close()
    return np.mean(episode_returns)

def train_dqn(
        env,
        policy_net, 
        target_net,
        memory_capacity = 10000,  
        budget = 1e5,
        gamma = 0.99,
        epsilon = 0.05,
        tau = 5e-3,
        batch_size = 128,
        lr = 3e-4,
        TN = True,
        ER = True
    ):
    memory = ExperienceReplay(memory_capacity)
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    eval_timesteps = []
    eval_returns = []
    counter = 0
    if not ER:
        batch_size = 1
    while counter < budget:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        episode_return = 0
        while counter < budget and not terminated:
            action = epsilon_greedy(policy_net,state,epsilon)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            terminated = terminated or truncated
            reward = torch.tensor([reward], device=device)
            episode_return += reward

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
                batch_size=batch_size,
                gamma=gamma,
                device=device,
                optim=optimizer
            )
            update_target_net(
                target_net=target_net,
                policy_net=policy_net,
                tau=tau,
                TN=TN
            )
            
            counter += 1

            if counter % 1000 == 0:
                eval_timesteps.append(counter)
                episode_return = evla_policy(policy_net)
                eval_returns.append(episode_return)
                print(f'Episode return: {episode_return}')
    return eval_timesteps, eval_returns

        


eval_timesteps, eval_returns = train_dqn(env=env,policy_net=policy_net,target_net=target_net)
df = pd.DataFrame({"eval_timesteps": eval_timesteps, "eval_returns":eval_returns})
df.to_csv('Data.csv')
print('done')