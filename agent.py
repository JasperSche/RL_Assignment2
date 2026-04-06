import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
from q_net import DQN
from tqdm import tqdm
import time

trans = namedtuple('trans',('s', 'a', 's_next', 'r'))


class ExperienceReplay(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(trans(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self, *args):
        """Save a transition"""
        self.memory.clear()
        
    def __len__(self):
        return len(self.memory)
    


def epsilon_greedy(model, state, epsilon, env, device):
    sample = random.random()
    if sample > epsilon:
        with torch.no_grad():
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def evla_policy(policy_net, device, eval_iterations = 3):
    episode_returns = []
    eval_env = gym.make("CartPole-v1")
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

def optimize_model(policy_net:DQN, target_net:DQN, memory:ExperienceReplay, batch_size:int, gamma:float, device:torch.device, optim):
    if len(memory) < batch_size:
        return policy_net, optim
    
    transitions = memory.sample(batch_size)
    batch = trans(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_next)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.s_next if s is not None])

    state_batch = torch.cat(batch.s)
    action_batch = torch.cat(batch.a)
    reward_batch = torch.cat(batch.r)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * gamma) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optim.step()
    return policy_net, optim

def optimize_model_no_ER(policy_net:DQN, target_net:DQN, memory:ExperienceReplay, gamma:float, device:torch.device, optim):
    transitions = memory.sample(1)
    batch = trans(*zip(*transitions))
    s = torch.cat(batch.s)
    a = torch.cat(batch.a)
    r = torch.cat(batch.r)

    if type(batch.s_next) != torch.tensor:
        expected_state_action_value = r
    else:
        sp = torch.cat(batch.s_next)
        with torch.no_grad():
            next_state_values = target_net(sp).max(1).values
        expected_state_action_value = (next_state_values * gamma) + r

    state_action_value = policy_net(s).gather(1, a)
    criterion = nn.MSELoss()
    loss = criterion(state_action_value, expected_state_action_value.unsqueeze(1))

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optim.step()
    return policy_net, optim

def train_dqn(
        env,
        policy_net, 
        target_net,
        device,
        update_data_ratio = 4,
        memory_capacity = 10000,  
        budget = 1e6,
        gamma = 0.99,
        epsilon = 0.05,
        batch_size = 128,
        lr = 3e-4,
        TN = True,
        ER = True,
        verbose = 1,
        eval_rate = 250
    ):
    memory = ExperienceReplay(memory_capacity)
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    eval_timesteps = []
    eval_returns = []
    counter = 0
    pbar = tqdm(total=budget)
    if not TN: update_data_ratio = 1
    while counter < budget:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        terminated = False
        episode_return = 0
        while counter < budget and not terminated:
            action = epsilon_greedy(policy_net, state, epsilon, env, device)
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

            if not ER:
                policy_net,optimizer = optimize_model_no_ER(
                    policy_net=policy_net,
                    target_net=target_net,
                    memory=memory,
                    gamma=gamma,
                    device=device,
                    optim=optimizer
                )
            else:
                policy_net, optimizer = optimize_model(
                    policy_net=policy_net,
                    target_net=target_net,
                    memory=memory,
                    batch_size=batch_size,
                    gamma=gamma,
                    device=device,
                    optim=optimizer
                )
           
            if counter % update_data_ratio == 0:
                target_net.load_state_dict(policy_net.state_dict())

            counter += 1

            if counter % eval_rate == 0:
                pbar.update(eval_rate)
                eval_timesteps.append(counter)
                eval_return = evla_policy(policy_net, device)
                eval_returns.append(eval_return)
                if verbose == 1:
                    print(f'Episode return: {eval_return}')
    return eval_timesteps, eval_returns
