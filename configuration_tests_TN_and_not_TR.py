from tqdm import tqdm
import numpy as np
import torch
import gymnasium as gym
from Q_net import DQN
from agent import train_dqn
import pandas as pd
from tqdm import tqdm

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

#Experiment COnfigurations
ER = False
TN = True

n_observations = 4
n_actions = 2

#Best settings:
net_length = 2
lr = 6.25e-5
epsilon = 0.05
update_to_data_ratio = 100

file_path = f'Full_Run_Results/no_ER_and_TN.csv'
print(f'Running experiment: {file_path}')

results = []
for i in range(5):
    curr_env = gym.make("CartPole-v1")
    policy_net = DQN(n_observations, n_actions, net_length).to(device)
    target_net = DQN(n_observations, n_actions, net_length).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    curr_eval_timesteps, curr_eval_returns = train_dqn(
        env=curr_env,
        policy_net=policy_net,
        target_net=target_net,
        device=device,
        update_data_ratio=update_to_data_ratio,
        epsilon=epsilon,
        lr=lr,
        TN=TN,
        ER=ER,
        verbose=0,
        eval_rate=250
    )
    results.append(np.array(curr_eval_returns))
results = np.array(results)
df = pd.DataFrame({
    "eval_timesteps": curr_eval_timesteps,
    "eval_mean_returns":np.mean(results, axis=0),
    "eval_std_returns":np.std(results, axis=0)
    })
df.to_csv(file_path)
