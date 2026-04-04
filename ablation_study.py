import numpy as np
import torch
import gymnasium as gym
import random
from Q_net import DQN
from agent import train_dqn
import pandas as pd
import os
from tqdm import tqdm


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

net_lengths = [1,2,3]
lr_set = [1e-3,6.25e-5,1e-7]
epsilon_values = [0.1,0.05,0.01 ]
update_to_data_ratios = [10,100,1000]

n_actions = 2
n_observations = 4

random.seed(120)

for net_length in net_lengths:
    for lr in lr_set:
        for epsilon in epsilon_values:
            for update_to_data_ratio in update_to_data_ratios:
                file_path = f'Results/Results_{net_length}_{lr}_{epsilon}_{update_to_data_ratio}.csv'
                if not os.path.exists(file_path):
                    print(f'Running experiment: Net_size:{net_length} Lr:{lr} Epsilon:{epsilon} Update/Data Ratio:{update_to_data_ratio}')
                    results = []
                    for i in tqdm(range(5)):
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
                            budget=20000,
                            epsilon=epsilon,
                            lr=lr,
                            TN=True,
                            ER=True,
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