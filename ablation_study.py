import numpy as np
import torch
import gymnasium as gym
import random
from q_net import DQN
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
lr_set = [1e-3,1e-4,6.25e-5,1e-6,1e-7]
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


ablation_results = []
for net_length in net_lengths:
    for lr in lr_set:
        for epsilon in epsilon_values:
            for update_to_data_ratio in update_to_data_ratios:
                file_path = f'Results/Results_{net_length}_{lr}_{epsilon}_{update_to_data_ratio}.csv'
                ## Grade settings based on Performance (avg return) Effifciency (avg gain) and stability(std over last 10 evaluations)
                df = pd.read_csv(file_path)
                mean_returns = df['eval_mean_returns'].to_numpy()
                std_returns = df['eval_std_returns'].to_numpy()
                ablation_results.append(
                    {
                        'net_depth':net_length,
                        'learning_rate':lr,
                        'epsilon':epsilon,
                        'update_to_data_ratio':update_to_data_ratio,
                        'avg_return':np.mean(mean_returns),
                        # 'avg_return_gain':np.mean([mean_returns[idx]-mean_returns[idx-1] for idx in range(1,len(mean_returns))]),
                        'avg_std_last_10':np.mean(std_returns[-10])
                    }
                )

results = pd.DataFrame(ablation_results)
# print(results)

###Show results:
import matplotlib as plt
import matplotlib.pyplot as plt

def plot_avarages(setting_name:str):
    returns = []
    std_last_10 = []
    uniqe_settings = sorted(np.unique(results[setting_name]))
    for setting in uniqe_settings:
        returns.append(results[results[setting_name] == setting]['avg_return'].mean())
        std_last_10.append(results[results[setting_name] == setting]['avg_std_last_10'].mean())
    fig, ax = plt.subplots()
    ax1 = ax.twinx()
    ax.set_xlabel('Setting values')
    ax.set_ylabel('Average Returns')
    ax1.set_ylabel('Average standart deviation over last 10 runs')
    ax.set_title(f'Ablation Study {setting_name} result')

    lns1 = ax.plot(uniqe_settings, returns, color='red', label = 'Avg Returns')
    lns2 = ax1.plot(uniqe_settings, std_last_10, label = 'Std. last 10 returns')
    leg = lns1 + lns2
    labs = [l.get_label() for l in leg]
    ax.legend(leg, labs, loc=0)
    plt.savefig(f"Ablation_Plots/{setting_name}_result.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_avarages('net_depth')
plot_avarages('learning_rate')
plot_avarages('epsilon')
plot_avarages('update_to_data_ratio')



