# RL_Assignment2
This repository contains the source code for the 2. Assignment for Reinforcement Learning by Jasper Scheel, and Santiago Moreno Mercado
The source code to train the agent is contained in the `agent.py` file, and the Deep Q Net is contained in the `q_net.py` file.
## Ablation study
To run the ablation study, execute the file `ablation_study.py`. It includes a file check, so it won't rerun already completed experiments. This makes it possible to add settings without having to rerun every single experiment. If the file is executed as is, the results of the ablation study will be shown.
## Configurations Experiment
The configurations experiment is contained in the `configuration_tests.py` file. Running this file will **only execute the currently defined experiment.** All the results of the configuration tests can be visulized using the `plot_configuration_test.py` script.
## Training an agent
In order to train an agent, please do the following:
1. Import required libaries
```Python
import torch
import gymnasium as gym
from q_net import DQN
from agent import train_dqn
```
2. Initilize CartPole environment
```Python
env = gym.make("CartPole-v1")
```
3. Create the DQN models (policy and target)
```Python
policy_net = DQN(n_observations, n_actions, net_length).to(device)
target_net = DQN(n_observations, n_actions, net_length).to(device)
target_net.load_state_dict(policy_net.state_dict())
```
4. Pass Hyperparameters, DQN models and environment to training function
```Python
curr_eval_timesteps, curr_eval_returns = train_dqn(
    env=env,
    policy_net=policy_net,
    target_net=target_net,
    device="cpu", #set to "cuda" if torch.cuda.is_available() == True
    update_data_ratio=100,
    budget=1e6,
    epsilon=0.05,
    lr=6.25e-5,
    TN=True,
    ER=True,
    verbose=0, #Print out evaluation results each eval_rate iterations
    eval_rate=250
)
```
