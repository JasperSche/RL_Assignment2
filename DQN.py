import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.input = nn.Linear(n_observations, 128)
        self.h1 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.input(observations))
        x = F.relu(self.h1(x))
        return self.out(x)