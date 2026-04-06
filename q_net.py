import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers = 1):
        super(DQN, self).__init__()
        self.n_layers = n_layers
        self.input = nn.Linear(n_observations, 128)
        for idx in range(self.n_layers):
            setattr(self, 'layer%d' % (idx+1), nn.Linear(128,128))
        self.out = nn.Linear(128, n_actions)

    def forward(self, observations):
        x = F.relu(self.input(observations))
        for idx in range(self.n_layers):
            x = F.relu(getattr(self, 'layer%d' % (idx+1))(x))
        return self.out(x)
