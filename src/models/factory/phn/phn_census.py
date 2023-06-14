# Adapted from https://github.com/ruchtem/cosmos and https://github.com/AvivNavon/pareto-hypernetworks
import torch.nn as nn
import torch.nn.functional as F


def measure_target_network_params(params):
    # for sanity check
    num_params = sum([v.numel() for k, v in params.items()])
    print("NUM_PARAMS=", num_params)


class CensusHyper(nn.Module):
    def __init__(self, ray_hidden_dim=100, num_tasks=2):
        super().__init__()

        self.feature_dim = 468

        self.ray_mlp = nn.Sequential(
            nn.Linear(num_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.fc_0_weights = nn.Linear(ray_hidden_dim, 256 * self.feature_dim)
        self.fc_0_bias = nn.Linear(ray_hidden_dim, 256)
        self.fc_1_weights = nn.Linear(ray_hidden_dim, 2 * 256)
        self.fc_1_bias = nn.Linear(ray_hidden_dim, 2)
        self.fc_2_weights = nn.Linear(ray_hidden_dim, 2 * 256)
        self.fc_2_bias = nn.Linear(ray_hidden_dim, 2)

    def forward(self, ray):
        x = self.ray_mlp(ray)
        out_dict = {
            "fc.weights": self.fc_0_weights(x).reshape(256, self.feature_dim),
            "fc.bias": self.fc_0_bias(x),
            "fc1.weights": self.fc_1_weights(x).reshape(2, 256),
            "fc1.bias": self.fc_1_bias(x),
            "fc2.weights": self.fc_2_weights(x).reshape(2, 256),
            "fc2.bias": self.fc_2_bias(x),
        }
        # measure_target_network_params(out_dict)
        return out_dict


class CensusTarget(nn.Module):
    def forward(self, x, weights):
        x = F.linear(x, weight=weights["fc.weights"], bias=weights["fc.bias"])
        x = F.relu(x)
        x1 = F.linear(x, weight=weights["fc1.weights"], bias=weights["fc1.bias"])
        x2 = F.linear(x, weight=weights["fc2.weights"], bias=weights["fc2.bias"])
        return x1, x2
