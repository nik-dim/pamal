# Adapted from https://github.com/ruchtem/cosmos and https://github.com/AvivNavon/pareto-hypernetworks
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


def measure_target_network_params(params):
    # for sanity check
    num_params = sum([v.numel() for k, v in params.items()])
    print("NUM_PARAMS=", num_params)


class LeNetHyper(nn.Module):
    """Hypernetwork"""

    def __init__(self, ray_hidden_dim=100, num_tasks=2):
        super().__init__()
        self.ray_hidden_dim = ray_hidden_dim
        self.num_tasks = num_tasks
        self.ray_mlp = nn.Sequential(
            nn.Linear(num_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            # nn.ReLU(inplace=True),
        )

        self.conv_1_weights = nn.Linear(ray_hidden_dim, 10 * 5 * 5)
        self.conv_1_bias = nn.Linear(ray_hidden_dim, 10)
        self.conv_2_weights = nn.Linear(ray_hidden_dim, 20 * 10 * 5 * 5)
        self.conv_2_bias = nn.Linear(ray_hidden_dim, 20)
        self.fc_weights = nn.Linear(ray_hidden_dim, 50 * 320)
        self.fc_bias = nn.Linear(ray_hidden_dim, 50)

        self.task_1_fc1_weights = nn.Linear(ray_hidden_dim, 50 * 50)
        self.task_1_fc1_bias = nn.Linear(ray_hidden_dim, 50)
        self.task_1_fc2_weights = nn.Linear(ray_hidden_dim, 10 * 50)
        self.task_1_fc2_bias = nn.Linear(ray_hidden_dim, 10)

        self.task_2_fc1_weights = nn.Linear(ray_hidden_dim, 50 * 50)
        self.task_2_fc1_bias = nn.Linear(ray_hidden_dim, 50)
        self.task_2_fc2_weights = nn.Linear(ray_hidden_dim, 10 * 50)
        self.task_2_fc2_bias = nn.Linear(ray_hidden_dim, 10)

        if self.num_tasks == 3:
            self.task_3_fc1_weights = nn.Linear(ray_hidden_dim, 50 * 50)
            self.task_3_fc1_bias = nn.Linear(ray_hidden_dim, 50)
            self.task_3_fc2_weights = nn.Linear(ray_hidden_dim, 10 * 50)
            self.task_3_fc2_bias = nn.Linear(ray_hidden_dim, 10)

    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if "task" not in n])

    def forward(self, ray):
        features = self.ray_mlp(ray)
        out_dict = {
            "encoder.conv1.weight": self.conv_1_weights(features),
            "encoder.conv1.bias": self.conv_1_bias(features),
            "encoder.conv2.weight": self.conv_2_weights(features),
            "encoder.conv2.bias": self.conv_2_bias(features),
            "encoder.fc.weight": self.fc_weights(features),
            "encoder.fc.bias": self.fc_bias(features),
            "decoders.0.fc1.weight": self.task_1_fc1_weights(features),
            "decoders.0.fc1.bias": self.task_1_fc1_bias(features),
            "decoders.0.fc2.weight": self.task_1_fc2_weights(features),
            "decoders.0.fc2.bias": self.task_1_fc2_bias(features),
            "decoders.1.fc1.weight": self.task_2_fc1_weights(features),
            "decoders.1.fc1.bias": self.task_2_fc1_bias(features),
            "decoders.1.fc2.weight": self.task_2_fc2_weights(features),
            "decoders.1.fc2.bias": self.task_2_fc2_bias(features),
        }
        if self.num_tasks == 3:
            out_dict.update(
                {
                    "decoders.2.fc1.weight": self.task_3_fc1_weights(features),
                    "decoders.2.fc1.bias": self.task_3_fc1_bias(features),
                    "decoders.2.fc2.weight": self.task_3_fc2_weights(features),
                    "decoders.2.fc2.bias": self.task_3_fc2_bias(features),
                }
            )

        # measure_target_network_params(out_dict)
        return out_dict


class LeNetTarget(nn.Module):
    """Target network"""

    def __init__(self, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks

    def forward(self, x, weights=None):
        x = F.conv2d(
            x,
            weight=weights["encoder.conv1.weight"].reshape(10, 1, 5, 5),
            bias=weights["encoder.conv1.bias"],
            stride=1,
        )
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.conv2d(
            x,
            weight=weights["encoder.conv2.weight"].reshape(20, 10, 5, 5),
            bias=weights["encoder.conv2.bias"],
            stride=1,
        )
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = F.linear(
            x,
            weight=weights["encoder.fc.weight"].reshape(50, 320),
            bias=weights["encoder.fc.bias"],
        )

        logits = []
        for j in range(self.num_tasks):
            y_hat = F.linear(
                x,
                weight=weights[f"decoders.{j}.fc1.weight"].reshape(50, 50),
                bias=weights[f"decoders.{j}.fc1.bias"],
            )
            y_hat = F.relu(y_hat)
            y_hat = F.linear(
                x,
                weight=weights[f"decoders.{j}.fc2.weight"].reshape(10, 50),
                bias=weights[f"decoders.{j}.fc2.bias"],
            )

            logits.append(y_hat)

        return logits
