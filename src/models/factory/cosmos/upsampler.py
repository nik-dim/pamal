# Adapted from https://github.com/ruchtem/cosmos
from typing import Iterator

import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class Upsampler(BaseModel):
    def __init__(self, K, child_model, input_dim):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling)
        In case of image data: use a transposed CNN for the sampled rays.
        """
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:]),
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions")

        self.child_model = child_model
        self.ray = None

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return []

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return []

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return []

    def forward(self, x, return_embedding=False):
        b = x.shape[0]
        a = self.ray.repeat(b, 1)

        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(self.ray), 1, 1)
            a = self.transposed_cnn(a)

        x = torch.cat((x, a), dim=1)
        return self.child_model(x, return_embedding)
