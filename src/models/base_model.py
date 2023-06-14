import copy
import logging
from typing import Iterator

import torch.nn as nn
from src.utils.callbacks.base_callback import BaseCallback


class BaseModel(nn.Module, BaseCallback):
    """The BaseModel class has the functionality of the typical nn.Module from PyTorch as
    well as callback functionality provided from BaseCallback."""

    def __init__(self):
        super().__init__()


class BaseModelWrapper(BaseModel):
    def __init__(self, model, config, *args, **kwargs):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, x):
        return self.model(x)


class SharedBottom(BaseModel):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_tasks,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder = encoder
        if isinstance(decoder, list):
            assert len(decoder) == self.num_tasks
            self.decoders = nn.ModuleList(decoder)
        elif isinstance(decoder, dict):
            assert len(decoder) == self.num_tasks
            self.decoders = nn.ModuleList(decoder.values())
        else:
            self.decoders = nn.ModuleList([self._create_new(decoder) for _ in range(num_tasks)])

    def _create_new(self, module: nn.Module, reinit=True):
        new_module = copy.deepcopy(module)
        if reinit:
            if hasattr(new_module, "reset_parameters"):
                new_module.reset_parameters()
            else:
                for m in new_module.children():
                    m.reset_parameters()
        return new_module

    def forward(self, x, return_embedding=False):
        embedding = self.encoder(x)

        task_outs = [d(embedding) for d in self.decoders]

        if return_embedding:
            return task_outs, embedding
        else:
            return task_outs

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.encoder.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.decoders.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        # TODO: fix this for MGDA
        return self.encoder.get_last_layer().parameters()
