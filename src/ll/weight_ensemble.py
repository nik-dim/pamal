import copy
import logging
import random
from typing import Dict, Iterator, List, Tuple, Union

import meshzoo
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from src.utils.callbacks.base_callback import BaseCallback

from .subspace_modules import SubspaceBatchNorm2d, SubspaceConv, SubspaceLinear
import matplotlib.pyplot as plt


class _BaseWeightEnsemble(nn.Module, BaseCallback):
    """The BaseModel class has the functionality of the typical nn.Module from PyTorch as
    well as callback functionality provided from BaseCallback."""

    reinit_flag: bool

    def __init__(self):
        super().__init__()
        self._counter = 0
        self.cat_metric = torchmetrics.CatMetric()

    def reset_counter(self):
        self._counter = 0

    def on_before_forward(self, *args, **kwargs):
        """In case of training, an alpha is sampled. For evaluation, the alpha is set manually."""
        if self.training:
            self.set_alpha_for_batch(alpha=None)
            self.cat_metric(self.alpha)

    # def on_before_training_step(self, *args, **kwargs):
    #     self.set_alpha_for_batch(alpha=None)

    # def on_after_training_step(self, *args, **kwargs):
    #     self.cat_metric(self.alpha)

    def on_after_fit(self, *args, **kwargs):
        samples = torch.stack(self.cat_metric.value, dim=1)[0].detach().cpu().numpy()
        samples = np.array(samples).squeeze()
        fig = plt.figure(figsize=(6, 6))
        n, bins, patches = plt.hist(samples, 50, density=True, facecolor="g", alpha=0.75)

        plt.xlabel(r"$\alpha$", fontsize=16)
        plt.ylabel("Histogram", fontsize=16)
        plt.grid(True)

        plt.savefig("sampling.png")
        return super().on_after_fit(*args, **kwargs)

    def set_alpha_for_batch(self, alpha=None):
        if alpha is None:
            alpha = self.sample_alpha()
        self.set_alpha(alpha)

    def sample_alpha(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def set_task_weights_list(self, task_weights_list):
        return torch.vstack([torch.Tensor(t) for t in task_weights_list])

    def make_subspace_compatible(self, module, name=""):
        for name, immediate_child_module in module.named_children():
            if isinstance(immediate_child_module, nn.Conv2d):
                m = getattr(module, name)
                setattr(module, name, SubspaceConv(m, self.n, self.reinit_flag))
            elif isinstance(immediate_child_module, nn.Linear):
                m = getattr(module, name)
                setattr(module, name, SubspaceLinear(m, self.n, self.reinit_flag))
            elif isinstance(immediate_child_module, nn.BatchNorm2d):
                m = getattr(module, name)
                setattr(module, name, SubspaceBatchNorm2d(m, self.n, self.reinit_flag))
            elif isinstance(immediate_child_module, SubspaceConv):
                break
            elif isinstance(immediate_child_module, SubspaceLinear):
                break
            else:
                self.make_subspace_compatible(immediate_child_module, name)

    def combine_losses(self, losses):
        if self.num_tasks == 1:
            return losses
        return sum([w * l for w, l in zip(self.task_weights, losses)])

    def set_alpha_recursively(self, alpha, module: nn.Module):
        for k, v in module.named_children():
            if isinstance(v, (SubspaceConv, SubspaceLinear, SubspaceBatchNorm2d)):
                setattr(v, f"alpha", alpha)
            else:
                self.set_alpha_recursively(alpha, v)

    def set_alpha(self, alpha):
        """Sets alpha on task weights and then traverses the model like a tree to set the alpha on all modules."""
        self.alpha = alpha
        self.task_weights = sum([tw * a for tw, a in zip(self.task_weights_list, self.alpha)])
        self.set_alpha_recursively(alpha=alpha, module=self)

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.shared_parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.task_specific_parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.model.last_shared_parameters()


class BaseWeightEnsemble(_BaseWeightEnsemble):
    def __init__(self, model, num_tasks, task_weights_list, reinit_flag):
        super().__init__()
        self.model: nn.Module = copy.deepcopy(model)
        self.num_tasks = num_tasks
        self.register_buffer("task_weights_list", self.set_task_weights_list(task_weights_list))
        self.reinit_flag = reinit_flag
        self.n = len(task_weights_list)
        self.num_members = len(task_weights_list)
        self.make_subspace_compatible(self.model)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def next(self):
        # set alpha
        index = self._counter
        alpha = self.points[index]
        self.set_alpha(alpha)

        # increment counter
        num_points = len(self.points)
        self._counter += 1
        self._counter = self._counter % num_points

        return index, alpha


class SimplexWeightEnsemble(BaseWeightEnsemble):
    def __init__(
        self,
        model,
        num_tasks,
        reinit_flag,
        validate_models=10,
        p: Union[float, torch.Tensor] = 1.0,
    ):
        task_weights_list = self.generate_single_task_weights_list(num_tasks)
        super().__init__(model, num_tasks, task_weights_list, reinit_flag)
        self.validate_models = validate_models
        self.points = self.create_interpolated_points()
        logging.info(self.points)
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
            self.p = torch.ones(self.num_tasks) * p
            self.sampling_fn = torch.distributions.Dirichlet(self.p)

    @staticmethod
    def generate_single_task_weights_list(num_tasks):
        task_weights_list = np.eye(num_tasks).tolist()
        return task_weights_list

    def sample_alpha(self):
        return self.sampling_fn.rsample()

    def create_interpolated_points(self) -> List[List[float]]:
        if self.validate_models == -1:
            # only validate the midpoint
            midpoint = np.ones(self.num_members) / self.num_members
            return [midpoint.tolist()]
        elif self.validate_models == 0 or self.num_members > 3:
            logging.warn("For more than 3 members, we only evaluate the 'clean' and the midpoint models.")
            points = []
            for m in range(self.num_members):
                alpha = np.zeros(self.num_members)
                alpha[m] = 1
                points.append(alpha.tolist())

            midpoint = np.ones(self.num_members) / self.num_members
            points.append(midpoint.tolist())
            return points
        if self.num_members == 2:
            points = np.linspace(0, 1, self.validate_models).tolist()
            points = [[p, 1 - p] for p in points]
            return points[::-1]
        elif self.num_members == 3:
            points, _ = meshzoo.triangle(self.validate_models - 1)
            return points.T
