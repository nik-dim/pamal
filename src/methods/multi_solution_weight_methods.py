from typing import List, Tuple, Union

import torch
from src.models.factory.phn.solvers import EPOSolver, LinearScalarizationSolver

from .weight_methods import WeightMethod


def num_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


class Cosmos(WeightMethod):
    def __init__(self, num_tasks: int, lamda=8):
        super().__init__(num_tasks)
        self.lamda = lamda

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.ray)
        cossim = torch.nn.functional.cosine_similarity(losses, self.ray, dim=0)
        loss -= self.lamda * cossim
        return loss, dict(weights=self.ray, cossim=cossim)


class HypernetMethod(WeightMethod):
    def __init__(self, num_tasks, hnet, internal_solver, **kwargs):
        self.objectives = num_tasks
        self.num_tasks = num_tasks

        if internal_solver == "linear":
            self.solver = LinearScalarizationSolver(num_tasks=self.num_tasks)
        elif internal_solver == "epo":
            print(hnet)
            self.solver = EPOSolver(num_tasks=self.num_tasks, n_params=num_parameters(hnet))

    def connect_device(self, trainer):
        super().connect_device(trainer)

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        loss = self.solver(losses, self.ray, list(shared_parameters))
        return loss, dict(ray=self.ray)
