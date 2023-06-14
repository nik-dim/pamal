import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class PamalWeightMethod:
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        logging.warn(f"Initializing {self.__repr__()}")

    def connect_device(self, device):
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        if grad_scaler:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        task_specific_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        return []


class PamalLinearScalarization(PamalWeightMethod):
    def __init__(self, num_tasks):
        super().__init__(num_tasks)

    def get_weighted_loss(self, losses, task_weights: torch.Tensor, **kwargs):
        loss = sum([l * tw for l, tw in zip(losses, task_weights)])
        return loss, dict(weights=task_weights)

    def __repr__(self) -> str:
        return f"PamalLinearScalarization()"


class PaMaLRunningWeightAverage(PamalWeightMethod):
    def __init__(self, num_tasks, iteration_window: int = 25, temp=2.0):
        super().__init__(num_tasks)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, task_weights, **kwargs):
        losses = torch.stack(losses)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            ws = self.costs[self.iteration_window :, :].mean(0)
            self.weights = 1 - np.abs(ws) / np.abs(ws).sum()

        runnin_avg_task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        task_weights = task_weights.to(losses.device)
        loss = (runnin_avg_task_weights * losses * task_weights).sum()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)


class PamalGradientNormalization(PamalWeightMethod):
    def __init__(self, num_tasks: int):
        super().__init__(num_tasks)

    def get_weighted_loss(self, losses, task_weights, shared_parameters, **kwargs):
        grads = {}
        norm_grads = {}
        norm_terms = {}

        for i, loss in enumerate(losses):
            g = list(torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True))
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)
            norm_terms[i] = norm_term

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        xi = 0.001
        loss = sum([losses[i] * task_weights[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=task_weights)


class PaMaL_GL(PamalWeightMethod):
    def __init__(self, num_tasks, iteration_window: int = 25, temp=2.0):
        super().__init__(num_tasks)
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, num_tasks), dtype=np.float32)
        self.weights = np.ones(num_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, task_weights, shared_parameters, **kwargs):
        losses = torch.stack(losses)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[: self.iteration_window, :].mean(0)
            ws = self.costs[self.iteration_window :, :].mean(0)
            self.weights = 1 - np.abs(ws) / np.abs(ws).sum()

        runnin_avg_task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(losses.device)
        task_weights = task_weights.to(losses.device)
        self.running_iterations += 1

        losses = runnin_avg_task_weights * losses

        grads = {}
        norm_grads = {}
        norm_terms = {}

        for i, loss in enumerate(losses):
            g = list(
                torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
            )
            grad = torch.cat([torch.flatten(grad) for grad in g])
            norm_term = torch.norm(grad)
            norm_terms[i] = norm_term

            grads[i] = grad
            norm_grads[i] = grad / norm_term

        xi = 0.001
        loss = sum([losses[i] * task_weights[i] / (norm_terms[i] + xi) for i in range(len(losses))])
        return loss, dict(weights=task_weights)


PaMaLMETHODS = dict(
    ls=PamalLinearScalarization,
    rwa=PaMaLRunningWeightAverage,
    gradnorm=PamalGradientNormalization,
    full=PaMaL_GL,
)
