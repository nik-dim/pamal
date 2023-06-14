import logging

import torch

from src.ll.weight_ensemble import SimplexWeightEnsemble
from src.methods.multi_solution_weight_methods import Cosmos, HypernetMethod
from src.methods.pamal_weight_methods import PaMaLMETHODS
from src.methods.weight_methods import METHODS
from src.trainer.base_trainer import BaseTrainer
from src.trainer.ensemble_trainer import EnsembleTrainer
from src.trainer.multi_forward_ensemble_trainer import MultiForwardEnsembleTrainer
from src.trainer.multi_solution_trainer import MultiSolutionTrainer
from src.utils.callbacks.cityscapes_metric_cb import CityscapesMetricCallback
from src.utils.callbacks.mtl_metric_callback import (
    ClassificationMultiTaskMetricCallback,
    UTKFaceMultiTaskMetricCallback,
)
from src.utils.callbacks.save_model import SaveModelCallback


def get_callbacks(config, num_tasks):
    metric_cb = ClassificationMultiTaskMetricCallback(num_tasks=num_tasks, use_task_names=True)
    if config.data.dataset == "utkface":
        metric_cb = UTKFaceMultiTaskMetricCallback()
    if config.data.dataset == "cityscapes":
        metric_cb = CityscapesMetricCallback()

    save_cb = SaveModelCallback()
    callbacks = [metric_cb, save_cb]

    return callbacks


def get_trainer(config, trainer_kwargs, num_tasks, model=None):
    if config.method.name == "pamal":
        weight_method = PaMaLMETHODS[config.method.inner_method](num_tasks=num_tasks)
        trainer_kwargs.update(dict(method=weight_method, validate_every_n=getattr(config, "validate_every_n", 2)))
        if getattr(config.method, "num", 1) == 1:
            trainer = EnsembleTrainer(**trainer_kwargs)
        else:
            trainer = MultiForwardEnsembleTrainer(
                **trainer_kwargs,
                num=config.method.num,
                temperature=config.method.temperature,
                reg_coefficient=config.method.reg_coefficient,
            )
    elif config.method.name == "cosmos":
        trainer_kwargs["method"] = Cosmos(num_tasks, config.method.lamda)
        trainer = MultiSolutionTrainer(config.method.alpha, **trainer_kwargs)
    elif config.method.name == "phn":
        trainer_kwargs["method"] = HypernetMethod(num_tasks, model, config.method.solver)
        trainer = MultiSolutionTrainer(config.method.alpha, **trainer_kwargs)
    else:
        if config.method.name == "ls":
            kwargs = dict(task_weights=[1 / num_tasks] * num_tasks)
        elif config.method.name == "stl":
            kwargs = dict(main_task=config.method.main_task)
        else:
            kwargs = {}
        weight_method = METHODS[config.method.name](num_tasks=num_tasks, **kwargs)

        trainer_kwargs["optimizer"].add_param_group({"params": weight_method.parameters()})
        trainer = BaseTrainer(method=weight_method, **trainer_kwargs)

    return trainer


def get_optimizer(config, param_groups):
    if config.optimizer.type == "Adam":
        optimizer = torch.optim.Adam(param_groups, lr=config.optimizer.lr)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=config.optimizer.lr, momentum=config.optimizer.momentum)

    return optimizer


def get_ensemble_model(base_model, num_tasks, config):
    return SimplexWeightEnsemble(
        model=base_model,
        num_tasks=num_tasks,
        validate_models=config.method.validate_models,
        reinit_flag=config.method.reinit_flag,
        p=config.method.p,
    )
