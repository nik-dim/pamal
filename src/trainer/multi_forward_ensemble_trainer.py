import logging
from typing import List, Optional

import torch
import wandb
from src.datasets.base_data_module import BaseDataModule
from src.ll.weight_ensemble import BaseWeightEnsemble
from src.methods.weight_methods import WeightMethod
from src.utils.callbacks.base_callback import BaseCallback
from src.utils.loggers.base_logger import BaseLogger

from .ensemble_trainer import EnsembleTrainer


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


class MultiForwardEnsembleTrainer(EnsembleTrainer):
    model: BaseWeightEnsemble

    def __init__(
        self,
        num,
        temperature,
        reg_coefficient,
        model: BaseWeightEnsemble,
        benchmark: BaseDataModule,
        method: WeightMethod = None,
        gpu=None,
        use_amp=False,
        optimizer=None,
        scheduler=None,
        loss_fn=None,
        scheduler_step_on_epoch=False,
        validate_every_n=1,
        callbacks: List[BaseCallback] = None,
        loggers: Optional[BaseLogger] = None,
        verbose=False,
    ) -> None:
        super().__init__(
            model=model,
            benchmark=benchmark,
            method=method,
            gpu=gpu,
            use_amp=use_amp,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scheduler_step_on_epoch=scheduler_step_on_epoch,
            validate_every_n=validate_every_n,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.num = num
        self.temperature = temperature
        self.reg_coefficient = reg_coefficient
        self.verbose = verbose
        self.num_tasks = self.benchmark.num_tasks

    def calculate_and_backward_loss(self):
        self.losses = self.loss_fn(self.y_hat, self.y)
        self.loss, self.loss_extra_outputs = self.method.get_weighted_loss(
            losses=self.losses,
            task_weights=self.model.task_weights,
            shared_parameters=list(self.model.shared_parameters()),
            task_specific_parameters=list(self.model.task_specific_parameters()),
            last_shared_parameters=list(self.model.last_shared_parameters()),
            representation=self.features,
            # grad_scaler=self.grad_scaler,
        )

    def training_step(self):
        self.zero_grad_optimizer()

        NUM = self.num
        TEMPERATURE = self.temperature
        reg_coefficient = self.reg_coefficient
        losses = {i: {} for i in range(self.num_tasks)}
        total_loss = 0
        mask = []
        reg_term = 0
        for _ in range(NUM):
            self.on_before_forward()
            self.model.on_before_forward(self)
            self.on_before_forward_callbacks()
            self.y_hat = self.forward()
            self.on_after_forward()
            self.model.on_after_forward()
            self.on_after_forward_callbacks()

            self.calculate_and_backward_loss()
            total_loss += self.loss
            # print([round(a, 3) for a in self.model.alpha.tolist()], [round(l.detach().item(), 6) for l in self.losses])
            for i in range(self.num_tasks):
                losses[i][self.model.alpha[i].item()] = self.losses[i]

        losses = {k: dict(sorted(v.items(), reverse=True)) for k, v in losses.items()}

        if self.num > 1:
            for task_id in range(self.model.num_tasks):
                task_losses = list(losses[task_id].values())
                diff = torch.stack(task_losses).diff()
                _reg_term = torch.nn.functional.relu(diff).mul(TEMPERATURE).exp().sum().div(NUM - 1).log()
                reg_term += _reg_term
                mask.append(torch.nn.functional.relu(diff) > 0)

            if reg_term > 0.002:
                # print(reg_coefficient * reg_term < total_loss)
                wandb.log({"train/reg_loss": reg_term.item()})
                # logging.info(reg_term)
                if self.verbose:
                    _losses = {k: [round(vv.item(), 5) for vv in v] for k, v in losses.items()}
                    for i, (k, v) in enumerate(_losses.items()):
                        if i == 0:
                            m = ["-" for _ in range(len(_losses))]
                        else:
                            m = [mask[i][i - 1].item() for i in range(len(_losses))]
                        m = "\t".join(map(str, m))
                        logging.info(f"{round(k, 5)}:\t{v} {m}")
            desc = self.tqdm_dl.desc
            if " R=" in desc:
                desc = desc.split(" R=")[0]
            self.tqdm_dl.set_description_str(desc + f" R={reg_term.item():.3f}")
            if reg_coefficient * reg_term < total_loss:
                total_loss += reg_coefficient * reg_term

        total_loss.backward()

        self.on_before_optimizer_step()
        self.model.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.step_scheduler()
        self.on_after_optimizer_step()
        self.model.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()
