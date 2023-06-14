# adapted from https://github.com/lorenmt/auto-lambda
import copy
import logging

import torch
from torch.optim import Adam

from src.trainer.base_trainer import BaseTrainer
from src.utils.callbacks.callback import Callback


class AutoLambdaCallback(Callback):
    def __init__(self, meta_lr, weight_init=0.1):
        logging.info("Initializing AutoLambdaCallback.")
        self.weight_init = weight_init
        self.meta_lr = meta_lr

    def connect(self, trainer: BaseTrainer, *args, **kwargs):
        self.num_tasks = trainer.benchmark.num_tasks
        self.device = trainer.device
        self.model = trainer.model
        self.model_ = copy.deepcopy(trainer.model)
        self.meta_weights = torch.tensor([self.weight_init] * self.num_tasks, requires_grad=True, device=self.device)

        self.meta_optimizer = Adam([self.meta_weights], lr=self.meta_lr)

    def get_lr(self, trainer: BaseTrainer):
        if trainer.scheduler is None:
            return trainer.optimizer.param_groups[0]["lr"]
        else:
            return trainer.scheduler.get_last_lr()[0]

    def virtual_step(self, trainer, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        if type(train_x) == list:  # multi-domain setting [many-to-many]
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:  # single-domain setting [one-to-many]
            train_pred = self.model(train_x)

        train_loss = self.model_fit(trainer, train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if "momentum" in model_optim.param_groups[0].keys():  # used in SGD with momentum
                    m = model_optim.state[weight].get("momentum_buffer", 0.0) * model_optim.param_groups[0]["momentum"]
                else:
                    m = 0
                weight_.copy_(weight - alpha * (m + grad + model_optim.param_groups[0]["weight_decay"] * weight))

    def unrolled_backward(self, trainer: BaseTrainer, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(trainer, train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        # TODO: no primary tasks atm.
        # pri_weights = [1,]
        # for t in self.train_tasks:
        #     if t in self.pri_tasks:
        #         pri_weights += [1.0]
        #     else:
        #         pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(trainer, val_pred, val_y)
        loss = sum(val_loss)

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(trainer, d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = -alpha * h

    def compute_hessian(self, trainer, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(trainer, train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(trainer, train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2.0 * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian

    def model_fit(self, trainer: BaseTrainer, pred, targets):
        return trainer.loss_fn(pred, targets)

    def on_before_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        logging.info(f"Meta-weights: {self.meta_weights}")
        logging.info("Setting meta dataloader")
        self.meta_dataloader = iter(trainer.benchmark.train_dataloader())

    def on_before_forward(self, trainer: BaseTrainer, *args, **kwargs):
        if trainer.model.training:
            batch = next(self.meta_dataloader)
            if len(batch) == 3:
                # TODO: hack for cityscapes. the api is not consistent
                x_meta, y_meta = batch[0], batch[1:]
            else:
                x_meta, y_meta = batch[0], batch[1]
            x_meta = x_meta.to(self.device)
            y_meta = [yy.to(self.device) for yy in y_meta]

            # update meta-weights with Auto-Lambda
            last_lr = self.get_lr(trainer)
            self.meta_optimizer.zero_grad()
            x, y = trainer.x, trainer.y
            self.unrolled_backward(trainer, x, y, x_meta, y_meta, last_lr, trainer.optimizer)
            self.meta_optimizer.step()

            # register meta-weights to trainer
            trainer.meta_weights = self.meta_weights

            # log meta-weights per task at each iteration
            meta_weights = self.meta_weights.detach().cpu().numpy()
            for t, w in enumerate(meta_weights):
                trainer.log(key=f"meta-weights/task-{t}", value=w)
