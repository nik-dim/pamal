import logging
import sys
from typing import List, Optional, Tuple
import torch
import wandb
from src.datasets.base_data_module import BaseDataModule
from src.methods.weight_methods import WeightMethod
from src.utils.callbacks.base_callback import BaseCallback
from src.utils.logging_utils import install_logging
from tqdm import tqdm
from src.models.base_model import BaseModel
from src.models.factory.rotograd import RotogradWrapper

from .callback_hooks import TrainerCallbackHookMixin
from .state_manager import TrainerStateManagerMixin
from src.utils.loggers.base_logger import BaseLogger


class BaseTrainer(TrainerStateManagerMixin, TrainerCallbackHookMixin, BaseCallback):
    def __init__(
        self,
        model: torch.nn.Module,
        benchmark: BaseDataModule,
        method: WeightMethod,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[BaseCallback]] = None,
        loggers: Optional[BaseLogger] = None,
        gpu=None,
        use_amp: bool = False,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_step_on_epoch: Optional[bool] = None,
    ) -> None:
        self.install_logging()
        self.device = "cpu" if gpu is None else f"cuda:{gpu}"

        self.model: BaseModel = model.to(self.device)

        self.method = method
        if self.method is not None:
            # method is None in case of ensemble training
            self.method.connect_device(self.device)

        self.benchmark = benchmark

        self.callbacks = callbacks
        self.loggers = loggers
        self.setup_callbacks()

        self.optimizer = optimizer
        self.scheduler, self.scheduler_step_on_epoch = scheduler, scheduler_step_on_epoch
        if scheduler is not None:
            assert (
                scheduler_step_on_epoch is not None
            ), "scheduler_step_on_epoch must be provided if scheduler is not None"

        self.use_amp = use_amp
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.loss_fn = loss_fn

        logging.info(f"Running on {self.device}")
        logging.info(f"The model has {self.count_parameters()/1e6:.3f}m parameters")
        logging.info(method)
        logging.info(f"torch.cuda.autocast set to {self.use_amp}")

    def install_logging(self):
        install_logging()

    def count_parameters(self):
        return sum([p.numel() for p in self.model.parameters() if p.requires_grad])

    def log(self, key: str, value):
        # for logger in self.loggers:
        #     logger.log(key, value)
        wandb.log({key: value})

    def setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = []
        for cb in self.callbacks:
            cb.connect(self)

    def setup(self):
        pass

    #     self.loss_fn = self.configure_loss_fn()
    # self.configure_logger()

    def _parse_config(self, config):
        self.config = config
        self.logging_freq = config.logging.freq
        self.epochs = config.training.epochs
        self.num_tasks = config.data.num_tasks

    def forward(self):
        """Calls the forward function of the model"""
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output, self.features = self.model(self.x, return_embedding=True)

        return output

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], (list, tuple, dict)):
            y = batch[1]
        else:
            y = batch[1:]
        self.x = x.to(self.device)

        if isinstance(y, torch.Tensor):
            self.y = y.to(self.device)
        elif isinstance(y, tuple) or isinstance(y, list):
            self.y = tuple(yy.to(self.device) for yy in y)
        elif isinstance(y, dict):
            self.y = {k: v.to(self.device) for k, v in y.items()}
        else:
            raise NotImplementedError

    def compute_loss(self):
        losses = self.loss_fn(self.y_hat, self.y)

        loss = sum(losses) / len(losses)
        return loss, losses

    def zero_grad_optimizer(self):
        self.optimizer.zero_grad()

    def step_optimizer(self):
        self.optimizer.step()

    def step_scheduler(self, EPOCH_FINISHED=False):
        if self.scheduler is not None:
            if EPOCH_FINISHED and self.scheduler_step_on_epoch:
                old_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                new_lr = self.scheduler.get_last_lr()
                if new_lr != old_lr:
                    logging.info(f"The LR changed from {old_lr} to {new_lr}.")
            elif not EPOCH_FINISHED and not self.scheduler_step_on_epoch:
                self.scheduler.step()
                new_lr = self.scheduler.get_last_lr()
                desc = self.tqdm_dl.desc.split(",")[0]
                self.tqdm_dl.set_description(f"{desc}, lr={new_lr[0]:.4f}")

            else:
                pass

    def calculate_and_backward_loss(self):
        self.losses = self.loss_fn(self.y_hat, self.y)
        # TODO: unify the API below.
        if isinstance(self.model, RotogradWrapper):
            self.model.backward(self.losses)
            self.loss = sum(self.losses) / len(self.losses)
        elif getattr(self, "meta_weights", None) is not None:
            # Auto-lamdba case
            self.loss = sum([w * l for w, l in zip(self.meta_weights, self.losses)])
            self.loss.backward()
            # print(self.losses, self.meta_weights, self.loss)
        else:
            self.losses = torch.stack(self.losses)
            self.loss, self.loss_extra_outputs = self.method.backward(
                losses=self.losses,
                shared_parameters=list(self.model.shared_parameters()),
                task_specific_parameters=list(self.model.task_specific_parameters()),
                last_shared_parameters=list(self.model.last_shared_parameters()),
                representation=self.features,
                grad_scaler=self.grad_scaler,
            )

    def training_step(self):
        """The training step, i.e. training for each batch. Goes through the usual hoops of zeroing out the optimizer,
        forwarding the input, computing the loss, backpropagating and updating the weights. For each different steps,
        callbacks are offered for maximum versatility and ease of use."""
        self.zero_grad_optimizer()

        self.on_before_forward()
        self.model.on_before_forward(self)
        self.on_before_forward_callbacks()
        self.y_hat = self.forward()
        self.on_after_forward()
        self.model.on_after_forward()
        self.on_after_forward_callbacks()

        # ----------------
        # TODO: The bulk of the code has been moved to calculate_and_backward_loss to be compatible with the weight
        # methods retrieved from https://github.com/AvivNavon/nash-mtl. This renders all the code below as well as the
        # corresponding callbacks obsolete. It would be code to reinstate the commented out code and move the weight
        # method logic to dedicated callbacks.
        # ----------------

        # self.loss, self.losses = self.compute_loss()

        # self.on_before_backward()
        # self.model.on_before_backward()
        # self.on_before_backward_callbacks()
        # self.grad_scaler.scale(self.loss).backward()
        # self.on_after_backward()
        # self.model.on_after_backward()
        # self.on_after_backward_callbacks()
        self.calculate_and_backward_loss()
        # self.losses = self.loss_fn(self.y_hat, self.y)
        # self.losses = torch.stack(self.losses)
        # self.loss, self.loss_extra_outputs = self.method.backward(
        #     losses=self.losses,
        #     shared_parameters=list(self.model.shared_parameters()),
        #     task_specific_parameters=list(self.model.task_specific_parameters()),
        #     last_shared_parameters=list(self.model.last_shared_parameters()),
        #     representation=self.features,
        #     grad_scaler=self.grad_scaler,
        # )

        self.on_before_optimizer_step()
        self.model.on_before_optimizer_step()
        self.on_before_optimizer_step_callbacks()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.step_scheduler()
        self.on_after_optimizer_step()
        self.model.on_after_optimizer_step()
        self.on_after_optimizer_step_callbacks()

    def validation_step(self):
        """Performs the validation step. Callbacks are offered for each step of the process."""
        with torch.no_grad():
            self.on_before_forward()
            self.model.on_before_forward()
            self.on_before_forward_callbacks()
            self.y_hat = self.forward()
            self.on_after_forward()
            self.model.on_after_forward()
            self.on_after_forward_callbacks()
            self.loss, self.losses = self.compute_loss()

    def testing_step(self):
        self.validation_step()

    def _tqdm(self, dataloader, mode="train", leave=True):
        if mode == "train":
            msg = f"Epoch {self.epoch}"
        elif mode == "eval":
            msg = f"Validating epoch {self.epoch}"
        elif mode == "test":
            msg = f"TESTING"
        return tqdm(
            dataloader,
            desc=msg,
            file=sys.stdout,  # so that tqdm does not print out of order,
            leave=leave,
            # disable=True,
        )

    def update_tqdm(self, msg):
        self.tqdm_dl.set_postfix(msg)

    def train_epoch(self):
        """Trains the model for a single epoch. Callbacks are offered for each method."""
        self.model.train()
        self.tqdm_dl = self._tqdm(self.train_loader, mode="train")
        for self.batch_idx, batch in enumerate(self.tqdm_dl):
            self._unpack_batch(batch)
            self.model.on_before_training_step()
            self.on_before_training_step()
            self.on_before_training_step_callbacks()
            self._tick_step()
            self.training_step()
            self.model.on_after_training_step()
            self.on_after_training_step()
            self.on_after_training_step_callbacks()

        self.step_scheduler(EPOCH_FINISHED=True)

    def eval_epoch(self, leave_tqdm=True):
        """Performs the evaluation of the model on the validation set. If no validation dataloader is provided, the
        method returns without any computation."""
        self.model.eval()
        if self.val_loader is None:
            return

        self.tqdm_dl = self._tqdm(self.val_loader, mode="eval", leave=leave_tqdm)
        for self.batch_idx, batch in enumerate(self.tqdm_dl):
            self._unpack_batch(batch)
            self.on_before_validation_step()
            self.model.on_before_validation_step()
            self.on_before_validation_step_callbacks()
            self._tick_step()
            self.validation_step()
            self.model.on_after_validation_step()
            self.on_after_validation_step()
            self.on_after_validation_step_callbacks()

    def test_epoch(self, leave_tqdm=True):
        """Performs the evaluation of the model on the validation set."""
        self._set_test()
        self.model.eval()
        self.tqdm_dl = self._tqdm(self.test_loader, mode="test", leave=leave_tqdm)
        for self.batch_idx, batch in enumerate(self.tqdm_dl):
            self._unpack_batch(batch)
            self.on_before_testing_step()
            self.model.on_before_testing_step()
            self.on_before_testing_step_callbacks()
            self._tick_step()
            self.testing_step()
            self.on_after_testing_step()
            self.model.on_after_testing_step()
            self.on_after_testing_step_callbacks()

    def predict(self, test_loader, leave_tqdm=True):
        """Performs the evaluation of the provided test dataloader.

        Args:
            test_dataloader (DataLoader): the dataloader to be evaluated.
        """
        if test_loader is None:
            # some datasets (e.g. Cityscapes) do not have predefined test datasets.
            return
        self.test_loader = test_loader
        self._set_test()
        self.on_before_testing_epoch()
        self.model.on_before_testing_epoch()
        self.on_before_testing_epoch_callbacks()
        self.test_epoch(leave_tqdm=leave_tqdm)
        self.on_after_testing_epoch()
        self.model.on_after_testing_epoch()
        self.on_after_testing_epoch_callbacks()

        return self.test_metrics

    def _train_loop(self):
        self.on_before_training_epoch()
        self.model.on_before_training_epoch()
        self.on_before_training_epoch_callbacks()
        self._tick_epoch()
        self.log("epoch", self.current_epoch)
        self.train_epoch()
        self.on_after_training_epoch()
        self.model.on_after_training_epoch()
        self.on_after_training_epoch_callbacks()

    def _val_loop(self, leave_tqdm=True):
        self.on_before_eval_epoch()
        self.model.on_before_eval_epoch()
        self.on_before_eval_epoch_callbacks()
        self.eval_epoch(leave_tqdm)
        self.on_after_eval_epoch()
        self.model.on_after_eval_epoch()
        self.on_after_eval_epoch_callbacks()

        return self.val_metrics

    def _fit(self):
        for self.epoch in range(1, self.epochs + 1):
            self._set_train()
            self._train_loop()
            if self.val_loader is not None:
                self._set_val()
                self._val_loop()

    def fit(self, epochs):
        """The fit method of the Trainer."""
        self.epochs = epochs
        self.train_loader = self.benchmark.train_dataloader()
        self.val_loader = self.benchmark.val_dataloader()
        self.setup()

        self.on_before_fit()
        self.model.on_before_fit()
        self.on_before_fit_callbacks()
        self._fit()
        self.on_after_fit()
        self.model.on_after_fit()
        self.on_after_fit_callbacks()

    #     self.teardown()

    # def teardown(self):
    #     for logger in self.loggers:
    #         logger.terminate()
