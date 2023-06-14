from .base_callback import BaseCallback
from src.trainer.base_trainer import BaseTrainer


class Callback(BaseCallback):
    def connect(self, trainer: BaseTrainer, *args, **kwargs):
        self.trainer = trainer

    def on_before_setup(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_setup(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_teardown(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_teardown(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_fit(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_training_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_eval_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_testing_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_testing_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_training_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_backward(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_backward(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_forward(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_forward(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_optimizer_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_optimizer_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_validation_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_validation_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_testing_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_after_testing_step(self, trainer: BaseTrainer, *args, **kwargs):
        pass
