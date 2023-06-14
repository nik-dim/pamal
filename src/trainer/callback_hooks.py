from abc import ABC
from src.utils.callbacks.base_callback import BaseCallback
from typing import List


class TrainerCallbackHookMixin(ABC):
    """Generic class that provides the hooks for all callbacks."""

    callbacks: List[BaseCallback] = []

    def on_before_setup_callbacks(self):
        """Callbacks before the setup."""
        for cb in self.callbacks:
            cb.on_before_setup(self)

    def on_after_setup_callbacks(self):
        """Callbacks after the setup."""
        for cb in self.callbacks:
            cb.on_after_setup(self)

    def on_before_teardown_callbacks(self):
        """Callbacks before the teardown."""
        for cb in self.callbacks:
            cb.on_before_teardown(self)

    def on_after_teardown_callbacks(self):
        """Callbacks after the teardown."""
        for cb in self.callbacks:
            cb.on_after_teardown(self)

    def on_before_fit_callbacks(self):
        """Callbacks before fitting the data."""
        for cb in self.callbacks:
            cb.on_before_fit(self)

    def on_after_fit_callbacks(self):
        """Callbacks after fitting the data."""
        for cb in self.callbacks:
            cb.on_after_fit(self)

    def on_before_training_epoch_callbacks(self):
        """Callbacks before training one epoch."""
        for cb in self.callbacks:
            cb.on_before_training_epoch(self)

    def on_after_training_epoch_callbacks(self):
        """Callbacks after training one epoch."""
        for cb in self.callbacks:
            cb.on_after_training_epoch(self)

    def on_before_eval_epoch_callbacks(self):
        """Callbacks before the validation epoch."""
        for cb in self.callbacks:
            cb.on_before_eval_epoch(self)

    def on_after_eval_epoch_callbacks(self):
        """Callbacks after the validation epoch."""
        for cb in self.callbacks:
            cb.on_after_eval_epoch(self)

    def on_before_testing_epoch_callbacks(self):
        """Callbacks before the testing epoch."""
        for cb in self.callbacks:
            cb.on_before_testing_epoch(self)

    def on_after_testing_epoch_callbacks(self):
        """Callbacks after the testing epoch."""
        for cb in self.callbacks:
            cb.on_after_testing_epoch(self)

    def on_before_training_step_callbacks(self):
        """Callbacks before the training step (single batch step)."""
        for cb in self.callbacks:
            cb.on_before_training_step(self)

    def on_after_training_step_callbacks(self):
        """Callbacks after the training step."""
        for cb in self.callbacks:
            cb.on_after_training_step(self)

    def on_before_backward_callbacks(self):
        """Callbacks before backpropagation."""
        for cb in self.callbacks:
            cb.on_before_backward(self)

    def on_after_backward_callbacks(self):
        """Callbacks after backpropagation."""
        for cb in self.callbacks:
            cb.on_after_backward(self)

    def on_before_forward_callbacks(self):
        """Callbacks before the forward pass."""
        for cb in self.callbacks:
            cb.on_before_forward(self)

    def on_after_forward_callbacks(self):
        """Callbacks after the forward pass."""
        for cb in self.callbacks:
            cb.on_after_forward(self)

    def on_before_optimizer_step_callbacks(self):
        """Callbacks before the optimizer step (weight update)."""
        for cb in self.callbacks:
            cb.on_before_optimizer_step(self)

    def on_after_optimizer_step_callbacks(self):
        """Callbacks after the optimizer step (weight update).."""
        for cb in self.callbacks:
            cb.on_after_optimizer_step(self)

    def on_before_validation_step_callbacks(self):
        """Callbacks before the validation step."""
        for cb in self.callbacks:
            cb.on_before_validation_step(self)

    def on_after_validation_step_callbacks(self):
        """Callbacks after the validation step."""
        for cb in self.callbacks:
            cb.on_after_validation_step(self)

    def on_before_testing_step_callbacks(self):
        """Callbacks before the testing step."""
        for cb in self.callbacks:
            cb.on_before_testing_step(self)

    def on_after_testing_step_callbacks(self):
        """Callbacks after the testing step."""
        for cb in self.callbacks:
            cb.on_after_testing_step(self)

    # ONLY USED BY INTERPOLATED MODELS
    def on_before_validating_interpolations_callbacks(self):
        for cb in self.callbacks:
            cb.on_before_validating_interpolations(self)

    def on_after_validating_interpolations_callbacks(self):
        for cb in self.callbacks:
            cb.on_after_validating_interpolations(self)

    def on_before_predicting_interpolations_callbacks(self):
        for cb in self.callbacks:
            cb.on_before_predicting_interpolations(self)

    def on_after_predicting_interpolations_callbacks(self):
        for cb in self.callbacks:
            cb.on_after_predicting_interpolations(self)
