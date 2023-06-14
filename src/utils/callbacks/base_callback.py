class BaseCallback:
    """
    Base class for callbacks.
    """

    def __init__(self):
        pass

    def on_before_setup(self, *args, **kwargs):
        pass

    def on_after_setup(self, *args, **kwargs):
        pass

    def on_before_teardown(self, *args, **kwargs):
        pass

    def on_after_teardown(self, *args, **kwargs):
        pass

    def on_before_fit(self, *args, **kwargs):
        pass

    def on_after_fit(self, *args, **kwargs):
        pass

    def on_before_training_epoch(self, *args, **kwargs):
        pass

    def on_after_training_epoch(self, *args, **kwargs):
        pass

    def on_before_eval_epoch(self, *args, **kwargs):
        pass

    def on_after_eval_epoch(self, *args, **kwargs):
        pass

    def on_before_testing_epoch(self, *args, **kwargs):
        pass

    def on_after_testing_epoch(self, *args, **kwargs):
        pass

    def on_before_training_step(self, *args, **kwargs):
        pass

    def on_after_training_step(self, *args, **kwargs):
        pass

    def on_before_backward(self, *args, **kwargs):
        pass

    def on_after_backward(self, *args, **kwargs):
        pass

    def on_before_forward(self, *args, **kwargs):
        pass

    def on_after_forward(self, *args, **kwargs):
        pass

    def on_before_optimizer_step(self, *args, **kwargs):
        pass

    def on_after_optimizer_step(self, *args, **kwargs):
        pass

    def on_before_validation_step(self, *args, **kwargs):
        pass

    def on_after_validation_step(self, *args, **kwargs):
        pass

    def on_before_testing_step(self, *args, **kwargs):
        pass

    def on_after_testing_step(self, *args, **kwargs):
        pass

    # ONLY USED BY INTERPOLATED MODELS
    def on_before_validating_interpolations(self, *args, **kwargs):
        pass

    def on_after_validating_interpolations(self, *args, **kwargs):
        pass

    def on_before_predicting_interpolations(self, *args, **kwargs):
        pass

    def on_after_predicting_interpolations(self, *args, **kwargs):
        pass
