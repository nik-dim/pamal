class BaseLogger:
    """The BaseLogger class. All loggers inherit this class. It provides functions for logging data in various formats, such as key-value pairs, figures, hyperparameters and more."""

    def __init__(self):
        pass

    def _build_experiment(self):
        raise NotImplementedError

    def log(self, key, value):
        raise NotImplementedError

    def log_parameters(self, params: dict, prefix=None):
        raise NotImplementedError

    def log_metric(self, key: str, value, step=None, prefix=None):
        raise NotImplementedError

    def log_figure(self, figure, name, step=None, prefix=None):
        raise NotImplementedError

    def log_text(self, text, prefix=None):
        raise NotImplementedError

    def log_folder(self, folder_path, prefix=None):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError
