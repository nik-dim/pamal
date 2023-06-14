import logging

import torch
import torch.nn as nn
from src.trainer.base_trainer import BaseTrainer
from src.utils.metrics import CrossEntropyLossMetric, HuberLossMetric
from torchmetrics import Accuracy, JaccardIndex, MeanMetric, MetricCollection

from .callback import Callback


class MultiTaskMetricCallback(Callback):
    """Handles the computation and logging of metrics. Callback hooks after train/val/test steps/epochs etc. Inherits from Callback."""

    def __init__(
        self,
        metrics: MetricCollection,
        num_tasks,
        use_task_names=False,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.use_task_names = use_task_names
        self.orig_metrics = metrics
        self.logging_freq = 1

    def configure_task_names_and_ids(self, task_names, task_ids):
        task_ids = task_ids if task_ids is not None else list(range(self.num_tasks))
        if task_names is None:
            assert all([isinstance(t, int) for t in task_ids])
            task_names = [f"task-{task_ids[i]}" for i in range(self.num_tasks)]

        self.map_range_to_names = {i: task_names[t] for i, t in enumerate(task_ids)}
        return task_names, task_ids

    def setup_metrics(self, metrics):
        if isinstance(metrics, MetricCollection):
            # logging.info("All tasks use the same metrics!")
            metrics = {self.get_task_name(i): metrics for i in range(self.num_tasks)}

        self.train_metrics = nn.ModuleList(
            metrics[self.get_task_name(i)].clone(postfix=f"/{self.get_task_name(i)}") for i in range(self.num_tasks)
        )
        self.val_metrics = nn.ModuleList(
            metrics[self.get_task_name(i)].clone(postfix=f"/{self.get_task_name(i)}") for i in range(self.num_tasks)
        )
        self.test_metrics = nn.ModuleList(
            metrics[self.get_task_name(i)].clone(postfix=f"/{self.get_task_name(i)}") for i in range(self.num_tasks)
        )

        # logging.info(self.train_metrics)
        self.train_avg_loss = MeanMetric(compute_on_step=False)
        self.val_avg_loss = MeanMetric(compute_on_step=False)
        self.test_avg_loss = MeanMetric(compute_on_step=False)

    def get_task_name(self, task_id):
        return f"{self.task_names[task_id]}"

    def connect(self, trainer: BaseTrainer, *args, **kwargs):
        pass

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        super().connect(trainer, *args, **kwargs)
        if self.use_task_names:
            self.task_names = trainer.benchmark.task_names
        else:
            self.task_names = [f"task-{i}" for i in range(self.num_tasks)]
        self.setup_metrics(self.orig_metrics)
        self.move_to_device(trainer.device)

    def move_to_device(self, device):
        self.train_metrics = self.train_metrics.to(device)
        self.val_metrics = self.val_metrics.to(device)
        self.test_metrics = self.test_metrics.to(device)
        self.train_avg_loss = self.train_avg_loss.to(device)
        self.val_avg_loss = self.val_avg_loss.to(device)
        self.test_avg_loss = self.test_avg_loss.to(device)

    def _reset_metrics(self):
        # logging.info("Reseting metrics")
        self.train_avg_loss.reset()
        self.val_avg_loss.reset()
        self.test_avg_loss.reset()

        for m in self.train_metrics:
            m.reset()

        for m in self.val_metrics:
            m.reset()

        for m in self.test_metrics:
            m.reset()

    def log(self, trainer, key, value):
        trainer.log(key, value)

    def msg_process(self, msg):
        msg = {k: v for k, v in msg.items() if "loss/" not in k}
        return msg

    def update_metrics(self, metrics, trainer: BaseTrainer):
        for task_id, (yy_hat, y) in enumerate(zip(trainer.y_hat, trainer.y)):
            metrics[task_id](yy_hat, y)

    def compute_metrics(self, metrics, avg_metric):
        msg = dict()
        msg["avg_loss"] = avg_metric.compute().item()
        for task_id in range(len(metrics)):
            res = metrics[task_id].compute()
            res = {k: v.item() for k, v in res.items()}
            msg.update(res)
        return msg

    # ------- STEPS -------
    def on_after_training_step(self, trainer: BaseTrainer):
        self.train_avg_loss(trainer.loss)
        self.update_metrics(self.train_metrics, trainer)

        if trainer.batch_idx % self.logging_freq == 0:
            msg = self.compute_metrics(self.train_metrics, self.train_avg_loss)
            for k, v in msg.items():
                trainer.log(f"train/{k}", v)

            msg = self.msg_process(msg)
            trainer.update_tqdm(msg)

    def on_after_validation_step(self, trainer):
        self.val_avg_loss(trainer.loss)
        self.update_metrics(self.val_metrics, trainer)

        if trainer.batch_idx % self.logging_freq == 0:
            msg = self.compute_metrics(self.val_metrics, self.val_avg_loss)
            msg = self.msg_process(msg)
            trainer.update_tqdm(msg)

    def on_after_testing_step(self, trainer: BaseTrainer):
        self.test_avg_loss(trainer.loss)
        self.update_metrics(self.test_metrics, trainer)

        if trainer.batch_idx % self.logging_freq == 0:
            msg = self.compute_metrics(self.test_metrics, self.test_avg_loss)
            msg = self.msg_process(msg)
            trainer.update_tqdm(msg)

    # ------- EPOCHS - before -------
    def on_before_training_epoch(self, trainer):
        self._reset_metrics()

    def on_before_eval_epoch(self, trainer):
        self._reset_metrics()

    def on_before_testing_epoch(self, trainer):
        self._reset_metrics()

    # ------- EPOCHS - after -------
    # def on_after_training_epoch(self, trainer):

    def on_after_eval_epoch(self, trainer: BaseTrainer):
        results = self.compute_metrics(self.val_metrics, self.val_avg_loss)
        self.register_results_to_trainer(trainer, "val_metrics", results)
        for k, v in results.items():
            trainer.log(f"val/{k}", v)
        msg = self.msg_process(results)
        trainer.update_tqdm(msg)

    def on_after_testing_epoch(self, trainer: BaseTrainer):
        results = self.compute_metrics(self.test_metrics, self.test_avg_loss)
        self.register_results_to_trainer(trainer, "test_metrics", results)
        for k, v in results.items():
            trainer.log(f"test/{k}", v)
        msg = self.msg_process(results)
        trainer.update_tqdm(msg)

    def log_best_interpolation_results(self, trainer, prefix):
        res = {key: [m[key] for m in trainer.results.values()] for key in trainer.results[0].keys()}

        for key, values in res.items():
            if "loss" in key or "huber" in key:
                best = min(values)
            elif "acc" in key:
                best = max(values)
            else:
                raise NotImplementedError

            trainer.log(f"{prefix}/best/{key}", best)

    def on_after_validating_interpolations(self, trainer: BaseTrainer):
        logging.info("Logging best results out of interpolations for validation dataset.")
        self.log_best_interpolation_results(trainer, prefix="val")

    def on_after_predicting_interpolations(self, trainer: BaseTrainer):
        logging.info("Logging best results out of interpolations for test dataset.")
        self.log_best_interpolation_results(trainer, prefix="test")

    def register_results_to_trainer(self, trainer: BaseTrainer, results_name, results_dict):
        setattr(trainer, results_name, results_dict)


acc_metrics = MetricCollection(
    {
        "acc": Accuracy(compute_on_step=False),
        "loss": CrossEntropyLossMetric(compute_on_step=False),
    }
)


class ClassificationMultiTaskMetricCallback(MultiTaskMetricCallback):
    def __init__(self, num_tasks, use_task_names=True):
        super().__init__(metrics=acc_metrics, num_tasks=num_tasks, use_task_names=use_task_names)


class UTKFaceMultiTaskMetricCallback(MultiTaskMetricCallback):
    utk_face_metrics = {
        "age": MetricCollection({"huber": HuberLossMetric()}),
        "gender": acc_metrics,
        "race": acc_metrics,
    }

    def __init__(self):
        super().__init__(metrics=self.utk_face_metrics, num_tasks=3, use_task_names=True)
