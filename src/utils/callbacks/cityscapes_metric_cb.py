import copy
import logging

import pandas as pd
import torch
import wandb
from src.trainer.base_trainer import BaseTrainer
from torch import Tensor
from torchmetrics import Accuracy, JaccardIndex, MeanMetric, MetricCollection

from .callback import Callback


class ModifiedJaccardIndex(JaccardIndex):
    def update(self, preds: Tensor, target: Tensor) -> None:
        mask = target >= 0
        preds = preds[mask]
        target = target[mask]
        return super().update(preds, target)


class ModifiedAccuracy(Accuracy):
    def update(self, preds: Tensor, target: Tensor) -> None:
        mask = target >= 0
        preds = preds[mask]
        target = target[mask]
        return super().update(preds, target)


def get_metrics(device):
    _metrics = torch.nn.ModuleDict(
        {
            "sem": MetricCollection(
                {
                    "loss": MeanMetric(),
                    "iou": ModifiedJaccardIndex(num_classes=7).to(device),
                    "pix_acc": ModifiedAccuracy(num_classes=7).to(device),
                }
            ),
            "depth": MetricCollection(
                {
                    "loss": MeanMetric(),
                    "abs_err": MeanMetric(),
                    "rel_err": MeanMetric(),
                }
            ),
        }
    )
    return _metrics


class CityscapesMetricCallback(Callback):
    def __init__(self, use_amp=False, logging_freq=1, verbose=True):
        super().__init__()
        self.use_amp = use_amp
        self.logging_freq = logging_freq
        self.verbose = verbose

    def connect(self, trainer: BaseTrainer, *args, **kwargs):
        super().connect(trainer, *args, **kwargs)
        self.move_to_device(trainer.device)

    def depth_error(self, x_pred, x_output):
        with torch.cuda.amp.autocast(enabled=self.use_amp):

            device = x_pred.device
            binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
            x_pred_true = x_pred.masked_select(binary_mask)
            x_output_true = x_output.masked_select(binary_mask)
            abs_err = torch.abs(x_pred_true - x_output_true)
            rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
            a = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
            r = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
            return a, r

    def move_to_device(self, device):
        self.device = device

    def _reset_metrics(self):
        pass
        # self.train_metrics = copy.deepcopy(get_metrics(self.device))
        # self.val_metrics = copy.deepcopy(get_metrics(self.device))

    def log(self, trainer, key, value):
        trainer.log(key, value)

    def msg_process(self, msg):
        msg = {k: v for k, v in msg.items() if "loss/" not in k}
        if self.single_task_mode:
            msg = {k: v for k, v in msg.items() if "avg_loss" in k or self.only_task_name in k}
        return msg

    # ------- STEPS -------
    def on_after_training_step(self, trainer: BaseTrainer):
        # capture losses
        self.train_metrics["sem"]["loss"](trainer.losses[0].item())
        self.train_metrics["depth"]["loss"](trainer.losses[1].item())

        # semantic segmentation metrics
        y_hat_processed = trainer.y_hat[0].argmax(1).flatten()
        y_processed = trainer.y[0].long().flatten()
        self.train_metrics["sem"]["pix_acc"](y_hat_processed, y_processed)
        self.train_metrics["sem"]["iou"](y_hat_processed, y_processed)

        # depth metrics
        abs_err, rel_err = self.depth_error(trainer.y_hat[1], trainer.y[1])
        self.train_metrics["depth"]["abs_err"](abs_err)
        self.train_metrics["depth"]["rel_err"](rel_err)

        trainer.update_tqdm(
            {
                "S": f"{trainer.losses[0].item():.3f}",
                "D": f"{trainer.losses[1].item():.3f}",
                "avg": f"{trainer.loss.item():.3f}",
            }
        )

        _logs = {
            "loss/avg_loss": trainer.loss.item(),
            "loss/semantic": trainer.losses[0].item(),
            "loss/depth": trainer.losses[1].item(),
            "loss/depth1": abs_err,
            "loss/depth2": rel_err,
        }
        _logs = {f"train/{k}": v for k, v in _logs.items()}
        wandb.log(_logs)

    def on_after_validation_step(self, trainer):
        # capture losses
        self.val_metrics["sem"]["loss"](trainer.losses[0].item())
        self.val_metrics["depth"]["loss"](trainer.losses[1].item())

        # semantic segmentation metrics
        y_hat_processed = trainer.y_hat[0].argmax(1).flatten()
        y_processed = trainer.y[0].long().flatten()
        self.val_metrics["sem"]["pix_acc"](y_hat_processed, y_processed)
        self.val_metrics["sem"]["iou"](y_hat_processed, y_processed)

        # depth metrics
        abs_err, rel_err = self.depth_error(trainer.y_hat[1], trainer.y[1])
        self.val_metrics["depth"]["abs_err"](abs_err)
        self.val_metrics["depth"]["rel_err"](rel_err)

    def on_after_testing_step(self, trainer: BaseTrainer, *args, **kwargs):
        # capture losses
        self.test_metrics["sem"]["loss"](trainer.losses[0].item())
        self.test_metrics["depth"]["loss"](trainer.losses[1].item())

        # semantic segmentation metrics
        y_hat_processed = trainer.y_hat[0].argmax(1).flatten()
        y_processed = trainer.y[0].long().flatten()
        self.test_metrics["sem"]["pix_acc"](y_hat_processed, y_processed)
        self.test_metrics["sem"]["iou"](y_hat_processed, y_processed)

        # depth metrics
        abs_err, rel_err = self.depth_error(trainer.y_hat[1], trainer.y[1])
        self.test_metrics["depth"]["abs_err"](abs_err)
        self.test_metrics["depth"]["rel_err"](rel_err)

    # ------- EPOCHS - before -------
    def on_before_training_epoch(self, trainer):
        self.train_metrics = copy.deepcopy(get_metrics(self.device))
        # self._reset_metrics()

    def on_before_eval_epoch(self, trainer):
        self.val_metrics = copy.deepcopy(get_metrics(self.device))

    def on_before_testing_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        self.test_metrics = copy.deepcopy(get_metrics(self.device))

    @staticmethod
    def prepare_metrics(metrics):
        results = {k: v.compute() for k, v in metrics.items()}
        results = pd.json_normalize(results, sep="/").to_dict(orient="records")[0]
        results = {k: v.cpu().item() for k, v in results.items()}
        return results

    def on_after_eval_epoch(self, trainer: BaseTrainer):
        train_results = self.prepare_metrics(self.train_metrics)
        val_results = self.prepare_metrics(self.val_metrics)

        if self.verbose:
            logging.info(f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR ")
            logging.info(
                f"Epoch: {trainer.epoch:04d} | TRAIN: {train_results['sem/loss']:.4f} {train_results['sem/iou']:.4f} {train_results['sem/pix_acc']:.4f} | "
                f"{train_results['depth/loss']:.4f} {train_results['depth/abs_err']:.4f} {train_results['depth/rel_err']:.4f}"
            )
            logging.info(
                f"Epoch: {trainer.epoch:04d} | VAL/TEST: {val_results['sem/loss']:.4f} {val_results['sem/iou']:.4f} {val_results['sem/pix_acc']:.4f} | "
                f"{val_results['depth/loss']:.4f} {val_results['depth/abs_err']:.4f} {val_results['depth/rel_err']:.4f} "
            )

        self.register_results_to_trainer(trainer, "val_metrics", val_results)
        val_results = {f"val/{k}": v for k, v in val_results.items()}
        wandb.log(val_results)

    def on_after_testing_epoch(self, trainer: BaseTrainer, *args, **kwargs):
        test_results = self.prepare_metrics(self.test_metrics)

        if self.verbose:
            # logging.info(f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR ")

            logging.info(
                f"Epoch: {trainer.epoch:04d} | TEST: {test_results['sem/loss']:.4f} {test_results['sem/iou']:.4f} {test_results['sem/pix_acc']:.4f} | "
                f"{test_results['depth/loss']:.4f} {test_results['depth/abs_err']:.4f} {test_results['depth/rel_err']:.4f}"
            )
        self.register_results_to_trainer(trainer, "test_metrics", test_results)
        test_results = {f"test/{k}": v for k, v in test_results.items()}
        wandb.log(test_results)

    def log_best_interpolation_results(self, trainer, prefix):
        res = {key: [m[key] for m in trainer.results.values()] for key in trainer.results[0].keys()}

        for key, values in res.items():
            if "loss" in key or "depth" in key:
                best = min(values)
            elif "iou" in key or "pix_acc" in key:
                best = max(values)
            else:
                raise NotImplementedError

            trainer.log(f"{prefix}/best/{key}", best)

    def on_after_validating_interpolations(self, trainer: BaseTrainer):
        self.log_best_interpolation_results(trainer, prefix="val")

    def on_after_predicting_interpolations(self, trainer: BaseTrainer):
        self.log_best_interpolation_results(trainer, prefix="test")

    def register_results_to_trainer(self, trainer: BaseTrainer, results_name, results_dict):
        setattr(trainer, results_name, results_dict)
