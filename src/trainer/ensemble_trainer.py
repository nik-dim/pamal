import pickle
from typing import List, Optional
from tqdm import tqdm
import logging
import torch
import wandb
from src.ll.weight_ensemble import BaseWeightEnsemble
from src.datasets.base_data_module import BaseDataModule
from src.methods.weight_methods import WeightMethod
from src.utils.callbacks.base_callback import BaseCallback
from src.utils.loggers.base_logger import BaseLogger
from .base_trainer import BaseTrainer
import torch.nn.functional as F


class ResultAggregator:
    def __init__(self) -> None:
        self.results = {}

    def update(self, results, key):
        self.results[key] = results

    def log_to_wandb(self):
        output = open("results.pkl", "wb")
        pickle.dump(self.results, output)
        output.close()
        artifact = wandb.Artifact(name="results", type="results")
        artifact.add_file("results.pkl")
        wandb.log_artifact(artifact)


class EnsembleTrainer(BaseTrainer):
    model: BaseWeightEnsemble

    def __init__(
        self,
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
            callbacks=callbacks,
            loggers=loggers,
        )
        self.setup_task_ids_and_names()
        self.results_aggregator = ResultAggregator()
        self.validate_every_n = validate_every_n

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

        self.loss.backward()

    def setup_task_ids_and_names(self):
        self.task_names = self.benchmark.task_names

    def _check_validation_condition(self):
        return self.epoch == self.epochs or self.epoch % self.validate_every_n == 0

    def _fit(self):
        for self.epoch in range(1, self.epochs + 1):
            self._set_train()
            self._train_loop()
            if self.val_loader is not None and self._check_validation_condition():
                self._set_val()
                self.on_before_validating_interpolations()
                self.on_before_validating_interpolations_callbacks()
                self.results = self._validate_interpolations()
                self.on_after_validating_interpolations()
                self.on_after_validating_interpolations_callbacks()

    def _validate_interpolations(self):
        self.model.reset_counter()
        results = {}
        iterator = tqdm(range(len(self.model.points)), disable=(self.model.num_members == 1))
        for i, _ in enumerate(iterator):
            index, alpha = self.model.next()
            super()._val_loop(leave_tqdm=False)
            results[index] = self.val_metrics

        if self.model.num_members > 1:
            self.compute_members_cosine_similarities()

        for k, v in results.items():
            _res = {kk: round(vv, 4) for kk, vv in v.items()}
            logging.info(f"{k}: {_res}")

        return results

    def on_after_validating_interpolations(self, *args, **kwargs):
        self.log_interpolations(prefix="val")

    def log_interpolations(self, prefix):
        for key, val in self.results.items():
            if "avg_loss" in val:
                val.pop("avg_loss")
            for metric_name, metric_value in val.items():
                wandb.log({f"interpolations/{prefix}/{metric_name}-{round(key,2)}": metric_value})

    def predict_interpolations(self, dataloader):
        self._set_test()
        self.on_before_predicting_interpolations()
        self.on_before_predicting_interpolations_callbacks()
        self.results = self._predict_interpolations(dataloader)
        self.on_after_predicting_interpolations()
        self.on_after_predicting_interpolations_callbacks()

    def _predict_interpolations(self, dataloader):
        self.model.reset_counter()
        results = {}
        iterator = tqdm(range(len(self.model.points)), disable=(self.model.num_members == 1) or True)
        for i, _ in enumerate(iterator):
            index, alpha = self.model.next()
            results[index] = super().predict(dataloader, leave_tqdm=False)

        for k, v in results.items():
            _res = {kk: round(vv, 4) for kk, vv in v.items()}
            logging.info(f"TEST -- {k}: {_res}")

        return results

    def on_after_predicting_interpolations(self, *args, **kwargs):
        self.log_interpolations(prefix="test")

    # def predict_interpolation(self, dataloader, prefix):
    #     self.model.reset_counter()
    #     results = {}
    #     iterator = tqdm(range(len(self.model.points)), disable=(self.model.num_members == 1))
    #     for i, _ in enumerate(iterator):
    #         index, alpha = self.model.next()
    #         results[index] = self.predict(dataloader, leave_tqdm=False)

    #     for k, v in results.items():
    #         _res = {kk: round(vv, 4) for kk, vv in v.items()}
    #         logging.info(f"{k}: {_res}")

    #     self.log_interpolations(prefix="test")
    #     return results

    def compute_members_cosine_similarities_2_models(self, i=0, j=1):
        if i > j:
            temp = i
            i = j
            j = temp
        weights = {i: [] for i in range(self.model.num_members)}
        for k, v in self.model.named_parameters():
            if "members." in k:
                member_id = int(k.split("members.")[1].split(".")[0])
                weights[member_id].append(v.flatten())

        weights = {i: torch.cat(w) for i, w in weights.items()}
        value = F.cosine_similarity(weights[i], weights[j], dim=0)
        wandb.log({f"similarities/cosine_{i}-{j}": value})
        return value

    def compute_members_cosine_similarities(self):
        weights = {i: [] for i in range(self.model.num_members)}
        for k, v in self.model.named_parameters():
            if "members." in k:
                member_id = int(k.split("members.")[1].split(".")[0])
                weights[member_id].append(v.detach().flatten())

        weights = {i: torch.cat(w) for i, w in weights.items()}
        for i in range(self.model.num_members):
            for j in range(i + 1, self.model.num_members):
                value = F.cosine_similarity(weights[i], weights[j], dim=0).item()
                wandb.log({f"similarities/cosine_{i}-{j}": value})

    def on_after_fit(self, *args, **kwargs):
        self.results_aggregator.log_to_wandb()

    def on_after_training_epoch(self, *args, **kwargs):
        cosine_similarity = self.compute_members_cosine_similarities_2_models().detach().cpu().item()
        logging.info(f"Cosine similaity of models is {cosine_similarity:.6f}")
