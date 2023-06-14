import logging
import os

import numpy as np
import torch
import wandb

from src.trainer.base_trainer import BaseTrainer

from .callback import Callback


class SaveModelCallback(Callback):
    def __init__(self, wandb=False, inputs=-1):
        super().__init__()
        self.wandb = wandb
        self.filename = "model.pth"
        self.inputs = inputs
        assert isinstance(self.inputs, int)
        if self.inputs > 10:
            logging.info("Will save 10 images and not more.")
            self.inputs = 10

    def on_before_fit(self, trainer: BaseTrainer, *args, **kwargs):
        assert isinstance(self.inputs, int)
        if self.inputs > 0 and self.wandb:
            train_dataset = trainer.train_loader.dataset
            random_indices = np.random.choice(len(train_dataset), size=self.inputs, replace=False).tolist()
            for index in random_indices:
                img = train_dataset[index][0]
                # logging.info(img)
                # logging.info(img.dim())
                if img.dim() == 1:
                    # not an image
                    return
                img = wandb.Image(img)
                wandb.log({f"inputs/{index}": img})

    def on_after_fit(self, trainer: BaseTrainer, *args, **kwargs):
        torch.save(
            f=self.filename,
            obj={
                "epoch": trainer.current_epoch,
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                # "config": trainer.config,
                # "results_dict": trainer.results_aggregator.results,
            },
        )
        logging.info(f"Saved model as {os.getcwd()}/{self.filename}")
        if self.wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(self.filename)
            wandb.run.log_artifact(artifact)
            logging.info("Saved model to Weights&Biases!")
