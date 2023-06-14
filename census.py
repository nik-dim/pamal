import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.utils._selectors import get_callbacks, get_ensemble_model, get_optimizer, get_trainer
from src.datasets.census import TestCensusDataModule
from src.models.factory.cosmos.upsampler import Upsampler
from src.models.factory.mlp import MultiTaskMLP
from src.models.factory.phn.phn_wrappers import HyperModel
from src.utils import set_seed
from src.utils.callbacks.auto_lambda_callback import AutoLambdaCallback
from src.utils.logging_utils import install_logging, initialize_wandb
from src.models.factory.rotograd import RotogradWrapper
from src.utils.losses import MultiTaskCrossEntropyLoss


@hydra.main(config_path="configs/experiment/census", config_name="census")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    initialize_wandb(config)

    set_seed(config.seed)
    dm = TestCensusDataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        income=config.data.income,
        age=config.data.age,
        education=config.data.education,
        never_married=config.data.never_married,
    )
    logging.info(f"I am using the following benchmark {dm.name}")
    if config.method.name == "phn":
        model = HyperModel(dm.name)
    elif config.method.name == "cosmos":
        model = MultiTaskMLP(
            in_features=dm.num_features + 2,
            num_tasks=dm.num_tasks,
            encoder_specs=config.model.encoder_specs,
            decoder_specs=config.model.decoder_specs,
        )
    elif config.method.name == "rotograd":
        m = MultiTaskMLP(
            in_features=dm.num_features,
            num_tasks=dm.num_tasks,
            encoder_specs=config.model.encoder_specs,
            decoder_specs=config.model.decoder_specs,
        )
        backbone = m.encoder
        head1, head2 = m.decoders[0], m.decoders[1]
        model = RotogradWrapper(backbone=backbone, heads=[head1, head2], latent_size=256)
    else:
        model = MultiTaskMLP(
            in_features=dm.num_features,
            num_tasks=dm.num_tasks,
            encoder_specs=config.model.encoder_specs,
            decoder_specs=config.model.decoder_specs,
        )

    logging.info(model)

    if config.method.name == "pamal":
        model = get_ensemble_model(model, dm.num_tasks, config)
    elif config.method.name == "cosmos":
        model = Upsampler(dm.num_tasks, model, input_dim=dm.input_dims)
    param_groups = model.parameters()

    optimizer = get_optimizer(config, param_groups)
    if config.method.name == "rotograd":
        optimizer = torch.optim.Adam(
            [{"params": m.parameters()} for m in [backbone, head1, head2]]
            + [{"params": model.parameters(), "lr": config.optimizer.lr * 0.1}],
            lr=config.optimizer.lr,
        )
    callbacks = get_callbacks(config, dm.num_tasks)
    if config.method.name == "autol":
        callbacks.append(AutoLambdaCallback(config.method.meta_lr))

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=MultiTaskCrossEntropyLoss(),
    )

    trainer = get_trainer(config, trainer_kwargs, dm.num_tasks, model)
    trainer.fit(epochs=config.training.epochs)

    if config.method.name == "pamal":
        trainer.predict_interpolations(dm.test_dataloader())
    else:
        trainer.predict(test_loader=dm.test_dataloader())

    wandb.finish()


if __name__ == "__main__":
    my_app()
