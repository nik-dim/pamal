import logging

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.datasets.multimnist3digits import MultiMnistThreeDataModule
from src.models.base_model import SharedBottom
from src.models.factory.cosmos.upsampler import Upsampler
from src.models.factory.lenet import MultiLeNetO, MultiLeNetR
from src.models.factory.phn.phn_wrappers import HyperModel
from src.models.factory.rotograd import RotogradWrapper
from src.utils import set_seed
from src.utils._selectors import get_callbacks, get_ensemble_model, get_optimizer, get_trainer
from src.utils.callbacks.auto_lambda_callback import AutoLambdaCallback
from src.utils.logging_utils import initialize_wandb, install_logging
from src.utils.losses import MultiTaskCrossEntropyLoss


@hydra.main(config_path="configs/experiment/multimnist3", config_name="multimnist3")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)

    initialize_wandb(config)

    dm = MultiMnistThreeDataModule(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    logging.info(f"I am using the following benchmark {dm.name}")
    if config.method.name == "phn":
        model = HyperModel(dm.name)
    elif config.method.name == "cosmos":
        model = SharedBottom(encoder=MultiLeNetR(in_channels=4), decoder=MultiLeNetO(), num_tasks=3)
    elif config.method.name == "rotograd":
        backbone = MultiLeNetR(in_channels=1)
        head1, head2, head3 = MultiLeNetO(), MultiLeNetO(), MultiLeNetO()
        model = RotogradWrapper(backbone=backbone, heads=[head1, head2, head3], latent_size=50)
    else:
        model = SharedBottom(
            encoder=MultiLeNetR(in_channels=1), decoder=[MultiLeNetO(), MultiLeNetO(), MultiLeNetO()], num_tasks=3
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
