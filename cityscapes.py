import logging
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.datasets.cityscapes2 import Cityscapes2DataModule
from src.models.factory.cosmos.upsampler import Upsampler
from src.models.factory.phn.phn_wrappers import HyperModel
from src.models.factory.rotograd import RotogradWrapper
from src.models.factory.segnet_cityscapes import (
    SegNet,
    SegNetDepthDecoder,
    SegNetMtan,
    SegNetSegmentationDecoder,
    SegNetSplitEncoder,
)
from src.utils import set_seed
from src.utils._selectors import get_ensemble_model, get_trainer
from src.utils.callbacks.auto_lambda_callback import AutoLambdaCallback
from src.utils.callbacks.cityscapes_metric_cb import CityscapesMetricCallback
from src.utils.callbacks.save_model import SaveModelCallback
from src.utils.logging_utils import initialize_wandb, install_logging
from src.utils.losses import CityscapesTwoTaskLoss


@hydra.main(config_path="configs/experiment/cityscapes", config_name="cityscapes")
def my_app(config: DictConfig) -> None:
    import warnings

    warnings.filterwarnings(
        "ignore", message="Note that order of the arguments: ceil_mode and return_indices will change"
    )
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)
    initialize_wandb(config)
    wandb.run.tags = ["372"]

    dm = Cityscapes2DataModule(
        **(dict() if config.data.root is None else dict(root=Path(config.data.root))),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        apply_augmentation=config.data.apply_augmentation,
    )
    logging.info(f"I am using the following benchmark {dm.name}")
    in_channels = 5 if config.method.name == "cosmos" else 3
    model = dict(segnetmtan=SegNetMtan(in_channels), segnet=SegNet(in_channels))[config.model.type]

    if config.method.name == "pamal":
        model = get_ensemble_model(model, dm.num_tasks, config)
    elif config.method.name == "cosmos":
        model = Upsampler(dm.num_tasks, model, input_dim=dm.input_dims)
    elif config.method.name == "rotograd":
        backbone = SegNetSplitEncoder(in_channels=in_channels, rotograd=True)
        head1, head2 = SegNetSegmentationDecoder(rotograd=True), SegNetDepthDecoder(rotograd=True)
        model = RotogradWrapper(backbone=backbone, heads=[head1, head2], latent_size=50)
    param_groups = model.parameters()

    logging.info(model)
    optimizer = torch.optim.Adam(param_groups, lr=config.optimizer.lr)
    if config.method.name == "rotograd":
        optimizer = torch.optim.Adam(
            [{"params": m.parameters()} for m in [backbone, head1, head2]]
            + [{"params": model.parameters(), "lr": config.optimizer.lr * 0.1}],
            lr=config.optimizer.lr,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler.step,
        gamma=config.scheduler.gamma,
    )
    logging.info(scheduler)
    logging.info(scheduler.__dict__)

    callbacks = [CityscapesMetricCallback(), SaveModelCallback()]
    if config.method.name == "autol":
        callbacks.append(AutoLambdaCallback(config.method.meta_lr))

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        loss_fn=CityscapesTwoTaskLoss(),
        gpu=0,
        scheduler=scheduler,
        scheduler_step_on_epoch=True,
        callbacks=callbacks,
    )

    trainer = get_trainer(config, trainer_kwargs, dm.num_tasks)
    trainer.fit(epochs=config.training.epochs)

    if config.method.name == "pamal":
        trainer.predict_interpolations(dm.test_dataloader())
    else:
        trainer.predict(test_loader=dm.test_dataloader())
    wandb.finish()


if __name__ == "__main__":
    my_app()


"""

"""
