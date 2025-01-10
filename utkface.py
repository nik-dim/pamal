import logging
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.datasets.utkface import UTKFaceDataModule
from src.models.factory.cosmos.upsampler import Upsampler
from src.models.factory.phn.phn_wrappers import HyperModel
from src.models.factory.resnet import BasicBlock, MLPDecoder, ResNetEncoder, UtkFaceResnet
from src.models.factory.rotograd import RotogradWrapper
from src.utils import set_seed
from src.utils._selectors import get_ensemble_model, get_optimizer, get_trainer
from src.utils.callbacks.auto_lambda_callback import AutoLambdaCallback
from src.utils.callbacks.mtl_metric_callback import UTKFaceMultiTaskMetricCallback
from src.utils.callbacks.save_model import SaveModelCallback
from src.utils.logging_utils import initialize_wandb, install_logging
from src.utils.losses import UTKFaceMultiTaskLoss


@hydra.main(config_path="configs/experiment/utkface", config_name="utkface")
def my_app(config: DictConfig) -> None:
    install_logging()
    logging.info(OmegaConf.to_yaml(config))
    set_seed(config.seed)
    initialize_wandb(config)

    dm = UTKFaceDataModule(
        **(dict() if config.data.root is None else dict(root=config.data.root)),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )
    logging.info(f"I am using the following benchmark {dm.name}")
    # model = UtkFaceResnet()

    if config.method.name == "phn":
        model = HyperModel(dm.name)
    elif config.method.name == "cosmos":
        model = UtkFaceResnet(in_channels=6)
    elif config.method.name == "rotograd":
        backbone = ResNetEncoder(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_channels=3)
        head1, head2, head3 = MLPDecoder(num_classes=1), MLPDecoder(num_classes=2), MLPDecoder(num_classes=5)
        model = RotogradWrapper(backbone=backbone, heads=[head1, head2, head3], latent_size=500)
    else:
        model = UtkFaceResnet()

    if config.method.name == "pamal":
        model = get_ensemble_model(model, dm.num_tasks, config)
    elif config.method.name == "cosmos":
        model = Upsampler(dm.num_tasks, model, input_dim=dm.input_dims)
    param_groups = model.parameters()

    optimizer = get_optimizer(config, param_groups)

    callbacks = [UTKFaceMultiTaskMetricCallback(), SaveModelCallback()]
    if config.method.name == "autol":
        callbacks.append(AutoLambdaCallback(config.method.meta_lr))

    trainer_kwargs = dict(
        model=model,
        benchmark=dm,
        optimizer=optimizer,
        gpu=0,
        callbacks=callbacks,
        loss_fn=UTKFaceMultiTaskLoss(),
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
