import logging

import coloredlogs
import hydra
import matplotlib.pyplot as plt
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def install_logging(level=logging.DEBUG):
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES
    level_styles["info"] = {"color": "yellow"}

    coloredlogs.install(
        level=level,
        fmt="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
        level_styles=level_styles,
    )

    # hide the gazillion cryptic debug messages coming from PIL
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.category").disabled = True
    logging.getLogger("numba").setLevel(logging.WARNING)
    # plt.set_loglevel("WARNING")


def initialize_wandb(config):
    wandb.init(
        entity=getattr(config.wandb, "entity", None),
        project=config.wandb.project,
        config=OmegaConf.to_container(config),
        group=getattr(config.wandb, "group", None),
        mode=getattr(config.wandb, "mode", "online"),
    )
    if HydraConfig.initialized():
        orig_cwd = hydra.utils.get_original_cwd()
        wandb.run.log_code(orig_cwd)
