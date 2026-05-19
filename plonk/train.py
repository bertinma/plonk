# Fix for compatibility with Python 3.14
# /!\ DO NOT REMOVE /!\
import utils.fix_py314  # noqa: F401 # isort: skip # /!\ DO NOT REMOVE /!\

import hydra
import torch
from omegaconf import OmegaConf

from plonk.models.module import DiffGeolocalizer
from plonk.training_utils import (
    callback_init,
    configure_mlflow_logger,
    create_trainer,
    init_datamodule,
    load_checkpoint_if_exists,
    project_init,
)

torch.set_float32_matmul_precision("high")  # TODO do we need that?

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations in hydra:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)


def load_model(cfg, dict_config, callbacks, logger):
    """Load or create DiffGeolocalizer model and trainer."""
    directory = cfg.checkpoints.dirpath

    ckpt_path, checkpoint_exists = load_checkpoint_if_exists(directory) if cfg.resume else (None, False)

    if checkpoint_exists and ckpt_path is not None:
        model = DiffGeolocalizer.load_from_checkpoint(ckpt_path, cfg=cfg.model, weights_only=False)
    else:
        model = DiffGeolocalizer(cfg.model)

    trainer = create_trainer(cfg, logger, callbacks)

    # log_hyperparams must be called after trainer is created so the MLflow run is active
    logger.log_hyperparams(dict_config)

    return trainer, model, ckpt_path


def hydra_boilerplate(cfg):
    dict_config = OmegaConf.to_container(cfg, resolve=True)
    callbacks = callback_init(cfg)
    datamodule = init_datamodule(cfg)
    project_init(cfg)
    logger = configure_mlflow_logger(cfg)
    trainer, model, ckpt_path = load_model(cfg, dict_config, callbacks, logger)
    return trainer, model, datamodule, ckpt_path


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    trainer, model, datamodule, ckpt_path = hydra_boilerplate(cfg)
    model.datamodule = datamodule
    if cfg.mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif cfg.mode == "eval":
        trainer.test(model, datamodule=datamodule)
    elif cfg.mode == "traineval":
        cfg.mode = "train"
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        cfg.mode = "test"
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
