"""Common utilities for training scripts."""

import os
from os.path import isfile, join
from shutil import copyfile

import mlflow
from callbacks import EMACallback, FixNANinGrad, IncreaseDataEpoch
from hydra.utils import instantiate
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning.callbacks import LearningRateMonitor


def configure_mlflow_logger(cfg) -> MLFlowLogger:
    """Configure MLflow tracking and return a configured MLFlowLogger.

    Sets up:
    - Tracking URI
    - Workspace (if specified)
    - Experiment (creates if doesn't exist)
    - System metrics sampling interval (if specified)

    Args:
        cfg: Hydra config with logger.mlflow section

    Returns:
        Configured MLFlowLogger instance
    """
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)

    if hasattr(cfg.logger, "workspace_name"):
        mlflow.set_workspace(cfg.logger.workspace_name)

    if mlflow.get_experiment_by_name(cfg.logger.experiment_name) is None:
        mlflow.create_experiment(cfg.logger.experiment_name)
    mlflow.set_experiment(cfg.logger.experiment_name)

    if hasattr(cfg.logger, "system_metrics_sampling_interval"):
        mlflow.set_system_metrics_sampling_interval(cfg.logger.system_metrics_sampling_interval)

    return MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=cfg.logger.run_name,
        log_model=cfg.logger.log_model,
        tags=dict(cfg.logger.tags) if hasattr(cfg.logger, "tags") else None,
    )


def project_init(cfg):
    """Initialize project directories and copy Hydra config.

    Args:
        cfg: Hydra config with checkpoints.dirpath
    """
    print("Working directory set to {}".format(os.getcwd()))
    directory = cfg.checkpoints.dirpath
    os.makedirs(directory, exist_ok=True)
    copyfile(".hydra/config.yaml", join(directory, "config.yaml"))


def callback_init(cfg):
    """Initialize PyTorch Lightning callbacks.

    Args:
        cfg: Hydra config with checkpoints, progress_bar, and model sections

    Returns:
        List of configured callbacks
    """
    checkpoint_callback = instantiate(cfg.checkpoints)
    progress_bar = instantiate(cfg.progress_bar)
    lr_monitor = LearningRateMonitor()
    ema_callback = EMACallback(
        "network",
        "ema_network",
        decay=cfg.model.ema_decay,
        start_ema_step=cfg.model.start_ema_step,
        init_ema_random=False,
    )
    fix_nan_callback = FixNANinGrad(
        monitor=["train/loss"],
    )
    increase_data_epoch_callback = IncreaseDataEpoch()

    callbacks = [
        checkpoint_callback,
        progress_bar,
        lr_monitor,
        ema_callback,
        fix_nan_callback,
        increase_data_epoch_callback,
    ]
    return callbacks


def init_datamodule(cfg):
    """Initialize the datamodule from config.

    Args:
        cfg: Hydra config with datamodule section

    Returns:
        Instantiated datamodule
    """
    datamodule = instantiate(cfg.datamodule)
    return datamodule


def load_checkpoint_if_exists(directory: str) -> tuple[str | None, bool]:
    """Check if a checkpoint exists and return its path.

    Args:
        directory: Directory to check for last.ckpt

    Returns:
        Tuple of (checkpoint_path, checkpoint_exists)
    """
    checkpoint_path = join(directory, "last.ckpt")
    if isfile(checkpoint_path):
        print(f"Loading from checkpoint ... {checkpoint_path}")
        return checkpoint_path, True
    return None, False


def create_trainer(cfg, logger, callbacks):
    """Create a PyTorch Lightning Trainer.

    Args:
        cfg: Hydra config with trainer section
        logger: Configured logger instance
        callbacks: List of callbacks

    Returns:
        Configured Trainer instance
    """
    trainer, strategy = cfg.trainer, cfg.trainer.strategy

    trainer = instantiate(
        trainer,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
    )
    return trainer
