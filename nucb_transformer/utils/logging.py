# Logging setup for training runs.
# Provides a standard Python logger for stdout and optional W&B / TensorBoard
# integration; both are opt-in via the config so the package has no hard
# dependency on wandb or tensorboard.

import logging
from typing import Any


def setup_logger(name: str = "nucb_transformer", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with timestamps."""
    ...


def setup_wandb(cfg: dict) -> Any | None:
    """
    Initialize a W&B run using settings from the training config.
    Returns the wandb run object, or None if log_wandb is false in cfg.
    """
    ...


def log_metrics(metrics: dict, step: int, wandb_run=None, logger: logging.Logger | None = None):
    """Write metrics to stdout logger and optionally to W&B."""
    ...
