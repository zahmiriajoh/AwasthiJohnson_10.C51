# Logging setup for training runs.
# Provides a standard Python logger for stdout and optional W&B integration;
# both are opt-in via the config so the package has no hard dependency on wandb.

from __future__ import annotations

import logging
from typing import Any


def setup_logger(name: str = "nucb_transformer", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout with timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_wandb(cfg: dict) -> Any | None:
    """
    Initialize a W&B run using settings from the training config.
    Returns the wandb run object, or None if log_wandb is false in cfg.
    """
    if not cfg.get("training", {}).get("log_wandb", False):
        return None
    try:
        import wandb
        return wandb.init(
            project=cfg["training"].get("wandb_project", "nuclease_transformer"),
            config=cfg,
        )
    except ImportError:
        logging.getLogger("nucb_transformer").warning(
            "log_wandb=true but wandb is not installed; skipping."
        )
        return None


def log_metrics(
    metrics: dict,
    step: int,
    wandb_run=None,
    logger: logging.Logger | None = None,
):
    """Write metrics to stdout logger and optionally to W&B."""
    if logger:
        msg = "  ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in metrics.items())
        logger.info(f"step {step} | {msg}")
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
