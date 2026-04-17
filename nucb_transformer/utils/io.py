# Checkpoint save / load helpers.
# Saves model weights, optimizer state, epoch, and config together so a run
# can be fully resumed or a specific checkpoint loaded for evaluation.

import torch
import os
from nucb_transformer.models.transformer import NucleaseTransformer


def save_checkpoint(
    model: NucleaseTransformer,
    optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    path: str,
):
    """Save a full training checkpoint to path (creates parent dirs if needed)."""
    ...


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[NucleaseTransformer, dict]:
    """
    Restore model weights from a saved checkpoint.
    Returns (model, checkpoint_dict) so callers can also access saved metrics / config.
    """
    ...


def best_checkpoint_path(checkpoint_dir: str) -> str:
    """Return the path of the checkpoint with the highest val accuracy in checkpoint_dir."""
    ...
