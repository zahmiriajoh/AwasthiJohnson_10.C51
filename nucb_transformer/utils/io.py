# Checkpoint save / load helpers.
# Saves model weights, optimizer state, epoch, and config together so a run
# can be fully resumed or a specific checkpoint loaded for evaluation.

import os
import re
import glob
import torch
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[NucleaseTransformer, dict]:
    """
    Restore model weights from a saved checkpoint.
    Returns (model, checkpoint_dict) so callers can also access saved metrics / config.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = NucleaseTransformer(**ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def best_checkpoint_path(checkpoint_dir: str) -> str:
    """Return the path of the checkpoint with the highest val accuracy in checkpoint_dir."""
    paths = glob.glob(os.path.join(checkpoint_dir, "epoch_*_acc*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def _acc_from_name(path: str) -> float:
        match = re.search(r"_acc([0-9.]+)\.pt$", os.path.basename(path))
        return float(match.group(1)) if match else 0.0

    return max(paths, key=_acc_from_name)
