"""
CLI entrypoint for test-set evaluation.
Loads a checkpoint, runs inference on the held-out test split,
and prints accuracy, macro-F1, hit-rate, and a confusion matrix.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config configs/default.yaml
"""

import argparse
import torch
from nucb_transformer.utils.io import load_checkpoint
from nucb_transformer.data.dataset import make_dataloaders, run_preprocessing_pipeline
from nucb_transformer.data.splits import split_dataset
from nucb_transformer.training.metrics import compute_metrics, confusion_matrix_df
from nucb_transformer.training.trainer import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained NucleaseTransformer checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    model, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    # load data → split → eval → print results
    ...
