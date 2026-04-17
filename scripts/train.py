"""
CLI entrypoint for training.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/sweep.yaml --run_name sweep_lr1e4
"""

import argparse
from nucb_transformer.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train NucleaseTransformer")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--run_name", default=None, help="Override W&B run name")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config, run_name=args.run_name)
