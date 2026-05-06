"""
CLI entrypoint for training.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
"""

import argparse
from nucb_transformer.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train NucleaseTransformer")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config)
