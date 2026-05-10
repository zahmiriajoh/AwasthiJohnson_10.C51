"""
CLI entrypoint for training the CNN.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --config configs/default_cnn.yaml
    python scripts/train_cnn.py --config configs/best_cnn.yaml
"""

import argparse
from nucb_cnn.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train NucleaseConvNet")
    p.add_argument("--config", default="configs/default_cnn.yaml", help="Path to YAML config")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config)
