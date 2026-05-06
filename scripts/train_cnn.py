"""
CLI entrypoint for training the CNN.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --config configs/cnn_default.yaml
"""

import argparse
from nucb_cnn.training.trainer import train


def parse_args():
    p = argparse.ArgumentParser(description="Train NucleaseConvNet")
    p.add_argument("--config", default="configs/cnn_default.yaml", help="Path to YAML config")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(config_path=args.config)
