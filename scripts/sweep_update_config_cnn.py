"""
Extract the best hyperparameters from a completed CNN Optuna sweep and write
them to configs/best_cnn.yaml for use in a full training run.

Usage:
    python scripts/sweep_update_config_cnn.py
    python scripts/sweep_update_config_cnn.py --study_name my_cnn_study
    python scripts/sweep_update_config_cnn.py --output configs/custom_cnn.yaml
"""

import argparse
import os

import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DB_PATH     = "checkpoints/sweep_cnn/optuna.db"
_BASE_CONFIG = "configs/cnn_default.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", default="nucb_cnn_sweep")
    p.add_argument("--output", default="configs/best_cnn.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(_DB_PATH):
        raise FileNotFoundError(f"No Optuna DB found at {_DB_PATH}. Run the sweep first.")

    study = optuna.load_study(
        study_name=args.study_name,
        storage=f"sqlite:///{_DB_PATH}",
    )

    best = study.best_trial
    p    = best.params

    print(f"Best trial:     #{best.number}")
    print(f"val_hit_rate:   {best.value:.4f}")
    print(f"Params:         {p}")

    with open(_BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["num_filters"]     = p["num_filters"]
    cfg["model"]["kernel_size"]     = p["kernel_size"]
    cfg["model"]["num_conv_layers"] = p["num_conv_layers"]
    cfg["model"]["hidden_dim"]      = p["hidden_dim"]
    cfg["model"]["dropout"]         = p["dropout"]
    cfg["training"]["learning_rate"] = p["learning_rate"]
    cfg["training"]["weight_decay"]  = p["weight_decay"]

    with open(args.output, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig written to {args.output}")
    print(f"Train with: python scripts/train_cnn.py --config {args.output}")
