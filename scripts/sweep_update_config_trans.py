"""
Extract the best hyperparameters from a completed Optuna sweep and write
them to configs/best.yaml for use in a full training run.

Usage:
    python scripts/sweep_update_config.py
    python scripts/sweep_update_config.py --study_name my_study
    python scripts/sweep_update_config.py --output configs/custom.yaml
"""

import argparse
import os

import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)

_DB_PATH     = "results/260502_sweep_results_1/optuna.db"
_BASE_CONFIG = "configs/default.yaml"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", default="nucb_sweep")
    p.add_argument("--output", default="configs/best.yaml")
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
    p = best.params

    print(f"Best trial:     #{best.number}")
    print(f"val_hit_rate:   {best.value:.4f}")
    print(f"Params:         {p}")

    with open(_BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["d_model"]    = p["d_model"]
    cfg["model"]["num_heads"]  = p["num_heads"]
    cfg["model"]["num_layers"] = p["num_layers"]
    cfg["model"]["ffn_dim"]    = p["d_model"] * 2
    cfg["model"]["dropout"]    = p["dropout"]
    cfg["training"]["learning_rate"] = p["learning_rate"]
    cfg["training"]["weight_decay"]  = p["weight_decay"]

    with open(args.output, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nConfig written to {args.output}")
    print(f"Train with: python scripts/train.py --config {args.output}")
