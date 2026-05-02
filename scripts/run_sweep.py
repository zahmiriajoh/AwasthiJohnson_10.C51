"""
Optuna hyperparameter sweep runner with cluster-safe checkpointing.

Each trial saves a checkpoint after every epoch to checkpoints/sweep/current_trial.pt.
If the job is killed and restarted (e.g. Slurm preemption + --requeue), the interrupted
trial resumes automatically from its last saved epoch before starting new trials.

Results are written to checkpoints/sweep/results.csv and to an Optuna SQLite DB.
Multiple Slurm jobs can attach to the same study by pointing at the same DB file.

Usage:
    python scripts/run_sweep.py              # create/join study + run 20 trials
    python scripts/run_sweep.py --count 50
    python scripts/run_sweep.py --study_name my_study
"""

import argparse
import copy
import csv
import json
import os
import time

import optuna
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nucb_transformer.data.dataset import load_landscape, make_dataloaders
from nucb_transformer.data.splits import mutation_count_split
from nucb_transformer.models.transformer import NucleaseTransformer
from nucb_transformer.training.losses import WeightedActivityLoss
from nucb_transformer.training.metrics import ACTIVITY_CLASSES, hit_rate, spearman_rho
from nucb_transformer.training.trainer import _warmup_cosine_lr, load_config

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CKPT_DIR      = "checkpoints/sweep"
_RESUME_FILE   = os.path.join(_CKPT_DIR, ".resume.json")
_CURRENT_CKPT  = os.path.join(_CKPT_DIR, "current_trial.pt")
_RESULTS_CSV   = os.path.join(_CKPT_DIR, "results.csv")
_DB_PATH       = os.path.join(_CKPT_DIR, "optuna.db")

_CSV_FIELDS = [
    "trial_number", "val_hit_rate", "val_macro_f1", "val_spearman_rho",
    "val_loss", "val_accuracy", "epochs",
    "learning_rate", "weight_decay", "d_model", "num_heads", "num_layers", "dropout",
]

# Set once in main, read-only inside objective.
_BASE_CFG = None


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip, train: bool):
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(labels)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return {
        "loss":         total_loss / len(loader.dataset),
        "accuracy":     float((preds_np == labels_np).mean()),
        "macro_f1":     float(f1_score(labels_np, preds_np, average="macro", zero_division=0)),
        "hit_rate":     hit_rate(preds_np, labels_np),
        "spearman_rho": spearman_rho(preds_np.astype(float), labels_np.astype(float)),
    }


def _save_checkpoint(epoch, model, optimizer, scheduler, cfg):
    os.makedirs(_CKPT_DIR, exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cfg":                  cfg,
    }, _CURRENT_CKPT)


def _load_checkpoint(device):
    if os.path.isfile(_CURRENT_CKPT):
        return torch.load(_CURRENT_CKPT, map_location=device)
    return None


def _append_csv(row: dict):
    os.makedirs(_CKPT_DIR, exist_ok=True)
    write_header = not os.path.exists(_RESULTS_CSV)
    with open(_RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    cfg = copy.deepcopy(_BASE_CFG)
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    # Sample hyperparameters
    d_model    = trial.suggest_categorical("d_model",    [64, 128, 256])
    num_heads  = trial.suggest_categorical("num_heads",  [2, 4, 8])
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 6])
    dropout    = trial.suggest_float("dropout",        0.0, 0.2)
    lr         = trial.suggest_float("learning_rate",  1e-4, 1e-3, log=True)
    wd         = trial.suggest_float("weight_decay",   1e-5, 1e-1, log=True)

    cfg["model"]["d_model"]    = d_model
    cfg["model"]["num_heads"]  = num_heads
    cfg["model"]["num_layers"] = num_layers
    cfg["model"]["ffn_dim"]    = d_model * 2
    cfg["model"]["dropout"]    = dropout
    tcfg["learning_rate"]      = lr
    tcfg["weight_decay"]       = wd

    print(
        f"\n[Trial {trial.number}] "
        f"lr={lr:.2e}  wd={wd:.2e}  d_model={d_model}  "
        f"heads={num_heads}  layers={num_layers}  dropout={dropout:.3f}",
        flush=True,
    )

    # Persist params so a requeued job can resume this trial.
    os.makedirs(_CKPT_DIR, exist_ok=True)
    with open(_RESUME_FILE, "w") as f:
        json.dump({"trial_number": trial.number, "params": dict(trial.params)}, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg["seed"])

    df = load_landscape()
    train_df, val_df, _ = mutation_count_split(
        df,
        max_train_mutations=dcfg.get("max_train_mutations", 2),
        val_frac=dcfg["val_frac"],
        seed=tcfg["seed"],
    )
    train_loader, val_loader, _ = make_dataloaders(
        train_df, val_df, val_df,
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
    )

    model = NucleaseTransformer(**cfg["model"]).to(device)
    class_counts = [int((train_df["activity_level"] == c).sum()) for c in ACTIVITY_CLASSES]
    criterion = WeightedActivityLoss(class_counts, device=str(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine_lr(tcfg["warmup_steps"], total_steps),
    )

    # Restore from checkpoint if this trial was interrupted.
    start_epoch = 1
    ckpt = _load_checkpoint(device)
    if ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}", flush=True)

    best_hit_rate = 0.0
    best_val = {}

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        t0 = time.time()
        train_m = _run_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, tcfg["grad_clip"], train=True,
        )
        val_m = _run_epoch(
            model, val_loader, None, None,
            criterion, device, tcfg["grad_clip"], train=False,
        )
        elapsed = time.time() - t0

        print(
            f"  epoch {epoch:3d}/{tcfg['epochs']} "
            f"| train loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f} "
            f"| val loss={val_m['loss']:.4f}  hit_rate={val_m['hit_rate']:.4f} "
            f"f1={val_m['macro_f1']:.4f}  rho={val_m['spearman_rho']:.4f} "
            f"| {elapsed:.1f}s",
            flush=True,
        )

        if val_m["hit_rate"] > best_hit_rate:
            best_hit_rate = val_m["hit_rate"]
            best_val = val_m

        _save_checkpoint(epoch, model, optimizer, scheduler, cfg)

        # Prune unpromising trials early.
        trial.report(val_m["hit_rate"], epoch)
        if trial.should_prune():
            print(f"  Pruned at epoch {epoch}", flush=True)
            raise optuna.TrialPruned()

    _append_csv({
        "trial_number":     trial.number,
        "val_hit_rate":     best_hit_rate,
        "val_macro_f1":     best_val.get("macro_f1", ""),
        "val_spearman_rho": best_val.get("spearman_rho", ""),
        "val_loss":         best_val.get("loss", ""),
        "val_accuracy":     best_val.get("accuracy", ""),
        "epochs":           tcfg["epochs"],
        "learning_rate":    lr,
        "weight_decay":     wd,
        "d_model":          d_model,
        "num_heads":        num_heads,
        "num_layers":       num_layers,
        "dropout":          dropout,
    })

    # Clean up per-trial state.
    for path in (_CURRENT_CKPT, _RESUME_FILE):
        if os.path.exists(path):
            os.remove(path)

    return best_hit_rate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--study_name", default="nucb_sweep")
    p.add_argument("--count", type=int, default=20, help="Number of trials to run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(_CKPT_DIR, exist_ok=True)

    _BASE_CFG = load_config("configs/default.yaml")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=f"sqlite:///{_DB_PATH}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    # Any trial left RUNNING from a killed job will never finish — mark it failed.
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            try:
                study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass

    # If the last job was preempted mid-trial, re-queue those params so the
    # next ask() picks them up and the checkpoint can be reused.
    if os.path.exists(_RESUME_FILE):
        with open(_RESUME_FILE) as f:
            resume_info = json.load(f)
        print(
            f"Found interrupted trial {resume_info['trial_number']} — "
            "re-queuing params for resumption",
            flush=True,
        )
        study.enqueue_trial(resume_info["params"])

    print(
        f"Study '{args.study_name}' — "
        f"{len(study.trials)} existing trials, running {args.count} more.",
        flush=True,
    )

    study.optimize(objective, n_trials=args.count, show_progress_bar=True)

    print("\n=== Sweep complete ===")
    best = study.best_trial
    print(f"Best trial: #{best.number}  val_hit_rate={best.value:.4f}")
    print("Params:", json.dumps(best.params, indent=2))
    print(f"All results saved to {_RESULTS_CSV}")
    print(f"Optuna DB: {_DB_PATH}  (query with optuna-dashboard or study.trials_dataframe())")
