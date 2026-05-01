"""
Weights & Biases hyperparameter sweep runner with cluster-safe checkpointing.

Each trial saves a checkpoint after every epoch to checkpoints/sweep/<run_id>/latest.pt.
If the job is killed and restarted, the interrupted trial resumes automatically before
the agent requests a new trial from the sweep server.

Usage:
    python scripts/run_sweep.py              # create sweep + run 20 trials
    python scripts/run_sweep.py --count 50
    python scripts/run_sweep.py --sweep_id <id>  # attach to an existing sweep
"""

import argparse
import os
import torch
import yaml
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nucb_transformer.training.trainer import load_config, _warmup_cosine_lr
from nucb_transformer.training.losses import WeightedActivityLoss
from nucb_transformer.training.metrics import ACTIVITY_CLASSES, hit_rate, spearman_rho
from nucb_transformer.data.dataset import load_landscape, make_dataloaders
from nucb_transformer.data.splits import mutation_count_split
from nucb_transformer.models.transformer import NucleaseTransformer
from sklearn.metrics import f1_score

# Persists the run_id of any interrupted trial across job restarts.
_RESUME_FILE = "checkpoints/sweep/.resume_run_id"
_SWEEP_CKPT_DIR = "checkpoints/sweep"


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


def _save_checkpoint(run_id, epoch, model, optimizer, scheduler, cfg):
    path = os.path.join(_SWEEP_CKPT_DIR, run_id, "latest.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "cfg":                  cfg,
    }, path)


def _load_checkpoint(run_id, device):
    path = os.path.join(_SWEEP_CKPT_DIR, run_id, "latest.pt")
    if os.path.isfile(path):
        return torch.load(path, map_location=device)
    return None


def sweep_train(resume_id=None):
    init_kwargs = {"resume": "allow"}
    if resume_id:
        init_kwargs["id"] = resume_id

    with wandb.init(**init_kwargs) as run:
        # Record this run_id so a restart can find and resume it.
        os.makedirs(os.path.dirname(_RESUME_FILE), exist_ok=True)
        with open(_RESUME_FILE, "w") as f:
            f.write(run.id)

        wc = wandb.config
        cfg = load_config("configs/default.yaml")
        cfg["model"]["d_model"]    = wc.d_model
        cfg["model"]["num_heads"]  = wc.num_heads
        cfg["model"]["num_layers"] = wc.num_layers
        cfg["model"]["ffn_dim"]    = wc.d_model * 2
        cfg["model"]["dropout"]    = wc.dropout
        cfg["training"]["learning_rate"] = wc.learning_rate
        cfg["training"]["weight_decay"]  = wc.weight_decay

        tcfg = cfg["training"]
        dcfg = cfg["data"]
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

        optimizer = AdamW(
            model.parameters(),
            lr=tcfg["learning_rate"],
            weight_decay=tcfg["weight_decay"],
        )
        total_steps = tcfg["epochs"] * len(train_loader)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lr(tcfg["warmup_steps"], total_steps),
        )

        # Restore from checkpoint if this trial was interrupted.
        start_epoch = 1
        best_hit_rate = 0.0
        ckpt = _load_checkpoint(run.id, device)
        if ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed trial {run.id} from epoch {ckpt['epoch']}")

        for epoch in range(start_epoch, tcfg["epochs"] + 1):
            train_m = _run_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, device, tcfg["grad_clip"], train=True,
            )
            val_m = _run_epoch(
                model, val_loader, None, None,
                criterion, device, tcfg["grad_clip"], train=False,
            )

            wandb.log({
                "epoch":            epoch,
                "train/loss":       train_m["loss"],
                "train/accuracy":   train_m["accuracy"],
                "train/hit_rate":   train_m["hit_rate"],
                "val/loss":         val_m["loss"],
                "val/accuracy":     val_m["accuracy"],
                "val/macro_f1":     val_m["macro_f1"],
                "val/hit_rate":     val_m["hit_rate"],
                "val/spearman_rho": val_m["spearman_rho"],
            })

            best_hit_rate = max(best_hit_rate, val_m["hit_rate"])
            _save_checkpoint(run.id, epoch, model, optimizer, scheduler, cfg)

        wandb.summary["best_val_hit_rate"] = best_hit_rate

    # Trial finished cleanly — clear the resume marker and checkpoint.
    if os.path.exists(_RESUME_FILE):
        os.remove(_RESUME_FILE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", default=None, help="Attach to an existing sweep instead of creating one")
    p.add_argument("--count", type=int, default=20, help="Number of trials to run")
    p.add_argument("--project", default="nuclease_transformer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        with open("configs/sweep.yaml") as f:
            sweep_cfg = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep_cfg, project=args.project)
        print(f"Created sweep: {sweep_id}")

    # If a previous run was interrupted, resume it before asking the sweep
    # server for a new trial — otherwise that trial's progress is lost.
    resume_id = None
    if os.path.exists(_RESUME_FILE):
        with open(_RESUME_FILE) as f:
            resume_id = f.read().strip()
        print(f"Found interrupted trial {resume_id} — resuming before starting new trials")
        sweep_train(resume_id=resume_id)

    wandb.agent(sweep_id, function=sweep_train, count=args.count, project=args.project)
