# Training loop with LR warmup, cosine decay, gradient clipping, and checkpointing.
# Weighted cross-entropy handles the severe class imbalance in the landscape
# ('activity > WT' and 'activity > A73R' are rare relative to 'non-functional').

import os
import math
import logging
import torch
import yaml
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from nucb_cnn.models.cnn import NucleaseConvNet
from nucb_cnn.data.dataset import load_landscape, make_dataloaders
from nucb_cnn.data.splits import mutation_count_split
from nucb_cnn.training.metrics import compute_metrics, ACTIVITY_CLASSES
from nucb_cnn.training.losses import WeightedActivityLoss

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("nucb_cnn")


def load_config(path: str) -> dict:
    """Load hyperparameters from a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> NucleaseConvNet:
    """Instantiate NucleaseConvNet from the model section of a config dict."""
    return NucleaseConvNet(**cfg["model"])


def _warmup_cosine_lr(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay; returned as a LambdaLR scale function."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def train_one_epoch(
    model: NucleaseConvNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    criterion: torch.nn.Module,
    device: torch.device,
    grad_clip: float,
) -> dict:
    """Single training epoch; returns dict with loss and metrics."""
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    batch_bar = tqdm(loader, leave=False, desc="  batches")
    for tokens, labels in batch_bar:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(tokens)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * len(labels)
        all_logits.append(logits.detach())
        all_labels.append(labels)
        batch_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def validate(
    model: NucleaseConvNet,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Full evaluation pass; returns the same metrics dict as train_one_epoch."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(config_path: str = "configs/default_cnn.yaml"):
    """
    Full training run:
      1. Load config and data
      2. Build model, optimizer, LR scheduler
      3. Epoch loop: train → validate → save best checkpoint
      4. Log metrics to stdout
    """
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tcfg["seed"])

    # ── data ──────────────────────────────────────────────────────────────────
    df = load_landscape()
    train_df, val_df, _ = mutation_count_split(
        df,
        max_train_mutations=dcfg.get("max_train_mutations", 2),
        val_frac=dcfg["val_frac"],
        seed=tcfg["seed"],
    )
    train_loader, val_loader, _ = make_dataloaders(
        train_df, val_df, val_df,   # val reused as test placeholder — test eval is separate
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
    )

    # ── model + loss ──────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    class_counts = [int((train_df["activity_level"] == c).sum()) for c in ACTIVITY_CLASSES]
    criterion = WeightedActivityLoss(class_counts, device=str(device))

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    total_steps = tcfg["epochs"] * len(train_loader)
    scheduler   = LambdaLR(
        optimizer,
        lr_lambda=_warmup_cosine_lr(tcfg["warmup_steps"], total_steps),
    )

    # ── epoch loop ────────────────────────────────────────────────────────────
    os.makedirs(tcfg["checkpoint_dir"], exist_ok=True)
    best_val_acc = 0.0
    epoch_bar = tqdm(range(1, tcfg["epochs"] + 1), desc="epochs")

    for epoch in epoch_bar:
        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, tcfg["grad_clip"],
        )
        val_m = validate(model, val_loader, criterion, device)

        epoch_bar.set_postfix({
            "train_loss": f"{train_m['loss']:.4f}",
            "val_loss":   f"{val_m['loss']:.4f}",
            "val_acc":    f"{val_m['accuracy']:.4f}",
            "hit_rate":   f"{val_m['hit_rate']:.4f}",
        })

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            ckpt_path = os.path.join(
                tcfg["checkpoint_dir"],
                f"epoch_{epoch:03d}_acc{best_val_acc:.4f}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_m,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"  → new best val accuracy {best_val_acc:.4f}, saved to {ckpt_path}")

    return model
