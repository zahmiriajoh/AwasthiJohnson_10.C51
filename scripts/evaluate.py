"""
CLI entrypoint for test-set evaluation.
Loads a checkpoint, runs inference on the held-out test split,
and prints accuracy, macro-F1, hit-rate, and a confusion matrix.

Usage:
    # auto-find best checkpoint in default dir
    python scripts/evaluate.py

    # specify checkpoint explicitly
    python scripts/evaluate.py --checkpoint checkpoints/epoch_012_acc0.8234.pt
"""

import argparse
import torch
from nucb_transformer.utils.io import load_checkpoint, best_checkpoint_path
from nucb_transformer.data.dataset import load_landscape, make_dataloaders
from nucb_transformer.data.splits import split_dataset
from nucb_transformer.training.metrics import (
    compute_metrics, confusion_matrix_df, bootstrap_hit_rate, ACTIVITY_CLASSES,
)
from nucb_transformer.training.trainer import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained NucleaseTransformer checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt file; omit to auto-find best in --checkpoint_dir")
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    device = torch.device(args.device)

    ckpt_path = args.checkpoint or best_checkpoint_path(args.checkpoint_dir)
    model, _ = load_checkpoint(ckpt_path, device=args.device)
    model.eval()

    df = load_landscape()
    _, _, test_df = split_dataset(
        df,
        val_frac=dcfg["val_frac"],
        test_frac=dcfg["test_frac"],
        cluster_aware=dcfg["cluster_aware_split"],
        seed=tcfg["seed"],
    )
    _, _, test_loader = make_dataloaders(
        test_df, test_df, test_df,
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for tokens, labels in test_loader:
            all_logits.append(model(tokens.to(device)).cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds = all_logits.argmax(dim=-1).numpy()
    labels_np = all_labels.numpy()

    metrics = compute_metrics(all_logits, all_labels)
    ci_lo, ci_hi = bootstrap_hit_rate(preds, labels_np)

    print(f"\nCheckpoint : {ckpt_path}")
    print(f"Test size  : {len(test_df):,}")
    print(f"  accuracy     {metrics['accuracy']:.4f}")
    print(f"  macro_f1     {metrics['macro_f1']:.4f}")
    print(f"  hit_rate     {metrics['hit_rate']:.4f}  (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
    print(f"  spearman_rho {metrics['spearman_rho']:.4f}")
    print()
    for cls in ACTIVITY_CLASSES:
        print(f"  f1_{cls:<20} {metrics[f'f1_{cls}']:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix_df(preds, labels_np).to_string())
