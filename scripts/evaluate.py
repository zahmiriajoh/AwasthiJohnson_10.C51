"""
CLI entrypoint for test-set evaluation.
Loads a checkpoint, runs inference on the held-out test split,
and prints accuracy, macro-F1, hit-rate, and a confusion matrix.

Inference accumulates only argmax class indices per batch (O(N) ints),
not full logit tensors, so it stays memory-efficient on large test sets.

Usage:
    # auto-find best checkpoint in default dir
    python scripts/evaluate.py

    # specify checkpoint explicitly
    python scripts/evaluate.py --checkpoint checkpoints/epoch_012_acc0.8234.pt

    # save per-sequence predictions to CSV
    python scripts/evaluate.py --save_predictions results/test_predictions.csv
"""

import argparse
import numpy as np
import torch
from sklearn.metrics import f1_score

from nucb_transformer.utils.io import load_checkpoint, best_checkpoint_path
from nucb_transformer.data.dataset import load_landscape, make_dataloaders
from nucb_transformer.data.splits import mutation_count_split
from nucb_transformer.training.metrics import (
    confusion_matrix_df, bootstrap_hit_rate, hit_rate, spearman_rho, ACTIVITY_CLASSES,
)
from nucb_transformer.training.trainer import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained NucleaseTransformer checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt file; omit to auto-find best in --checkpoint_dir")
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_predictions", default=None, metavar="CSV",
                   help="Save per-sequence predictions to this CSV path")
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
    _, _, test_df = mutation_count_split(
        df,
        max_train_mutations=dcfg.get("max_train_mutations", 2),
        val_frac=dcfg["val_frac"],
        seed=tcfg["seed"],
    )
    _, _, test_loader = make_dataloaders(
        test_df, test_df, test_df,
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
    )

    # Store only the argmax per batch — O(N) ints instead of O(N*4) floats.
    all_preds, all_labels = [], []
    with torch.no_grad():
        for tokens, labels in test_loader:
            logits = model(tokens.to(device))
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # All metrics derived from class indices (no logits needed).
    accuracy  = float((preds_np == labels_np).mean())
    macro_f1  = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
    per_cls_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
    hr        = hit_rate(preds_np, labels_np)
    rho       = spearman_rho(preds_np.astype(float), labels_np.astype(float))
    ci_lo, ci_hi = bootstrap_hit_rate(preds_np, labels_np)

    print(f"\nCheckpoint : {ckpt_path}")
    print(f"Test size  : {len(test_df):,}")
    print(f"  accuracy     {accuracy:.4f}")
    print(f"  macro_f1     {macro_f1:.4f}")
    print(f"  hit_rate     {hr:.4f}  (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
    print(f"  spearman_rho {rho:.4f}")
    print()
    for i, cls in enumerate(ACTIVITY_CLASSES):
        print(f"  f1_{cls:<20} {per_cls_f1[i]:.4f}")
    print()
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix_df(preds_np, labels_np).to_string())

    # Attach predictions back to test_df for downstream analysis.
    test_df = test_df.reset_index(drop=True)
    test_df["predicted_class"] = [ACTIVITY_CLASSES[i] for i in preds_np]
    test_df["correct"] = test_df["predicted_class"] == test_df["activity_level"]

    print(f"\nSample predictions (first 10 rows):")
    print(test_df[["sequence", "activity_level", "predicted_class", "correct"]].head(10).to_string())

    if args.save_predictions:
        import os
        os.makedirs(os.path.dirname(args.save_predictions) or ".", exist_ok=True)
        test_df.to_csv(args.save_predictions, index=False)
        print(f"\nFull predictions saved → {args.save_predictions}")
