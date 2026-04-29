# Evaluation metrics — mirrors what the DeepMind CNN paper reports
# so results are directly comparable.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix

ACTIVITY_CLASSES = ['activity > 0', 'activity > A73R', 'activity > WT', 'non-functional']


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Master metrics function called after each epoch.
    Returns: accuracy, macro-F1, per-class F1, hit_rate, spearman_rho.
    """
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = float((preds == labels_np).mean())
    macro_f1 = float(f1_score(labels_np, preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **{f"f1_{c}": float(per_class_f1[i]) for i, c in enumerate(ACTIVITY_CLASSES)},
        "hit_rate": hit_rate(preds, labels_np),
        "spearman_rho": spearman_rho(preds.astype(float), labels_np.astype(float)),
    }


def hit_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    positive_classes: list[str] = ["activity > WT", "activity > A73R"],
) -> float:
    """
    Fraction of true high-activity variants correctly identified.
    Primary bio-relevant metric; the CNN paper reports it per sublibrary.
    """
    pos_indices = [ACTIVITY_CLASSES.index(c) for c in positive_classes]
    true_pos_mask = np.isin(labels, pos_indices)
    if true_pos_mask.sum() == 0:
        return 0.0
    return float(np.isin(preds[true_pos_mask], pos_indices).mean())


def bootstrap_hit_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% confidence interval on hit_rate via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(preds)
    rates = [
        hit_rate(preds[idx := rng.integers(0, n, size=n)], labels[idx])
        for _ in range(n_bootstrap)
    ]
    return float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def spearman_rho(preds: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank correlation between predicted class indices and true class indices."""
    rho, _ = spearmanr(preds, targets)
    return float(rho)


def confusion_matrix_df(preds: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Labeled confusion matrix DataFrame for notebook display."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(ACTIVITY_CLASSES))))
    return pd.DataFrame(cm, index=ACTIVITY_CLASSES, columns=ACTIVITY_CLASSES)
