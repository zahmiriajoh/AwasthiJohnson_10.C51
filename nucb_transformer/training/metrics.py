# Evaluation metrics — mirrors what the DeepMind CNN paper reports
# so results are directly comparable.

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from nucb_transformer.data.encoding import ACTIVITY_CLASSES


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Master metrics function called after each epoch.
    Returns: accuracy, macro-F1, per-class F1, hit_rate, spearman_rho.
    """
    ...


def hit_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    positive_classes: list[str] = [">WT", ">=A73R"],
) -> float:
    """
    Fraction of true high-activity variants correctly identified.
    Primary bio-relevant metric; the CNN paper reports it per sublibrary.
    """
    ...


def bootstrap_hit_rate(
    preds: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% confidence interval on hit_rate via bootstrap resampling."""
    ...


def spearman_rho(preds: np.ndarray, targets: np.ndarray) -> float:
    """Spearman rank correlation between predicted scores and enrichment values."""
    ...


def confusion_matrix_df(preds: np.ndarray, labels: np.ndarray):
    """Labeled confusion matrix DataFrame for notebook display."""
    ...
