# Tests for evaluation metrics: hit rate, bootstrap CI, Spearman correlation.

import torch
import numpy as np
import pytest
from nucb_transformer.training.metrics import (
    hit_rate, bootstrap_hit_rate, spearman_rho, compute_metrics, ACTIVITY_CLASSES,
)

# In the current sorted ACTIVITY_CLASSES:
#   0: 'activity > 0'
#   1: 'activity > A73R'   ← positive
#   2: 'activity > WT'     ← positive
#   3: 'non-functional'
POS_INDICES = [
    ACTIVITY_CLASSES.index("activity > A73R"),
    ACTIVITY_CLASSES.index("activity > WT"),
]


def test_hit_rate_perfect():
    labels = np.array(POS_INDICES * 2)
    preds  = np.array(POS_INDICES * 2)
    assert hit_rate(preds, labels) == pytest.approx(1.0)


def test_hit_rate_zero():
    labels = np.array(POS_INDICES)
    preds  = np.array([0, 0])   # predicted non-positive for all true positives
    assert hit_rate(preds, labels) == pytest.approx(0.0)


def test_hit_rate_no_true_positives_returns_zero():
    labels = np.array([0, 3])   # no positive-class samples
    preds  = np.array([0, 3])
    assert hit_rate(preds, labels) == 0.0


def test_bootstrap_returns_ordered_interval():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    preds  = rng.integers(0, 4, size=200)
    lo, hi = bootstrap_hit_rate(preds, labels, n_bootstrap=100)
    assert 0.0 <= lo <= hi <= 1.0


def test_spearman_perfect_rank():
    x = np.arange(10, dtype=float)
    assert spearman_rho(x, x) == pytest.approx(1.0)


def test_spearman_reverse_rank():
    x = np.arange(10, dtype=float)
    assert spearman_rho(x, x[::-1]) == pytest.approx(-1.0)


def test_compute_metrics_keys():
    logits = torch.randn(20, 4)
    labels = torch.randint(0, 4, (20,))
    metrics = compute_metrics(logits, labels)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "hit_rate" in metrics
    assert "spearman_rho" in metrics
    for cls in ACTIVITY_CLASSES:
        assert f"f1_{cls}" in metrics


def test_compute_metrics_accuracy_range():
    logits = torch.randn(50, 4)
    labels = torch.randint(0, 4, (50,))
    metrics = compute_metrics(logits, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0
