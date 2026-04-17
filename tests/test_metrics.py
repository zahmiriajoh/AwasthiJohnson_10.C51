# Tests for evaluation metrics: hit rate, bootstrap CI, Spearman correlation.

import numpy as np
import pytest
from nucb_transformer.training.metrics import hit_rate, bootstrap_hit_rate, spearman_rho
from nucb_transformer.data.encoding import ACTIVITY_CLASSES


def test_hit_rate_perfect():
    labels = np.array([2, 3, 2, 3])   # all '>WT' or '>=A73R'
    preds  = np.array([2, 3, 2, 3])
    assert hit_rate(preds, labels) == 1.0


def test_hit_rate_zero():
    labels = np.array([2, 3])          # true positives
    preds  = np.array([0, 0])          # all predicted '<WT'
    assert hit_rate(preds, labels) == 0.0


def test_bootstrap_returns_valid_interval():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=200)
    preds  = rng.integers(0, 4, size=200)
    lo, hi = bootstrap_hit_rate(preds, labels, n_bootstrap=100)
    assert 0.0 <= lo <= hi <= 1.0


def test_spearman_perfect_rank():
    x = np.arange(10, dtype=float)
    assert spearman_rho(x, x) == pytest.approx(1.0)
