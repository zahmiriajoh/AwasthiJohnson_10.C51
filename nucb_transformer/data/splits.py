# Train / val / test split logic.
# Supports random splitting and Hamming-distance-aware clustering splits
# (matching the DeepMind CNN evaluation methodology) to prevent similar
# sequences from leaking across splits and inflating held-out accuracy.

import pandas as pd
import numpy as np


def split_dataset(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    cluster_aware: bool = True,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split labeled landscape DataFrame into train / val / test.

    cluster_aware=True:
      Computes pairwise Hamming distances, runs hierarchical clustering
      (complete linkage), and assigns whole clusters to splits so that
      sequence-similar variants never span the boundary.

    cluster_aware=False:
      Simple random split, useful for quick iteration and unit tests.
    """
    ...


def hamming_distance_matrix(sequences: list[str]) -> np.ndarray:
    """Return an (N, N) pairwise Hamming distance matrix for a list of AA strings."""
    ...


def cluster_sequences(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agglomerative clustering with complete linkage; returns cluster label array."""
    ...
