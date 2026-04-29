# Train / val / test split logic.
# Supports random splitting and Hamming-distance-aware clustering splits
# (matching the DeepMind CNN evaluation methodology) to prevent similar
# sequences from leaking across splits and inflating held-out accuracy.

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# ── sublibrary control ────────────────────────────────────────────────────────

SUBLIBRARY = None  # e.g. "g3_unmatched" — None keeps all sublibraries

def filter_by_sublibrary(df: pd.DataFrame, sublibrary: str | None = SUBLIBRARY) -> pd.DataFrame:
    """Return rows where sublibrary appears in sublibrary_names. None returns all rows."""
    if sublibrary is None:
        return df.reset_index(drop=True)
    mask = df["sublibrary_names"].apply(lambda names: sublibrary in names)
    return df[mask].reset_index(drop=True)


# ── mutation-count split ──────────────────────────────────────────────────────

def mutation_count_split(
    df: pd.DataFrame,
    max_train_mutations: int = 2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by num_mutations: variants with <= max_train_mutations go to train/val,
    variants with > max_train_mutations go to test.
    Validation is a random subset of the train pool.
    """
    train_pool = df[df["num_mutations"] <= max_train_mutations].reset_index(drop=True)
    test_df    = df[df["num_mutations"] >  max_train_mutations].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * val_frac)

    val_df   = train_pool.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = train_pool.iloc[idx[n_val:]].reset_index(drop=True)

    return train_df, val_df, test_df


# ── splitting ─────────────────────────────────────────────────────────────────

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
    rng = np.random.default_rng(seed)

    if not cluster_aware:
        idx = rng.permutation(len(df))
        n_test = int(len(df) * test_frac)
        n_val = int(len(df) * val_frac)
        test  = df.iloc[idx[:n_test]].reset_index(drop=True)
        val   = df.iloc[idx[n_test:n_test + n_val]].reset_index(drop=True)
        train = df.iloc[idx[n_test + n_val:]].reset_index(drop=True)
        return train, val, test

    sequences = df["sequence"].tolist()
    n_clusters = max(20, int(1 / min(val_frac, test_frac)))
    dist = hamming_distance_matrix(sequences)
    labels = cluster_sequences(dist, n_clusters)

    cluster_ids = np.unique(labels)
    rng.shuffle(cluster_ids)

    test_mask = np.zeros(len(df), dtype=bool)
    val_mask  = np.zeros(len(df), dtype=bool)

    for cid in cluster_ids:
        members = np.where(labels == cid)[0]
        if test_mask.sum() / len(df) < test_frac:
            test_mask[members] = True
        elif val_mask.sum() / len(df) < val_frac:
            val_mask[members] = True

    train_mask = ~test_mask & ~val_mask
    return (
        df[train_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )


def hamming_distance_matrix(sequences: list[str]) -> np.ndarray:
    """Return an (N, N) pairwise Hamming distance matrix for a list of AA strings."""
    arr = np.array([[ord(c) for c in s] for s in sequences], dtype=np.uint8)
    n = len(arr)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i] = (arr[i] != arr).sum(axis=1)
    return dist


def cluster_sequences(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agglomerative clustering with complete linkage; returns cluster label array."""
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="complete"
    )
    return model.fit_predict(dist_matrix)
