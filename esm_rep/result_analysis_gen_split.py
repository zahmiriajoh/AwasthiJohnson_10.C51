"""result_analysis.py — stratified confusion matrices for the trained model.

Standalone: replays the same load → split → embed → fit pipeline as train.py
(reusing the embedding cache, so it's fast on a second run) and writes
heatmap PNGs to ./output/ stratified by:

  - overall test set
  - mutation count: 1, 2, 3-5, 6-10, 11-15, 16+
  - minimum linear distance between mutated positions
    (only for variants with >= 3 mutations)

Usage:
    python result_analysis.py --config config.yaml

Bin definitions are at the top of the file — edit there to change them.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utils import load_and_split, load_config
from embed import embed_sequences


# ---------------------------------------------------------------------------
# Bin definitions — edit these to change the stratification.
# Each entry: (display_label, predicate(int) -> bool). Predicates are checked
# in order; first match wins, so make sure they're disjoint.
# ---------------------------------------------------------------------------

MUTATION_COUNT_BINS = [
    ("1",     lambda n: n == 1),
    ("2",     lambda n: n == 2),
    ("3-5",   lambda n: 3 <= n <= 5),
    ("6-10",  lambda n: 6 <= n <= 10),
    ("11-15", lambda n: 11 <= n <= 15),
    ("16+",   lambda n: n >= 16),
]

# Bins for the minimum linear (sequence) distance between any two mutated
# residues, applied only to variants with >= 3 mutations.
MIN_DISTANCE_BINS = [
    ("1",    lambda d: d == 1),
    ("2-5",  lambda d: 2 <= d <= 5),
    ("6-15", lambda d: 6 <= d <= 15),
    ("16+",  lambda d: d >= 16),
]


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------

# The `mutations` column is a stringified tuple of (wt_aa, position, mut_aa)
# triples, e.g. "(('A', 30, 'S'),)" or "(('A', 30, 'S'), ('L', 67, 'P'))".


def parse_positions(mutations_cell) -> list[int]:
    """Extract integer positions from a `mutations` cell.

    Returns [] for NaN/empty/wildtype/unparseable cells.
    """
    if pd.isna(mutations_cell):
        return []
    s = str(mutations_cell).strip()
    if not s or s.lower() in {"wt", "wildtype", "none", "()", "(,)"}:
        return []
    try:
        obj = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(obj, (tuple, list)):
        return []
    positions = []
    for item in obj:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            # (wt_aa, position, mut_aa) — position is the 2nd element.
            try:
                positions.append(int(item[1]))
            except (TypeError, ValueError):
                continue
    return positions


def min_linear_distance(positions: list[int]) -> int | None:
    """Smallest |p_i - p_j| over all i != j; None if fewer than 2 positions."""
    if len(positions) < 2:
        return None
    sorted_p = sorted(positions)
    return min(b - a for a, b in zip(sorted_p, sorted_p[1:]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Row-normalized CM heatmap with both the recall fraction and the raw
    count annotated in each cell. Rows with zero examples stay blank.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = int(cm.sum())
    if n == 0:
        print(f"  skip: {title} has 0 examples")
        return

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums,
        where=row_sums > 0,
        out=np.zeros_like(cm, dtype=float),
    )

    k = len(labels)
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * k, 1.2 + 0.7 * k))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\n(n = {n})")

    for i in range(k):
        for j in range(k):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center", fontsize=9, color=color,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def safe_filename(label: str) -> str:
    """Make a bin label safe for filenames: '16+' -> '16plus', etc."""
    return label.replace("+", "plus").replace(" ", "_")


# ---------------------------------------------------------------------------
# Pipeline replay
# ---------------------------------------------------------------------------

def get_predictions(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Replay split + embed + fit; return (df_test, y_true, y_pred, labels)."""
    seed = cfg.get("seed", 42)
    seq_col = cfg["data"]["sequence_col"]
    target_col = cfg["data"]["target_col"]

    df_train, df_test, _ = load_and_split(cfg)

    # Mirror train.py's behaviour for the random_holdout case.
    if df_test is None:
        test_size = cfg["split"].get("test_size", 0.2)
        labels_full = df_train[target_col].astype(str).to_numpy()
        idx = np.arange(len(df_train))
        idx_tr, idx_te = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=labels_full if len(set(labels_full)) > 1 else None,
        )
        df_test = df_train.iloc[idx_te].reset_index(drop=True)
        df_train = df_train.iloc[idx_tr].reset_index(drop=True)
        print(f"Random holdout: train={len(df_train)}, test={len(df_test)}")
    else:
        print(f"Generation split: train={len(df_train)}, test={len(df_test)}")

    sequences = (
        df_train[seq_col].astype(str).tolist()
        + df_test[seq_col].astype(str).tolist()
    )
    X = embed_sequences(sequences, cfg)
    n_train = len(df_train)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr = df_train[target_col].astype(str).to_numpy()
    y_te = df_test[target_col].astype(str).to_numpy()

    c = cfg["classifier"]
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegressionCV(
        Cs=list(c["Cs"]),
        cv=c["cv_folds"],
        max_iter=c["max_iter"],
        n_jobs=-1,
        scoring=c["scoring"],
        class_weight=c["class_weight"],
    ).fit(scaler.transform(X_tr), y_tr)

    y_pred = clf.predict(scaler.transform(X_te))
    labels = sorted(set(y_tr) | set(y_te))
    return df_test, y_te, y_pred, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-dir", default="output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_test, y_true, y_pred, labels = get_predictions(cfg)
    print(f"Class labels: {labels}")

    # --- Overall confusion matrix --------------------------------------
    print("\nOverall:")
    plot_confusion_heatmap(
        y_true, y_pred, labels,
        title="Overall test set",
        out_path=out_dir / "confusion_matrix_overall.png",
    )

    # --- Per mutation-count bucket -------------------------------------
    if "num_mutations" not in df_test.columns:
        print("\n'num_mutations' column not found — skipping mutation-count plots.")
    else:
        # Coerce to int, treating NaN as -1 so it won't match any bin.
        num_mut = df_test["num_mutations"].to_numpy()
        num_mut_int = np.array([
            int(n) if not pd.isna(n) else -1 for n in num_mut
        ])

        print("\nBy mutation count:")
        for label, predicate in MUTATION_COUNT_BINS:
            mask = np.array([predicate(int(n)) for n in num_mut_int])
            if mask.sum() == 0:
                print(f"  skip: mutation count {label} has 0 examples")
                continue
            plot_confusion_heatmap(
                y_true[mask], y_pred[mask], labels,
                title=f"Mutation count: {label}",
                out_path=out_dir / f"confusion_matrix_mut_{safe_filename(label)}.png",
            )

    # --- Per minimum-linear-distance bucket (for >= 3 mutations) -------
    if "mutations" not in df_test.columns:
        print("\n'mutations' column not found — skipping linear-distance plots.")
        return
    if "num_mutations" not in df_test.columns:
        return

    high_mut_mask = num_mut_int >= 3
    mutations_col = df_test["mutations"].to_numpy()

    # Only compute distances where we have >= 3 mutations.
    min_dists = np.full(len(df_test), -1, dtype=int)
    parse_failures = 0
    for i in np.where(high_mut_mask)[0]:
        positions = parse_positions(mutations_col[i])
        if len(positions) < 2:
            parse_failures += 1
            continue
        min_dists[i] = min_linear_distance(positions)

    valid = min_dists >= 0
    print(f"\nVariants with >=3 mutations: {high_mut_mask.sum()} "
          f"({valid.sum()} parseable; {parse_failures} unparseable).")

    if valid.sum() > 0:
        print("By minimum linear distance between mutated positions:")
        for label, predicate in MIN_DISTANCE_BINS:
            bin_mask = valid & np.array([
                predicate(int(d)) if d >= 0 else False for d in min_dists
            ])
            if bin_mask.sum() == 0:
                print(f"  skip: min distance {label} has 0 examples")
                continue
            plot_confusion_heatmap(
                y_true[bin_mask], y_pred[bin_mask], labels,
                title=f"≥3 mutations, min linear distance: {label}",
                out_path=out_dir / f"confusion_matrix_dist_{safe_filename(label)}.png",
            )

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()