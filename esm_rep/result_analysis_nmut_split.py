"""result_analysis_nmut_split.py — analysis tailored to the mutation-count split.

Companion to result_analysis.py, designed for the setting in which the
classifier was trained on simple variants (e.g. <=2 mutations from WT) and
evaluated on more complex ones (>2 mutations). This is the same train/test
protocol used by Thomas et al. (2025) for their CNN activity classifier
baseline (see Methods: 'CNN activity classifier' and Table S5).

Outputs are written to <output-dir>/<subdir>/ — defaults to ./output/nmut_split/.

Plots produced:
  1. Overall confusion matrix on the test set.
  2. Confusion matrix stratified by mutation count of the test variant
     (3, 4-5, 6-10, 11-15, 16+; the 1- and 2-mutation buckets are training).
  3. Confusion matrix stratified by minimum linear distance between any
     two mutated positions (1, 2-5, 6-15, 16+), applied only to variants
     with >= 3 mutations — tests whether dense-mutation variants suffer
     more than spread-out ones.
  4. Hits@1000 at two activity thresholds (>WT, >A73R) — the precision-at-k
     metric the paper uses for ranking-quality comparisons (Table S5).
  5. A train/test class-distribution plot, broken down by mutation count.
     Visually shows the OOD-class problem (higher-activity classes don't
     appear in training).

Usage:
    python result_analysis_nmut_split.py --config config.yaml
    python result_analysis_nmut_split.py --config config.yaml \\
        --output-dir results --subdir nmut_split
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
from sklearn.preprocessing import StandardScaler

from data_utils import load_and_split, load_config
from embed import embed_sequences


# ---------------------------------------------------------------------------
# Bin definitions — edit to change stratification
# ---------------------------------------------------------------------------

# Mutation-count bins for the test set. Excludes 1 and 2 because those are
# the training distribution under the <=2 / >2 split.
MUTATION_COUNT_BINS = [
    ("3",     lambda n: n == 3),
    ("4-5",   lambda n: 4 <= n <= 5),
    ("6-10",  lambda n: 6 <= n <= 10),
    ("11-15", lambda n: 11 <= n <= 15),
    ("16+",   lambda n: n >= 16),
]

MIN_DISTANCE_BINS = [
    ("1",    lambda d: d == 1),
    ("2-5",  lambda d: 2 <= d <= 5),
    ("6-15", lambda d: 6 <= d <= 15),
    ("16+",  lambda d: d >= 16),
]

# Threshold names used in hits@k. These correspond to the paper's Table S5
# columns (hits@1000_wt, hits@1000_A73R). The class-name strings must match
# what's in the activity_level column. The released landscape.csv has 4
# classes: 'non-functional', 'activity > 0', 'activity > WT', 'activity > A73R'.
HITS_AT_K = 1000
HIT_THRESHOLDS = {
    # name : set of class labels that count as a "hit" at this threshold
    "wt":   {"activity > WT", "activity > A73R"},
    "a73r": {"activity > A73R"},
}

# Reference values from the paper's Table S5 (CNN classifier baselines).
# Used to draw horizontal reference lines on the hits@k bar chart, matched
# per-bar to the same threshold name.
PAPER_BASELINES = {
    "wt":   {"LR": 0.44, "CNN": 0.82},
    "a73r": {"LR": 0.09, "CNN": 0.21},
}


# ---------------------------------------------------------------------------
# Mutation parsing — same logic as result_analysis.py
# ---------------------------------------------------------------------------

def parse_positions(mutations_cell) -> list[int]:
    """Extract integer positions from a `mutations` cell.

    The cell format is a stringified tuple of (wt_aa, position, mut_aa)
    triples, e.g. "(('A', 30, 'S'),)" or "(('A', 30, 'S'), ('L', 67, 'P'))".
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


def safe_filename(label: str) -> str:
    return label.replace("+", "plus").replace(" ", "_").replace(",", "_")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Row-normalized CM heatmap with both recall fraction and raw count."""
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
    print(f"  saved {out_path.name}")


def plot_hits_at_k(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    out_path: Path,
    k: int = HITS_AT_K,
    thresholds: dict[str, set[str]] = HIT_THRESHOLDS,
) -> dict[str, float]:
    """Compute and bar-plot hits@k for each named threshold.

    For each threshold, score every test variant by the sum of the model's
    predicted probabilities on the classes that count as a hit. Take the
    top-k by that score and report the fraction that are real hits. This
    is the same metric reported in Table S5 of Thomas et al. (2025).
    """
    results = {}
    n_test = len(y_true)
    k_eff = min(k, n_test)

    for name, hit_classes in thresholds.items():
        # Score = P(any hit class). If none of the hit classes are in the
        # model's known classes, skip — model can't possibly score them.
        relevant_idx = [i for i, c in enumerate(classes) if c in hit_classes]
        if not relevant_idx:
            print(f"  hits@{k}_{name}: no relevant classes in model output, skipping")
            continue
        score = proba[:, relevant_idx].sum(axis=1)
        # Top-k indices
        top_idx = np.argsort(-score)[:k_eff]
        is_hit = np.array([y_true[i] in hit_classes for i in top_idx])
        n_hits = int(is_hit.sum())
        # Total hits available in test set (for context)
        n_total_hits = int(np.array([y in hit_classes for y in y_true]).sum())
        results[name] = {
            "hits_at_k": n_hits / k_eff if k_eff else 0.0,
            "n_hits_in_topk": n_hits,
            "n_total_hits": n_total_hits,
            "k_used": k_eff,
        }

    if not results:
        return {}

    # Bar chart
    names = list(results.keys())
    vals = [results[n]["hits_at_k"] for n in names]
    fig, ax = plt.subplots(figsize=(4 + 0.6 * len(names), 4))
    bars = ax.bar(names, vals, color=["#3b7dd8", "#d8893b"][: len(names)])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Hits@{k_eff} (precision-at-k)")
    ax.set_title(f"Ranking quality at top-{k_eff}")
    for bar, name, v in zip(bars, names, vals):
        r = results[name]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.3f}\n({r['n_hits_in_topk']}/{r['k_used']})",
            ha="center", va="bottom", fontsize=10,
        )

    # Per-bar paper reference lines (Table S5). Drawn as short horizontal
    # ticks above each bar so each baseline is unambiguously paired with
    # its threshold rather than spanning the whole axis.
    for bar, name in zip(bars, names):
        if name not in PAPER_BASELINES:
            continue
        x_left = bar.get_x()
        x_right = x_left + bar.get_width()
        for label, ref in PAPER_BASELINES[name].items():
            ls = "--" if label == "LR" else ":"
            ax.hlines(ref, x_left, x_right, color="gray", ls=ls, lw=1)
            ax.text(
                x_right + 0.02,
                ref,
                f"paper {label}: {ref:.2f}",
                color="gray", fontsize=8, va="center",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return results


def plot_class_distribution(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    out_path: Path,
    n_mut_col: str = "num_mutations",
) -> None:
    """Stacked bars showing class distribution across mutation counts.

    Two side-by-side panels (train, test) make the OOD-class problem visible
    at a glance: classes that are absent from train but present in test will
    show up as colored bars with zero counterpart in the train panel.

    Stack order is least-to-most active from bottom to top: non-functional
    sits at the bottom, then activity > 0, > WT, > A73R.
    """
    # Explicit ordinal order (least -> most active), bottom of stack first.
    ORDINAL_ORDER = [
        "non-functional",
        "activity > 0",
        "activity > WT",
        "activity > A73R",
    ]
    present = set(df_train[target_col]) | set(df_test[target_col])
    all_classes = [c for c in ORDINAL_ORDER if c in present]
    # Append any unexpected labels at the end so we don't silently drop them.
    extras = sorted(present - set(ORDINAL_ORDER))
    if extras:
        print(f"  WARNING: unexpected activity labels {extras} appended to stack")
        all_classes.extend(extras)
    # Choose mutation-count bins that cover both panels
    max_mut = int(max(df_train[n_mut_col].max(), df_test[n_mut_col].max()))
    edges = [0, 1, 2, 3, 5, 10, 15, max(20, max_mut + 1)]
    edges = sorted(set(e for e in edges if e <= max_mut + 1))
    bin_labels = [
        f"{edges[i]}" if edges[i + 1] - edges[i] == 1 else f"{edges[i]}-{edges[i + 1] - 1}"
        for i in range(len(edges) - 1)
    ]
    bin_labels[-1] = f"{edges[-2]}+"

    def counts(df):
        out = np.zeros((len(bin_labels), len(all_classes)), dtype=int)
        nm = df[n_mut_col].astype(int).values
        cls = df[target_col].astype(str).values
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (nm >= lo) & (nm < hi)
            for j, c in enumerate(all_classes):
                out[i, j] = int((cls[mask] == c).sum())
        return out

    cnt_tr, cnt_te = counts(df_train), counts(df_test)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(all_classes)))
    for ax, cnt, name in [(axes[0], cnt_tr, "Train"), (axes[1], cnt_te, "Test")]:
        bottom = np.zeros(len(bin_labels))
        for j, c in enumerate(all_classes):
            ax.bar(bin_labels, cnt[:, j], bottom=bottom, color=cmap[j], label=c)
            bottom += cnt[:, j]
        ax.set_xlabel("Mutations from WT")
        ax.set_ylabel("Variant count")
        ax.set_title(f"{name} pool (n = {int(cnt.sum())})")
        ax.tick_params(axis="x", rotation=0)
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Activity class",
        fontsize=8,
    )
    fig.suptitle("Class distribution by mutation count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Pipeline replay
# ---------------------------------------------------------------------------

def get_predictions(cfg: dict):
    """Replay split + embed + fit; return all the bits needed for plotting."""
    seq_col = cfg["data"]["sequence_col"]
    target_col = cfg["data"]["target_col"]

    df_train, df_test, _ = load_and_split(cfg)
    if df_test is None:
        raise ValueError(
            "This script requires an explicit train/test split (e.g. "
            "split.test_set='complement'); got 'random_holdout' instead."
        )
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

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
    print(f"Best C per class: {clf.C_}")

    Xte_s = scaler.transform(X_te)
    y_pred = clf.predict(Xte_s)
    proba = clf.predict_proba(Xte_s)
    classes = list(clf.classes_)
    return df_train, df_test, y_te, y_pred, proba, classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--subdir", default="nmut_split_unbalanced_bigmodel")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}/")

    df_train, df_test, y_true, y_pred, proba, classes = get_predictions(cfg)
    # All labels seen anywhere — used as a stable axis order for CMs that
    # may include OOD classes in test.
    all_labels = sorted(set(y_true) | set(y_pred) | set(classes))
    print(f"Class labels (test ∪ predicted ∪ trained): {all_labels}")

    target_col = cfg["data"]["target_col"]

    # ---- 1. Overall confusion matrix --------------------------------------
    print("\n[1/5] Overall confusion matrix:")
    plot_confusion_heatmap(
        y_true, y_pred, all_labels,
        title="Overall test set",
        out_path=out_dir / "confusion_matrix_overall.png",
    )

    # ---- 2. Stratified by mutation count of the test variant --------------
    print("\n[2/5] Stratified by mutation count:")
    if "num_mutations" not in df_test.columns:
        print("  'num_mutations' column missing — skipping")
    else:
        nmut = df_test["num_mutations"].astype(int).to_numpy()
        for label, predicate in MUTATION_COUNT_BINS:
            mask = np.array([predicate(int(n)) for n in nmut])
            if mask.sum() == 0:
                print(f"  skip: {label} mutations has 0 examples")
                continue
            plot_confusion_heatmap(
                y_true[mask], y_pred[mask], all_labels,
                title=f"Mutation count: {label}",
                out_path=out_dir / f"confusion_matrix_mut_{safe_filename(label)}.png",
            )

    # ---- 3. Stratified by min linear distance (>=3 mutations only) --------
    print("\n[3/5] Stratified by min linear distance (>=3 mutations):")
    if "mutations" not in df_test.columns or "num_mutations" not in df_test.columns:
        print("  required columns missing — skipping")
    else:
        nmut = df_test["num_mutations"].astype(int).to_numpy()
        muts_col = df_test["mutations"].to_numpy()
        min_dists = np.full(len(df_test), -1, dtype=int)
        for i in np.where(nmut >= 3)[0]:
            positions = parse_positions(muts_col[i])
            if len(positions) >= 2:
                min_dists[i] = min_linear_distance(positions)
        n_parsed = int((min_dists >= 0).sum())
        n_high = int((nmut >= 3).sum())
        print(f"  {n_parsed} of {n_high} variants with >=3 mutations parsed")
        for label, predicate in MIN_DISTANCE_BINS:
            mask = np.array([
                d >= 0 and predicate(int(d)) for d in min_dists
            ])
            if mask.sum() == 0:
                print(f"  skip: distance {label} has 0 examples")
                continue
            plot_confusion_heatmap(
                y_true[mask], y_pred[mask], all_labels,
                title=f"≥3 mutations, min linear distance: {label}",
                out_path=out_dir / f"confusion_matrix_dist_{safe_filename(label)}.png",
            )

    # ---- 4. Hits@1000 at thresholds (paper Table S5 metric) ---------------
    print("\n[4/5] Hits@k metric (matches paper Table S5):")
    hits_results = plot_hits_at_k(
        y_true, proba, classes,
        out_path=out_dir / "hits_at_k.png",
    )
    if hits_results:
        print("\n  Detailed hits@k breakdown:")
        for name, r in hits_results.items():
            print(f"    hits@{r['k_used']}_{name}: {r['hits_at_k']:.4f} "
                  f"({r['n_hits_in_topk']}/{r['k_used']} in top-k; "
                  f"{r['n_total_hits']} total hits in test set)")

    # ---- 5. Class distribution by mutation count, train vs test -----------
    print("\n[5/5] Class distribution by mutation count:")
    plot_class_distribution(
        df_train, df_test, target_col,
        out_path=out_dir / "class_distribution.png",
    )

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()