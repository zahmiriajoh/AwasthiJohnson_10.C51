"""tsne_wt_vs_a73r.py — visualize WT vs A73R in ESM embedding space.

Standalone diagnostic for hypothesis H6 (writeup §3.7): are the ESM-2
embeddings of WT and the A73R single mutant distinguishable from each
other and from the rest of the landscape? If WT and A73R sit in obviously
different regions of the embedding manifold, the model has captured
*something* about the activity-relevant difference between them. If they
sit on top of each other and the activity classes are uniformly mixed,
no downstream classifier on these embeddings can separate them.

What this script does:

  1. Loads the FULL landscape (all 55,760 variants — no train/test split).
  2. Embeds every variant via embed.embed_sequences (cached on disk).
  3. PCA -> 50 dims, then t-SNE -> 2D. The 2D coords are cached separately
     so re-running with --reuse-coords skips the slow t-SNE step.
  4. Plots t-SNE colored by activity class with WT and A73R highlighted.
  5. Reports cosine and Euclidean distances between WT and A73R in the
     ORIGINAL embedding space, with percentile ranks against the
     distribution of distances from each reference to all other variants.
     (t-SNE 2D distance is not interpretable — only the original-space
     numbers answer "how different are they relative to other sequences".)

Usage:
    python tsne_wt_vs_a73r.py --config config.yaml
    python tsne_wt_vs_a73r.py --config config.yaml --reuse-coords
    python tsne_wt_vs_a73r.py --config config.yaml --perplexity 50

Outputs (under --output-dir, default ./output/):
    tsne_wt_vs_a73r.png       The plot.
    tsne_coords_<key>.npy     Cached 2D coords keyed on hyperparameters.
    tsne_distances.txt        Numerical distance summary.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from data_utils import load_config
from embed import embed_sequences


# ---------------------------------------------------------------------------
# Activity class palette — kept consistent with result_analysis.py styling.
# ---------------------------------------------------------------------------

CLASS_ORDER = [
    "non-functional",
    "activity > 0",
    "activity > WT",
    "activity > A73R",
]

CLASS_COLORS = {
    "non-functional":  "#bdbdbd",
    "activity > 0":    "#74c476",
    "activity > WT":   "#2171b5",
    "activity > A73R": "#cb181d",
}


# ---------------------------------------------------------------------------
# Reference-row identification
# ---------------------------------------------------------------------------

def _parse_mutations(s: str):
    """Parse the `mutations` column entry, e.g. \"(('A', 73, 'R'),)\" -> tuple."""
    try:
        return ast.literal_eval(s) or ()
    except (ValueError, SyntaxError):
        return ()


def find_reference_rows(df: pd.DataFrame) -> dict[str, int]:
    """Return positional indices of WT and the A73R single mutant in df.

    Raises ValueError if either reference cannot be uniquely located.
    """
    refs: dict[str, int] = {}

    wt_idx = df.index[df["num_mutations"] == 0].tolist()
    if not wt_idx:
        raise ValueError("Could not find WT (num_mutations == 0) in landscape.")
    refs["WT"] = int(wt_idx[0])

    parsed = df["mutations"].apply(_parse_mutations)
    is_a73r_single = parsed.apply(
        lambda t: len(t) == 1
        and t[0][0] == "A"
        and int(t[0][1]) == 73
        and t[0][2] == "R"
    )
    a73r_idx = df.index[is_a73r_single].tolist()
    if not a73r_idx:
        raise ValueError("Could not find A73R single mutant in landscape.")
    refs["A73R"] = int(a73r_idx[0])
    return refs


# ---------------------------------------------------------------------------
# t-SNE with on-disk cache
# ---------------------------------------------------------------------------

def _coords_cache_key(cfg: dict, n_seqs: int, pca_dim: int,
                      perplexity: float, seed: int) -> str:
    """Hash key for the t-SNE coord cache.

    Includes the embedding model name (so swapping models invalidates the
    cache), the number of sequences (so changes in landscape size invalidate
    it), and all the t-SNE hyperparameters that would change the output.
    """
    h = hashlib.sha256()
    h.update(cfg["embedding"]["model_name"].encode())
    h.update(str(n_seqs).encode())
    h.update(str(pca_dim).encode())
    h.update(f"{perplexity:.4f}".encode())
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def compute_or_load_tsne(
    X: np.ndarray,
    cache_path: Path,
    pca_dim: int,
    perplexity: float,
    seed: int,
    reuse: bool,
) -> np.ndarray:
    """PCA -> t-SNE, with on-disk cache of the 2D coords."""
    if reuse and cache_path.exists():
        print(f"Loading cached t-SNE coords: {cache_path}")
        return np.load(cache_path)

    print(f"PCA -> {pca_dim} dims...")
    pca = PCA(n_components=min(pca_dim, X.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X)
    print(f"  cumulative variance explained: "
          f"{pca.explained_variance_ratio_.sum():.3f}")

    print(f"t-SNE (perplexity={perplexity}, n={X_pca.shape[0]}). "
          "This is the slow step.")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X_pca)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X_2d)
    print(f"Saved t-SNE coords to {cache_path}")
    return X_2d


# ---------------------------------------------------------------------------
# Distance analysis in the ORIGINAL embedding space
# ---------------------------------------------------------------------------

def distance_summary(X: np.ndarray, refs: dict[str, int]) -> dict:
    """Compare the WT<->A73R distance to the distribution of distances from
    each reference to every other variant.

    Reported as percentile ranks: a low percentile means the WT<->A73R
    distance is *small* relative to typical distances from WT (so they
    look similar), high means they're far apart relative to the average
    other variant.
    """
    wt = X[refs["WT"]][None, :]
    a = X[refs["A73R"]][None, :]

    # Cosine and Euclidean distances from each reference to every variant.
    cos_from_wt = cosine_distances(wt, X).ravel()
    cos_from_a = cosine_distances(a, X).ravel()
    euc_from_wt = np.linalg.norm(X - wt, axis=1)
    euc_from_a = np.linalg.norm(X - a, axis=1)

    cos_wt_a = float(cos_from_wt[refs["A73R"]])
    euc_wt_a = float(euc_from_wt[refs["A73R"]])

    # Percentile of WT<->A73R among (WT-to-others) distances.
    others_mask = np.ones(X.shape[0], dtype=bool)
    others_mask[refs["WT"]] = False
    others_mask[refs["A73R"]] = False

    pct_cos_wt = (cos_from_wt[others_mask] < cos_wt_a).mean() * 100
    pct_cos_a = (cos_from_a[others_mask] < cos_wt_a).mean() * 100
    pct_euc_wt = (euc_from_wt[others_mask] < euc_wt_a).mean() * 100
    pct_euc_a = (euc_from_a[others_mask] < euc_wt_a).mean() * 100

    return {
        "cosine_wt_a73r": cos_wt_a,
        "euclidean_wt_a73r": euc_wt_a,
        "cosine_pct_among_wt_to_others": float(pct_cos_wt),
        "cosine_pct_among_a73r_to_others": float(pct_cos_a),
        "euclidean_pct_among_wt_to_others": float(pct_euc_wt),
        "euclidean_pct_among_a73r_to_others": float(pct_euc_a),
        "median_cos_wt_to_others": float(np.median(cos_from_wt[others_mask])),
        "median_cos_a73r_to_others": float(np.median(cos_from_a[others_mask])),
    }


def format_distance_report(summary: dict) -> str:
    return (
        "WT <-> A73R distance in original ESM embedding space\n"
        "-----------------------------------------------------\n"
        f"  cosine    : {summary['cosine_wt_a73r']:.4f}\n"
        f"  euclidean : {summary['euclidean_wt_a73r']:.4f}\n"
        "\n"
        "Percentile of this distance among distances from each reference\n"
        "to every other variant in the landscape (lower = MORE similar\n"
        "than typical, higher = MORE different than typical):\n"
        "\n"
        "  cosine, vs WT->others    : "
        f"{summary['cosine_pct_among_wt_to_others']:6.2f}th percentile\n"
        "  cosine, vs A73R->others  : "
        f"{summary['cosine_pct_among_a73r_to_others']:6.2f}th percentile\n"
        "  euclid, vs WT->others    : "
        f"{summary['euclidean_pct_among_wt_to_others']:6.2f}th percentile\n"
        "  euclid, vs A73R->others  : "
        f"{summary['euclidean_pct_among_a73r_to_others']:6.2f}th percentile\n"
        "\n"
        "Reference distributions (median cosine distance to other variants):\n"
        f"  median(WT   -> others)   : {summary['median_cos_wt_to_others']:.4f}\n"
        f"  median(A73R -> others)   : {summary['median_cos_a73r_to_others']:.4f}\n"
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_tsne(
    X_2d: np.ndarray,
    labels: np.ndarray,
    refs: dict[str, int],
    summary: dict,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))

    # Background: variants colored by activity class. Plot non-functional
    # first so it sits underneath the more interesting active classes.
    class_counts: dict[str, int] = {}
    for cls in CLASS_ORDER:
        mask = labels == cls
        n = int(mask.sum())
        class_counts[cls] = n
        if n == 0:
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=4,
            c=CLASS_COLORS[cls],
            alpha=0.45,
            edgecolors="none",
            rasterized=True,
        )

    # Reference points: WT (black star) and A73R (red star), drawn on top.
    wt_xy = X_2d[refs["WT"]]
    a_xy = X_2d[refs["A73R"]]

    ax.scatter(*wt_xy, s=420, marker="*", c="black",
               edgecolors="white", linewidths=2.0, zorder=10)
    ax.scatter(*a_xy, s=420, marker="*", c="#cb181d",
               edgecolors="white", linewidths=2.0, zorder=10)

    ax.annotate("WT", wt_xy, xytext=(10, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color="black")
    ax.annotate("A73R", a_xy, xytext=(10, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color="#cb181d")

    # Build proxy handles so the legend isn't dominated by the giant star
    # markers used in the data layer.
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=CLASS_COLORS[cls], markeredgecolor="none",
               label=f"{cls} (n={class_counts[cls]})")
        for cls in CLASS_ORDER if class_counts[cls] > 0
    ] + [
        Line2D([0], [0], marker="*", linestyle="", markersize=14,
               markerfacecolor="black", markeredgecolor="white",
               markeredgewidth=1.0, label="WT"),
        Line2D([0], [0], marker="*", linestyle="", markersize=14,
               markerfacecolor="#cb181d", markeredgecolor="white",
               markeredgewidth=1.0, label="A73R single mutant"),
    ]

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "ESM-2 embeddings, t-SNE projection — WT vs A73R in landscape context\n"
       # f"WT<->A73R cosine distance = {summary['cosine_wt_a73r']:.4f} "
       # f"({summary['cosine_pct_among_wt_to_others']:.1f}th pct of WT->others)"
    )
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n")[0],
    )
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--pca-dim", type=int, default=50,
                    help="PCA dim before t-SNE (50 is the standard choice)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-coords", action="store_true",
                    help="Skip t-SNE and load coords from the matching cache")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load the FULL landscape (no train/test split) ----------------
    csv_path = cfg["data"]["path"]
    seq_col = cfg["data"]["sequence_col"]
    target_col = cfg["data"]["target_col"]
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[seq_col, target_col]).reset_index(drop=True)
    sequences = df[seq_col].astype(str).tolist()
    labels = df[target_col].astype(str).to_numpy()
    print(f"  {len(df)} variants total")

    refs = find_reference_rows(df)
    print(f"  WT   row index: {refs['WT']:6d}  (label={labels[refs['WT']]})")
    print(f"  A73R row index: {refs['A73R']:6d}  (label={labels[refs['A73R']]})")

    # ---- 2. Embed every variant (cached by embed.py) ---------------------
    print("\nEmbedding (will hit cache if previously computed in this order)...")
    X = embed_sequences(sequences, cfg)
    print(f"  Embedding shape: {X.shape}")

    # ---- 3. PCA -> t-SNE, with coord cache -------------------------------
    coord_key = _coords_cache_key(
        cfg, len(sequences), args.pca_dim, args.perplexity, args.seed
    )
    coord_cache = out_dir / f"tsne_coords_{coord_key}.npy"
    X_2d = compute_or_load_tsne(
        X, coord_cache, args.pca_dim, args.perplexity, args.seed,
        reuse=args.reuse_coords,
    )

    # ---- 4. Distance analysis in ORIGINAL embedding space ----------------
    print("\nComputing reference-distance statistics in embedding space...")
    summary = distance_summary(X, refs)
    report = format_distance_report(summary)
    print("\n" + report)
    (out_dir / "tsne_distances.txt").write_text(report)
    (out_dir / "tsne_distances.json").write_text(json.dumps(summary, indent=2))

    # ---- 5. Plot ---------------------------------------------------------
    plot_path = out_dir / "tsne_wt_vs_a73r.png"
    plot_tsne(X_2d, labels, refs, summary, plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()