"""t-SNE of delta-from-WT ESM features for the full landscape,
with WT, A73R, A33C highlighted. Companion to tsne_wt_vs_a73r.py
which does the same on raw mean-pooled features.

Reuses compute_delta_features from train_delta.py, so the cache hits
whatever train_delta.py already computed (or this script is the first
caller and seeds the cache).
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from data_utils import load_config
from train_delta import compute_delta_features

CLASS_ORDER = ["non-functional", "activity > 0", "activity > WT", "activity > A73R"]
CLASS_COLORS = {
    "non-functional":  "#bdbdbd",
    "activity > 0":    "#74c476",
    "activity > WT":   "#2171b5",
    "activity > A73R": "#cb181d",
}

cfg = load_config("config.yaml")
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

# Load full landscape and pick the three reference rows.
df = pd.read_csv(cfg["data"]["path"])
df = df.dropna(subset=["sequence", "activity_level"]).reset_index(drop=True)
parsed = df["mutations"].apply(lambda s: ast.literal_eval(s) or ())

wt_idx   = int(df.index[df["num_mutations"] == 0][0])
a73r_idx = int(df.index[parsed.apply(lambda t: t == (("A", 73, "R"),))][0])
a33c_idx = int(df.index[parsed.apply(lambda t: t == (("A", 33, "C"),))][0])

refs = [
    ("WT",   wt_idx,   "#000000"),
    ("A73R", a73r_idx, "#cb181d"),
    ("A33C", a33c_idx, "#BF00FF"),
]
print(f"WT row {wt_idx}, A73R row {a73r_idx}, A33C row {a33c_idx}")

# Compute delta features for all variants (cached on disk).
sequences = df["sequence"].astype(str).tolist()
labels = df["activity_level"].astype(str).to_numpy()
wt_seq = sequences[wt_idx]
X = compute_delta_features(sequences, wt_seq, cfg)
print(f"Delta-feature shape: {X.shape}")

# PCA -> 50 -> t-SNE.
print("PCA -> 50 dims, then t-SNE...")
X_pca = PCA(n_components=50, random_state=42).fit_transform(X)
X_2d = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
            random_state=42, n_jobs=-1, verbose=1).fit_transform(X_pca)
np.save(out_dir / "tsne_delta_coords.npy", X_2d)

# Pairwise cosine distances among the three references.
print("\nCosine distances in delta-feature space:")
for i, (n_i, idx_i, _) in enumerate(refs):
    for n_j, idx_j, _ in refs[i + 1:]:
        d = float(cosine_distances(X[idx_i:idx_i + 1], X[idx_j:idx_j + 1])[0, 0])
        print(f"  {n_i:<4} <-> {n_j:<4}: {d:.4f}")

# Plot.
fig, ax = plt.subplots(figsize=(11, 9))
for cls in CLASS_ORDER:
    m = labels == cls
    ax.scatter(X_2d[m, 0], X_2d[m, 1], s=4, c=CLASS_COLORS[cls],
               alpha=0.45, edgecolors="none", rasterized=True)

for name, idx, color in refs:
    xy = X_2d[idx]
    ax.scatter(*xy, s=420, marker="*", c=color, edgecolors="white",
               linewidths=2.0, zorder=10)
    ax.annotate(name, xy, xytext=(10, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color=color)

handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                  markerfacecolor=CLASS_COLORS[c], markeredgecolor="none",
                  label=f"{c} (n={int((labels == c).sum())})")
           for c in CLASS_ORDER]
handles += [Line2D([0], [0], marker="*", linestyle="", markersize=14,
                   markerfacecolor=color, markeredgecolor="white",
                   markeredgewidth=1.0, label=name)
            for name, _, color in refs]

ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("Max-abs delta-from-WT ESM features — full landscape t-SNE")
ax.set_aspect("equal", adjustable="datalim")
plt.tight_layout()
plt.savefig(out_dir / "tsne_delta_landscape.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_dir / 'tsne_delta_landscape.png'}")