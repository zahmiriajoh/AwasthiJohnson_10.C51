"""Per-residue ESM-2 embeddings (no pooling) for WT, A73R, A33C.

Produces a per-position cosine-distance plot and a 2D t-SNE of the
stacked per-residue embeddings.
"""

import ast
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

from data_utils import load_config

cfg = load_config("config.yaml")
out_dir = Path("output/tsne_v2_bigmodel")
out_dir.mkdir(parents=True, exist_ok=True)

# Load CSV and pick the three reference rows.
df = pd.read_csv(cfg["data"]["path"])
df = df.dropna(subset=["sequence", "activity_level"]).reset_index(drop=True)
parsed = df["mutations"].apply(lambda s: ast.literal_eval(s) or ())

wt_idx   = int(df.index[df["num_mutations"] == 0][0])
a73r_idx = int(df.index[parsed.apply(lambda t: t == (("A", 73, "R"),))][0])
a33c_idx = int(df.index[parsed.apply(lambda t: t == (("A", 33, "C"),))][0])

refs = [
    {"name": "WT",   "seq": str(df.loc[wt_idx,   "sequence"]), "color": "#000000", "mut": None},
    {"name": "A73R", "seq": str(df.loc[a73r_idx, "sequence"]), "color": "#cb181d", "mut": 73},
    {"name": "A33C", "seq": str(df.loc[a33c_idx, "sequence"]), "color": "#1f78b4", "mut": 33},
]

# Per-residue embeddings (no pooling).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {cfg['embedding']['model_name']} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(cfg["embedding"]["model_name"])
model = AutoModel.from_pretrained(
    cfg["embedding"]["model_name"], add_pooling_layer=False,
).to(device).eval()

per_res = []
with torch.inference_mode():
    for r in refs:
        enc = tokenizer(r["seq"], return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state[0]  # (L+2, D)
        last = int(enc["attention_mask"][0].sum().item()) - 1
        per_res.append(h[1:last].float().cpu().numpy())  # (L, D), drops <cls>/<eos>

L, D = per_res[0].shape
print(f"L = {L}, D = {D}")

# Per-position cosine distance from WT.
wt_unit = per_res[0] / np.linalg.norm(per_res[0], axis=1, keepdims=True).clip(min=1e-12)
dists = []
for arr in per_res[1:]:
    a_unit = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    dists.append(1.0 - (wt_unit * a_unit).sum(axis=1))

# Plot 1: per-position distance.
fig, ax = plt.subplots(figsize=(12, 5))
positions = np.arange(1, L + 1)
for i, d in enumerate(dists):
    r = refs[i + 1]
    ax.plot(positions, d, color=r["color"], linewidth=1.5, alpha=0.9, label=r["name"])
    ax.axvline(r["mut"], color=r["color"], linestyle=":", linewidth=1.0, alpha=0.5)

ax.set_xlabel("Residue position")
ax.set_ylabel("Cosine distance from WT (per-residue ESM)")
ax.set_title("Per-position embedding divergence from WT")
ax.set_xlim(0, L + 1)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=10)
plt.tight_layout()
plt.savefig(out_dir / "per_position_distance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_dir / 'per_position_distance.png'}")

for i, d in enumerate(dists):
    r = refs[i + 1]
    print(f"  {r['name']}: median={np.median(d):.4f}, max={d.max():.4f} "
          f"at position {int(np.argmax(d)) + 1} (mutated at {r['mut']})")

# # t-SNE on stacked per-residue embeddings.
# X = np.concatenate(per_res, axis=0)  # (3L, D)
# print(f"\nt-SNE on {X.shape[0]} residue embeddings...")
# X_2d = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
#             random_state=42, n_jobs=-1, verbose=1).fit_transform(X)

# # Plot 2: t-SNE.
# fig, ax = plt.subplots(figsize=(11, 9))
# for i, r in enumerate(refs):
#     coords = X_2d[i * L:(i + 1) * L]
#     ax.scatter(coords[:, 0], coords[:, 1], s=18, c=r["color"], alpha=0.7,
#                edgecolors="none", zorder=2)

# # WT residue numbers as navigation anchors (every 10).
# wt_coords = X_2d[:L]
# for pos in range(10, L + 1, 10):
#     xy = wt_coords[pos - 1]
#     txt = ax.text(xy[0], xy[1], str(pos), fontsize=7, color="#222222",
#                   zorder=6, ha="center", va="center", fontweight="bold")
#     txt.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground="white"),
#                           path_effects.Normal()])

# # Mutated positions, drawn last so they sit on top.
# for i, r in enumerate(refs):
#     if r["mut"] is None:
#         continue
#     coords = X_2d[i * L:(i + 1) * L]
#     xy = coords[r["mut"] - 1]
#     ax.scatter(*xy, s=380, marker="*", c=r["color"], edgecolors="white",
#                linewidths=1.8, zorder=10)
#     ann = ax.annotate(r["name"], xy, xytext=(10, 10), textcoords="offset points",
#                       fontsize=11, fontweight="bold", color=r["color"], zorder=11)
#     ann.set_path_effects([path_effects.Stroke(linewidth=2.5, foreground="white"),
#                           path_effects.Normal()])

# handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
#                   markerfacecolor=r["color"], markeredgecolor="none",
#                   label=f"{r['name']} (L={L})") for r in refs]
# handles += [Line2D([0], [0], marker="*", linestyle="", markersize=14,
#                    markerfacecolor="gray", markeredgecolor="white",
#                    markeredgewidth=1.0, label="mutated position")]
# ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92)
# ax.set_xlabel("t-SNE 1")
# ax.set_ylabel("t-SNE 2")
# ax.set_title("Per-residue ESM-2 embeddings t-SNE")
# ax.set_aspect("equal", adjustable="datalim")
# plt.tight_layout()
# plt.savefig(out_dir / "tsne_per_residue.png", dpi=150, bbox_inches="tight")
# plt.close()
# print(f"Saved {out_dir / 'tsne_per_residue.png'}")