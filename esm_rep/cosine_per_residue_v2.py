"""Per-residue ESM-2 embeddings (no pooling) for WT, A73R, A33C.

Produces a per-position cosine-distance plot.
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
out_dir = Path("output/tsne_v2")
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

# Plot per-position distance.
fig, ax = plt.subplots(figsize=(12, 5))
positions = np.arange(1, L + 1)
for i, d in enumerate(dists):
    r = refs[i + 1]
    ax.plot(positions, d, color=r["color"], linewidth=1.5, alpha=0.9, label=r["name"])
    ax.axvline(r["mut"], color=r["color"], linestyle=":", linewidth=1.0, alpha=0.5)

ax.set_xlabel("Residue position")
ax.set_ylabel("Cosine distance from WT (per residue)")
ax.set_title("Per-residue embedding divergence from WT (ESM-2, 8M parameters)")
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
