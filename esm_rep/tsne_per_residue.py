"""tsne_per_residue.py — per-residue ESM embeddings (no mean pooling).

Companion to tsne_wt_vs_a73r.py for the per-position view. Mean-pooling
averages a single-residue mutation's effect into 1/L of the final vector,
which can dilute the signal beyond detection. This script instead works
with per-residue embeddings of three reference sequences:

  - WT
  - the A73R single mutant
  - one non-functional single mutant (auto-picked, or set via --nf-mutation)

It produces two complementary plots:

  1. tsne_per_residue.png — 2D t-SNE of all (3 * L) residue embeddings,
     with WT residues numbered every --label-step positions (default 10)
     so you can navigate the cloud, and mutated positions called out
     with stars + labels.

  2. per_position_distance.png — cosine distance between WT and each
     other sequence's residue at the same position, plotted against
     position. This is the more direct answer to "where, and how widely,
     does ESM respond to the substitution?" — a near-zero baseline with
     a spike at the mutation site, whose width measures how far attention
     propagates the perturbation.

Usage:
    python tsne_per_residue.py --config config.yaml
    python tsne_per_residue.py --config config.yaml --nf-mutation A30S
    python tsne_per_residue.py --config config.yaml --label-step 5
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_utils import load_config


# ---------------------------------------------------------------------------
# Sequence-identity palette
# ---------------------------------------------------------------------------

SEQ_COLORS = {
    "WT":   "#000000",
    "A73R": "#cb181d",
    "NF":   "#1f78b4",
}


# ---------------------------------------------------------------------------
# Pick the three reference sequences
# ---------------------------------------------------------------------------

def _parse_mutations(s: str):
    try:
        return ast.literal_eval(s) or ()
    except (ValueError, SyntaxError):
        return ()


def _format_mut(mut_tuple) -> str:
    """('A', 73, 'R') -> 'A73R'."""
    return f"{mut_tuple[0]}{int(mut_tuple[1])}{mut_tuple[2]}"


def find_three_sequences(
    df: pd.DataFrame,
    nf_mutation_arg: str | None,
) -> list[dict]:
    """Return [{name, sequence, mut_position, color_key, label}] for WT,
    A73R, and a non-functional single mutant."""
    parsed = df["mutations"].apply(_parse_mutations)

    wt_mask = df["num_mutations"] == 0
    if not wt_mask.any():
        raise ValueError("No WT found (no row with num_mutations == 0).")
    wt_row = df[wt_mask].iloc[0]

    is_a73r = parsed.apply(
        lambda t: len(t) == 1
        and t[0][0] == "A"
        and int(t[0][1]) == 73
        and t[0][2] == "R"
    )
    if not is_a73r.any():
        raise ValueError("No A73R single mutant found.")
    a73r_row = df[is_a73r].iloc[0]

    if nf_mutation_arg:
        m = re.match(r"^([A-Z])(\d+)([A-Z])$", nf_mutation_arg.strip())
        if not m:
            raise ValueError(
                f"Could not parse --nf-mutation '{nf_mutation_arg}'. "
                "Expected e.g. 'A30S'."
            )
        old_aa, pos, new_aa = m.group(1), int(m.group(2)), m.group(3)
        nf_mask = parsed.apply(
            lambda t: (
                len(t) == 1
                and t[0][0] == old_aa
                and int(t[0][1]) == pos
                and t[0][2] == new_aa
            )
        )
        if not nf_mask.any():
            raise ValueError(
                f"Mutation '{nf_mutation_arg}' not found as a single mutant."
            )
        nf_row = df[nf_mask].iloc[0]
    else:
        nf_mask = (
            (df["num_mutations"] == 1)
            & (df["activity_level"] == "non-functional")
            & parsed.apply(lambda t: len(t) == 1 and int(t[0][1]) != 73)
        )
        if not nf_mask.any():
            raise ValueError("No non-functional single mutant found.")
        nf_row = df[nf_mask].iloc[0]

    nf_label = _format_mut(parsed[nf_row.name][0])
    nf_pos = int(parsed[nf_row.name][0][1])

    return [
        {"name": "WT",
         "sequence": str(wt_row["sequence"]),
         "mut_position": None,
         "color_key": "WT",
         "label": "WT"},
        {"name": "A73R",
         "sequence": str(a73r_row["sequence"]),
         "mut_position": 73,
         "color_key": "A73R",
         "label": "A73R"},
        {"name": nf_label,
         "sequence": str(nf_row["sequence"]),
         "mut_position": nf_pos,
         "color_key": "NF",
         "label": f"{nf_label} (non-functional)"},
    ]


# ---------------------------------------------------------------------------
# Per-residue embedding extraction (no mean pooling)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def per_residue_embeddings(
    sequences: list[str],
    model_name: str,
    device: str | None = None,
    max_length: int = 1024,
) -> list[np.ndarray]:
    """Forward-pass each sequence and return per-residue hidden states.

    Returns a list of (L_i, D) arrays — <cls> and <eos> dropped, so each
    row corresponds 1:1 to a real residue.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        .to(device)
        .eval()
    )

    out = []
    for seq in tqdm(sequences, desc="Per-residue embedding"):
        enc = tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        h = model(**enc).last_hidden_state[0]  # (L+2, D)
        attn = enc["attention_mask"][0]
        last = int(attn.sum().item()) - 1
        per_res = h[1:last]  # (L, D)
        out.append(per_res.float().cpu().numpy())
    return out


# ---------------------------------------------------------------------------
# Per-position distance from WT
# ---------------------------------------------------------------------------

def compute_per_position_distances(
    per_res_list: list[np.ndarray],
    refs: list[dict],
) -> list[dict]:
    """Cosine distance between WT[i] and other_seq[i] at every position i.

    Sequences are assumed to be the same length (all single-substitutions
    of the same WT). Raises ValueError if not.
    """
    wt = per_res_list[0]
    L = wt.shape[0]
    wt_norm = np.linalg.norm(wt, axis=1, keepdims=True).clip(min=1e-12)
    wt_unit = wt / wt_norm

    out = []
    for i in range(1, len(per_res_list)):
        seq = per_res_list[i]
        if seq.shape != wt.shape:
            raise ValueError(
                f"Length mismatch: WT={wt.shape}, "
                f"{refs[i]['name']}={seq.shape}. "
                "This script assumes single-substitution variants of WT."
            )
        seq_norm = np.linalg.norm(seq, axis=1, keepdims=True).clip(min=1e-12)
        seq_unit = seq / seq_norm
        cos_sim = (wt_unit * seq_unit).sum(axis=1)
        cos_dist = 1.0 - cos_sim
        out.append({
            "name": refs[i]["name"],
            "label": refs[i]["label"],
            "color": SEQ_COLORS[refs[i]["color_key"]],
            "mut_position": refs[i]["mut_position"],
            "cos_dist": cos_dist,
            "L": L,
        })
    return out


# ---------------------------------------------------------------------------
# Plot 1: per-residue t-SNE with WT residue numbering
# ---------------------------------------------------------------------------

def plot_per_residue_tsne(
    X_2d: np.ndarray,
    refs: list[dict],
    lengths: list[int],
    out_path: Path,
    label_step: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))

    offsets = np.cumsum([0] + lengths)

    # Plot all three sequences' residues as dots — no connecting ribbon.
    for i, ref in enumerate(refs):
        start, end = offsets[i], offsets[i + 1]
        coords = X_2d[start:end]
        color = SEQ_COLORS[ref["color_key"]]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=18, c=color, alpha=0.7, edgecolors="none", zorder=2,
        )

    # WT residue numbers as navigation anchors. Stroked white outline so
    # they're readable on top of the colored dots.
    if label_step > 0:
        wt_coords = X_2d[offsets[0]:offsets[1]]
        L = lengths[0]
        for pos in range(label_step, L + 1, label_step):
            idx = pos - 1
            if 0 <= idx < L:
                xy = wt_coords[idx]
                txt = ax.text(
                    xy[0], xy[1], str(pos),
                    fontsize=7, color="#222222", zorder=6,
                    ha="center", va="center", fontweight="bold",
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=2.5, foreground="white"),
                    path_effects.Normal(),
                ])

    # Mutated positions: big star + outlined name, drawn last so they
    # sit on top of any number labels they happen to overlap.
    for i, ref in enumerate(refs):
        if ref["mut_position"] is None:
            continue
        start, end = offsets[i], offsets[i + 1]
        coords = X_2d[start:end]
        pos_idx = ref["mut_position"] - 1
        if 0 <= pos_idx < (end - start):
            color = SEQ_COLORS[ref["color_key"]]
            xy = coords[pos_idx]
            ax.scatter(
                *xy, s=380, marker="*", c=color,
                edgecolors="white", linewidths=1.8, zorder=10,
            )
            ann = ax.annotate(
                ref["label"].split(" ")[0],
                xy, xytext=(10, 10),
                textcoords="offset points",
                fontsize=11, fontweight="bold", color=color, zorder=11,
            )
            ann.set_path_effects([
                path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal(),
            ])

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=SEQ_COLORS[r["color_key"]],
               markeredgecolor="none",
               label=f"{r['label']}  (L={lengths[i]})")
        for i, r in enumerate(refs)
    ] + [
        Line2D([0], [0], marker="*", linestyle="", markersize=14,
               markerfacecolor="gray", markeredgecolor="white",
               markeredgewidth=1.0, label="mutated position"),
    ]
    if label_step > 0:
        handles.append(
            Line2D([0], [0], marker="$N$", linestyle="", markersize=10,
                   markerfacecolor="#222222", markeredgecolor="none",
                   label=f"WT residue numbers (every {label_step})")
        )
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        "Per-residue ESM-2 embeddings, t-SNE projection\n"
        "Each point is one residue. WT positions labeled as "
        "navigation anchors."
    )
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: per-position cosine distance from WT
# ---------------------------------------------------------------------------

def plot_per_position_distances(
    distance_results: list[dict],
    out_path: Path,
) -> None:
    L = distance_results[0]["L"]
    positions = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    for d in distance_results:
        ax.plot(
            positions, d["cos_dist"],
            color=d["color"], linewidth=1.5, alpha=0.9,
            label=d["label"],
        )
        if d["mut_position"] is not None:
            ax.axvline(
                d["mut_position"],
                color=d["color"], linestyle=":", linewidth=1.0, alpha=0.5,
            )
            ax.scatter(
                [d["mut_position"]],
                [d["cos_dist"][d["mut_position"] - 1]],
                s=80, marker="o", facecolors="none",
                edgecolors=d["color"], linewidths=1.8, zorder=5,
            )

    ax.set_xlabel("Residue position")
    ax.set_ylabel("Cosine distance from WT (per-residue ESM)")
    ax.set_title(
        "Per-position embedding divergence from WT\n"
        "Spike location = where ESM responds to the substitution; "
        "spike width ≈ how far attention propagates the perturbation"
    )
    ax.set_xlim(0, L + 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
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
    ap.add_argument("--nf-mutation", default=None,
                    help="Specific NF single mutant, e.g. 'A30S'. "
                         "If omitted, an NF single mutant at a position "
                         "other than 73 is auto-selected.")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-step", type=int, default=10,
                    help="Annotate every Nth WT residue with its position "
                         "number on the t-SNE plot. Set to 0 to disable.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Pick three reference sequences ------------------------------
    csv_path = cfg["data"]["path"]
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sequence", "activity_level"]).reset_index(drop=True)

    refs = find_three_sequences(df, args.nf_mutation)
    print("Reference sequences:")
    for r in refs:
        print(f"  {r['label']:<32}  L={len(r['sequence'])}  "
              f"mut_pos={r['mut_position']}")

    # ---- 2. Per-residue embeddings (no mean pooling) --------------------
    sequences = [r["sequence"] for r in refs]
    per_res_list = per_residue_embeddings(
        sequences, cfg["embedding"]["model_name"]
    )
    lengths = [arr.shape[0] for arr in per_res_list]

    # ---- 3. Per-position distance plot (cheap, do this first) -----------
    distance_results = compute_per_position_distances(per_res_list, refs)
    dist_path = out_dir / "per_position_distance.png"
    plot_per_position_distances(distance_results, dist_path)
    print(f"\nPer-position distance plot saved to {dist_path}")

    print("\nPer-position distance summary:")
    for d in distance_results:
        cd = d["cos_dist"]
        argmax = int(np.argmax(cd)) + 1
        print(f"  {d['name']:<8}  median={np.median(cd):.4f}  "
              f"max={cd.max():.4f} at position {argmax}  "
              f"(mutated at {d['mut_position']})")

    # ---- 4. t-SNE on stacked per-residue embeddings ---------------------
    X = np.concatenate(per_res_list, axis=0)
    print(f"\nStacked per-residue array: {X.shape}  "
          f"(=sum(L) x D, residues from all {len(refs)} sequences)")

    print(f"t-SNE (perplexity={args.perplexity}, n={X.shape[0]})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        n_jobs=-1,
        verbose=1,
    )
    X_2d = tsne.fit_transform(X)

    tsne_path = out_dir / "tsne_per_residue.png"
    plot_per_residue_tsne(
        X_2d, refs, lengths, tsne_path,
        label_step=args.label_step,
    )
    print(f"t-SNE plot saved to {tsne_path}")


if __name__ == "__main__":
    main()