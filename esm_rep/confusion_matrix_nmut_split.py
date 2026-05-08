"""Result analysis for the mutation-count split (train: ≤2 mutations, test: >2).

Outputs to ./output/nmut_split/:
  - confusion_matrix_overall.png
  - confusion_matrix_mut_<bin>.png        (stratified by test mutation count)
  - confusion_matrix_dist_<bin>.png       (stratified by min linear distance,
                                           ≥3-mutation variants only)
  - hits_at_k.png                         (matches paper Table S5)
"""

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
HITS_AT_K = 1000
HIT_THRESHOLDS = {
    "wt":   {"activity > WT", "activity > A73R"},
    "a73r": {"activity > A73R"},
}
ORDINAL_ORDER = [
    "non-functional", "activity > 0", "activity > WT", "activity > A73R",
]


def plot_cm(y_true, y_pred, labels, title, out_path):
    """Row-normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = int(cm.sum())
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums > 0,
                        out=np.zeros_like(cm, dtype=float))

    k = len(labels)
    fig, ax = plt.subplots(figsize=(1.5 + 0.9 * k, 1.2 + 0.7 * k))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\n(n = {n})")
    for i in range(k):
        for j in range(k):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# -----------------------------------------------------------------------------
# Train + predict
# -----------------------------------------------------------------------------

cfg = load_config("config.yaml")
out_dir = Path("output") / "nmut_split_big_final"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Writing outputs to {out_dir}/")

seq_col = cfg["data"]["sequence_col"]
target_col = cfg["data"]["target_col"]
df_train, df_test = load_and_split(cfg)
print(f"Train: {len(df_train)} | Test: {len(df_test)}")

sequences = (df_train[seq_col].astype(str).tolist()
             + df_test[seq_col].astype(str).tolist())
X = embed_sequences(sequences, cfg)
n_train = len(df_train)
y_tr = df_train[target_col].astype(str).to_numpy()
y_te = df_test[target_col].astype(str).to_numpy()

clf_cfg = cfg["classifier"]
scaler = StandardScaler().fit(X[:n_train])
clf = LogisticRegressionCV(
    Cs=list(clf_cfg["Cs"]), cv=clf_cfg["cv_folds"],
    max_iter=clf_cfg["max_iter"], n_jobs=-1,
    scoring=clf_cfg["scoring"], class_weight=clf_cfg["class_weight"],
).fit(scaler.transform(X[:n_train]), y_tr)
print(f"Best C per class: {clf.C_}")

X_te_s = scaler.transform(X[n_train:])
y_pred = clf.predict(X_te_s)
proba = clf.predict_proba(X_te_s)
classes = list(clf.classes_)
all_labels = sorted(set(y_te) | set(y_pred) | set(classes))

# -----------------------------------------------------------------------------
# 1. Overall confusion matrix
# -----------------------------------------------------------------------------

print("\n[1/5] Overall confusion matrix:")
plot_cm(y_te, y_pred, all_labels, "ESM-2, 35M parameters, overall test set",
        out_dir / "confusion_matrix_overall.png")

# -----------------------------------------------------------------------------
# 2. Stratified by test-variant mutation count
# -----------------------------------------------------------------------------

print("\n[2/5] Stratified by mutation count:")
nmut = df_test["num_mutations"].astype(int).to_numpy()
for label, predicate in MUTATION_COUNT_BINS:
    mask = np.array([predicate(int(n)) for n in nmut])
    if mask.sum() == 0:
        continue
    plot_cm(y_te[mask], y_pred[mask], all_labels,
            f"ESM-2, 35M parameters, mutation count: {label}",
            out_dir / f"confusion_matrix_mut_{label.replace('+', 'plus')}.png")

# -----------------------------------------------------------------------------
# 3. Stratified by min linear distance between mutations (≥3 mut only)
# -----------------------------------------------------------------------------

print("\n[3/5] Stratified by min linear distance (≥3 mutations):")
muts_col = df_test["mutations"].to_numpy()
min_dists = np.full(len(df_test), -1, dtype=int)
for i in np.where(nmut >= 3)[0]:
    positions = sorted(t[1] for t in ast.literal_eval(muts_col[i]))
    min_dists[i] = min(b - a for a, b in zip(positions[:-1], positions[1:]))

for label, predicate in MIN_DISTANCE_BINS:
    mask = np.array([d >= 0 and predicate(int(d)) for d in min_dists])
    if mask.sum() == 0:
        continue
    plot_cm(y_te[mask], y_pred[mask], all_labels,
            f"ESM-2, 35M parameters, ≥3 mutations, min linear distance: {label}",
            out_dir / f"confusion_matrix_dist_{label.replace('+', 'plus')}.png")

# -----------------------------------------------------------------------------
# 4. Hits@1000 at >WT and >A73R thresholds (paper Table S5 metric)
# -----------------------------------------------------------------------------

print("\n[4/5] Hits@k:")
k_eff = min(HITS_AT_K, len(y_te))
hits = {}
for name, hit_classes in HIT_THRESHOLDS.items():
    relevant = [i for i, c in enumerate(classes) if c in hit_classes]
    if not relevant:
        continue
    score = proba[:, relevant].sum(axis=1)
    top_idx = np.argsort(-score)[:k_eff]
    n_hits = int(sum(y_te[i] in hit_classes for i in top_idx))
    n_total = int(sum(y in hit_classes for y in y_te))
    hits[name] = (n_hits / k_eff, n_hits, n_total)
    print(f"  hits@{k_eff}_{name}: {n_hits/k_eff:.4f} "
          f"({n_hits}/{k_eff} in top-k, {n_total} total hits in test)")

if hits:
    fig, ax = plt.subplots(figsize=(4 + 0.6 * len(hits), 4))
    names = list(hits.keys())
    vals = [hits[n][0] for n in names]
    bars = ax.bar(names, vals, color=["#3b7dd8", "#d8893b"][:len(names)])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Hits@{k_eff}")
    ax.set_title(f"ESM-2, 35M parameters, top {k_eff} hits in test set")
    for bar, name in zip(bars, names):
        v, n_hits, _ = hits[name]
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}\n({n_hits}/{k_eff})",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "hits_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved hits_at_k.png")

print(f"\nAll outputs in {out_dir}/")