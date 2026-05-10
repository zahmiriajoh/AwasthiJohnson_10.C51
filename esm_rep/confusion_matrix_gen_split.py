"""stratified confusion matrices for the trained model.

Replays load → split → embed → fit (reusing the embedding cache) and
writes heatmap PNGs stratified by:

  - overall test set
  - mutation count (1, 2, 3-5, 6-10, 11-15, 16+)
  - minimum linear distance between mutated positions
    (only for variants with >= 3 mutations)
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
    ("1",     lambda n: n == 1),
    ("2",     lambda n: n == 2),
    ("3-5",   lambda n: 3 <= n <= 5),
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
out_dir = Path("output") / "gen_split"
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
classes = list(clf.classes_)
present = set(y_te) | set(y_pred) | set(classes)
all_labels = ([c for c in ORDINAL_ORDER if c in present]
              + sorted(present - set(ORDINAL_ORDER)))

# -----------------------------------------------------------------------------
# 1. Overall confusion matrix
# -----------------------------------------------------------------------------

print("\n[1/3] Overall confusion matrix:")
plot_cm(y_te, y_pred, all_labels, "ESM-2 (8M parameters), generation-wise split",
        out_dir / "confusion_matrix_overall.png")

# -----------------------------------------------------------------------------
# 2. Stratified by test-variant mutation count
# -----------------------------------------------------------------------------

print("\n[2/3] Stratified by mutation count:")
nmut = df_test["num_mutations"].astype(int).to_numpy()
for label, predicate in MUTATION_COUNT_BINS:
    mask = np.array([predicate(int(n)) for n in nmut])
    if mask.sum() == 0:
        continue
    plot_cm(y_te[mask], y_pred[mask], all_labels,
            f"Mutation count: {label}",
            out_dir / f"confusion_matrix_mut_{label.replace('+', 'plus')}.png")

# -----------------------------------------------------------------------------
# 3. Stratified by min linear distance between mutations (≥3 mut only)
# -----------------------------------------------------------------------------

print("\n[3/3] Stratified by min linear distance (≥3 mutations):")
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
            f"≥3 mutations, min linear distance: {label}",
            out_dir / f"confusion_matrix_dist_{label.replace('+', 'plus')}.png")

print(f"\nAll outputs in {out_dir}/")