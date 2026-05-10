"""Train classifier on delta-from-WT pooled ESM features.

Per-residue ESM embeddings minus WT per-residue embeddings, then
max-abs-pooled across positions: for each ESM dimension, take the
largest absolute perturbation across the L residue positions.

Why max-abs and not mean: mean-pooling commutes with subtraction, so
mean_pool(E_v − E_wt) = mean_pool(E_v) − mean_pool(E_wt), where the
second term is a constant across variants. StandardScaler then absorbs
that constant, making the delta + mean-pool features mathematically
identical to raw mean-pool features. Max-abs is a non-linear aggregation
that breaks this equivalence — its value at dim k is the largest
position-wise |delta| in dim k, which depends nonlinearly on which
position(s) were perturbed and by how much.

Standalone — does not modify embed.py or train.py.

Output goes to output/training_loss_delta/ (loss curve + values), parallel
to train.py's output/training_loss/ so you can compare directly.

Usage:
    python train_delta.py --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_utils import load_and_split, load_config

ORDINAL_ORDER = ["non-functional", "activity > 0", "activity > WT", "activity > A73R"]


@torch.inference_mode()
def compute_delta_features(sequences, wt_seq, cfg) -> np.ndarray:
    """Max-abs pool of (per-residue ESM - WT per-residue ESM) over positions.
    Returns (N, D) float32. Resumable chunk-saving like embed.py.

    Assumes all sequences in the batch are the same length (true for the
    NucB landscape — all variants are 142 residues). Variable lengths
    would need explicit handling of attention_mask before pooling.
    """
    e = cfg["embedding"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cache key: model + WT + sequences. Changing any invalidates the cache.
    sig = hashlib.sha256(e["model_name"].encode())
    sig.update(wt_seq.encode())
    for s in sequences:
        sig.update(s.encode())
    key = sig.hexdigest()[:16]
    run_dir = Path(e["cache_dir"]) / f"delta_maxabs_{key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "all.npy"

    if final.exists():
        print(f"Loading cached delta features: {final}")
        return np.load(final)

    print(f"Loading {e['model_name']} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(e["model_name"])
    model = (AutoModel.from_pretrained(e["model_name"], add_pooling_layer=False)
             .to(device).eval())

    # WT per-residue embedding, drop <cls> and <eos>.
    enc = tokenizer(wt_seq, return_tensors="pt").to(device)
    h = model(**enc).last_hidden_state[0]
    last = int(enc["attention_mask"][0].sum().item()) - 1
    wt_per_res = h[1:last]  # (L, D)
    print(f"WT per-residue embedding: {tuple(wt_per_res.shape)}")

    n = len(sequences)
    chunk = e["chunk_size"]
    n_chunks = (n + chunk - 1) // chunk
    for ci in tqdm(range(n_chunks), desc="Delta chunks"):
        path = run_dir / f"chunk_{ci:05d}.npy"
        if path.exists():
            continue
        s, t = ci * chunk, min((ci + 1) * chunk, n)
        chunk_features = []
        for i in range(s, t, e["batch_size"]):
            j = min(i + e["batch_size"], t)
            batch = list(sequences[i:j])
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=e["max_length"]).to(device)
            h = model(**enc).last_hidden_state  # (B, L+2, D)
            # Drop <cls> at idx 0 and <eos> at idx -1. Valid because all
            # NucB variants have the same length, so no padding is added.
            per_res = h[:, 1:-1, :]                       # (B, L, D)
            delta = per_res - wt_per_res.unsqueeze(0)     # (B, L, D)
            features = delta.abs().max(dim=1).values      # max-abs pool: (B, D)
            chunk_features.append(features.float().cpu().numpy())
        np.save(path, np.concatenate(chunk_features, axis=0))

    out = np.concatenate(
        [np.load(run_dir / f"chunk_{ci:05d}.npy") for ci in range(n_chunks)],
        axis=0,
    )
    np.save(final, out)
    for ci in range(n_chunks):
        (run_dir / f"chunk_{ci:05d}.npy").unlink(missing_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seq_col = cfg["data"]["sequence_col"]
    target_col = cfg["data"]["target_col"]

    # ---- Load + split ----
    df_train, df_test = load_and_split(cfg)
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    # ---- Find WT (for delta computation) ----
    df_full = pd.read_csv(cfg["data"]["path"])
    df_full = df_full.dropna(subset=[seq_col, target_col]).reset_index(drop=True)
    wt_seq = str(df_full[df_full["num_mutations"] == 0].iloc[0][seq_col])
    print(f"WT sequence length: {len(wt_seq)}")

    sequences = (df_train[seq_col].astype(str).tolist()
                 + df_test[seq_col].astype(str).tolist())
    labels = np.concatenate([
        df_train[target_col].astype(str).to_numpy(),
        df_test[target_col].astype(str).to_numpy(),
    ])
    print(f"Class counts (full pool): {pd.Series(labels).value_counts().to_dict()}")

    # ---- Compute delta features ----
    X = compute_delta_features(sequences, wt_seq, cfg)
    print(f"Delta-feature shape: {X.shape}")

    # ---- Slice into train/test ----
    n_train = len(df_train)
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = labels[:n_train], labels[n_train:]

    # ---- Fit ----
    clf_cfg = cfg["classifier"]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegressionCV(
        Cs=list(clf_cfg["Cs"]), cv=clf_cfg["cv_folds"],
        max_iter=clf_cfg["max_iter"], n_jobs=-1,
        scoring=clf_cfg["scoring"], class_weight=clf_cfg["class_weight"],
    ).fit(X_tr_s, y_tr)
    print(f"Best C per class: {clf.C_}")

    # ---- Training loss curve (separate output dir) ----
    print("\nComputing training loss curve...")
    loss_dir = Path("output/training_loss_delta")
    loss_dir.mkdir(parents=True, exist_ok=True)

    best_C = float(np.atleast_1d(clf.C_).mean())
    checkpoints = [c for c in
                   [1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
                   if c <= clf_cfg["max_iter"]]

    classes_idx = None
    losses = []
    for n_iter in checkpoints:
        d_clf = LogisticRegression(
            C=best_C, max_iter=n_iter,
            class_weight=clf_cfg["class_weight"], n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            d_clf.fit(X_tr_s, y_tr)
        proba = d_clf.predict_proba(X_tr_s)
        if classes_idx is None:
            classes_ = list(d_clf.classes_)
            classes_idx = np.array([classes_.index(y) for y in y_tr])
        loss = float(-np.log(proba[np.arange(len(y_tr)), classes_idx] + 1e-15).mean())
        losses.append(loss)
        print(f"  iter={n_iter:5d}: train cross-entropy = {loss:.4f}")

    pd.DataFrame({"iteration": checkpoints, "loss": losses}).to_csv(
        loss_dir / "loss_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(checkpoints, losses, marker="o", linewidth=1.5, color="#cb181d")
    ax.set_xscale("log")
    ax.set_xlabel("Solver iterations (log scale)")
    ax.set_ylabel("Training cross-entropy loss")
    ax.set_title("Logistic regression training loss curve (delta-from-WT features)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {loss_dir}/")

    # ---- Evaluate ----
    preds = clf.predict(X_te_s)
    print("\n=== Test set metrics (delta-from-WT features) ===")
    print(f"  accuracy:    {accuracy_score(y_te, preds):.4f}")
    print(f"  f1_macro:    {f1_score(y_te, preds, average='macro'):.4f}")
    print(f"  f1_weighted: {f1_score(y_te, preds, average='weighted'):.4f}")

    print("\nClassification report:")
    print(classification_report(y_te, preds, zero_division=0))

    present = set(y_tr) | set(y_te)
    label_order = ([c for c in ORDINAL_ORDER if c in present]
                   + sorted(present - set(ORDINAL_ORDER)))
    cm = pd.DataFrame(
        confusion_matrix(y_te, preds, labels=label_order),
        index=[f"true:{c}" for c in label_order],
        columns=[f"pred:{c}" for c in label_order],
    )
    print("Confusion matrix:")
    print(cm)

    # ---- Save predictions CSV (landscape with delta-feature predictions) ----
    preds_tr = clf.predict(X_tr_s)
    proba_tr = clf.predict_proba(X_tr_s)
    proba_te = clf.predict_proba(X_te_s)

    df_out = pd.concat([df_train, df_test], ignore_index=True)
    df_out["split"] = ["train"] * len(df_train) + ["test"] * len(df_test)
    df_out["predicted_class"] = np.concatenate([preds_tr, preds])
    proba_all = np.vstack([proba_tr, proba_te])
    for i, c in enumerate(clf.classes_):
        df_out[f"prob_{c}"] = proba_all[:, i]
    pred_path = Path("output/predictions_delta.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")


if __name__ == "__main__":
    main()