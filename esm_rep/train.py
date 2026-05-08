"""Train + evaluate a logistic regression classifier on cached ESM embeddings.

Also produces a training loss curve under output/training_loss/ for diagnostic.

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from data_utils import load_and_split, load_config
from embed import embed_sequences


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

    sequences = (df_train[seq_col].astype(str).tolist()
                 + df_test[seq_col].astype(str).tolist())
    labels = np.concatenate([
        df_train[target_col].astype(str).to_numpy(),
        df_test[target_col].astype(str).to_numpy(),
    ])
    print(f"Class counts (full pool): {pd.Series(labels).value_counts().to_dict()}")

    # ---- Embed ----
    X = embed_sequences(sequences, cfg)
    print(f"Embedding shape: {X.shape}")

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

    # ---- Training loss curve (diagnostic; doesn't change clf above) ----
    print("\nComputing training loss curve...")
    loss_dir = Path("output/training_loss")
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
    ax.plot(checkpoints, losses, marker="o", linewidth=1.5, color="#2171b5")
    ax.set_xscale("log")
    ax.set_xlabel("Solver iterations (log scale)")
    ax.set_ylabel("Training cross-entropy loss")
    ax.set_title("Logistic regression training loss curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {loss_dir}/")

    # ---- Evaluate ----
    preds = clf.predict(X_te_s)
    print("\n=== Test set metrics ===")
    print(f"  accuracy:    {accuracy_score(y_te, preds):.4f}")
    print(f"  f1_macro:    {f1_score(y_te, preds, average='macro'):.4f}")
    print(f"  f1_weighted: {f1_score(y_te, preds, average='weighted'):.4f}")

    print("\nClassification report:")
    print(classification_report(y_te, preds, zero_division=0))

    label_order = sorted(set(y_tr) | set(y_te))
    cm = pd.DataFrame(
        confusion_matrix(y_te, preds, labels=label_order),
        index=[f"true:{c}" for c in label_order],
        columns=[f"pred:{c}" for c in label_order],
    )
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()