"""Train + evaluate a logistic regression classifier on cached ESM embeddings.

Usage:
    python train.py --config config.yaml

If embeddings haven't been computed yet, this script will compute them on the
fly (i.e. running embed.py first is optional). All split logic and embedding
caching are shared via data_utils.py and embed.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_utils import load_and_split, load_config
from embed import embed_sequences


def fit_classifier(X_train: np.ndarray, y_train: np.ndarray, cfg: dict):
    """Standard-scale features, then fit LogisticRegressionCV with k-fold CV."""
    c = cfg["classifier"]
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    clf = LogisticRegressionCV(
        Cs=list(c["Cs"]),
        cv=c["cv_folds"],
        max_iter=c["max_iter"],
        n_jobs=-1,
        scoring=c["scoring"],
        class_weight=c["class_weight"],
    )
    clf.fit(Xs, y_train)
    print(f"Best C per class: {clf.C_}")
    return clf, scaler


def evaluate(clf, scaler, X: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray]:
    """Compute classification metrics on a held-out set."""
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "f1_macro": f1_score(y, preds, average="macro"),
        "f1_weighted": f1_score(y, preds, average="weighted"),
    }
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xs)
        try:
            metrics["auroc_ovr_macro"] = roc_auc_score(
                y, proba,
                multi_class="ovr",
                average="macro",
                labels=clf.classes_,
            )
        except ValueError as e:
            print(f"AUROC skipped: {e}")
    return metrics, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    seq_col = cfg["data"]["sequence_col"]
    target_col = cfg["data"]["target_col"]

    # --- Load + split ----------------------------------------------------
    df_train, df_test, _ = load_and_split(cfg)
    print(f"Train pool: {len(df_train)} rows")
    if df_test is not None:
        print(f"Test pool:  {len(df_test)} rows (no overlap with train)")

    if df_test is None:
        sequences = df_train[seq_col].astype(str).tolist()
        labels = df_train[target_col].astype(str).to_numpy()
    else:
        sequences = (
            df_train[seq_col].astype(str).tolist()
            + df_test[seq_col].astype(str).tolist()
        )
        labels = np.concatenate(
            [
                df_train[target_col].astype(str).to_numpy(),
                df_test[target_col].astype(str).to_numpy(),
            ]
        )
    print(f"Class counts (full pool): {pd.Series(labels).value_counts().to_dict()}")

    # --- Embed (or load from cache) -------------------------------------
    X = embed_sequences(sequences, cfg)
    print(f"Embedding shape: {X.shape}")

    # --- Build train / test splits --------------------------------------
    if df_test is None:
        test_size = cfg["split"].get("test_size", 0.2)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        print(f"Train: {len(X_tr)} | Random-holdout test: {len(X_te)}")
    else:
        n_train = len(df_train)
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = labels[:n_train], labels[n_train:]
        print(f"Train: {len(X_tr)} | External test: {len(X_te)}")

    # --- Train + evaluate -----------------------------------------------
    clf, scaler = fit_classifier(X_tr, y_tr, cfg)
    metrics, preds = evaluate(clf, scaler, X_te, y_te)

    print("\n=== Test set metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

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