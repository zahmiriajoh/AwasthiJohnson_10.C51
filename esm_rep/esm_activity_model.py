"""
ESM-2 embeddings + logistic regression for the Google DeepMind NucB landscape,
trained on G1 (initial error-prone PCR library).

Dataset: https://github.com/google-deepmind/nuclease_design
Paper:   Thomas, Belanger, Colwell et al., Cell Systems 2025
         "Engineering highly active nuclease enzymes with ML and HT screening"

Download `data/landscape.csv` from the repo first (it's ~15 MB).

The target column in landscape.csv is an ordinal multi-class activity label
(roughly: inactive < WT-active < A73R-active < A73R,D74S-active). We treat it
as a classification problem but also report Spearman rank correlation against
the ordinal integer codes, so we reward models that get the order right even
when they miss the exact bucket.

*G1 caveat*: The initial epPCR library has ~11% functional and ~2% >WT variants,
and A73R/A73R,D74S weren't observed until G2+. So when filtered to G1, the
higher activity classes will have very few or zero examples — it's effectively
near-binary (inactive vs. >=WT-active). This is exactly the setting the paper
used to train its round-2 design models.

CPU-friendly design choices:
    - Smallest ESM-2 (t6_8M_UR50D) by default.
    - fp32, small batch size, no_grad inference.
    - Embedding extraction is resumable: each chunk is saved to disk so you
      can kill and restart without losing progress.
    - Mean pooling over residue positions, ignoring CLS/EOS/pad tokens.

Usage:
    pip install torch transformers scikit-learn scipy pandas numpy tqdm

    # 1. inspect the CSV to confirm column names and generation labels:
    python esm_activity_model.py --data landscape.csv --inspect

    # 2. train on G1 only (the default):
    python esm_activity_model.py --data landscape.csv

    # 2b. if the CSV uses a different column or value for generation:
    python esm_activity_model.py --data landscape.csv \
        --generation-col round --generations 1

    # train on G1, but evaluate on G2 (paper-style forward-prediction test):
    python esm_activity_model.py --data landscape.csv \
        --generations G1 --test-generations G2

    # faster iteration on a random subsample of G1:
    python esm_activity_model.py --data landscape.csv --subsample 3000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Embedding extraction (CPU-friendly, resumable)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _embed_chunk(
    sequences: Sequence[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Embed a list of sequences; returns (N, hidden_dim) float32 array."""
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = list(sequences[i : i + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        h = model(**enc).last_hidden_state  # (B, L, D)

        # Mask out <cls>, <eos>, and padding before mean-pooling.
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0  # <cls>
        last = enc["attention_mask"].sum(dim=1) - 1
        mask[torch.arange(mask.size(0)), last] = 0  # <eos>
        mask = mask.unsqueeze(-1).float()

        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_with_resume(
    sequences: Sequence[str],
    cache_dir: str | Path,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    batch_size: int = 8,
    chunk_size: int = 1024,
    max_length: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """Compute ESM embeddings chunk-by-chunk, saving each chunk to disk.

    If the process is interrupted, re-running resumes from the last saved
    chunk. Final array is assembled from all chunks.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: model + all sequences (so edits bust the cache).
    sig = hashlib.sha256(model_name.encode())
    for s in sequences:
        sig.update(s.encode())
    key = sig.hexdigest()[:16]
    run_dir = cache_dir / f"esm_{key}"
    run_dir.mkdir(exist_ok=True)

    final_path = run_dir / "all.npy"
    if final_path.exists():
        print(f"Loading cached embeddings: {final_path}")
        return np.load(final_path)

    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    n = len(sequences)
    n_chunks = (n + chunk_size - 1) // chunk_size
    for ci in tqdm(range(n_chunks), desc="Embedding chunks"):
        chunk_path = run_dir / f"chunk_{ci:05d}.npy"
        if chunk_path.exists():
            continue
        start, end = ci * chunk_size, min((ci + 1) * chunk_size, n)
        embs = _embed_chunk(
            sequences[start:end],
            tokenizer, model, device,
            batch_size, max_length,
        )
        np.save(chunk_path, embs)

    # Stitch chunks in order.
    all_embs = np.concatenate(
        [np.load(run_dir / f"chunk_{ci:05d}.npy") for ci in range(n_chunks)],
        axis=0,
    )
    np.save(final_path, all_embs)
    # Clean up per-chunk files now that the combined file is saved.
    for ci in range(n_chunks):
        (run_dir / f"chunk_{ci:05d}.npy").unlink(missing_ok=True)
    return all_embs


# ---------------------------------------------------------------------------
# Top models
# ---------------------------------------------------------------------------

def fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    Cs: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0),
    n_folds: int = 5,
) -> tuple[LogisticRegressionCV, StandardScaler]:
    """Multinomial logistic regression with k-fold CV over C."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    clf = LogisticRegressionCV(
        Cs=list(Cs),
        cv=n_folds,
        max_iter=5000,
        n_jobs=-1,
        scoring="f1_macro",
        class_weight="balanced",  # activity levels are imbalanced
    )
    clf.fit(X_train_s, y_train)
    print(f"LogReg best C: {clf.C_}")
    return clf, scaler


def fit_ridge_ordinal(X_train, y_train, alphas=(1e-2, 1e-1, 1, 10, 100)):
    """Treat ordinal labels as integers and fit Ridge.

    Often the best 'classifier' for ordinal data is a regressor — its
    predictions respect the order explicitly.
    """
    scaler = StandardScaler().fit(X_train)
    reg = RidgeCV(alphas=list(alphas), cv=5).fit(scaler.transform(X_train), y_train)
    return reg, scaler


def evaluate_classifier(clf, scaler, X: np.ndarray, y_int: np.ndarray):
    Xs = scaler.transform(X)
    preds = clf.predict(Xs)
    metrics = {
        "accuracy": accuracy_score(y_int, preds),
        "f1_macro": f1_score(y_int, preds, average="macro"),
        "f1_weighted": f1_score(y_int, preds, average="weighted"),
        # Ordinal-aware: Spearman between predicted class index and true.
        "spearman_on_predicted_class": spearmanr(preds, y_int).correlation,
    }
    # With probabilities, compute rank correlation of the *expected* class
    # index — rewards probability mass on nearby ordinal classes.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xs)
        classes = clf.classes_.astype(float)
        expected = proba @ classes
        metrics["spearman_on_expected_class"] = spearmanr(expected, y_int).correlation
        try:
            metrics["auroc_ovr_macro"] = roc_auc_score(
                y_int, proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            pass
    return metrics, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to landscape.csv")
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--target-col", default="activity_level",
                   help="Column with ordinal activity class labels")
    p.add_argument("--generation-col", default="generations",
                   help="Column identifying the experimental round (G1..G4). "
                        "Common alternatives: 'round', 'campaign_round'.")
    p.add_argument("--generations", nargs="+", default=["G1"],
                   help="Which generation(s) to train on. Default: G1 only. "
                        "Pass multiple, e.g. --generations G1 G2.")
    p.add_argument("--test-generations", nargs="+", default=None,
                   help="If set, evaluate on these generations instead of a "
                        "random holdout (paper-style forward-prediction).")
    p.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D",
                   help="ESM-2 variant. t6_8M is ~8M params (CPU-friendly); "
                        "t12_35M is a good step up if you have patience.")
    p.add_argument("--cache-dir", default="./esm_cache")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--subsample", type=int, default=None,
                   help="Randomly subsample this many rows (speeds iteration)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=1024,
                   help="Sequences per checkpoint file during embedding")
    p.add_argument("--inspect", action="store_true",
                   help="Print columns + head and exit (run this first)")
    p.add_argument("--also-regression", action="store_true",
                   help="Also fit Ridge on integer-coded labels as ordinal proxy")
    args = p.parse_args()

    # --- Load & inspect --------------------------------------------------
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    if args.inspect:
        print(df.head())
        if args.target_col in df.columns:
            print(f"\n'{args.target_col}' value counts:")
            print(df[args.target_col].value_counts(dropna=False))
        if args.generation_col in df.columns:
            print(f"\n'{args.generation_col}' value counts:")
            print(df[args.generation_col].value_counts(dropna=False))
            if args.target_col in df.columns:
                print(f"\nCross-tab ({args.generation_col} x {args.target_col}):")
                print(pd.crosstab(df[args.generation_col], df[args.target_col]))
        else:
            print(f"\nWARNING: generation column '{args.generation_col}' not "
                  f"found. Rerun --inspect after picking the right column "
                  f"name from the list above.")
        return

    # --- Filter to requested generation(s) -------------------------------
    if args.generation_col not in df.columns:
        raise ValueError(
            f"Column '{args.generation_col}' not in CSV. Available: "
            f"{list(df.columns)}. Run with --inspect to look at the data, "
            f"then set --generation-col accordingly."
        )

    # Each row in the generation column may be either a plain string ('g1')
    # or a stringified tuple of generations the variant was screened in
    # (e.g. "('g1', 'g2')"). We parse to a frozenset of lowercase tokens so
    # that membership checks work uniformly.
    import ast

    def _parse_gens(cell) -> frozenset:
        if pd.isna(cell):
            return frozenset()
        s = str(cell).strip()
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            obj = s
        if isinstance(obj, (tuple, list, set, frozenset)):
            return frozenset(str(x).lower() for x in obj)
        return frozenset({str(obj).lower()})

    gen_sets = df[args.generation_col].apply(_parse_gens)

    train_gens = {str(g).lower() for g in args.generations}
    test_gens = {str(g).lower() for g in args.test_generations} if args.test_generations else None

    # Catalogue of every individual generation token that appears anywhere
    # in the column — used both for diagnostics and for validation.
    available_tokens = sorted({tok for s in gen_sets for tok in s})
    missing = [g for g in train_gens if g not in available_tokens]
    if missing:
        raise ValueError(
            f"Requested generation(s) {missing} not found. Available tokens: "
            f"{available_tokens}. (Note: matching is case-insensitive and "
            f"checks set-membership, so 'G1' matches any row whose generation "
            f"tuple contains 'g1'.)"
        )

    # A row is in the train pool if its generation set intersects the
    # requested train generations. So passing G1 picks up every variant
    # that was screened in G1, including those carried forward to later
    # rounds.
    train_mask = gen_sets.apply(lambda s: bool(s & train_gens))
    df_train_pool = df[train_mask].copy()
    print(f"\nTraining pool: {len(df_train_pool)} rows from generation(s) {sorted(train_gens)}")

    df_test_pool = None
    if test_gens is not None:
        missing_te = [g for g in test_gens if g not in available_tokens]
        if missing_te:
            raise ValueError(f"Test generation(s) {missing_te} not found. Available tokens: {available_tokens}.")
        # For an honest forward-prediction test, exclude variants that were
        # also in the training generations.
        test_mask = gen_sets.apply(lambda s: bool(s & test_gens) and not (s & train_gens))
        df_test_pool = df[test_mask].copy()
        print(f"External test set: {len(df_test_pool)} rows from generation(s) "
              f"{sorted(test_gens)} (excluding any that overlap with train).")

    # Work with the combined set for a single embedding pass, then split.
    if df_test_pool is not None:
        df_train_pool = df_train_pool.assign(__is_train=True)
        df_test_pool = df_test_pool.assign(__is_train=False)
        df_full = pd.concat([df_train_pool, df_test_pool], ignore_index=True)
    else:
        df_full = df_train_pool.assign(__is_train=True).reset_index(drop=True)

    df_full = df_full.dropna(subset=[args.sequence_col, args.target_col]).reset_index(drop=True)
    is_train = df_full["__is_train"].to_numpy() if df_test_pool is not None else None

    if args.subsample and args.subsample < len(df_full):
        # Only subsample from within the TRAIN pool when test set is external.
        if is_train is not None:
            train_idx = np.where(is_train)[0]
            keep_train = np.random.RandomState(args.seed).choice(
                train_idx, size=min(args.subsample, len(train_idx)), replace=False
            )
            keep = np.concatenate([keep_train, np.where(~is_train)[0]])
            df_full = df_full.iloc[keep].reset_index(drop=True)
            is_train = is_train[keep]
            print(f"Subsampled train pool to {is_train.sum()} rows "
                  f"(test pool unchanged at {(~is_train).sum()}).")
        else:
            df_full = df_full.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
            print(f"Subsampled to {len(df_full)} rows.")

    sequences = df_full[args.sequence_col].astype(str).tolist()
    y_raw = df_full[args.target_col].values

    # Map ordinal labels -> integer codes preserving natural order if strings.
    # The landscape file typically has labels like 'NO_ACTIVITY', 'WT', 'A73R',
    # 'A73R,D74S'. If they're already numeric (0/1/2/3), the order is kept.
    ORDER = ["non-functional", "activity > 0", "activity > WT", "activity > A73R", "activity > A73R,D74S"]
    present = [c for c in ORDER if c in set(y_raw)]
    cat = pd.Categorical(y_raw, categories=present, ordered=True)
    y_int = cat.codes.astype(int)
    class_names = list(cat.categories)
    print(f"Classes (in ascending order): {class_names}")
    print(f"Class counts: {pd.Series(y_raw).value_counts().to_dict()}")
    lens = [len(s) for s in sequences]
    print(f"Sequence lengths: min={min(lens)}, max={max(lens)}, median={int(np.median(lens))}")

    # --- Embeddings ------------------------------------------------------
    X = embed_with_resume(
        sequences,
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )
    print(f"Embedding shape: {X.shape}")

    # --- Split -----------------------------------------------------------
    if is_train is not None:
        # Use the external test generation(s) as the test set.
        X_tr, X_te = X[is_train], X[~is_train]
        y_tr, y_te = y_int[is_train], y_int[~is_train]
        print(f"Train: {len(X_tr)} rows; External test: {len(X_te)} rows.")
    else:
        # Stratified random holdout from within the training generation(s).
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_int,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y_int if len(np.unique(y_int)) > 1 else None,
        )
        print(f"Train: {len(X_tr)} rows; Random-holdout test: {len(X_te)} rows.")

    # --- Classifier ------------------------------------------------------
    clf, scaler = fit_logreg(X_tr, y_tr)

    print("\n=== Logistic regression: test set ===")
    test_metrics, test_preds = evaluate_classifier(clf, scaler, X_te, y_te)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nPer-class report:")
    print(classification_report(
        y_te, test_preds,
        target_names=[str(c) for c in class_names],
        zero_division=0,
    ))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_te, test_preds))

    # --- Optional ordinal proxy ------------------------------------------
    if args.also_regression:
        reg, reg_scaler = fit_ridge_ordinal(X_tr, y_tr)
        reg_preds = reg.predict(reg_scaler.transform(X_te))
        print("\n=== Ridge-on-integers (ordinal proxy): test set ===")
        print(f"  spearman: {spearmanr(reg_preds, y_te).correlation:.4f}")


if __name__ == "__main__":
    main()