"""Load the NucB landscape CSV and partition it into train / test pools.

Two split modes are supported, selected via `cfg["split"]["mode"]`:

* `generation` — the original setup. Variants are routed by which experimental
  round(s) they appeared in. Each row's `generations` cell is a stringified
  tuple of round tokens (e.g. "('g1', 'g2')"); we parse it to a frozenset and
  test set-membership.

* `mutation_count` — split on the integer `num_mutations` column. Train pool
  is `num_mutations < threshold`; test pool is `num_mutations >= threshold`.
  Default threshold = 2 (single-mutants and WT for training; multi-mutants
  for testing).

In both modes, the same physical variant is never put in both splits.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def parse_generations(cell) -> frozenset:
    """Parse one `generations` cell into a frozenset of lowercase tokens.

    Accepts either a stringified tuple like "('g1', 'g2')" or a plain string
    like 'g1'. Returns an empty frozenset for NaN/missing.
    """
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


# ---------------------------------------------------------------------------
# Split implementations
# ---------------------------------------------------------------------------

def _split_by_generation(
    df: pd.DataFrame, d: dict, s: dict
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Train/test split on the `generations` column."""
    if d["generation_col"] not in df.columns:
        raise ValueError(
            f"Column '{d['generation_col']}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    gen_sets = df[d["generation_col"]].apply(parse_generations)
    all_tokens = sorted({tok for g in gen_sets for tok in g})

    train_gens = {str(g).lower() for g in s["train_generations"]}
    missing = [g for g in train_gens if g not in all_tokens]
    if missing:
        raise ValueError(
            f"Train generation(s) {missing} not found. Available: {all_tokens}"
        )
    train_mask = gen_sets.apply(lambda g: bool(g & train_gens))
    df_train = df[train_mask].reset_index(drop=True)

    tg = s["test_generations"]
    if tg == "random_holdout":
        return df_train, None, all_tokens

    if tg == "all_others":
        test_gens = set(all_tokens) - train_gens
    else:
        test_gens = {str(g).lower() for g in tg}
        missing_te = [g for g in test_gens if g not in all_tokens]
        if missing_te:
            raise ValueError(
                f"Test generation(s) {missing_te} not found. Available: {all_tokens}"
            )

    # Variants in both pools go to TRAIN only.
    test_mask = gen_sets.apply(
        lambda g: bool(g & test_gens) and not (g & train_gens)
    )
    df_test = df[test_mask].reset_index(drop=True)
    return df_train, df_test, all_tokens


def _split_by_mutation_count(
    df: pd.DataFrame, d: dict, s: dict
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Train/test split on the `num_mutations` column.

    Train pool = num_mutations < threshold, test pool = num_mutations >= threshold.
    """
    col = s.get("mutation_count_col", "num_mutations")
    threshold = int(s.get("mutation_count_threshold", 2))

    if col not in df.columns:
        raise ValueError(
            f"Mutation-count column '{col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    n_mut = pd.to_numeric(df[col], errors="coerce")
    bad = n_mut.isna().sum()
    if bad:
        print(f"  WARNING: dropping {bad} rows with non-numeric '{col}'")
        df = df.loc[n_mut.notna()].reset_index(drop=True)
        n_mut = n_mut.loc[n_mut.notna()].reset_index(drop=True)
    n_mut = n_mut.astype(int)

    df_train = df[n_mut < threshold].reset_index(drop=True)

    tg = s.get("test_set", "complement")
    if tg == "random_holdout":
        return df_train, None, []
    if tg != "complement":
        raise ValueError(
            f"For split.mode='mutation_count', test_set must be 'complement' "
            f"or 'random_holdout' (got {tg!r})."
        )

    df_test = df[n_mut >= threshold].reset_index(drop=True)
    return df_train, df_test, []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_and_split(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Return (train_df, test_df_or_None, all_generation_tokens) per `cfg`.

    `test_df` is None iff the test set is a random holdout drawn from the
    train pool — the caller is responsible for making that holdout.
    The third element is only meaningful for `mode='generation'`.
    """
    d = cfg["data"]
    s = cfg["split"]

    df = pd.read_csv(d["path"])
    df = df.dropna(subset=[d["sequence_col"], d["target_col"]]).reset_index(drop=True)

    mode = s.get("mode", "generation")
    if mode == "generation":
        return _split_by_generation(df, d, s)
    if mode == "mutation_count":
        return _split_by_mutation_count(df, d, s)
    raise ValueError(
        f"Unknown split.mode={mode!r}. Use 'generation' or 'mutation_count'."
    )