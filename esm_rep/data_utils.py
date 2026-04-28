"""Load the NucB landscape CSV and partition it into train / test pools.

The `generations` column in landscape.csv stores a tuple (as a Python literal
string) of every experimental round each variant was screened in. We parse
each cell into a frozenset of lowercase tokens so set-membership tests work
regardless of formatting.

A variant ends up in the TRAIN pool iff its generation set intersects the
train generations. A variant ends up in the TEST pool iff its set intersects
the test generations AND does NOT intersect the train generations — this
guarantees the same physical variant is never in both splits.
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


def load_and_split(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Return (train_df, test_df_or_None, all_generation_tokens) per `cfg`.

    `test_df` is None iff `split.test_generations == 'random_holdout'`, in
    which case the caller should make a stratified holdout from `train_df`.
    """
    d = cfg["data"]
    s = cfg["split"]

    df = pd.read_csv(d["path"])
    df = df.dropna(subset=[d["sequence_col"], d["target_col"]]).reset_index(drop=True)

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

    # Variants that appear in both pools go to TRAIN only.
    test_mask = gen_sets.apply(
        lambda g: bool(g & test_gens) and not (g & train_gens)
    )
    df_test = df[test_mask].reset_index(drop=True)
    return df_train, df_test, all_tokens