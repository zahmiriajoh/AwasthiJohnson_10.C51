"""Load the NucB landscape CSV and partition it into train / test pools.

Two split modes selected via cfg["split"]["mode"]:

  * "generation" — variants routed by which experimental round(s) they
    appeared in. cfg["split"]["train_generations"] is a list of tokens;
    cfg["split"]["test_generations"] is either a list of tokens or
    "all_others". Variants in both pools go to TRAIN only.

  * "mutation_count" — variants split on the integer num_mutations column.
    Train = num_mutations < threshold, test = >= threshold.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _split_by_generation(df, d, s):
    parse = lambda cell: frozenset(str(x).lower() for x in ast.literal_eval(cell))
    gen_sets = df[d["generation_col"]].apply(parse)
    all_tokens = sorted({tok for g in gen_sets for tok in g})

    train_gens = {str(g).lower() for g in s["train_generations"]}
    train_mask = gen_sets.apply(lambda g: bool(g & train_gens))
    df_train = df[train_mask].reset_index(drop=True)

    tg = s["test_generations"]
    if tg == "all_others":
        test_gens = set(all_tokens) - train_gens
    else:
        test_gens = {str(g).lower() for g in tg}

    # Variants in both pools go to TRAIN only.
    test_mask = gen_sets.apply(
        lambda g: bool(g & test_gens) and not (g & train_gens)
    )
    df_test = df[test_mask].reset_index(drop=True)
    return df_train, df_test


def _split_by_mutation_count(df, d, s):
    threshold = int(s["mutation_count_threshold"])
    n_mut = df["num_mutations"].astype(int)
    df_train = df[n_mut < threshold].reset_index(drop=True)
    df_test = df[n_mut >= threshold].reset_index(drop=True)
    return df_train, df_test


def load_and_split(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) per cfg["split"]["mode"]."""
    d = cfg["data"]
    s = cfg["split"]

    df = pd.read_csv(d["path"])
    df = df.dropna(subset=[d["sequence_col"], d["target_col"]]).reset_index(drop=True)

    if s["mode"] == "generation":
        return _split_by_generation(df, d, s)
    return _split_by_mutation_count(df, d, s)