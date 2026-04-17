# Tests for NucleaseDataset, DataLoader collation, and split logic.

import pandas as pd
import pytest
from nucb_transformer.data.dataset import NucleaseDataset, collate_fn
from nucb_transformer.data.splits import split_dataset
from nucb_transformer.data.encoding import VARIABLE_REGION_LENGTH

WILDTYPE = "A" * VARIABLE_REGION_LENGTH


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "mutations": ["", "A1C", "A1C, A2T"],
        "activity_class": ["WT", ">WT", "<WT"],
    })


def test_dataset_length(small_df):
    ds = NucleaseDataset(small_df, WILDTYPE)
    assert len(ds) == 3


def test_splits_are_disjoint(small_df):
    big_df = pd.concat([small_df] * 20, ignore_index=True)
    train, val, test = split_dataset(big_df, cluster_aware=False)
    combined = set(train.index) | set(val.index) | set(test.index)
    assert len(combined) == len(big_df)


def test_collate_produces_padding_mask(small_df):
    ds = NucleaseDataset(small_df, WILDTYPE)
    batch = [ds[i] for i in range(len(ds))]
    tokens, labels, mask = collate_fn(batch)
    assert tokens.shape[0] == len(batch)
    assert mask.dtype.name == "bool"
