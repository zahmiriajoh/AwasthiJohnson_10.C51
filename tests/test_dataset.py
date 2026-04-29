# Tests for NucleaseDataset and split logic.

import torch
import pandas as pd
import pytest
from nucb_transformer.data.dataset import NucleaseDataset
from nucb_transformer.data.splits import split_dataset, filter_by_sublibrary
from nucb_transformer.data.encoding import ACTIVITY_CLASSES

SEQ_LEN = 10   # short sequences for speed; NucleaseDataset has no fixed-length requirement
SEQ = "ACDEFGHIKL"


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "sequence": [SEQ] * 4,
        "activity_level": ACTIVITY_CLASSES,
        "sublibrary_names": [("g1",), ("g1",), ("g2",), ("g2",)],
    })


def test_dataset_length(small_df):
    ds = NucleaseDataset(small_df)
    assert len(ds) == 4


def test_getitem_token_shape(small_df):
    ds = NucleaseDataset(small_df)
    tokens, _ = ds[0]
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape == (SEQ_LEN,)
    assert tokens.dtype == torch.long


def test_getitem_label_in_range(small_df):
    ds = NucleaseDataset(small_df)
    for i in range(len(ds)):
        _, label = ds[i]
        assert 0 <= label < len(ACTIVITY_CLASSES)


def test_class_to_idx_covers_all_classes(small_df):
    ds = NucleaseDataset(small_df)
    assert set(ds.class_to_idx.keys()) == set(ACTIVITY_CLASSES)


def test_splits_sum_to_total(small_df):
    big_df = pd.concat([small_df] * 20, ignore_index=True)
    train, val, test = split_dataset(big_df, cluster_aware=False)
    assert len(train) + len(val) + len(test) == len(big_df)


def test_splits_have_no_overlap(small_df):
    big_df = pd.concat([small_df] * 20, ignore_index=True)
    big_df["_orig_idx"] = big_df.index
    train, val, test = split_dataset(big_df, cluster_aware=False)
    train_ids = set(train["_orig_idx"])
    val_ids   = set(val["_orig_idx"])
    test_ids  = set(test["_orig_idx"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_filter_by_sublibrary(small_df):
    filtered = filter_by_sublibrary(small_df, sublibrary="g1")
    assert len(filtered) == 2
    assert all("g1" in names for names in filtered["sublibrary_names"])


def test_filter_by_sublibrary_none_returns_all(small_df):
    assert len(filter_by_sublibrary(small_df, sublibrary=None)) == len(small_df)
