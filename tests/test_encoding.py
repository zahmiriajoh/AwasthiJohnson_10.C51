# Tests for amino acid encoding utilities.

import numpy as np
import pytest
from nucb_transformer.data.encoding import (
    tokenize, seq_to_one_hot, AA_TO_IDX, ACTIVITY_CLASSES, amino_acids,
)

SEQ = "ACDEFGHIKLMNPQRSTVWY"   # all 20 canonical AAs


def test_all_canonical_aas_in_vocab():
    for aa in amino_acids:
        assert aa in AA_TO_IDX


def test_tokenize_length_matches_sequence():
    assert len(tokenize(SEQ)) == len(SEQ)


def test_tokenize_indices_in_range():
    tokens = tokenize(SEQ)
    assert all(0 <= t < len(amino_acids) for t in tokens)


def test_tokenize_unknown_aa_raises():
    with pytest.raises(KeyError):
        tokenize("X")


def test_seq_to_one_hot_shape():
    ohe = seq_to_one_hot(SEQ)
    assert ohe.shape == (len(SEQ) * len(amino_acids),)


def test_seq_to_one_hot_is_binary():
    ohe = seq_to_one_hot("ACDE")
    assert set(ohe.tolist()).issubset({0.0, 1.0})


def test_seq_to_one_hot_one_hot_per_position():
    ohe = seq_to_one_hot("AC")
    # each position sums to 1
    assert ohe[:20].sum() == pytest.approx(1.0)
    assert ohe[20:].sum() == pytest.approx(1.0)


def test_activity_classes_has_four_entries():
    assert len(ACTIVITY_CLASSES) == 4


def test_activity_classes_are_sorted():
    assert ACTIVITY_CLASSES == sorted(ACTIVITY_CLASSES)
