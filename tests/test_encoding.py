# Tests for vocabulary constants and amino acid encoding utilities.
# Validates coverage, one-hot shape/values, padding logic, and mutation application.

import numpy as np
import pytest
from nucb_transformer.data.encoding import (
    tokenize, to_one_hot, pad_or_truncate, mutation_string_to_sequence,
    AA_TO_IDX, VOCAB_SIZE, VARIABLE_REGION_LENGTH,
)

WILDTYPE = "A" * VARIABLE_REGION_LENGTH


def test_all_canonical_aas_in_vocab():
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert aa in AA_TO_IDX, f"{aa} missing from vocabulary"


def test_tokenize_returns_valid_indices():
    tokens = tokenize("ACDEFGHIKLMNPQRSTVWY")
    assert all(0 <= t < VOCAB_SIZE for t in tokens)


def test_one_hot_shape():
    tokens = pad_or_truncate(tokenize(WILDTYPE))
    oh = to_one_hot(tokens)
    assert oh.shape == (VARIABLE_REGION_LENGTH, VOCAB_SIZE)


def test_one_hot_is_binary():
    oh = to_one_hot(pad_or_truncate(tokenize("AAAA")))
    assert set(oh.flatten().tolist()).issubset({0, 1})


def test_padding_fills_to_length():
    padded = pad_or_truncate(tokenize("AC"), length=VARIABLE_REGION_LENGTH)
    assert len(padded) == VARIABLE_REGION_LENGTH
    assert padded[2:] == [0] * (VARIABLE_REGION_LENGTH - 2)


def test_mutation_string_applies_correctly():
    seq = mutation_string_to_sequence("A1C", WILDTYPE)
    assert seq[0] == "C"
    assert seq[1:] == WILDTYPE[1:]
