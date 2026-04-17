# Smoke tests for NucleaseTransformer: shape checks, softmax sanity, padding mask.

import torch
import pytest
from nucb_transformer.models.transformer import NucleaseTransformer
from nucb_transformer.data.encoding import NUM_CLASSES, VARIABLE_REGION_LENGTH


@pytest.fixture
def model():
    return NucleaseTransformer(d_model=64, num_heads=2, num_layers=2, ffn_dim=128)


def test_output_shape(model):
    tokens = torch.randint(1, 21, (4, VARIABLE_REGION_LENGTH))
    assert model(tokens).shape == (4, NUM_CLASSES)


def test_softmax_sums_to_one(model):
    tokens = torch.randint(1, 21, (2, VARIABLE_REGION_LENGTH))
    probs = model.predict_proba(tokens)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_padding_mask_does_not_crash(model):
    tokens = torch.randint(1, 21, (3, VARIABLE_REGION_LENGTH))
    mask = torch.zeros(3, VARIABLE_REGION_LENGTH, dtype=torch.bool)
    mask[:, -2:] = True
    model(tokens, padding_mask=mask)
