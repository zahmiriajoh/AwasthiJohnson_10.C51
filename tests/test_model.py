# Smoke tests for NucleaseTransformer: output shape, predict_proba, padding mask.

import torch
import pytest
from nucb_transformer.models.transformer import NucleaseTransformer
from nucb_transformer.data.encoding import ACTIVITY_CLASSES, amino_acids

BATCH   = 4
SEQ_LEN = 20   # shorter than full 142 for speed; transformer has no fixed-length constraint
VOCAB   = len(amino_acids)    # 20
N_CLS   = len(ACTIVITY_CLASSES)  # 4


@pytest.fixture
def model():
    return NucleaseTransformer(d_model=32, num_heads=2, num_layers=2, ffn_dim=64)


def tokens():
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


def test_forward_output_shape(model):
    assert model(tokens()).shape == (BATCH, N_CLS)


def test_predict_proba_shape(model):
    assert model.predict_proba(tokens()).shape == (BATCH, N_CLS)


def test_predict_proba_sums_to_one(model):
    probs = model.predict_proba(tokens())
    assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


def test_predict_proba_non_negative(model):
    probs = model.predict_proba(tokens())
    assert (probs >= 0).all()


def test_padding_mask_accepted(model):
    t = tokens()
    mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    mask[:, -3:] = True   # mark last 3 positions as padding
    out = model(t, padding_mask=mask)
    assert out.shape == (BATCH, N_CLS)


def test_sinusoidal_and_learned_produce_same_shape():
    t = tokens()
    for enc in ("sinusoidal", "learned"):
        m = NucleaseTransformer(d_model=32, num_heads=2, num_layers=1, ffn_dim=64, pos_encoding=enc)
        assert m(t).shape == (BATCH, N_CLS)
