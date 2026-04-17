# Positional encodings — injected after the token embedding so the transformer
# knows residue order. NucB variants are short (<=158 AAs), so both sinusoidal
# and learned options work; the config selects which to use at runtime.

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sin/cos encoding (Vaswani et al. 2017).
    No learned parameters; generalises to any sequence length up to max_len.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        # Pre-computes the (max_len, d_model) encoding matrix at init time.
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) — adds encoding slice and applies dropout.
        ...


class LearnedPositionalEncoding(nn.Module):
    """
    Trainable embedding table indexed by position (0 … max_len-1).
    Slightly more expressive than sinusoidal for fixed-length inputs.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


def get_positional_encoding(name: str, d_model: int, **kwargs):
    """Factory: returns the encoding module selected by name ('sinusoidal' | 'learned')."""
    options = {"sinusoidal": SinusoidalPositionalEncoding, "learned": LearnedPositionalEncoding}
    if name not in options:
        raise ValueError(f"Unknown positional encoding '{name}'. Choose from {list(options)}")
    return options[name](d_model, **kwargs)
