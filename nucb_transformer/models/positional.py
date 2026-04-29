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
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)              # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                           # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)               # even dims
        pe[:, 1::2] = torch.cos(position * div_term)               # odd dims
        self.register_buffer("pe", pe.unsqueeze(0))                # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) — adds encoding slice and applies dropout.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Trainable embedding table indexed by position (0 … max_len-1).
    Slightly more expressive than sinusoidal for fixed-length inputs.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)      # (seq_len,)
        x = x + self.pe(positions)                                 # broadcasts over batch
        return self.dropout(x)


def get_positional_encoding(name: str, d_model: int, **kwargs):
    """Factory: returns the encoding module selected by name ('sinusoidal' | 'learned')."""
    options = {"sinusoidal": SinusoidalPositionalEncoding, "learned": LearnedPositionalEncoding}
    if name not in options:
        raise ValueError(f"Unknown positional encoding '{name}'. Choose from {list(options)}")
    return options[name](d_model, **kwargs)
