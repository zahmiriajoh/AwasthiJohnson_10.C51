# Multi-head self-attention, encoder block, and the top-level NucleaseTransformer.
# Replaces the CNN's local conv kernels with global self-attention so the model
# can capture epistatic interactions between non-adjacent residue positions.

import torch
import torch.nn as nn
from nucb_transformer.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from nucb_transformer.models.heads import ClassificationHead
from nucb_transformer.data.encoding import VOCAB_SIZE, NUM_CLASSES


# ── attention ─────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention over sequence positions."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        # Projects Q, K, V; splits into heads; applies scaled dot-product;
        # concatenates and projects back to d_model.
        ...

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # mask: (batch, 1, 1, seq_len) — True where padding, set to -inf before softmax
        ...


# ── encoder block ─────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Pre-norm encoder layer: self-attention → add & norm → FFN → add & norm."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        # FFN is a 2-layer MLP with GELU activation.
        # Pre-norm (LayerNorm before each sub-layer) improves training stability.
        ...

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        ...


# ── full model ────────────────────────────────────────────────────────────────

class NucleaseTransformer(nn.Module):
    """
    Sequence → 4-class activity classifier.

    CNN analogue:
      Conv layers (kernel=5, local)  ↔  TransformerEncoderBlocks (global attention)
      Flatten + Dense(64)            ↔  mean pool over positions → linear projection
      Dense(4, softmax)              ↔  ClassificationHead(4) + softmax at inference
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,          # mirrors CNN's 3 conv blocks
        ffn_dim: int = 256,
        dropout: float = 0.05,        # matches CNN's 5% dropout default
        pos_encoding: str = "sinusoidal",   # "sinusoidal" | "learned"
        vocab_size: int = VOCAB_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        # 1. Token embedding:  int IDs → d_model vectors
        # 2. Positional encoding (type selected by pos_encoding)
        # 3. N × TransformerEncoderBlock
        # 4. LayerNorm after final block
        # 5. Mean pool across sequence positions
        # 6. ClassificationHead → num_classes logits
        ...

    def forward(
        self,
        tokens: torch.Tensor,                      # (batch, seq_len) int64
        padding_mask: torch.Tensor | None = None,  # (batch, seq_len) bool, True = pad
    ) -> torch.Tensor:
        # Returns raw logits (batch, num_classes).
        # Apply softmax externally for inference; use with cross-entropy for training.
        ...

    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns softmax probabilities (batch, num_classes)."""
        ...
