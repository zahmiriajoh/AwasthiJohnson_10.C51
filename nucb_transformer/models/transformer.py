# Multi-head self-attention, encoder block, and the top-level NucleaseTransformer.
# Replaces the CNN's local conv kernels with global self-attention so the model
# can capture epistatic interactions between non-adjacent residue positions.

from __future__ import annotations

import torch
import torch.nn as nn
from nucb_transformer.models.positional import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from nucb_transformer.models.heads import ClassificationHead

_VOCAB_SIZE = 20   # 20 canonical amino acids, no pad token
_NUM_CLASSES = 4   # non-functional | activity > 0 | activity > WT | activity > A73R


# ── attention ─────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention over sequence positions."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        # mask: (batch, seq_len) bool, True = pad → ignored by attention
        out, attn_weights = self.attn(x, x, x, key_padding_mask=mask, need_weights=return_weights)
        if return_weights:
            return out, attn_weights
        return out


# ── encoder block ─────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Pre-norm encoder layer: self-attention → add & norm → FFN → add & norm."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_weights:
            attn_out, attn_weights = self.attn(self.norm1(x), mask, return_weights=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x), mask)
            attn_weights = None
        x = x + self.ffn(self.norm2(x))
        if return_weights:
            return x, attn_weights
        return x


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
        vocab_size: int = _VOCAB_SIZE,
        num_classes: int = _NUM_CLASSES,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        if pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(d_model, num_classes, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,                      # (batch, seq_len) int64
        padding_mask: torch.Tensor | None = None,  # (batch, seq_len) bool, True = pad
        return_attention: bool = False,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple:
        # Returns raw logits (batch, num_classes).
        # Apply softmax externally for inference; use with cross-entropy for training.
        # return_attention=True  → (logits, attn_list) where attn_list is one
        #                          (batch, seq_len, seq_len) tensor per layer.
        # return_embeddings=True → (logits, embeddings) where embeddings is the
        #                          mean-pooled (batch, d_model) vector before the head.
        x = self.embedding(tokens)       # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        attn_list = []
        for layer in self.layers:
            if return_attention:
                x, w = layer(x, padding_mask, return_weights=True)
                attn_list.append(w)
            else:
                x = layer(x, padding_mask)
        x = self.norm(x)
        embeddings = x.mean(dim=1)       # (batch, d_model)
        logits = self.head(embeddings)
        if return_attention:
            return logits, attn_list
        if return_embeddings:
            return logits, embeddings
        return logits

    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns softmax probabilities (batch, num_classes)."""
        return torch.softmax(self.forward(tokens), dim=-1)
