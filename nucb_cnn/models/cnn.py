# 1D convolutional encoder and top-level NucleaseConvNet.
# Architecture mirrors the DeepMind G4 CNN:
#   3 × Conv1d(32 filters, k=5, same-padding) → flatten → Dense(64) + dropout → logits(4).
# One-hot encoding is done inside forward() so the token interface matches the transformer,
# enabling both models to share the same Dataset and DataLoader without changes.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from nucb_cnn.models.heads import ClassificationHead

_VOCAB_SIZE = 20   # 20 canonical amino acids, no pad token
_NUM_CLASSES = 4   # non-functional | activity > 0 | activity > WT | activity > A73R
_SEQ_LEN = 142     # full NucB sequence length (no variable-region trimming)


# ── conv block ────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Single Conv1d → ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        # padding = kernel_size // 2 reproduces TensorFlow padding='same' for odd kernels.
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len) → same shape out
        return self.relu(self.conv(x))


# ── full model ────────────────────────────────────────────────────────────────

class NucleaseConvNet(nn.Module):
    """
    Sequence → 4-class activity classifier using 1D convolutions.

    Transformer analogue:
      TransformerEncoderBlocks (global attention)  ↔  ConvBlocks (local kernels)
      mean pool over positions → linear projection  ↔  flatten → Dense(hidden_dim)
      ClassificationHead(4)                         ↔  ClassificationHead(4)

    The flatten + position-specific dense layer is intentional: it matches the
    DeepMind G4 CNN and gives the model sensitivity to absolute position, which
    is appropriate for the fixed-length NucB sequence.
    """

    def __init__(
        self,
        num_filters: int = 32,       # conv channels per layer (DeepMind default)
        kernel_size: int = 5,        # receptive field per conv (DeepMind default)
        num_conv_layers: int = 3,    # conv blocks — mirrors transformer's 3 encoder blocks
        hidden_dim: int = 64,        # dense layer units after flatten (DeepMind default)
        dropout: float = 0.05,       # 5% dropout, same as transformer and DeepMind defaults
        vocab_size: int = _VOCAB_SIZE,
        num_classes: int = _NUM_CLASSES,
        seq_len: int = _SEQ_LEN,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        layers: list[nn.Module] = []
        in_channels = vocab_size   # first layer reads from one-hot channels
        for _ in range(num_conv_layers):
            layers.append(ConvBlock(in_channels, num_filters, kernel_size))
            in_channels = num_filters
        self.conv_stack = nn.Sequential(*layers)

        # Flatten → Dense(hidden_dim, relu) → Dropout → logits(num_classes).
        # Passing flat_dim directly into ClassificationHead gives exactly the
        # DeepMind architecture: one hidden layer, not two.
        flat_dim = num_filters * seq_len
        self.flatten = nn.Flatten()
        self.head = ClassificationHead(flat_dim, num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # tokens: (batch, seq_len) int64
        # Returns raw logits (batch, num_classes).
        # return_embeddings=True → (logits, embeddings) where embeddings is the
        #                          flattened conv representation (batch, num_filters*seq_len)
        #                          before the classification head.
        x = F.one_hot(tokens, num_classes=self.vocab_size).float()  # (batch, seq_len, vocab)
        x = x.permute(0, 2, 1)          # (batch, vocab, seq_len) — Conv1d is channels-first
        x = self.conv_stack(x)           # (batch, num_filters, seq_len)
        embeddings = self.flatten(x)     # (batch, num_filters * seq_len)
        logits = self.head(embeddings)   # (batch, num_classes)
        if return_embeddings:
            return logits, embeddings
        return logits

    def predict_proba(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns softmax probabilities (batch, num_classes)."""
        return torch.softmax(self.forward(tokens), dim=-1)
