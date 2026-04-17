# Classification and regression output heads.
# Keeping heads separate from the encoder body lets you swap objectives
# (e.g. switch from 4-class classification to continuous enrichment regression)
# without touching the transformer stack.

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Linear → dropout → linear → logits.
    Matches the CNN's dense-64 + softmax head;
    call with cross-entropy loss during training (no softmax here).
    """

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.05):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) pooled representation → (batch, num_classes) logits
        ...


class RegressionHead(nn.Module):
    """
    Linear → ReLU → linear → scalar.
    For predicting continuous enrichment scores (MBO-DNN style objective)
    instead of the discrete 4-class labels. Use with MSE or Huber loss.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.05):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) → (batch, 1) enrichment prediction
        ...
