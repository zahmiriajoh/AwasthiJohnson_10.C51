# Classification output head.
# Keeping heads separate from the encoder body lets you swap objectives
# (e.g. switch from 4-class classification to continuous enrichment regression)
# without touching the transformer stack. But not actually necessary for this simple model.

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Linear → dropout → linear → logits.
    Matches the CNN's dense-64 + softmax head;
    call with cross-entropy loss during training (no softmax here).
    """

    def __init__(self, d_model: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model) pooled representation → (batch, num_classes) logits
        return self.net(x)

