# Classification output head — shared structure with the transformer.
# Keeping the head separate lets you swap objectives without touching the conv stack.

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Linear → ReLU → dropout → linear → logits.
    Matches the DeepMind CNN's dense output head;
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
        # x: (batch, d_model) → (batch, num_classes) logits
        return self.net(x)
