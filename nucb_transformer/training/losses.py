# Loss functions for classification and regression objectives.
# WeightedActivityLoss is the primary loss: it up-weights the rare high-activity
# classes ('>WT', '>=A73R') so the model doesn't collapse to predicting '<WT'.

import torch
import torch.nn as nn
from nucb_transformer.data.encoding import ACTIVITY_CLASSES


class ActivityCrossEntropy(nn.Module):
    """Standard cross-entropy over the 4 activity classes (no re-weighting)."""

    def __init__(self):
        ...

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ...


class WeightedActivityLoss(nn.Module):
    """
    Cross-entropy with inverse-frequency class weights computed from training labels.
    Pass class_counts (list of 4 ints in ACTIVITY_CLASSES order) at init time.
    """

    def __init__(self, class_counts: list[int], device: str = "cpu"):
        # weights = total / (num_classes * count_per_class)
        ...

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ...


class EnrichmentMSELoss(nn.Module):
    """
    MSE loss for the regression head operating on continuous enrichment scores.
    Huber loss (delta configurable) reduces sensitivity to outlier variants.
    """

    def __init__(self, delta: float = 1.0):
        ...

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ...
