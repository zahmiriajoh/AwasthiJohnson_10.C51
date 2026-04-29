# Loss functions for classification and regression objectives.
# WeightedActivityLoss is the primary loss: it up-weights the rare high-activity
# classes ('activity > WT', 'activity > A73R') so the model doesn't collapse to
# predicting 'non-functional'.

import torch
import torch.nn as nn

_NUM_CLASSES = 4  # non-functional | activity > 0 | activity > WT | activity > A73R


class ActivityCrossEntropy(nn.Module):
    """Standard cross-entropy over the 4 activity classes (no re-weighting)."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)


class WeightedActivityLoss(nn.Module):
    """
    Cross-entropy with inverse-frequency class weights computed from training labels.
    Pass class_counts (list of 4 ints in ACTIVITY_CLASSES order) at init time.
    """

    def __init__(self, class_counts: list[int], device: str = "cpu"):
        # weights = total / (num_classes * count_per_class)
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (_NUM_CLASSES * counts)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, labels)
