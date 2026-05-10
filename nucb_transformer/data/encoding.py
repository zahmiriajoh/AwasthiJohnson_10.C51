# Vocabulary constants and amino acid encoding utilities.
# Single source of truth for the AA alphabet, token IDs, and sequence length;
# also provides tokenization and one-hot conversion consumed by the Dataset.

import numpy as np
from sklearn.preprocessing import LabelBinarizer

# ── constants ────────────────────────────────────────────────────────────────

ACTIVITY_CLASSES = ['activity > 0', 'activity > A73R', 'activity > WT', 'non-functional']
NUM_CLASSES = len(ACTIVITY_CLASSES)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(amino_acids)}

# ── tokenization ──────────────────────────────────────────────────────────────

def tokenize(sequence: str) -> list[int]:
    """Map an AA string to a list of integer token IDs (0–19)."""
    return [AA_TO_IDX[aa] for aa in sequence]

# ── OHE ─────────────────────────────────────────────────────────────────
_lb = LabelBinarizer().fit(list(amino_acids))

def seq_to_one_hot(sequence: str) -> np.ndarray:
    """Convert an AA string to a (length*20,) one-hot array."""

    struct_ohe = _lb.transform(list(sequence)).astype(np.float32)
    ohe = struct_ohe.ravel()

    return ohe
