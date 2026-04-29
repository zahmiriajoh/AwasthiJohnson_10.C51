# Vocabulary constants and amino acid encoding utilities.
# Single source of truth for the AA alphabet, token IDs, and sequence length;
# also provides tokenization and one-hot conversion consumed by the Dataset.

import numpy as np
from sklearn.preprocessing import LabelBinarizer

# ── constants ────────────────────────────────────────────────────────────────

ACTIVITY_CLASSES = ['activity > 0', 'non-functional', 'activity > WT',
       'activity > A73R']
NUM_CLASSES = len(ACTIVITY_CLASSES)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

# ── OHE ─────────────────────────────────────────────────────────────────
_lb = LabelBinarizer().fit(list(amino_acids))

def seq_to_one_hot(sequence: str) -> np.ndarray:
    """Convert an AA string to a (length*20,) one-hot array."""

    struct_ohe = _lb.transform(list(sequence)).astype(np.float32)
    ohe = struct_ohe.ravel()

    return ohe
