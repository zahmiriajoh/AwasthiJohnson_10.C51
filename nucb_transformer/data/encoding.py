# Vocabulary constants and amino acid encoding utilities.
# Single source of truth for the AA alphabet, token IDs, and sequence length;
# also provides tokenization and one-hot conversion consumed by the Dataset.

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

VARIABLE_REGION_LENGTH = 14       # length of the mutagenized region
FULL_SEQUENCE_LENGTH = 142        # full NucB amino acid count

ACTIVITY_CLASSES = ["<WT", "WT", ">WT", ">=A73R"]
NUM_CLASSES = len(ACTIVITY_CLASSES)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "-"
VOCAB = PAD_TOKEN + AMINO_ACIDS   # index 0 = pad, 1-20 = canonical AAs
VOCAB_SIZE = len(VOCAB)           # 21

# ── lookup tables ─────────────────────────────────────────────────────────────

AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(VOCAB)}

# ── functions ─────────────────────────────────────────────────────────────────

def tokenize(sequence: str) -> list[int]:
    """Map a variable-region AA string to a list of integer token IDs."""
    ...


def pad_or_truncate(tokens: list[int], length: int = VARIABLE_REGION_LENGTH) -> list[int]:
    """Right-pad with 0 (pad token) or truncate to a fixed length."""
    ...


def to_one_hot(tokens: list[int], vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    """Convert token IDs to a (length, vocab_size) one-hot array."""
    ...


def mutation_string_to_sequence(mutation_str: str, wildtype: str) -> str:
    """Apply comma-delimited mutation string (e.g. 'A45C, A56T') to a wildtype sequence."""
    ...
