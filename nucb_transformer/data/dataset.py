# PyTorch Dataset and DataLoader factory.
# Also contains the NGS preprocessing pipeline that converts raw landscape.csv
# (counts per sort bin) into a labeled DataFrame ready for training.

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from nuclease_design import utils as _nd_utils
from nucb_transformer.data.encoding import (
    tokenize, pad_or_truncate, mutation_string_to_sequence,
    VARIABLE_REGION_LENGTH, ACTIVITY_CLASSES,
)

# ── preprocessing ─────────────────────────────────────────────────────────────

def load_landscape(path: str = None) -> pd.DataFrame:
    """
    Loads the raw landscape.csv which has estimated enzyme activity for 55,760 NucB variants

    For each variant, the following is provided:
- mutations: e.g. ((A,33,E),(T,50,I))
- num_mutations: e.g. 2
- sublibrary_names: e.g. (g3_unmatched,)
- generations: e.g. (g3,)
- activity_level: e.g. "non-functional"
- is_functional: e.g. False
- sequence

    """
    df = _nd_utils.load_landscape()

    df['activity_level'] = (
    df['activity_level'].str.replace('_greater_than_', ' > ')
) #later, these labels will be converted to integers, but this makes them more readable for now.
    return df

# There is no other preprocessing currently written.
# May need to consider any preprocessing done in for the CNN demonstration
# but this is sufficient for relating sequence to activity label.

# ── dataset ───────────────────────────────────────────────────────────────────

class NucleaseDataset(Dataset):
    """
    Maps each row of the landscape CSV (mutation string + activity label)
    to a (token_tensor, label_index) pair.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        classes = sorted(df["activity_level"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # X: tokenize the full sequence directly
        tokens = tokenize(row["sequence"])
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        # y: map activity_level string to integer index
        label_index = self.class_to_idx[row["activity_level"]]

        return token_tensor, label_index
# ── dataloader factory ────────────────────────────────────────────────────────

## Not sure there's a need for collate_fn. All sequences are 142 cause we're not selecting region. Trying to see long distance effects.
# def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Pads sequences in a batch to equal length and builds boolean padding masks."""
#     ...


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap each split in a NucleaseDataset and return three DataLoaders."""

    train_dataset = NucleaseDataset(train_df)
    val_dataset = NucleaseDataset(val_df)
    test_dataset = NucleaseDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader