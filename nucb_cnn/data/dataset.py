# PyTorch Dataset and DataLoader factory.
# Returns (token_tensor, label_index) pairs — the same interface as the transformer dataset
# so both models can share a single DataLoader in comparison experiments.
# One-hot encoding is handled inside NucleaseConvNet.forward(), not here.

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nuclease_design import utils as _nd_utils
from nucb_cnn.data.encoding import tokenize


# ── preprocessing ─────────────────────────────────────────────────────────────

def load_landscape(path: str = None) -> pd.DataFrame:
    """
    Loads the raw landscape.csv which has estimated enzyme activity for 55,760 NucB variants.

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
    )
    return df


# ── dataset ───────────────────────────────────────────────────────────────────

class NucleaseDataset(Dataset):
    """
    Maps each row of the landscape CSV (sequence + activity label)
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
        tokens = tokenize(row["sequence"])
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        label_index = self.class_to_idx[row["activity_level"]]
        return token_tensor, label_index


# ── dataloader factory ────────────────────────────────────────────────────────

def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap each split in a NucleaseDataset and return three DataLoaders."""
    train_dataset = NucleaseDataset(train_df)
    val_dataset   = NucleaseDataset(val_df)
    test_dataset  = NucleaseDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
