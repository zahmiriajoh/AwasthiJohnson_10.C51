"""
CLI entrypoint for scoring arbitrary sequences or mutation strings.
Reads a CSV with a 'mutations' column, applies each mutation to the wildtype
sequence, and writes a CSV of predicted activity classes and probabilities.

Usage:
    python scripts/predict.py \
        --checkpoint checkpoints/best.pt \
        --input_csv my_variants.csv \
        --output_csv predictions.csv \
        --wildtype <WILDTYPE_AA_SEQUENCE>
"""

import argparse
import torch
import pandas as pd
from nucb_transformer.utils.io import load_checkpoint
from nucb_transformer.data.encoding import tokenize, pad_or_truncate, mutation_string_to_sequence
from nucb_transformer.data.encoding import ACTIVITY_CLASSES, VARIABLE_REGION_LENGTH


def predict_sequences(model, sequences: list[str], device: str, batch_size: int = 512) -> pd.DataFrame:
    """
    Score a list of variable-region AA strings.
    Returns DataFrame: sequence, predicted_class, prob_<WT, prob_WT, prob_>WT, prob_>=A73R.
    """
    ...


def predict_mutations(model, mutation_strings: list[str], wildtype_seq: str, **kwargs) -> pd.DataFrame:
    """Convert mutation strings → sequences → predict_sequences."""
    ...


def parse_args():
    p = argparse.ArgumentParser(description="Score NucB variants with a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_csv", required=True, help="CSV with a 'mutations' column")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--wildtype", required=True, help="Full wildtype AA sequence")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    df = pd.read_csv(args.input_csv)
    results = predict_mutations(model, df["mutations"].tolist(), args.wildtype, device=args.device)
    results.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")
