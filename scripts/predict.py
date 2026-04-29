"""
CLI entrypoint for scoring novel sequences.
Reads a CSV with either a 'sequence' column or a 'mutations' column,
and writes a CSV of predicted activity classes and per-class probabilities.

Usage:
    # input CSV has a 'sequence' column
    python scripts/predict.py \
        --checkpoint checkpoints/epoch_012_acc0.8234.pt \
        --input_csv my_variants.csv \
        --output_csv predictions.csv

    # input CSV has a 'mutations' column (standard notation e.g. "A33E,T50I")
    python scripts/predict.py \
        --checkpoint checkpoints/epoch_012_acc0.8234.pt \
        --input_csv my_variants.csv \
        --output_csv predictions.csv \
        --wildtype <FULL_WILDTYPE_AA_SEQUENCE>
"""

import argparse
import torch
import pandas as pd
from nucb_transformer.utils.io import load_checkpoint
from nucb_transformer.data.encoding import tokenize, ACTIVITY_CLASSES


def _apply_mutations(mutation_str: str, wildtype: str) -> str:
    """Apply a comma-separated mutation string (e.g. 'A33E,T50I') to a wildtype sequence."""
    seq = list(wildtype)
    for mut in mutation_str.split(","):
        mut = mut.strip()
        if not mut:
            continue
        pos = int(mut[1:-1]) - 1   # 1-indexed in standard notation
        seq[pos] = mut[-1]
    return "".join(seq)


def predict_sequences(
    model,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Score a list of AA strings; returns a DataFrame with class and per-class probs."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tokens = torch.tensor(
                [tokenize(s) for s in batch], dtype=torch.long
            ).to(device)
            probs = model.predict_proba(tokens).cpu()
            all_probs.append(probs)

    all_probs = torch.cat(all_probs).numpy()
    predicted_classes = [ACTIVITY_CLASSES[i] for i in all_probs.argmax(axis=1)]

    return pd.DataFrame(
        {
            "sequence": sequences,
            "predicted_class": predicted_classes,
            **{f"prob_{cls}": all_probs[:, i] for i, cls in enumerate(ACTIVITY_CLASSES)},
        }
    )


def parse_args():
    p = argparse.ArgumentParser(description="Score NucB variants with a trained checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input_csv", required=True,
                   help="CSV with a 'sequence' column, or a 'mutations' column + --wildtype")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--wildtype", default=None,
                   help="Full wildtype AA sequence; required when input CSV has 'mutations' column")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=args.device)

    df = pd.read_csv(args.input_csv)

    if "sequence" in df.columns:
        sequences = df["sequence"].tolist()
    elif "mutations" in df.columns:
        if not args.wildtype:
            raise ValueError("--wildtype is required when input CSV has a 'mutations' column")
        sequences = [_apply_mutations(m, args.wildtype) for m in df["mutations"]]
    else:
        raise ValueError("Input CSV must have a 'sequence' or 'mutations' column")

    results = predict_sequences(model, sequences, device)
    results.to_csv(args.output_csv, index=False)
    print(f"Scored {len(results):,} variants → {args.output_csv}")
