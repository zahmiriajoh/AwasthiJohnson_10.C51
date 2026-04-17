# nucb_transformer

A transformer-based sequence-to-activity classifier for NucB nuclease variants, built for comparison to the CNN in [google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design)

---

## What this does

The original CNN takes variable-region amino acid sequences from NucB variants and classifies each into one of four activity tiers:

| Class | Meaning |
|---|---|
| `<WT` | Below wildtype activity |
| `WT` | Wildtype-level activity |
| `>WT` | Above wildtype activity |
| `>=A73R` | At or above the A73R benchmark variant |

This repo replaces the CNN with a **transformer encoder** that attends globally over residue positions, hypothetically letting it capture long-range epistatic interactions that local convolution misses. The label space, preprocessing pipeline, and evaluation metrics are kept identical for direct comparison.

### Architecture mapping

## Repo structure

## Quickstart NOT ACTUALLY WRITTEN YET, WILL NOT WORK

```bash
# 1. Install (editable)
pip install -e ".[dev]"

# 2. Download data — see data/README.md for instructions

# 3. Train
python scripts/train.py --config configs/default.yaml

# 4. Evaluate on the held-out test set
python scripts/evaluate.py --checkpoint checkpoints/best.pt

# 5. Predict on new variants
python scripts/predict.py \
    --checkpoint checkpoints/best.pt \
    --input_csv my_variants.csv \
    --output_csv predictions.csv \
    --wildtype <WILDTYPE_AA_SEQUENCE>

# 6. Run tests
pytest tests/
```

## Key hyperparameters NEED TO CHECK THIS

See [configs/default.yaml](configs/default.yaml) for the full list. Defaults are chosen to match the CNN's capacity:

| Parameter | Value | Rationale |
|---|---|---|
| `num_layers` | 3 | Mirrors 3 CNN conv blocks |
| `d_model` | 128 | Comparable parameter count to conv-32 + dense-64 |
| `num_heads` | 4 | Divides `d_model` evenly |
| `dropout` | 0.05 | Matches CNN default |
| `pos_encoding` | sinusoidal | No extra parameters; works well on short sequences |

To run a hyperparameter sweep: `wandb sweep configs/sweep.yaml`.
