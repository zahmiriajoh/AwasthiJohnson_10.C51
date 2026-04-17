# nucb_transformer

A transformer-based sequence-to-activity classifier for NucB nuclease variants, built for comparison to the CNN described in:

Reference CNN implementation: [google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design)

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

## Repo structure

nucb_transformer/
│
├── README.md
├── LICENSE
├── pyproject.toml           # or setup.py / setup.cfg
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── default.yaml         # model hyperparams, training settings
│   └── sweep.yaml           # hyperparameter sweep config (e.g. for W&B)
│
├── data/
│   ├── README.md            # how to download landscape.csv from GCS
│   └── .gitkeep             # keep dir tracked; actual data not committed
│
├── nucb_transformer/        # installable package
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # PyTorch Dataset: loads landscape.csv, one-hot encodes
│   │   ├── encoding.py      # one-hot encoder (AA vocab, padding logic)
│   │   └── splits.py        # train/val/test split logic (round-aware, as in paper)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py   # TransformerEncoder + regression/classification head
│   │   ├── positional.py    # sinusoidal or learned positional embeddings
│   │   └── heads.py         # regression head (MBO-DNN style) and/or class head
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # training loop, validation, early stopping
│   │   ├── losses.py        # MSE for regression, cross-entropy for class labels
│   │   └── metrics.py       # Spearman ρ, hit rate, enrichment factor
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py            # checkpoint save/load helpers
│       └── logging.py       # W&B / TensorBoard setup
│
├── scripts/
│   ├── train.py             # CLI entrypoint: reads config, launches training
│   ├── evaluate.py          # loads checkpoint, runs test-set eval
│   └── predict.py           # scores arbitrary FASTA / CSV of sequences
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb   # compare transformer vs CNN baseline
│   └── 03_results_figures.ipynb    # reproduce paper-style figures
│
└── tests/
    ├── test_encoding.py
    ├── test_dataset.py
    ├── test_model.py        # forward pass shape checks, no NaN outputs
    └── test_metrics.py