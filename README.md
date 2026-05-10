# nucb_transformer

A transformer-based sequence-to-activity classifier for NucB nuclease variants, built for comparison to the CNN described in:

[google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design)

---

## What this does

The original CNN takes variable-region amino acid sequences from NucB variants and classifies each into one of four activity tiers:

| Class | Meaning |
|---|---|
| `non-functional` | No function |
| `activity > 0` | Low function, below WT |
| `activity > WT` | Above WT activity |
| `activity > A73R` | Above the A73R benchmark variant |

This repo replaces the CNN with a **transformer encoder** that attends globally over residue positions, hypothetically letting it capture long-range epistatic interactions that local convolution misses. The label space, preprocessing pipeline, and evaluation metrics are kept identical for direct comparison.


## Quickstart

```bash
# 1. Set up nucb environment

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# This installs both packages (nucb_transformer and nucb_cnn) as editable installs, along with all dependencies including the DeepMind nuclease_design package (which includes the data).

# 2. Train
python scripts/train_trans.py --config configs/best_trans.yaml   # transformer
python scripts/train_cnn.py   --config configs/best_cnn.yaml     # CNN
# The "best" configs are from the hyperparameter sweep
# Checkpoints are saved to checkpoints/training_trans/ and checkpoints/training_cnn/ as epoch_NNN_accX.XXXX.pt

# 3. Evaluate on the held-out test set (automatically uses best training checkpoint)
python scripts/evaluate_trans.py --save_predictions results/trans_test_predictions.csv
python scripts/evaluate_cnn.py   --save_predictions results/cnn_test_predictions.csv

# 4. Predict on new variants
# The 03_predict_sandbox.ipynb contains the code to poke around the data and test on a few novel sequences


## Repo structure
```bash
AwasthiJohnson_10.C51/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/ #Reflect the defaults or sweep results
│   ├── default_trans.yaml
│   ├── default_cnn.yaml
│   ├── best_trans.yaml
│   ├── best_cnn.yaml
│
├── data/
│   ├── README.md
│   └── .gitkeep
│
├── checkpoints/ # Contain the best checkpoints from sweeping and training
│   ├── training_trans/
│   ├── training_cnn/
│   ├── sweep_trans/
│   └── sweep_cnn/
│
├── nucb_transformer/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── encoding.py
│   │   └── splits.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── positional.py
│   │   └── heads.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── logging.py
│
├── nucb_cnn/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── encoding.py
│   │   └── splits.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   └── heads.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── logging.py
│
├── esm_rep/
│   ├── config.yaml
│   ├── data_utils.py
│   ├── embed.py
│   ├── train.py
│   ├── train_maxpool.py
│   ├── confusion_matrix_gen_split.py
│   ├── confusion_matrix_nmut_split.py
│   ├── confusion_matrix_nmut_split_maxpool.py
│   ├── cosine_per_residue_v2.py
│   ├── hamr_distance.py
│   ├── tsne_maxpool.py
│   ├── tsne_wt_vs_a73r_v2.py
│   └── output/
│
├── scripts/ #Use to train and use the CNN and transformer
│   ├── train_trans.py
│   ├── train_cnn.py
│   ├── evaluate_trans.py
│   ├── evaluate_cnn.py
│   ├── predict_trans.py
│   ├── predict_cnn.py
│
├── notebooks/ #Use to replicate figures and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_predict_sandbox.ipynb
│
└── tests/
    ├── test_encoding.py
    ├── test_dataset.py
    ├── test_model.py
    └── test_metrics.py

    ```