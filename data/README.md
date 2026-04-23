# Data

Actual data files are not committed to this repository. Download them from the
DeepMind nuclease_design release and place them here before running any scripts.

## Required files

| File | Source | Description |
|---|---|---|
| `landscape.csv` | [google-deepmind/nuclease_design](https://github.com/google-deepmind/nuclease_design) | NGS counts for all 55,760 NucB variants across sort bins |
| `wildtype.fasta` | Same release | Full 158-AA NucB wildtype sequence in FASTA format |
| `liquid_culture.csv` | Same release | Liquid culture growth measurements per variant |

## Preprocessing

After downloading, run the preprocessing pipeline to produce the labeled CSV used by the DataLoader:

```bash
python scripts/train.py --config configs/default.yaml
# preprocessing runs automatically, or run it standalone:
python -c "from nucb_transformer.data.dataset import run_preprocessing_pipeline; run_preprocessing_pipeline('data/landscape.csv', 'data/landscape_labeled.csv')"
```

This produces `data/landscape_labeled.csv` with columns:
- `mutations` — comma-delimited mutation string (empty = wildtype)
- `activity_class` — one of `<WT`, `WT`, `>WT`, `>=A73R`
- `enrichment` — continuous log-enrichment score
