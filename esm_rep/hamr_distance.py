import pandas as pd
import numpy as np
import ast
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Load data ─────────────────────────────────────────────────────────
pred = pd.read_csv('output/predictions.csv')
df = pred[(pred['split'] == 'test') & (pred['num_mutations'] == 3)].copy()


# ── Helper functions ───────────────────────────────────────────────────
def parse_positions(mut_str):
    """Return list of residue positions from a mutations tuple-of-tuples string."""
    try:
        tups = ast.literal_eval(mut_str)
        return [t[1] for t in tups] if tups else []
    except Exception:
        return []


def min_pairwise_dist(positions):
    """Minimum sequence distance between any two mutated residues."""
    if len(positions) < 2:
        return np.nan
    return min(abs(a - b) for a, b in itertools.combinations(positions, 2))


# ── Feature engineering ────────────────────────────────────────────────
df['positions'] = df['mutations'].apply(parse_positions)
df['min_dist']  = df['positions'].apply(min_pairwise_dist).astype('Int64')

# Subset to high-activity true classes
HIGH_ACTIVITY_CLASSES = ['activity > WT', 'activity > A73R']
df_high = df[df['activity_level'].isin(HIGH_ACTIVITY_CLASSES)].copy()
df_high['misclassified'] = (df_high['predicted_class'] == 'non-functional').astype(int)


# ── Distance bins ──────────────────────────────────────────────────────
dist_bins   = [0,  1,  2,  3,  5,  8, 12, 18, 100]
dist_labels = ['1', '2', '3', '4–5', '6–8', '9–12', '13–18', '19+']

df_high['dist_bin'] = pd.cut(df_high['min_dist'], bins=dist_bins, labels=dist_labels)


# ── Compute misclassification rate + 95% CI per bin ───────────────────
rows = []
for dlabel in dist_labels:
    sub = df_high[df_high['dist_bin'] == dlabel]
    n   = len(sub)
    k   = int(sub['misclassified'].sum())
    if n == 0:
        continue
    rate          = k / n
    rows.append({'dist_bin': dlabel, 'n': n, 'k': k,
                 'rate': rate})

res = pd.DataFrame(rows)


# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x     = np.arange(len(res))
color = '#D94F3D'

ax.bar(x, res['rate'], color=color, alpha=0.75, zorder=3, width=0.6)


# Annotate sample size above each bar
for xi, (_, row) in zip(x, res.iterrows()):
    ax.text(xi, row['rate'] + 0.002, f"n={row['n']}",
            ha='center', va='bottom', fontsize=7.5, color='#444')

ax.set_xticks(x)
ax.set_xticklabels(res['dist_bin'], fontsize=10)
ax.set_xlabel('Minimum distance between any two mutated residues', fontsize=10)
ax.set_ylabel('High-activity misclassification rate', fontsize=10)
ax.set_title(
    'High-activity triple mutants misclassified as non-functional', 
    fontsize=11.5, pad=10,
)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('output/triple_mutant_misclassification_vs_mindist.png', dpi=160, bbox_inches='tight')
print("Saved: triple_mutant_misclassification_vs_mindist.png")