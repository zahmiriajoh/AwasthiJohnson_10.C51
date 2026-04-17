# nucb_transformer: sequence-to-activity transformer for NucB nuclease variants.
# Mirrors the 4-class classification task of the DeepMind CNN
# (below-WT / WT / above-WT / >=A73R) but uses attention instead of convolution.

from nucb_transformer.models.transformer import NucleaseTransformer
