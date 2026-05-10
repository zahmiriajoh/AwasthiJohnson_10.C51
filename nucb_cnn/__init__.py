# nucb_cnn: sequence-to-activity CNN for NucB nuclease variants.
# Mirrors the 4-class classification task of the DeepMind G4 CNN
# (below-WT / WT / above-WT / >=A73R) using 1D convolutions instead of attention.

from nucb_cnn.models.cnn import NucleaseConvNet
