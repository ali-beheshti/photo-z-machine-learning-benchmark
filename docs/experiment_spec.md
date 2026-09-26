# Experiment specification

The analysis compares supervised regression methods for estimating galaxy
photometric redshift from imaging-derived features.

## Data described in the project notes

- DESI / Legacy Imaging Surveys Bright Galaxy Survey sample
- roughly 75,000–80,000 objects
- redshift range approximately `0 < z < 0.9`
- spectroscopic redshift used as the target
- photometric, size, morphology, photometric-system, reddening, Sérsic,
  ellipticity, and fit-quality information used as candidate features

## Evaluation

The main metrics are NMAD, median normalized bias, catastrophic-outlier
fraction at `|Δz|/(1+z) > 0.15`, and normalized RMSE.

The training-size study uses a fixed 20% test sample and training fractions
from 1.25% through 80% of the full sample, increasing by factors of two.

## Feature ablation

Four progressively richer feature sets are evaluated:

1. six colors + two magnitudes
2. colors + magnitudes + half-light radius
3. colors + magnitudes + categorical information
4. all available structural / reddening features

The implementation keeps the raw-column mapping configurable because the
project notes do not fully specify every original column choice or filtering cut.

## Preserved model settings

Settings explicitly recorded in the project material include:

- Random Forest: 20 trees, maximum depth 20
- KNN: best outlier-fraction configurations at 20 neighbors (uniform)
  and 30 neighbors (distance weighting)
- CatBoost: 40 estimators
- MLP: 50 maximum iterations, random state 2
- Keras: two hidden layers, ReLU, L2 regularization, batch size 200,
  40 epochs, 60 units
- Gaussian-process study: KISS-GP / LOVE, 30 training iterations,
  grid size 200

For settings not recorded in the project notes, the implementation uses
library defaults or exposes the choice in code rather than assigning a
historical value.
