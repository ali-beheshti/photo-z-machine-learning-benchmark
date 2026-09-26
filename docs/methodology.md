# Methodology

## Data

The analysis uses a Bright Galaxy Survey sample from the DESI Legacy Imaging Surveys, with spectroscopic redshift as the regression target and imaging-derived galaxy properties as input features.

The feature groups include photometric colors and magnitudes, half-light radius, morphology, photometric system, Sérsic index, reddening, ellipticity, and fit-quality information.

## Metrics

Model performance is evaluated using normalized redshift residuals,

\[
\Delta z_{\rm norm} = \frac{z_{\rm phot}-z_{\rm spec}}{1+z_{\rm spec}},
\]

with four main diagnostics:

- **NMAD:** \(1.48\,\mathrm{median}(|\Delta z_{\rm norm}|)\)
- **Bias:** \(\mathrm{median}(\Delta z_{\rm norm})\)
- **Outlier fraction:** fraction with \(|\Delta z_{\rm norm}| > 0.15\)
- **RMSE:** root-mean-square normalized residual

## Feature analysis

Four progressively richer feature sets are compared:

1. six colors + two magnitudes
2. colors + magnitudes + half-light radius
3. colors + magnitudes + categorical features
4. full photometric, structural, and reddening feature set

## Training-size scaling

A fixed 20% test sample is used while the training sample is increased from 1.25% to 80% of the full catalog in factors of two. NMAD and outlier fraction are fit as functions of training-set size.

## Models

The benchmark includes:

- Random Forest
- weighted and unweighted K-Nearest Neighbors
- XGBoost
- CatBoost
- Multi-Layer Perceptron
- Keras neural network
- Gaussian Process Regression

Selected model settings used in the analysis are stored in `results/model_settings.csv`.
