# Photometric Redshift ML Benchmark

Machine-learning comparison for estimating **photometric redshifts (photo-z)** from galaxy photometry and structural properties.

The project benchmarks multiple supervised-regression methods and studies how performance changes with training-set size, feature selection, preprocessing, and model hyperparameters.

**Tech:** Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, TensorFlow/Keras, SciPy, Matplotlib

## Overview

Spectroscopic redshifts are accurate but expensive to obtain at large scale. Photometric redshift estimation uses broadband imaging measurements to infer galaxy redshift efficiently.

This project evaluates several regression approaches on Bright Galaxy Survey data from the DESI Legacy Imaging Surveys:

- Random Forest
- K-Nearest Neighbors
- XGBoost
- CatBoost
- Multi-Layer Perceptron
- Keras neural network
- Gaussian Process Regression

The analysis focuses on four questions:

1. How do different ML models compare on the same prediction task?
2. Which galaxy features contribute most to prediction accuracy?
3. How does performance scale with training-set size?
4. How sensitive are the results to model hyperparameters?

## Metrics

For spectroscopic redshift \(z_{\rm spec}\) and predicted photometric redshift \(z_{\rm phot}\),

\[
\Delta z_{\rm norm} =
\frac{z_{\rm phot}-z_{\rm spec}}{1+z_{\rm spec}}.
\]

Performance is evaluated using:

- **NMAD:** \(1.48\,\mathrm{median}(|\Delta z_{\rm norm}|)\)
- **Bias:** median normalized residual
- **Outlier fraction:** fraction with \(|\Delta z_{\rm norm}| > 0.15\)
- **RMSE:** root-mean-square normalized residual

## Feature sets

The feature-ablation analysis uses progressively richer inputs:

```text
colors + magnitudes
        |
        + half-light radius
        |
        + categorical information
        |
        + structural / reddening features
```

Features include broadband and fiber magnitudes, galaxy size, morphology, photometric system, Sérsic index, reddening, ellipticity, and fit-quality information.

## Selected results

The full-feature Random Forest and CatBoost models gave the strongest results in the feature-ablation study.

| Model | NMAD | Outliers |
|---|---:|---:|
| Random Forest | 0.0250 | 0.863% |
| Weighted KNN | 0.0320 | 1.088% |
| XGBoost | 0.0328 | 1.204% |
| CatBoost | 0.0248 | 0.927% |
| MLP | 0.0308 | 1.062% |

The training-size analysis also showed an approximately power-law improvement in photo-z accuracy as the training sample increased.

Additional tables are available in `results/`.

## Repository structure

```text
src/photoz_ml/
  data.py          # loading and train/test preparation
  features.py      # feature engineering and feature sets
  metrics.py       # NMAD, bias, outlier fraction, RMSE
  models.py        # model definitions
  evaluation.py    # fitting, scoring, and diagnostic plots

experiments/
  benchmark_models.py
  feature_ablation.py
  training_size_scaling.py
  generate_synthetic_demo.py
  run_demo.sh

results/
  feature_ablation.csv
  model_settings.csv

docs/
  methodology.md

tests/
  test_metrics.py
  test_features.py
```

## Quick start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A small synthetic catalog is included as a convenient way to exercise the pipeline without downloading survey data:

```bash
python experiments/generate_synthetic_demo.py
```

Run the model comparison:

```bash
python experiments/benchmark_models.py \
  --data data/synthetic_galaxies.csv \
  --models rf knn_distance xgb catboost mlp
```

Run the feature-ablation analysis:

```bash
python experiments/feature_ablation.py \
  --data data/synthetic_galaxies.csv \
  --model rf
```

Run the training-size scaling experiment:

```bash
python experiments/training_size_scaling.py \
  --data data/synthetic_galaxies.csv \
  --model rf
```

Or run the compact demo:

```bash
bash experiments/run_demo.sh
```

Outputs are written to `outputs/`.

## Input format

The loader accepts CSV or Parquet tables with `zspec` as the default target column.

Typical photometric columns are:

```text
g_mag, r_mag, z_mag, w1_mag, w2_mag,
fiber_g_mag, fiber_r_mag, fiber_z_mag
```

Optional structural and categorical columns include:

```text
hlr, morphology, photsys, sersic, ebv, ellipticity, dchi2
```

Column mappings and feature groups can be edited in `src/photoz_ml/features.py`.

## Acknowledgments

Developed with **Yasha Kaushal** under the supervision of **Prof. Jeffrey A. Newman** at the University of Pittsburgh.
