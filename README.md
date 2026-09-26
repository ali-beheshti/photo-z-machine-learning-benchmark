# Photometric Redshift ML Benchmark

A reproducible machine-learning workflow for estimating **photometric redshifts (photo-z)** from galaxy photometry and structural properties.

The project compares tree-based models, nearest-neighbor regression, gradient boosting, neural networks, and Gaussian-process regression using astronomy-specific diagnostics such as **NMAD**, bias, RMSE, and catastrophic-outlier fraction.

**Tech:** Python, pandas, NumPy, scikit-learn, XGBoost, CatBoost, TensorFlow/Keras (optional), SciPy, Matplotlib

## Analysis workflow

The benchmark is organized around four questions:

1. **Model comparison** — how accurately do different regression algorithms predict spectroscopic redshift?
2. **Feature ablation** — how does performance change as size, categorical, and structural information are added?
3. **Training-size scaling** — how quickly does prediction quality improve as the training sample grows?
4. **Hyperparameter sensitivity** — how strongly do model choices such as tree depth, estimator count, neighbor count, and network size affect the results?

Models supported by the code include:

- Random Forest
- weighted / unweighted K-Nearest Neighbors
- XGBoost
- CatBoost
- scikit-learn MLP
- Keras neural network (optional)
- Gaussian Process Regression

## Metrics

For spectroscopic redshift \(z_\mathrm{spec}\) and predicted photometric redshift \(z_\mathrm{phot}\),

\[
\Delta z_\mathrm{norm} = \frac{z_\mathrm{phot}-z_\mathrm{spec}}{1+z_\mathrm{spec}}.
\]

The main diagnostics are:

- **NMAD:** \(1.48\,\mathrm{median}(|\Delta z_\mathrm{norm}|)\)
- **Bias:** \(\mathrm{median}(\Delta z_\mathrm{norm})\)
- **Outlier fraction:** fraction with \(|\Delta z_\mathrm{norm}| > 0.15\)
- **RMSE:** root-mean-square normalized redshift error

## Feature sets

The pipeline supports progressive feature sets similar to those used in the analysis:

```text
colors + magnitudes
        |
        + half-light radius
        |
        + categorical information
        |
        + structural / reddening features
```

The default feature builder derives six colors from eight magnitude-like inputs and adds:

- half-light radius
- morphology
- photometric system
- Sérsic index
- reddening (EBV)
- ellipticity
- Δχ²

Column names and feature groups are configurable in `src/photoz_ml/features.py`. See `docs/experiment_spec.md` for the experiment specification preserved from the project material.

## Repository structure

```text
src/photoz_ml/
  data.py          # table loading and train/test preparation
  features.py      # feature engineering and feature-set definitions
  metrics.py       # photo-z metrics
  models.py        # model registry
  evaluation.py    # fitting, scoring, and diagnostic plots

experiments/
  benchmark_models.py
  feature_ablation.py
  training_size_scaling.py
  generate_synthetic_demo.py

results/
  reference_feature_ablation.csv
  reference_hyperparameters.csv

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

Generate a synthetic galaxy catalog:

```bash
python experiments/generate_synthetic_demo.py
```

Run a compact benchmark:

```bash
python experiments/benchmark_models.py \
  --data data/synthetic_galaxies.csv \
  --models rf knn_distance xgb catboost mlp
```

Run the feature-ablation study:

```bash
python experiments/feature_ablation.py \
  --data data/synthetic_galaxies.csv \
  --model rf
```

Run the training-size scaling analysis:

```bash
python experiments/training_size_scaling.py \
  --data data/synthetic_galaxies.csv \
  --model rf
```

Generated tables and plots are written to `outputs/`.

For a one-command smoke test:

```bash
bash experiments/run_demo.sh
```

## Input data

The loader accepts CSV or Parquet tables. The target column defaults to:

```text
zspec
```

A typical table can contain magnitude columns such as

```text
g_mag, r_mag, z_mag, w1_mag, w2_mag,
fiber_g_mag, fiber_r_mag, fiber_z_mag
```

plus optional structural/categorical columns such as

```text
hlr, morphology, photsys, sersic, ebv, ellipticity, dchi2
```

The synthetic-data generator creates this schema automatically.

## Reference results

`results/reference_feature_ablation.csv` records the feature-ablation metrics from the original analysis for comparison with future runs. The values are **reference measurements**, not hard-coded model outputs.

The original analysis found that Random Forest and CatBoost benefited most from the full feature set, while KNN performed best after adding size information. Random Forest and the Keras network gave the strongest overall photo-z performance among the tested configurations.

## Notes on reproducibility

Exact numerical results depend on the source catalog, filtering cuts, train/test realization, preprocessing, and library versions. The code is therefore designed to reproduce the **analysis workflow and diagnostics** rather than force a particular set of metric values.


## Acknowledgments

The original graduate project was completed with **Yasha Kaushal** under the supervision of **Prof. Jeffrey A. Newman** at the University of Pittsburgh.
