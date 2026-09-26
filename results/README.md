# Results

This directory contains summary tables from the photometric-redshift model comparison.

- `feature_ablation.csv` — NMAD and outlier fraction for progressively richer feature sets
- `model_settings.csv` — selected model configurations used in the analysis

The feature-ablation study shows that Random Forest and CatBoost benefit strongly from the full feature set, while KNN performs best after adding galaxy size information.
