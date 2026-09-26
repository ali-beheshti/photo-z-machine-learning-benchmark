#!/usr/bin/env bash
set -euo pipefail

python experiments/generate_synthetic_demo.py --n 5000
python experiments/benchmark_models.py \
  --data data/synthetic_galaxies.csv \
  --models rf knn_distance xgb catboost mlp
python experiments/feature_ablation.py \
  --data data/synthetic_galaxies.csv \
  --model rf
