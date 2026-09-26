#!/usr/bin/env python
"""Compare multiple regression models on a fixed 80/20 split."""

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from photoz_ml.data import load_table, clean_table, fixed_train_test_split
from photoz_ml.features import get_feature_columns
from photoz_ml.evaluation import fit_predict, save_photoz_plot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--feature-set", default="all")
    p.add_argument(
        "--models", nargs="+",
        default=["rf", "knn_distance", "xgb", "catboost", "mlp"]
    )
    p.add_argument("--target", default="zspec")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/benchmark")
    args = p.parse_args()

    df = clean_table(load_table(args.data), args.target)
    train, test = fixed_train_test_split(df, args.target, random_state=args.seed)
    features = get_feature_columns(args.feature_set)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in args.models:
        pred, metrics = fit_predict(
            train, test, features, model_name,
            target=args.target, random_state=args.seed
        )
        rows.append({"model": model_name, **metrics})
        save_photoz_plot(
            test[args.target].to_numpy(),
            pred,
            outdir / f"{model_name}_photoz_vs_specz.png",
            title=model_name,
        )
        print(model_name, metrics)

    pd.DataFrame(rows).to_csv(outdir / "model_metrics.csv", index=False)


if __name__ == "__main__":
    main()
