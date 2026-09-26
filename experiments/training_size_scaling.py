#!/usr/bin/env python
"""Measure photo-z performance as a function of training-set size."""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from photoz_ml.data import load_table, clean_table, fixed_train_test_split
from photoz_ml.features import get_feature_columns
from photoz_ml.evaluation import fit_predict


FRACTIONS = [0.0125, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80]


def scaling_model(n, floor, amplitude, alpha):
    return floor + amplitude * np.power(n, alpha)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="rf")
    p.add_argument("--feature-set", default="all")
    p.add_argument("--target", default="zspec")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/scaling")
    args = p.parse_args()

    df = clean_table(load_table(args.data), args.target)
    train_pool, test = fixed_train_test_split(
        df, args.target, test_size=0.20, random_state=args.seed
    )
    features = get_feature_columns(args.feature_set)

    rng = np.random.default_rng(args.seed)
    rows = []
    for frac_total in FRACTIONS:
        n = max(100, int(round(frac_total * len(df))))
        n = min(n, len(train_pool))
        idx = rng.choice(len(train_pool), size=n, replace=False)
        train = train_pool.iloc[idx].reset_index(drop=True)

        _, metrics = fit_predict(
            train, test, features, args.model,
            target=args.target, random_state=args.seed
        )
        rows.append({"train_fraction_total": frac_total, "train_size": n, **metrics})
        print(n, metrics)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "training_size_metrics.csv", index=False)

    x = result["train_size"].to_numpy(float)
    for metric in ["nmad", "outlier_fraction"]:
        y = result[metric].to_numpy(float)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, label="measured")

        try:
            popt, _ = curve_fit(
                scaling_model, x, y,
                p0=(max(0.0, y.min() * .8), max(y.max(), 1e-4), -0.3),
                maxfev=20000,
            )
            xx = np.geomspace(x.min(), x.max(), 200)
            ax.plot(
                xx, scaling_model(xx, *popt),
                label=f"{popt[0]:.3g} + {popt[1]:.3g} N^{popt[2]:.2f}"
            )
        except Exception:
            pass

        ax.set_xscale("log")
        ax.set_xlabel("training sample size")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"{metric}_vs_training_size.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
