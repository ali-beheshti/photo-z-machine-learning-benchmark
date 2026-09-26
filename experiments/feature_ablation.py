#!/usr/bin/env python
"""Evaluate one model across progressively richer feature sets."""

from pathlib import Path
import argparse
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from photoz_ml.data import load_table, clean_table, fixed_train_test_split
from photoz_ml.features import FEATURE_SETS
from photoz_ml.evaluation import fit_predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="rf")
    p.add_argument("--target", default="zspec")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="outputs/feature_ablation.csv")
    args = p.parse_args()

    df = clean_table(load_table(args.data), args.target)
    train, test = fixed_train_test_split(df, args.target, random_state=args.seed)

    rows = []
    for feature_set, columns in FEATURE_SETS.items():
        _, metrics = fit_predict(
            train, test, columns, args.model,
            target=args.target, random_state=args.seed
        )
        rows.append({"model": args.model, "feature_set": feature_set, **metrics})
        print(feature_set, metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
