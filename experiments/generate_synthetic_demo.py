#!/usr/bin/env python
"""Generate a galaxy-like tabular regression data set for smoke testing."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/synthetic_galaxies.csv")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    z = np.clip(rng.beta(2.0, 5.0, args.n) * 0.95, 0, 0.9)

    # Smooth nonlinear photometric trends plus heteroscedastic noise.
    base = 20.2 - 3.2 * z + 0.35 * np.sin(7 * z)
    r = base + rng.normal(0, 0.10 + 0.08 * z, args.n)
    g = r + 0.45 + 1.25 * z + rng.normal(0, 0.08, args.n)
    zmag = r - 0.25 - 0.75 * z + rng.normal(0, 0.07, args.n)
    w1 = zmag - 0.55 - 0.45 * z + rng.normal(0, 0.09, args.n)
    w2 = w1 - 0.10 - 0.15 * z + rng.normal(0, 0.08, args.n)

    fg = g + rng.normal(0.7, 0.20, args.n)
    fr = r + rng.normal(0.55, 0.18, args.n)
    fz = zmag + rng.normal(0.45, 0.18, args.n)

    df = pd.DataFrame({
        "g_mag": g, "r_mag": r, "z_mag": zmag, "w1_mag": w1, "w2_mag": w2,
        "fiber_g_mag": fg, "fiber_r_mag": fr, "fiber_z_mag": fz,
        "hlr": np.exp(rng.normal(-0.1 + 0.5*z, 0.25, args.n)),
        "morphology": rng.choice(["EXP", "DEV", "REX", "PSF"], args.n, p=[.35,.25,.35,.05]),
        "photsys": rng.choice(["N", "S"], args.n, p=[.45,.55]),
        "sersic": np.clip(rng.normal(2.0 + 1.0*z, 0.7, args.n), 0.2, 6),
        "ebv": np.abs(rng.normal(0.035, 0.018, args.n)),
        "ellipticity": np.clip(rng.beta(2, 5, args.n), 0, 0.95),
        "dchi2": rng.lognormal(2.0 + 0.5*z, 0.6, args.n),
        "zspec": z,
    })

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")


if __name__ == "__main__":
    main()
