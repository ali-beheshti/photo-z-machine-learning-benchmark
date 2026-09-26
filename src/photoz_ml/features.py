"""Feature engineering for photometric-redshift regression."""

from __future__ import annotations
import pandas as pd

MAG_COLUMNS = [
    "g_mag", "r_mag", "z_mag", "w1_mag", "w2_mag",
    "fiber_g_mag", "fiber_r_mag", "fiber_z_mag",
]

COLOR_DEFINITIONS = {
    "g_r": ("g_mag", "r_mag"),
    "r_z": ("r_mag", "z_mag"),
    "z_w1": ("z_mag", "w1_mag"),
    "w1_w2": ("w1_mag", "w2_mag"),
    "fiber_g_r": ("fiber_g_mag", "fiber_r_mag"),
    "fiber_r_z": ("fiber_r_mag", "fiber_z_mag"),
}

# Two representative magnitude features used alongside the six colors.
BASE_MAGNITUDES = ["r_mag", "w1_mag"]

FEATURE_SETS = {
    "colors_mags": list(COLOR_DEFINITIONS) + BASE_MAGNITUDES,
    "colors_mags_hlr": list(COLOR_DEFINITIONS) + BASE_MAGNITUDES + ["hlr"],
    "colors_mags_cat": list(COLOR_DEFINITIONS) + BASE_MAGNITUDES + [
        "morphology", "photsys"
    ],
    "all": list(COLOR_DEFINITIONS) + BASE_MAGNITUDES + [
        "hlr", "morphology", "photsys", "sersic", "ebv",
        "ellipticity", "dchi2",
    ],
}


def add_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (a, b) in COLOR_DEFINITIONS.items():
        if a not in out or b not in out:
            raise KeyError(f"Cannot build {name}: missing {a!r} or {b!r}")
        out[name] = out[a] - out[b]
    return out


def get_feature_columns(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise KeyError(
            f"Unknown feature set {feature_set!r}. "
            f"Choose from {sorted(FEATURE_SETS)}"
        )
    return FEATURE_SETS[feature_set]


def split_feature_types(columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [c for c in columns if c in {"morphology", "photsys"}]
    numerical = [c for c in columns if c not in categorical]
    return numerical, categorical
