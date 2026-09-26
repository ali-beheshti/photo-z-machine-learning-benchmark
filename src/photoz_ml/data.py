"""Data loading helpers."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import add_colors


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Supported formats: CSV and Parquet")
    return add_colors(df)


def clean_table(df: pd.DataFrame, target: str = "zspec") -> pd.DataFrame:
    if target not in df:
        raise KeyError(f"Target column {target!r} not found")
    out = df.replace([float("inf"), float("-inf")], pd.NA)
    return out.dropna(subset=[target]).reset_index(drop=True)


def fixed_train_test_split(
    df: pd.DataFrame,
    target: str = "zspec",
    test_size: float = 0.20,
    random_state: int = 42,
):
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
