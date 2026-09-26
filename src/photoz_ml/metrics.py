"""Photo-z performance metrics."""

from __future__ import annotations
import numpy as np


def normalized_residual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return (y_pred - y_true) / (1.0 + y_true)


def nmad(y_true, y_pred) -> float:
    """Normalized median absolute deviation."""
    dz = normalized_residual(y_true, y_pred)
    return float(1.48 * np.median(np.abs(dz)))


def bias(y_true, y_pred) -> float:
    """Median normalized redshift residual."""
    return float(np.median(normalized_residual(y_true, y_pred)))


def outlier_fraction(y_true, y_pred, threshold: float = 0.15) -> float:
    """Fraction of objects with |Δz|/(1+z) above threshold."""
    dz = np.abs(normalized_residual(y_true, y_pred))
    return float(np.mean(dz > threshold))


def rmse(y_true, y_pred) -> float:
    """Root-mean-square normalized redshift residual."""
    dz = normalized_residual(y_true, y_pred)
    return float(np.sqrt(np.mean(dz**2)))


def score_photoz(y_true, y_pred) -> dict[str, float]:
    return {
        "nmad": nmad(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "outlier_fraction": outlier_fraction(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
