import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from photoz_ml.metrics import nmad, bias, outlier_fraction, rmse


def test_perfect_predictions():
    z = np.array([0.1, 0.2, 0.4])
    assert nmad(z, z) == 0
    assert bias(z, z) == 0
    assert outlier_fraction(z, z) == 0
    assert rmse(z, z) == 0
