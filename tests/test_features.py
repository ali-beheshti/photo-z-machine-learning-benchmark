import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from photoz_ml.features import add_colors


def test_color_generation():
    df = pd.DataFrame({
        "g_mag": [20.0], "r_mag": [19.5], "z_mag": [19.2],
        "w1_mag": [18.8], "w2_mag": [18.7],
        "fiber_g_mag": [21.0], "fiber_r_mag": [20.4],
        "fiber_z_mag": [20.0],
    })
    out = add_colors(df)
    assert abs(out.loc[0, "g_r"] - 0.5) < 1e-12
    assert abs(out.loc[0, "w1_w2"] - 0.1) < 1e-12
