from pathlib import Path
import pandas as pd

from muni_core.config import AppConfig
from muni_core.curves.loader import build_default_curve_config  # or whatever you used to fix AppConfig(curves=...)
from muni_core.curves.hw_curve_builders import build_hw_curve_bundle

def test_build_hw_curve_bundle_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    history_df = pd.read_parquet(hist_path)

    curves_cfg = build_default_curve_config()   # use the same pattern you used when the pytest error disappeared
    cfg = AppConfig(curves=curves_cfg)

    bundle = build_hw_curve_bundle(history_df=history_df, app_cfg=cfg)

    assert len(bundle.dense_df) > 0
    assert len(bundle.theta_df) > 0
    assert bundle.asof is not None
