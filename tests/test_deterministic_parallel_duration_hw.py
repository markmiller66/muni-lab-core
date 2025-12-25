# tests/test_deterministic_parallel_duration_hw.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math
import pandas as pd

from muni_core.config import AppConfig
from muni_core.model import Bond, CallFeature
from muni_core.risk.deterministic_parallel_duration_hw import compute_parallel_duration_hw_det


def test_deterministic_parallel_duration_hw_moves_price():
    repo_root = Path(__file__).resolve().parents[1]
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    assert hist_path.exists(), f"Missing history file: {hist_path}"

    history_df = pd.read_parquet(hist_path)
    assert len(history_df) > 0

    # Minimal AppConfig: only provide cfg.curves.curve_asof_date
    # Set curve_asof_date to make the test deterministic.
    cfg = AppConfig(curves=SimpleNamespace(curve_asof_date="2025-11-26"))

    bond = Bond(
        cusip="01025QAX9",
        rating="A1",
        rating_num=17,
        basis="1",
        settle_date=pd.to_datetime("2024-02-05").date(),
        maturity_date=pd.to_datetime("2051-09-01").date(),
        coupon=0.04,
        clean_price=94.1234,
        quantity=100.0,
        call_feature=CallFeature(call_date=pd.to_datetime("2031-09-01").date(), call_price=100.0),
    )

    det = compute_parallel_duration_hw_det(
        bond=bond,
        history_df=history_df,
        app_cfg=cfg,
        curve_key="AAA_MUNI_SPOT",
        bump_bp=1.0,
        z_spread_bp=0.0,
        step_years=0.5,
        q=0.5,
        coupon_freq=2,
        face=100.0,
        debug=True,   # turns on bump sanity checks in your module
    )

    # Core contract: price must move under a parallel bump
    assert math.isfinite(det.base_price)
    assert math.isfinite(det.price_up)
    assert math.isfinite(det.price_down)

    # This is the actual point of the test:
    assert abs(det.price_down - det.price_up) > 1e-8, (
        f"Price did not move under parallel bump. "
        f"base={det.base_price}, up={det.price_up}, down={det.price_down}"
    )

    # DV01 should be non-zero if prices move
    assert abs(det.dv01_bp) > 1e-12
