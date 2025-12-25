from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd

from muni_core.curves.history import build_dense_zero_curve_for_date
from muni_core.curves.build_curve import build_dense_curve_from_history


def test_build_dense_curve_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    assert hist_path.exists()

    history_df = pd.read_parquet(hist_path)
    assert len(history_df) > 0

    # pick an asof you know exists (or compute max date externally)
    asof = date(2025, 11, 26)
    curve_key = "AAA_MUNI_SPOT"

    built = build_dense_curve_from_history(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=0.5,
        build_dense_zero_curve_for_date=build_dense_zero_curve_for_date,
    )

    df = built.df
    assert "tenor_yrs" in df.columns
    assert "rate_dec" in df.columns
    assert df["tenor_yrs"].min() >= 0
    assert df["rate_dec"].notna().any()

    # sanity: sorted by tenor
    assert (df["tenor_yrs"].values[:-1] <= df["tenor_yrs"].values[1:]).all()

    # optional: print a quick peek when running interactively
    print("\n[Dense Curve]")
    print(df[["tenor_yrs", "rate_dec"]].head(5).to_string(index=False))
    print("...")
    print(df[["tenor_yrs", "rate_dec"]].tail(5).to_string(index=False))
