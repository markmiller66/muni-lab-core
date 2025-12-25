from __future__ import annotations

"""
FILE: tests/test_curve_bundle_build.py

PURPOSE
-------
Smoke test for CurveBundle construction (Option A: from parquet curve history).

Validates:
- Dense curve DataFrame is present and well-formed
- ZeroCurve object is created (API not asserted here)

DEPENDENCIES (EXPLICIT)
-----------------------
- muni_core.curves.build_curve_bundle.build_curve_bundle

DESIGN RULES
------------
- Deterministic asof date (fixture) to keep tests stable
- Do not assert ZeroCurve internals here (keep smoke-level)
"""

from datetime import date
import pandas as pd

from muni_core.curves.build_curve_bundle import build_curve_bundle


def test_curve_bundle_builds_dense_and_zero_curve(history_df: pd.DataFrame, asof_date: date):
    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key="AAA_MUNI_SPOT",
        asof=asof_date,
        step_years=0.5,
    )

    df = bundle.dense_df
    assert "tenor_yrs" in df.columns
    assert "rate_dec" in df.columns
    assert df["rate_dec"].notna().all()
    assert (df["tenor_yrs"].values[:-1] <= df["tenor_yrs"].values[1:]).all()

    # Smoke: confirm curve object exists
    assert bundle.zero_curve is not None

    assert pd.api.types.is_numeric_dtype(df["tenor_yrs"])
    assert pd.api.types.is_numeric_dtype(df["rate_dec"])

    print("\n[CurveBundle] head:")
    print(df[["tenor_yrs", "rate_dec"]].head(5).to_string(index=False))
out_path = Path("data/3-OUTPUT/curve_bundle_dense_df.xlsx")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(out_path, index=False)
print(f"Wrote: {out_path.resolve()}")
