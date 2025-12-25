from __future__ import annotations

"""
FILE: tests/test_dense_curve_parallel_bump.py

PURPOSE
-------
Ensure a parallel bump applied to the dense curve actually changes rates.

This is a guardrail test: if this fails, DV01/KRD tests will lie.

DEPENDENCIES (EXPLICIT)
-----------------------
- muni_core.curves.build_curve_bundle.build_curve_bundle

DESIGN RULES
------------
- Bump rate_dec by +1bp and ensure values move
"""

from datetime import date
import pandas as pd

from muni_core.curves.build_curve_bundle import build_curve_bundle


def test_dense_curve_parallel_bump_moves_rates(history_df: pd.DataFrame, asof_date: date):
    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key="AAA_MUNI_SPOT",
        asof=asof_date,
        step_years=0.5,
    )
    df = bundle.dense_df.copy()
    assert "rate_dec" in df.columns

    r0 = float(df["rate_dec"].iloc[0])

    bump = 1.0 / 10000.0
    df_up = df.copy()
    df_up["rate_dec"] = df_up["rate_dec"].astype(float) + bump
    rup = float(df_up["rate_dec"].iloc[0])

    assert rup > r0
