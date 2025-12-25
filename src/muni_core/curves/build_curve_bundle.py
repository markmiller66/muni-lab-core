from __future__ import annotations

"""
FILE: src/muni_core/curves/build_curve_bundle.py

PURPOSE
-------
Build a CurveBundle containing BOTH:
  - dense_df: canonical debug curve grid (tenor_yrs, rate_dec)
  - zero_curve: production ZeroCurve object (interpolation/discounting)

This module is intentionally production-only (no tests, no printing).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CurveBundle:
    """
    A "both" object:
      - dense_df: canonical debug grid (tenor_yrs, rate_dec)
      - zero_curve: production curve object (interpolation/discounting)
    """
    asof: date
    curve_key: str
    dense_df: pd.DataFrame
    zero_curve: Any  # loose to avoid coupling


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_dense_curve_df(dense_df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonicalize curve df to:
      - tenor_yrs: float
      - rate_dec: float (decimal)

    Keeps extras, sorts by tenor, drops NaNs.
    """
    df = dense_df.copy()

    # tenor_yrs
    if "tenor_yrs" not in df.columns:
        ten_col = _pick_col(df, ["tenor_years", "TenorY", "tenor", "tenor_yr", "tenorY"])
        if ten_col is not None:
            df["tenor_yrs"] = pd.to_numeric(df[ten_col], errors="coerce")
    df["tenor_yrs"] = pd.to_numeric(df.get("tenor_yrs"), errors="coerce")

    # rate_dec
    if "rate_dec" not in df.columns:
        rate_col = _pick_col(df, ["zero_rate", "ZeroRate", "spot_rate", "rate", "zr", "spot"])
        if rate_col is not None:
            df["rate_dec"] = pd.to_numeric(df[rate_col], errors="coerce")
    df["rate_dec"] = pd.to_numeric(df.get("rate_dec"), errors="coerce")

    if df["tenor_yrs"].isna().all():
        raise KeyError(f"Could not identify tenor column. cols={list(df.columns)}")
    if df["rate_dec"].isna().all():
        raise KeyError(f"Could not identify rate column. cols={list(df.columns)}")

    df = (
        df.dropna(subset=["tenor_yrs", "rate_dec"])
          .sort_values("tenor_yrs")
          .reset_index(drop=True)
    )

    if not pd.api.types.is_numeric_dtype(df["tenor_yrs"]):
        raise TypeError("tenor_yrs must be numeric")
    if not pd.api.types.is_numeric_dtype(df["rate_dec"]):
        raise TypeError("rate_dec must be numeric")

    if (df["tenor_yrs"].values[:-1] > df["tenor_yrs"].values[1:]).any():
        raise ValueError("tenor_yrs must be non-decreasing after sort (unexpected).")

    return df


from typing import Any
...
def dense_df_to_zero_curve(dense_df: pd.DataFrame) -> Any:

    """
    Convert canonical dense_df -> project's ZeroCurve.
    """
    # Import lazily so this module doesn’t create dependency loops.
    from muni_core.curves import make_zero_curve_from_pairs

    pairs = list(
        zip(
            dense_df["tenor_yrs"].astype(float).tolist(),
            dense_df["rate_dec"].astype(float).tolist(),
        )
    )
    return make_zero_curve_from_pairs(pairs)


def build_curve_bundle(
    *,
    history_df: pd.DataFrame,
    curve_key: str,
    asof: date,
    step_years: float,
) -> CurveBundle:
    """
    Uses your existing history builder, then returns BOTH:
      - dense_df (canonical)
      - zero_curve (from pairs)
    """
    from muni_core.curves.history import build_dense_zero_curve_for_date

    dense_raw = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )
    dense_df = normalize_dense_curve_df(dense_raw)
    zc = dense_df_to_zero_curve(dense_df)

    return CurveBundle(
        asof=asof,
        curve_key=curve_key,
        dense_df=dense_df,
        zero_curve=zc,
    )
