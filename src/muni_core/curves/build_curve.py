from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import pandas as pd


@dataclass(frozen=True)
class BuiltDenseCurve:
    asof: date
    curve_key: str
    df: pd.DataFrame  # must include tenor_yrs, rate_dec (decimal)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_dense_curve_df(dense_df: pd.DataFrame) -> pd.DataFrame:
    """
    Force a canonical shape for dense curves:
      - tenor_yrs (float)
      - rate_dec  (float, decimal)
    Keeps existing extra cols.
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

    # basic guards
    if df["tenor_yrs"].isna().all():
        raise KeyError(f"Could not identify tenor column. cols={list(df.columns)}")
    if df["rate_dec"].isna().all():
        raise KeyError(f"Could not identify rate column. cols={list(df.columns)}")

    # sort + drop unusable rows
    df = df.dropna(subset=["tenor_yrs", "rate_dec"]).sort_values("tenor_yrs").reset_index(drop=True)

    return df


def build_dense_curve_from_history(
    *,
    history_df: pd.DataFrame,
    curve_key: str,
    asof: date,
    step_years: float = 0.5,
    # you already have this function in your codebase:
    build_dense_zero_curve_for_date,
) -> BuiltDenseCurve:
    """
    Adapter wrapper around your existing build_dense_zero_curve_for_date
    that guarantees canonical cols.
    """
    dense_raw = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )
    dense_df = normalize_dense_curve_df(dense_raw)
    return BuiltDenseCurve(asof=asof, curve_key=curve_key, df=dense_df)
