"""
FILE: src/muni_core/risk/deterministic_parallel_duration_hw.py

PURPOSE
-------
Deterministic (non-option) parallel duration using the HW/Ho-Lee lattice engine.

- Parallel +1 bp shift of the spot/zero curve
- Deterministic (non-callable) lattice pricing (allow_call=False)
- Produces DV01 per bp and modified duration

This is a primitive risk measure. No summaries here.

DEPENDENCIES (EXPLICIT)
-----------------------
- muni_core.curves.history.build_dense_zero_curve_for_date
- muni_core.config.AppConfig
- muni_core.model.Bond
- muni_core.pricing.hw_bond_pricer_override.price_callable_bond_hw_from_bond_dense_override

DESIGN RULES
------------
- No call exercise (allow_call=False)
- No OAS solving; optional z_spread_bp input may be provided (default 0)
- Return raw prices and computed DV01/mod duration for inspection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import pandas as pd

from muni_core.config import AppConfig
from muni_core.model import Bond
from muni_core.curves.build_curve_bundle import build_curve_bundle


from muni_core.pricing.hw_bond_pricer_override import (
    price_callable_bond_hw_from_bond_dense_override,
)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None



@dataclass(frozen=True)
class DeterministicParallelDurationResult:
    bump_bp: float
    z_spread_bp: float

    base_price: float
    price_up: float      # curve + bump_bp (yields up => price down)
    price_down: float    # curve - bump_bp (yields down => price up)

    dv01_bp: float
    mod_duration: float

    asof: date
    curve_key: str
    a: float
    sigma: float
    step_years: float
    q: float


def _get_asof(cfg: AppConfig, history_df: pd.DataFrame) -> date:
    if getattr(cfg.curves, "curve_asof_date", None):
        return pd.to_datetime(cfg.curves.curve_asof_date).date()
    tmp = history_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
    return tmp["date"].max()


def _get_hw_params(cfg: AppConfig) -> tuple[float, float]:
    a_raw = None
    sigma_raw = None
    if hasattr(cfg, "get_control_value"):
        try:
            a_raw = cfg.get_control_value("HW_A", default=None)
            sigma_raw = cfg.get_control_value("HW_SIGMA_BASE", default=None)
        except Exception:
            a_raw = None
            sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01
    return a, sigma

def _normalize_dense_df(dense_df: pd.DataFrame) -> pd.DataFrame:
    df = dense_df.copy()

    if "tenor_yrs" not in df.columns:
        for c in ["tenor_years", "TenorY", "tenor", "tenor_yr"]:
            if c in df.columns:
                df["tenor_yrs"] = df[c].astype(float)
                break

    # --- tenor normalization (optional, for debug friendliness) ---
    if "tenor_years" not in df.columns and "tenor_yrs" in df.columns:
        df["tenor_years"] = df["tenor_yrs"].astype(float)

    # --- create canonical zero_rate if missing ---
    if "zero_rate" not in df.columns:
        for c in ["rate_dec", "ZeroRate", "zero", "rate", "spot_rate", "spot", "zr"]:
            if c in df.columns:
                df["zero_rate"] = df[c].astype(float)
                break

    if "zero_rate" not in df.columns:
        raise KeyError(f"dense_df missing 'zero_rate'. Columns: {list(df.columns)}")

    # --- now force canonical rate_dec to exist ---
    if "rate_dec" not in df.columns:
        df["rate_dec"] = df["zero_rate"].astype(float)

    return df



def _bump_parallel(dense_df: pd.DataFrame, bump_bp: float) -> pd.DataFrame:
    out = dense_df.copy()
    bump = bump_bp / 10000.0

    if "rate_dec" in out.columns:
        out["rate_dec"] = out["rate_dec"].astype(float) + bump
    if "zero_rate" in out.columns:
        out["zero_rate"] = out["zero_rate"].astype(float) + bump
    if "ZeroRate" in out.columns:
        out["ZeroRate"] = out["ZeroRate"].astype(float) + bump

    # hard guard
    if "rate_dec" not in out.columns and "zero_rate" not in out.columns and "ZeroRate" not in out.columns:
        raise KeyError(f"Can't find rate column to bump. cols={list(out.columns)}")

    return out





def compute_parallel_duration_hw_det(
    *,
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    bump_bp: float = 1.0,
    z_spread_bp: float = 0.0,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    coupon_freq: int = 2,
    face: float = 100.0,
    time_tolerance: float = 1e-6,
    debug: bool = False,
) -> DeterministicParallelDurationResult:
    """
    Deterministic (non-call) parallel duration using HW lattice pricing.

    - Builds dense curve at asof
    - Prices at base, curve+bump, curve-bump
    - DV01 per bp = (P_down - P_up) / 2
    - ModDuration = DV01_bp * 10000 / P_base
    """
    asof = _get_asof(app_cfg, history_df)
    a, sigma = _get_hw_params(app_cfg)

    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )
    dense_df = bundle.dense_df

    dense_up = _bump_parallel(dense_df, +abs(bump_bp))
    dense_dn = _bump_parallel(dense_df, -abs(bump_bp))
    if debug:
        r0 = float(dense_df["rate_dec"].iloc[0])
        rup = float(dense_up["rate_dec"].iloc[0])
        rdn = float(dense_dn["rate_dec"].iloc[0])
        if not (rup > r0 > rdn):
            raise RuntimeError("Parallel bump failed: rate_dec did not shift as expected.")
    if debug:
        for name, df in [("base", dense_df), ("up", dense_up), ("dn", dense_dn)]:
            ten_col = _pick_col(df, ["tenor_yrs", "tenor_years", "TenorY", "tenor"])
            rate_col = _pick_col(df, ["rate_dec", "zero_rate", "ZeroRate", "spot_rate", "rate"])
            if ten_col and rate_col:
                print(name, df[[ten_col, rate_col]].head(3).to_string(index=False))
            else:
                print(name, "cols=", list(df.columns))

    # Base price (deterministic = allow_call=False)
    p0 = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_df,
        a=a,
        sigma=sigma,
        oas_bp=float(z_spread_bp),     # used as a plain spread input; OAS is not solved here
        freq_per_year=coupon_freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )

    pup = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_up,
        a=a,
        sigma=sigma,
        oas_bp=float(z_spread_bp),
        freq_per_year=coupon_freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )

    pdn = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_dn,
        a=a,
        sigma=sigma,
        oas_bp=float(z_spread_bp),
        freq_per_year=coupon_freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )



    p0 = float(p0)
    pup = float(pup)
    pdn = float(pdn)

    dv01_bp = (pdn - pup) / 2.0
    mod_dur = float("nan")
    if p0 > 0 and math.isfinite(p0):
        mod_dur = dv01_bp * 10000.0 / p0

    return DeterministicParallelDurationResult(
        bump_bp=float(bump_bp),
        z_spread_bp=float(z_spread_bp),
        base_price=p0,
        price_up=pup,
        price_down=pdn,
        dv01_bp=float(dv01_bp),
        mod_duration=float(mod_dur),
        asof=asof,
        curve_key=str(curve_key),
        a=float(a),
        sigma=float(sigma),
        step_years=float(step_years),
        q=float(q),
    )
