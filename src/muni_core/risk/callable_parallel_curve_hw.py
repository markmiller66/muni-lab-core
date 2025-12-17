# FILE: src/muni_core/risk/callable_parallel_curve_hw.py
#
# PURPOSE:
#   Parallel-curve DV01 / duration for callable munis using HW lattice.
#   Holds OAS fixed; bumps ALL curve nodes +1bp / -1bp.
#
# IMPORTS FROM:
#   - src/muni_core/curves/history.py : build_dense_zero_curve_for_date
#   - src/muni_core/pricing/hw_bond_pricer_override.py : price_callable_bond_hw_from_bond_dense_override

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd

from muni_core.model import Bond
from muni_core.config.loader import AppConfig
from muni_core.curves.history import build_dense_zero_curve_for_date
from muni_core.pricing.hw_bond_pricer_override import price_callable_bond_hw_from_bond_dense_override


@dataclass
class CallableParallelCurveResult:
    base_price: float
    bump_bp: float
    price_up: float
    price_down: float
    dv01_bp: float
    mod_duration: float


def _rate_col(df: pd.DataFrame) -> str:
    if "rate_dec" in df.columns:
        return "rate_dec"
    if "zero_rate" in df.columns:
        return "zero_rate"
    raise KeyError(f"Expected rate_dec or zero_rate; got columns={list(df.columns)}")


def _bump_dense_parallel(dense_df: pd.DataFrame, bump_bp: float) -> pd.DataFrame:
    out = dense_df.copy()
    rc = _rate_col(out)
    out[rc] = out[rc].astype(float) + (float(bump_bp) / 10000.0)
    return out


def compute_callable_parallel_curve_hw(
    *,
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    base_oas_bp: float,
    bump_bp: float = 1.0,
    freq_per_year: int = 2,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> CallableParallelCurveResult:
    curves_cfg = app_cfg.curves
    if not curves_cfg.curve_asof_date:
        raise ValueError("app_cfg.curves.curve_asof_date must be set.")
    asof: date = pd.to_datetime(curves_cfg.curve_asof_date).date()

    a = float(getattr(app_cfg, "get_control_value", lambda *_a, **_k: 0.10)("HW_A", default=0.10))
    sigma = float(getattr(app_cfg, "get_control_value", lambda *_a, **_k: 0.015)("HW_SIGMA_BASE", default=0.015))

    dense_base = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    p0 = float(price_callable_bond_hw_from_bond_dense_override(
        bond=bond, asof=asof, dense_df=dense_base, a=a, sigma=sigma, oas_bp=base_oas_bp,
        freq_per_year=freq_per_year, step_years=step_years, q=q, time_tolerance=time_tolerance
    ))
    if p0 <= 0:
        raise ValueError(f"Base price must be > 0; got {p0}")

    up_df = _bump_dense_parallel(dense_base, +bump_bp)
    dn_df = _bump_dense_parallel(dense_base, -bump_bp)

    p_up = float(price_callable_bond_hw_from_bond_dense_override(
        bond=bond, asof=asof, dense_df=up_df, a=a, sigma=sigma, oas_bp=base_oas_bp,
        freq_per_year=freq_per_year, step_years=step_years, q=q, time_tolerance=time_tolerance
    ))
    p_dn = float(price_callable_bond_hw_from_bond_dense_override(
        bond=bond, asof=asof, dense_df=dn_df, a=a, sigma=sigma, oas_bp=base_oas_bp,
        freq_per_year=freq_per_year, step_years=step_years, q=q, time_tolerance=time_tolerance
    ))

    bump_dec = float(bump_bp) / 10000.0
    dv01_bp = float((p_dn - p_up) / 2.0)              # per 1bp symmetric if bump_bp=1
    mod_dur = float(dv01_bp / (p0 * bump_dec))        # duration = -dP/P / dY; symmetric approx

    return CallableParallelCurveResult(
        base_price=p0,
        bump_bp=float(bump_bp),
        price_up=p_up,
        price_down=p_dn,
        dv01_bp=dv01_bp,
        mod_duration=mod_dur,
    )
