# FILE: src/muni_core/risk/callable_krd_hw_triangular.py
#
# PURPOSE:
#   Proper Key-Rate Duration (KRD) and Key-Rate Convexity (KRC) for callable munis
#   using a HW lattice and *triangular* localized curve bumps.
#
# MODEL:
#   - Hold OAS fixed
#   - For each key tenor, apply a triangular bump across dense nodes between adjacent key tenors
#   - Compute KRD/KRC from symmetric differences
#
# IMPORTS FROM:
#   - src/muni_core/curves/history.py : build_dense_zero_curve_for_date
#   - src/muni_core/pricing/hw_bond_pricer_override.py : price_callable_bond_hw_from_bond_dense_override
#   - src/muni_core/risk/callable_parallel_curve_hw.py : compute_callable_parallel_curve_hw  (optional sanity)

# FILE: src/muni_core/risk/callable_krd_hw_triangular.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from muni_core.model import Bond
from muni_core.config.loader import AppConfig
from muni_core.curves.build_curve_bundle import build_curve_bundle
from muni_core.pricing.hw_bond_pricer_override import price_callable_bond_hw_from_bond_dense_override


@dataclass
class CallableKRDResult:
    base_price: float
    bump_bp: float
    key_tenors: List[float]
    krd: Dict[float, float]          # key tenor -> KRD
    krc: Dict[float, float]          # key tenor -> KRC
    price_up: Dict[float, float]     # key tenor -> price_up
    price_down: Dict[float, float]   # key tenor -> price_down

    # Optional sanity vs parallel bump (filled if requested)
    curve_dv01_bp: float | None = None
    curve_mod_duration: float | None = None
    curve_price_up: float | None = None
    curve_price_down: float | None = None


def _tenor_col(df: pd.DataFrame) -> str:
    if "tenor_years" in df.columns:
        return "tenor_years"
    if "tenor_yrs" in df.columns:
        return "tenor_yrs"
    raise KeyError(f"Expected tenor_years or tenor_yrs; got columns={list(df.columns)}")


def _rate_col(df: pd.DataFrame) -> str:
    if "rate_dec" in df.columns:
        return "rate_dec"
    if "zero_rate" in df.columns:
        return "zero_rate"
    raise KeyError(f"Expected rate_dec or zero_rate; got columns={list(df.columns)}")


def _apply_triangular_bump(
    dense_df: pd.DataFrame,
    *,
    key_tenors: List[float],
    key_idx: int,
    bump_bp: float,
) -> pd.DataFrame:
    out = dense_df.copy()
    tc = _tenor_col(out)
    rc = _rate_col(out)

    ks = [float(x) for x in key_tenors]
    k = float(ks[key_idx])

    left = ks[key_idx - 1] if key_idx > 0 else None
    right = ks[key_idx + 1] if key_idx < (len(ks) - 1) else None

    t = out[tc].astype(float).values
    weights = [0.0] * len(t)

    for i, ti in enumerate(t):
        w = 0.0
        if left is None and right is not None:
            if k <= ti <= right:
                w = (right - ti) / (right - k) if right != k else 0.0
        elif right is None and left is not None:
            if left <= ti <= k:
                w = (ti - left) / (k - left) if k != left else 0.0
        else:
            if left <= ti <= k:
                w = (ti - left) / (k - left) if k != left else 0.0
            elif k < ti <= right:
                w = (right - ti) / (right - k) if right != k else 0.0

        weights[i] = max(0.0, min(1.0, w))

    bump_dec = float(bump_bp) / 10000.0
    out[rc] = out[rc].astype(float) + pd.Series(weights, index=out.index) * bump_dec
    return out


def compute_callable_krd_hw(
    *,
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    base_oas_bp: float,
    key_tenors: Optional[List[float]] = None,
    bump_bp: float = 1.0,
    freq_per_year: int = 2,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
    include_parallel_sanity: bool = True,
) -> CallableKRDResult:
    if key_tenors is None:
        key_tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    key_tenors = [float(x) for x in key_tenors]

    curves_cfg = app_cfg.curves
    if not curves_cfg.curve_asof_date:
        raise ValueError("app_cfg.curves.curve_asof_date must be set for callable KRD.")
    asof: date = pd.to_datetime(curves_cfg.curve_asof_date).date()

    a = float(getattr(app_cfg, "get_control_value", lambda *_a, **_k: 0.10)("HW_A", default=0.10))
    sigma = float(getattr(app_cfg, "get_control_value", lambda *_a, **_k: 0.015)("HW_SIGMA_BASE", default=0.015))

    # --- NEW: build BOTH (dense + curve object), use dense_df for KRD bumps ---
    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )
    dense_base = bundle.dense_df

    p0 = float(
        price_callable_bond_hw_from_bond_dense_override(
            bond=bond,
            asof=asof,
            dense_df=dense_base,
            a=a,
            sigma=sigma,
            oas_bp=base_oas_bp,
            freq_per_year=freq_per_year,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )
    if p0 <= 0.0:
        raise ValueError(f"Base price p0 must be > 0; got p0={p0}")

    bump_dec = float(bump_bp) / 10000.0

    price_up: Dict[float, float] = {}
    price_down: Dict[float, float] = {}
    krd: Dict[float, float] = {}
    krc: Dict[float, float] = {}

    for j, k in enumerate(key_tenors):
        df_up = _apply_triangular_bump(dense_base, key_tenors=key_tenors, key_idx=j, bump_bp=+bump_bp)
        df_dn = _apply_triangular_bump(dense_base, key_tenors=key_tenors, key_idx=j, bump_bp=-bump_bp)

        pu = float(
            price_callable_bond_hw_from_bond_dense_override(
                bond=bond,
                asof=asof,
                dense_df=df_up,
                a=a,
                sigma=sigma,
                oas_bp=base_oas_bp,
                freq_per_year=freq_per_year,
                step_years=step_years,
                q=q,
                time_tolerance=time_tolerance,
            )
        )
        pdn = float(
            price_callable_bond_hw_from_bond_dense_override(
                bond=bond,
                asof=asof,
                dense_df=df_dn,
                a=a,
                sigma=sigma,
                oas_bp=base_oas_bp,
                freq_per_year=freq_per_year,
                step_years=step_years,
                q=q,
                time_tolerance=time_tolerance,
            )
        )

        price_up[k] = pu
        price_down[k] = pdn
        krd[k] = float(-(pu - pdn) / (2.0 * p0 * bump_dec))
        krc[k] = float((pu + pdn - 2.0 * p0) / (p0 * bump_dec * bump_dec))

    res = CallableKRDResult(
        base_price=p0,
        bump_bp=float(bump_bp),
        key_tenors=key_tenors,
        krd=krd,
        krc=krc,
        price_up=price_up,
        price_down=price_down,
    )

    if include_parallel_sanity:
        from muni_core.risk.callable_parallel_curve_hw import compute_callable_parallel_curve_hw

        par = compute_callable_parallel_curve_hw(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            base_oas_bp=base_oas_bp,
            bump_bp=bump_bp,
            freq_per_year=freq_per_year,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
        res.curve_dv01_bp = float(par.dv01_bp)
        res.curve_mod_duration = float(par.mod_duration)
        res.curve_price_up = float(par.price_up)
        res.curve_price_down = float(par.price_down)

    return res
