from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
import pandas as pd

from muni_core.config import AppConfig
from muni_core.curves.history import build_dense_zero_curve_for_date
from muni_core.pricing.hw_bond_pricer_override import (
    build_hw_theta_from_dense,
    build_binomial_lattice_from_hw,
    build_state_price_tree_from_lattice,
)

@dataclass(frozen=True)
class HWCurveBundle:
    asof: date_type
    curve_key: str
    dense_df: pd.DataFrame
    theta_df: pd.DataFrame
    lattice: object
    state_prices: object
    a: float
    sigma: float
    step_years: float
    q: float

def build_hw_curve_bundle(
    *,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
) -> HWCurveBundle:
    # --- asof ---
    curves_cfg = app_cfg.curves
    if getattr(curves_cfg, "curve_asof_date", None):
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        tmp = history_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        asof = tmp["date"].max()

    # --- HW params (Controls or defaults) ---
    a_raw = sigma_raw = None
    if hasattr(app_cfg, "get_control_value"):
        try:
            a_raw = app_cfg.get_control_value("HW_A", default=None)
            sigma_raw = app_cfg.get_control_value("HW_SIGMA_BASE", default=None)
        except Exception:
            pass
    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01

    # --- build dense curve ---
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # --- build theta + lattice + state prices ---
    theta_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)
    lattice = build_binomial_lattice_from_hw(theta_df, a=a, sigma=sigma, dt_years=step_years)
    state_prices = build_state_price_tree_from_lattice(lattice)

    return HWCurveBundle(
        asof=asof,
        curve_key=curve_key,
        dense_df=dense_df,
        theta_df=theta_df,
        lattice=lattice,
        state_prices=state_prices,
        a=a,
        sigma=sigma,
        step_years=step_years,
        q=q,
    )
