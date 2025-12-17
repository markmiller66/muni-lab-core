from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from muni_core.model import Bond
from muni_core.config import AppConfig
from muni_core.pricing.hw_bond_pricer import price_callable_bond_hw_from_bond
from muni_core.oas.callable_oas_hw import solve_callable_oas_hw


@dataclass
class OASDV01Result:
    base_oas_bp: float
    bump_bp: float
    price_base: float
    price_up: float
    price_down: float
    dv01_bp: float          # price change per 1bp OAS (positive number typically)
    mod_duration: float     # approx modified duration in years


def compute_callable_oas_dv01_hw(
    *,
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    base_oas_bp: Optional[float] = None,
    bump_bp: float = 1.0,
    coupon_freq: int = 2,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> OASDV01Result:
    """
    Callable OAS DV01 using central differences:
      DV01 ≈ (P(oas-bump) - P(oas+bump)) / 2

    Returns dv01_bp as the price change per 1bp OAS (positive for normal bonds).
    """
    if base_oas_bp is None:
        sol = solve_callable_oas_hw(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            target_price=bond.clean_price,
            coupon_freq=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
        base_oas_bp = float(sol.oas_bp)

    o0 = float(base_oas_bp)
    b = float(bump_bp)

    p0 = float(
        price_callable_bond_hw_from_bond(
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=o0,
            bond=bond,
            freq_per_year=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )
    p_up = float(
        price_callable_bond_hw_from_bond(
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=o0 + b,
            bond=bond,
            freq_per_year=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )
    p_dn = float(
        price_callable_bond_hw_from_bond(
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=o0 - b,
            bond=bond,
            freq_per_year=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )

    # central difference (per 1bp)
    dv01 = (p_dn - p_up) / (2.0 * b) * 1.0  # already “per bp” since bump is in bp
    mod_dur = dv01 / p0 * 10_000.0  # because dv01 is per bp; duration ≈ (1/P)*dP/dy where dy=0.0001

    return OASDV01Result(
        base_oas_bp=o0,
        bump_bp=b,
        price_base=p0,
        price_up=p_up,
        price_down=p_dn,
        dv01_bp=float(dv01),
        mod_duration=float(mod_dur),
    )
