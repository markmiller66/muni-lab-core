# src/muni_core/oas/callable_oas_hw.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Any

import pandas as pd

from muni_core.model import Bond
from muni_core.config.loader import AppConfig
from muni_core.pricing.hw_bond_pricer import price_callable_bond_hw_from_bond
from muni_core.oas.simple_oas import OASResult


def solve_callable_oas_hw(
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,

    target_price: Optional[float] = None,   # per 100
    coupon_freq: int = 2,
    today: Optional[date] = None,           # kept for signature symmetry; not used yet

    bp_low: float = -2000.0,
    bp_high: float = 2000.0,
    tol: float = 1e-6,
    max_iter: int = 50,

    # Extra knobs passed through to price_callable_bond_hw_from_bond
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> OASResult:
    """
    Solve for the constant callable OAS (in bp) such that the HW-lattice
    callable price matches the target clean price.

    This mirrors solve_oas_for_price in simple_oas.py, but:

      - Pricing is via Hull–White lattice + Bermudan call logic.
      - Non-call value is implicitly "with no call dates" if you want,
        but this solver assumes the *issuer* has the call rights
        embedded in the bond.

    Inputs:
      bond         : muni_core.model.Bond (must have maturity_date, coupon, clean_price)
      history_df   : curves history table (AAA_MUNI_SPOT etc.)
      app_cfg      : AppConfig with curve + controls

      target_price : per-100 clean price (if None, uses bond.clean_price)
      coupon_freq  : coupon frequency (2 = semiannual, 1 = annual)
      today        : currently unused here (as-of comes from app_cfg.curves.curve_asof_date)

      bp_low       : lower bound for OAS search (bp)
      bp_high      : upper bound for OAS search (bp)
      tol          : tolerance on price residual
      max_iter     : max bisection iterations

      curve_key    : which curve to use for dense zero -> HW lattice
      step_years   : HW lattice time step
      q            : up-move probability in HW binomial lattice
      time_tolerance: tolerance for maturity / horizon alignment

    Returns:
      OASResult: (oas_bp, model_price, clean_price, residual, iterations, converged)
    """

    # --- Normalize target_price --------------------------------------
    if target_price is None:
        if bond.clean_price is None:
            raise ValueError(
                "solve_callable_oas_hw: target_price is None and bond.clean_price is None; "
                "cannot solve callable OAS."
            )
        target_price = float(bond.clean_price)
    else:
        target_price = float(target_price)

    if bond.maturity_date is None:
        raise ValueError("Bond.maturity_date is required for callable OAS pricing")

    # We keep 'today' in the signature for symmetry with solve_oas_for_price,
    # but the HW lattice pricer uses curves_cfg.curve_asof_date / max(history_df.date)
    # internally via price_callable_bond_hw_from_bond, so 'today' is not used here.

    def f(oas_bp: float) -> float:
        """
        Objective function: callable_model_price(oas_bp) - target_price.
        """
        price = price_callable_bond_hw_from_bond(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=oas_bp,
            freq_per_year=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
        if price is None or target_price is None:
            raise TypeError(
                f"Internal callable OAS f(): price={price!r}, "
                f"target_price={target_price!r}"
            )
        return price - target_price

    # --- Evaluate at endpoints ---------------------------------------
    f_low = f(bp_low)
    f_high = f(bp_high)

    # Exact hits at boundaries
    if f_low == 0.0:
        return OASResult(
            oas_bp=bp_low,
            model_price=target_price,
            clean_price=target_price,
            residual=0.0,
            iterations=0,
            converged=True,
        )
    if f_high == 0.0:
        return OASResult(
            oas_bp=bp_high,
            model_price=target_price,
            clean_price=target_price,
            residual=0.0,
            iterations=0,
            converged=True,
        )

    # No sign change: fall back to midpoint, report converged = False
    if f_low * f_high > 0:
        mid_oas = 0.5 * (bp_low + bp_high)
        model_price = price_callable_bond_hw_from_bond(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=mid_oas,
            freq_per_year=coupon_freq,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
        residual = model_price - target_price
        return OASResult(
            oas_bp=mid_oas,
            model_price=model_price,
            clean_price=target_price,
            residual=residual,
            iterations=0,
            converged=False,
        )

    # --- Bisection loop ----------------------------------------------
    low, high = bp_low, bp_high
    it = 0
    for it in range(1, max_iter + 1):
        mid = 0.5 * (low + high)
        f_mid = f(mid)

        if abs(f_mid) < tol:
            model_price = target_price + f_mid
            return OASResult(
                oas_bp=mid,
                model_price=model_price,
                clean_price=target_price,
                residual=f_mid,
                iterations=it,
                converged=True,
            )

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    # Max iterations reached; return best midpoint
    mid = 0.5 * (low + high)
    f_mid = f(mid)
    model_price = target_price + f_mid
    return OASResult(
        oas_bp=mid,
        model_price=model_price,
        clean_price=target_price,
        residual=f_mid,
        iterations=it,
        converged=False,
    )
