# src/muni_core/oas/simple_oas.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, List
import math

from muni_core.model import Bond
from muni_core.curves.types import ZeroCurve


# ---------------------------------------------------------------------------
# Basic utilities (duplicated minimally here to avoid circular deps)
# ---------------------------------------------------------------------------

def _year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction, consistent with the rest of the project.
    """
    days = (end - start).days
    return days / 365.25


def _build_coupon_times(
    settle_date: date,
    end_date: date,
    freq: int = 2,
) -> List[float]:
    """
    Build coupon times (in years) from settle_date to end_date.

    Example: freq = 2 -> semiannual coupons at ~0.5, 1.0, 1.5, ...
    """
    t_end = _year_fraction(settle_date, end_date)
    if t_end <= 0:
        return []

    step = 1.0 / float(freq)
    t = step
    times: List[float] = []
    eps = 1e-8

    while t <= t_end + eps:
        times.append(t)
        t += step

    return times


# ---------------------------------------------------------------------------
# Public OAS result container
# ---------------------------------------------------------------------------

@dataclass
class OASResult:
    """
    Simple container for an OAS solve.

    All prices are expressed per 100 notional (investor convention).
    """
    oas_bp: float          # solved option-adjusted spread, in basis points
    model_price: float     # model price at that OAS (per 100)
    clean_price: float     # target clean price (per 100)
    residual: float        # model_price - clean_price
    iterations: int        # number of iterations used
    converged: bool        # whether we hit tolerance within max_iter


# ---------------------------------------------------------------------------
# Core: price a *non-callable* bond at a given OAS
# ---------------------------------------------------------------------------

def price_bond_with_oas(
    bond: Bond,
    curve: ZeroCurve,
    oas_bp: float,
    *,
    coupon_freq: int = 2,
    today: Optional[date] = None,
) -> float:
    """
    Price a *non-callable* fixed-rate bond with a constant OAS added
    to the zero curve.

    This is investor-view, per 100 notional. For callable bonds we will
    layer a HW/Ho-Lee lattice on top later; this function is the
    non-call "backbone" and is also useful for Z/OAS KRDs.
    """
    if bond.maturity_date is None:
        raise ValueError("Bond.maturity_date is required for OAS pricing")

    # Choose today: prefer settle_date, fall back to maturity_date for safety
    if today is None:
        today = bond.settle_date or bond.maturity_date

    # Build coupon times to maturity
    times = _build_coupon_times(today, bond.maturity_date, freq=coupon_freq)
    if not times:
        # If there are no coupon times left, treat as an immediate redemption
        return 100.0

    notional = 100.0  # price per 100 convention
    coupon_rate = float(bond.coupon or 0.0)
    cpn_per_period = coupon_rate / float(coupon_freq) * notional

    # Convert bp -> decimal
    oas_dec = oas_bp / 10_000.0

    def df_oas(t: float) -> float:
        # Zero rate from curve + constant OAS
        r0 = curve.zero_rate(t)
        r = r0 + oas_dec
        return math.exp(-r * t)

    t_mat = _year_fraction(today, bond.maturity_date)

    pv = 0.0
    for t in times:
        pv += cpn_per_period * df_oas(t)
    pv += notional * df_oas(t_mat)

    return pv


# ---------------------------------------------------------------------------
# Solver: find OAS that matches market clean price
# ---------------------------------------------------------------------------

def solve_oas_for_price(
    bond: Bond,
    curve: ZeroCurve,
    *,
    target_price: Optional[float] = None,   # per 100
    coupon_freq: int = 2,
    today: Optional[date] = None,
    bp_low: float = -2000.0,
    bp_high: float = 2000.0,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> OASResult:
    """
    Solve for the constant OAS (in bp) such that the model price matches
    the target clean price.

    - Uses a simple bisection method over [bp_low, bp_high].
    - If f(low)*f(high) > 0, it falls back to the midpoint and reports
      converged = False, but still returns a best-effort OAS.

    This is for *non-callable* pricing; callable OAS will later plug into
    a HW/Ho-Lee lattice with the same external signature.
    """
    if target_price is None:
        if bond.clean_price is None:
            raise ValueError("Either target_price or bond.clean_price must be provided")
        target_price = float(bond.clean_price)

    # Default 'today' to settle_date if available
    if today is None:
        today = bond.settle_date or bond.maturity_date

    # Objective function: model_price(oas) - target_price
    def f(oas_bp: float) -> float:
        return price_bond_with_oas(
            bond,
            curve,
            oas_bp,
            coupon_freq=coupon_freq,
            today=today,
        ) - target_price

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

    # If there is no sign change, we can't guarantee a root; return midpoint
    if f_low * f_high > 0:
        mid_oas = 0.5 * (bp_low + bp_high)
        model_price = price_bond_with_oas(
            bond,
            curve,
            mid_oas,
            coupon_freq=coupon_freq,
            today=today,
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
