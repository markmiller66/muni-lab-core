# src/muni_core/npv/__init__.py

from __future__ import annotations

from datetime import date
from typing import List, Optional

import math

from ..model import Bond
from ..curves.types import ZeroCurve
from .issuer import IssuerCallResult, compute_issuer_call_npv

__all__ = [
    # legacy / investor-view helpers used by calls.npv_call_test
    "pv_to_maturity",
    "pv_to_call",
    # new issuer-view engine (Path 3)
    "IssuerCallResult",
    "compute_issuer_call_npv",
]


# ---- shared helpers --------------------------------------------------------


def _year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction.
    """
    days = (end - start).days
    return days / 365.25


def _build_coupon_times(
    settle_date: date,
    end_date: date,
    freq: int = 2,
) -> List[float]:
    """
    Build coupon times from settle_date to end_date in year fractions.

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


def _pv_simple(
    bond: Bond,
    curve: ZeroCurve,
    *,
    end_date: date,
    coupon_freq: int = 2,
    today: Optional[date] = None,
) -> float:
    """
    Simple investor-view PV of a fixed coupon bond from today to end_date
    (either call date or maturity), using the provided ZeroCurve.

    This is a generic engine used by pv_to_maturity and pv_to_call.
    """
    if today is None:
        if bond.settle_date is not None:
            today = bond.settle_date
        else:
            # very defensive fallback
            today = end_date

    # If end_date is already in the past, treat all value as occurring at t=0
    if end_date <= today:
        return float(bond.quantity or 100_000.0)

    if bond.coupon is None:
        coupon_rate = 0.0
    else:
        coupon_rate = float(bond.coupon)

    notional = float(bond.quantity) if bond.quantity is not None else 100_000.0
    times = _build_coupon_times(settle_date=today, end_date=end_date, freq=coupon_freq)
    if not times:
        return notional  # trivial case

    cpn_per_period = coupon_rate / float(coupon_freq) * notional

    def df(t: float) -> float:
        return curve.discount_factor(t)

    t_end = _year_fraction(today, end_date)

    pv = 0.0
    for t in times:
        pv += cpn_per_period * df(t)
    pv += notional * df(t_end)

    return pv


# ---- public legacy helpers -------------------------------------------------


def pv_to_maturity(
    bond: Bond,
    curve: ZeroCurve,
    coupon_freq: int = 2,
    freq_per_year: Optional[int] = None,
    today: Optional[date] = None,
) -> float:
    """
    Investor-view PV of the bond to its maturity date.

    This restores the legacy API used by muni_core.calls.npv_call_test:
        from muni_core.npv import pv_to_maturity, pv_to_call

    Accepts either `coupon_freq` or `freq_per_year` for compatibility.
    """
    # If caller used freq_per_year, let it override coupon_freq
    if freq_per_year is not None:
        coupon_freq = int(freq_per_year)

    if bond.maturity_date is None:
        raise ValueError("Bond.maturity_date is required for pv_to_maturity")

    return _pv_simple(
        bond,
        curve,
        end_date=bond.maturity_date,
        coupon_freq=coupon_freq,
        today=today,
    )



def pv_to_call(
    bond: Bond,
    curve: ZeroCurve,
    coupon_freq: int = 2,
    freq_per_year: Optional[int] = None,
    today: Optional[date] = None,
) -> float:
    """
    Investor-view PV of the bond to its (first) call date.

    If there is no call feature, falls back to pv_to_maturity.

    Accepts either `coupon_freq` or `freq_per_year` for compatibility.
    """
    # If caller used freq_per_year, let it override coupon_freq
    if freq_per_year is not None:
        coupon_freq = int(freq_per_year)

    if bond.call_feature is None or bond.call_feature.call_date is None:
        # No call => treat as PV to maturity
        return pv_to_maturity(
            bond,
            curve,
            coupon_freq=coupon_freq,
            today=today,
        )

    return _pv_simple(
        bond,
        curve,
        end_date=bond.call_feature.call_date,
        coupon_freq=coupon_freq,
        today=today,
    )



# ---- issuer-view engine is imported from issuer.py -------------------------
# IssuerCallResult, compute_issuer_call_npv are already imported & exported
