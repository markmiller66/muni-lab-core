from __future__ import annotations

from datetime import date
from typing import List, Tuple, Optional

from muni_core.model import Bond
from .types import ZeroCurve


def year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction.

    We keep this here to avoid importing from muni_core.npv
    (which would create a circular dependency).
    """
    days = (end - start).days
    return days / 365.25


def forward_rate_between_dates(
    settle_date: date,
    start_date: date,
    end_date: date,
    curve: ZeroCurve,
) -> float:
    """
    Implied continuously compounded forward rate between start_date and end_date,
    using settlement date as time zero.
    """
    t1 = year_fraction(settle_date, start_date)
    t2 = year_fraction(settle_date, end_date)
    return curve.forward_rate(t1, t2)


def forward_rate_to_date(
    settle_date: date,
    target_date: date,
    curve: ZeroCurve,
    window_years: float = 1.0,
) -> float:
    """
    Approximate forward rate "ending at" target_date over a backward-looking window.

    Example:
        window_years = 1.0
        -> forward rate between (target - 1Y, target).
    """
    t2 = year_fraction(settle_date, target_date)
    t1 = max(0.0, t2 - float(window_years))
    return curve.forward_rate(t1, t2)


def forward_curve_grid(
    settle_date: date,
    tenors_years: List[float],
    curve: ZeroCurve,
    window_years: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Build a simple forward curve over a set of tenors, using a fixed window_years.

    Returns:
        list of (T, F) where:
          - T is the tenor in years
          - F is the forward rate over [T - window_years, T]
    """
    out: List[Tuple[float, float]] = []
    for T in tenors_years:
        t2_days = int(round(T * 365.25))
        target_date = date.fromordinal(settle_date.toordinal() + t2_days)
        f = forward_rate_to_date(settle_date, target_date, curve, window_years=window_years)
        out.append((T, f))
    return out


# -------------------------------
# Bond-aware forward diagnostics
# -------------------------------


def forward_at_call(
    bond: Bond,
    curve: ZeroCurve,
    window_years: float = 1.0,
) -> Optional[float]:
    """
    Forward rate "ending at" the call date, viewed from the issuer's perspective.

    Returns None if the bond has no call_feature or call_date.
    """
    if bond.settle_date is None or not bond.has_call() or bond.call_feature.call_date is None:
        return None

    settle = bond.settle_date
    call_date = bond.call_feature.call_date
    return forward_rate_to_date(settle, call_date, curve, window_years=window_years)


def forward_after_call(
    bond: Bond,
    curve: ZeroCurve,
    window_years: float = 1.0,
    offset_years: float = 1.0,
) -> Optional[float]:
    """
    Forward rate "ending after" the call date by offset_years.

    Example:
        window_years = 1.0, offset_years = 1.0
        -> forward rate over [call+0Y, call+1Y] approx.

    Returns None if no call_feature / call_date.
    """
    if bond.settle_date is None or not bond.has_call() or bond.call_feature.call_date is None:
        return None

    settle = bond.settle_date
    call_date = bond.call_feature.call_date

    # approximate target date = call_date + offset_years
    offset_days = int(round(offset_years * 365.25))
    target_after = date.fromordinal(call_date.toordinal() + offset_days)

    return forward_rate_to_date(settle, target_after, curve, window_years=window_years)


def forward_slope_around_call_bp(
    bond: Bond,
    curve: ZeroCurve,
    window_years: float = 1.0,
    offset_years: float = 1.0,
) -> Optional[float]:
    """
    Simple slope measure around the call date in basis points:

        slope_bp = (F_after - F_at_call) * 10_000

    where F_at_call is the forward ending at call_date,
    and F_after is the forward ending at (call_date + offset_years).

    Returns None if forwards cannot be computed.
    """
    f_call = forward_at_call(bond, curve, window_years=window_years)
    f_after = forward_after_call(bond, curve, window_years=window_years, offset_years=offset_years)

    if f_call is None or f_after is None:
        return None

    return (f_after - f_call) * 10_000.0
