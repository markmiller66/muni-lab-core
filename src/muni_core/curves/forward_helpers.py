from __future__ import annotations

from datetime import date
from typing import List, Tuple

from .types import ZeroCurve


def year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction.

    We duplicate this here instead of importing npv._year_fraction
    to avoid a circular dependency (curves <-> npv).
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

    Converts dates to year fractions and calls ZeroCurve.forward_rate(t1, t2).
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
    Approximate forward rate "ending at" target_date with a backward-looking window.

    Example:
        window_years = 1.0
        -> forward rate between (target - 1Y, target).

    This is often close to how people think about "forward 1Y rate at T".
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
        target_date = settle_date.fromordinal(settle_date.toordinal() + t2_days)
        f = forward_rate_to_date(settle_date, target_date, curve, window_years=window_years)
        out.append((T, f))
    return out
