from __future__ import annotations

from datetime import date, timedelta
from typing import List, Tuple

from muni_core.model import Bond


Cashflow = Tuple[float, float]  # (t_years, amount)


def _year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction.
    Good enough for NPV engine v1.
    """
    days = (end - start).days
    return days / 365.25


def _generate_schedule(
    start: date,
    end: date,
    freq_per_year: int = 2,
) -> List[date]:
    """
    Generate coupon payment dates from 'start' to 'end' (inclusive),
    assuming fixed interval in time by year fraction. This is a v1 simplification:
    we march forward in equal year steps rather than exact calendar logic.

    For example:
    - freq_per_year = 2 -> step = 0.5 years
    - freq_per_year = 1 -> step = 1.0 years
    """
    if end <= start:
        return []

    step_years = 1.0 / float(freq_per_year)
    step_days = int(round(step_years * 365.25))

    dates: List[date] = []
    current = start + timedelta(days=step_days)
    while current < end:
        dates.append(current)
        current = current + timedelta(days=step_days)

    dates.append(end)  # include final maturity/call date
    return dates


def cashflows_to_maturity(
    bond: Bond,
    freq_per_year: int = 2,
) -> List[Cashflow]:
    """
    Generate cashflows (t_years, amount) from settlement to maturity.

    - Coupons are based on coupon % and quantity.
    - Principal is repaid at maturity.
    """
    if bond.settle_date is None or bond.maturity_date is None:
        raise ValueError("Bond must have settle_date and maturity_date set for cashflow generation.")

    coupon_rate_decimal = bond.coupon / 100.0
    coupon_per_year = coupon_rate_decimal * bond.quantity
    coupon_per_period = coupon_per_year / float(freq_per_year)

    schedule = _generate_schedule(bond.settle_date, bond.maturity_date, freq_per_year)

    cashflows: List[Cashflow] = []
    for dt in schedule:
        t = _year_fraction(bond.settle_date, dt)
        amount = coupon_per_period
        if dt == bond.maturity_date:
            amount += bond.quantity  # principal
        cashflows.append((t, amount))

    return cashflows


def cashflows_to_call(
    bond: Bond,
    freq_per_year: int = 2,
) -> List[Cashflow]:
    """
    Generate cashflows (t_years, amount) from settlement to CALL DATE
    using the bond's call_feature (if present).

    If no call_feature exists, raises ValueError.
    """
    if bond.settle_date is None or bond.call_feature is None or bond.call_feature.call_date is None:
        raise ValueError("Bond must have settle_date and a valid call_feature.call_date for call cashflows.")

    call_date = bond.call_feature.call_date

    coupon_rate_decimal = bond.coupon / 100.0
    coupon_per_year = coupon_rate_decimal * bond.quantity
    coupon_per_period = coupon_per_year / float(freq_per_year)

    schedule = _generate_schedule(bond.settle_date, call_date, freq_per_year)

    cashflows: List[Cashflow] = []
    for dt in schedule:
        t = _year_fraction(bond.settle_date, dt)
        amount = coupon_per_period
        if dt == call_date:
            # Principal at call date, using call_price % of par
            call_price_fraction = bond.call_feature.call_price / 100.0
            amount += bond.quantity * call_price_fraction
        cashflows.append((t, amount))

    return cashflows
