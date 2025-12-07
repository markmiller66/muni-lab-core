from __future__ import annotations

from typing import List, Tuple

from muni_core.model import Bond
from muni_core.curves import ZeroCurve
from .cashflows import cashflows_to_maturity, cashflows_to_call, Cashflow


def present_value(
    cashflows: List[Cashflow],
    curve: ZeroCurve,
) -> float:
    """
    Present value of a list of (t_years, amount) using DF from ZeroCurve.
    """
    pv = 0.0
    for t, amt in cashflows:
        df = curve.discount_factor(t)
        pv += amt * df
    return pv


def pv_to_maturity(
    bond: Bond,
    curve: ZeroCurve,
    freq_per_year: int = 2,
) -> float:
    """
    PV of bond cashflows to maturity (in currency units, not per 100).
    """
    cfs = cashflows_to_maturity(bond, freq_per_year=freq_per_year)
    return present_value(cfs, curve)


def pv_to_call(
    bond: Bond,
    curve: ZeroCurve,
    freq_per_year: int = 2,
) -> float:
    """
    PV of bond cashflows to first call date (in currency units, not per 100).

    Raises ValueError if no call_feature exists.
    """
    cfs = cashflows_to_call(bond, freq_per_year=freq_per_year)
    return present_value(cfs, curve)


def price_per_100_from_pv(
    pv: float,
    quantity: float,
) -> float:
    """
    Convert a PV in dollars to a price per 100 of par.
    """
    if quantity == 0:
        raise ValueError("Quantity must be non-zero to compute price per 100.")
    return pv * 100.0 / float(quantity)


def price_to_maturity(
    bond: Bond,
    curve: ZeroCurve,
    freq_per_year: int = 2,
) -> float:
    """
    Model price (per 100 of par) if held to maturity.
    """
    pv = pv_to_maturity(bond, curve, freq_per_year=freq_per_year)
    return price_per_100_from_pv(pv, bond.quantity)


def price_to_call(
    bond: Bond,
    curve: ZeroCurve,
    freq_per_year: int = 2,
) -> float:
    """
    Model price (per 100 of par) if bond is redeemed at first call.
    """
    pv = pv_to_call(bond, curve, freq_per_year=freq_per_year)
    return price_per_100_from_pv(pv, bond.quantity)
