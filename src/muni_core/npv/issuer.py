# src/muni_core/npv/issuer.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple, List

import math

from ..model import Bond
from ..curves.types import ZeroCurve


def _year_fraction(start: date, end: date) -> float:
    """
    Simple Actual/365.25 year fraction.

    Kept local to avoid circular imports.
    """
    days = (end - start).days
    return days / 365.25


def _build_coupon_times_from_call(
    settle_date: date,
    call_date: date,
    maturity_date: date,
    freq: int = 2,
) -> Tuple[List[float], float, float]:
    """
    Build a simple schedule of coupon times (in years from settle_date)
    from the first coupon *after* call_date out to maturity_date.

    Returns:
        times:   list of t_i in years from settle_date
        t_call:  time of call (years from settle_date)
        t_mat:   time of maturity (years from settle_date)
    """
    t_call = _year_fraction(settle_date, call_date)
    t_mat = _year_fraction(settle_date, maturity_date)

    # If maturity is at/ before call, nothing to compare
    if t_mat <= t_call:
        return [], t_call, t_mat

    step = 1.0 / float(freq)

    times: List[float] = []
    t = t_call + step
    # Allow small epsilon beyond t_mat
    eps = 1e-8
    while t <= t_mat + eps:
        times.append(t)
        t += step

    return times, t_call, t_mat


@dataclass
class IssuerCallResult:
    """
    Issuer-side view of call economics.

    All NPV values are from the issuer's perspective, discounted to settle_date
    using the provided ZeroCurve.
    """

    npv_keep: float          # NPV of keeping old coupon after call date
    npv_refi: float          # NPV of refinancing at call date
    savings: float           # npv_keep - npv_refi  (positive => savings if call)
    savings_pct: float       # savings / npv_keep
    new_coupon: float        # assumed coupon on the new issue
    threshold_pct: float     # decision threshold (e.g. 0.03)
    label: str               # "LikelyCall" / "MaybeCall" / "NoCall"


def compute_issuer_call_npv(
    bond: Bond,
    curve: ZeroCurve,
    *,
    threshold_pct: float = 0.03,
    coupon_freq: int = 2,
    new_issue_spread_bp: float = 50.0,
    issue_cost_bp: float = 50.0,
    today: Optional[date] = None,
) -> IssuerCallResult:
    """
    Compute NPV savings from the issuer's perspective if they call at the first
    call date and refinance at a new coupon implied by the current curve.

    Assumptions:
      - Issuer compares only cashflows *after the call date*.
      - New issue has the same final maturity as the old bond.
      - New issue coupon ~= zero_curve_yield(remaining_term) + new_issue_spread_bp.
      - Issuance cost = issue_cost_bp * notional at call date.
      - Coupon frequency is fixed (semiannual by default).

    This deliberately ignores *investor* market price; it models only issuer
    interest cost vs refi cost.
    """
    if today is None:
        # fall back to bond.settle_date, then to maturity minus a small buffer
        if bond.settle_date is not None:
            today = bond.settle_date
        else:
            # extremely defensive: use "today" as just before maturity
            if bond.maturity_date is None:
                raise ValueError("Bond must have either settle_date or maturity_date set")
            today = bond.maturity_date

    if bond.maturity_date is None:
        raise ValueError("Bond must have maturity_date for issuer NPV")

    if bond.call_feature is None or bond.call_feature.call_date is None:
        # No call feature => issuer cannot exercise; treat as NoCall with zero savings
        return IssuerCallResult(
            npv_keep=0.0,
            npv_refi=0.0,
            savings=0.0,
            savings_pct=0.0,
            new_coupon=0.0,
            threshold_pct=threshold_pct,
            label="NoCall",
        )

    call_date = bond.call_feature.call_date
    call_price = bond.call_feature.call_price

    # If call date is in the past relative to "today", treat today as effective call boundary
    if call_date < today:
        call_date = today

    # Build simple time grid from call date to maturity
    times, t_call, t_mat = _build_coupon_times_from_call(
        settle_date=today,
        call_date=call_date,
        maturity_date=bond.maturity_date,
        freq=coupon_freq,
    )

    if not times:
        # Nothing meaningful after call date
        return IssuerCallResult(
            npv_keep=0.0,
            npv_refi=0.0,
            savings=0.0,
            savings_pct=0.0,
            new_coupon=0.0,
            threshold_pct=threshold_pct,
            label="NoCall",
        )

    # Notional: use bond.quantity if present; else assume 100_000
    notional = float(bond.quantity) if bond.quantity is not None else 100_000.0

    # Old coupon and per-period coupon payment
    old_coupon = float(bond.coupon)  # e.g. 0.04 = 4%
    old_cpn_per_period = old_coupon / float(coupon_freq) * notional

    # Discount factors for all relevant times
    def df(t: float) -> float:
        return curve.discount_factor(t)

    # NPV of KEEPING the old coupon from call date to maturity
    npv_keep = 0.0
    for t in times:
        npv_keep += old_cpn_per_period * df(t)
    # principal at maturity
    npv_keep += notional * df(t_mat)

    # New issue coupon: approximate from curve using remaining term
    remaining_term = max(1e-6, t_mat - t_call)
    base_yield = curve.zero_rate(remaining_term)
    new_coupon = base_yield + new_issue_spread_bp / 10_000.0
    new_cpn_per_period = new_coupon / float(coupon_freq) * notional

    # NPV of REFINANCING at call date
    npv_refi = 0.0
    for t in times:
        npv_refi += new_cpn_per_period * df(t)
    # principal at maturity for the new issue
    npv_refi += notional * df(t_mat)

    # Issuance cost at call date (in bp of notional)
    issue_cost = issue_cost_bp / 10_000.0 * notional
    npv_refi += issue_cost * df(t_call)

    savings = npv_keep - npv_refi
    if npv_keep > 0:
        savings_pct = savings / npv_keep
    else:
        savings_pct = 0.0

    # Map savings_pct to a label
    if savings_pct >= threshold_pct:
        label = "LikelyCall"
    elif savings_pct >= 0.5 * threshold_pct:
        label = "MaybeCall"
    else:
        label = "NoCall"

    return IssuerCallResult(
        npv_keep=npv_keep,
        npv_refi=npv_refi,
        savings=savings,
        savings_pct=savings_pct,
        new_coupon=new_coupon,
        threshold_pct=threshold_pct,
        label=label,
    )
