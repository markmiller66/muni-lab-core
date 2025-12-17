from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, List

from muni_core.model import Bond
from muni_core.curves import (
    ZeroCurve,
    forward_at_call,
    forward_after_call,
    forward_slope_around_call_bp,
    year_fraction,
)
from muni_core.npv import pv_to_maturity, pv_to_call


@dataclass
class NPVCallResult:
    pv_no_call: float
    pv_call: Optional[float]
    savings_pct: Optional[float]
    threshold_pct: float
    label: str


@dataclass
class NPVForwardCallResult:
    """
    Extended call diagnostic that includes forward curve information
    at and around the call date.
    """
    pv_no_call: float
    pv_call: Optional[float]
    savings_pct: Optional[float]
    threshold_pct: float
    label: str

    forward_at_call: Optional[float]          # decimal (e.g. 0.035 = 3.5%)
    forward_after_call: Optional[float]       # decimal
    forward_slope_bp: Optional[float]         # basis points (after - at_call)


def evaluate_call_npv(
    bond: Bond,
    curve: ZeroCurve,
    threshold_pct: float = 0.03,  # 3% savings threshold
    freq_per_year: int = 2,
) -> NPVCallResult:
    """
    Simple NPV-based call test from the issuer's point of view.

    pv_no_call: PV of existing bond cashflows to maturity.
    pv_call:    PV of existing bond cashflows to call date
                (assuming redeemed at call_price).
                If no call feature, this is None.

    savings_pct: (pv_no_call - pv_call) / pv_no_call.
                 Higher means more incentive to call.

    label: "NoCall", "MaybeCall", "LikelyCall"
    """
    pv_no_call = pv_to_maturity(bond, curve, freq_per_year=freq_per_year)

    if not bond.has_call():
        return NPVCallResult(
            pv_no_call=pv_no_call,
            pv_call=None,
            savings_pct=None,
            threshold_pct=threshold_pct,
            label="NoCall (no call feature)",
        )

    pv_call = pv_to_call(bond, curve, freq_per_year=freq_per_year)

    # If pv_call >= pv_no_call, there is no savings in calling.
    savings = pv_no_call - pv_call
    savings_pct = savings / pv_no_call if pv_no_call != 0 else None

    if savings_pct is None or savings_pct <= 0:
        label = "NoCall"
    elif savings_pct < threshold_pct:
        label = "MaybeCall"
    else:
        label = "LikelyCall"

    return NPVCallResult(
        pv_no_call=pv_no_call,
        pv_call=pv_call,
        savings_pct=savings_pct,
        threshold_pct=threshold_pct,
        label=label,
    )


def evaluate_call_with_forwards(
    bond: Bond,
    curve: ZeroCurve,
    threshold_pct: float = 0.03,
    freq_per_year: int = 2,
    fwd_window_years: float = 1.0,
    fwd_offset_years: float = 1.0,
) -> NPVForwardCallResult:
    """
    Combined NPV + forward-curve diagnostic.

    Adds:
      - forward_at_call:  approx 1Y forward ending at call date
      - forward_after_call: approx 1Y forward ending at call+offset_years
      - forward_slope_bp:  (after - at_call) * 10_000

    This is a good place to encode your "issuer forward view" logic.
    """
    base = evaluate_call_npv(
        bond,
        curve,
        threshold_pct=threshold_pct,
        freq_per_year=freq_per_year,
    )

    f_call = forward_at_call(bond, curve, window_years=fwd_window_years)
    f_after = forward_after_call(
        bond,
        curve,
        window_years=fwd_window_years,
        offset_years=fwd_offset_years,
    )
    slope_bp = forward_slope_around_call_bp(
        bond,
        curve,
        window_years=fwd_window_years,
        offset_years=fwd_offset_years,
    )

    return NPVForwardCallResult(
        pv_no_call=base.pv_no_call,
        pv_call=base.pv_call,
        savings_pct=base.savings_pct,
        threshold_pct=base.threshold_pct,
        label=base.label,
        forward_at_call=f_call,
        forward_after_call=f_after,
        forward_slope_bp=slope_bp,
    )


# ---------------------------------------------------------------------------
# Bermudan 7-date call schedule (3-year window, 6-month steps)
# ---------------------------------------------------------------------------

def _add_months(d: date, months: int) -> date:
    """
    Add a number of months to a date, clamping to the last valid day
    of the target month.

    This avoids pulling in dateutil just for relativedelta and is
    sufficient for standard muni call dates (1st / 15th / EOM).
    """
    new_month = d.month - 1 + months
    new_year = d.year + new_month // 12
    new_month = new_month % 12 + 1

    day = d.day
    while True:
        try:
            return date(new_year, new_month, day)
        except ValueError:
            day -= 1
            if day <= 0:
                # Defensive fallback; should never occur in practice.
                return date(new_year, new_month, 1)


def build_bermudan_call_times_3yr_window(
    asof_date: date,
    first_call_date: Optional[date],
    maturity_date: date,
) -> List[float]:
    """
    Build Bermudan call exercise times over a 3-year window after first call.

    Rules:
    - Call 0 = first_call_date
    - Then every 6 months for 3 years (max 7 dates total: 0..6)
    - Stop early if we pass maturity_date
    - Return ONLY future exercise times (t > 0) as ACT/365.25
      year fractions from asof_date.

    Parameters
    ----------
    asof_date : date
        Valuation / pricing date (today).
    first_call_date : date or None
        First optional call date; if None or invalid, returns [].
    maturity_date : date
        Final maturity date of the bond.

    Returns
    -------
    List[float]
        List of exercise times (in years) relative to asof_date.
        These are the candidate Bermudan exercise times for the HW lattice.
    """
    if first_call_date is None:
        return []

    # If first call is on/after maturity, there is effectively no call option.
    if first_call_date >= maturity_date:
        return []

    times: List[float] = []
    current = first_call_date

    max_steps = 7      # first_call + up to 6 more
    step_months = 6    # 6-month intervals

    for _ in range(max_steps):
        if current > maturity_date:
            break

        t = year_fraction(asof_date, current)

        # Only keep *future* decision times.
        if t > 0.0:
            times.append(t)

        # Move to the next potential call date
        current = _add_months(current, step_months)

    return times
