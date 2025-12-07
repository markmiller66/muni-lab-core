from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from muni_core.model import Bond
from muni_core.curves import ZeroCurve, forward_at_call, forward_after_call, forward_slope_around_call_bp
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
    base = evaluate_call_npv(bond, curve, threshold_pct=threshold_pct, freq_per_year=freq_per_year)

    f_call = forward_at_call(bond, curve, window_years=fwd_window_years)
    f_after = forward_after_call(bond, curve, window_years=fwd_window_years, offset_years=fwd_offset_years)
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
