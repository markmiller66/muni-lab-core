from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from muni_core.model import Bond
from muni_core.curves import ZeroCurve
from muni_core.npv import pv_to_maturity, pv_to_call


@dataclass
class NPVCallResult:
    pv_no_call: float
    pv_call: Optional[float]
    savings_pct: Optional[float]
    threshold_pct: float
    label: str


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
