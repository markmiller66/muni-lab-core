# src/muni_core/curves/short_rate.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import math

from .types import ZeroCurve, CurvePoint


# ---------------------------------------------------------------------------
# 1) SanitizedCurve: defensive wrapper around ZeroCurve
# ---------------------------------------------------------------------------

@dataclass
class SanitizedCurve:
    """
    Defensive wrapper around ZeroCurve for short-rate calibration.

    Goals:
      * Never mutate the original ZeroCurve.
      * Clamp / dedupe tenors if needed.
      * Provide a stable view of (t, r) for building time grids.

    In practice your AAA curve is usually clean, but this layer gives
    you a single place to enforce invariants before plugging into
    Hull–White or Ho–Lee.
    """

    _curve: ZeroCurve
    _points: List[CurvePoint]

    @classmethod
    def from_zero_curve(
        cls,
        curve: ZeroCurve,
        *,
        min_tenor: float = 0.0,
        max_tenor: Optional[float] = None,
    ) -> "SanitizedCurve":
        """
        Copy points from an existing ZeroCurve and enforce:

          * sorted by tenor
          * optional min/max tenor clipping
          * no duplicate tenors (keep the last value for each)

        The original `curve` object is not mutated.
        """
        # Copy + sort
        pts = list(curve.points)
        pts.sort(key=lambda p: p.tenor_years)

        # Optional clipping
        if min_tenor is not None and min_tenor > 0.0:
            pts = [p for p in pts if p.tenor_years >= min_tenor]
        if max_tenor is not None:
            pts = [p for p in pts if p.tenor_years <= max_tenor]

        if not pts:
            raise ValueError("SanitizedCurve: no points left after clipping")

        # Dedupe tenors (keep last rate for any repeated tenor)
        dedup: dict[float, CurvePoint] = {}
        for p in pts:
            dedup[float(p.tenor_years)] = CurvePoint(
                tenor_years=float(p.tenor_years),
                zero_rate=float(p.zero_rate),
            )

        final_points = list(dedup.values())
        final_points.sort(key=lambda p: p.tenor_years)

        # Build a *new* ZeroCurve so downstream code can treat this as
        # an independent object if needed.
        safe_curve = ZeroCurve(points=list(final_points))
        return cls(_curve=safe_curve, _points=final_points)

    @property
    def curve(self) -> ZeroCurve:
        """Access the internal safe ZeroCurve."""
        return self._curve

    @property
    def points(self) -> Sequence[CurvePoint]:
        """Read-only view of sanitized points."""
        return tuple(self._points)

    def zero_rate(self, t_years: float) -> float:
        """Delegate to the internal ZeroCurve."""
        return self._curve.zero_rate(t_years)

    def discount_factor(self, t_years: float) -> float:
        """DF(t) = exp(-r(t) * t), via ZeroCurve."""
        return self._curve.discount_factor(t_years)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Forward rate between t1 and t2 via ZeroCurve.forward_rate."""
        return self._curve.forward_rate(t1, t2)


# ---------------------------------------------------------------------------
# 2) Generic time / DF / forward grids for calibration
# ---------------------------------------------------------------------------

def make_time_grid(
    max_years: float,
    dt: float,
    *,
    include_zero: bool = True,
) -> List[float]:
    """
    Build a simple time grid [0, dt, 2*dt, ..., max_years].

    This is the natural grid for a short-rate lattice or for sampling
    theta(t) in Hull–White.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    n_steps = int(math.floor(max_years / dt + 1e-9))
    times: List[float] = []
    if include_zero:
        times.append(0.0)
    for k in range(1, n_steps + 1):
        t = k * dt
        if t <= max_years + 1e-9:
            times.append(t)
    return times


def discount_factor_grid(
    curve: SanitizedCurve,
    times: Sequence[float],
) -> List[Tuple[float, float]]:
    """
    Build a grid of (t, DF(t)) from a sanitized curve.
    """
    out: List[Tuple[float, float]] = []
    for t in times:
        df = curve.discount_factor(t)
        out.append((float(t), float(df)))
    return out


def forward_rate_grid(
    curve: SanitizedCurve,
    tenors: Sequence[float],
    window_years: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    For each tenor T in `tenors`, compute the forward rate over
    [T - window_years, T].

    This mirrors the forward-window logic you already use in the
    call-probability paths and is a convenient input for HW theta
    calibration.
    """
    out: List[Tuple[float, float]] = []
    w = float(window_years)
    if w <= 0.0:
        raise ValueError("window_years must be positive")

    for T in tenors:
        T = float(T)
        t2 = max(T, 0.0)
        t1 = max(0.0, t2 - w)
        if t2 <= t1:
            # Degenerate window; just use the spot zero
            f = curve.zero_rate(t2)
        else:
            f = curve.forward_rate(t1, t2)
        out.append((T, float(f)))
    return out


# ---------------------------------------------------------------------------
# 3) Hull–White parameter container (engine comes next)
# ---------------------------------------------------------------------------

@dataclass
class HullWhite1FParams:
    """
    Simple container for 1-factor Hull–White parameters.

    This is *not* the full model implementation yet; it exists so that
    calibration and lattice-building code can pass around (a, sigma)
    plus some metadata in a structured way.

        dr(t) = [theta(t) - a * r(t)] dt + sigma dW(t)
    """

    a: float              # mean reversion speed
    sigma: float          # short-rate volatility
    curve: SanitizedCurve # sanitized discount curve

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("HullWhite1FParams: a must be positive")
        if self.sigma <= 0.0:
            raise ValueError("HullWhite1FParams: sigma must be positive")
