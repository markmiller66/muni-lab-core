

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import math

@dataclass
class CurvePoint:
    """
    A single point on a zero curve.

    tenor_years: time to maturity in years (e.g. 1.0, 5.0, 10.0)
    zero_rate:   continuously compounded zero rate (decimal, e.g. 0.0225 = 2.25%)
    """
    tenor_years: float
    zero_rate: float


@dataclass
class ZeroCurve:
    """
    Simple zero curve built from (tenor_years, zero_rate) points.

    Provides:
    - zero_rate(t): linear interpolation in time
    - discount_factor(t): exp(-r(t) * t)
    - forward_rate(t1, t2): implied forward between t1 and t2
    """

    points: List[CurvePoint]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("ZeroCurve requires at least one point")
        # Sort by tenor
        self.points.sort(key=lambda p: p.tenor_years)

    @classmethod
    def from_pairs(cls, pairs: Iterable[Tuple[float, float]]) -> "ZeroCurve":
        pts = [CurvePoint(float(t), float(r)) for t, r in pairs]
        return cls(points=pts)

    def zero_rate(self, t_years: float) -> float:
        """
        Linear interpolation of the zero rate at time t_years.
        Extrapolates flat beyond the last point.
        """
        t = float(t_years)
        pts = self.points

        if t <= pts[0].tenor_years:
            return pts[0].zero_rate
        if t >= pts[-1].tenor_years:
            return pts[-1].zero_rate

        # Find bounding segment
        for i in range(1, len(pts)):
            left = pts[i - 1]
            right = pts[i]
            if left.tenor_years <= t <= right.tenor_years:
                w = (t - left.tenor_years) / (right.tenor_years - left.tenor_years)
                return left.zero_rate + w * (right.zero_rate - left.zero_rate)

        # Fallback (should not hit)
        return pts[-1].zero_rate

    def discount_factor(self, t_years: float) -> float:
        """
        DF(t) = exp(-r(t) * t), using continuously compounded zero rate.
        """
        r = self.zero_rate(t_years)
        # could import math, but this avoids adding another import
        return 2.718281828459045 ** (-r * float(t_years))

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Forward rate between t1 and t2 (year fractions).

        Robust to bad inputs:
          - clamp negative times to 0
          - if t2 <= t1, nudge t2 slightly forward so we don't explode

        This keeps diagnostics like "forward at call" from blowing up if
        the call date is in the past or if start/end dates coincide.
        """
        # Clamp to non-negative times
        t1 = max(0.0, float(t1))
        t2 = max(0.0, float(t2))

        # Ensure t2 > t1 by at least a tiny epsilon
        if t2 <= t1:
            t2 = t1 + 1e-6  # effectively instantaneous forward

        # Use whatever your discount method is called: discount() or discount_factor()
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)

        if df1 <= 0 or df2 <= 0:
            raise ValueError(f"Non-positive discount factors: df1={df1}, df2={df2}")

        return -math.log(df2 / df1) / (t2 - t1)

