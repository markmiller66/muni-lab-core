from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


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
        Simple implied forward rate between t1 and t2 (continuously compounded).

        Formula:
            DF(t) = exp(-r(t)*t)
            F(t1, t2) solves DF(t2) = DF(t1) * exp(-F * (t2 - t1))
        So:
            F = (r(t2)*t2 - r(t1)*t1) / (t2 - t1)
        """
        t1 = float(t1)
        t2 = float(t2)
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1 for forward rate")

        r1 = self.zero_rate(t1)
        r2 = self.zero_rate(t2)

        return (r2 * t2 - r1 * t1) / (t2 - t1)
