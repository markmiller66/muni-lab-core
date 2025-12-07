from __future__ import annotations

from typing import Iterable, Tuple

from .types import ZeroCurve, CurvePoint


def make_zero_curve_from_pairs(pairs: Iterable[Tuple[float, float]]) -> ZeroCurve:
    """
    Convenience helper so callers don't need to import ZeroCurve directly.
    """
    return ZeroCurve.from_pairs(pairs)


__all__ = ["ZeroCurve", "CurvePoint", "make_zero_curve_from_pairs"]
