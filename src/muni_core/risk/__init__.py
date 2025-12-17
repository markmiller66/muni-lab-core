# FILE: src/muni_core/risk/__init__.py
"""
Risk analytics for muni_core.

Scope:
- curve-space risk: parallel bump DV01/duration
- curve-shape risk: key-rate duration/convexity (KRD/KRC)

Notes:
- callable_krd_hw.py is the canonical "triangular bump" KRD.
- callable_parallel_curve_hw.py is a debug/sanity tool (nearest-node bump).
"""

from .callable_krd_hw import CallableKRDResult, compute_callable_krd_hw
from .callable_parallel_curve_hw import compute_callable_parallel_curve_hw, CallableParallelCurveResult
from .callable_krd_hw import compute_callable_krd_hw, CallableKRDResult

from .krd_summary import KRDSummary, summarize_callable_krd, krd_summary_to_frame


# If/when you add a parallel curve bump module:
# from .callable_parallel_curve_hw import CallableParallelResult, compute_callable_parallel_curve_hw


__all__ = [
    "compute_callable_parallel_curve_hw",
    "CallableParallelCurveResult",
    "compute_callable_krd_hw",
    "CallableKRDResult",
    "summarize_callable_krd",
    "KRDSummary",
    "summarize_callable_krd",
    "krd_summary_to_frame",


]

