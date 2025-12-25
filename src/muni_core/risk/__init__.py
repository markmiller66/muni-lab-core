from __future__ import annotations# FILE: src/muni_core/risk/__init__.py
"""
Risk analytics for muni_core.

Scope:
- curve-space risk: parallel bump DV01/duration
- curve-shape risk: key-rate duration/convexity (KRD/KRC)

Notes:
- callable_krd_hw.py is a backwards-compatible shim.
- callable_krd_hw_triangular.py is the canonical triangular bump KRD engine.
"""
"""
Risk analytics for muni_core.
"""



from .callable_parallel_curve_hw import CallableParallelCurveResult, compute_callable_parallel_curve_hw
from .callable_krd_hw_triangular import CallableKRDResult, compute_callable_krd_hw
from .krd_summary import KRDSummary, summarize_callable_krd, krd_summary_to_frame

__all__ = [
    "CallableParallelCurveResult",
    "compute_callable_parallel_curve_hw",
    "CallableKRDResult",
    "compute_callable_krd_hw",
    "KRDSummary",
    "summarize_callable_krd",
    "krd_summary_to_frame",
]
