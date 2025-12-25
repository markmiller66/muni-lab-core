"""
Backwards-compatible shim.

The canonical triangular-bump callable KRD implementation moved to:
- callable_krd_hw_triangular.py

Do NOT add logic here. Keep this file as a re-export layer so
older imports remain stable.
"""

from __future__ import annotations

from .callable_krd_hw_triangular import CallableKRDResult, compute_callable_krd_hw

__all__ = ["CallableKRDResult", "compute_callable_krd_hw"]
