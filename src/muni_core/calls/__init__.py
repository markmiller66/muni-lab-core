"""
Call-related utilities for muni_core.
"""

from .npv_call_test import (
    NPVCallResult,
    NPVForwardCallResult,
    evaluate_call_npv,
    evaluate_call_with_forwards,
)

__all__ = [
    "NPVCallResult",
    "NPVForwardCallResult",
    "evaluate_call_npv",
    "evaluate_call_with_forwards",
]
