# FILE: src/muni_core/oas/callable_krd_hw_triangular.py
"""
Backward-compatibility shim.

Historically, callable KRD lived under muni_core.oas.
Canonical location is now muni_core.risk.
"""

from muni_core.risk.callable_krd_hw_triangular import *  # noqa: F401,F403
