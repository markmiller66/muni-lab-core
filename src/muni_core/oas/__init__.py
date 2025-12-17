# src/muni_core/oas/__init__.py

from __future__ import annotations
from .callable_krd_hw import CallableKRDResult, compute_callable_krd_hw
from .simple_oas import (
    OASResult,
    solve_oas_for_price,
    price_bond_with_oas,
)

# Callable OAS via HW lattice
from .callable_oas_hw import solve_callable_oas_hw

__all__ = [
    "OASResult",
    "price_bond_with_oas",
    "solve_oas_for_price",
    "solve_callable_oas_hw",
    "CallableKRDResult", "compute_callable_krd_hw"
]
