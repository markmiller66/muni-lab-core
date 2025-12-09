# src/muni_core/oas/__init__.py

from .simple_oas import (
    OASResult,
    price_bond_with_oas,
    solve_oas_for_price,
)

__all__ = [
    "OASResult",
    "price_bond_with_oas",
    "solve_oas_for_price",
]
