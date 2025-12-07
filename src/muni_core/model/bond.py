from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class CallFeature:
    """
    Represents the main (first) call feature of a bond.
    We can extend this later with full schedules.
    """
    call_date: Optional[date]
    call_price: float = 100.0  # percent of par, e.g. 100 = par, 101 = 101%


@dataclass
class Bond:
    """
    Minimal muni bond representation for now.

    Conventions:
    - coupon: percent per year (5.0 means 5% annual coupon)
    - clean_price: percent of par (e.g. 99.25 means 99.25% of par)
    - quantity: par amount in dollars (e.g. 250000 means 250k face)
    """

    cusip: str
    rating: str
    rating_num: int

    basis: str  # e.g. "Actual/Actual"
    settle_date: Optional[date]
    maturity_date: Optional[date]

    coupon: float
    clean_price: float
    quantity: float

    call_feature: Optional[CallFeature] = None

    def has_call(self) -> bool:
        return self.call_feature is not None
