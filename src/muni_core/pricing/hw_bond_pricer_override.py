from __future__ import annotations

from datetime import date
import pandas as pd

from muni_core.model import Bond
from muni_core.pricing.hw_bond_pricer import price_callable_bond_from_lattice
from muni_core.curves.short_rate_lattice import build_binomial_lattice_from_hw
from muni_core.curves.history import build_hw_theta_from_dense
from muni_core.calls.call_schedule import build_bermudan_call_times_3yr_window


def price_callable_bond_hw_from_bond_dense_override(
    bond: Bond,
    asof: date,
    dense_df: pd.DataFrame,
    a: float,
    sigma: float,
    oas_bp: float,
    allow_call: bool = True,
    freq_per_year: int = 2,
    face: float = 100.0,
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> float:

    """
    Same callable pricer, but uses caller-provided dense curve (so KRD bumps work).

    IMPORTANT:
      build_hw_theta_from_dense() in this repo expects:
        - tenor_yrs
        - rate_dec  (decimal zero/spot)
      So this function normalizes inputs to that schema.
    """

    if bond.maturity_date is None:
        raise ValueError("Bond.maturity_date is required.")
    if bond.maturity_date <= asof:
        raise ValueError(f"maturity {bond.maturity_date} must be after asof {asof}")

    dense = dense_df.copy()

    # ------------------------------------------------------------------
    # Normalize TENOR column -> tenor_yrs (what build_hw_theta_from_dense expects)
    # ------------------------------------------------------------------
    if "tenor_yrs" not in dense.columns:
        if "tenor_years" in dense.columns:
            dense = dense.rename(columns={"tenor_years": "tenor_yrs"})
        elif "tenor" in dense.columns:
            dense = dense.rename(columns={"tenor": "tenor_yrs"})
        elif "tenor_yr" in dense.columns:
            dense = dense.rename(columns={"tenor_yr": "tenor_yrs"})
        elif "TenorY" in dense.columns:
            dense = dense.rename(columns={"TenorY": "tenor_yrs"})

    if "tenor_yrs" not in dense.columns:
        raise KeyError(f"Expected tenor_yrs; got columns={list(dense.columns)}")

    # ------------------------------------------------------------------
    # Normalize RATE column -> rate_dec (what build_hw_theta_from_dense expects)
    # ------------------------------------------------------------------
    if "rate_dec" not in dense.columns:
        # common variants we may produce elsewhere
        for c in (
            "zero_rate",
            "ZeroRate",
            "ZeroRateDec",
            "rate",
            "spot_rate",
            "zr",
            "zero",
            "zeroRate",
        ):
            if c in dense.columns:
                dense = dense.rename(columns={c: "rate_dec"})
                break

    if "rate_dec" not in dense.columns:
        raise KeyError(f"Expected rate_dec; got columns={list(dense.columns)}")

    # Apply constant OAS (bp) ONCE to the curve used by theta-builder
    dense["rate_dec"] = dense["rate_dec"].astype(float) + (oas_bp / 10000.0)

    # Optional alias (harmless; helps other utilities)
    dense["zero_rate"] = dense["rate_dec"]

    # ------------------------------------------------------------------
    # Build theta + lattice
    # ------------------------------------------------------------------
    hw_df = build_hw_theta_from_dense(dense, a=a, sigma=sigma)
    lattice = build_binomial_lattice_from_hw(hw_df, a=a, sigma=sigma, dt_years=step_years)

    maturity_years = (bond.maturity_date - asof).days / 365.25
    coupon_rate = float(bond.coupon or 0.0)


    # Bermudan call times (call exercise can be disabled for deterministic runs)
    call_times_yrs: list[float] = []
    call_price = face
    if allow_call and getattr(bond, "call_feature", None) is not None:
        call_dt = bond.call_feature.call_date
        if call_dt is not None and call_dt > asof:
            call_times_yrs = build_bermudan_call_times_3yr_window(
                asof_date=asof,
                first_call_date=call_dt,
                maturity_date=bond.maturity_date,
            )
            call_price = float(getattr(bond.call_feature, "call_price", face) or face)

    return float(
        price_callable_bond_from_lattice(
            lattice=lattice,
            maturity_years=maturity_years,
            coupon_rate=coupon_rate,
            freq_per_year=freq_per_year,
            call_times_yrs=call_times_yrs,
            face=face,
            call_price=call_price,
            q=q,
            time_tolerance=time_tolerance,
                    )
    )
