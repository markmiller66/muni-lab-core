# src/muni_core/risk/oas_dv01_callable_hw.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from muni_core.model import Bond
from muni_core.config.loader import AppConfig
from muni_core.pricing.hw_bond_pricer import price_callable_bond_hw_from_bond


@dataclass
class OASDV01Result:
    """
    Parallel OAS DV01 for a callable bond on the HW lattice.

    All prices are per 100 face (investor convention).

    dv01_bp:   price change per 1 bp OAS *decrease* (should be > 0)
    mod_dur:   modified duration implied by that DV01
    """
    base_oas_bp: float
    bump_bp: float

    price_base: float
    price_up: float
    price_down: float

    dv01_bp: float      # dollars per 100 per 1 bp
    mod_duration: float # duration in years (approx)


def oas_dv01_callable_hw_for_bond(
    *,
    bond: Bond,
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    base_oas_bp: float,
    bump_bp: float = 1.0,       # size of OAS bump in bp (both + and -)
    freq_per_year: int = 2,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> OASDV01Result:
    """
    Compute an OAS-based DV01 for a callable bond using the HW lattice.

    We hold the curve and call schedule fixed and shift ONLY the OAS:

        base:  P(OAS)
        up:    P(OAS + bump_bp)
        down:  P(OAS - bump_bp)

    Then:

        dv01_bp      ≈ (P_down - P_up) / (2 * bump_bp)
        mod_duration ≈ dv01_bp / (P_base * 0.0001)

    (dv01_bp is per 1 bp; mod_duration is in years.)
    """

    if base_oas_bp is None:
        raise ValueError("base_oas_bp is required for OAS DV01.")

    bump_bp = float(bump_bp)
    if bump_bp <= 0.0:
        raise ValueError("bump_bp must be > 0.")

    # --- Base price at base OAS --------------------------------------
    p_base = float(
        price_callable_bond_hw_from_bond(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=base_oas_bp,
            freq_per_year=freq_per_year,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )

    # --- Prices at OAS +/- bump --------------------------------------
    p_up = float(
        price_callable_bond_hw_from_bond(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=base_oas_bp + bump_bp,
            freq_per_year=freq_per_year,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )

    p_down = float(
        price_callable_bond_hw_from_bond(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            oas_bp=base_oas_bp - bump_bp,
            freq_per_year=freq_per_year,
            curve_key=curve_key,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
    )

    # --- Central difference DV01 (per 1 bp) --------------------------
    # bump_bp is in bp, so divide by (2 * bump_bp) to get per-1bp change.
    dv01_bp = (p_down - p_up) / (2.0 * bump_bp)

    # Standard relationship:
    #   DV01 = Duration_mod * Price * 0.0001
    # => Duration_mod ≈ DV01 / (Price * 0.0001)
    if p_base <= 0.0:
        mod_duration = 0.0
    else:
        mod_duration = dv01_bp / (p_base * 0.0001)

    return OASDV01Result(
        base_oas_bp=float(base_oas_bp),
        bump_bp=bump_bp,
        price_base=p_base,
        price_up=p_up,
        price_down=p_down,
        dv01_bp=dv01_bp,
        mod_duration=mod_duration,
    )
