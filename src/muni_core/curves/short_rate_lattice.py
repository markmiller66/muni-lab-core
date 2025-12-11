# src/muni_core/curves/short_rate_lattice.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class ShortRateLattice:
    """
    Simple recombining short-rate lattice:

        rates[t][j] = short rate at time step t, node j

    with constant time step dt_years.
    """
    dt_years: float
    rates: List[List[float]]  # rates[time_step][node]


def build_hw_short_rate_lattice(
    dense_df: pd.DataFrame,
    a: float,
    sigma: float,
    dt_years: float = 1.0,
) -> ShortRateLattice:
    """
    Build a very simple Hull–White-style short-rate lattice from a dense
    zero curve.

    dense_df is expected to have columns:
        - tenor_yrs (float)
        - rate_dec  (zero yield in decimal)

    This is intentionally *simple* and meant as a wiring + sanity-check
    implementation, not a production-calibrated HW model:

      1. We take the dense zero curve, sorted by tenor_yrs.
      2. For each time step t_k = (k+1) * dt_years, we linearly interpolate
         the zero rate z(t_k).
      3. That interpolated rate is used as a "base" short rate at that step.
      4. Node j at level k gets:

             r(k, j) = base_k + (2*j - k) * sigma * sqrt(dt_years)

         which creates a recombining binomial tree with volatility sigma.

    The 'a' parameter (mean reversion) is accepted here but not yet used,
    so that in the future you can upgrade this to a proper calibrated
    Hull–White construction without changing the public interface.
    """
    if dense_df.empty:
        raise ValueError("dense_df is empty; cannot build short-rate lattice.")

    df = dense_df.copy()
    df = df.sort_values("tenor_yrs")

    tenors = df["tenor_yrs"].to_numpy(dtype=float)
    rates = df["rate_dec"].to_numpy(dtype=float)

    if len(tenors) < 2:
        raise ValueError("Need at least 2 tenor points in dense_df to build lattice.")

    if dt_years <= 0.0:
        raise ValueError(f"dt_years must be positive, got {dt_years}.")

    # We'll build steps up to the max tenor covered by the dense curve.
    max_T = float(tenors.max())
    n_steps = int(max_T // dt_years)
    if n_steps < 1:
        raise ValueError(
            f"Not enough horizon for dt_years={dt_years}; max tenor is {max_T}."
        )

    # Interpolator for zero rates z(t)
    times = np.asarray(tenors, dtype=float)
    zr = np.asarray(rates, dtype=float)

    def z_of_t(t: float) -> float:
        # clip to the available range to avoid NaNs at edges
        t_clipped = np.clip(t, times.min(), times.max())
        return float(np.interp(t_clipped, times, zr))

    lattice_rates: List[List[float]] = []

    # sigma * sqrt(dt) used as the per-step shift magnitude
    shift_unit = sigma * np.sqrt(dt_years)

    for k in range(n_steps):
        t_k = (k + 1) * dt_years  # time at this step
        base_k = z_of_t(t_k)
        level_rates: List[float] = []
        for j in range(k + 1):
            # simple symmetric binomial around base_k
            shift = (2 * j - k) * shift_unit
            level_rates.append(base_k + shift)
        lattice_rates.append(level_rates)

    return ShortRateLattice(dt_years=dt_years, rates=lattice_rates)


def build_state_price_tree_from_lattice(
    lattice: ShortRateLattice,
    p_up: float = 0.5,
) -> List[List[float]]:
    """
    Build a state price tree from a short-rate lattice.

    We assume a recombining binomial structure:

      - From node (t, j) you can go to:
            (t+1, j)   with probability (1 - p_up)
            (t+1, j+1) with probability p_up

      - Discounting uses the short rate r(t, j) at the *start* of the period:

            state_price(t+1, node) += state_price(t, j)
                                      * prob
                                      * exp(-r(t,j) * dt)

    This is *not* a fully calibrated risk-neutral HW model, but it
    provides a clean and deterministic mapping from (rates, dt) ->
    state prices that you can plug into the bond pricer.

    Returns:
        state_prices[t][j] = state price at time step t, node j.
    """
    dt = lattice.dt_years
    rates = lattice.rates

    n_levels = len(rates)
    if n_levels == 0:
        raise ValueError("ShortRateLattice has no levels.")

    # Initialize state price tree
    state_prices: List[List[float]] = []
    state_prices.append([1.0])  # at t=0, single node with state price 1.0

    for t in range(n_levels):
        current_rates = rates[t]
        current_states = state_prices[t]
        n_nodes = len(current_states)

        # next level has n_nodes + 1 nodes
        next_states = [0.0] * (n_nodes + 1)

        for j in range(n_nodes):
            r_tj = current_rates[j]
            pi_tj = current_states[j]

            # discount factor for this step
            disc = float(np.exp(-r_tj * dt))

            # down move: (t+1, j)
            next_states[j] += pi_tj * (1.0 - p_up) * disc
            # up move: (t+1, j+1)
            next_states[j + 1] += pi_tj * p_up * disc

        state_prices.append(next_states)

    return state_prices

def build_short_rate_path_from_hw(
    dense_df: pd.DataFrame,
    a: float,
    sigma: float,
    dt_years: float = 1.0,
):
    """
    Convenience helper retained for backward compatibility with older code.

    Given a dense zero curve (tenor_yrs, rate_dec) and Hull–White params
    (a, sigma, dt_years), this:

      1) Builds a ShortRateLattice via build_hw_short_rate_lattice.
      2) Extracts a single "central" short-rate path r(t_k):
            - time grid t_k = (k+1) * dt_years
            - r_path[k] = rate at the middle node of level k

    Returns:
        times  (np.ndarray): shape (n_steps,), time in years
        r_path (np.ndarray): shape (n_steps,), short rate at each step
    """
    # 1) Build the lattice
    lattice = build_hw_short_rate_lattice(
        dense_df=dense_df,
        a=a,
        sigma=sigma,
        dt_years=dt_years,
    )

    levels = lattice.rates
    n_steps = len(levels)
    if n_steps == 0:
        raise ValueError("ShortRateLattice has no levels in build_short_rate_path_from_hw.")

    # 2) Time grid: t_k = (k+1) * dt
    times = np.array([(k + 1) * lattice.dt_years for k in range(n_steps)], dtype=float)

    # 3) "Central" path: middle node at each level
    r_path = []
    for level in levels:
        j_mid = len(level) // 2
        r_path.append(float(level[j_mid]))

    r_path = np.array(r_path, dtype=float)

    return times, r_path
def build_binomial_lattice_from_hw(
    dense_df: pd.DataFrame,
    a: float,
    sigma: float,
    dt_years: float = 1.0,
):
    """
    Backward-compatibility alias for older code that expected a
    'build_binomial_lattice_from_hw' helper.

    In this muni-lab-core version, we treat it as a thin wrapper around
    build_hw_short_rate_lattice and simply return the ShortRateLattice
    object.

    Args:
        dense_df: DataFrame with columns [tenor_yrs, rate_dec] for a single date.
        a:       Hull–White mean reversion speed.
        sigma:   Hull–White short rate volatility.
        dt_years: Time step in years (1.0 = annual, 0.5 = semi-annual).

    Returns:
        ShortRateLattice instance.
    """
    return build_hw_short_rate_lattice(
        dense_df=dense_df,
        a=a,
        sigma=sigma,
        dt_years=dt_years,
    )
