# src/muni_core/pricing/hw_bond_pricer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Core cashflow representation
# -------------------------------------------------------------------


@dataclass
class BondCashflowSchedule:
    """
    Simple container for a level-coupon bond's cashflow schedule.

    t_yrs         : payment times in years
    amounts       : cashflow amounts at those times
    face_value    : principal repaid at maturity
    coupon_rate   : annual coupon rate (decimal)
    freq_per_year : number of coupon payments per year
    maturity_years: final maturity in years
    """
    t_yrs: List[float]
    amounts: List[float]
    face_value: float
    coupon_rate: float
    freq_per_year: int
    maturity_years: float

    @classmethod
    def from_arrays(
        cls,
        t_yrs: Sequence[float],
        amounts: Sequence[float],
        face_value: float,
        coupon_rate: float,
        freq_per_year: int,
        maturity_years: float,
    ) -> "BondCashflowSchedule":
        return cls(
            list(t_yrs),
            list(amounts),
            float(face_value),
            float(coupon_rate),
            int(freq_per_year),
            float(maturity_years),
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"t_yrs": self.t_yrs, "amount": self.amounts})


# -------------------------------------------------------------------
# Schedule builder
# -------------------------------------------------------------------


def build_level_coupon_schedule(
    maturity_years: float,
    coupon_rate: float,
    freq_per_year: int = 2,
    dt_years: Optional[float] = None,
    face_value: float = 100.0,
) -> BondCashflowSchedule:
    """
    Build a simple level-coupon bond schedule.

    Args:
        maturity_years: final maturity in years.
        coupon_rate:    annual coupon rate (decimal, e.g. 0.04).
        freq_per_year:  number of coupon payments per year (2 = semi-annual).
        dt_years:       OPTIONAL time step (years). If provided, we infer
                        freq_per_year = round(1 / dt_years).
        face_value:     principal repaid at maturity (default 100.0).

    Returns:
        BondCashflowSchedule with t_yrs, amounts, and face_value.
    """
    # --- interpret dt_years if provided ---
    if dt_years is not None:
        if dt_years <= 0:
            raise ValueError("dt_years must be positive.")
        freq = int(round(1.0 / dt_years))
        if freq <= 0:
            freq = 1
        freq_per_year = freq

    face = float(face_value)

    # Time between coupons
    dt = 1.0 / float(freq_per_year)

    # Coupon per period (level coupon)
    coupon = face * float(coupon_rate) / float(freq_per_year)

    # Build times and cashflows
    n_periods = int(round(maturity_years * freq_per_year))
    if n_periods <= 0:
        raise ValueError("maturity_years * freq_per_year must be > 0")

    t_yrs: list[float] = []
    amounts: list[float] = []

    for k in range(1, n_periods + 1):
        t = k * dt
        t_yrs.append(t)

        cf = coupon
        if k == n_periods:
            cf += face  # add principal at maturity
        amounts.append(cf)

    return BondCashflowSchedule(
        t_yrs=t_yrs,
        amounts=amounts,
        face_value=face,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        maturity_years=maturity_years,
    )


# -------------------------------------------------------------------
# Pricing from state-price tree
# -------------------------------------------------------------------


def _discount_factors_from_state_tree(
    state_tree_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Arrow–Debreu prices at each time step into a single
    discount factor P(0, t):

        DF(t_n) = sum_j state_price(n, j)

    Input:
      state_tree_df: must include columns 't_yrs' and 'state_price'

    Returns:
      DataFrame with columns: t_yrs, df
    """
    if "state_price" not in state_tree_df.columns:
        raise ValueError("state_tree_df must contain 'state_price' column.")
    if "t_yrs" not in state_tree_df.columns:
        raise ValueError("state_tree_df must contain 't_yrs' column.")

    df = state_tree_df.copy()
    df = df.groupby("t_yrs", as_index=False)["state_price"].sum()
    df.rename(columns={"state_price": "df"}, inplace=True)
    df.sort_values("t_yrs", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def price_cashflows_from_state_tree(
    state_tree_df: pd.DataFrame,
    schedule: BondCashflowSchedule,
    time_tolerance: float = 1e-6,
) -> float:
    """
    Price a set of cashflows using the HW state-price tree.

    Conceptually:

        Price = sum_k CF_k * DF(0, t_k)

    where DF(0, t) is reconstructed from the Arrow–Debreu state prices:

        DF(0, t_n) = sum_j A(n,j)

    Implementation details:
      - We first aggregate state prices into DF(t) per t_yrs.
      - For each cashflow time t_k, we find the *nearest* tree tenor t_n
        and use DF(t_n), with an optional sanity check that the gap
        is not too large (time_tolerance).
    """
    df_curve = _discount_factors_from_state_tree(state_tree_df)

    tree_times = df_curve["t_yrs"].to_numpy(dtype=float)
    tree_df = df_curve["df"].to_numpy(dtype=float)

    if tree_times.size == 0:
        raise ValueError("No tenors in state-price DF curve.")

    price = 0.0
    for t_k, cf in zip(schedule.t_yrs, schedule.amounts):
        # Find nearest tenor in tree
        idx = int(np.argmin(np.abs(tree_times - t_k)))
        t_near = tree_times[idx]
        df_near = tree_df[idx]

        if abs(t_near - t_k) > time_tolerance:
            print(
                f"[WARN] Cashflow at t={t_k:.6f} mapped to nearest lattice tenor "
                f"t={t_near:.6f}. Consider aligning step_years / maturity."
            )

        price += cf * df_near

    return float(price)


# -------------------------------------------------------------------
# Pricing directly from dense zero curve (for cross-checks)
# -------------------------------------------------------------------


def price_cashflows_from_dense_zero(
    dense_df: pd.DataFrame,
    schedule: BondCashflowSchedule,
) -> float:
    """
    Price cashflows using the dense zero curve *without* the lattice,
    purely as a deterministic discounting benchmark.

    We assume zero curve is given as:

        dense_df: columns 'tenor_yrs', 'rate_dec' (zero yield in decimal)

    We use continuous compounding (consistent with typical HW builds):

        DF(0, t) = exp(-z(t) * t)

    and interpolate z(t) with PCHIP to avoid weird shape artifacts.
    """
    if dense_df.empty:
        raise ValueError("dense_df is empty.")
    if "tenor_yrs" not in dense_df.columns or "rate_dec" not in dense_df.columns:
        raise ValueError("dense_df must contain 'tenor_yrs' and 'rate_dec' columns.")

    dense = dense_df.copy()
    dense["tenor_yrs"] = dense["tenor_yrs"].astype(float)
    dense.sort_values("tenor_yrs", inplace=True)
    dense.reset_index(drop=True, inplace=True)

    from scipy.interpolate import PchipInterpolator

    tenors = dense["tenor_yrs"].to_numpy(dtype=float)
    rates = dense["rate_dec"].to_numpy(dtype=float)

    interpolator = PchipInterpolator(tenors, rates, extrapolate=True)

    price = 0.0
    for t_k, cf in zip(schedule.t_yrs, schedule.amounts):
        z_k = float(interpolator(t_k))
        df_k = np.exp(-z_k * t_k)
        price += cf * df_k

    return float(price)


# -------------------------------------------------------------------
# Thin wrapper for tests: price via HW state-price tree
# -------------------------------------------------------------------


def price_level_coupon_bond_hw(
    schedule: BondCashflowSchedule,
    state_prices,
    time_tolerance: float = 1e-6,
) -> float:
    """
    Thin wrapper used in tests:

      - 'schedule' is a BondCashflowSchedule
      - 'state_prices' can be either:
          * a pandas DataFrame with columns ['t_yrs', 'state_price'], OR
          * a list-of-lists of Arrow–Debreu prices per time step
            (levels 0..N), where level n has n+1 state prices.

    We convert whatever we get into a DataFrame and then delegate to
    price_cashflows_from_state_tree.
    """
    # Case 1: already a DataFrame with the right columns
    if isinstance(state_prices, pd.DataFrame):
        df = state_prices
        # if it already has 't_yrs' and 'state_price', just use it
        if {"t_yrs", "state_price"}.issubset(df.columns):
            return price_cashflows_from_state_tree(
                state_tree_df=df,
                schedule=schedule,
                time_tolerance=time_tolerance,
            )
        # Otherwise try to coerce
        elif "time" in df.columns and "price" in df.columns:
            df2 = df.rename(columns={"time": "t_yrs", "price": "state_price"})[
                ["t_yrs", "state_price"]
            ]
            return price_cashflows_from_state_tree(
                state_tree_df=df2,
                schedule=schedule,
                time_tolerance=time_tolerance,
            )
        else:
            raise ValueError(
                "state_prices DataFrame must contain either "
                "['t_yrs', 'state_price'] or ['time', 'price']."
            )

    # Case 2: list-like input (what tests are currently using)
    # Expected shape: levels[0..N], each level is a list of state prices
    if isinstance(state_prices, (list, tuple)):
        # If it's a list of dicts, try DataFrame directly
        if state_prices and isinstance(state_prices[0], dict):
            df = pd.DataFrame(state_prices)
            if {"t_yrs", "state_price"}.issubset(df.columns):
                return price_cashflows_from_state_tree(
                    state_tree_df=df,
                    schedule=schedule,
                    time_tolerance=time_tolerance,
                )
            else:
                raise ValueError(
                    "List-of-dicts state_prices must have keys 't_yrs' and 'state_price'."
                )

        # Otherwise we assume list-of-lists: levels of the tree
        levels = state_prices
        n_levels = len(levels) - 1  # level 0..N => N time steps
        if n_levels <= 0:
            raise ValueError("state_prices list must contain at least 2 levels.")

        # Use the bond's maturity to infer dt
        maturity_years = float(schedule.maturity_years)
        dt = maturity_years / float(n_levels)

        records = []
        for level_idx, row in enumerate(levels):
            t = level_idx * dt  # time at this level
            # row is expected to be a sequence of state prices at this level
            for j, sp in enumerate(row):
                records.append(
                    {
                        "t_yrs": float(t),
                        "state_price": float(sp),
                        "level": int(level_idx),
                        "state_index": int(j),
                    }
                )

        df_states = pd.DataFrame.from_records(records)

        return price_cashflows_from_state_tree(
            state_tree_df=df_states,
            schedule=schedule,
            time_tolerance=time_tolerance,
        )

    # Fallback: unsupported type
    raise TypeError(
        f"Unsupported state_prices type: {type(state_prices)}. "
        "Expected DataFrame or list-of-lists."
    )
