# src/muni_core/pricing/hw_bond_pricer.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from typing import Sequence, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from muni_core.config.loader import AppConfig
from muni_core.curves.history import (
    build_dense_zero_curve_for_date,
    build_hw_theta_from_dense,
)
from muni_core.curves.short_rate_lattice import (
    build_short_rate_path_from_hw,
    build_binomial_lattice_from_hw,
    build_state_price_tree_from_lattice,
)


# -------------------------------------------------------------------
# Core cashflow representation
# -------------------------------------------------------------------

@dataclass
class BondCashflowSchedule:
    """
    Simple representation of a bond's projected cashflows
    relative to the as-of date (in *years*).

    t_yrs: array of payment times in years (e.g., 0.5, 1.0, ..., 10.0)
    amounts: array of cashflows at those times (coupon + principal)
    """
    t_yrs: np.ndarray
    amounts: np.ndarray

    def __post_init__(self) -> None:
        self.t_yrs = np.asarray(self.t_yrs, dtype=float)
        self.amounts = np.asarray(self.amounts, dtype=float)
        if self.t_yrs.shape != self.amounts.shape:
            raise ValueError(
                f"t_yrs and amounts must have same shape, got "
                f"{self.t_yrs.shape} vs {self.amounts.shape}"
            )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "BondCashflowSchedule":
        """
        Expect columns: 't_yrs' and 'amount'
        """
        if "t_yrs" not in df.columns or "amount" not in df.columns:
            raise ValueError("DataFrame must contain 't_yrs' and 'amount' columns.")
        return cls(
            t_yrs=df["t_yrs"].to_numpy(dtype=float),
            amounts=df["amount"].to_numpy(dtype=float),
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"t_yrs": self.t_yrs, "amount": self.amounts})


def build_level_coupon_schedule(
    maturity_years: float,
    coupon_rate: float,
    freq_per_year: int = 2,
    face: float = 100.0,
) -> BondCashflowSchedule:
    """
    Build a standard *level coupon* schedule:

      - maturity_years: final maturity (time from as-of in years)
      - coupon_rate   : annual coupon rate (e.g. 0.04 for 4%)
      - freq_per_year : coupon frequency (2 = semiannual)
      - face          : principal amount

    This ignores stub periods and assumes exact integer number of periods:

      n_periods = round(maturity_years * freq_per_year)

      t_k = k / freq_per_year, k=1..n_periods
      CF_k = coupon_rate * face / freq_per_year
      CF_last += face
    """
    if maturity_years <= 0:
        raise ValueError("maturity_years must be > 0")
    if freq_per_year <= 0:
        raise ValueError("freq_per_year must be > 0")

    n_periods = int(round(maturity_years * freq_per_year))
    if n_periods <= 0:
        raise ValueError(
            f"Computed n_periods={n_periods} from maturity_years={maturity_years}, "
            f"freq_per_year={freq_per_year}"
        )

    t_yrs = np.arange(1, n_periods + 1, dtype=float) / float(freq_per_year)
    coupon = face * coupon_rate / float(freq_per_year)
    amounts = np.full(n_periods, coupon, dtype=float)
    amounts[-1] += face

    return BondCashflowSchedule(t_yrs=t_yrs, amounts=amounts)


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

    Parameters
    ----------
    state_tree_df : DataFrame
        Columns include at least: 't_yrs', 'state_price'.
    schedule : BondCashflowSchedule
        Cashflow times and amounts (in years from as-of).
    time_tolerance : float
        If the nearest tree tenor differs from t_k by more than this,
        we *warn* via print; pricing still proceeds but it’s a flag
        for misalignment.

    Returns
    -------
    float
        Present value of the cashflow schedule.
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

    We use continuous compounding consistent with the HW build:

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

    # piecewise monotone interpolation of zero rates
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
# High-level convenience: build HW lattice + price coupon bond
# -------------------------------------------------------------------

def price_level_coupon_bond_hw(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    maturity_years: float,
    coupon_rate: float,
    freq_per_year: int = 2,
    face: float = 100.0,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
) -> float:
    """
    End-to-end helper for a *toy* level-coupon bond:

      1. Choose as-of date (CURVE_ASOF_DATE or max(history_df.date)).
      2. Build dense zero curve for (asof, curve_key).
      3. Build HW theta grid from dense.
      4. Build short-rate path + binomial lattice.
      5. Build state-price tree matching the dense zero DF.
      6. Build level coupon schedule from 0..maturity_years.
      7. Price with state prices.

    This does *not* yet know about real bonds, call schedules,
    or actual calendar dates. It’s meant as a “lab-core” pricing
    building block and sanity check that the lattice is consistent.

    Parameters
    ----------
    history_df : DataFrame
        Long-form historical curve table built by build_historical_curves.
    app_cfg : AppConfig
        Configuration object with CURVE_ASOF_DATE and HW params.
    maturity_years : float
        Time from as-of to final maturity (years).
    coupon_rate : float
        Annual coupon rate (e.g. 0.04 for 4%).
    freq_per_year : int
        Coupon frequency (2 = semiannual).
    face : float
        Face value.
    curve_key : str
        Which curve in history_df to use (default AAA_MUNI_SPOT).
    step_years : float
        Lattice time step (0.5 = semiannual).
    q : float
        Baseline up probability in HW tree (shape parameter, not DF).

    Returns
    -------
    float
        HW-lattice price of the level coupon bond.
    """
    curves_cfg = app_cfg.curves

    # --- As-of date ---
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        history_df_local = history_df.copy()
        history_df_local["date"] = pd.to_datetime(history_df_local["date"]).dt.date
        asof = history_df_local["date"].max()

    # --- HW parameters from Controls (or defaults) ---
    a_raw: Optional[str] = None
    sigma_raw: Optional[str] = None
    try:
        if hasattr(app_cfg, "get_control_value"):
            a_raw = app_cfg.get_control_value("HW_A", default=None)
            sigma_raw = app_cfg.get_control_value("HW_SIGMA_BASE", default=None)
    except Exception:
        a_raw = None
        sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01

    # --- Dense zero curve for as-of / curve_key ---
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # --- HW theta grid and short-rate machinery ---
    hw_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)
    _path_df = build_short_rate_path_from_hw(hw_df)
    lattice_df = build_binomial_lattice_from_hw(hw_df, sigma=sigma, dt=step_years)
    state_tree_df = build_state_price_tree_from_lattice(
        lattice_df=lattice_df,
        dense_df=dense_df,
        dt=step_years,
        q=q,
    )

    # --- Build coupon schedule and price ---
    schedule = build_level_coupon_schedule(
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        face=face,
    )

    price_hw = price_cashflows_from_state_tree(state_tree_df, schedule)

    return price_hw
