# src/muni_core/pricing/hw_bond_pricer.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


from muni_core.config.loader import AppConfig
from muni_core.curves.history import (
    build_dense_zero_curve_for_date,
    build_hw_theta_from_dense,
)
from muni_core.curves.short_rate_lattice import (
    ShortRateLattice,
    build_binomial_lattice_from_hw,
    build_state_price_tree_from_lattice,
)
from muni_core.model import Bond
from muni_core.calls import build_bermudan_call_times_3yr_window
from datetime import date as date_type
from muni_core.calls.call_schedule import build_bermudan_call_times_3yr_window


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


def price_level_coupon_bond_hw_from_state_tree(
    schedule: BondCashflowSchedule,
    state_prices,
    time_tolerance: float = 1e-6,
) -> float:
    """
    Test-friendly wrapper to price a level-coupon bond from a
    Hull–White state-price tree.

    Parameters
    ----------
    schedule : BondCashflowSchedule
        Cashflow times and amounts.
    state_prices :
        Either:
          * a pandas DataFrame with columns ['t_yrs', 'state_price'], OR
          * a list-of-lists of Arrow–Debreu prices per time step
            (levels[0..N], where level n has n+1 state prices).

    time_tolerance : float
        Passed through to price_cashflows_from_state_tree to warn
        if cashflow times do not align well with lattice tenors.

    Returns
    -------
    float
        Present value of the bond.
    """
    # ---- Case 1: already a DataFrame ----
    if isinstance(state_prices, pd.DataFrame):
        df = state_prices

        # Ideal case: already has the right columns.
        if {"t_yrs", "state_price"}.issubset(df.columns):
            return price_cashflows_from_state_tree(
                state_tree_df=df,
                schedule=schedule,
                time_tolerance=time_tolerance,
            )

        # Fallback rename case: ['time', 'price'] -> ['t_yrs', 'state_price']
        if {"time", "price"}.issubset(df.columns):
            df2 = df.rename(columns={"time": "t_yrs", "price": "state_price"})[
                ["t_yrs", "state_price"]
            ]
            return price_cashflows_from_state_tree(
                state_tree_df=df2,
                schedule=schedule,
                time_tolerance=time_tolerance,
            )

        raise ValueError(
            "state_prices DataFrame must contain either "
            "['t_yrs', 'state_price'] or ['time', 'price'] columns."
        )

    # ---- Case 2: list-like (what our HW lattice builder returns) ----
    if isinstance(state_prices, (list, tuple)):
        # If it's list-of-dicts, try to DataFrame it directly
        if state_prices and isinstance(state_prices[0], dict):
            df = pd.DataFrame(state_prices)
            if {"t_yrs", "state_price"}.issubset(df.columns):
                return price_cashflows_from_state_tree(
                    state_tree_df=df,
                    schedule=schedule,
                    time_tolerance=time_tolerance,
                )
            raise ValueError(
                "List-of-dicts state_prices must have keys 't_yrs' and 'state_price'."
            )

        # Otherwise: assume list-of-lists per level
        levels = state_prices
        n_levels = len(levels) - 1  # level 0..N => N time steps
        if n_levels <= 0:
            raise ValueError("state_prices list must contain at least 2 levels.")

        # Use bond maturity to infer the time step dt
        maturity_years = float(schedule.maturity_years)
        dt = maturity_years / float(n_levels)

        records = []
        for level_idx, row in enumerate(levels):
            t = level_idx * dt  # time at this level
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

    # ---- Fallback: unsupported ----
    raise TypeError(
        f"Unsupported state_prices type: {type(state_prices)}. "
        "Expected DataFrame or list-of-lists."
    )


# -------------------------------------------------------------------
# Thin wrapper for tests: price via HW state-price tree
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
    time_tolerance: float = 1e-6,
) -> float:
    """
    High-level Hull–White bond pricer used by PATH 3 and
    price_bullet_bond_hw_from_config.

    1) Pick as-of date from CURVE_ASOF_DATE or max(history_df.date).
    2) Build dense zero curve for (asof, curve_key).
    3) Build HW theta grid from dense curve.
    4) Build HW binomial short-rate lattice.
    5) Build state-price tree to match dense zero DFs.
    6) Build level coupon schedule from 0..maturity_years.
    7) Price via state prices.
    """
    curves_cfg = app_cfg.curves

    # --- as-of date ---
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    # --- HW parameters from Controls (or defaults) ---
    a_raw = None
    sigma_raw = None
    if hasattr(app_cfg, "get_control_value"):
        try:
            a_raw = app_cfg.get_control_value("HW_A", default=None)
            sigma_raw = app_cfg.get_control_value("HW_SIGMA_BASE", default=None)
        except Exception:
            a_raw = None
            sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01

    # --- dense zero curve for as-of / curve_key ---
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # --- HW theta grid + lattice + state-price tree ---
    hw_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)

    lattice = build_binomial_lattice_from_hw(
        hw_df,
        a=a,
        sigma=sigma,
        dt_years=step_years,
    )

    # NOTE: this returns a list-of-lists of state prices
    state_prices = build_state_price_tree_from_lattice(lattice)

    # --- build level coupon schedule ---
    schedule = build_level_coupon_schedule(
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        dt_years=None,
        face_value=face,
    )

    # --- final price via state-price wrapper that handles list-of-lists ---
    return price_level_coupon_bond_hw_from_state_tree(
        schedule=schedule,
        state_prices=state_prices,
        time_tolerance=time_tolerance,
    )



def price_bullet_bond_hw_from_config(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    *,
    coupon_rate: float,
    maturity_date: date_type,
    asof_date: Optional[date_type] = None,
    freq_per_year: int = 2,
    face: float = 100.0,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> float:
    """
    Convenience wrapper: price a *bullet* level-coupon bond using HW,
    driven by actual calendar dates.

    - Uses CURVE_ASOF_DATE from AppConfig.controls if asof_date is None,
      otherwise uses the provided asof_date.
    - Converts (asof -> maturity_date) into maturity_years.
    - Delegates to price_level_coupon_bond_hw(...) under the hood.

    Parameters
    ----------
    history_df : DataFrame
        Long-form curves table (output of build_historical_curves).
    app_cfg : AppConfig
        Global configuration (curves + controls).
    coupon_rate : float
        Annual coupon rate (e.g. 0.04 for 4%).
    maturity_date : date
        Calendar maturity date of the bond.
    asof_date : Optional[date]
        Pricing date. If None, uses CURVE_ASOF_DATE or max(history_df.date).
    freq_per_year : int
        Coupon frequency (2 = semiannual).
    face : float
        Face value.
    curve_key : str
        Which curve to use (default AAA_MUNI_SPOT).
    step_years : float
        Lattice time step in years (0.5 = semiannual).
    q : float
        Up-move probability parameter in the HW lattice.
    time_tolerance : float
        Tolerance for matching cashflow times to lattice tenors
        (passed through to the underlying pricer).

    Returns
    -------
    float
        HW price of the dated level-coupon bond.
    """
    # --- Determine as-of date ---
    curves_cfg = app_cfg.curves

    if asof_date is not None:
        asof = asof_date
    else:
        if curves_cfg.curve_asof_date:
            asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
        else:
            history_df_local = history_df.copy()
            history_df_local["date"] = pd.to_datetime(history_df_local["date"]).dt.date
            asof = history_df_local["date"].max()

    if maturity_date <= asof:
        raise ValueError(
            f"maturity_date {maturity_date} must be after as-of date {asof}"
        )

    # Convert calendar dates to year fraction (simple ACT/365.25 style)
    delta_days = (maturity_date - asof).days
    maturity_years = float(delta_days) / 365.25

    # Delegate to the core HW pricer
    price = price_level_coupon_bond_hw(
        history_df=history_df,
        app_cfg=app_cfg,
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        face=face,
        curve_key=curve_key,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
    )

    return float(price)

def price_callable_bond_hw_from_bond(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    oas_bp: float,
    *,
    bond: Bond,
    freq_per_year: int = 2,
    face: float = 100.0,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> float:
    """
    High-level Hull–White pricer for a *single Bond* using the
    Bermudan call schedule helper.
    """
    curves_cfg = app_cfg.curves

    # --- as-of date, same logic as other helpers ---
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    if bond.maturity_date is None:
        raise ValueError("Bond.maturity_date is required for HW pricing.")

    if bond.maturity_date <= asof:
        raise ValueError(
            f"Bond.maturity_date {bond.maturity_date} must be after as-of date {asof}"
        )

    # --- dense zero curve for this as-of / curve_key ---
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # --- apply callable OAS as a parallel shift to the zero curve ---
    oas_dec = oas_bp / 10_000.0
    if oas_dec != 0.0:
        dense_df = dense_df.copy()
        if "rate_dec" not in dense_df.columns:
            raise ValueError("dense_df must contain 'rate_dec' column for OAS shift.")
        dense_df["rate_dec"] = dense_df["rate_dec"].astype(float) + oas_dec

    # --- HW parameters from Controls (or defaults), same as price_level_coupon_bond_hw ---
    a_raw = None
    sigma_raw = None
    if hasattr(app_cfg, "get_control_value"):
        try:
            a_raw = app_cfg.get_control_value("HW_A", default=None)
            sigma_raw = app_cfg.get_control_value("HW_SIGMA_BASE", default=None)
        except Exception:
            a_raw = None
            sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01

    # --- HW theta grid + lattice ---
    hw_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)

    lattice = build_binomial_lattice_from_hw(
        hw_df,
        a=a,
        sigma=sigma,
        dt_years=step_years,
    )

    # --- maturity in years (simple ACT/365.25 like other helpers) ---
    delta_days = (bond.maturity_date - asof).days
    maturity_years = float(delta_days) / 365.25

    # Bond coupon is stored as decimal (e.g. 0.04 for 4%).
    coupon_rate = float(bond.coupon or 0.0)

    # --- Bermudan call schedule (or none if no call) ---
    call_times_yrs: list[float]
    call_price = face

    if getattr(bond, "call_feature", None) is not None:
        call_dt = bond.call_feature.call_date
        if call_dt is not None and call_dt > asof:
            call_times_yrs = build_bermudan_call_times_3yr_window(
                asof_date=asof,
                first_call_date=call_dt,
                maturity_date=bond.maturity_date,
            )
            call_price = float(getattr(bond.call_feature, "call_price", face) or face)
        else:
            # call date in the past or missing -> treat as non-callable going forward
            call_times_yrs = []
    else:
        call_times_yrs = []

    # --- Delegate to lattice-based callable pricer ---
    price = price_callable_bond_from_lattice(
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

    return float(price)



# -------------------------------------------------------------------
# Callable bond pricing on a HW short-rate lattice
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Callable bond pricing on a HW short-rate lattice (low-level core)
# -------------------------------------------------------------------


def price_callable_bond_from_lattice(
    lattice,
    maturity_years: float,
    coupon_rate: float,
    freq_per_year: int = 2,
    call_times_yrs: Optional[Sequence[float]] = None,
    face: float = 100.0,
    call_price: float = 100.0,
    q: float = 0.5,
    time_tolerance: float = 1e-6,
) -> float:
    """
    Price a level-coupon bond on a Hull–White short-rate lattice
    with an **issuer** Bermudan call (investor is short the option).

    The lattice is:
        lattice.dt_years : time step (years)
        lattice.rates[t][j] : short rate at step t, node j

    We only use as many steps as needed to reach maturity_years; if the
    lattice is longer (e.g. 30Y) but the bond is 10Y, we truncate to
    the first N steps where N * dt ≈ maturity_years.

    Coupons:
        We assume coupons align with the lattice step size, i.e.
            dt ≈ 1 / freq_per_year
        and set coupon_per_period = face * coupon_rate * dt.
        This matches build_level_coupon_schedule when dt_years is used.

    Early-exercise rule (issuer advantage, investor worst case):
        At a call date:
            V_t(j) = min( continuation_value, call_price )

    so the option reduces the investor's value.
    """
    import math

    if call_times_yrs is None:
        call_times_yrs = []

    dt = float(lattice.dt_years)
    levels = lattice.rates
    n_steps_total = len(levels)
    if n_steps_total == 0:
        raise ValueError("ShortRateLattice has no levels in price_callable_bond_from_lattice.")

    # --- determine how many steps we actually need for maturity ---
    if dt <= 0.0:
        raise ValueError(f"Invalid lattice.dt_years={dt}")

    n_steps_use = int(round(maturity_years / dt))
    if n_steps_use < 1:
        raise ValueError(
            f"maturity_years={maturity_years} with dt={dt} implies <1 step; "
            "cannot build callable pricer."
        )
    if n_steps_use > n_steps_total:
        raise ValueError(
            f"Lattice only has horizon T={n_steps_total * dt:.6f}, "
            f"but maturity_years={maturity_years:.6f} requires more steps."
        )

    T_horizon = n_steps_use * dt
    if abs(T_horizon - maturity_years) > time_tolerance:
        logger.debug(
            "Effective horizon T=%.6f from n_steps=%s and dt=%.6f differs from maturity_years=%.6f.",
            T_horizon,
            n_steps_use,
            dt,
            maturity_years,
        )

    levels_use = levels[:n_steps_use]

    # Coupons: consistent with dt (dt ≈ 1/freq_per_year)
    coupon_per_period = face * float(coupon_rate) * dt

    # --- map call times (in years) -> step indices within [0, n_steps_use-1] ---
    call_steps: set[int] = set()
    for t_call in call_times_yrs:
        if t_call <= 0:
            continue
        step_f = t_call / dt
        k = int(round(step_f))
        # interpret k as "around time t = k * dt"; we place the call
        # at step index k-1 (just before that time), if in range
        k_idx = k - 1
        if 0 <= k_idx < n_steps_use:
            call_steps.add(k_idx)

    # --- terminal payoffs at maturity (last used step) ---
    last_idx = n_steps_use - 1
    last_level = levels_use[last_idx]
    n_nodes_T = len(last_level)

    # At maturity: final coupon + principal
    V_next = [face + coupon_per_period] * n_nodes_T

    # --- backward induction over the truncated lattice ---
    for t in reversed(range(n_steps_use - 1)):  # t = last_idx-1, ..., 0
        rates_t = levels_use[t]
        n_nodes_t = len(rates_t)

        if len(V_next) != n_nodes_t + 1:
            raise ValueError(
                f"Dimension mismatch at step {t}: "
                f"have {len(V_next)} next states, expected {n_nodes_t + 1}."
            )

        V_curr = [0.0] * n_nodes_t

        for j in range(n_nodes_t):
            r_tj = float(rates_t[j])

            # risk-neutral expectation of next-step values
            continuation = (1.0 - q) * V_next[j] + q * V_next[j + 1]

            # discount back one step
            disc = math.exp(-r_tj * dt)
            value_t = continuation * disc

            # add coupon at this step (first coupon at t=dt)
            value_t += coupon_per_period

            V_curr[j] = value_t

        # issuer call: investor gets the WORSE of continuation or call_price
        if t in call_steps:
            V_curr = [min(v, call_price) for v in V_curr]

        V_next = V_curr

    # root node at t=0
    return float(V_next[0])


