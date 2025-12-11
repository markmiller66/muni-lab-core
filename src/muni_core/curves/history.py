# src/muni_core/curves/history.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type, datetime
from pathlib import Path
from typing import Optional, Literal, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from muni_core.config.loader import AppConfig
from .types import CurvePoint, ZeroCurve
from .zero_curve import make_zero_curve_from_pairs


CurveKey = Literal["AAA_MUNI_PAR", "AAA_MUNI_SPOT", "UST_PAR", "UST_SPOT"]


# ---------- History config ----------


@dataclass
class CurveHistoryConfig:
    """
    Simple wrapper around AppConfig.curves for convenience.
    """
    history_file: Path

    @classmethod
    def from_app_config(cls, app_cfg: AppConfig) -> "CurveHistoryConfig":
        if app_cfg.curves.history_file is None:
            raise ValueError("curves.history_file is not set in AppConfig/YAML.")
        return cls(history_file=app_cfg.curves.history_file)


# ---------- Helpers for output folders ----------


def _ensure_output_subdirs(base: Path) -> Dict[str, Path]:
    """
    Given a base directory (typically app_cfg.dated_output_root),
    ensure the standard curve subdirectories exist:

        spot/    : par & spot curves + spreads
        dense/   : dense zero curves (semi-annual)
        forward/ : forward matrices

    Returns a dict {"spot": spot_dir, "dense": dense_dir, "forward": fwd_dir}.
    """
    spot_dir = base / "spot"
    dense_dir = base / "dense"
    fwd_dir = base / "forward"

    for d in (spot_dir, dense_dir, fwd_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {"spot": spot_dir, "dense": dense_dir, "forward": fwd_dir}


# ---------- Par → spot bootstrap ----------


def bootstrap_spot_from_par(
    par_df: pd.DataFrame,
    curve_key_in: str,
    curve_key_out: str,
) -> pd.DataFrame:
    """
    Very simple par -> spot bootstrap assuming ANNUAL coupons and
    integer-year tenors.

    par_df must have columns: date, curve_key, tenor_yrs, rate_dec (par yield in DECIMAL).

    For each (date, curve_key_in), we:
      - sort by tenor_yrs
      - bootstrap discount factors DF_t using par pricing at par=1
      - convert DF_t to spot zero rate via: DF_t = (1 + z_t)^(-t)

    Returns a long DataFrame with columns: date, curve_key, tenor_yrs, rate_dec,
    where rate_dec is the SPOT zero yield (decimal).
    """
    records: list[dict] = []

    # Filter to the specific curve_key_in
    df = par_df[par_df["curve_key"] == curve_key_in].copy()

    for dt, group in df.groupby("date"):
        group = group.sort_values("tenor_yrs")

        # Discount factors indexed by integer year t
        dfs: dict[int, float] = {}

        for _, row in group.iterrows():
            tenor = float(row["tenor_yrs"])
            y = float(row["rate_dec"])  # par yield in decimal

            # assume integer-year tenor
            t = int(round(tenor))
            if t <= 0:
                continue

            if t == 1:
                # 1 = (1 + y) * DF_1
                DF_t = 1.0 / (1.0 + y)
            else:
                # Annual coupons:
                # 1 = y * sum_{i=1}^{t-1} DF_i + (1 + y) * DF_t
                coupon_sum = 0.0
                for i in range(1, t):
                    if i not in dfs:
                        coupon_sum = None
                        break
                    coupon_sum += dfs[i]

                if coupon_sum is None:
                    continue

                DF_t = (1.0 - y * coupon_sum) / (1.0 + y)

            if DF_t <= 0.0:
                # ignore pathological cases
                continue

            dfs[t] = DF_t

            # Convert DF_t to annual spot z_t: DF_t = (1 + z_t)^(-t)
            spot = DF_t ** (-1.0 / t) - 1.0

            records.append(
                {
                    "date": dt,
                    "curve_key": curve_key_out,
                    "tenor_yrs": float(t),
                    "rate_dec": spot,
                }
            )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["date", "tenor_yrs"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


# ---------- Build historical curves (par + spot) ----------


def build_historical_curves(
    treas_df: pd.DataFrame,
    muni_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame],
    app_cfg: AppConfig,
) -> pd.DataFrame:
    """
    Build a long-form historical curves table from raw Tradeweb/Fed data.

    For now, this version:
      - Treats the muni CSV columns as par yields for tenors 1Y, 2Y, ..., N
      - Treats the treasury CSV as spot yields from Svensson columns (SVENYxx)
      - Produces a long DataFrame with columns:
            date, curve_key, tenor_yrs, rate_dec
    """
    ch_cfg = CurveHistoryConfig.from_app_config(app_cfg)

    # --- Muni side ---
    muni = muni_df.copy()

    # Normalize column names to numeric tenor in years, e.g. "1 yr" -> 1.0
    tenor_map = {}
    for col in muni.columns:
        col_str = str(col).strip().lower()
        try:
            n = int(col_str.split()[0])
            tenor_map[col] = float(n)
        except Exception:
            continue

    muni_long_records = []
    for dt, row in muni.iterrows():
        for col, tenor_yrs in tenor_map.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            # Assume val is in percent; convert to decimal
            rate_dec = float(val) / 100.0
            muni_long_records.append(
                {
                    "date": dt.normalize(),
                    "curve_key": "AAA_MUNI_PAR",
                    "tenor_yrs": tenor_yrs,
                    "rate_dec": rate_dec,
                }
            )

    muni_long = pd.DataFrame.from_records(muni_long_records)

    # --- Muni SPOT (bootstrapped from AAA_MUNI_PAR) ---
    muni_spot_long = bootstrap_spot_from_par(
        muni_long,
        curve_key_in="AAA_MUNI_PAR",
        curve_key_out="AAA_MUNI_SPOT",
    )

    # --- Treasury side ---
    treas = treas_df.copy()

    treas_tenor_map = {}
    for col in treas.columns:
        col_str = str(col).upper()
        if col_str.startswith("SVENY"):
            # SVENY01 -> 1, SVENY10 -> 10, etc.
            try:
                n = int(col_str.replace("SVENY", ""))
                treas_tenor_map[col] = float(n)
            except Exception:
                continue

    treas_long_records = []
    for dt, row in treas.iterrows():
        for col, tenor_yrs in treas_tenor_map.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            rate_dec = float(val) / 100.0  # SVENYxx are zero yields in percent
            treas_long_records.append(
                {
                    "date": dt.normalize(),
                    "curve_key": "UST_SPOT",
                    "tenor_yrs": tenor_yrs,
                    "rate_dec": rate_dec,
                }
            )

    treas_long = pd.DataFrame.from_records(treas_long_records)

    # --- Combine ---
    frames = [muni_long, muni_spot_long, treas_long]
    all_long = pd.concat(frames, ignore_index=True)

    # Sort for sanity
    all_long.sort_values(["date", "curve_key", "tenor_yrs"], inplace=True)
    all_long.reset_index(drop=True, inplace=True)

    # Save to history_file
    ch_cfg.history_file.parent.mkdir(parents=True, exist_ok=True)
    if ch_cfg.history_file.suffix.lower() == ".parquet":
        all_long.to_parquet(ch_cfg.history_file)
    else:
        all_long.to_csv(ch_cfg.history_file, index=False)

    return all_long


# ---------- Pivot helpers & forward curves ----------


def _pivot_curve(df: pd.DataFrame, value_col: str = "rate_dec") -> pd.DataFrame:
    """
    Convert long format into wide format:

        date | 0.5 | 1.0 | ... | 30.0

    - Removes timestamps (keeps YYYY-MM-DD)
    - Sorts with most recent date *on top*
    """
    if df.empty:
        return pd.DataFrame(columns=["date"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    pivot = df.pivot(index="date", columns="tenor_yrs", values=value_col)
    pivot = pivot.sort_index(ascending=False)  # latest date first
    pivot = pivot.reset_index()
    pivot.columns.name = None

    return pivot


def _compute_forward_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a long-format zero curve:

        date, tenor_yrs, rate_dec

    compute annual forward rates between adjacent tenors:

        f_{i-1,i} = (DF(T_{i-1}) / DF(T_i))^(1/(T_i - T_{i-1})) - 1

    Returns long-format:

        date, tenor_yrs, fwd_rate_dec

    where tenor_yrs is the END tenor (T_i) of the forward interval.
    """
    records: list[dict] = []

    if df.empty:
        return pd.DataFrame(columns=["date", "tenor_yrs", "fwd_rate_dec"])

    for dt, group in df.groupby("date"):
        g = group.sort_values("tenor_yrs")
        tenors = g["tenor_yrs"].to_numpy(dtype=float)
        z = g["rate_dec"].to_numpy(dtype=float)

        DF = (1.0 + z) ** (-tenors)

        for i in range(1, len(tenors)):
            t0 = tenors[i - 1]
            t1 = tenors[i]
            if t1 <= t0:
                continue

            df0 = DF[i - 1]
            df1 = DF[i]
            if df1 <= 0.0 or df0 <= 0.0:
                continue

            dt_years = t1 - t0
            fwd = (df0 / df1) ** (1.0 / dt_years) - 1.0

            records.append(
                {
                    "date": dt,
                    "tenor_yrs": t1,
                    "fwd_rate_dec": fwd,
                }
            )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["date", "tenor_yrs"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def _compute_horizon_forward_curve(
    df: pd.DataFrame,
    horizon_years: float,
) -> pd.DataFrame:
    """
    Compute a *fixed-horizon* forward rate f(T -> T + horizon_years) for each tenor T
    where data exist at both T and T + horizon_years.

    Result columns:
        date, tenor_yrs, fwd_rate_dec
    where tenor_yrs = T (the start of the horizon).
    """
    records: list[dict] = []

    if df.empty:
        return pd.DataFrame(columns=["date", "tenor_yrs", "fwd_rate_dec"])

    for dt, group in df.groupby("date"):
        g = group.sort_values("tenor_yrs")
        tenors = g["tenor_yrs"].to_numpy(dtype=float)
        z = g["rate_dec"].to_numpy(dtype=float)

        DF = (1.0 + z) ** (-tenors)
        df_map = {float(t): float(d) for t, d in zip(tenors, DF)}

        for t_start in tenors:
            t_end = t_start + horizon_years
            if t_end not in df_map:
                continue

            df1 = df_map[t_start]
            df2 = df_map[t_end]
            if df1 <= 0.0 or df2 <= 0.0:
                continue

            fwd = (df1 / df2) ** (1.0 / horizon_years) - 1.0
            records.append(
                {
                    "date": dt,
                    "tenor_yrs": float(t_start),
                    "fwd_rate_dec": float(fwd),
                }
            )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["date", "tenor_yrs"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


# ---------- Spot curves, spreads, and simple forwards ----------


def export_spot_curves_and_spreads(history_df: pd.DataFrame, app_cfg: AppConfig) -> None:
    """
    Export AAA_MUNI_SPOT, UST_SPOT, AAA_MUNI_PAR, their spreads,
    and annual forward (short-rate) curves into date-partitioned folders:

        <repo_root>/output/curves/YYYY-MM-DD/spot/

    Parquet (engine-facing, long format):
        - aaa_muni_spot.parquet
        - ust_spot.parquet
        - aaa_muni_par.parquet
        - spot_spreads.parquet
        - aaa_muni_fwd.parquet
        - ust_fwd.parquet
        - aaa_muni_fwd5.parquet
        - ust_fwd5.parquet

    Excel (human-facing, wide format) in the same folder:
        - spot_curves_and_spreads_<timestamp>.xlsx
    """
    base = app_cfg.dated_output_root
    dirs = _ensure_output_subdirs(base)
    spot_dir = dirs["spot"]

    # --- Extract spot + par curves in long format ---
    muni_spot = history_df[history_df["curve_key"] == "AAA_MUNI_SPOT"].copy()
    ust_spot = history_df[history_df["curve_key"] == "UST_SPOT"].copy()
    muni_par = history_df[history_df["curve_key"] == "AAA_MUNI_PAR"].copy()

    muni_spot = muni_spot[["date", "tenor_yrs", "rate_dec"]]
    ust_spot = ust_spot[["date", "tenor_yrs", "rate_dec"]]
    muni_par = muni_par[["date", "tenor_yrs", "rate_dec"]]

    # --- Compute spot spreads: AAA_MUNI_SPOT - UST_SPOT ---
    merged = pd.merge(
        muni_spot,
        ust_spot,
        on=["date", "tenor_yrs"],
        suffixes=("_muni", "_ust"),
    )

    merged["spread_zero_yield"] = merged["rate_dec_muni"] - merged["rate_dec_ust"]
    merged["spread_bp"] = merged["spread_zero_yield"] * 10_000.0

    spreads_out = merged[
        [
            "date",
            "tenor_yrs",
            "rate_dec_muni",
            "rate_dec_ust",
            "spread_zero_yield",
            "spread_bp",
        ]
    ].copy()

    # --- Forward curves ---
    muni_fwd = _compute_forward_curve(muni_spot)
    ust_fwd = _compute_forward_curve(ust_spot)

    # 5Y horizon forwards (optional richer structure)
    muni_fwd5 = _compute_horizon_forward_curve(muni_spot, horizon_years=5.0)
    ust_fwd5 = _compute_horizon_forward_curve(ust_spot, horizon_years=5.0)

    # --- Parquet exports ---
    muni_spot.to_parquet(spot_dir / "aaa_muni_spot.parquet")
    ust_spot.to_parquet(spot_dir / "ust_spot.parquet")
    muni_par.to_parquet(spot_dir / "aaa_muni_par.parquet")
    spreads_out.to_parquet(spot_dir / "spot_spreads.parquet")
    muni_fwd.to_parquet(spot_dir / "aaa_muni_fwd.parquet")
    ust_fwd.to_parquet(spot_dir / "ust_fwd.parquet")
    muni_fwd5.to_parquet(spot_dir / "aaa_muni_fwd5.parquet")
    ust_fwd5.to_parquet(spot_dir / "ust_fwd5.parquet")

    # --- Wide-format Excel (human-readable) ---
    muni_spot_x = _pivot_curve(muni_spot, value_col="rate_dec")
    ust_spot_x = _pivot_curve(ust_spot, value_col="rate_dec")
    muni_par_x = _pivot_curve(muni_par, value_col="rate_dec")

    spread_x = _pivot_curve(
        spreads_out[["date", "tenor_yrs", "spread_bp"]],
        value_col="spread_bp",
    )

    muni_fwd_x = _pivot_curve(muni_fwd, value_col="fwd_rate_dec")
    ust_fwd_x = _pivot_curve(ust_fwd, value_col="fwd_rate_dec")
    muni_fwd5_x = _pivot_curve(muni_fwd5, value_col="fwd_rate_dec")
    ust_fwd5_x = _pivot_curve(ust_fwd5, value_col="fwd_rate_dec")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = spot_dir / f"spot_curves_and_spreads_{run_ts}.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        muni_spot_x.to_excel(writer, sheet_name="AAA_MUNI_SPOT", index=False)
        ust_spot_x.to_excel(writer, sheet_name="UST_SPOT", index=False)
        muni_par_x.to_excel(writer, sheet_name="AAA_MUNI_PAR", index=False)
        spread_x.to_excel(writer, sheet_name="SPREADS", index=False)
        muni_fwd_x.to_excel(writer, sheet_name="AAA_MUNI_FWD", index=False)
        ust_fwd_x.to_excel(writer, sheet_name="UST_FWD", index=False)
        muni_fwd5_x.to_excel(writer, sheet_name="AAA_MUNI_FWD5", index=False)
        ust_fwd5_x.to_excel(writer, sheet_name="UST_FWD5", index=False)

    print(f"[OK] Exported spot curves, spreads, and forwards to {xlsx_path}")


# ---------- Dense curve + full forward matrix (single as-of date) ----------


def resolve_curve_asof_date(app_cfg: AppConfig, history_df: pd.DataFrame) -> date_type:
    """
    Decide which curve date to use, in priority order:

      1) Controls sheet: CURVE_ASOF_DATE
      2) YAML: curves.curve_asof_date
      3) Latest date in history_df
    """
    asof: Optional[date_type] = None

    # 1) Controls sheet override
    try:
        ctrl_val = app_cfg.get_control_value("CURVE_ASOF_DATE", default=None)
    except Exception:
        ctrl_val = None

    if ctrl_val is not None and str(ctrl_val).strip() != "":
        try:
            asof = pd.to_datetime(ctrl_val).date()
        except Exception:
            asof = None

    # 2) YAML fallback
    if asof is None and app_cfg.curves.curve_asof_date:
        try:
            asof = pd.to_datetime(app_cfg.curves.curve_asof_date).date()
        except Exception:
            asof = None

    # 3) Last date in history_df
    if asof is None:
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    return asof


def build_dense_zero_curve_for_date(
    history_df: pd.DataFrame,
    curve_key: str,
    target_date: date_type,
    step_years: float = 0.5,
) -> pd.DataFrame:
    """
    From the long-form history_df (annual points):

        date, curve_key, tenor_yrs, rate_dec

    build a *dense* zero curve at semi-annual (or custom) spacing for a
    single (date, curve_key), using PCHIP interpolation.

    Returns a DataFrame:

        date, tenor_yrs, rate_dec

    where tenor_yrs = 0.5, 1.0, 1.5, ..., T_max
    """
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    mask = (df["date"] == target_date) & (df["curve_key"] == curve_key)
    sub = df.loc[mask].copy()

    if sub.empty:
        raise ValueError(
            f"No curve rows found in history for date={target_date} and curve_key={curve_key}"
        )

    sub.sort_values("tenor_yrs", inplace=True)

    tenors = sub["tenor_yrs"].to_numpy(dtype=float)
    rates = sub["rate_dec"].to_numpy(dtype=float)

    if len(tenors) < 2:
        raise ValueError(
            f"Need at least 2 tenor points to interpolate; got {len(tenors)} for {curve_key} on {target_date}"
        )

    interpolator = PchipInterpolator(tenors, rates, extrapolate=False)

    t_max = float(tenors.max())
    grid = np.arange(step_years, t_max + 1e-9, step_years)

    dense_rates = interpolator(grid)
    mask_valid = ~np.isnan(dense_rates)

    grid = grid[mask_valid]
    dense_rates = dense_rates[mask_valid]

    out = pd.DataFrame(
        {
            "date": target_date,
            "tenor_yrs": grid,
            "rate_dec": dense_rates,
        }
    )
    out.sort_values("tenor_yrs", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out

def build_dense_zero_curve_all_dates(
    history_df: pd.DataFrame,
    curve_key: str,
    step_years: float = 0.5,
) -> pd.DataFrame:
    """
    Build dense zero curves for *all* dates for a given curve_key.

    Returns long-format:

        date, tenor_yrs, rate_dec

    where each date has tenors: step_years, 2*step_years, ..., T_max(date).
    """
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    unique_dates = sorted(df[df["curve_key"] == curve_key]["date"].unique())
    records: list[pd.DataFrame] = []

    for dt in unique_dates:
        try:
            dense_dt = build_dense_zero_curve_for_date(
                history_df=history_df,
                curve_key=curve_key,
                target_date=dt,
                step_years=step_years,
            )
            records.append(dense_dt)
        except Exception:
            # skip dates that fail interpolation for any reason
            continue

    if not records:
        return pd.DataFrame(columns=["date", "tenor_yrs", "rate_dec"])

    out = pd.concat(records, ignore_index=True)
    out.sort_values(["date", "tenor_yrs"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def build_forward_matrix_from_dense(
    dense_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Given a *single-date* dense zero curve:

        date, tenor_yrs, rate_dec

    compute the full forward-rate matrix f(t1 -> t2) for all t1 < t2
    on that tenor grid.

    DF(t) = (1 + z(t))^(-t)
    f(t1 -> t2) = (DF(t1) / DF(t2))^(1/(t2 - t1)) - 1

    Returns a long-form DataFrame:

        date, start_yrs, end_yrs, fwd_rate_dec
    """
    if dense_df.empty:
        raise ValueError("dense_df is empty; cannot build forward matrix.")

    dense_df = dense_df.copy()
    dense_df["date"] = pd.to_datetime(dense_df["date"]).dt.date
    unique_dates = dense_df["date"].unique()
    if len(unique_dates) != 1:
        raise ValueError(
            f"build_forward_matrix_from_dense expects one date, got {unique_dates}"
        )
    dt = unique_dates[0]

    g = dense_df.sort_values("tenor_yrs")
    tenors = g["tenor_yrs"].to_numpy(dtype=float)
    z = g["rate_dec"].to_numpy(dtype=float)

    DF = (1.0 + z) ** (-tenors)
    df_map = {float(t): float(d) for t, d in zip(tenors, DF)}

    records: list[dict] = []

    for i, t1 in enumerate(tenors):
        df1 = df_map[t1]
        if df1 <= 0.0:
            continue
        for j in range(i + 1, len(tenors)):
            t2 = tenors[j]
            df2 = df_map[t2]
            if df2 <= 0.0:
                continue

            delta_t = t2 - t1
            if delta_t <= 0.0:
                continue

            fwd = (df1 / df2) ** (1.0 / delta_t) - 1.0

            records.append(
                {
                    "date": dt,
                    "start_yrs": float(t1),
                    "end_yrs": float(t2),
                    "fwd_rate_dec": float(fwd),
                }
            )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["start_yrs", "end_yrs"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out

def export_hw_theta_for_asof(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    For the current CURVE_ASOF_DATE, build a dense zero curve for `curve_key`
    (default AAA_MUNI_SPOT), derive the instantaneous forward curve f(0,t)
    and Hull–White theta(t), and export to:

        <repo_root>/output/curves/<CURVE_ASOF_DATE>/hw/

    Controls used (in MUNI_MASTER_BUCKET Controls):

        CURVE_ASOF_DATE   (required, YYYY-MM-DD)
        HW_A              (optional, mean reversion; default 0.10)
        HW_SIGMA_BASE     (optional, volatility; default 0.01)

    Outputs:

        - hw_theta_<curve_key>_<asof>.parquet
        - hw_theta_<curve_key>_<asof>.xlsx
    """
    curves_cfg = app_cfg.curves

    # Resolve as-of date: from CURVE_ASOF_DATE in YAML or Controls
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        # fall back to max date in history
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    # HW parameters from Controls, with safe defaults
    a_raw = None
    sigma_raw = None
    try:
        if hasattr(app_cfg, "get_control_value"):
            a_raw = app_cfg.get_control_value("HW_A", default=None)
            sigma_raw = app_cfg.get_control_value("HW_SIGMA_BASE", default=None)
    except Exception:
        a_raw = None
        sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01

    print(f"[INFO] HW params: a={a:.6f}, sigma={sigma:.6f}, asof={asof}")

    # Dense zero curve at semi-annual spacing
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    hw_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)

    base = app_cfg.dated_output_root
    outdir = base / "hw"
    outdir.mkdir(parents=True, exist_ok=True)

    safe_key = curve_key.replace(" ", "_")
    date_str = asof.isoformat()

    parquet_path = outdir / f"hw_theta_{safe_key}_{date_str}.parquet"
    xlsx_path = outdir / f"hw_theta_{safe_key}_{date_str}.xlsx"

    hw_df.to_parquet(parquet_path)

    # Excel for inspection
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        hw_df.to_excel(writer, sheet_name="HW_THETA", index=False)

    print(
        f"[OK] Exported HW theta curve for {curve_key} @ {asof} to\n"
        f"      {parquet_path}\n"
        f"      {xlsx_path}"
    )


def build_hw_theta_from_dense(
    dense_df: pd.DataFrame,
    a: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Given a *single-date* dense zero curve:

        date, tenor_yrs, rate_dec   (rate_dec = zero yield in DECIMAL)

    build:

        - discount factors P(0,t) using continuous compounding: P = exp(-z * t)
        - instantaneous forward curve f(0,t) = - d/dt ln P(0,t)
        - Hull–White theta(t) satisfying the initial term structure:

              theta(t) = df/dt + a * f(t) + (sigma^2 / (2a)) * (1 - exp(-2a t))

    Returns a DataFrame with columns:

        date, tenor_yrs, rate_dec, df, inst_fwd, df_dt, theta
    """
    if dense_df.empty:
        raise ValueError("dense_df is empty; cannot build HW theta.")

    df = dense_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    unique_dates = df["date"].unique()
    if len(unique_dates) != 1:
        raise ValueError(
            f"build_hw_theta_from_dense expects one date, got {unique_dates}"
        )
    dt0 = unique_dates[0]

    g = df.sort_values("tenor_yrs")
    t = g["tenor_yrs"].to_numpy(dtype=float)  # in years
    z = g["rate_dec"].to_numpy(dtype=float)   # zero yields (decimal)

    if len(t) < 2:
        raise ValueError("Need at least 2 tenor points to build HW theta.")

    # Discount factors with continuous compounding: P(0,t) = exp(-z * t)
    lnP = -z * t
    P = np.exp(lnP)

    # Instantaneous forward: f(0,t) = - d/dt ln P(0,t)
    dlnP_dt = np.gradient(lnP, t)
    f_inst = -dlnP_dt

    # df/dt term
    df_dt = np.gradient(f_inst, t)

    if a <= 0.0:
        raise ValueError(f"Hull-White mean reversion 'a' must be > 0, got {a}.")
    if sigma < 0.0:
        raise ValueError(f"Hull-White sigma must be >= 0, got {sigma}.")

    # Hull–White theta(t)
    # theta(t) = df/dt + a * f(t) + (sigma^2 / (2a)) * (1 - exp(-2 a t))
    theta = df_dt + a * f_inst + (sigma * sigma / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))

    out = pd.DataFrame(
        {
            "date": dt0,
            "tenor_yrs": t,
            "rate_dec": z,
            "df": P,
            "inst_fwd": f_inst,
            "df_dt": df_dt,
            "theta": theta,
        }
    )
    out.sort_values("tenor_yrs", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def export_dense_curve_and_forward_matrix(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    For a chosen curve_key (default AAA_MUNI_SPOT), build:

      (1) A dense semi-annual zero curve for *all dates*.
      (2) A full forward-rate matrix f(t1 -> t2) for a single as-of date.

    Exports into:

        <repo_root>/output/curves/YYYY-MM-DD/dense/
        <repo_root>/output/curves/YYYY-MM-DD/forward/

    Files:

      DENSE (all dates)
        - dense_zero_<curve_key>_all.parquet
        - dense_zero_<curve_key>_all.xlsx   (date × tenor_yrs, horizontal)

      FORWARD (as-of date)
        - forward_matrix_<curve_key>_<YYYY-MM-DD>.parquet
        - dense_and_forward_<curve_key>_<YYYY-MM-DD>.xlsx
            * Sheet 'DENSE_ZERO' : one row, tenors as columns
            * Sheet 'FWD_MATRIX' : matrix rows=start_yrs, cols=end_yrs
    """
    import numpy as np  # ensure numpy is imported at top of file if not already

    # Base dated folder like: output/curves/2025-11-26
    base = app_cfg.dated_output_root
    dense_dir = base / "dense"
    fwd_dir = base / "forward"
    dense_dir.mkdir(parents=True, exist_ok=True)
    fwd_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Dense zero curves for *all* dates ----------
    dense_all = build_dense_zero_curve_all_dates(
        history_df=history_df,
        curve_key=curve_key,
        step_years=step_years,
    )

    safe_key = curve_key.replace(" ", "_")

    if not dense_all.empty:
        # Parquet: engine-facing, long format
        dense_all_parquet = dense_dir / f"dense_zero_{safe_key}_all.parquet"
        dense_all.to_parquet(dense_all_parquet)

        # Excel: human-facing, wide (dates rows, tenors columns)
        dense_all_wide = _pivot_curve(dense_all, value_col="rate_dec")
        dense_all_xlsx = dense_dir / f"dense_zero_{safe_key}_all.xlsx"
        with pd.ExcelWriter(dense_all_xlsx, engine="openpyxl") as writer:
            dense_all_wide.to_excel(writer, sheet_name="DENSE_ZERO_ALL", index=False)

    # ---------- 2) Forward matrix for a single as-of date ----------

    # Resolve as-of date: Controls CURVE_ASOF_DATE > YAML curves.curve_asof_date > latest in history
    # Try Controls first if available
    asof: Optional[date_type] = None
    ctrl_raw: Optional[str] = None
    try:
        if hasattr(app_cfg, "get_control_value"):
            ctrl_raw = app_cfg.get_control_value("CURVE_ASOF_DATE", default=None)
    except Exception:
        ctrl_raw = None

    if ctrl_raw:
        asof = pd.to_datetime(ctrl_raw).date()
    elif app_cfg.curves.curve_asof_date:
        asof = pd.to_datetime(app_cfg.curves.curve_asof_date).date()
    else:
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    # Dense curve for the as-of date only
    dense_asof = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # Forward matrix from that dense curve
    fwd_df = build_forward_matrix_from_dense(dense_asof)

    date_str = asof.isoformat()

    # Parquet: forward matrix long format
    fwd_parquet_path = fwd_dir / f"forward_matrix_{safe_key}_{date_str}.parquet"
    fwd_df.to_parquet(fwd_parquet_path)

    # Excel: one workbook for the as-of date
    #   Sheet 1: dense zero curve HORIZONTAL (tenors as columns)
    dense_asof_wide = _pivot_curve(dense_asof, value_col="rate_dec")

    #   Sheet 2: forward matrix (rows=start_yrs, cols=end_yrs)
    fwd_matrix = fwd_df.pivot(index="start_yrs", columns="end_yrs", values="fwd_rate_dec")
    fwd_matrix.sort_index(axis=0, inplace=True)
    fwd_matrix.sort_index(axis=1, inplace=True)
    fwd_matrix.columns.name = None
    fwd_matrix.index.name = "start_yrs"

    xlsx_path = fwd_dir / f"dense_and_forward_{safe_key}_{date_str}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        dense_asof_wide.to_excel(writer, sheet_name="DENSE_ZERO", index=False)
        fwd_matrix.to_excel(writer, sheet_name="FWD_MATRIX")

    print(
        f"[OK] Exported dense zero curves for ALL dates to {dense_dir}, "
        f"and forward matrix for {curve_key} @ {date_str} to {xlsx_path}"
    )


def export_forward_matrix_range(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    Build forward matrices f(t1 -> t2) for a RANGE of dates and
    export a single long-format Parquet file for Python/ML.

    Controls used (in MUNI_MASTER_BUCKET Controls sheet):

        CURVE_RANGE_FROM   (optional, YYYY-MM-DD)
        CURVE_RANGE_TO     (optional, YYYY-MM-DD)

    If either is missing, we fall back to the min/max available dates
    in history_df for the given curve_key.

    Output:

        <repo_root>/output/curves/<CURVE_ASOF_DATE>/forward_range/
            forward_matrix_<curve_key>_<from>_<to>.parquet

    Parquet columns:

        date, start_yrs, end_yrs, fwd_rate_dec
    """
    import pandas as pd

    # Base dated folder like: output/curves/2025-11-26
    base = app_cfg.dated_output_root
    outdir = base / "forward_range"
    outdir.mkdir(parents=True, exist_ok=True)

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["curve_key"] == curve_key]

    if df.empty:
        print(f"[WARN] No history rows found for curve_key={curve_key}; skipping forward matrix range export.")
        return

    # ---- Resolve date range from Controls or disable if not set ----
    start_raw = None
    end_raw = None
    try:
        if hasattr(app_cfg, "get_control_value"):
            start_raw = app_cfg.get_control_value("CURVE_RANGE_FROM", default=None)
            end_raw = app_cfg.get_control_value("CURVE_RANGE_TO", default=None)
    except Exception:
        start_raw = None
        end_raw = None

    # If BOTH are missing/empty -> treat as "feature off"
    if (not start_raw) and (not end_raw):
        print(
            "[INFO] Forward matrix RANGE export disabled: "
            "no CURVE_RANGE_FROM / CURVE_RANGE_TO set in Controls."
        )
        return

    # Otherwise, use provided bounds, falling back to history min/max
    if start_raw:
        start_date = pd.to_datetime(start_raw).date()
    else:
        start_date = df["date"].min()

    if end_raw:
        end_date = pd.to_datetime(end_raw).date()
    else:
        end_date = df["date"].max()

    # Ensure ordering
    if end_date < start_date:
        start_date, end_date = end_date, start_date


    # Filter dates within range
    mask_range = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask_range]

    if df.empty:
        print(
            f"[WARN] No dates in history for curve_key={curve_key} "
            f"between {start_date} and {end_date}; skipping forward matrix range export."
        )
        return

    unique_dates = sorted(df["date"].unique())
    frames: list[pd.DataFrame] = []

    for dt in unique_dates:
        try:
            dense_dt = build_dense_zero_curve_for_date(
                history_df=history_df,
                curve_key=curve_key,
                target_date=dt,
                step_years=step_years,
            )
            fwd_dt = build_forward_matrix_from_dense(dense_dt)
            frames.append(fwd_dt)
        except Exception as exc:
            print(f"[WARN] Skipping date {dt} in forward-matrix range export: {exc}")
            continue

    if not frames:
        print(
            f"[WARN] All dates failed during forward-matrix construction for "
            f"{curve_key} between {start_date} and {end_date}."
        )
        return

    all_fwd = pd.concat(frames, ignore_index=True)
    all_fwd.sort_values(["date", "start_yrs", "end_yrs"], inplace=True)
    all_fwd.reset_index(drop=True, inplace=True)

    safe_key = curve_key.replace(" ", "_")
    fname = f"forward_matrix_{safe_key}_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    out_path = outdir / fname

    all_fwd.to_parquet(out_path)

    print(
        f"[OK] Exported forward matrix RANGE for {curve_key} from {start_date} "
        f"to {end_date} to {out_path}"
    )

def export_dense_zero_panel_range(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key_muni: str = "AAA_MUNI_SPOT",
    curve_key_ust: str = "UST_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    Build a dense semi-annual zero-curve PANEL for a RANGE of dates,
    for both AAA_MUNI_SPOT and UST_SPOT, plus their spreads.

    Controls used (in MUNI_MASTER_BUCKET Controls sheet):

        CURVE_RANGE_FROM   (optional, YYYY-MM-DD)
        CURVE_RANGE_TO     (optional, YYYY-MM-DD)

    If BOTH are blank/missing, we treat this feature as OFF.

    Output:

        <repo_root>/output/curves/<CURVE_ASOF_DATE>/dense_panel/
            dense_panel_<muni>_<ust>_<from>_<to>.parquet

    Parquet columns:

        date, tenor_yrs, rate_muni_dec, rate_ust_dec, spread_bp
    """
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Filter only the two curve keys we care about
    df = df[df["curve_key"].isin([curve_key_muni, curve_key_ust])]
    if df.empty:
        print(
            f"[WARN] No history rows found for curve_key in "
            f"({curve_key_muni}, {curve_key_ust}); skipping dense panel export."
        )
        return

    # ---- Resolve date range from Controls or disable if not set ----
    start_raw = None
    end_raw = None
    try:
        if hasattr(app_cfg, "get_control_value"):
            start_raw = app_cfg.get_control_value("CURVE_RANGE_FROM", default=None)
            end_raw = app_cfg.get_control_value("CURVE_RANGE_TO", default=None)
    except Exception:
        start_raw = None
        end_raw = None

    # If BOTH are missing/empty -> feature OFF
    if (not start_raw) and (not end_raw):
        print(
            "[INFO] Dense zero PANEL range export disabled: "
            "no CURVE_RANGE_FROM / CURVE_RANGE_TO set in Controls."
        )
        return

    # Otherwise, use provided bounds, falling back to history min/max
    # (using only the muni curve for range if you like, but here we use df)
    if start_raw:
        start_date = pd.to_datetime(start_raw).date()
    else:
        start_date = df["date"].min()

    if end_raw:
        end_date = pd.to_datetime(end_raw).date()
    else:
        end_date = df["date"].max()

    # Ensure ordering
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    mask_range = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask_range]

    if df.empty:
        print(
            f"[WARN] No dates in history for curve panel between "
            f"{start_date} and {end_date}; skipping dense panel export."
        )
        return

    unique_dates = sorted(df["date"].unique())

    muni_frames: list[pd.DataFrame] = []
    ust_frames: list[pd.DataFrame] = []

    for dt in unique_dates:
        # muni
        try:
            dense_muni = build_dense_zero_curve_for_date(
                history_df=history_df,
                curve_key=curve_key_muni,
                target_date=dt,
                step_years=step_years,
            )
            muni_frames.append(dense_muni)
        except Exception as exc:
            print(f"[WARN] Skipping muni dense curve for {dt}: {exc}")

        # ust
        try:
            dense_ust = build_dense_zero_curve_for_date(
                history_df=history_df,
                curve_key=curve_key_ust,
                target_date=dt,
                step_years=step_years,
            )
            ust_frames.append(dense_ust)
        except Exception as exc:
            print(f"[WARN] Skipping UST dense curve for {dt}: {exc}")

    if not muni_frames or not ust_frames:
        print(
            f"[WARN] Dense panel export: no valid muni or ust frames for "
            f"({curve_key_muni}, {curve_key_ust}) in {start_date}–{end_date}."
        )
        return

    muni_all = pd.concat(muni_frames, ignore_index=True)
    ust_all = pd.concat(ust_frames, ignore_index=True)

    muni_all.rename(columns={"rate_dec": "rate_muni_dec"}, inplace=True)
    ust_all.rename(columns={"rate_dec": "rate_ust_dec"}, inplace=True)

    # Merge on date + tenor_yrs
    merged = pd.merge(
        muni_all[["date", "tenor_yrs", "rate_muni_dec"]],
        ust_all[["date", "tenor_yrs", "rate_ust_dec"]],
        on=["date", "tenor_yrs"],
        how="inner",
    )

    if merged.empty:
        print(
            f"[WARN] No overlapping tenors between muni and UST dense curves "
            f"in range {start_date}–{end_date}; skipping dense panel export."
        )
        return

    merged["spread_bp"] = (merged["rate_muni_dec"] - merged["rate_ust_dec"]) * 10_000.0

    merged.sort_values(["date", "tenor_yrs"], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # Output path under dated folder
    base = app_cfg.dated_output_root
    outdir = base / "dense_panel"
    outdir.mkdir(parents=True, exist_ok=True)

    muni_safe = curve_key_muni.replace(" ", "_")
    ust_safe = curve_key_ust.replace(" ", "_")

    fname = (
        f"dense_panel_{muni_safe}_{ust_safe}_"
        f"{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    )
    out_path = outdir / fname

    merged.to_parquet(out_path)

    print(
        f"[OK] Exported dense zero PANEL for {curve_key_muni} vs {curve_key_ust} "
        f"from {start_date} to {end_date} to {out_path}"
    )


# ---------- Dense curves for ALL dates ----------


def build_dense_zero_curves_all_dates(
    history_df: pd.DataFrame,
    curve_key: str,
    step_years: float = 0.5,
) -> pd.DataFrame:
    """
    Build dense semi-annual zero curves for *all* dates for a given curve_key.

    Input (history_df):
        date, curve_key, tenor_yrs, rate_dec  [long format]

    Output:
        date, tenor_yrs, rate_dec  [long format]
    """
    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    sub = df[df["curve_key"] == curve_key].copy()
    if sub.empty:
        raise ValueError(f"No rows found in history for curve_key={curve_key}")

    records: list[dict] = []

    for dt, group in sub.groupby("date"):
        group = group.sort_values("tenor_yrs")

        tenors = group["tenor_yrs"].to_numpy(dtype=float)
        rates = group["rate_dec"].to_numpy(dtype=float)

        if len(tenors) < 2:
            continue

        interpolator = PchipInterpolator(tenors, rates, extrapolate=False)

        t_max = float(tenors.max())
        grid = np.arange(step_years, t_max + 1e-9, step_years)

        dense_rates = interpolator(grid)
        mask_valid = ~np.isnan(dense_rates)

        grid = grid[mask_valid]
        dense_rates = dense_rates[mask_valid]

        for t, r in zip(grid, dense_rates):
            records.append(
                {
                    "date": dt,
                    "tenor_yrs": float(t),
                    "rate_dec": float(r),
                }
            )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out.sort_values(["date", "tenor_yrs"], inplace=True)
        out.reset_index(drop=True, inplace=True)

    return out


def export_dense_curves_all_dates(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    Build dense semi-annual zero curves for *all* dates for the given curve_key
    and export to:

        <repo_root>/output/curves/YYYY-MM-DD/dense/

    Outputs:
      - dense_zero_<curve_key>_ALL.parquet       (long format)
      - dense_zero_<curve_key>_ALL_<timestamp>.xlsx
          * Sheet 'DENSE_ZERO_ALL': date × tenor_yrs (wide)
    """
    base = app_cfg.dated_output_root
    dirs = _ensure_output_subdirs(base)
    dense_dir = dirs["dense"]

    dense_df = build_dense_zero_curves_all_dates(
        history_df=history_df,
        curve_key=curve_key,
        step_years=step_years,
    )

    safe_key = curve_key.replace(" ", "_")

    parquet_path = dense_dir / f"dense_zero_{safe_key}_ALL.parquet"
    dense_df.to_parquet(parquet_path)

    dense_wide = _pivot_curve(dense_df, value_col="rate_dec")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = dense_dir / f"dense_zero_{safe_key}_ALL_{run_ts}.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        dense_wide.to_excel(writer, sheet_name="DENSE_ZERO_ALL", index=False)

    print(
        f"[OK] Exported dense zero curves for ALL dates for {curve_key} to\n"
        f"  {parquet_path}\n  {xlsx_path}"
    )


# ---------- ZeroCurve loader for a given date ----------


def get_zero_curve_for_date(
    target_date: date_type,
    curve_key: CurveKey,
    app_cfg: AppConfig,
) -> ZeroCurve:
    """
    Load the historical curves table and return a ZeroCurve for the
    requested date and curve_key.

    This uses the 'rate_dec' column as the zero yield (decimal).
    """
    ch_cfg = CurveHistoryConfig.from_app_config(app_cfg)

    if not ch_cfg.history_file.exists():
        raise FileNotFoundError(
            f"Curve history file does not exist: {ch_cfg.history_file}"
        )

    if ch_cfg.history_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(ch_cfg.history_file)
    else:
        df = pd.read_csv(ch_cfg.history_file, parse_dates=["date"])

    df["date"] = pd.to_datetime(df["date"]).dt.date

    mask = (df["date"] == target_date) & (df["curve_key"] == curve_key)
    sub = df.loc[mask].copy()

    if sub.empty:
        raise ValueError(
            f"No curve rows found for date={target_date} and curve_key={curve_key}"
        )

    sub.sort_values("tenor_yrs", inplace=True)

    pairs = [
        (float(row["tenor_yrs"]), float(row["rate_dec"]))
        for _, row in sub.iterrows()
    ]

    return make_zero_curve_from_pairs(pairs)
