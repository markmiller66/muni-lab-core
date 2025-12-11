# src/muni_core/curves/short_rate_lattice.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from muni_core.config.loader import AppConfig
from muni_core.curves.history import (
    build_dense_zero_curve_for_date,
    build_hw_theta_from_dense,
)


def build_short_rate_path_from_hw(hw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a *deterministic* short-rate path from the HW output.

    We approximate the short rate r(t) by the instantaneous forward rate
    f(0,t) in the 'inst_fwd' column. If 'inst_fwd' is missing, we fall
    back to the zero yield 'rate_dec'.

    Parameters
    ----------
    hw_df : DataFrame
        Columns: date, tenor_yrs, rate_dec, df, inst_fwd, df_dt, theta

    Returns
    -------
    DataFrame
        Columns:
            step     : integer time step index (0..N-1)
            t_yrs    : time in years
            r_short  : short rate at that step (decimal)
    """
    if hw_df.empty:
        raise ValueError("hw_df is empty; cannot build short-rate path.")

    df = hw_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("tenor_yrs").reset_index(drop=True)

    if "inst_fwd" in df.columns and not df["inst_fwd"].isna().all():
        r = df["inst_fwd"].to_numpy(dtype=float)
    else:
        # fallback: use zero yields
        r = df["rate_dec"].to_numpy(dtype=float)

    t = df["tenor_yrs"].to_numpy(dtype=float)

    steps = np.arange(len(t), dtype=int)

    out = pd.DataFrame(
        {
            "step": steps,
            "t_yrs": t,
            "r_short": r,
        }
    )
    return out


def build_binomial_lattice_from_hw(
    hw_df: pd.DataFrame,
    sigma: float,
    dt: float,
) -> pd.DataFrame:
    """
    Build a simple *recombining binomial* short-rate lattice using
    a Hull–White-style baseline path from f(0,t) (inst_fwd) and a
    constant volatility sigma.

    This is an initial lattice suitable for experimentation and will
    later be refined (e.g., full HW calibration, probabilities, etc.).

    Construction:
      - For each time step n, define a "mid" rate r_mid(n) from inst_fwd.
      - Node indices j = -n, ..., +n
      - Short rate at node (n, j):
            r(n, j) = r_mid(n) + j * (sigma * sqrt(dt))

      - This produces a recombining tree in j.

    Parameters
    ----------
    hw_df : DataFrame
        HW output for a single as-of date (one date), columns including:
            tenor_yrs, inst_fwd (or rate_dec as fallback)
    sigma : float
        Short-rate volatility (decimal per year).
    dt : float
        Time step size in years (e.g., 0.5).

    Returns
    -------
    DataFrame
        Columns:
            step     : integer time step index (0..N-1)
            t_yrs    : time in years at this step
            j        : node index at this step (-step .. +step)
            r_short  : short rate at that node (decimal)
    """
    if hw_df.empty:
        raise ValueError("hw_df is empty; cannot build binomial lattice.")

    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}.")

    df = hw_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("tenor_yrs").reset_index(drop=True)

    if "inst_fwd" in df.columns and not df["inst_fwd"].isna().all():
        r_mid_all = df["inst_fwd"].to_numpy(dtype=float)
    else:
        r_mid_all = df["rate_dec"].to_numpy(dtype=float)

    t_all = df["tenor_yrs"].to_numpy(dtype=float)
    n_steps = len(t_all)

    up_step = sigma * np.sqrt(dt)

    records: list[dict] = []

    for n in range(n_steps):
        t_n = t_all[n]
        r_mid = r_mid_all[n]

        # recombining node indices for this step
        # j = -n, ..., +n
        for j in range(-n, n + 1):
            r_ij = r_mid + j * up_step

            records.append(
                {
                    "step": n,
                    "t_yrs": t_n,
                    "j": j,
                    "r_short": r_ij,
                }
            )

    lattice_df = pd.DataFrame.from_records(records)
    lattice_df.sort_values(["step", "j"], inplace=True)
    lattice_df.reset_index(drop=True, inplace=True)
    return lattice_df


def export_hw_lattice_for_asof(
    history_df: pd.DataFrame,
    app_cfg: AppConfig,
    curve_key: str = "AAA_MUNI_SPOT",
    step_years: float = 0.5,
) -> None:
    """
    High-level helper:
      - takes historical curves table
      - chooses CURVE_ASOF_DATE
      - builds dense zero curve
      - builds HW theta grid
      - builds deterministic short-rate path
      - builds simple binomial lattice
      - exports all to the dated HW folder.

    Controls used (MUNI_MASTER_BUCKET Controls sheet):

        CURVE_ASOF_DATE
        HW_A
        HW_SIGMA_BASE

    Outputs into:
        <repo_root>/output/curves/<CURVE_ASOF_DATE>/hw/

      - hw_theta_<curve_key>_<asof>.parquet   (already produced elsewhere)
      - hw_lattice_path_<curve_key>_<asof>.parquet
      - hw_lattice_tree_<curve_key>_<asof>.parquet
      - hw_lattice_<curve_key>_<asof>.xlsx
          * PATH sheet  : step, t_yrs, r_short
          * LATTICE sheet: step, t_yrs, j, r_short
    """
    curves_cfg = app_cfg.curves

    # Resolve as-of date: prefer CURVE_ASOF_DATE from YAML / Controls
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        df_local = history_df.copy()
        df_local["date"] = pd.to_datetime(df_local["date"]).dt.date
        asof = df_local["date"].max()

    # HW parameters from Controls, with same defaults used elsewhere
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

    print(f"[INFO] HW lattice params: a={a:.6f}, sigma={sigma:.6f}, asof={asof}, dt={step_years}")

    # --- Build dense zero curve for this date/curve ---
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )

    # --- Build HW theta grid from dense curve ---
    hw_df = build_hw_theta_from_dense(dense_df, a=a, sigma=sigma)

    # --- Deterministic short-rate path ---
    path_df = build_short_rate_path_from_hw(hw_df)

    # --- Simple binomial lattice around that path ---
    lattice_df = build_binomial_lattice_from_hw(hw_df, sigma=sigma, dt=step_years)

    # --- Export under dated HW folder ---
    base = app_cfg.dated_output_root
    outdir = base / "hw"
    outdir.mkdir(parents=True, exist_ok=True)

    safe_key = curve_key.replace(" ", "_")
    date_str = asof.isoformat()

    path_parquet = outdir / f"hw_lattice_path_{safe_key}_{date_str}.parquet"
    tree_parquet = outdir / f"hw_lattice_tree_{safe_key}_{date_str}.parquet"
    xlsx_path = outdir / f"hw_lattice_{safe_key}_{date_str}.xlsx"

    path_df.to_parquet(path_parquet)
    lattice_df.to_parquet(tree_parquet)

    # Excel for human inspection
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        path_df.to_excel(writer, sheet_name="PATH", index=False)
        lattice_df.to_excel(writer, sheet_name="LATTICE", index=False)

    print(
        f"[OK] Exported HW short-rate path and binomial lattice for {curve_key} @ {asof} to\n"
        f"      {path_parquet}\n"
        f"      {tree_parquet}\n"
        f"      {xlsx_path}"
    )
