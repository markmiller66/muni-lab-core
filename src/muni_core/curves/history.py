# src/muni_core/curves/history.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

from muni_core.config.loader import AppConfig
from .types import CurvePoint, ZeroCurve
from .zero_curve import make_zero_curve_from_pairs


# You can adjust/extend these later as needed
CurCurveKey = Literal["AAA_MUNI_PAR", "AAA_MUNI_SPOT", "UST_PAR", "UST_SPOT"]



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
                        # if missing, we bail out for this term
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
      - Treats the treasury CSV as par yields where available (you can refine
        this later to use the Fed spot columns if you like)
      - Produces a long DataFrame with columns:
            date, curve_key, tenor_yrs, rate_dec
      - Does NOT yet run a full par->spot bootstrap. That is where you'll
        graft in the 'brains' from your original spot.py.

    Once this is stable, you can:
      - Add par->spot bootstrap
      - Add HW theta / short-rate columns
      - Add muni/treas spread columns
    """
    ch_cfg = CurveHistoryConfig.from_app_config(app_cfg)

    # --- Muni side ---
    # Your muni_df currently has date index and columns like "1 yr", "2 yr", ...
    muni = muni_df.copy()

    # Normalize column names to numeric tenor in years, e.g. "1 yr" -> 1.0
    tenor_map = {}
    for col in muni.columns:
        col_str = str(col).strip().lower()
        # crude parse: look for leading integer
        try:
            n = int(col_str.split()[0])
            tenor_map[col] = float(n)
        except Exception:
            # skip non-tenor columns for now
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
    # For now, grab some standard Svensson/spot columns if they exist.
    # You can refine this to match your actual usage later.
    treas = treas_df.copy()

    # Example: if you have columns like "SVENY01", "SVENY02", ... (zero yields)
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
    # Combine: muni PAR, muni SPOT, UST SPOT
    frames = [muni_long, muni_spot_long, treas_long]
    all_long = pd.concat(frames, ignore_index=True)


    # Optional: sort for sanity
    all_long.sort_values(["date", "curve_key", "tenor_yrs"], inplace=True)
    all_long.reset_index(drop=True, inplace=True)

    # Save to history_file
    ch_cfg.history_file.parent.mkdir(parents=True, exist_ok=True)
    if ch_cfg.history_file.suffix.lower() == ".parquet":
        all_long.to_parquet(ch_cfg.history_file)
    else:
        all_long.to_csv(ch_cfg.history_file, index=False)

    return all_long


def get_zero_curve_for_date(
    target_date: date_type,
    curve_key: CurveKey,
    app_cfg: AppConfig,
) -> ZeroCurve:
    """
    Load the historical curves table and return a ZeroCurve for the
    requested date and curve_key.

    Right now, this uses the 'rate_dec' directly as if they were zero
    yields. Once you have a true spot bootstrap, you'll either:
      - Build a separate 'AAA_MUNI_SPOT' curve_key, or
      - Transform these par yields into spots here.
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

    # Normalize date to date-only
    df["date"] = pd.to_datetime(df["date"]).dt.date

    mask = (df["date"] == target_date) & (df["curve_key"] == curve_key)
    sub = df.loc[mask].copy()

    if sub.empty:
        raise ValueError(
            f"No curve rows found for date={target_date} and curve_key={curve_key}"
        )

    sub.sort_values("tenor_yrs", inplace=True)

    # ZeroCurve.from_pairs expects an iterable of (tenor, rate) pairs
    pairs = [
        (float(row["tenor_yrs"]), float(row["rate_dec"]))
        for _, row in sub.iterrows()
    ]

    return make_zero_curve_from_pairs(pairs)


