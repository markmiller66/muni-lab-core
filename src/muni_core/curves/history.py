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
CurveKey = Literal["AAA_MUNI_PAR", "UST_PAR"]


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
            rate_dec = float(val) / 100.0
            treas_long_records.append(
                {
                    "date": dt.normalize(),
                    "curve_key": "UST_PAR",
                    "tenor_yrs": tenor_yrs,
                    "rate_dec": rate_dec,
                }
            )

    treas_long = pd.DataFrame.from_records(treas_long_records)

    # --- Combine ---
    all_long = pd.concat([muni_long, treas_long], ignore_index=True)

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


