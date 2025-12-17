"""
run_broker_positions_pipeline.py

Convenience runner for the broker positions ingestion + MSRB enrichment pipeline.

Runs:
  1) muni_core.io.broker_positions.build_positions.build_positions()
  2) muni_core.io.broker_positions.enrich_with_msrb.enrich_with_msrb()

Usage (from repo root):
  python run_broker_positions_pipeline.py

Or (preferred):
  python -m run_broker_positions_pipeline

Notes:
- Assumes paths are configured in:
    muni_core.io.broker_positions.config
- Designed to be safe to run repeatedly; outputs overwrite by default.
"""

from __future__ import annotations


import os
import sys
from pathlib import Path

import pandas as pd

from muni_core.io.broker_positions.config import (
    BROKER_DIRS,
    MSRB_FILE,
    POSITIONS_OUT,
    ENRICHED_OUT,
)
from muni_core.io.broker_positions.build_positions import build_positions
from muni_core.io.broker_positions.enrich_with_msrb import enrich_with_msrb


def _check_paths() -> None:
    missing_dirs = [d for d in BROKER_DIRS if not os.path.exists(d)]
    if missing_dirs:
        print("\n[WARN] Missing broker directories:")
        for d in missing_dirs:
            print("  -", d)
        print("Proceeding anyway (will load what exists).\n")

    if not os.path.exists(MSRB_FILE):
        raise FileNotFoundError(f"MSRB_FILE not found: {MSRB_FILE}")

    # Ensure output folder exists (in case config changed)
    Path(POSITIONS_OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(ENRICHED_OUT).parent.mkdir(parents=True, exist_ok=True)


def _coverage_report(df: pd.DataFrame) -> None:
    def pct_filled(col: str) -> str:
        if col not in df.columns:
            return "n/a"
        return f"{100.0 * df[col].notna().mean():.2f}%"

    cols = [
        "CUSIP", "DESCRIPTION", "QTY", "COUPON", "MATURITY",
        "BASIS PRICE", "BASIS VALUE", "MRKT VALUE", "MRKT PRICE",
        "MOODYS", "S&P", "CALL_DATE", "CALL_PRICE",
        "ACQ DATE", "BROKER", "PRICING DATE",
    ]

    print("\n[Coverage]")
    for c in cols:
        print(f"  {c:12s}: {pct_filled(c)}")

    # Common sanity checks
    if "CUSIP" in df.columns:
        bad = df["CUSIP"].isna().sum()
        print(f"\n[Sanity] Missing CUSIP rows: {bad}")
    if "BROKER" in df.columns:
        print("\n[Counts by BROKER]")
        print(df["BROKER"].value_counts(dropna=False).to_string())
from muni_core.io.broker_positions.interest_calendar import build_monthly_interest_calendar




def main() -> int:
    try:
        _check_paths()

        print("[1/2] Building standardized combined positions...")
        pos_df = build_positions()
        print(f"    -> Wrote: {POSITIONS_OUT}")

        print("\n[2/2] Enriching with MSRB reference...")
        enriched_df = enrich_with_msrb()
        print(f"    -> Wrote: {ENRICHED_OUT}")

        print("\n[Interest Calendar - field coverage by broker]")
        needed = ["QTY", "COUPON", "MATURITY"]
        for c in needed:
            if c not in enriched_df.columns:
                print(f"  MISSING COLUMN: {c}")

        if all(c in enriched_df.columns for c in needed):
            tmp = enriched_df.copy()
            tmp["QTY_NUM"] = pd.to_numeric(tmp["QTY"].astype(str).str.replace(",", "", regex=False), errors="coerce")
            print(
                tmp.assign(
                    MISS_QTY=tmp["QTY_NUM"].isna(),
                    MISS_COUPON=tmp["COUPON"].isna(),
                    MISS_MATURITY=tmp["MATURITY"].isna(),
                )
                .groupby("BROKER")[["MISS_QTY", "MISS_COUPON", "MISS_MATURITY"]]
                .mean()
                .sort_index()
                .to_string()
            )

        _coverage_report(enriched_df)
        print("\n[3/3] Building monthly interest calendar...")
        build_monthly_interest_calendar()

        print("\nDone.")
        return 0

    except Exception as e:
        print("\n[ERROR]", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
