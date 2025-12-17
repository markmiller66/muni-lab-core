"""
build_positions.py

Stage 1 pipeline: build standardized broker position table.

Workflow:
- Load broker files from configured directories
- Apply broker-specific loaders
- Normalize to canonical schema
- Write combined positions output

Dependency contract:
- Allowed imports: config, loaders, utils, pandas, standard library
- Forbidden imports: enrich_with_msrb
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from numpy import ndarray, dtype, integer, floating
from pandas import Series

from muni_core.io.broker_positions.config import BROKER_DIRS, POSITIONS_OUT, STANDARD_COLUMNS
from muni_core.io.broker_positions.loaders import load_one_file, standardize_positions


from pathlib import Path
import os
import pandas as pd

def build_positions() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # helper defined ONCE, not inside loops
    def _to_num_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str)
             .str.replace("$", "", regex=False)
             .str.replace(",", "", regex=False)
             .str.strip(),
            errors="coerce"
        )

    for directory in BROKER_DIRS:
        if not os.path.exists(directory):
            print(f"Missing directory: {directory}")
            continue

        broker = os.path.basename(directory).upper().strip()
        files = [f for f in os.listdir(directory) if f.lower().endswith((".csv", ".xls", ".xlsx"))]
        print(f"[{broker}] files={len(files)}")

        for f in files:
            path = os.path.join(directory, f)
            try:
                raw = load_one_file(broker, path)
                if raw.empty:
                    print(f"  - {f}: empty after load")
                    continue

                std = standardize_positions(raw, broker, f)

                # enforce standard schema (adds missing cols)
                std = std.reindex(columns=STANDARD_COLUMNS)

                # --- Derive MRKT PRICE when missing ---
                if {"MRKT PRICE", "MRKT VALUE", "QTY"}.issubset(std.columns):
                    qty = _to_num_series(std["QTY"])
                    mv  = _to_num_series(std["MRKT VALUE"])
                    mp  = _to_num_series(std["MRKT PRICE"])

                    # treat small QTY as bond-count for RJ/FID (count * 1000 par)
                    denom = qty.copy()
                    if broker in ("RJ", "FID"):
                        denom = denom.where(denom >= 1000, denom * 1000)

                    derived_price = 100.0 * (mv / denom)

                    mp_missing = (mp.isna() | (mp.fillna(0) == 0)) & mv.notna() & denom.notna() & (denom != 0)
                    std.loc[mp_missing, "MRKT PRICE"] = derived_price.loc[mp_missing].round(5)

                # drop rows that are fully empty except broker/pricing date
                cols_check = [c for c in STANDARD_COLUMNS if c not in ("BROKER", "PRICING DATE")]
                std = std.dropna(how="all", subset=cols_check)

                if std.empty:
                    print(f"  - {f}: empty after standardization")
                    continue

                frames.append(std)
                print(f"  - {f}: rows={len(std)}")

            except Exception as e:
                print(f"  - {f}: ERROR {type(e).__name__}: {e}")

    if not frames:
        raise RuntimeError("No broker files loaded. Check BROKER_DIRS and file formats.")

    combined = pd.concat(frames, ignore_index=True)

    from muni_core.io.broker_positions.utils import (
        normalize_date_columns,
        normalize_price_to_percent_of_par,
    )

    combined = normalize_date_columns(combined, cols=["PRICING DATE", "ACQ DATE", "CALL_DATE", "MATURITY"])

    # ✅ normalize WF (fraction-of-par) prices to percent-of-par before writing
    combined = normalize_price_to_percent_of_par(combined)

    combined.to_excel(POSITIONS_OUT, index=False)
    print(f"Wrote: {POSITIONS_OUT} rows={len(combined)}")
    return combined


def main() -> None:
    build_positions()


if __name__ == "__main__":
    main()
