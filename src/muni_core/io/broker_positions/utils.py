"""
utils.py

Stateless helper utilities for broker/MSRB ingestion.

Includes:
- Column name normalization
- CUSIP normalization
- Filename date extraction
- Safe CSV/Excel readers
- Numeric coercion helpers

Dependency contract:
- Allowed imports: standard library, pandas
- Forbidden imports: config, loaders, build_positions, enrich_with_msrb
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import pandas as pd


def normalize_col(c: str) -> str:
    if c is None:
        return ""
    s = str(c).replace("\ufeff", "")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s.strip())
    return s.upper()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    # WF: normalize prices that come as fraction-of-par (0.9875) to percent-of-par (98.75)
    df = normalize_price_to_percent_of_par(df)

    return df



import re
import pandas as pd

def clean_cusip(x) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).upper().strip()
    s = s.replace("\u00A0", "")  # non-breaking space
    s = re.sub(r"[^0-9A-Z]", "", s)  # keep only alnum
    if len(s) < 9:
        return None
    return s[:9]



def try_read_csv(path: str) -> pd.DataFrame:
    # Prefer dtype=str so we don't lose leading zeros / formatting
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, dtype=str, encoding="latin1", errors="replace")


def try_read_excel(path: str) -> pd.DataFrame:
    engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
    return pd.read_excel(path, dtype=str, engine=engine)

import re
from datetime import datetime

def parse_pricing_date_from_filename(filename: str) -> Optional[str]:
    patterns = [
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{2}-\d{2}-\d{4})",
        r"(\d{2}/\d{2}/\d{4})",
        r"([A-Za-z]{3}-\d{2}-\d{4})",
        r"(\d{8})",
        r"(\d{2}\.\d{2}\.\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, filename)

        if not m:
            continue
        token = m.group(1)
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
                dt = datetime.strptime(token, "%Y-%m-%d")
            elif re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", token):
                dt = datetime.strptime(token, "%m.%d.%Y")
            elif re.fullmatch(r"\d{2}-\d{2}-\d{4}", token):
                dt = datetime.strptime(token, "%m-%d-%Y")
            elif re.fullmatch(r"\d{2}/\d{2}/\d{4}", token):
                dt = datetime.strptime(token, "%m/%d/%Y")
            elif re.fullmatch(r"[A-Za-z]{3}-\d{2}-\d{4}", token):
                dt = datetime.strptime(token, "%b-%d-%Y")
            elif re.fullmatch(r"\d{8}", token):
                # Interpret YYYYMMDD if first 4 digits look like a year
                y = int(token[:4])
                if 2000 <= y <= 2100:
                    dt = datetime.strptime(token, "%Y%m%d")
                else:
                    dt = datetime.strptime(token, "%m%d%Y")
            else:
                continue
            if 2000 <= dt.year <= 2100:
                return dt.strftime("%Y-%m-%d")

        except Exception:
            continue
    return None


def coerce_coupon_from_desc(desc) -> Optional[float]:
    if desc is None or (isinstance(desc, float) and pd.isna(desc)):
        return None

    text = str(desc)

    # Require an explicit percent sign to avoid years / series numbers
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if not m:
        return None

    v = float(m.group(1))

    # sanity bounds for muni coupons
    if v <= 0 or v > 20:
        return None

    return v / 100.0

def coerce_maturity_from_desc(desc) -> Optional[str]:
    if desc is None or (isinstance(desc, float) and pd.isna(desc)):
        return None
    s = str(desc)

    # Common patterns:
    # "DUE 09/01/2051" or "DUE 11/01/2047"
    m = re.search(r"\bDUE\s+(\d{2})/(\d{2})/(\d{4})\b", s, flags=re.IGNORECASE)
    if m:
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    # "DUE 2051-09-01"
    m = re.search(r"\bDUE\s+(\d{4})-(\d{2})-(\d{2})\b", s, flags=re.IGNORECASE)
    if m:
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"

    return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

import pandas as pd
import numpy as np
import re

_NUMERIC_COLUMNS_DEFAULT = [
    "QTY",
    "COUPON",
    "BASIS PRICE",
    "BASIS VALUE",
    "MRKT PRICE",
    "MRKT VALUE",
    "ANNUAL_INTEREST",
]

import pandas as pd
import numpy as np

def normalize_date_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Normalize date columns to YYYY-MM-DD.

    Handles:
      - normal date strings (YYYY-MM-DD, MM/DD/YYYY, etc.)
      - Excel serial dates (e.g., 45234) when files were read with dtype=str
    """
    out = df.copy()

    for c in cols:
        if c not in out.columns:
            continue

        s = out[c].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})

        # First pass: parse typical date strings
        dt = pd.to_datetime(s, errors="coerce")

        # Second pass: Excel serial numbers (common if read with dtype=str)
        # Excel serial is usually 5 digits (but can be 4-6 depending on range)
        needs_excel = dt.isna() & s.str.fullmatch(r"\d{4,6}", na=False)
        if needs_excel.any():
            serial = pd.to_numeric(s[needs_excel], errors="coerce")
            # Excel's day 0 is 1899-12-30 for pandas' origin convention
            dt_excel = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
            dt.loc[needs_excel] = dt_excel

        out[c] = dt.dt.strftime("%Y-%m-%d")

    return out


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Convert common money/number/text fields to real numeric dtype.

    Handles:
      - commas: "25,000"
      - currency: "$25,007.25"
      - percents: "5.15%" -> 5.15 (caller can /100 if desired)
      - parentheses negatives: "(1,234.56)" -> -1234.56
      - blanks/"—"/"N/A" -> NaN
    """
    cols = cols or _NUMERIC_COLUMNS_DEFAULT
    out = df.copy()

    for c in cols:
        if c not in out.columns:
            continue

        s = out[c].astype(str).str.strip()

        # standard missing tokens
        s = s.replace(
            {"": np.nan, "nan": np.nan, "None": np.nan, "N/A": np.nan, "NA": np.nan, "--": np.nan, "—": np.nan}
        )

        # (123.45) -> -123.45
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

        # remove $ and commas
        s = s.str.replace("$", "", regex=False).str.replace(",", "", regex=False)

        # keep trailing % for now; remove it (value remains in percent units)
        s = s.str.replace("%", "", regex=False)

        out[c] = pd.to_numeric(s, errors="coerce")

    return out
# src/muni_core/io/broker_positions/utils.py

import numpy as np
import pandas as pd

PRICE_COLS = ["MRKT PRICE", "BASIS PRICE", "ACQ PRICE", "OUTFLOW PRICE"]

def normalize_price_to_percent_of_par(out: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price columns so that 100.00 = par across ALL brokers.
    WF can represent prices in:
      - percent-of-par (98.75)
      - fraction-of-par (0.9875)  => x100
      - decimal-percent (0.009875) => x10000
    We use MRKT VALUE / PAR to infer the correct scale per-row.
    """
    df = out.copy()

    if "BROKER" not in df.columns:
        return df

    wf_mask = df["BROKER"].astype(str).str.upper().str.startswith("WF")
    if not wf_mask.any():
        return df

    # Need these to infer implied price
    need_cols = {"MRKT VALUE", "QTY"}
    if not need_cols.issubset(df.columns):
        return df

    mv = pd.to_numeric(df["MRKT VALUE"], errors="coerce")
    par = pd.to_numeric(df["QTY"], errors="coerce")

    # implied clean price (percent of par)
    implied = 100.0 * (mv / par)
    implied = implied.where((mv.notna()) & (par.notna()) & (par != 0))

    scale_candidates = (1.0, 100.0, 10000.0)

    for c in PRICE_COLS:
        if c not in df.columns:
            continue

        s = pd.to_numeric(df[c], errors="coerce")

        # only adjust WF rows where we can infer implied price and price is present
        m = wf_mask & s.notna() & implied.notna()

        if not m.any():
            continue

        s_m = s.loc[m]
        implied_m = implied.loc[m]

        # pick the scale that minimizes absolute error vs implied
        best = None
        best_err = None
        for k in scale_candidates:
            cand = s_m * k
            err = (cand - implied_m).abs()
            if best is None:
                best = cand
                best_err = err
            else:
                pick = err < best_err
                best = best.where(~pick, cand)
                best_err = best_err.where(~pick, err)

        df.loc[m, c] = best.round(6)

    return df

