"""
loaders.py

Broker-specific file loaders and standardization logic.

Responsibilities:
- Read raw broker files (CSV/XLS/XLSX)
- Normalize column names
- Apply broker-specific rename maps
- Return standardized DataFrames

Dependency contract:
- Allowed imports: utils, pandas, standard library
- Forbidden imports: config, build_positions, enrich_with_msrb
"""

from __future__ import annotations

import os
import re
from io import StringIO

import pandas as pd

from muni_core.io.broker_positions.utils import (
    normalize_columns,
    try_read_csv,
    try_read_excel,
    parse_pricing_date_from_filename,
    coerce_coupon_from_desc,
    coerce_maturity_from_desc,
)


# IMPORTANT: Keys should match normalized_columns (UPPER)
RENAME_MAP: dict[str, dict[str, str]] = {
    "CHASE": {
        "CUSIP": "CUSIP",
        "DESCRIPTION": "DESCRIPTION",
        "QUANTITY": "QTY",
        "COUPON RATE": "COUPON",
        "MATURITY DATE": "MATURITY",
        "MOODY RATING": "MOODYS",
        "S&P RATING": "S&P",
        "BUY/CALL AMOUNT": "CALL_PRICE",  # not a date
        "ACQUISITION DATE": "ACQ DATE",
        "PRICING DATE": "PRICING DATE",
        "PRICE": "MRKT PRICE",
        "VALUE": "MRKT VALUE",
        "LOCAL PRICE": "MRKT PRICE",
        "LOCAL VALUE": "MRKT VALUE",
        "LOCAL UNIT COST": "BASIS PRICE",
        "ORIG COST (LOCAL)": "BASIS VALUE",

    },
    "FID": {
        "SYMBOL": "CUSIP",
        "DESCRIPTION": "DESCRIPTION",
        "QUANTITY": "QTY",
        "AVERAGE COST BASIS": "BASIS PRICE",
        "COST BASIS TOTAL": "BASIS VALUE",
        "CURRENT VALUE": "MRKT VALUE",
        "LAST PRICE": "MRKT PRICE",
    },
    "MERRILL": {
        "CUSIP #": "CUSIP",
        "SECURITY DESCRIPTION": "DESCRIPTION",
        "QUANTITY": "QTY",
        "UNIT COST ($)": "BASIS PRICE",
        "COST BASIS ($)": "BASIS VALUE",
        "VALUE ($)": "MRKT VALUE",
        "ACQUISITION DATE": "ACQ DATE",
        "COB DATE": "PRICING DATE",
    },
    "RJ": {
    "SYMBOL/CUSIP": "CUSIP",
    "DESCRIPTION": "DESCRIPTION",
    "QUANTITY": "QTY",
    "DELAYED PRICE": "MRKT PRICE",
    "CURRENT VALUE": "MRKT VALUE",
    "UNIT COST": "BASIS PRICE",
    "TOTAL COST BASIS(†)": "BASIS VALUE",
    # Optional if you want income:
    "ESTIMATED ANNUAL INCOME": "ANNUAL_INCOME_RJ",
    "TIME HELD": "TIME_HELD_RJ",
    "PRODUCT TYPE": "PRODUCT_TYPE_RJ",
},
    "SCHWAB": {
        "SYMBOL": "CUSIP",
        "DESCRIPTION": "DESCRIPTION",
        "QTY (QUANTITY)": "QTY",
        "QUANTITY": "QTY",
        "COST/SHARE": "BASIS PRICE",
        "COST BASIS": "BASIS VALUE",
        "MKT VAL (MARKET VALUE)": "MRKT VALUE",
        "MARKET VALUE": "MRKT VALUE",
        "PRICE": "MRKT PRICE",
        "EXP/MAT (EXPIRATION/MATURITY)": "MATURITY",
        "EXPIRATION/MATURITY": "MATURITY",
        "RATINGS": "RATINGS_RAW",
    },
    "WF": {
        "CUSIP": "CUSIP",
        "DESCRIPTION": "DESCRIPTION",
        "QUANTITY": "QTY",
        "COUPON RATE": "COUPON",
        "MATURITY DATE": "MATURITY",
        "TRADE PRICE (%)": "BASIS PRICE",
        "ORIGINAL TOTAL COST": "BASIS VALUE",
        "ESTIMATED MARKET VALUE": "MRKT VALUE",
        "ESTIMATED PRICE (%)": "MRKT PRICE",
        "CALL DATE": "CALL_DATE",
        "TRADE DATE1": "ACQ DATE",
        "PRICED AS OF DATE": "PRICING DATE",
        "S&P/MOODY'S RATING": "RATINGS_RAW",
    },
}


def load_schwab(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    header_indices = []
    for i, line in enumerate(lines):
        l = line.strip().lstrip("\ufeff")
        if re.search(r"\bSymbol\b", l, re.I) and re.search(r"\bDescription\b", l, re.I) and "," in l:
            header_indices.append(i)

    parts: list[pd.DataFrame] = []
    for start in header_indices:
        end = start + 1
        while end < len(lines) and "Account Total" not in lines[end]:
            end += 1

        block = lines[start:end]
        if len(block) <= 1:
            continue

        df = pd.read_csv(StringIO("".join(block)), dtype=str)
        df = normalize_columns(df)

        if "SYMBOL" in df.columns:
            sym = df["SYMBOL"].astype(str)
            df = df[df["SYMBOL"].notna() & ~sym.str.contains(
                r"Cash & Cash Investments|Account Total|Positions for",
                case=False, regex=True, na=False
            )]

        if not df.empty:
            parts.append(df)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_rj(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str)
    header_row = None
    for idx in raw.index:
        row = raw.loc[idx].astype(str).str.upper().str.strip().tolist()
        if "SYMBOL/CUSIP" in row and "DESCRIPTION" in row:
            header_row = idx
            break
    if header_row is None:
        return pd.DataFrame()

    header = raw.loc[header_row].tolist()
    df = pd.DataFrame(raw.loc[header_row + 1:].values, columns=header)
    return normalize_columns(df)


def load_wf(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, sep="\t", encoding="latin1", header=None, dtype=str)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", encoding="latin1", header=None, dtype=str)
    else:
        df = pd.read_excel(path, header=None, dtype=str,
                           engine=("xlrd" if path.lower().endswith(".xls") else "openpyxl"))

    # Find "Fixed Income" section
    mask_fixed = df.apply(lambda r: r.astype(str).str.contains("Fixed Income", na=False).any(), axis=1)
    fixed_rows = df[mask_fixed].index
    if fixed_rows.empty:
        return pd.DataFrame()

    fixed_income_row = fixed_rows[0]
    header_row = fixed_income_row + 1

    mask_total = df.apply(lambda r: r.astype(str).str.contains("Total Fixed Income", na=False).any(), axis=1)
    total_rows = df[mask_total].index
    total_row = total_rows[0] if not total_rows.empty else len(df)

    header = [c for c in df.iloc[header_row].tolist() if pd.notna(c)]
    data = df.iloc[header_row + 1: total_row, 0:len(header)].values
    out = pd.DataFrame(data, columns=header)
    out = normalize_columns(out)


    if "CUSIP" in out.columns:
        out = out[out["CUSIP"].notna()]

    return out


def load_one_file(broker: str, path: str) -> pd.DataFrame:
    broker = broker.upper().strip()

    if broker == "SCHWAB":
        return load_schwab(path)
    if broker == "RJ":
        return load_rj(path)
    if broker == "WF":
        return load_wf(path)

    if path.lower().endswith(".csv"):
        df = try_read_csv(path)
    else:
        df = try_read_excel(path)
    return normalize_columns(df)

def standardize_positions(df: pd.DataFrame, broker: str, filename: str) -> pd.DataFrame:
    broker = broker.upper().strip()
    df = normalize_columns(df)

    rename_map = RENAME_MAP.get(broker)
    if not rename_map:
        raise ValueError(f"No rename mapping for broker={broker}. Columns={df.columns.tolist()}")

    # Apply mapping
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Pricing date fallback from filename
    if "PRICING DATE" not in df.columns or df["PRICING DATE"].isna().all():
        p = parse_pricing_date_from_filename(filename)
        if p:
            df["PRICING DATE"] = p

    # CHASE: market fields sometimes vary; ensure MRKT fields populated
    if broker == "CHASE":
        if "MRKT PRICE" not in df.columns:
            df["MRKT PRICE"] = pd.NA
        if "MRKT VALUE" not in df.columns:
            df["MRKT VALUE"] = pd.NA

        # Prefer already-mapped MRKT values; otherwise fill from alternates if they exist
        for alt in ("PRICE", "LOCAL PRICE"):
            if alt in df.columns:
                df["MRKT PRICE"] = df["MRKT PRICE"].combine_first(df[alt])
                break

        for alt in ("VALUE", "LOCAL VALUE"):
            if alt in df.columns:
                df["MRKT VALUE"] = df["MRKT VALUE"].combine_first(df[alt])
                break

    # Coupon fallback from description (MSRB will override/fill later anyway)
    if ("COUPON" not in df.columns or df["COUPON"].isna().all()) and "DESCRIPTION" in df.columns:
        # Coupon fallback from description:
        # - Skip RJ because it has no coupon column and description parsing can misfire on years/series
        # - Let MSRB enrichment fill RJ coupons
        if broker != "RJ":
            if ("COUPON" not in df.columns or df["COUPON"].isna().all()) and "DESCRIPTION" in df.columns:
                df["COUPON"] = df["DESCRIPTION"].map(coerce_coupon_from_desc)
        else:
            # Ensure COUPON exists but stays empty so MSRB can fill later
            if "COUPON" not in df.columns:
                df["COUPON"] = pd.NA

    df["BROKER"] = broker

    # Maturity fallback from description (helps RJ/FID/MERRILL a lot)
    if ("MATURITY" not in df.columns or df["MATURITY"].isna().all()) and "DESCRIPTION" in df.columns:
        df["MATURITY"] = df["DESCRIPTION"].map(coerce_maturity_from_desc)

    return df
