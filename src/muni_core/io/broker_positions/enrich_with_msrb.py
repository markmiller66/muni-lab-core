"""
enrich_with_msrb.py

Stage 2 pipeline: enrich broker positions with MSRB reference data.

Workflow:
- Load standardized broker positions
- Load MSRB reference file
- Merge on cleaned CUSIP
- Fill missing reference fields (coupon/call/maturity/ratings)
- Write enriched output

Dependency contract:
- Allowed imports: config, utils, pandas, standard library
- Forbidden imports: loaders, build_positions
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from muni_core.io.broker_positions.config import MSRB_FILE, POSITIONS_OUT, ENRICHED_OUT
from muni_core.io.broker_positions.utils import (
    coerce_numeric_columns,
    normalize_columns,
    clean_cusip,
    try_read_csv,
    try_read_excel,
)


def build_msrb_append_template(punch: pd.DataFrame) -> pd.DataFrame:
    """
    Build rows to append into the MSRB reference template.
    Keeps the MSRB columns you care about and fills what we can from broker data.
    """

    cols_needed = ["CUSIP", "DESCRIPTION", "COUPON", "CALL_DATE", "MATURITY", "MOODYS", "S&P"]
    for c in cols_needed:
        if c not in punch.columns:
            punch[c] = pd.NA

    # These are the MSRB-style columns (normalized / uppercase)
    cols = [
        "CUSIP",
        "DESCRIPTION",
        "COUPON",
        "CALL",
        "MATURITY",
        "MOODYS CURRENT",
        "S&P CURRENT",
    ]



    # Build template rows (blank coupon/call unless broker already had them)
    tmpl = punch[["CUSIP", "DESCRIPTION", "COUPON", "CALL_DATE", "MATURITY", "MOODYS", "S&P"]].copy()
    tmpl = tmpl.rename(columns={
        "CALL_DATE": "CALL",
        "MOODYS": "MOODYS CURRENT",
        "S&P": "S&P CURRENT",
    })

    # Ensure required columns exist
    for c in cols:
        if c not in tmpl.columns:
            tmpl[c] = pd.NA

    # Keep only desired ordering
    tmpl = tmpl[cols]

    # Dedup by CUSIP
    tmpl["CUSIP_CLEAN"] = tmpl["CUSIP"].map(clean_cusip)
    tmpl = tmpl.dropna(subset=["CUSIP_CLEAN"]).copy()
    tmpl = tmpl.sort_values("CUSIP_CLEAN").drop_duplicates(subset=["CUSIP_CLEAN"], keep="first")
    tmpl = tmpl.drop(columns=["CUSIP_CLEAN"])

    return tmpl



def load_msrb(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = try_read_csv(path)
    else:
        df = try_read_excel(path)

    df = normalize_columns(df)

    # -------------------------
    # Build MSRB reference safely (no duplicate columns)
    # -------------------------
    ref = pd.DataFrame()

    # CUSIP (required)
    if "CUSIP" not in df.columns:
        raise ValueError("MSRB reference missing CUSIP column after normalization.")
    ref["CUSIP_MSRB"] = df["CUSIP"]

    # Coupon
    if "COUPON" in df.columns:
        ref["COUPON_MSRB"] = df["COUPON"]

    # Call date (prefer explicit CALL DATE variants)
    for c in ["CALL", "CALL DATE"]:
        if c in df.columns:
            ref["CALL_DATE_MSRB"] = df[c]
            break

    # Maturity (THIS IS THE KEY FIX)
    for c in ["MATURITY", "MATURITY DATE", "EXTRACT MATURITY", "FINAL MATURITY"]:
        if c in df.columns:
            ref["MATURITY_MSRB"] = df[c]
            break

    # Ratings
    if "MOODYS CURRENT" in df.columns:
        ref["MOODYS_MSRB"] = df["MOODYS CURRENT"]

    if "S&P CURRENT" in df.columns:
        ref["S&P_MSRB"] = df["S&P CURRENT"]

    # clean key
    ref["CUSIP_CLEAN"] = ref["CUSIP_MSRB"].map(clean_cusip)
    ref = ref.dropna(subset=["CUSIP_CLEAN"]).copy()

    # normalize coupon to numeric + decimal (0.0515)
    if "COUPON_MSRB" in ref.columns:
        ref["COUPON_MSRB"] = pd.to_numeric(ref["COUPON_MSRB"], errors="coerce")
        ref.loc[ref["COUPON_MSRB"] > 1, "COUPON_MSRB"] = ref["COUPON_MSRB"] / 100.0

    # De-dupe by CUSIP
    ref = ref.groupby("CUSIP_CLEAN", as_index=False).first()
    return ref



def enrich_with_msrb() -> pd.DataFrame:
    pos = pd.read_excel(POSITIONS_OUT, dtype=str)
    pos = normalize_columns(pos)
    pos["CUSIP_CLEAN"] = pos["CUSIP"].map(clean_cusip)

    msrb = load_msrb(MSRB_FILE)
    out = pos.merge(msrb, on="CUSIP_CLEAN", how="left")

    # MSRB is source of truth
    if "COUPON_MSRB" in out.columns:
        out["COUPON"] = out["COUPON_MSRB"]

    if "CALL_DATE_MSRB" in out.columns:
        out["CALL_DATE"] = out["CALL_DATE_MSRB"]

    if "MATURITY_MSRB" in out.columns:
        out["MATURITY"] = out["MATURITY_MSRB"]

    # normalize dates after forcing
    try:
        from muni_core.io.broker_positions.utils import normalize_date_columns
        out = normalize_date_columns(out, cols=["PRICING DATE", "ACQ DATE", "CALL_DATE", "MATURITY"])
    except Exception:
        pass

    # Treat blank strings as missing so combine_first can fill from MSRB
    for c in ["COUPON", "CALL_DATE", "MATURITY", "MOODYS", "S&P"]:
        if c in out.columns:
            out[c] = out[c].replace(r"^\s*$", pd.NA, regex=True)

    # -------------------------
    # Punch sheet + auto-append template
    # -------------------------
    from muni_core.io.broker_positions.config import (
        PUNCH_OUT, MSRB_PUNCH_OUT, MSRB_APPEND_TEMPLATE_OUT, MSRB_REFERENCE_UPDATED_OUT
    )

    punch_cols = [
        "CUSIP", "CUSIP_CLEAN", "BROKER", "DESCRIPTION", "PRICING DATE",
        "COUPON", "CALL_DATE", "MATURITY",
        "MOODYS", "S&P",
        "COUPON_MSRB", "CALL_DATE_MSRB", "MATURITY_MSRB",
        "MOODYS_MSRB", "S&P_MSRB",
    ]
    for c in punch_cols:
        if c not in out.columns:
            out[c] = pd.NA

    needs_msrb = (
        out["COUPON_MSRB"].isna()
        | out["CALL_DATE_MSRB"].isna()
        | out["MATURITY_MSRB"].isna()
    )

    punch = out.loc[needs_msrb, punch_cols].copy()

    punch["MISSING_COUPON_MSRB"] = punch["COUPON_MSRB"].isna()
    punch["MISSING_CALLDATE_MSRB"] = punch["CALL_DATE_MSRB"].isna()
    punch["MISSING_MATURITY_MSRB"] = punch["MATURITY_MSRB"].isna()
    punch["MISSING_MOODYS_MSRB"] = punch["MOODYS_MSRB"].isna()
    punch["MISSING_SP_MSRB"] = punch["S&P_MSRB"].isna()

    punch = punch.sort_values(["CUSIP", "BROKER"]).drop_duplicates(subset=["CUSIP"], keep="first")

    # Write punch sheets
    punch.to_excel(MSRB_PUNCH_OUT, index=False)
    print(f"Wrote: {MSRB_PUNCH_OUT} rows={len(punch)}")

    punch.to_excel(PUNCH_OUT, index=False)
    print(f"Wrote: {PUNCH_OUT} rows={len(punch)}")

    # Build append-ready template and updated ref
    append_rows = build_msrb_append_template(punch)
    append_rows.to_excel(MSRB_APPEND_TEMPLATE_OUT, index=False)
    print(f"Wrote: {MSRB_APPEND_TEMPLATE_OUT} rows={len(append_rows)}")

    updated_ref = write_updated_msrb_reference(MSRB_FILE, append_rows)
    updated_ref.to_excel(MSRB_REFERENCE_UPDATED_OUT, index=False)
    print(f"Wrote: {MSRB_REFERENCE_UPDATED_OUT} rows={len(updated_ref)}")

    # -------------------------
    # Apply enrichment from MSRB onto broker fields
    # -------------------------

    # RJ coupons are junk (your 20.xx issue) -> always allow MSRB to fill
    if "BROKER" in out.columns:
        out.loc[out["BROKER"].eq("RJ"), "COUPON"] = pd.NA

    # -------------------------
    # Coupon numeric cleanup (FIXED)
    # -------------------------
    # RJ coupons are junk (your 20.xx issue) -> always allow MSRB to fill
    if "BROKER" in out.columns:
        out.loc[out["BROKER"].eq("RJ"), "COUPON"] = pd.NA

    # Parse numeric
    out["COUPON"] = pd.to_numeric(out.get("COUPON"), errors="coerce")
    out["COUPON_MSRB"] = pd.to_numeric(out.get("COUPON_MSRB"), errors="coerce")

    # Normalize MSRB coupon percent-points -> decimal
    out.loc[out["COUPON_MSRB"] > 1, "COUPON_MSRB"] = out["COUPON_MSRB"] / 100.0

    # For NON-RJ brokers, treat 1..20 as percent coupons and convert to decimal
    if "BROKER" in out.columns:
        non_rj = ~out["BROKER"].eq("RJ")
    else:
        non_rj = pd.Series(True, index=out.index)

    pct_mask = non_rj & out["COUPON"].between(1.0, 20.0, inclusive="both")
    out.loc[pct_mask, "COUPON"] = out.loc[pct_mask, "COUPON"] / 100.0

    # Still treat 0 or >20 as missing (sanity)
    out.loc[out["COUPON"].fillna(0).eq(0), "COUPON"] = np.nan
    out.loc[out["COUPON"] > 0.20, "COUPON"] = np.nan  # catches junk like 20.xx, 99, etc


    # Fill from MSRB when broker coupon missing
    out["COUPON"] = out["COUPON"].combine_first(out["COUPON_MSRB"])

    # Call date
    if "CALL_DATE" not in out.columns:
        out["CALL_DATE"] = pd.NA
    out["CALL_DATE"] = out["CALL_DATE"].combine_first(out.get("CALL_DATE_MSRB"))

    # Maturity
    if "MATURITY" not in out.columns:
        out["MATURITY"] = pd.NA
    out["MATURITY"] = out["MATURITY"].combine_first(out.get("MATURITY_MSRB"))

    # Ratings
    if "MOODYS" not in out.columns:
        out["MOODYS"] = pd.NA
    if "S&P" not in out.columns:
        out["S&P"] = pd.NA
    out["MOODYS"] = out["MOODYS"].combine_first(out.get("MOODYS_MSRB"))
    out["S&P"] = out["S&P"].combine_first(out.get("S&P_MSRB"))

    # Annual interest (par * coupon)
    if "QTY" in out.columns and "COUPON" in out.columns:
        qty_num = pd.to_numeric(out["QTY"].astype(str).str.replace(",", "", regex=False), errors="coerce")
        out["ANNUAL_INTEREST"] = qty_num * out["COUPON"]
    else:
        out["ANNUAL_INTEREST"] = pd.NA

    # Drop helper columns you don't want in final
    drop_cols = [c for c in [
        "CUSIP_CLEAN", "CUSIP_MSRB",
        "COUPON_MSRB", "CALL_DATE_MSRB", "MATURITY_MSRB",
        "MOODYS_MSRB", "S&P_MSRB",
    ] if c in out.columns]
    out = out.drop(columns=drop_cols)

    # Coerce numerics for Excel math
    out = coerce_numeric_columns(out, cols=[
        "QTY", "COUPON",
        "BASIS PRICE", "BASIS VALUE",
        "MRKT PRICE", "MRKT VALUE",
        "ANNUAL_INTEREST",
    ])

    # Date standardization (if you have normalize_date_columns)
    try:
        from muni_core.io.broker_positions.utils import normalize_date_columns
        out = normalize_date_columns(out, cols=["PRICING DATE", "ACQ DATE", "CALL_DATE", "MATURITY"])
    except Exception:
        # keep going even if that helper isn't present yet
        pass

    out.to_excel(ENRICHED_OUT, index=False)
    print(f"    -> Wrote: {ENRICHED_OUT} rows={len(out)}")

    return out



def write_updated_msrb_reference(existing_path: str, append_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Create a NEW updated MSRB reference file by appending missing CUSIPs.
    (Does not overwrite your original reference.)
    """
    base = try_read_excel(existing_path)
    base = normalize_columns(base)

    # normalize columns of append_rows to match base
    app = append_rows.copy()
    app = normalize_columns(app)

    # Ensure CUSIP column exists
    if "CUSIP" not in base.columns:
        raise ValueError("MSRB reference file missing CUSIP column after normalization.")
    if "CUSIP" not in app.columns:
        raise ValueError("Append rows missing CUSIP column after normalization.")

    base["CUSIP_CLEAN"] = base["CUSIP"].map(clean_cusip)
    app["CUSIP_CLEAN"] = app["CUSIP"].map(clean_cusip)

    # Only add truly missing cusips
    missing = app.loc[~app["CUSIP_CLEAN"].isin(set(base["CUSIP_CLEAN"].dropna())), :].copy()

    updated = pd.concat([base, missing], ignore_index=True)

    # Dedup by CUSIP (prefer existing base row)
    updated = updated.sort_values("CUSIP_CLEAN").drop_duplicates(subset=["CUSIP_CLEAN"], keep="first")
    updated = updated.drop(columns=["CUSIP_CLEAN"])

    return updated



def main() -> None:
    enrich_with_msrb()


if __name__ == "__main__":
    main()
