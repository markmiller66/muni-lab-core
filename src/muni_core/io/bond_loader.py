from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from muni_core.config import AppConfig
from muni_core.model import Bond, CallFeature


@dataclass
class BondInputConfig:
    """
    Configuration describing how to map Excel columns to Bond fields.

    You can either:
      - construct this manually, or
      - use BondInputConfig.from_app_config(AppConfig) to load from
        the MUNI_MASTER_BUCKET ColumnMap sheet.
    """
    cusip_col: str = "CUSIP"
    rating_col: str = "RATING"
    rating_num_col: str = "RatingNum"
    basis_col: str = "BasisText"

    settle_date_col: Optional[str] = "SettleDate"
    maturity_date_col: str = "MATURITY"

    call_date_col: Optional[str] = "CALL_DATE"
    call_price_col: Optional[str] = "CALL_PRICE"  # optional; defaults to 100 if missing

    coupon_col: str = "Coupon"
    clean_price_col: str = "MarketPrice_Clean"
    quantity_col: Optional[str] = "Quantity"      # optional; defaults to 100_000 if missing

    default_quantity: float = 100_000.0

    def normalize_dates(self, df: pd.DataFrame, col: Optional[str]) -> Optional[pd.Series]:
        """
        Convert a column to datetime.date if present; otherwise return None.
        """
        if col is None or col not in df.columns:
            return None
        dt = pd.to_datetime(df[col], errors="coerce")
        return dt.dt.date

    @classmethod
    def from_app_config(cls, app_cfg: AppConfig) -> "BondInputConfig":
        """
        Build BondInputConfig from the ColumnMap sheet in MUNI_MASTER_BUCKET.

        The ColumnMap sheet should map logical names (e.g. 'cusip_col') to
        the actual Excel header names in your bond list.
        """
        colmap = app_cfg.load_column_map()

        return cls(
            cusip_col=colmap.get("cusip_col", "CUSIP"),
            rating_col=colmap.get("rating_col", "RATING"),
            rating_num_col=colmap.get("rating_num_col", "RatingNum"),
            basis_col=colmap.get("basis_col", "BasisText"),
            settle_date_col=colmap.get("settle_date_col", "SettleDate"),
            maturity_date_col=colmap.get("maturity_date_col", "MATURITY"),
            call_date_col=colmap.get("call_date_col", "CALL_DATE"),
            call_price_col=colmap.get("call_price_col", "CALL_PRICE"),
            coupon_col=colmap.get("coupon_col", "Coupon"),
            clean_price_col=colmap.get("clean_price_col", "MarketPrice_Clean"),
            quantity_col=colmap.get("quantity_col", "Quantity"),
        )


def _safe_get(row: pd.Series, col: Optional[str], default=None):
    if col is None or col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


def load_bonds_from_excel(
    file_path: Path,
    sheet_name: str = 0,
    cfg: Optional[BondInputConfig] = None,
) -> Tuple[List[Bond], pd.DataFrame]:
    """
    Load bonds from an Excel workbook and convert to a list of Bond objects.

    Returns:
        bonds: list[Bond]
        df:    the underlying DataFrame (with parsed date columns attached)
    """
    if cfg is None:
        cfg = BondInputConfig()

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Normalize date columns (add derived *_parsed columns for safety)
    settle_parsed = cfg.normalize_dates(df, cfg.settle_date_col)
    mat_parsed = cfg.normalize_dates(df, cfg.maturity_date_col)
    call_parsed = cfg.normalize_dates(df, cfg.call_date_col)

    if settle_parsed is not None:
        df["_SettleDate_parsed"] = settle_parsed
    if mat_parsed is not None:
        df["_Maturity_parsed"] = mat_parsed
    if call_parsed is not None:
        df["_CallDate_parsed"] = call_parsed

    bonds: List[Bond] = []

    today = date.today()

    for _, row in df.iterrows():
        cusip = str(_safe_get(row, cfg.cusip_col, "")).strip()
        if not cusip:
            # skip rows without CUSIP
            continue

        rating = str(_safe_get(row, cfg.rating_col, "")).strip()

        # We now assume your bond file has a summary rating number column
        # (e.g. SUMMARY_R_NUM), mapped via ColumnMap to rating_num_col.
        rating_num_val = _safe_get(row, cfg.rating_num_col, None)
        try:
            rating_num = int(rating_num_val) if rating_num_val is not None else 0
        except (TypeError, ValueError):
            rating_num = 0

        basis = str(_safe_get(row, cfg.basis_col, "Actual/Actual"))

        # Dates
        settle_dt: Optional[date] = None
        mat_dt: Optional[date] = None
        call_dt: Optional[date] = None

        if "_SettleDate_parsed" in df.columns:
            settle_dt = row["_SettleDate_parsed"]
        if "_Maturity_parsed" in df.columns:
            mat_dt = row["_Maturity_parsed"]
        if "_CallDate_parsed" in df.columns:
            call_dt = row["_CallDate_parsed"]

        # If maturity is missing, we really cannot price → skip this bond
        if mat_dt is None:
            continue

        # If settle date is missing, default to "today" for this portfolio run
        if settle_dt is None:
            settle_dt = today

        # Coupon and prices
        coupon_val = _safe_get(row, cfg.coupon_col, 0.0)
        try:
            coupon = float(coupon_val)
        except (TypeError, ValueError):
            coupon = 0.0

        clean_price_val = _safe_get(row, cfg.clean_price_col, 100.0)
        try:
            clean_price = float(clean_price_val)
        except (TypeError, ValueError):
            clean_price = 100.0

        qty_val = _safe_get(row, cfg.quantity_col, cfg.default_quantity)
        try:
            quantity = float(qty_val)
        except (TypeError, ValueError):
            quantity = cfg.default_quantity

        # Call feature
        call_price_val = _safe_get(row, cfg.call_price_col, 100.0)
        try:
            call_price = float(call_price_val)
        except (TypeError, ValueError):
            call_price = 100.0

        call_feature: Optional[CallFeature] = None
        if call_dt is not None:
            call_feature = CallFeature(call_date=call_dt, call_price=call_price)

        bond = Bond(
            cusip=cusip,
            rating=rating,
            rating_num=rating_num,
            basis=basis,
            settle_date=settle_dt,
            maturity_date=mat_dt,
            coupon=coupon,
            clean_price=clean_price,
            quantity=quantity,
            call_feature=call_feature,
        )
        bonds.append(bond)

    return bonds, df
