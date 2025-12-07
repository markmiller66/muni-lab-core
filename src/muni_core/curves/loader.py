from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .types import ZeroCurve
from muni_core.config import AppConfig, CurvesConfig


@dataclass
class CurveConfig:
    """
    Simple config object describing where to find curve data.
    """
    wide_curve_file: Path
    wide_curve_sheet: str
    spot_curve_file: Path
    spot_curve_sheet: str
    curve_strategy: str = "excel_curves_wide"


def load_zero_curve_from_spot_excel(
    file_path: Path,
    sheet_name: str = "AAA_Spot",
    tenor_col: str = "TenorY",
    rate_col: str = "ZeroRate",
) -> ZeroCurve:
    """
    Load AAA spot/zero curve from an Excel sheet with columns:

    - tenor_col: years to maturity (float)
    - rate_col: zero rate in DECIMAL (e.g. 0.02203 for 2.203%)
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df[[tenor_col, rate_col]].dropna()

    pairs = list(zip(df[tenor_col].astype(float), df[rate_col].astype(float)))
    return ZeroCurve.from_pairs(pairs)


def build_default_curve_config(root: Optional[Path] = None) -> CurveConfig:
    """
    Simple fallback helper if we are NOT using YAML yet.

    Assumes an Excel file at:
        <repo root>/data/AAA_MUNI_CURVE/aaa_curves.xlsx
    """
    if root is None:
        # repo root ~ 3 levels up from this file: curves -> muni_core -> src -> repo
        root = Path(__file__).resolve().parents[3]

    curve_file = root / "data" / "AAA_MUNI_CURVE" / "aaa_curves.xlsx"

    return CurveConfig(
        wide_curve_file=curve_file,
        wide_curve_sheet="Curves_Wide",
        spot_curve_file=curve_file,
        spot_curve_sheet="AAA_Spot",
        curve_strategy="excel_curves_wide",
    )


def load_zero_curve_from_app_config(app_cfg: AppConfig) -> ZeroCurve:
    """
    Convenience function: given an AppConfig, load the AAA spot zero curve.
    """
    curves: CurvesConfig = app_cfg.curves
    return load_zero_curve_from_spot_excel(
        file_path=curves.spot_curve_file,
        sheet_name=curves.spot_curve_sheet,
    )
