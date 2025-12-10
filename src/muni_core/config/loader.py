from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import pandas as pd


# ---------- Curves & Sources ----------

@dataclass
class CurvesConfig:
    wide_curve_file: Path
    wide_curve_sheet: str
    spot_curve_file: Path
    spot_curve_sheet: str
    curve_strategy: str

    # NEW: optional historical spot cache + as-of date
    # e.g. history_file = data/AAA_MUNI_CURVE/aaa_muni_treas_history.parquet
    # curve_asof_date = "2025-11-26"
    history_file: Optional[Path] = None
    curve_asof_date: Optional[str] = None  # keep as string; parse later where needed


@dataclass
class CurveSourcesConfig:
    """
    Raw data sources for building spot curves from Tradeweb / Fed / VIX.
    These are what your (former) spot.py will read.
    """
    data_root: Path              # e.g. "data"
    treasury_file: str           # e.g. "feds200628.csv"
    muni_file: str               # e.g. "Tradeweb_MUNI_data (4).csv"
    vix_file: Optional[str] = None  # e.g. "muni_vix_data (2).csv"

    treasury_skiprows: int = 9   # Fed CSV header rows, as in your script


# ---------- Master Bucket ----------

@dataclass
class MasterBucketConfig:
    """
    Configuration for the MUNI_MASTER_BUCKET workbook.

    Right now we primarily use the ColumnMap sheet to map logical
    field names (cusip_col, maturity_date_col, etc.) to actual Excel
    column headers in your bond list.
    """
    file: Path
    column_map_sheet: str = "ColumnMap"
    controls_sheet: str = "Controls"
    rating_map_sheet: str = "RatingMap"


@dataclass
class AppConfig:
    """
    Top-level configuration object for muni_core.

    - curves: where to find the AAA curves & history parquet
    - master_bucket: where to find MUNI_MASTER_BUCKET.xlsx
    - curve_sources: raw Tradeweb/Fed/VIX CSV sources
    """
    curves: CurvesConfig
    master_bucket: Optional[MasterBucketConfig] = None
    curve_sources: Optional[CurveSourcesConfig] = None

    # ---------- path helpers ----------

    @property
    def repo_root(self) -> Path:
        """
        Best-effort guess at the repo root.

        We assume curves.wide_curve_file looks like:
            <repo_root>/data/AAA_MUNI_CURVE/aaa_curves.xlsx
        so repo_root is three levels up from that file.
        """
        return self.curves.wide_curve_file.parent.parent.parent

    @property
    def output_root(self) -> Path:
        """
        Base output directory for everything:

            <repo_root>/output
        """
        out = self.repo_root / "output"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def dated_output_root(self) -> Path:
        """
        Base folder for this run’s curve-related outputs, partitioned by date:

            <repo_root>/output/curves/YYYY-MM-DD

        Date resolution precedence:
          1) Controls:    CURVE_ASOF_DATE  (if present)
          2) YAML:        curves.curve_asof_date
          3) Fallback:    today()
        Any time component (e.g. '2025-11-28 00:00:00') is stripped off.
        """
        from datetime import date
        import pandas as pd

        asof_raw: Optional[str] = None

        # 1) try Controls (if helper exists)
        try:
            if hasattr(self, "get_control_value"):
                val = self.get_control_value("CURVE_ASOF_DATE", default=None)
                if val:
                    asof_raw = str(val)
        except Exception:
            asof_raw = None

        # 2) YAML curves.curve_asof_date
        if not asof_raw and self.curves.curve_asof_date:
            asof_raw = str(self.curves.curve_asof_date)

        # 3) fallback: today
        if not asof_raw:
            asof_dt = date.today()
        else:
            # Normalize to date-only, in case we get 'YYYY-MM-DD 00:00:00'
            asof_dt = pd.to_datetime(asof_raw).date()

        asof_str = asof_dt.isoformat()  # always 'YYYY-MM-DD'

        out = self.output_root / "curves" / asof_str
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ---------- YAML loader and helpers follow here ----------
    # (keep your existing from_yaml, load_column_map, get_control_value, etc.)


    # ---------- constructors ----------

    @classmethod
    def from_yaml(cls, cfg_path: Path) -> "AppConfig":
        """
        Load configuration from a YAML file.

        Paths in the YAML are interpreted as relative to the repo root.
        We assume this file lives in: <repo root>/config/example_config.yaml
        """
        text = cfg_path.read_text(encoding="utf-8")
        data: Dict[str, Any] = yaml.safe_load(text) or {}

        # repo root ~ parent of the "config" directory
        repo_root = cfg_path.parent.parent

        def resolve_path(p: str) -> Path:
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            return path.resolve()

        # ----- Curves -----
        curves_data: Dict[str, Any] = data.get("curves", {}) or {}

        wide_curve_file = resolve_path(
            curves_data.get("wide_curve_file", "data/AAA_MUNI_CURVE/aaa_curves.xlsx")
        )
        spot_curve_file = resolve_path(
            curves_data.get("spot_curve_file", "data/AAA_MUNI_CURVE/aaa_curves.xlsx")
        )

        history_file_raw = curves_data.get("history_file")
        history_file: Optional[Path]
        if history_file_raw:
            history_file = resolve_path(history_file_raw)
        else:
            history_file = None

        curves_cfg = CurvesConfig(
            wide_curve_file=wide_curve_file,
            wide_curve_sheet=curves_data.get("wide_curve_sheet", "Curves_Wide"),
            spot_curve_file=spot_curve_file,
            spot_curve_sheet=curves_data.get("spot_curve_sheet", "AAA_Spot"),
            curve_strategy=curves_data.get("curve_strategy", "excel_curves_wide"),
            history_file=history_file,
            curve_asof_date=curves_data.get("curve_asof_date"),  # may be None
        )

        # ----- Master Bucket -----
        mb_cfg: Optional[MasterBucketConfig] = None
        mb_data: Dict[str, Any] = data.get("master_bucket", {}) or {}
        if mb_data:
            mb_file = resolve_path(mb_data.get("file", "config/MUNI_MASTER_BUCKET.xlsx"))
            mb_cfg = MasterBucketConfig(
                file=mb_file,
                column_map_sheet=mb_data.get("column_map_sheet", "ColumnMap"),
                controls_sheet=mb_data.get("controls_sheet", "Controls"),
                rating_map_sheet=mb_data.get("rating_map_sheet", "RatingMap"),
            )

        # ----- Curve Sources (Tradeweb / Fed / VIX) -----
        cs_cfg: Optional[CurveSourcesConfig] = None
        cs_data: Dict[str, Any] = data.get("curve_sources", {}) or {}

        if cs_data:
            data_root = resolve_path(cs_data.get("data_root", "data"))
            treasury_file = cs_data.get("treasury_file", "feds200628.csv")
            muni_file = cs_data.get("muni_file", "Tradeweb_MUNI_data (4).csv")
            vix_file = cs_data.get("vix_file")  # may be None
            treasury_skiprows = int(cs_data.get("treasury_skiprows", 9))

            cs_cfg = CurveSourcesConfig(
                data_root=data_root,
                treasury_file=treasury_file,
                muni_file=muni_file,
                vix_file=vix_file,
                treasury_skiprows=treasury_skiprows,
            )

        return cls(curves=curves_cfg, master_bucket=mb_cfg, curve_sources=cs_cfg)

    # -------------- ColumnMap helpers --------------

    def load_column_map(self) -> Dict[str, str]:
        """
        Load the ColumnMap sheet from the MUNI_MASTER_BUCKET workbook and
        return a mapping: {logical_name: excel_column_name}.
        """
        if self.master_bucket is None:
            raise ValueError("master_bucket configuration not set in YAML.")

        mb = self.master_bucket
        df = pd.read_excel(mb.file, sheet_name=mb.column_map_sheet)

        def find_col(candidates, cols):
            for c in candidates:
                if c in cols:
                    return c
            raise KeyError(
                f"Could not find any of {candidates} in ColumnMap sheet columns: {list(cols)}"
            )

        logical_col = find_col(
            ["LogicalName", "Logical Name", "logical_name", "LOGICAL_NAME"],
            df.columns,
        )
        excel_col = find_col(
            ["ExcelColumn", "Excel Column", "excel_column", "EXCEL_COLUMN"],
            df.columns,
        )

        mapping: Dict[str, str] = {}
        for _, row in df.iterrows():
            logical = str(row[logical_col]).strip()
            excel_name = str(row[excel_col]).strip()
            if logical and excel_name:
                mapping[logical] = excel_name

        return mapping

    def load_controls_df(self) -> pd.DataFrame:
        """
        Load the Controls sheet from MUNI_MASTER_BUCKET.

        Expected columns (flexible naming):
          - Control / ControlName / Name / Key
          - Value
        """
        if self.master_bucket is None:
            raise ValueError("master_bucket configuration not set in YAML.")

        mb = self.master_bucket
        df = pd.read_excel(mb.file, sheet_name=mb.controls_sheet)
        return df

    def get_control_value(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Look up a value in the Controls sheet by control name.

        Example usage:
            app_cfg.get_control_value("CURVE_ASOF_DATE")

        Returns string or default if not found.
        """
        try:
            df = self.load_controls_df()
        except Exception:
            return default

        # Normalise column names
        norm = {c.lower().replace(" ", ""): c for c in df.columns}

        key_col = None
        val_col = None
        for candidate in ("control", "controlname", "name", "key"):
            if candidate in norm:
                key_col = norm[candidate]
                break
        for candidate in ("value", "val"):
            if candidate in norm:
                val_col = norm[candidate]
                break

        if key_col is None or val_col is None:
            return default

        mask = df[key_col].astype(str).str.strip().str.upper() == name.upper()
        sub = df.loc[mask]
        if sub.empty:
            return default

        val = sub.iloc[0][val_col]
        if pd.isna(val):
            return default

        return str(val)
