from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class CurvesConfig:
    wide_curve_file: Path
    wide_curve_sheet: str
    spot_curve_file: Path
    spot_curve_sheet: str
    curve_strategy: str


@dataclass
class AppConfig:
    """
    Top-level configuration object for muni_core.

    Right now it only knows about curves, but we can expand later
    (npv thresholds, sigma settings, paths, etc.).
    """
    curves: CurvesConfig

    @classmethod
    def from_yaml(cls, cfg_path: Path) -> "AppConfig":
        """
        Load configuration from a YAML file.

        Paths in the YAML are interpreted as relative to the **repo root**.
        We assume this file lives in: <repo root>/config/example_config.yaml
        """
        text = cfg_path.read_text(encoding="utf-8")
        data: Dict[str, Any] = yaml.safe_load(text) or {}

        # Infer repo root: config/ is one level below project root.
        repo_root = cfg_path.parent.parent

        curves_data: Dict[str, Any] = data.get("curves", {})

        def resolve_path(p: str) -> Path:
            # Allow absolute paths too, but default to repo-root-relative.
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            return path.resolve()

        wide_curve_file = resolve_path(curves_data.get("wide_curve_file", "data/AAA_MUNI_CURVE/aaa_curves.xlsx"))
        spot_curve_file = resolve_path(curves_data.get("spot_curve_file", "data/AAA_MUNI_CURVE/aaa_curves.xlsx"))

        curves_cfg = CurvesConfig(
            wide_curve_file=wide_curve_file,
            wide_curve_sheet=curves_data.get("wide_curve_sheet", "Curves_Wide"),
            spot_curve_file=spot_curve_file,
            spot_curve_sheet=curves_data.get("spot_curve_sheet", "AAA_Spot"),
            curve_strategy=curves_data.get("curve_strategy", "excel_curves_wide"),
        )

        return cls(curves=curves_cfg)
