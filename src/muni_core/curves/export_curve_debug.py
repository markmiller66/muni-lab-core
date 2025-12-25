from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import pandas as pd

from muni_core.curves.build_curve_bundle import build_curve_bundle


def _bump_parallel_df(dense_df: pd.DataFrame, bump_bp: float) -> pd.DataFrame:
    """Assumes canonical columns tenor_yrs, rate_dec exist."""
    out = dense_df.copy()
    bump = float(bump_bp) / 10000.0
    out["rate_dec"] = pd.to_numeric(out["rate_dec"], errors="coerce") + bump
    return out


def export_curve_overlay_to_excel(
    *,
    history_df: pd.DataFrame,
    curve_key: str,
    asof: date,
    step_years: float = 0.5,
    bump_bp: float = 1.0,
    out_dir: Path | str = "data/curve_debug",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )
    base = bundle.dense_df[["tenor_yrs", "rate_dec"]].copy()
    up = _bump_parallel_df(base.rename(columns={"rate_dec": "rate_dec"}), +abs(bump_bp))
    dn = _bump_parallel_df(base.rename(columns={"rate_dec": "rate_dec"}), -abs(bump_bp))

    # Rename for clarity
    base = base.rename(columns={"rate_dec": "rate_base"})
    up = up.rename(columns={"rate_dec": "rate_up"})
    dn = dn.rename(columns={"rate_dec": "rate_dn"})

    overlay = base.merge(up[["tenor_yrs", "rate_up"]], on="tenor_yrs").merge(
        dn[["tenor_yrs", "rate_dn"]], on="tenor_yrs"
    )

    overlay["delta_up_bp"] = (overlay["rate_up"] - overlay["rate_base"]) * 10000.0
    overlay["delta_dn_bp"] = (overlay["rate_base"] - overlay["rate_dn"]) * 10000.0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"curve_overlay_{curve_key}_{asof}_step{step_years}_bump{bump_bp}bp_{stamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        overlay.to_excel(xw, sheet_name="overlay", index=False)
        base.to_excel(xw, sheet_name="base", index=False)
        up.to_excel(xw, sheet_name="up", index=False)
        dn.to_excel(xw, sheet_name="dn", index=False)

    return out_path
