from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from muni_core.curves.build_curve_bundle import build_curve_bundle


def export_curve_bundle_to_excel(
    *,
    history_path: Path,
    out_dir: Path,
    curve_key: str = "AAA_MUNI_SPOT",
    asof: date = date(2025, 11, 26),
    step_years: float = 0.5,
) -> Path:
    history_df = pd.read_parquet(history_path)

    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"curve_{curve_key}_{asof.isoformat()}_step{step_years}_{stamp}.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        bundle.dense_df.to_excel(xw, sheet_name="dense_df", index=False)

    return out_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    out_dir = repo_root / "data" / "3-OUTPUT"

    out = export_curve_bundle_to_excel(
        history_path=hist_path,
        out_dir=out_dir,
        curve_key="AAA_MUNI_SPOT",
        asof=date(2025, 11, 26),
        step_years=0.5,
    )
    print(f"Wrote: {out}")
