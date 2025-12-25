from __future__ import annotations

from pathlib import Path
from datetime import date, datetime

import pandas as pd

from muni_core.risk.krd_curve_overlays import KRDCurveOverlayConfig, build_krd_curve_overlay_table


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"

    history_df = pd.read_parquet(hist_path)

    cfg = KRDCurveOverlayConfig(
        asof=date(2025, 11, 26),
        curve_key="AAA_MUNI_SPOT",
        step_years=0.5,
        bump_bp=1.0,
    )

    df = build_krd_curve_overlay_table(history_df=history_df, cfg=cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = repo_root / "data" / "debug" / f"krd_curve_overlay_{cfg.curve_key}_{cfg.asof}_b{cfg.bump_bp:g}bp_{ts}.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="overlay", index=False)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
