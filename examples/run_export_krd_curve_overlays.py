from __future__ import annotations

from datetime import date
from pathlib import Path
import argparse

import pandas as pd

from muni_core.risk.krd_curve_overlays import (
    build_krd_curve_overlay_bundle,
    export_krd_curve_overlays_to_excel,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-key", default="AAA_MUNI_SPOT")
    ap.add_argument("--asof", default="2025-11-26")          # YYYY-MM-DD
    ap.add_argument("--step-years", type=float, default=0.5)
    ap.add_argument("--bump-bp", type=float, default=1.0)
    ap.add_argument("--out", default="D:/BONDS/3-OUTPUT/krd_overlays.xlsx")
    ap.add_argument(
        "--history-parquet",
        default=str(Path("data/AAA_MUNI_CURVE/aaa_muni_treas_history.parquet")),
    )
    args = ap.parse_args()

    asof = pd.to_datetime(args.asof).date()

    hist_path = Path(args.history_parquet)
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history parquet: {hist_path.resolve()}")

    history_df = pd.read_parquet(hist_path)
    if history_df.empty:
        raise ValueError("history_df is empty")

    bundle = build_krd_curve_overlay_bundle(
        history_df=history_df,
        curve_key=args.curve_key,
        asof=asof,
        step_years=float(args.step_years),
        bump_bp=float(args.bump_bp),
        key_tenors=[1, 2, 3, 5, 7, 10, 15, 20, 30],
    )

    out_path = Path(args.out)
    export_krd_curve_overlays_to_excel(overlay_bundle=bundle, out_path=out_path)

    print(f"Wrote: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
