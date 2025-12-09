# tests/test_config_and_sources.py

from pathlib import Path
import sys
import pandas as pd

# --- ensure src/ is on sys.path so "muni_core" can be imported ---
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]        # .../muni-lab-core
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Now we can import from the real package
from muni_core.config.loader import AppConfig
from muni_core.curves.spot_sources import (
    SpotSourceConfig,
    load_treasury_raw,
    load_muni_raw,
    load_vix_raw,
)


def main() -> None:
    # 1) Load AppConfig from YAML
    cfg_path = REPO_ROOT / "config" / "example_config.yaml"
    print(f"[INFO] Loading AppConfig from: {cfg_path}")
    app_cfg = AppConfig.from_yaml(cfg_path)

    # 2) Show what the loader thinks the repo root + key paths are
    print("\n[CONFIG] Curves:")
    print(f"  wide_curve_file : {app_cfg.curves.wide_curve_file}")
    print(f"  wide_curve_sheet: {app_cfg.curves.wide_curve_sheet}")
    print(f"  spot_curve_file : {app_cfg.curves.spot_curve_file}")
    print(f"  spot_curve_sheet: {app_cfg.curves.spot_curve_sheet}")
    print(f"  curve_strategy  : {app_cfg.curves.curve_strategy}")
    print(f"  history_file    : {app_cfg.curves.history_file}")
    print(f"  curve_asof_date : {app_cfg.curves.curve_asof_date}")

    if app_cfg.master_bucket:
        print("\n[CONFIG] Master bucket:")
        print(f"  file            : {app_cfg.master_bucket.file}")
        print(f"  column_map_sheet: {app_cfg.master_bucket.column_map_sheet}")
        print(f"  controls_sheet  : {app_cfg.master_bucket.controls_sheet}")
        print(f"  rating_map_sheet: {app_cfg.master_bucket.rating_map_sheet}")
    else:
        print("\n[WARN] master_bucket is not configured")

    # 3) Curve sources (Tradeweb / Fed / VIX)
    if app_cfg.curve_sources is None:
        print("\n[ERROR] curve_sources is not configured in YAML.")
        return

    cs = app_cfg.curve_sources
    print("\n[CONFIG] Curve sources:")
    print(f"  data_root         : {cs.data_root}")
    print(f"  treasury_file     : {cs.treasury_file}")
    print(f"  muni_file         : {cs.muni_file}")
    print(f"  vix_file          : {cs.vix_file}")
    print(f"  treasury_skiprows : {cs.treasury_skiprows}")

    # 4) Build SpotSourceConfig and try loading each raw dataset
    spot_cfg = SpotSourceConfig(
        data_root=cs.data_root,
        treasury_file=cs.treasury_file,
        muni_file=cs.muni_file,
        vix_file=cs.vix_file,
        treasury_skiprows=cs.treasury_skiprows,
    )

    print("\n[TEST] Loading Treasury raw data...")
    treas_df = load_treasury_raw(spot_cfg)
    print(f"  Treasury rows: {len(treas_df)}")
    print(f"  Treasury index range: {treas_df.index.min()} -> {treas_df.index.max()}")
    print(f"  Treasury columns: {list(treas_df.columns)[:8]} ...")

    print("\n[TEST] Loading Muni raw data...")
    muni_df = load_muni_raw(spot_cfg)
    print(f"  Muni rows: {len(muni_df)}")
    print(f"  Muni index range: {muni_df.index.min()} -> {muni_df.index.max()}")
    print(f"  Muni columns: {list(muni_df.columns)[:8]} ...")

    if cs.vix_file:
        print("\n[TEST] Loading VIX raw data...")
        vix_df = load_vix_raw(spot_cfg)
        print(f"  VIX rows: {len(vix_df)}")
        print(f"  VIX index range: {vix_df.index.min()} -> {vix_df.index.max()}")
        print(f"  VIX columns: {list(vix_df.columns)[:8]} ...")

    from muni_core.curves.history import (
        build_historical_curves,
        get_zero_curve_for_date,
    )

    # ... after your existing prints/tests ...

    print("\n[TEST] Building historical curves table (par version)...")
    history_df = build_historical_curves(
        treas_df=treas_df,
        muni_df=muni_df,
        vix_df=vix_df,
        app_cfg=app_cfg,
    )
    print(f"  History rows: {len(history_df)}")
    print(f"  History date range: {history_df['date'].min()} -> {history_df['date'].max()}")
    print(f"  History curve_keys: {history_df['curve_key'].unique()}")

    # Try to build a ZeroCurve for the configured as-of date
    from datetime import datetime

    if app_cfg.curves.curve_asof_date:
        asof = datetime.fromisoformat(app_cfg.curves.curve_asof_date).date()
        print(f"\n[TEST] Building ZeroCurve for {asof} / AAA_MUNI_SPOT ...")
        zc = get_zero_curve_for_date(asof, "AAA_MUNI_SPOT", app_cfg)
        print(f"  ZeroCurve has {len(zc.points)} points.")

    else:
        print("\n[WARN] curves.curve_asof_date is not set; skipping ZeroCurve test.")

    print("\n[OK] Config, raw data loading, and history/ZeroCurve wiring appear to work.")

    print("\n[OK] Config + raw data loading appear to work.")


if __name__ == "__main__":
    main()
