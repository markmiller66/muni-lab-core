# tests/test_config_and_sources.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from muni_core.config.loader import AppConfig
from muni_core.curves.history import (
    build_historical_curves,
    get_zero_curve_for_date,
    export_spot_curves_and_spreads,
    export_dense_curve_and_forward_matrix,
)

# -------------------------------------------------------------------
# Local raw-data loaders (replaces nonexistent muni_core.sources)
# -------------------------------------------------------------------


def load_treasury_raw(app_cfg: AppConfig) -> pd.DataFrame:
    """
    Load raw Treasury curve data from Fed CSV using curve_sources
    section of AppConfig.
    """
    cs = app_cfg.curve_sources
    if cs is None:
        raise ValueError("AppConfig.curve_sources is not configured.")

    path = cs.data_root / cs.treasury_file
    print(f"[TEST] Loading Treasury raw data from: {path}")

    df = pd.read_csv(path, skiprows=cs.treasury_skiprows, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    print(f"  Treasury rows: {len(df)}")
    print(f"  Treasury index range: {df.index.min()} -> {df.index.max()}")
    print(f"  Treasury columns: {list(df.columns[:8])} ...")
    return df


def load_muni_raw(app_cfg: AppConfig) -> pd.DataFrame:
    """
    Load raw Tradeweb AAA muni curve data from CSV.
    """
    cs = app_cfg.curve_sources
    if cs is None:
        raise ValueError("AppConfig.curve_sources is not configured.")

    path = cs.data_root / cs.muni_file
    print(f"[TEST] Loading Muni raw data from: {path}")

    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["Date Of"] = pd.to_datetime(df["Date Of"])
    df = df.sort_values("Date Of").set_index("Date Of")

    print(f"  Muni rows: {len(df)}")
    print(f"  Muni index range: {df.index.min()} -> {df.index.max()}")
    print(f"  Muni columns: {list(df.columns[:8])} ...")
    return df


def load_vix_raw(app_cfg: AppConfig) -> pd.DataFrame:
    """
    Load U Chicago muni VIX data if configured; otherwise return empty df.
    """
    cs = app_cfg.curve_sources
    if cs is None or cs.vix_file is None:
        print("[TEST] No VIX file configured; skipping VIX load.")
        return pd.DataFrame()

    path = cs.data_root / cs.vix_file
    print(f"[TEST] Loading VIX raw data from: {path}")

    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    print(f"  VIX rows: {len(df)}")
    print(f"  VIX index range: {df.index.min()} -> {df.index.max()}")
    print(f"  VIX columns: {list(df.columns[:8])} ...")
    return df


# -------------------------------------------------------------------
# Main test harness
# -------------------------------------------------------------------


def main() -> None:
    # --- Load config ---
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "example_config.yaml"
    print(f"[INFO] Loading AppConfig from: {cfg_path}")
    app_cfg = AppConfig.from_yaml(cfg_path)

    # --- Print curves config ---
    curves_cfg = app_cfg.curves
    print("\n[CONFIG] Curves:")
    print(f"  wide_curve_file : {curves_cfg.wide_curve_file}")
    print(f"  wide_curve_sheet: {curves_cfg.wide_curve_sheet}")
    print(f"  spot_curve_file : {curves_cfg.spot_curve_file}")
    print(f"  spot_curve_sheet: {curves_cfg.spot_curve_sheet}")
    print(f"  curve_strategy  : {curves_cfg.curve_strategy}")
    print(f"  history_file    : {curves_cfg.history_file}")
    print(f"  curve_asof_date : {curves_cfg.curve_asof_date}")

    # --- Print master bucket config ---
    mb_cfg = app_cfg.master_bucket
    if mb_cfg is not None:
        print("\n[CONFIG] Master bucket:")
        print(f"  file            : {mb_cfg.file}")
        print(f"  column_map_sheet: {mb_cfg.column_map_sheet}")
        print(f"  controls_sheet  : {mb_cfg.controls_sheet}")
        print(f"  rating_map_sheet: {mb_cfg.rating_map_sheet}")
    else:
        print("\n[CONFIG] Master bucket: None")

    # --- Print curve sources config ---
    cs_cfg = app_cfg.curve_sources
    if cs_cfg is not None:
        print("\n[CONFIG] Curve sources:")
        print(f"  data_root         : {cs_cfg.data_root}")
        print(f"  treasury_file     : {cs_cfg.treasury_file}")
        print(f"  muni_file         : {cs_cfg.muni_file}")
        print(f"  vix_file          : {cs_cfg.vix_file}")
        print(f"  treasury_skiprows : {cs_cfg.treasury_skiprows}")
    else:
        print("\n[CONFIG] Curve sources: None")

    # --- Load raw data ---
    treas_df = load_treasury_raw(app_cfg)
    muni_df = load_muni_raw(app_cfg)
    vix_df = load_vix_raw(app_cfg)

    # --- Build historical curves ---
    print("\n[TEST] Building historical curves table (par/spot version)...")
    history_df = build_historical_curves(
        treas_df=treas_df,
        muni_df=muni_df,
        vix_df=vix_df,
        app_cfg=app_cfg,
    )

    print(f"  History rows: {len(history_df)}")
    print(
        f"  History date range: "
        f"{pd.to_datetime(history_df['date']).min()} -> "
        f"{pd.to_datetime(history_df['date']).max()}"
    )
    print(f"  History curve_keys: {history_df['curve_key'].unique()}")

    # --- Test ZeroCurve wiring for the configured as-of date ---
    if curves_cfg.curve_asof_date:
        asof = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        asof = pd.to_datetime(history_df["date"]).dt.date.max()

    print(f"\n[TEST] Building ZeroCurve for {asof} / AAA_MUNI_SPOT ...")
    zc = get_zero_curve_for_date(asof, "AAA_MUNI_SPOT", app_cfg)
    print(f"  ZeroCurve has {len(zc.points)} points.")

    # --- Export annual spot curves + spreads ---
    print("\n[TEST] Exporting spot curves + spreads...")
    export_spot_curves_and_spreads(history_df, app_cfg)

    # --- Export dense semi-annual curve + full forward matrix for AAA_MUNI_SPOT ---
    print("\n[TEST] Exporting dense semi-annual curve + full forward matrix...")
    export_dense_curve_and_forward_matrix(
        history_df,
        app_cfg,
        curve_key="AAA_MUNI_SPOT",
        step_years=0.5,
    )

    print("\n[OK] Config + raw data loading and curve exports appear to work.")


if __name__ == "__main__":
    main()
