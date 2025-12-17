# tests/test_config_and_sources.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from muni_core.config.loader import AppConfig
from muni_core.curves.history import (
    build_historical_curves,
    get_zero_curve_for_date,
    export_spot_curves_and_spreads,
    export_dense_curve_and_forward_matrix,
    build_dense_zero_curve_for_date,
)
from muni_core.curves.short_rate_lattice import (
    build_hw_short_rate_lattice,
    build_state_price_tree_from_lattice,
)
from muni_core.pricing import (
    BondCashflowSchedule,
    build_level_coupon_schedule,
    price_cashflows_from_state_tree,
    price_cashflows_from_dense_zero,
    price_level_coupon_bond_hw,
    price_bullet_bond_hw_from_config,
)
from muni_core.pricing.hw_bond_pricer import (
    price_callable_bond_from_lattice,
    price_callable_bond_hw_from_bond,
)
from muni_core.model import Bond, CallFeature



# ----------------------------------------------------------------------
# Small helper: load Treasury / Muni / VIX directly from CSVs
# using AppConfig.curve_sources (no muni_core.sources dependency).
# ----------------------------------------------------------------------
def load_treasury_muni_vix_raw_inline(app_cfg: AppConfig):
    cs = app_cfg.curve_sources
    if cs is None:
        raise ValueError("curve_sources is not configured in app_config.yaml")

    data_root = cs.data_root

    # --- Treasury ---
    treas_path = data_root / cs.treasury_file
    treas_df = pd.read_csv(
        treas_path,
        skiprows=cs.treasury_skiprows,
        low_memory=False,
    )
    # Expect a 'Date' column as in your original spot.py
    treas_df["Date"] = pd.to_datetime(treas_df["Date"])
    treas_df.set_index("Date", inplace=True)
    treas_df.attrs["source_path"] = str(treas_path)

    # --- Muni ---
    muni_path = data_root / cs.muni_file
    muni_df = pd.read_csv(
        muni_path,
        low_memory=False,
        on_bad_lines="skip",
    )
    # Expect 'Date Of' column as in your original Tradeweb CSV
    muni_df["Date Of"] = pd.to_datetime(muni_df["Date Of"])
    muni_df = muni_df.sort_values("Date Of").set_index("Date Of")
    muni_df.attrs["source_path"] = str(muni_path)

    # --- VIX (optional) ---
    vix_df = None
    if cs.vix_file:
        vix_path = data_root / cs.vix_file
        vix_df = pd.read_csv(
            vix_path,
            low_memory=False,
        )
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_df.set_index("date", inplace=True)
        vix_df.attrs["source_path"] = str(vix_path)

    return treas_df, muni_df, vix_df


def print_config(app_cfg: AppConfig) -> None:
    """
    Pretty-print the important config pieces so we can confirm wiring.
    """
    curves = app_cfg.curves
    mb = app_cfg.master_bucket
    cs = app_cfg.curve_sources

    print("\n[CONFIG] Curves:")
    print(f"  wide_curve_file : {curves.wide_curve_file}")
    print(f"  wide_curve_sheet: {curves.wide_curve_sheet}")
    print(f"  spot_curve_file : {curves.spot_curve_file}")
    print(f"  spot_curve_sheet: {curves.spot_curve_sheet}")
    print(f"  curve_strategy  : {curves.curve_strategy}")
    print(f"  history_file    : {curves.history_file}")
    print(f"  curve_asof_date : {curves.curve_asof_date}")

    if mb is not None:
        print("\n[CONFIG] Master bucket:")
        print(f"  file            : {mb.file}")
        print(f"  column_map_sheet: {mb.column_map_sheet}")
        print(f"  controls_sheet  : {mb.controls_sheet}")
        print(f"  rating_map_sheet: {mb.rating_map_sheet}")
    else:
        print("\n[CONFIG] Master bucket: <None>")

    if cs is not None:
        print("\n[CONFIG] Curve sources:")
        print(f"  data_root         : {cs.data_root}")
        print(f"  treasury_file     : {cs.treasury_file}")
        print(f"  muni_file         : {cs.muni_file}")
        print(f"  vix_file          : {cs.vix_file}")
        print(f"  treasury_skiprows : {cs.treasury_skiprows}")
    else:
        print("\n[CONFIG] Curve sources: <None>")


def main() -> None:
    # ------------------------------------------------------------------
    # 1) Load AppConfig from YAML
    # ------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "example_config.yaml"

    print(f"[INFO] Loading AppConfig from: {cfg_path}")
    app_cfg = AppConfig.from_yaml(cfg_path)

    print_config(app_cfg)

    # ------------------------------------------------------------------
    # 2) Load raw Treasury / Muni / VIX data (inline loader)
    # ------------------------------------------------------------------
    treas_df, muni_df, vix_df = load_treasury_muni_vix_raw_inline(app_cfg)

    print("\n[TEST] Loading Treasury raw data from:", treas_df.attrs.get("source_path", "<unknown>"))
    print(f"  Treasury rows: {len(treas_df)}")
    print(f"  Treasury index range: {treas_df.index.min()} -> {treas_df.index.max()}")
    print(f"  Treasury columns: {list(treas_df.columns[:8])} ...")

    print("\n[TEST] Loading Muni raw data from:", muni_df.attrs.get("source_path", "<unknown>"))
    print(f"  Muni rows: {len(muni_df)}")
    print(f"  Muni index range: {muni_df.index.min()} -> {muni_df.index.max()}")
    print(f"  Muni columns: {list(muni_df.columns[:8])} ...")

    if vix_df is not None:
        print("\n[TEST] Loading VIX raw data from:", vix_df.attrs.get("source_path", "<unknown>"))
        print(f"  VIX rows: {len(vix_df)}")
        print(f"  VIX index range: {vix_df.index.min()} -> {vix_df.index.max()}")
        print(f"  VIX columns: {list(vix_df.columns[:8])} ...")
    else:
        print("\n[TEST] VIX raw data: <None>")

    # ------------------------------------------------------------------
    # 3) Build historical curves table (UST_SPOT, AAA_MUNI_PAR, AAA_MUNI_SPOT)
    # ------------------------------------------------------------------
    print("\n[TEST] Building historical curves table (par/spot version)...")
    history_df = build_historical_curves(
        treas_df=treas_df,
        muni_df=muni_df,
        vix_df=vix_df,
        app_cfg=app_cfg,
    )

    print(f"  History rows: {len(history_df)}")
    print(f"  History date range: {history_df['date'].min()} -> {history_df['date'].max()}")
    print(f"  History curve_keys: {history_df['curve_key'].unique()}")

    # ------------------------------------------------------------------
    # 4) Build a ZeroCurve for the configured as-of date / AAA_MUNI_SPOT
    # ------------------------------------------------------------------
    curves_cfg = app_cfg.curves
    if curves_cfg.curve_asof_date:
        asof_date = pd.to_datetime(curves_cfg.curve_asof_date).date()
    else:
        tmp = history_df.copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
        asof_date = tmp["date"].max()

    curve_key = "AAA_MUNI_SPOT"

    print(f"\n[TEST] Building ZeroCurve for {asof_date} / {curve_key} ...")
    zc = get_zero_curve_for_date(asof_date, curve_key, app_cfg)
    print(f"  ZeroCurve has {len(zc.points)} points.")

    # ------------------------------------------------------------------
    # 5) Export spot curves + spreads + 1Y / 5Y forwards
    # ------------------------------------------------------------------
    print("\n[TEST] Exporting spot curves + spreads...")
    export_spot_curves_and_spreads(history_df, app_cfg)

    # ------------------------------------------------------------------
    # 6) Export dense semi-annual curve + full forward matrix
    #    (for the as-of date / AAA_MUNI_SPOT)
    # ------------------------------------------------------------------
    print("\n[TEST] Exporting dense semi-annual curve + full forward matrix...")
    export_dense_curve_and_forward_matrix(
        history_df=history_df,
        app_cfg=app_cfg,
        curve_key=curve_key,
        step_years=0.5,
    )

    # ------------------------------------------------------------------
    # 7) HW short-rate pricer sanity check
    # ------------------------------------------------------------------
    print("\n[TEST] HW short-rate pricer sanity check...")

    # Build dense zero curve for this date / curve_key
    dense_df = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof_date,
        step_years=0.5,
    )

    print(f"  Dense zero curve rows for {curve_key} @ {asof_date}: {len(dense_df)}")

    # Build HW short-rate lattice
    lattice = build_hw_short_rate_lattice(
        dense_df=dense_df,
        a=0.1,
        sigma=0.01,
        dt_years=1.0,
    )

    print(f"  Short-rate lattice levels: {len(lattice.rates)}")
    if lattice.rates:
        print(f"  Level 0 rate: {lattice.rates[0][0]:.6f}")

    # Build state-price tree from lattice
    state_prices = build_state_price_tree_from_lattice(lattice)
    print(f"  State price tree levels: {len(state_prices)}")
    if state_prices:
        print(f"  Level 0 state price: {state_prices[0][0]:.6f}")

    # Build a toy 10Y 4% level coupon bond schedule (annual)
    schedule = build_level_coupon_schedule(
        maturity_years=10.0,
        coupon_rate=0.04,
        dt_years=1.0,
    )

    # Convert list-of-lists state_prices -> DataFrame with t_yrs + state_price
    levels = state_prices           # list of levels, each a list of state prices
    n_levels = len(levels) - 1      # 0..N => N time steps

    if n_levels <= 0:
        raise ValueError("state_prices must contain at least 2 levels.")

    maturity_years = float(schedule.maturity_years)
    dt_years = maturity_years / float(n_levels)

    records = []
    for level_idx, row in enumerate(levels):
        t = level_idx * dt_years
        for j, sp in enumerate(row):
            records.append(
                {
                    "t_yrs": float(t),
                    "state_price": float(sp),
                    "level": int(level_idx),
                    "state_index": int(j),
                }
            )

    state_tree_df = pd.DataFrame.from_records(records)

    # Price via HW state-price tree helper
    hw_price = price_cashflows_from_state_tree(
        state_tree_df=state_tree_df,
        schedule=schedule,
        time_tolerance=1e-6,
    )

    print(
        f"  HW price for toy 10Y 4% bond "
        f"@ {asof_date} ({curve_key}): {hw_price:.6f}"
    )

    # ------------------------------------------------------------------
    # 8) Callable HW pricer sanity check (lattice-level)
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 8) Callable HW pricer sanity check (lattice-level)
    # ------------------------------------------------------------------
    print("\n[TEST] Callable HW pricer sanity check (lattice-level)...")

    # Use the SAME lattice as above, but build both:
    #   - non-call price (no Bermudan rights)
    #   - callable price (issuer call at 5y)
    maturity_years = 10.0
    coupon_rate = 0.04
    freq_per_year = 1  # matches dt_years = 1.0 in lattice build
    call_times_yrs = [5.0]  # single call date at 5y

    # Non-call price on the same HW lattice (call_times_yrs = [])
    noncall_hw_price = price_callable_bond_from_lattice(
        lattice=lattice,
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        call_times_yrs=[],  # <--- no call feature
        face=100.0,
        call_price=100.0,
        q=0.5,
        time_tolerance=1e-6,
    )

    # Callable price with issuer Bermudan call at 5y
    callable_hw_price = price_callable_bond_from_lattice(
        lattice=lattice,
        maturity_years=maturity_years,
        coupon_rate=coupon_rate,
        freq_per_year=freq_per_year,
        call_times_yrs=call_times_yrs,
        face=100.0,
        call_price=100.0,
        q=0.5,
        time_tolerance=1e-6,
    )

    print(f"  Non-call HW price (lattice): {noncall_hw_price:.6f}")
    print(f"  Callable HW price (call @5y): {callable_hw_price:.6f}")

    if callable_hw_price > noncall_hw_price + 1e-6:
        print("  [WARN] Callable price is not lower than non-call price; investigate.")
    else:
        print("  [OK] Callable price is <= non-call price (embedded option cost present).")

    print("\n[OK] Config, raw data loading, curve history, exports, and HW pricer wiring appear to work.\n")


if __name__ == "__main__":
    main()
