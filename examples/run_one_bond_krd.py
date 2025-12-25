# examples/run_one_bond_krd.py
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from muni_core.risk.callable_krd_hw_triangular import compute_callable_krd_hw
from muni_core.config.loader import AppConfig, CurvesConfig
from muni_core.oas.callable_oas_hw import solve_callable_oas_hw


def _load_history_df(repo_root: Path) -> pd.DataFrame:
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing curve history file: {hist_path}")
    return pd.read_parquet(hist_path)


def _make_app_cfg(repo_root: Path, asof: str) -> AppConfig:
    curve_xlsx = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_curves.xlsx"

    curves = CurvesConfig(
        curve_asof_date=asof,
        wide_curve_file=str(curve_xlsx),
        wide_curve_sheet="Curves_Wide",
        spot_curve_file=str(curve_xlsx),
        spot_curve_sheet="AAA_Spot",
        curve_strategy="excel_curves_wide",
    )
    return AppConfig(curves=curves)


def _make_one_bond():
    # Known-good bond shape (matches your Bond + CallFeature dataclasses)
    from muni_core.model import Bond, CallFeature

    return Bond(
        cusip="01025QAX9",
        rating="A1",
        rating_num=17,
        basis="1",
        settle_date=date(2024, 2, 5),
        maturity_date=date(2051, 9, 1),
        coupon=0.04,
        clean_price=94.1234,
        quantity=100.0,
        call_feature=CallFeature(
            call_date=date(2031, 9, 1),
            call_price=100.0,
        ),
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    asof = "2025-11-26"

    history_df = _load_history_df(repo_root)
    app_cfg = _make_app_cfg(repo_root, asof)
    bond = _make_one_bond()

    # If you have a target market price, set it here to solve OAS before KRD.
    target_price = None  # e.g. 101.25

    if target_price is None:
        base_oas_bp = 0.0
        print("\n[run_one_bond_krd] Using base_oas_bp=0.0 (no OAS solve) for first sanity pass.")
    else:
        oas_res = solve_callable_oas_hw(
            bond=bond,
            history_df=history_df,
            app_cfg=app_cfg,
            target_price=float(target_price),
            curve_key="AAA_MUNI_SPOT",
            step_years=0.5,
            q=0.5,
            time_tolerance=1e-6,
        )
        base_oas_bp = float(oas_res.oas_bp)
        print(f"\n[run_one_bond_krd] Solved callable OAS: {base_oas_bp:.3f} bp (target_price={target_price})")

    # Compute callable KRD/KRC (triangular bumps, OAS held fixed)
    res = compute_callable_krd_hw(
        bond=bond,
        history_df=history_df,
        app_cfg=app_cfg,
        base_oas_bp=base_oas_bp,
        key_tenors=[1, 2, 3, 5, 7, 10, 15, 20, 30],
        bump_bp=1.0,
        freq_per_year=2,
        curve_key="AAA_MUNI_SPOT",
        step_years=0.5,
        q=0.5,
        time_tolerance=1e-6,
        include_parallel_sanity=True,
    )

    # Print results
    print("\n=== Callable KRD (triangular) ===")
    print(f"base_price: {res.base_price:.6f}")
    print(f"bump_bp:    {res.bump_bp:.3f}")
    print("tenor |   KRD      |   KRC")
    for k in res.key_tenors:
        print(f"{k:>5.1f} | {res.krd[k]:>10.6f} | {res.krc[k]:>10.6f}")

    krd_sum = sum(res.krd.values())
    print(f"\nKRD sum: {krd_sum:.6f}")

    if res.curve_mod_duration is not None:
        gap = abs(krd_sum - res.curve_mod_duration)
        print(f"KRD sum - parallel gap: {gap:.6f}")

        print("\n=== Parallel sanity (callable) ===")
        print(f"curve_mod_duration: {res.curve_mod_duration:.6f}")
        print(f"curve_dv01_bp:      {res.curve_dv01_bp:.6f}")
        print(f"curve_price_up:     {res.curve_price_up:.6f}")
        print(f"curve_price_down:   {res.curve_price_down:.6f}")

    # Export pack AFTER res exists
    from muni_core.risk.export_one_bond_krd_pack import export_one_bond_krd_pack_to_excel

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("D:/BONDS/3-OUTPUT") / f"ONE_BOND_KRD_PACK_{bond.cusip}_{asof}_{stamp}.xlsx"

    export_one_bond_krd_pack_to_excel(
        out_path=out,
        history_df=history_df,
        asof=pd.to_datetime(asof).date(),
        curve_key="AAA_MUNI_SPOT",
        step_years=0.5,
        key_tenors=res.key_tenors,
        bump_bp=1.0,
        krd_res=res,
    )

    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
