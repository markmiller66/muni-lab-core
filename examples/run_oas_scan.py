# examples/run_oas_scan.py

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd

from muni_core.config import AppConfig
from muni_core.curves import load_zero_curve_from_app_config
from muni_core.io.bond_loader import BondInputConfig, load_bonds_from_excel
from muni_core.oas import solve_oas_for_price


def find_repo_root() -> Path:
    """
    Resolve the repo root assuming this file lives in <repo>/examples/.
    """
    return Path(__file__).resolve().parents[1]


def main() -> None:
    repo_root = find_repo_root()
    print(f"[DEBUG] Repo root resolved to: {repo_root}")

    # ------------------------------------------------------------------
    # 1) Load AppConfig (for curves + column map)
    # ------------------------------------------------------------------
    # Adjust this name if your YAML is called something else.
    cfg_path = repo_root / "config" / "app_config.yaml"
    if not cfg_path.exists():
        # Fallback to example_config.yaml if that's what you're using
        fallback = repo_root / "config" / "example_config.yaml"
        if fallback.exists():
            cfg_path = fallback
        else:
            print(f"[ERROR] Could not find config YAML at {cfg_path} or {fallback}")
            sys.exit(1)

    print(f"[DEBUG] Using config file: {cfg_path}")
    app_cfg = AppConfig.from_yaml(cfg_path)

    # ------------------------------------------------------------------
    # 2) Load ZeroCurve from AppConfig
    # ------------------------------------------------------------------
    try:
        zc = load_zero_curve_from_app_config(app_cfg)
        print("[DEBUG] ZeroCurve loaded from AppConfig.")
    except Exception as e:
        print(f"[ERROR] Failed to load ZeroCurve from AppConfig: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3) Load bonds from your working Excel file
    # ------------------------------------------------------------------
    # For now we mirror the path used in run_portfolio_call_scan.py.
    # You can later route this through AppConfig if desired.
    bonds_file = repo_root / "data" / "working" / "my_300_bonds.xlsx"
    if not bonds_file.exists():
        print(f"[ERROR] Bonds file not found at {bonds_file}")
        sys.exit(1)

    # Use ColumnMap via AppConfig if configured; otherwise BondInputConfig defaults
    try:
        bond_cfg = BondInputConfig.from_app_config(app_cfg)
        print("[DEBUG] BondInputConfig loaded from ColumnMap in MUNI_MASTER_BUCKET.")
    except Exception as e:
        print(f"[WARN] Could not load BondInputConfig from AppConfig ({e}); "
              f"falling back to default column names.")
        bond_cfg = BondInputConfig()

    bonds, source_df = load_bonds_from_excel(
        bonds_file,
        sheet_name=0,
        cfg=bond_cfg,
    )

    print(f"Loaded {len(bonds)} bonds from {bonds_file}")

    # ------------------------------------------------------------------
    # 4) Solve a constant spread (Z-style) for each bond
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    for bond in bonds:
        try:
            # Solve for spread (Z/OAS backbone) that matches the clean price
            res = solve_oas_for_price(
                bond,
                zc,
                coupon_freq=2,  # semiannual; adjust if you ever add basis-aware freq
            )

            rows.append(
                {
                    "CUSIP": bond.cusip,
                    "Rating": bond.rating,
                    "RatingNum": bond.rating_num,
                    "Coupon": bond.coupon,
                    "CleanPrice_Input": bond.clean_price,
                    "HasCall": bond.has_call() if hasattr(bond, "has_call") else None,

                    # Z-style constant spread on top of ZeroCurve
                    "Z_spread_bp": res.oas_bp,
                    "ModelPrice_at_Z": res.model_price,
                    "Z_residual": res.residual,
                    "Z_converged": res.converged,
                    "Z_iterations": res.iterations,

                    "Error": None,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "CUSIP": bond.cusip,
                    "Rating": bond.rating,
                    "RatingNum": bond.rating_num,
                    "Coupon": bond.coupon,
                    "CleanPrice_Input": bond.clean_price,
                    "HasCall": bond.has_call() if hasattr(bond, "has_call") else None,

                    "Z_spread_bp": None,
                    "ModelPrice_at_Z": None,
                    "Z_residual": None,
                    "Z_converged": False,
                    "Z_iterations": 0,

                    "Error": str(e),
                }
            )

    result_df = pd.DataFrame(rows)
    print("\nResult columns:", list(result_df.columns))

    # Quick sanity counts
    print("\nZ_converged value counts:")
    print(result_df["Z_converged"].value_counts(dropna=False))

    if "Error" in result_df.columns:
        errors = result_df["Error"].dropna()
        if not errors.empty:
            print("\nUnique errors (top 10):")
            print(errors.value_counts().head(10))

    # ------------------------------------------------------------------
    # 5) Write timestamped Excel output
    # ------------------------------------------------------------------
    output_dir = repo_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_oas_scan_{timestamp}.xlsx"
    out_path = output_dir / filename

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="OAS_Scan", index=False)

    print(f"\nWrote OAS scan to {out_path}")


if __name__ == "__main__":
    main()
