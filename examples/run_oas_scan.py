# examples/run_oas_scan.py

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

from muni_core.config import AppConfig
from muni_core.curves import load_zero_curve_from_app_config
from muni_core.io.bond_loader import BondInputConfig, load_bonds_from_excel
from muni_core.oas import solve_oas_for_price, solve_callable_oas_hw
from muni_core.oas.simple_oas import price_bond_with_oas
from muni_core.risk.oas_dv01_callable_hw import oas_dv01_callable_hw_for_bond

def find_repo_root() -> Path:
    """
    Resolve the repo root assuming this file lives in <repo>/examples/>.
    """
    return Path(__file__).resolve().parents[1]


def main() -> None:
    repo_root = find_repo_root()
    print(f"[DEBUG] Repo root resolved to: {repo_root}")

    # ------------------------------------------------------------------
    # 1) Load AppConfig (for curves + column map)
    # ------------------------------------------------------------------
    cfg_path = repo_root / "config" / "app_config.yaml"
    if not cfg_path.exists():
        fallback = repo_root / "config" / "example_config.yaml"
        if fallback.exists():
            cfg_path = fallback
        else:
            print(f"[ERROR] Could not find config YAML at {cfg_path} or {fallback}")
            sys.exit(1)

    print(f"[DEBUG] Using config file: {cfg_path}")
    app_cfg = AppConfig.from_yaml(cfg_path)

    curves_cfg = app_cfg.curves

    # ------------------------------------------------------------------
    # 2) Load ZeroCurve from AppConfig (for Z-style OAS backbone)
    # ------------------------------------------------------------------
    try:
        zc = load_zero_curve_from_app_config(app_cfg)
        print("[DEBUG] ZeroCurve loaded from AppConfig.")
    except Exception as e:
        print(f"[ERROR] Failed to load ZeroCurve from AppConfig: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3) Load historical curves table (for HW lattice / callable OAS)
    # ------------------------------------------------------------------
    history_df: Optional[pd.DataFrame] = None
    try:
        history_path = curves_cfg.history_file
        if history_path is None:
            raise ValueError("curves.history_file is not configured in AppConfig.")

        history_path = Path(history_path)
        if not history_path.exists():
            raise FileNotFoundError(
                f"History parquet not found at {history_path}. "
                f"Run your curve history builder first."
            )

        history_df = pd.read_parquet(history_path)
        print(f"[DEBUG] Loaded history_df from {history_path}")
        print(
            f"[DEBUG] history_df: rows={len(history_df)}, "
            f"date range {history_df['date'].min()} -> {history_df['date'].max()}"
        )

    except Exception as e:
        print(f"[WARN] Could not load history_df for callable OAS: {e}")
        print("[WARN] Callable OAS will be skipped; only Z-spread will be computed.")
        history_df = None

    # Resolve as-of date for callable HW pricer (same logic as tests)
    asof_date = None
    if history_df is not None:
        if curves_cfg.curve_asof_date:
            asof_date = pd.to_datetime(curves_cfg.curve_asof_date).date()
        else:
            tmp = history_df.copy()
            tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
            asof_date = tmp["date"].max()
        print(f"[DEBUG] Callable HW as-of date: {asof_date}")

    # ------------------------------------------------------------------
    # 4) Load bonds from your working Excel file
    # ------------------------------------------------------------------
    bonds_file = repo_root / "data" / "working" / "my_300_bonds.xlsx"
    if not bonds_file.exists():
        print(f"[ERROR] Bonds file not found at {bonds_file}")
        sys.exit(1)

    try:
        bond_cfg = BondInputConfig.from_app_config(app_cfg)
        print("[DEBUG] BondInputConfig loaded from ColumnMap in MUNI_MASTER_BUCKET.")
    except Exception as e:
        print(
            f"[WARN] Could not load BondInputConfig from AppConfig ({e}); "
            f"falling back to default column names."
        )
        bond_cfg = BondInputConfig()

    bonds, source_df = load_bonds_from_excel(
        bonds_file,
        sheet_name=0,
        cfg=bond_cfg,
    )

    print(f"Loaded {len(bonds)} bonds from {bonds_file}")
    print("First 5 bond clean prices:",
          [b.clean_price for b in bonds[:5]])

    # ------------------------------------------------------------------
    # 5) Solve Z-style spread + callable OAS for each bond
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    for bond in bonds:
        try:
            # --- Guards on required fields -----------------------------
            if bond.maturity_date is None:
                raise ValueError("Bond.maturity_date is None; cannot price.")

            if bond.clean_price is None:
                raise ValueError(
                    "Bond.clean_price is None (no market clean price loaded from Excel)."
                )

            target_price = float(bond.clean_price)

            # ----------------------------------------------------------
            # 1) Constant Z-style spread (non-call backbone)
            # ----------------------------------------------------------
            res = solve_oas_for_price(
                bond,
                zc,
                target_price=target_price,  # explicit target
                coupon_freq=2,              # semiannual
            )

            # Base Z results
            z_spread_bp = res.oas_bp
            z_model_price = res.model_price
            z_residual = res.residual
            z_converged = res.converged
            z_iterations = res.iterations

            # ----------------------------------------------------------
            # 2) Callable OAS via HW lattice (if bond has a call)
            # ----------------------------------------------------------
            callable_res = None
            embedded_bp = None

            callable_dv01 = None
            callable_mod_dur = None
            callable_price_base = None
            callable_price_up = None
            callable_price_down = None

            if bond.has_call():
                try:
                    callable_res = solve_callable_oas_hw(
                        bond=bond,
                        history_df=history_df,
                        app_cfg=app_cfg,
                        target_price=target_price,
                        coupon_freq=2,
                        curve_key=app_cfg.curves.curve_strategy and "AAA_MUNI_SPOT" or "AAA_MUNI_SPOT",
                        step_years=0.5,
                        q=0.5,
                    )

                    if callable_res is not None and callable_res.converged:
                        # Embedded option cost in bp (callable OAS – Z spread)
                        embedded_bp = callable_res.oas_bp - z_spread_bp

                        # --------------------------------------------------
                        # 3) Callable OAS DV01 (around the callable OAS)
                        # --------------------------------------------------
                        dv01_res = oas_dv01_callable_hw_for_bond(
                            bond=bond,
                            history_df=history_df,
                            app_cfg=app_cfg,
                            base_oas_bp=callable_res.oas_bp,
                            bump_bp=1.0,             # +1 bp bump
                            freq_per_year=2,
                            curve_key="AAA_MUNI_SPOT",
                            step_years=0.5,
                            q=0.5,
                        )

                        callable_dv01 = dv01_res.dv01_bp
                        callable_mod_dur = dv01_res.mod_duration
                        callable_price_base = dv01_res.price_base
                        callable_price_up = dv01_res.price_up
                        callable_price_down = dv01_res.price_down

                except Exception as e_call:
                    # Keep going; record error text only in Error field below
                    callable_res = None

            # ----------------------------------------------------------
            # 4) Assemble row
            # ----------------------------------------------------------
            rows.append(
                {
                    "CUSIP": bond.cusip,
                    "Rating": bond.rating,
                    "RatingNum": bond.rating_num,
                    "Coupon": bond.coupon,
                    "CleanPrice_Input": bond.clean_price,
                    "HasCall": bond.has_call() if hasattr(bond, "has_call") else None,

                    "Z_spread_bp": z_spread_bp,
                    "ModelPrice_at_Z": z_model_price,
                    "Z_residual": z_residual,
                    "Z_converged": z_converged,
                    "Z_iterations": z_iterations,

                    "Callable_OAS_bp": callable_res.oas_bp if callable_res is not None else None,
                    "Callable_ModelPrice": callable_res.model_price if callable_res is not None else None,
                    "Callable_OAS_residual": callable_res.residual if callable_res is not None else None,
                    "Callable_OAS_converged": callable_res.converged if callable_res is not None else None,
                    "Callable_OAS_iterations": callable_res.iterations if callable_res is not None else None,

                    "EmbeddedOption_bp": embedded_bp,

                    # New DV01 outputs
                    "Callable_DV01_bp": callable_dv01,
                    "Callable_ModDuration": callable_mod_dur,
                    "Callable_DV01_Price_Base": callable_price_base,
                    "Callable_DV01_Price_Up": callable_price_up,
                    "Callable_DV01_Price_Down": callable_price_down,

                    "Error": None,
                }
            )

        except Exception as e:
            rows.append(
                {
                    "CUSIP": getattr(bond, "cusip", None),
                    "Rating": getattr(bond, "rating", None),
                    "RatingNum": getattr(bond, "rating_num", None),
                    "Coupon": getattr(bond, "coupon", None),
                    "CleanPrice_Input": getattr(bond, "clean_price", None),
                    "HasCall": bond.has_call() if hasattr(bond, "has_call") else None,

                    "Z_spread_bp": None,
                    "ModelPrice_at_Z": None,
                    "Z_residual": None,
                    "Z_converged": False,
                    "Z_iterations": 0,

                    "Callable_OAS_bp": None,
                    "Callable_ModelPrice": None,
                    "Callable_OAS_residual": None,
                    "Callable_OAS_converged": None,
                    "Callable_OAS_iterations": 0,

                    "EmbeddedOption_bp": None,
                    "Callable_DV01_bp": None,
                    "Callable_ModDuration": None,
                    "Callable_DV01_Price_Base": None,
                    "Callable_DV01_Price_Up": None,
                    "Callable_DV01_Price_Down": None,

                    "Error": str(e),
                }
            )



    result_df = pd.DataFrame(rows)
    print("\nResult columns:", list(result_df.columns))

    print("\nZ_converged value counts:")
    print(result_df["Z_converged"].value_counts(dropna=False))

    if "Callable_OAS_converged" in result_df.columns:
        print("\nCallable_OAS_converged value counts:")
        print(result_df["Callable_OAS_converged"].value_counts(dropna=False))

    if "Error" in result_df.columns:
        errors = result_df["Error"].dropna()
        if not errors.empty:
            print("\nUnique errors (top 10):")
            print(errors.value_counts().head(10))

    # ------------------------------------------------------------------
    # 6) Write timestamped Excel output
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
