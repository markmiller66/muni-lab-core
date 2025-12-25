from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from muni_core.curves.build_curve_bundle import build_curve_bundle
from muni_core.curves.history import build_dense_zero_curve_for_date
from muni_core.risk.callable_krd_hw_triangular import _apply_triangular_bump  # ok: internal use
from muni_core.risk.callable_krd_hw_triangular import CallableKRDResult


def export_one_bond_krd_pack_to_excel(
    *,
    out_path: str | Path,
    history_df: pd.DataFrame,
    asof: date,
    curve_key: str,
    step_years: float,
    key_tenors: Iterable[float],
    bump_bp: float,
    krd_res: CallableKRDResult,
) -> Path:
    out_path = Path(out_path)

    # Base dense curve (canonical)
    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )
    dense_base = bundle.dense_df.copy()

    # Summary sheet
    summary = {
        "base_price": krd_res.base_price,
        "bump_bp": krd_res.bump_bp,
        "krd_sum": float(sum(krd_res.krd.values())),
        "curve_mod_duration": krd_res.curve_mod_duration,
        "curve_dv01_bp": krd_res.curve_dv01_bp,
        "curve_price_up": krd_res.curve_price_up,
        "curve_price_down": krd_res.curve_price_down,
    }
    summary_df = pd.DataFrame([summary])

    # KRD table sheet
    kt = [float(x) for x in key_tenors]
    rows = []
    for k in kt:
        rows.append(
            {
                "key_tenor": k,
                "krd": krd_res.krd.get(k),
                "krc": krd_res.krc.get(k),
                "price_up": krd_res.price_up.get(k),
                "price_down": krd_res.price_down.get(k),
            }
        )
    krd_df = pd.DataFrame(rows)

    # Write workbook
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        summary_df.to_excel(xw, sheet_name="SUMMARY", index=False)
        krd_df.to_excel(xw, sheet_name="KRD_TABLE", index=False)
        dense_base.to_excel(xw, sheet_name="CURVE_BASE", index=False)

        # One overlay per key tenor
        for j, k in enumerate(kt):
            # We want weight diagnostics too, so rebuild weights by calling the same bump function
            up = _apply_triangular_bump(dense_base, key_tenors=kt, key_idx=j, bump_bp=+abs(bump_bp))
            dn = _apply_triangular_bump(dense_base, key_tenors=kt, key_idx=j, bump_bp=-abs(bump_bp))

            # Add handy diffs (bp)
            df = pd.DataFrame(
                {
                    "tenor_yrs": dense_base["tenor_yrs"].astype(float),
                    "rate_base": dense_base["rate_dec"].astype(float),
                    "rate_up": up["rate_dec"].astype(float),
                    "rate_dn": dn["rate_dec"].astype(float),
                }
            )
            df["delta_up_bp"] = (df["rate_up"] - df["rate_base"]) * 10000.0
            df["delta_dn_bp"] = (df["rate_dn"] - df["rate_base"]) * 10000.0
            df["up_minus_dn_bp"] = (df["rate_up"] - df["rate_dn"]) * 10000.0

            sheet = f"KRD_{k:g}Y"
            # Excel sheet names max 31 chars; these are fine.
            df.to_excel(xw, sheet_name=sheet, index=False)

    return out_path
