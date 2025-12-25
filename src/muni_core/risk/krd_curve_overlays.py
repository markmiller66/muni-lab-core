from __future__ import annotations

"""
FILE: src/muni_core/risk/krd_curve_overlays.py

PURPOSE
-------
Build and export Option-A (triangular) KRD curve overlays:

- Base curve (dense_df)
- For each key tenor:
    - triangular weights across dense nodes
    - bumped-up curve (+bump_bp * weight)
    - bumped-down curve (-bump_bp * weight)

This module is DEBUG/EXPORT oriented only.
It does NOT do pricing; it only constructs curve overlays for inspection.

DEPENDENCIES
------------
- muni_core.curves.build_curve_bundle.build_curve_bundle
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class KRDOverlayBundle:
    asof: date
    curve_key: str
    bump_bp: float
    key_tenors: List[float]
    base_df: pd.DataFrame
    overlays: dict[float, pd.DataFrame]  # tenor -> overlay df


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Have: {list(df.columns)}")


def _triangular_weights(tenors: pd.Series, *, key_tenors: list[float], key_idx: int) -> pd.Series:
    """
    Returns weights in [0,1] for each dense node tenor.

    Interior key:
        support [left, right], peak at center
    Edge key:
        half-triangle toward neighbor
    """
    ks = [float(x) for x in key_tenors]
    k = float(ks[key_idx])

    left = ks[key_idx - 1] if key_idx > 0 else None
    right = ks[key_idx + 1] if key_idx < (len(ks) - 1) else None

    tvals = tenors.astype(float).to_numpy()
    w = [0.0] * len(tvals)

    for i, ti in enumerate(tvals):
        wi = 0.0
        if left is None and right is not None:
            # left edge: half triangle on [k, right], peak at k
            if k <= ti <= right:
                wi = (right - ti) / (right - k) if right != k else 0.0
        elif right is None and left is not None:
            # right edge: half triangle on [left, k], peak at k
            if left <= ti <= k:
                wi = (ti - left) / (k - left) if k != left else 0.0
        else:
            # interior full triangle
            if left <= ti <= k:
                wi = (ti - left) / (k - left) if k != left else 0.0
            elif k < ti <= right:
                wi = (right - ti) / (right - k) if right != k else 0.0

        w[i] = max(0.0, min(1.0, wi))

    return pd.Series(w, index=tenors.index, dtype="float64")


def build_krd_curve_overlay_bundle(
    *,
    history_df: pd.DataFrame,
    curve_key: str,
    asof: date,
    step_years: float,
    key_tenors: list[float] | None = None,
    bump_bp: float = 1.0,
) -> KRDOverlayBundle:
    """
    Build base + all key-tenor overlays for triangular-bump KRD.

    Output overlay df columns per key tenor:
      - tenor_yrs
      - rate_base
      - weight
      - bump_dec
      - rate_up
      - rate_dn
      - delta_up_bp
      - delta_dn_bp
      - up_minus_dn_bp   (should be 2 * bump_bp * weight)
    """
    from muni_core.curves.build_curve_bundle import build_curve_bundle

    if key_tenors is None:
        key_tenors = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    key_tenors = [float(x) for x in key_tenors]

    bundle = build_curve_bundle(
        history_df=history_df,
        curve_key=curve_key,
        asof=asof,
        step_years=step_years,
    )
    base_df = bundle.dense_df.copy()
    _require_cols(base_df, ["tenor_yrs", "rate_dec"])

    overlays: dict[float, pd.DataFrame] = {}
    bump_dec = float(bump_bp) / 10000.0

    for j, kt in enumerate(key_tenors):
        w = _triangular_weights(base_df["tenor_yrs"], key_tenors=key_tenors, key_idx=j)

        df = pd.DataFrame(
            {
                "tenor_yrs": base_df["tenor_yrs"].astype(float),
                "rate_base": base_df["rate_dec"].astype(float),
                "weight": w.astype(float),
                "bump_dec": bump_dec,
            }
        )

        df["rate_up"] = df["rate_base"] + df["weight"] * bump_dec
        df["rate_dn"] = df["rate_base"] - df["weight"] * bump_dec

        df["delta_up_bp"] = (df["rate_up"] - df["rate_base"]) * 10000.0
        df["delta_dn_bp"] = (df["rate_dn"] - df["rate_base"]) * 10000.0
        df["up_minus_dn_bp"] = (df["rate_up"] - df["rate_dn"]) * 10000.0  # expect 2*bump_bp*weight

        overlays[float(kt)] = df

    return KRDOverlayBundle(
        asof=asof,
        curve_key=str(curve_key),
        bump_bp=float(bump_bp),
        key_tenors=key_tenors,
        base_df=base_df,
        overlays=overlays,
    )


def export_krd_curve_overlays_to_excel(
    *,
    overlay_bundle: KRDOverlayBundle,
    out_path: Path,
) -> Path:
    """
    Writes:
      - Base sheet
      - Summary sheet
      - One sheet per key tenor
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = overlay_bundle.base_df.copy()
    _require_cols(base, ["tenor_yrs", "rate_dec"])
    base_sheet = base[["tenor_yrs", "rate_dec"]].rename(columns={"rate_dec": "rate_base"})

    # Summary table: how big is each key bump overall?
    rows = []
    for kt, df in overlay_bundle.overlays.items():
        w = df["weight"]
        rows.append(
            {
                "key_tenor": kt,
                "max_weight": float(w.max()),
                "min_weight": float(w.min()),
                "nonzero_nodes": int((w > 0).sum()),
                "sum_weights": float(w.sum()),
                "max_up_minus_dn_bp": float(df["up_minus_dn_bp"].max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("key_tenor").reset_index(drop=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        base_sheet.to_excel(xw, sheet_name="BASE", index=False)
        summary.to_excel(xw, sheet_name="SUMMARY", index=False)

        # Key tenor sheets
        for kt in overlay_bundle.key_tenors:
            df = overlay_bundle.overlays[float(kt)]
            name = f"KRD_{kt:g}Y"
            # Excel sheet name limit = 31 chars; we're safe.
            df.to_excel(xw, sheet_name=name[:31], index=False)

    return out_path
