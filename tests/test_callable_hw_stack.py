"""
FILE: tests/test_callable_hw_stack.py

PURPOSE
-------
End-to-end integration test for the callable Hull–White / Ho–Lee analytics stack
(Option A: full-curve KRD).

This test validates correctness and internal consistency across:
- Callable pricing vs OAS
- Callable OAS solver
- Callable DV01 / modified duration
- Callable Key Rate Duration (KRD) — full curve
- KRD summarization (post-process only, no math inside engine)

This file is intentionally SELF-CONTAINED and EXECUTABLE.


DEPENDENCY CONTRACT (DO NOT VIOLATE)
-----------------------------------
This test relies on the following production modules ONLY:

Pricing / OAS:
- muni_core.pricing.hw_bond_pricer_override
- muni_core.oas.callable_oas_hw
- muni_core.oas.callable_dv01_hw

Risk:
- muni_core.risk.callable_krd_hw          (Option A full-curve KRD engine)
- muni_core.risk.krd_summary              (post-process summarization ONLY)

Curves / Config:
- muni_core.curves.history
- muni_core.config.AppConfig
- muni_core.model (Bond, CallFeature)

No shims, no silent imports, no monkey patching.


NAMING RULES (CRITICAL — ENFORCED BY CONVENTION)
------------------------------------------------
To prevent module/scalar shadowing bugs:

- Module imports MUST end in `_mod` or `_module`


- Numeric scalars MUST end in `_val` or be obvious attributes
    e.g.:
        krd_sum_val = summary.sum_krd
        curve_dur_val = summary.curve_mod_duration

- NEVER alias a module to a scalar name
    (e.g. `krd_sum = <module>` is forbidden)

- Summary objects are always named `summary`
- Raw KRD engine output is always named `krd_res`


DESIGN GUARANTEES (DO NOT CHANGE)
--------------------------------
- Option A = full-curve KRD only

- KRD summarization is a pure post-process
- sum(KRD) ≈ curve modified duration (within tolerance)
- No business logic in tests beyond sanity assertions


DEBUGGING POLICY
----------------
- Print statements are allowed in this file
- This test may be run interactively
- Failures should be explainable via printed diagnostics


LAST KNOWN GOOD STATE
---------------------
- Callable OAS converges and is monotonic
- Callable DV01 / ModDuration are consistent
- sum(KRD) ≈ curve_mod_duration ≈ 8.48
- Near vs far KRD split validated (≈ 69% / 31%)

EXECUTION
---------
- As pytest:   pytest -q tests/test_callable_hw_stack.py
- As module:   python -m tests.test_callable_hw_stack
"""
from __future__ import annotations

from pathlib import Path
from datetime import date
import math

import pandas as pd

from muni_core.config import AppConfig
from muni_core.model import Bond, CallFeature

from muni_core.curves.history import build_dense_zero_curve_for_date
from muni_core.pricing.hw_bond_pricer_override import (
    price_callable_bond_hw_from_bond_dense_override,
)

from muni_core.oas.callable_oas_hw import solve_callable_oas_hw
from muni_core.oas.callable_dv01_hw import compute_callable_oas_dv01_hw
from muni_core.risk.callable_krd_hw_triangular import compute_callable_krd_hw

# KRD summarization is post-process only (no math inside KRD engine)
from muni_core.risk import summarize_callable_krd, krd_summary_to_frame






# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _get_asof(cfg: AppConfig, history_df: pd.DataFrame) -> date:
    if cfg.curves.curve_asof_date:
        return pd.to_datetime(cfg.curves.curve_asof_date).date()
    tmp = history_df.copy()
    tmp["date"] = pd.to_datetime(tmp["date"]).dt.date
    return tmp["date"].max()


def _normalize_dense_df(dense_df: pd.DataFrame) -> pd.DataFrame:
    df = dense_df.copy()

    # tenor normalization
    if "tenor_years" not in df.columns:
        if "tenor_yrs" in df.columns:
            df["tenor_years"] = df["tenor_yrs"].astype(float)

    # zero-rate normalization
    if "zero_rate" not in df.columns:
        for c in ["rate_dec", "ZeroRate", "zero", "rate", "spot_rate", "spot", "zr"]:
            if c in df.columns:
                df["zero_rate"] = df[c].astype(float)
                break

    if "zero_rate" not in df.columns:
        raise KeyError(f"dense_df missing 'zero_rate'. Columns: {list(df.columns)}")

    return df


def _get_hw_params(cfg: AppConfig) -> tuple[float, float]:
    # Prefer Controls if AppConfig exposes get_control_value
    a_raw = None
    sigma_raw = None
    if hasattr(cfg, "get_control_value"):
        try:
            a_raw = cfg.get_control_value("HW_A", default=None)
            sigma_raw = cfg.get_control_value("HW_SIGMA_BASE", default=None)
        except Exception:
            a_raw = None
            sigma_raw = None

    a = float(a_raw) if a_raw not in (None, "") else 0.10
    sigma = float(sigma_raw) if sigma_raw not in (None, "") else 0.01
    return a, sigma


def _sum_krd(krd_obj) -> float:
    """
    Sum KRD contributions (finite-safe) regardless of return shape.

    Accepts:
      - CallableKRDResult (uses .krd)
      - dict tenor->value
      - pandas Series/DataFrame
      - list of numbers or (tenor,value)
    """
    if krd_obj is None:
        return float("nan")

    # CallableKRDResult-like
    if hasattr(krd_obj, "krd"):
        return _sum_krd(getattr(krd_obj, "krd"))

    # dict tenor->value
    if isinstance(krd_obj, dict):
        vals = []
        for v in krd_obj.values():
            if v is None:
                continue
            fv = float(v)
            if math.isfinite(fv):
                vals.append(fv)
        return float(sum(vals)) if vals else float("nan")

    # pandas Series / DataFrame
    if hasattr(krd_obj, "ndim"):
        # Series
        if krd_obj.ndim == 1:
            vals = []
            for v in krd_obj.values:
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
            return float(sum(vals)) if vals else float("nan")

        # DataFrame
        if hasattr(krd_obj, "columns"):
            for col in ("krd", "dv01", "value"):
                if col in krd_obj.columns:
                    s = pd.to_numeric(krd_obj[col], errors="coerce")
                    s = s[pd.notna(s)]
                    s = s[s.apply(lambda x: math.isfinite(float(x)))]
                    return float(s.sum()) if len(s) else float("nan")

            dfnum = krd_obj.select_dtypes("number")
            return float(dfnum.sum().sum()) if not dfnum.empty else float("nan")

    # list/tuple of numbers or (tenor,val)
    if isinstance(krd_obj, (list, tuple)):
        total = 0.0
        any_ok = False
        for item in krd_obj:
            if isinstance(item, (int, float)):
                fv = float(item)
                if math.isfinite(fv):
                    total += fv
                    any_ok = True
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                fv = float(item[1])
                if math.isfinite(fv):
                    total += fv
                    any_ok = True
        return float(total) if any_ok else float("nan")

    return float("nan")


# ---------------------------------------------------------------------
# Main test (pytest-friendly) + executable entrypoint
# ---------------------------------------------------------------------

def test_callable_hw_stack() -> None:
    # Config + history
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "example_config.yaml"

    cfg = AppConfig.from_yaml(cfg_path)
    history_df = pd.read_parquet(cfg.curves.history_file)

    print("History file:", cfg.curves.history_file)
    print("History rows:", len(history_df))

    # Test bond (known-good from earlier runs)
    bond = Bond(
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

    print("Bond:", bond)

    # Tier 1: Build dense curve + sanity price monotonicity vs OAS (override pricer)
    asof = _get_asof(cfg, history_df)
    curve_key = "AAA_MUNI_SPOT"
    step_years = 0.5
    q = 0.5
    freq = 2
    face = 100.0
    time_tolerance = 1e-6

    a, sigma = _get_hw_params(cfg)

    dense_raw = build_dense_zero_curve_for_date(
        history_df=history_df,
        curve_key=curve_key,
        target_date=asof,
        step_years=step_years,
    )
    dense_df = _normalize_dense_df(dense_raw)

    # sanity: check the curve bump changes something (debug only)
    dense_chk1 = dense_df.copy()
    dense_chk1["zero_rate"] = dense_chk1["zero_rate"] + 0.01  # +100bp
    if dense_df["zero_rate"].equals(dense_chk1["zero_rate"]):
        print("[WARN] Dense curve bump sanity check failed (zero_rate unchanged).")

    print(f"\n[Dense] asof={asof} curve_key={curve_key} rows={len(dense_df)}")
    print(f"[HW] a={a} sigma={sigma} step_years={step_years} q={q}")

    oas_grid = [-2000.0, -1000.0, 0.0, 1000.0, 2000.0]

    prices = []
    for bp in oas_grid:
        p = price_callable_bond_hw_from_bond_dense_override(
            bond=bond,
            asof=asof,
            dense_df=dense_df,
            a=a,
            sigma=sigma,
            oas_bp=float(bp),
            freq_per_year=freq,
            face=face,
            step_years=step_years,
            q=q,
            time_tolerance=time_tolerance,
        )
        prices.append(float(p))
        print(f"[Monotonicity] OAS {bp:8.1f} bp -> price {p:12.6f}")

    mono_ok = all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))
    moved = (max(prices) - min(prices)) > 1e-6

    print("[Monotonicity] non-increasing vs OAS:", mono_ok)
    print("[Monotonicity] price moved across grid:", moved)

    if not moved:
        print(
            "[WARN] Prices did not change across OAS grid — "
            "override pricer may not be feeding OAS into theta/lattice."
        )
    if not mono_ok:
        print("[WARN] Price is not monotone decreasing in OAS. Investigate curve bumping or call logic.")

    def _bump_parallel_rate_dec(df: pd.DataFrame, bump_bp: float) -> pd.DataFrame:
        out = df.copy()
        # bump the curve itself (NOT the oas_bp input)
        out["zero_rate"] = out["zero_rate"].astype(float) + (bump_bp / 10000.0)
        return out

    dense_up = _bump_parallel_rate_dec(dense_df, +1.0)
    dense_dn = _bump_parallel_rate_dec(dense_df, -1.0)


    p0_det = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_df,
        a=a,
        sigma=sigma,
        oas_bp=0.0,
        freq_per_year=freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )
    pup_det = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_up,
        a=a,
        sigma=sigma,
        oas_bp=0.0,
        freq_per_year=freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )
    pdn_det = price_callable_bond_hw_from_bond_dense_override(
        bond=bond,
        asof=asof,
        dense_df=dense_dn,
        a=a,
        sigma=sigma,
        oas_bp=0.0,
        freq_per_year=freq,
        face=face,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
        allow_call=False,
    )

    dv01_det = (float(pdn_det) - float(pup_det)) / 2.0
    mod_dur_det = dv01_det * 10000.0 / float(p0_det)



    # Tier 2: Solve callable OAS
    oas_res = solve_callable_oas_hw(
        bond=bond,
        history_df=history_df,
        app_cfg=cfg,
        target_price=bond.clean_price,
        coupon_freq=freq,
        bp_low=-2000.0,
        bp_high=2000.0,
        tol=1e-6,
        max_iter=60,
        curve_key=curve_key,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
    )

    print("\n[Callable OAS] ", oas_res)
    assert oas_res is not None
    assert oas_res.converged, "Callable OAS did not converge"
    base_oas = float(oas_res.oas_bp)

    # Tier 3: Callable OAS DV01 / duration
    dv01_res = compute_callable_oas_dv01_hw(
        bond=bond,
        history_df=history_df,
        app_cfg=cfg,
        base_oas_bp=base_oas,
        bump_bp=1.0,
        coupon_freq=freq,
        curve_key=curve_key,
        step_years=step_years,
        q=q,
        time_tolerance=time_tolerance,
    )

    print("\n[Callable DV01] ", dv01_res)

    # Basic reasonableness checks
    assert dv01_res.price_up <= dv01_res.price_base + 1e-9, "Price should fall when OAS increases"
    assert dv01_res.price_down >= dv01_res.price_base - 1e-9, "Price should rise when OAS decreases"
    assert dv01_res.dv01_bp >= 0.0, "DV01 should be non-negative for normal bonds"
    assert dv01_res.mod_duration > 0.0, "ModDuration should be positive"

    # Tier 4: Callable KRD (Option A) + summary (post-process only)
    krd_res = compute_callable_krd_hw(
        bond=bond,
        history_df=history_df,
        app_cfg=cfg,
        base_oas_bp=base_oas,
        curve_key=curve_key,
        step_years=step_years,
        q=q,
    )

    print("\n[Callable KRD] ", krd_res)

    # ---- Pretty KRD/KRC table (by tenor) ----
    rows = []
    for t in krd_res.key_tenors:
        k = float(krd_res.krd.get(t, float("nan")))
        c = float(krd_res.krc.get(t, float("nan"))) if hasattr(krd_res, "krc") and krd_res.krc else float("nan")
        pu = float(krd_res.price_up.get(t, float("nan")))
        pdown = float(krd_res.price_down.get(t, float("nan")))

        rows.append({"tenor_y": float(t), "krd": k, "krc": c, "price_up": pu, "price_down": pdown})

    krd_df = pd.DataFrame(rows).sort_values("tenor_y").reset_index(drop=True)
    sum_krd_val = float(krd_df["krd"].sum())
    krd_df["pct_of_sum"] = krd_df["krd"] / sum_krd_val if sum_krd_val and math.isfinite(sum_krd_val) else float("nan")

    print("\n[KRD Nodes] (tenor bumps, Option A)")
    print(krd_df.to_string(index=False, formatters={
        "tenor_y": "{:.1f}".format,
        "krd": "{:.6f}".format,
        "krc": "{:.6f}".format,
        "price_up": "{:.6f}".format,
        "price_down": "{:.6f}".format,
        "pct_of_sum": "{:.3%}".format,
    }))
    print(f"\n[KRD Nodes] sum(krd)={sum_krd_val:.6f}  curve_mod_duration={krd_res.curve_mod_duration:.6f}")

    summary = summarize_callable_krd(krd_res, near_cutoff_years=7.0)

    print("\n[KRD Summary]")
    print(f"  sum_krd             = {summary.sum_krd:.6f}")
    print(f"  curve_mod_duration  = {summary.curve_mod_duration:.6f}")
    print(f"  ratio(sum/curve)    = {summary.ratio_sum_to_curve:.6f}")
    print(f"  near<=7Y pct        = {summary.near_pct:.3%}")
    print(f"  far>7Y pct          = {summary.far_pct:.3%}")

    print("\n[KRD Buckets]")
    for label, pct in summary.bucket_pct.items():
        print(f"  {label:>6s}: {pct:.3%}  (krd={summary.bucket_krd[label]:.6f})")

    print("\n[KRD Summary Table]")
    print(krd_summary_to_frame(summary).to_string(index=False))

    # Hard invariant: Option A full-curve KRD sum matches curve duration (within tolerance)
    assert summary.curve_mod_duration > 0.0
    assert abs(summary.ratio_sum_to_curve - 1.0) < 1e-3, "Sum(KRD) should match curve duration in Option A"

    # Informational: compare KRD sum to OAS duration (not apples-to-apples, so no strict assert)
    if dv01_res.mod_duration > 0:
        ratio_vs_oas = summary.sum_krd / dv01_res.mod_duration
        print(f"\n[Info] sumKRD / oasModDur ≈ {ratio_vs_oas:.3f}")

    print("\n[OK] Callable HW stack test completed.\n")


if __name__ == "__main__":
    # Allows: python -m tests.test_callable_hw_stack
    test_callable_hw_stack()
