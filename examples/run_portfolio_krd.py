# examples/run_portfolio_krd.py
from __future__ import annotations

"""
Portfolio-level Callable KRD runner (Hull–White lattice + triangular key-rate bumps).

For each bond row:
  1) (Optional) solve callable OAS to match market price
  2) compute callable KRD pack (triangular bumps)
  3) write summary + long-form per-tenor KRD table

Outputs (CSV):
  - portfolio_krd_summary__YYYYmmdd_HHMMSS.csv
  - portfolio_krd_tenors__YYYYmmdd_HHMMSS.csv
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from muni_core.model import Bond
from muni_core.config.loader import AppConfig
from muni_core.oas.callable_oas_hw import solve_callable_oas_hw
from muni_core.risk.callable_krd_hw_triangular import compute_callable_krd_hw


# -----------------------------
# Config / Settings
# -----------------------------

@dataclass(frozen=True)
class RunSettings:
    input_path: Path
    out_dir: Path
    config_path: Path | None
    sheet_name: str | None

    solve_oas: bool
    oas_initial_bp: float

    bump_bp: float
    step_years: float

    # compute_callable_krd_hw knobs
    freq_per_year: int = 2
    curve_key: str = "AAA_MUNI_SPOT"
    q: float = 0.5
    include_parallel_sanity: bool = True
    time_tolerance: float = 1e-6

    # Master controls workbook (optional)
    controls_path: Path | None = None
    controls_sheet: str = "Controls"


# Validate CANONICAL columns (after aliasing)
REQUIRED_COLS = ["CUSIP", "COUPON", "MATURITY", "CALL_DATE", "MARKET_PRICE"]

COL_ALIASES = {
    # canonical -> file column
    "COUPON": "Coupon",
    "MARKET_PRICE": "MarketPrice_Clean",
    "QTY": "qty",
    "SETTLE_DATE": "SETTLE_DATE",
    "FACE_VALUE": "FACE_VALUE",
    "CALL_PRICE": "CALL_PRICE",
    "RATING": "SUMMARY_RATING",
    "RATING_NUM": "SUMMARY_R_NUM",
    "BASIS": "BASISTEXT",
}


# -----------------------------
# Small utils
# -----------------------------

def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_portfolio(path: Path, sheet: str | None = None) -> pd.DataFrame:
    p = Path(str(path)).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suf = p.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(p, sheet_name=sheet or 0)
    elif suf == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported input type: {suf} (expected .csv or .xlsx/.xlsm/.xls)")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for canonical, src in COL_ALIASES.items():
        if canonical not in out.columns and src in out.columns:
            out[canonical] = out[src]
    return out


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _safe_float(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return float(x)
    except Exception:
        return None


def _get_attr(res: Any, name: str) -> Any:
    if hasattr(res, name):
        return getattr(res, name)
    if isinstance(res, dict):
        return res.get(name)
    return None


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["CUSIP"] = out["CUSIP"].astype(str).str.strip()

    cpn = pd.to_numeric(out.get("COUPON"), errors="coerce")
    out["COUPON_DEC"] = cpn.where(cpn <= 1.0, cpn / 100.0)

    out["MATURITY_DT"] = pd.to_datetime(out.get("MATURITY"), errors="coerce")
    out["CALL_DATE_DT"] = pd.to_datetime(out.get("CALL_DATE"), errors="coerce")

    out["MARKET_PRICE"] = pd.to_numeric(out.get("MARKET_PRICE"), errors="coerce")
    out["SETTLE_DATE_DT"] = pd.to_datetime(out.get("SETTLE_DATE"), errors="coerce")

    out["RATING"] = out.get("RATING")
    out["RATING"] = out["RATING"].astype(str).where(out["RATING"].notna(), "NR")

    out["RATING_NUM"] = pd.to_numeric(out.get("RATING_NUM"), errors="coerce").fillna(-1).astype(int)

    out["BASIS"] = out.get("BASIS")
    out["BASIS"] = out["BASIS"].astype(str).where(out["BASIS"].notna(), "")

    out["QTY"] = pd.to_numeric(out.get("QTY"), errors="coerce").fillna(0.0)

    if "CALL_PRICE" in out.columns:
        out["CALL_PRICE"] = pd.to_numeric(out["CALL_PRICE"], errors="coerce")

    return out


def _infer_par_amount_row(row: pd.Series) -> float | None:
    # 1) direct par fields (if present)
    for k in ("PAR_AMT", "PAR", "PAR_AMOUNT"):
        v = _safe_float(row.get(k))
        if v is not None and v > 0:
            return v

    # 2) common quantity fields
    qty = _safe_float(row.get("QTY"))
    if qty is None:
        qty = _safe_float(row.get("qty"))
    if qty is None:
        qty = _safe_float(row.get("quantity"))

    if qty is None or qty <= 0:
        return None

    face = _safe_float(row.get("FACE_VALUE"))
    if face is None:
        face = 100.0  # default muni convention

    # Heuristic: if qty is already "par dollars" (e.g., 35000), keep it.
    if qty >= 1000:
        return qty

    # Else treat as #bonds * face
    return qty * face


# -----------------------------
# Controls (optional)
# -----------------------------

def _read_controls_xlsx(path: Path, sheet: str = "Controls") -> dict[str, Any]:
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if "Key" not in df.columns or "Value" not in df.columns:
        raise ValueError(f"Controls sheet must have columns Key/Value. Found: {list(df.columns)}")

    kv: dict[str, Any] = {}
    for _, r in df.iterrows():
        k = str(r["Key"]).strip()
        if not k or k.lower() == "nan":
            continue
        kv[k] = r["Value"]
    return kv


# -----------------------------
# AppConfig + history_df
# -----------------------------

def _load_app_config(config_path: Path | None) -> AppConfig:
    if config_path is None:
        if hasattr(AppConfig, "build_default"):
            return AppConfig.build_default()  # type: ignore[attr-defined]
        raise ValueError("No --config provided and no AppConfig.build_default() found.")

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {Path(config_path).resolve()}")

    if hasattr(AppConfig, "from_yaml"):
        return AppConfig.from_yaml(config_path)  # type: ignore[attr-defined]
    if hasattr(AppConfig, "load"):
        return AppConfig.load(config_path)  # type: ignore[attr-defined]
    if hasattr(AppConfig, "from_path"):
        return AppConfig.from_path(config_path)  # type: ignore[attr-defined]

    raise ValueError("Could not load AppConfig: no supported loader method found on AppConfig.")


def _load_history_df(repo_root: Path) -> pd.DataFrame:
    # Adjust if your history file path differs
    hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing curve history file: {hist_path}")
    return pd.read_parquet(hist_path)


# -----------------------------
# Bond builder
# -----------------------------

def _bond_from_row(row: pd.Series) -> Bond:
    cusip = str(row["CUSIP"]).strip()

    settle = pd.to_datetime(row.get("SETTLE_DATE_DT"), errors="coerce")
    settle_date = settle.date() if pd.notna(settle) else None

    mat = pd.to_datetime(row.get("MATURITY_DT"), errors="coerce")
    maturity_date = mat.date() if pd.notna(mat) else None

    return Bond(
        cusip=cusip,
        rating=str(row.get("RATING", "NR")),
        rating_num=int(row.get("RATING_NUM", -1)),
        basis=str(row.get("BASIS", "")),
        settle_date=settle_date,
        maturity_date=maturity_date,
        coupon=float(row["COUPON_DEC"]),
        clean_price=float(row["MARKET_PRICE"]),
        quantity=float(row.get("QTY", 0.0)),
        call_feature=None,
    )
def _call_window_dv01_points(
    tenor_dv01_map: dict[float, float],
    call_years: float | None,
    window_years: float = 2.0
) -> float | None:
    if call_years is None:
        return None
    lo = max(call_years - window_years, 0.0)
    hi = call_years + window_years
    return sum(v for t, v in tenor_dv01_map.items() if lo <= float(t) <= hi)


from datetime import date


def _years_to_call(row: pd.Series, today: date) -> float | None:
    call_dt = row.get("CALL_DATE")
    if call_dt is None or pd.isna(call_dt):
        return None
    call_dt = pd.to_datetime(call_dt, errors="coerce")
    if pd.isna(call_dt):
        return None
    call_d = call_dt.date()
    return max((call_d - today).days / 365.25, 0.0)


# -----------------------------
# Main runner
# -----------------------------

def run_portfolio(settings: RunSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    app = _load_app_config(settings.config_path)

    repo_root = Path(__file__).resolve().parents[1]
    history_df = _load_history_df(repo_root)

    if settings.controls_path is not None:
        _ = _read_controls_xlsx(settings.controls_path, settings.controls_sheet)  # loaded for future use

    df_raw = _read_portfolio(settings.input_path, settings.sheet_name)
    df_raw = _apply_aliases(df_raw)
    _validate_input(df_raw)
    df = _coerce_types(df_raw)

    summary_rows: list[dict[str, Any]] = []
    tenor_rows: list[dict[str, Any]] = []
    today = date.today()

    for i, row in df.iterrows():
        cusip = str(row.get("CUSIP", f"row_{i}")).strip()

        try:
            call_years = _years_to_call(row, today)







            # 1) market price
            mkt = _safe_float(row.get("MARKET_PRICE"))
            if mkt is None:
                summary_rows.append({"CUSIP": cusip, "status": "SKIP", "reason": "missing MARKET_PRICE"})
                continue

            # 2) bond
            bond_obj = _bond_from_row(row)

            # 3) par + scaling
            par_amt = _infer_par_amount_row(row)
            pos_scale = (par_amt / 100.0) if (par_amt is not None and par_amt > 0) else None

            # 4) OAS solve (optional)
            if settings.solve_oas:
                oas_res = solve_callable_oas_hw(
                    bond=bond_obj,
                    history_df=history_df,
                    app_cfg=app,
                    target_price=float(mkt),
                    coupon_freq=settings.freq_per_year,
                    curve_key=settings.curve_key,
                    step_years=settings.step_years,
                    q=settings.q,
                )
                base_oas_bp = _safe_float(_get_attr(oas_res, "oas_bp")) or _safe_float(_get_attr(oas_res, "bp"))
                if base_oas_bp is None:
                    raise ValueError(f"OAS solve returned no oas_bp/bp field. Got type={type(oas_res)}")
            else:
                base_oas_bp = float(settings.oas_initial_bp)

            # 5) KRD pack (triangular bumps) - EXACT signature match ✅
            res = compute_callable_krd_hw(
                bond=bond_obj,
                history_df=history_df,
                app_cfg=app,
                base_oas_bp=float(base_oas_bp),
                key_tenors=None,  # or list[float]
                bump_bp=float(settings.bump_bp),
                freq_per_year=int(settings.freq_per_year),
                curve_key=str(settings.curve_key),
                step_years=float(settings.step_years),
                q=float(settings.q),
                time_tolerance=float(settings.time_tolerance),
                include_parallel_sanity=bool(settings.include_parallel_sanity),
            )

            base_price = _safe_float(_get_attr(res, "base_price"))
            krd_map = _get_attr(res, "krd") or {}
            krc_map = _get_attr(res, "krc") or {}
            pu_map = _get_attr(res, "price_up") or {}
            pd_map = _get_attr(res, "price_down") or {}

            curve_dv01_bp = _safe_float(_get_attr(res, "curve_dv01_bp"))
            curve_mod_duration = _safe_float(_get_attr(res, "curve_mod_duration"))
            curve_price_up = _safe_float(_get_attr(res, "curve_price_up"))
            curve_price_down = _safe_float(_get_attr(res, "curve_price_down"))

            # 6) position-scaled diagnostics
            krd_sum = sum(float(v) for v in krd_map.values()) if isinstance(krd_map, dict) else None
            dur_gap = (krd_sum - curve_mod_duration) if (krd_sum is not None and curve_mod_duration is not None) else None

            pos_base_price_points = (base_price * pos_scale) if (base_price is not None and pos_scale is not None) else None
            pos_dv01_price_points = (curve_dv01_bp * pos_scale) if (curve_dv01_bp is not None and pos_scale is not None) else None

            # 7) per-tenor DV01 contribution maps (only if base_price exists)
            tenor_dv01_map: dict[float, float] = {}
            tenor_krc_points_map: dict[float, float] = {}

            if isinstance(krd_map, dict) and base_price is not None and pos_scale is not None:
                for t in sorted(krd_map.keys()):
                    krd_val = _safe_float(krd_map.get(t))
                    if krd_val is None:
                        continue
                    tenor_dv01_map[float(t)] = base_price * float(krd_val) * 1e-4 * pos_scale

            # NOTE: KRC meaning varies; here we store a position-scaled proxy
            if isinstance(krc_map, dict) and pos_scale is not None:
                for t in sorted(krc_map.keys()):
                    krc_val = _safe_float(krc_map.get(t))
                    if krc_val is None:
                        continue
                    tenor_krc_points_map[float(t)] = float(krc_val) * pos_scale

            pos_krd_dv01_points_sum = sum(tenor_dv01_map.values()) if tenor_dv01_map else None

            # call-window DV01 points (±2y around call date)
            call_window_dv01_points = _call_window_dv01_points(
                tenor_dv01_map=tenor_dv01_map,
                call_years=call_years,
                window_years=2.0,
            )

            # NOTE: KRC meaning varies; here we store a position-scaled proxy
            if isinstance(krc_map, dict) and pos_scale is not None:
                for t in sorted(krc_map.keys()):
                    krc_val = _safe_float(krc_map.get(t))
                    if krc_val is None:
                        continue
                    tenor_krc_points_map[float(t)] = float(krc_val) * pos_scale

            pos_krd_dv01_points_sum = sum(tenor_dv01_map.values()) if tenor_dv01_map else None

            # -------- Step A: build tenor_dv01_map (NOW you have inputs) --------
            tenor_dv01_map: dict[float, float] = {}
            if isinstance(krd_map, dict) and base_price is not None and pos_scale is not None:
                for t in sorted(krd_map.keys()):
                    krd_val = _safe_float(krd_map.get(t))
                    if krd_val is None:
                        continue
                    tenor_dv01_map[float(t)] = base_price * float(krd_val) * 1e-4 * pos_scale

            pos_krd_dv01_points_sum = sum(tenor_dv01_map.values()) if tenor_dv01_map else None

            # -------- Step B: call-window DV01 points --------
            call_window_dv01_points = _call_window_dv01_points(
                tenor_dv01_map=tenor_dv01_map,
                call_years=call_years,
                window_years=2.0,
            )


            # 8) summary row (NO undefined `tenor` fields)
            summary_rows.append({
                "CUSIP": cusip,
                "status": "OK",
                "MARKET_PRICE": float(mkt),
                "base_oas_bp": float(base_oas_bp),
                "base_price_model": base_price,
                "price_error_model_minus_mkt": (base_price - float(mkt)) if base_price is not None else None,

                "bump_bp": float(settings.bump_bp),
                "step_years": float(settings.step_years),
                "curve_key": str(settings.curve_key),

                "krd_sum": krd_sum,
                "curve_mod_duration": curve_mod_duration,
                "dur_gap": dur_gap,

                "curve_dv01_bp": curve_dv01_bp,
                "curve_price_up": curve_price_up,
                "curve_price_down": curve_price_down,

                "PAR_AMT": par_amt,
                "pos_scale_par_over_100": pos_scale,
                "pos_base_price_points": pos_base_price_points,
                "pos_dv01_price_points": pos_dv01_price_points,
                "pos_krd_dv01_points_sum": pos_krd_dv01_points_sum,
                "call_years": call_years,
                "call_window_dv01_points": call_window_dv01_points,
            })

            # 9) tenor rows
            if isinstance(krd_map, dict) and krd_map:
                for tenor in sorted(krd_map.keys()):
                    t = float(tenor)
                    tenor_rows.append({
                        "CUSIP": cusip,
                        "base_oas_bp": float(base_oas_bp),
                        "bump_bp": float(settings.bump_bp),
                        "tenor": t,
                        "krd": _safe_float(krd_map.get(tenor)),
                        "krc": _safe_float(krc_map.get(tenor)) if isinstance(krc_map, dict) else None,
                        "price_up": _safe_float(pu_map.get(tenor)) if isinstance(pu_map, dict) else None,
                        "price_down": _safe_float(pd_map.get(tenor)) if isinstance(pd_map, dict) else None,
                        "pos_scale_par_over_100": pos_scale,
                        "pos_tenor_dv01_points": tenor_dv01_map.get(t),
                        "pos_tenor_krc_points": tenor_krc_points_map.get(t),
                    })


        except Exception as e:
            summary_rows.append({"CUSIP": cusip, "status": "ERROR", "reason": str(e)})

    return pd.DataFrame(summary_rows), pd.DataFrame(tenor_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="CSV/XLSX with portfolio rows")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (if input is xlsx)")
    ap.add_argument("--out-dir", default=Path("output"), type=Path)

    ap.add_argument("--config", default=None, type=Path, help="AppConfig YAML/JSON/etc")
    ap.add_argument("--solve-oas", action="store_true", help="Solve OAS per bond to match MARKET_PRICE")
    ap.add_argument("--oas-initial-bp", type=float, default=0.0)

    ap.add_argument("--bump-bp", type=float, default=1.0)
    ap.add_argument("--step-years", type=float, default=0.5)

    ap.add_argument("--freq-per-year", type=int, default=2)
    ap.add_argument("--curve-key", type=str, default="AAA_MUNI_SPOT")
    ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--time-tolerance", type=float, default=1e-6)
    ap.add_argument("--no-parallel-sanity", action="store_true")

    ap.add_argument("--controls", default=None, type=Path, help="MUNI_MASTER_BUCKET.xlsx (Controls)")
    ap.add_argument("--controls-sheet", default="Controls", help="Controls sheet name (Key/Value)")

    args = ap.parse_args()

    settings = RunSettings(
        input_path=args.input,
        out_dir=args.out_dir,
        config_path=args.config,
        sheet_name=args.sheet,
        solve_oas=bool(args.solve_oas),
        oas_initial_bp=float(args.oas_initial_bp),
        bump_bp=float(args.bump_bp),
        step_years=float(args.step_years),
        freq_per_year=int(args.freq_per_year),
        curve_key=str(args.curve_key),
        q=float(args.q),
        time_tolerance=float(args.time_tolerance),
        include_parallel_sanity=not bool(args.no_parallel_sanity),
        controls_path=args.controls,
        controls_sheet=str(args.controls_sheet),
    )

    _ensure_out_dir(settings.out_dir)

    summary_df, tenors_df = run_portfolio(settings)

    if "status" in summary_df.columns:
        print(summary_df["status"].value_counts(dropna=False))
        if (summary_df["status"] == "ERROR").any():
            print("\nTop ERROR reasons:")
            print(summary_df.loc[summary_df["status"] == "ERROR", ["CUSIP", "reason"]].head(30).to_string(index=False))

    tag = _timestamp_tag()
    summary_path = settings.out_dir / f"portfolio_krd_summary__{tag}.csv"
    tenors_path = settings.out_dir / f"portfolio_krd_tenors__{tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    tenors_df.to_csv(tenors_path, index=False)

    # --- Aggregate per CUSIP (OK rows only) ---
    ok_df = summary_df[summary_df.get("status", "OK") == "OK"].copy() if "status" in summary_df.columns else summary_df.copy()
    if ok_df.empty:
        raise RuntimeError("No OK rows to aggregate.")

    want_sum = ["PAR_AMT", "pos_base_price_points", "pos_dv01_price_points", "pos_krd_dv01_points_sum"]
    want_first = ["curve_key", "step_years", "bump_bp"]

    sum_cols = [c for c in want_sum if c in ok_df.columns]
    first_cols = [c for c in want_first if c in ok_df.columns]
    agg_spec = {**{c: "sum" for c in sum_cols}, **{c: "first" for c in first_cols}}

    summary_by_cusip = ok_df.groupby("CUSIP", as_index=False).agg(agg_spec)

    summary_by_cusip["pos_scale_par_over_100"] = summary_by_cusip["PAR_AMT"] / 100.0
    summary_by_cusip["base_price_per_100_implied"] = (
        summary_by_cusip["pos_base_price_points"] / summary_by_cusip["pos_scale_par_over_100"]
    ).where(summary_by_cusip["pos_scale_par_over_100"] != 0, None)

    summary_by_cusip["dv01_gap_points"] = (
        summary_by_cusip["pos_krd_dv01_points_sum"] - summary_by_cusip["pos_dv01_price_points"]
    )
    summary_by_cusip["dv01_gap_pct"] = (
        summary_by_cusip["dv01_gap_points"] / summary_by_cusip["pos_dv01_price_points"]
    ).where(summary_by_cusip["pos_dv01_price_points"] != 0, None)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {tenors_path}")
    print(f"Rows: summary={len(summary_df)} tenors={len(tenors_df)}")

    # Optionally save aggregate:
    # agg_path = settings.out_dir / f"portfolio_krd_by_cusip__{tag}.csv"
    # summary_by_cusip.to_csv(agg_path, index=False)
    # print(f"Wrote: {agg_path}")
    # --- Portfolio DV01 contribution by LOT (rows) ---
    ok_df = summary_df[summary_df.get("status") == "OK"].copy()

    if not ok_df.empty and "pos_dv01_price_points" in ok_df.columns:
        port_dv01 = ok_df["pos_dv01_price_points"].sum()
        ok_df["dv01_contrib_pct"] = ok_df["pos_dv01_price_points"] / port_dv01 if port_dv01 else None

        contrib_path = settings.out_dir / f"portfolio_krd_lot_contrib__{tag}.csv"
        ok_df.to_csv(contrib_path, index=False)
        print(f"Wrote: {contrib_path} (portfolio DV01 points={port_dv01:.6f})")

    # --- DV01 contribution by CUSIP (aggregated lots) ---
    want = ["PAR_AMT", "pos_base_price_points", "pos_dv01_price_points", "pos_krd_dv01_points_sum"]
    have = [c for c in want if c in ok_df.columns]

    if not ok_df.empty and have:
        by_cusip = ok_df.groupby("CUSIP", as_index=False)[have].sum()
        port_dv01 = by_cusip["pos_dv01_price_points"].sum()
        by_cusip["dv01_contrib_pct"] = by_cusip["pos_dv01_price_points"] / port_dv01 if port_dv01 else None

        cusip_path = settings.out_dir / f"portfolio_krd_cusip_contrib__{tag}.csv"
        by_cusip.to_csv(cusip_path, index=False)
        print(f"Wrote: {cusip_path}")

    # --- Tenor DV01 contribution (portfolio-level) ---
    ok_ten = tenors_df[tenors_df.get("pos_tenor_dv01_points").notna()].copy()
    if not ok_ten.empty:
        by_tenor = ok_ten.groupby("tenor", as_index=False)["pos_tenor_dv01_points"].sum()
        port_dv01 = by_tenor["pos_tenor_dv01_points"].sum()
        by_tenor["dv01_contrib_pct"] = by_tenor["pos_tenor_dv01_points"] / port_dv01 if port_dv01 else None

        tenor_path = settings.out_dir / f"portfolio_krd_tenor_contrib__{tag}.csv"
        by_tenor.to_csv(tenor_path, index=False)
        print(f"Wrote: {tenor_path}")


if __name__ == "__main__":
    main()
