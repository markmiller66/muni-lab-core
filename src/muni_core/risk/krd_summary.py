# FILE: src/muni_core/risk/krd_summary.py
#
# PURPOSE:
#   Summarize callable HW KRD output into decision-support diagnostics:
#     - sum_krd (should ≈ curve_mod_duration for proper KRD)
#     - bucketed KRD (% of total)
#     - near-call vs far exposure
#
# IMPORTS FROM:
#   - stdlib + pandas only
#
# CALLED BY:
#   - tests/test_callable_hw_stack.py
#   - future portfolio scan / reporting pipelines

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
import math
import pandas as pd


BucketDef = Tuple[float, float, str]  # (lo, hi, label)


DEFAULT_BUCKETS: List[BucketDef] = [
    (0.0, 3.0, "0-3Y"),
    (3.0, 7.0, "3-7Y"),
    (7.0, 15.0, "7-15Y"),
    (15.0, 1000.0, "15Y+"),
]


@dataclass(frozen=True)
class KRDSummary:
    # Core
    sum_krd: float
    curve_mod_duration: float
    curve_dv01_bp: float

    # Sanity (how close is sum_krd to curve duration)
    ratio_sum_to_curve: float

    # Buckets: label -> (bucket_sum, pct_of_total)
    bucket_krd: Dict[str, float]
    bucket_pct: Dict[str, float]

    # Near vs far
    near_cutoff_years: float
    near_krd: float
    far_krd: float
    near_pct: float
    far_pct: float

    # Raw (useful for debug / plots)
    krd_by_tenor: Dict[float, float]


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _clean_krd_map(krd: Any) -> Dict[float, float]:
    """
    Accepts:
      - dict tenor->value
      - object with .krd dict
    Returns a cleaned dict[float,float] with finite values only.
    """
    if krd is None:
        return {}

    # CallableKRDResult-like
    if hasattr(krd, "krd"):
        return _clean_krd_map(getattr(krd, "krd"))

    if isinstance(krd, dict):
        out: Dict[float, float] = {}
        for k, v in krd.items():
            if not _is_finite(k) or not _is_finite(v):
                continue
            out[float(k)] = float(v)
        return out

    # Unknown shape
    return {}


def _sum_krd_map(m: Dict[float, float]) -> float:
    vals = [v for v in m.values() if _is_finite(v)]
    return float(sum(vals)) if vals else float("nan")


def _bucketize(
    krd_map: Dict[float, float],
    buckets: Sequence[BucketDef],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
      bucket_sum[label] = sum of KRD in that bucket
      bucket_pct[label] = bucket_sum / total (if total finite and non-zero)
    """
    bucket_sum: Dict[str, float] = {label: 0.0 for _, _, label in buckets}
    for tenor, val in krd_map.items():
        for lo, hi, label in buckets:
            if lo <= tenor < hi:
                bucket_sum[label] += float(val)
                break

    total = _sum_krd_map(krd_map)
    bucket_pct: Dict[str, float] = {}
    if _is_finite(total) and abs(total) > 1e-12:
        for label, s in bucket_sum.items():
            bucket_pct[label] = float(s / total)
    else:
        for label in bucket_sum.keys():
            bucket_pct[label] = float("nan")

    return bucket_sum, bucket_pct


def summarize_callable_krd(
    result: Any,
    *,
    near_cutoff_years: float = 7.0,
    buckets: Optional[Sequence[BucketDef]] = None,
) -> KRDSummary:
    """
    Summarize a CallableKRDResult-like object.

    Expected (best case):
      result.krd: dict tenor->krd
      result.curve_mod_duration: float
      result.curve_dv01_bp: float

    If curve_* diagnostics are missing, we fall back safely:
      - curve_mod_duration := sum_krd
      - curve_dv01_bp := NaN
      - ratio := 1.0 (or NaN if total not finite)
    """
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    krd_map = _clean_krd_map(result)
    sum_krd = _sum_krd_map(krd_map)

    # Pull diagnostics if present
    curve_mod_duration = getattr(result, "curve_mod_duration", float("nan"))
    curve_dv01_bp = getattr(result, "curve_dv01_bp", float("nan"))

    # Fallbacks (older result shapes)
    if not _is_finite(curve_mod_duration):
        curve_mod_duration = sum_krd

    if not _is_finite(curve_dv01_bp):
        curve_dv01_bp = float("nan")

    # Ratio
    if _is_finite(sum_krd) and _is_finite(curve_mod_duration) and abs(curve_mod_duration) > 1e-12:
        ratio = float(sum_krd / curve_mod_duration)
    else:
        ratio = float("nan")

    # Buckets
    bucket_sum, bucket_pct = _bucketize(krd_map, buckets)

    # Near vs far
    near = 0.0
    far = 0.0
    for tenor, val in krd_map.items():
        if tenor <= float(near_cutoff_years):
            near += float(val)
        else:
            far += float(val)

    total = sum_krd
    if _is_finite(total) and abs(total) > 1e-12:
        near_pct = float(near / total)
        far_pct = float(far / total)
    else:
        near_pct = float("nan")
        far_pct = float("nan")

    return KRDSummary(
        sum_krd=float(sum_krd),
        curve_mod_duration=float(curve_mod_duration),
        curve_dv01_bp=float(curve_dv01_bp),
        ratio_sum_to_curve=float(ratio),
        bucket_krd={k: float(v) for k, v in bucket_sum.items()},
        bucket_pct={k: float(v) for k, v in bucket_pct.items()},
        near_cutoff_years=float(near_cutoff_years),
        near_krd=float(near),
        far_krd=float(far),
        near_pct=float(near_pct),
        far_pct=float(far_pct),
        krd_by_tenor=dict(sorted(krd_map.items(), key=lambda kv: kv[0])),
    )


def krd_summary_to_frame(summary: KRDSummary) -> pd.DataFrame:
    """
    Convenience: represent summary as a tidy DataFrame for printing/exporting.
    """
    rows = []

    # headline
    rows.append(("sum_krd", summary.sum_krd, 1.0))
    rows.append(("curve_mod_duration", summary.curve_mod_duration, 1.0))
    rows.append(("ratio_sum_to_curve", summary.ratio_sum_to_curve, 1.0))
    rows.append(("curve_dv01_bp", summary.curve_dv01_bp, 1.0))
    rows.append((f"near_<=_{summary.near_cutoff_years:g}Y", summary.near_krd, summary.near_pct))
    rows.append((f"far_>_{summary.near_cutoff_years:g}Y", summary.far_krd, summary.far_pct))

    # buckets
    for label in summary.bucket_krd.keys():
        rows.append((f"bucket:{label}", summary.bucket_krd[label], summary.bucket_pct[label]))

    df = pd.DataFrame(rows, columns=["metric", "value", "pct_of_total"])
    return df
