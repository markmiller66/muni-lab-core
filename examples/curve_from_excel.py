from __future__ import annotations

from datetime import date
from pathlib import Path

from muni_core.config import AppConfig
from muni_core.curves import (
    load_zero_curve_from_app_config,
    forward_rate_to_date,
    forward_curve_grid,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "example_config.yaml"

    app_cfg = AppConfig.from_yaml(cfg_path)
    zc = load_zero_curve_from_app_config(app_cfg)

    # Assume today's "settle date" for illustration
    settle = date.today()

    # Example: check zero rates at 1Y, 5Y, 10Y
    for T in [1.0, 5.0, 10.0]:
        r = zc.zero_rate(T)
        print(f"Zero rate at {T:.1f}Y: {r:.4%}")

    # Example: forward 1Y ending at year 5 (roughly)
    # We'll create a fake target date 5Y from now:
    t5_days = int(round(5.0 * 365.25))
    target_5y = date.fromordinal(settle.toordinal() + t5_days)

    f_4_to_5 = forward_rate_to_date(settle, target_5y, zc, window_years=1.0)
    print(f"Approx 1Y forward ending at 5Y (4→5Y): {f_4_to_5:.4%}")

    # Build a small forward curve grid (1Y, 3Y, 5Y, 7Y, 10Y)
    tenors = [1.0, 3.0, 5.0, 7.0, 10.0]
    fwd_grid = forward_curve_grid(settle, tenors, zc, window_years=1.0)

    print("\nForward curve grid (1Y window):")
    for T, F in fwd_grid:
        print(f"  F({T-1:.1f}->{T:.1f}Y): {F:.4%}")


if __name__ == "__main__":
    main()
