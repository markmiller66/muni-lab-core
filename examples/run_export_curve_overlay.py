from datetime import date
from pathlib import Path
import pandas as pd

from muni_core.curves.export_curve_debug import export_curve_overlay_to_excel

# --- paths ---
repo_root = Path(__file__).resolve().parents[1]
hist_path = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"

history_df = pd.read_parquet(hist_path)

out_path = export_curve_overlay_to_excel(
    history_df=history_df,
    curve_key="AAA_MUNI_SPOT",
    asof=date(2025, 11, 26),
    step_years=0.5,
    bump_bp=1.0,
)



print("Wrote:", out_path)
