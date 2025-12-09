import os
import pandas as pd
import numpy as np
from scipy.optimize import newton, brentq
from datetime import datetime, timedelta

# Constants
CURRENT_DATE = datetime(2025, 12, 8)  # As per query
FACE_VALUE = 100.0


# Step 1: Load Data from Files (robust paths: bonds in data/working/)
def load_data(bonds_excel=os.path.join('data', 'working', 'my_300_bonds.xlsx'),
              yield_csv=os.path.join('data', 'Tradeweb_MUNI_data (4).csv')):
    # Compute project root from script location (src/muni_core/curves/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # Up to muni-lab-core

    # Full paths
    bonds_path = os.path.join(project_root, bonds_excel)
    yield_path = os.path.join(project_root, yield_csv)

    # Bonds: From 'Bonds' sheet
    bonds_df = pd.read_excel(bonds_path, sheet_name='Bonds', engine='openpyxl')
    bonds_df = bonds_df.dropna(subset=['CUSIP'])  # Drop empty rows

    # Convert serial dates to datetime (handle if already datetime)
    def serial_to_date(serial):
        if pd.isna(serial):
            return None
        if isinstance(serial, (datetime, pd.Timestamp)):
            return serial  # Already datetime
        try:
            base = datetime(1899, 12, 30)  # Excel 1900 start + leap adjust
            return base + timedelta(days=float(serial))
        except (ValueError, TypeError):
            return None  # Invalid, skip

    bonds_df['MATURITY_DATE'] = bonds_df['MATURITY'].apply(serial_to_date)
    bonds_df['MATURITY_YEARS'] = bonds_df.apply(
        lambda row: ((row['MATURITY_DATE'] - CURRENT_DATE).days / 365.25) if row['MATURITY_DATE'] else np.nan,
        axis=1
    ).clip(lower=0)

    # Tradeweb: Latest par yields (for curve)
    tradeweb_df = pd.read_csv(yield_path, parse_dates=['Date Of'])
    tradeweb_df = tradeweb_df.sort_values('Date Of').reset_index(drop=True)
    latest_yields = tradeweb_df.iloc[-1].drop('Date Of').astype(float).values / 100
    annual_mats = np.arange(1, 31)

    return bonds_df, latest_yields, annual_mats


# Step 2: Bootstrap Spot Curve (semi-annual, continuous compounding)
def build_spot_curve(par_yields, annual_mats):
    semi_mats = np.arange(0.5, 30.5, 0.5)
    semi_par_yields = np.interp(semi_mats, annual_mats, par_yields)

    n = len(semi_mats)
    dfs = np.zeros(n + 1)
    dfs[0] = 1.0

    for i in range(1, n + 1):
        y_annual = semi_par_yields[i - 1]
        y_semi = y_annual / 2

        def price_diff(df_new):
            cfs = np.full(i, y_semi)
            cfs[-1] += 1.0
            pv_coupons = np.sum(cfs[:-1] * dfs[1:i]) if i > 1 else 0.0
            pv_last = cfs[-1] * df_new
            return pv_coupons + pv_last - 1.0

        dfs[i] = newton(price_diff, 0.9)

    spot_rates = -np.log(dfs[1:]) / semi_mats
    return spot_rates, semi_mats


# Step 3: Compute Z-Spread for a Bond (semi-annual coupons, face=100, use clean price)
def compute_z_spread(clean_price, coupon_annual, maturity_years, spot_rates, semi_mats, liquidity_adjust=0.0):
    if maturity_years <= 0 or pd.isna(maturity_years) or pd.isna(clean_price):
        return np.nan

    num_periods = int(2 * maturity_years)
    coupon_semi = (coupon_annual / 2) * FACE_VALUE
    cfs = np.full(num_periods, coupon_semi)
    cfs[-1] += FACE_VALUE

    times = np.arange(0.5, maturity_years + 0.0001, 0.5)

    def pv_diff(z):
        times_clipped = np.clip(times, semi_mats[0], semi_mats[-1])
        adj_spots = np.interp(times_clipped, semi_mats, spot_rates) + np.clip(z, -0.10, 0.10)
        discounts = np.exp(-adj_spots * times_clipped)
        pv = np.sum(cfs * discounts)
        return pv - clean_price  # Clean price (accrued separate in file)

    pv_diff0 = pv_diff(0)
    rough_dur = maturity_years
    a = -0.05 if pv_diff0 < 0 else -0.20
    b = 0.05 if pv_diff0 > 0 else 0.20

    try:
        z_base = brentq(pv_diff, a, b)
    except ValueError:
        z_base = np.nan

    return (z_base + liquidity_adjust) * 10000 if not np.isnan(z_base) else np.nan  # bps


# Main: Compute Z-Spreads and Output to Excel
if __name__ == "__main__":
    bonds_df, par_yields, annual_mats = load_data()
    spot_rates, semi_mats = build_spot_curve(par_yields, annual_mats)

    # Compute Z-spreads (liquidity_adjust=0 for now; adapt if needed)
    bonds_df['Z_SPREAD_BPS'] = bonds_df.apply(
        lambda row: compute_z_spread(row['MarketPrice_Clean'], row['Coupon'], row['MATURITY_YEARS'], spot_rates,
                                     semi_mats),
        axis=1
    )

    # Output to new Excel in data/working/ (append date-time)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    output_dir = os.path.join(project_root, 'data', 'working')
    output_file = os.path.join(output_dir, f"bonds_with_zspreads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    bonds_df[['CUSIP', 'Coupon', 'MATURITY_YEARS', 'MarketPrice_Clean', 'Z_SPREAD_BPS']].to_excel(output_file,
                                                                                                  index=False)
    print(f"Output saved to: {output_file}")