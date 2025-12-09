import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

# Load Treasury data (adjust path if needed)
treasury_df = pd.read_csv(r'data/feds200628.csv', skiprows=9, low_memory=False)
treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
treasury_df.set_index('Date', inplace=True)

# Load Muni data (adapt path if needed)
muni_df = pd.read_csv(r'data/Tradeweb_MUNI_data (4).csv', low_memory=False, on_bad_lines='skip')
muni_df['Date Of'] = pd.to_datetime(muni_df['Date Of'])
muni_df = muni_df.sort_values('Date Of').set_index('Date Of')

# Load U Chicago VIX (adapt path if needed)
vix_df = pd.read_csv(r'data/muni_vix_data (2).csv', low_memory=False)
vix_df['date'] = pd.to_datetime(vix_df['date'])
vix_df.set_index('date', inplace=True)


# Function to bootstrap spot curve from par yields (semi-annual, continuous)
def bootstrap_spot(par_yields):
    annual_mats = np.arange(1, 31)
    semi_mats = np.arange(0.5, 30.5, 0.5)
    semi_par_yields = np.interp(semi_mats, annual_mats, par_yields)

    n = len(semi_mats)
    dfs = [1.0]

    for i in range(1, n + 1):
        y_semi = semi_par_yields[i - 1] / 2
        sum_prev = sum(y_semi * dfs[j] for j in range(1, i))
        df_new = (1 - sum_prev) / (1 + y_semi)

        if df_new <= 0 or np.isnan(df_new):
            # Fallback: use current interpolated par as approximate spot for DF
            df_new = np.exp(-semi_par_yields[i - 1] * semi_mats[i - 1])

        dfs.append(df_new)

    dfs = np.array(dfs)

    # Compute continuous spots
    with np.errstate(invalid='ignore'):
        spot_rates = -np.log(dfs[1:]) / semi_mats
    spot_rates = np.nan_to_num(spot_rates, nan=0.0, posinf=0.0, neginf=0.0)

    return spot_rates[1::2], dfs[2::2]  # Annual spots and DFs (2,4,...60 semis → 1y to 30y)


# Function to compute (discrete) instantaneous forward rates from spots
def compute_forwards(spot_rates):
    """
    spot_rates: array-like of annual zero rates r(T) in DECIMAL (e.g. 0.01 = 1%)
    returns: f(0,1Y), f(1Y,2Y), ..., f(29Y,30Y) in decimal
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    T = np.arange(1, len(spot_rates) + 1, dtype=float)  # 1..N in years

    forwards = np.zeros_like(spot_rates)
    forwards[0] = spot_rates[0]  # use first spot as 0–1Y forward

    # f_{i-1,i} = (T_i r_i - T_{i-1} r_{i-1}) / (T_i - T_{i-1})
    forwards[1:] = (T[1:] * spot_rates[1:] - T[:-1] * spot_rates[:-1]) / (T[1:] - T[:-1])

    return forwards



# Function to calibrate Hull-White params and generate short rate curve (mean paths)
def calibrate_hw(spot_rates, forwards, date, num_paths=100, years=30, is_treasury=False):
    """
    spot_rates: annual zero rates r(1Y..N) in decimal
    forwards:   discrete forwards between years (same length, decimal)
    date:       pandas Timestamp (for looking up muni VIX)
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    forwards = np.asarray(forwards, dtype=float)

    # --- 1. Choose sigma -------------------------------------------
    if not is_treasury:
        # Pull volatility proxy from muni VIX table
        if date in vix_df.index:
            vix = float(vix_df.loc[date, 'Weighted Average'])
        else:
            prev_dates = vix_df.index[vix_df.index < date]
            if len(prev_dates) > 0:
                vix = float(vix_df.loc[prev_dates[-1], 'Weighted Average'])
            else:
                vix = float(vix_df['Weighted Average'].iloc[-1])

        # Assume "Weighted Average" is a % annualized vol (e.g. 50 → 50%)
        # Use some fraction of it as the short-rate sigma (tunable).
        sigma = (vix / 100.0) * 0.5   # e.g. 50% index → σ = 0.25
    else:
        # Simple starting point for Treasuries; you can calibrate later
        sigma = 0.01                  # 1% annual short-rate vol

    # --- 2. Mean reversion -----------------------------------------
    a = 0.05  # 5% mean reversion speed (can be calibrated later)

    # Limit years so arrays line up
    years = min(years, len(spot_rates), len(forwards))
    T = np.arange(1, years + 1, dtype=float)      # 1..years
    f = forwards[:years].copy()

    # --- 3. Approximate df/dt and θ(t) -----------------------------
    dfdt = np.zeros_like(f)
    dfdt[1:] = (f[1:] - f[:-1]) / (T[1:] - T[:-1])  # discrete derivative
    dfdt[0] = dfdt[1]  # reuse second slope for the first point

    # θ(t) ≈ df/dt + a f(t) + (σ² / (2a)) (1 - e^{-2 a t})
    theta = dfdt + a * f + (sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * T))

    # Prevent crazy θ values
    theta = np.clip(theta, 0.0, 0.10)  # 0% to 10%

    # --- 4. Simulate HW short rate paths ---------------------------
    dt = 1.0 / 12.0                    # monthly steps
    num_steps = int(years * 12)
    paths = np.zeros((num_paths, num_steps + 1))

    # Start at the 1Y spot as a proxy for r(0)
    paths[:, 0] = spot_rates[0]

    for t_step in range(1, num_steps + 1):
        # Map monthly index to year index for θ(t)
        year_idx = min(t_step // 12, len(theta) - 1)
        theta_t = theta[year_idx]

        dr = (theta_t - a * paths[:, t_step - 1]) * dt \
             + sigma * np.sqrt(dt) * np.random.normal(0.0, 1.0, num_paths)

        # Mild clipping to stop insane outliers
        dr = np.clip(dr, -0.01, 0.01)  # +/- 1% per month
        r_new = paths[:, t_step - 1] + dr
        paths[:, t_step] = np.clip(r_new, 0.0, 0.20)  # keep in [0%, 20%]

    # Annual mean short rates: r(0Y), r(1Y), ..., r(years)
    mean_short_rates = np.mean(paths, axis=0)[::12]  # step 0,12,24,...

    return a, sigma, mean_short_rates


# Process Muni historic curves (add HW)
muni_spots = {}
muni_forwards = {}
muni_discounts = {}
hw_params_muni = {}
hw_short_muni = {}
for date, row in muni_df.iterrows():
    par_yields = row.values / 100  # Divide by 100 to convert percent to decimal
    spots, discounts = bootstrap_spot(par_yields)
    forwards = compute_forwards(spots)
    muni_spots[date] = spots
    muni_forwards[date] = forwards
    muni_discounts[date] = discounts

    # Add HW for muni
    a, sigma, mean_short = calibrate_hw(spots, forwards, date)
    hw_params_muni[date] = [a, sigma]
    hw_short_muni[date] = mean_short

muni_spots_df = pd.DataFrame.from_dict(muni_spots, orient='index')
muni_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]
muni_forwards_df = pd.DataFrame.from_dict(muni_forwards, orient='index')
muni_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]
muni_discounts_df = pd.DataFrame.from_dict(muni_discounts, orient='index')
muni_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]
hw_params_muni_df = pd.DataFrame.from_dict(hw_params_muni, orient='index', columns=['HW_a', 'HW_sigma'])
hw_short_muni_df = pd.DataFrame.from_dict(hw_short_muni, orient='index')
hw_short_muni_df.columns = [f'HW_Short_{y}Y' for y in range(31)]  # 0 to 30Y (31 points)

## Process Treasury (add HW for treasury)
treasury_spots = {}
treasury_forwards = {}
treasury_discounts = {}
hw_params_treas = {}
hw_short_treas = {}

for date in treasury_df.index:
    # SVENYxx and SVENFxx are in PERCENT → convert to decimal
    spots = treasury_df.loc[date, [f'SVENY{y:02d}' for y in range(1, 31)]].values / 100.0
    forwards = treasury_df.loc[date, [f'SVENF{y:02d}' for y in range(1, 31)]].values / 100.0

    discounts = np.exp(-spots * np.arange(1, 31, dtype=float))

    treasury_spots[date] = spots
    treasury_forwards[date] = forwards
    treasury_discounts[date] = discounts

    # Add HW for treasury (note is_treasury=True)
    a, sigma, mean_short = calibrate_hw(spots, forwards, date, is_treasury=True)
    hw_params_treas[date] = [a, sigma]
    hw_short_treas[date] = mean_short

treasury_spots_df = pd.DataFrame.from_dict(treasury_spots, orient='index')
treasury_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]
treasury_forwards_df = pd.DataFrame.from_dict(treasury_forwards, orient='index')
treasury_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]
treasury_discounts_df = pd.DataFrame.from_dict(treasury_discounts, orient='index')
treasury_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]
hw_params_treas_df = pd.DataFrame.from_dict(hw_params_treas, orient='index', columns=['HW_a', 'HW_sigma'])
hw_short_treas_df = pd.DataFrame.from_dict(hw_short_treas, orient='index')
hw_short_treas_df.columns = [f'HW_Short_{y}Y' for y in range(31)]  # 0 to 30Y (31 points)

# Find common dates (2021-2025 overlap)
common_dates = treasury_spots_df.index.intersection(muni_spots_df.index)
treasury_spots_df = treasury_spots_df.loc[common_dates]
treasury_forwards_df = treasury_forwards_df.loc[common_dates]
treasury_discounts_df = treasury_discounts_df.loc[common_dates]
hw_params_treas_df = hw_params_treas_df.loc[common_dates]
hw_short_treas_df = hw_short_treas_df.loc[common_dates]
muni_spots_df = muni_spots_df.loc[common_dates]
muni_forwards_df = muni_forwards_df.loc[common_dates]
muni_discounts_df = muni_discounts_df.loc[common_dates]
hw_params_muni_df = hw_params_muni_df.loc[common_dates]
hw_short_muni_df = hw_short_muni_df.loc[common_dates]

# Compute spreads (Muni - Treasury, in percent)
spot_spreads_df = muni_spots_df - treasury_spots_df
forward_spreads_df = muni_forwards_df - treasury_forwards_df
discount_spreads_df = muni_discounts_df - treasury_discounts_df  # Optional: DF spreads

# Save to Excel
output_file = f'historic_curves_spreads_{datetime.now().strftime("%Y%m%d")}.xlsx'
with pd.ExcelWriter(output_file) as writer:
    muni_spots_df.to_excel(writer, sheet_name='Muni_Spots')
    muni_forwards_df.to_excel(writer, sheet_name='Muni_Forwards')
    muni_discounts_df.to_excel(writer, sheet_name='Muni_Discounts')
    treasury_spots_df.to_excel(writer, sheet_name='Treasury_Spots')
    treasury_forwards_df.to_excel(writer, sheet_name='Treasury_Forwards')
    treasury_discounts_df.to_excel(writer, sheet_name='Treasury_Discounts')
    spot_spreads_df.to_excel(writer, sheet_name='Spot_Spreads')
    forward_spreads_df.to_excel(writer, sheet_name='Forward_Spreads')
    discount_spreads_df.to_excel(writer, sheet_name='Discount_Spreads')
    hw_params_muni_df.to_excel(writer, sheet_name='Muni_HW_Params')
    hw_short_muni_df.to_excel(writer, sheet_name='Muni_HW_Short_Curves')
    hw_params_treas_df.to_excel(writer, sheet_name='Treasury_HW_Params')
    hw_short_treas_df.to_excel(writer, sheet_name='Treasury_HW_Short_Curves')

print(f"Generated {output_file} with data for {len(common_dates)} common dates (2021-2025).")