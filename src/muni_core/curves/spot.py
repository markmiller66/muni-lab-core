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
                for T = 1,2,...,N years.
    Returns forward rates f_{i-1,i} (0–1Y, 1–2Y, ..., (N-1)–N) in decimal.
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    T = np.arange(1, len(spot_rates) + 1, dtype=float)  # 1..N in years

    forwards = np.zeros_like(spot_rates)
    forwards[0] = spot_rates[0]  # use 1Y spot as 0–1Y forward proxy

    # f_{i-1,i} = (T_i r_i - T_{i-1} r_{i-1}) / (T_i - T_{i-1})
    forwards[1:] = (T[1:] * spot_rates[1:] - T[:-1] * spot_rates[:-1]) / (T[1:] - T[:-1])

    return forwards


# Deterministic Hull–White calibration and mean short-rate path
def calibrate_hw(spot_rates, forwards, date,
                 years=30, is_treasury=False,
                 a=0.05):
    """
    Deterministic Hull–White short-rate term structure.

    Returns
    -------
    a         : float
    sigma     : float
    theta     : np.ndarray, shape (years,)
        HW theta(t) at t = 1..years (annual grid).
    r_mean    : np.ndarray, shape (years+1,)
        Deterministic expected short rate E[r(t)] at t = 0..years.
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    forwards = np.asarray(forwards, dtype=float)

    # Limit horizon to available curve length
    years = int(min(years, len(spot_rates), len(forwards)))
    T = np.arange(1, years + 1, dtype=float)   # 1..years
    f = forwards[:years].copy()

    # ---------- 1. Choose sigma ----------
    if is_treasury:
        # Simple fixed vol for now (can be calibrated later)
        sigma = 0.01      # 1% short-rate vol
    else:
        # Pull muni "VIX" and scale; assume VIX column is % annual vol
        if date in vix_df.index:
            vix = float(vix_df.loc[date, 'Weighted Average'])
        else:
            prev_dates = vix_df.index[vix_df.index < date]
            if len(prev_dates) > 0:
                vix = float(vix_df.loc[prev_dates[-1], 'Weighted Average'])
            else:
                vix = float(vix_df['Weighted Average'].iloc[-1])

        # Turn e.g. 40–60% "VIX" into a more modest HW σ (tunable knob)
        sigma = (vix / 100.0) * 0.0002  # 0.5% when vix = 50
    # e.g. 50% index → σ = 0.25

    # ---------- 2. df/dt (finite difference) ----------
    dfdt = np.zeros_like(f)
    if years > 1:
        dfdt[1:] = (f[1:] - f[:-1]) / (T[1:] - T[:-1])
        dfdt[0] = dfdt[1]      # reuse 2nd slope for t=1Y
    else:
        dfdt[0] = 0.0

    # ---------- 3. Theta(t) using HW no-arbitrage condition ----------
    # θ(t) = f(0,t) + (1/a) f'(0,t) + (σ² / (2 a²)) (1 - e^{-2 a t})
    theta = f + (1.0 / a) * dfdt + (sigma ** 2 / (2.0 * a ** 2)) * (1.0 - np.exp(-2.0 * a * T))

    # Clip to keep in a realistic range (0–10%) for our muni use-case
    theta = np.clip(theta, 0.0, 0.10)

    # ---------- 4. Deterministic mean short rate path E[r(t)] ----------
    # ---------- 4. Deterministic mean short rate path E[r(t)] ----------
    dt = 1.0  # 1-year step
    r_mean = np.zeros(years + 1)

    # Start r(0) as 1Y forward or 1Y spot (both are fine as a proxy)
    r_mean[0] = f[0]

    for i in range(1, years + 1):
        theta_i = theta[i - 1]       # θ at year i
        r_prev = r_mean[i - 1]
        # HW canonical form: dr = a(θ - r) dt
        r_mean[i] = r_prev + a * (theta_i - r_prev) * dt



    return a, sigma, theta, r_mean


# -------------------------
# Process Muni historic curves (add HW)
# -------------------------
muni_spots = {}
muni_forwards = {}
muni_discounts = {}
hw_params_muni = {}
hw_theta_muni = {}      # NEW
hw_short_muni = {}

for date, row in muni_df.iterrows():
    par_yields = row.values / 100.0  # percent → decimal
    spots, discounts = bootstrap_spot(par_yields)
    forwards = compute_forwards(spots)

    muni_spots[date] = spots
    muni_forwards[date] = forwards
    muni_discounts[date] = discounts

    # Deterministic HW for muni
    a, sigma, theta, mean_short = calibrate_hw(spots, forwards, date, is_treasury=False)
    hw_params_muni[date] = [a, sigma]
    hw_theta_muni[date] = theta           # θ(t) at 1..years
    hw_short_muni[date] = mean_short      # E[r(t)] at 0..years

muni_spots_df = pd.DataFrame.from_dict(muni_spots, orient='index')
muni_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]

muni_forwards_df = pd.DataFrame.from_dict(muni_forwards, orient='index')
muni_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]

muni_discounts_df = pd.DataFrame.from_dict(muni_discounts, orient='index')
muni_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]

hw_params_muni_df = pd.DataFrame.from_dict(hw_params_muni, orient='index',
                                           columns=['HW_a', 'HW_sigma'])

hw_theta_muni_df = pd.DataFrame.from_dict(hw_theta_muni, orient='index')
hw_theta_muni_df.columns = [f'HW_Theta_{y}Y' for y in range(1, hw_theta_muni_df.shape[1] + 1)]

hw_short_muni_df = pd.DataFrame.from_dict(hw_short_muni, orient='index')
hw_short_muni_df.columns = [f'HW_Short_{y}Y' for y in range(hw_short_muni_df.shape[1])]  # 0..years


# -------------------------
# Process Treasury (add HW)
# -------------------------
treasury_spots = {}
treasury_forwards = {}
treasury_discounts = {}
hw_params_treas = {}
hw_theta_treas = {}     # NEW
hw_short_treas = {}

for date in treasury_df.index:
    # SVENYxx and SVENFxx are in PERCENT → convert to decimal
    spots = treasury_df.loc[date, [f'SVENY{y:02d}' for y in range(1, 31)]].values / 100.0
    forwards = treasury_df.loc[date, [f'SVENF{y:02d}' for y in range(1, 31)]].values / 100.0

    discounts = np.exp(-spots * np.arange(1, 31, dtype=float))

    treasury_spots[date] = spots
    treasury_forwards[date] = forwards
    treasury_discounts[date] = discounts

    # Deterministic HW for Treasury
    a, sigma, theta, mean_short = calibrate_hw(spots, forwards, date, is_treasury=True)
    hw_params_treas[date] = [a, sigma]
    hw_theta_treas[date] = theta
    hw_short_treas[date] = mean_short

treasury_spots_df = pd.DataFrame.from_dict(treasury_spots, orient='index')
treasury_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]

treasury_forwards_df = pd.DataFrame.from_dict(treasury_forwards, orient='index')
treasury_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]

treasury_discounts_df = pd.DataFrame.from_dict(treasury_discounts, orient='index')
treasury_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]

hw_params_treas_df = pd.DataFrame.from_dict(hw_params_treas, orient='index',
                                            columns=['HW_a', 'HW_sigma'])

hw_theta_treas_df = pd.DataFrame.from_dict(hw_theta_treas, orient='index')
hw_theta_treas_df.columns = [f'HW_Theta_{y}Y' for y in range(1, hw_theta_treas_df.shape[1] + 1)]

hw_short_treas_df = pd.DataFrame.from_dict(hw_short_treas, orient='index')
hw_short_treas_df.columns = [f'HW_Short_{y}Y' for y in range(hw_short_treas_df.shape[1])]  # 0..years


# -------------------------
# Align on common dates
# -------------------------
common_dates = treasury_spots_df.index.intersection(muni_spots_df.index)

treasury_spots_df = treasury_spots_df.loc[common_dates]
treasury_forwards_df = treasury_forwards_df.loc[common_dates]
treasury_discounts_df = treasury_discounts_df.loc[common_dates]
hw_params_treas_df = hw_params_treas_df.loc[common_dates]
hw_theta_treas_df = hw_theta_treas_df.loc[common_dates]
hw_short_treas_df = hw_short_treas_df.loc[common_dates]

muni_spots_df = muni_spots_df.loc[common_dates]
muni_forwards_df = muni_forwards_df.loc[common_dates]
muni_discounts_df = muni_discounts_df.loc[common_dates]
hw_params_muni_df = hw_params_muni_df.loc[common_dates]
hw_theta_muni_df = hw_theta_muni_df.loc[common_dates]
hw_short_muni_df = hw_short_muni_df.loc[common_dates]

# -------------------------
# Compute spreads (Muni - Treasury, in decimal)
# -------------------------
spot_spreads_df = muni_spots_df - treasury_spots_df
forward_spreads_df = muni_forwards_df - treasury_forwards_df
discount_spreads_df = muni_discounts_df - treasury_discounts_df  # Optional: DF spreads


# -------------------------
# Save to Excel
# -------------------------
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
    hw_theta_muni_df.to_excel(writer, sheet_name='Muni_HW_Theta')
    hw_short_muni_df.to_excel(writer, sheet_name='Muni_HW_Short')

    hw_params_treas_df.to_excel(writer, sheet_name='Treasury_HW_Params')
    hw_theta_treas_df.to_excel(writer, sheet_name='Treasury_HW_Theta')
    hw_short_treas_df.to_excel(writer, sheet_name='Treasury_HW_Short')

print(f"Generated {output_file} with data for {len(common_dates)} common dates (2021-2025).")
