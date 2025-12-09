import pandas as pd
import numpy as np
from datetime import datetime

# Load Treasury data (adjust path if needed)
treasury_df = pd.read_csv(r'data/feds200628.csv', skiprows=9, low_memory=False)
treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
treasury_df.set_index('Date', inplace=True)

# Load Muni data (adapt path if needed)
muni_df = pd.read_csv(r'data/Tradeweb_MUNI_data (4).csv', low_memory=False, on_bad_lines='skip')
muni_df['Date Of'] = pd.to_datetime(muni_df['Date Of'])
muni_df = muni_df.sort_values('Date Of').set_index('Date Of')


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


# Function to compute instantaneous forward rates from spots
def compute_forwards(spot_rates):
    annual_mats = np.arange(1, 31)
    forwards = np.zeros(30)
    forwards[0] = spot_rates[0]
    for i in range(1, 30):
        if spot_rates[i] == 0 and i > 0:
            forwards[i] = forwards[i - 1]  # Extrapolate if zero
        else:
            forwards[i] = (annual_mats[i] * spot_rates[i] - annual_mats[i - 1] * spot_rates[i - 1])
    return forwards


# Process Muni historic curves
muni_spots = {}
muni_forwards = {}
muni_discounts = {}
for date, row in muni_df.iterrows():
    par_yields = row.values / 100  # Divide by 100 to convert percent to decimal
    spots, discounts = bootstrap_spot(par_yields)
    forwards = compute_forwards(spots)
    muni_spots[date] = spots
    muni_forwards[date] = forwards
    muni_discounts[date] = discounts

muni_spots_df = pd.DataFrame.from_dict(muni_spots, orient='index')
muni_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]
muni_forwards_df = pd.DataFrame.from_dict(muni_forwards, orient='index')
muni_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]
muni_discounts_df = pd.DataFrame.from_dict(muni_discounts, orient='index')
muni_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]

# Treasury spots, forwards, and discounts (pre-computed spots/forwards; compute DFs)
treasury_spots_df = treasury_df[[f'SVENY{y:02d}' for y in range(1, 31)]]
treasury_forwards_df = treasury_df[[f'SVENF{y:02d}' for y in range(1, 31)]]
treasury_discounts_df = np.exp(-treasury_spots_df * np.arange(1, 31))  # Annual DFs from spots

# Find common dates (2021-2025 overlap)
common_dates = treasury_spots_df.index.intersection(muni_spots_df.index)
treasury_spots_df = treasury_spots_df.loc[common_dates]
treasury_spots_df.columns = [f'Spot_{y}Y' for y in range(1, 31)]
treasury_forwards_df = treasury_forwards_df.loc[common_dates]
treasury_forwards_df.columns = [f'Forward_{y}Y' for y in range(1, 31)]
treasury_discounts_df = treasury_discounts_df.loc[common_dates]
treasury_discounts_df.columns = [f'Discount_{y}Y' for y in range(1, 31)]
muni_spots_df = muni_spots_df.loc[common_dates]
muni_forwards_df = muni_forwards_df.loc[common_dates]
muni_discounts_df = muni_discounts_df.loc[common_dates]

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

print(f"Generated {output_file} with data for {len(common_dates)} common dates (2021-2025).")