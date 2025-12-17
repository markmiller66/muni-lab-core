# src/muni_core/calls/call_schedule.py

from __future__ import annotations

from datetime import date
from typing import List, Optional

from muni_core.curves import year_fraction


def _add_months(d: date, months: int) -> date:
    """
    Add a number of months to a date, clamping to last valid day of month.

    This avoids pulling in dateutil just for relativedelta and is
    more than sufficient for standard muni call dates (1st / 15th / EOM).
    """
    # Compute target year and month
    new_month = d.month - 1 + months
    new_year = d.year + new_month // 12
    new_month = new_month % 12 + 1

    # Clamp day to last day in target month
    # (handles 28/29/30/31 and leap years).
    # We use a simple loop instead of calendar.monthrange to avoid an extra import.
    day = d.day
    while True:
        try:
            return date(new_year, new_month, day)
        except ValueError:
            day -= 1
            if day <= 0:
                # Should never happen in practice; defensive fallback.
                return date(new_year, new_month, 1)


def build_bermudan_call_times_3yr_window(
    asof_date: date,
    first_call_date: Optional[date],
    maturity_date: date,
) -> List[float]:
    """
    Build Bermudan call exercise times over a 3-year window after first call.

    Rules:
    - Call 0 = first_call_date
    - Then every 6 months for 3 years (max 7 dates total: 0..6)
    - Stop early if we pass maturity_date
    - Return ONLY future exercise times (t > 0) as ACT/365.25
      year fractions from asof_date (issuer's decision times).

    Parameters
    ----------
    asof_date : date
        Valuation / pricing date (today).
    first_call_date : date or None
        First optional call date; if None or invalid, returns [].
    maturity_date : date
        Final maturity date of the bond.

    Returns
    -------
    List[float]
        List of exercise times (in years) relative to asof_date.
        These are the candidate Bermudan exercise times for the HW lattice.
    """
    if first_call_date is None:
        return []

    # If first call is on/after maturity, there is effectively no call option.
    if first_call_date >= maturity_date:
        return []

    times: List[float] = []
    current = first_call_date

    # Maximum 7 call dates: first_call_date + 6 monthly steps
    max_steps = 7
    step_months = 6

    for _ in range(max_steps):
        if current > maturity_date:
            break

        t = year_fraction(asof_date, current)

        # Only keep *future* decision times.
        if t > 0.0:
            times.append(t)

        # Move to the next potential call date
        current = _add_months(current, step_months)

    return times
