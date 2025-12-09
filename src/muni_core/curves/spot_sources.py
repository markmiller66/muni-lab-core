# src/muni_core/curves/spot_sources.py

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SpotSourceConfig:
    """
    Lightweight config for raw curve sources.

    In practice you'll probably populate this from your AppConfig /
    MUNI_MASTER_BUCKET rather than hard-coding paths.
    """
    data_root: Path
    treasury_file: str              # e.g. "feds200628.csv"
    muni_file: str                  # e.g. "Tradeweb_MUNI_data (4).csv"
    vix_file: Optional[str] = None  # VIX is optional
    treasury_skiprows: int = 9


def load_treasury_raw(cfg: SpotSourceConfig) -> pd.DataFrame:
    """
    Load raw Fed Treasury CSV and return a Date-indexed DataFrame.
    """
    path = cfg.data_root / cfg.treasury_file

    df = pd.read_csv(path, skiprows=cfg.treasury_skiprows, low_memory=False)
    # You used 'Date' as the column name in your script
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    return df


def load_muni_raw(cfg: SpotSourceConfig) -> pd.DataFrame:
    """
    Load raw Tradeweb AAA muni CSV and return a Date-indexed DataFrame.
    """
    path = cfg.data_root / cfg.muni_file

    df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
    # You used 'Date Of' as the date column
    df['Date Of'] = pd.to_datetime(df['Date Of'])
    df = df.sort_values('Date Of').set_index('Date Of')

    return df


def load_vix_raw(cfg: SpotSourceConfig) -> pd.DataFrame:
    """
    Load U Chicago muni VIX csv, if provided. Returns Date-indexed DataFrame.
    """
    if cfg.vix_file is None:
        raise ValueError("vix_file is not configured in SpotSourceConfig")

    path = cfg.data_root / cfg.vix_file

    df = pd.read_csv(path, low_memory=False)
    # You used 'date' as the column name
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    return df
