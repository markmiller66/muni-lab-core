from __future__ import annotations

from pathlib import Path
from datetime import date

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def history_parquet_path(repo_root: Path) -> Path:
    p = repo_root / "data" / "AAA_MUNI_CURVE" / "aaa_muni_treas_history.parquet"
    assert p.exists(), f"Missing history file: {p}"
    return p


@pytest.fixture(scope="session")
def history_df(history_parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(history_parquet_path)
    assert len(df) > 0
    return df


@pytest.fixture(scope="session")
def asof_date() -> date:
    # Keep deterministic for tests (matches your HW stack output)
    return date(2025, 11, 26)
