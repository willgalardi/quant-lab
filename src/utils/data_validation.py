"""Data validation utilities for OHLCV DataFrames."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def validate_ohlcv_data(df: pd.DataFrame) -> None:
    """Validate that a DataFrame is a well-formed OHLCV timeseries.

    Checks performed:
    - `df` is a pandas DataFrame
    - index is a pandas.DatetimeIndex
    - index is monotonic increasing and contains no duplicates
    - required columns are present
    - required columns contain no missing values

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")

    # Index validations
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonic increasing")

    if df.index.duplicated().any():
        raise ValueError("DataFrame index contains duplicate timestamps")

    # Column presence
    missing: Iterable[str] = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    # Missing values in required columns
    if df.loc[:, REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("required columns contain missing values")

    # All checks passed; return None
    return None
