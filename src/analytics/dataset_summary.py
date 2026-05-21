"""Dataset summary utilities for OHLCV DataFrames."""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd


def summarize_ohlcv_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a concise summary of an OHLCV DataFrame.

    Parameters
    ----------
    df:
        A pandas DataFrame indexed by timestamps containing OHLCV columns.

    Returns
    -------
    Dict[str, Any]
        Summary with keys: `row_count`, `column_count`, `start_date`,
        `end_date`, `missing_values`, and `duplicate_timestamps`.

    Raises
    ------
    ValueError
        If `df` is not a DataFrame, has a non-datetime index, or is empty.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex")

    if df.empty:
        raise ValueError("DataFrame is empty")

    # Basic counts
    row_count: int = int(len(df))
    column_count: int = int(df.shape[1])

    # ISO date strings for start/end
    start_ts = df.index.min()
    end_ts = df.index.max()
    start_date: str = start_ts.isoformat()
    end_date: str = end_ts.isoformat()

    # Missing values across the DataFrame
    missing_values: int = int(df.isnull().sum().sum())

    # Count of duplicate timestamps (excluding first occurrences)
    duplicate_timestamps: int = int(df.index.duplicated().sum())

    return {
        "row_count": row_count,
        "column_count": column_count,
        "start_date": start_date,
        "end_date": end_date,
        "missing_values": missing_values,
        "duplicate_timestamps": duplicate_timestamps,
    }
