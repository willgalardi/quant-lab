"""Signal computations for returns series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Compute simple daily close-to-close returns.

    Parameters
    ----------
    df:
        Input OHLCV DataFrame containing a `close` column and a DatetimeIndex.

    Returns
    -------
    pd.Series
        Series of simple daily returns named "daily_return" with the initial
        NaN removed. The series is indexed by the timestamps corresponding to
        each return (one fewer row than the input).

    Raises
    ------
    ValueError
        If the input is invalid (missing `close`, empty DataFrame, or non-datetime index).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("input DataFrame is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex")

    if "close" not in df.columns:
        raise ValueError("required column 'close' is missing")

    close = df["close"]

    # Ensure numeric dtype for arithmetic
    if not pd.api.types.is_numeric_dtype(close):
        try:
            close = pd.to_numeric(close, errors="raise")
        except Exception as exc:
            raise ValueError("'close' column must be numeric") from exc

    # Simple close-to-close returns: P_t / P_{t-1} - 1
    returns = close.pct_change()

    # Name the series and drop the initial NaN
    returns.name = "daily_return"
    returns = returns.dropna()

    # Ensure numpy float dtype
    returns = returns.astype(float)

    return returns
