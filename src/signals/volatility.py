"""Volatility signal computations.

Provides a deterministic rolling volatility computation for return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling sample standard deviation of returns.

    Parameters
    ----------
    returns:
        A pandas Series of returns indexed by a pandas.DatetimeIndex.
    window:
        Rolling window length (number of periods). Must be > 1.

    Returns
    -------
    pd.Series
        Rolling volatility series named "rolling_volatility" with initial
        NaN values (from the window) dropped.

    Raises
    ------
    ValueError
        If the input is invalid: empty series, non-datetime index, or
        invalid `window` value.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("input must be a pandas Series of returns")

    if returns.empty:
        raise ValueError("input returns series is empty")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a pandas.DatetimeIndex")

    if not isinstance(window, int) or window <= 1:
        raise ValueError("window must be an integer greater than 1")

    # Compute rolling sample standard deviation with ddof=1
    rolling_std = returns.rolling(window=window).std(ddof=1)

    # Name the output and drop initial NaNs caused by the rolling window
    rolling_std.name = "rolling_volatility"
    rolling_std = rolling_std.dropna()

    # Ensure float dtype
    rolling_std = rolling_std.astype(float)

    return rolling_std
