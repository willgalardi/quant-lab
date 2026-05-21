"""Moving average signal utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Compute the simple moving average (SMA) for a series.

    Parameters
    ----------
    series:
        Input pandas Series indexed by a pandas.DatetimeIndex.
    window:
        Window length (n) for the SMA. Must be an integer > 0.

    Returns
    -------
    pd.Series
        The SMA series named "sma_{window}" with initial NaN values
        (from the window) dropped.

    Raises
    ------
    ValueError
        If input is invalid (empty series, non-datetime index, missing/invalid window).
    """
    if not isinstance(series, pd.Series):
        raise ValueError("input must be a pandas Series")

    if series.empty:
        raise ValueError("input series is empty")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series index must be a pandas.DatetimeIndex")

    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be an integer greater than 0")

    # Ensure numeric dtype for arithmetic
    if not pd.api.types.is_numeric_dtype(series):
        try:
            series = pd.to_numeric(series, errors="raise")
        except Exception as exc:
            raise ValueError("series must contain numeric values") from exc

    sma = series.rolling(window=window, min_periods=window).mean()
    sma.name = f"sma_{window}"

    # Drop initial NaN values introduced by the rolling window
    sma = sma.dropna()

    # Ensure float dtype
    sma = sma.astype(float)

    return sma
