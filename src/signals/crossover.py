"""SMA crossover signal generator.

Provides a deterministic, long-only binary crossover signal based on two
simple moving average series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sma_crossover_signal(short_sma: pd.Series, long_sma: pd.Series) -> pd.Series:
    """Generate a binary long-only SMA crossover signal.

    signal_t = 1 if short_sma_t > long_sma_t else 0

    Parameters
    ----------
    short_sma:
        Short-window SMA as a pandas Series indexed by a pandas.DatetimeIndex.
    long_sma:
        Long-window SMA as a pandas Series indexed by a pandas.DatetimeIndex.

    Returns
    -------
    pd.Series
        Binary signal named "crossover_signal" indexed identically to the inputs.

    Raises
    ------
    ValueError
        If inputs are invalid (empty, mismatched indices, or non-datetime indices).
    """
    if not isinstance(short_sma, pd.Series) or not isinstance(long_sma, pd.Series):
        raise ValueError("both inputs must be pandas Series")

    if short_sma.empty or long_sma.empty:
        raise ValueError("input series must not be empty")

    if not isinstance(short_sma.index, pd.DatetimeIndex) or not isinstance(long_sma.index, pd.DatetimeIndex):
        raise ValueError("both series must have a pandas.DatetimeIndex")

    # Ensure indices match exactly
    if not short_sma.index.equals(long_sma.index):
        raise ValueError("indices of short_sma and long_sma must match exactly")

    # Compute binary signal where short > long
    try:
        comp = short_sma > long_sma
    except Exception as exc:
        raise ValueError("failed to compare SMA series") from exc

    signal = comp.astype(int)
    signal.name = "crossover_signal"

    # Ensure integer dtype (numpy int64)
    signal = signal.astype(int)

    return signal
