"""Position generation utilities for backtesting.

Provides deterministic position creation by shifting signals forward one bar to
prevent look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_positions(signal: pd.Series) -> pd.Series:
    """Generate positions by shifting a signal forward one bar.

    Parameters
    ----------
    signal:
        A pandas Series indexed by a pandas.DatetimeIndex representing a raw trading signal.

    Returns
    -------
    pd.Series
        The position series named "position" with the same index as the input.

    Raises
    ------
    ValueError
        If the input signal is empty or does not use a DatetimeIndex.
    """
    if not isinstance(signal, pd.Series):
        raise ValueError("input must be a pandas Series")

    if signal.empty:
        raise ValueError("input signal series is empty")

    if not isinstance(signal.index, pd.DatetimeIndex):
        raise ValueError("signal index must be a pandas.DatetimeIndex")

    positions = signal.shift(1)
    positions.name = "position"

    # Use 0 as the default prior position on the first bar
    positions = positions.fillna(0)

    return positions.astype(int)
