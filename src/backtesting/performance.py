"""Performance calculation utilities for backtesting.

Provides explicit alignment and deterministic strategy return computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_strategy_returns(
    positions: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Compute strategy returns by multiplying positions and asset returns.

    Parameters
    ----------
    positions:
        Position series indexed by a pandas.DatetimeIndex.
    returns:
        Asset returns series indexed by a pandas.DatetimeIndex.

    Returns
    -------
    pd.Series
        Series of strategy returns named "strategy_return".

    Raises
    ------
    ValueError
        If inputs are invalid, empty, or alignment produces no data.
    """
    if not isinstance(positions, pd.Series) or not isinstance(returns, pd.Series):
        raise ValueError("positions and returns must both be pandas Series")

    if positions.empty or returns.empty:
        raise ValueError("positions and returns series must not be empty")

    if not isinstance(positions.index, pd.DatetimeIndex) or not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("both series must use a pandas.DatetimeIndex")

    aligned_positions, aligned_returns = positions.align(returns, join="inner")

    if aligned_positions.empty or aligned_returns.empty:
        raise ValueError("aligned positions and returns data is empty")

    strategy_return = aligned_positions * aligned_returns
    strategy_return.name = "strategy_return"

    return strategy_return.astype(float)
