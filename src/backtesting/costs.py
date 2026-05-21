"""Transaction cost utilities for backtesting.

Applies explicit cost deductions when positions change between bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_transaction_costs(
    strategy_returns: pd.Series,
    positions: pd.Series,
    cost_per_trade: float = 0.001,
) -> pd.Series:
    """Apply transaction costs to strategy returns based on position changes.

    Parameters
    ----------
    strategy_returns:
        Strategy returns indexed by a pandas.DatetimeIndex.
    positions:
        Position series indexed by a pandas.DatetimeIndex.
    cost_per_trade:
        Cost per trade executed when the position changes.

    Returns
    -------
    pd.Series
        Net strategy returns named "net_strategy_return".

    Raises
    ------
    ValueError
        If inputs are invalid, empty, non-datetime indexed, or cost_per_trade is negative.
    """
    if not isinstance(strategy_returns, pd.Series) or not isinstance(positions, pd.Series):
        raise ValueError("strategy_returns and positions must both be pandas Series")

    if strategy_returns.empty or positions.empty:
        raise ValueError("strategy_returns and positions must not be empty")

    if not isinstance(strategy_returns.index, pd.DatetimeIndex) or not isinstance(positions.index, pd.DatetimeIndex):
        raise ValueError("both series must use a pandas.DatetimeIndex")

    if not isinstance(cost_per_trade, (float, int)) or cost_per_trade < 0:
        raise ValueError("cost_per_trade must be a non-negative number")

    aligned_returns, aligned_positions = strategy_returns.align(positions, join="inner")

    if aligned_returns.empty or aligned_positions.empty:
        raise ValueError("aligned strategy_returns and positions data is empty")

    trade = aligned_positions.diff().abs().fillna(0)
    cost = trade * float(cost_per_trade)

    net_strategy_return = aligned_returns - cost
    net_strategy_return.name = "net_strategy_return"

    return net_strategy_return.astype(float)
