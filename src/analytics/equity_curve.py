"""Equity curve calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_equity_curve(
    strategy_returns: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """Compute the compounded equity curve from strategy returns.

    Parameters
    ----------
    strategy_returns:
        Strategy returns indexed by a pandas.DatetimeIndex.
    initial_capital:
        Starting capital for the equity curve. Must be greater than zero.

    Returns
    -------
    pd.Series
        The equity curve series named "equity_curve" with the same index as the input.

    Raises
    ------
    ValueError
        If inputs are invalid, empty, or non-positive.
    """
    if not isinstance(strategy_returns, pd.Series):
        raise ValueError("strategy_returns must be a pandas Series")

    if strategy_returns.empty:
        raise ValueError("strategy_returns series is empty")

    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise ValueError("strategy_returns index must be a pandas.DatetimeIndex")

    if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
        raise ValueError("initial_capital must be a positive number")

    # Compute compounded growth: Equity_t = Equity_{t-1} * (1 + r_t)
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    equity_curve.name = "equity_curve"

    return equity_curve.astype(float)
