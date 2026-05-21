"""Sharpe ratio calculation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute an annualized Sharpe ratio assuming zero risk-free rate.

    Parameters
    ----------
    returns:
        Return series indexed by a pandas.DatetimeIndex.
    periods_per_year:
        Number of return periods per year for annualization.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Raises
    ------
    ValueError
        If inputs are invalid, empty, or standard deviation is zero.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")

    if returns.empty:
        raise ValueError("returns series is empty")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns index must be a pandas.DatetimeIndex")

    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be an integer greater than 0")

    mean_return = returns.mean()
    std_return = returns.std(ddof=1)

    if std_return == 0 or np.isclose(std_return, 0.0):
        raise ValueError("standard deviation of returns is zero")

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return float(sharpe)
