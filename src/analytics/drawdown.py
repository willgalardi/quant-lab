"""Drawdown analytics utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the maximum drawdown from an equity curve.

    Parameters
    ----------
    equity_curve:
        A pandas Series indexed by a pandas.DatetimeIndex representing equity values.

    Returns
    -------
    float
        The minimum drawdown value (negative for losses).

    Raises
    ------
    ValueError
        If the input series is invalid, empty, not datetime indexed, or contains non-positive values.
    """
    if not isinstance(equity_curve, pd.Series):
        raise ValueError("equity_curve must be a pandas Series")

    if equity_curve.empty:
        raise ValueError("equity_curve series is empty")

    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise ValueError("equity_curve index must be a pandas.DatetimeIndex")

    if (equity_curve <= 0).any():
        raise ValueError("equity_curve values must be strictly positive")

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak

    return float(drawdown.min())
