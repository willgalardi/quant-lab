"""Tests for transaction cost application in backtesting."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.costs import apply_transaction_costs


def test_apply_transaction_costs_deducts_costs_on_trades() -> None:
    """Transaction costs should only apply when positions change."""
    index = pd.date_range(start="2024-01-01", periods=4, freq="D")
    positions = pd.Series([0, 1, 1, 0], index=index)
    strategy_returns = pd.Series([0.0, 0.01, -0.02, 0.0], index=index)
    cost_per_trade = 0.001

    net_returns = apply_transaction_costs(strategy_returns, positions, cost_per_trade=cost_per_trade)

    expected = pd.Series([0.0, 0.009, -0.02, -0.001], index=index, name="net_strategy_return")

    pd.testing.assert_series_equal(net_returns, expected)


def test_apply_transaction_costs_negative_cost_raises() -> None:
    """Negative transaction cost should raise a ValueError."""
    index = pd.date_range(start="2024-01-01", periods=4, freq="D")
    positions = pd.Series([0, 1, 1, 0], index=index)
    strategy_returns = pd.Series([0.0, 0.01, -0.02, 0.0], index=index)

    with pytest.raises(ValueError, match="cost_per_trade must be a non-negative number"):
        apply_transaction_costs(strategy_returns, positions, cost_per_trade=-0.001)
