"""Run a deterministic SMA crossover backtest for AAPL.

This script loads historical AAPL OHLCV data, computes SMA crossover signals,
applies transaction costs, and reports performance metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analytics.drawdown import compute_max_drawdown
from src.analytics.equity_curve import compute_equity_curve
from src.backtesting.costs import apply_transaction_costs
from src.backtesting.performance import compute_strategy_returns
from src.backtesting.positions import generate_positions
from src.ingestion.data_loader import load_parquet_ohlcv
from src.signals.crossover import generate_sma_crossover_signal
from src.signals.returns import compute_daily_returns
from src.signals.moving_averages import compute_simple_moving_average
from src.config.settings import RAW_DATA_DIR

SHORT_WINDOW: int = 5
LONG_WINDOW: int = 20
TRANSACTION_COST: float = 0.001
INITIAL_CAPITAL: float = 1.0


def main() -> None:
    """Execute the SMA crossover backtest and print summary metrics."""
    file_path = RAW_DATA_DIR / "aapl_daily.parquet"
    df = load_parquet_ohlcv(file_path)

    returns = compute_daily_returns(df)

    short_sma = compute_simple_moving_average(df["close"], SHORT_WINDOW)
    long_sma = compute_simple_moving_average(df["close"], LONG_WINDOW)

    # Align SMAs to the same timestamps before generating crossover signals
    short_sma, long_sma = short_sma.align(long_sma, join="inner")
    crossover_signal = generate_sma_crossover_signal(short_sma, long_sma)

    positions = generate_positions(crossover_signal)
    gross_strategy_returns = compute_strategy_returns(positions, returns)
    net_strategy_returns = apply_transaction_costs(
        gross_strategy_returns,
        positions,
        cost_per_trade=TRANSACTION_COST,
    )

    equity_curve = compute_equity_curve(net_strategy_returns, initial_capital=INITIAL_CAPITAL)
    max_drawdown = compute_max_drawdown(equity_curve)

    total_return = float(equity_curve.iloc[-1] / INITIAL_CAPITAL - 1)
    number_of_trades = int(positions.diff().abs().sum())

    print(f"Total return: {total_return:.6f}")
    print(f"Max drawdown: {max_drawdown:.6f}")
    print(f"Number of trades: {number_of_trades}")


if __name__ == "__main__":
    main()
