# quant-lab
quantitative finance research platform - signal research, backtesting, risk governance, and paper trading

## Current Architecture

### Data Layer

- Yahoo Finance ingestion for historical OHLCV data.
- Parquet persistence for local raw data storage.
- Validation layer for path, schema, and OHLCV integrity.

### Signal Layer

- Daily close-to-close return computation.
- Rolling volatility calculation with sample standard deviation.
- Simple moving averages for short and long windows.
- SMA crossover signal generation.

### Backtesting Layer

- Look-ahead-safe position shifting by one bar.
- Gross strategy return calculation from positions and returns.
- Transaction cost modeling on position changes.
- Net strategy return computation after costs.

### Analytics Layer

- Equity curve generation via compounded returns.
- Dataset summaries for row/column counts and data quality.
- Maximum drawdown calculation for downside risk analysis.

### Engineering Principles

- Deterministic pipelines with explicit execution order.
- Modular components organized by data, signals, backtesting, and analytics.
- Explicit validation at each transformation boundary.
- Causal execution handling to avoid look-ahead bias.
- Transaction-cost-aware research for realistic performance estimates.
