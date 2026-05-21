"""Yahoo Finance ingestion utilities."""

from __future__ import annotations

from typing import Final

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download and normalize OHLCV data for a ticker from Yahoo Finance.

    Parameters
    ----------
    symbol:
        The ticker symbol to download.
    start_date:
        The inclusive start date in ISO format.
    end_date:
        The inclusive end date in ISO format.

    Returns
    -------
    pd.DataFrame
        A normalized OHLCV DataFrame indexed by pandas.DatetimeIndex.

    Raises
    ------
    ValueError
        If inputs are invalid or data is missing.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    if not start_date or not start_date.strip():
        raise ValueError("start_date must be a non-empty string")

    if not end_date or not end_date.strip():
        raise ValueError("end_date must be a non-empty string")

    try:
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("start_date and end_date must be valid date strings") from exc

    if start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date")

    raw_df = yf.download(symbol, start=start_ts.date(), end=end_ts.date(), progress=False)
    if raw_df is None or raw_df.empty:
        raise ValueError(f"no data returned for symbol: {symbol}")

    df = raw_df.copy()
    # Normalize column names to simple lowercase strings.
    # If a column is a tuple (MultiIndex), use its first element.
    def _col_to_str(col) -> str:
        if isinstance(col, tuple):
            first = col[0]
        else:
            first = col
        return str(first).lower().strip()

    df.columns = [_col_to_str(c) for c in df.columns]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    df = df.loc[:, REQUIRED_COLUMNS]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    if df.empty:
        raise ValueError(f"normalized data is empty for symbol: {symbol}")

    return df
