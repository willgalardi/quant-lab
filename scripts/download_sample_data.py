"""Download a sample AAPL OHLCV dataset and save as Parquet.

This script fetches AAPL daily OHLCV data for a fixed date range
and writes it to `data/raw/aapl_daily.parquet` using the pyarrow engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pyarrow  # ensure pyarrow engine is available
import pandas as pd

from src.ingestion.yahoo_client import fetch_ohlcv
from src.config.settings import RAW_DATA_DIR

OUTPUT_FILE: Final[Path] = RAW_DATA_DIR / "aapl_daily.parquet"


def save_df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a non-empty DataFrame to Parquet using pyarrow.

    Raises a RuntimeError on failure.
    """
    if df is None or df.empty:
        raise ValueError("cannot save empty or None DataFrame")

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, engine="pyarrow", index=True)
    except Exception as exc:
        raise RuntimeError(f"failed to write parquet file to {path}") from exc


def main() -> None:
    """Fetch AAPL OHLCV and save to a parquet file, printing results."""
    start_date = "2024-01-01"
    end_date = "2024-03-01"

    df = fetch_ohlcv("AAPL", start_date, end_date)

    # Basic validation
    if df is None or df.empty:
        raise RuntimeError("no data fetched for AAPL in the requested range")

    row_count = len(df)
    save_df_to_parquet(df, OUTPUT_FILE)

    print(f"Row count: {row_count}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
