"""Parquet loader for OHLCV datasets.

Provides a narrow, deterministic loader that reads a Parquet file
into a pandas DataFrame, normalizes the index, sorts rows by timestamp,
and validates the result using the project's validation utility.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.data_validation import validate_ohlcv_data


def load_parquet_ohlcv(file_path: Path) -> pd.DataFrame:
    """Load and validate an OHLCV Parquet file.

    Parameters
    ----------
    file_path:
        Path to the parquet file to load.

    Returns
    -------
    pd.DataFrame
        Validated OHLCV DataFrame with a pandas.DatetimeIndex sorted ascending.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ValueError
        If the loaded data is empty or fails schema/validation checks.
    RuntimeError
        If parquet cannot be read.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"parquet file not found: {file_path}")

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        raise RuntimeError(f"failed to read parquet file: {file_path}") from exc

    if df is None or df.empty:
        raise ValueError(f"loaded data is empty: {file_path}")

    # Ensure index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception as exc:
            raise ValueError("failed to convert index to DatetimeIndex") from exc

    # Sort rows by timestamp ascending
    df = df.sort_index()

    # Validate schema and content
    validate_ohlcv_data(df)

    return df
