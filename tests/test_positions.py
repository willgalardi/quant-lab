"""Tests for backtesting position generation."""

import numpy as np
import pandas as pd

from src.backtesting.positions import generate_positions


def test_generate_positions_shifts_signal_forward() -> None:
    """The position series should shift the signal forward by one bar."""
    index = pd.date_range(start="2024-01-01", periods=4, freq="D")
    signal = pd.Series([1, 1, 0, 1], index=index)

    positions = generate_positions(signal)

    expected = pd.Series([0, 1, 1, 0], index=index, name="position")

    pd.testing.assert_series_equal(positions, expected)


def test_generate_positions_empty_signal_raises() -> None:
    """Empty signal input should raise a ValueError."""
    empty_signal = pd.Series([], dtype=float)

    try:
        generate_positions(empty_signal)
    except ValueError as exc:
        assert str(exc) == "input signal series is empty"
    else:
        raise AssertionError("ValueError was not raised for empty signal")
