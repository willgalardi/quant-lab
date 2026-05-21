"""Core path settings for the quant-lab project."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Root directory of the repository."""

DATA_DIR: Path = PROJECT_ROOT / "data"
"""Top-level data directory."""

RAW_DATA_DIR: Path = DATA_DIR / "raw"
"""Directory for raw input data."""

PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
"""Directory for processed data outputs."""

METADATA_DIR: Path = DATA_DIR / "metadata"
"""Directory for data metadata files."""
