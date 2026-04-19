"""Price data loading and synthetic generation."""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")

# HistData Generic ASCII M1: timestamps are EST with no DST adjustment (UTC-5).
HISTDATA_TZ = "Etc/GMT+5"


def _expand_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Accept a single path, a glob pattern, a directory, or an iterable."""
    if isinstance(paths, (str, Path)):
        items: list[str | Path] = [paths]
    else:
        items = list(paths)

    resolved: list[Path] = []
    for item in items:
        s = str(item)
        # Directory: take all CSVs inside
        p = Path(s)
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.csv")))
            continue
        # Glob pattern (contains wildcards)
        if any(ch in s for ch in "*?[]"):
            matched = [Path(m) for m in _glob.glob(s)]
            resolved.extend(sorted(matched))
            continue
        resolved.append(p)

    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise FileNotFoundError(f"No files matched: {paths!r}")
    return unique


def load_histdata(
    paths: str | Path | Iterable[str | Path],
    tz: str = HISTDATA_TZ,
) -> pd.DataFrame:
    """Load one or more HistData Generic ASCII M1 CSVs.

    Accepts a single file, glob (e.g. 'C:/Users/.../DAT_ASCII_USDJPY_M1_*.csv'),
    directory, or list of paths. Files are concatenated, deduplicated and sorted
    by timestamp. The raw timestamps are EST-without-DST (UTC-5) and are
    converted to UTC so the rest of the pipeline stays consistent.

    HistData format (no header, semicolon-separated):
        YYYYMMDD HHMMSS;open;high;low;close;volume
    """
    files = _expand_paths(paths)
    frames: list[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(
            fp,
            sep=";",
            header=None,
            names=["timestamp", "open", "high", "low", "close", "volume"],
            dtype={"timestamp": str},
        )
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%Y%m%d %H%M%S", errors="coerce"
        )
        if df["timestamp"].isna().any():
            bad = df["timestamp"].isna().sum()
            raise ValueError(f"{fp}: {bad} timestamps failed to parse")
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz).dt.tz_convert("UTC")
        frames.append(df[["timestamp", "open", "high", "low", "close"]])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp")
    merged = merged.set_index("timestamp")
    return merged[list(REQUIRED_COLS)].astype(float)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC data to a coarser frequency (e.g. '5min', '1h', '1d')."""
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return out.dropna(how="any")


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLC data from CSV.

    Expected columns: timestamp, open, high, low, close.
    Timestamp becomes the index (UTC if parseable).
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("CSV must have a 'timestamp' column")
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps failed to parse")
    df = df.set_index("timestamp").sort_index()
    return df[list(REQUIRED_COLS)].astype(float)


def synthetic_ohlc(
    bars: int = 5000,
    start_price: float = 150.0,
    mu: float = 0.0,
    sigma: float = 0.0015,
    seed: int = 42,
    freq: str = "1h",
) -> pd.DataFrame:
    """Generate a synthetic OHLC series via geometric Brownian motion.

    Defaults approximate an hourly USD/JPY-like series.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mu, scale=sigma, size=bars)
    close = start_price * np.exp(np.cumsum(returns))

    # Build O/H/L around close with small intrabar noise
    noise = rng.normal(loc=0.0, scale=sigma * 0.5, size=bars)
    open_ = np.concatenate([[start_price], close[:-1]]) * np.exp(noise * 0.3)
    high = np.maximum(open_, close) * np.exp(np.abs(noise) * 0.5)
    low = np.minimum(open_, close) * np.exp(-np.abs(noise) * 0.5)

    index = pd.date_range("2024-01-01", periods=bars, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=index,
    )
