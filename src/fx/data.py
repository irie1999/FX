"""Price data loading and synthetic generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ("open", "high", "low", "close")


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
