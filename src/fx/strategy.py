"""Signal generation: SMA crossover with RSI trend filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    fast: int = 20
    slow: int = 50
    rsi_period: int = 14
    rsi_upper: float = 70.0
    rsi_lower: float = 30.0


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing ≈ EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    # Edge cases: avg_loss==0 & avg_gain>0 -> 100; avg_gain==0 & avg_loss>0 -> 0; both 0 -> 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(avg_gain != 0, 0.0)
    out = out.where(~both_zero, 50.0)
    return out.fillna(50.0)


def generate_signals(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Return a DataFrame with indicator columns and a `signal` in {-1, 0, +1}.

    Rules:
      - Long (+1) when fast SMA > slow SMA AND RSI < rsi_upper
      - Short (-1) when fast SMA < slow SMA AND RSI > rsi_lower
      - Otherwise flat (0)

    The `signal` is the desired position for the *next* bar; the backtest
    shifts it to avoid look-ahead bias.
    """
    if params.fast <= 0 or params.slow <= 0 or params.fast >= params.slow:
        raise ValueError("Require 0 < fast < slow")

    close = df["close"]
    sma_fast = close.rolling(params.fast, min_periods=params.fast).mean()
    sma_slow = close.rolling(params.slow, min_periods=params.slow).mean()
    r = rsi(close, params.rsi_period)

    long_cond = (sma_fast > sma_slow) & (r < params.rsi_upper)
    short_cond = (sma_fast < sma_slow) & (r > params.rsi_lower)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.mask(long_cond, 1).mask(short_cond & ~long_cond, -1)

    # Warm-up: no signal until indicators are ready
    warmup = max(params.slow, params.rsi_period)
    signal.iloc[:warmup] = 0

    return pd.DataFrame(
        {
            "close": close,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "rsi": r,
            "signal": signal,
        }
    )
