"""Signal generation: SMA crossover with RSI/ADX trend filters."""

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
    adx_period: int = 14
    adx_threshold: float = 0.0     # 0.0 = disabled (no trend-strength filter)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing. Needs high/low/close."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index (0-100).

    Measures trend strength regardless of direction.
    <20  = flat / range
    20-25 = weak trend / transitioning
    25+  = trending
    40+  = strong trend

    Needs `high`, `low`, `close` columns.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr_w = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period
    ).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * plus_dm_s / atr_w
        minus_di = 100.0 * minus_dm_s / atr_w
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / denom

    adx_s = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_s.fillna(0.0)


def generate_signals(df: pd.DataFrame, params: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Return a DataFrame with indicator columns and a `signal` in {-1, 0, +1}.

    Rules:
      - Long (+1) when fast SMA > slow SMA AND RSI < rsi_upper
      - Short (-1) when fast SMA < slow SMA AND RSI > rsi_lower
      - If `adx_threshold > 0`, both conditions also require ADX > threshold
        (trend-strength filter). When ADX falls below the threshold the
        signal reverts to 0, exiting the position at the next bar.

    `signal` is the desired position for the *next* bar; the backtest
    shifts it to avoid look-ahead bias.
    """
    if params.fast <= 0 or params.slow <= 0 or params.fast >= params.slow:
        raise ValueError("Require 0 < fast < slow")

    close = df["close"]
    sma_fast = close.rolling(params.fast, min_periods=params.fast).mean()
    sma_slow = close.rolling(params.slow, min_periods=params.slow).mean()
    r = rsi(close, params.rsi_period)

    has_ohlc = {"high", "low"} <= set(df.columns)
    a = atr(df, params.rsi_period) if has_ohlc else pd.Series(0.0, index=df.index)
    adx_s = adx(df, params.adx_period) if has_ohlc else pd.Series(0.0, index=df.index)

    adx_ok = pd.Series(True, index=df.index)
    if params.adx_threshold > 0:
        adx_ok = adx_s > params.adx_threshold

    long_cond = (sma_fast > sma_slow) & (r < params.rsi_upper) & adx_ok
    short_cond = (sma_fast < sma_slow) & (r > params.rsi_lower) & adx_ok

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.mask(long_cond, 1).mask(short_cond & ~long_cond, -1)

    # Warm-up: no signal until indicators are ready
    warmup = max(params.slow, params.rsi_period, params.adx_period)
    signal.iloc[:warmup] = 0

    out = pd.DataFrame(
        {
            "close": close,
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "rsi": r,
            "atr": a,
            "adx": adx_s,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
