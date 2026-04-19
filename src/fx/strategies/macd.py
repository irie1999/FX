"""MACD (Moving Average Convergence Divergence) crossover.

  MACD line   = EMA(fast) - EMA(slow)
  Signal line = EMA(signal_period) of MACD line
  Histogram   = MACD - Signal

Long when MACD crosses *above* the signal line (histogram turns positive).
Short when MACD crosses *below* the signal line (histogram turns negative).

This is one of the most widely-used swing-trading signals; EMAs react
faster than simple moving averages so the system tends to catch the
start of moves earlier than SMA crossovers at the cost of more
whipsaws during chop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "macd"
DISPLAY = "MACD クロス"


@dataclass(frozen=True)
class Params:
    fast: int = 12
    slow: int = 26
    signal_period: int = 9
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if params.fast >= params.slow:
        raise ValueError("MACD requires fast < slow")

    close = df["close"].astype(float)
    ema_fast = close.ewm(span=params.fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=params.signal_period, adjust=False).mean()
    hist = macd - signal

    sig = pd.Series(0, index=df.index, dtype=int)
    sig = sig.mask(hist > 0, 1).mask(hist < 0, -1)

    warmup = params.slow + params.signal_period
    sig.iloc[:warmup] = 0

    has_hl = {"high", "low"} <= set(df.columns)
    a = _atr(df, params.atr_period) if has_hl else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "close": close,
            "macd": macd,
            "macd_signal": signal,
            "macd_hist": hist,
            "atr": a,
            "signal": sig,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
