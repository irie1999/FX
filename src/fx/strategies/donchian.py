"""Donchian channel breakout.

Long when today's close makes a new N-bar high.
Short when today's close makes a new N-bar low.
Exit when the opposite side is breached (reverse signal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "donchian"
DISPLAY = "Donchian ブレイクアウト"


@dataclass(frozen=True)
class Params:
    channel_period: int = 20
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("Donchian needs high/low/close columns")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Shift by 1 so the breakout level is computed from *prior* bars (no look-ahead)
    upper = high.rolling(params.channel_period, min_periods=params.channel_period).max().shift(1)
    lower = low.rolling(params.channel_period, min_periods=params.channel_period).min().shift(1)

    long_cond = close > upper
    short_cond = close < lower

    # Stateful: flip between +1 and -1 on breakouts; otherwise hold.
    sig = np.zeros(len(close), dtype=int)
    pos = 0
    lc = long_cond.values
    sc = short_cond.values
    for i in range(len(close)):
        if lc[i]:
            pos = 1
        elif sc[i]:
            pos = -1
        sig[i] = pos

    signal = pd.Series(sig, index=df.index, dtype=int)

    a = _atr(df, params.atr_period)

    out = pd.DataFrame(
        {
            "close": close,
            "donchian_upper": upper,
            "donchian_lower": lower,
            "atr": a,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
