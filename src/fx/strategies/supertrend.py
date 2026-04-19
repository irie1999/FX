"""Supertrend indicator (ATR-based dynamic stop / trend switch).

Classical Supertrend with Wilder-smoothed ATR:

    hl2          = (high + low) / 2
    basic_upper  = hl2 + multiplier * ATR
    basic_lower  = hl2 - multiplier * ATR

The "final" bands are trailing versions of the basic bands:

    final_upper[i] = basic_upper[i]  if basic_upper[i] < final_upper[i-1]
                                       or close[i-1] > final_upper[i-1]
                    else final_upper[i-1]
    final_lower[i] = basic_lower[i]  if basic_lower[i] > final_lower[i-1]
                                       or close[i-1] < final_lower[i-1]
                    else final_lower[i-1]

Direction flips when close crosses the active band. The supertrend
value itself is that active band (acts as a trailing stop).

Signal: +1 while the trend is up, -1 while down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "supertrend"
DISPLAY = "Supertrend"


@dataclass(frozen=True)
class Params:
    atr_period: int = 10
    multiplier: float = 3.0


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("Supertrend needs high/low/close columns")

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values

    atr_s = _atr(df, params.atr_period).values
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + params.multiplier * atr_s
    basic_lower = hl2 - params.multiplier * atr_s

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction = np.ones(n, dtype=int)        # +1 up, -1 down
    super_line = np.zeros(n)

    for i in range(n):
        if i == 0 or np.isnan(atr_s[i]):
            final_upper[i] = basic_upper[i] if not np.isnan(basic_upper[i]) else 0.0
            final_lower[i] = basic_lower[i] if not np.isnan(basic_lower[i]) else 0.0
            direction[i] = 1
            super_line[i] = final_lower[i]
            continue

        # Trailing upper band: only loosens when price breaks above the previous upper
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Trailing lower band: only loosens when price breaks below the previous lower
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction flip rule
        prev_dir = direction[i - 1]
        if prev_dir == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1

        super_line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    # Suppress signals until ATR has warmed up
    warmup = params.atr_period
    sig = pd.Series(direction, index=df.index, dtype=int)
    sig.iloc[:warmup] = 0

    out = pd.DataFrame(
        {
            "close": df["close"].astype(float),
            "atr": pd.Series(atr_s, index=df.index),
            "supertrend": pd.Series(super_line, index=df.index),
            "supertrend_upper": pd.Series(final_upper, index=df.index),
            "supertrend_lower": pd.Series(final_lower, index=df.index),
            "signal": sig,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
