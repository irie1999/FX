"""Opening Range Breakout (ORB).

For each UTC calendar day:
  1. First `range_bars` bars define the opening range
     (range_high = max(high), range_low = min(low))
  2. After the range is set, a close above range_high = long signal
     A close below range_low = short signal
  3. Position stays until the next day (day-trade rules will flatten EOD)

Signals are "desired position for the next bar", matching the rest of
the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "orb"
DISPLAY = "Opening Range Breakout"


@dataclass(frozen=True)
class Params:
    range_bars: int = 4      # e.g. 4 × 15min = first hour
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("ORB needs high/low/close columns")
    if df.index.tz is None:
        raise ValueError("ORB needs tz-aware index to detect calendar days")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    utc = df.index.tz_convert("UTC")
    day_key = utc.floor("1D")
    bar_of_day = pd.Series(df.groupby(day_key).cumcount().values, index=df.index)

    # Running per-day high/low over only the opening window
    in_range_window = bar_of_day < params.range_bars
    roll_high = high.where(in_range_window)
    roll_low = low.where(in_range_window)

    # Establish the range once per day (max/min over the opening window so far)
    range_high_so_far = roll_high.groupby(day_key).cummax()
    range_low_so_far = roll_low.groupby(day_key).cummin()

    # Once we leave the opening window the range is fixed -> forward-fill
    range_high = range_high_so_far.groupby(day_key).ffill()
    range_low = range_low_so_far.groupby(day_key).ffill()

    # Only active after the opening window has ended
    active = bar_of_day >= params.range_bars

    long_cond = active & (close > range_high)
    short_cond = active & (close < range_low)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.mask(long_cond, 1).mask(short_cond & ~long_cond, -1)

    a = _atr(df, params.atr_period)

    out = pd.DataFrame(
        {
            "close": close,
            "range_high": range_high,
            "range_low": range_low,
            "atr": a,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
