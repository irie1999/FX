"""Bollinger Band mean reversion.

Long when price pierces the lower band; short when price pierces the
upper band. Exit when price returns to the middle band (SMA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "bollinger"
DISPLAY = "Bollinger 逆張り"


@dataclass(frozen=True)
class Params:
    period: int = 20
    num_std: float = 2.0
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    close = df["close"].astype(float)
    mid = close.rolling(params.period, min_periods=params.period).mean()
    std = close.rolling(params.period, min_periods=params.period).std(ddof=0)
    upper = mid + params.num_std * std
    lower = mid - params.num_std * std

    # Step-wise position using stateful loop (desired position at end of bar).
    # Long while below mid after a dip to lower band, short while above mid
    # after a spike to upper band. Reset to 0 when price crosses mid.
    sig = np.zeros(len(close), dtype=int)
    pos = 0
    c = close.values
    m = mid.values
    u = upper.values
    l = lower.values
    for i in range(len(c)):
        if np.isnan(m[i]):
            sig[i] = 0
            continue
        if pos == 0:
            if c[i] < l[i]:
                pos = 1
            elif c[i] > u[i]:
                pos = -1
        elif pos > 0 and c[i] >= m[i]:
            pos = 0
        elif pos < 0 and c[i] <= m[i]:
            pos = 0
        sig[i] = pos

    signal = pd.Series(sig, index=df.index, dtype=int)

    has_hl = {"high", "low"} <= set(df.columns)
    a = _atr(df, params.atr_period) if has_hl else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "close": close,
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "atr": a,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
