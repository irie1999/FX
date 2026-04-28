"""Bollinger Band Squeeze breakout (BBS).

  John Bollinger, "Bollinger on Bollinger Bands" (2001)
  + Larry Williams' volatility-breakout framing.

When the bands' width contracts to a multi-month minimum the market is
unusually quiet — energy is being stored. We trade the *direction* of
the eventual breakout out of the squeeze, exiting when price returns
to the bands' middle line.

Mechanics:
  upper / lower / mid  : BB(period, num_std)
  bandwidth            : upper - lower
  is_squeeze[i]        : bandwidth[i] is the minimum over the prior
                          `squeeze_lookback` bars
  recent_squeeze[i]    : at least one squeeze bar fired in the last
                          `release_window` bars (so we don't enter
                          months later when the squeeze is irrelevant)

Position state machine:
  flat      → long  on (close > upper) AND recent_squeeze
            → short on (close < lower) AND recent_squeeze
  long      → flat  on close <= mid (mean-reversion exit)
            → short on (close < lower) AND recent_squeeze (flip)
  short     → flat  on close >= mid
            → long  on (close > upper) AND recent_squeeze (flip)

Hold periods are short (until the move plays out); win-rate is low but
the average winner is much larger than the average loser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "bbs"
DISPLAY = "BB スクイーズ・ブレイクアウト"


@dataclass(frozen=True)
class Params:
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_lookback: int = 120        # 6 months of daily, or ~1 month of 4h
    squeeze_quantile: float = 0.25     # bandwidth in the lowest N% qualifies as squeeze
    release_window: int = 20           # how long the "post-squeeze" zone lasts
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    close = df["close"].astype(float)
    sma = close.rolling(params.bb_period, min_periods=params.bb_period).mean()
    std = close.rolling(params.bb_period, min_periods=params.bb_period).std(ddof=0)
    upper = sma + params.bb_std * std
    lower = sma - params.bb_std * std
    bandwidth = upper - lower

    # Squeeze: current bandwidth is in the lowest `squeeze_quantile` slice of
    # the rolling lookback window. Captures the entire flat-vol period, not
    # just the single bar that hit the absolute minimum.
    bandwidth_thresh = bandwidth.rolling(
        params.squeeze_lookback, min_periods=params.squeeze_lookback
    ).quantile(params.squeeze_quantile)
    is_squeeze = (bandwidth <= bandwidth_thresh) & bandwidth.notna()

    # Recent-squeeze: any squeeze bar in the last `release_window` bars.
    recent_squeeze = is_squeeze.rolling(params.release_window, min_periods=1).max().fillna(0).astype(bool)

    # Stateful pass to build the position series.
    n = len(df)
    sig = np.zeros(n, dtype=int)
    pos = 0
    c = close.values
    u = upper.values
    l = lower.values
    m = sma.values
    rs = recent_squeeze.values

    for i in range(n):
        if np.isnan(m[i]):
            sig[i] = 0
            continue
        breakout_up = c[i] > u[i] and rs[i]
        breakout_dn = c[i] < l[i] and rs[i]

        if pos == 0:
            if breakout_up:
                pos = 1
            elif breakout_dn:
                pos = -1
        elif pos == 1:
            if breakout_dn:
                pos = -1
            elif c[i] <= m[i]:
                pos = 0
        else:  # pos == -1
            if breakout_up:
                pos = 1
            elif c[i] >= m[i]:
                pos = 0

        sig[i] = pos

    signal = pd.Series(sig, index=df.index, dtype=int)

    has_hl = {"high", "low"} <= set(df.columns)
    a = _atr(df, params.atr_period) if has_hl else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "close": close,
            "bbs_upper": upper,
            "bbs_lower": lower,
            "bbs_mid": sma,
            "bandwidth": bandwidth,
            "is_squeeze": is_squeeze.astype(int),
            "atr": a,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
