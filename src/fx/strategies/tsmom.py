"""Time-Series Momentum (TSMOM).

  Moskowitz, Ooi, Pedersen (2012)
  "Time Series Momentum"
  Journal of Financial Economics 104, 228-250

Rule: the sign of the cumulative return over the past `lookback_bars`
predicts the next bar's direction. If the asset is up over the lookback
window, go long; if down, go short. Volatility-targeted position sizing
is the original paper's main innovation, but our backtest engine uses
fixed unit sizing — sizing is delegated to the user (see notes below).

Notes on sizing:
  The paper scales each leg by `target_vol / realized_vol` to equalize
  risk across assets. With our integer signal {-1, 0, +1} we can't
  express continuous sizing, so the user should pick `--size` such that
  each trade is a similar % of equity. For mixed-pair operation the
  multi_pair_backtest tool is the right wrapper.

Why it works (paper's findings):
  - 1985-2009, 58 instruments, Sharpe ≈ 1.5
  - Strong autocorrelation at 1-12 month horizons
  - Reversal beyond ~18 months → only use lookback ≤ 12 months
  - Robust across asset classes (FX, equities, commodities, bonds)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "tsmom"
DISPLAY = "時系列モメンタム (TSMOM)"


@dataclass(frozen=True)
class Params:
    lookback_bars: int = 252       # ~1 year of daily bars
    skip_recent_bars: int = 0      # skip the most recent N bars (avoids 1-bar mean-reversion)
    atr_period: int = 14


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if params.lookback_bars <= 0:
        raise ValueError("lookback_bars must be > 0")

    close = df["close"].astype(float)
    # Past return up to (and not including) the most recent `skip_recent_bars`
    end_close = close.shift(params.skip_recent_bars)
    start_close = end_close.shift(params.lookback_bars)
    past_return = (end_close / start_close) - 1.0

    signal = pd.Series(0, index=df.index, dtype=int)
    signal = signal.mask(past_return > 0, 1).mask(past_return < 0, -1)

    warmup = params.lookback_bars + params.skip_recent_bars
    signal.iloc[:warmup] = 0

    has_hl = {"high", "low"} <= set(df.columns)
    a = _atr(df, params.atr_period) if has_hl else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "close": close,
            "past_return": past_return,
            "atr": a,
            "signal": signal,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
