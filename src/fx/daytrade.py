"""Day-trading helpers: force positions flat at an end-of-day cutoff.

The backtest engine already knows how to honour a signal value of 0 as
"flat". We reuse that by overwriting the raw signal with 0 at every bar
whose timestamp is at or after the daily cutoff (configurable, default
NY close 21:00 UTC). The next-day session reopens normally.

This keeps the vectorized backtest intact while producing positions that
close every day — i.e., true day trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dtime

import pandas as pd

DEFAULT_EOD_UTC = dtime(21, 0)   # 21:00 UTC = NY close (approx; ignores DST)


@dataclass(frozen=True)
class DayTradeConfig:
    eod_utc: dtime = DEFAULT_EOD_UTC
    # Optional: also blank out signals during the first few bars after open,
    # letting the market settle. 0 disables this.
    warmup_bars_per_day: int = 0


def apply_daytrade_rules(signals: pd.DataFrame, config: DayTradeConfig = DayTradeConfig()) -> pd.DataFrame:
    """Return a new signals DataFrame with positions forced flat at EOD.

    `signals` must have a tz-aware DatetimeIndex and a 'signal' column.
    """
    if "signal" not in signals.columns:
        raise ValueError("signals must have a 'signal' column")
    if signals.index.tz is None:
        raise ValueError("signals index must be timezone-aware for day-trade rules")

    out = signals.copy()
    # EOD cutoff: any bar at or after the cutoff within a UTC day is flat.
    times = out.index.tz_convert("UTC").time
    eod = config.eod_utc
    flat_mask = pd.Series([t >= eod for t in times], index=out.index)
    out.loc[flat_mask, "signal"] = 0

    # Optional per-day warmup
    if config.warmup_bars_per_day > 0:
        # Within each UTC calendar day, zero the first N bars.
        day_key = out.index.tz_convert("UTC").floor("1D")
        grouped = out.groupby(day_key).cumcount()
        out.loc[grouped < config.warmup_bars_per_day, "signal"] = 0

    return out
