"""SMA crossover + RSI filter — re-uses fx.strategy.generate_signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..strategy import StrategyParams, generate_signals as _generate

NAME = "sma_rsi"
DISPLAY = "SMA クロス + RSI"


@dataclass(frozen=True)
class Params:
    fast: int = 5
    slow: int = 20
    rsi_period: int = 14
    rsi_upper: float = 70.0
    rsi_lower: float = 30.0


DEFAULT = Params()


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    sp = StrategyParams(
        fast=params.fast, slow=params.slow,
        rsi_period=params.rsi_period,
        rsi_upper=params.rsi_upper, rsi_lower=params.rsi_lower,
    )
    return _generate(df, sp)
