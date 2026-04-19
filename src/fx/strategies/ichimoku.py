"""一目均衡表 (Ichimoku Kinko Hyo) — cloud breakout.

Components (classical 9 / 26 / 52 settings):
    転換線 (Tenkan-sen)   = (9-bar high  + 9-bar low)  / 2
    基準線 (Kijun-sen)    = (26-bar high + 26-bar low) / 2
    先行スパンA (Senkou A) = (Tenkan + Kijun) / 2 shifted forward 26 bars
    先行スパンB (Senkou B) = (52-bar high + 52-bar low) / 2 shifted forward 26
    遅行スパン (Chikou)   = close shifted back 26 bars

Trading rule (雲ブレイク):
    long  : close > max(Senkou A, Senkou B)  AND  Tenkan > Kijun
    short : close < min(Senkou A, Senkou B)  AND  Tenkan < Kijun
    flat  : otherwise (inside the cloud or contradicting signals)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategy import atr as _atr

NAME = "ichimoku"
DISPLAY = "一目均衡表 (雲ブレイク)"


@dataclass(frozen=True)
class Params:
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    cloud_shift: int = 26
    atr_period: int = 14


DEFAULT = Params()


def _midline(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    return (high.rolling(period, min_periods=period).max()
            + low.rolling(period, min_periods=period).min()) / 2.0


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    if not {"high", "low", "close"} <= set(df.columns):
        raise ValueError("Ichimoku needs high/low/close columns")

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    tenkan = _midline(high, low, params.tenkan_period)
    kijun = _midline(high, low, params.kijun_period)
    senkou_a = ((tenkan + kijun) / 2.0).shift(params.cloud_shift)
    senkou_b = _midline(high, low, params.senkou_b_period).shift(params.cloud_shift)

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

    long_cond = (close > cloud_top) & (tenkan > kijun)
    short_cond = (close < cloud_bottom) & (tenkan < kijun)

    sig = pd.Series(0, index=df.index, dtype=int)
    sig = sig.mask(long_cond, 1).mask(short_cond & ~long_cond, -1)

    warmup = params.senkou_b_period + params.cloud_shift
    sig.iloc[:warmup] = 0

    a = _atr(df, params.atr_period)

    out = pd.DataFrame(
        {
            "close": close,
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "atr": a,
            "signal": sig,
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
