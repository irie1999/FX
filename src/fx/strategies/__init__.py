"""Strategy registry for comparison.

Each module exposes:
    Params         — dataclass with defaults
    DEFAULT        — Params()
    generate(df, params) -> DataFrame with columns suitable for run_backtest
                    (must include 'close', 'signal'; OHLC / atr carried when
                     available)
    NAME           — short identifier
    DISPLAY        — human-readable name (Japanese)
"""

from __future__ import annotations

from . import bollinger, donchian, ichimoku, macd, orb, sma_rsi, supertrend
from . import adaptive   # imports peer modules above, so must come last

REGISTRY = {
    sma_rsi.NAME: sma_rsi,
    orb.NAME: orb,
    bollinger.NAME: bollinger,
    donchian.NAME: donchian,
    macd.NAME: macd,
    supertrend.NAME: supertrend,
    ichimoku.NAME: ichimoku,
    adaptive.NAME: adaptive,
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name!r}. Choices: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY.keys())
