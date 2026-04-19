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

from . import bollinger, donchian, orb, sma_rsi

REGISTRY = {
    sma_rsi.NAME: sma_rsi,
    orb.NAME: orb,
    bollinger.NAME: bollinger,
    donchian.NAME: donchian,
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name!r}. Choices: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY.keys())
