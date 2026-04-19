"""Adaptive meta-strategy: pick the best recent performer.

Every `refresh_every` bars, look back `lookback_bars` bars and score each
candidate strategy by its cumulative PnL over that window. Use the
winner's signal until the next refresh. If the best score falls below
`flat_threshold` (default 0), go flat for the interval — no strategy
deserves our capital right now.

This is NOT curve fitting: at bar t we only look at data from [t-lookback, t]
and act from bar t onwards. The winner is re-evaluated in real time as
regimes change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Import peer strategy modules directly to avoid a circular import via
# strategies/__init__.py which itself imports this module.
from . import bollinger, donchian, ichimoku, macd, sma_rsi, supertrend
from ..strategy import atr as _atr

NAME = "adaptive"
DISPLAY = "適応型 (直近の勝者を選択)"


_CANDIDATE_MODULES = {
    "sma_rsi": sma_rsi,
    "supertrend": supertrend,
    "ichimoku": ichimoku,
    "donchian": donchian,
    "bollinger": bollinger,
    "macd": macd,
}


@dataclass(frozen=True)
class Params:
    candidates: tuple[str, ...] = (
        "sma_rsi", "supertrend", "ichimoku", "donchian", "bollinger", "macd",
    )
    lookback_bars: int = 60       # ~3 months of daily, ~2 weeks of 4h
    refresh_every: int = 10       # re-pick every 10 bars
    flat_threshold: float = 0.0   # go flat if best score < threshold
    spread: float = 0.02
    size: float = 10_000.0
    atr_period: int = 14


DEFAULT = Params()


def _pnl_per_bar(signals_df: pd.DataFrame, spread: float, size: float) -> pd.Series:
    """Quick per-bar net-PnL reconstruction mirroring run_backtest math."""
    close = signals_df["close"].astype(float)
    signal = signals_df["signal"].astype(int)
    position = signal.shift(1).fillna(0).astype(int)
    price_diff = close.diff().fillna(0.0)
    bar_pnl = position * price_diff * size
    pos_change = position.diff().abs().fillna(position.abs())
    cost = 0.5 * spread * size * pos_change
    return bar_pnl - cost


def generate(df: pd.DataFrame, params: Params = DEFAULT) -> pd.DataFrame:
    for name in params.candidates:
        if name not in _CANDIDATE_MODULES:
            raise ValueError(
                f"Unknown candidate {name!r}. Available: {sorted(_CANDIDATE_MODULES)}"
            )

    n = len(df)
    close = df["close"].astype(float)

    # Pre-compute each candidate's signal matrix and per-bar PnL.
    candidate_signals: dict[str, pd.Series] = {}
    candidate_pnls: dict[str, pd.Series] = {}
    for name in params.candidates:
        mod = _CANDIDATE_MODULES[name]
        sig_df = mod.generate(df)
        candidate_signals[name] = sig_df["signal"].astype(int)
        candidate_pnls[name] = _pnl_per_bar(sig_df, params.spread, params.size)

    # Rolling sum of each candidate's PnL over the lookback window.
    rolling = pd.DataFrame(
        {name: s.rolling(params.lookback_bars, min_periods=params.lookback_bars).sum()
         for name, s in candidate_pnls.items()}
    )

    # Pick a winner only at refresh boundaries; reuse between refreshes.
    selected = pd.Series([None] * n, index=df.index, dtype=object)
    current_winner: str | None = None
    for i in range(n):
        if i < params.lookback_bars:
            continue
        # Re-evaluate on a refresh boundary (every `refresh_every` bars after warmup)
        if current_winner is None or (i - params.lookback_bars) % params.refresh_every == 0:
            scores = rolling.iloc[i - 1]         # use last completed bar's rolling score
            best_name = scores.idxmax()
            best_score = float(scores.loc[best_name])
            if not np.isfinite(best_score) or best_score < params.flat_threshold:
                current_winner = None            # go flat until next refresh
            else:
                current_winner = best_name
        selected.iloc[i] = current_winner

    # Build combined signal: pick from winner's signal; 0 where flat.
    combined_arr = np.zeros(n, dtype=int)
    selected_arr = selected.values
    for name in params.candidates:
        mask = selected_arr == name
        if mask.any():
            combined_arr[mask] = candidate_signals[name].values[mask]
    combined = pd.Series(combined_arr, index=df.index, dtype=int)

    a = _atr(df, params.atr_period) if {"high", "low"} <= set(df.columns) else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "close": close,
            "atr": a,
            "signal": combined,
            "selected_strategy": selected.astype(object),
        }
    )
    for col in ("open", "high", "low"):
        if col in df.columns:
            out[col] = df[col]
    return out
