"""Vectorized FX backtest with spread/transaction cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    size: float = 10_000.0   # units of base currency per position
    spread: float = 0.02     # price units, e.g. 0.02 JPY ≈ 2 pips on USD/JPY
    initial_equity: float = 1_000_000.0


@dataclass
class BacktestResult:
    equity: pd.Series           # equity curve in quote currency
    returns: pd.Series          # per-bar PnL in quote currency
    position: pd.Series         # position held during each bar {-1,0,+1}
    trades: pd.DataFrame        # entry/exit log with pnl
    signals: pd.DataFrame       # strategy output used


def _extract_trades(
    position: pd.Series,
    close: pd.Series,
    size: float,
    spread: float,
) -> pd.DataFrame:
    """Reconstruct discrete trades from a (shifted) position series.

    Because `position = signal.shift(1)`, a change from pos_prev to pos at bar i
    was decided at close of bar i-1. That close is the effective execution price,
    which keeps trade PnL consistent with the vectorized equity curve.
    """
    rows: list[dict] = []
    times = position.index
    closes = close.values
    positions = position.values.astype(int)

    entry_i: int | None = None
    entry_pos = 0

    def _close_trade(exec_i: int, exit_ts):
        if entry_i is None:
            return
        entry_price = float(closes[entry_i])
        exit_price = float(closes[exec_i])
        gross = (exit_price - entry_price) * entry_pos * size
        pnl = gross - spread * size  # half-spread per side = one full spread round-trip
        rows.append(
            {
                "entry_time": times[entry_i],
                "exit_time": exit_ts,
                "side": entry_pos,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            }
        )

    pos_prev = 0
    for i in range(len(positions)):
        pos = positions[i]
        if pos != pos_prev:
            # Execution bar for this transition is i-1 (prior close)
            exec_i = max(i - 1, 0)
            if pos_prev != 0:
                _close_trade(exec_i, times[i])
                entry_i = None
                entry_pos = 0
            if pos != 0:
                entry_i = exec_i
                entry_pos = int(pos)
        pos_prev = pos

    # Mark-to-market close for any position still open on the last bar
    if entry_i is not None and pos_prev != 0:
        _close_trade(len(positions) - 1, times[-1])

    return pd.DataFrame(rows)


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Run a vectorized backtest.

    `signals` must contain 'close' and 'signal' (desired position in {-1,0,+1}).
    Positions are shifted by one bar to avoid look-ahead.
    PnL per bar = position * (close_t - close_{t-1}) * size.
    Transaction cost = spread * size applied whenever position changes.
    """
    if "signal" not in signals or "close" not in signals:
        raise ValueError("signals must contain 'signal' and 'close' columns")

    close = signals["close"].astype(float)
    # Desired position produced at bar t is executed at bar t+1
    position = signals["signal"].shift(1).fillna(0).astype(int)

    price_diff = close.diff().fillna(0.0)
    bar_pnl = position * price_diff * config.size

    # Apply spread cost when position changes (including open from flat)
    pos_change = position.diff().abs().fillna(position.abs())
    # Half-spread per side; a flip (|Δpos|=2) costs two half-spreads = full spread
    cost = 0.5 * config.spread * config.size * pos_change
    net_pnl = bar_pnl - cost

    equity = config.initial_equity + net_pnl.cumsum()
    trades = _extract_trades(position, close, config.size, config.spread)

    return BacktestResult(
        equity=equity,
        returns=net_pnl,
        position=position,
        trades=trades,
        signals=signals,
    )
