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


@dataclass(frozen=True)
class StopConfig:
    """ATR-based stop-loss.

    Stop distance = atr_mult * ATR measured at entry.
    enabled=False keeps the original vectorized (no-stop) path.
    """
    enabled: bool = False
    atr_mult: float = 2.0


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


def _apply_stops(
    signals: pd.DataFrame,
    stops: StopConfig,
) -> pd.Series:
    """Return an adjusted position series with ATR stops applied bar-by-bar.

    Convention: `position = signal.shift(1)` — i.e., position at bar i was
    decided at close of bar i-1, and the "entry price" is close[i-1].
    When a stop is hit during bar i (high/low breach), the position is
    forced flat for the *remainder* of the bar. A new signal at bar i+1
    may re-enter.
    """
    sig = signals["signal"].astype(int).values
    close = signals["close"].astype(float).values
    atr = signals.get("atr", pd.Series(0.0, index=signals.index)).astype(float).values
    high = signals.get("high", signals["close"]).astype(float).values
    low = signals.get("low", signals["close"]).astype(float).values

    n = len(sig)
    position = np.zeros(n, dtype=int)
    pos = 0
    entry_price = 0.0
    entry_atr = 0.0

    for i in range(1, n):
        stopped_this_bar = False
        # 1) intrabar stop check for any open position
        if pos != 0 and entry_atr > 0:
            stop_dist = stops.atr_mult * entry_atr
            if pos > 0 and low[i] <= entry_price - stop_dist:
                pos = 0
                entry_price = 0.0
                entry_atr = 0.0
                stopped_this_bar = True
            elif pos < 0 and high[i] >= entry_price + stop_dist:
                pos = 0
                entry_price = 0.0
                entry_atr = 0.0
                stopped_this_bar = True

        # 2) Reconcile with desired signal — but skip if a stop fired this bar
        # (a re-entry on the same bar as a stop-out would defeat the stop).
        if not stopped_this_bar:
            desired = sig[i - 1]
            if desired != pos:
                if desired != 0:
                    entry_price = close[i - 1]
                    entry_atr = atr[i - 1] if not np.isnan(atr[i - 1]) else 0.0
                    pos = int(desired)
                else:
                    pos = 0
                    entry_price = 0.0
                    entry_atr = 0.0

        position[i] = pos

    return pd.Series(position, index=signals.index, dtype=int)


def run_backtest(
    signals: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
    stops: StopConfig = StopConfig(),
) -> BacktestResult:
    """Run a vectorized backtest.

    `signals` must contain 'close' and 'signal' (desired position in {-1,0,+1}).
    Positions are shifted by one bar to avoid look-ahead.
    PnL per bar = position * (close_t - close_{t-1}) * size.
    Transaction cost = spread * size applied whenever position changes.

    If stops.enabled and signals has 'atr', 'high', 'low' columns, an
    ATR-based stop-loss is applied: positions are force-closed during the
    bar when the intrabar high/low breaches entry_price ± atr_mult*ATR.
    """
    if "signal" not in signals or "close" not in signals:
        raise ValueError("signals must contain 'signal' and 'close' columns")

    close = signals["close"].astype(float)
    if stops.enabled:
        position = _apply_stops(signals, stops)
    else:
        position = signals["signal"].shift(1).fillna(0).astype(int)

    price_diff = close.diff().fillna(0.0)
    bar_pnl = position * price_diff * config.size

    pos_change = position.diff().abs().fillna(position.abs())
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
