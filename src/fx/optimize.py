"""Parameter optimization primitives shared by grid search and walk-forward."""

from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

from .backtest import BacktestConfig, StopConfig, run_backtest
from .metrics import compute_performance
from .strategy import StrategyParams, generate_signals


@dataclass(frozen=True)
class Combo:
    fast: int
    slow: int
    rsi_period: int
    rsi_upper: float
    rsi_lower: float
    stop_atr: float | None   # None = stops disabled

    def describe(self) -> str:
        stop = f"{self.stop_atr:.2f}×ATR" if self.stop_atr is not None else "off"
        return (
            f"fast={self.fast} slow={self.slow} rsi={self.rsi_period} "
            f"U/L={self.rsi_upper:.0f}/{self.rsi_lower:.0f} stop={stop}"
        )


def parse_param_list(raw: str, cast):
    """Comma-separated value parser. 'none'/'off'/'-' map to Python None."""
    items = [s.strip() for s in raw.split(",") if s.strip()]
    out = []
    for s in items:
        if s.lower() in ("none", "off", "-"):
            out.append(None)
        else:
            out.append(cast(s))
    return out


def generate_combos(
    fast: list[int],
    slow: list[int],
    rsi_period: list[int],
    rsi_upper: list[float],
    rsi_lower: list[float],
    stop_atr: list[float | None],
) -> list[Combo]:
    combos: list[Combo] = []
    for f, s, rp, ru, rl, sa in itertools.product(
        fast, slow, rsi_period, rsi_upper, rsi_lower, stop_atr
    ):
        if f >= s:
            continue                 # Require fast < slow
        if rl >= ru:
            continue                 # Require lower < upper
        combos.append(
            Combo(fast=f, slow=s, rsi_period=rp, rsi_upper=ru, rsi_lower=rl, stop_atr=sa)
        )
    return combos


def run_combo(df: pd.DataFrame, combo: Combo, cfg: BacktestConfig):
    """Run one backtest. Returns (BacktestResult, Performance)."""
    params = StrategyParams(
        fast=combo.fast,
        slow=combo.slow,
        rsi_period=combo.rsi_period,
        rsi_upper=combo.rsi_upper,
        rsi_lower=combo.rsi_lower,
    )
    signals = generate_signals(df, params)
    stops = StopConfig(
        enabled=combo.stop_atr is not None,
        atr_mult=combo.stop_atr or 0.0,
    )
    result = run_backtest(signals, cfg, stops=stops)
    perf = compute_performance(
        result.equity, result.returns, result.trades,
        cfg.initial_equity, position=result.position,
    )
    return result, perf


def combo_row(combo: Combo, perf, extra: dict | None = None) -> dict:
    """Flat dict suitable for a results DataFrame."""
    mar = perf.cagr / abs(perf.max_drawdown) if perf.max_drawdown < 0 else float("inf")
    row = {
        "fast": combo.fast,
        "slow": combo.slow,
        "rsi": combo.rsi_period,
        "rsi_upper": combo.rsi_upper,
        "rsi_lower": combo.rsi_lower,
        "stop_atr": combo.stop_atr if combo.stop_atr is not None else float("nan"),
        "num_trades": perf.num_trades,
        "win_rate": perf.win_rate,
        "pf": perf.profit_factor,
        "sharpe": perf.sharpe,
        "cagr": perf.cagr,
        "total_return": perf.total_return,
        "max_dd": perf.max_drawdown,
        "mar": mar,
        "net_profit": perf.net_profit,
        "rr": perf.risk_reward,
        "best": perf.best_trade,
        "worst": perf.worst_trade,
    }
    if extra:
        row.update(extra)
    return row


def grid_search(
    df: pd.DataFrame,
    combos: Iterable[Combo],
    cfg: BacktestConfig,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    combos = list(combos)
    start = time.time()
    for i, combo in enumerate(combos, 1):
        try:
            _, perf = run_combo(df, combo, cfg)
            rows.append(combo_row(combo, perf))
        except Exception as exc:
            print(f"[warn] combo {combo} failed: {exc}", file=sys.stderr)
        if progress is not None:
            progress(i, len(combos))
        elif i % 20 == 0 or i == len(combos):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            eta = (len(combos) - i) / max(rate, 1e-9)
            print(f"  {i:>5}/{len(combos)}  ({rate:.1f}/s, ETA {eta:.1f}s)", file=sys.stderr)
    return pd.DataFrame(rows)


SORT_COLUMNS = {
    "pf": "pf",
    "sharpe": "sharpe",
    "cagr": "cagr",
    "mar": "mar",
    "net_profit": "net_profit",
}


def best_by(results: pd.DataFrame, metric: str) -> pd.Series | None:
    col = SORT_COLUMNS[metric]
    finite = results[results[col].apply(lambda v: v == v and v != float("inf"))]
    if len(finite) == 0:
        return None
    return finite.sort_values(col, ascending=False).iloc[0]
