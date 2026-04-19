"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Performance:
    total_return: float        # final PnL / initial equity
    cagr: float                # annualized return
    sharpe: float              # annualized Sharpe ratio
    max_drawdown: float        # max drawdown (negative number)
    win_rate: float            # fraction of winning trades
    num_trades: int
    profit_factor: float       # sum(wins) / sum(|losses|)
    avg_trade_pnl: float

    def as_dict(self) -> dict:
        return asdict(self)


def _bars_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return 252.0
    deltas = pd.Series(index).diff().dropna()
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta.total_seconds() <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / median_delta.total_seconds()


def compute_performance(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    initial_equity: float,
) -> Performance:
    if len(equity) == 0:
        raise ValueError("Empty equity curve")

    total_return = float((equity.iloc[-1] - initial_equity) / initial_equity)

    bars_per_year = _bars_per_year(equity.index)
    years = len(equity) / bars_per_year if bars_per_year > 0 else 0.0
    if years > 0 and equity.iloc[-1] > 0:
        cagr = float((equity.iloc[-1] / initial_equity) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    # Sharpe on per-bar simple returns of equity
    eq_ret = equity.pct_change().dropna()
    if eq_ret.std(ddof=0) > 0 and len(eq_ret) > 1:
        sharpe = float(eq_ret.mean() / eq_ret.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    if len(trades) > 0:
        wins = trades[trades["pnl"] > 0]["pnl"]
        losses = trades[trades["pnl"] < 0]["pnl"]
        win_rate = float(len(wins) / len(trades))
        loss_sum = float(-losses.sum())
        profit_factor = float(wins.sum() / loss_sum) if loss_sum > 0 else float("inf")
        avg_trade_pnl = float(trades["pnl"].mean())
        num_trades = int(len(trades))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        num_trades = 0

    return Performance(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
    )


def format_performance(p: Performance) -> str:
    pf = f"{p.profit_factor:.2f}" if np.isfinite(p.profit_factor) else "inf"
    lines = [
        f"Total return : {p.total_return:>10.2%}",
        f"CAGR         : {p.cagr:>10.2%}",
        f"Sharpe       : {p.sharpe:>10.2f}",
        f"Max drawdown : {p.max_drawdown:>10.2%}",
        f"Win rate     : {p.win_rate:>10.2%}",
        f"# trades     : {p.num_trades:>10d}",
        f"Profit factor: {pf:>10}",
        f"Avg trade PnL: {p.avg_trade_pnl:>10.2f}",
    ]
    return "\n".join(lines)
