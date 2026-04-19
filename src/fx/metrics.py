"""Performance metrics for backtest results."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

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

    # Extended stats -----------------------------------------------------
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_profit: float = 0.0        # final - initial (quote currency)
    gross_profit: float = 0.0      # sum of winning trades
    gross_loss: float = 0.0        # sum of losing trades (negative)
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    risk_reward: float = 0.0       # |avg_win / avg_loss|
    expectancy: float = 0.0        # avg PnL per trade (same as avg_trade_pnl)
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_holding_bars: float = 0.0
    exposure: float = 0.0          # fraction of bars with non-flat position
    num_wins: int = 0
    num_losses: int = 0

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


def _max_streak(mask: pd.Series) -> int:
    """Longest run of True in a boolean series."""
    if mask.empty:
        return 0
    best = cur = 0
    for v in mask.values:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _avg_holding_bars(trades: pd.DataFrame, index: pd.Index) -> float:
    """Average number of bars between entry and exit for each trade."""
    if len(trades) == 0 or len(index) == 0:
        return 0.0
    if "entry_time" not in trades.columns or "exit_time" not in trades.columns:
        return 0.0
    pos_by_time = pd.Series(range(len(index)), index=index)
    durations = []
    for entry, exit_ in zip(trades["entry_time"], trades["exit_time"]):
        try:
            i0 = int(pos_by_time.loc[entry])
            i1 = int(pos_by_time.loc[exit_])
            durations.append(max(i1 - i0, 0))
        except KeyError:
            continue
    return float(np.mean(durations)) if durations else 0.0


def compute_performance(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    initial_equity: float,
    position: pd.Series | None = None,
) -> Performance:
    if len(equity) == 0:
        raise ValueError("Empty equity curve")

    final_equity = float(equity.iloc[-1])
    net_profit = final_equity - float(initial_equity)
    total_return = net_profit / float(initial_equity)

    bars_per_year = _bars_per_year(equity.index)
    years = len(equity) / bars_per_year if bars_per_year > 0 else 0.0
    if years > 0 and final_equity > 0:
        cagr = float((final_equity / initial_equity) ** (1.0 / years) - 1.0)
    else:
        cagr = 0.0

    eq_ret = equity.pct_change().dropna()
    if eq_ret.std(ddof=0) > 0 and len(eq_ret) > 1:
        sharpe = float(eq_ret.mean() / eq_ret.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    # Trade stats
    if len(trades) > 0:
        pnl = trades["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        num_trades = int(len(trades))
        num_wins = int(len(wins))
        num_losses = int(len(losses))
        win_rate = num_wins / num_trades if num_trades else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(losses.sum())
        profit_factor = (gross_profit / -gross_loss) if gross_loss < 0 else float("inf")
        avg_trade_pnl = float(pnl.mean())
        best_trade = float(pnl.max())
        worst_trade = float(pnl.min())
        avg_win = float(wins.mean()) if num_wins else 0.0
        avg_loss = float(losses.mean()) if num_losses else 0.0
        risk_reward = float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0.0
        expectancy = avg_trade_pnl
        max_win_streak = _max_streak(pnl > 0)
        max_loss_streak = _max_streak(pnl < 0)
        avg_holding_bars = _avg_holding_bars(trades, equity.index)
    else:
        num_trades = num_wins = num_losses = 0
        win_rate = profit_factor = avg_trade_pnl = 0.0
        gross_profit = gross_loss = 0.0
        best_trade = worst_trade = 0.0
        avg_win = avg_loss = risk_reward = expectancy = 0.0
        max_win_streak = max_loss_streak = 0
        avg_holding_bars = 0.0

    if position is not None and len(position) > 0:
        exposure = float((position != 0).mean())
    else:
        exposure = 0.0

    return Performance(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        initial_equity=float(initial_equity),
        final_equity=final_equity,
        net_profit=net_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        best_trade=best_trade,
        worst_trade=worst_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        risk_reward=risk_reward,
        expectancy=expectancy,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        avg_holding_bars=avg_holding_bars,
        exposure=exposure,
        num_wins=num_wins,
        num_losses=num_losses,
    )


def monthly_pnl(equity: pd.Series, initial_equity: float) -> pd.DataFrame:
    """Return a DataFrame with monthly PnL and return.

    Columns: ['month', 'pnl', 'return']. For the first month the reference
    is `initial_equity`; for subsequent months it's the previous month's
    closing equity.
    """
    if len(equity) == 0:
        return pd.DataFrame(columns=["month", "pnl", "return"])
    monthly_last = equity.resample("MS").last().ffill()
    if len(monthly_last) == 0:
        return pd.DataFrame(columns=["month", "pnl", "return"])
    prev = monthly_last.shift(1)
    prev.iloc[0] = float(initial_equity)
    diffs = monthly_last - prev
    returns = diffs / prev.replace(0, np.nan)
    df = pd.DataFrame(
        {
            "month": monthly_last.index.strftime("%Y-%m"),
            "pnl": diffs.values,
            "return": returns.values,
        }
    )
    return df.reset_index(drop=True)


def format_performance(p: Performance) -> str:
    pf = f"{p.profit_factor:.2f}" if np.isfinite(p.profit_factor) else "inf"
    lines = [
        f"Initial equity : {p.initial_equity:>12,.2f}",
        f"Final equity   : {p.final_equity:>12,.2f}",
        f"Net profit     : {p.net_profit:>12,.2f}",
        f"Total return   : {p.total_return:>12.2%}",
        f"CAGR           : {p.cagr:>12.2%}",
        f"Sharpe         : {p.sharpe:>12.2f}",
        f"Max drawdown   : {p.max_drawdown:>12.2%}",
        f"Win rate       : {p.win_rate:>12.2%}  ({p.num_wins}W / {p.num_losses}L)",
        f"# trades       : {p.num_trades:>12d}",
        f"Profit factor  : {pf:>12}",
        f"Best trade     : {p.best_trade:>12,.2f}",
        f"Worst trade    : {p.worst_trade:>12,.2f}",
        f"Avg win        : {p.avg_win:>12,.2f}",
        f"Avg loss       : {p.avg_loss:>12,.2f}",
        f"Risk/Reward    : {p.risk_reward:>12.2f}",
        f"Max win streak : {p.max_win_streak:>12d}",
        f"Max loss streak: {p.max_loss_streak:>12d}",
        f"Avg hold (bars): {p.avg_holding_bars:>12.1f}",
        f"Exposure       : {p.exposure:>12.2%}",
    ]
    return "\n".join(lines)
