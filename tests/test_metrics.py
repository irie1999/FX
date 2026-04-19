import numpy as np
import pandas as pd

from fx.metrics import compute_performance, monthly_pnl


def _equity(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_total_return_and_drawdown():
    eq = _equity([1000, 1100, 1050, 1200, 1150])
    returns = eq.diff().fillna(0.0)
    trades = pd.DataFrame(
        [
            {"pnl": 100.0},
            {"pnl": -50.0},
            {"pnl": 150.0},
            {"pnl": -50.0},
        ]
    )
    p = compute_performance(eq, returns, trades, initial_equity=1000.0)
    assert p.total_return == 0.15
    # Running max: [1000,1100,1100,1200,1200]; DD at idx 1->2: -50/1100, idx 3->4: -50/1200
    assert p.max_drawdown < 0
    assert abs(p.max_drawdown - (-50.0 / 1100.0)) < 1e-9
    assert p.num_trades == 4
    assert p.win_rate == 0.5
    # profit factor = (100+150)/(50+50) = 2.5
    assert abs(p.profit_factor - 2.5) < 1e-9


def test_empty_trades():
    eq = _equity([1000, 1000, 1000])
    returns = eq.diff().fillna(0.0)
    p = compute_performance(eq, returns, pd.DataFrame(columns=["pnl"]), initial_equity=1000.0)
    assert p.num_trades == 0
    assert p.win_rate == 0.0
    assert p.profit_factor == 0.0
    assert p.total_return == 0.0


def test_extended_stats():
    eq = _equity([1000, 1100, 1050, 1200, 1150])
    returns = eq.diff().fillna(0.0)
    trades = pd.DataFrame(
        [
            {"pnl": 100.0},
            {"pnl": -50.0},
            {"pnl": 150.0},
            {"pnl": -50.0},
        ]
    )
    p = compute_performance(eq, returns, trades, initial_equity=1000.0)
    assert p.initial_equity == 1000.0
    assert p.final_equity == 1150.0
    assert p.net_profit == 150.0
    assert p.gross_profit == 250.0
    assert p.gross_loss == -100.0
    assert p.best_trade == 150.0
    assert p.worst_trade == -50.0
    assert p.avg_win == 125.0
    assert p.avg_loss == -50.0
    assert abs(p.risk_reward - 2.5) < 1e-9
    assert p.num_wins == 2
    assert p.num_losses == 2
    # Alternating W/L/W/L -> max streak both = 1
    assert p.max_win_streak == 1
    assert p.max_loss_streak == 1


def test_streaks_long_runs():
    eq = _equity([1000, 1000, 1000, 1000, 1000, 1000])
    returns = eq.diff().fillna(0.0)
    trades = pd.DataFrame(
        [{"pnl": v} for v in [10, 20, 30, -5, -5, -5, -5, 40]]
    )
    p = compute_performance(eq, returns, trades, initial_equity=1000.0)
    assert p.max_win_streak == 3
    assert p.max_loss_streak == 4


def test_exposure_uses_position():
    eq = _equity([1000, 1000, 1000, 1000, 1000])
    returns = eq.diff().fillna(0.0)
    pos = pd.Series([0, 1, 1, 0, -1], index=eq.index)
    p = compute_performance(
        eq, returns, pd.DataFrame(columns=["pnl"]), initial_equity=1000.0, position=pos
    )
    assert abs(p.exposure - 0.6) < 1e-9  # 3 of 5 bars have non-zero position


def test_monthly_pnl_breakdown():
    # 3 hourly bars per calendar month over 3 months
    idx = pd.DatetimeIndex(
        [
            "2024-01-15 00:00", "2024-01-15 01:00",
            "2024-02-10 00:00", "2024-02-10 01:00",
            "2024-03-05 00:00", "2024-03-05 01:00",
        ],
        tz="UTC",
    )
    eq = pd.Series([1_000, 1_100, 1_050, 1_200, 1_180, 1_300], index=idx, dtype=float)
    m = monthly_pnl(eq, initial_equity=1_000.0)
    assert list(m["month"]) == ["2024-01", "2024-02", "2024-03"]
    # Jan: 1000 -> 1100 = +100
    # Feb: 1100 -> 1200 = +100
    # Mar: 1200 -> 1300 = +100
    assert list(m["pnl"]) == [100.0, 100.0, 100.0]
