import numpy as np
import pandas as pd

from fx.metrics import compute_performance


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
