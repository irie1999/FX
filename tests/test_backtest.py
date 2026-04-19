import numpy as np
import pandas as pd

from fx.backtest import BacktestConfig, run_backtest


def _make_signals(closes, signals):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame({"close": closes, "signal": signals}, index=idx)


def test_long_trade_profit_with_spread():
    # Flat, then long for two bars while price rises by 1.0 per bar
    closes = [100.0, 100.0, 101.0, 102.0, 102.0]
    signals = [0, 1, 1, 0, 0]   # position effective shifted: [0,0,1,1,0]
    df = _make_signals(closes, signals)
    cfg = BacktestConfig(size=1000.0, spread=0.1, initial_equity=0.0)
    r = run_backtest(df, cfg)

    # Bar pnl: position*diff*size
    # position after shift: [0,0,1,1,0] ; diff: [0,0,1,1,0]
    # gross = sum(position*diff*size) = 1000*1 + 1000*1 = 2000
    # Entry cost at bar 2 (pos 0->1): 0.5*spread*size*1 = 50
    # Exit cost at bar 4 (pos 1->0): 0.5*spread*size*1 = 50
    # Net = 2000 - 100 = 1900
    assert float(r.equity.iloc[-1]) == 1900.0
    assert len(r.trades) == 1
    assert r.trades.iloc[0]["pnl"] == 1900.0


def test_short_trade_profit():
    closes = [100.0, 100.0, 99.0, 98.0, 98.0]
    signals = [0, -1, -1, 0, 0]
    df = _make_signals(closes, signals)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    r = run_backtest(df, cfg)
    # position shifted: [0,0,-1,-1,0] ; diff: [0,0,-1,-1,0]
    # gross = (-1)*(-1)*1000 + (-1)*(-1)*1000 = 2000
    assert float(r.equity.iloc[-1]) == 2000.0


def test_no_lookahead():
    closes = [100.0, 110.0, 110.0]
    # Signal only at bar 0 (before the jump). With shift, position is 0 at bar 1.
    signals = [1, 0, 0]
    df = _make_signals(closes, signals)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    r = run_backtest(df, cfg)
    # pos shifted: [0,1,0]; bar1 diff=10 -> pnl=10000; then flips back, cost=0
    assert float(r.equity.iloc[-1]) == 10000.0


def test_flip_costs_full_spread():
    closes = [100.0, 100.0, 100.0]
    signals = [1, -1, 0]
    df = _make_signals(closes, signals)
    cfg = BacktestConfig(size=1000.0, spread=0.1, initial_equity=0.0)
    r = run_backtest(df, cfg)
    # pos shifted: [0, 1, -1]
    # position-change magnitudes: [0, 1, 2] (first NaN filled by |pos[0]|=0)
    # cost = 0.5 * 0.1 * 1000 * 3 = 150 ; price diffs zero => final equity = -150
    assert float(r.equity.iloc[-1]) == -150.0
