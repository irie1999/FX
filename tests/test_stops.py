import pandas as pd

from fx.backtest import BacktestConfig, StopConfig, run_backtest
from fx.strategy import atr


def _make_signals(closes, highs, lows, signals, atrs):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "close": closes,
            "high": highs,
            "low": lows,
            "signal": signals,
            "atr": atrs,
        },
        index=idx,
    )


def test_atr_is_positive_on_real_movement():
    df = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.0, 105.0, 106.0, 105.0,
                      106.5, 107.0, 108.0, 107.5, 109.0],
            "high": [100.5, 101.5, 102.5, 102.2, 103.5, 104.5, 103.8, 105.5, 106.5, 105.5,
                     107.0, 107.5, 108.5, 108.0, 109.5],
            "low": [99.5, 100.5, 101.5, 101.0, 102.5, 103.5, 102.5, 104.5, 105.5, 104.5,
                    106.0, 106.5, 107.5, 107.0, 108.5],
        }
    )
    a = atr(df, period=5)
    assert a.iloc[-1] > 0
    # ATR should be roughly between bar ranges
    assert 0.5 < a.iloc[-1] < 2.5


def test_stop_closes_losing_long():
    # Long signal produced at bar 0; entry at close[0]=100
    # ATR=1.0, stop_mult=2 -> stop at 98
    # At bar 2 price crashes: low=97.5 which breaches the stop
    # Position must flip to 0 at bar 2 despite the signal still being long
    closes = [100.0, 100.0, 99.0, 100.0, 101.0]
    highs = [100.5, 100.5, 99.5, 100.5, 101.5]
    lows = [99.5, 99.5, 97.5, 99.5, 100.5]   # bar 2 breaches stop
    signals = [1, 1, 1, 1, 1]                # strategy still wants to be long
    atrs = [1.0, 1.0, 1.0, 1.0, 1.0]

    df = _make_signals(closes, highs, lows, signals, atrs)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    stops = StopConfig(enabled=True, atr_mult=2.0)
    r = run_backtest(df, cfg, stops=stops)

    # Expected position series (shift convention):
    #   bar 0: 0 (no signal yet)
    #   bar 1: 1 (sig[0]=1 -> position effective here)
    #   bar 2: 0 (stop hit intrabar)
    #   bar 3: 1 (re-entered via ongoing signal)
    #   bar 4: 1
    assert list(r.position.values) == [0, 1, 0, 1, 1]


def test_stop_closes_losing_short():
    closes = [100.0, 100.0, 101.0, 100.0, 99.0]
    highs = [100.5, 100.5, 102.5, 100.5, 99.5]   # bar 2 breaches short stop
    lows = [99.5, 99.5, 100.5, 99.5, 98.5]
    signals = [-1, -1, -1, -1, -1]
    atrs = [1.0, 1.0, 1.0, 1.0, 1.0]
    df = _make_signals(closes, highs, lows, signals, atrs)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    stops = StopConfig(enabled=True, atr_mult=2.0)
    r = run_backtest(df, cfg, stops=stops)
    assert list(r.position.values) == [0, -1, 0, -1, -1]


def test_stop_not_triggered_when_within_range():
    closes = [100.0, 100.0, 99.5, 100.5, 101.0]
    highs = [100.5, 100.5, 100.0, 101.0, 101.5]
    lows = [99.5, 99.5, 99.0, 100.0, 100.5]    # never breaches stop at 98
    signals = [1, 1, 1, 1, 1]
    atrs = [1.0, 1.0, 1.0, 1.0, 1.0]
    df = _make_signals(closes, highs, lows, signals, atrs)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    stops = StopConfig(enabled=True, atr_mult=2.0)
    r = run_backtest(df, cfg, stops=stops)
    assert list(r.position.values) == [0, 1, 1, 1, 1]


def test_disabled_stops_matches_baseline():
    closes = [100.0, 100.0, 99.0, 100.0, 101.0]
    highs = [100.5, 100.5, 99.5, 100.5, 101.5]
    lows = [99.5, 99.5, 97.5, 99.5, 100.5]
    signals = [1, 1, 1, 1, 1]
    atrs = [1.0, 1.0, 1.0, 1.0, 1.0]
    df = _make_signals(closes, highs, lows, signals, atrs)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=0.0)
    r_off = run_backtest(df, cfg, stops=StopConfig(enabled=False))
    # Position follows signal.shift(1) with no stops
    assert list(r_off.position.values) == [0, 1, 1, 1, 1]
