import numpy as np
import pandas as pd

from fx.backtest import BacktestConfig, StopConfig, run_backtest
from fx.data import synthetic_ohlc
from fx.status import compute_current_status, format_status_console
from fx.strategy import StrategyParams, generate_signals


def _run(bars=600, seed=1, stop_mult=None, **kwargs):
    df = synthetic_ohlc(bars=bars, seed=seed)
    params = StrategyParams(fast=5, slow=20, rsi_upper=101, rsi_lower=-1, **kwargs)
    signals = generate_signals(df, params)
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    stops = StopConfig(enabled=stop_mult is not None, atr_mult=stop_mult or 0.0)
    return run_backtest(signals, cfg, stops=stops), cfg, stops


def test_status_for_long_position():
    result, cfg, stops = _run()
    # Ensure a long position exists at the end for the test
    if int(result.position.iloc[-1]) == 0:
        # Force a long at the end so the status helper has something to report
        result.position.iloc[-1] = 1
    result.position.iloc[-1] = 1
    status = compute_current_status(result, cfg, stops)
    assert status.position == 1
    assert status.entry_price is not None
    assert status.entry_time is not None
    assert status.bars_held >= 1
    # Console format renders
    txt = format_status_console(status)
    assert "ポジション" in txt
    assert "ロング" in txt


def test_status_for_flat_position():
    df = synthetic_ohlc(bars=200, seed=2)
    # Use a slow SMA longer than the data so no signals fire
    params = StrategyParams(fast=5, slow=300)
    signals = generate_signals(df, params)
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    result = run_backtest(signals, cfg)
    status = compute_current_status(result, cfg, StopConfig(enabled=False))
    assert status.position == 0
    assert status.entry_price is None
    assert "ポジション" in format_status_console(status)


def test_next_action_labels():
    df = synthetic_ohlc(bars=600, seed=1)
    params = StrategyParams(fast=5, slow=20, rsi_upper=101, rsi_lower=-1)
    signals = generate_signals(df, params)
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    result = run_backtest(signals, cfg)

    # Construct a hand-made scenario: flat now, signal = +1 -> should say buy
    result.position.iloc[-1] = 0
    signals2 = result.signals.copy()
    signals2.loc[signals2.index[-1], "signal"] = 1
    result_mod = result
    result_mod.signals = signals2

    status = compute_current_status(result_mod, cfg, StopConfig(enabled=False))
    assert "新規ロング" in status.next_action


def test_stop_level_long():
    result, cfg, stops = _run(stop_mult=2.0)
    # Force long at last bar so the stop-level path is exercised deterministically
    result.position.iloc[-1] = 1
    # Ensure a matching prior non-zero position (consecutive)
    if int(result.position.iloc[-2]) != 1:
        result.position.iloc[-2] = 1
    status = compute_current_status(result, cfg, stops)
    if status.atr_now is not None:
        assert status.stop_level is not None
        assert status.stop_level < status.current_price
