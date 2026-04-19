import numpy as np
import pandas as pd

from fx.strategy import StrategyParams, generate_signals, rsi


def _price_series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC")
    close = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_rsi_all_up_is_100():
    close = pd.Series(np.arange(1, 101, dtype=float))
    r = rsi(close, period=14)
    assert r.iloc[-1] > 99.0


def test_rsi_all_down_is_0():
    close = pd.Series(np.arange(100, 0, -1, dtype=float))
    r = rsi(close, period=14)
    assert r.iloc[-1] < 1.0


def test_signals_uptrend_goes_long():
    df = _price_series(np.linspace(100, 200, 300))
    sig = generate_signals(df, StrategyParams(fast=5, slow=20, rsi_period=14))
    # RSI in a pure uptrend saturates near 100, so RSI filter blocks longs.
    # Relax the rsi_upper to confirm long condition fires when allowed.
    sig2 = generate_signals(df, StrategyParams(fast=5, slow=20, rsi_period=14, rsi_upper=101.0))
    assert (sig2["signal"].iloc[-10:] == 1).all()


def test_signals_downtrend_goes_short():
    df = _price_series(np.linspace(200, 100, 300))
    sig = generate_signals(
        df, StrategyParams(fast=5, slow=20, rsi_period=14, rsi_lower=-1.0)
    )
    assert (sig["signal"].iloc[-10:] == -1).all()


def test_signal_values_are_in_set():
    df = _price_series(np.random.default_rng(0).normal(100, 1, 500).cumsum() + 1000)
    sig = generate_signals(df)
    assert set(sig["signal"].unique()).issubset({-1, 0, 1})


def test_warmup_is_zero():
    df = _price_series(np.linspace(100, 200, 300))
    params = StrategyParams(fast=5, slow=20, rsi_period=14)
    sig = generate_signals(df, params)
    warmup = max(params.slow, params.rsi_period)
    assert (sig["signal"].iloc[:warmup] == 0).all()


def test_invalid_params_raise():
    df = _price_series(np.linspace(100, 200, 100))
    import pytest

    with pytest.raises(ValueError):
        generate_signals(df, StrategyParams(fast=50, slow=20))
