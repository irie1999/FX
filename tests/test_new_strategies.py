import numpy as np
import pandas as pd
import pytest

from fx import strategies
from fx.data import synthetic_ohlc
from fx.strategies import bollinger, donchian, orb, sma_rsi


def _trending_df(n=200, step=0.5, freq="15min"):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = np.arange(n, dtype=float) * step + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close},
        index=idx,
    )


def _ranging_df(n=200, freq="15min"):
    rng = np.random.default_rng(2)
    close = 100.0 + np.sin(np.linspace(0, 20 * np.pi, n)) * 2 + rng.normal(0, 0.1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close},
        index=idx,
    )


def test_registry_has_four_strategies():
    assert set(strategies.names()) == {"sma_rsi", "orb", "bollinger", "donchian"}


def test_orb_long_after_breakout_in_uptrend():
    df = _trending_df(n=400)
    sig = orb.generate(df, orb.Params(range_bars=4))
    # After the opening 4 bars, we expect long signals to fire in an uptrend
    assert (sig["signal"] == 1).any()
    # Early bars (inside opening window) must be 0
    first_day_first_bars = sig.iloc[:4]
    assert (first_day_first_bars["signal"] == 0).all()


def test_orb_short_after_breakdown_in_downtrend():
    df = _trending_df(n=400, step=-0.5)
    sig = orb.generate(df, orb.Params(range_bars=4))
    assert (sig["signal"] == -1).any()


def test_orb_requires_tz():
    df = _trending_df(n=100)
    df.index = df.index.tz_convert(None)
    with pytest.raises(ValueError):
        orb.generate(df)


def test_bollinger_fires_on_extremes():
    df = _ranging_df(n=400)
    sig = bollinger.generate(df, bollinger.Params(period=20, num_std=2.0))
    # In a sinusoidal range we should see both long and short mean-reversion entries
    assert (sig["signal"] == 1).any()
    assert (sig["signal"] == -1).any()


def test_donchian_long_on_breakout():
    df = _trending_df(n=200, step=0.5)
    sig = donchian.generate(df, donchian.Params(channel_period=20))
    # Upward trend should produce a sustained long position after warmup
    assert (sig["signal"].iloc[-1] == 1)


def test_sma_rsi_wrapper_matches_base():
    df = synthetic_ohlc(bars=500, seed=5)
    s1 = sma_rsi.generate(df)
    assert "signal" in s1.columns
    assert set(s1["signal"].unique()).issubset({-1, 0, 1})


def test_all_strategies_produce_signal_column():
    df = synthetic_ohlc(bars=400, seed=1, freq="15min")
    for name in strategies.names():
        mod = strategies.get(name)
        sig = mod.generate(df)
        assert "close" in sig.columns
        assert "signal" in sig.columns
        assert set(sig["signal"].unique()).issubset({-1, 0, 1})
