import numpy as np
import pandas as pd

from fx.strategy import StrategyParams, adx, generate_signals


def _trend_df(n=120, start=100.0, step=1.0):
    """Strong uptrend OHLC."""
    close = np.arange(n, dtype=float) * step + start
    high = close + 0.3
    low = close - 0.3
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def _flat_df(n=120, price=100.0):
    """Tight ranging market."""
    rng = np.random.default_rng(0)
    close = price + rng.normal(0, 0.05, size=n)
    high = close + 0.05
    low = close - 0.05
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )


def test_adx_is_high_in_trend():
    df = _trend_df()
    a = adx(df, period=14)
    # Strong sustained trend should drive ADX well above 25
    assert a.iloc[-1] > 40


def test_adx_is_low_in_range():
    df = _flat_df()
    a = adx(df, period=14)
    # Random range noise should keep ADX below 25 most of the time
    assert a.iloc[-1] < 25


def test_adx_filter_blocks_low_trend_signals():
    df = _flat_df()
    # Without ADX filter: some signals fire based on SMA crossovers
    no_filter = generate_signals(df, StrategyParams(fast=5, slow=20,
                                                    rsi_upper=101, rsi_lower=-1))
    # With a high ADX threshold: all signals should be suppressed
    filtered = generate_signals(df, StrategyParams(fast=5, slow=20,
                                                    rsi_upper=101, rsi_lower=-1,
                                                    adx_threshold=30))
    assert (no_filter["signal"] != 0).any()
    assert (filtered["signal"] == 0).all()


def test_adx_filter_allows_strong_trend_signals():
    df = _trend_df()
    filtered = generate_signals(df, StrategyParams(fast=5, slow=20,
                                                    rsi_upper=101, rsi_lower=-1,
                                                    adx_threshold=25))
    # In a strong trend the ADX filter should not suppress all long signals
    assert (filtered["signal"] == 1).any()


def test_adx_filter_zero_threshold_acts_as_disabled():
    df = _trend_df()
    off = generate_signals(df, StrategyParams(fast=5, slow=20,
                                               rsi_upper=101, rsi_lower=-1,
                                               adx_threshold=0.0))
    baseline = generate_signals(df, StrategyParams(fast=5, slow=20,
                                                    rsi_upper=101, rsi_lower=-1))
    # Should produce identical signals when threshold is 0
    pd.testing.assert_series_equal(off["signal"], baseline["signal"])
