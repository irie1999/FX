import numpy as np
import pandas as pd
import pytest

from fx import strategies
from fx.strategies import ichimoku, macd, supertrend


def _uptrend(n=200, step=0.5):
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = np.arange(n, dtype=float) * step + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close},
        index=idx,
    )


def _downtrend(n=200, step=-0.5):
    return _uptrend(n=n, step=step)


def _ranging(n=400, freq="1D"):
    rng = np.random.default_rng(7)
    x = np.linspace(0, 12 * np.pi, n)
    close = 100.0 + np.sin(x) * 3 + rng.normal(0, 0.1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close},
        index=idx,
    )


# ------------------------------------------------------- MACD

def test_macd_long_signal_in_uptrend():
    sig = macd.generate(_uptrend(n=200))
    # After warmup, sustained uptrend should result in long signals
    warmup = macd.DEFAULT.slow + macd.DEFAULT.signal_period
    assert (sig["signal"].iloc[warmup:] == 1).any()


def test_macd_short_signal_in_downtrend():
    sig = macd.generate(_downtrend())
    warmup = macd.DEFAULT.slow + macd.DEFAULT.signal_period
    assert (sig["signal"].iloc[warmup:] == -1).any()


def test_macd_requires_fast_lt_slow():
    with pytest.raises(ValueError):
        macd.generate(_uptrend(), macd.Params(fast=50, slow=20))


def test_macd_output_columns():
    out = macd.generate(_uptrend())
    for col in ("close", "macd", "macd_signal", "macd_hist", "signal"):
        assert col in out.columns


# ----------------------------------------------------- Supertrend

def test_supertrend_locks_uptrend():
    sig = supertrend.generate(_uptrend(n=300), supertrend.Params(atr_period=10, multiplier=3.0))
    # After the warmup, a strong uptrend should keep direction = +1 most of the time
    tail = sig["signal"].iloc[50:]
    assert (tail == 1).mean() > 0.7


def test_supertrend_flips_on_reversal():
    # Create a sequence that rises then falls
    up = _uptrend(n=150, step=0.5)
    down = _downtrend(n=150, step=-0.5).copy()
    # Continue price from where up ended
    offset = up["close"].iloc[-1] - down["close"].iloc[0]
    down[["open", "high", "low", "close"]] += offset
    down.index = pd.date_range(up.index[-1] + pd.Timedelta(days=1),
                                periods=len(down), freq="1D", tz="UTC")
    df = pd.concat([up, down])
    sig = supertrend.generate(df, supertrend.Params(atr_period=10, multiplier=3.0))
    # Should see +1 earlier, -1 later
    assert (sig["signal"].iloc[50:150] == 1).any()
    assert (sig["signal"].iloc[-50:] == -1).any()


def test_supertrend_output_has_band_columns():
    out = supertrend.generate(_uptrend())
    for col in ("supertrend", "supertrend_upper", "supertrend_lower", "signal"):
        assert col in out.columns


# ----------------------------------------------------- Ichimoku

def test_ichimoku_long_in_strong_uptrend():
    sig = ichimoku.generate(_uptrend(n=300), ichimoku.Params())
    # Once past the cloud warmup, a strong uptrend should drive long signals
    warmup = ichimoku.DEFAULT.senkou_b_period + ichimoku.DEFAULT.cloud_shift
    tail = sig["signal"].iloc[warmup + 20:]
    assert (tail == 1).any()


def test_ichimoku_warmup_is_zero():
    sig = ichimoku.generate(_uptrend(n=200))
    warmup = ichimoku.DEFAULT.senkou_b_period + ichimoku.DEFAULT.cloud_shift
    assert (sig["signal"].iloc[:warmup] == 0).all()


def test_ichimoku_requires_high_low_close():
    df = pd.DataFrame(
        {"close": np.arange(100)}, index=pd.date_range("2024-01-01", periods=100, freq="1D", tz="UTC")
    )
    with pytest.raises(ValueError):
        ichimoku.generate(df)


# ----------------------------------------------------- Registry

def test_registry_contains_all_strategies():
    assert set(strategies.names()) == {
        "sma_rsi", "orb", "bollinger", "donchian",
        "macd", "supertrend", "ichimoku", "adaptive",
        "tsmom", "bbs",
    }


def test_all_registered_strategies_produce_valid_signals():
    df = _uptrend(n=300)
    for name in strategies.names():
        sig = strategies.get(name).generate(df)
        assert "close" in sig.columns
        assert "signal" in sig.columns
        assert set(sig["signal"].unique()).issubset({-1, 0, 1}), name
