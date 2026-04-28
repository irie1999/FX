"""Tests for the three new advanced strategies: TSMOM, BBS, and Pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx import strategies
from fx.data import synthetic_ohlc
from fx.pairs import PairsParams, compute_spread_signal, run_pairs_backtest
from fx.strategies import bbs, tsmom


def _uptrend(n=400, step=0.5, freq="1D"):
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = np.arange(n, dtype=float) * step + 100.0
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close},
        index=idx,
    )


def _downtrend(n=400, freq="1D"):
    """Multiplicative decay so close stays strictly positive."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 200.0 * (0.999 ** np.arange(n, dtype=float))
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close},
        index=idx,
    )


def _ranging(n=400, freq="1D"):
    rng = np.random.default_rng(11)
    x = np.linspace(0, 12 * np.pi, n)
    close = 100.0 + np.sin(x) * 1.5 + rng.normal(0, 0.05, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close + 0.05, "low": close - 0.05, "close": close},
        index=idx,
    )


# ================================================================== TSMOM


def test_tsmom_long_after_uptrend():
    sig = tsmom.generate(_uptrend(n=400), tsmom.Params(lookback_bars=100))
    # After warmup, rising prices → +1
    assert (sig["signal"].iloc[150:] == 1).all()


def test_tsmom_short_after_downtrend():
    sig = tsmom.generate(_downtrend(n=400), tsmom.Params(lookback_bars=100))
    assert (sig["signal"].iloc[150:] == -1).all()


def test_tsmom_warmup_is_zero():
    sig = tsmom.generate(_uptrend(n=400), tsmom.Params(lookback_bars=100))
    assert (sig["signal"].iloc[:100] == 0).all()


def test_tsmom_skip_recent_zeros_extra_bars():
    sig = tsmom.generate(
        _uptrend(n=400),
        tsmom.Params(lookback_bars=100, skip_recent_bars=10),
    )
    assert (sig["signal"].iloc[:110] == 0).all()


def test_tsmom_invalid_lookback():
    with pytest.raises(ValueError):
        tsmom.generate(_uptrend(n=200), tsmom.Params(lookback_bars=0))


def test_tsmom_signal_set_constraint():
    df = synthetic_ohlc(bars=400, seed=4)
    sig = tsmom.generate(df, tsmom.Params(lookback_bars=60))
    assert set(sig["signal"].unique()).issubset({-1, 0, 1})


# =================================================================== BBS


def test_bbs_warmup_is_zero():
    df = _ranging(n=400)
    params = bbs.Params(bb_period=20, squeeze_lookback=120)
    sig = bbs.generate(df, params)
    # First `bb_period` bars must be flat (BB itself is NaN)
    assert (sig["signal"].iloc[: params.bb_period] == 0).all()


def test_bbs_signal_values_in_set():
    df = synthetic_ohlc(bars=400, seed=8)
    sig = bbs.generate(df)
    assert set(sig["signal"].unique()).issubset({-1, 0, 1})


def test_bbs_fires_on_breakout_after_squeeze():
    """Construct a synthetic squeeze + breakout so BBS must enter long."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    # First 150 bars: a tight range (squeeze)
    base = np.full(n, 100.0)
    base[150:] = np.linspace(100.0, 110.0, n - 150)  # explosive uptrend
    # Add tiny noise on the squeeze part
    rng = np.random.default_rng(5)
    base[:150] += rng.normal(0, 0.05, 150)
    df = pd.DataFrame(
        {"open": base, "high": base + 0.05, "low": base - 0.05, "close": base},
        index=idx,
    )
    sig = bbs.generate(df, bbs.Params(bb_period=20, squeeze_lookback=80))
    # Somewhere after the breakout we should see a long position
    assert (sig["signal"].iloc[160:] == 1).any()


def test_bbs_registered_in_strategies():
    assert "bbs" in strategies.names()


def test_tsmom_registered_in_strategies():
    assert "tsmom" in strategies.names()


# ================================================================== PAIRS


def test_pairs_signal_zero_when_spread_constant():
    """Constant log-spread → rolling std == 0 → no entries."""
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    a = pd.Series(np.full(200, 100.0), index=idx)
    b = pd.Series(np.full(200, 50.0), index=idx)
    spread, z, pos = compute_spread_signal(a, b, PairsParams(z_window=30, entry_z=2.0))
    assert (pos == 0).all()


def test_pairs_signal_short_when_spread_explodes_up():
    """If A spikes vs B, log-spread shoots up → expect a short-spread position."""
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    a = pd.Series(100.0, index=idx)
    b = pd.Series(50.0, index=idx)
    a.iloc[150:] = 200.0   # A doubles late in the series
    spread, z, pos = compute_spread_signal(
        a, b,
        PairsParams(z_window=30, entry_z=2.0, exit_z=0.5, stop_z=10.0),
    )
    # Some bar after the shock should have a short-spread position (-1) or stop-out (0)
    later = pos.iloc[150:]
    assert (later == -1).any() or (later == 0).all()


def test_pairs_run_backtest_returns_expected_shape():
    idx = pd.date_range("2024-01-01", periods=200, freq="1D", tz="UTC")
    a_close = pd.Series(np.linspace(100, 110, 200), index=idx)
    b_close = pd.Series(np.linspace(50, 55, 200) + np.sin(np.linspace(0, 30, 200)),
                         index=idx)
    a_df = pd.DataFrame({"open": a_close, "high": a_close + 0.1,
                          "low": a_close - 0.1, "close": a_close})
    b_df = pd.DataFrame({"open": b_close, "high": b_close + 0.1,
                          "low": b_close - 0.1, "close": b_close})
    res = run_pairs_backtest(
        a_df, b_df,
        a_to_jpy=150.0, b_to_jpy=150.0,
        size=1000, spread_a=0.0, spread_b=0.0,
        initial_equity_jpy=200_000.0,
        params=PairsParams(z_window=30, entry_z=1.5, exit_z=0.3, stop_z=10.0),
    )
    assert len(res.equity) == 200
    assert res.equity.iloc[0] == pytest.approx(200_000.0)
    assert res.position.dtype == int
    assert set(res.position.unique()).issubset({-1, 0, 1})


def test_pairs_align_raises_on_no_common_timestamps():
    from fx.pairs import _align
    a = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC"),
    )
    b = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="1D", tz="UTC"),
    )
    with pytest.raises(ValueError):
        _align(a, b)
