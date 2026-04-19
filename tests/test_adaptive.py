import numpy as np
import pandas as pd
import pytest

from fx import strategies
from fx.data import synthetic_ohlc
from fx.strategies import adaptive


def _df(bars=800, seed=1, freq="1D"):
    return synthetic_ohlc(bars=bars, seed=seed, freq=freq)


def test_adaptive_registered():
    assert "adaptive" in strategies.names()


def test_adaptive_produces_valid_signals():
    df = _df(bars=500)
    out = adaptive.generate(df)
    assert "close" in out.columns
    assert "signal" in out.columns
    assert "selected_strategy" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_adaptive_picks_a_candidate_after_warmup():
    df = _df(bars=500)
    out = adaptive.generate(
        df, adaptive.Params(lookback_bars=60, refresh_every=20, flat_threshold=-1e18)
    )
    sel = out["selected_strategy"].dropna()
    assert len(sel) > 0
    # The selected names must be known candidates
    assert set(sel.unique()) <= {"sma_rsi", "supertrend", "ichimoku", "donchian", "bollinger", "macd"}


def test_adaptive_warmup_zero():
    df = _df(bars=200)
    params = adaptive.Params(lookback_bars=80)
    out = adaptive.generate(df, params)
    assert (out["signal"].iloc[:params.lookback_bars] == 0).all()
    assert out["selected_strategy"].iloc[:params.lookback_bars].isna().all()


def test_adaptive_respects_candidate_subset():
    df = _df(bars=400)
    out = adaptive.generate(
        df, adaptive.Params(candidates=("sma_rsi", "supertrend"),
                             lookback_bars=60, refresh_every=20)
    )
    sel = out["selected_strategy"].dropna().unique()
    assert set(sel) <= {"sma_rsi", "supertrend"}


def test_adaptive_flat_threshold_forces_flat():
    df = _df(bars=400)
    # Set flat_threshold huge so nothing ever qualifies -> always flat
    out = adaptive.generate(
        df, adaptive.Params(lookback_bars=60, refresh_every=20,
                             flat_threshold=1e18)
    )
    # After warmup, selected_strategy should be None / NaN for every bar
    post_warmup = out.iloc[60:]
    assert post_warmup["selected_strategy"].isna().all()
    assert (post_warmup["signal"] == 0).all()


def test_adaptive_rejects_unknown_candidate():
    df = _df(bars=200)
    with pytest.raises(ValueError, match="Unknown candidate"):
        adaptive.generate(df, adaptive.Params(candidates=("xxx_bad",)))
