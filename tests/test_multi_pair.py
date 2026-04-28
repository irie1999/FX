import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "multi_pair_cli", ROOT / "tools" / "multi_pair_backtest.py"
)
mp = importlib.util.module_from_spec(spec)
sys.modules["multi_pair_cli"] = mp
spec.loader.exec_module(mp)


def test_quote_currency_extracts_last_three():
    assert mp.quote_currency("USDJPY") == "JPY"
    assert mp.quote_currency("eurusd") == "USD"
    assert mp.quote_currency("GBPJPY") == "JPY"


def test_to_jpy_known_currencies():
    assert mp.to_jpy("USDJPY") == 1.0
    assert mp.to_jpy("EURUSD") == mp.QUOTE_TO_JPY["USD"]
    assert mp.to_jpy("EURGBP") == mp.QUOTE_TO_JPY["GBP"]


def test_to_jpy_unknown_returns_default():
    # Not in map -> default 150 (treated as USD-quoted ish)
    assert mp.to_jpy("XXXYYY") == 150.0


def test_default_spread_jpy_pair():
    assert mp.default_spread("USDJPY") == 0.02
    assert mp.default_spread("usdjpy") == 0.02


def test_default_spread_falls_back():
    assert mp.default_spread("XYZUSD") == 0.0003


def test_parse_spread_overrides_basic():
    out = mp.parse_spread_overrides("EURUSD=0.0001,GBPUSD=0.0002")
    assert out == {"EURUSD": 0.0001, "GBPUSD": 0.0002}


def test_parse_spread_overrides_handles_whitespace():
    out = mp.parse_spread_overrides("  USDJPY = 0.015 , EURUSD = 0.00015 ")
    assert out == {"USDJPY": 0.015, "EURUSD": 0.00015}


def test_parse_spread_overrides_empty():
    assert mp.parse_spread_overrides("") == {}
    assert mp.parse_spread_overrides(None) == {}


def test_parse_spread_overrides_bad_format_raises():
    with pytest.raises(ValueError):
        mp.parse_spread_overrides("USDJPY:0.02")


def test_aggregate_portfolio_sums_pnl():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    # Fake two pair results
    pair1_eq = pd.Series([100_000, 100_500, 101_000, 100_800, 101_200], index=idx, dtype=float)
    pair2_eq = pd.Series([100_000, 100_300, 100_600, 100_400, 100_900], index=idx, dtype=float)
    r1 = mp.PairResult(
        pair="USDJPY", spread=0.02, bars=5,
        period_start=idx[0], period_end=idx[-1],
        perf_jpy={}, equity_jpy=pair1_eq, raw_perf=None,
        trades=pd.DataFrame(columns=["pnl"]),
    )
    r2 = mp.PairResult(
        pair="EURUSD", spread=0.0002, bars=5,
        period_start=idx[0], period_end=idx[-1],
        perf_jpy={}, equity_jpy=pair2_eq, raw_perf=None,
        trades=pd.DataFrame(columns=["pnl"]),
    )
    portfolio = mp.aggregate_portfolio([r1, r2], initial_total_jpy=200_000.0)
    eq = portfolio["equity"]
    # Final portfolio = initial + sum of pnl moves
    # Pair1 net: +1200, Pair2 net: +900 => total 2100
    assert eq.iloc[-1] == pytest.approx(202_100.0)
    # First bar should be at the initial value (no pnl yet on bar 0)
    assert eq.iloc[0] == pytest.approx(200_000.0)


def test_run_pair_skips_missing_files(tmp_path):
    out = mp.run_pair(
        pair="USDJPY",
        histdata_dir=tmp_path,           # empty
        resample="1d",
        strategy_name="adaptive",
        size=1000,
        spread=0.02,
        initial_equity_jpy=100_000.0,
        stop_atr=1.25,
        start=None, end=None,
    )
    assert out is None
