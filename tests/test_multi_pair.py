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


def _trades_df(rows):
    return pd.DataFrame(rows)


def test_format_trades_table_empty():
    out = mp._format_trades_table(pd.DataFrame())
    assert "トレードはありません" in out


def test_format_trades_table_with_pair_column():
    df = _trades_df([
        {
            "entry_time": pd.Timestamp("2024-01-01", tz="UTC"),
            "exit_time": pd.Timestamp("2024-01-05", tz="UTC"),
            "side": 1, "entry_price": 150.0, "exit_price": 151.0,
            "pnl": 1000.0, "pair": "USDJPY",
        },
        {
            "entry_time": pd.Timestamp("2024-01-10", tz="UTC"),
            "exit_time": pd.Timestamp("2024-01-12", tz="UTC"),
            "side": -1, "entry_price": 1.10, "exit_price": 1.09,
            "pnl": 100.0, "pair": "EURUSD",
        },
    ])
    out = mp._format_trades_table(df, limit=10, show_pair=True)
    assert "通貨ペア" in out
    assert "USDJPY" in out
    assert "EURUSD" in out
    assert "買い" in out
    assert "売り" in out
    # Most recent first: EURUSD (2024-01-12) should appear before USDJPY (2024-01-05)
    assert out.index("EURUSD") < out.index("USDJPY")


def test_format_trades_table_limit_truncates():
    rows = [
        {
            "entry_time": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i),
            "exit_time": pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=i),
            "side": 1, "entry_price": 100.0, "exit_price": 101.0,
            "pnl": 100.0, "pair": "USDJPY",
        } for i in range(5)
    ]
    df = _trades_df(rows)
    out = mp._format_trades_table(df, limit=2, show_pair=True)
    assert "直近 2 件 / 全 5 件を表示しています" in out


def test_render_html_includes_trade_sections():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    pair1_eq = pd.Series([100_000, 100_500, 101_000, 100_800, 101_200], index=idx, dtype=float)
    trades_df = pd.DataFrame([
        {
            "entry_time": idx[0], "exit_time": idx[2],
            "side": 1, "entry_price": 150.0, "exit_price": 151.0, "pnl": 1000.0,
        }
    ])
    r = mp.PairResult(
        pair="USDJPY", spread=0.02, bars=5,
        period_start=idx[0], period_end=idx[-1],
        perf_jpy={
            "pf": 2.0, "sharpe": 1.0, "cagr": 0.1, "total_return": 0.012,
            "max_dd": -0.01, "win_rate": 1.0, "num_trades": 1,
            "net_profit": 1200.0, "rr": 1.0, "best": 1000.0, "worst": 0.0,
        },
        equity_jpy=pair1_eq, raw_perf=None,
        trades=trades_df,
    )
    portfolio = mp.aggregate_portfolio([r], initial_total_jpy=100_000.0)
    html = mp.render_html([r], portfolio, "adaptive",
                          period="2024-01-01 → 2024-01-05",
                          title="Test")
    assert "直近トレード一覧" in html
    assert "通貨ペア別トレード詳細" in html
    assert "USDJPY" in html
