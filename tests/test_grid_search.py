import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "grid_search_cli", ROOT / "tools" / "grid_search.py"
)
gs = importlib.util.module_from_spec(spec)
sys.modules["grid_search_cli"] = gs
spec.loader.exec_module(gs)

from fx.optimize import generate_combos, grid_search, parse_param_list


def test_parse_list_ints():
    assert parse_param_list("10,20,30", int) == [10, 20, 30]


def test_parse_list_floats_with_none():
    assert parse_param_list("none,1.25,2.0", float) == [None, 1.25, 2.0]


def test_parse_list_handles_whitespace():
    assert parse_param_list(" 10 , 20 ", int) == [10, 20]


def test_generate_combos_filters_invalid():
    combos = generate_combos(
        fast=[10, 50],
        slow=[20, 50],
        rsi_period=[14],
        rsi_upper=[70.0],
        rsi_lower=[30.0],
        stop_atr=[None, 2.0],
    )
    assert len(combos) == 4
    for c in combos:
        assert c.fast < c.slow
        assert c.rsi_lower < c.rsi_upper


def test_generate_combos_filters_rsi_bounds():
    combos = generate_combos(
        fast=[10], slow=[50], rsi_period=[14],
        rsi_upper=[30.0], rsi_lower=[70.0], stop_atr=[None],
    )
    assert combos == []


def test_grid_search_runs_end_to_end():
    from fx.data import synthetic_ohlc
    from fx.backtest import BacktestConfig

    df = synthetic_ohlc(bars=500, seed=1)
    combos = generate_combos(
        fast=[5, 10], slow=[20, 40], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0], stop_atr=[None, 1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    results = grid_search(df, combos, cfg)
    assert len(results) == len(combos)
    for col in ("fast", "slow", "pf", "sharpe", "cagr", "max_dd", "mar", "num_trades"):
        assert col in results.columns
    assert len(results.sort_values("pf", ascending=False).head(1)) == 1


def test_render_html_produces_document():
    from fx.data import synthetic_ohlc
    from fx.backtest import BacktestConfig

    df = synthetic_ohlc(bars=500, seed=1)
    combos = generate_combos(
        fast=[5], slow=[20], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0], stop_atr=[None, 1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    results = grid_search(df, combos, cfg)
    html = gs.render_html(
        results,
        sort_by="pf",
        top=10,
        title="Test grid",
        context={"start": df.index[0], "end": df.index[-1], "bars": len(df)},
    )
    assert html.startswith("<!doctype html>")
    assert "上位パラメータ一覧" in html
    assert "Test grid" in html
