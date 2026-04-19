import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "grid_search", ROOT / "tools" / "grid_search.py"
)
gs = importlib.util.module_from_spec(spec)
sys.modules["grid_search"] = gs
spec.loader.exec_module(gs)


def test_parse_list_ints():
    assert gs._parse_list("10,20,30", int) == [10, 20, 30]


def test_parse_list_floats_with_none():
    assert gs._parse_list("none,1.25,2.0", float) == [None, 1.25, 2.0]


def test_parse_list_handles_whitespace():
    assert gs._parse_list(" 10 , 20 ", int) == [10, 20]


def test_generate_combos_filters_invalid():
    combos = gs.generate_combos(
        fast=[10, 50],
        slow=[20, 50],
        rsi_period=[14],
        rsi_upper=[70.0],
        rsi_lower=[30.0],
        stop_atr=[None, 2.0],
    )
    # Valid: (10,20), (10,50) x 2 stops = 4; (50,*) fails fast<slow
    assert len(combos) == 4
    for c in combos:
        assert c.fast < c.slow
        assert c.rsi_lower < c.rsi_upper


def test_generate_combos_filters_rsi_bounds():
    combos = gs.generate_combos(
        fast=[10], slow=[50],
        rsi_period=[14],
        rsi_upper=[30.0],   # lower >= upper -> skip
        rsi_lower=[70.0],
        stop_atr=[None],
    )
    assert combos == []


def test_grid_search_runs_end_to_end():
    # Short synthetic data to keep test fast
    from fx.data import synthetic_ohlc
    from fx.backtest import BacktestConfig

    df = synthetic_ohlc(bars=500, seed=1)
    combos = gs.generate_combos(
        fast=[5, 10],
        slow=[20, 40],
        rsi_period=[14],
        rsi_upper=[70.0],
        rsi_lower=[30.0],
        stop_atr=[None, 1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    results = gs.grid_search(df, combos, cfg)
    # All combos produced a row
    assert len(results) == len(combos)
    # Required columns present
    for col in ("fast", "slow", "pf", "sharpe", "cagr", "max_dd", "mar", "num_trades"):
        assert col in results.columns
    # Sorting by pf works
    top = results.sort_values("pf", ascending=False).head(1)
    assert len(top) == 1


def test_render_html_produces_document():
    from fx.data import synthetic_ohlc
    from fx.backtest import BacktestConfig

    df = synthetic_ohlc(bars=500, seed=1)
    combos = gs.generate_combos(
        fast=[5], slow=[20], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0], stop_atr=[None, 1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    results = gs.grid_search(df, combos, cfg)
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
