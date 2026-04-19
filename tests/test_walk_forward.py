import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "walk_forward_cli", ROOT / "tools" / "walk_forward.py"
)
wf = importlib.util.module_from_spec(spec)
sys.modules["walk_forward_cli"] = wf
spec.loader.exec_module(wf)

from fx.backtest import BacktestConfig
from fx.data import synthetic_ohlc
from fx.optimize import generate_combos


def test_build_windows_non_overlap():
    windows = wf.build_windows(n_bars=1000, train=300, test=100, step=100)
    assert len(windows) == 7          # 300..400, then step 100 each, last test_end=1000
    first = windows[0]
    assert first.train_start == 0
    assert first.train_end == 300
    assert first.test_start == 300
    assert first.test_end == 400
    last = windows[-1]
    assert last.test_end == 1000


def test_build_windows_overlap_step_smaller_than_test():
    windows = wf.build_windows(n_bars=600, train=200, test=100, step=50)
    assert len(windows) > 0
    # Tests overlap by 50
    assert windows[1].test_start - windows[0].test_start == 50


def test_build_windows_respects_warmup():
    windows = wf.build_windows(n_bars=1000, train=300, test=100, step=100, min_warmup=50)
    assert windows[0].train_start == 50


def test_build_windows_too_short_raises():
    with pytest.raises(ValueError):
        wf.build_windows(n_bars=100, train=80, test=80, step=20)


def test_run_walk_forward_end_to_end():
    df = synthetic_ohlc(bars=800, seed=3)
    windows = wf.build_windows(n_bars=len(df), train=300, test=100, step=100)
    combos = generate_combos(
        fast=[5, 10], slow=[20, 40], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0], stop_atr=[None, 1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    result = wf.run_walk_forward(df, windows, combos, cfg, sort_by="pf", progress=False)

    # Per-window records exist for all windows
    assert len(result["per_window"]) == len(windows)
    # Stitched equity spans the test segments back-to-back
    assert len(result["stitched_equity"]) == len(windows) * 100
    # oos_perf uses the stitched curve
    assert result["oos_perf"].initial_equity == 100_000.0


def test_render_html_contains_sections():
    df = synthetic_ohlc(bars=800, seed=3)
    windows = wf.build_windows(n_bars=len(df), train=300, test=100, step=100)
    combos = generate_combos(
        fast=[5], slow=[20], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0], stop_atr=[1.5],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    result = wf.run_walk_forward(df, windows, combos, cfg, sort_by="pf", progress=False)
    html = wf.render_html(
        result, "WFA Test",
        context={"start": df.index[0], "end": df.index[-1]},
        sort_by="pf",
    )
    assert html.startswith("<!doctype html>")
    assert "窓ごとの結果" in html
    assert "OOS 資産推移" in html
    assert "パラメータ安定性" in html
