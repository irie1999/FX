import pytest

from fx.backtest import BacktestConfig, run_backtest
from fx.data import synthetic_ohlc
from fx.metrics import compute_performance
from fx.strategy import StrategyParams, generate_signals

try:
    from fx.report import render_html, write_html
except ImportError:  # pragma: no cover
    pytest.skip("matplotlib not available", allow_module_level=True)


def _run_small_backtest():
    df = synthetic_ohlc(bars=500, seed=1)
    params = StrategyParams(fast=5, slow=20, rsi_period=14, rsi_upper=101.0, rsi_lower=-1.0)
    signals = generate_signals(df, params)
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    result = run_backtest(signals, cfg)
    perf = compute_performance(result.equity, result.returns, result.trades, cfg.initial_equity)
    return result, perf, params, cfg


def test_render_html_returns_valid_document():
    result, perf, params, cfg = _run_small_backtest()
    doc = render_html(result, perf, params, cfg, title="Unit Test Run")
    assert doc.startswith("<!doctype html>")
    assert "Unit Test Run" in doc
    assert "損益サマリー" in doc
    assert "詳細指標" in doc
    assert "パラメータ" in doc
    assert "月次損益" in doc
    # Embedded images are base64 PNGs
    assert "data:image/png;base64," in doc
    # Core yen figures and metrics
    assert "初期資金" in doc
    assert "最終資金" in doc
    assert "純損益" in doc
    assert "シャープレシオ" in doc
    assert "最大利益トレード" in doc


def test_write_html_creates_file(tmp_path):
    result, perf, params, cfg = _run_small_backtest()
    target = tmp_path / "sub" / "report.html"
    out = write_html(target, result, perf, params, cfg)
    assert out == target
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert len(content) > 10_000  # includes embedded images


def test_report_handles_no_trades(tmp_path):
    # Shorter than the slow-SMA warmup -> no signals fire -> no trades
    df = synthetic_ohlc(bars=30, seed=7)
    params = StrategyParams(fast=5, slow=60, rsi_period=14)
    signals = generate_signals(df, params)
    cfg = BacktestConfig(size=1000.0, spread=0.0, initial_equity=100_000.0)
    result = run_backtest(signals, cfg)
    perf = compute_performance(result.equity, result.returns, result.trades, cfg.initial_equity)
    doc = render_html(result, perf, params, cfg)
    assert "トレードはありません" in doc
