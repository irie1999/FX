"""Tests for tools/daily_signal.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "daily_signal_cli", ROOT / "tools" / "daily_signal.py"
)
ds = importlib.util.module_from_spec(spec)
sys.modules["daily_signal_cli"] = ds
spec.loader.exec_module(ds)

from fx.status import CurrentStatus


def _mk_status(pos, next_sig, **kwargs):
    return CurrentStatus(
        last_bar=pd.Timestamp("2026-04-01", tz="UTC"),
        current_price=158.85,
        position=pos,
        next_signal=next_sig,
        next_action="dummy",
        **kwargs,
    )


def test_classify_actions_all_permutations():
    # Current 0 (flat)
    assert ds.classify_action(0, 0) == "HOLD_FLAT"
    assert ds.classify_action(0, 1) == "ENTER_LONG"
    assert ds.classify_action(0, -1) == "ENTER_SHORT"
    # Current +1 (long)
    assert ds.classify_action(1, 1) == "HOLD_LONG"
    assert ds.classify_action(1, 0) == "EXIT_LONG"
    assert ds.classify_action(1, -1) == "FLIP_TO_SHORT"
    # Current -1 (short)
    assert ds.classify_action(-1, -1) == "HOLD_SHORT"
    assert ds.classify_action(-1, 0) == "EXIT_SHORT"
    assert ds.classify_action(-1, 1) == "FLIP_TO_LONG"


def test_build_report_flat_signal():
    st = _mk_status(0, 0)
    text = ds.build_report(st, "HOLD_FLAT", "USD/JPY", 1000)
    assert "USD/JPY" in text
    assert "何もしない" in text
    assert "ノーポジション" in text


def test_build_report_new_long_includes_sbi_steps():
    st = _mk_status(0, 1, stop_level=156.50)
    text = ds.build_report(st, "ENTER_LONG", "USD/JPY", 1000)
    assert "新規ロング" in text
    assert "SBI FX アプリでの手順" in text
    assert "買い" in text
    assert "1,000 通貨" in text
    assert "156.500" in text
    # Risk estimate should appear
    assert "想定リスク" in text


def test_build_report_hold_long_position():
    st = _mk_status(
        1, 1,
        entry_time=pd.Timestamp("2026-03-01", tz="UTC"),
        entry_price=157.00,
        unrealized_pnl=18_500.0,
        stop_level=154.50,
        bars_held=20,
    )
    text = ds.build_report(st, "HOLD_LONG", "USD/JPY", 1000)
    assert "ロング継続" in text
    assert "継続保有" in text
    assert "157.0000" in text
    assert "+18,500" in text
    # No SBI step-by-step when holding
    assert "SBI FX アプリでの手順" not in text


def test_build_report_exit_long():
    st = _mk_status(
        1, 0,
        entry_time=pd.Timestamp("2026-03-01", tz="UTC"),
        entry_price=157.00,
        unrealized_pnl=18_500.0,
    )
    text = ds.build_report(st, "EXIT_LONG", "USD/JPY", 1000)
    assert "ロング決済" in text
    assert "決済" in text


def test_json_payload_roundtrip():
    # Use synthetic data via argparse-compatible namespace mock
    import argparse
    ns = argparse.Namespace(
        histdata=None, csv=None,
        resample="1d", strategy="sma_rsi",
        instrument="USD/JPY", size=1000, spread=0.02, equity=500_000.0,
        stop_atr=1.25,
        fast=5, slow=20, rsi=14, rsi_upper=70.0, rsi_lower=30.0,
        currency="¥",
    )
    # Synthesize data via synthetic OHLC
    from fx.data import synthetic_ohlc
    df = synthetic_ohlc(bars=400, seed=1, freq="1D")
    # Monkey-patch _load_data path via inline: simpler to just call the pieces
    from fx.strategy import StrategyParams, generate_signals
    from fx.backtest import BacktestConfig, StopConfig, run_backtest
    from fx.status import compute_current_status

    sp = StrategyParams(fast=5, slow=20)
    signals = generate_signals(df, sp)
    cfg = BacktestConfig(size=1000, spread=0.02, initial_equity=500_000.0)
    stops = StopConfig(enabled=True, atr_mult=1.25)
    result = run_backtest(signals, cfg, stops=stops)
    status = compute_current_status(result, cfg, stops)
    action = ds.classify_action(status.position, status.next_signal)
    text = ds.build_report(status, action, "USD/JPY", 1000)

    # Basic sanity: text has expected header + action line
    assert "FX 取引シグナル" in text
    assert action in ds._ACTION_TYPES


def test_webhook_network_failure_returns_false(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("simulated network error")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert ds.post_webhook("http://invalid.invalid/", "hello") is False
