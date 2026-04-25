"""Tests for the day-trade live runner stop-loss wiring.

Covers:
  - PaperBroker stop registration / breach simulation / flip semantics
  - OandaBroker stopLossOnFill payload injection
  - tools/live_trade.py run_once end-to-end with a paper broker
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fx.broker import OandaBroker, PaperBroker, PriceBar
from fx.daytrade import DayTradeConfig
from fx.strategy import StrategyParams

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "live_trade_cli", ROOT / "tools" / "live_trade.py"
)
lt = importlib.util.module_from_spec(spec)
sys.modules["live_trade_cli"] = lt
spec.loader.exec_module(lt)


# ---------------------------------------------------------------- PaperBroker


def test_entry_attaches_stop_long():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    r = br.market_order("USD_JPY", 10_000, stop_distance=0.5)
    assert r.success
    assert br._stops["USD_JPY"] == (pytest.approx(149.5), 1)


def test_entry_attaches_stop_short():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    r = br.market_order("USD_JPY", -10_000, stop_distance=0.4)
    assert r.success
    stop_price, side = br._stops["USD_JPY"]
    assert stop_price == pytest.approx(150.4)
    assert side == -1


def test_no_stop_when_distance_omitted():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000)
    assert "USD_JPY" not in br._stops


def test_no_stop_when_distance_zero():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000, stop_distance=0.0)
    assert "USD_JPY" not in br._stops


def test_intrabar_breach_forces_flat_long():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000, stop_distance=0.5)
    # Bar pierces stop on the low side
    res = br.check_stops("USD_JPY", high=150.7, low=149.0)
    assert res is not None
    assert res.filled_units == -10_000
    assert res.filled_price == pytest.approx(149.5)
    assert br.get_position("USD_JPY").units == 0
    assert "USD_JPY" not in br._stops


def test_intrabar_breach_forces_flat_short():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", -10_000, stop_distance=0.4)
    res = br.check_stops("USD_JPY", high=150.6, low=149.5)
    assert res is not None
    assert res.filled_units == 10_000
    assert res.filled_price == pytest.approx(150.4)
    assert br.get_position("USD_JPY").units == 0


def test_check_stops_no_breach_returns_none():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000, stop_distance=0.5)
    res = br.check_stops("USD_JPY", high=150.6, low=149.7)  # low > 149.5
    assert res is None
    assert br.get_position("USD_JPY").units == 10_000
    assert "USD_JPY" in br._stops


def test_flip_replaces_stop():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000, stop_distance=0.5)
    assert br._stops["USD_JPY"] == (pytest.approx(149.5), 1)

    # Flip long -> short with a fresh stop on the new side
    br.set_last_price("USD_JPY", 151.0)
    br.market_order("USD_JPY", -20_000, stop_distance=0.4)
    stop_price, side = br._stops["USD_JPY"]
    assert side == -1
    assert stop_price == pytest.approx(151.4)


def test_close_clears_stop():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000, stop_distance=0.5)
    br.market_order("USD_JPY", -10_000)
    assert "USD_JPY" not in br._stops


# ---------------------------------------------------------------- OandaBroker


def test_oanda_payload_includes_stoploss(monkeypatch):
    br = OandaBroker(token="x", account="y", env="practice")
    captured: dict = {}

    def fake_request(method, path, params=None, body=None):
        captured["method"] = method
        captured["path"] = path
        captured["body"] = body
        return {
            "orderFillTransaction": {
                "id": "42", "units": "10000", "price": "150.123",
            }
        }

    monkeypatch.setattr(br, "_request", fake_request)
    r = br.market_order("USD_JPY", 10_000, stop_distance=0.25)
    assert r.success
    sof = captured["body"]["order"]["stopLossOnFill"]
    assert sof == {"distance": "0.25000", "timeInForce": "GTC"}


def test_oanda_payload_omits_stoploss_when_none(monkeypatch):
    br = OandaBroker(token="x", account="y", env="practice")
    captured: dict = {}

    def fake_request(method, path, params=None, body=None):
        captured["body"] = body
        return {
            "orderFillTransaction": {
                "id": "1", "units": "10000", "price": "150.0",
            }
        }

    monkeypatch.setattr(br, "_request", fake_request)
    br.market_order("USD_JPY", 10_000)
    assert "stopLossOnFill" not in captured["body"]["order"]


def test_oanda_payload_omits_stoploss_when_zero(monkeypatch):
    br = OandaBroker(token="x", account="y", env="practice")
    captured: dict = {}

    monkeypatch.setattr(
        br, "_request",
        lambda *a, **kw: captured.update(body=kw.get("body") or a[-1])
        or {"orderFillTransaction": {"id": "1", "units": "10000", "price": "150.0"}},
    )
    br.market_order("USD_JPY", 10_000, stop_distance=0.0)
    assert "stopLossOnFill" not in captured["body"]["order"]


# ---------------------------------------------------------------- run_once


def _trending_candles(n: int, start: float = 150.0, step: float = 0.05) -> list[PriceBar]:
    """Synthetic monotonically rising OHLC; guarantees a long signal once
    fast SMA > slow SMA and the RSI/ADX filters pass."""
    bars: list[PriceBar] = []
    base = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n):
        c = start + step * i
        bars.append(
            PriceBar(
                time=base + pd.Timedelta(minutes=15 * i),
                open=c - step / 4,
                high=c + step / 4,
                low=c - step / 2,
                close=c,
            )
        )
    return bars


def _make_runner_inputs():
    # rsi_upper > 100 so the strict-rise synthetic data (RSI=100) clears the
    # filter; keep adx_threshold=0 (default) so ADX warmup doesn't matter.
    params = StrategyParams(fast=3, slow=5, rsi_period=3, rsi_upper=101.0, rsi_lower=-1.0)
    dt_config = DayTradeConfig()  # 21:00 UTC EOD
    return params, dt_config


def test_run_once_attaches_stop_paper():
    br = PaperBroker(spread=0.0)
    br.set_candles("USD_JPY", _trending_candles(40))
    params, dt_config = _make_runner_inputs()
    logger = logging.getLogger("test.live")

    lt.run_once(
        broker=br,
        instrument="USD_JPY",
        granularity="M15",
        params=params,
        size=10_000,
        dt_config=dt_config,
        logger=logger,
        dry_run=False,
        lookback=200,
        stop_atr=1.5,
        min_stop_distance=0.0,
    )
    pos = br.get_position("USD_JPY")
    assert pos.units == 10_000
    assert "USD_JPY" in br._stops
    stop_price, side = br._stops["USD_JPY"]
    assert side == 1
    assert stop_price < pos.avg_price


def test_run_once_no_stop_when_disabled():
    br = PaperBroker(spread=0.0)
    br.set_candles("USD_JPY", _trending_candles(40))
    params, dt_config = _make_runner_inputs()

    lt.run_once(
        broker=br,
        instrument="USD_JPY",
        granularity="M15",
        params=params,
        size=10_000,
        dt_config=dt_config,
        logger=logging.getLogger("test.live"),
        dry_run=False,
        lookback=200,
        stop_atr=0.0,
        min_stop_distance=0.0,
    )
    assert br.get_position("USD_JPY").units == 10_000
    assert "USD_JPY" not in br._stops


def test_run_once_skips_entry_when_atr_not_ready(monkeypatch):
    """If ATR is NaN/0 on the latest bar but the signal is non-zero, refuse
    to open an unprotected position when --stop-atr is enabled."""
    br = PaperBroker(spread=0.0)
    br.set_candles("USD_JPY", _trending_candles(40))
    params, dt_config = _make_runner_inputs()

    real_gen = lt.generate_signals

    def fake_generate(df, p):
        out = real_gen(df, p)
        out.loc[out.index[-1], "atr"] = float("nan")
        out.loc[out.index[-1], "signal"] = 1
        return out

    monkeypatch.setattr(lt, "generate_signals", fake_generate)

    lt.run_once(
        broker=br,
        instrument="USD_JPY",
        granularity="M15",
        params=params,
        size=10_000,
        dt_config=dt_config,
        logger=logging.getLogger("test.live"),
        dry_run=False,
        lookback=200,
        stop_atr=1.5,
        min_stop_distance=0.0,
    )
    assert br.get_position("USD_JPY").units == 0
    assert "USD_JPY" not in br._stops


def test_run_once_paper_simulates_breach():
    """If the latest bar's low pierces the registered stop, the paper broker
    is forced flat before the new signal is processed."""
    br = PaperBroker(spread=0.0)
    base_candles = _trending_candles(40)
    br.set_candles("USD_JPY", base_candles)
    params, dt_config = _make_runner_inputs()

    # First iteration: opens long with a stop.
    lt.run_once(
        broker=br,
        instrument="USD_JPY",
        granularity="M15",
        params=params,
        size=10_000,
        dt_config=dt_config,
        logger=logging.getLogger("test.live"),
        dry_run=False,
        lookback=200,
        stop_atr=1.5,
        min_stop_distance=0.0,
    )
    stop_price, _ = br._stops["USD_JPY"]
    last_close = base_candles[-1].close

    # Append a bar whose low pierces the stop; close stays elevated to keep
    # the strategy signal long (so any re-entry attempt would be visible).
    breach_bar = PriceBar(
        time=base_candles[-1].time + pd.Timedelta(minutes=15),
        open=last_close,
        high=last_close + 0.05,
        low=stop_price - 0.02,
        close=last_close,
    )
    br.set_candles("USD_JPY", base_candles + [breach_bar])

    lt.run_once(
        broker=br,
        instrument="USD_JPY",
        granularity="M15",
        params=params,
        size=10_000,
        dt_config=dt_config,
        logger=logging.getLogger("test.live"),
        dry_run=False,
        lookback=200,
        stop_atr=1.5,
        min_stop_distance=0.0,
    )
    # Stop fires, then the still-long signal re-enters on the next bar.
    # We mainly assert the stop fired (cash advanced from realized loss).
    assert br.cash < 500_000.0  # starting cash, lost on the stop-out
