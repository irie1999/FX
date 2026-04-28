"""Tests for the recent-data merge helper and fetch_recent symbol mapping."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fetch_recent_cli", ROOT / "tools" / "fetch_recent.py"
)
fr = importlib.util.module_from_spec(spec)
sys.modules["fetch_recent_cli"] = fr
spec.loader.exec_module(fr)

from fx.data import merge_recent


def _bars(start: str, n: int, value: float = 100.0, freq: str = "1D") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": value, "high": value + 0.5, "low": value - 0.5, "close": value},
        index=idx,
    )


def test_merge_recent_appends_after():
    h = _bars("2024-01-01", 5, value=100.0)
    r = _bars("2024-01-06", 3, value=110.0)
    out = merge_recent(h, r)
    assert len(out) == 8
    assert out.index[0] == h.index[0]
    assert out.index[-1] == r.index[-1]
    # Non-overlapping ranges so historical values are preserved
    assert out.loc[h.index[0], "close"] == 100.0
    assert out.loc[r.index[-1], "close"] == 110.0


def test_merge_recent_recent_wins_on_overlap():
    h = _bars("2024-01-01", 5, value=100.0)
    r = _bars("2024-01-04", 3, value=110.0)  # overlaps Jan 4-5, extends Jan 6
    out = merge_recent(h, r)
    assert len(out) == 6
    # Overlapping bars should pick up the recent value
    assert out.loc[r.index[0], "close"] == 110.0
    assert out.loc[r.index[1], "close"] == 110.0


def test_merge_recent_handles_empty():
    h = _bars("2024-01-01", 5)
    empty = h.iloc[:0].copy()
    assert len(merge_recent(h, empty)) == 5
    assert len(merge_recent(empty, h)) == 5


def test_merge_recent_returns_sorted_unique():
    # Out-of-order recent bars should still produce a sorted output
    h = _bars("2024-01-01", 3, value=100.0)
    r = _bars("2024-01-05", 2, value=110.0)
    out = merge_recent(h, r.iloc[::-1])
    assert out.index.is_monotonic_increasing
    assert len(out) == 5


def test_yf_symbol_mapping_known_pairs():
    assert fr.yf_symbol("USDJPY") == "USDJPY=X"
    assert fr.yf_symbol("EURUSD") == "EURUSD=X"
    assert fr.yf_symbol("eurjpy") == "EURJPY=X"


def test_yf_symbol_unknown_falls_back():
    assert fr.yf_symbol("XYZABC") == "XYZABC=X"
