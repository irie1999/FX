from datetime import time as dtime

import pandas as pd

from fx.daytrade import DayTradeConfig, apply_daytrade_rules


def _signals(df_index, signals):
    return pd.DataFrame({"signal": signals}, index=df_index)


def test_forces_flat_after_eod():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 15:00", "2024-01-02 20:00",
            "2024-01-02 21:00", "2024-01-02 22:00",
        ],
        tz="UTC",
    )
    sig = _signals(idx, [1, 1, 1, 1])
    out = apply_daytrade_rules(sig, DayTradeConfig(eod_utc=dtime(21, 0)))
    # 21:00 and 22:00 should be flattened
    assert list(out["signal"]) == [1, 1, 0, 0]


def test_pre_eod_unchanged():
    idx = pd.date_range("2024-01-02 12:00", periods=4, freq="1h", tz="UTC")
    sig = _signals(idx, [1, -1, 1, -1])
    out = apply_daytrade_rules(sig, DayTradeConfig(eod_utc=dtime(21, 0)))
    assert list(out["signal"]) == [1, -1, 1, -1]


def test_warmup_per_day():
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 00:00", "2024-01-02 01:00", "2024-01-02 02:00",
            "2024-01-03 00:00", "2024-01-03 01:00", "2024-01-03 02:00",
        ],
        tz="UTC",
    )
    sig = _signals(idx, [1, 1, 1, 1, 1, 1])
    out = apply_daytrade_rules(
        sig, DayTradeConfig(eod_utc=dtime(23, 0), warmup_bars_per_day=2)
    )
    # First 2 bars of each UTC calendar day -> 0
    assert list(out["signal"]) == [0, 0, 1, 0, 0, 1]


def test_requires_tz_aware_index():
    idx = pd.date_range("2024-01-02", periods=3, freq="1h")
    sig = _signals(idx, [1, 1, 1])
    try:
        apply_daytrade_rules(sig)
    except ValueError as e:
        assert "timezone-aware" in str(e)
    else:
        raise AssertionError("expected ValueError for naive index")
