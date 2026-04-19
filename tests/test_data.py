import pandas as pd
import pytest

from fx.data import load_histdata, resample_ohlc


HIST_SAMPLE_1 = (
    "20230102 170000;130.850;130.870;130.810;130.830;0\n"
    "20230102 170100;130.830;130.890;130.825;130.880;0\n"
    "20230102 170200;130.880;130.900;130.870;130.895;0\n"
)

HIST_SAMPLE_2 = (
    "20230103 170000;131.000;131.020;130.990;131.010;0\n"
    "20230103 170100;131.010;131.050;131.000;131.040;0\n"
)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_load_single_histdata_file(tmp_path):
    f = _write(tmp_path, "DAT_ASCII_USDJPY_M1_2023_01.csv", HIST_SAMPLE_1)
    df = load_histdata(f)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert len(df) == 3
    assert str(df.index.tz) == "UTC"
    # 17:00 EST (UTC-5) == 22:00 UTC
    assert df.index[0] == pd.Timestamp("2023-01-02 22:00:00+00:00")
    assert df["close"].iloc[-1] == pytest.approx(130.895)


def test_load_multiple_histdata_via_glob(tmp_path):
    _write(tmp_path, "DAT_ASCII_USDJPY_M1_2023_01.csv", HIST_SAMPLE_1)
    _write(tmp_path, "DAT_ASCII_USDJPY_M1_2023_02.csv", HIST_SAMPLE_2)
    df = load_histdata(str(tmp_path / "DAT_ASCII_USDJPY_M1_*.csv"))
    assert len(df) == 5
    assert df.index.is_monotonic_increasing


def test_load_histdata_from_directory(tmp_path):
    _write(tmp_path, "DAT_ASCII_USDJPY_M1_2023_01.csv", HIST_SAMPLE_1)
    _write(tmp_path, "DAT_ASCII_USDJPY_M1_2023_02.csv", HIST_SAMPLE_2)
    df = load_histdata(tmp_path)
    assert len(df) == 5


def test_load_histdata_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_histdata(str(tmp_path / "does-not-exist-*.csv"))


def test_resample_m1_to_m5(tmp_path):
    f = _write(tmp_path, "sample.csv", HIST_SAMPLE_1)
    df = load_histdata(f)
    r = resample_ohlc(df, "5min")
    assert len(r) == 1
    row = r.iloc[0]
    assert row["open"] == pytest.approx(130.850)
    assert row["high"] == pytest.approx(130.900)
    assert row["low"] == pytest.approx(130.810)
    assert row["close"] == pytest.approx(130.895)
