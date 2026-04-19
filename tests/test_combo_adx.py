from fx.backtest import BacktestConfig
from fx.data import synthetic_ohlc
from fx.optimize import Combo, generate_combos, grid_search


def test_combo_default_adx_zero():
    c = Combo(fast=10, slow=30, rsi_period=14, rsi_upper=70, rsi_lower=30, stop_atr=None)
    assert c.adx_threshold == 0.0


def test_generate_combos_sweeps_adx_axis():
    combos = generate_combos(
        fast=[5], slow=[20], rsi_period=[14],
        rsi_upper=[70.0], rsi_lower=[30.0],
        stop_atr=[None],
        adx_threshold=[0.0, 20.0, 25.0],
    )
    assert len(combos) == 3
    assert {c.adx_threshold for c in combos} == {0.0, 20.0, 25.0}


def test_grid_search_emits_adx_threshold_column():
    df = synthetic_ohlc(bars=500, seed=4)
    combos = generate_combos(
        fast=[5], slow=[20], rsi_period=[14],
        rsi_upper=[101.0], rsi_lower=[-1.0],
        stop_atr=[None],
        adx_threshold=[0.0, 25.0],
    )
    cfg = BacktestConfig(size=1000.0, spread=0.02, initial_equity=100_000.0)
    results = grid_search(df, combos, cfg)
    assert "adx_threshold" in results.columns
    assert set(results["adx_threshold"].unique()) == {0.0, 25.0}


def test_combo_describe_omits_adx_when_disabled():
    c = Combo(fast=10, slow=30, rsi_period=14, rsi_upper=70, rsi_lower=30, stop_atr=1.5)
    assert "adx" not in c.describe()


def test_combo_describe_includes_adx_when_enabled():
    c = Combo(fast=10, slow=30, rsi_period=14, rsi_upper=70, rsi_lower=30,
              stop_atr=1.5, adx_threshold=25.0)
    assert "adx>25" in c.describe()
