import pandas as pd
import pytest

from fx.broker import OandaBroker, OrderResult, PaperBroker, Position, PriceBar, make_broker


def test_paper_broker_buy_then_sell():
    br = PaperBroker(spread=0.02, starting_cash=100_000.0)
    br.set_last_price("USD_JPY", 150.0)

    r1 = br.market_order("USD_JPY", 10_000)
    assert r1.success and r1.filled_units == 10_000
    assert br.get_position("USD_JPY").units == 10_000
    # Buy fills at mid + half-spread
    assert br.get_position("USD_JPY").avg_price == pytest.approx(150.01)

    br.set_last_price("USD_JPY", 151.0)
    # Unrealized PnL = (151 - 150.01) * 10000 = 9900
    pos = br.get_position("USD_JPY")
    assert pos.unrealized_pnl == pytest.approx(9900.0)

    r2 = br.market_order("USD_JPY", -10_000)
    assert r2.success
    assert br.get_position("USD_JPY").units == 0


def test_paper_broker_flip_long_to_short():
    br = PaperBroker(spread=0.0)
    br.set_last_price("USD_JPY", 150.0)
    br.market_order("USD_JPY", 10_000)
    # Flip by selling 20k: 10k closes long, 10k opens short
    br.set_last_price("USD_JPY", 151.0)
    r = br.market_order("USD_JPY", -20_000)
    assert r.success
    pos = br.get_position("USD_JPY")
    assert pos.units == -10_000
    assert pos.avg_price == pytest.approx(151.0)


def test_paper_broker_rejects_zero_order():
    br = PaperBroker()
    br.set_last_price("USD_JPY", 150.0)
    r = br.market_order("USD_JPY", 0)
    assert not r.success


def test_paper_broker_candles():
    br = PaperBroker()
    bars = [
        PriceBar(pd.Timestamp("2024-01-01", tz="UTC"), 150.0, 150.5, 149.5, 150.2),
        PriceBar(pd.Timestamp("2024-01-02", tz="UTC"), 150.2, 150.8, 150.1, 150.6),
    ]
    br.set_candles("USD_JPY", bars)
    got = br.get_candles("USD_JPY", "D", 5)
    assert len(got) == 2


def test_make_broker_factory():
    assert isinstance(make_broker("paper"), PaperBroker)


def test_oanda_broker_requires_token(monkeypatch):
    monkeypatch.delenv("OANDA_TOKEN", raising=False)
    monkeypatch.delenv("OANDA_ACCOUNT", raising=False)
    with pytest.raises(ValueError, match="OANDA_TOKEN"):
        OandaBroker(env="practice")


def test_oanda_broker_invalid_env():
    with pytest.raises(ValueError, match="practice"):
        OandaBroker(token="x", account="y", env="staging")
