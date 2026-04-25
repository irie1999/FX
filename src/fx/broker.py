"""Broker interface: paper (simulated) and OANDA v20 (real / demo).

Keep it minimal: enough to place market orders, read the current position,
and fetch recent OHLC candles. stdlib only.

Use paper broker for dry runs and unit tests. Switch to OANDA only when
you actually want to route orders (demo-practice or live).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


# ---------------------------------------------------------------- interfaces


@dataclass
class PriceBar:
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


@dataclass
class Position:
    units: int                      # signed: +ve long, -ve short, 0 flat
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    filled_units: int = 0
    filled_price: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)


class Broker:
    """Abstract broker interface."""

    name: str = "base"

    def get_candles(self, instrument: str, granularity: str, count: int) -> list[PriceBar]:
        raise NotImplementedError

    def get_position(self, instrument: str) -> Position:
        raise NotImplementedError

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_distance: float | None = None,
    ) -> OrderResult:
        """Place a market order. Positive units = buy, negative = sell.

        If `stop_distance` is set (price units, > 0), attach a server-side
        stop-loss at `fill_price ∓ stop_distance` (− for long, + for short).
        Returns OrderResult reflecting fill state.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Override for cleanup (closing HTTP sessions etc)."""
        return None


# ---------------------------------------------------------------- paper broker


class PaperBroker(Broker):
    """In-memory broker that simulates fills on the supplied close prices.

    Useful for:
      - Unit tests
      - Dry runs of the live loop without hitting any external API
    """

    name = "paper"

    def __init__(self, spread: float = 0.02, starting_cash: float = 500_000.0):
        self.spread = spread
        self.cash = starting_cash
        self._positions: dict[str, Position] = {}
        self._last_prices: dict[str, float] = {}
        self._candles: dict[str, list[PriceBar]] = {}
        self._order_counter = 0
        # instrument -> (stop_price, side)  side = +1 long, -1 short
        self._stops: dict[str, tuple[float, int]] = {}

    # helpers to feed data for tests / offline dry runs
    def set_candles(self, instrument: str, candles: list[PriceBar]) -> None:
        self._candles[instrument] = list(candles)
        if candles:
            self._last_prices[instrument] = candles[-1].close

    def set_last_price(self, instrument: str, price: float) -> None:
        self._last_prices[instrument] = price

    def get_candles(self, instrument: str, granularity: str, count: int) -> list[PriceBar]:
        data = self._candles.get(instrument, [])
        return data[-count:]

    def get_position(self, instrument: str) -> Position:
        pos = self._positions.get(instrument, Position(units=0))
        price = self._last_prices.get(instrument, pos.avg_price)
        if pos.units != 0:
            pos.unrealized_pnl = (price - pos.avg_price) * pos.units
        return pos

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_distance: float | None = None,
    ) -> OrderResult:
        if units == 0:
            return OrderResult(success=False, error="units=0")
        price = self._last_prices.get(instrument)
        if price is None:
            return OrderResult(success=False, error="no last price")

        # Apply half-spread cost in the worse direction
        half = self.spread / 2.0
        fill_price = price + half if units > 0 else price - half

        pos = self._positions.get(instrument, Position(units=0))
        if pos.units == 0:
            new_units = units
            new_avg = fill_price
        elif (pos.units > 0 and units > 0) or (pos.units < 0 and units < 0):
            # Adding to an existing position
            total_units = pos.units + units
            if total_units == 0:
                new_units = 0
                new_avg = 0.0
            else:
                new_avg = (pos.avg_price * pos.units + fill_price * units) / total_units
                new_units = total_units
        else:
            # Offsetting / flipping
            if abs(units) <= abs(pos.units):
                # Partial or full close
                realized = (fill_price - pos.avg_price) * (-units if pos.units > 0 else units)
                # Actually realized pnl = direction * (exit - entry) * closed_units
                closed = min(abs(units), abs(pos.units))
                realized = (fill_price - pos.avg_price) * (1 if pos.units > 0 else -1) * closed
                self.cash += realized
                new_units = pos.units + units
                new_avg = pos.avg_price if new_units != 0 else 0.0
            else:
                # Flip through zero
                closed = abs(pos.units)
                realized = (fill_price - pos.avg_price) * (1 if pos.units > 0 else -1) * closed
                self.cash += realized
                # Remaining units open new position in opposite direction
                remaining = units + pos.units
                new_units = remaining
                new_avg = fill_price

        self._positions[instrument] = Position(units=new_units, avg_price=new_avg)

        # Manage attached stop. Closing or flipping invalidates the prior one;
        # a new stop is registered against whatever side the new position takes.
        if new_units == 0:
            self._stops.pop(instrument, None)
        elif stop_distance is not None and stop_distance > 0:
            side = 1 if new_units > 0 else -1
            stop_price = fill_price - side * stop_distance
            self._stops[instrument] = (stop_price, side)
        elif (pos.units > 0) != (new_units > 0) or pos.units == 0:
            # New position (entry or flip) without a stop_distance — clear any
            # stale stop from the previous side.
            self._stops.pop(instrument, None)

        self._order_counter += 1
        return OrderResult(
            success=True,
            order_id=f"paper-{self._order_counter}",
            filled_units=units,
            filled_price=fill_price,
        )

    def check_stops(
        self,
        instrument: str,
        high: float,
        low: float,
    ) -> OrderResult | None:
        """Simulate intrabar stop-out using a bar's high/low.

        Mirrors backtest semantics: a long is stopped if `low <= stop_price`,
        a short if `high >= stop_price`. On trigger, the position is forced
        flat at the stop price (slippage = 0; tightens the simulation versus
        the half-spread fill the real broker would suffer, but is consistent
        with the backtest's `_apply_stops` accounting).
        """
        entry = self._stops.get(instrument)
        if entry is None:
            return None
        stop_price, side = entry
        triggered = (side > 0 and low <= stop_price) or (side < 0 and high >= stop_price)
        if not triggered:
            return None

        pos = self._positions.get(instrument, Position(units=0))
        if pos.units == 0:
            self._stops.pop(instrument, None)
            return None

        closed_units = -pos.units
        realized = (stop_price - pos.avg_price) * (1 if pos.units > 0 else -1) * abs(pos.units)
        self.cash += realized
        self._positions[instrument] = Position(units=0)
        self._stops.pop(instrument, None)
        self._last_prices[instrument] = stop_price
        self._order_counter += 1
        return OrderResult(
            success=True,
            order_id=f"paper-stop-{self._order_counter}",
            filled_units=closed_units,
            filled_price=stop_price,
        )


# ---------------------------------------------------------------- OANDA v20


class OandaBroker(Broker):
    """OANDA v20 REST broker (demo 'practice' or real 'live').

    Minimal implementation using stdlib urllib. Supports candles, position
    query, and market orders. Streaming is not implemented; poll instead.

    Environment variables expected (can be overridden by constructor args):
      OANDA_TOKEN      - API token from https://www.oanda.jp/ (demo or live)
      OANDA_ACCOUNT    - Account ID (xxx-xxx-xxxxxx-xxx)
      OANDA_ENV        - 'practice' (demo, default) or 'live'
    """

    name = "oanda"

    HOSTS = {
        "practice": "https://api-fxpractice.oanda.com",
        "live": "https://api-fxtrade.oanda.com",
    }

    def __init__(
        self,
        token: str | None = None,
        account: str | None = None,
        env: str | None = None,
    ):
        self.token = token or os.environ.get("OANDA_TOKEN", "")
        self.account = account or os.environ.get("OANDA_ACCOUNT", "")
        env_name = env or os.environ.get("OANDA_ENV", "practice")
        if env_name not in self.HOSTS:
            raise ValueError(f"OANDA_ENV must be 'practice' or 'live', got {env_name!r}")
        self.env = env_name
        self.base = self.HOSTS[env_name]

        if not self.token:
            raise ValueError(
                "Missing OANDA_TOKEN. Get a demo token at https://www.oanda.jp/ → "
                "個人向け → デモ口座 → API token."
            )
        if not self.account:
            raise ValueError("Missing OANDA_ACCOUNT.")

    # ---- HTTP helpers ------------------------------------------------
    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | None = None) -> Any:
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OANDA {method} {path} failed: {e.code} {detail}") from None

    # ---- Broker methods ---------------------------------------------
    def get_candles(self, instrument: str, granularity: str, count: int) -> list[PriceBar]:
        """`granularity` uses OANDA codes: M1, M5, M15, M30, H1, H4, D."""
        data = self._request(
            "GET",
            f"/v3/instruments/{instrument}/candles",
            params={
                "granularity": granularity,
                "count": str(count),
                "price": "M",   # mid prices
            },
        )
        bars: list[PriceBar] = []
        for c in data.get("candles", []):
            if not c.get("complete"):
                continue
            mid = c["mid"]
            bars.append(
                PriceBar(
                    time=pd.Timestamp(c["time"], tz="UTC"),
                    open=float(mid["o"]),
                    high=float(mid["h"]),
                    low=float(mid["l"]),
                    close=float(mid["c"]),
                    volume=int(c.get("volume", 0)),
                )
            )
        return bars

    def get_position(self, instrument: str) -> Position:
        data = self._request(
            "GET", f"/v3/accounts/{self.account}/positions/{instrument}"
        )
        pos = data.get("position", {})
        long_units = int(pos.get("long", {}).get("units", 0))
        short_units = int(pos.get("short", {}).get("units", 0))
        if long_units > 0:
            avg = float(pos["long"]["averagePrice"])
            upl = float(pos["long"].get("unrealizedPL", 0))
            return Position(units=long_units, avg_price=avg, unrealized_pnl=upl)
        if short_units < 0:
            avg = float(pos["short"]["averagePrice"])
            upl = float(pos["short"].get("unrealizedPL", 0))
            return Position(units=short_units, avg_price=avg, unrealized_pnl=upl)
        return Position(units=0)

    def market_order(
        self,
        instrument: str,
        units: int,
        stop_distance: float | None = None,
    ) -> OrderResult:
        if units == 0:
            return OrderResult(success=False, error="units=0")
        order: dict[str, Any] = {
            "instrument": instrument,
            "units": str(units),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_distance is not None and stop_distance > 0:
            order["stopLossOnFill"] = {
                "distance": f"{stop_distance:.5f}",
                "timeInForce": "GTC",
            }
        body = {"order": order}
        try:
            data = self._request(
                "POST", f"/v3/accounts/{self.account}/orders", body=body
            )
        except RuntimeError as exc:
            return OrderResult(success=False, error=str(exc))

        fill = data.get("orderFillTransaction")
        if fill is None:
            return OrderResult(
                success=False,
                error=f"no fill: {data}",
                raw=data,
            )
        return OrderResult(
            success=True,
            order_id=str(fill.get("id", "")),
            filled_units=int(fill.get("units", units)),
            filled_price=float(fill.get("price", 0.0)),
            raw=data,
        )


# ---------------------------------------------------------------- factory


def make_broker(kind: str = "paper", **kwargs) -> Broker:
    kind = kind.lower()
    if kind == "paper":
        return PaperBroker(**kwargs)
    if kind == "oanda":
        return OandaBroker(**kwargs)
    raise ValueError(f"Unknown broker kind: {kind!r}")
