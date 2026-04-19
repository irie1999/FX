"""Live / paper trading runner.

Loop:
  1. Sleep until near the next candle close
  2. Fetch recent candles from the broker
  3. Run signal generator + day-trade EOD close
  4. Compute desired position (signed units)
  5. Reconcile against current broker position; place market orders as needed
  6. Log everything
  7. Repeat

Default broker is `paper` (no real money). Switch to OANDA with `--broker oanda`
AND only then set `--live` to route real orders on your OANDA live account.
If `--broker oanda` is used without `--live`, the OANDA practice (demo) env
is selected, which is itself safe (no real money).

Setup:
  1. Open a free demo account at https://www.oanda.jp/
  2. Generate an API token from the account console
  3. Set env vars:
        OANDA_TOKEN=...
        OANDA_ACCOUNT=101-001-xxxxxxx-001
  4. python tools/live_trade.py --broker oanda --instrument USD_JPY --granularity M15
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal as _signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd                                           # noqa: E402

from fx.broker import Broker, OandaBroker, PaperBroker, PriceBar, make_broker  # noqa: E402
from fx.daytrade import DayTradeConfig, apply_daytrade_rules  # noqa: E402
from fx.strategy import StrategyParams, generate_signals      # noqa: E402


GRANULARITY_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D": 86400,
}


def _setup_logging(log_file: Path | None) -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")
    logger = logging.getLogger("fx.live")
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
    return logger


def candles_to_df(candles: list[PriceBar]) -> pd.DataFrame:
    rows = [
        {"time": c.time, "open": c.open, "high": c.high, "low": c.low, "close": c.close}
        for c in candles
    ]
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df


def seconds_until_next_bar(gran_seconds: int) -> float:
    """Wait until ~2s after the next bar close aligned to UTC wall-clock."""
    now = time.time()
    period_start = (now // gran_seconds) * gran_seconds
    next_close = period_start + gran_seconds
    delay = max(0.0, next_close + 2.0 - now)
    return delay


def reconcile_position(
    broker: Broker,
    instrument: str,
    desired_units: int,
    logger: logging.Logger,
    dry_run: bool = False,
) -> None:
    pos = broker.get_position(instrument)
    delta = desired_units - pos.units
    if delta == 0:
        logger.info("position aligned (units=%d) — no order", pos.units)
        return
    verb = "BUY" if delta > 0 else "SELL"
    logger.info(
        "reconcile: current=%d desired=%d -> %s %d",
        pos.units, desired_units, verb, abs(delta),
    )
    if dry_run:
        logger.info("[dry-run] skipping order submission")
        return
    result = broker.market_order(instrument, delta)
    if result.success:
        logger.info(
            "filled order id=%s units=%+d @ %.5f",
            result.order_id, result.filled_units, result.filled_price,
        )
    else:
        logger.error("order failed: %s", result.error)


def run_once(
    broker: Broker,
    instrument: str,
    granularity: str,
    params: StrategyParams,
    size: int,
    dt_config: DayTradeConfig,
    logger: logging.Logger,
    dry_run: bool,
    lookback: int,
) -> None:
    logger.info("fetching %d %s candles for %s…", lookback, granularity, instrument)
    candles = broker.get_candles(instrument, granularity, lookback)
    if not candles:
        logger.warning("no candles returned; skipping")
        return
    df = candles_to_df(candles)
    last_bar = df.index[-1]
    last_close = df["close"].iloc[-1]

    signals = generate_signals(df, params)
    signals = apply_daytrade_rules(signals, dt_config)

    last_signal = int(signals["signal"].iloc[-1])
    desired_units = last_signal * size

    logger.info(
        "bar %s  close=%.5f  signal=%+d  desired_units=%+d",
        last_bar, last_close, last_signal, desired_units,
    )
    reconcile_position(broker, instrument, desired_units, logger, dry_run)


# ----------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FX live/paper trading runner")

    p.add_argument(
        "--broker",
        choices=("paper", "oanda"),
        default="paper",
        help="Broker backend (default: paper, no real money)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="(OANDA only) route orders to the LIVE production account. "
             "Requires --broker oanda. Without --live, the practice (demo) "
             "environment is used.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals but don't place any orders (even on paper).",
    )

    p.add_argument("--instrument", default="USD_JPY", help="OANDA-style instrument code, e.g. USD_JPY")
    p.add_argument("--granularity", default="M15", choices=tuple(GRANULARITY_SECONDS),
                   help="Candle granularity (default M15)")
    p.add_argument("--size", type=int, default=10_000, help="Trade size in units (abs value)")
    p.add_argument("--lookback", type=int, default=200,
                   help="Number of recent candles to fetch per iteration")

    p.add_argument("--fast", type=int, default=5)
    p.add_argument("--slow", type=int, default=20)
    p.add_argument("--rsi", type=int, default=14)
    p.add_argument("--rsi-upper", type=float, default=70.0)
    p.add_argument("--rsi-lower", type=float, default=30.0)
    p.add_argument("--adx-threshold", type=float, default=0.0)

    p.add_argument("--eod-utc", default="21:00",
                   help="End-of-day close time (UTC, HH:MM). Positions go flat from this time.")
    p.add_argument("--once", action="store_true",
                   help="Run a single iteration and exit (useful for cron / diagnostics).")
    p.add_argument("--log-file", type=Path, default=Path("logs/live.log"))

    return p


def _parse_time(s: str):
    from datetime import time as dtime
    h, m = s.split(":")
    return dtime(int(h), int(m))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.live and args.broker != "oanda":
        print("--live requires --broker oanda", file=sys.stderr)
        return 2

    logger = _setup_logging(args.log_file)

    if args.broker == "oanda":
        env = "live" if args.live else "practice"
        os.environ.setdefault("OANDA_ENV", env)
        try:
            broker: Broker = OandaBroker(env=env)
        except Exception as exc:
            logger.error("OANDA init failed: %s", exc)
            return 1
        mode = f"OANDA {env}"
    else:
        broker = PaperBroker()
        mode = "paper"

    params = StrategyParams(
        fast=args.fast, slow=args.slow,
        rsi_period=args.rsi,
        rsi_upper=args.rsi_upper, rsi_lower=args.rsi_lower,
        adx_threshold=args.adx_threshold,
    )
    dt_config = DayTradeConfig(eod_utc=_parse_time(args.eod_utc))

    logger.info("=" * 60)
    logger.info("fx.live_trade starting")
    logger.info("mode          : %s%s", mode, " [DRY-RUN]" if args.dry_run else "")
    logger.info("instrument    : %s", args.instrument)
    logger.info("granularity   : %s", args.granularity)
    logger.info("size          : %d units", args.size)
    logger.info("params        : fast=%d slow=%d rsi=%d", params.fast, params.slow, params.rsi_period)
    logger.info("EOD cutoff    : %s UTC", args.eod_utc)
    logger.info("=" * 60)

    gran_seconds = GRANULARITY_SECONDS[args.granularity]
    stop = {"flag": False}

    def _sigint(*_):
        logger.info("SIGINT received — stopping after this iteration")
        stop["flag"] = True

    _signal.signal(_signal.SIGINT, _sigint)

    try:
        while not stop["flag"]:
            try:
                run_once(
                    broker=broker,
                    instrument=args.instrument,
                    granularity=args.granularity,
                    params=params,
                    size=args.size,
                    dt_config=dt_config,
                    logger=logger,
                    dry_run=args.dry_run,
                    lookback=args.lookback,
                )
            except Exception as exc:
                logger.exception("iteration failed: %s", exc)

            if args.once:
                break
            delay = seconds_until_next_bar(gran_seconds)
            logger.info("sleeping %.0fs until next %s close…", delay, args.granularity)
            # Sleep in short chunks so SIGINT is responsive
            slept = 0.0
            while slept < delay and not stop["flag"]:
                time.sleep(min(1.0, delay - slept))
                slept += 1.0
    finally:
        broker.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
