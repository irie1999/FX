"""Fetch recent FX OHLC data via yfinance.

HistData publishes only completed months, so to get the most recent
days (e.g. for tonight's signal) we top up with Yahoo Finance daily
bars. The resulting CSV is in the same shape as our other CSVs:

    timestamp,open,high,low,close

with a UTC tz-aware index. Combine with HistData via
`daily_signal.py --extra-csv` (or set --csv directly).

Requirements:
    pip install yfinance

Usage:
    python tools/fetch_recent.py --pair USDJPY --period 60d
    python tools/fetch_recent.py --pair EURUSD --period 1y --interval 1d
    # Save to a custom path:
    python tools/fetch_recent.py --pair GBPUSD --out data/recent/GBPUSD.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Yahoo Finance ticker symbols for FX pairs (=X suffix for major FX).
YF_SYMBOLS: dict[str, str] = {
    "USDJPY": "USDJPY=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "NZDJPY": "NZDJPY=X",
    "CADJPY": "CADJPY=X",
    "CHFJPY": "CHFJPY=X",
    "EURGBP": "EURGBP=X",
    "EURAUD": "EURAUD=X",
}


def yf_symbol(pair: str) -> str:
    """Map our internal pair code (e.g. USDJPY) to a Yahoo ticker."""
    return YF_SYMBOLS.get(pair.upper(), f"{pair.upper()}=X")


def fetch_yfinance(pair: str, period: str = "60d", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLC daily bars from Yahoo Finance for the given pair."""
    try:
        import yfinance as yf
    except ImportError:
        print(
            "yfinance is not installed. Install it with:  pip install yfinance",
            file=sys.stderr,
        )
        sys.exit(1)

    symbol = yf_symbol(pair)
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    # yfinance >=0.2.40 returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    df = df[["open", "high", "low", "close"]].copy().astype(float)
    df.index.name = "timestamp"
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch recent FX data from Yahoo Finance")
    p.add_argument("--pair", required=True, help="FX pair, e.g. USDJPY, EURUSD, GBPUSD")
    p.add_argument(
        "--period", default="60d",
        help="yfinance period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default 60d)",
    )
    p.add_argument(
        "--interval", default="1d",
        help="yfinance interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo (default 1d)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path (default: data/recent/<PAIR>.csv)",
    )
    args = p.parse_args(argv)

    if args.out is None:
        args.out = Path("data/recent") / f"{args.pair.upper()}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_yfinance(args.pair, args.period, args.interval)
    df.to_csv(args.out)
    print(
        f"Saved {len(df)} {args.interval} bars for {args.pair.upper()} "
        f"({df.index[0]} -> {df.index[-1]}) to {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
