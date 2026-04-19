"""CLI entrypoint: python -m fx.main ..."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as data_mod
from .backtest import BacktestConfig, run_backtest
from .metrics import compute_performance, format_performance
from .strategy import StrategyParams, generate_signals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FX SMA+RSI strategy backtest")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to OHLC CSV (timestamp,open,high,low,close)")
    src.add_argument("--synthetic", action="store_true", help="Use synthetic OHLC data")

    p.add_argument("--bars", type=int, default=5000, help="Synthetic bars (default 5000)")
    p.add_argument("--seed", type=int, default=42, help="Synthetic data seed")

    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--rsi", type=int, default=14)
    p.add_argument("--rsi-upper", type=float, default=70.0)
    p.add_argument("--rsi-lower", type=float, default=30.0)

    p.add_argument("--size", type=float, default=10_000.0, help="Units per trade")
    p.add_argument("--spread", type=float, default=0.02, help="Spread in price units")
    p.add_argument("--equity", type=float, default=1_000_000.0, help="Initial equity")

    p.add_argument("--save-equity", type=Path, help="Optional CSV path to save equity curve")
    p.add_argument("--save-trades", type=Path, help="Optional CSV path to save trades")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.synthetic:
        df = data_mod.synthetic_ohlc(bars=args.bars, seed=args.seed)
    else:
        df = data_mod.load_csv(args.csv)

    params = StrategyParams(
        fast=args.fast,
        slow=args.slow,
        rsi_period=args.rsi,
        rsi_upper=args.rsi_upper,
        rsi_lower=args.rsi_lower,
    )
    signals = generate_signals(df, params)

    cfg = BacktestConfig(size=args.size, spread=args.spread, initial_equity=args.equity)
    result = run_backtest(signals, cfg)

    perf = compute_performance(result.equity, result.returns, result.trades, cfg.initial_equity)

    print(f"Bars       : {len(df)}")
    print(f"Period     : {df.index[0]} -> {df.index[-1]}")
    print(f"Params     : fast={params.fast} slow={params.slow} rsi={params.rsi_period}")
    print(f"Spread     : {cfg.spread}  Size: {cfg.size}  Equity0: {cfg.initial_equity:.0f}")
    print("-" * 40)
    print(format_performance(perf))

    if args.save_equity:
        result.equity.to_csv(args.save_equity, header=["equity"])
    if args.save_trades:
        result.trades.to_csv(args.save_trades, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
