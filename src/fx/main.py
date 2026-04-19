"""CLI entrypoint: python -m fx.main ..."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data as data_mod
from . import strategies as strategies_pkg
from .backtest import BacktestConfig, BacktestResult, StopConfig, run_backtest
from .metrics import compute_performance, format_performance
from .status import compute_current_status, format_status_console
from .strategy import StrategyParams, generate_signals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FX SMA+RSI strategy backtest")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to OHLC CSV (timestamp,open,high,low,close)")
    src.add_argument(
        "--histdata",
        nargs="+",
        help=(
            "HistData Generic ASCII M1 files. Accepts one or many paths, glob "
            "patterns, or directories. e.g. "
            'C:\\Users\\you\\Downloads\\DAT_ASCII_USDJPY_M1_*.csv'
        ),
    )
    src.add_argument("--synthetic", action="store_true", help="Use synthetic OHLC data")

    p.add_argument(
        "--resample",
        default=None,
        help="Resample loaded data to this pandas rule (e.g. '5min','1h','1d'). Useful for M1 sources.",
    )
    p.add_argument(
        "--start",
        default=None,
        help="Keep bars with timestamp >= START (inclusive). Format: YYYY-MM-DD or ISO 8601.",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Keep bars with timestamp <= END (inclusive). Format: YYYY-MM-DD or ISO 8601.",
    )
    p.add_argument("--bars", type=int, default=5000, help="Synthetic bars (default 5000)")
    p.add_argument("--seed", type=int, default=42, help="Synthetic data seed")

    p.add_argument(
        "--strategy",
        default="sma_rsi",
        choices=strategies_pkg.names(),
        help="Strategy to run (default: sma_rsi). --fast/--slow/--rsi only apply to sma_rsi.",
    )
    p.add_argument(
        "--eval-days",
        type=int,
        default=None,
        help="Only evaluate performance on the last N days (the strategy still "
             "warms up on all prior data). Useful for checking recent behavior.",
    )

    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--rsi", type=int, default=14)
    p.add_argument("--rsi-upper", type=float, default=70.0)
    p.add_argument("--rsi-lower", type=float, default=30.0)
    p.add_argument("--adx-period", type=int, default=14)
    p.add_argument(
        "--adx-threshold", type=float, default=0.0,
        help="Only enter when ADX > threshold (trend-strength filter). 0 = disabled.",
    )

    p.add_argument("--size", type=float, default=10_000.0, help="Units per trade")
    p.add_argument("--spread", type=float, default=0.02, help="Spread in price units")
    p.add_argument("--equity", type=float, default=500_000.0, help="Initial equity (default 500,000)")

    p.add_argument(
        "--stop-atr",
        type=float,
        default=None,
        metavar="N",
        help="Enable ATR-based stop-loss at N * ATR(rsi_period) distance from entry. "
             "e.g. --stop-atr 2 uses a 2-ATR stop. Omit to disable.",
    )

    p.add_argument("--save-equity", type=Path, help="Optional CSV path to save equity curve")
    p.add_argument("--save-trades", type=Path, help="Optional CSV path to save trades")
    p.add_argument(
        "--html",
        type=Path,
        nargs="?",
        const=Path("results/report.html"),
        help="Write an HTML report to PATH (default: results/report.html)",
    )
    p.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the HTML report in a browser after writing (default: on). Use --no-open to disable.",
    )
    p.add_argument("--title", default="FX バックテストレポート", help="Title in the HTML report")
    p.add_argument(
        "--currency", default="¥",
        help="Currency symbol shown in the HTML (default ¥). Use '$' for EURUSD etc.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.synthetic:
        df = data_mod.synthetic_ohlc(bars=args.bars, seed=args.seed)
    elif args.histdata:
        df = data_mod.load_histdata(args.histdata)
    else:
        df = data_mod.load_csv(args.csv)

    if args.resample:
        df = data_mod.resample_ohlc(df, args.resample)

    # Slice by date range (after resample so boundaries match aggregated bars)
    if args.start or args.end:
        df = data_mod.slice_period(df, args.start, args.end)

    if len(df) == 0:
        print("No bars after applying --start/--end filter.")
        return 1

    # Keep StrategyParams for HTML/backcompat; only sma_rsi actually honors them.
    params = StrategyParams(
        fast=args.fast,
        slow=args.slow,
        rsi_period=args.rsi,
        rsi_upper=args.rsi_upper,
        rsi_lower=args.rsi_lower,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
    )

    if args.strategy == "sma_rsi":
        signals = generate_signals(df, params)
        strategy_display = "SMA クロス + RSI"
    else:
        strat_mod = strategies_pkg.get(args.strategy)
        signals = strat_mod.generate(df)
        strategy_display = strat_mod.DISPLAY

    cfg = BacktestConfig(size=args.size, spread=args.spread, initial_equity=args.equity)
    stops = StopConfig(enabled=args.stop_atr is not None, atr_mult=args.stop_atr or 0.0)
    result = run_backtest(signals, cfg, stops=stops)

    # Optionally restrict evaluation to the last N days. The strategy already
    # consumed its full warmup above, so slicing here just hides earlier bars
    # from metrics / report / HTML.
    if args.eval_days is not None:
        end = result.equity.index[-1]
        start = end - pd.Timedelta(days=args.eval_days)
        mask = result.equity.index >= start
        if not mask.any():
            print(f"No bars within the last {args.eval_days} days.")
            return 1

        eq_slice = result.equity[mask]
        start_eq = float(eq_slice.iloc[0])
        adjusted_equity = cfg.initial_equity + (eq_slice - start_eq)
        sliced_returns = result.returns[mask]
        sliced_position = result.position[mask]
        sliced_signals = result.signals.loc[eq_slice.index]
        if len(result.trades) > 0 and "exit_time" in result.trades.columns:
            sliced_trades = result.trades[result.trades["exit_time"] >= start].reset_index(drop=True)
        else:
            sliced_trades = result.trades

        result = BacktestResult(
            equity=adjusted_equity,
            returns=sliced_returns,
            position=sliced_position,
            trades=sliced_trades,
            signals=sliced_signals,
        )

    perf = compute_performance(
        result.equity,
        result.returns,
        result.trades,
        cfg.initial_equity,
        position=result.position,
    )

    print(f"Bars       : {len(result.equity)}" + (f"  (last {args.eval_days}d)" if args.eval_days else ""))
    print(f"Period     : {result.equity.index[0]} -> {result.equity.index[-1]}")
    print(f"Strategy   : {strategy_display}")
    if args.strategy == "sma_rsi":
        print(f"Params     : fast={params.fast} slow={params.slow} rsi={params.rsi_period}")
    stop_txt = f"{stops.atr_mult}*ATR" if stops.enabled else "off"
    print(f"Spread     : {cfg.spread}  Size: {cfg.size}  Equity0: {cfg.initial_equity:.0f}  Stop: {stop_txt}")
    print("-" * 40)
    print(format_performance(perf))

    status = compute_current_status(result, cfg, stops)
    print()
    print(format_status_console(status, currency=args.currency))

    if args.save_equity:
        result.equity.to_csv(args.save_equity, header=["equity"])
    if args.save_trades:
        result.trades.to_csv(args.save_trades, index=False)

    if args.html:
        # Lazy import so matplotlib isn't required unless the user wants HTML.
        from .report import write_html

        out = write_html(
            args.html, result, perf, params, cfg,
            title=args.title, stops=stops, currency=args.currency,
            status=status,
        )
        print(f"HTML report: {out.resolve()}")
        if args.open:
            import webbrowser

            webbrowser.open(out.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
