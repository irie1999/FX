"""Pairs / statistical-arbitrage backtester (cointegrated FX pairs).

Usage:
    python tools/pairs_trading.py \\
        --histdata-dir data/raw \\
        --pair-a EURUSD --pair-b GBPUSD \\
        --resample 1d --size 1000 \\
        --html results/pairs_eurgbp.html

This complements multi_pair_backtest.py:
  * multi_pair_backtest = same strategy on N pairs in parallel
  * pairs_trading       = ONE strategy that trades 2 pairs *together*

Defaults to a z-score mean-reversion rule on the log spread (Avellaneda
& Lee 2010 style). Output reports the spread, z-score, position track,
per-leg PnL and a stitched equity curve.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd                                   # noqa: E402

from fx import data as data_mod                       # noqa: E402
from fx.metrics import compute_performance            # noqa: E402
from fx.pairs import PairsParams, run_pairs_backtest  # noqa: E402

# Re-use spread & JPY-conversion defaults from multi_pair_backtest.
import importlib.util
_mp_spec = importlib.util.spec_from_file_location(
    "multi_pair_helpers", ROOT / "tools" / "multi_pair_backtest.py"
)
_mp = importlib.util.module_from_spec(_mp_spec)
# Register in sys.modules so dataclasses inside that module can resolve
# their own __module__ during decoration.
sys.modules["multi_pair_helpers"] = _mp
_mp_spec.loader.exec_module(_mp)


def _find_files(histdata_dir: Path, pair: str, extra_dir: Path | None) -> tuple[list[Path], Path | None]:
    pattern = str(histdata_dir / f"DAT_ASCII_{pair.upper()}_M1_*.csv")
    files = sorted(Path(p) for p in glob(pattern))
    extra_csv = extra_dir / f"{pair.upper()}.csv" if extra_dir else None
    if extra_csv is not None and not extra_csv.exists():
        extra_csv = None
    return files, extra_csv


def _load_pair(
    pair: str, histdata_dir: Path, resample: str, extra_dir: Path | None
) -> pd.DataFrame:
    files, extra_csv = _find_files(histdata_dir, pair, extra_dir)
    if not files:
        raise FileNotFoundError(
            f"No HistData files for {pair} in {histdata_dir}"
        )
    df = data_mod.load_histdata(files)
    if resample:
        df = data_mod.resample_ohlc(df, resample)
    if extra_csv is not None:
        recent = data_mod.load_csv(extra_csv)
        if resample:
            recent = data_mod.resample_ohlc(recent, resample)
        df = data_mod.merge_recent(df, recent)
    return df


# ---------------------------------------------------------------- HTML


_CSS = """
:root {
  --bg:#0e1117; --panel:#161b22; --panel-2:#1c222c; --border:#2a313c;
  --text:#e6edf3; --muted:#8b949e; --pos:#3fb950; --neg:#f85149; --accent:#58a6ff;
}
html, body { background: var(--bg); color: var(--text); }
body {
  font-family: "Yu Gothic UI","Yu Gothic","Meiryo","Hiragino Sans","Noto Sans CJK JP",
               -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1300px; margin: 2rem auto; padding: 0 1rem;
}
h1 { margin-bottom: 0.2rem; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
.muted { color: var(--muted); font-size: 0.9em; }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
.card { background: var(--panel); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.8rem 1rem; }
.card .label { color: var(--muted); font-size: 0.85em; }
.card .value { font-size: 1.3em; font-weight: 600; margin-top: 0.3rem; }
table { border-collapse: collapse; width: 100%; background: var(--panel);
        border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th { background: var(--panel-2); color: var(--muted); font-weight: 500; }
th:first-child, td:first-child { text-align: left; }
img { max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }
.section { margin: 2rem 0; }
"""


def _plot_spread_z(spread, z, position) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 2]},
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        for sp in ax.spines.values():
            sp.set_color("#2a313c")
        ax.grid(True, color="#2a313c", alpha=0.6)

    ax1.plot(spread.index, spread.values, color="#58a6ff", linewidth=0.8, label="spread")
    ax1.set_title("log(A) - log(B)", color="#e6edf3")
    ax1.legend(loc="best", facecolor="#161b22", edgecolor="#2a313c", labelcolor="#e6edf3")

    ax2.plot(z.index, z.values, color="#f0883e", linewidth=0.8, label="z-score")
    ax2.axhline(0, color="#8b949e", linewidth=0.5)
    ax2.axhline(2, color="#f85149", linewidth=0.5, linestyle="--", alpha=0.6)
    ax2.axhline(-2, color="#3fb950", linewidth=0.5, linestyle="--", alpha=0.6)
    long_idx = position[position == 1].index
    short_idx = position[position == -1].index
    ax2.scatter(long_idx, z.loc[long_idx], color="#3fb950", s=4, label="long spread")
    ax2.scatter(short_idx, z.loc[short_idx], color="#f85149", s=4, label="short spread")
    ax2.set_title("z-score (entry ±2, exit ±0.5)", color="#e6edf3")
    ax2.legend(loc="best", facecolor="#161b22", edgecolor="#2a313c", labelcolor="#e6edf3", fontsize=8)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_equity(equity) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#e6edf3")
    for sp in ax.spines.values():
        sp.set_color("#2a313c")
    ax.grid(True, color="#2a313c", alpha=0.6)
    ax.plot(equity.index, equity.values, color="#58a6ff", linewidth=1.2)
    ax.set_title("Pairs strategy equity (JPY)", color="#e6edf3")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_yen(v):
    try:
        if pd.isna(v): return "—"
    except Exception:
        pass
    return f"¥{v:,.0f}"


def render_html(args, perf, result, period_str) -> str:
    cards = [
        ("通貨ペア", f"{args.pair_a} vs {args.pair_b}"),
        ("純損益", _fmt_yen(perf.net_profit)),
        ("Sharpe", f"{perf.sharpe:.2f}"),
        ("Max DD", f"{perf.max_drawdown:.2%}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html_lib.escape(l)}</div>'
        f'<div class="value">{html_lib.escape(v)}</div></div>'
        for l, v in cards
    )

    eq_chart = _plot_equity(result.equity)
    z_chart = _plot_spread_z(result.spread, result.z_score, result.position)

    title = f"Pairs Trading — {args.pair_a} vs {args.pair_b}"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html_lib.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="muted">統計的裁定取引 — z-score 平均回帰戦略 — {period_str}</p>
<div class="summary">{cards_html}</div>
<div class="section">
  <h2>資産推移 (JPY)</h2>
  <img alt="equity" src="data:image/png;base64,{eq_chart}">
</div>
<div class="section">
  <h2>スプレッド & z-score</h2>
  <img alt="spread" src="data:image/png;base64,{z_chart}">
  <p class="muted">
    緑のドット = ロング・ザ・スプレッド (A 買い + B 売り)。
    赤のドット = ショート・ザ・スプレッド (A 売り + B 買い)。
    z が ±2 を超えたらエントリー、|z| < 0.5 で利確、|z| > 4 で逆指値。
  </p>
</div>
</body></html>"""


# ---------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pairs / stat-arb backtester for two FX pairs")
    p.add_argument("--histdata-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--extra-dir", type=Path, default=None)
    p.add_argument("--pair-a", required=True, help="e.g. EURUSD")
    p.add_argument("--pair-b", required=True, help="e.g. GBPUSD")
    p.add_argument("--resample", default="1d")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--size", type=int, default=1_000,
                   help="Units per leg (default 1000)")
    p.add_argument("--equity", type=float, default=200_000.0,
                   help="Initial equity for both legs combined (default 200,000 JPY)")

    p.add_argument("--z-window", type=int, default=60)
    p.add_argument("--entry-z", type=float, default=2.0)
    p.add_argument("--exit-z", type=float, default=0.5)
    p.add_argument("--stop-z", type=float, default=4.0)

    p.add_argument("--html", type=Path, nargs="?",
                   const=Path("results/pairs.html"))
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--csv-out", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    a_df = _load_pair(args.pair_a, args.histdata_dir, args.resample, args.extra_dir)
    b_df = _load_pair(args.pair_b, args.histdata_dir, args.resample, args.extra_dir)

    if args.start or args.end:
        a_df = data_mod.slice_period(a_df, args.start, args.end)
        b_df = data_mod.slice_period(b_df, args.start, args.end)

    a_jpy = _mp.to_jpy(args.pair_a)
    b_jpy = _mp.to_jpy(args.pair_b)
    spread_a = _mp.default_spread(args.pair_a)
    spread_b = _mp.default_spread(args.pair_b)

    params = PairsParams(
        z_window=args.z_window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
    )

    print(f"A           : {args.pair_a}  {a_df.index[0]} -> {a_df.index[-1]}", file=sys.stderr)
    print(f"B           : {args.pair_b}  {b_df.index[0]} -> {b_df.index[-1]}", file=sys.stderr)
    print(f"z-score     : window={args.z_window}, entry=±{args.entry_z}, exit=±{args.exit_z}, stop=±{args.stop_z}", file=sys.stderr)
    print(f"Sizing      : {args.size} units/leg, init equity ¥{args.equity:,.0f}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    result = run_pairs_backtest(
        a_df, b_df,
        a_to_jpy=a_jpy, b_to_jpy=b_jpy,
        size=args.size,
        spread_a=spread_a, spread_b=spread_b,
        initial_equity_jpy=args.equity,
        params=params,
    )

    # Build trade list from position changes (one trade = one round-trip)
    pos = result.position
    trades_rows = []
    open_idx = None
    for i in range(1, len(pos)):
        if pos.iloc[i] != pos.iloc[i - 1]:
            # Position changed — close any open trade, possibly open new
            if pos.iloc[i - 1] != 0 and open_idx is not None:
                # Realized PnL between open_idx and i - 1
                seg = result.pnl.iloc[open_idx + 1 : i + 1]
                trades_rows.append({
                    "entry_time": pos.index[open_idx],
                    "exit_time": pos.index[i],
                    "side": int(pos.iloc[i - 1]),
                    "pnl": float(seg.sum()),
                })
                open_idx = None
            if pos.iloc[i] != 0:
                open_idx = i
    if open_idx is not None and pos.iloc[-1] != 0:
        seg = result.pnl.iloc[open_idx + 1 :]
        trades_rows.append({
            "entry_time": pos.index[open_idx],
            "exit_time": pos.index[-1],
            "side": int(pos.iloc[-1]),
            "pnl": float(seg.sum()),
        })
    trades = pd.DataFrame(trades_rows)

    perf = compute_performance(
        result.equity, result.pnl, trades, args.equity, position=pos,
    )

    print()
    print(f"Trades         : {perf.num_trades}")
    print(f"Win rate       : {perf.win_rate:.2%}")
    print(f"PF             : {perf.profit_factor:.2f}")
    print(f"Sharpe         : {perf.sharpe:.2f}")
    print(f"CAGR           : {perf.cagr:.2%}")
    print(f"Max drawdown   : {perf.max_drawdown:.2%}")
    print(f"Net profit     : ¥{perf.net_profit:,.0f}")

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        trades.to_csv(args.csv_out, index=False)

    if args.html:
        period_str = f"{a_df.index[0]} -> {a_df.index[-1]}"
        html = render_html(args, perf, result, period_str)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML saved: {args.html.resolve()}")
        if args.open:
            import webbrowser
            webbrowser.open(args.html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
