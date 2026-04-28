"""Benchmark multiple day-trading strategies on the same data.

For every strategy in fx.strategies.REGISTRY:
  1. Generate signals from the loaded OHLC
  2. Apply EOD flattening (day-trade rules) unless --swing
  3. Run the vectorized backtest (optional ATR stop)
  4. Compute performance metrics

Results are printed as a ranked table, optionally saved to CSV, and
rendered into a self-contained dark-theme HTML dashboard showing:
  - Summary cards for the winner
  - Full ranking table
  - Side-by-side equity curves
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import sys
import time
from datetime import time as dtime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from fx import data as data_mod                              # noqa: E402
from fx import strategies                                    # noqa: E402
from fx.backtest import BacktestConfig, StopConfig, run_backtest  # noqa: E402
from fx.daytrade import DayTradeConfig, apply_daytrade_rules # noqa: E402
from fx.metrics import compute_performance                   # noqa: E402


SORT_COLUMNS = {
    "pf": "pf",
    "sharpe": "sharpe",
    "cagr": "cagr",
    "mar": "mar",
    "net_profit": "net_profit",
}


def run_strategy(
    name: str,
    df: pd.DataFrame,
    cfg: BacktestConfig,
    stops: StopConfig,
    dt_config: DayTradeConfig | None,
) -> dict:
    mod = strategies.get(name)
    signals = mod.generate(df)
    if dt_config is not None:
        signals = apply_daytrade_rules(signals, dt_config)
    result = run_backtest(signals, cfg, stops=stops)
    perf = compute_performance(
        result.equity, result.returns, result.trades,
        cfg.initial_equity, position=result.position,
    )
    mar = perf.cagr / abs(perf.max_drawdown) if perf.max_drawdown < 0 else float("inf")
    return {
        "name": name,
        "display": mod.DISPLAY,
        "pf": perf.profit_factor,
        "sharpe": perf.sharpe,
        "cagr": perf.cagr,
        "total_return": perf.total_return,
        "max_dd": perf.max_drawdown,
        "mar": mar,
        "net_profit": perf.net_profit,
        "num_trades": perf.num_trades,
        "win_rate": perf.win_rate,
        "rr": perf.risk_reward,
        "best": perf.best_trade,
        "worst": perf.worst_trade,
        "equity": result.equity,
    }


# ---------------------------------------------------------- HTML


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
tr.best td { background: rgba(88,166,255,0.12); font-weight: 600; }
td.pos { color: var(--pos); font-weight: 500; }
td.neg { color: var(--neg); font-weight: 500; }
img { max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }
.section { margin: 2rem 0; }
"""


def _fmt_num(v, spec=".2f"):
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    if v == float("inf"):
        return "∞"
    return format(v, spec)


def _fmt_pct(v):
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    return f"{v:.2%}"


def _fmt_yen(v, currency="¥"):
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    return f"{currency}{v:,.0f}"


def _cell(v, formatter, positive_good=True):
    cls = ""
    if isinstance(v, (int, float)):
        if positive_good:
            cls = "pos" if v > 0 else "neg" if v < 0 else ""
        else:
            cls = "pos" if v < 0 else "neg" if v > 0 else ""
    return f"<td class='{cls}'>{html_lib.escape(formatter(v))}</td>"


def _plot_equity_compare(rows: list[dict]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#e6edf3")
    for sp in ax.spines.values():
        sp.set_color("#2a313c")
    ax.grid(True, color="#2a313c", alpha=0.6)

    colors = ["#58a6ff", "#3fb950", "#f0883e", "#f85149", "#bc8cff", "#e8c547"]
    for i, r in enumerate(rows):
        ax.plot(r["equity"].index, r["equity"].values,
                color=colors[i % len(colors)], linewidth=1.1, label=r["display"])
    ax.set_title("戦略別 資産推移", color="#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    leg = ax.legend(loc="best", fontsize=9, facecolor="#161b22", edgecolor="#2a313c")
    for t in leg.get_texts():
        t.set_color("#e6edf3")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(rows: list[dict], sort_by: str, context: dict,
                title: str, currency: str) -> str:
    ranked = sorted(rows, key=lambda r: r[sort_by], reverse=True)
    best = ranked[0]
    cards = [
        ("勝者", best["display"]),
        (f"ベスト {sort_by.upper()}", _fmt_num(best[sort_by])),
        ("ベスト純損益", _fmt_yen(best["net_profit"], currency)),
        ("期間", f"{context['start']} → {context['end']}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html_lib.escape(l)}</div>'
        f'<div class="value">{html_lib.escape(v)}</div></div>'
        for l, v in cards
    )

    header = (
        "<tr><th>順位</th><th>戦略</th><th>PF</th><th>Sharpe</th>"
        "<th>CAGR</th><th>MaxDD</th><th>MAR</th>"
        "<th>トレード数</th><th>勝率</th><th>RR</th><th>純損益</th></tr>"
    )
    body = []
    for i, r in enumerate(ranked):
        cells = [
            f"<td>{i+1}</td>",
            f"<td>{html_lib.escape(r['display'])}</td>",
            _cell(r["pf"], lambda v: _fmt_num(v, ".2f")),
            _cell(r["sharpe"], lambda v: _fmt_num(v, ".2f")),
            _cell(r["cagr"], _fmt_pct),
            _cell(r["max_dd"], _fmt_pct, positive_good=False),
            _cell(r["mar"], lambda v: _fmt_num(v, ".2f")),
            f"<td>{r['num_trades']:,}</td>",
            _cell(r["win_rate"], _fmt_pct),
            _cell(r["rr"], lambda v: _fmt_num(v, ".2f")),
            _cell(r["net_profit"], lambda v: _fmt_yen(v, currency)),
        ]
        cls = " class='best'" if i == 0 else ""
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    table_html = "<table><thead>" + header + "</thead><tbody>" + "".join(body) + "</tbody></table>"

    chart_b64 = _plot_equity_compare(ranked)

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html_lib.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="muted">戦略比較 — {sort_by.upper()} 降順</p>
<div class="summary">{cards_html}</div>
<div class="section">
  <h2>資産推移比較</h2>
  <img alt="equity" src="data:image/png;base64,{chart_b64}">
</div>
<div class="section">
  <h2>ランキング</h2>
  {table_html}
</div>
</body></html>"""


# ---------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare day-trading strategies")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--histdata", nargs="+")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--bars", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample", default="15min",
                   help="Resample rule for the data (default 15min for day-trading)")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--size", type=float, default=10_000.0)
    p.add_argument("--spread", type=float, default=0.02)
    p.add_argument("--equity", type=float, default=500_000.0)
    p.add_argument("--stop-atr", type=float, default=None,
                   help="ATR stop multiplier (omit = disabled)")
    p.add_argument("--swing", action="store_true",
                   help="Do NOT force-flatten at EOD (swing trading mode)")
    p.add_argument("--eod-utc", default="21:00")

    p.add_argument("--strategies", default=",".join(strategies.names()),
                   help="Comma list of strategies to run")
    p.add_argument("--extra-csv", type=Path, default=None,
                   help="Optional recent OHLC CSV appended onto histdata "
                        "(e.g. fetched via tools/fetch_recent.py)")
    p.add_argument("--sort-by", choices=tuple(SORT_COLUMNS), default="pf")
    p.add_argument("--html", type=Path, nargs="?",
                   const=Path("results/compare.html"))
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--csv-out", type=Path)
    p.add_argument("--title", default="FX 戦略比較レポート")
    p.add_argument("--currency", default="¥")
    return p


def _load_data(args) -> pd.DataFrame:
    if args.synthetic:
        df = data_mod.synthetic_ohlc(bars=args.bars, seed=args.seed, freq="15min")
    elif args.histdata:
        df = data_mod.load_histdata(args.histdata)
    else:
        df = data_mod.load_csv(args.csv)
    if args.resample:
        df = data_mod.resample_ohlc(df, args.resample)
    extra = getattr(args, "extra_csv", None)
    if extra is not None:
        recent = data_mod.load_csv(extra)
        if args.resample:
            recent = data_mod.resample_ohlc(recent, args.resample)
        df = data_mod.merge_recent(df, recent)
    if args.start or args.end:
        df = data_mod.slice_period(df, args.start, args.end)
    return df


def _parse_eod(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    df = _load_data(args)
    if len(df) == 0:
        print("No bars after filters.", file=sys.stderr)
        return 1

    cfg = BacktestConfig(size=args.size, spread=args.spread, initial_equity=args.equity)
    stops = StopConfig(enabled=args.stop_atr is not None, atr_mult=args.stop_atr or 0.0)
    dt_config = None if args.swing else DayTradeConfig(eod_utc=_parse_eod(args.eod_utc))

    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    print(f"Bars        : {len(df):,}", file=sys.stderr)
    print(f"Period      : {df.index[0]} -> {df.index[-1]}", file=sys.stderr)
    print(f"Mode        : {'swing' if args.swing else 'day-trade'}", file=sys.stderr)
    print(f"Strategies  : {', '.join(names)}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    rows: list[dict] = []
    t0 = time.time()
    for name in names:
        try:
            res = run_strategy(name, df, cfg, stops, dt_config)
            rows.append(res)
            print(f"  {name:>10}  PF={res['pf']:.2f}  Sharpe={res['sharpe']:.2f}  "
                  f"trades={res['num_trades']}  net={res['net_profit']:,.0f}",
                  file=sys.stderr)
        except Exception as exc:
            print(f"  {name:>10}  FAILED: {exc}", file=sys.stderr)
    print(f"\nDone in {time.time()-t0:.1f}s\n", file=sys.stderr)

    if not rows:
        print("No strategies produced results.", file=sys.stderr)
        return 1

    sort_col = SORT_COLUMNS[args.sort_by]
    ranked = sorted(rows, key=lambda r: r[sort_col], reverse=True)

    # Console ranked table
    print("Ranked by", args.sort_by)
    print("=" * 72)
    print(f"{'rank':<5}{'strategy':<32}{'PF':>7}{'Sharpe':>8}{'CAGR':>10}{'Trades':>8}{'Net':>12}")
    print("-" * 72)
    for i, r in enumerate(ranked):
        print(
            f"{i+1:<5}{r['display']:<32}{r['pf']:>7.2f}{r['sharpe']:>8.2f}"
            f"{r['cagr']:>10.2%}{r['num_trades']:>8}{r['net_profit']:>12,.0f}"
        )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{k: v for k, v in r.items() if k != "equity"} for r in ranked]
        ).to_csv(args.csv_out, index=False)
        print(f"\nCSV saved: {args.csv_out.resolve()}")

    if args.html:
        context = {"start": df.index[0], "end": df.index[-1]}
        html = render_html(rows, args.sort_by, context, args.title, args.currency)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML saved: {args.html.resolve()}")
        if args.open:
            import webbrowser
            webbrowser.open(args.html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
