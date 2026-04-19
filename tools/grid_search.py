"""Grid search over strategy parameters.

Usage:
    python tools/grid_search.py \\
        --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \\
        --resample 1d --equity 500000 \\
        --fast 10,15,20,30 --slow 30,50,75,100 \\
        --rsi-period 10,14,21 \\
        --stop-atr none,1.0,1.25,1.5,2.0 \\
        --top 30 --sort-by pf --html results/grid.html

Reuses the existing data loader, strategy, backtest and metrics modules.
Runs are serialized (single-process) — the backtester is vectorized, so a
grid of a few hundred combinations completes in seconds on a laptop.
"""

from __future__ import annotations

import argparse
import html as html_lib
import itertools
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Make `fx.*` importable whether run from repo root or anywhere else.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from fx import data as data_mod  # noqa: E402
from fx.backtest import BacktestConfig, StopConfig, run_backtest  # noqa: E402
from fx.metrics import compute_performance  # noqa: E402
from fx.strategy import StrategyParams, generate_signals  # noqa: E402


SORT_CHOICES = ("pf", "sharpe", "cagr", "mar", "net_profit")


@dataclass(frozen=True)
class Combo:
    fast: int
    slow: int
    rsi_period: int
    rsi_upper: float
    rsi_lower: float
    stop_atr: float | None   # None = stops disabled


def _parse_list(raw: str, cast):
    items = [s.strip() for s in raw.split(",") if s.strip()]
    out = []
    for s in items:
        if s.lower() in ("none", "off", "-"):
            out.append(None)
        else:
            out.append(cast(s))
    return out


def generate_combos(
    fast: list[int],
    slow: list[int],
    rsi_period: list[int],
    rsi_upper: list[float],
    rsi_lower: list[float],
    stop_atr: list[float | None],
) -> list[Combo]:
    combos: list[Combo] = []
    for f, s, rp, ru, rl, sa in itertools.product(
        fast, slow, rsi_period, rsi_upper, rsi_lower, stop_atr
    ):
        if f >= s:
            continue                 # Require fast < slow
        if rl >= ru:
            continue                 # Require lower < upper
        combos.append(
            Combo(fast=f, slow=s, rsi_period=rp, rsi_upper=ru, rsi_lower=rl, stop_atr=sa)
        )
    return combos


def _run_one(df: pd.DataFrame, combo: Combo, cfg: BacktestConfig) -> dict:
    params = StrategyParams(
        fast=combo.fast,
        slow=combo.slow,
        rsi_period=combo.rsi_period,
        rsi_upper=combo.rsi_upper,
        rsi_lower=combo.rsi_lower,
    )
    signals = generate_signals(df, params)
    stops = StopConfig(
        enabled=combo.stop_atr is not None,
        atr_mult=combo.stop_atr or 0.0,
    )
    result = run_backtest(signals, cfg, stops=stops)
    perf = compute_performance(
        result.equity,
        result.returns,
        result.trades,
        cfg.initial_equity,
        position=result.position,
    )
    pf = perf.profit_factor
    mar = perf.cagr / abs(perf.max_drawdown) if perf.max_drawdown < 0 else float("inf")
    return {
        "fast": combo.fast,
        "slow": combo.slow,
        "rsi": combo.rsi_period,
        "rsi_upper": combo.rsi_upper,
        "rsi_lower": combo.rsi_lower,
        "stop_atr": combo.stop_atr if combo.stop_atr is not None else float("nan"),
        "num_trades": perf.num_trades,
        "win_rate": perf.win_rate,
        "pf": pf,
        "sharpe": perf.sharpe,
        "cagr": perf.cagr,
        "total_return": perf.total_return,
        "max_dd": perf.max_drawdown,
        "mar": mar,
        "net_profit": perf.net_profit,
        "rr": perf.risk_reward,
        "best": perf.best_trade,
        "worst": perf.worst_trade,
    }


def grid_search(df: pd.DataFrame, combos: Iterable[Combo], cfg: BacktestConfig) -> pd.DataFrame:
    rows: list[dict] = []
    combos = list(combos)
    start = time.time()
    for i, combo in enumerate(combos, 1):
        try:
            rows.append(_run_one(df, combo, cfg))
        except Exception as exc:   # one bad combo shouldn't kill the sweep
            print(f"[warn] combo {combo} failed: {exc}", file=sys.stderr)
        if i % 20 == 0 or i == len(combos):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            eta = (len(combos) - i) / max(rate, 1e-9)
            print(f"  {i:>5}/{len(combos)}  ({rate:.1f}/s, ETA {eta:.1f}s)", file=sys.stderr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- HTML output
_CSS = """
:root {
  --bg:#0e1117; --panel:#161b22; --panel-2:#1c222c; --border:#2a313c;
  --text:#e6edf3; --muted:#8b949e; --pos:#3fb950; --neg:#f85149; --accent:#58a6ff;
}
html, body { background: var(--bg); color: var(--text); }
body {
  font-family: "Yu Gothic UI","Yu Gothic","Meiryo","Hiragino Sans","Noto Sans CJK JP",
               -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  max-width: 1400px; margin: 2rem auto; padding: 0 1rem;
}
h1, h2 { letter-spacing: 0.02em; }
h1 { margin-bottom: 0.2rem; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
.muted { color: var(--muted); font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; background: var(--panel);
        border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
        font-size: 0.9em; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th { background: var(--panel-2); color: var(--muted); font-weight: 500; }
th:first-child, td:first-child { text-align: center; }
tbody tr:hover { background: var(--panel-2); }
td.pos { color: var(--pos); font-weight: 500; }
td.neg { color: var(--neg); font-weight: 500; }
tr.best td { background: rgba(88,166,255,0.1); }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
           margin: 1rem 0; }
.card { background: var(--panel); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.8rem 1rem; }
.card .label { color: var(--muted); font-size: 0.85em; }
.card .value { font-size: 1.3em; font-weight: 600; margin-top: 0.3rem; }
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


def _fmt_yen(v):
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    return f"¥{v:,.0f}"


def _cell(v, fmt, positive_good=True):
    txt = fmt(v)
    if isinstance(v, (int, float)) and not math.isnan(v) if isinstance(v, float) else True:
        cls = ""
        if isinstance(v, (int, float)):
            if positive_good and v > 0:
                cls = "pos"
            elif positive_good and v < 0:
                cls = "neg"
            elif not positive_good and v < 0:
                cls = "pos"   # negative is good (e.g. drawdown)
            elif not positive_good and v > 0:
                cls = "neg"
        return f"<td class='{cls}'>{html_lib.escape(str(txt))}</td>"
    return f"<td>{html_lib.escape(str(txt))}</td>"


def render_html(
    results: pd.DataFrame,
    sort_by: str,
    top: int,
    title: str,
    context: dict,
) -> str:
    if len(results) == 0:
        return "<html><body>No results.</body></html>"

    sort_col = {
        "pf": "pf",
        "sharpe": "sharpe",
        "cagr": "cagr",
        "mar": "mar",
        "net_profit": "net_profit",
    }[sort_by]
    # Replace inf with a very large number so sorting works predictably
    ranked = results.sort_values(sort_col, ascending=False).head(top).reset_index(drop=True)

    best = ranked.iloc[0]

    header_cards = [
        ("検証組み合わせ数", f"{len(results):,}"),
        ("期間", f"{context['start']} → {context['end']} ({context['bars']:,} 本)"),
        (f"ベスト {sort_by.upper()}", _fmt_num(best[sort_col])),
        ("ベスト純損益", _fmt_yen(best["net_profit"])),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html_lib.escape(l)}</div>'
        f'<div class="value">{html_lib.escape(v)}</div></div>'
        for l, v in header_cards
    )

    def _stop_cell(r):
        sa = r["stop_atr"]
        if pd.isna(sa):
            return "<td>—</td>"
        return f"<td>{sa:.2f}×ATR</td>"

    columns = [
        ("順位", lambda r, i: f"<td>{i+1}</td>"),
        ("fast", lambda r, i: _cell(int(r["fast"]), lambda v: str(v))),
        ("slow", lambda r, i: _cell(int(r["slow"]), lambda v: str(v))),
        ("rsi", lambda r, i: _cell(int(r["rsi"]), lambda v: str(v))),
        ("rsi_U/L", lambda r, i: f"<td>{r['rsi_upper']:.0f}/{r['rsi_lower']:.0f}</td>"),
        ("stop", lambda r, i: _stop_cell(r)),
        ("PF", lambda r, i: _cell(r["pf"], lambda v: _fmt_num(v, ".2f"))),
        ("Sharpe", lambda r, i: _cell(r["sharpe"], lambda v: _fmt_num(v, ".2f"))),
        ("CAGR", lambda r, i: _cell(r["cagr"], _fmt_pct)),
        ("MaxDD", lambda r, i: _cell(r["max_dd"], _fmt_pct, positive_good=False)),
        ("MAR", lambda r, i: _cell(r["mar"], lambda v: _fmt_num(v, ".2f"))),
        ("Trades", lambda r, i: _cell(int(r["num_trades"]), lambda v: str(v))),
        ("Win%", lambda r, i: _cell(r["win_rate"], _fmt_pct)),
        ("RR", lambda r, i: _cell(r["rr"], lambda v: _fmt_num(v, ".2f"))),
        ("Net", lambda r, i: _cell(r["net_profit"], _fmt_yen)),
    ]

    thead = "<tr>" + "".join(f"<th>{html_lib.escape(c[0])}</th>" for c in columns) + "</tr>"
    rows = []
    for i, row in ranked.iterrows():
        cells = "".join(c[1](row, i) for c in columns)
        cls = " class='best'" if i == 0 else ""
        rows.append(f"<tr{cls}>{cells}</tr>")
    table_html = f"<table><thead>{thead}</thead><tbody>{''.join(rows)}</tbody></table>"

    return f"""<!doctype html>
<html lang="ja">
<head><meta charset="utf-8"><title>{html_lib.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="muted">
  グリッドサーチ結果 — {sort_by.upper()} 降順、上位 {top} 件
</p>
<div class="summary">{cards_html}</div>
<div class="section">
  <h2>上位パラメータ一覧</h2>
  {table_html}
</div>
<p class="muted">
  全 {len(results):,} 通りのうち {len(ranked)} 件を表示。CSV で全件を保存するには <code>--csv</code> を指定。
</p>
</body>
</html>"""


# ------------------------------------------------------------------------ CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid search over strategy parameters")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--histdata", nargs="+")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--bars", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--fast", default="10,15,20,30", help="Comma list of fast SMA periods")
    p.add_argument("--slow", default="30,50,75,100", help="Comma list of slow SMA periods")
    p.add_argument("--rsi-period", default="14", help="Comma list of RSI periods")
    p.add_argument("--rsi-upper", default="70", help="Comma list of RSI upper thresholds")
    p.add_argument("--rsi-lower", default="30", help="Comma list of RSI lower thresholds")
    p.add_argument("--stop-atr", default="none,1.25,2.0", help="Comma list of stop mults, or 'none'")

    p.add_argument("--size", type=float, default=10_000.0)
    p.add_argument("--spread", type=float, default=0.02)
    p.add_argument("--equity", type=float, default=500_000.0)

    p.add_argument("--sort-by", choices=SORT_CHOICES, default="pf")
    p.add_argument("--top", type=int, default=30)
    p.add_argument(
        "--html",
        type=Path,
        nargs="?",
        const=Path("results/grid.html"),
        help="Write ranking HTML (default: results/grid.html)",
    )
    p.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the HTML in a browser after writing",
    )
    p.add_argument("--csv-out", type=Path, help="Optional CSV file of all combos")
    p.add_argument("--title", default="FX グリッドサーチ結果")
    return p


def _load_data(args) -> pd.DataFrame:
    if args.synthetic:
        df = data_mod.synthetic_ohlc(bars=args.bars, seed=args.seed)
    elif args.histdata:
        df = data_mod.load_histdata(args.histdata)
    else:
        df = data_mod.load_csv(args.csv)
    if args.resample:
        df = data_mod.resample_ohlc(df, args.resample)
    if args.start or args.end:
        df = data_mod.slice_period(df, args.start, args.end)
    return df


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    df = _load_data(args)
    if len(df) == 0:
        print("No bars after applying filters.", file=sys.stderr)
        return 1

    combos = generate_combos(
        fast=_parse_list(args.fast, int),
        slow=_parse_list(args.slow, int),
        rsi_period=_parse_list(args.rsi_period, int),
        rsi_upper=_parse_list(args.rsi_upper, float),
        rsi_lower=_parse_list(args.rsi_lower, float),
        stop_atr=_parse_list(args.stop_atr, float),
    )
    if not combos:
        print("No valid combinations (check fast<slow and rsi_lower<rsi_upper).", file=sys.stderr)
        return 1

    print(f"Bars       : {len(df):,}", file=sys.stderr)
    print(f"Period     : {df.index[0]} -> {df.index[-1]}", file=sys.stderr)
    print(f"Combos     : {len(combos):,}", file=sys.stderr)
    print(f"Sort by    : {args.sort_by}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    cfg = BacktestConfig(size=args.size, spread=args.spread, initial_equity=args.equity)
    results = grid_search(df, combos, cfg)

    # Console top 10
    sort_col = {"pf": "pf", "sharpe": "sharpe", "cagr": "cagr",
                "mar": "mar", "net_profit": "net_profit"}[args.sort_by]
    ranked = results.sort_values(sort_col, ascending=False).head(10)
    print("\nTop 10:")
    print(
        ranked.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2%}".format,
                "pf": "{:.2f}".format,
                "sharpe": "{:.2f}".format,
                "cagr": "{:.2%}".format,
                "total_return": "{:.2%}".format,
                "max_dd": "{:.2%}".format,
                "mar": "{:.2f}".format,
                "net_profit": "{:,.0f}".format,
                "rr": "{:.2f}".format,
                "best": "{:,.0f}".format,
                "worst": "{:,.0f}".format,
            },
        )
    )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.csv_out, index=False)
        print(f"\nCSV saved: {args.csv_out.resolve()}")

    if args.html:
        context = {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
        }
        html = render_html(results, args.sort_by, args.top, args.title, context)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML saved: {args.html.resolve()}")
        if args.open:
            import webbrowser
            webbrowser.open(args.html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
