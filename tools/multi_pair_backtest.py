"""Multi-pair (portfolio) backtest.

Runs the same strategy on N currency pairs in parallel, with per-pair
spreads and unit sizing. Per-bar PnL is converted to JPY using rough
constants so the portfolio can be aggregated for a Japanese retail
account.

Usage:
    python tools/multi_pair_backtest.py \\
        --histdata-dir data/raw \\
        --pairs USDJPY,EURUSD,GBPUSD,AUDUSD,EURJPY \\
        --resample 1d --strategy adaptive --stop-atr 1.25 \\
        --size 1000 --html results/multi_pair.html

Spread defaults are SBI-FX-Trade-like retail values; override via
--spread-overrides "EURUSD=0.0001,GBPUSD=0.0002".
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import sys
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd                                                       # noqa: E402

from fx import data as data_mod                                           # noqa: E402
from fx import strategies as strategies_pkg                               # noqa: E402
from fx.backtest import BacktestConfig, StopConfig, run_backtest          # noqa: E402
from fx.metrics import compute_performance                                # noqa: E402
from fx.strategy import StrategyParams, generate_signals                  # noqa: E402


# --------------------------------------------------- pair-specific defaults


# Approximate retail spreads in price units (typical SBI / OANDA values).
DEFAULT_SPREADS: dict[str, float] = {
    # JPY pairs (price in JPY, 1 pip = 0.01)
    "USDJPY": 0.02,
    "EURJPY": 0.03,
    "GBPJPY": 0.04,
    "AUDJPY": 0.03,
    "NZDJPY": 0.03,
    "CADJPY": 0.03,
    "CHFJPY": 0.03,
    # Major USD pairs (price in USD, 1 pip = 0.0001)
    "EURUSD": 0.0002,
    "GBPUSD": 0.0003,
    "AUDUSD": 0.0003,
    "NZDUSD": 0.0003,
    "USDCAD": 0.0003,
    "USDCHF": 0.0003,
    # Crosses
    "EURGBP": 0.0004,
    "EURAUD": 0.0005,
}


# Rough conversion of one unit of the pair's quote currency to JPY.
# These are approximations for portfolio aggregation, NOT live FX rates.
QUOTE_TO_JPY: dict[str, float] = {
    "JPY": 1.0,
    "USD": 150.0,    # ~current USD/JPY mid
    "EUR": 165.0,
    "GBP": 195.0,
    "AUD": 100.0,
    "NZD": 90.0,
    "CAD": 110.0,
    "CHF": 170.0,
}


def quote_currency(pair: str) -> str:
    """Last 3 chars of the pair name. e.g. USDJPY -> JPY."""
    return pair.upper()[-3:]


def to_jpy(pair: str) -> float:
    """Conversion factor: 1 unit of quote currency = X JPY."""
    return QUOTE_TO_JPY.get(quote_currency(pair), 150.0)


def default_spread(pair: str) -> float:
    return DEFAULT_SPREADS.get(pair.upper(), 0.0003)


def parse_spread_overrides(s: str | None) -> dict[str, float]:
    if not s:
        return {}
    out: dict[str, float] = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad override (expected PAIR=VALUE): {chunk!r}")
        pair, val = chunk.split("=", 1)
        out[pair.strip().upper()] = float(val)
    return out


# --------------------------------------------------- per-pair runner


@dataclass
class PairResult:
    pair: str
    spread: float
    bars: int
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    perf_jpy: dict
    equity_jpy: pd.Series   # equity converted to JPY
    raw_perf: object        # original Performance object
    trades: pd.DataFrame


def find_files(histdata_dir: Path, pair: str) -> list[Path]:
    pattern = str(histdata_dir / f"DAT_ASCII_{pair.upper()}_M1_*.csv")
    return sorted(Path(p) for p in glob(pattern))


def run_pair(
    pair: str,
    histdata_dir: Path,
    resample: str,
    strategy_name: str,
    size: int,
    spread: float,
    initial_equity_jpy: float,
    stop_atr: float | None,
    start: str | None,
    end: str | None,
) -> PairResult | None:
    files = find_files(histdata_dir, pair)
    if not files:
        print(f"[skip] {pair}: no files matching DAT_ASCII_{pair.upper()}_M1_*.csv",
              file=sys.stderr)
        return None

    df = data_mod.load_histdata(files)
    if resample:
        df = data_mod.resample_ohlc(df, resample)
    if start or end:
        df = data_mod.slice_period(df, start, end)
    if len(df) == 0:
        print(f"[skip] {pair}: no bars after filters", file=sys.stderr)
        return None

    if strategy_name == "sma_rsi":
        signals = generate_signals(df, StrategyParams())
    else:
        mod = strategies_pkg.get(strategy_name)
        signals = mod.generate(df)

    cfg_jpy_factor = to_jpy(pair)
    # Run the per-pair backtest in QUOTE-currency units (BacktestConfig uses
    # whatever currency the price is in). Convert to JPY afterwards.
    cfg = BacktestConfig(size=size, spread=spread,
                         initial_equity=initial_equity_jpy / cfg_jpy_factor)
    stops = StopConfig(enabled=stop_atr is not None and stop_atr > 0,
                       atr_mult=stop_atr or 0.0)
    result = run_backtest(signals, cfg, stops=stops)

    # Convert PnL series to JPY
    equity_in_quote = result.equity
    equity_jpy = (equity_in_quote - cfg.initial_equity) * cfg_jpy_factor + initial_equity_jpy
    perf = compute_performance(
        equity_jpy,
        result.returns * cfg_jpy_factor,
        # Trades' pnl converted to JPY
        result.trades.assign(
            pnl=lambda d: d["pnl"] * cfg_jpy_factor
        ) if len(result.trades) > 0 else result.trades,
        initial_equity_jpy,
        position=result.position,
    )

    return PairResult(
        pair=pair.upper(),
        spread=spread,
        bars=len(df),
        period_start=df.index[0],
        period_end=df.index[-1],
        perf_jpy={
            "pf": perf.profit_factor,
            "sharpe": perf.sharpe,
            "cagr": perf.cagr,
            "total_return": perf.total_return,
            "max_dd": perf.max_drawdown,
            "win_rate": perf.win_rate,
            "num_trades": perf.num_trades,
            "net_profit": perf.net_profit,
            "rr": perf.risk_reward,
            "best": perf.best_trade,
            "worst": perf.worst_trade,
        },
        equity_jpy=equity_jpy,
        raw_perf=perf,
        trades=result.trades,
    )


# --------------------------------------------------- portfolio aggregation


def aggregate_portfolio(results: list[PairResult], initial_total_jpy: float) -> dict:
    """Sum each pair's JPY-converted PnL into a single portfolio equity curve."""
    if not results:
        return {}
    # Align indices: union of all bar timestamps; forward-fill missing
    eq_frames = []
    for r in results:
        # Per-pair PnL in JPY: equity - initial_per_pair
        pnl_jpy = r.equity_jpy.diff().fillna(0.0)
        eq_frames.append(pnl_jpy.rename(r.pair))
    pnl_df = pd.concat(eq_frames, axis=1).sort_index().fillna(0.0)
    portfolio_pnl = pnl_df.sum(axis=1)
    portfolio_equity = initial_total_jpy + portfolio_pnl.cumsum()

    # Combined trades (just append, mark with pair)
    trade_dfs = []
    for r in results:
        if len(r.trades) > 0:
            t = r.trades.copy()
            t["pair"] = r.pair
            t["pnl"] = t["pnl"] * to_jpy(r.pair)  # ensure JPY
            trade_dfs.append(t)
    combined_trades = (pd.concat(trade_dfs, ignore_index=True)
                       if trade_dfs else pd.DataFrame(columns=["pnl", "pair"]))

    perf = compute_performance(
        portfolio_equity, portfolio_pnl, combined_trades, initial_total_jpy,
    )
    return {
        "equity": portfolio_equity,
        "pnl_per_bar": portfolio_pnl,
        "trades": combined_trades,
        "perf": perf,
    }


# --------------------------------------------------- HTML output


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
tr.total td { background: var(--panel-2); font-weight: 600; }
td.pos { color: var(--pos); font-weight: 500; }
td.neg { color: var(--neg); font-weight: 500; }
img { max-width: 100%; border: 1px solid var(--border); border-radius: 6px; }
.section { margin: 2rem 0; }
details { background: var(--panel); border: 1px solid var(--border);
          border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.8rem; }
details summary { cursor: pointer; padding: 0.3rem 0; color: var(--text);
                  list-style: revert; }
details[open] summary { border-bottom: 1px solid var(--border); margin-bottom: 0.6rem; }
details-stack details + details { margin-top: 0.5rem; }
"""


def _fmt(v, kind="num"):
    try:
        if pd.isna(v):
            return "—"
    except Exception:
        pass
    if kind == "yen":
        return f"¥{v:,.0f}"
    if kind == "pct":
        return f"{v:.2%}"
    if kind == "int":
        return f"{int(v):,}"
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


def _cell(v, kind, positive_good=True):
    cls = ""
    if isinstance(v, (int, float)):
        if positive_good and v > 0: cls = "pos"
        elif positive_good and v < 0: cls = "neg"
        elif not positive_good and v < 0: cls = "pos"
        elif not positive_good and v > 0: cls = "neg"
    return f"<td class='{cls}'>{html_lib.escape(_fmt(v, kind))}</td>"


def _format_trades_table(trades_df: pd.DataFrame, limit: int = 50,
                          show_pair: bool = True) -> str:
    if trades_df is None or len(trades_df) == 0:
        return "<p class='muted'>トレードはありません</p>"

    # Most recent first
    df = trades_df.copy()
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time", ascending=False)
    shown = df.head(limit)

    cols = []
    if show_pair and "pair" in shown.columns:
        cols.append("pair")
    cols.extend(["entry_time", "exit_time", "side", "entry_price", "exit_price", "pnl"])

    headers_jp = {
        "pair": "通貨ペア",
        "entry_time": "エントリー時刻",
        "exit_time": "決済時刻",
        "side": "方向",
        "entry_price": "エントリー価格",
        "exit_price": "決済価格",
        "pnl": "損益 (JPY)",
    }
    thead = "<tr>" + "".join(
        f"<th>{html_lib.escape(headers_jp[c])}</th>" for c in cols
    ) + "</tr>"

    body_rows = []
    for _, row in shown.iterrows():
        cells: list[str] = []
        for c in cols:
            if c == "pair":
                cells.append(f"<td>{html_lib.escape(str(row[c]))}</td>")
            elif c == "side":
                side_label = "買い" if int(row[c]) == 1 else "売り"
                cells.append(f"<td>{side_label}</td>")
            elif c in ("entry_time", "exit_time"):
                ts = row[c]
                cells.append(
                    f"<td>{ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else html_lib.escape(str(ts))}</td>"
                )
            elif c in ("entry_price", "exit_price"):
                cells.append(f"<td>{row[c]:.4f}</td>")
            elif c == "pnl":
                cells.append(_cell(row[c], "yen"))
            else:
                cells.append(f"<td>{html_lib.escape(str(row[c]))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    extra = ""
    if len(df) > limit:
        extra = (f"<p class='muted'>直近 {limit} 件 / 全 {len(df):,} 件を表示しています。</p>")
    return (f"<table><thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table>"
            + extra)


def _plot_per_trade_pnl(trades_df: pd.DataFrame) -> str | None:
    if trades_df is None or len(trades_df) == 0:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = trades_df.sort_values("exit_time") if "exit_time" in trades_df.columns else trades_df
    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#e6edf3")
    for sp in ax.spines.values():
        sp.set_color("#2a313c")
    ax.grid(True, color="#2a313c", alpha=0.6, axis="y")

    pnls = df["pnl"].values
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, width=0.9)
    ax.axhline(0, color="#8b949e", linewidth=0.6)
    ax.set_xlabel("Trade #", color="#e6edf3")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_combined_equity(results, portfolio_equity) -> str:
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

    colors = ["#58a6ff", "#3fb950", "#f0883e", "#f85149", "#bc8cff",
              "#e8c547", "#56d4dd", "#ff7b72"]
    for i, r in enumerate(results):
        ax.plot(r.equity_jpy.index, r.equity_jpy.values,
                color=colors[i % len(colors)], linewidth=0.9,
                alpha=0.7, label=r.pair)
    if portfolio_equity is not None and len(portfolio_equity) > 0:
        ax.plot(portfolio_equity.index, portfolio_equity.values,
                color="white", linewidth=1.6, label="Portfolio (合計)")
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


def render_html(results, portfolio, strategy_name, period, title) -> str:
    pf_perf = portfolio.get("perf")

    cards = [
        ("通貨ペア数", str(len(results))),
        ("総トレード数", _fmt(sum(r.perf_jpy["num_trades"] for r in results), "int")),
        ("ポートフォリオ純損益", _fmt(pf_perf.net_profit if pf_perf else 0, "yen")),
        ("ポートフォリオ Sharpe", _fmt(pf_perf.sharpe if pf_perf else 0, "num")),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html_lib.escape(l)}</div>'
        f'<div class="value">{html_lib.escape(v)}</div></div>'
        for l, v in cards
    )

    # Per-pair table
    header = ("<tr><th>通貨ペア</th><th>トレード</th><th>勝率</th><th>PF</th>"
              "<th>Sharpe</th><th>CAGR</th><th>MaxDD</th><th>純損益</th></tr>")
    body = []
    for r in sorted(results, key=lambda x: x.perf_jpy["net_profit"], reverse=True):
        cells = [
            f"<td>{r.pair}</td>",
            _cell(r.perf_jpy["num_trades"], "int"),
            _cell(r.perf_jpy["win_rate"], "pct"),
            _cell(r.perf_jpy["pf"], "num"),
            _cell(r.perf_jpy["sharpe"], "num"),
            _cell(r.perf_jpy["cagr"], "pct"),
            _cell(r.perf_jpy["max_dd"], "pct", positive_good=False),
            _cell(r.perf_jpy["net_profit"], "yen"),
        ]
        body.append("<tr>" + "".join(cells) + "</tr>")
    if pf_perf is not None:
        body.append(
            "<tr class='total'>"
            + "<td>ポートフォリオ合計</td>"
            + _cell(pf_perf.num_trades, "int")
            + _cell(pf_perf.win_rate, "pct")
            + _cell(pf_perf.profit_factor, "num")
            + _cell(pf_perf.sharpe, "num")
            + _cell(pf_perf.cagr, "pct")
            + _cell(pf_perf.max_drawdown, "pct", positive_good=False)
            + _cell(pf_perf.net_profit, "yen")
            + "</tr>"
        )
    table_html = f"<table>{header}<tbody>{''.join(body)}</tbody></table>"

    chart_b64 = _plot_combined_equity(
        results, portfolio.get("equity") if portfolio else None
    )

    # Per-trade PnL bar chart (combined, all pairs)
    combined_trades = portfolio.get("trades") if portfolio else None
    trade_pnl_b64 = _plot_per_trade_pnl(combined_trades)
    trade_pnl_section = (
        f'<div class="section"><h2>トレード別損益（全通貨）</h2>'
        f'<img alt="trade-pnl" src="data:image/png;base64,{trade_pnl_b64}"></div>'
        if trade_pnl_b64 else ""
    )

    # Recent combined trades (last 50 across pairs)
    recent_trades_table = _format_trades_table(combined_trades, limit=50, show_pair=True)

    # Per-pair trade detail (last 20 per pair, with JPY-converted PnL)
    per_pair_blocks = []
    for r in sorted(results, key=lambda x: x.perf_jpy["net_profit"], reverse=True):
        if len(r.trades) == 0:
            continue
        t = r.trades.copy()
        t["pnl"] = t["pnl"] * to_jpy(r.pair)
        t["pair"] = r.pair
        block = (
            f"<details><summary><strong>{r.pair}</strong> "
            f"({r.perf_jpy['num_trades']:,} トレード, "
            f"純損益 {_fmt(r.perf_jpy['net_profit'], 'yen')})</summary>"
            f"{_format_trades_table(t, limit=20, show_pair=False)}"
            f"</details>"
        )
        per_pair_blocks.append(block)
    per_pair_html = "<div class='details-stack'>" + "".join(per_pair_blocks) + "</div>"

    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>{html_lib.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="muted">マルチ通貨バックテスト — 戦略: {html_lib.escape(strategy_name)} — {period}</p>
<div class="summary">{cards_html}</div>
<div class="section">
  <h2>資産推移（通貨別 + 合計）</h2>
  <img alt="equity" src="data:image/png;base64,{chart_b64}">
</div>
<div class="section">
  <h2>通貨別パフォーマンス</h2>
  {table_html}
</div>
{trade_pnl_section}
<div class="section">
  <h2>直近トレード一覧（全通貨、新しい順）</h2>
  {recent_trades_table}
</div>
<div class="section">
  <h2>通貨ペア別トレード詳細</h2>
  <p class="muted">各通貨ペアの最新 20 件。クリックで展開。</p>
  {per_pair_html}
</div>
</body></html>"""


# ---------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-pair (portfolio) backtest")
    p.add_argument("--histdata-dir", type=Path, default=Path("data/raw"),
                   help="Directory containing DAT_ASCII_<PAIR>_M1_*.csv files")
    p.add_argument("--pairs", required=True,
                   help="Comma-separated pairs, e.g. USDJPY,EURUSD,GBPUSD")
    p.add_argument("--resample", default="1d")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--strategy", default="adaptive",
                   choices=strategies_pkg.names())
    p.add_argument("--stop-atr", type=float, default=1.25)
    p.add_argument("--size", type=int, default=1_000,
                   help="Trade size per pair in units (default 1000)")
    p.add_argument("--equity-per-pair", type=float, default=100_000.0,
                   help="Initial equity allocated to each pair, in JPY (default 100,000)")

    p.add_argument("--spread-overrides",
                   help="Override default spreads, e.g. 'EURUSD=0.0001,GBPUSD=0.0002'")

    p.add_argument("--html", type=Path, nargs="?",
                   const=Path("results/multi_pair.html"))
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--csv-out", type=Path)
    p.add_argument("--title", default="FX マルチ通貨バックテスト")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        print("No pairs specified", file=sys.stderr)
        return 1

    overrides = parse_spread_overrides(args.spread_overrides)

    print(f"Strategy   : {args.strategy}", file=sys.stderr)
    print(f"Pairs      : {', '.join(pairs)}", file=sys.stderr)
    print(f"Equity/pair: ¥{args.equity_per_pair:,.0f}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    t0 = time.time()
    results: list[PairResult] = []
    for pair in pairs:
        spread = overrides.get(pair, default_spread(pair))
        try:
            r = run_pair(
                pair=pair,
                histdata_dir=args.histdata_dir,
                resample=args.resample,
                strategy_name=args.strategy,
                size=args.size,
                spread=spread,
                initial_equity_jpy=args.equity_per_pair,
                stop_atr=args.stop_atr,
                start=args.start,
                end=args.end,
            )
        except Exception as exc:
            print(f"[error] {pair}: {exc}", file=sys.stderr)
            continue
        if r is None:
            continue
        print(
            f"  {pair}: trades={r.perf_jpy['num_trades']:>4}  "
            f"PF={r.perf_jpy['pf']:.2f}  Sharpe={r.perf_jpy['sharpe']:.2f}  "
            f"net=¥{r.perf_jpy['net_profit']:>+10,.0f}",
            file=sys.stderr,
        )
        results.append(r)
    print(f"\nDone in {time.time()-t0:.1f}s\n", file=sys.stderr)

    if not results:
        print("No pair produced any results.", file=sys.stderr)
        return 1

    initial_total_jpy = args.equity_per_pair * len(results)
    portfolio = aggregate_portfolio(results, initial_total_jpy)

    pf_perf = portfolio["perf"]
    print("===== Portfolio summary =====")
    print(f"Pairs           : {len(results)}")
    print(f"Total trades    : {sum(r.perf_jpy['num_trades'] for r in results):,}")
    print(f"Portfolio Sharpe: {pf_perf.sharpe:.2f}")
    print(f"Portfolio CAGR  : {pf_perf.cagr:.2%}")
    print(f"Portfolio MaxDD : {pf_perf.max_drawdown:.2%}")
    print(f"Net profit      : ¥{pf_perf.net_profit:,.0f}")

    if args.csv_out:
        rows = [
            {
                "pair": r.pair, "spread": r.spread, "bars": r.bars,
                **{k: r.perf_jpy[k] for k in
                   ("num_trades", "win_rate", "pf", "sharpe", "cagr",
                    "total_return", "max_dd", "net_profit")},
            } for r in results
        ]
        rows.append({
            "pair": "TOTAL", "spread": None, "bars": None,
            "num_trades": pf_perf.num_trades,
            "win_rate": pf_perf.win_rate,
            "pf": pf_perf.profit_factor,
            "sharpe": pf_perf.sharpe,
            "cagr": pf_perf.cagr,
            "total_return": pf_perf.total_return,
            "max_dd": pf_perf.max_drawdown,
            "net_profit": pf_perf.net_profit,
        })
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.csv_out, index=False)
        print(f"\nCSV saved: {args.csv_out.resolve()}")

    if args.html:
        period = (
            f"{min(r.period_start for r in results)} → "
            f"{max(r.period_end for r in results)}"
        )
        html = render_html(results, portfolio, args.strategy, period, args.title)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML saved: {args.html.resolve()}")
        if args.open:
            import webbrowser
            webbrowser.open(args.html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
