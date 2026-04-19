"""Walk-forward analysis.

Splits the data into sequential (train, test) windows. For each window we:
  1. Grid-search the parameter space on the *train* segment
  2. Pick the best combo by the selected metric (pf / sharpe / mar / ...)
  3. Run those fixed params on the *test* segment — this is out-of-sample
Then we stitch the test segments together to form an honest equity curve
that the strategy would have produced if re-optimized on rolling history.

Usage:
    python tools/walk_forward.py \\
        --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \\
        --resample 1d --equity 500000 \\
        --train 540 --test 180 --step 180 \\
        --fast 10,15,20,30 --slow 50,75,100 \\
        --rsi-period 14,21 --rsi-upper 60,70 --rsi-lower 30 \\
        --stop-atr 1.0,1.25,1.5,2.0 \\
        --sort-by pf --html results/wfa.html

Window sizes are in BARS (after --resample), not calendar days.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from fx import data as data_mod                                 # noqa: E402
from fx.backtest import BacktestConfig                          # noqa: E402
from fx.metrics import compute_performance                      # noqa: E402
from fx.optimize import (                                       # noqa: E402
    SORT_COLUMNS,
    Combo,
    best_by,
    generate_combos,
    grid_search,
    parse_param_list,
    run_combo,
)


# ---------------------------------------------------------------- windowing


@dataclass(frozen=True)
class Window:
    idx: int
    train_start: int   # inclusive bar index
    train_end: int     # exclusive
    test_start: int
    test_end: int


def build_windows(n_bars: int, train: int, test: int, step: int, min_warmup: int = 0) -> list[Window]:
    """Slide train+test windows across the data.

    The first train window starts at `min_warmup` so indicator warmup can
    be satisfied. Subsequent windows advance by `step`. We stop once the
    test window would fall off the end of the series.
    """
    if train <= 0 or test <= 0 or step <= 0:
        raise ValueError("train/test/step must be positive")
    if train + test > n_bars - min_warmup:
        raise ValueError("Not enough bars for even one train+test window")

    windows: list[Window] = []
    ts = min_warmup
    i = 0
    while True:
        train_end = ts + train
        test_start = train_end
        test_end = test_start + test
        if test_end > n_bars:
            break
        windows.append(Window(i, ts, train_end, test_start, test_end))
        i += 1
        ts += step
    return windows


# -------------------------------------------------------------- WFA engine


def _test_equity_segment(
    df_test: pd.DataFrame,
    combo: Combo,
    cfg: BacktestConfig,
) -> pd.Series:
    """Run the backtest on the test slice and return its *net PnL per bar*
    series (not equity). Stitching accumulates these across windows.
    """
    result, _ = run_combo(df_test, combo, cfg)
    return result.returns


def run_walk_forward(
    df: pd.DataFrame,
    windows: list[Window],
    combos: list[Combo],
    cfg: BacktestConfig,
    sort_by: str,
    progress=True,
) -> dict:
    per_window: list[dict] = []
    stitched_pnl = pd.Series(dtype=float)
    t0 = time.time()
    for w in windows:
        train_df = df.iloc[w.train_start:w.train_end]
        test_df = df.iloc[w.test_start:w.test_end]

        grid = grid_search(train_df, combos, cfg)
        best = best_by(grid, sort_by)
        if best is None:
            if progress:
                print(f"[warn] window {w.idx}: no finite best result; skipping", file=sys.stderr)
            continue

        best_combo = Combo(
            fast=int(best["fast"]),
            slow=int(best["slow"]),
            rsi_period=int(best["rsi"]),
            rsi_upper=float(best["rsi_upper"]),
            rsi_lower=float(best["rsi_lower"]),
            stop_atr=None if pd.isna(best["stop_atr"]) else float(best["stop_atr"]),
        )

        # Out-of-sample run on test slice
        result, perf = run_combo(test_df, best_combo, cfg)
        stitched_pnl = pd.concat([stitched_pnl, result.returns])

        per_window.append(
            {
                "window": w.idx,
                "train_start": df.index[w.train_start],
                "train_end": df.index[w.train_end - 1],
                "test_start": df.index[w.test_start],
                "test_end": df.index[w.test_end - 1],
                "combo": best_combo,
                "train_metric": float(best[SORT_COLUMNS[sort_by]]),
                "train_pf": float(best["pf"]),
                "test_pf": perf.profit_factor,
                "test_sharpe": perf.sharpe,
                "test_return": perf.total_return,
                "test_net": perf.net_profit,
                "test_max_dd": perf.max_drawdown,
                "test_trades": perf.num_trades,
                "test_win_rate": perf.win_rate,
            }
        )

        if progress:
            elapsed = time.time() - t0
            print(
                f"  window {w.idx+1}/{len(windows)}  "
                f"params={best_combo.describe()}  "
                f"train {sort_by}={best[SORT_COLUMNS[sort_by]]:.2f}  "
                f"test PF={perf.profit_factor:.2f} "
                f"({elapsed:.1f}s)",
                file=sys.stderr,
            )

    # Build the stitched OOS equity curve
    stitched_equity = cfg.initial_equity + stitched_pnl.cumsum()
    oos_perf = compute_performance(
        stitched_equity,
        stitched_pnl,
        pd.DataFrame(columns=["pnl"]),   # trade-level records not stitched here
        cfg.initial_equity,
    )

    return {
        "per_window": per_window,
        "stitched_equity": stitched_equity,
        "stitched_pnl": stitched_pnl,
        "oos_perf": oos_perf,
    }


# ----------------------------------------------------------------- HTML report


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
        font-size: 0.88em; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
th { background: var(--panel-2); color: var(--muted); font-weight: 500; }
th:first-child, td:first-child { text-align: center; }
tbody tr:hover { background: var(--panel-2); }
td.pos { color: var(--pos); font-weight: 500; }
td.neg { color: var(--neg); font-weight: 500; }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
.card { background: var(--panel); border: 1px solid var(--border);
        border-radius: 6px; padding: 0.8rem 1rem; }
.card .label { color: var(--muted); font-size: 0.85em; }
.card .value { font-size: 1.3em; font-weight: 600; margin-top: 0.3rem; }
.section { margin: 2rem 0; }
img { max-width: 100%; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); }
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


def _plot_stitched_equity(equity: pd.Series) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#e6edf3")
        for sp in ax.spines.values():
            sp.set_color("#2a313c")
        ax.grid(True, color="#2a313c", alpha=0.6)
        ax.yaxis.label.set_color("#e6edf3")
        ax.xaxis.label.set_color("#e6edf3")
        ax.title.set_color("#e6edf3")

    ax1.plot(equity.index, equity.values, color="#58a6ff", linewidth=1.2)
    ax1.set_title("Stitched OOS equity")
    ax1.set_ylabel("Equity")
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#f85149", alpha=0.5)
    ax2.set_title("OOS drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0e1117", edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(wfa: dict, title: str, context: dict, sort_by: str) -> str:
    perf = wfa["oos_perf"]
    per_window = wfa["per_window"]

    pf_vals = [w["test_pf"] for w in per_window if math.isfinite(w["test_pf"])]
    median_test_pf = float(np.median(pf_vals)) if pf_vals else float("nan")
    pos_windows = sum(1 for w in per_window if w["test_return"] > 0)

    cards = [
        ("期間", f"{context['start']} → {context['end']}"),
        ("窓数 (train + test)", f"{len(per_window)}"),
        ("OOS Sharpe", _fmt_num(perf.sharpe, ".2f")),
        ("OOS 最大 DD", _fmt_pct(perf.max_drawdown)),
        ("OOS 総リターン", _fmt_pct(perf.total_return)),
        ("OOS 純損益", _fmt_yen(perf.net_profit)),
        ("Test PF (中央値)", _fmt_num(median_test_pf, ".2f")),
        ("プラス窓 / 全窓", f"{pos_windows} / {len(per_window)}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{html_lib.escape(l)}</div>'
        f'<div class="value">{html_lib.escape(v)}</div></div>'
        for l, v in cards
    )

    # Stitched equity chart
    stitched_img = ""
    if len(wfa["stitched_equity"]) > 0:
        b64 = _plot_stitched_equity(wfa["stitched_equity"])
        stitched_img = (
            f'<img alt="stitched equity" src="data:image/png;base64,{b64}">'
        )

    # Per-window table
    def _cls(v, positive_good=True):
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            if positive_good and v > 0: return "pos"
            if positive_good and v < 0: return "neg"
            if not positive_good and v < 0: return "pos"
            if not positive_good and v > 0: return "neg"
        return ""

    thead = (
        "<tr><th>#</th><th>Train 開始</th><th>Train 終了</th>"
        "<th>Test 開始</th><th>Test 終了</th><th>パラメータ</th>"
        f"<th>Train {sort_by.upper()}</th><th>Test PF</th>"
        "<th>Test Sharpe</th><th>Test リターン</th>"
        "<th>Test 純損益</th><th>Test MaxDD</th>"
        "<th>Trades</th><th>Win%</th></tr>"
    )
    body_rows = []
    for w in per_window:
        rows = [
            f"<td>{w['window']+1}</td>",
            f"<td>{w['train_start'].strftime('%Y-%m-%d')}</td>",
            f"<td>{w['train_end'].strftime('%Y-%m-%d')}</td>",
            f"<td>{w['test_start'].strftime('%Y-%m-%d')}</td>",
            f"<td>{w['test_end'].strftime('%Y-%m-%d')}</td>",
            f"<td style='text-align:left'>{html_lib.escape(w['combo'].describe())}</td>",
            f"<td>{_fmt_num(w['train_metric'], '.2f')}</td>",
            f"<td class='{_cls(w['test_pf'] - 1)}'>{_fmt_num(w['test_pf'], '.2f')}</td>",
            f"<td class='{_cls(w['test_sharpe'])}'>{_fmt_num(w['test_sharpe'], '.2f')}</td>",
            f"<td class='{_cls(w['test_return'])}'>{_fmt_pct(w['test_return'])}</td>",
            f"<td class='{_cls(w['test_net'])}'>{_fmt_yen(w['test_net'])}</td>",
            f"<td class='{_cls(w['test_max_dd'], False)}'>{_fmt_pct(w['test_max_dd'])}</td>",
            f"<td>{w['test_trades']}</td>",
            f"<td>{_fmt_pct(w['test_win_rate'])}</td>",
        ]
        body_rows.append("<tr>" + "".join(rows) + "</tr>")
    table_html = f"<table><thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table>"

    # Parameter stability (how often each value was picked)
    stability_blocks = []
    for axis in ("fast", "slow", "rsi_period", "rsi_upper", "rsi_lower", "stop_atr"):
        values = [getattr(w["combo"], axis) for w in per_window]
        counts: dict = {}
        for v in values:
            key = "—" if v is None else str(v)
            counts[key] = counts.get(key, 0) + 1
        items = "".join(
            f"<tr><td>{html_lib.escape(str(k))}</td><td>{cnt}</td></tr>"
            for k, cnt in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        stability_blocks.append(
            f"<div class='card'><div class='label'>{html_lib.escape(axis)}</div>"
            f"<table>{items}</table></div>"
        )
    stability_html = (
        '<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:1rem;">'
        + "".join(stability_blocks)
        + "</div>"
    )

    return f"""<!doctype html>
<html lang="ja">
<head><meta charset="utf-8"><title>{html_lib.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html_lib.escape(title)}</h1>
<p class="muted">
  ウォークフォワード結果 — 各窓で再最適化した OOS パフォーマンスのみを集計
</p>

<div class="summary">{cards_html}</div>

<div class="section">
  <h2>OOS 資産推移とドローダウン</h2>
  {stitched_img}
</div>

<div class="section">
  <h2>窓ごとの結果</h2>
  {table_html}
</div>

<div class="section">
  <h2>パラメータ安定性</h2>
  <p class="muted">窓ごとに選ばれた値の頻度。同じ値が多いほど頑健。</p>
  {stability_html}
</div>

</body>
</html>"""


# ---------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward analysis")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path)
    src.add_argument("--histdata", nargs="+")
    src.add_argument("--synthetic", action="store_true")
    p.add_argument("--bars", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resample", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    p.add_argument("--train", type=int, required=True, help="Train window size in bars")
    p.add_argument("--test", type=int, required=True, help="Test window size in bars")
    p.add_argument("--step", type=int, default=None,
                   help="Step in bars between windows (default: same as --test, non-overlap)")

    p.add_argument("--fast", default="10,15,20,30")
    p.add_argument("--slow", default="50,75,100")
    p.add_argument("--rsi-period", default="14,21")
    p.add_argument("--rsi-upper", default="70")
    p.add_argument("--rsi-lower", default="30")
    p.add_argument("--stop-atr", default="1.0,1.25,1.5,2.0")

    p.add_argument("--size", type=float, default=10_000.0)
    p.add_argument("--spread", type=float, default=0.02)
    p.add_argument("--equity", type=float, default=500_000.0)

    p.add_argument("--sort-by", choices=tuple(SORT_COLUMNS.keys()), default="pf",
                   help="Metric used to pick best params on train slice")
    p.add_argument(
        "--html", type=Path, nargs="?", const=Path("results/wfa.html"),
        help="Write HTML report (default: results/wfa.html)",
    )
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--csv-out", type=Path, help="Per-window summary CSV")
    p.add_argument("--title", default="FX ウォークフォワード結果")
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
        fast=parse_param_list(args.fast, int),
        slow=parse_param_list(args.slow, int),
        rsi_period=parse_param_list(args.rsi_period, int),
        rsi_upper=parse_param_list(args.rsi_upper, float),
        rsi_lower=parse_param_list(args.rsi_lower, float),
        stop_atr=parse_param_list(args.stop_atr, float),
    )
    if not combos:
        print("No valid combinations.", file=sys.stderr)
        return 1

    step = args.step if args.step else args.test
    try:
        windows = build_windows(len(df), args.train, args.test, step)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Bars       : {len(df):,}", file=sys.stderr)
    print(f"Period     : {df.index[0]} -> {df.index[-1]}", file=sys.stderr)
    print(f"Train/Test : {args.train}/{args.test} bars (step {step})", file=sys.stderr)
    print(f"Windows    : {len(windows)}", file=sys.stderr)
    print(f"Combos     : {len(combos):,} per window", file=sys.stderr)
    print(f"Sort by    : {args.sort_by}", file=sys.stderr)
    print("-" * 40, file=sys.stderr)

    cfg = BacktestConfig(size=args.size, spread=args.spread, initial_equity=args.equity)
    wfa = run_walk_forward(df, windows, combos, cfg, args.sort_by)
    perf = wfa["oos_perf"]

    print("\n===== OOS summary =====")
    print(f"Windows completed : {len(wfa['per_window'])}")
    print(f"OOS total return  : {perf.total_return:.2%}")
    print(f"OOS CAGR          : {perf.cagr:.2%}")
    print(f"OOS Sharpe        : {perf.sharpe:.2f}")
    print(f"OOS Max drawdown  : {perf.max_drawdown:.2%}")
    print(f"OOS net profit    : {perf.net_profit:,.0f}")

    pw_df = pd.DataFrame(
        [
            {
                "window": w["window"] + 1,
                "test_start": w["test_start"].strftime("%Y-%m-%d"),
                "test_end": w["test_end"].strftime("%Y-%m-%d"),
                "params": w["combo"].describe(),
                "train_pf": w["train_pf"],
                "test_pf": w["test_pf"],
                "test_return": w["test_return"],
                "test_net": w["test_net"],
                "test_trades": w["test_trades"],
            }
            for w in wfa["per_window"]
        ]
    )
    print("\nPer-window:")
    print(
        pw_df.to_string(
            index=False,
            formatters={
                "train_pf": "{:.2f}".format,
                "test_pf": "{:.2f}".format,
                "test_return": "{:.2%}".format,
                "test_net": "{:,.0f}".format,
            },
        )
    )

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        pw_df.to_csv(args.csv_out, index=False)
        print(f"\nCSV saved: {args.csv_out.resolve()}")

    if args.html:
        context = {"start": df.index[0], "end": df.index[-1]}
        html = render_html(wfa, args.title, context, args.sort_by)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML saved: {args.html.resolve()}")
        if args.open:
            import webbrowser
            webbrowser.open(args.html.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
