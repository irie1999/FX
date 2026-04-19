"""Render a self-contained HTML report for a backtest run."""

from __future__ import annotations

import base64
import html
import io
from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestConfig, BacktestResult  # noqa: E402
from .metrics import Performance  # noqa: E402
from .strategy import StrategyParams  # noqa: E402


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_equity_drawdown(equity: pd.Series) -> str:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.2)
    ax1.set_title("Equity curve")
    ax1.grid(alpha=0.3)
    ax1.set_ylabel("Equity")

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.5)
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.3)
    ax2.set_ylabel("DD")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_price_signals(signals: pd.DataFrame, max_points: int = 4000) -> str:
    df = signals
    if len(df) > max_points:
        stride = int(np.ceil(len(df) / max_points))
        df = df.iloc[::stride]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["close"], color="#333", linewidth=0.8, label="close")
    if "sma_fast" in df:
        ax.plot(df.index, df["sma_fast"], color="#2ca02c", linewidth=0.9, label="SMA fast")
    if "sma_slow" in df:
        ax.plot(df.index, df["sma_slow"], color="#ff7f0e", linewidth=0.9, label="SMA slow")

    longs = df[df["signal"] == 1]
    shorts = df[df["signal"] == -1]
    ax.scatter(longs.index, longs["close"], marker="^", s=12, color="#2ca02c", alpha=0.5, label="long")
    ax.scatter(shorts.index, shorts["close"], marker="v", s=12, color="#d62728", alpha=0.5, label="short")

    ax.set_title("Price with signals")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_trade_pnl(trades: pd.DataFrame) -> str | None:
    if trades is None or len(trades) == 0:
        return None
    fig, ax = plt.subplots(figsize=(11, 3))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in trades["pnl"]]
    ax.bar(range(len(trades)), trades["pnl"], color=colors, width=0.9)
    ax.axhline(0, color="#333", linewidth=0.5)
    ax.set_title("Per-trade PnL")
    ax.set_xlabel("Trade #")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _metrics_table(perf: Performance) -> str:
    pf = f"{perf.profit_factor:.2f}" if np.isfinite(perf.profit_factor) else "∞"
    rows = [
        ("Total return", f"{perf.total_return:.2%}"),
        ("CAGR", f"{perf.cagr:.2%}"),
        ("Sharpe", f"{perf.sharpe:.2f}"),
        ("Max drawdown", f"{perf.max_drawdown:.2%}"),
        ("Win rate", f"{perf.win_rate:.2%}"),
        ("# trades", f"{perf.num_trades:d}"),
        ("Profit factor", pf),
        ("Avg trade PnL", f"{perf.avg_trade_pnl:,.2f}"),
    ]
    items = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{items}</table>"


def _params_table(params: StrategyParams, cfg: BacktestConfig, period: tuple) -> str:
    rows = [
        ("Period start", str(period[0])),
        ("Period end", str(period[1])),
        ("Bars", str(period[2])),
        ("SMA fast", str(params.fast)),
        ("SMA slow", str(params.slow)),
        ("RSI period", str(params.rsi_period)),
        ("RSI upper / lower", f"{params.rsi_upper} / {params.rsi_lower}"),
        ("Size", f"{cfg.size:,.0f}"),
        ("Spread", f"{cfg.spread}"),
        ("Initial equity", f"{cfg.initial_equity:,.0f}"),
    ]
    items = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{items}</table>"


def _trades_table(trades: pd.DataFrame, limit: int = 50) -> str:
    if trades is None or len(trades) == 0:
        return "<p><em>No trades.</em></p>"
    shown = trades.head(limit).copy()
    shown["side"] = shown["side"].map({1: "LONG", -1: "SHORT"}).fillna("")
    shown["entry_price"] = shown["entry_price"].map(lambda v: f"{v:.4f}")
    shown["exit_price"] = shown["exit_price"].map(lambda v: f"{v:.4f}")
    shown["pnl"] = shown["pnl"].map(lambda v: f"{v:,.2f}")
    thead = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in shown.columns) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    extra = ""
    if len(trades) > limit:
        extra = f"<p class='muted'>Showing first {limit} of {len(trades)} trades.</p>"
    return f"<table class='trades'><thead>{thead}</thead><tbody>{body}</tbody></table>{extra}"


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: #222; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
h1 { margin-bottom: 0.2rem; }
.muted { color: #777; font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }
table.kv { border-collapse: collapse; width: 100%; }
table.kv th, table.kv td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }
table.kv th { color: #555; font-weight: 500; width: 45%; }
table.trades { border-collapse: collapse; width: 100%; font-size: 0.88em; }
table.trades th, table.trades td { text-align: right; padding: 4px 8px; border-bottom: 1px solid #eee; }
table.trades th:first-child, table.trades td:first-child,
table.trades th:nth-child(2), table.trades td:nth-child(2) { text-align: left; }
img { max-width: 100%; height: auto; display: block; margin: 0.5rem 0; }
section { margin: 2rem 0; }
"""


def render_html(
    result: BacktestResult,
    perf: Performance,
    params: StrategyParams,
    cfg: BacktestConfig,
    title: str = "FX Backtest Report",
) -> str:
    """Return a fully self-contained HTML document as a string."""
    eq_b64 = _plot_equity_drawdown(result.equity)
    price_b64 = _plot_price_signals(result.signals)
    trade_b64 = _plot_trade_pnl(result.trades)

    period = (result.equity.index[0], result.equity.index[-1], len(result.equity))
    metrics = _metrics_table(perf)
    paramsT = _params_table(params, cfg, period)
    trades_html = _trades_table(result.trades)

    trade_section = (
        f"<section><h2>Per-trade PnL</h2><img alt='pnl' src='data:image/png;base64,{trade_b64}'></section>"
        if trade_b64
        else ""
    )

    html_doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="muted">Generated by fx.main — SMA crossover + RSI filter strategy</p>

<div class="grid">
  <div><h2>Performance</h2>{metrics}</div>
  <div><h2>Parameters</h2>{paramsT}</div>
</div>

<section>
  <h2>Equity & drawdown</h2>
  <img alt="equity" src="data:image/png;base64,{eq_b64}">
</section>

<section>
  <h2>Price & signals</h2>
  <img alt="price" src="data:image/png;base64,{price_b64}">
</section>

{trade_section}

<section>
  <h2>Trades</h2>
  {trades_html}
</section>

</body>
</html>
"""
    return html_doc


def write_html(
    path: str | Path,
    result: BacktestResult,
    perf: Performance,
    params: StrategyParams,
    cfg: BacktestConfig,
    title: str = "FX Backtest Report",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, perf, params, cfg, title), encoding="utf-8")
    return path
