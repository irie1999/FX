"""Render a self-contained HTML report for a backtest run (Japanese)."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestConfig, BacktestResult  # noqa: E402
from .metrics import Performance, monthly_pnl  # noqa: E402
from .strategy import StrategyParams  # noqa: E402


# Candidate Japanese-capable fonts on Windows / macOS / Linux
_JP_FONT_CANDIDATES = (
    "Yu Gothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "MS PGothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic Pro",
    "Hiragino Maru Gothic Pro",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
)


def _configure_jp_font() -> bool:
    """Pick a Japanese-capable font if available and return True on success."""
    try:
        available = {f.name for f in fm.fontManager.ttflist}
    except Exception:
        return False
    for name in _JP_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


_JP_OK = _configure_jp_font()


def _t(jp: str, en: str) -> str:
    """Use Japanese chart labels if a JP font is available, else English."""
    return jp if _JP_OK else en


# Chart palette matching the dark HTML theme
_DARK = {
    "bg": "#161b22",        # panel
    "fig_bg": "#0e1117",    # page
    "text": "#e6edf3",
    "grid": "#2a313c",
    "eq": "#58a6ff",
    "dd": "#f85149",
    "sma_fast": "#3fb950",
    "sma_slow": "#f0883e",
    "long": "#3fb950",
    "short": "#f85149",
    "zero": "#8b949e",
    "close": "#c9d1d9",
}


def _apply_dark_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor(_DARK["bg"])
        ax.tick_params(colors=_DARK["text"])
        for spine in ax.spines.values():
            spine.set_color(_DARK["grid"])
        ax.grid(True, color=_DARK["grid"], alpha=0.6)
        ax.yaxis.label.set_color(_DARK["text"])
        ax.xaxis.label.set_color(_DARK["text"])
        ax.title.set_color(_DARK["text"])


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        dpi=110,
        facecolor=_DARK["fig_bg"],
        edgecolor="none",
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_equity_drawdown(equity: pd.Series) -> str:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(_DARK["fig_bg"])
    ax1.plot(equity.index, equity.values, color=_DARK["eq"], linewidth=1.2)
    ax1.set_title(_t("資産推移", "Equity curve"))
    ax1.set_ylabel(_t("資産", "Equity"))

    ax2.fill_between(drawdown.index, drawdown.values, 0, color=_DARK["dd"], alpha=0.5)
    ax2.set_title(_t("ドローダウン", "Drawdown"))
    ax2.set_ylabel(_t("DD", "DD"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    _apply_dark_axes(ax1, ax2)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_price_signals(signals: pd.DataFrame, max_points: int = 4000) -> str:
    df = signals
    if len(df) > max_points:
        stride = int(np.ceil(len(df) / max_points))
        df = df.iloc[::stride]

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(_DARK["fig_bg"])
    ax.plot(df.index, df["close"], color=_DARK["close"], linewidth=0.8, label=_t("終値", "close"))
    if "sma_fast" in df:
        ax.plot(df.index, df["sma_fast"], color=_DARK["sma_fast"], linewidth=0.9, label=_t("短期 SMA", "SMA fast"))
    if "sma_slow" in df:
        ax.plot(df.index, df["sma_slow"], color=_DARK["sma_slow"], linewidth=0.9, label=_t("長期 SMA", "SMA slow"))

    longs = df[df["signal"] == 1]
    shorts = df[df["signal"] == -1]
    ax.scatter(longs.index, longs["close"], marker="^", s=14, color=_DARK["long"], alpha=0.7, label=_t("買い", "long"))
    ax.scatter(shorts.index, shorts["close"], marker="v", s=14, color=_DARK["short"], alpha=0.7, label=_t("売り", "short"))

    ax.set_title(_t("価格とシグナル", "Price with signals"))
    leg = ax.legend(loc="best", fontsize=8, facecolor=_DARK["bg"], edgecolor=_DARK["grid"])
    for text in leg.get_texts():
        text.set_color(_DARK["text"])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _apply_dark_axes(ax)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_trade_pnl(trades: pd.DataFrame) -> str | None:
    if trades is None or len(trades) == 0:
        return None
    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor(_DARK["fig_bg"])
    colors = [_DARK["long"] if v >= 0 else _DARK["short"] for v in trades["pnl"]]
    ax.bar(range(len(trades)), trades["pnl"], color=colors, width=0.9)
    ax.axhline(0, color=_DARK["zero"], linewidth=0.6)
    ax.set_title(_t("トレード別損益", "Per-trade PnL"))
    ax.set_xlabel(_t("トレード番号", "Trade #"))
    _apply_dark_axes(ax)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _fmt_yen(v: float) -> str:
    return f"¥{v:,.0f}"


def _summary_table(perf: Performance) -> str:
    """Core money-focused summary shown first."""
    rows = [
        ("初期資金", _fmt_yen(perf.initial_equity)),
        ("最終資金", _fmt_yen(perf.final_equity)),
        ("純損益", _fmt_yen(perf.net_profit)),
        ("総リターン", f"{perf.total_return:.2%}"),
        ("年率リターン (CAGR)", f"{perf.cagr:.2%}"),
        ("最大ドローダウン", f"{perf.max_drawdown:.2%}"),
    ]
    items = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{items}</table>"


def _metrics_table(perf: Performance) -> str:
    pf = f"{perf.profit_factor:.2f}" if np.isfinite(perf.profit_factor) else "∞"
    rows = [
        ("シャープレシオ", f"{perf.sharpe:.2f}"),
        ("トレード数", f"{perf.num_trades:,d}  ({perf.num_wins}勝 / {perf.num_losses}敗)"),
        ("勝率", f"{perf.win_rate:.2%}"),
        ("プロフィットファクター", pf),
        ("総利益 (勝ちトレード合計)", _fmt_yen(perf.gross_profit)),
        ("総損失 (負けトレード合計)", _fmt_yen(perf.gross_loss)),
        ("平均トレード損益", _fmt_yen(perf.avg_trade_pnl)),
        ("平均利益 / 平均損失", f"{_fmt_yen(perf.avg_win)} / {_fmt_yen(perf.avg_loss)}"),
        ("リスクリワード比", f"{perf.risk_reward:.2f}"),
        ("最大利益トレード", _fmt_yen(perf.best_trade)),
        ("最大損失トレード", _fmt_yen(perf.worst_trade)),
        ("最大連勝 / 最大連敗", f"{perf.max_win_streak} / {perf.max_loss_streak}"),
        ("平均保有本数", f"{perf.avg_holding_bars:.1f} bars"),
        ("建玉時間比率 (相場にいた割合)", f"{perf.exposure:.2%}"),
    ]
    items = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{items}</table>"


def _monthly_table(monthly: pd.DataFrame) -> str:
    if monthly is None or len(monthly) == 0:
        return "<p class='muted'>月次データなし</p>"
    thead = "<tr><th>月</th><th>損益</th><th>月次リターン</th></tr>"
    rows_html = []
    for _, row in monthly.iterrows():
        pnl = float(row["pnl"])
        ret = float(row["return"])
        color = "pos" if pnl >= 0 else "neg"
        rows_html.append(
            f"<tr><td>{html.escape(str(row['month']))}</td>"
            f"<td class='{color}'>{_fmt_yen(pnl)}</td>"
            f"<td class='{color}'>{ret:.2%}</td></tr>"
        )
    return (
        "<table class='trades monthly'><thead>" + thead + "</thead><tbody>"
        + "".join(rows_html) + "</tbody></table>"
    )


def _params_table(params: StrategyParams, cfg: BacktestConfig, period: tuple) -> str:
    rows = [
        ("期間開始", str(period[0])),
        ("期間終了", str(period[1])),
        ("本数", f"{period[2]:,}"),
        ("短期 SMA", str(params.fast)),
        ("長期 SMA", str(params.slow)),
        ("RSI 期間", str(params.rsi_period)),
        ("RSI 上限 / 下限", f"{params.rsi_upper} / {params.rsi_lower}"),
        ("取引枚数", f"{cfg.size:,.0f}"),
        ("スプレッド", f"{cfg.spread}"),
        ("初期資金", f"{cfg.initial_equity:,.0f}"),
    ]
    items = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in rows
    )
    return f"<table class='kv'>{items}</table>"


_TRADE_COL_JP = {
    "entry_time": "エントリー時刻",
    "exit_time": "決済時刻",
    "side": "方向",
    "entry_price": "エントリー価格",
    "exit_price": "決済価格",
    "pnl": "損益",
}


def _trades_table(trades: pd.DataFrame, limit: int = 50) -> str:
    if trades is None or len(trades) == 0:
        return "<p><em>トレードはありません。</em></p>"
    shown = trades.head(limit).copy()
    shown["side"] = shown["side"].map({1: "買い", -1: "売り"}).fillna("")
    shown["entry_price"] = shown["entry_price"].map(lambda v: f"{v:.4f}")
    shown["exit_price"] = shown["exit_price"].map(lambda v: f"{v:.4f}")
    shown["pnl"] = shown["pnl"].map(lambda v: f"{v:,.2f}")
    headers = [_TRADE_COL_JP.get(c, c) for c in shown.columns]
    thead = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    extra = ""
    if len(trades) > limit:
        extra = (
            f"<p class='muted'>先頭 {limit} 件 / 全 {len(trades):,} 件を表示しています。</p>"
        )
    return f"<table class='trades'><thead>{thead}</thead><tbody>{body}</tbody></table>{extra}"


_CSS = """
:root {
  --bg: #0e1117;
  --panel: #161b22;
  --panel-2: #1c222c;
  --border: #2a313c;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --pos: #3fb950;
  --neg: #f85149;
}
html, body { background: var(--bg); }
body { font-family: "Yu Gothic UI", "Yu Gothic", "Meiryo", "Hiragino Sans",
       "Noto Sans CJK JP", -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, Helvetica, Arial, sans-serif;
       color: var(--text); max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }
h1, h2 { color: var(--text); letter-spacing: 0.02em; }
h1 { margin-bottom: 0.2rem; }
h2 { border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
.muted, p.muted { color: var(--muted); font-size: 0.9em; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }
table.kv, table.trades {
  border-collapse: collapse; width: 100%;
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  overflow: hidden;
}
table.kv th, table.kv td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
table.kv tr:last-child th, table.kv tr:last-child td { border-bottom: none; }
table.kv th { color: var(--muted); font-weight: 500; width: 45%; background: var(--panel-2); }
table.kv td { color: var(--text); }
table.trades { font-size: 0.88em; }
table.trades th, table.trades td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--border); }
table.trades thead th { background: var(--panel-2); color: var(--muted); font-weight: 500; }
table.trades th:first-child, table.trades td:first-child,
table.trades th:nth-child(2), table.trades td:nth-child(2) { text-align: left; }
table.trades tbody tr:hover { background: var(--panel-2); }
img { max-width: 100%; height: auto; display: block; margin: 0.5rem 0;
      border: 1px solid var(--border); border-radius: 6px; background: var(--panel); }
section { margin: 2rem 0; }
td.pos { color: var(--pos); font-weight: 500; }
td.neg { color: var(--neg); font-weight: 500; }
.summary table.kv th { width: 55%; }
.summary table.kv td { font-weight: 600; font-size: 1.1em; color: var(--text); }
.monthly th, .monthly td { text-align: right; }
.monthly th:first-child, .monthly td:first-child { text-align: left; }
::selection { background: rgba(88,166,255,0.35); }
"""


def render_html(
    result: BacktestResult,
    perf: Performance,
    params: StrategyParams,
    cfg: BacktestConfig,
    title: str = "FX バックテストレポート",
) -> str:
    """Return a fully self-contained Japanese HTML document as a string."""
    eq_b64 = _plot_equity_drawdown(result.equity)
    price_b64 = _plot_price_signals(result.signals)
    trade_b64 = _plot_trade_pnl(result.trades)

    period = (result.equity.index[0], result.equity.index[-1], len(result.equity))
    summary = _summary_table(perf)
    metrics = _metrics_table(perf)
    paramsT = _params_table(params, cfg, period)
    monthly = monthly_pnl(result.equity, perf.initial_equity)
    monthly_html = _monthly_table(monthly)
    trades_html = _trades_table(result.trades)

    trade_section = (
        f"<section><h2>トレード別損益</h2><img alt='pnl' src='data:image/png;base64,{trade_b64}'></section>"
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
<p class="muted">fx.main による出力 — SMA クロス + RSI フィルタ戦略</p>

<section class="summary">
  <h2>損益サマリー</h2>
  {summary}
</section>

<div class="grid">
  <div><h2>詳細指標</h2>{metrics}</div>
  <div><h2>パラメータ</h2>{paramsT}</div>
</div>

<section>
  <h2>資産推移とドローダウン</h2>
  <img alt="equity" src="data:image/png;base64,{eq_b64}">
</section>

<section>
  <h2>月次損益</h2>
  {monthly_html}
</section>

<section>
  <h2>価格とシグナル</h2>
  <img alt="price" src="data:image/png;base64,{price_b64}">
</section>

{trade_section}

<section>
  <h2>トレード一覧</h2>
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
    title: str = "FX バックテストレポート",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result, perf, params, cfg, title), encoding="utf-8")
    return path
