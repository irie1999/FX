"""Current-position status for live-trading-style output.

Reads the latest backtest result and tells you:
  - what position you're holding "now" (as of the last bar)
  - what the strategy wants you to do on the next bar (= tomorrow for daily)
  - entry price, unrealized PnL, and stop level if a position is open
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, StopConfig


@dataclass
class CurrentStatus:
    last_bar: pd.Timestamp
    current_price: float
    position: int               # -1 / 0 / +1
    next_signal: int            # -1 / 0 / +1 (what strategy wants for the next bar)
    next_action: str            # human-readable recommendation
    entry_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    bars_held: int = 0
    unrealized_pnl: float = 0.0
    stop_level: Optional[float] = None
    distance_to_stop: Optional[float] = None   # current_price - stop (positive = safety margin)
    atr_now: Optional[float] = None


def _position_run_start(position: pd.Series) -> int:
    """Index of the first bar in the latest contiguous non-zero run (or -1 if flat)."""
    n = len(position)
    if n == 0:
        return -1
    last = int(position.iloc[-1])
    if last == 0:
        return -1
    i = n - 1
    while i > 0 and int(position.iloc[i - 1]) == last:
        i -= 1
    return i


def _action_label(current_pos: int, next_signal: int, size: float) -> str:
    size_txt = f"{size:,.0f} 通貨"
    if next_signal == current_pos:
        if current_pos == 0:
            return "何もしない (ポジションなし、シグナルもなし)"
        direction = "ロング" if current_pos > 0 else "ショート"
        return f"{direction}を継続保有 (シグナル変化なし)"
    if next_signal == 0:
        direction = "ロング" if current_pos > 0 else "ショート"
        return f"{direction}を決済 ({size_txt}を翌寄付で反対売買)"
    if next_signal == 1:
        if current_pos == 0:
            return f"新規ロング ({size_txt}を翌寄付で買い)"
        return f"ショートを決済してロングへドテン ({size_txt}ずつ)"
    # next_signal == -1
    if current_pos == 0:
        return f"新規ショート ({size_txt}を翌寄付で売り)"
    return f"ロングを決済してショートへドテン ({size_txt}ずつ)"


def compute_current_status(
    result: BacktestResult,
    cfg: BacktestConfig,
    stops: StopConfig | None = None,
) -> CurrentStatus:
    signals = result.signals
    position = result.position.astype(int)

    last_bar = signals.index[-1]
    current_price = float(signals["close"].iloc[-1])
    current_pos = int(position.iloc[-1])
    next_signal = int(signals["signal"].iloc[-1])

    status = CurrentStatus(
        last_bar=last_bar,
        current_price=current_price,
        position=current_pos,
        next_signal=next_signal,
        next_action=_action_label(current_pos, next_signal, cfg.size),
    )

    if current_pos == 0:
        return status

    entry_i = _position_run_start(position)
    # With position = signal.shift(1), the execution happened at close of bar (entry_i - 1)
    price_idx = max(entry_i - 1, 0)
    entry_price = float(signals["close"].iloc[price_idx])
    entry_time = signals.index[price_idx]
    bars_held = len(position) - entry_i

    unrealized = current_pos * (current_price - entry_price) * cfg.size
    status.entry_time = entry_time
    status.entry_price = entry_price
    status.bars_held = bars_held
    status.unrealized_pnl = unrealized

    if stops is not None and stops.enabled and "atr" in signals.columns:
        atr_at_entry = signals["atr"].iloc[price_idx]
        if not np.isnan(atr_at_entry) and atr_at_entry > 0:
            stop_distance = stops.atr_mult * float(atr_at_entry)
            if current_pos > 0:
                stop = entry_price - stop_distance
                margin = current_price - stop
            else:
                stop = entry_price + stop_distance
                margin = stop - current_price
            status.stop_level = stop
            status.distance_to_stop = margin
            status.atr_now = float(signals["atr"].iloc[-1])

    return status


def format_status_console(status: CurrentStatus, currency: str = "¥") -> str:
    lines = []
    lines.append("=" * 40)
    lines.append(f"現在の状態 (データ最終: {status.last_bar})")
    lines.append("=" * 40)

    pos_label = {1: "ロング (+1)", -1: "ショート (-1)", 0: "ノーポジション (0)"}[status.position]
    lines.append(f"ポジション    : {pos_label}")
    lines.append(f"現在値        : {status.current_price:.4f}")

    if status.position != 0:
        assert status.entry_time is not None and status.entry_price is not None
        lines.append(f"エントリー日  : {status.entry_time.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"エントリー値  : {status.entry_price:.4f}")
        lines.append(f"保有本数      : {status.bars_held} バー")
        lines.append(f"含み損益      : {currency}{status.unrealized_pnl:+,.0f}")
        if status.stop_level is not None:
            lines.append(f"ストップ水準  : {status.stop_level:.4f}")
            lines.append(f"ストップまで  : {status.distance_to_stop:+.4f}  (価格単位)")

    lines.append("")
    lines.append(f"次のシグナル  : {status.next_signal:+d}")
    lines.append(f"翌バーの行動  : {status.next_action}")

    return "\n".join(lines)


def render_status_html(status: CurrentStatus, currency: str = "¥") -> str:
    """Self-contained dark-theme card block for embedding in the report."""
    pos_label = {1: "ロング (+1)", -1: "ショート (-1)", 0: "ノーポジション (0)"}[status.position]
    pos_color = {1: "var(--pos)", -1: "var(--neg)", 0: "var(--muted)"}[status.position]

    rows = [
        ("最終バー", str(status.last_bar)),
        ("現在値", f"{status.current_price:.4f}"),
        ("ポジション", f'<span style="color: {pos_color}; font-weight: 600">{pos_label}</span>'),
    ]
    if status.position != 0 and status.entry_price is not None and status.entry_time is not None:
        rows += [
            ("エントリー日時", status.entry_time.strftime("%Y-%m-%d %H:%M")),
            ("エントリー価格", f"{status.entry_price:.4f}"),
            ("保有本数", f"{status.bars_held} バー"),
            (
                "含み損益",
                f'<span style="color: {"var(--pos)" if status.unrealized_pnl >= 0 else "var(--neg)"}">'
                f'{currency}{status.unrealized_pnl:+,.0f}</span>',
            ),
        ]
        if status.stop_level is not None:
            rows += [
                ("ストップ水準", f"{status.stop_level:.4f}"),
                ("ストップまで", f"{status.distance_to_stop:+.4f} (価格)"),
            ]
    rows += [
        ("次のシグナル", f"{status.next_signal:+d}"),
        (
            "翌バーの行動",
            f'<span style="color: var(--accent); font-weight: 600">{status.next_action}</span>',
        ),
    ]
    items = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{items}</table>"
