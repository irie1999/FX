"""Daily trading signal for manual execution at SBI FX (or any broker).

Runs the selected strategy on the loaded data, then reports the current
position and the action to take at the next session open. Formatted to
be easy to glance at and translate into an order ticket on the broker's
app / website.

Schedule this via Windows Task Scheduler (or cron on macOS/Linux) to
run once a day shortly after the NY close (JST 07:00 during EDT,
06:00 during EST) so it reads the just-finished daily candle.

Usage:
    python tools/daily_signal.py \\
        --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \\
        --resample 1d --strategy adaptive \\
        --size 1000 --stop-atr 1.25 --spread 0.02 \\
        --output results/signal.txt \\
        --json results/signal.json

Optional notifications (free, no LINE Notify — deprecated 2025-04):
    --webhook  URL    POST the signal to a Discord or Slack incoming-
                      webhook URL. Message text is Japanese and respects
                      both services' simple payload format.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd                                              # noqa: E402

from fx import data as data_mod                                  # noqa: E402
from fx import strategies as strategies_pkg                      # noqa: E402
from fx.backtest import BacktestConfig, StopConfig, run_backtest # noqa: E402
from fx.status import CurrentStatus, compute_current_status      # noqa: E402
from fx.strategy import StrategyParams, generate_signals         # noqa: E402


# ---------------------------------------------------------------- formatting


_ACTION_TYPES = {
    "HOLD_FLAT": "何もしない",
    "HOLD_LONG": "ロング継続",
    "HOLD_SHORT": "ショート継続",
    "ENTER_LONG": "新規ロング",
    "ENTER_SHORT": "新規ショート",
    "EXIT_LONG": "ロング決済",
    "EXIT_SHORT": "ショート決済",
    "FLIP_TO_LONG": "ショート→ロングへドテン",
    "FLIP_TO_SHORT": "ロング→ショートへドテン",
}


def classify_action(current_pos: int, next_signal: int) -> str:
    if next_signal == current_pos:
        return {1: "HOLD_LONG", -1: "HOLD_SHORT", 0: "HOLD_FLAT"}[current_pos]
    if next_signal == 0:
        return "EXIT_LONG" if current_pos > 0 else "EXIT_SHORT"
    if next_signal == 1:
        return "ENTER_LONG" if current_pos == 0 else "FLIP_TO_LONG"
    return "ENTER_SHORT" if current_pos == 0 else "FLIP_TO_SHORT"


def _fmt_money(v: float, currency: str = "¥") -> str:
    return f"{currency}{v:+,.0f}"


JST = timezone(timedelta(hours=9))


_RESAMPLE_HOURS = {
    "1min": 1/60, "5min": 5/60, "15min": 0.25, "30min": 0.5,
    "1h": 1.0, "4h": 4.0, "1d": 24.0, "d": 24.0,
}


def _next_close(last_bar, resample: str):
    """Estimate the next bar close from the last completed bar.

    Maps common resample rules to a Timedelta; falls back to 1d for
    anything unfamiliar. We avoid pandas offset arithmetic because
    different pandas versions disagree on which case ('d' vs 'D') is
    valid.
    """
    import pandas as pd

    hours = _RESAMPLE_HOURS.get(resample.lower(), 24.0)
    return pd.Timestamp(last_bar) + pd.Timedelta(hours=hours)


def _hours_until(ts) -> float:
    import pandas as pd
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (pd.Timestamp(ts) - now).total_seconds() / 3600.0)


def _format_sbi_instructions(action: str, instrument: str, size: int,
                             stop_level: float | None) -> list[str]:
    """SBI-flavored step-by-step instructions in Japanese."""
    lines: list[str] = []
    if action in ("ENTER_LONG", "ENTER_SHORT"):
        side = "買い" if action == "ENTER_LONG" else "売り"
        lines.append(f"  1. {instrument} を選択")
        lines.append("  2. 『新規』成行注文")
        lines.append(f"  3. 『{side}』 {size:,} 通貨")
        if stop_level is not None:
            lines.append(f"  4. 逆指値 (ストップ) を {stop_level:.3f} にセット")
    elif action in ("EXIT_LONG", "EXIT_SHORT"):
        side = "売り" if action == "EXIT_LONG" else "買い"
        lines.append(f"  1. {instrument} 保有ポジションを選択")
        lines.append(f"  2. 『決済』成行注文 ({side} {size:,} 通貨)")
    elif action in ("FLIP_TO_LONG", "FLIP_TO_SHORT"):
        close_side = "買い" if action == "FLIP_TO_LONG" else "売り"
        new_side = "買い" if action == "FLIP_TO_LONG" else "売り"
        lines.append("  1. 現在のポジションを成行決済")
        lines.append(f"  2. 同時に『新規』{new_side} {size:,} 通貨 を発注")
        if stop_level is not None:
            lines.append(f"  3. 新規ポジションに逆指値 (ストップ) {stop_level:.3f} をセット")
    return lines


def build_report(
    status: CurrentStatus,
    action: str,
    instrument: str,
    size: int,
    currency: str = "¥",
    resample: str = "1d",
) -> str:
    now_jst = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f" 📅 FX 取引シグナル — 作成: {now_jst}")
    lines.append("=" * 60)
    lines.append(f" 通貨ペア      : {instrument}")
    last_jst = status.last_bar.tz_convert(JST)
    lines.append(f" データ最終    : {last_jst.strftime('%Y-%m-%d %H:%M JST')}")

    # Next bar close info — so the user knows when to place orders.
    try:
        import pandas as pd
        next_close = _next_close(status.last_bar, resample)
        hours_remaining = _hours_until(next_close)
        next_close_jst = next_close.tz_convert(JST)
        if hours_remaining > 0:
            lines.append(
                f" 次バー締め    : {next_close_jst.strftime('%Y-%m-%d %H:%M JST')} "
                f"(あと {hours_remaining:.1f} h)"
            )
        else:
            lines.append(
                f" 次バー締め    : {next_close_jst.strftime('%Y-%m-%d %H:%M JST')} (既に経過、データ更新待ち)"
            )
    except Exception:
        pass

    lines.append(f" 現在値        : {status.current_price:.4f}")

    pos_label = {1: "ロング", -1: "ショート", 0: "ノーポジション"}[status.position]
    lines.append(f" 現在のポジション: {pos_label}"
                 + (f" {size:,} 通貨" if status.position != 0 else ""))

    if status.position != 0 and status.entry_price is not None and status.entry_time is not None:
        lines.append(f"   エントリー日  : {status.entry_time.strftime('%Y-%m-%d')}")
        lines.append(f"   エントリー値  : {status.entry_price:.4f}")
        lines.append(f"   含み損益      : {_fmt_money(status.unrealized_pnl, currency)}")
        if status.stop_level is not None:
            lines.append(f"   ストップ水準  : {status.stop_level:.4f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append(f" 📌 翌バーのアクション: {_ACTION_TYPES[action]}")
    lines.append("-" * 60)

    if action == "HOLD_FLAT":
        lines.append("  → 特に操作不要。新規エントリーシグナルなし。")
    elif action in ("HOLD_LONG", "HOLD_SHORT"):
        lines.append("  → 既存ポジションを継続保有。ストップ注文はそのまま維持。")
    else:
        lines.append(" 【SBI FX アプリでの手順】")
        lines.extend(_format_sbi_instructions(action, instrument, size, status.stop_level))

    # Risk estimate
    if action in ("ENTER_LONG", "ENTER_SHORT", "FLIP_TO_LONG", "FLIP_TO_SHORT") \
            and status.stop_level is not None:
        distance = abs(status.current_price - status.stop_level)
        risk = distance * size
        lines.append("")
        lines.append(f" 💰 想定リスク (ストップ到達時): {currency}-{risk:,.0f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------- notifications


def post_webhook(url: str, text: str) -> bool:
    """POST to a Slack or Discord incoming-webhook URL.

    Both platforms accept {"text": "..."} so one payload covers both.
    """
    data = json.dumps({"text": text, "content": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        print(f"Webhook HTTPError: {exc.code} {exc.reason}", file=sys.stderr)
    except Exception as exc:
        print(f"Webhook error: {exc}", file=sys.stderr)
    return False


# ---------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily trading signal for manual execution")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV (timestamp,open,high,low,close)")
    src.add_argument("--histdata", nargs="+",
                     help="HistData Generic ASCII M1 files / glob / directory")

    p.add_argument("--resample", default="1d",
                   help="Pandas resample rule (default 1d)")
    p.add_argument("--strategy", default="adaptive",
                   choices=strategies_pkg.names(),
                   help="Strategy (default: adaptive)")
    p.add_argument("--instrument", default="USD/JPY")
    p.add_argument("--size", type=int, default=1_000,
                   help="Trade size in units (default 1000 = micro lot)")
    p.add_argument("--spread", type=float, default=0.02)
    p.add_argument("--equity", type=float, default=500_000.0)
    p.add_argument("--stop-atr", type=float, default=1.25,
                   help="ATR stop multiplier (default 1.25)")

    # SMA+RSI specific (only used when --strategy sma_rsi)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--rsi", type=int, default=14)
    p.add_argument("--rsi-upper", type=float, default=70.0)
    p.add_argument("--rsi-lower", type=float, default=30.0)

    p.add_argument("--output", type=Path, help="Write the report to PATH (text)")
    p.add_argument("--json", type=Path, help="Write a JSON payload to PATH")
    p.add_argument("--webhook", help="Slack/Discord webhook URL to POST the report to")
    p.add_argument("--currency", default="¥")
    return p


def _load_data(args) -> pd.DataFrame:
    if args.histdata:
        df = data_mod.load_histdata(args.histdata)
    else:
        df = data_mod.load_csv(args.csv)
    if args.resample:
        df = data_mod.resample_ohlc(df, args.resample)
    return df


def generate_signal_payload(args) -> dict:
    df = _load_data(args)
    if len(df) == 0:
        raise RuntimeError("No bars loaded")

    if args.strategy == "sma_rsi":
        sp = StrategyParams(
            fast=args.fast, slow=args.slow,
            rsi_period=args.rsi,
            rsi_upper=args.rsi_upper, rsi_lower=args.rsi_lower,
        )
        signals = generate_signals(df, sp)
    else:
        mod = strategies_pkg.get(args.strategy)
        signals = mod.generate(df)

    cfg = BacktestConfig(size=args.size, spread=args.spread,
                         initial_equity=args.equity)
    stops = StopConfig(enabled=args.stop_atr is not None and args.stop_atr > 0,
                       atr_mult=args.stop_atr or 0.0)
    result = run_backtest(signals, cfg, stops=stops)
    status = compute_current_status(result, cfg, stops)
    action = classify_action(status.position, status.next_signal)

    report_text = build_report(
        status, action, args.instrument, args.size,
        currency=args.currency, resample=args.resample,
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "instrument": args.instrument,
        "strategy": args.strategy,
        "as_of": status.last_bar.isoformat(),
        "current_price": status.current_price,
        "position": status.position,
        "action": action,
        "action_jp": _ACTION_TYPES[action],
        "next_signal": status.next_signal,
        "size": args.size,
        "entry_time": status.entry_time.isoformat() if status.entry_time else None,
        "entry_price": status.entry_price,
        "unrealized_pnl": status.unrealized_pnl,
        "stop_level": status.stop_level,
        "distance_to_stop": status.distance_to_stop,
        "report_text": report_text,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = generate_signal_payload(args)
    except Exception as exc:
        print(f"Signal generation failed: {exc}", file=sys.stderr)
        return 1

    print(payload["report_text"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload["report_text"], encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.webhook:
        ok = post_webhook(args.webhook, payload["report_text"])
        print(f"webhook: {'ok' if ok else 'failed'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
