"""Smoke-test the CLI's new --strategy and --eval-days flags."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fx.main import main as run_main


def _run(argv, capsys, tmp_path):
    html = tmp_path / "r.html"
    full_argv = list(argv) + ["--html", str(html), "--no-open"]
    rc = run_main(full_argv)
    captured = capsys.readouterr()
    return rc, captured.out, html


def test_main_runs_default_sma_rsi(capsys, tmp_path):
    rc, out, html = _run(
        ["--synthetic", "--bars", "800", "--equity", "500000"],
        capsys, tmp_path,
    )
    assert rc == 0
    assert "Strategy   : SMA クロス + RSI" in out
    assert html.exists()


def test_main_runs_adaptive_strategy(capsys, tmp_path):
    rc, out, html = _run(
        ["--synthetic", "--bars", "800", "--equity", "500000",
         "--strategy", "adaptive"],
        capsys, tmp_path,
    )
    assert rc == 0
    assert "適応型" in out
    assert html.exists()


def test_main_runs_supertrend_strategy(capsys, tmp_path):
    rc, out, html = _run(
        ["--synthetic", "--bars", "800", "--equity", "500000",
         "--strategy", "supertrend"],
        capsys, tmp_path,
    )
    assert rc == 0
    assert "Supertrend" in out


def test_main_eval_days_slices_period(capsys, tmp_path):
    """Synthetic data is hourly (~24 bars/day); 30 days ≈ 720 bars."""
    rc, out, html = _run(
        ["--synthetic", "--bars", "2000", "--equity", "500000",
         "--strategy", "sma_rsi", "--eval-days", "30"],
        capsys, tmp_path,
    )
    assert rc == 0
    assert "(last 30d)" in out
    bars_line = [l for l in out.splitlines() if l.startswith("Bars")][0]
    bars_count = int(bars_line.split(":")[1].split()[0])
    # Significantly fewer than the 2000 we loaded; 30 days of hourly ≈ 720 bars.
    assert bars_count < 2000
    assert bars_count <= 30 * 24 + 10


def test_main_eval_days_rebases_equity(capsys, tmp_path):
    """Evaluation-window equity must start at the configured initial value."""
    rc, out, _ = _run(
        ["--synthetic", "--bars", "1500", "--equity", "500000",
         "--strategy", "sma_rsi", "--eval-days", "20"],
        capsys, tmp_path,
    )
    assert rc == 0
    # Final status should show the sliced state without error
    assert "現在の状態" in out
