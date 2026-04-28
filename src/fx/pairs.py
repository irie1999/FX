"""Statistical arbitrage / pairs trading engine.

Source material:
  Gatev, Goetzmann, Rouwenhorst (2006)
  "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
  Review of Financial Studies, 19(3): 797-827

  Avellaneda & Lee (2010)
  "Statistical Arbitrage in the U.S. Equities Market"
  Quantitative Finance, 10(7): 761-782

Mechanics (z-score style):
  1. Take two cointegrated / correlated price series A and B
     (e.g. EURUSD and GBPUSD).
  2. Compute log spread:
        spread[t] = log(A[t]) - beta * log(B[t])
     beta defaults to 1 (simple log-ratio); a rolling regression beta
     can be enabled when there is a clear linear relationship.
  3. Z-score the spread on a rolling window:
        z[t] = (spread[t] - rolling_mean) / rolling_std
  4. When z is far from zero the spread is "stretched" and tends to
     revert. Take opposing positions in A and B:
        z >  +entry_z   -> short A, long  B   (= sell the spread)
        z <  -entry_z   -> long  A, short B   (= buy the spread)
  5. Close when |z| < exit_z (mean-reversion done) or when |z| breaches
     the stop_z (relationship has broken).

This module produces a *spread* equity curve in JPY by combining the
two pairs' bar PnL after FX conversion. It returns positions and per-bar
PnL for both legs; the CLI tool aggregates and reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairsParams:
    z_window: int = 60        # rolling window for mean / std
    entry_z: float = 2.0      # |z| threshold to open a position
    exit_z: float = 0.5       # |z| threshold to close a position
    stop_z: float = 4.0       # |z| at which we bail out (relationship broken)
    use_log: bool = True      # operate on log prices (default = yes, scale-free)


@dataclass
class PairsResult:
    spread: pd.Series         # the spread series we modelled
    z_score: pd.Series        # rolling z-score
    position: pd.Series       # +1 = long the spread, -1 = short the spread, 0 = flat
    legA_pnl: pd.Series       # per-bar JPY PnL on the A leg
    legB_pnl: pd.Series       # per-bar JPY PnL on the B leg
    pnl: pd.Series            # leg sum
    equity: pd.Series         # cumulative equity from initial


def _align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join two OHLC frames on their indices."""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise ValueError("A and B share no common timestamps")
    return a.loc[common], b.loc[common]


def compute_spread_signal(
    a_close: pd.Series,
    b_close: pd.Series,
    params: PairsParams = PairsParams(),
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (spread, z_score, position) series."""
    if params.use_log:
        spread = np.log(a_close) - np.log(b_close)
    else:
        spread = a_close - b_close

    mu = spread.rolling(params.z_window, min_periods=params.z_window).mean()
    sd = spread.rolling(params.z_window, min_periods=params.z_window).std(ddof=0)
    z = (spread - mu) / sd

    n = len(spread)
    pos = np.zeros(n, dtype=int)
    cur = 0
    z_arr = z.values
    for i in range(n):
        if np.isnan(z_arr[i]):
            pos[i] = 0
            continue

        zi = z_arr[i]
        # Stop-out: relationship blown out
        if cur != 0 and abs(zi) >= params.stop_z:
            cur = 0
        # Exit on mean-reversion
        elif cur != 0 and abs(zi) <= params.exit_z:
            cur = 0
        # Entry (only when flat)
        elif cur == 0:
            if zi >= params.entry_z:
                cur = -1     # spread is too high → short the spread
            elif zi <= -params.entry_z:
                cur = +1     # spread is too low  → long the spread
        pos[i] = cur

    position = pd.Series(pos, index=spread.index, dtype=int)
    return spread, z, position


def run_pairs_backtest(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    *,
    a_to_jpy: float,
    b_to_jpy: float,
    size: int = 1_000,
    spread_a: float = 0.02,
    spread_b: float = 0.0002,
    initial_equity_jpy: float = 200_000.0,
    params: PairsParams = PairsParams(),
) -> PairsResult:
    """Run a 2-leg pairs backtest.

    `a_to_jpy` / `b_to_jpy`: factor to convert one unit of each pair's
    quote-currency PnL into JPY. e.g. EURUSD trades in USD → ~150.

    Position convention:
      pos = +1  → long A, short B  (we expect A/B to rise)
      pos = -1  → short A, long B
    """
    a, b = _align(a_df, b_df)
    spread, z, position = compute_spread_signal(a["close"], b["close"], params)

    # Shift by 1 to avoid look-ahead (act on the next bar's open close).
    eff_pos = position.shift(1).fillna(0).astype(int)

    # Per-bar PnL on each leg in its quote currency, then convert to JPY.
    a_diff = a["close"].diff().fillna(0.0)
    b_diff = b["close"].diff().fillna(0.0)
    leg_a_units = eff_pos * size
    leg_b_units = -eff_pos * size

    legA_quote_pnl = leg_a_units * a_diff
    legB_quote_pnl = leg_b_units * b_diff
    legA_jpy = legA_quote_pnl * a_to_jpy
    legB_jpy = legB_quote_pnl * b_to_jpy

    # Transaction costs: each position change pays half-spread on both legs.
    pos_change = position.diff().abs().fillna(position.abs())
    cost_a = 0.5 * spread_a * size * pos_change * a_to_jpy
    cost_b = 0.5 * spread_b * size * pos_change * b_to_jpy

    pnl = legA_jpy + legB_jpy - cost_a - cost_b
    equity = initial_equity_jpy + pnl.cumsum()

    return PairsResult(
        spread=spread,
        z_score=z,
        position=position,
        legA_pnl=legA_jpy - cost_a,
        legB_pnl=legB_jpy - cost_b,
        pnl=pnl,
        equity=equity,
    )
