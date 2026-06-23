"""
indicators.py — Rolling OLS regression bands + EMA crossover.

Entry/exit logic (verified against strategy1_dualbasket1.py):

  ENTRY (OR gate, regression priority):
    reg_entry  : prev_close < prev_lower2  AND  yest_close >= yest_lower2
    ema_entry  : ema_fast[-2] < ema_slow[-2]
                 AND ema_fast[-1] >= ema_slow[-1]
                 AND close[-1] > ema_slow[-1]

  EXIT (earliest fires, reg +2σ has priority if both same bar):
    reg_exit   : yest_close >= yest_upper2   (level touch, not a crossing)
    ema_exit   : ema_fast[-2] > ema_slow[-2]
                 AND ema_fast[-1] <= ema_slow[-1]   (death cross, no close filter)

  Trailing SL removed entirely; EMA death cross replaces it.
  All signals at bar[-1] → execute at bar[0] open (next bar).
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    ROLLING_WINDOW, MIN_ROLLING_POINTS,
    EMA_FAST, EMA_SLOW,
)


# ── Rolling OLS regression bands ─────────────────────────────────────────────

def rolling_regression_bands(close: pd.Series,
                              window: int = ROLLING_WINDOW,
                              min_points: int = MIN_ROLLING_POINTS) -> pd.DataFrame:
    """
    Rolling OLS regression bands — identical to backtest.
    Returns: trend_line, std_res, lower2, upper1, upper2.
    """
    close = close.dropna()
    n     = len(close)
    trend   = np.full(n, np.nan)
    std_res = np.full(n, np.nan)

    for i in range(min_points - 1, n):
        start      = max(0, i - window + 1)
        y          = close.iloc[start:i + 1].values.astype(float)
        if len(y) < min_points:
            continue
        x          = np.arange(len(y), dtype=float)
        sl, ic, *_ = stats.linregress(x, y)
        fitted     = sl * x + ic
        trend[i]   = fitted[-1]
        std_res[i] = (y - fitted).std()

    return pd.DataFrame({
        "trend_line": trend,
        "std_res":    std_res,
        "lower2":     trend - 2 * std_res,
        "upper1":     trend + 1 * std_res,
        "upper2":     trend + 2 * std_res,
    }, index=close.index)


# ── EMA signals (buy + death-cross sell) ─────────────────────────────────────

def ema_crossover_signals(close: pd.Series,
                          ema_fast: int = EMA_FAST,
                          ema_slow: int = EMA_SLOW) -> pd.DataFrame:
    """
    Golden cross buy + death cross sell, both confirmed on bar[-1].

    buy_signal[i]:
        ema_fast[-2] <  ema_slow[-2]          # was below
        ema_fast[-1] >= ema_slow[-1]          # crossed up
        close[-1]    >  ema_slow[-1]          # close filter

    sell_signal[i]:
        ema_fast[-2] >  ema_slow[-2]          # was above
        ema_fast[-1] <= ema_slow[-1]          # death cross (no close filter)
    """
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    buy = (
        (ema_f.shift(2) <  ema_s.shift(2)) &
        (ema_f.shift(1) >= ema_s.shift(1)) &
        (close.shift(1) >  ema_s.shift(1))
    )
    sell = (
        (ema_f.shift(2) >  ema_s.shift(2)) &
        (ema_f.shift(1) <= ema_s.shift(1))
    )
    return pd.DataFrame({
        "Close":       close,
        "EMA_fast":    ema_f,
        "EMA_slow":    ema_s,
        "buy_signal":  buy.fillna(False),
        "sell_signal": sell.fillna(False),
    }, index=close.index)


# ── OR-gate combined signals ──────────────────────────────────────────────────

def or_gate_combined_signals(close: pd.Series,
                              bands: pd.DataFrame,
                              ema_sig: pd.DataFrame) -> pd.DataFrame:
    """
    ENTRY  — OR gate:
        reg_entry OR ema_golden_cross  (regression takes priority label)

    EXIT   — earliest fires, reg +2σ takes priority if both same bar:
        reg_exit  : yest_close >= yest_upper2   (level touch)
        ema_exit  : ema death cross on bar[-1]  (replaces trailing SL)

    exit_type label: "regression_2sd" | "ema_death_cross"
    """
    yest_close  = close.shift(1)
    prev_close  = close.shift(2)
    yest_lower2 = bands["lower2"].shift(1)
    prev_lower2 = bands["lower2"].shift(2)
    yest_upper2 = bands["upper2"].shift(1)

    # ── Entry legs ────────────────────────────────────────────────────────────
    reg_buy = (
        yest_lower2.notna() &
        (yest_close >= yest_lower2) &
        (prev_close <  prev_lower2)
    )
    ema_buy = ema_sig["buy_signal"]

    reg_buy_f = reg_buy.fillna(False)
    ema_buy_f = ema_buy.fillna(False)

    buy_signal = reg_buy_f | ema_buy_f
    entry_type = np.where(reg_buy_f, "regression",
                 np.where(ema_buy_f, "ema_crossover", ""))

    # ── Exit legs ─────────────────────────────────────────────────────────────
    reg_sell = yest_upper2.notna() & (yest_close >= yest_upper2)
    ema_sell = ema_sig["sell_signal"]       # death cross from ema_crossover_signals

    reg_sell_f = reg_sell.fillna(False)
    ema_sell_f = ema_sell.fillna(False)

    # reg +2σ wins if both fire same bar
    sell_signal = reg_sell_f | ema_sell_f
    exit_type   = np.where(reg_sell_f, "regression_2sd",
                  np.where(ema_sell_f, "ema_death_cross", ""))

    out = ema_sig.copy()
    out["buy_signal"]  = buy_signal
    out["sell_signal"] = sell_signal
    out["entry_type"]  = entry_type
    out["exit_type"]   = exit_type
    return out


# ── Incremental cache ─────────────────────────────────────────────────────────

class IndicatorCache:
    """
    Per-basket indicator cache.  Recomputes fully on each new bar
    (regression requires full-window recompute anyway).
    """

    def __init__(self, basket_id):
        self.basket_id  = basket_id
        self._bands:   pd.DataFrame | None = None
        self._signals: pd.DataFrame | None = None
        self._last_len: int = 0

    def update(self, close: pd.Series) -> bool:
        """Recompute if series grew. Returns True when signals are valid."""
        import logging
        if len(close) == self._last_len:
            return self._signals is not None
            
        if len(close) < MIN_ROLLING_POINTS:
            logging.debug(f"[IndicatorCache] {self.basket_id} has {len(close)} points, needs {MIN_ROLLING_POINTS}. Not ready.")
            return False

        self._bands    = rolling_regression_bands(close)
        ema_sig        = ema_crossover_signals(close)
        self._signals  = or_gate_combined_signals(close, self._bands, ema_sig)
        self._last_len = len(close)
        return True

    @property
    def bands(self) -> pd.DataFrame | None:
        return self._bands

    @property
    def signals(self) -> pd.DataFrame | None:
        return self._signals

    def prev_bar_signal(self) -> dict:
        """
        Full indicator snapshot at the last completed bar.
        buy_signal / sell_signal here mean: signal confirmed on this bar,
        execute at NEXT bar open.
        """
        if self._signals is None or not len(self._signals):
            return {}

        last = self._signals.iloc[-1]
        brow = self._bands.iloc[-1] if self._bands is not None else {}

        def _b(key):
            try:    return float(brow[key])
            except: return float("nan")

        return {
            # ── signal flags ──────────────────────────────────────────────
            "buy_signal":  bool(last["buy_signal"]),
            "sell_signal": bool(last["sell_signal"]),
            "entry_type":  str(last["entry_type"]),
            "exit_type":   str(last["exit_type"]),
            "bar_time":    self._signals.index[-1],
            # ── price / EMA ───────────────────────────────────────────────
            "close":       float(last["Close"]),
            "ema_fast":    float(last["EMA_fast"]),
            "ema_slow":    float(last["EMA_slow"]),
            # ── regression bands ──────────────────────────────────────────
            "trend_line":  _b("trend_line"),
            "std_res":     _b("std_res"),
            "lower2":      _b("lower2"),
            "upper1":      _b("upper1"),
            "upper2":      _b("upper2"),
        }