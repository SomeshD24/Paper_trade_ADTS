"""
portfolio_engine.py — Live dual-slot portfolio with correlation-based eviction.

EXIT LOGIC (updated):
  - Trailing SL removed entirely.
  - sell_signal from indicator engine now covers BOTH reg +2σ AND ema death cross.
    portfolio engine just checks the flag — no SL math here.

EVICTION LOGIC (unchanged from backtest):
  - Both slots full + new entry signal → compute mean pairwise correlation among
    {held_A, held_B, incoming}. The member with the HIGHEST mean correlation is
    the most redundant:
      - If it's a held basket → evict it, enter incoming.
      - If it's the incoming basket → skip (swap wouldn't diversify).
  - Multiple candidates, more than free slots → keep least-correlated subset.
"""

import logging
from itertools import combinations
from datetime import datetime

import numpy as np
import pandas as pd

from config import N_SLOTS, POSITION_SIZE, CORR_LOOKBACK

logger = logging.getLogger(__name__)


# ── Correlation helpers ───────────────────────────────────────────────────────

def _basket_return_window(basket_id: int, basket_close_series: dict,
                          end_time: pd.Timestamp,
                          lookback: int = CORR_LOOKBACK) -> pd.Series | None:
    if basket_id not in basket_close_series:
        return None
    close = basket_close_series[basket_id]
    avail = close.index[close.index <= end_time]
    if avail.empty:
        return None
    window = close.loc[:avail[-1]].iloc[-(lookback + 1):]
    if len(window) < max(10, lookback // 3):
        return None
    rets = window.pct_change().dropna()
    return rets if not rets.empty else None


def _pairwise_correlation(a: int, b: int, basket_close_series: dict,
                          end_time: pd.Timestamp) -> float:
    if a == b:
        return 1.0
    ra = _basket_return_window(a, basket_close_series, end_time)
    rb = _basket_return_window(b, basket_close_series, end_time)
    if ra is None or rb is None:
        return 0.0
    aligned = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(corr) if not np.isnan(corr) else 0.0


def _eviction_target(held_ids: list, new_bid: int,
                     basket_close_series: dict,
                     now: pd.Timestamp) -> int | None:
    """
    Among {held_ids + [new_bid]}, find the member with the highest mean
    pairwise correlation (most redundant).

    Returns the index into held_ids of the basket to evict, or None if
    the incoming basket itself is the most redundant (no eviction).
    """
    members = list(held_ids) + [new_bid]
    n = len(members)
    if n < 2:
        return None

    corr = {}
    for a, b in combinations(range(n), 2):
        c = _pairwise_correlation(members[a], members[b], basket_close_series, now)
        corr[(a, b)] = c
        corr[(b, a)] = c

    mean_corrs = [
        sum(corr[(i, j)] for j in range(n) if j != i) / (n - 1)
        for i in range(n)
    ]
    worst = int(np.argmax(mean_corrs))

    # worst == n-1 means the incoming basket is most redundant → skip
    return None if worst == n - 1 else worst


def _select_least_correlated_subset(basket_ids: list, k: int,
                                    basket_close_series: dict,
                                    now: pd.Timestamp) -> list:
    """
    From basket_ids pick the k-subset with the lowest total pairwise correlation.
    Used when multiple entry signals arrive and there are fewer free slots.
    """
    if k <= 0 or len(basket_ids) <= k:
        return basket_ids
    best_subset, best_score = basket_ids[:k], np.inf
    for combo in combinations(basket_ids, k):
        score = sum(
            _pairwise_correlation(a, b, basket_close_series, now)
            for a, b in combinations(combo, 2)
        )
        if score < best_score:
            best_score, best_subset = score, list(combo)
    return best_subset


# ── Portfolio state ───────────────────────────────────────────────────────────

class PortfolioEngine:
    """
    Live dual-slot portfolio engine.

    Slot state schema:
        basket_id, entry_time, entry_type, tickers,
        quantities, entry_prices, investment, capital_allocated,
        returns_ref_value, returns_ref_time, entry_indicator,
        exit_indicator (set just before close execution)
    """

    def __init__(self, initial_capital: float = POSITION_SIZE,
                 n_slots: int = N_SLOTS):
        self.initial_capital = initial_capital
        self.n_slots         = n_slots
        self.slots: list     = [None] * n_slots
        self.realized_pnl    = 0.0
        self.trade_log: list = []

        self._pending_exits:   list = []   # [(slot_idx, reason), ...]
        self._pending_entries: list = []   # [(bid, entry_type, capital, needs_evict, evict_idx, snap), ...]

        # Per-basket close price series for correlation (updated each bar)
        self.basket_close_series: dict = {}

    # ── Capital helpers ───────────────────────────────────────────────────────

    @property
    def base_capital(self) -> float:
        return self.initial_capital + self.realized_pnl

    @property
    def available_cash(self) -> float:
        invested = sum(s["investment"] for s in self.slots if s is not None)
        return self.base_capital - invested

    @property
    def active_basket_ids(self) -> set:
        return {s["basket_id"] for s in self.slots if s is not None}

    def both_empty(self) -> bool:
        return all(s is None for s in self.slots)

    # ── Bar-close: update close series (no SL tracking needed) ───────────────

    def on_bar_close(self, bar_time: pd.Timestamp,
                     basket_close_prices: dict):
        """
        Update basket_close_series for correlation computation.
        No trailing SL tracking — exit signals come entirely from indicators.
        """
        for bid, price in basket_close_prices.items():
            if bid not in self.basket_close_series:
                self.basket_close_series[bid] = pd.Series(dtype=float)
            self.basket_close_series[bid].at[bar_time] = price

    # ── Exit signal check ─────────────────────────────────────────────────────

    def check_exit_signals(self, sell_signals: dict,
                           basket_close_prices: dict) -> list:
        """
        sell_signals: {basket_id: bool}
            True when EITHER reg +2σ OR ema death cross fired on this bar.
            exit_type label is embedded in the slot's exit_indicator (set by runner
            just before this call).

        Queues exits for execution at next bar open.
        Returns list of (slot_idx, reason) for logging.
        """
        exits = []
        for idx, slot in enumerate(self.slots):
            if slot is None:
                continue
            bid = slot["basket_id"]

            if sell_signals.get(bid, False):
                # Derive reason from exit_indicator if already attached,
                # otherwise use generic label resolved at log time
                reason = slot.get("exit_indicator", {}).get("exit_type") or "signal_exit"
                self.queue_exit(idx, reason)
                exits.append((idx, reason))
                close_px = basket_close_prices.get(bid, "?")
                logger.info(f"  EXIT QUEUED B{bid} ({reason})  close={close_px}")

        return exits

    def queue_exit(self, slot_idx: int, reason: str):
        if self.slots[slot_idx] is not None:
            self._pending_exits.append((slot_idx, reason))

    # ── Entry queueing ────────────────────────────────────────────────────────

    def queue_entries(self, signals: list, bar_time: pd.Timestamp,
                      indicator_snapshots: dict | None = None):
        """
        signals: [(basket_id, entry_type), ...]
        indicator_snapshots: {basket_id: snapshot_dict}

        Applies:
          1. Multiple candidates > free slots → keep least-correlated subset.
          2. All slots full → evict highest mean-pairwise-correlated held basket
             (if incoming is most redundant, skip).
        """
        if not signals:
            return

        indicator_snapshots = indicator_snapshots or {}
        active     = self.active_basket_ids
        candidates = [(bid, et) for bid, et in signals if bid not in active]
        if not candidates:
            return

        n_free     = sum(1 for s in self.slots if s is None)
        both_empty = self.both_empty()

        # Trim candidates to least-correlated subset when more arrive than slots
        if n_free >= 2 and len(candidates) > n_free:
            keep = set(_select_least_correlated_subset(
                [bid for bid, _ in candidates], n_free,
                self.basket_close_series, bar_time,
            ))
            candidates = [(bid, et) for bid, et in candidates if bid in keep]

        eviction_done = False
        for bid, entry_type in candidates:
            if bid in self.active_basket_ids:
                continue

            snap           = indicator_snapshots.get(bid, {})
            free_slots_now = [i for i in range(self.n_slots) if self.slots[i] is None]

            if free_slots_now:
                capital = (self.base_capital / self.n_slots) if both_empty else self.available_cash
                self._pending_entries.append((bid, entry_type, capital, False, -1, snap))
            else:
                if eviction_done:
                    continue
                held_ids  = [self.slots[i]["basket_id"] for i in range(self.n_slots)]
                worst_idx = _eviction_target(held_ids, bid, self.basket_close_series, bar_time)
                if worst_idx is None:
                    logger.info(f"  Skip B{bid}: incoming most correlated, no eviction")
                    continue
                self._pending_entries.append((bid, entry_type, None, True, worst_idx, snap))
                eviction_done = True

    # ── Execute pending orders (next bar open) ────────────────────────────────

    def execute_pending(self, exec_prices_by_basket: dict,
                        bar_time: pd.Timestamp,
                        basket_info: dict) -> list:
        """
        Execute all queued exits then entries at given open prices.
        Returns list of closed trade dicts.
        """
        new_trades = []

        # Exits first
        for slot_idx, reason in self._pending_exits:
            slot = self.slots[slot_idx]
            if slot is None:
                continue
            prices = exec_prices_by_basket.get(slot["basket_id"])
            if prices is None:
                logger.warning(f"No exec prices for B{slot['basket_id']} exit, skipping")
                continue
            trade = self._close_slot(slot_idx, prices, bar_time, reason)
            if trade:
                new_trades.append(trade)
                self.slots[slot_idx] = None
        self._pending_exits.clear()

        # Entries
        for (bid, entry_type, capital, needs_eviction, evict_idx, entry_snap) in self._pending_entries:
            if bid in self.active_basket_ids:
                continue

            prices = exec_prices_by_basket.get(bid)
            if prices is None:
                logger.warning(f"No exec prices for B{bid} entry, skipping")
                continue

            # Eviction
            if needs_eviction and evict_idx >= 0 and self.slots[evict_idx] is not None:
                evict_bid    = self.slots[evict_idx]["basket_id"]
                evict_prices = exec_prices_by_basket.get(evict_bid)
                if evict_prices is None:
                    continue
                trade = self._close_slot(evict_idx, evict_prices, bar_time, "evicted_new_entry")
                if trade is None:
                    continue
                new_trades.append(trade)
                self.slots[evict_idx] = None
                capital = self.available_cash

            if capital is None:
                capital = self.available_cash

            target = next((i for i in range(self.n_slots) if self.slots[i] is None), None)
            if target is None:
                logger.warning(f"B{bid}: no free slot after eviction, skip")
                continue

            tickers = basket_info.get(bid, {}).get("tickers", [])
            if not tickers:
                continue

            state = self._open_slot(bid, entry_type, tickers, prices, capital, bar_time, entry_snap)
            if state:
                self.slots[target] = state
                for peer in range(self.n_slots):
                    if peer != target and self.slots[peer] is not None:
                        self._reset_returns_ref(self.slots[peer], bar_time)

        self._pending_entries.clear()
        return new_trades

    # ── Slot helpers ──────────────────────────────────────────────────────────

    def _open_slot(self, basket_id: int, entry_type: str, tickers: list,
                   exec_prices: dict, capital: float,
                   bar_time: pd.Timestamp,
                   indicator_snapshot: dict | None = None) -> dict | None:
        per_stock = capital / len(tickers)
        quantities, entry_prices = {}, {}
        for t in tickers:
            p = exec_prices.get(t)
            if p is None or p <= 0:
                return None
            quantities[t]   = int(per_stock / p)
            entry_prices[t] = p

        investment = sum(quantities[t] * entry_prices[t] for t in tickers)
        logger.info(f"  OPEN B{basket_id} ({entry_type}) at {bar_time}  invest={investment:.0f}")
        return {
            "basket_id":         basket_id,
            "entry_time":        bar_time,
            "entry_type":        entry_type,
            "tickers":           tickers,
            "quantities":        quantities,
            "entry_prices":      entry_prices,
            "investment":        investment,
            "capital_allocated": capital,
            "returns_ref_value": investment,
            "returns_ref_time":  bar_time,
            "entry_indicator":   indicator_snapshot or {},
            "exit_indicator":    {},          # populated by runner before exit execution
        }

    def _close_slot(self, slot_idx: int, exec_prices: dict,
                    bar_time: pd.Timestamp, reason: str) -> dict | None:
        slot = self.slots[slot_idx]
        if slot is None:
            return None
        if reason != "mtm" and bar_time == slot["entry_time"]:
            return None   # same-bar entry/exit guard

        exit_value = sum(
            slot["quantities"][t] * exec_prices.get(t, slot["entry_prices"][t])
            for t in slot["tickers"]
        )
        pnl      = exit_value - slot["investment"]
        hold_min = (bar_time - slot["entry_time"]).total_seconds() / 60

        if reason != "mtm":
            self.realized_pnl += pnl

        trade = {
            "basket_id":       slot["basket_id"],
            "entry_time":      slot["entry_time"],
            "exit_time":       bar_time,
            "entry_type":      slot["entry_type"],
            "close_reason":    reason,
            "investment":      round(slot["investment"], 2),
            "exit_value":      round(exit_value, 2),
            "pnl":             round(pnl, 2),
            "pnl_pct":         round(pnl / slot["investment"] * 100, 2) if slot["investment"] else 0.0,
            "hold_minutes":    round(hold_min, 1),
            "status":          "open (MTM)" if reason == "mtm" else "closed",
            "tickers":         slot["tickers"],
            "entry_prices":    slot["entry_prices"],
            "exit_prices":     {t: exec_prices.get(t) for t in slot["tickers"]},
            "entry_indicator": slot.get("entry_indicator", {}),
            "exit_indicator":  slot.get("exit_indicator", {}),
        }
        logger.info(
            f"  CLOSE B{slot['basket_id']} ({reason}) at {bar_time}  "
            f"pnl={pnl:+.0f} ({trade['pnl_pct']:+.2f}%)"
        )
        self.trade_log.append(trade)
        return trade

    def _reset_returns_ref(self, slot: dict, reset_time: pd.Timestamp):
        slot["returns_ref_value"] = slot["investment"]
        slot["returns_ref_time"]  = reset_time

    # ── MTM snapshot ─────────────────────────────────────────────────────────

    def mtm_snapshot(self, basket_close_prices: dict) -> dict:
        unrealized = 0.0
        slot_info  = []
        for slot in self.slots:
            if slot is None:
                slot_info.append(None)
                continue
            bid   = slot["basket_id"]
            price = basket_close_prices.get(bid)

            if price is None:
                mtm_value = slot["investment"]
            else:
                # Approximate: scale investment by price / entry_basket_price
                entry_snap  = slot.get("entry_indicator", {})
                entry_close = entry_snap.get("close") or price
                scale       = price / entry_close if entry_close else 1.0
                mtm_value   = slot["investment"] * scale

            u_pnl = mtm_value - slot["investment"]
            unrealized += u_pnl
            slot_info.append({
                "basket_id":      bid,
                "entry_time":     slot["entry_time"],
                "investment":     slot["investment"],
                "mtm_value":      round(mtm_value, 2),
                "unrealized_pnl": round(u_pnl, 2),
            })
        return {
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl":   round(self.realized_pnl, 2),
            "total_equity":   round(self.base_capital + unrealized, 2),
            "slots":          slot_info,
        }