"""
signal_engine.py — Evaluates per-basket OR-gate signals after each 5-min bar.

Flow per bar close:
  1. New bar appended to TickerBuffer for each stock.
  2. BasketSignalEngine.on_bar_close() called for each basket:
     a. Rebuilds basket OHLC from buffers.
     b. Recomputes indicators (IndicatorCache).
     c. Reads prev_bar_signal → records entry/exit signals.
  3. Returns aggregated signals for portfolio engine.
"""

import logging

import pandas as pd

from config import (
    EMA_FAST, EMA_SLOW, ROLLING_WINDOW, MIN_ROLLING_POINTS,
    POSITION_SIZE, BARS_PER_DAY, MIN_ALIGNED_BARS,
)
from data_manager import build_basket_5m_ohlc, TickerBuffer
from indicators import IndicatorCache, rolling_regression_bands, ema_crossover_signals, or_gate_combined_signals

logger = logging.getLogger(__name__)


class BasketSignalEngine:
    """
    Manages indicator state for ALL baskets.
    One IndicatorCache per basket, updated each bar.
    """

    def __init__(self, basket_info: dict, ticker_buffers: dict,
                 instrument_map: dict, position_size: float = POSITION_SIZE):
        """
        basket_info:     {basket_id: {"tickers": [...], "symbols": [...]}}
        ticker_buffers:  {yf_ticker: TickerBuffer}
        instrument_map:  {yf_ticker: {"token": int, "trading_symbol": str}}
        """
        self.basket_info    = basket_info
        self.ticker_buffers = ticker_buffers
        self.instrument_map = instrument_map
        self.position_size  = position_size

        self._caches: dict  = {bid: IndicatorCache(bid) for bid in basket_info}
        self._basket_ohlc: dict  = {}   # {basket_id: latest OHLC DataFrame}
        self._last_bar_time: pd.Timestamp | None = None

    # ── Per-bar update ────────────────────────────────────────────────────────

    def on_bar_close(self, bar_time: pd.Timestamp) -> dict:
        """
        Called after each completed 5-min bar has been appended to ticker buffers.

        Returns {
            "entry_signals": [(basket_id, entry_type), ...],
            "sell_signals":  {basket_id: bool},
            "basket_closes": {basket_id: float},
            "indicator_ready": {basket_id: bool},
        }
        """
        self._last_bar_time = bar_time

        entry_signals = []
        sell_signals  = {}
        basket_closes = {}
        ready_map     = {}

        for bid, info in self.basket_info.items():
            tickers = info["tickers"]

            # Build basket OHLC from rolling buffers
            ohlc = build_basket_5m_ohlc(self.ticker_buffers, tickers, self.position_size)
            if ohlc is None or ohlc.empty:
                ready_map[bid] = False
                # Diagnostic: show per-ticker bar counts
                bar_counts = {
                    t: self.ticker_buffers[t].n_bars
                    for t in tickers
                    if t in self.ticker_buffers
                }
                min_bars = min(bar_counts.values()) if bar_counts else 0
                logger.debug(
                    f"  B{bid} not ready: basket OHLC failed "
                    f"(min ticker bars={min_bars}, need aligned >= {MIN_ALIGNED_BARS})"
                )
                continue
            self._basket_ohlc[bid] = ohlc

            # Update indicator cache
            cache = self._caches[bid]
            n_close = len(ohlc["Close"])
            ready = cache.update(ohlc["Close"])
            ready_map[bid] = ready

            if not ready:
                logger.debug(
                    f"  B{bid} not ready: {n_close} aligned bars, "
                    f"need >= {MIN_ROLLING_POINTS} for indicators"
                )
                continue

            # Record basket close for portfolio engine
            basket_closes[bid] = float(ohlc["Close"].iloc[-1])

            # Read signal at current bar (will be "prev bar" from next bar's perspective)
            sig = cache.prev_bar_signal()
            if not sig:
                continue

            if sig.get("buy_signal"):
                entry_type = sig.get("entry_type", "")
                entry_signals.append((bid, entry_type))
                logger.info(f"  SIGNAL ENTRY B{bid} ({entry_type}) at {bar_time}")

            sell_signals[bid] = bool(sig.get("sell_signal", False))

        return {
            "entry_signals":    entry_signals,
            "sell_signals":     sell_signals,
            "basket_closes":    basket_closes,
            "indicator_ready":  ready_map,
        }

    def get_exec_prices(self, bar_time: pd.Timestamp) -> dict:
        """
        Returns {basket_id: {ticker: open_price}} using the most recent OPEN
        bar in each ticker's buffer. Called at bar N+1 open for order execution.
        """
        result = {}
        for bid, info in self.basket_info.items():
            tickers = info["tickers"]
            prices  = {}
            for t in tickers:
                buf = self.ticker_buffers.get(t)
                if buf is None or buf.df.empty:
                    continue
                # Use most recent bar open (= bar N+1 open, which is now)
                prices[t] = float(buf.df["Open"].iloc[-1])
            if len(prices) == len(tickers):
                result[bid] = prices
        return result

    def get_basket_close_prices(self, bar_time: pd.Timestamp | None = None) -> dict:
        """Return {basket_id: latest_close} from cached basket OHLC."""
        result = {}
        for bid, ohlc in self._basket_ohlc.items():
            if not ohlc.empty:
                result[bid] = float(ohlc["Close"].iloc[-1])
        return result

    def n_ready_baskets(self) -> int:
        return sum(1 for c in self._caches.values() if c.signals is not None)

    def warmup_indicators(self) -> int:
        """
        Pre-compute indicator caches for ALL baskets from the seeded warmup data.
        Call once at startup after ticker buffers are seeded.
        Returns number of baskets that became ready.
        """
        n_ready = 0
        for bid, info in self.basket_info.items():
            tickers = info["tickers"]
            ohlc = build_basket_5m_ohlc(self.ticker_buffers, tickers, self.position_size)
            if ohlc is None or ohlc.empty:
                logger.warning(
                    f"  B{bid}: warmup OHLC failed — "
                    f"ticker bar counts: { {t: self.ticker_buffers[t].n_bars for t in tickers} }"
                )
                continue
            self._basket_ohlc[bid] = ohlc
            n_bars = len(ohlc["Close"])
            ready = self._caches[bid].update(ohlc["Close"])
            if ready:
                n_ready += 1
                logger.info(f"  B{bid}: {n_bars} aligned bars → READY")
            else:
                logger.warning(f"  B{bid}: {n_bars} aligned bars < {MIN_ROLLING_POINTS} needed — NOT READY")
        return n_ready


    def warmup_status(self) -> str:
        ready = self.n_ready_baskets()
        total = len(self.basket_info)
        # Show per-ticker bar counts for diagnosis
        ticker_bars = {
            t: buf.n_bars
            for t, buf in self.ticker_buffers.items()
        }
        if ticker_bars:
            min_bars = min(ticker_bars.values())
            max_bars = max(ticker_bars.values())
            thin_tickers = [t for t, n in ticker_bars.items() if n < MIN_ROLLING_POINTS]
            bar_info = f"ticker bars min={min_bars}/max={max_bars}"
            if thin_tickers:
                bar_info += f", {len(thin_tickers)} tickers thin (<{MIN_ROLLING_POINTS})"
        else:
            bar_info = "no ticker data"
        return (
            f"{ready}/{total} baskets ready "
            f"(need {MIN_ROLLING_POINTS} indicator bars, {MIN_ALIGNED_BARS} aligned) "
            f"| {bar_info}"
        )


# ── Basket info builder ───────────────────────────────────────────────────────

def build_basket_info(config: dict) -> dict:
    """
    Extract basket_info dict from a loaded basket config.
    Returns {basket_id: {"tickers": [...], "symbols": [...], "sectors": [...]}}
    """
    members = config["members"]
    info = {}
    for bid, grp in members.groupby("basket_id"):
        grp = grp.sort_values("stock_position")
        info[int(bid)] = {
            "tickers":  grp["ticker"].tolist(),
            "symbols":  grp["symbol"].tolist(),
            "sectors":  grp["sector"].tolist() if "sector" in grp.columns else [],
            "companies": grp["company_name"].tolist() if "company_name" in grp.columns else [],
        }
    return info
