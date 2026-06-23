"""
signal_engine_daily.py — Per-basket signal evaluation on daily OHLCV.

Identical OR-gate logic to the 5-min signal_engine.py; the only
difference is it calls build_basket_daily_ohlc() instead of
build_basket_5m_ohlc().

Basket info dict format:
    {basket_id: {"tickers": [...], "symbols": [...], "sectors": [...]}}
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_daily import (
    EMA_FAST, EMA_SLOW, ROLLING_WINDOW, MIN_ROLLING_POINTS,
    POSITION_SIZE, MIN_ALIGNED_DAYS,
)
from data_manager_daily import build_basket_daily_ohlc, DailyTickerBuffer

# Reuse parent indicator engine unchanged
from indicators import IndicatorCache

logger = logging.getLogger(__name__)


class DailySignalEngine:
    """
    Manages daily indicator state for all baskets.
    Called once per day at EOD after today's bar is appended.
    """

    def __init__(self, basket_info: dict, ticker_buffers: dict,
                 instrument_map: dict, position_size: float = POSITION_SIZE):
        self.basket_info    = basket_info
        self.ticker_buffers = ticker_buffers   # {ticker: DailyTickerBuffer}
        self.instrument_map = instrument_map
        self.position_size  = position_size

        self._caches: dict      = {bid: IndicatorCache(bid) for bid in basket_info}
        self._basket_ohlc: dict = {}

        # Pre-warm indicator caches from already-seeded buffers so that
        # warmup_status() reflects the true readiness at startup.
        self._prewarm_caches()

    # ── Startup pre-warm ──────────────────────────────────────────────────────

    def _prewarm_caches(self):
        """
        Run indicator computation for every basket using the already-seeded
        ticker buffers.  Called once at __init__ so warmup_status() reports
        the correct number of ready baskets immediately after startup.
        """
        ready = 0
        for bid, info in self.basket_info.items():
            tickers = info["tickers"]
            ohlc = build_basket_daily_ohlc(self.ticker_buffers, tickers, self.position_size)
            if ohlc is None or ohlc.empty:
                continue
            self._basket_ohlc[bid] = ohlc
            if self._caches[bid].update(ohlc["Close"]):
                ready += 1
        logger.info(f"  Pre-warmed indicators: {ready}/{len(self.basket_info)} baskets ready")

    # ── EOD evaluation ────────────────────────────────────────────────────────

    def on_day_close(self, day_date: pd.Timestamp) -> dict:
        """
        Called once per day after today's bar is appended to all buffers.

        Returns:
            entry_signals : [(basket_id, entry_type), ...]
            sell_signals  : {basket_id: bool}
            basket_closes : {basket_id: float}
            indicator_ready: {basket_id: bool}
        """
        entry_signals = []
        sell_signals  = {}
        basket_closes = {}
        ready_map     = {}

        for bid, info in self.basket_info.items():
            tickers = info["tickers"]

            ohlc = build_basket_daily_ohlc(self.ticker_buffers, tickers, self.position_size)
            if ohlc is None or ohlc.empty:
                ready_map[bid] = False
                continue
            self._basket_ohlc[bid] = ohlc

            cache = self._caches[bid]
            ready = cache.update(ohlc["Close"])
            ready_map[bid] = ready
            if not ready:
                continue

            basket_closes[bid] = float(ohlc["Close"].iloc[-1])

            sig = cache.prev_bar_signal()
            if not sig:
                continue

            if sig.get("buy_signal"):
                entry_type = sig.get("entry_type", "")
                entry_signals.append((bid, entry_type))
                logger.info(f"  SIGNAL ENTRY B{bid} ({entry_type}) at {day_date.date()}")

            sell_signals[bid] = bool(sig.get("sell_signal", False))

        return {
            "entry_signals":    entry_signals,
            "sell_signals":     sell_signals,
            "basket_closes":    basket_closes,
            "indicator_ready":  ready_map,
        }

    def get_exec_prices(self, open_prices: dict) -> dict:
        """
        Build {basket_id: {ticker: open_price}} from today's open prices.
        open_prices: {ticker: float} fetched at 09:17 from 1-min bar.
        """
        result = {}
        for bid, info in self.basket_info.items():
            tickers = info["tickers"]
            prices  = {t: open_prices[t] for t in tickers if t in open_prices}
            if len(prices) == len(tickers):
                result[bid] = prices
        return result

    def get_basket_close_prices(self) -> dict:
        result = {}
        for bid, ohlc in self._basket_ohlc.items():
            if not ohlc.empty:
                result[bid] = float(ohlc["Close"].iloc[-1])
        return result

    def n_ready_baskets(self) -> int:
        return sum(1 for c in self._caches.values() if c.signals is not None)

    def warmup_status(self) -> str:
        return (f"{self.n_ready_baskets()}/{len(self.basket_info)} baskets ready "
                f"(need {MIN_ROLLING_POINTS} daily bars)")


def build_basket_info(config: dict) -> dict:
    """Extract {basket_id: {"tickers", "symbols", "sectors"}} from basket config."""
    members = config["members"]
    info = {}
    for bid, grp in members.groupby("basket_id"):
        grp = grp.sort_values("stock_position")
        info[int(bid)] = {
            "tickers":   grp["ticker"].tolist(),
            "symbols":   grp["symbol"].tolist(),
            "sectors":   grp["sector"].tolist() if "sector" in grp.columns else [],
            "companies": grp["company_name"].tolist() if "company_name" in grp.columns else [],
        }
    return info
