"""
symbol_mapper.py — Maps basket yfinance tickers (.NS) to pyintegrate
trading_symbol + token using conn.symbols (the built-in symbol master).

conn.symbols is a property on ConnectToIntegrate that downloads the NSE
symbol master and yields dicts:
    {"segment": "NSE", "trading_symbol": "RELIANCE-EQ", "token": "2885", ...}

We cache the full generator into a list once per process so we can search it
without re-downloading on every lookup.
"""

import logging
import re
from functools import lru_cache

from config import EXCHANGE

logger = logging.getLogger(__name__)

_EQUITY_SERIES = {"EQ", "BE", "SM", "N1", "N2", "N3", "N4", "NA", "NB", "NC"}

# Module-level cache: populated by load_symbol_master(conn) at startup.
_SYMBOL_CACHE: list[dict] = []


def load_symbol_master(conn) -> list[dict]:
    """
    Download and cache the full symbol master from ConnectToIntegrate.symbols.
    Call once at startup.  Filters to NSE equity series only.

    conn.symbols is a Generator[dict[str, str], None, None] that yields:
        {"segment": str, "trading_symbol": str, "token": str, ...}
    The exact extra keys may vary; only segment/trading_symbol/token are relied on.
    """
    global _SYMBOL_CACHE
    if _SYMBOL_CACHE:
        return _SYMBOL_CACHE

    logger.info("Downloading symbol master from broker…")
    rows = []
    for sym in conn.symbols:
        seg = str(sym.get("segment", "")).upper()
        ts  = str(sym.get("trading_symbol", "")).strip()
        if seg != EXCHANGE:
            continue
        # Keep only equity series (trading_symbol ends with -EQ, -BE, etc.)
        suffix = ts.rsplit("-", 1)[-1].upper() if "-" in ts else ""
        if suffix in _EQUITY_SERIES:
            rows.append({
                "segment":        seg,
                "trading_symbol": ts,
                "token":          str(sym.get("token", "")),
                "instrument_name": ts.rsplit("-", 1)[0].upper() if "-" in ts else ts.upper(),
                "series":         suffix,
            })

    _SYMBOL_CACHE = rows
    logger.info(f"Symbol master loaded: {len(rows)} NSE equity instruments")
    return rows


def _yf_to_nse_name(yf_ticker: str) -> str:
    """Strip .NS suffix and uppercase."""
    return re.sub(r"\.NS$", "", yf_ticker, flags=re.IGNORECASE).upper().strip()


def ticker_to_trading_symbol(yf_ticker: str) -> tuple[str, str] | None:
    """
    Return (trading_symbol, token) for a yfinance .NS ticker.
    Returns None if not found.  Prefers EQ series.
    Uses cached symbol master (load_symbol_master must be called first).
    """
    if not _SYMBOL_CACHE:
        raise RuntimeError("Symbol master not loaded. Call load_symbol_master(conn) first.")

    name = _yf_to_nse_name(yf_ticker)
    matches = [r for r in _SYMBOL_CACHE if r["instrument_name"] == name]
    if not matches:
        return None

    # Prefer EQ series
    eq = [r for r in matches if r["series"] == "EQ"]
    best = eq[0] if eq else matches[0]
    return best["trading_symbol"], best["token"]


def build_basket_instrument_map(basket_tickers: list[str]) -> dict:
    """
    Returns {yf_ticker: {"trading_symbol": str, "token": str}} for each ticker.
    Logs a warning for unmappable tickers.
    """
    result = {}
    for t in basket_tickers:
        mapped = ticker_to_trading_symbol(t)
        if mapped is None:
            logger.warning(f"  No NSE equity symbol found for ticker '{t}'")
            continue
        ts, tok = mapped
        result[t] = {"trading_symbol": ts, "token": tok}
        logger.debug(f"  {t} → {ts} (token={tok})")
    return result


def all_unique_tickers_from_configs(configs: list[dict]) -> list[str]:
    """Flatten all tickers from all basket configs (deduped)."""
    tickers = set()
    for cfg in configs:
        tickers.update(cfg["members"]["ticker"].tolist())
    return sorted(tickers)
