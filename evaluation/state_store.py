"""
state_store.py — Persist and restore portfolio state to/from JSON.

State file includes:
  - Open slot positions (basket_id, entry_time, quantities, prices, etc.)
  - Realized PnL
  - Full trade log
  - Per-ticker rolling buffer (last N 5-min bars) for indicator continuity
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import STATE_FILE, TRADE_LOG_FILE, IST

logger = logging.getLogger(__name__)


# ── JSON serialization helpers ────────────────────────────────────────────────

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return {str(k): v for k, v in obj.items()}
        return super().default(obj)


def _decode_ts(s: str | None) -> pd.Timestamp | None:
    if s is None:
        return None
    try:
        ts = pd.Timestamp(s)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        return ts
    except Exception:
        return None


# ── Save ──────────────────────────────────────────────────────────────────────

def _serialize_slot(slot: dict) -> dict:
    if slot is None:
        return None
    return {
        "basket_id":         slot["basket_id"],
        "entry_time":        slot["entry_time"].isoformat(),
        "entry_type":        slot["entry_type"],
        "tickers":           slot["tickers"],
        "quantities":        slot["quantities"],
        "entry_prices":      slot["entry_prices"],
        "investment":        slot["investment"],
        "capital_allocated": slot["capital_allocated"],
        "returns_ref_value": slot["returns_ref_value"],
        "returns_ref_time":  slot["returns_ref_time"].isoformat() if slot.get("returns_ref_time") else None,
    }


def _serialize_buffer(buf_df: pd.DataFrame) -> list:
    """Serialize a 5-min OHLCV DataFrame to a list of records."""
    if buf_df is None or buf_df.empty:
        return []
    df = buf_df.copy()
    # Drop any rows with NaT or non-datetime index before serializing
    if hasattr(df.index, 'notna'):
        df = df[df.index.notna()]
    if df.empty:
        return []
    def _safe_ts(x):
        try:
            return x.isoformat()
        except Exception:
            return None
    df.index = df.index.map(_safe_ts)
    df = df[df.index.notna()]   # drop any that failed
    if df.empty:
        return []
    return df.reset_index().rename(columns={"index": "datetime"}).to_dict(orient="records")


def save_state(portfolio_engine, ticker_buffers: dict,
               state_file: str = STATE_FILE, trade_log_file: str = TRADE_LOG_FILE):
    """
    Persist portfolio engine state + all ticker buffers.
    Creates parent dirs if needed.
    """
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)

    # Serialize slots
    slots_raw = [_serialize_slot(s) for s in portfolio_engine.slots]

    # Serialize basket_close_series
    bcs = {}
    for bid, series in portfolio_engine.basket_close_series.items():
        bcs[str(bid)] = {str(k): float(v) for k, v in series.items()}

    # Serialize ticker buffers — only save the last MAX_BARS bars.
    # On restart, the full 6-month warmup fetch replaces these anyway;
    # state buffer is only a fallback if the broker API is unavailable.
    from data_manager import TickerBuffer
    buffers_raw = {}
    for ticker, buf in ticker_buffers.items():
        df = buf.df
        if len(df) > TickerBuffer.MAX_BARS:
            df = df.iloc[-TickerBuffer.MAX_BARS:]
        buffers_raw[ticker] = _serialize_buffer(df)

    state = {
        "saved_at":        datetime.now(IST).isoformat(),
        "realized_pnl":    portfolio_engine.realized_pnl,
        "slots":           slots_raw,
        "trade_log":       portfolio_engine.trade_log,
        "basket_close_series": bcs,
        "ticker_buffers":  buffers_raw,
    }

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, cls=_Encoder)
    logger.info(f"State saved → {state_file}")

    # Also append/write trade log CSV
    if portfolio_engine.trade_log:
        _save_trade_log(portfolio_engine.trade_log, trade_log_file)


def _save_trade_log(trade_log: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(trade_log)
    # Flatten entry/exit prices (dicts) to strings for CSV
    for col in ["entry_prices", "exit_prices"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df.to_csv(path, index=False)
    logger.debug(f"Trade log → {path}")


# ── Load ──────────────────────────────────────────────────────────────────────

def _deserialize_slot(raw: dict | None) -> dict | None:
    if raw is None:
        return None
    return {
        "basket_id":         raw["basket_id"],
        "entry_time":        _decode_ts(raw["entry_time"]),
        "entry_type":        raw.get("entry_type", ""),
        "tickers":           raw["tickers"],
        "quantities":        raw["quantities"],
        "entry_prices":      raw["entry_prices"],
        "investment":        float(raw["investment"]),
        "capital_allocated": float(raw["capital_allocated"]),
        "returns_ref_value": float(raw.get("returns_ref_value", raw["investment"])),
        "returns_ref_time":  _decode_ts(raw.get("returns_ref_time")),
    }


def _deserialize_buffer(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(records)
    if "datetime" in df.columns:
        # Use errors='coerce' to convert any bad/legacy datetime strings (e.g. "0",
        # "NaT", integer indices) to NaT instead of crashing, then drop those rows.
        parsed = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
        df.index = parsed
        df.drop(columns=["datetime"], inplace=True, errors="ignore")
        df = df[df.index.notna()]   # drop rows with NaT timestamps
        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        # Ensure tz-aware IST index
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
        else:
            df.index = df.index.tz_convert(IST)
        df = df[df.index.notna()]   # drop any NaT introduced by tz_localize
    return df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce").dropna()


def load_state(portfolio_engine, ticker_buffers: dict,
               state_file: str = STATE_FILE) -> bool:
    """
    Restore portfolio engine and ticker buffers from state file.
    Returns True if state was loaded, False if no file found.
    """
    from data_manager import TickerBuffer

    if not Path(state_file).exists():
        logger.info(f"No state file at {state_file}, starting fresh")
        return False

    try:
        with open(state_file) as f:
            state = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return False

    portfolio_engine.realized_pnl = float(state.get("realized_pnl", 0.0))
    portfolio_engine.slots = [
        _deserialize_slot(s) for s in state.get("slots", [None, None])
    ]
    # Pad/trim slots to n_slots
    while len(portfolio_engine.slots) < portfolio_engine.n_slots:
        portfolio_engine.slots.append(None)
    portfolio_engine.trade_log = state.get("trade_log", [])

    # Restore basket close series
    bcs_raw = state.get("basket_close_series", {})
    for bid_str, kv in bcs_raw.items():
        bid = int(bid_str)
        s = pd.Series({pd.Timestamp(k): float(v) for k, v in kv.items()})
        portfolio_engine.basket_close_series[bid] = s

    # Restore ticker buffers — but ONLY if the freshly-seeded warmup buffer is
    # empty or shorter.  The warmup already pulled the full broker history (up
    # to 6 months), which is always richer than what was saved in state.
    # Overwriting it with the old state would throw away bars and break readiness.
    buffers_raw = state.get("ticker_buffers", {})
    for ticker, records in buffers_raw.items():
        live_buf = ticker_buffers.get(ticker)
        live_bars = live_buf.n_bars if live_buf is not None else 0
        state_df  = _deserialize_buffer(records)
        state_bars = len(state_df)

        if live_bars == 0 and state_bars > 0:
            # No warmup data available — fall back to state
            if ticker not in ticker_buffers:
                ticker_buffers[ticker] = TickerBuffer(ticker)
            ticker_buffers[ticker].seed(state_df)
            logger.debug(f"  {ticker}: restored {state_bars} bars from state (no live warmup)")
        elif state_bars > live_bars:
            # State somehow has more bars (shouldn't happen after 6-month fetch)
            ticker_buffers[ticker].seed(state_df)
            logger.debug(f"  {ticker}: restored {state_bars} bars from state (state richer than warmup {live_bars})")
        else:
            # Live warmup is richer — keep it, discard state buffer
            logger.debug(f"  {ticker}: keeping {live_bars} live warmup bars (state had {state_bars})")

    saved_at = state.get("saved_at", "unknown")
    active = sum(1 for s in portfolio_engine.slots if s is not None)
    logger.info(f"State loaded from {state_file} (saved {saved_at})  "
                f"active_slots={active}  realized_pnl={portfolio_engine.realized_pnl:.0f}")
    return True
