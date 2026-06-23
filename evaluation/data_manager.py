"""
data_manager.py — Fetch 1-min candles from pyintegrate IntegrateData,
resample to 5-min, maintain a per-ticker rolling buffer.

Key API facts (from official docs):
  from integrate import ConnectToIntegrate, IntegrateData
  ic = IntegrateData(conn)
  gen = ic.historical_data(
      exchange=conn.EXCHANGE_TYPE_NSE,   # "NSE"
      trading_symbol="RELIANCE-EQ",
      timeframe=conn.TIMEFRAME_TYPE_MIN, # "minute"
      start=datetime_obj,
      end=datetime_obj,
  )
  # gen yields dicts; we normalise key names to Open/High/Low/Close/Volume

Resampling (1-min → 5-min):
  open   = first bar of window
  high   = max
  low    = min
  close  = last bar of window
  volume = sum
  timestamp = window open time (09:15, 09:20, …)
"""

import logging
import time
from datetime import datetime, timedelta, time as dtime

import pandas as pd
import numpy as np

from config import (
    BAR_MINUTES, IST,
    MARKET_OPEN_H, MARKET_OPEN_M, MARKET_CLOSE_H, MARKET_CLOSE_M,
    EXCHANGE, WARMUP_BARS, WARMUP_DAYS, MIN_ALIGNED_BARS, MAX_FETCH_DAYS,
)

logger = logging.getLogger(__name__)


# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_1m_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a 1-min OHLCV DataFrame (DatetimeIndex, IST-aware or naive)
    to 5-min bars aligned to market open (09:15).

    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    Only includes bars whose open time falls in [09:15, 15:25].
    """
    if df_1m.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = df_1m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert(IST)
    else:
        df.index = df.index.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")

    # Filter to market hours [09:15, 15:30)
    market_open  = dtime(MARKET_OPEN_H,  MARKET_OPEN_M)
    market_close = dtime(MARKET_CLOSE_H, MARKET_CLOSE_M)
    mask = (df.index.time >= market_open) & (df.index.time < market_close)
    df = df[mask]
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Anchor origin to 09:15 of the first date present
    anchor = df.index[0].replace(
        hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0
    )
    ohlcv = df.resample(
        f"{BAR_MINUTES}min",
        closed="left",
        label="left",
        origin=anchor,
    ).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna(subset=["Close"])

    # Keep only bars within market hours
    mask5 = (ohlcv.index.time >= market_open) & (ohlcv.index.time < market_close)
    return ohlcv[mask5]


# ── Raw historical fetch via IntegrateData ─────────────────────────────────────

def _parse_hist_row(row: dict) -> dict | None:
    """
    Normalise a single dict from ic.historical_data() generator.

    The broker yields dicts whose exact keys aren't documented publicly.
    Common candidates: datetime/time/date, open/o, high/h, low/l, close/c,
    volume/v/vol.  We try both capitalized and lower-case.
    Returns None if any essential field is missing.
    """
    # ── timestamp ─────────────────────────────────────────────────────────────
    ts = (row.get("datetime") or row.get("time") or
          row.get("date")     or row.get("ts"))
    if ts is None:
        return None

    # ── OHLCV ─────────────────────────────────────────────────────────────────
    def _f(keys):
        for k in keys:
            v = row.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    o = _f(["open",   "Open",   "o"])
    h = _f(["high",   "High",   "h"])
    l = _f(["low",    "Low",    "l"])
    c = _f(["close",  "Close",  "c"])
    v = _f(["volume", "Volume", "v", "vol"])

    if None in (o, h, l, c):
        return None

    return {"ts": ts, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v or 0.0}


def fetch_1m_historical(
    ic,                     # IntegrateData instance
    conn,                   # ConnectToIntegrate instance (for constants)
    trading_symbol: str,
    from_dt: datetime,
    to_dt: datetime,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Fetch 1-min candles from IntegrateData.historical_data() for one symbol.

    Parameters
    ----------
    ic            : IntegrateData instance
    conn          : ConnectToIntegrate instance (provides EXCHANGE_TYPE_NSE,
                    TIMEFRAME_TYPE_MIN constants)
    trading_symbol: e.g. "RELIANCE-EQ"
    from_dt / to_dt: datetime objects (naive or IST-aware)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (IST-aware) and columns Open/High/Low/Close/Volume.
    Empty DataFrame on failure.
    """
    # Strip tz for broker call (it expects naive datetime)
    if hasattr(from_dt, "tzinfo") and from_dt.tzinfo is not None:
        from_dt = from_dt.replace(tzinfo=None)
    if hasattr(to_dt, "tzinfo") and to_dt.tzinfo is not None:
        to_dt = to_dt.replace(tzinfo=None)

    records = []
    for attempt in range(max_retries):
        try:
            gen = ic.historical_data(
                exchange=conn.EXCHANGE_TYPE_NSE,
                trading_symbol=trading_symbol,
                timeframe=conn.TIMEFRAME_TYPE_MIN,   # "minute"
                start=from_dt,
                end=to_dt,
            )
            # Drain the generator inside the try block to catch errors during iteration
            for row in gen:
                parsed = _parse_hist_row(row)
                if parsed:
                    records.append(parsed)
            # If we get here without an exception, the fetch succeeded (even if empty)
            break
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt
                    logger.warning(f"Rate limited (429) for {trading_symbol}. Retrying in {sleep_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
            logger.error(f"historical_data failed for {trading_symbol}: {e}")
            return pd.DataFrame()

    if not records:
        logger.debug(f"No 1-min data returned for {trading_symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df["ts"])
    df.drop(columns=["ts"], inplace=True)

    # Localise to IST
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert(IST)

    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return df.dropna(subset=["Close"]).sort_index()


# ── Rolling buffer ─────────────────────────────────────────────────────────────

class TickerBuffer:
    """
    Maintains a rolling buffer of completed 5-min bars for one ticker.
    Keeps at most MAX_BARS bars (oldest trimmed on append).
    """
    MAX_BARS = WARMUP_BARS + 50

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._df: pd.DataFrame = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )

    def seed(self, df_5m: pd.DataFrame):
        """Initialize buffer with ALL historical 5-min bars (no truncation at warmup).
        The full history is needed for the rolling-window indicator.
        Only live append() enforces MAX_BARS to cap memory."""
        self._df = df_5m.copy()  # keep every bar fetched

    def append_bar(self, bar: dict):
        """
        Append one completed 5-min bar.
        bar must contain: datetime (pd.Timestamp), Open, High, Low, Close, Volume.
        """
        ts = bar.get("datetime")
        if ts is None:
            return
        new_row = pd.DataFrame(
            [[bar["Open"], bar["High"], bar["Low"], bar["Close"], bar.get("Volume", 0)]],
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=[ts],
        )
        self._df = pd.concat([self._df, new_row])
        self._df = self._df[~self._df.index.duplicated(keep="last")]
        self._df.sort_index(inplace=True)
        if len(self._df) > self.MAX_BARS:
            self._df = self._df.iloc[-self.MAX_BARS:]

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def n_bars(self) -> int:
        return len(self._df)


# ── Basket OHLC builder ────────────────────────────────────────────────────────

def build_basket_5m_ohlc(
    ticker_buffers: dict,       # {yf_ticker: TickerBuffer}
    tickers: list[str],
    position_size: float,
) -> pd.DataFrame | None:
    """
    Build an equal-weight basket 5-min OHLC DataFrame from per-ticker buffers.
    Returns None if insufficient aligned bars.
    """
    if not all(t in ticker_buffers for t in tickers):
        return None

    dfs = {}
    for t in tickers:
        buf = ticker_buffers[t].df
        if buf.empty:
            return None
        dfs[t] = buf

    # Forward-fill up to 1 bar to bridge minor per-ticker gaps (e.g. one 5-min
    # window with no trades for a single stock) before taking the intersection.
    # This prevents losing entire aligned rows due to one ticker's missing bar.
    closes_raw = pd.DataFrame({t: dfs[t]["Close"] for t in tickers})
    closes = closes_raw.ffill(limit=1).dropna()
    if len(closes) < MIN_ALIGNED_BARS:
        return None

    first_close = closes.iloc[0].astype(float)
    quantities  = (1 / len(tickers)) * position_size / first_close  # fractional shares

    common_idx = closes.index
    basket = pd.DataFrame(index=common_idx)

    for field in ["Open", "High", "Low", "Close"]:
        basket[field] = sum(
            dfs[t][field].reindex(common_idx).ffill(limit=1).astype(float) * quantities[t]
            for t in tickers
        )
    basket["Volume"] = sum(
        dfs[t]["Volume"].reindex(common_idx).fillna(0).astype(float)
        for t in tickers
    )
    basket.dropna(inplace=True)

    # Attach component data for correlation
    basket.attrs["component_close"] = pd.DataFrame(
        {t: dfs[t]["Close"].reindex(basket.index) for t in tickers}
    )
    basket.attrs["component_open"] = pd.DataFrame(
        {t: dfs[t]["Open"].reindex(basket.index) for t in tickers}
    )
    return basket


# ── Warm-up loader ─────────────────────────────────────────────────────────────

def warmup_ticker_buffers(
    ic,
    conn,
    instrument_map: dict,       # {yf_ticker: {"trading_symbol": str, "token": str}}
    warmup_days: int = WARMUP_DAYS,
) -> dict:
    """
    Fetch historical 1-min data for all tickers, resample to 5-min,
    and seed TickerBuffer objects.

    Returns {yf_ticker: TickerBuffer}.
    """
    now     = datetime.now(IST)
    # Fetch the full broker history window (6 months = 180 calendar days).
    # The broker stores up to MAX_FETCH_DAYS of 1-min data; we pull all of it
    # so the rolling-window indicator has the maximum possible history.
    fetch_days = MAX_FETCH_DAYS
    from_dt = now - timedelta(days=fetch_days)
    to_dt   = now
    logger.info(
        f"Warmup: fetching full {fetch_days}-day broker history "
        f"({from_dt.strftime('%Y-%m-%d')} -> {to_dt.strftime('%Y-%m-%d')})"
    )

    buffers: dict[str, TickerBuffer] = {}
    for ticker, info in instrument_map.items():
        trading_symbol = info["trading_symbol"]
        logger.info(f"  Warming up {ticker} ({trading_symbol})...")

        df_1m = fetch_1m_historical(ic, conn, trading_symbol, from_dt, to_dt)
        time.sleep(0.5)  # Pause between warmup fetches to avoid rate limits
        buf   = TickerBuffer(ticker)
        if df_1m.empty:
            logger.warning(f"  {ticker}: no historical data returned")
        else:
            df_5m = resample_1m_to_5m(df_1m)
            buf.seed(df_5m)
            date_range = (
                f"{df_5m.index[0].strftime('%Y-%m-%d')} -> {df_5m.index[-1].strftime('%Y-%m-%d')}"
                if not df_5m.empty else "empty"
            )
            logger.info(
                f"  {ticker}: {buf.n_bars} 5-min bars seeded  [{date_range}]"
            )
        buffers[ticker] = buf

    return buffers


# ── Live polling ───────────────────────────────────────────────────────────────

class LiveBarPoller:
    """
    Polls IntegrateData.historical_data() for new 1-min bars since last poll,
    assembles completed 5-min bars, and appends them to TickerBuffers.
    """

    def __init__(self, ic, conn, instrument_map: dict, ticker_buffers: dict):
        self.ic             = ic
        self.conn           = conn
        self.instrument_map = instrument_map
        self.ticker_buffers = ticker_buffers
        self._last_poll: dict[str, datetime] = {}

    @staticmethod
    def _to_ist_ts(dt) -> pd.Timestamp:
        """
        Convert any datetime-like value to a tz-aware IST pd.Timestamp.
        Handles: naive datetime, tz-aware datetime, naive pd.Timestamp,
        and tz-aware pd.Timestamp.  Never raises a TypeError on comparison
        with the IST-aware index produced by resample_1m_to_5m().
        """
        ts = pd.Timestamp(dt) if not isinstance(dt, pd.Timestamp) else dt
        if ts.tzinfo is None:
            return ts.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
        return ts.tz_convert(IST)

    def poll_and_update(self, now: datetime) -> dict[str, pd.DataFrame]:
        """
        Fetch new 1-min bars for all tickers since last poll.
        Assembles complete 5-min bars and appends to buffers.
        Returns {ticker: completed_5min_bars_df}.
        """
        new_5m: dict[str, pd.DataFrame] = {}

        for ticker, info in self.instrument_map.items():
            ts_sym = info["trading_symbol"]

            # Poll from (last_poll - 1 min) to now to avoid gaps
            last_ts = self._last_poll.get(
                ticker,
                now - timedelta(minutes=BAR_MINUTES * 3)
            )
            from_dt = last_ts - timedelta(minutes=1)

            df_1m = fetch_1m_historical(self.ic, self.conn, ts_sym, from_dt, now)
            time.sleep(0.2)  # Small pause to avoid rate limits during polling
            if df_1m.empty:
                continue

            df_5m = resample_1m_to_5m(df_1m)
            if df_5m.empty:
                continue

            # Only keep bars whose window is fully closed (bar_open + 5min <= now)
            now_ts   = self._to_ist_ts(now)
            complete = df_5m[
                df_5m.index + pd.Timedelta(minutes=BAR_MINUTES) <= now_ts
            ]
            if complete.empty:
                continue

            buf = self.ticker_buffers[ticker]
            for ts, row in complete.iterrows():
                bar = {
                    "datetime": ts,
                    "Open":   float(row["Open"]),
                    "High":   float(row["High"]),
                    "Low":    float(row["Low"]),
                    "Close":  float(row["Close"]),
                    "Volume": float(row["Volume"]),
                }
                buf.append_bar(bar)

            self._last_poll[ticker] = complete.index[-1]
            new_5m[ticker] = complete

        return new_5m
