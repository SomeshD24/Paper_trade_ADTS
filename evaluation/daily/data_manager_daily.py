"""
data_manager_daily.py — Daily OHLCV data via IntegrateData.

Key differences from the 5-min data_manager:
  - Uses conn.TIMEFRAME_TYPE_DAY  → daily bars directly, no resampling.
  - Warmup: ~1300 calendar days (covers 756+ trading days for regression).
  - EOD update: called at 15:35; appends today's completed daily bar.
  - Open-price fetch: called at 09:17 via 1-min bar for execution price.
  - Basket OHLC built from daily buffers (identical equal-weight logic).
"""

import logging
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd

from config_daily import (
    IST, EXCHANGE, MARKET_OPEN_H, MARKET_OPEN_M,
    WARMUP_TRADING_DAYS, WARMUP_CALENDAR_DAYS,
    HISTORICAL_TF_DAY, HISTORICAL_TF_MIN,
    MIN_ALIGNED_DAYS, POSITION_SIZE,
)

logger = logging.getLogger(__name__)


# ── Response parser (reused from 5-min engine) ────────────────────────────────

def _parse_hist_row(row: dict) -> dict | None:
    """Normalise one dict from ic.historical_data() — handles varied key names."""
    ts = (row.get("datetime") or row.get("time") or
          row.get("date")     or row.get("ts"))
    if ts is None:
        return None

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


def _records_to_df(records: list, tz=IST) -> pd.DataFrame:
    """Convert parsed row-list to a DatetimeIndex OHLCV DataFrame (IST-aware)."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df["ts"])
    df.drop(columns=["ts"], inplace=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert(tz)
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"]).sort_index()


# ── Daily OHLCV fetch ─────────────────────────────────────────────────────────

def fetch_daily_historical(
    ic,
    conn,
    trading_symbol: str,
    from_dt: datetime,
    to_dt: datetime,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars via IntegrateData.historical_data() with
    timeframe = conn.TIMEFRAME_TYPE_DAY.

    Returns DataFrame with DatetimeIndex (IST) and Open/High/Low/Close/Volume.
    """
    if hasattr(from_dt, "tzinfo") and from_dt.tzinfo is not None:
        from_dt = from_dt.replace(tzinfo=None)
    if hasattr(to_dt, "tzinfo") and to_dt.tzinfo is not None:
        to_dt = to_dt.replace(tzinfo=None)

    try:
        gen = ic.historical_data(
            exchange=conn.EXCHANGE_TYPE_NSE,
            trading_symbol=trading_symbol,
            timeframe=conn.TIMEFRAME_TYPE_DAY,
            start=from_dt,
            end=to_dt,
        )
    except Exception as e:
        logger.error(f"daily historical_data failed for {trading_symbol}: {e}")
        return pd.DataFrame()

    records = [_parse_hist_row(r) for r in gen]
    records = [r for r in records if r]
    return _records_to_df(records)


def fetch_todays_open(
    ic,
    conn,
    trading_symbol: str,
) -> float | None:
    """
    Fetch today's opening price via the first completed 1-min bar (09:15–09:16).
    Called at 09:17 IST for paper order execution.
    Returns the Open of today's first 1-min bar, or None on failure.
    """
    now     = datetime.now(IST)
    from_dt = now.replace(
        hour=MARKET_OPEN_H, minute=MARKET_OPEN_M - 1, second=0, microsecond=0,
        tzinfo=None,
    )
    to_dt = now.replace(tzinfo=None)

    try:
        gen = ic.historical_data(
            exchange=conn.EXCHANGE_TYPE_NSE,
            trading_symbol=trading_symbol,
            timeframe=conn.TIMEFRAME_TYPE_MIN,
            start=from_dt,
            end=to_dt,
        )
    except Exception as e:
        logger.error(f"fetch_todays_open failed for {trading_symbol}: {e}")
        return None

    records = [_parse_hist_row(r) for r in gen]
    records = [r for r in records if r]
    if not records:
        return None

    df = _records_to_df(records)
    if df.empty:
        return None

    # Filter to 09:15 bar only
    market_open_t = dtime(MARKET_OPEN_H, MARKET_OPEN_M)
    mask = df.index.time == market_open_t
    if mask.any():
        return float(df.loc[mask]["Open"].iloc[0])

    # Fallback: first bar of the day
    return float(df["Open"].iloc[0])


def fetch_live_ltp(
    ic,
    conn,
    trading_symbol: str,
) -> float | None:
    """
    Fetch the latest traded price (LTP) using the most recent completed 1-min bar.
    Called during market hours to compute unrealized PnL.
    Returns the Close of the last available 1-min bar, or None on failure.
    """
    now     = datetime.now(IST)
    from_dt = (now - timedelta(minutes=10)).replace(tzinfo=None)
    to_dt   = now.replace(tzinfo=None)

    try:
        gen = ic.historical_data(
            exchange=conn.EXCHANGE_TYPE_NSE,
            trading_symbol=trading_symbol,
            timeframe=conn.TIMEFRAME_TYPE_MIN,
            start=from_dt,
            end=to_dt,
        )
    except Exception as e:
        logger.error(f"fetch_live_ltp failed for {trading_symbol}: {e}")
        return None

    records = [_parse_hist_row(r) for r in gen]
    records = [r for r in records if r]
    if not records:
        return None

    df = _records_to_df(records)
    if df.empty:
        return None

    # Return close of the latest bar
    return float(df["Close"].iloc[-1])


# ── Rolling daily buffer ──────────────────────────────────────────────────────

class DailyTickerBuffer:
    """
    Rolling buffer of completed daily OHLCV bars for one ticker.
    Keeps at most MAX_BARS bars (trims oldest on append).
    """
    MAX_BARS = WARMUP_TRADING_DAYS + 50

    def __init__(self, ticker: str):
        self.ticker = ticker
        self._df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def seed(self, df_daily: pd.DataFrame):
        self._df = df_daily.copy().tail(self.MAX_BARS)

    def append_bar(self, bar: dict):
        """
        Append one completed daily bar.
        bar keys: datetime (pd.Timestamp), Open, High, Low, Close, Volume.
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

    def append_today(self, ic, conn, trading_symbol: str):
        """
        Fetch and append today's completed daily bar.
        Called at EOD (15:35 IST) after market close.
        Returns True if a new bar was appended.
        """
        now  = datetime.now(IST)
        from_dt = now - timedelta(days=3)
        to_dt   = now

        df = fetch_daily_historical(ic, conn, trading_symbol, from_dt, to_dt)
        if df.empty:
            return False

        today = now.date()
        today_bars = df[df.index.date == today]
        if today_bars.empty:
            logger.debug(f"{trading_symbol}: today's daily bar not yet available")
            return False

        row = today_bars.iloc[-1]
        bar = {
            "datetime": today_bars.index[-1],
            "Open":   float(row["Open"]),
            "High":   float(row["High"]),
            "Low":    float(row["Low"]),
            "Close":  float(row["Close"]),
            "Volume": float(row["Volume"]),
        }
        self.append_bar(bar)
        return True

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def n_bars(self) -> int:
        return len(self._df)


# ── Basket OHLC builder (daily) ───────────────────────────────────────────────

def build_basket_daily_ohlc(
    ticker_buffers: dict,       # {yf_ticker: DailyTickerBuffer}
    tickers: list[str],
    position_size: float = POSITION_SIZE,
) -> pd.DataFrame | None:
    """
    Equal-weight basket daily OHLC — EXACT port of build_equal_weight_basket_ohlc()
    from strategy1_dualbasket2.py.

    Key invariants (matching the backtest):
      1. Build a MultiIndex-style panel by concat of all ticker OHLCV DataFrames.
      2. panel.dropna() → inner join on ALL fields across ALL tickers simultaneously.
         This gives the exact same common-start-date as the backtest.
      3. first_close = panel[(t, 'Close')].iloc[0]  ← first row of JOINT panel.
      4. quantities[t] = (1/N * position_size) / first_close[t]
      5. basket field = sum(panel[(t, field)] * quantities[t])
      6. Volume = simple sum (no weighting).

    Returns None if insufficient aligned data.
    """
    if not all(t in ticker_buffers for t in tickers):
        return None

    fields = ["Open", "High", "Low", "Close", "Volume"]
    dfs    = {t: ticker_buffers[t].df for t in tickers}
    if any(d.empty for d in dfs.values()):
        return None

    # Build full MultiIndex panel — exactly as in backtest:
    #   loaded = [load_ohlc(t)[fields].rename(columns={c: (t,c) for c in fields}) for t in tickers]
    #   panel  = pd.concat(loaded, axis=1).dropna()
    loaded = [
        dfs[t][fields].rename(columns={c: (t, c) for c in fields})
        for t in tickers
    ]
    panel = pd.concat(loaded, axis=1).dropna()

    if len(panel) < MIN_ALIGNED_DAYS:
        return None

    import json
    from pathlib import Path
    
    # Load EXACT backtest quantities from JSON (to prevent rolling window drift)
    q_file = Path("state/basket_quantities_6.json")
    if not q_file.exists():
        return None
        
    with open(q_file) as f:
        all_q = json.load(f)
        
    quantities = None
    for bid, q_map in all_q.items():
        if set(q_map.keys()) == set(tickers):
            quantities = q_map
            break
            
    if quantities is None:
        return None  # Missing fixed quantities for this basket

    basket = pd.DataFrame(index=panel.index)
    for field in ["Open", "High", "Low", "Close"]:
        basket[field] = sum(panel[(t, field)] * quantities[t] for t in tickers)
    basket["Volume"] = sum(panel[(t, "Volume")] for t in tickers)

    basket = basket.dropna().sort_index()

    basket.attrs["component_close"] = pd.DataFrame(
        {t: panel[(t, "Close")] for t in tickers}, index=basket.index
    ).sort_index()
    basket.attrs["component_open"] = pd.DataFrame(
        {t: panel[(t, "Open")] for t in tickers}, index=basket.index
    ).sort_index()
    return basket


# ── Warmup loader ─────────────────────────────────────────────────────────────

def warmup_daily_buffers(
    ic,
    conn,
    instrument_map: dict,
    calendar_days: int = WARMUP_CALENDAR_DAYS,
) -> dict:
    """
    Fetch historical daily bars for all tickers and seed DailyTickerBuffer objects.
    Returns {yf_ticker: DailyTickerBuffer}.
    """
    now     = datetime.now(IST)
    from_dt = now - timedelta(days=calendar_days)
    to_dt   = now

    buffers = {}
    for ticker, info in instrument_map.items():
        ts_sym = info["trading_symbol"]
        logger.info(f"  Warming up daily {ticker} ({ts_sym})…")
        df = fetch_daily_historical(ic, conn, ts_sym, from_dt, to_dt)
        buf = DailyTickerBuffer(ticker)
        if df.empty:
            logger.warning(f"  {ticker}: no daily history returned")
        else:
            buf.seed(df)
            logger.info(f"  {ticker}: {buf.n_bars} daily bars seeded")
        buffers[ticker] = buf

    return buffers
