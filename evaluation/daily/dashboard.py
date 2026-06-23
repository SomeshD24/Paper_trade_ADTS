"""
dashboard.py — ADTS Daily Paper Trader • Live Dashboard

Fixes applied:
  • days_back = 1300 (matches WARMUP_CALENDAR_DAYS — enough for 756+ trading days)
  • LTP priority: 1-min close (during market) → daily close (after close / fallback)
  • After 15:30 IST: always shows daily close as LTP
  • Auto-refresh via JS window.location.reload() (avoids Python 3.14 event-loop bug)
  • Full engine state: open slots, pending orders, basket members, indicators
  • Trade log shows basket members + entry/exit prices per ticker
  • Basket chart uses exact build_equal_weight_basket_ohlc() formula
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# ── Engine Process Management ──────────────────────────────────────────────────

PID_FILE = Path(__file__).resolve().parent.parent.parent / "state" / "engine.pid"

def is_engine_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode()
        return str(pid) in output
    except Exception:
        return False

def start_engine():
    if not is_engine_running():
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_daily.py")]
        CREATE_NO_WINDOW = 0x08000000
        p = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent.parent.parent), creationflags=CREATE_NO_WINDOW)
        with open(PID_FILE, "w") as f:
            f.write(str(p.pid))

def stop_engine():
    if PID_FILE.exists():
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            subprocess.run(f'taskkill /F /PID {pid}', shell=True)
            PID_FILE.unlink()
        except Exception:
            pass

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent      # …/evaluation/daily/
_ROOT = _HERE.parent                          # …/evaluation/
_PROJ = _ROOT.parent                          # project root (stock_trend_predictor)
for p in [str(_HERE), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ADTS Daily Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp{background:#0d1117;color:#e6edf3;}
  .block-container{padding-top:1rem;}
  header[data-testid="stHeader"]{background:transparent;}
  section[data-testid="stSidebar"]>div{background:#0d1117;border-right:1px solid #21262d;}
  button[kind="secondary"]{background:#161b22!important;border:1px solid #30363d!important;color:#e6edf3!important;}
  button[kind="secondary"]:hover{border-color:#58a6ff!important;}

  .mc{background:linear-gradient(135deg,#161b22,#1c2128);border:1px solid #30363d;
      border-radius:12px;padding:16px 18px;text-align:center;margin-bottom:4px;}
  .mc:hover{border-color:#58a6ff;}
  .ml{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;}
  .mv{font-size:1.45rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
  .ms{font-size:.75rem;color:#8b949e;margin-top:3px;}
  .pos{color:#3fb950;} .neg{color:#f85149;} .neu{color:#58a6ff;}

  .sh{font-size:.88rem;font-weight:600;color:#8b949e;text-transform:uppercase;
      letter-spacing:.1em;border-bottom:1px solid #21262d;padding-bottom:5px;margin:18px 0 10px 0;}
  .tag{background:#21262d;border-radius:4px;padding:2px 7px;font-size:.75rem;color:#8b949e;margin-right:4px;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, prefix="₹", sign=True) -> str:
    s = "+" if (sign and v >= 0) else ""
    return f"{s}{prefix}{v:,.0f}" if prefix else f"{s}{v:,.0f}"

def _pct(v: float) -> str:
    return f"{'+' if v>=0 else ''}{v:.2f}%"

def _cls(v: float) -> str:
    return "pos" if v >= 0 else "neg"

def _metric(col, label: str, val_html: str, sub: str = ""):
    col.markdown(
        f'<div class="mc"><div class="ml">{label}</div>'
        f'<div class="mv">{val_html}</div>'
        + (f'<div class="ms">{sub}</div>' if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# State / config loaders
# ──────────────────────────────────────────────────────────────────────────────

def _state_path() -> Path | None:
    for p in [
        _ROOT / "state" / "daily_portfolio_state.json",
        _PROJ / "state" / "daily_portfolio_state.json",
        Path("state/daily_portfolio_state.json"),
    ]:
        if p.exists():
            return p
    return None


def _load_state() -> dict | None:
    p = _state_path()
    if p is None:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_trade_log() -> pd.DataFrame:
    for p in [
        _ROOT / "state" / "daily_trade_log.csv",
        _PROJ / "state" / "daily_trade_log.csv",
    ]:
        if p.exists():
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _basket_info(csv_path: str, size: int) -> dict:
    try:
        df = pd.read_csv(csv_path)
        if "basket_size" in df.columns:
            df = df[df["basket_size"] == size]
        info: dict = {}
        for bid, grp in df.groupby("basket_id"):
            grp = grp.sort_values("stock_position")
            info[int(bid)] = {
                "tickers":   grp["ticker"].tolist(),
                "symbols":   grp["symbol"].tolist() if "symbol" in grp.columns else grp["ticker"].tolist(),
                "companies": grp["company_name"].tolist() if "company_name" in grp.columns else [],
                "sectors":   grp["sector"].tolist()       if "sector"       in grp.columns else [],
            }
        return info
    except Exception as e:
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Broker connection
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _broker():
    """Return (conn, ic, err_msg) — conn/ic may be None if unavailable."""
    try:
        from dotenv import load_dotenv, find_dotenv
        env_file = find_dotenv(filename=".env", raise_error_if_not_found=False)
        if not env_file:
            # Search upwards from project root
            for candidate in [_PROJ / ".env", _ROOT / ".env", _HERE / ".env"]:
                if candidate.exists():
                    env_file = str(candidate)
                    break
        if env_file:
            load_dotenv(env_file, override=True)
    except ImportError:
        pass

    api_token  = os.environ.get("INTEGRATE_API_TOKEN",  "")
    api_secret = os.environ.get("INTEGRATE_API_SECRET", "")
    uid        = os.environ.get("INTEGRATE_UID",              "")
    actid      = os.environ.get("INTEGRATE_ACTID",            "")
    api_sk     = os.environ.get("INTEGRATE_API_SESSION_KEY",  "")
    ws_sk      = os.environ.get("INTEGRATE_WS_SESSION_KEY",   "")
    saved_at   = os.environ.get("INTEGRATE_SESSION_SAVED_AT", "")

    if not api_token or not api_secret:
        return None, None, "No API credentials in .env"

    try:
        from integrate import ConnectToIntegrate, IntegrateData
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        conn = ConnectToIntegrate()
        conn._verify = False

        if uid and saved_at:
            try:
                age_h = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds() / 3600
                if age_h < 23:
                    conn.set_session_keys(uid, actid, api_sk, ws_sk)
                    return conn, IntegrateData(conn), None
            except Exception:
                pass

        return None, None, "AUTH_REQUIRED"
    except Exception as e:
        return None, None, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Symbol map
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _instrument_map(tickers: tuple) -> tuple[dict, str]:
    """Returns (map, error_msg)."""
    conn, _, _ = _broker()
    if conn is None:
        return {t: t for t in tickers}, "Broker offline"
    try:
        from symbol_mapper import load_symbol_master, build_basket_instrument_map
        load_symbol_master(conn)
        m = build_basket_instrument_map(list(tickers))
        for t in tickers:
            if t not in m:
                m[t] = {"trading_symbol": t}
        return m, None
    except Exception as e:
        return {t: {"trading_symbol": t} for t in tickers}, str(e)


def _sym(imap: dict, ticker: str) -> str:
    entry = imap.get(ticker, {})
    if isinstance(entry, dict):
        return entry.get("trading_symbol", ticker)
    return ticker


# ──────────────────────────────────────────────────────────────────────────────
# Data fetchers  (days_back=1300 matches WARMUP_CALENDAR_DAYS for 756+ bars)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_rows(gen) -> list:
    """Consume a historical_data generator and parse all rows."""
    from data_manager_daily import _parse_hist_row
    rows = []
    try:
        for r in gen:
            parsed = _parse_hist_row(r)
            if parsed:
                rows.append(parsed)
    except Exception:
        pass
    return rows


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Convert parsed row list → IST-indexed OHLCV DataFrame."""
    from data_manager_daily import _records_to_df
    return _records_to_df(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_daily(trading_symbol: str, days_back: int = 1300) -> tuple[pd.DataFrame, str | None]:
    """
    Fetch daily OHLCV bars. Returns (df, error_msg).
    days_back=1300 gives 756+ trading days, matching engine warmup.
    """
    conn, ic, err = _broker()
    if ic is None:
        return pd.DataFrame(), f"Broker: {err}"
    try:
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        now     = datetime.now(IST).replace(tzinfo=None)
        from_dt = now - timedelta(days=days_back)
        gen = ic.historical_data(
            exchange=conn.EXCHANGE_TYPE_NSE,
            trading_symbol=trading_symbol,
            timeframe=conn.TIMEFRAME_TYPE_DAY,
            start=from_dt,
            end=now,
        )
        rows = _parse_rows(gen)
        if not rows:
            return pd.DataFrame(), f"{trading_symbol}: 0 daily rows returned"
        df = _rows_to_df(rows)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_1min(trading_symbol: str, days_back: int = 1) -> tuple[pd.DataFrame, str | None]:
    """Fetch 1-min intraday bars. Returns (df, error_msg). Cached 60s."""
    conn, ic, err = _broker()
    if ic is None:
        return pd.DataFrame(), f"Broker: {err}"
    try:
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        now     = datetime.now(IST).replace(tzinfo=None)
        from_dt = now - timedelta(days=days_back)
        gen = ic.historical_data(
            exchange=conn.EXCHANGE_TYPE_NSE,
            trading_symbol=trading_symbol,
            timeframe=conn.TIMEFRAME_TYPE_MIN,
            start=from_dt,
            end=now,
        )
        rows = _parse_rows(gen)
        if not rows:
            return pd.DataFrame(), f"{trading_symbol}: 0 1-min rows"
        df = _rows_to_df(rows)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ──────────────────────────────────────────────────────────────────────────────
# LTP logic
# ──────────────────────────────────────────────────────────────────────────────

def _get_ltp(
    ticker: str,
    sym: str,
    is_market_open: bool,
    df_daily: pd.DataFrame,
    df_1min: pd.DataFrame,
) -> tuple[float | None, str]:
    """
    Return (ltp, source_label).
    Priority:
      Market open  → 1-min last close, then daily close
      Market closed → daily close directly (no 1-min needed)
    """
    if is_market_open and not df_1min.empty:
        v = float(df_1min["Close"].iloc[-1])
        return v, "1-min"

    # After market close or 1-min unavailable → daily close
    if not df_daily.empty:
        v = float(df_daily["Close"].iloc[-1])
        return v, "daily-close"

    return None, "—"


# ──────────────────────────────────────────────────────────────────────────────
# Basket price builder — EXACT port of build_equal_weight_basket_ohlc()
# ──────────────────────────────────────────────────────────────────────────────

def _build_basket_df(
    ticker_daily: dict[str, pd.DataFrame],
    ticker_1min:  dict[str, pd.DataFrame],
    tickers: list[str],
    position_size: float,
    is_market_open: bool,
) -> tuple[pd.DataFrame | None, str]:
    """
    Exact port of build_equal_weight_basket_ohlc() from strategy1_dualbasket2.py.

    Steps:
      1. Build OHLCV MultiIndex panel, inner-join → dropna() (common start date).
      2. quantities = (1/N * position_size) / first_close[t]
      3. basket[field] = sum(panel[(t,field)] * quantities[t])
      4. If market is open: stitch today's 1-min bars as synthetic daily bar.

    Returns (basket_df | None, debug_str).
    """
    fields = ["Open", "High", "Low", "Close", "Volume"]
    missing = [t for t in tickers if t not in ticker_daily or ticker_daily[t].empty]
    if missing:
        return None, f"Missing daily data: {missing}"

    loaded = [
        ticker_daily[t][fields].rename(columns={c: (t, c) for c in fields})
        for t in tickers
    ]
    panel = pd.concat(loaded, axis=1).dropna()
    if panel.empty:
        return None, "Panel empty after inner join"

    quantities = _basket_quantities(ticker_daily, tickers, position_size)
    if not quantities:
        return None, "Missing fixed quantities in JSON"

    basket = pd.DataFrame(index=panel.index)
    for field in ["Open", "High", "Low", "Close"]:
        basket[field] = sum(panel[(t, field)] * quantities[t] for t in tickers)
    basket["Volume"] = sum(panel[(t, "Volume")] for t in tickers)
    basket = basket.dropna().sort_index()
    if basket.empty:
        return None, "Basket empty after dropna"

    # Stitch today's 1-min bars if market is open
    if is_market_open:
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        today = datetime.now(IST).date()
        if basket.index[-1].date() < today:
            today_rows: dict = {}
            all_ok = True
            for t in tickers:
                df1 = ticker_1min.get(t, pd.DataFrame())
                if df1.empty:
                    all_ok = False; break
                td = df1[df1.index.date == today]
                if td.empty:
                    all_ok = False; break
                today_rows[t] = {
                    "Open":   float(td["Open"].iloc[0]),
                    "High":   float(td["High"].max()),
                    "Low":    float(td["Low"].min()),
                    "Close":  float(td["Close"].iloc[-1]),
                    "Volume": float(td["Volume"].sum()),
                }
            if all_ok:
                today_ts = pd.Timestamp(today).tz_localize(IST)
                nr: dict = {}
                for field in ["Open", "High", "Low", "Close"]:
                    nr[field] = sum(today_rows[t][field] * quantities[t] for t in tickers)
                nr["Volume"] = sum(today_rows[t]["Volume"] for t in tickers)
                basket = pd.concat([basket, pd.DataFrame([nr], index=[today_ts])])

    return basket, f"{len(basket)} bars, from {basket.index[0].date()} to {basket.index[-1].date()}"


def _basket_quantities(ticker_daily: dict, tickers: list, position_size: float) -> dict:
    """Load exact backtest quantities from JSON."""
    import json
    from pathlib import Path
    
    # Try multiple possible paths to be safe when running from different CWDs
    q_file = None
    for p in [
        Path("state/basket_quantities_6.json"),
        Path(__file__).resolve().parent.parent.parent / "state" / "basket_quantities_6.json"
    ]:
        if p.exists():
            q_file = p
            break
            
    if not q_file:
        return {}
        
    try:
        with open(q_file) as f:
            all_q = json.load(f)
        for bid, q_map in all_q.items():
            if set(q_map.keys()) == set(tickers):
                return q_map
    except Exception:
        pass
        
    return {}


def _build_intraday_basket(ticker_1min: dict, tickers: list, quantities: dict) -> pd.DataFrame | None:
    """Minute-level basket using pre-computed quantities."""
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    today = datetime.now(IST).date()
    dfs = {}
    for t in tickers:
        df = ticker_1min.get(t, pd.DataFrame())
        if df.empty: return None
        td = df[df.index.date == today]
        if td.empty: return None
        dfs[t] = td

    idx = dfs[tickers[0]].index
    for t in tickers[1:]:
        idx = idx.intersection(dfs[t].index)
    if idx.empty: return None

    basket = pd.DataFrame(index=idx)
    for field in ["Open", "High", "Low", "Close"]:
        basket[field] = sum(dfs[t][field].reindex(idx).astype(float) * quantities[t] for t in tickers)
    basket["Volume"] = sum(dfs[t]["Volume"].reindex(idx).fillna(0).astype(float) for t in tickers)
    basket.dropna(inplace=True)
    return basket if not basket.empty else None


# ──────────────────────────────────────────────────────────────────────────────
# Indicator overlays
# ──────────────────────────────────────────────────────────────────────────────

def _compute_overlays(close: pd.Series, ema_fast: int, ema_slow: int,
                      show_bands: bool, show_ema: bool, window: int, min_pts: int):
    ef = close.ewm(span=ema_fast, adjust=False).mean() if show_ema else None
    es = close.ewm(span=ema_slow, adjust=False).mean() if show_ema else None
    bands = None
    if show_bands and len(close) >= min_pts:
        try:
            from indicators import rolling_regression_bands
            bands = rolling_regression_bands(close, window, min_pts)
        except Exception:
            pass
    return ef, es, bands


# ──────────────────────────────────────────────────────────────────────────────
# Chart builder
# ──────────────────────────────────────────────────────────────────────────────

def _chart(df: pd.DataFrame, title: str,
           entry_time=None, entry_price=None,
           bands=None, ema_f=None, ema_s=None,
           height: int = 450) -> go.Figure:

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing=dict(line=dict(color="#3fb950"), fillcolor="#152a15"),
        decreasing=dict(line=dict(color="#f85149"), fillcolor="#2a1515"),
    ), row=1, col=1)

    if bands is not None and not bands.empty:
        bi = bands.index.intersection(df.index)
        if not bi.empty:
            b = bands.loc[bi].dropna()
            fig.add_trace(go.Scatter(x=b.index, y=b["upper2"], name="+2σ",
                line=dict(color="rgba(248,81,73,.55)", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=b.index, y=b["trend_line"], name="Trend",
                line=dict(color="rgba(88,166,255,.65)", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=b.index, y=b["lower2"], name="-2σ",
                line=dict(color="rgba(63,185,80,.55)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(88,166,255,0.04)"), row=1, col=1)

    if ema_f is not None:
        fig.add_trace(go.Scatter(x=df.index, y=ema_f.reindex(df.index), name="EMA Fast",
            line=dict(color="#e3b341", width=1.2)), row=1, col=1)
    if ema_s is not None:
        fig.add_trace(go.Scatter(x=df.index, y=ema_s.reindex(df.index), name="EMA Slow",
            line=dict(color="#bc8cff", width=1.4)), row=1, col=1)

    if entry_time is not None and entry_price is not None:
        fig.add_trace(go.Scatter(
            x=[entry_time], y=[entry_price], mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#3fb950"),
            text=["BUY"], textposition="top center",
            textfont=dict(color="#3fb950", size=10), name="Entry",
        ), row=1, col=1)

    colors = ["#3fb950" if c >= o else "#f85149"
              for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.45, showlegend=False), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e6edf3")),
        height=height, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    font=dict(size=9, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8, r=8, t=48, b=8),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(
            gridcolor="#21262d", tickfont=dict(color="#8b949e", size=9))})
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh via JS (avoids Python 3.14 event-loop close bug)
# ──────────────────────────────────────────────────────────────────────────────

def _js_autorefresh(ms: int):
    components.html(
        f'<script>setTimeout(()=>window.parent.location.reload(), {ms});</script>',
        height=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(IST)

    # ── Config ────────────────────────────────────────────────────────────────
    from config_daily import (
        BASKET_CSV_PATH, TARGET_BASKET_SIZE, POSITION_SIZE,
        EMA_FAST, EMA_SLOW, ROLLING_WINDOW, MIN_ROLLING_POINTS,
    )
    csv_path = str(_PROJ / BASKET_CSV_PATH)

    # ── Market status ─────────────────────────────────────────────────────────
    mkt_open  = now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    mkt_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    is_market_open = (now_ist.weekday() < 5 and mkt_open <= now_ist <= mkt_close)

    # ── Broker ────────────────────────────────────────────────────────────────
    conn, ic, broker_err = _broker()
    broker_ok = conn is not None

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚡ ADTS Daily")
        
        # Engine Control
        st.markdown("---")
        engine_running = is_engine_running()
        if engine_running:
            st.success("🟢 Engine is RUNNING")
            if st.button("⏹ Stop Engine", use_container_width=True):
                stop_engine()
                st.rerun()
        else:
            st.error("🔴 Engine is STOPPED")
            if st.button("▶️ Start Engine", use_container_width=True):
                start_engine()
                st.rerun()

        st.markdown("---")
        refresh_sec = st.slider("Auto-refresh (s)", 30, 300, 60, 15)
        show_bands  = st.toggle("OLS bands",   value=True)
        show_ema    = st.toggle("EMAs",         value=True)
        chart_mode  = st.radio("Charts", ["Basket + Stocks", "Basket only", "Stocks only"], index=0)
        st.markdown("---")
        st.markdown(
            f"**Broker:** {'🟢 connected' if broker_ok else f'🔴 offline'}",
            unsafe_allow_html=True,
        )
        if broker_err:
            st.caption(f"⚠️ {broker_err[:80]}")
        st.markdown(f"**Market:** {'🟢 Open' if is_market_open else '🔴 Closed'}")
        st.markdown(f"`{now_ist.strftime('%H:%M:%S IST')}`")
        st.markdown("---")
        if st.button("🔄 Clear cache & refresh"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Refresh in {refresh_sec}s")

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([4, 1])
    h1.markdown("# 📈 ADTS Daily Paper Trader")
    h2.markdown(
        f"<p style='color:#8b949e;font-size:.82rem;text-align:right;padding-top:14px'>"
        f"{now_ist.strftime('%d %b %Y  %H:%M:%S IST')}</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Authentication ────────────────────────────────────────────────────────
    if broker_err == "AUTH_REQUIRED":
        st.error("🔒 **Broker Authentication Required**")
        st.markdown("A daily TOTP login is required to fetch live market data and execute trades.")
        with st.form("totp_form"):
            totp = st.text_input("Enter 6-digit TOTP", max_chars=6)
            submit = st.form_submit_button("Login")
            
            if submit and totp:
                try:
                    from integrate import ConnectToIntegrate
                    from dotenv import set_key
                    import os
                    
                    c = ConnectToIntegrate()
                    c._verify = False
                    
                    api_token = os.environ.get("INTEGRATE_API_TOKEN")
                    api_secret = os.environ.get("INTEGRATE_API_SECRET")
                    
                    c.login(api_token=api_token, api_secret=api_secret, totp=totp)
                    uid, actid, api_sk, ws_sk = c.get_session_keys()
                    
                    env_file = _PROJ / ".env"
                    set_key(str(env_file), "INTEGRATE_UID", uid)
                    set_key(str(env_file), "INTEGRATE_ACTID", actid)
                    set_key(str(env_file), "INTEGRATE_API_SESSION_KEY", api_sk)
                    set_key(str(env_file), "INTEGRATE_WS_SESSION_KEY", ws_sk)
                    set_key(str(env_file), "INTEGRATE_SESSION_SAVED_AT", datetime.now().isoformat())
                    
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
        return  # Stop rendering the rest of the dashboard

    # ── Load state ────────────────────────────────────────────────────────────
    state = _load_state()
    if state is None:
        st.error("No state file found. Run the daily engine first.")
        _js_autorefresh(refresh_sec * 1000)
        return

    saved_at     = state.get("saved_at", "—")
    slots        = state.get("slots", [None, None])
    trade_log_st = state.get("trade_log", [])
    realized     = float(state.get("realized_pnl", 0.0))
    pending_ent  = state.get("pending_entries", [])
    pending_ex   = state.get("pending_exits",   [])
    basket_cs    = state.get("basket_close_series", {})

    binfo = _basket_info(csv_path, TARGET_BASKET_SIZE)

    active_slots    = [s for s in slots if s is not None]
    held_basket_ids = {s["basket_id"] for s in active_slots}

    # All tickers across held baskets
    all_tickers: set[str] = set()
    for bid in held_basket_ids:
        all_tickers.update(binfo.get(bid, {}).get("tickers", []))

    imap, imap_err = _instrument_map(tuple(sorted(all_tickers))) if all_tickers else ({}, None)

    # ── Fetch data ────────────────────────────────────────────────────────────
    ticker_daily: dict[str, pd.DataFrame] = {}
    ticker_1min:  dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, str]          = {}

    if all_tickers:
        ph = st.empty()
        ph.info(f"⏳ Fetching data for {len(all_tickers)} tickers (1300 days daily)…")
        for ticker in sorted(all_tickers):
            sym = _sym(imap, ticker)
            df_d, err_d = _fetch_daily(sym, days_back=1300)
            ticker_daily[ticker] = df_d
            if err_d:
                fetch_errors[ticker] = err_d
            if is_market_open:
                df_m, err_m = _fetch_1min(sym, days_back=1)
                ticker_1min[ticker] = df_m
                if err_m and ticker not in fetch_errors:
                    fetch_errors[ticker] = err_m
        ph.empty()

    # ── LTP computation ───────────────────────────────────────────────────────
    live_prices: dict[str, float] = {}
    ltp_sources: dict[str, str]   = {}
    for ticker in all_tickers:
        sym = _sym(imap, ticker)
        ltp, src = _get_ltp(
            ticker, sym,
            is_market_open,
            ticker_daily.get(ticker, pd.DataFrame()),
            ticker_1min.get(ticker,  pd.DataFrame()),
        )
        if ltp is not None:
            live_prices[ticker] = ltp
            ltp_sources[ticker] = src

    # ── PnL per slot ──────────────────────────────────────────────────────────
    total_invest   = 0.0
    total_live_val = 0.0
    slot_info_list = []

    for slot in active_slots:
        bid     = slot["basket_id"]
        tickers = slot.get("tickers", [])
        qtys    = {k: int(v)   for k, v in slot.get("quantities",   {}).items()}
        epx     = {k: float(v) for k, v in slot.get("entry_prices", {}).items()}
        invest  = float(slot.get("investment", 0))
        total_invest += invest

        slot_live   = 0.0
        all_ltp_ok  = True
        ticker_rows = []

        for t in tickers:
            ep   = epx.get(t, 0.0)
            qty  = qtys.get(t, 0)
            ltp  = live_prices.get(t)
            src  = ltp_sources.get(t, "—")
            cost = ep * qty
            if ltp is None:
                all_ltp_ok = False
                ticker_rows.append(dict(ticker=t, ep=ep, qty=qty, ltp=None, cost=cost, mkt=None, pnl=None, src=src))
            else:
                mkt = ltp * qty
                slot_live += mkt
                ticker_rows.append(dict(ticker=t, ep=ep, qty=qty, ltp=ltp, cost=cost, mkt=mkt, pnl=mkt - cost, src=src))

        s_pnl = (slot_live - invest) if all_ltp_ok else float("nan")
        s_pct = (s_pnl / invest * 100) if (all_ltp_ok and invest) else float("nan")
        if all_ltp_ok:
            total_live_val += slot_live

        bi      = binfo.get(bid, {})
        companies = bi.get("companies", [])
        tck_list  = bi.get("tickers", tickers)

        slot_info_list.append(dict(
            basket_id  = bid,
            entry_time = slot.get("entry_time"),
            entry_type = slot.get("entry_type", ""),
            invest     = invest,
            live_val   = slot_live if all_ltp_ok else None,
            pnl        = s_pnl,
            pnl_pct    = s_pct,
            tickers    = ticker_rows,
            companies  = companies,
            tck_list   = tck_list,
        ))

    unrealized   = (total_live_val - total_invest) if total_live_val else 0.0
    day_pnl      = realized + unrealized
    total_equity = POSITION_SIZE + realized + unrealized

    # ══════════════════════════════════════════════════════════════════════════
    # ① METRIC CARDS
    # ══════════════════════════════════════════════════════════════════════════
    c = st.columns(5)
    _metric(c[0], "Total Equity",
            f"<span class='neu'>₹{total_equity:,.0f}</span>",
            f"Capital ₹{POSITION_SIZE:,.0f}")
    _metric(c[1], "Unrealized P&L",
            f"<span class='{_cls(unrealized)}'>{_fmt(unrealized)}</span>",
            f"{_pct(unrealized/total_invest*100) if total_invest else '—'} on cost")
    _metric(c[2], "Realized P&L",
            f"<span class='{_cls(realized)}'>{_fmt(realized)}</span>",
            f"{len(trade_log_st)} closed trades")
    _metric(c[3], "Day P&L",
            f"<span class='{_cls(day_pnl)}'>{_fmt(day_pnl)}</span>",
            _pct(day_pnl / POSITION_SIZE * 100))
    _metric(c[4], "Open Slots",
            f"<span class='neu'>{len(active_slots)}/2</span>",
            "🟢 Market open" if is_market_open else "🔴 Market closed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ② ENGINE STATE (pending orders, basket close series)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">⚙️ Engine State</div>', unsafe_allow_html=True)

    ec1, ec2 = st.columns([1, 1])
    with ec1:
        st.markdown(f"**State saved:** `{saved_at}`")
        if pending_ent:
            st.markdown("**📥 Pending ENTRIES (execute at 09:17):**")
            for pe in pending_ent:
                bid   = pe.get("basket_id", "?")
                etype = pe.get("entry_type", "?")
                tkrs  = binfo.get(bid, {}).get("tickers", [])
                st.success(f"🧺 Basket {bid} · {etype} · {', '.join(tkrs)}")
        else:
            st.info("No pending entry orders")

        if pending_ex:
            st.markdown("**📤 Pending EXITS (execute at 09:17):**")
            for px in pending_ex:
                bid    = px.get("basket_id", "?")
                reason = px.get("reason", "?")
                st.warning(f"🧺 Basket {bid} · {reason}")
        else:
            st.info("No pending exit orders")

    with ec2:
        if basket_cs:
            st.markdown("**📊 Basket Close Series (last 3 values):**")
            bcs_rows = []
            for bid_str, series in basket_cs.items():
                if isinstance(series, list) and len(series) >= 2:
                    last3 = series[-3:]
                    bcs_rows.append({
                        "Basket": int(bid_str),
                        "Last Close": round(float(last3[-1]), 2) if last3 else "—",
                        "Prev Close": round(float(last3[-2]), 2) if len(last3) >= 2 else "—",
                        "Bars": len(series),
                    })
            if bcs_rows:
                st.dataframe(pd.DataFrame(bcs_rows).set_index("Basket"),
                             use_container_width=True, height=120)
        else:
            st.info("No basket close series in state")

    # ══════════════════════════════════════════════════════════════════════════
    # ③ HOLDINGS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">🗂 Holdings</div>', unsafe_allow_html=True)

    if not active_slots:
        st.info("No open positions.")
    else:
        for si in slot_info_list:
            bid    = si["basket_id"]
            pnl    = si["pnl"]
            pct    = si["pnl_pct"]
            invest = si["invest"]
            live   = si["live_val"]
            etype  = si["entry_type"].replace("_", " ").title()
            et_str = str(si["entry_time"] or "")[:16]
            pnl_ok = not (isinstance(pnl, float) and np.isnan(pnl))
            companies = si["companies"]

            exp_label = (
                f"🧺 Basket {bid}  ·  {etype}  ·  Entered {et_str}  ·  "
                + (_fmt(pnl) + " (" + _pct(pct) + ")" if pnl_ok else "⚠️ LTP unavailable")
            )
            with st.expander(exp_label, expanded=True):
                hdr = st.columns([2.2, 2, 1, 1.3, 1.3, 1.3, 1.8, 1])
                for col, h in zip(hdr, ["Ticker", "Company", "Qty", "Entry ₹", "LTP ₹", "Cost", "P&L", "Src"]):
                    col.markdown(f"**{h}**")

                for i, tr in enumerate(si["tickers"]):
                    row = st.columns([2.2, 2, 1, 1.3, 1.3, 1.3, 1.8, 1])
                    comp = companies[i] if i < len(companies) else ""
                    ltp_s = f"₹{tr['ltp']:,.2f}" if tr["ltp"] else "—"
                    pnl_s  = _fmt(tr["pnl"]) if tr["pnl"] is not None else "—"
                    pnl_cl = _cls(tr["pnl"]) if tr["pnl"] is not None else "neu"
                    row[0].markdown(f"**{tr['ticker']}**")
                    row[1].markdown(comp[:22])
                    row[2].markdown(f"{tr['qty']:,}")
                    row[3].markdown(f"₹{tr['ep']:,.2f}")
                    row[4].markdown(ltp_s)
                    row[5].markdown(f"₹{tr['cost']:,.0f}")
                    row[6].markdown(f"<span class='{pnl_cl}'>{pnl_s}</span>", unsafe_allow_html=True)
                    row[7].markdown(f"<span style='color:#8b949e;font-size:.75rem'>{tr['src']}</span>",
                                    unsafe_allow_html=True)

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Invested",   f"₹{invest:,.0f}")
                m2.metric("Live Value", f"₹{live:,.0f}" if live else "—")
                if pnl_ok:
                    m3.metric("Unrealized", _fmt(pnl), delta=_pct(pct))
                else:
                    m3.metric("Unrealized", "N/A")

    # ══════════════════════════════════════════════════════════════════════════
    # ④ CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">📊 Charts</div>', unsafe_allow_html=True)

    if not held_basket_ids:
        st.info("Charts appear when a position is held.")
    else:
        tabs = st.tabs([f"Basket {bid}" for bid in sorted(held_basket_ids)])
        for tab, bid in zip(tabs, sorted(held_basket_ids)):
            with tab:
                bi      = binfo.get(bid, {})
                tickers = bi.get("tickers", [])
                companies = bi.get("companies", [])
                if not tickers:
                    st.warning(f"Basket {bid} not in CSV")
                    continue

                slot = next((s for s in slots if s and s["basket_id"] == bid), None)
                entry_ts = None
                if slot and slot.get("entry_time"):
                    try:
                        entry_ts = pd.Timestamp(slot["entry_time"])
                        if entry_ts.tzinfo is None:
                            entry_ts = entry_ts.tz_localize(IST)
                    except Exception:
                        entry_ts = None

                qtys_basket = _basket_quantities(ticker_daily, tickers, POSITION_SIZE)

                show_basket = chart_mode in ("Basket + Stocks", "Basket only")
                show_stocks = chart_mode in ("Basket + Stocks", "Stocks only")

                # ── Daily basket chart ────────────────────────────────────────
                if show_basket:
                    basket_df, binfo_str = _build_basket_df(
                        ticker_daily, ticker_1min, tickers, POSITION_SIZE, is_market_open)

                    st.markdown(f"#### Basket {bid} — Daily (history + today)")
                    st.caption(f"📐 {binfo_str}")

                    if basket_df is not None and not basket_df.empty:
                        close = basket_df["Close"]
                        ef, es, bands = _compute_overlays(
                            close, EMA_FAST, EMA_SLOW,
                            show_bands, show_ema, ROLLING_WINDOW, MIN_ROLLING_POINTS)

                        entry_px = None
                        if entry_ts is not None:
                            avail = basket_df.index[basket_df.index <= entry_ts]
                            if not avail.empty:
                                entry_px = float(basket_df.loc[avail[-1], "Close"])

                        fig = _chart(
                            basket_df,
                            f"Basket {bid}  ·  {', '.join(tickers[:5])}{'…' if len(tickers) > 5 else ''}",
                            entry_time=entry_ts if entry_px else None,
                            entry_price=entry_px,
                            bands=bands, ema_f=ef, ema_s=es, height=500,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"Insufficient daily data — {binfo_str}")

                    # Intraday 1-min basket (market open only)
                    if is_market_open and qtys_basket:
                        st.markdown(f"#### Basket {bid} — Intraday 1-min")
                        ib = _build_intraday_basket(ticker_1min, tickers, qtys_basket)
                        if ib is not None:
                            st.plotly_chart(_chart(ib, f"Basket {bid} — Intraday", height=320),
                                            use_container_width=True)
                        else:
                            st.info("Intraday basket data not available yet.")

                # ── Individual stock charts ───────────────────────────────────
                if show_stocks and tickers:
                    st.markdown(f"#### Individual Stocks — Basket {bid}")
                    n_cols = max(1, min(len(tickers), 3))
                    cols   = st.columns(n_cols)
                    for i, ticker in enumerate(tickers):
                        with cols[i % n_cols]:
                            company = companies[i] if i < len(companies) else ticker
                            ltp     = live_prices.get(ticker)
                            src     = ltp_sources.get(ticker, "")
                            pnl_t   = next((tr["pnl"] for si in slot_info_list
                                           if si["basket_id"] == bid
                                           for tr in si["tickers"]
                                           if tr["ticker"] == ticker), None)

                            # Choose best df: 1-min (today, market open) else daily
                            df_s = pd.DataFrame()
                            if is_market_open:
                                df_s = ticker_1min.get(ticker, pd.DataFrame())
                            if df_s.empty:
                                df_s = ticker_daily.get(ticker, pd.DataFrame())

                            title_parts = [ticker]
                            if company:
                                title_parts.append(company[:18])
                            if ltp:
                                title_parts.append(f"₹{ltp:,.2f} [{src}]")
                            if pnl_t is not None:
                                title_parts.append(f"{'▲' if pnl_t>=0 else '▼'}{_fmt(pnl_t)}")

                            if df_s.empty:
                                err = fetch_errors.get(ticker, "no data")
                                st.warning(f"{ticker}: {err}")
                            else:
                                st.plotly_chart(
                                    _chart(df_s, "  ·  ".join(title_parts), height=300),
                                    use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ⑤ TRADE LOG
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">📋 Trade Log</div>', unsafe_allow_html=True)

    tl = _load_trade_log()
    if tl.empty and trade_log_st:
        tl = pd.DataFrame(trade_log_st)

    # Convert active slots to open trade log entries
    open_rows = []
    for si in slot_info_list:
        et = si.get("entry_time")
        try:
            et_dt = pd.Timestamp(et).tz_localize(IST) if et and pd.Timestamp(et).tzinfo is None else pd.Timestamp(et)
            hold_days = (now_ist - et_dt).days if pd.notna(et_dt) else 0
        except Exception:
            hold_days = 0

        open_rows.append({
            "entry_time": et,
            "exit_time": "—",
            "basket_id": si["basket_id"],
            "basket_tickers": ", ".join(si["tck_list"]),
            "entry_type": si["entry_type"],
            "close_reason": "—",
            "investment": si["invest"],
            "exit_value": si.get("live_val"),
            "pnl": si.get("pnl"),
            "pnl_pct": si.get("pnl_pct"),
            "hold_days": hold_days,
            "status": "open (MTM)"
        })

    if open_rows:
        open_df = pd.DataFrame(open_rows)
        tl = pd.concat([open_df, tl], ignore_index=True) if not tl.empty else open_df

    if not tl.empty:
        # Enrich with basket member tickers from binfo
        if "basket_id" in tl.columns and "basket_tickers" not in tl.columns:
            tl["basket_tickers"] = tl["basket_id"].apply(
                lambda bid: ", ".join(binfo.get(int(bid), {}).get("tickers", []))
                if pd.notna(bid) else ""
            )

        show_cols = [c for c in [
            "status", "entry_time", "exit_time", "basket_id", "basket_tickers",
            "entry_type", "close_reason",
            "investment", "exit_value", "pnl", "pnl_pct", "hold_days",
        ] if c in tl.columns]

        def _pnl_style(v):
            try:
                return f"color: {'#3fb950' if float(v) >= 0 else '#f85149'}"
            except Exception:
                return ""

        styled = tl[show_cols].style
        for col in ["pnl", "pnl_pct"]:
            if col in show_cols:
                styled = styled.map(_pnl_style, subset=[col])
        st.dataframe(styled, use_container_width=True, height=280)

        # Cumulative PnL chart
        if "pnl" in tl.columns and len(tl) > 1:
            tl_closed = tl[tl.get("status", "closed") != "open (MTM)"] if "status" in tl.columns else tl
            if len(tl_closed) > 1:
                cum = tl_closed["pnl"].astype(float).cumsum()
                colors_pnl = ["#3fb950" if v >= 0 else "#f85149" for v in tl_closed["pnl"]]
                fig_pnl = go.Figure(go.Scatter(
                    y=cum.values, mode="lines+markers",
                    line=dict(color="#58a6ff", width=2),
                    marker=dict(size=7, color=colors_pnl),
                    fill="tozeroy", fillcolor="rgba(88,166,255,0.07)",
                ))
                fig_pnl.update_layout(
                    title="Cumulative Realized PnL", height=240,
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    margin=dict(l=8, r=8, t=38, b=8),
                    yaxis=dict(gridcolor="#21262d", tickprefix="₹",
                               tickfont=dict(color="#8b949e")),
                    xaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("No closed trades yet.")

    # ══════════════════════════════════════════════════════════════════════════
    # ⑥ DEBUG PANEL (sidebar expandable)
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        with st.expander("🔍 Debug info"):
            st.json({
                "broker_ok":     broker_ok,
                "broker_err":    broker_err,
                "imap_err":      imap_err,
                "market_open":   is_market_open,
                "tickers":       sorted(all_tickers),
                "fetch_errors":  fetch_errors,
                "ltp_sources":   ltp_sources,
                "daily_bars":    {t: len(df) for t, df in ticker_daily.items()},
                "1min_bars":     {t: len(df) for t, df in ticker_1min.items()},
                "state_saved_at": saved_at,
            })

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(f"State: {saved_at}  ·  Paper mode only — no real orders  ·  Next refresh in {refresh_sec}s")

    # ── JS auto-refresh (avoids Python 3.14 asyncio event-loop close bug) ────
    _js_autorefresh(refresh_sec * 1000)


if __name__ == "__main__":
    main()
