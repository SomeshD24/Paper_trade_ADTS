"""
run_paper.py — ADTS 5-min paper trading runner.

Requires:
    pip install pyintegrate pandas numpy scipy pytz

Usage:
    python run_paper.py --api-token TOKEN --api-secret SECRET [options]

Options:
    --basket-csv   path to basket CSV  (default: data/baskets_nifty200_all_sizes.csv)
    --basket-size  int                 (default: 6)
    --capital      float               (default: 100000)
    --fresh        ignore saved state; start clean
    --totp         6-digit TOTP if 2FA is enabled on account

Login flow (ConnectToIntegrate):
    Session keys (uid, actid, api_session_key, ws_session_key) are valid for
    24 hours.  On first run they are obtained via conn.login() and cached in
    state/session_keys.json.  On subsequent runs within 24 h the cached keys
    are reused via conn.set_session_keys() to avoid re-login.

Bar cadence (NSE 5-min):
  ┌─ bar N open ─── bar N close ─ +3s eval ──── signals queued
  │                                              │
  └─ bar N+1 open ─ +2s execution ──────────────┘

All order execution in this file is PAPER ONLY (no real orders placed).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    IST, BAR_MINUTES, EVAL_DELAY_SECS, EXEC_DELAY_SECS,
    MARKET_OPEN_H, MARKET_OPEN_M, MARKET_CLOSE_H, MARKET_CLOSE_M,
    POSITION_SIZE, N_SLOTS, TARGET_BASKET_SIZE,
    BASKET_CSV_PATH, STATE_FILE, TRADE_LOG_FILE, LOG_LEVEL,
    WARMUP_DAYS,
)
from symbol_mapper import (
    load_symbol_master, build_basket_instrument_map,
    all_unique_tickers_from_configs,
)
from data_manager import warmup_ticker_buffers, TickerBuffer, LiveBarPoller
from signal_engine import BasketSignalEngine, build_basket_info
from portfolio_engine import PortfolioEngine
from state_store import save_state, load_state

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,   # Use DEBUG to see warmup/readiness diagnostics; change back to LOG_LEVEL after warmup
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_paper")

DOTENV_FILE  = ".env"


# ── Basket config loader ──────────────────────────────────────────────────────

def _load_basket_config(csv_path: str, target_size: int) -> dict:
    import pandas as pd

    REQUIRED = {"basket_id", "stock_position", "symbol", "ticker", "company_name", "sector"}
    df = pd.read_csv(csv_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Basket CSV missing columns: {missing}")

    if "basket_size" in df.columns:
        df = df[df["basket_size"] == target_size]
    if df.empty:
        raise ValueError(f"No baskets found for basket_size={target_size}")

    return {"label": f"{target_size}-stock", "basket_size": target_size, "members": df}


# ── pyintegrate session management ───────────────────────────────────────────

def _load_env() -> dict:
    """
    Load credentials and session keys from .env file.
    Requires: pip install python-dotenv
    Returns dict of all INTEGRATE_* env vars (may be empty strings if unset).
    """
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(filename=DOTENV_FILE, raise_error_if_not_found=False)
                    or DOTENV_FILE, override=True)
    except ImportError:
        logger.warning("python-dotenv not installed; reading environment only. "
                       "Run: pip install python-dotenv")
    import os
    return {k: os.environ.get(k, "") for k in (
        "INTEGRATE_API_TOKEN", "INTEGRATE_API_SECRET", "INTEGRATE_TOTP",
        "INTEGRATE_UID", "INTEGRATE_ACTID",
        "INTEGRATE_API_SESSION_KEY", "INTEGRATE_WS_SESSION_KEY",
        "INTEGRATE_SESSION_SAVED_AT",
    )}


def _save_session_to_env(uid: str, actid: str, api_sk: str, ws_sk: str):
    """
    Persist session keys back into .env so the next run can reuse them.
    Only the session-key keys are touched; credentials are left intact.
    """
    try:
        from dotenv import set_key, find_dotenv
        env_path = find_dotenv(filename=DOTENV_FILE, raise_error_if_not_found=False) or DOTENV_FILE
        now_iso  = datetime.now().isoformat()
        set_key(env_path, "INTEGRATE_UID",             uid)
        set_key(env_path, "INTEGRATE_ACTID",           actid)
        set_key(env_path, "INTEGRATE_API_SESSION_KEY", api_sk)
        set_key(env_path, "INTEGRATE_WS_SESSION_KEY",  ws_sk)
        set_key(env_path, "INTEGRATE_SESSION_SAVED_AT", now_iso)
        logger.info(f"Session keys saved to {env_path}")
    except Exception as e:
        logger.warning(f"Could not write session keys to .env: {e}")


def _init_connection(api_token: str, api_secret: str, totp: str | None) -> tuple:
    """
    Create ConnectToIntegrate and IntegrateData instances.

    Session key persistence (stored in .env):
      1. If .env has INTEGRATE_UID + fresh INTEGRATE_SESSION_SAVED_AT (<23 h),
         call conn.set_session_keys() — no network login needed.
      2. Otherwise call conn.login(api_token, api_secret, totp) and write the
         new session keys back to .env via set_key().

    Returns (conn, ic):
      conn : ConnectToIntegrate
      ic   : IntegrateData
    """
    from integrate import ConnectToIntegrate, IntegrateData

    conn = ConnectToIntegrate()
    conn._verify = False
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    env  = _load_env()

    # ── Try to restore session from .env ──────────────────────────────────────
    restored  = False
    saved_at  = env.get("INTEGRATE_SESSION_SAVED_AT", "")
    uid_saved = env.get("INTEGRATE_UID", "")

    if saved_at and uid_saved:
        try:
            age_h = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds() / 3600
            if age_h < 23:
                conn.set_session_keys(
                    uid_saved,
                    env["INTEGRATE_ACTID"],
                    env["INTEGRATE_API_SESSION_KEY"],
                    env["INTEGRATE_WS_SESSION_KEY"],
                )
                logger.info(f"Session keys restored from .env (age {age_h:.1f} h)")
                restored = True
            else:
                logger.info(f"Session keys in .env are {age_h:.1f} h old — re-logging in.")
        except Exception as e:
            logger.warning(f"Could not restore session from .env: {e}")

    # ── Fresh login ───────────────────────────────────────────────────────────
    if not restored:
        if totp is None:
            import getpass
            totp = getpass.getpass(
                "Enter 6-digit TOTP (2FA code from your authenticator app): "
            ).strip()
        logger.info("Logging in to Definedge Integrate…")
        conn.login(api_token=api_token, api_secret=api_secret, totp=totp or None)
        uid, actid, api_sk, ws_sk = conn.get_session_keys()
        _save_session_to_env(uid, actid, api_sk, ws_sk)
        logger.info("Login successful.")

    ic = IntegrateData(conn)
    return conn, ic


# ── Timing helpers ────────────────────────────────────────────────────────────

def _to_ist_ts(dt) -> pd.Timestamp:
    """
    Convert any datetime-like to a tz-aware IST pd.Timestamp.
    Safe for: naive datetime, aware datetime, naive Timestamp, aware Timestamp.
    """
    ts = pd.Timestamp(dt) if not isinstance(dt, pd.Timestamp) else dt
    if ts.tzinfo is None:
        return ts.tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    return ts.tz_convert(IST)


def _is_market_day(dt: datetime) -> bool:
    return dt.weekday() < 5      # Mon–Fri (no holiday calendar; add if needed)


def _is_in_market_hours(dt: datetime) -> bool:
    t = dt.time()
    return dtime(MARKET_OPEN_H, MARKET_OPEN_M) <= t < dtime(MARKET_CLOSE_H, MARKET_CLOSE_M)


def _bar_open_time(dt: datetime) -> datetime:
    """Return the open timestamp of the 5-min bar that contains dt."""
    minutes_since_open = (
        dt.hour * 60 + dt.minute
    ) - (MARKET_OPEN_H * 60 + MARKET_OPEN_M)
    bar_idx = max(0, minutes_since_open // BAR_MINUTES)
    return dt.replace(
        hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0
    ) + timedelta(minutes=bar_idx * BAR_MINUTES)


def _bar_close_time(dt: datetime) -> datetime:
    return _bar_open_time(dt) + timedelta(minutes=BAR_MINUTES)


# ── Paper trading runner ──────────────────────────────────────────────────────

class PaperTradingRunner:
    """
    Orchestrates the full 5-min paper trading loop.
    """

    def __init__(self, conn, ic, config: dict, instrument_map: dict,
                 ticker_buffers: dict, capital: float = POSITION_SIZE):
        self.conn           = conn
        self.ic             = ic
        self.config         = config
        self.instrument_map = instrument_map
        self.ticker_buffers = ticker_buffers

        basket_info       = build_basket_info(config)
        slot_capital      = capital / N_SLOTS

        self.portfolio    = PortfolioEngine(initial_capital=capital, n_slots=N_SLOTS)
        self.signal_eng   = BasketSignalEngine(
            basket_info, ticker_buffers, instrument_map,
            position_size=slot_capital,
        )
        self.poller       = LiveBarPoller(ic, conn, instrument_map, ticker_buffers)
        self._last_eval_bar: pd.Timestamp | None = None
        self._last_heartbeat: pd.Timestamp | None = None   # for 5-min alive log

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self):
        logger.info("=" * 60)
        logger.info("ADTS 5-min Paper Trader — LIVE")
        logger.info(f"  Capital: {self.portfolio.base_capital:,.0f}   Slots: {N_SLOTS}")
        logger.info("=" * 60)

        # Pre-compute indicators from seeded warmup data so the engine is
        # immediately ready for signals on the very first bar close.
        logger.info("Pre-computing indicators from warmup data...")
        n_ready = self.signal_eng.warmup_indicators()
        logger.info(
            f"  {n_ready}/{len(self.signal_eng.basket_info)} baskets ready at startup  "
            f"| {self.signal_eng.warmup_status()}"
        )

        try:
            while True:
                now = datetime.now(IST)

                if not _is_market_day(now):
                    logger.info("Weekend / non-trading day. Sleeping 1 h.")
                    time.sleep(3600)
                    continue

                if not _is_in_market_hours(now):
                    self._end_of_day(now)
                    # Sleep until next market open
                    next_open = now.replace(
                        hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                        second=0, microsecond=0,
                    )
                    if now >= next_open:
                        next_open += timedelta(days=1)
                    sleep_secs = (next_open - now).total_seconds()
                    logger.info(f"Market closed. Sleeping {sleep_secs / 3600:.1f} h.")
                    time.sleep(min(sleep_secs, 3600))
                    continue

                bar_open  = _bar_open_time(now)   # open of the bar that contains now
                bar_ts    = _to_ist_ts(bar_open)

                # The bar we want to evaluate is the one that JUST COMPLETED —
                # i.e. the bar whose close == bar_open (current bar's open).
                # We wait EVAL_DELAY_SECS after it closed before evaluating.
                prev_bar_ts = bar_ts - pd.Timedelta(minutes=BAR_MINUTES)
                eval_ts     = bar_ts + pd.Timedelta(seconds=EVAL_DELAY_SECS)

                # ── Execute any pending orders at current bar open + EXEC_DELAY ─
                exec_threshold = bar_ts + pd.Timedelta(seconds=EXEC_DELAY_SECS)
                now_ts = _to_ist_ts(now)
                if (now_ts >= exec_threshold and
                        (self.portfolio._pending_exits or
                         self.portfolio._pending_entries)):
                    self._execute_pending(bar_ts)

                # ── Evaluate the PREVIOUS (just-completed) bar ────────────────
                if (now_ts >= eval_ts and
                        self._last_eval_bar != prev_bar_ts):
                    self._on_bar_close(prev_bar_ts, now)
                    self._last_eval_bar = prev_bar_ts

                # ── Heartbeat: print status every 5 min even if no bar fired ─
                if (self._last_heartbeat is None or
                        (now_ts - self._last_heartbeat).total_seconds() >= 300):
                    n_open = sum(1 for s in self.portfolio.slots if s is not None)
                    logger.info(
                        f"[HEARTBEAT] {now_ts.strftime('%H:%M:%S')}  "
                        f"{self.signal_eng.warmup_status()}  "
                        f"open_slots={n_open}/{self.portfolio.n_slots}  "
                        f"last_eval_bar={self._last_eval_bar.strftime('%H:%M') if self._last_eval_bar is not None else 'none'}"
                    )
                    self._last_heartbeat = now_ts

                # Sleep toward the next bar evaluation
                # next eval fires EVAL_DELAY_SECS after the next bar opens
                next_eval_ts = bar_ts + pd.Timedelta(minutes=BAR_MINUTES) + pd.Timedelta(seconds=EVAL_DELAY_SECS)
                secs_to_next = (next_eval_ts - now_ts).total_seconds()
                sleep_secs = max(1, min(secs_to_next - 2, 30))
                time.sleep(sleep_secs)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — saving state and exiting.")
            save_state(self.portfolio, self.ticker_buffers)

    # ── Bar-close handler ─────────────────────────────────────────────────────

    def _on_bar_close(self, bar_ts: pd.Timestamp, now: datetime):
        logger.info(f"── Bar {bar_ts.strftime('%H:%M')} close ──────────────────")

        # 1. Poll broker for new 1-min bars → update 5-min buffers
        self.poller.poll_and_update(now)

        # 2. Signal evaluation for all baskets
        result        = self.signal_eng.on_bar_close(bar_ts)
        basket_closes = result["basket_closes"]

        # 3. Update trailing SL peaks + extend basket close series
        self.portfolio.on_bar_close(bar_ts, basket_closes)

        # 4. Exit signal check (reg +2σ or trailing SL)
        self.portfolio.check_exit_signals(result["sell_signals"], basket_closes)

        # 5. Queue new entries
        self.portfolio.queue_entries(result["entry_signals"], bar_ts)

        # 6. MTM log
        snap = self.portfolio.mtm_snapshot(basket_closes)
        logger.info(
            f"  MTM  realized={snap['realized_pnl']:+,.0f}  "
            f"unrealized={snap['unrealized_pnl']:+,.0f}  "
            f"equity={snap['total_equity']:,.0f}"
        )
        logger.info(f"  {self.signal_eng.warmup_status()}")

        # 7. Persist state every bar
        save_state(self.portfolio, self.ticker_buffers)

    # ── Order execution (next bar open) ───────────────────────────────────────

    def _execute_pending(self, bar_ts: pd.Timestamp):
        exec_prices = self.signal_eng.get_exec_prices(bar_ts)
        basket_info = self.signal_eng.basket_info
        new_trades  = self.portfolio.execute_pending(exec_prices, bar_ts, basket_info)
        if new_trades:
            for t in new_trades:
                status = t.get("close_reason", "?")
                logger.info(
                    f"  EXEC {status} B{t['basket_id']} "
                    f"pnl={t['pnl']:+.0f} ({t['pnl_pct']:+.2f}%)"
                )
            save_state(self.portfolio, self.ticker_buffers)

    # ── End-of-day ────────────────────────────────────────────────────────────

    def _end_of_day(self, now: datetime):
        closes = self.signal_eng.get_basket_close_prices()
        snap   = self.portfolio.mtm_snapshot(closes)
        logger.info("=" * 60)
        logger.info("END OF DAY")
        logger.info(f"  Realized PnL:   {snap['realized_pnl']:+,.2f}")
        logger.info(f"  Unrealized PnL: {snap['unrealized_pnl']:+,.2f}")
        logger.info(f"  Total Equity:   {snap['total_equity']:,.2f}")
        for s in [s for s in snap["slots"] if s is not None]:
            logger.info(
                f"  Open B{s['basket_id']}: invest={s['investment']:,.0f}  "
                f"mtm={s['mtm_value']:,.0f}  u_pnl={s['unrealized_pnl']:+,.0f}"
            )
        logger.info("=" * 60)
        save_state(self.portfolio, self.ticker_buffers)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ADTS 5-min Paper Trader",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Credentials are read from .env (copy .env.example → .env and fill in).\n"
            "CLI flags override .env values if both are provided.\n"
            "Session keys are stored back into .env automatically after login."
        ),
    )
    parser.add_argument("--basket-csv",   default=BASKET_CSV_PATH)
    parser.add_argument("--basket-size",  type=int,   default=TARGET_BASKET_SIZE)
    parser.add_argument("--capital",      type=float, default=POSITION_SIZE)
    parser.add_argument("--api-token",    default=None,
                        help="Overrides INTEGRATE_API_TOKEN in .env")
    parser.add_argument("--api-secret",   default=None,
                        help="Overrides INTEGRATE_API_SECRET in .env")
    parser.add_argument("--totp",         default=None,
                        help="6-digit TOTP (required for login; prompted if omitted)")
    parser.add_argument("--fresh",        action="store_true",
                        help="Ignore saved portfolio state; start clean")
    args = parser.parse_args()

    Path("state").mkdir(exist_ok=True)

    # ── 1. Resolve credentials (.env → CLI override) ──────────────────────────
    env = _load_env()
    api_token  = args.api_token  or env.get("INTEGRATE_API_TOKEN",  "")
    api_secret = args.api_secret or env.get("INTEGRATE_API_SECRET", "")
    totp       = args.totp or None   # may be None; prompted inside _init_connection if login needed

    if not api_token or not api_secret:
        parser.error(
            "API token and secret are required.\n"
            "Set INTEGRATE_API_TOKEN and INTEGRATE_API_SECRET in .env, "
            "or pass --api-token / --api-secret."
        )

    # ── 2. Connect ────────────────────────────────────────────────────────────
    conn, ic = _init_connection(api_token, api_secret, totp)

    # ── 3. Load basket config ─────────────────────────────────────────────────
    logger.info(f"Loading basket config from {args.basket_csv} (size={args.basket_size})")
    config = _load_basket_config(args.basket_csv, args.basket_size)
    all_tickers = all_unique_tickers_from_configs([config])
    logger.info(f"  {len(build_basket_info(config))} baskets, {len(all_tickers)} unique tickers")

    # ── 4. Map tickers → trading symbols via conn.symbols ────────────────────
    logger.info("Building symbol map from broker master…")
    load_symbol_master(conn)               # download + cache
    instrument_map = build_basket_instrument_map(all_tickers)
    unmapped = set(all_tickers) - set(instrument_map)
    if unmapped:
        logger.warning(f"  {len(unmapped)} tickers unmapped: {sorted(unmapped)}")
    logger.info(f"  {len(instrument_map)} tickers mapped")

    # ── 5. Warm up rolling buffers with historical 1-min data ────────────────
    logger.info(f"Fetching {WARMUP_DAYS}-day warm-up history for {len(instrument_map)} instruments…")
    ticker_buffers = warmup_ticker_buffers(ic, conn, instrument_map, warmup_days=WARMUP_DAYS)

    # ── 6. Build runner ───────────────────────────────────────────────────────
    runner = PaperTradingRunner(
        conn=conn, ic=ic, config=config,
        instrument_map=instrument_map,
        ticker_buffers=ticker_buffers,
        capital=args.capital,
    )

    # ── 7. Restore or fresh state ─────────────────────────────────────────────
    if not args.fresh:
        if load_state(runner.portfolio, ticker_buffers):
            logger.info("Portfolio state restored.")
    else:
        logger.info("Fresh start (--fresh).")

    # ── 8. Run ────────────────────────────────────────────────────────────────
    runner.run_forever()


if __name__ == "__main__":
    main()
