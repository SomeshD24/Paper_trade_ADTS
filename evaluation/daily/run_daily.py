"""
run_daily.py — ADTS daily paper trading engine.

Requires:
    pip install pyintegrate pandas numpy scipy pytz python-dotenv

Usage:
    python daily/run_daily.py [--basket-csv path] [--basket-size 6]
                               [--capital 100000] [--fresh] [--totp 123456]

Logic (identical to strategy1_dualbasket1.py backtest):
  ┌─ Day N market close (15:30) ──────────────────────────────────────────────┐
  │  15:35  fetch today's completed daily bar → append to buffers             │
  │         recompute indicators → evaluate OR-gate signals                   │
  │         queue exits (reg +2σ or trailing SL) + queue entries              │
  │         save state                                                        │
  └───────────────────────────────────────────────────────────────────────────┘
  ┌─ Day N+1 market open (09:15) ─────────────────────────────────────────────┐
  │  09:17  fetch today's 09:15 open via 1-min bar for each pending ticker    │
  │         execute queued exits + entries at those open prices               │
  │         save state                                                        │
  └───────────────────────────────────────────────────────────────────────────┘

All orders are PAPER ONLY — no real orders placed via IntegrateOrders.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))           # daily/
sys.path.insert(0, str(_HERE.parent))    # paper_trading/  (shared modules)

from config_daily import (
    IST, MARKET_OPEN_H, MARKET_OPEN_M, MARKET_CLOSE_H, MARKET_CLOSE_M,
    EOD_EVAL_H, EOD_EVAL_M, MORNING_EXEC_H, MORNING_EXEC_M,
    POSITION_SIZE, N_SLOTS, TARGET_BASKET_SIZE,
    BASKET_CSV_PATH, STATE_FILE, TRADE_LOG_FILE, LOG_LEVEL,
    DOTENV_FILE, WARMUP_CALENDAR_DAYS,
)
from data_manager_daily import (
    warmup_daily_buffers, DailyTickerBuffer, fetch_todays_open, fetch_live_ltp,
)
from signal_engine_daily import DailySignalEngine, build_basket_info
from symbol_mapper import (
    load_symbol_master, build_basket_instrument_map,
    all_unique_tickers_from_configs,
)

# Reuse shared portfolio + state modules unchanged
from portfolio_engine import PortfolioEngine
from state_store import save_state, load_state

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_daily")


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
        raise ValueError(f"No baskets for size={target_size}")
    return {"label": f"{target_size}-stock", "basket_size": target_size, "members": df}


# ── .env / credential helpers ────────────────────────────────────────────────

def _load_env() -> dict:
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(
            find_dotenv(filename=DOTENV_FILE, raise_error_if_not_found=False) or DOTENV_FILE,
            override=True,
        )
    except ImportError:
        logger.warning("python-dotenv not installed — reading os.environ only.")
    import os
    return {k: os.environ.get(k, "") for k in (
        "INTEGRATE_API_TOKEN", "INTEGRATE_API_SECRET",
        "INTEGRATE_UID", "INTEGRATE_ACTID",
        "INTEGRATE_API_SESSION_KEY", "INTEGRATE_WS_SESSION_KEY",
        "INTEGRATE_SESSION_SAVED_AT",
    )}


def _save_session_to_env(uid, actid, api_sk, ws_sk):
    try:
        from dotenv import set_key, find_dotenv
        env_path = (
            find_dotenv(filename=DOTENV_FILE, raise_error_if_not_found=False) or DOTENV_FILE
        )
        set_key(env_path, "INTEGRATE_UID",              uid)
        set_key(env_path, "INTEGRATE_ACTID",            actid)
        set_key(env_path, "INTEGRATE_API_SESSION_KEY",  api_sk)
        set_key(env_path, "INTEGRATE_WS_SESSION_KEY",   ws_sk)
        set_key(env_path, "INTEGRATE_SESSION_SAVED_AT", datetime.now().isoformat())
        logger.info(f"Session keys saved to {env_path}")
    except Exception as e:
        logger.warning(f"Could not write session keys to .env: {e}")


def _init_connection(api_token: str, api_secret: str, totp: str | None):
    from integrate import ConnectToIntegrate, IntegrateData
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    conn = ConnectToIntegrate()
    conn._verify = False
    
    # If TOTP was provided via CLI arg, use it immediately
    if totp is not None:
        logger.info("Logging in to Definedge Integrate with provided TOTP…")
        conn.login(api_token=api_token, api_secret=api_secret, totp=totp)
        uid, actid, api_sk, ws_sk = conn.get_session_keys()
        _save_session_to_env(uid, actid, api_sk, ws_sk)
        logger.info("Login successful.")
        return conn, IntegrateData(conn)

    # Cloud deployment friendly: wait for Streamlit UI to handle auth
    logger.info("Checking for valid session in .env...")
    while True:
        env = _load_env()
        saved_at  = env.get("INTEGRATE_SESSION_SAVED_AT", "")
        uid_saved = env.get("INTEGRATE_UID", "")
        
        if saved_at and uid_saved:
            try:
                age_h = (datetime.now() - datetime.fromisoformat(saved_at)).total_seconds() / 3600
                if age_h < 23:
                    conn.set_session_keys(
                        uid_saved, env["INTEGRATE_ACTID"],
                        env["INTEGRATE_API_SESSION_KEY"],
                        env["INTEGRATE_WS_SESSION_KEY"],
                    )
                    logger.info(f"Session keys restored from .env (age {age_h:.1f} h)")
                    return conn, IntegrateData(conn)
            except Exception as e:
                pass
                
        logger.info("No valid session found. Waiting for authentication via Streamlit dashboard...")
        time.sleep(30)


# ── Timing helpers ────────────────────────────────────────────────────────────

def _is_market_day(dt: datetime) -> bool:
    return dt.weekday() < 5

def _now_ist() -> datetime:
    return datetime.now(IST)

def _next_occurrence(h: int, m: int, after: datetime | None = None) -> datetime:
    """Next datetime with hour:minute >= now (or after), possibly tomorrow."""
    base = (after or _now_ist()).replace(
        hour=h, minute=m, second=0, microsecond=0
    )
    now = after or _now_ist()
    if base <= now:
        base += timedelta(days=1)
    # Skip weekends
    while base.weekday() >= 5:
        base += timedelta(days=1)
    return base

def _sleep_until(target: datetime):
    """Sleep in short chunks until target time, logging countdown."""
    while True:
        remaining = (target - _now_ist()).total_seconds()
        if remaining <= 0:
            return
        chunk = min(remaining, 300)   # log every 5 min at most
        if remaining > 60:
            logger.debug(f"  Sleeping {remaining/60:.1f} min until {target.strftime('%H:%M:%S %Z')}")
        time.sleep(max(chunk, 1))


# ── Daily trading runner ──────────────────────────────────────────────────────

class DailyTradingRunner:
    """
    Orchestrates the daily paper trading loop.

    State machine per day:
      PHASE_WAIT_EOD   → sleep until 15:35
      PHASE_EOD_EVAL   → fetch bar, compute signals, queue orders
      PHASE_WAIT_OPEN  → sleep until next trading day 09:17
      PHASE_EXEC       → fetch opens, execute pending orders
      back to PHASE_WAIT_EOD
    """

    def __init__(self, conn, ic, config: dict, instrument_map: dict,
                 ticker_buffers: dict, capital: float = POSITION_SIZE):
        self.conn           = conn
        self.ic             = ic
        self.instrument_map = instrument_map
        self.ticker_buffers = ticker_buffers   # {ticker: DailyTickerBuffer}

        basket_info = build_basket_info(config)
        self.portfolio  = PortfolioEngine(initial_capital=capital, n_slots=N_SLOTS)
        self.signal_eng = DailySignalEngine(
            basket_info, ticker_buffers, instrument_map,
            position_size=capital / N_SLOTS,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    # ── Unrealized PnL helpers ────────────────────────────────────────────────

    def _fetch_current_prices(self) -> dict:
        """
        Fetch the latest LTP for every ticker held in open slots.
        Returns {ticker: ltp_float}.  Tickers that fail return nothing.
        """
        needed: set[str] = set()
        for slot in self.portfolio.slots:
            if slot is not None:
                needed.update(slot["tickers"])
        prices = {}
        for ticker in needed:
            info = self.instrument_map.get(ticker)
            if not info:
                continue
            ltp = fetch_live_ltp(self.ic, self.conn, info["trading_symbol"])
            if ltp is not None:
                prices[ticker] = ltp
        return prices

    def _log_unrealized_pnl(self):
        """
        Fetch live LTPs and log a detailed unrealized PnL breakdown for
        each open slot.  Safe to call any time during market hours.
        """
        active_slots = [s for s in self.portfolio.slots if s is not None]
        if not active_slots:
            logger.info("  Unrealized PnL: no open positions.")
            return

        live = self._fetch_current_prices()
        if not live:
            logger.info("  Unrealized PnL: could not fetch live prices.")
            return

        now_str = _now_ist().strftime("%H:%M:%S")
        logger.info(f"── Unrealized PnL snapshot {now_str} {'─' * 32}")

        total_invest = 0.0
        total_live   = 0.0

        for slot in active_slots:
            bid      = slot["basket_id"]
            tickers  = slot["tickers"]
            entry_dt = slot["entry_time"]
            entry_date = entry_dt.strftime("%d-%b") if hasattr(entry_dt, 'strftime') else str(entry_dt)[:10]
            invest   = slot["investment"]

            slot_live_val = 0.0
            all_prices_ok = True
            ticker_lines  = []

            for t in tickers:
                ep = slot["entry_prices"].get(t, 0.0)
                qty = slot["quantities"].get(t, 0.0)
                ltp = live.get(t)
                if ltp is None:
                    all_prices_ok = False
                    ticker_lines.append(f"    {t:20s}  entry={ep:>9.2f}  ltp=N/A")
                    continue
                mkt_val  = qty * ltp
                cost_val = qty * ep
                tpnl     = mkt_val - cost_val
                tpnl_pct = tpnl / cost_val * 100 if cost_val else 0.0
                slot_live_val += mkt_val
                sign = "+" if tpnl >= 0 else ""
                ticker_lines.append(
                    f"    {t:20s}  entry={ep:>9.2f}  ltp={ltp:>9.2f}  "
                    f"qty={qty:>8.2f}  pnl={sign}{tpnl:>10,.0f} ({sign}{tpnl_pct:.2f}%)"
                )

            slot_pnl     = slot_live_val - invest if all_prices_ok else float("nan")
            slot_pnl_pct = slot_pnl / invest * 100 if (all_prices_ok and invest) else float("nan")
            pnl_str = (f"{'+' if slot_pnl >= 0 else ''}{slot_pnl:>10,.0f} "
                       f"({'+' if slot_pnl >= 0 else ''}{slot_pnl_pct:.2f}%)"
                       if all_prices_ok else "N/A")

            entry_type = slot.get("entry_type", "")
            logger.info(
                f"  Basket {bid:2d} [{entry_type:15s}] entered {entry_date}  "
                f"invest={invest:>12,.0f}  unrealized={pnl_str}"
            )
            for line in ticker_lines:
                logger.info(line)

            total_invest += invest
            if all_prices_ok:
                total_live += slot_live_val

        if total_invest > 0 and total_live > 0:
            total_pnl     = total_live - total_invest
            total_pnl_pct = total_pnl / total_invest * 100
            sign = "+" if total_pnl >= 0 else ""
            logger.info(
                f"  {'─' * 60}\n"
                f"  TOTAL  invest={total_invest:>12,.0f}  "
                f"live={total_live:>12,.0f}  "
                f"unrealized={sign}{total_pnl:>10,.0f} ({sign}{total_pnl_pct:.2f}%)"
            )
        logger.info(f"{'─' * 64}")

    def _sleep_with_pnl_updates(self, target: datetime, interval_min: int = 30):
        """
        Sleep until `target`, waking every `interval_min` minutes to print
        unrealized PnL if there are open positions.
        """
        while True:
            remaining = (target - _now_ist()).total_seconds()
            if remaining <= 0:
                return
            chunk = min(remaining, interval_min * 60)
            if remaining > 60:
                logger.debug(
                    f"  Sleeping {remaining/60:.1f} min until "
                    f"{target.strftime('%H:%M:%S %Z')}"
                )
            time.sleep(max(chunk, 1))
            # Wake up: print PnL if still in market hours and positions are open
            now = _now_ist()
            if now < target and any(s is not None for s in self.portfolio.slots):
                self._log_unrealized_pnl()

    def run_forever(self):
        logger.info("=" * 60)
        logger.info("ADTS Daily Paper Trader — LIVE")
        logger.info(f"  {self.signal_eng.warmup_status()}")
        logger.info(f"  Capital: {self.portfolio.base_capital:,.0f}   Slots: {N_SLOTS}")
        logger.info("=" * 60)
        # Show current unrealized PnL at startup if positions exist and market is open
        now_startup = _now_ist()
        market_open_today  = now_startup.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,  second=0, microsecond=0)
        market_close_today = now_startup.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
        if (market_open_today <= now_startup <= market_close_today
                and _is_market_day(now_startup)
                and any(s is not None for s in self.portfolio.slots)):
            self._log_unrealized_pnl()

        try:
            while True:
                now = _now_ist()

                # Skip non-trading days
                if not _is_market_day(now):
                    next_mon = _next_occurrence(EOD_EVAL_H, EOD_EVAL_M)
                    logger.info(f"Weekend. Sleeping until {next_mon.strftime('%a %d-%b %H:%M')}.")
                    _sleep_until(next_mon)
                    continue

                eod_today  = now.replace(
                    hour=EOD_EVAL_H, minute=EOD_EVAL_M, second=0, microsecond=0
                )
                exec_today = now.replace(
                    hour=MORNING_EXEC_H, minute=MORNING_EXEC_M, second=0, microsecond=0
                )

                # ── Case 1: Before 09:17 — pending orders may need execution ─
                if now < exec_today:
                    if self.portfolio._pending_exits or self.portfolio._pending_entries:
                        logger.info(f"Waiting for market open to execute pending orders…")
                        _sleep_until(exec_today)
                        self._morning_execution()
                    else:
                        # Nothing pending; sleep to EOD eval
                        logger.info(f"No pending orders. Sleeping to EOD eval {eod_today.strftime('%H:%M')}.")
                        _sleep_until(eod_today)
                        self._eod_evaluation()

                # ── Case 2: Between 09:17 and 15:35 — market is open ─────────
                elif exec_today <= now < eod_today:
                    # Execute any still-pending orders (e.g. runner restarted mid-day)
                    if self.portfolio._pending_exits or self.portfolio._pending_entries:
                        self._morning_execution()
                    # Show PnL immediately if positions are open
                    if any(s is not None for s in self.portfolio.slots):
                        self._log_unrealized_pnl()
                    logger.info(f"Market open. Sleeping to EOD eval {eod_today.strftime('%H:%M')}.")
                    self._sleep_with_pnl_updates(eod_today)
                    self._eod_evaluation()

                # ── Case 3: After 15:35 — run EOD if not yet done today ───────
                else:
                    self._eod_evaluation()
                    # Now sleep until next trading day's morning exec
                    next_exec = _next_occurrence(MORNING_EXEC_H, MORNING_EXEC_M)
                    logger.info(f"EOD done. Sleeping until {next_exec.strftime('%a %d-%b %H:%M')}.")
                    _sleep_until(next_exec)
                    self._morning_execution()

        except KeyboardInterrupt:
            logger.info("Interrupted — saving state.")
            save_state(
                self.portfolio,
                self.ticker_buffers,
                STATE_FILE, TRADE_LOG_FILE,
            )

    # ── EOD: fetch bar + evaluate signals ────────────────────────────────────

    def _eod_evaluation(self):
        now = _now_ist()
        day_ts = pd.Timestamp(now.date()).tz_localize('Asia/Kolkata')
        logger.info(f"── EOD Eval {now.strftime('%d-%b-%Y')} ──────────────────────────────")

        # 1. Append today's daily bar to all buffers
        fetched = 0
        for ticker, info in self.instrument_map.items():
            buf = self.ticker_buffers[ticker]
            ok  = buf.append_today(self.ic, self.conn, info["trading_symbol"])
            if ok:
                fetched += 1
        logger.info(f"  Daily bars fetched: {fetched}/{len(self.instrument_map)}")

        # 2. Evaluate signals
        result        = self.signal_eng.on_day_close(day_ts)
        basket_closes = result["basket_closes"]

        # 3. Update trailing peaks + close series
        self.portfolio.on_bar_close(day_ts, basket_closes)

        # 4. Exit signals
        self.portfolio.check_exit_signals(result["sell_signals"], basket_closes)

        # 5. Entry signals
        self.portfolio.queue_entries(result["entry_signals"], day_ts)

        # 6. MTM log
        snap = self.portfolio.mtm_snapshot(basket_closes)
        logger.info(
            f"  MTM  realized={snap['realized_pnl']:+,.0f}  "
            f"unrealized={snap['unrealized_pnl']:+,.0f}  "
            f"equity={snap['total_equity']:,.0f}"
        )
        logger.info(f"  {self.signal_eng.warmup_status()}")
        pending_e = len(self.portfolio._pending_entries)
        pending_x = len(self.portfolio._pending_exits)
        logger.info(f"  Pending orders: {pending_e} entries  {pending_x} exits → execute at 09:17 tomorrow")

        # 7. Save state
        self._save()

    # ── Morning: execute pending orders at today's open ───────────────────────

    def _morning_execution(self):
        if not (self.portfolio._pending_exits or self.portfolio._pending_entries):
            # Still log PnL at open even with nothing pending
            if any(s is not None for s in self.portfolio.slots):
                self._log_unrealized_pnl()
            return

        now = _now_ist()
        exec_ts = pd.Timestamp(now).tz_convert('Asia/Kolkata')
        logger.info(f"── Morning Exec {now.strftime('%d-%b-%Y %H:%M')} ─────────────────────")

        # Collect all tickers needed for pending orders
        needed_tickers: set[str] = set()
        for slot in self.portfolio.slots:
            if slot is not None:
                needed_tickers.update(slot["tickers"])
        for bid, etype, capital, needs_evict, evict_idx in self.portfolio._pending_entries:
            info = self.signal_eng.basket_info.get(bid, {})
            needed_tickers.update(info.get("tickers", []))
        for slot_idx, reason in self.portfolio._pending_exits:
            slot = self.portfolio.slots[slot_idx]
            if slot:
                needed_tickers.update(slot["tickers"])

        # Fetch today's open price for each needed ticker
        open_prices: dict[str, float] = {}
        for ticker in needed_tickers:
            info = self.instrument_map.get(ticker)
            if not info:
                continue
            price = fetch_todays_open(self.ic, self.conn, info["trading_symbol"])
            if price is not None:
                open_prices[ticker] = price
                logger.info(f"  {ticker}: open={price:.2f}")
            else:
                logger.warning(f"  {ticker}: could not fetch open price")

        if not open_prices:
            logger.warning("  No open prices fetched — deferring execution.")
            return

        exec_prices = self.signal_eng.get_exec_prices(open_prices)
        basket_info = self.signal_eng.basket_info
        new_trades  = self.portfolio.execute_pending(exec_prices, exec_ts, basket_info)

        if new_trades:
            for t in new_trades:
                logger.info(
                    f"  EXEC {t['close_reason']} B{t['basket_id']} "
                    f"pnl={t['pnl']:+.0f} ({t['pnl_pct']:+.2f}%)"
                )
        else:
            logger.info("  No trades executed (missing prices?).")

        self._save()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        # state_store expects {ticker: TickerBuffer} with .df property —
        # DailyTickerBuffer has the same interface.
        save_state(
            self.portfolio,
            self.ticker_buffers,
            STATE_FILE,
            TRADE_LOG_FILE,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ADTS Daily Paper Trader",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Credentials are read from .env (one level up from daily/).\n"
            "Session keys are written back into .env automatically after login."
        ),
    )
    parser.add_argument("--basket-csv",  default=BASKET_CSV_PATH)
    parser.add_argument("--basket-size", type=int,   default=TARGET_BASKET_SIZE)
    parser.add_argument("--capital",     type=float, default=POSITION_SIZE)
    parser.add_argument("--totp",        default=None,
                        help="6-digit TOTP (prompted if omitted and login needed)")
    parser.add_argument("--fresh",       action="store_true",
                        help="Ignore saved portfolio state; start clean")
    args = parser.parse_args()

    Path("state").mkdir(exist_ok=True)

    # ── 1. Credentials ────────────────────────────────────────────────────────
    env        = _load_env()
    api_token  = env.get("INTEGRATE_API_TOKEN",  "")
    api_secret = env.get("INTEGRATE_API_SECRET", "")
    if not api_token or not api_secret:
        parser.error(
            "Set INTEGRATE_API_TOKEN and INTEGRATE_API_SECRET in .env "
            "(see .env.example in the paper_trading/ root)."
        )
    conn, ic = _init_connection(api_token, api_secret, args.totp)

    # ── 2. Basket config ──────────────────────────────────────────────────────
    logger.info(f"Loading basket config: {args.basket_csv} (size={args.basket_size})")
    config = _load_basket_config(args.basket_csv, args.basket_size)
    all_tickers = all_unique_tickers_from_configs([config])
    basket_info = build_basket_info(config)
    logger.info(f"  {len(basket_info)} baskets  {len(all_tickers)} unique tickers")

    # ── 3. Symbol map ─────────────────────────────────────────────────────────
    logger.info("Building symbol map from broker master…")
    load_symbol_master(conn)
    instrument_map = build_basket_instrument_map(all_tickers)
    unmapped = set(all_tickers) - set(instrument_map)
    if unmapped:
        logger.warning(f"  {len(unmapped)} tickers unmapped: {sorted(unmapped)}")
    logger.info(f"  {len(instrument_map)} tickers mapped")

    # ── 4. Warm up daily buffers ──────────────────────────────────────────────
    logger.info(f"Fetching {WARMUP_CALENDAR_DAYS}-day daily history…")
    ticker_buffers = warmup_daily_buffers(ic, conn, instrument_map, WARMUP_CALENDAR_DAYS)

    # ── 5. Build runner ───────────────────────────────────────────────────────
    runner = DailyTradingRunner(
        conn=conn, ic=ic, config=config,
        instrument_map=instrument_map,
        ticker_buffers=ticker_buffers,
        capital=args.capital,
    )

    # ── 6. Restore / fresh state ──────────────────────────────────────────────
    if not args.fresh:
        if load_state(runner.portfolio, ticker_buffers, STATE_FILE):
            logger.info("Daily portfolio state restored.")
    else:
        logger.info("Fresh start (--fresh).")

    # ── 7. Run ────────────────────────────────────────────────────────────────
    runner.run_forever()


if __name__ == "__main__":
    main()
