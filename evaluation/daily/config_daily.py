"""
config_daily.py — Config for ADTS daily paper trading engine.

Indicator parameters are IDENTICAL to the original backtest
(strategy1_dualbasket1.py).  Only timing constants differ from the
5-min engine.

Daily bar cadence (NSE):
  EOD eval   : 15:35 IST  (5 min after close; daily bar is complete)
  Morning exec: 09:17 IST  (2 min after open; fetch today's open via 1-min bar)
"""

import pytz

# ── Timezone ──────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

# ── Market hours ──────────────────────────────────────────────────────────────
MARKET_OPEN_H    = 9
MARKET_OPEN_M    = 15
MARKET_CLOSE_H   = 15
MARKET_CLOSE_M   = 30
EXCHANGE         = "NSE"

# ── Indicator windows (daily bars — identical to original backtest) ────────────
EMA_FAST             = 20
EMA_SLOW             = 100
ROLLING_WINDOW       = 756      # ≈ 3 trading years
MIN_ROLLING_POINTS   = 500
MIN_SLOPE_PCT        = 0.01
CORR_LOOKBACK        = 60       # trading days (~3 months)
MIN_ALIGNED_DAYS     = 500

# ── Portfolio ──────────────────────────────────────────────────────────────────
N_SLOTS          = 2
POSITION_SIZE    = 10_000_000

# ── Data ──────────────────────────────────────────────────────────────────────
# Need 756 + buffer daily bars ≈ 3.5 trading years
WARMUP_TRADING_DAYS  = ROLLING_WINDOW + 200     # bars to request
WARMUP_CALENDAR_DAYS = 1300                      # calendar days to cover that many trading days
HISTORICAL_TF_DAY    = "day"                     # conn.TIMEFRAME_TYPE_DAY value
HISTORICAL_TF_MIN    = "1"                       # conn.TIMEFRAME_TYPE_MIN — for open-price fetch

# ── Scheduler timing ──────────────────────────────────────────────────────────
EOD_EVAL_H    = 15
EOD_EVAL_M    = 35   # fetch completed daily bar + run signals
MORNING_EXEC_H = 9
MORNING_EXEC_M = 17  # fetch today's 1-min open bar → execute pending orders

# ── State + logging ───────────────────────────────────────────────────────────
STATE_FILE      = "state/daily_portfolio_state.json"
TRADE_LOG_FILE  = "state/daily_trade_log.csv"
LOG_LEVEL       = "INFO"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASKET_CSV_PATH      = "data/baskets_nifty200_all_sizes.csv"
TARGET_BASKET_SIZE   = 6
DOTENV_FILE          = ".env"