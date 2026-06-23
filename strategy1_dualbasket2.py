"""
Basket OR Gate — Dual-Basket Portfolio (EOD Signal -> Next-Open Execution)
Strategy1's OR-gate entry/exit signals run through strategy2a's correlation-based
dual-basket (2-slot) portfolio engine, instead of strategy1's original single-slot sim.

Signal timing : ALL signals computed on PREVIOUS day's close -> trade executes on
                  NEXT day's open (unchanged from strategy1.py).
                  - EMA crossover : detected via ema20.shift(1)/shift(2) -> confirmed EOD yesterday
                  - Regression    : close[i-1] vs bands[i-1] crossover  -> confirmed EOD yesterday

Entry         : regression -2sigma crossover OR EMA{EMA_FAST}/{EMA_SLOW} golden cross
                (regression priority same day). Per-basket entries are sequential
                (one virtual position per basket at a time), computed by run_or_gate_strategy()
                exactly as in strategy1.py.
Exit          : EARLIEST exit signal fires, regardless of entry type:
                  - regression_2sd  - prev close crosses above upper2 (+2sigma)
                  - ema_death_cross - EMA{EMA_FAST} crosses below EMA{EMA_SLOW}
                Same-bar priority: regression_2sd > ema_death_cross.

Portfolio     : Two-slot dual-basket simulator, 50/50 capital split (engine copied
                from strategy2a.py, correlation-based selection/eviction):
                  - New signal takes a free slot if one exists (peer's return baseline resets).
                  - If multiple baskets signal entry on the same day and there are more
                    candidates than free slots, the mutually LEAST-correlated subset is kept.
                  - If both slots full, mean pairwise correlation is computed among the
                    group {held basket 1, held basket 2, incoming new basket}. Whichever
                    member has the HIGHEST mean correlation to the other two is the most
                    redundant. If that's a held basket, it's evicted and replaced by the
                    new entry. If it's the new (incoming) basket itself, NO eviction
                    happens and the new entry is skipped (swap would not diversify).
                  - Same basket_id can't occupy two slots; ties broken by basket_id order.

Multi-size    : load_basket_configs() — identical to strategy1.py / strategy2a.py.
"""

import warnings
import math
warnings.filterwarnings('ignore')

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = 'plotly_dark'

ACCENT = '#00D4FF'
GREEN  = '#00FF88'
RED    = '#FF4466'
YELLOW = '#FFD700'
ORANGE = '#FF8C00'
PURPLE = '#BB86FC'
TEAL   = '#00CED1'
LIME   = '#ADFF2F'
PINK   = '#FF69B4'
CORAL  = '#FF6B6B'
BG     = '#0D1117'
GRID   = '#1E2730'
TEXT   = '#C9D1D9'

SIZE_COLORS = [ACCENT, ORANGE, GREEN, PURPLE, YELLOW, PINK, TEAL, LIME, CORAL, RED]

LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family='monospace', color=TEXT),
    xaxis=dict(gridcolor=GRID, showgrid=True),
    yaxis=dict(gridcolor=GRID, showgrid=True),
    legend=dict(bgcolor='#1A222C', bordercolor=GRID),
    margin=dict(l=60, r=60, t=70, b=50),
)

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

# ── Config ─────────────────────────────────────────────────────────────────────
START_DATE          = '2003-01-01'
END_DATE             = None
POSITION_SIZE        = 100_000
MIN_ALIGNED_DAYS     = 500
ROLLING_WINDOW       = 756        # ≈ 3 trading years
MIN_ROLLING_POINTS   = 500

EMA_FAST             = 20
EMA_SLOW             = 100

N_SLOTS              = 2          # dual-basket portfolio (50/50 split)
CORR_LOOKBACK        = 60         # trading days of returns used for basket-to-basket correlation

RF_ANNUAL            = 0.06       # India risk-free proxy (long-run RBI repo rate)
RF_DAILY             = (1 + RF_ANNUAL) ** (1 / 252) - 1   # compounded daily RF, for Sharpe/Sortino

DATA_CACHE = {}
MIN_SLOPE_PCT = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def normalize_yf_ohlc(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(c not in raw.columns for c in required):
        return pd.DataFrame()
    return raw[required].dropna().sort_index()


def load_ohlc(ticker, start=START_DATE, end=END_DATE):
    if ticker in DATA_CACHE:
        return DATA_CACHE[ticker]
    for label, fn in [
        ('download_start_end', lambda: yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)),
        ('history_start_end',  lambda: yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)),
        ('download_max',       lambda: yf.download(ticker, period='max', auto_adjust=True, progress=False)),
        ('history_max',        lambda: yf.Ticker(ticker).history(period='max', auto_adjust=True)),
    ]:
        try:
            df = normalize_yf_ohlc(fn())
            if not df.empty and start is not None:
                df = df.loc[pd.to_datetime(start):]
            if not df.empty:
                DATA_CACHE[ticker] = df
                print(f'  [{ticker}] {len(df):,} days via {label}')
                return df
        except Exception:
            pass
    raise ValueError(f'No data for {ticker}')


def build_equal_weight_basket_ohlc(tickers, total_amount=POSITION_SIZE):
    """Equal-weight basket OHLC for any number of stocks."""
    fields  = ['Open', 'High', 'Low', 'Close', 'Volume']
    weights = np.repeat(1 / len(tickers), len(tickers))
    loaded  = [load_ohlc(t)[fields].rename(columns={c: (t, c) for c in fields}) for t in tickers]
    panel   = pd.concat(loaded, axis=1).dropna()
    if len(panel) < MIN_ALIGNED_DAYS:
        raise ValueError(f'Only {len(panel)} aligned days (need {MIN_ALIGNED_DAYS})')
    first_close = pd.Series({t: panel[(t, 'Close')].iloc[0] for t in tickers}, dtype=float)
    quantities  = (weights * total_amount) / first_close
    basket = pd.DataFrame(index=panel.index)
    for field in ['Open', 'High', 'Low', 'Close']:
        basket[field] = sum(panel[(t, field)] * quantities[t] for t in tickers)
    basket['Volume'] = sum(panel[(t, 'Volume')] for t in tickers)
    comp_close = pd.DataFrame({t: panel[(t, 'Close')] for t in tickers}, index=panel.index)
    comp_open  = pd.DataFrame({t: panel[(t, 'Open')]  for t in tickers}, index=panel.index)
    basket.attrs['component_close'] = comp_close.sort_index()
    basket.attrs['component_open']  = comp_open.sort_index()
    return basket.dropna().sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# BASKET CONFIG LOADER
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLS = {'basket_id', 'stock_position', 'symbol', 'ticker', 'company_name', 'sector'}


def _make_configs_from_df(df, source_label):
    """
    Split a DataFrame into one config per basket_size.
    If the DataFrame has no basket_size column, treat the whole file as one config
    and infer basket_size from the mode of stocks-per-basket.
    """
    configs = []

    if 'basket_size' in df.columns:
        for size, grp in df.groupby('basket_size'):
            grp = grp.sort_values(['basket_id', 'stock_position']).reset_index(drop=True)
            label = f'{int(size)}-stock'
            configs.append({
                'label':       label,
                'basket_size': int(size),
                'source':      source_label,
                'members':     grp,
            })
    else:
        size_counts = df.groupby('basket_id')['ticker'].count()
        basket_size = int(size_counts.mode().iloc[0])
        df = df.sort_values(['basket_id', 'stock_position']).reset_index(drop=True)
        label = f'{basket_size}-stock ({source_label})'
        configs.append({
            'label':       label,
            'basket_size': basket_size,
            'source':      source_label,
            'members':     df,
        })

    return configs


def load_basket_configs(csv_paths=None, search_dirs=None):
    """
    Load basket configs from one or more CSV files.

    Priority:
      1. Explicit `csv_paths` list (Path or str).
      2. Glob for baskets_*.csv / *basket*.csv in each dir of `search_dirs`.
      3. Default fallback: baskets_nifty200_all_sizes.csv and baskets_nifty200_diversified.csv
         in CWD and data/.

    A CSV with a `basket_size` column is split into one config per unique size.
    A CSV without that column is treated as one config (size inferred from data).

    Returns list of config dicts sorted by basket_size, then source.
    """
    candidates: list[Path] = []

    if csv_paths:
        for p in csv_paths:
            candidates.append(Path(p))

    if search_dirs:
        for d in search_dirs:
            d = Path(d)
            if d.exists():
                for pat in ['baskets_*.csv', '*basket*.csv']:
                    candidates.extend(sorted(d.glob(pat)))

    # default fallback
    for fb in [
        'baskets_nifty200_all_sizes.csv',
        'data/baskets_nifty200_all_sizes.csv',
        'baskets_nifty200_diversified.csv',
        'data/baskets_nifty200_diversified.csv',
    ]:
        candidates.append(Path(fb))

    # deduplicate
    seen, unique = set(), []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            unique.append(p)

    if not unique:
        raise FileNotFoundError(
            'No basket CSV found. Place baskets_nifty200_all_sizes.csv in the working dir '
            'or pass csv_paths explicitly.'
        )

    all_configs = []
    for csv_path in unique:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f'[SKIP] {csv_path}: {e}')
            continue
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            print(f'[SKIP] {csv_path}: missing columns {missing}')
            continue
        configs = _make_configs_from_df(df, csv_path.name)
        for c in configs:
            n = c['members']['basket_id'].nunique()
            print(f'  {c["label"]:20s}  {n:3d} baskets  ← {csv_path.name}')
        all_configs.extend(configs)

    if not all_configs:
        raise ValueError('No valid basket configs loaded.')

    all_configs.sort(key=lambda c: (c['basket_size'], c['source']))
    return all_configs


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS  (previous-close signals — zero lookahead)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_regression_bands(close, window=ROLLING_WINDOW, min_points=MIN_ROLLING_POINTS):
    """
    Rolling OLS regression bands computed up to and including day i.
    Strategy loop reads iloc[i-1] → no lookahead.
    """
    close = close.dropna()
    n     = len(close)
    trend   = np.full(n, np.nan)
    std_res = np.full(n, np.nan)
    for i in range(min_points - 1, n):
        start = max(0, i - window + 1)
        y     = close.iloc[start:i + 1].values.astype(float)
        if len(y) < min_points:
            continue
        x = np.arange(len(y), dtype=float)
        sl, ic, *_ = stats.linregress(x, y)
        fitted     = sl * x + ic
        trend[i]   = fitted[-1]
        std_res[i] = (y - fitted).std()
    return pd.DataFrame({
        'trend_line': trend,
        'std_res':    std_res,
        'lower2':     trend - 2 * std_res,
        'upper1':     trend + 1 * std_res,
        'upper2':     trend + 2 * std_res,
    }, index=close.index)


def ema_crossover_signals(close, ema_fast=EMA_FAST, ema_slow=EMA_SLOW):
    """
    EMA fast/slow signals on PREVIOUS close (unchanged logic from strategy1.py,
    just parameterised so EMA_FAST/EMA_SLOW can be swapped — e.g. 50/100).

      buy_signal[i]  = crossover (fast>slow) confirmed EOD at bar i-1 → execute open[i]
      sell_signal[i] = death-cross (fast<slow) confirmed EOD at bar i-1 → execute open[i]
    """
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    buy  = (
        (ema_f.shift(2) <  ema_s.shift(2)) &
        (ema_f.shift(1) >= ema_s.shift(1)) &
        (close.shift(1) >  ema_s.shift(1))
    )
    sell = (
        (ema_f.shift(2) >  ema_s.shift(2)) &
        (ema_f.shift(1) <= ema_s.shift(1))
    )
    return pd.DataFrame({
        'Close':       close,
        'EMA_fast':    ema_f,
        'EMA_slow':    ema_s,
        'buy_signal':  buy.fillna(False),
        'sell_signal': sell.fillna(False),
    }, index=close.index)


def or_gate_combined_signals(close, bands, ema_sig):
    """
    Combine the regression-band and EMA-crossover legs into a single OR-gate
    buy/sell signal series, indexed at EXECUTION day (matches ema_sig's shift
    convention). Used by:
      (a) the dual-basket portfolio engine's daily "natural exit" check, and
      (b) the candlestick plot.

    Entry (buy_signal)  = reg_entry OR ema_buy   (regression priority for the label)
    Exit  (sell_signal) = reg_exit  OR ema_sell   (regression priority for the label)
      reg_entry : yest_close crosses up through yest_lower2 (matches run_or_gate_strategy)
      reg_exit  : yest_close >= yest_upper2 (level check, not a crossing — matches
                  run_or_gate_strategy's in-trade exit test)
    """
    yest_close  = close.shift(1)
    prev_close  = close.shift(2)
    yest_lower2 = bands['lower2'].shift(1)
    prev_lower2 = bands['lower2'].shift(2)
    yest_upper2 = bands['upper2'].shift(1)

    reg_buy  = yest_lower2.notna() & (yest_close >= yest_lower2) & (prev_close < prev_lower2)
    reg_sell = yest_upper2.notna() & (yest_close >= yest_upper2)

    ema_buy  = ema_sig['buy_signal']
    ema_sell = ema_sig['sell_signal']

    reg_buy_f, ema_buy_f   = reg_buy.fillna(False),  ema_buy.fillna(False)
    reg_sell_f, ema_sell_f = reg_sell.fillna(False), ema_sell.fillna(False)

    buy_signal  = reg_buy_f | ema_buy_f
    sell_signal = reg_sell_f | ema_sell_f

    entry_type = np.where(reg_buy_f, 'regression', np.where(ema_buy_f, 'ema_crossover', ''))
    exit_type  = np.where(reg_sell_f, 'regression_2sd', np.where(ema_sell_f, 'ema_death_cross', ''))

    out = ema_sig.copy()
    out['buy_signal']  = buy_signal
    out['sell_signal'] = sell_signal
    out['entry_type']  = entry_type
    out['exit_type']   = exit_type
    return out


# ══════════════════════════════════════════════════════════════════════════════
# OR GATE STRATEGY — EARLIEST EXIT  (per-basket sequential trade builder)
# ══════════════════════════════════════════════════════════════════════════════

def run_or_gate_strategy(close, open_prices, bands, signals):
    """
    Entry  : regression -2σ crossover OR EMA golden cross (regression priority).
    Exit   : EARLIEST of regression_2sd or ema_death_cross, regardless of entry.
    All signal checks reference bar i-1 (previous close); execution at open[i].
    """
    n, dates = len(close), close.index
    trades   = []
    in_trade = buy_price = buy_date = entry_src = None
    in_trade = False

    for i in range(2, n):
        date_i     = dates[i]
        exec_price = open_prices.iloc[i]            # next open after signal

        yest_close  = close.iloc[i - 1]             # signal evaluation: previous close
        prev_close  = close.iloc[i - 2]
        yest_lower2 = bands['lower2'].iloc[i - 1]
        prev_lower2 = bands['lower2'].iloc[i - 2]
        yest_upper2 = bands['upper2'].iloc[i - 1]
        ema_buy     = signals['buy_signal'].iloc[i]
        ema_sell    = signals['sell_signal'].iloc[i]

        if not in_trade:
            reg_entry = (
                not np.isnan(yest_lower2) and
                yest_close >= yest_lower2 and
                prev_close <  prev_lower2
            )
            if reg_entry:
                buy_price, buy_date, in_trade, entry_src = exec_price, date_i, True, 'regression'
            elif ema_buy:
                buy_price, buy_date, in_trade, entry_src = exec_price, date_i, True, 'ema_crossover'
        else:
            fired = exit_type = None
            if not np.isnan(yest_upper2) and yest_close >= yest_upper2:
                fired, exit_type = True, 'regression_2sd'
            elif ema_sell:
                fired, exit_type = True, 'ema_death_cross'

            if fired:
                trades.append({
                    'buy_date':   buy_date,
                    'sell_date':  date_i,
                    'buy_price':  round(buy_price, 2),
                    'sell_price': round(exec_price, 2),
                    'pnl':        round(exec_price - buy_price, 2),
                    'hold_days':  (date_i - buy_date).days,
                    'entry_type': entry_src,
                    'exit_type':  exit_type,
                    'status':     'closed',
                })
                in_trade = False

    if in_trade:
        trades.append({
            'buy_date':   buy_date,
            'sell_date':  dates[-1],
            'buy_price':  round(buy_price, 2),
            'sell_price': round(open_prices.iloc[-1], 2),
            'pnl':        round(open_prices.iloc[-1] - buy_price, 2),
            'hold_days':  (dates[-1] - buy_date).days,
            'entry_type': entry_src,
            'exit_type':  'open_mtm',
            'status':     'open (MTM)',
        })
    return trades


# ══════════════════════════════════════════════════════════════════════════════
# DUAL-BASKET PORTFOLIO SIMULATION  (N_SLOTS slots, 50/50 split, correlation-based eviction/selection)
# Engine copied from strategy2a.py — slope_pct metadata replaced with entry_type
# (strategy1's OR-gate has no slope feature; entry_type labels 'regression' / 'ema_crossover').
# ══════════════════════════════════════════════════════════════════════════════

def _open_slot(basket_id, entry_date, entry_type, capital_to_deploy, results):
    """
    Open a basket position using today's OPEN as execution price.
    Signal was generated from yesterday's close.
    """
    comp_close = results[basket_id]['component_close']
    comp_open  = results[basket_id]['component_open']

    entry_ts = pd.Timestamp(entry_date)
    if entry_ts not in comp_open.index:
        avail = comp_open.index[comp_open.index >= entry_ts]
        if avail.empty:
            return None
        entry_ts = avail[0]

    entry_prices      = comp_open.loc[entry_ts].astype(float)
    per_stock_capital = capital_to_deploy / len(entry_prices)
    quantities        = np.floor(per_stock_capital / entry_prices).astype(int)
    investment        = float((quantities * entry_prices).sum())

    if investment <= 0:
        return None

    return {
        'basket_id':         basket_id,
        'entry_date':        entry_ts,
        'entry_type':        entry_type or '',
        'capital_allocated': round(capital_to_deploy, 2),
        'quantities':        quantities,
        'investment':        investment,
        'component_close':   comp_close,
        'component_open':    comp_open,
        'returns_ref_date':  entry_ts,
        'returns_ref_value': investment,
    }


def _reset_returns_reference(slot_state, reset_date):
    """Reset return-tracking baseline for the surviving slot after a new peer enters."""
    comp_close = slot_state['component_close']
    reset_ts   = pd.Timestamp(reset_date)
    if reset_ts not in comp_close.index:
        avail = comp_close.index[comp_close.index <= reset_ts]
        if avail.empty:
            return
        reset_ts = avail[-1]
    reset_value = float((slot_state['quantities'] * comp_close.loc[reset_ts].astype(float)).sum())
    if reset_value > 0:
        slot_state['returns_ref_date']  = reset_ts
        slot_state['returns_ref_value'] = reset_value


def _close_slot(slot_state, close_date, close_reason, results):
    """
    Close an active slot at today's OPEN price.
    close_reason: 'or_gate_exit' | 'evicted_new_entry' | 'mtm'
    Returns (trade_dict, realized_pnl_delta).
    """
    comp_open = slot_state['component_open']
    close_ts  = pd.Timestamp(close_date)
    if close_ts not in comp_open.index:
        avail = comp_open.index[comp_open.index >= close_ts]
        if avail.empty:
            comp_open = slot_state['component_close']
            avail2 = comp_open.index[comp_open.index <= close_ts]
            if avail2.empty:
                return None, 0.0
            close_ts = avail2[-1]
        else:
            close_ts = avail[0]

    is_mtm = (close_reason == 'mtm')
    if not is_mtm and close_ts == slot_state['entry_date']:
        return None, 0.0

    basket_id   = slot_state['basket_id']
    quantities  = slot_state['quantities']
    exit_prices = comp_open.loc[close_ts].astype(float)
    exit_value  = float((quantities * exit_prices).sum())
    investment  = slot_state['investment']
    pnl         = exit_value - investment

    trade = {
        'basket_id':         basket_id,
        'basket_symbols':    ', '.join(results[basket_id]['symbols']),
        'buy_date':          slot_state['entry_date'],
        'sell_date':         close_ts,
        'close_reason':      close_reason,
        'entry_type':        slot_state['entry_type'],
        'capital_allocated': slot_state['capital_allocated'],
        'investment':        round(investment, 2),
        'unused_cash':       round(slot_state['capital_allocated'] - investment, 2),
        'exit_value':        round(exit_value, 2),
        'pnl':               round(pnl, 2),
        'pnl_pct':           round(pnl / investment * 100, 2) if investment > 0 else 0.0,
        'hold_days':         (close_ts - slot_state['entry_date']).days,
        'status':            'open (MTM)' if is_mtm else 'closed',
    }
    realized_delta = 0.0 if is_mtm else pnl
    return trade, realized_delta


def _basket_return_window(basket_id, results, end_date, lookback=CORR_LOOKBACK):
    """
    Trailing daily-return window for a basket's combined Close price, ending
    at or just before `end_date`. Returns None if there isn't enough history
    to estimate a meaningful correlation yet.
    """
    close  = results[basket_id]['signals']['Close']
    end_ts = pd.Timestamp(end_date)
    avail  = close.index[close.index <= end_ts]
    if avail.empty:
        return None
    end_ts = avail[-1]
    window = close.loc[:end_ts].iloc[-(lookback + 1):]
    if len(window) < max(10, lookback // 3):
        return None
    rets = window.pct_change().dropna()
    return rets if not rets.empty else None


def _pairwise_correlation(basket_id_a, basket_id_b, results, end_date, lookback=CORR_LOOKBACK):
    """
    Pearson correlation of trailing daily returns between two baskets, as of
    `end_date`. Falls back to a neutral 0.0 when correlation can't be reliably
    estimated (insufficient/non-overlapping history) so missing data doesn't
    bias eviction/selection decisions.
    """
    if basket_id_a == basket_id_b:
        return 1.0
    ra = _basket_return_window(basket_id_a, results, end_date, lookback)
    rb = _basket_return_window(basket_id_b, results, end_date, lookback)
    if ra is None or rb is None:
        return 0.0
    aligned = pd.concat([ra, rb], axis=1, join='inner').dropna()
    if len(aligned) < 10:
        return 0.0
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(corr) if not np.isnan(corr) else 0.0


def _eviction_target(slot_basket_ids, new_bid, results, end_date, lookback=CORR_LOOKBACK):
    """
    Decide eviction among the group {held baskets..., incoming new basket}.

    For each member of this group, compute its MEAN pairwise correlation to
    the *other* members of the same group. The member with the highest mean
    correlation is the most redundant exposure in the group.

    - If a HELD basket is the most redundant -> return its index into
      `slot_basket_ids` (that slot gets evicted, new basket takes its place).
    - If the INCOMING new basket is itself the most redundant member (i.e.
      it would add more overlap than either held basket already does) ->
      return None. No eviction happens and the new entry is skipped, since
      swapping it in would not improve diversification.
    """
    members = list(slot_basket_ids) + [new_bid]
    n = len(members)
    if n < 2:
        return None

    corr = {}
    for a, b in combinations(range(n), 2):
        c = _pairwise_correlation(members[a], members[b], results, end_date, lookback)
        corr[(a, b)] = c
        corr[(b, a)] = c

    mean_corrs = [
        sum(corr[(i, j)] for j in range(n) if j != i) / (n - 1)
        for i in range(n)
    ]

    worst_member = int(np.argmax(mean_corrs))
    new_idx = n - 1  # incoming basket is always last in `members`

    if worst_member == new_idx:
        return None  # incoming basket is the most redundant -> skip entry

    return worst_member  # index into slot_basket_ids / slots


def _select_least_correlated_subset(basket_ids, k, results, end_date, lookback=CORR_LOOKBACK):
    """
    From `basket_ids`, choose the size-k subset that is mutually LEAST
    correlated (minimises the sum of pairwise correlations among members).

    Used when multiple new entries land on the same day and there are more
    candidates than free slots (e.g. both slots empty, n_slots=2 candidates
    arrive together) — keeps the most-diversified pair/group instead of
    picking by basket_id order.
    """
    basket_ids = list(basket_ids)
    if k <= 0 or len(basket_ids) <= k:
        return basket_ids

    best_subset, best_score = basket_ids[:k], np.inf
    for combo in combinations(basket_ids, k):
        score = sum(
            _pairwise_correlation(a, b, results, end_date, lookback)
            for a, b in combinations(combo, 2)
        )
        if score < best_score:
            best_score, best_subset = score, combo

    return list(best_subset)


def simulate_dual_basket_portfolio(results, strategy_key='or_gate',
                                    initial_capital=POSITION_SIZE, n_slots=N_SLOTS):
    """
    N-slot simulator (default 2 → 'dual basket'). Engine copied from strategy2a.py.
    - Executes at today's OPEN (signal from yesterday's close).
    - When multiple baskets signal on the same day and there are more
      candidates than free slots, the mutually LEAST-correlated subset of
      candidates is kept (most diversified group), not basket_id order.
    - Equal-split slots: when full, mean correlation among {held1, held2, new}
      decides eviction — the most-redundant member is removed. If that's the
      new basket itself, no eviction occurs and the entry is skipped. Return-
      reset on the surviving peer whenever a new position opens.
    - Natural exits use results[basket_id]['signals']['sell_signal'] — the
      OR-gate combined exit (regression_2sd OR ema_death_cross) produced by
      or_gate_combined_signals().

    Returns (port_trades, equity Series of CUMULATIVE REALIZED PnL indexed by sell_date).
    """
    all_signals = []
    for basket_id, res in results.items():
        for trade in res[strategy_key]['trades']:
            entry_type = trade.get('entry_type', '')
            all_signals.append((pd.Timestamp(trade['buy_date']), basket_id, entry_type))
    all_signals.sort(key=lambda x: (x[0], x[1]))

    if not all_signals:
        return [], pd.Series(dtype=float)

    basket_signals = {bid: res['signals'] for bid, res in results.items()}
    all_dates = sorted(set().union(*[set(sig.index) for sig in basket_signals.values()]))

    signals_by_date = defaultdict(list)
    for date, bid, entry_type in all_signals:
        signals_by_date[date].append((bid, entry_type))

    slots            = [None] * n_slots
    portfolio_trades = []
    equity_points    = []
    realized_pnl     = 0.0

    def base_capital():
        return initial_capital + realized_pnl

    def available_cash():
        invested = sum(s['investment'] for s in slots if s is not None)
        return base_capital() - invested

    for date in all_dates:
        new_entries          = signals_by_date.get(date, [])
        eviction_done_today  = False

        # ── Natural exits on this date ────────────────────────────────────────
        for idx in range(n_slots):
            if slots[idx] is None:
                continue
            basket_id = slots[idx]['basket_id']
            sig_df    = results[basket_id]['signals']
            if date in sig_df.index and sig_df.loc[date, 'sell_signal']:
                trade, delta = _close_slot(slots[idx], date, 'or_gate_exit', results)
                if trade:
                    realized_pnl += delta
                    portfolio_trades.append(trade)
                    equity_points.append((date, realized_pnl))
                slots[idx] = None

        both_empty_at_entry_phase = all(s is None for s in slots)

        free_slots_at_phase_start = [i for i in range(n_slots) if slots[i] is None]

        if len(free_slots_at_phase_start) >= 2 and len(new_entries) > len(free_slots_at_phase_start):
            keep_ids = set(_select_least_correlated_subset(
                [bid for bid, _ in new_entries], len(free_slots_at_phase_start), results, date
            ))
            new_entries = [(bid, et) for (bid, et) in new_entries if bid in keep_ids]

        # ── New entries for this date ────────────────────────────────────────
        for (new_bid, new_entry_type) in new_entries:
            active_ids = {s['basket_id'] for s in slots if s is not None}
            if new_bid in active_ids:
                continue

            free_slots = [i for i in range(n_slots) if slots[i] is None]

            if free_slots:
                capital_to_deploy = (
                    base_capital() / n_slots if both_empty_at_entry_phase else available_cash()
                )
                state = _open_slot(new_bid, date, new_entry_type, capital_to_deploy, results)
                if state:
                    target = free_slots[0]
                    slots[target] = state
                    for peer_idx in range(n_slots):
                        if peer_idx != target and slots[peer_idx] is not None:
                            _reset_returns_reference(slots[peer_idx], date)

            else:
                if eviction_done_today:
                    continue

                held_ids  = [slots[i]['basket_id'] for i in range(n_slots)]
                worst_idx = _eviction_target(held_ids, new_bid, results, date)

                if worst_idx is None:
                    # incoming basket is the most redundant member of the
                    # {held..., new} group -> no eviction, skip this entry
                    continue

                trade, delta = _close_slot(slots[worst_idx], date, 'evicted_new_entry', results)
                if trade is None:
                    continue

                realized_pnl += delta
                slots[worst_idx] = None
                portfolio_trades.append(trade)
                equity_points.append((date, realized_pnl))
                eviction_done_today = True

                state = _open_slot(new_bid, date, new_entry_type, available_cash(), results)
                if state:
                    slots[worst_idx] = state
                    for peer_idx in range(n_slots):
                        if peer_idx != worst_idx and slots[peer_idx] is not None:
                            _reset_returns_reference(slots[peer_idx], date)

    # ── MTM close of any still-open slots ───────────────────────────────────
    if all_dates:
        last = all_dates[-1]
        for idx in range(n_slots):
            if slots[idx] is not None:
                trade, _ = _close_slot(slots[idx], last, 'mtm', results)
                if trade:
                    portfolio_trades.append(trade)

    equity = pd.Series(
        [p[1] for p in equity_points],
        index=[p[0] for p in equity_points],
        dtype=float,
    ).sort_index()

    return portfolio_trades, equity


# ══════════════════════════════════════════════════════════════════════════════
# METRICS  (identical contract to strategy1.py — equity must be ABSOLUTE capital)
# ══════════════════════════════════════════════════════════════════════════════

def _daily_equity_curve(equity, initial_capital, start, end):
    """
    Forward-fill the sparse, trade-event-indexed REALIZED-PnL equity series
    to a business-day daily series (flat between trade-close dates, since
    PnL is only realized when a slot closes — this is NOT a mark-to-market
    curve). Mirrors the report script's _daily_equity() so Sharpe/Sortino
    here match what gets shown in generated reports.
    """
    s = equity.groupby(level=0).last()          # same-day closes -> keep last
    idx = pd.bdate_range(start=start, end=end)
    return s.reindex(idx).ffill().fillna(float(initial_capital))


def compute_metrics(port_trades, equity, initial_capital=POSITION_SIZE):
    """
    Returns dict:
      total_trades, closed_trades, win_rate_pct,
      total_return_pct, cagr_pct, max_drawdown_pct,
      avg_hold_days, avg_pnl_pct, total_pnl,
      first_trade, last_trade, years,
      sharpe_ratio, sortino_ratio

    Sharpe / Sortino methodology (stdev definition):
      1. equity (absolute capital, sparse @ trade-close dates) is forward-
         filled to a daily business-day series via _daily_equity_curve().
      2. daily_rets = that daily series' pct_change().
      3. excess = daily_rets - RF_DAILY   (RF_DAILY = (1+RF_ANNUAL)**(1/252)-1,
         RF_ANNUAL = 6% p.a.)
      4. Sharpe  stdev = excess.std()                  — SAMPLE std (ddof=1,
         pandas default) of ALL daily excess returns.
      5. Sortino stdev = excess[excess < 0].std()       — SAMPLE std (ddof=1)
         of the DOWNSIDE-ONLY subset of daily excess returns.
      6. Both ratios annualised via * sqrt(252) on the mean/std ratio.
    """
    _nan = {k: np.nan for k in [
        'total_trades', 'closed_trades', 'win_rate_pct', 'total_return_pct',
        'cagr_pct', 'max_drawdown_pct', 'avg_hold_days', 'avg_pnl_pct',
        'total_pnl', 'first_trade', 'last_trade', 'years',
        'sharpe_ratio', 'sortino_ratio',
    ]}
    if not port_trades:
        return _nan

    df     = pd.DataFrame(port_trades)
    df['buy_date']  = pd.to_datetime(df['buy_date'])
    df['sell_date'] = pd.to_datetime(df['sell_date'])
    closed = df[df['status'] == 'closed']

    total_trades  = len(df)
    closed_trades = len(closed)
    win_rate      = (closed['pnl'] > 0).mean() * 100 if closed_trades else np.nan
    avg_pnl_pct   = closed['pnl_pct'].mean()          if closed_trades else np.nan
    total_pnl     = closed['pnl'].sum()                if closed_trades else 0.0

    first_date = df['buy_date'].min()
    last_date  = df['sell_date'].max()
    years      = max((last_date - first_date).days / 365.25, 1e-6)

    sharpe_ratio  = np.nan
    sortino_ratio = np.nan

    if not equity.empty:
        final_cap     = equity.iloc[-1]
        total_ret_pct = (final_cap / initial_capital - 1) * 100
        cagr_pct      = ((final_cap / initial_capital) ** (1 / years) - 1) * 100
        running_max   = equity.cummax()
        drawdowns     = (equity - running_max) / running_max * 100
        max_dd_pct    = drawdowns.min()

        # ── Sharpe / Sortino (see docstring for exact stdev definitions) ───
        daily      = _daily_equity_curve(equity, initial_capital, first_date, last_date)
        daily_rets = daily.pct_change().dropna()
        excess     = daily_rets - RF_DAILY
        ex_mean    = float(excess.mean())
        ex_std     = float(excess.std())            # sample std, ALL excess returns

        if ex_std > 0:
            sharpe_ratio = round(ex_mean / ex_std * math.sqrt(252), 3)

        downside = excess[excess < 0]
        if len(downside) > 1:
            dn_std = float(downside.std())           # sample std, DOWNSIDE-ONLY excess returns
            if dn_std > 0:
                sortino_ratio = round(ex_mean / dn_std * math.sqrt(252), 3)
    else:
        final_cap     = initial_capital + total_pnl
        total_ret_pct = total_pnl / initial_capital * 100
        cagr_pct      = ((final_cap / initial_capital) ** (1 / years) - 1) * 100
        max_dd_pct    = np.nan

    avg_hold = df['hold_days'].mean()

    return {
        'total_trades':     total_trades,
        'closed_trades':    closed_trades,
        'win_rate_pct':     round(win_rate, 1),
        'total_return_pct': round(total_ret_pct, 2),
        'cagr_pct':         round(cagr_pct, 2),
        'max_drawdown_pct': round(max_dd_pct, 2) if not np.isnan(max_dd_pct) else np.nan,
        'avg_hold_days':    round(avg_hold, 1),
        'avg_pnl_pct':      round(avg_pnl_pct, 2),
        'total_pnl':        round(total_pnl, 2),
        'first_trade':      first_date.date(),
        'last_trade':       last_date.date(),
        'years':            round(years, 2),
        'sharpe_ratio':      sharpe_ratio,
        'sortino_ratio':     sortino_ratio,
    }


def run_backtest_for_config(config, initial_capital=POSITION_SIZE, n_slots=N_SLOTS):
    """
    Full backtest for one basket-size config.
    Returns (results, bdata, port_trades, equity[ABSOLUTE capital], metrics).
    """
    members = config['members']
    label   = config['label']
    results, bdata, skipped = {}, {}, []

    print(f'\n{"─"*64}')
    print(f'  {label}  ({members["basket_id"].nunique()} baskets)')
    print(f'{"─"*64}')

    for basket_id, group in members.groupby('basket_id'):
        group     = group.sort_values('stock_position')
        tickers   = group['ticker'].tolist()
        symbols   = group['symbol'].tolist()
        sectors   = group['sector'].tolist()
        companies = group['company_name'].tolist()
        try:
            df           = build_equal_weight_basket_ohlc(tickers)
            close        = df['Close']
            opens        = df['Open']
            bands        = rolling_regression_bands(close)
            ema_sig_raw  = ema_crossover_signals(close, EMA_FAST, EMA_SLOW)
            trades       = run_or_gate_strategy(close, opens, bands, ema_sig_raw)
            combined_sig = or_gate_combined_signals(close, bands, ema_sig_raw)

            bdata[basket_id]   = df
            results[basket_id] = {
                'basket_id':       basket_id,
                'symbols':         symbols,
                'tickers':         tickers,
                'sectors':         sectors,
                'companies':       companies,
                'bands':           bands,
                'signals':         combined_sig,
                'component_close': df.attrs['component_close'],
                'component_open':  df.attrs['component_open'],
                'or_gate':         {'trades': trades},
            }
            reg_t = sum(1 for t in trades if t['entry_type'] == 'regression')
            ema_t = sum(1 for t in trades if t['entry_type'] == 'ema_crossover')
            print(f'  B{basket_id:02d} [{", ".join(symbols)}]  '
                  f'{len(trades)} trades (reg={reg_t} ema={ema_t})')
        except Exception as e:
            skipped.append(basket_id)
            print(f'  B{basket_id:02d} SKIP – {e}')

    print(f'  → Completed {len(results)}  Skipped {len(skipped)}')

    port_trades, equity_pnl = simulate_dual_basket_portfolio(
        results, 'or_gate', initial_capital, n_slots
    )
    equity_abs = (equity_pnl + initial_capital) if not equity_pnl.empty else equity_pnl

    metrics = compute_metrics(port_trades, equity_abs, initial_capital)
    metrics.update({
        'basket_size': config['basket_size'],
        'label':       label,
        'n_baskets':   len(results),
    })

    print(f'  → {metrics["total_trades"]} trades | '
          f'Return {metrics["total_return_pct"]}% | '
          f'CAGR {metrics["cagr_pct"]}% | '
          f'MaxDD {metrics["max_drawdown_pct"]}% | '
          f'WinRate {metrics["win_rate_pct"]}%')

    return results, bdata, port_trades, equity_abs, metrics


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE + PLOTS  (same shapes as strategy1.py / strategy2a.py)
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_metrics):
    cols = [
        'basket_size', 'n_baskets',
        'total_trades', 'closed_trades', 'win_rate_pct',
        'total_return_pct', 'cagr_pct', 'max_drawdown_pct',
        'avg_hold_days', 'avg_pnl_pct', 'total_pnl',
        'years',
    ]
    rename = {
        'basket_size':      'Size',
        'n_baskets':        'Baskets',
        'total_trades':     'Trades',
        'closed_trades':    'Closed',
        'win_rate_pct':     'WinRate%',
        'total_return_pct': 'Return%',
        'cagr_pct':         'CAGR%',
        'max_drawdown_pct': 'MaxDD%',
        'avg_hold_days':    'AvgHold',
        'avg_pnl_pct':      'AvgPnL%',
        'total_pnl':        'TotalPnL',
        'years':            'Years',
    }
    df = (
        pd.DataFrame(all_metrics)[cols]
        .rename(columns=rename)
        .set_index('Size')
        .sort_index()
    )
    print(f'\n{"═"*84}')
    print(f'COMPARATIVE RESULTS — EMA{EMA_FAST}/{EMA_SLOW} + RegSlope≥{MIN_SLOPE_PCT}%/day | '
          f'Dual-Basket ({N_SLOTS} slots) | EOD signal → Next-Open exec')
    print(f'{"═"*84}')
    print(df.to_string())
    print(f'{"═"*84}\n')
    return df


def plot_comparative_equity(all_equities, all_metrics, initial_capital=POSITION_SIZE):
    """Overlay PnL equity curves for all basket-size configs."""
    fig = go.Figure()
    sorted_items = sorted(
        all_equities.items(),
        key=lambda kv: next((m['basket_size'] for m in all_metrics if m['label'] == kv[0]), 0)
    )
    for idx, (label, equity) in enumerate(sorted_items):
        if equity.empty:
            continue
        m     = next((m for m in all_metrics if m['label'] == label), {})
        color = SIZE_COLORS[idx % len(SIZE_COLORS)]
        name  = (
            f"{label}  "
            f"CAGR={m.get('cagr_pct','?')}%  "
            f"DD={m.get('max_drawdown_pct','?')}%  "
            f"WR={m.get('win_rate_pct','?')}%"
        )
        fig.add_trace(go.Scatter(
            x=(equity - initial_capital).index,
            y=(equity - initial_capital).values,
            mode='lines', line=dict(color=color, width=1.6), name=name,
        ))

    fig.add_hline(y=0, line_dash='dash', line_color=TEXT, opacity=0.4)
    fig.update_layout(
        **LAYOUT,
        title=f'EMA{EMA_FAST}/{EMA_SLOW} + RegSlope Dual-Basket — Equity Curves by Basket Size',
        height=560, hovermode='x unified',
    )
    fig.update_yaxes(title_text='Cumulative Realized PnL (Rs.)')
    fig.show()


def plot_metrics_bar(all_metrics):
    """2×2 bar grid: CAGR, Max Drawdown, Win Rate, Avg Hold Days."""
    df     = pd.DataFrame(all_metrics).sort_values('basket_size')
    labels = [f"{r['basket_size']}-stock" for _, r in df.iterrows()]

    panels = [
        ('cagr_pct',         'CAGR %',         ACCENT),
        ('max_drawdown_pct', 'Max Drawdown %',  RED),
        ('win_rate_pct',     'Win Rate %',      GREEN),
        ('avg_hold_days',    'Avg Hold Days',   ORANGE),
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[p[1] for p in panels],
        vertical_spacing=0.20,
        horizontal_spacing=0.10,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (col_key, col_title, color), (row, col) in zip(panels, positions):
        fig.add_trace(go.Bar(
            x=labels, y=df[col_key].tolist(),
            marker_color=color, name=col_title, showlegend=False,
        ), row=row, col=col)

    fig.update_layout(
        **LAYOUT,
        title=f'EMA{EMA_FAST}/{EMA_SLOW} + RegSlope — Key Metrics by Basket Size',
        height=620,
    )
    for ax in fig.layout:
        if ax.startswith('xaxis'):
            fig.layout[ax].tickangle = -30
    fig.show()


def plot_scatter_cagr_vs_dd(all_metrics):
    """Risk/return scatter: CAGR% (y) vs Max Drawdown% (x), bubble = win rate."""
    df = pd.DataFrame(all_metrics).sort_values('basket_size')
    fig = go.Figure()
    for idx, row in df.iterrows():
        color = SIZE_COLORS[idx % len(SIZE_COLORS)]
        fig.add_trace(go.Scatter(
            x=[row['max_drawdown_pct']],
            y=[row['cagr_pct']],
            mode='markers+text',
            marker=dict(
                size=max(row['win_rate_pct'] / 2, 8),
                color=color,
                opacity=0.85,
                line=dict(color=TEXT, width=1),
            ),
            text=[f"{row['basket_size']}-stock"],
            textposition='top center',
            name=f"{row['basket_size']}-stock  WR={row['win_rate_pct']}%",
        ))
    fig.update_layout(
        **LAYOUT,
        title='Risk / Return by Basket Size  (bubble size ∝ win rate)',
        height=500,
        xaxis_title='Max Drawdown %  (lower = better)',
        yaxis_title='CAGR %  (higher = better)',
    )
    fig.show()


# ══════════════════════════════════════════════════════════════════════════════
# CANDLESTICK (single basket verification — per-basket OR-gate trades)
# ══════════════════════════════════════════════════════════════════════════════

def plot_basket_candles_dual(basket_id, ohlc_df, bands, signals, trades, results, last_n_bars=300):
    df = ohlc_df.copy()
    if last_n_bars:
        df      = df.iloc[-last_n_bars:]
        bands   = bands.loc[df.index]
        signals = signals.loc[df.index]

    syms      = results.get(basket_id, {}).get('symbols', [])
    title_str = f"Basket {basket_id} | {', '.join(syms)} | OR Gate Dual-Basket (EMA {EMA_FAST}/{EMA_SLOW})"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04,
        subplot_titles=(title_str, 'Cumulative Trade PnL'),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        name='OHLC',
    ), row=1, col=1)

    for name_, key, color, dash in [
        ('Reg Trend', 'trend_line', PURPLE, 'dash'),
        ('-2σ',       'lower2',     RED,    'dot'),
        ('+2σ',       'upper2',     GREEN,  'dot'),
    ]:
        fig.add_trace(go.Scatter(x=bands.index, y=bands[key],
            mode='lines', line=dict(color=color, width=0.8, dash=dash), name=name_), row=1, col=1)

    fig.add_trace(go.Scatter(x=signals.index, y=signals['EMA_fast'],
        mode='lines', line=dict(color=ORANGE, width=1.0), name=f'EMA{EMA_FAST}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=signals.index, y=signals['EMA_slow'],
        mode='lines', line=dict(color=YELLOW, width=1.2), name=f'EMA{EMA_SLOW}'), row=1, col=1)

    if trades:
        tdf = pd.DataFrame(trades)
        tdf['buy_date']  = pd.to_datetime(tdf['buy_date'])
        tdf['sell_date'] = pd.to_datetime(tdf['sell_date'])
        om = df['Open']
        def gop(d):
            if d in om.index: return om[d]
            av = om.index[om.index >= d]
            return om[av[0]] if len(av) else np.nan
        tdf['ep'] = tdf['buy_date'].apply(gop)
        tdf['xp'] = tdf['sell_date'].apply(gop)

        for etype, color in [('regression', ACCENT), ('ema_crossover', PURPLE)]:
            s = tdf[tdf['entry_type'] == etype]
            if not s.empty:
                fig.add_trace(go.Scatter(x=s['buy_date'], y=s['ep'],
                    mode='markers', marker=dict(symbol='triangle-up', color=color, size=10),
                    name=f'Buy:{etype}'), row=1, col=1)

        for xtype, color in [('regression_2sd', ACCENT), ('ema_death_cross', RED)]:
            s = tdf[tdf['exit_type'] == xtype]
            if not s.empty:
                fig.add_trace(go.Scatter(x=s['sell_date'], y=s['xp'],
                    mode='markers', marker=dict(symbol='triangle-down', color=color, size=10),
                    name=f'Exit:{xtype}'), row=1, col=1)

        vis = tdf[tdf['buy_date'].isin(df.index) | tdf['sell_date'].isin(df.index)].copy()
        vis['cum'] = (vis['sell_price'] - vis['buy_price']).cumsum()
        fig.add_trace(go.Scatter(x=vis['sell_date'], y=vis['cum'],
            mode='lines+markers', line=dict(color=ORANGE, width=1.0), name='Cum PnL'), row=2, col=1)
        fig.add_hline(y=0, line_dash='dash', line_color=TEXT, opacity=0.4, row=2, col=1)

    fig.update_layout(**LAYOUT, height=820, hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text='Basket Price (Rs.)', row=1, col=1)
    fig.update_yaxes(title_text='Cum PnL (Rs.)',      row=2, col=1)
    fig.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(
    csv_paths=None,
    search_dirs=None,
    initial_capital=POSITION_SIZE,
    n_slots=N_SLOTS,
    plot_candles_per_config=3,
):
    print('Discovering basket CSVs...')
    configs = load_basket_configs(csv_paths=csv_paths, search_dirs=search_dirs)
    print(f'→ {len(configs)} config(s) found\n')

    all_metrics, all_equities, all_results = [], {}, {}

    for config in configs:
        results, bdata, port_trades, equity, metrics = run_backtest_for_config(
            config, initial_capital, n_slots
        )
        all_metrics.append(metrics)
        all_equities[config['label']] = equity
        all_results[config['label']]  = (results, bdata, port_trades)

        if port_trades:
            safe = config['label'].replace(' ', '_').replace('(', '').replace(')', '')
            out  = Path(f'trades_s1db_{safe}.csv')
            pd.DataFrame(port_trades).drop(
                columns=['quantities', 'component_open', 'component_close'], errors='ignore'
            ).to_csv(out, index=False)
            print(f'  Trade log → {out}')

    cmp_df = print_comparison_table(all_metrics)
    cmp_df.to_csv('comparison_results_s1_dualbasket.csv')
    print('Comparison table → comparison_results_s1_dualbasket.csv\n')

    if len(all_equities) >= 2:
        plot_comparative_equity(all_equities, all_metrics, initial_capital)
        plot_metrics_bar(all_metrics)
        plot_scatter_cagr_vs_dd(all_metrics)
    elif all_equities:
        label, equity = next(iter(all_equities.items()))
        if not equity.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity.index, y=(equity - initial_capital).values,
                mode='lines', line=dict(color=ACCENT, width=1.4)))
            fig.add_hline(y=0, line_dash='dash', line_color=TEXT, opacity=0.4)
            fig.update_layout(**LAYOUT,
                title=f'OR Gate Dual-Basket — {label}',
                height=480, hovermode='x unified')
            fig.update_yaxes(title_text='PnL (Rs.)')
            fig.show()

    if plot_candles_per_config > 0:
        for label, (results, bdata, _) in all_results.items():
            print(f'\nCandle charts → {label}')
            for bid in list(results.keys())[:plot_candles_per_config]:
                res = results[bid]
                plot_basket_candles_dual(
                    basket_id   = bid,
                    ohlc_df     = bdata[bid],
                    bands       = res['bands'],
                    signals     = res['signals'],
                    trades      = res['or_gate']['trades'],
                    results     = results,
                    last_n_bars = 400,
                )


if __name__ == '__main__':
    main(
        csv_paths=['data/baskets_nifty200_all_sizes.csv'],
        plot_candles_per_config=3,
    )