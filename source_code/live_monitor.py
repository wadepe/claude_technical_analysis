"""
live_monitor.py

Real-time rising-wedge monitor for SPY (or any ticker) using 1-minute bars.

Data source: yfinance (free, no credentials required, extended-hours support)
  Why not robin_stocks: requires Robinhood account login + 2-FA flow.
  To switch: see _fetch_bar_robinhood() stub at the bottom of this file.

Behaviour
---------
  - Polls the ticker once per minute while the market is open
    (pre-market 4:00 AM ET through after-hours close 8:00 PM ET, weekdays)
  - Rejects phantom prints: bars whose OHLC deviates more than
    --spike-threshold percent (default 4%) from the recent median close
    are skipped entirely (not written, not scored)
  - Appends each completed bar to  spy_data_1min.csv
  - Maintains rolling windows of 50 and 250 bars (filled from CSV on restart)
  - Normalises each window and scores it with the matching CNN model
  - Appends scores to  rising_wedge.csv

Output CSV schemas
------------------
  spy_data_1min.csv:
    timestamp (ISO 8601), open, high, low, close, volume

  rising_wedge.csv:
    timestamp, score_50bar, signal_50bar, score_250bar, signal_250bar,
    bars_50, bars_250,
    then per window (suffix _50bar / _250bar), zeros unless that window's
    score cleared the threshold this bar:
      proj_move_usd   projected |move| in dollars (linear model fit on the
                      2008-2021 SPY slope study; see PROJ_* constants)
      slope_upper     fitted upper trendline slope, $ per bar
      slope_lower     fitted lower trendline slope, $ per bar
      apex_min        minutes until the fitted trendlines cross; negative =
                      crossing already behind (pinched wedge); 0 = parallel
                      or diverging lines (no crossing)
      apex_price      price where they cross, $ (0 if no crossing)
      mid_travel      wedge midline tilt across the window as a fraction of
                      the window's price range (+ rising, - falling)
    (signal = 1 when score >= threshold AND the apex gate passes: converging
     lines with apex <= APEX_GATE_MAX_MIN ahead or already crossed. A bar with
     score >= threshold but signal = 0 was suppressed by the geometry gate.
     bars_* = current window depth)

Usage
-----
  # Live monitoring (default ticker=SPY, threshold=0.5)
  python live_monitor.py

  # Different ticker / stricter threshold
  python live_monitor.py --ticker QQQ --threshold 0.65

  # Replay an existing spy_data_1min.csv without waiting for the clock
  python live_monitor.py --replay

  # Dry-run: print what would be written without touching the CSVs
  python live_monitor.py --dry-run

Requirements
------------
  pip install numpy pandas yfinance
  pip install pandas_market_calendars   # NYSE holiday / half-day calendar

  On Ubuntu the system tz database is present, so zoneinfo resolves natively
  (no separate tzdata install needed). If pandas_market_calendars is absent the
  monitor still runs but only skips weekends, not holidays.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SPY_CSV      = PROJECT_ROOT / 'spy_data_1min.csv'
WEDGE_CSV    = PROJECT_ROOT / 'rising_wedge.csv'
CRASH_LOG    = PROJECT_ROOT / 'crash.log'

# ── Market hours (US/Eastern) ─────────────────────────────────────────────────
# Covers pre-market (4 AM), regular session (9:30 AM-4:00 PM), after-hours (to 8 PM).
# On a normal day the after-hours cutoff is the regular close (4 PM) +
# AFTERHOURS_BUFFER_H, which reproduces the historical 8 PM cutoff. On an
# early-close half-day (1 PM close) it shrinks to 5 PM automatically.
MARKET_OPEN_H       = 4    # 4:00 AM ET  pre-market open
MARKET_CLOSE_H      = 20   # 8:00 PM ET  fallback close (calendar unavailable)
AFTERHOURS_BUFFER_H = 4    # hours of after-hours polling past the regular close

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('live_monitor')

# Also mirror ERROR-level messages to crash.log on disk so crashes survive
# across systemd restarts and are visible in the GitHub repo.
_crash_file_handler = logging.FileHandler(CRASH_LOG, mode='a', encoding='utf-8')
_crash_file_handler.setLevel(logging.ERROR)
_crash_file_handler.setFormatter(logging.Formatter(
    '%(asctime)s  %(levelname)-8s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
))
log.addHandler(_crash_file_handler)

FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']

# ── Projected-move model (slope_move_study / nondud_projection, 2026-07-17) ──
# Linear fit of |4hr move|% vs mid-trendline travel, on 1,130 resolved
# (non-dud) 250-bar v2 detections over SPY 2008-2021:
#     projected |move|% = 1.034 - 0.146 * travel_mid
# where travel_mid = mean of upper/lower trendline travel across the window
# (fraction of window price range). Note R^2 is small (~0.007): this is a
# size *bias* given resolution, not a per-event forecast.
PROJ_INTERCEPT_PCT = 1.034
PROJ_TRAVEL_COEF   = -0.146

# ── Apex gate ─────────────────────────────────────────────────────────────────
# A bar only SIGNALS when, in addition to clearing the score threshold, its
# fitted trendlines converge with an apex no more than this many minutes ahead
# (already-crossed apexes pass: a fully pinched wedge is the most mature form).
# Near-parallel fits (apex further out) and diverging fits are suppressed —
# those channel-like shapes are left for dedicated channel models. Backtest
# (SPY 2008-2021): keeps 80% of signals at ~unchanged resolution rate, drops
# the 20% whose geometry is channel-like rather than wedge-like.
APEX_GATE_MAX_MIN = 360


# =============================================================================
# Timezone helper
# =============================================================================

_ET_ZONE = None   # cached America/New_York tzinfo


def _et_zone():
    """Return a cached America/New_York tzinfo (zoneinfo, or pytz fallback)."""
    global _ET_ZONE
    if _ET_ZONE is None:
        try:
            from zoneinfo import ZoneInfo
            _ET_ZONE = ZoneInfo('America/New_York')
        except Exception:
            # zoneinfo missing (Python < 3.9) OR no system tz database — the
            # latter is the norm on Windows, where ZoneInfo raises
            # ZoneInfoNotFoundError (a KeyError, not ImportError). Fall back to
            # pytz, which bundles its own copy of the tz data.
            import pytz
            _ET_ZONE = pytz.timezone('America/New_York')
    return _ET_ZONE


# =============================================================================
# Crash logging
# =============================================================================

def _write_crash_entry(exc: BaseException, context: str = '') -> None:
    """
    Append a formatted crash entry to crash.log.

    Called both from the inner exception handler (recoverable errors that are
    retried) and from the top-level sys.excepthook (fatal unhandled exceptions).
    Each entry is self-contained so the file remains readable after many crashes.
    """
    import traceback

    now_et  = datetime.now(_et_zone()).strftime('%Y-%m-%d %H:%M:%S ET')
    tb_str  = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    divider = '=' * 72

    entry = (
        f'\n{divider}\n'
        f'CRASH @ {now_et}\n'
        + (f'Context: {context}\n' if context else '')
        + f'{divider}\n'
        f'{tb_str}'
        f'{divider}\n'
    )

    try:
        with open(CRASH_LOG, 'a', encoding='utf-8') as fh:
            fh.write(entry)
    except Exception:
        pass   # never let crash-logging itself crash the process


# =============================================================================
# Model loading (both window sizes in one process)
# =============================================================================

def _load_model(window: int):
    """Build + load weights for a given window size. Returns (model, n_bars)."""
    os.environ['WEDGE_TOTAL_BARS'] = str(window)
    for mod in list(sys.modules.keys()):
        if mod == 'cnn_model':
            del sys.modules[mod]
    from cnn_model import build_model, N_BARS

    candidates = [
        PROJECT_ROOT / 'runs' / f'window_{window}bar' / 'models' / 'cnn_best.weights.h5',
        PROJECT_ROOT / 'models' / 'cnn_best.weights.h5',
    ]
    wp = next((p for p in candidates if p.exists()), None)
    if wp is None:
        log.warning(f'No weights found for window={window} — skipping this model.')
        return None, window

    model = build_model(print_summary=False)
    model.load_weights(str(wp))

    # Warm-up pass so first real inference isn't slow
    dummy = np.zeros((1, N_BARS, 5), dtype=np.float32)
    model.predict(dummy, verbose=0)

    log.info(f'Loaded {window}-bar model  [{wp.name}]')
    return model, N_BARS


# =============================================================================
# Per-window normalisation  (must match training pipeline exactly)
# =============================================================================

def _normalise(window: np.ndarray) -> np.ndarray:
    """
    Normalise a (n_bars, 5) OHLCV window to [0, 1].
    Prices share one min/max; volume normalised independently.
    """
    prices  = window[:, :4]
    p_min   = prices.min()
    p_range = prices.max() - p_min
    if p_range < 1e-10:
        p_range = 1.0
    prices_n = (prices - p_min) / p_range

    vol     = window[:, 4]
    vol_max = vol.max()
    if vol_max < 1e-10:
        vol_max = 1.0
    vol_n = (vol / vol_max).reshape(-1, 1)

    return np.concatenate([prices_n, vol_n], axis=1).astype(np.float32)


# =============================================================================
# Bad-tick / spike filter
# =============================================================================

class SpikeFilter:
    """
    Rejects bars containing phantom prints (e.g. the 2026-07-07 17:07 bar where
    SPY briefly 'traded' at 696.82, ~7% below the market). A bad bar poisons
    the rolling windows for the next 50/250 bars by stretching the
    normalisation range, muting the model's inputs.

    A bar is rejected when any OHLC field deviates more than `threshold`
    (fraction, e.g. 0.04 = 4%) from the median close of the last `window`
    accepted bars. To avoid rejecting a genuine large move forever, after
    `max_consecutive` rejections in a row the next bar is accepted and the
    reference re-anchors to it.
    """

    def __init__(self, threshold: float = 0.04, window: int = 10,
                 max_consecutive: int = 3):
        self.threshold       = threshold
        self.max_consecutive = max_consecutive
        self._closes         = deque(maxlen=window)
        self._rejects_in_row = 0

    def seed(self, closes) -> None:
        """Pre-fill the reference from historical closes (CSV tail on restart)."""
        for c in closes:
            self._closes.append(float(c))

    def check(self, bar: dict) -> bool:
        """Return True to accept the bar; False (and log) to reject it."""
        if not self._closes:
            self._closes.append(float(bar['close']))
            return True

        ref   = float(np.median(self._closes))
        worst = max(abs(float(bar[f]) / ref - 1.0)
                    for f in ('open', 'high', 'low', 'close'))

        if worst > self.threshold:
            self._rejects_in_row += 1
            if self._rejects_in_row <= self.max_consecutive:
                log.warning(
                    f'SPIKE FILTER: rejected bar {bar.get("timestamp", "?")} '
                    f'(deviates {worst*100:.2f}% from median {ref:.2f}, '
                    f'threshold {self.threshold*100:.1f}%, '
                    f'{self._rejects_in_row}/{self.max_consecutive} in a row)'
                )
                return False
            log.warning(
                f'SPIKE FILTER: accepting bar {bar.get("timestamp", "?")} after '
                f'{self._rejects_in_row - 1} consecutive rejections — treating '
                f'the move as genuine and re-anchoring.'
            )
            # Full re-anchor: drop the stale reference entirely, otherwise the
            # old median keeps rejecting bars at the new price level.
            self._closes.clear()

        self._rejects_in_row = 0
        self._closes.append(float(bar['close']))
        return True


# =============================================================================
# Market data fetching  (yfinance)
# =============================================================================

def _fetch_bar_yfinance(ticker: str) -> Optional[dict]:
    """
    Fetch the most recently *completed* 1-minute bar via yfinance.

    yfinance may include a still-forming bar at the end; we take the last bar
    whose timestamp is at least 60 seconds in the past (UTC).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit('yfinance not installed. Run: pip install yfinance')

    try:
        hist = yf.Ticker(ticker).history(
            period='1d', interval='1m', prepost=True
        )
    except Exception as exc:
        log.warning(f'yfinance fetch error: {exc}')
        return None

    if hist.empty:
        return None

    # Filter to bars that started at least 60 s ago
    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(seconds=60)
    hist.index = hist.index.tz_convert('UTC')
    completed = hist[hist.index <= cutoff]

    if completed.empty:
        return None

    bar = completed.iloc[-1]
    return {
        'timestamp': bar.name.tz_convert('America/New_York')
                        .strftime('%Y-%m-%d %H:%M:%S'),
        'open':   float(bar['Open']),
        'high':   float(bar['High']),
        'low':    float(bar['Low']),
        'close':  float(bar['Close']),
        'volume': float(bar['Volume']),
    }


# --------------------------------------------------------------------------- #
# robin_stocks stub — uncomment and fill credentials to use instead           #
# --------------------------------------------------------------------------- #
# def _fetch_bar_robinhood(ticker: str) -> Optional[dict]:
#     import robin_stocks.robinhood as rh
#     rh.login(username='YOUR_EMAIL', password='YOUR_PASSWORD')
#     bars = rh.stocks.get_stock_historicals(
#         ticker, interval='minute', span='hour', bounds='extended'
#     )
#     if not bars:
#         return None
#     b = bars[-2]   # -1 may still be forming
#     return {
#         'timestamp': b['begins_at'],
#         'open':   float(b['open_price']),
#         'high':   float(b['high_price']),
#         'low':    float(b['low_price']),
#         'close':  float(b['close_price']),
#         'volume': float(b['volume']),
#     }
# --------------------------------------------------------------------------- #


# =============================================================================
# Market-hours check  (NYSE trading calendar via pandas_market_calendars)
# =============================================================================

# Sentinel: the calendar package is unavailable, so callers should fall back
# to the original weekday-only check. Distinct from None (= closed today).
_CAL_UNAVAILABLE = object()

_NYSE_CAL        = None    # lazily-created pandas_market_calendars calendar
_NYSE_CAL_FAILED = False   # True once import/creation has failed (warn once)
_SESSION_CACHE   = {}      # date -> (open_et, close_et) | None ; one query per day


def _get_nyse_calendar():
    """Lazily create and cache the NYSE calendar. Returns None if unavailable."""
    global _NYSE_CAL, _NYSE_CAL_FAILED
    if _NYSE_CAL is not None:
        return _NYSE_CAL
    if _NYSE_CAL_FAILED:
        return None
    try:
        import pandas_market_calendars as mcal
        _NYSE_CAL = mcal.get_calendar('NYSE')
    except Exception as exc:
        _NYSE_CAL_FAILED = True
        log.warning(
            f'pandas_market_calendars unavailable ({exc}); falling back to '
            f'weekday-only check — HOLIDAYS WILL NOT BE SKIPPED. '
            f'Install with: pip install pandas_market_calendars'
        )
    return _NYSE_CAL


def _nyse_session_today(now_et):
    """
    Return today's regular NYSE session as (open_et, close_et) datetimes,
    None if the market is fully closed today (weekend/holiday), or
    _CAL_UNAVAILABLE if the calendar package could not be loaded.

    Cached per calendar date so the calendar is queried at most once a day.
    """
    cal = _get_nyse_calendar()
    if cal is None:
        return _CAL_UNAVAILABLE

    today = now_et.date()
    if today not in _SESSION_CACHE:
        sched = cal.schedule(start_date=str(today), end_date=str(today))
        if sched.empty:
            _SESSION_CACHE[today] = None                       # closed today
        else:
            tz = now_et.tzinfo
            _SESSION_CACHE[today] = (
                sched.iloc[0]['market_open'].tz_convert(tz),
                sched.iloc[0]['market_close'].tz_convert(tz),
            )
    return _SESSION_CACHE[today]


def _is_market_open() -> bool:
    """
    Return True if we should be polling right now.

    Skips weekends and NYSE holidays via the trading calendar, and shortens
    the after-hours window on early-close half-days (the cutoff is the regular
    session close + AFTERHOURS_BUFFER_H). Falls back to the original weekday +
    fixed-hours check if pandas_market_calendars is not installed.
    """
    now_et  = datetime.now(_et_zone())
    session = _nyse_session_today(now_et)

    if session is _CAL_UNAVAILABLE:
        # Calendar package missing — original behaviour (weekends only).
        if now_et.weekday() >= 5:        # Saturday=5, Sunday=6
            return False
        return MARKET_OPEN_H <= now_et.hour < MARKET_CLOSE_H

    if session is None:
        return False                     # weekend or market holiday

    # Trading day: poll from pre-market open through the after-hours buffer.
    _open_et, close_et = session
    premarket_open = now_et.replace(
        hour=MARKET_OPEN_H, minute=0, second=0, microsecond=0
    )
    post_close = close_et + timedelta(hours=AFTERHOURS_BUFFER_H)
    return premarket_open <= now_et < post_close


# =============================================================================
# CSV helpers
# =============================================================================

def _init_spy_csv() -> None:
    if not SPY_CSV.exists():
        pd.DataFrame(columns=['timestamp'] + FEATURE_COLS).to_csv(
            SPY_CSV, index=False
        )
        log.info(f'Created {SPY_CSV}')


_STAT_KEYS = ['proj_move_usd', 'slope_upper', 'slope_lower',
              'apex_min', 'apex_price', 'mid_travel']

WEDGE_COLUMNS = [
    'timestamp',
    'score_50bar', 'signal_50bar',
    'score_250bar', 'signal_250bar',
    'bars_50', 'bars_250',
] + [f'{k}_50bar' for k in _STAT_KEYS] + [f'{k}_250bar' for k in _STAT_KEYS]


def _init_wedge_csv() -> None:
    if WEDGE_CSV.exists():
        # Schema changed in v2 (wedge-geometry stat columns). Appending wider
        # rows to a v1-header file would misalign the CSV — archive it and
        # start fresh instead.
        try:
            header = pd.read_csv(WEDGE_CSV, nrows=0).columns.tolist()
        except Exception:
            header = []
        if header != WEDGE_COLUMNS:
            archived = WEDGE_CSV.with_name('rising_wedge_v1.csv')
            WEDGE_CSV.rename(archived)
            log.info(f'Schema change: archived old scores file to {archived}')
    if not WEDGE_CSV.exists():
        pd.DataFrame(columns=WEDGE_COLUMNS).to_csv(WEDGE_CSV, index=False)
        log.info(f'Created {WEDGE_CSV}')


def _last_timestamp() -> Optional[str]:
    """Return the most recent timestamp in spy_data_1min.csv, or None."""
    if not SPY_CSV.exists():
        return None
    try:
        df = pd.read_csv(SPY_CSV, usecols=['timestamp'])
        return df['timestamp'].iloc[-1] if len(df) else None
    except Exception:
        return None


def _append_spy_row(bar: dict, dry_run: bool) -> None:
    row = pd.DataFrame([{
        'timestamp': bar['timestamp'],
        'open':      bar['open'],
        'high':      bar['high'],
        'low':       bar['low'],
        'close':     bar['close'],
        'volume':    bar['volume'],
    }])
    if not dry_run:
        row.to_csv(SPY_CSV, mode='a', header=False, index=False)


def _append_wedge_row(ts: str, scores: dict, threshold: float,
                      window_depths: dict, dry_run: bool,
                      stats: Optional[dict] = None) -> None:
    stats = stats or {}

    def sig(s, w):
        if s is None or np.isnan(s) or s < threshold:
            return 0
        # Apex gate: geometry must be wedge-like (see APEX_GATE_MAX_MIN).
        # Missing stats (defensive) fall back to score-only behaviour.
        w_stats = stats.get(w)
        return 1 if (w_stats is None or w_stats.get('gate_ok')) else 0

    s50   = scores.get(50)
    s250  = scores.get(250)
    entry = {
        'timestamp':   ts,
        'score_50bar':  round(s50,  6) if s50  is not None else '',
        'signal_50bar': sig(s50, 50),
        'score_250bar': round(s250, 6) if s250 is not None else '',
        'signal_250bar':sig(s250, 250),
        'bars_50':  window_depths.get(50,  0),
        'bars_250': window_depths.get(250, 0),
    }
    for w in (50, 250):
        w_stats = stats.get(w, _ZERO_STATS)
        for k in _STAT_KEYS:
            entry[f'{k}_{w}bar'] = w_stats[k]

    row = pd.DataFrame([entry], columns=WEDGE_COLUMNS)
    if not dry_run:
        row.to_csv(WEDGE_CSV, mode='a', header=False, index=False)


# =============================================================================
# Rolling window init from existing CSV
# =============================================================================

def _load_rolling_window(n_bars: int) -> deque:
    """Pre-fill a rolling window from the tail of spy_data_1min.csv."""
    dq = deque(maxlen=n_bars)
    if SPY_CSV.exists():
        try:
            df = pd.read_csv(SPY_CSV, usecols=FEATURE_COLS)
            for _, row in df.tail(n_bars).iterrows():
                dq.append(row[FEATURE_COLS].values.astype(np.float32))
            log.info(f'  Pre-filled {n_bars}-bar window with {len(dq)} rows from CSV')
        except Exception as exc:
            log.warning(f'  Could not pre-fill window: {exc}')
    return dq


# =============================================================================
# Wedge geometry stats for signalling bars
# =============================================================================

_ZERO_STATS = {'proj_move_usd': 0.0, 'slope_upper': 0.0, 'slope_lower': 0.0,
               'apex_min': 0, 'apex_price': 0.0, 'mid_travel': 0.0,
               'gate_ok': False}


def _wedge_stats(window_deque: deque, score: Optional[float],
                 threshold: float) -> dict:
    """
    Fit trendlines on the raw window and derive the reported stats for a bar
    whose score clears the threshold. Below-threshold (or unavailable) bars
    get all-zero stats, per the output contract.

      proj_move_usd  projected |move| in dollars (linear model, see constants)
      slope_upper /  envelope trendline slopes in $ per bar
      slope_lower
      apex_min       minutes until the fitted lines cross; NEGATIVE when the
                     crossing is already behind (fully pinched wedge);
                     0 when the lines are parallel/diverging (no crossing)
      apex_price     price at that crossing ($; 0 if no crossing)
      gate_ok        apex gate verdict (internal, not written to the CSV):
                     converging AND apex_min <= APEX_GATE_MAX_MIN ahead.
                     The signal_* flag requires score >= threshold AND gate_ok,
                     so a high-score bar with signal=0 was geometry-gated.
    """
    if score is None or score < threshold:
        return dict(_ZERO_STATS)

    from classify_wedge import fit_wedge_lines
    arr = np.array(window_deque, dtype=np.float32)      # raw (n_bars, 5)
    g   = fit_wedge_lines(arr)
    n   = arr.shape[0]

    close      = float(arr[-1, 3])
    travel_mid = (g['travel_upper'] + g['travel_lower']) / 2.0
    proj_pct   = max(PROJ_INTERCEPT_PCT + PROJ_TRAVEL_COEF * travel_mid, 0.0)

    # Apex: crossing of upper/lower lines in raw price space.
    apex_min, apex_price, gate_ok = 0, 0.0, False
    db = g['b_upper'] - g['b_lower']
    if db < -1e-12:                                     # converging lines
        x_cross = (g['a_lower'] - g['a_upper']) / db    # bars from window start
        ahead   = x_cross - (n - 1)                     # bars past current bar
        apex_min   = int(round(ahead))                  # 1 bar = 1 minute
        apex_price = float(g['a_upper'] + g['b_upper'] * x_cross)
        gate_ok    = ahead <= APEX_GATE_MAX_MIN         # near or already pinched

    return {
        'proj_move_usd': round(close * proj_pct / 100.0, 4),
        'slope_upper':   round(g['b_upper'], 6),        # $ per bar
        'slope_lower':   round(g['b_lower'], 6),
        'apex_min':      apex_min,
        'apex_price':    round(apex_price, 4),
        # Wedge midline tilt across the window, as a fraction of the window's
        # price range (+1 = climbed the full range; - = falling wedge). Logged
        # raw so slope-based interpretations (e.g. the observed "steep-rising
        # = quiet continuation" regime) can be evaluated on live data post-hoc.
        'mid_travel':    round(travel_mid, 4),
        'gate_ok':       gate_ok,
    }


# =============================================================================
# Core inference
# =============================================================================

def _score_window(model, window_deque: deque, n_bars: int) -> Optional[float]:
    """Score a rolling window. Returns None if window is not yet full."""
    if len(window_deque) < n_bars:
        return None
    arr  = np.array(window_deque, dtype=np.float32)   # (n_bars, 5)
    norm = _normalise(arr)
    x    = norm[np.newaxis, :, :]                      # (1, n_bars, 5)
    return float(model.predict(x, verbose=0).squeeze())


# =============================================================================
# Main loop
# =============================================================================

def run_live(ticker: str, threshold: float, dry_run: bool,
             spike_threshold: float = 0.04) -> None:
    """Main monitoring loop — runs until keyboard interrupt."""

    # Load models
    log.info('Loading models ...')
    models   = {}
    n_bars_map = {}
    for w in (50, 250):
        m, nb = _load_model(w)
        if m is not None:
            models[w]    = m
            n_bars_map[w] = nb

    if not models:
        raise SystemExit('No models loaded. Run the pipeline first.')

    # Init CSVs
    _init_spy_csv()
    _init_wedge_csv()

    # Pre-fill rolling windows from existing data
    log.info('Pre-filling rolling windows from existing CSV data ...')
    windows = {w: _load_rolling_window(nb) for w, nb in n_bars_map.items()}

    # Seed the spike filter's reference from the freshest pre-filled window
    # (row layout is [open, high, low, close, volume] — close is index 3)
    spike_filter = SpikeFilter(threshold=spike_threshold)
    largest = max(windows.values(), key=len, default=None)
    if largest:
        spike_filter.seed(row[3] for row in list(largest)[-10:])

    last_ts = _last_timestamp()
    log.info(f'Last CSV timestamp: {last_ts or "none (fresh start)"}')
    log.info(f'Monitoring {ticker}  |  threshold={threshold}  |  '
             f'extended hours from {MARKET_OPEN_H}:00 ET')

    # Surface calendar status now so the operator knows whether holidays are
    # being skipped, instead of finding out at the first market-hours check.
    if _get_nyse_calendar() is not None:
        log.info('NYSE trading calendar active — weekends, holidays, and '
                 'early-close half-days will be skipped.')
    # (a warning is already logged by _get_nyse_calendar if it is unavailable)

    log.info('Press Ctrl-C to stop.\n')

    while True:
        try:
            # ── Sleep until next minute boundary (+5 s buffer for bar to close) ──
            now     = datetime.now()
            wait    = 60 - now.second - now.microsecond / 1e6 + 5
            if wait > 60:
                wait -= 60
            time.sleep(max(wait, 1))

            if not _is_market_open():
                log.debug('Market closed — sleeping 5 min')
                time.sleep(300)
                continue

            # ── Fetch latest completed bar ────────────────────────────────────
            bar = _fetch_bar_yfinance(ticker)
            if bar is None:
                log.warning('No bar returned — skipping this minute')
                continue

            # Deduplicate: skip if we already logged this bar
            if bar['timestamp'] == last_ts:
                log.debug(f'Duplicate bar {bar["timestamp"]} — skipping')
                continue
            last_ts = bar['timestamp']

            # Reject phantom prints before they reach the CSV or the windows.
            # A skipped bar behaves exactly like a minute where yfinance
            # returned nothing.
            if not spike_filter.check(bar):
                continue

            # ── Append to price CSV ───────────────────────────────────────────
            _append_spy_row(bar, dry_run)

            # ── Update rolling windows ────────────────────────────────────────
            bar_arr = np.array(
                [bar[c] for c in FEATURE_COLS], dtype=np.float32
            )
            for dq in windows.values():
                dq.append(bar_arr)

            # ── Score each window ─────────────────────────────────────────────
            scores, stats = {}, {}
            for w, model in models.items():
                scores[w] = _score_window(model, windows[w], n_bars_map[w])
                stats[w]  = _wedge_stats(windows[w], scores[w], threshold)

            depths = {w: len(windows[w]) for w in windows}

            # ── Log + write ───────────────────────────────────────────────────
            s50  = scores.get(50)
            s250 = scores.get(250)
            sig50  = (s50  is not None and s50  >= threshold
                      and stats.get(50,  {}).get('gate_ok', False))
            sig250 = (s250 is not None and s250 >= threshold
                      and stats.get(250, {}).get('gate_ok', False))

            score_str = (
                f'50bar={s50:.4f}{"*" if sig50 else " "}' if s50 is not None
                else '50bar=n/a '
            ) + '  ' + (
                f'250bar={s250:.4f}{"*" if sig250 else " "}' if s250 is not None
                else '250bar=n/a '
            )

            wedge_note = ''
            if sig50 or sig250:
                st = stats[250] if sig250 else stats[50]
                wedge_note = (
                    f'  <<< WEDGE SIGNAL  proj_move=${st["proj_move_usd"]:.2f}'
                    + (f'  apex in {st["apex_min"]}m @ ${st["apex_price"]:.2f}'
                       if st['apex_min'] else '')
                    + ' >>>'
                )

            level = logging.WARNING if (sig50 or sig250) else logging.INFO
            log.log(level,
                f'{bar["timestamp"]}  {ticker}  '
                f'C={bar["close"]:.2f}  {score_str}' + wedge_note
            )

            _append_wedge_row(bar['timestamp'], scores, threshold, depths,
                              dry_run, stats=stats)

        except KeyboardInterrupt:
            log.info('Stopped by user.')
            break
        except Exception as exc:
            context = f'Last bar: {bar["timestamp"] if bar else "none"}'
            log.error(f'Unhandled error: {exc}  [{context}]', exc_info=True)
            _write_crash_entry(exc, context=context)
            time.sleep(10)


# =============================================================================
# Replay mode  (feed spy_data_1min.csv through the models without live polling)
# =============================================================================

def run_replay(ticker: str, threshold: float, dry_run: bool,
               spike_threshold: float = 0.04) -> None:
    """
    Replay spy_data_1min.csv through both models as fast as possible.
    Useful for testing scoring logic without waiting for live data.
    Writes results to rising_wedge.csv (overwriting it for a clean replay).
    Applies the same spike filter as live mode, so bad ticks already stored
    in the CSV (e.g. 2026-07-07 17:07) are skipped during re-scoring.
    """
    if not SPY_CSV.exists():
        raise SystemExit(f'{SPY_CSV} not found. Nothing to replay.')

    log.info(f'Replaying {SPY_CSV} ...')
    df = pd.read_csv(SPY_CSV)
    log.info(f'  {len(df):,} bars  ({df["timestamp"].iloc[0]} to {df["timestamp"].iloc[-1]})')

    # Load models
    models     = {}
    n_bars_map = {}
    for w in (50, 250):
        m, nb = _load_model(w)
        if m is not None:
            models[w]    = m
            n_bars_map[w] = nb

    if not models:
        raise SystemExit('No models loaded.')

    # Reset output CSV
    if not dry_run:
        pd.DataFrame(columns=[
            'timestamp',
            'score_50bar', 'signal_50bar',
            'score_250bar', 'signal_250bar',
            'bars_50', 'bars_250',
        ]).to_csv(WEDGE_CSV, index=False)

    windows = {w: deque(maxlen=nb) for w, nb in n_bars_map.items()}
    signals = 0
    spike_filter = SpikeFilter(threshold=spike_threshold)
    skipped = 0

    for _, row in df.iterrows():
        if not spike_filter.check(row):
            skipped += 1
            continue
        bar_arr = row[FEATURE_COLS].values.astype(np.float32)
        for dq in windows.values():
            dq.append(bar_arr)

        scores = {w: _score_window(m, windows[w], n_bars_map[w])
                  for w, m in models.items()}
        stats  = {w: _wedge_stats(windows[w], scores[w], threshold)
                  for w in models}
        depths = {w: len(windows[w]) for w in windows}

        sig50  = (scores.get(50)  is not None and scores.get(50)  >= threshold
                  and stats.get(50,  {}).get('gate_ok', False))
        sig250 = (scores.get(250) is not None and scores.get(250) >= threshold
                  and stats.get(250, {}).get('gate_ok', False))
        if sig50 or sig250:
            signals += 1
            log.info(f'SIGNAL  {row["timestamp"]}  '
                     f'50bar={scores.get(50, "n/a")}  '
                     f'250bar={scores.get(250, "n/a")}')

        _append_wedge_row(row['timestamp'], scores, threshold, depths, dry_run,
                          stats=stats)

    log.info(f'Replay complete. {signals} signal bars written to {WEDGE_CSV}  '
             f'({skipped} bar(s) rejected by spike filter)')


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Live rising-wedge monitor (SPY 1-min bars via yfinance)'
    )
    parser.add_argument('--ticker',    default='SPY',
                        help='Ticker symbol to monitor (default: SPY)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Score threshold for a signal (default: 0.8 — '
                             'per the v2 held-out family analysis; at 0.5 the '
                             'v2 models fire on ordinary channels)')
    parser.add_argument('--replay',    action='store_true',
                        help='Replay spy_data_1min.csv instead of live polling')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print output without writing to CSV files')
    parser.add_argument('--spike-threshold', type=float, default=4.0,
                        metavar='PCT',
                        help='Reject bars whose OHLC deviates more than PCT%% '
                             'from the recent median close (default: 4.0)')
    args = parser.parse_args()

    if args.replay:
        run_replay(args.ticker, args.threshold, args.dry_run,
                   spike_threshold=args.spike_threshold / 100.0)
    else:
        run_live(args.ticker, args.threshold, args.dry_run,
                 spike_threshold=args.spike_threshold / 100.0)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Catch any exception that escapes main() and write it to crash.log."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    log.critical('Fatal unhandled exception', exc_info=(exc_type, exc_value, exc_tb))
    _write_crash_entry(exc_value, context='fatal — process exiting')


sys.excepthook = _excepthook


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Catches failures during startup (model load, CSV init, etc.)
        _write_crash_entry(exc, context='startup failure')
        raise
