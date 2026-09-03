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
  - Stores every completed bar in  wedge.db  (SQLite, extended hours included)
  - Scores only regular-session bars, 9:30 AM-4:00 PM ET: extended-hours bars
    carry zero volume from yfinance and are off-distribution for the models
    (see the regular-session gate notes below; --score-extended-hours overrides)
  - Maintains a rolling window of 250 regular-session bars (wedge dropped its
    50-bar model at v5; every formation is 250-bar now),
    stitched across days (filled from the database on restart)
  - Normalises each window and scores it with the matching CNN model
  - Writes bar + scores + signal geometry atomically, one transaction/minute

Output
------
  wedge.db (SQLite; see wedge_db.py for the full schema and rationale):
    bars(ts, open, high, low, close, volume)     every accepted bar
    scores(ts, window, score, signal, bars)      regular-session bars only
    signals(ts, window, proj_move_usd, slope_upper, slope_lower,
            apex_min, apex_price, mid_travel, convergence,
            touch_up, touch_lo, max_excursion)   rows exist only where
                                                 signal = 1
    (signal = 1 when score >= threshold AND both geometry gates pass: the apex
     gate — converging lines with apex <= APEX_GATE_MAX_MIN ahead or already
     crossed — and, for v5 wedges, the QUALITY gate, which rejects diverging
     envelopes, near-parallel channels, lines that never touch the price, and
     large breaches. A scores row with a high score and signal = 0 was
     suppressed by one of those; the quality columns are written on every
     signalling bar so which one can be told apart afterwards. 61.8% of raw v5
     detections over SPY 2008-2021 fail the quality gate. See _wedge_stats.)

  On first start after the CSV era, an empty database is populated
  automatically from spy_data_1min.csv / rising_wedge.csv if they exist
  (idempotent; see wedge_db.migrate_csvs). The CSVs are no longer written.

Usage
-----
  # Live monitoring (default ticker=SPY, threshold=0.5)
  python live_monitor.py

  # Different ticker / stricter threshold
  python live_monitor.py --ticker QQQ --threshold 0.65

  # Re-score the stored bar history without waiting for the clock
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
from datetime import datetime, timezone, timedelta, time as dtime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import wedge_db

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / 'wedge.db'
CRASH_LOG    = PROJECT_ROOT / 'crash.log'
# Legacy CSVs: no longer written; kept as the migration source and as a
# read fallback until the database exists.
SPY_CSV      = PROJECT_ROOT / 'spy_data_1min.csv'
WEDGE_CSV    = PROJECT_ROOT / 'rising_wedge.csv'

# ── Market hours (US/Eastern) ─────────────────────────────────────────────────
# Covers pre-market (4 AM), regular session (9:30 AM-4:00 PM), after-hours (to 8 PM).
# On a normal day the after-hours cutoff is the regular close (4 PM) +
# AFTERHOURS_BUFFER_H, which reproduces the historical 8 PM cutoff. On an
# early-close half-day (1 PM close) it shrinks to 5 PM automatically.
MARKET_OPEN_H       = 4    # 4:00 AM ET  pre-market open
MARKET_CLOSE_H      = 20   # 8:00 PM ET  fallback close (calendar unavailable)
AFTERHOURS_BUFFER_H = 4    # hours of after-hours polling past the regular close

# ── Regular-session gate ──────────────────────────────────────────────────────
# Model windows hold REGULAR-session bars only (9:30 AM-4:00 PM ET, or the
# early close on half-days), stitched across days. Extended-hours bars are still
# fetched and appended to spy_data_1min.csv — the price history and the daily
# chart keep full pre/post coverage — they are just never fed to the models and
# get no row in rising_wedge.csv. Override with --score-extended-hours.
#
# Why (measured on the first five v2 days, 2026-07-23..29):
#   * yfinance reports volume 0 for 100% of pre-market and 99.6% of after-hours
#     bars. _normalise then divides the volume channel by its 1.0 guard instead
#     of a real maximum, so the CNN sees a constant-zero 5th feature — whereas
#     every training window is scaled to a volume max of exactly 1.0 (see
#     _volume_profile in generate_wedges.py). Extended-hours input is therefore
#     off-distribution on that feature by construction.
#   * The tiny extended-hours price range is stretched across the full [0, 1]
#     price scale, promoting tick noise to pattern-scale structure.
#   * Both showed up in the scores: mean 50-bar score 0.71 after 4 PM vs 0.60
#     in session, and 68% of all 250-bar signals fired post-close on 22% of the
#     bars.
# Regular hours only also matches what the models were validated on: the SPY
# 2008-2021 backtest corpus is 390-bar regular sessions stitched across days,
# with zero extended-hours bars. Those are the sessions behind the held-out
# AUCs, the apex gate's keep rate, and the PROJ_* fit.
REGULAR_OPEN_TIME  = dtime(9, 30)   # fallback when the NYSE calendar is absent
REGULAR_CLOSE_TIME = dtime(16, 0)

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

# -- v5 quality gate ----------------------------------------------------------
# The v5 model is a far better detector than v2/v3/v4, but 61.8% of what it
# fires on over SPY 2008-2021 is not a wedge: 18.5% diverging, 29.2%
# near-parallel channels, 13.6% with the fitted line floating off the price
# action entirely. These four tests reject those, and every threshold is
# calibrated against the v5 corpus at fit_bars=100 rather than chosen by eye --
# the first hand-picked excursion threshold was wrong by 8x and rejected 100%
# of detections. Re-derive them if fit_bars changes.
QUALITY_MIN_CONVERGENCE = 0.5    # below this the envelope is a channel
QUALITY_MIN_TOUCHES     = 5      # per side; designed-wedge p10, walks reach 2
QUALITY_MAX_EXCURSION   = 1.83   # mean-widths; designed-wedge p95 at fit_bars=100
QUALITY_TOUCH_TOL_FRAC  = 0.18   # a touch is within this fraction of the width


# ── Formation registry ────────────────────────────────────────────────────────
# Every formation scored each minute. Adding one is an entry here plus its
# weights on disk — the loop, the writer and the database schema all treat
# `pattern` as data.
#
#   windows    window sizes with trained weights for this formation
#   run_dir    <project root>/<run_dir>/window_<N>bar/models/cnn_best.weights.h5
#   threshold  default score threshold (overridable per formation on the CLI)
#   apex_gate  require converging lines with a near apex before signalling
#   geometry   'envelope' fits upper/lower trendlines and writes a signals
#              row; None logs the score only
#   fit_bars   how many bars from the RIGHT EDGE to fit the envelope over.
#              None = the whole window.
#
# fit_bars exists because a corpus with entry context puts a long directional
# approach in front of the formation, and fitting across all 250 bars then
# describes the approach rather than the wedge. Measured on the v3 corpus,
# whole-window convergence correlates with the true value at r = -0.074;
# fitting the last 120 bars recovers r = +0.510, against a +0.593 ceiling from
# knowing the true start exactly.
#
# 120 is used rather than an estimated start because every estimator tried
# lost to it: searching candidate starts gave 48-55 bars median error, and
# anchoring the fit at the right edge and extrapolating backward gave 19 --
# all worse than the fixed window's 15, because a converging wedge's lines
# diverge going backward into a funnel that price rarely escapes.
#
# It stays None for the CURRENTLY DEPLOYED models: v2 wedges and channels were
# trained on a corpus whose formation fills 30-85% of the window with only
# structureless padding in front, so whole-window fitting is roughly right for
# them. Setting 120 here before v3's weights are deployed would mis-measure
# the model actually running.
#   log_only   True records detections but never raises signal=1 -- the
#              formation is being collected for study, not acted on
#
# Channels deliberately have NO apex gate: their lines are parallel by
# definition, so an apex test would reject nearly all of them. The wedge gate
# stays because it was validated on the 2008-2021 backtest.
#
# Channel threshold is 0.9 rather than the wedge's 0.8: the held-out family
# analysis put precision at 0.983 / recall 0.919 there versus 0.969 / 0.945 at
# 0.8, and the 2008-2021 scan showed the channel model is already 13x more
# selective than the wedge model, so the recall cost is cheap.
#
# hs / inverse_hs are LOG-ONLY. Both detect their shape well -- held-out AUC
# 0.9999 / 1.0000, only 0.4% / 0.3% leakage from triple tops and bottoms, and
# a sane 45 / 71 events per year on SPY 2008-2021 with near-zero overlap
# against the other formations. But the forward-return study found no edge to
# act on: signed returns run the WRONG way for hs (down-rate 40.0% at 2hr
# against a 46.5% baseline), inverse_hs's apparent 54.6% up-rate at 4hr is a
# 54.2% market drift, and both sit at 0.93-1.02x baseline |return| so there is
# not even the volatility signature wedges and channels have. They are logged
# so the data accumulates for a later look, not because they are tradeable.
#
# Their geometry is None because fit_wedge_lines fits an ENVELOPE, which is
# meaningless for a head and shoulders -- the meaningful line is the neckline
# through the two troughs, and that needs a dedicated fitter. Logging envelope
# slopes under a mid_travel column would be quietly wrong data.
FORMATIONS = {
    # v5 (deployed 2026-09-03). 250-bar only: the 50-bar model was never
    # retrained past v2 and barely discriminated live anyway (median score 0.65
    # against the 250-bar's 0.087, ~12 signal clusters/day), so running it
    # alongside v5 would have been a v2/v5 hybrid.
    #
    # fit_bars is 100, NOT the 120 the v3-era note below anticipated. 120 was
    # justified by r = +0.510 measured on the v3 corpus; on v4 that collapsed to
    # +0.014, and re-deriving it on the v5 corpus gives 100 (r = +0.4707,
    # against 0.3395 at 120 and negative past 140). It moves with the corpus and
    # must be re-derived, never inherited.
    'wedge':      {'windows': (250,),    'run_dir': 'runs',
                   'threshold': 0.8, 'apex_gate': True,
                   'geometry': 'envelope', 'log_only': False,
                   'quality_gate': True, 'fit_bars': 100},
    'channel':    {'windows': (250,),    'run_dir': 'runs_channel',
                   'threshold': 0.9, 'apex_gate': False,
                   'geometry': 'envelope', 'log_only': False,
                   'fit_bars': None},
    'hs':         {'windows': (250,),    'run_dir': 'runs_hs',
                   'threshold': 0.9, 'apex_gate': False,
                   'geometry': None,      'log_only': True,
                   'fit_bars': None},
    'inverse_hs': {'windows': (250,),    'run_dir': 'runs_inverse_hs',
                   'threshold': 0.9, 'apex_gate': False,
                   'geometry': None,      'log_only': True,
                   'fit_bars': None},
}


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

def _load_model(window: int, pattern: str = 'wedge'):
    """
    Build + load weights for one (formation, window size).
    Returns (model, n_bars); (None, window) if the weights are absent, so a
    formation without trained weights is skipped rather than fatal.
    """
    os.environ['WEDGE_TOTAL_BARS'] = str(window)
    for mod in list(sys.modules.keys()):
        if mod == 'cnn_model':
            del sys.modules[mod]
    from cnn_model import build_model, N_BARS

    run_dir = FORMATIONS[pattern]['run_dir']
    candidates = [
        PROJECT_ROOT / run_dir / f'window_{window}bar' / 'models' / 'cnn_best.weights.h5',
    ]
    if pattern == 'wedge':          # legacy single-model fallback
        candidates.append(PROJECT_ROOT / 'models' / 'cnn_best.weights.h5')
    wp = next((p for p in candidates if p.exists()), None)
    if wp is None:
        log.warning(f'No weights for {pattern}@{window} '
                    f'(looked in {run_dir}/) — skipping this model.')
        return None, window

    model = build_model(print_summary=False)
    model.load_weights(str(wp))

    # Warm-up pass so first real inference isn't slow
    dummy = np.zeros((1, N_BARS, 5), dtype=np.float32)
    model.predict(dummy, verbose=0)

    log.info(f'Loaded {pattern}@{window}-bar model  [{run_dir}/...]')
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
# Regular-session membership  (see the REGULAR_*_TIME notes above)
# =============================================================================

_SESSION_BOUNDS_CACHE = {}   # date -> (open_time, close_time) | None


def _session_bounds(d) -> Optional[tuple]:
    """
    Return the regular NYSE session on date `d` as naive-ET (open, close)
    `datetime.time` objects, or None if the market was closed that day.

    Honours early closes (a half-day returns a 13:00 close). Falls back to a
    weekday + 9:30-16:00 assumption when pandas_market_calendars is unavailable
    — the same degradation as _is_market_open, i.e. holidays are not skipped.
    Cached per date, so the calendar is queried once per trading day.
    """
    if d in _SESSION_BOUNDS_CACHE:
        return _SESSION_BOUNDS_CACHE[d]

    weekday_fallback = ((REGULAR_OPEN_TIME, REGULAR_CLOSE_TIME)
                        if d.weekday() < 5 else None)

    cal = _get_nyse_calendar()
    if cal is None:
        bounds = weekday_fallback
    else:
        try:
            sched = cal.schedule(start_date=str(d), end_date=str(d))
        except Exception as exc:
            # Don't cache a transient failure as "market closed" — that would
            # suppress scoring for the whole day.
            log.warning(f'Calendar query failed for {d} ({exc}); '
                        f'assuming regular hours')
            return weekday_fallback
        if sched.empty:
            bounds = None                        # weekend or holiday
        else:
            tz = _et_zone()
            bounds = (sched.iloc[0]['market_open'].tz_convert(tz).time(),
                      sched.iloc[0]['market_close'].tz_convert(tz).time())

    _SESSION_BOUNDS_CACHE[d] = bounds
    return bounds


def _is_regular_session_bar(ts) -> bool:
    """
    True if a naive-ET bar timestamp falls inside that day's regular session.

    The interval is half-open, [open, close): a bar labelled 09:30 covers
    09:30-09:31, so the last regular bar of a normal day is 15:59 and a full
    session is 390 bars — matching the backtest corpus.
    """
    t = ts if isinstance(ts, datetime) else pd.Timestamp(ts).to_pydatetime()
    bounds = _session_bounds(t.date())
    if bounds is None:
        return False
    open_t, close_t = bounds
    return open_t <= t.time() < close_t


def _regular_session_mask(timestamps) -> pd.Series:
    """
    Vectorised _is_regular_session_bar for a Series of naive-ET timestamps.
    One calendar lookup per distinct date rather than per bar.
    """
    ts     = pd.to_datetime(timestamps)
    dates  = ts.dt.date
    bounds = {d: _session_bounds(d) for d in pd.unique(dates)}

    keep = []
    for d, t in zip(dates, ts.dt.time):
        b = bounds[d]
        keep.append(b is not None and b[0] <= t < b[1])
    return pd.Series(keep, index=ts.index)


# =============================================================================
# Database helpers  (schema and write functions live in wedge_db.py)
# =============================================================================

_STAT_KEYS = wedge_db.STAT_KEYS


def _open_db(dry_run: bool):
    """
    Open wedge.db, auto-migrating the legacy CSVs into it on the first run
    after the CSV era (empty bars table + CSVs present). Migration is
    idempotent, so a crash halfway simply re-runs next start.

    In dry-run mode nothing is created or migrated; an existing database is
    opened read-only for pre-filling, and a missing one yields None (the CSV
    fallback in _load_bars_frame covers that).
    """
    if dry_run:
        if DB_PATH.exists():
            return wedge_db.connect(DB_PATH, readonly=True)
        log.info(f'[dry-run] {DB_PATH.name} absent — falling back to CSVs '
                 f'for pre-fill; no database will be created.')
        return None

    con = wedge_db.connect(DB_PATH)
    if wedge_db.bar_count(con) == 0 and SPY_CSV.exists():
        log.info(f'Empty database + legacy CSVs found — migrating ...')
        c = wedge_db.migrate_csvs(con, SPY_CSV, WEDGE_CSV)
        log.info(f'  Migrated {c["bars"]:,} bars, {c["scores"]:,} score rows, '
                 f'{c["signals"]:,} signal rows into {DB_PATH.name}')
    return con


def _load_bars_frame() -> pd.DataFrame:
    """
    Full bar history as a DataFrame (timestamp + OHLCV), preferring the
    database and falling back to the legacy CSV. Read-only.
    """
    if DB_PATH.exists():
        con = wedge_db.connect(DB_PATH, readonly=True)
        try:
            rows = wedge_db.all_bars(con)
        finally:
            con.close()
        if rows:
            return pd.DataFrame(rows, columns=['timestamp'] + FEATURE_COLS)
    if SPY_CSV.exists():
        try:
            return pd.read_csv(SPY_CSV, usecols=['timestamp'] + FEATURE_COLS)
        except Exception as exc:
            log.warning(f'Could not read {SPY_CSV}: {exc}')
    return pd.DataFrame(columns=['timestamp'] + FEATURE_COLS)


# =============================================================================
# Rolling window init from existing CSV
# =============================================================================

def _load_rolling_window(n_bars: int, regular_only: bool = True) -> deque:
    """
    Pre-fill a rolling window from the tail of the stored bar history
    (database, or legacy CSV before migration).

    With regular_only (the default) the extended-hours rows are filtered out
    first, so a restart during or after a session rebuilds the same window the
    live loop would have built, instead of seeding it with zero-volume bars.
    """
    dq = deque(maxlen=n_bars)
    try:
        df = _load_bars_frame()
        if df.empty:
            return dq
        if regular_only:
            keep    = _regular_session_mask(df['timestamp'])
            dropped = int((~keep).sum())
            df      = df[keep]
            log.info(f'  Skipped {dropped:,} extended-hours row(s) '
                     f'when pre-filling')
        for _, row in df.tail(n_bars).iterrows():
            dq.append(row[FEATURE_COLS].values.astype(np.float32))
        log.info(f'  Pre-filled {n_bars}-bar window with {len(dq)} rows')
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
                 threshold: float, pattern: str = 'wedge') -> dict:
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
    # Formations without an envelope geometry (see FORMATIONS) log the score
    # only: fitting upper/lower trendlines to a head and shoulders would put
    # meaningless numbers in columns named for wedge geometry.
    if score is None or score < threshold or \
            FORMATIONS[pattern]['geometry'] is None:
        return dict(_ZERO_STATS)

    from classify_wedge import fit_wedge_lines
    full = np.array(window_deque, dtype=np.float32)     # raw (n_bars, 5)

    # Fit the envelope over the right-hand fit_bars only (see FORMATIONS).
    # A corpus with entry context puts a long approach in front of the
    # formation, and fitting across the whole window then measures the
    # approach instead. Clamped to the window, so a 50-bar model is
    # unaffected however fit_bars is set.
    fb  = FORMATIONS[pattern].get('fit_bars')
    arr = full if not fb else full[-min(int(fb), full.shape[0]):]
    g   = fit_wedge_lines(arr)
    n   = arr.shape[0]

    close      = float(full[-1, 3])
    travel_mid = (g['travel_upper'] + g['travel_lower']) / 2.0

    # Projected move: the PROJ_* fit was estimated on WEDGE events only
    # (2008-2021 slope study). Applying it to another formation would invent
    # a number, so anything else reports NULL until it has its own fit.
    if pattern == 'wedge':
        proj_pct  = max(PROJ_INTERCEPT_PCT + PROJ_TRAVEL_COEF * travel_mid, 0.0)
        proj_move = round(close * proj_pct / 100.0, 4)
    else:
        proj_move = None

    # Apex: crossing of upper/lower lines in raw price space. Still computed
    # for every formation (it is informative when a fit happens to converge),
    # but only gates when the formation asks for it — a channel's lines are
    # parallel by definition, so an apex test would reject nearly all of them.
    apex_min, apex_price, converging = 0, 0.0, False
    db = g['b_upper'] - g['b_lower']
    if db < -1e-12:                                     # converging lines
        x_cross = (g['a_lower'] - g['a_upper']) / db    # bars from window start
        ahead   = x_cross - (n - 1)                     # bars past current bar
        apex_min   = int(round(ahead))                  # 1 bar = 1 minute
        apex_price = float(g['a_upper'] + g['b_upper'] * x_cross)
        converging = ahead <= APEX_GATE_MAX_MIN         # near or already pinched

    # Quality gate: is the thing we detected actually a wedge? Measured on the
    # v5 detection study over SPY 2008-2021, 61.8% of detections failed one of
    # these, so this is not a rounding-error filter -- it rejects most of them.
    #
    #   convergence < 0        a widening envelope is a megaphone
    #   convergence < 0.5      near-parallel, i.e. a CHANNEL. Channels ride
    #                          their rails perfectly, so the touch test cannot
    #                          catch one -- this is the only test that can
    #   touches < 5 per side   the line floats off the price action, anchored
    #                          by a stray spike. Designed wedges reach 5/5/8 at
    #                          p5/p10/median under this fit; random walks 2/2/5
    #   max_excursion > 1.83   breached worse than 95% of designed wedges
    #
    # Every threshold is calibrated against the v5 corpus AT fit_bars=100, the
    # value this formation deploys with. They move with the fit span: the 1.83
    # here is 1.98 at fit_bars=120.
    touch_up = touch_lo = max_exc = None
    quality_ok = True
    if FORMATIONS[pattern].get('quality_gate'):
        xq = np.arange(n, dtype=float)
        up = g['a_upper'] + g['b_upper'] * xq
        lo = g['a_lower'] + g['b_lower'] * xq
        width = up - lo
        mw  = max(float(np.mean(np.abs(width))), 1e-12)
        exc = np.maximum(np.maximum(arr[:, 1] - up, 0.0),
                         np.maximum(lo - arr[:, 2], 0.0)) / mw
        max_exc = float(exc.max())
        tol = QUALITY_TOUCH_TOL_FRAC * np.abs(width)

        def _runs(mask: np.ndarray) -> int:
            m = mask.astype(np.int8)
            return int(np.sum(m - np.concatenate([[0], m[:-1]]) == 1))

        touch_up = _runs(arr[:, 1] >= up - tol)
        touch_lo = _runs(arr[:, 2] <= lo + tol)
        quality_ok = (g['convergence'] >= QUALITY_MIN_CONVERGENCE
                      and touch_up >= QUALITY_MIN_TOUCHES
                      and touch_lo >= QUALITY_MIN_TOUCHES
                      and max_exc <= QUALITY_MAX_EXCURSION)

    apex_ok = converging if FORMATIONS[pattern]['apex_gate'] else True
    gate_ok = apex_ok and quality_ok

    return {
        'proj_move_usd': proj_move,
        'slope_upper':   round(g['b_upper'], 6),        # $ per bar
        'slope_lower':   round(g['b_lower'], 6),
        'apex_min':      apex_min,
        'apex_price':    round(apex_price, 4),
        # Wedge midline tilt across the window, as a fraction of the window's
        # price range (+1 = climbed the full range; - = falling wedge). Logged
        # raw so slope-based interpretations (e.g. the observed "steep-rising
        # = quiet continuation" regime) can be evaluated on live data post-hoc.
        'mid_travel':    round(travel_mid, 4),
        # Quality-gate inputs, recorded whether or not they passed, so a
        # suppressed signal can be told apart from a monitor that stopped.
        'convergence':   round(g['convergence'], 4),
        'touch_up':      touch_up,
        'touch_lo':      touch_lo,
        'max_excursion': None if max_exc is None else round(max_exc, 4),
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
             spike_threshold: float = 0.04, regular_only: bool = True,
             thresholds: Optional[dict] = None) -> None:
    """
    Main monitoring loop — runs until keyboard interrupt.

    thresholds: per-formation score thresholds. Defaults to each formation's
    registry value; `threshold` overrides every formation (kept so the old
    single --threshold flag still means something).
    """
    thresholds = thresholds or {p: threshold for p in FORMATIONS}

    # Load every formation's models, keyed by (pattern, window)
    log.info('Loading models ...')
    models, n_bars_map = {}, {}
    for pattern, cfg in FORMATIONS.items():
        for w in cfg['windows']:
            m, nb = _load_model(w, pattern)
            if m is not None:
                models[(pattern, w)]     = m
                n_bars_map[(pattern, w)] = nb

    if not models:
        raise SystemExit('No models loaded. Run the pipeline first.')

    # One rolling window per DISTINCT window size, shared across formations —
    # the bars are identical, only the models differ.
    window_sizes = sorted({w for _, w in models})

    # Open the database (auto-migrates the legacy CSVs on first run)
    con = _open_db(dry_run)

    # Pre-fill rolling windows from existing data
    log.info('Pre-filling rolling windows from stored bar history ...')
    windows = {w: _load_rolling_window(w, regular_only) for w in window_sizes}

    # Seed the spike filter's reference from the freshest pre-filled window
    # (row layout is [open, high, low, close, volume] — close is index 3)
    spike_filter = SpikeFilter(threshold=spike_threshold)
    largest = max(windows.values(), key=len, default=None)
    if largest:
        spike_filter.seed(row[3] for row in list(largest)[-10:])

    last_ts = wedge_db.last_bar_ts(con) if con is not None else None
    log.info(f'Last stored bar: {last_ts or "none (fresh start)"}')
    log.info(f'Monitoring {ticker}  |  extended hours from {MARKET_OPEN_H}:00 ET')
    log.info('Formations: ' + ',  '.join(
        f'{p}@{list(FORMATIONS[p]["windows"])} thr={thresholds[p]}'
        f'{" +apex-gate" if FORMATIONS[p]["apex_gate"] else ""}'
        for p in sorted(FORMATIONS)))
    log.info('Scoring: regular session only (9:30-16:00 ET); extended-hours '
             'bars are logged to the price CSV but not scored.'
             if regular_only else
             'Scoring: ALL polled bars, including extended hours '
             '(--score-extended-hours) — zero-volume bars are off-distribution '
             'for the models; see the regular-session gate notes.')

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

            # ── Regular-session gate ──────────────────────────────────────────
            # Every accepted bar is stored, in session or not: the bars table
            # keeps full pre/post-market coverage. But extended-hours bars
            # never reach the windows or the models, so they also cannot
            # contaminate a later in-session window.
            if regular_only and not _is_regular_session_bar(bar['timestamp']):
                if con is not None and not dry_run:
                    wedge_db.write_bar(con, bar)
                log.info(f'{bar["timestamp"]}  {ticker}  '
                         f'C={bar["close"]:.2f}  extended hours — stored, not scored')
                continue

            # ── Update rolling windows ────────────────────────────────────────
            bar_arr = np.array(
                [bar[c] for c in FEATURE_COLS], dtype=np.float32
            )
            for dq in windows.values():
                dq.append(bar_arr)

            # ── Score every formation on every one of its windows ─────────────
            # A DETECTION is a threshold crossing that passed the formation's
            # gate. A SIGNAL is a detection worth acting on. They differ for
            # log-only formations, which are recorded but never raise a signal.
            scores, stats, sigs, dets, depths = {}, {}, {}, {}, {}
            for key, model in models.items():
                pattern, w = key
                cfg = FORMATIONS[pattern]
                thr = thresholds[pattern]
                sc  = _score_window(model, windows[w], n_bars_map[key])
                st  = _wedge_stats(windows[w], sc, thr, pattern)
                # a formation without an apex gate always passes it; do not
                # read gate_ok from the zeroed stats of a geometry-less one
                gate_ok = st.get('gate_ok', False) if cfg['apex_gate'] else True
                over = bool(sc is not None and sc >= thr and gate_ok)
                scores[key] = sc
                stats[key]  = st
                dets[key]   = over
                sigs[key]   = 0 if cfg['log_only'] else int(over)
                depths[key] = len(windows[w])

            # ── Log + write ───────────────────────────────────────────────────
            # '*' = actionable signal, '.' = detection on a log-only formation
            def _mark(k):
                return '*' if sigs[k] else ('.' if dets[k] else ' ')

            score_str = '  '.join(
                f'{p}@{w}={scores[(p, w)]:.4f}{_mark((p, w))}'
                if scores[(p, w)] is not None else f'{p}@{w}=n/a '
                for (p, w) in sorted(models)
            )

            fired = [k for k in sorted(models) if sigs[k]]
            logged = [k for k in sorted(models) if dets[k] and not sigs[k]]
            note = ''
            if fired:
                parts = []
                for pattern, w in fired:
                    st = stats[(pattern, w)]
                    bits = [f'slope={st["mid_travel"]:+.3f}']
                    if st['proj_move_usd'] is not None:
                        bits.append(f'proj_move=${st["proj_move_usd"]:.2f}')
                    if st['apex_min']:
                        bits.append(f'apex {st["apex_min"]}m '
                                    f'@ ${st["apex_price"]:.2f}')
                    parts.append(f'{pattern}@{w} ' + ' '.join(bits))
                note = '  <<< ' + ' | '.join(parts) + ' >>>'
            if logged:
                note += ('  [log-only: '
                         + ', '.join(f'{p}@{w}' for p, w in logged) + ']')

            log.log(logging.WARNING if fired else logging.INFO,
                    f'{bar["timestamp"]}  {ticker}  '
                    f'C={bar["close"]:.2f}  {score_str}' + note)

            if con is not None and not dry_run:
                wedge_db.write_minute(con, bar, scores, sigs, stats, depths)

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
               spike_threshold: float = 0.04, regular_only: bool = True,
               thresholds: Optional[dict] = None) -> None:
    """
    Replay the stored bar history through both models as fast as possible.
    Useful for testing scoring logic without waiting for live data.
    Bars come from the database (or the legacy CSV before migration); a
    non-dry-run replay CLEARS the scores and signals tables first and
    rewrites them, leaving the bars table untouched. Applies the same spike
    filter as live mode, so bad ticks already stored (e.g. 2026-07-07 17:07)
    are skipped during re-scoring, and the same regular-session gate, so a
    replay reproduces live scoring.
    """
    thresholds = thresholds or {p: threshold for p in FORMATIONS}

    df = _load_bars_frame()
    if df.empty:
        raise SystemExit('No stored bars to replay (no database, no CSV).')

    log.info(f'Replaying stored bar history ...')
    log.info(f'  {len(df):,} bars  ({df["timestamp"].iloc[0]} to {df["timestamp"].iloc[-1]})')

    if regular_only:
        keep  = _regular_session_mask(df['timestamp'])
        n_ext = int((~keep).sum())
        df    = df[keep].reset_index(drop=True)
        log.info(f'  Regular-session gate: dropped {n_ext:,} extended-hours '
                 f'bar(s); {len(df):,} bar(s) to score')
        if df.empty:
            raise SystemExit('No regular-session bars to replay.')

    # Load every formation's models, keyed by (pattern, window)
    models, n_bars_map = {}, {}
    for pattern, cfg in FORMATIONS.items():
        for w in cfg['windows']:
            m, nb = _load_model(w, pattern)
            if m is not None:
                models[(pattern, w)]     = m
                n_bars_map[(pattern, w)] = nb

    if not models:
        raise SystemExit('No models loaded.')

    # Reset model output for a clean replay (bars are kept).
    con = None
    if not dry_run:
        con = wedge_db.connect(DB_PATH)
        wedge_db.clear_scores(con)
        log.info('  Cleared scores and signals tables for a clean replay.')

    window_sizes = sorted({w for _, w in models})
    windows = {w: deque(maxlen=w) for w in window_sizes}
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

        scores, stats, sigs, depths = {}, {}, {}, {}
        for key, m in models.items():
            pattern, w = key
            cfg = FORMATIONS[pattern]
            thr = thresholds[pattern]
            sc  = _score_window(m, windows[w], n_bars_map[key])
            st  = _wedge_stats(windows[w], sc, thr, pattern)
            gate_ok = st.get('gate_ok', False) if cfg['apex_gate'] else True
            over = bool(sc is not None and sc >= thr and gate_ok)
            scores[key], stats[key] = sc, st
            sigs[key]   = 0 if cfg['log_only'] else int(over)
            depths[key] = len(windows[w])

        if any(sigs.values()):
            signals += 1
            fired = ' '.join(f'{p}@{w}={scores[(p, w)]:.4f}'
                             for (p, w) in sorted(models) if sigs[(p, w)])
            log.info(f'SIGNAL  {row["timestamp"]}  {fired}')

        if con is not None:
            bar = {'timestamp': row['timestamp'],
                   **{c: float(row[c]) for c in FEATURE_COLS}}
            wedge_db.write_minute(con, bar, scores, sigs, stats, depths)

    if con is not None:
        con.close()

    # Say what actually happened: under --dry-run nothing was written, and
    # claiming otherwise reads as if the live scores had been overwritten.
    destination = ('nothing written (--dry-run)' if dry_run
                   else f'written to {DB_PATH.name}')
    log.info(f'Replay complete. {signals} signal bar(s); {destination}  '
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
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override the score threshold for EVERY '
                             'formation. Omit to use each formation\'s own '
                             'default: '
                             + ', '.join(f'{p}={c["threshold"]}'
                                         for p, c in sorted(FORMATIONS.items())))
    parser.add_argument('--pattern-threshold', action='append', default=[],
                        metavar='NAME=VALUE',
                        help='Override one formation only, repeatable '
                             '(e.g. --pattern-threshold channel=0.85)')
    parser.add_argument('--replay',    action='store_true',
                        help='Re-score the stored bar history instead of '
                             'live polling')
    parser.add_argument('--dry-run',   action='store_true',
                        help='Print output without writing to CSV files')
    parser.add_argument('--spike-threshold', type=float, default=4.0,
                        metavar='PCT',
                        help='Reject bars whose OHLC deviates more than PCT%% '
                             'from the recent median close (default: 4.0)')
    parser.add_argument('--score-extended-hours', action='store_true',
                        help='Also score pre/after-hours bars. Off by default: '
                             'yfinance reports zero volume outside the regular '
                             'session, which is off-distribution for the models '
                             '(see the regular-session gate notes). For '
                             'experiments only.')
    args = parser.parse_args()

    regular_only = not args.score_extended_hours

    # Per-formation thresholds: registry defaults, then --threshold applied to
    # all, then --pattern-threshold applied to individual formations.
    thresholds = {p: c['threshold'] for p, c in FORMATIONS.items()}
    if args.threshold is not None:
        thresholds = {p: args.threshold for p in thresholds}
    for spec in args.pattern_threshold:
        name, _, val = spec.partition('=')
        if name not in thresholds:
            raise SystemExit(f'Unknown formation {name!r} in '
                             f'--pattern-threshold; known: '
                             f'{", ".join(sorted(thresholds))}')
        thresholds[name] = float(val)

    fallback = args.threshold if args.threshold is not None else 0.8
    if args.replay:
        run_replay(args.ticker, fallback, args.dry_run,
                   spike_threshold=args.spike_threshold / 100.0,
                   regular_only=regular_only, thresholds=thresholds)
    else:
        run_live(args.ticker, fallback, args.dry_run,
                 spike_threshold=args.spike_threshold / 100.0,
                 regular_only=regular_only, thresholds=thresholds)


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
