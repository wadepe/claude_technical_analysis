"""
generate_wedges.py  (corpus v2)

Synthetic OHLCV corpus generator for the wedge-formation detector, v2.
Replaces generate_rising_wedge.py's corpus for model-v2 training. Key design
changes vs v1 (see git history on the model-v2 branch for rationale):

  1. ONE continuous wedge family, not just rising: both trendline slopes are
     sampled continuously (rising / falling / flat / symmetric all emerge),
     constrained only by convergence (m_lower > m_upper). Wedge *type* is
     determined post-hoc by fitting slopes, not by the model.

  2. Positives are FORMING wedges anchored at the window's RIGHT EDGE and
     truncated mid-formation (completion 50-95% of the run to apex), with NO
     breakout. The model's score means "a wedge is forming right now", not
     "a completed wedge exists somewhere in the window".

  3. HARD NEGATIVES alongside random walks:
       parallel channels  — structure without convergence
       megaphones         — divergence (convergence with the sign flipped)
       stale wedges       — completed wedge + breakout that resolved well
                            before the right edge (old news must score 0)

  4. An EXCLUSION BAND separates classes by construction: positives shrink
     channel width by >= 50%; channel negatives stay within +/-15%; megaphones
     expand >= 50%. No ambiguous boundary cases carry a hard label.

  5. Volume profiles are DECORRELATED from the label: v1 gave every positive
     declining volume and every negative flat volume, so the model could
     classify on volume fade alone (real SPY volume fades into lunch daily —
     the suspected cause of v1's midday-only signals). v2 mixes profiles:
     wedges usually-but-not-always fade; channels/megaphones sometimes fade.

Families and label
------------------
  forming_wedge   label=1   seed offset          0
  walk            label=0   seed offset  1,000,000   (v1-compatible dynamics)
  channel         label=0   seed offset  2,000,000
  megaphone       label=0   seed offset  3,000,000
  stale_wedge     label=0   seed offset  4,000,000

Column schema (identical to v1 so train_cnn.py works unchanged)
---------------------------------------------------------------
  bar, open, high, low, close, volume        normalised [0, 1]
  lower_trendline, upper_trendline           [0,1] on pattern bars, NaN off
  segment                                    0=noise pad, 1=pattern, 2=breakout
  label                                      1=forming wedge, 0=everything else

Usage
-----
  # Validation plots (3 examples per family) + parameter distribution stats
  python generate_wedges.py --validate

  # Full corpus into an isolated run dir (250-bar default; 50-bar via env)
  python generate_wedges.py --corpus --output-dir ../runs_v2/window_250bar
  WEDGE_TOTAL_BARS=50 python generate_wedges.py --corpus \
      --output-dir ../runs_v2/window_50bar

  # Custom family counts / worker processes
  python generate_wedges.py --corpus --n-wedge 200000 --n-walk 150000 \
      --n-channel 150000 --n-megaphone 100000 --n-stale 100000 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TOTAL_BARS = int(os.environ.get('WEDGE_TOTAL_BARS', '250'))
_SCALE     = 250.0 / TOTAL_BARS   # slope scaling relative to 250-bar baseline

SEED_OFFSETS = {
    'forming_wedge': 0,
    'walk':          1_000_000,
    'channel':       2_000_000,
    'megaphone':     3_000_000,
    'stale_wedge':   4_000_000,
    'compression_walk': 13_000_000,
    # peak-family formations (head and shoulders and its confusions)
    'hs':               5_000_000,
    'inverse_hs':       6_000_000,
    'triple_top':       7_000_000,
    'double_top':       8_000_000,
    'hs_stale':         9_000_000,
    'triple_bottom':   10_000_000,
    'double_bottom':   11_000_000,
    'inverse_hs_stale':12_000_000,
}

# Peak families as (shape, orientation). Orientation is a property of the
# family, not something derived from its name, because every confusion shape
# needs BOTH orientations: a triple top is a trivially easy negative for the
# inverse model (it is upside down from everything that model sees), while a
# triple BOTTOM is its genuinely hard case. Each corpus therefore pairs its
# positive with orientation-matched confusions.
_PEAK_SHAPE = {
    'hs':               ('hs',     +1),
    'inverse_hs':       ('hs',     -1),
    'hs_stale':         ('hs',     +1),
    'inverse_hs_stale': ('hs',     -1),
    'triple_top':       ('triple', +1),
    'triple_bottom':    ('triple', -1),
    'double_top':       ('double', +1),
    'double_bottom':    ('double', -1),
}
PEAK_FAMILIES = tuple(_PEAK_SHAPE)

# ── Class-separation geometry (the exclusion band) ────────────────────────────
WEDGE_COMPLETION_RANGE   = (0.50, 0.95)   # fraction of run-to-apex at right edge
CHANNEL_WIDTH_RATIO_RNG  = (0.85, 1.15)   # end/start width: essentially parallel
MEGAPHONE_WIDTH_RATIO_RNG = (1.5, 3.0)    # end/start width: clearly diverging
MIN_END_WIDTH            = 0.018          # absolute floor so the apex never pinches
                                          # below the noise scale

# Visible pattern length as a fraction of the window (all anchored families)
PATTERN_LEN_FRAC = ((0.36, 0.60) if os.environ.get('WEDGE_ENTRY_CONTEXT') == '1'
                    else (0.30, 0.85))

# -- v3: entry context --------------------------------------------------------
# A real formation is ENTERED: price arrives from outside the eventual
# trendlines and only then becomes bounded by them. A rising wedge is entered
# from below (it decelerates a prior advance), a falling wedge from above.
#
# v2 had no such structure -- _bridge_prepad starts at the formation entry
# price and random-walks backwards, so every positive was a formation already
# bounded by its lines with undifferentiated noise in front. Reviewing 20 real
# detections showed the cost: the detector scored 0.94-0.99 on plain uptrends,
# never having been shown what entering a formation looks like.
#
# Applied to ALL anchored families. Giving positives a structured approach
# while negatives keep the old bridge would teach "has an approach" rather
# than "is a wedge".
#
# With this on, PATTERN_LEN_FRAC targets 1.5-2.5 hour formations at 250 bars
# (90-150 bars) leaving 100-160 bars of approach.
ENTRY_CONTEXT    = os.environ.get('WEDGE_ENTRY_CONTEXT') == '1'
ENTRY_OFFSET_RNG = (0.35, 1.10)   # how far outside the boundary, x start width
ENTRY_CURVE_RNG  = (0.8, 2.2)     # approach shape; >1 converges late
ENTRY_TRAVEL_CAP = 1.60           # cap on backward extrapolation, x start width

# -- v4: volatility profile on EVERY family -----------------------------------
# v3 gave decaying volatility only to compression_walk and left positives
# stationary, making decay a near-perfect negative marker. The model learned
# "quiet = not a wedge" and collapsed to 3 events/year, selecting for
# compression harder than v2: median detection compression 0.284 against a
# population median of 0.964. That is backwards -- the midday lull is when
# participation pauses and consolidation genuinely happens.
#
# Drawing the SAME profile distribution for every family removes the label
# information while keeping the shape, so the model must separate a quiet
# window WITH boundary structure from a quiet one without.
VOL_PROFILE      = os.environ.get('WEDGE_VOL_PROFILE') == '1'
VOL_DECAY_PROB   = 0.75            # share of windows that go quiet
VOL_DECAY_RNG    = (0.20, 0.55)    # end/start sigma ratio when decaying
VOL_RISE_RNG     = (1.20, 3.00)    # ... and when building
VOL_CURVE_RNG    = (0.70, 1.80)

# -- v4: overnight gaps -------------------------------------------------------
# The corpus is continuous; real 250-bar windows are stitched across session
# boundaries and 64% contain an overnight jump. Measured on SPY 2008-2021
# regular-session bars (3,346 gaps) that jump is large relative to what the
# window otherwise shows: median 0.466 of the prior 250-bar RANGE, mean 0.630,
# p90 1.366, and 19% exceed the whole range.
#
# v3 responds to it in the wrong direction: 0.108% detection rate on
# gap-spanning windows against 0.019% gap-free, so 5x more likely to fire
# where a discontinuity sits in the approach it now treats as structure.
#
# GAP_PROB is not free: a 250-bar window inside a 390-bar session contains a
# boundary whenever it ends before session minute 249, i.e. 249/390 = 0.638,
# and the boundary lands uniformly across the window.
GAPS             = os.environ.get('WEDGE_GAPS') == '1'
GAP_PROB         = 0.638
GAP_LOGNORM      = (-0.76, 0.85)   # mu, sigma of ln(gap / window range)
GAP_MAX_REL      = 3.0             # clip the tail; p99 measured at 2.687
GAP_UP_PROB      = 0.54            # measured: 54% up, 45% down

# -- v5: the boundary stays in the APPROACH -----------------------------------
# v4 placed the gap uniformly across the window, which corrupted 22% of its own
# positives. Measured on 2,000 v4 forming_wedge windows: a gap at bar 130-199
# faked extreme convergence (fitted compression 0.065 against a designed 0.367),
# and one at 200-249 -- landing on the right-edge anchor the entire corpus design
# rests on -- faked divergence, with 60.1% fitting as DIVERGING when none were
# designed that way. Whole-corpus designed-vs-fitted correlation collapsed to
# r = +0.014 (v3: +0.432), which also invalidated live_monitor's fit_bars=120.
#
# On a real chart a large gap inside a forming wedge breaks the formation -- it
# stops being a forming wedge. _apply_gap instead shifts the trendlines with the
# price so the formation "stays inside its own lines", producing a shape that
# cannot occur in the data the model is deployed on.
#
# The cap is ONE constant applied to EVERY family, deliberately not each family's
# own formation start. Constraining per-family would make the gap POSITION
# distribution differ by label -- the same class of leak as the v3 volatility
# profile, and in the worse direction: it would teach "late gap = positive".
# 100 is the smallest pre_pad an anchored family can draw under ENTRY_CONTEXT
# (n_vis <= int(0.60 * 250) = 150), so no forming_wedge, channel or megaphone
# formation can contain a boundary. WEDGE_GAP_MAX_POS=250 restores v4's uniform
# placement for reproduction; <= 0 also means unconstrained.
GAP_MAX_POS      = int(os.environ.get('WEDGE_GAP_MAX_POS', '100'))

# -- v4: price roughness ------------------------------------------------------
# Measured on 1,500 random SPY 250-bar windows vs 400 synthetic ones, the
# corpus was markedly smoother than reality on every texture metric:
#
#                     bar step   bar range   wiggle
#   real SPY            0.0283      0.0531     14.1
#   forming_wedge       0.0177      0.0295      5.6
#   walk                0.0135      0.0238      4.9
#
# (bar step and range as a fraction of the window range; wiggle = path length
# over net travel, so 14.1 means real price covers fourteen times the ground
# it nets.) Every model so far learned formations against a smoother world
# than it is asked to work in.
#
# WIDENED rather than shifted: SPY is an ETF and among the smoothest
# instruments there is, so a corpus meant to generalise past it needs headroom
# above SPY texture, not a distribution centred on it.
NOISE_SIGMA_RNG = (0.010, 0.042)

# ── Which family carries label=1 ──────────────────────────────────────────────
# 'forming_wedge' reproduces the v2 wedge corpus exactly. 'channel' builds the
# channel-detector corpus, where wedges and megaphones become hard negatives
# (structure with the wrong width behaviour) -- the mirror image of the wedge
# corpus, sharing the same exclusion band so no window is ambiguous.
POSITIVE_FAMILY = os.environ.get('WEDGE_POSITIVE_FAMILY', 'forming_wedge')

# ── Channel touch constraint ──────────────────────────────────────────────────
# A channel is only identifiable if price actually reaches both boundaries at
# least twice -- you cannot draw the lines otherwise. _channel_closes alone
# mean-reverts toward the midline, so touches are emergent and often absent.
# When channels are the POSITIVE class we schedule touches explicitly and
# verify them; a sample that still falls short after retries is rejected.
#
# Enabled ONLY when channel is the positive family. Turning it on for the wedge
# corpus would change that corpus's channel NEGATIVES, silently breaking
# reproducibility of the deployed v2 wedge model, which was trained against the
# unconstrained ones.
CHANNEL_TOUCHES_REQUIRED = 2            # per side, minimum
CHANNEL_TOUCH_COUNT_RNG  = (2, 5)       # scheduled touches per side
CHANNEL_TOUCH_TOL_FRAC   = 0.18         # within this fraction of channel width
CHANNEL_TOUCH_MIN_GAP    = 0.06         # min separation, fraction of pattern len
CHANNEL_TOUCH_PULL       = 0.55         # attraction strength at a scheduled bar
CHANNEL_TOUCH_RETRIES    = 6


# =============================================================================
# Shared price/candle machinery (adapted from the v1 generator)
# =============================================================================

def _vol_profile(rng, n: int, base_sigma: float) -> np.ndarray:
    """
    Per-bar volatility across a window. Identical distribution for every
    family, so the profile carries no label information (see VOL_PROFILE).
    Returns a constant array when the feature is off, so callers are uniform.
    """
    if not VOL_PROFILE or n <= 0:
        return np.full(max(n, 0), base_sigma)
    ratio = (rng.uniform(*VOL_DECAY_RNG) if rng.random() < VOL_DECAY_PROB
             else rng.uniform(*VOL_RISE_RNG))
    curve = rng.uniform(*VOL_CURVE_RNG)
    scale = np.linspace(1.0, 0.0, n) ** curve
    return base_sigma * (ratio + (1.0 - ratio) * scale)


def _apply_gap(rng, closes: np.ndarray, *extra) -> tuple:
    """
    Insert one overnight gap: a level shift from a bar in the APPROACH to the
    end of the window. `extra` arrays (trendlines) shift identically, so the
    approach and the formation sit at a consistent level either side of it.

    The boundary is confined to bar < GAP_MAX_POS so it can never land inside a
    formation -- v4 placed it uniformly and corrupted 22% of its own positives
    that way. See GAP_MAX_POS for the measurements.

    Returns everything unchanged when the feature is off or no gap is drawn,
    so roughly a third of windows stay continuous and the gap carries no
    label information.
    """
    if not GAPS or len(closes) < 10 or rng.random() >= GAP_PROB:
        return (-1, closes, *extra)
    span = float(np.nanmax(closes) - np.nanmin(closes))
    if span <= 1e-9:
        return (-1, closes, *extra)
    mu, sd = GAP_LOGNORM
    rel = min(float(np.exp(rng.normal(mu, sd))), GAP_MAX_REL)
    amt = rel * span * (1.0 if rng.random() < GAP_UP_PROB else -1.0)
    # Uniform within the APPROACH only (see GAP_MAX_POS). Identical bound for
    # every family, so gap position carries no label information.
    hi = len(closes) if GAP_MAX_POS <= 0 else min(GAP_MAX_POS, len(closes))
    p = int(rng.randint(1, max(hi, 2)))
    out = [closes.copy()]
    out[0][p:] += amt
    for a in extra:
        b = a.copy()
        b[p:] += amt
        out.append(b)
    # Position is returned and recorded in every family's meta so the corpus
    # can be audited for label symmetry directly: gap RATE and gap POSITION
    # must match across families, or the model learns "gap = wedge".
    return (p, *out)


def _garch_step(rng, vol_t: float, base_sigma: float, persist: float,
                fat_prob: float, fat_mult: float) -> tuple[float, float]:
    """One GARCH-ish volatility update + a fat-tailed price step."""
    shock = rng.normal(0, base_sigma * 0.20)
    vol_t = base_sigma * (1 - persist) + persist * vol_t + shock
    vol_t = float(np.clip(vol_t, base_sigma * 0.25, base_sigma * 3.5))
    if rng.random() < fat_prob:
        step = rng.normal(0.0, vol_t * fat_mult)
    else:
        step = rng.normal(0.0, vol_t)
    return vol_t, step


def _channel_closes(rng, n: int, lower: np.ndarray, upper: np.ndarray,
                    noise_sigma: float, rev_strength: float, momentum_str: float,
                    vol_persist: float, fat_prob: float, fat_mult: float,
                    violation_prob: float,
                    touch_target: np.ndarray = None) -> np.ndarray:
    """
    Simulate closes bouncing inside a channel defined by per-bar lower/upper
    boundary arrays (works for converging, parallel, and diverging channels).

    touch_target: optional per-bar array of +1 (steer toward the upper
    boundary), -1 (toward the lower) or 0 (free). Where non-zero, the pull
    toward the midline is replaced by a pull toward that boundary, so the
    scheduled touches emerge from the same dynamics rather than being stamped
    on afterwards. None reproduces the original unconstrained behaviour.
    """
    mid    = (lower + upper) / 2.0
    slope  = np.diff(mid, prepend=mid[0])
    closes = np.empty(n)
    closes[0] = lower[0] + (upper[0] - lower[0]) * rng.uniform(0.30, 0.70)
    sig_arr = (np.full(n, noise_sigma) if np.isscalar(noise_sigma)
               else np.asarray(noise_sigma, dtype=float))
    vol_t = float(sig_arr[0])

    for i in range(1, n):
        vol_t, step = _garch_step(rng, vol_t, sig_arr[i], vol_persist,
                                  fat_prob, fat_mult)
        mom = momentum_str * (closes[i-1] - closes[i-2]) if i >= 2 else 0.0
        tgt = 0 if touch_target is None else touch_target[i]
        if tgt > 0:
            rev = CHANNEL_TOUCH_PULL * (upper[i-1] - closes[i-1])
        elif tgt < 0:
            rev = CHANNEL_TOUCH_PULL * (lower[i-1] - closes[i-1])
        else:
            rev = rev_strength * (mid[i-1] - closes[i-1])
        closes[i] = closes[i-1] + slope[i] + rev + mom + step

        # Boundary handling: reject back inside, or occasionally poke through
        if closes[i] > upper[i]:
            if rng.random() < violation_prob:
                closes[i] = upper[i] + abs(rng.normal(0, sig_arr[i] * 0.40))
            else:
                closes[i] = upper[i] - abs(rng.normal(0, sig_arr[i] * 0.15))
        elif closes[i] < lower[i]:
            if rng.random() < violation_prob:
                closes[i] = lower[i] - abs(rng.normal(0, sig_arr[i] * 0.40))
            else:
                closes[i] = lower[i] + abs(rng.normal(0, sig_arr[i] * 0.15))
    return closes


def _plan_touches(rng, n: int) -> tuple:
    """
    Schedule alternating boundary visits across a pattern of n bars.

    Returns (touch_target, n_up, n_lo): a per-bar steer array for
    _channel_closes, and how many visits were scheduled per side. Visits
    alternate sides so the path zig-zags between support and resistance the
    way a real channel does, are spaced at least CHANNEL_TOUCH_MIN_GAP apart,
    and each occupies a short run of bars so the approach is gradual.
    """
    n_up = int(rng.randint(*CHANNEL_TOUCH_COUNT_RNG))
    n_lo = int(rng.randint(*CHANNEL_TOUCH_COUNT_RNG))
    total = n_up + n_lo

    gap = max(int(n * CHANNEL_TOUCH_MIN_GAP), 2)
    # Evenly spread slots across the pattern, jittered, keeping the gap.
    span = n - 2 * gap
    if span <= 0 or total < 2:
        return None, 0, 0
    base = np.linspace(gap, n - gap - 1, total)
    jitter = rng.uniform(-gap * 0.4, gap * 0.4, total)
    slots = np.clip(np.round(base + jitter), 1, n - 1).astype(int)
    slots = np.unique(slots)

    # Alternate sides, starting from whichever side has more visits to place.
    start_up = n_up >= n_lo
    target = np.zeros(n, dtype=np.int8)
    used_up = used_lo = 0
    # Bars spent approaching each boundary. The floor of 3 matters: the pull
    # closes CHANNEL_TOUCH_PULL of the remaining gap per bar, so from mid-
    # channel a single bar only reaches 0.225 * width -- outside the 0.18 *
    # width tolerance. Short patterns would otherwise never register a touch.
    run = max(int(n * 0.025), 3)
    for k, s in enumerate(slots):
        want_up = (k % 2 == 0) == start_up
        if want_up and used_up < n_up:
            target[s:s + run] = 1
            used_up += 1
        elif not want_up and used_lo < n_lo:
            target[s:s + run] = -1
            used_lo += 1
        elif used_up < n_up:
            target[s:s + run] = 1
            used_up += 1
        elif used_lo < n_lo:
            target[s:s + run] = -1
            used_lo += 1
    return target[:n], used_up, used_lo


def _count_touches(closes: np.ndarray, lower: np.ndarray, upper: np.ndarray
                   ) -> tuple:
    """
    Count distinct visits to each boundary.

    A visit is a maximal run of bars within CHANNEL_TOUCH_TOL_FRAC of the
    channel width of that boundary; consecutive bars spent hugging a line
    count once, so a single long drift along support is not mistaken for
    several separate touches.
    """
    width = np.maximum(upper - lower, 1e-9)
    tol   = width * CHANNEL_TOUCH_TOL_FRAC
    near_up = closes >= upper - tol
    near_lo = closes <= lower + tol

    def runs(mask):
        return int(np.sum(mask.astype(np.int8) -
                          np.concatenate([[0], mask.astype(np.int8)[:-1]]) == 1))

    return runs(near_up), runs(near_lo)


def _fat_walk(rng, n: int, sigma: float, fat_prob: float, fat_mult: float
              ) -> np.ndarray:
    """Fat-tailed random-walk steps with mild momentum (for padding)."""
    if n == 0:
        return np.array([])
    steps = np.where(
        rng.random(n) < fat_prob,
        rng.normal(0, sigma * fat_mult, n),
        rng.normal(0, sigma, n),
    )
    for k in range(1, n):
        steps[k] += 0.10 * steps[k - 1]
    return steps


def _bridge_prepad(rng, n: int, target: float, sigma: float,
                   fat_prob: float, fat_mult: float) -> np.ndarray:
    """Brownian-bridge noise that pins cleanly onto the pattern entry price."""
    if n == 0:
        return np.array([])
    start = target + rng.normal(0, sigma * max(np.sqrt(n), 1) * 0.4)
    trend = np.linspace(start, target, n)
    walk  = np.cumsum(_fat_walk(rng, n, sigma, fat_prob, fat_mult))
    return trend + (walk - np.linspace(0, walk[-1], n))


def _approach_prepad(rng, n: int, m_low: float, m_up: float,
                     low0: float, up0: float, entry: float, m_mid: float,
                     p: dict) -> np.ndarray:
    """
    Price approaching the formation from OUTSIDE its boundaries.

    The formation trendlines are extrapolated backwards over the approach.
    Price tracks just outside the relevant one -- below for a rising
    formation, above for a falling one, either side for a flat one -- closing
    that gap as it nears the start, so it crosses in where the formation
    begins.

    Total travel is capped at ENTRY_TRAVEL_CAP x the start width: without it a
    steep formation extrapolated back over 160 bars produces an approach that
    dominates the window and squashes the formation after normalisation.
    """
    if n == 0:
        return np.array([])

    k = np.arange(-n, 0, dtype=float)
    width0 = max(up0 - low0, 1e-9)
    side = ((1.0 if rng.random() < 0.5 else -1.0) if abs(m_mid) < 1e-9
            else (-1.0 if m_mid > 0 else 1.0))
    slope = m_low if side < 0 else m_up
    anchor = low0 if side < 0 else up0
    boundary = anchor + slope * k
    cap = ENTRY_TRAVEL_CAP * width0
    boundary = np.clip(boundary, anchor - cap, anchor + cap)

    d0 = rng.uniform(*ENTRY_OFFSET_RNG) * width0
    curve = rng.uniform(*ENTRY_CURVE_RNG)
    offs = d0 * np.linspace(1.0, 0.0, n) ** curve
    skel = boundary + side * offs
    skel = skel + (entry - skel[-1]) * np.linspace(0.0, 1.0, n)

    return _skeleton_closes(rng, skel, p['noise_sigma'], p['rev_strength'],
                            p['momentum_str'], p['vol_persist'],
                            p['fat_prob'], p['fat_mult'])


def _ohlc_from_closes(rng, closes: np.ndarray, noise_sigma: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opens/highs/lows around a close series: gaps, pin bars, variable wicks."""
    n         = len(closes)
    bar_sigma = noise_sigma * 0.50
    gap_prob  = rng.uniform(0.02, 0.05)
    pin_prob  = rng.uniform(0.05, 0.12)

    opens    = np.empty(n)
    opens[0] = closes[0] + rng.normal(0, bar_sigma * 0.30)
    gap_mult = np.where(rng.random(n - 1) < gap_prob,
                        rng.uniform(2.5, 5.0, n - 1), 1.0)
    opens[1:] = closes[:-1] + rng.normal(0, bar_sigma * 0.22, n - 1) * gap_mult

    highs = np.empty(n)
    lows  = np.empty(n)
    for i in range(n):
        o, c = opens[i], closes[i]
        body = abs(c - o)
        if rng.random() < pin_prob:
            long_w  = abs(rng.normal(body * 2.0, body * 0.8 + bar_sigma))
            short_w = abs(rng.normal(bar_sigma * 0.20, bar_sigma * 0.10))
            if rng.random() < 0.5:
                highs[i], lows[i] = max(o, c) + long_w,  min(o, c) - short_w
            else:
                highs[i], lows[i] = max(o, c) + short_w, min(o, c) - long_w
        else:
            highs[i] = max(o, c) + abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
            lows[i]  = min(o, c) - abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
    return opens, highs, lows


def _volume_profile(rng, n_pad_pre: int, n_pattern: int, n_break: int,
                    n_pad_post: int, fading: bool, spike_prob: float
                    ) -> np.ndarray:
    """
    Volume series across [pre-pad | pattern | breakout | post-pad].
    `fading` selects the classic declining-volume profile over the pattern;
    otherwise pattern volume is flat-noisy (decorrelates volume from label).
    """
    vols = np.empty(n_pad_pre + n_pattern + n_break + n_pad_post)
    i0 = 0
    for i in range(n_pad_pre):
        vols[i0 + i] = 0.65 + abs(rng.normal(0, 0.20))
    i0 += n_pad_pre
    for i in range(n_pattern):
        base  = (1.0 - 0.45 * (i / max(n_pattern, 1))) if fading \
                else rng.uniform(0.55, 0.85)
        noise = rng.normal(0, 0.18)
        spike = rng.uniform(1.6, 3.5) if rng.random() < spike_prob else 1.0
        vols[i0 + i] = max(0.05, (base + noise) * spike)
    i0 += n_pattern
    for i in range(n_break):
        vols[i0 + i] = (1.45 + abs(rng.normal(0, 0.35))) * \
                       max(0.40, 1.0 - 0.20 * i / max(n_break, 1))
    i0 += n_break
    for i in range(n_pad_post):
        vols[i0 + i] = 0.65 + abs(rng.normal(0, 0.20))
    return vols


def _assemble(rng, closes: np.ndarray, noise_sigma: float, vols: np.ndarray,
              lower_full: np.ndarray, upper_full: np.ndarray,
              segment: np.ndarray, label: int) -> pd.DataFrame:
    """OHLC synthesis + min-max normalisation + final DataFrame."""
    opens, highs, lows = _ohlc_from_closes(rng, closes, noise_sigma)

    p_all   = np.concatenate([opens, highs, lows, closes])
    p_min   = p_all.min()
    p_range = p_all.max() - p_min
    if p_range == 0:
        p_range = 1.0

    def norm(a):
        return ((a - p_min) / p_range)

    return pd.DataFrame({
        'bar':             np.arange(TOTAL_BARS),
        'open':            norm(opens).astype(np.float32),
        'high':            norm(highs).astype(np.float32),
        'low':             norm(lows).astype(np.float32),
        'close':           norm(closes).astype(np.float32),
        'volume':          (vols / vols.max()).astype(np.float32),
        'lower_trendline': np.where(np.isnan(lower_full), np.nan,
                                    norm(lower_full)).astype(np.float32),
        'upper_trendline': np.where(np.isnan(upper_full), np.nan,
                                    norm(upper_full)).astype(np.float32),
        'segment':         segment.astype(np.int8),
        'label':           np.full(TOTAL_BARS, label, dtype=np.int8),
    })


def _shared_realism_params(rng) -> dict:
    return dict(
        noise_sigma    = rng.uniform(*NOISE_SIGMA_RNG),
        rev_strength   = rng.uniform(0.08, 0.18),
        momentum_str   = rng.uniform(0.05, 0.20),
        vol_persist    = rng.uniform(0.40, 0.75),
        fat_prob       = rng.uniform(0.04, 0.10),
        fat_mult       = rng.uniform(2.5, 5.0),
        violation_prob = rng.uniform(0.05, 0.12),
        vol_spike_prob = rng.uniform(0.04, 0.10),
    )


# =============================================================================
# Anchored channel-shape generator (wedge / channel / megaphone share this)
# =============================================================================

def _anchored_pattern(dataset_idx: int, family: str) -> tuple[pd.DataFrame, dict]:
    """
    Generate a right-edge-anchored channel pattern:
      forming_wedge  converging  (width shrinks to (1-completion) of start)
      channel        parallel    (width ratio in CHANNEL_WIDTH_RATIO_RNG)
      megaphone      diverging   (width ratio in MEGAPHONE_WIDTH_RATIO_RNG)
    The pattern's last bar is always the window's last bar.
    """
    rng = np.random.RandomState(SEED_OFFSETS[family] + dataset_idx)
    p   = _shared_realism_params(rng)

    n_vis   = int(rng.randint(int(TOTAL_BARS * PATTERN_LEN_FRAC[0]),
                              int(TOTAL_BARS * PATTERN_LEN_FRAC[1]) + 1))
    pre_pad = TOTAL_BARS - n_vis

    w0    = rng.uniform(0.10, 0.30)
    m_mid = rng.uniform(-0.0035, 0.0035) * _SCALE

    completion = None
    if family == 'forming_wedge':
        completion = rng.uniform(*WEDGE_COMPLETION_RANGE)
        # keep the pinch above the noise floor: shrink completion if needed
        max_completion = 1.0 - max(MIN_END_WIDTH, 2.2 * p['noise_sigma']) / w0
        completion     = min(completion, max_completion)
        w_end = w0 * (1.0 - completion)
    elif family == 'channel':
        w_end = w0 * rng.uniform(*CHANNEL_WIDTH_RATIO_RNG)
    elif family == 'megaphone':
        w_end = w0 * rng.uniform(*MEGAPHONE_WIDTH_RATIO_RNG)
    else:
        raise ValueError(family)

    # Per-bar boundaries over the visible pattern
    t      = np.arange(n_vis, dtype=float)
    widths = np.linspace(w0, w_end, n_vis)
    mid0   = rng.uniform(0.25, 0.55)
    mids   = mid0 + m_mid * t
    lower  = mids - widths / 2.0
    upper  = mids + widths / 2.0

    # Channel positives must visibly touch both boundaries at least twice, or
    # the lines are not drawable (see CHANNEL_TOUCHES_REQUIRED). Retry with a
    # fresh touch schedule; keep the best attempt if none fully satisfies.
    # Touches are required of whichever family is the POSITIVE class: a
    # formation you cannot draw the lines from is not that formation. This was
    # channel-only until v3, which let wedge positives converge without price
    # ever riding either boundary -- the loophole that let the detector score
    # plain narrowing as a wedge.
    # v5: enforced for BOTH families whichever is positive. Scoping it to the
    # positive class left channel NEGATIVES with price wandering mid-channel,
    # so their price envelope narrowed even though the designed lines are
    # parallel -- visually and by fit, indistinguishable from a wedge. Measured
    # on v5: channel fitted compression median 0.546 against a wedge median of
    # 0.374, with 56% of channels fitting more converging than 0.6. That is
    # label noise teaching the model that wedge-shaped price action is
    # sometimes negative, and it is the likely source of v4's 18.2% channel
    # false-positive rate. The principle above applies to negatives too: a
    # channel you cannot draw the lines from is not a channel.
    enforce = family in ('channel', 'forming_wedge', 'megaphone')
    n_touch_up = n_touch_lo = None
    sig_pat = _vol_profile(rng, n_vis, p['noise_sigma'])
    if not enforce:
        closes_pat = _channel_closes(
            rng, n_vis, lower, upper, sig_pat, p['rev_strength'],
            p['momentum_str'], p['vol_persist'], p['fat_prob'], p['fat_mult'],
            p['violation_prob'],
        )
    else:
        best, best_score = None, -1
        for _ in range(CHANNEL_TOUCH_RETRIES):
            target, _, _ = _plan_touches(rng, n_vis)
            cand = _channel_closes(
                rng, n_vis, lower, upper, sig_pat, p['rev_strength'],
                p['momentum_str'], p['vol_persist'], p['fat_prob'],
                p['fat_mult'], p['violation_prob'], touch_target=target,
            )
            t_up, t_lo = _count_touches(cand, lower, upper)
            if t_up >= CHANNEL_TOUCHES_REQUIRED and t_lo >= CHANNEL_TOUCHES_REQUIRED:
                closes_pat, n_touch_up, n_touch_lo = cand, t_up, t_lo
                break
            score = min(t_up, CHANNEL_TOUCHES_REQUIRED) + \
                    min(t_lo, CHANNEL_TOUCHES_REQUIRED)
            if score > best_score:
                best, best_score = (cand, t_up, t_lo), score
        else:
            closes_pat, n_touch_up, n_touch_lo = best

    # The approach. With ENTRY_CONTEXT the window shows price arriving from
    # OUTSIDE the formation boundaries and crossing in where it begins;
    # without it, v2's context-free bridge.
    m_half_pre = (w_end - w0) / (2.0 * max(n_vis - 1, 1))
    if ENTRY_CONTEXT:
        pre = _approach_prepad(rng, pre_pad,
                               m_mid - m_half_pre, m_mid + m_half_pre,
                               lower[0], upper[0], closes_pat[0], m_mid, p)
    else:
        pre = _bridge_prepad(rng, pre_pad, closes_pat[0],
                             p['noise_sigma'] * rng.uniform(0.70, 1.20),
                             p['fat_prob'], p['fat_mult'])
    closes = np.concatenate([pre, closes_pat])

    # ~75% of wedges fade volume, ~40% of channel-family negatives do too
    fading = rng.random() < (0.75 if family == 'forming_wedge' else 0.40)
    vols   = _volume_profile(rng, pre_pad, n_vis, 0, 0, fading,
                             p['vol_spike_prob'])

    lower_full = np.full(TOTAL_BARS, np.nan)
    upper_full = np.full(TOTAL_BARS, np.nan)
    lower_full[pre_pad:] = lower
    upper_full[pre_pad:] = upper
    segment = np.zeros(TOTAL_BARS, dtype=np.int8)
    segment[pre_pad:] = 1
    # trendlines shift with price so a formation straddling the boundary
    # stays inside its own lines
    gap_pos, closes, lower_full, upper_full = _apply_gap(rng, closes,
                                                         lower_full, upper_full)

    label = 1 if family == POSITIVE_FAMILY else 0
    df = _assemble(rng, closes, p['noise_sigma'], vols,
                   lower_full, upper_full, segment, label)

    m_half = (w_end - w0) / (2.0 * max(n_vis - 1, 1))
    meta = {
        'dataset_idx': dataset_idx,
        'gap_pos': gap_pos, 'family': family, 'label': label,
        'total_bars': TOTAL_BARS, 'n_visible': n_vis, 'pre_pad': pre_pad,
        'm_mid': round(m_mid, 6),
        'm_lower': round(m_mid - m_half, 6), 'm_upper': round(m_mid + m_half, 6),
        'width_start': round(w0, 4), 'width_end': round(w_end, 4),
        'width_ratio': round(w_end / w0, 4),
        'completion': round(completion, 4) if completion is not None else None,
        'volume_fading': bool(fading),
        'noise_sigma': round(p['noise_sigma'], 4),
        'n_touch_upper': n_touch_up, 'n_touch_lower': n_touch_lo,
    }
    return df, meta


# =============================================================================
# Peak-family formations  (head and shoulders + its confusions)
# =============================================================================
#
# Landmarks: left shoulder, trough, HEAD, trough, right shoulder. The NECKLINE
# is the line through the two troughs and may slope. Anchoring follows the v2
# convention: the right shoulder peaks HS_TAIL_RNG bars before the right edge
# and price is falling away from it but has NOT yet broken the neckline, so a
# score means "an H&S has just completed, the break may be coming" rather than
# "one happened somewhere in this window".
#
# The exclusion band is built on HEAD PROMINENCE, because a head and shoulders
# is a triple top with a taller middle peak and that is the only thing telling
# them apart:
#     hs / inverse_hs : head exceeds the taller shoulder by >= HS_HEAD_EXCESS
#     triple_top      : all three peaks within +/- TRIPLE_TOP_TOL
# 0.20 vs 0.10 leaves an unambiguous gap, exactly as the width-ratio band does
# for wedge / channel / megaphone.
#
# inverse_hs is the vertical mirror. It is a POSITIVE for its own model and a
# hard negative for the standard one (and vice versa) -- the two are far more
# informative negatives for each other than any random walk.

HS_SHOULDER_HEIGHT_RNG = (0.10, 0.28)      # shoulder height above the neckline
# Head prominence above the TALLER shoulder. The floor is 0.30 rather than
# 0.20 because noise erodes it: at 0.20 the REALISED excess (measured on the
# generated series at the true landmarks) had a 5th percentile of just +0.079
# against a triple_top 95th percentile of +0.129, leaving 8% of positives
# inside the negatives' range. The corpus's whole premise is that no ambiguous
# case carries a hard label.
HS_HEAD_EXCESS_RNG     = (0.30, 0.85)      # head above the TALLER shoulder
TRIPLE_TOP_TOL         = 0.10              # all peaks within +/- this
HS_SHOULDER_ASYM_RNG   = (-0.15, 0.15)     # right vs left shoulder height
HS_NECK_SLOPE_RNG      = (-0.0022, 0.0022) # per bar, scaled by _SCALE
HS_TAIL_RNG            = (5, 30)           # bars after the right-shoulder peak
HS_TAIL_RETRACE_RNG    = (0.15, 0.70)      # fraction of the way back to neckline
HS_SEG_JITTER          = (0.75, 1.30)      # per-leg duration jitter
# Peak families need much stronger reversion than the channel families. There,
# weak reversion (0.08-0.18) is the point: price should wander between the
# bounds. Here the skeleton IS the pattern, so weak reversion let the realised
# path drift off it -- measured at only 48% of right shoulders landing within
# 10 bars of the anchor, and a median of 6 spurious peaks instead of 3. Noise
# must be texture on the shape, not a competing signal.
HS_REV_STRENGTH_RNG    = (0.34, 0.58)
HS_NOISE_SCALE         = 0.55              # shape-relative noise damping
HS_BREAK_DEPTH_RNG     = (0.35, 1.10)      # stale: break depth, x head height
HS_STALE_POST_RNG      = (0.18, 0.45)      # stale: post-break bars, frac of window


def _peak_skeleton(rng, n_vis: int, family: str):
    """
    Build the piecewise-linear price skeleton for a peak-family pattern plus
    the per-bar neckline. Returns (skeleton, neckline, meta) in normalised
    price units; the caller adds noise around the skeleton.

    Shape and orientation both come from _PEAK_SHAPE, so one construction
    covers tops and bottoms.
    """
    shape, orient = _PEAK_SHAPE[family]
    sign = float(orient)

    neck0    = rng.uniform(0.35, 0.55)
    n_slope  = rng.uniform(*HS_NECK_SLOPE_RNG) * _SCALE
    h_sh     = rng.uniform(*HS_SHOULDER_HEIGHT_RNG)

    if shape == 'hs':
        asym    = rng.uniform(*HS_SHOULDER_ASYM_RNG)
        h_ls    = h_sh
        h_rs    = h_sh * (1.0 + asym)
        excess  = rng.uniform(*HS_HEAD_EXCESS_RNG)
        h_head  = max(h_ls, h_rs) * (1.0 + excess)
    elif shape == 'triple':
        # three extremes of essentially equal size -- the hard negative
        h_ls   = h_sh
        h_head = h_sh * (1.0 + rng.uniform(-TRIPLE_TOP_TOL, TRIPLE_TOP_TOL))
        h_rs   = h_sh * (1.0 + rng.uniform(-TRIPLE_TOP_TOL, TRIPLE_TOP_TOL))
    elif shape == 'double':
        h_ls, h_head, h_rs = h_sh, None, h_sh * (1.0 + rng.uniform(-0.12, 0.12))
    else:
        raise ValueError(family)

    # ── leg durations ────────────────────────────────────────────────────────
    tail = int(rng.randint(*HS_TAIL_RNG))
    n_legs = 4 if h_head is not None else 2        # rises+falls before the RS
    body = n_vis - tail
    base = body / (n_legs + 1)
    segs = np.maximum(
        (base * rng.uniform(*HS_SEG_JITTER, size=n_legs + 1)).astype(int), 4)
    segs = (segs * (body / segs.sum())).astype(int)
    segs[-1] += body - segs.sum()

    # ── landmark bars and their prices ───────────────────────────────────────
    # Heights are measured ABOVE THE NECKLINE, so a sloping neckline can
    # cancel them out in absolute price: with a falling neckline the head can
    # end up LOWER than the left shoulder while still being "taller" by the
    # parameter. A head that is not the highest point is not a head and
    # shoulders, and the same effect turns triple tops into accidental H&S
    # shapes. So the defining property is enforced on the ABSOLUTE landmark
    # prices, shrinking the neckline slope until it holds.
    heights = ([h_ls, 0.0, h_head, 0.0, h_rs] if h_head is not None
               else [h_ls, 0.0, h_rs])

    def _landmarks(slope):
        t_, p_ = [0], [neck0 - sign * rng.uniform(0.0, 0.06)]
        cursor = 0
        for k, hgt in enumerate(heights):
            cursor += segs[k] if k < len(segs) else segs[-1]
            cursor = min(cursor, body)
            t_.append(cursor)
            p_.append(neck0 + slope * cursor + sign * hgt)
        return t_, p_

    for _ in range(12):
        t, p = _landmarks(n_slope)
        peaks = [p[1], p[3], p[5]] if h_head is not None else [p[1], p[3]]
        pk = [sign * v for v in peaks]           # orientation-normalised
        if shape == 'hs':
            # head must clear BOTH shoulders by a visible margin
            margin = 0.25 * min(h_ls, h_rs)
            if pk[1] - max(pk[0], pk[2]) >= margin:
                break
        elif shape == 'triple':
            # no extreme may stand out: absolute spread stays inside the band
            if (max(pk) - min(pk)) <= TRIPLE_TOP_TOL * 2.0 * h_sh:
                break
        else:                                    # double
            if abs(pk[0] - pk[1]) <= 0.25 * h_sh:
                break
        n_slope *= 0.55                          # flatten and retry
    else:
        n_slope = 0.0
        t, p = _landmarks(n_slope)

    # right shoulder is the last landmark; the tail falls back toward the neck
    t_rs, p_rs = t[-1], p[-1]
    neck_end   = neck0 + n_slope * (n_vis - 1)
    retrace    = rng.uniform(*HS_TAIL_RETRACE_RNG)
    p_end      = p_rs - sign * retrace * abs(p_rs - neck_end)
    t.append(n_vis - 1)
    p.append(p_end)

    skeleton = np.interp(np.arange(n_vis), t, p)
    neckline = neck0 + n_slope * np.arange(n_vis, dtype=float)

    meta = {
        'neck0': neck0, 'neck_slope': n_slope, 'sign': sign,
        'h_left': h_ls, 'h_head': h_head, 'h_right': h_rs,
        't_landmarks': [int(x) for x in t], 'tail_bars': tail,
        't_right_shoulder': int(t_rs), 'tail_retrace': retrace,
    }
    return skeleton, neckline, meta


def _skeleton_closes(rng, skeleton, noise_sigma, rev_strength, momentum_str,
                     vol_persist, fat_prob, fat_mult):
    """Mean-reverting noise around a price skeleton (peak families)."""
    n = len(skeleton)
    sig_arr = (np.full(n, noise_sigma) if np.isscalar(noise_sigma)
               else np.asarray(noise_sigma, dtype=float))
    closes = np.empty(n)
    closes[0] = skeleton[0]
    vol_t = float(sig_arr[0])
    for i in range(1, n):
        vol_t, step = _garch_step(rng, vol_t, sig_arr[i], vol_persist,
                                  fat_prob, fat_mult)
        mom = momentum_str * (closes[i-1] - closes[i-2]) if i >= 2 else 0.0
        rev = rev_strength * (skeleton[i-1] - closes[i-1])
        closes[i] = closes[i-1] + (skeleton[i] - skeleton[i-1]) + rev + mom + step
    return closes


def generate_peak_pattern(dataset_idx: int, family: str):
    """
    Head-and-shoulders family generator: hs, inverse_hs, triple_top,
    double_top, hs_stale. Right-edge anchored (see the notes above).
    """
    rng = np.random.RandomState(SEED_OFFSETS[family] + dataset_idx)
    p   = _shared_realism_params(rng)

    stale = family in ('hs_stale', 'inverse_hs_stale')
    # a stale sample is built as its live counterpart, then broken through
    build = {'hs_stale': 'hs', 'inverse_hs_stale': 'inverse_hs'}.get(family,
                                                                     family)

    n_vis   = int(rng.randint(int(TOTAL_BARS * 0.55), int(TOTAL_BARS * 0.92) + 1))
    if stale:
        # leave room for the break and the aftermath inside the window
        post  = int(n_vis * rng.uniform(*HS_STALE_POST_RNG))
        n_vis = max(n_vis - post, 60)
    pre_pad = TOTAL_BARS - n_vis - (post if stale else 0)

    skeleton, neckline, m = _peak_skeleton(rng, n_vis, build)
    hs_sigma = _vol_profile(rng, n_vis, p['noise_sigma'] * HS_NOISE_SCALE)
    closes_pat = _skeleton_closes(rng, skeleton, hs_sigma,
                                  rng.uniform(*HS_REV_STRENGTH_RNG),
                                  p['momentum_str'],
                                  p['vol_persist'], p['fat_prob'], p['fat_mult'])

    # A positive must NOT have broken the neckline yet: the whole point of the
    # anchor is that the break is still ahead. Clamp any noise excursion that
    # would have broken it early.
    if not stale:
        sign = m['sign']
        floor = neckline + sign * 0.012
        closes_pat = np.where(sign * (closes_pat - floor) < 0, floor, closes_pat)

    segment = np.zeros(TOTAL_BARS, dtype=np.int8)
    lower_full = np.full(TOTAL_BARS, np.nan)
    upper_full = np.full(TOTAL_BARS, np.nan)

    parts = []
    if pre_pad > 0:
        parts.append(_bridge_prepad(
            rng, pre_pad, closes_pat[0],
            p['noise_sigma'] * rng.uniform(0.70, 1.20),
            p['fat_prob'], p['fat_mult']))
    parts.append(closes_pat)

    n_break = 0
    if stale:
        # break through the neckline, then drift on the far side
        sign  = m['sign']
        depth = rng.uniform(*HS_BREAK_DEPTH_RNG) * (m['h_head'] or m['h_left'])
        target = neckline[-1] - sign * depth
        n_break = post
        brk = np.linspace(closes_pat[-1], target, n_break) + \
            _fat_walk(rng, n_break, p['noise_sigma'] * 0.9,
                      p['fat_prob'], p['fat_mult']).cumsum() * 0.35
        parts.append(brk)

    closes = np.concatenate(parts)
    closes = closes[:TOTAL_BARS] if len(closes) >= TOTAL_BARS else np.concatenate(
        [closes, np.full(TOTAL_BARS - len(closes), closes[-1])])

    segment[pre_pad:pre_pad + n_vis] = 1
    if n_break:
        segment[pre_pad + n_vis:pre_pad + n_vis + n_break] = 2
    # the neckline is the meaningful reference line for this family
    lower_full[pre_pad:pre_pad + n_vis] = neckline

    fading = rng.random() < (0.55 if _PEAK_SHAPE[build][0] == 'hs' else 0.40)
    vols   = _volume_profile(rng, pre_pad, n_vis, n_break,
                             TOTAL_BARS - pre_pad - n_vis - n_break,
                             fading, p['vol_spike_prob'])

    label = 1 if family == POSITIVE_FAMILY else 0
    gap_pos, closes, lower_full, upper_full = _apply_gap(rng, closes, lower_full, upper_full)
    df = _assemble(rng, closes, p['noise_sigma'], vols,
                   lower_full, upper_full, segment, label)

    meta = {
        'dataset_idx': dataset_idx,
        'gap_pos': gap_pos, 'family': family, 'label': label,
        'total_bars': TOTAL_BARS, 'n_visible': n_vis, 'pre_pad': pre_pad,
        'n_break': n_break,
        'neck_slope': round(m['neck_slope'], 6),
        'h_left':  round(m['h_left'], 4),
        'h_head':  None if m['h_head'] is None else round(m['h_head'], 4),
        'h_right': round(m['h_right'], 4),
        'head_excess': (None if m['h_head'] is None else
                        round(m['h_head'] / max(m['h_left'], m['h_right']) - 1, 4)),
        'shoulder_asym': round(abs(m['h_right'] - m['h_left'])
                               / max(m['h_left'], 1e-9), 4),
        'tail_bars': m['tail_bars'], 'orientation': int(m['sign']),
        'noise_sigma': round(p['noise_sigma'], 4),
        # pattern-relative landmark bars: start, LS, trough, HEAD, trough, RS,
        # right edge (no head entry for double_top). Exposed so validation can
        # check the realised series at the intended landmarks instead of
        # guessing their positions.
        't_landmarks': m['t_landmarks'],
    }
    return df, meta


# =============================================================================
# Stale wedge (completed + broken out, resolved well before the right edge)
# =============================================================================

def generate_stale_wedge(dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.RandomState(SEED_OFFSETS['stale_wedge'] + dataset_idx)
    p   = _shared_realism_params(rng)

    # Budget the window: pre-pad | wedge | breakout | post-pad(>=15%)
    # v5: pre_pad is drawn EXACTLY as the anchored families draw it
    # (PATTERN_LEN_FRAC), then wedge/breakout/post-pad are budgeted inside the
    # remaining n_vis. This family previously budgeted the other way round and
    # its formation could start as early as bar 12 (median 40) against bar 100+
    # for every other family. With gaps capped at bar 99 (GAP_MAX_POS) that made
    # stale_wedge the ONLY family able to contain a boundary -- measured at
    # 55.3% against 0.0% for forming_wedge, channel and megaphone. That is a
    # label cue teaching "discontinuity inside structured price = negative",
    # which would reinforce exactly the gap sensitivity the v4 post-mortem
    # found. Equal pre_pad across families removes it by construction.
    n_vis    = int(rng.randint(int(TOTAL_BARS * PATTERN_LEN_FRAC[0]),
                               int(TOTAL_BARS * PATTERN_LEN_FRAC[1]) + 1))
    pre_pad  = TOTAL_BARS - n_vis
    n_break  = max(3, int(n_vis * rng.uniform(0.04, 0.10)))
    post_pad = int(rng.randint(int(n_vis * 0.15), int(n_vis * 0.40) + 1))
    n_wedge  = n_vis - n_break - post_pad
    if n_wedge < 30:                       # keep the wedge itself legible
        post_pad = max(0, n_vis - n_break - 30)
        n_wedge  = n_vis - n_break - post_pad

    w0         = rng.uniform(0.10, 0.30)
    completion = rng.uniform(0.55, 0.85)
    w_end      = max(w0 * (1.0 - completion),
                     max(MIN_END_WIDTH, 2.2 * p['noise_sigma']) * 0.8)
    m_mid      = rng.uniform(-0.0035, 0.0035) * _SCALE

    t      = np.arange(n_wedge, dtype=float)
    widths = np.linspace(w0, w_end, n_wedge)
    mids   = rng.uniform(0.30, 0.60) + m_mid * t
    lower  = mids - widths / 2.0
    upper  = mids + widths / 2.0

    closes_w = _channel_closes(
        rng, n_wedge, lower, upper,
        _vol_profile(rng, n_wedge, p['noise_sigma']), p['rev_strength'],
        p['momentum_str'], p['vol_persist'], p['fat_prob'], p['fat_mult'],
        p['violation_prob'],
    )

    # Breakout: rising wedges break down, falling break up, flat either way
    if m_mid > 0.0005 * _SCALE:
        direction = -1
    elif m_mid < -0.0005 * _SCALE:
        direction = +1
    else:
        direction = -1 if rng.random() < 0.5 else +1
    depth    = rng.uniform(0.06, 0.18)
    closes_b = np.empty(n_break)
    prev     = closes_w[-1]
    line_end = (lower[-1] if direction < 0 else upper[-1])
    m_line   = (lower[-1] - lower[-2]) if direction < 0 else (upper[-1] - upper[-2])
    for i in range(n_break):
        target = line_end + m_line * (i + 1) + direction * depth * (i + 1) / n_break
        prev  += 0.22 * (target - prev) + rng.normal(0, p['noise_sigma']
                                                     * rng.uniform(0.6, 1.1))
        closes_b[i] = prev

    # A stale wedge was also entered from somewhere; leaving it the only
    # anchored family without an approach would reintroduce the leak
    # ENTRY_CONTEXT exists to close.
    if ENTRY_CONTEXT:
        m_half_pre = (w_end - w0) / (2.0 * max(n_wedge - 1, 1))
        pre = _approach_prepad(rng, pre_pad,
                               m_mid - m_half_pre, m_mid + m_half_pre,
                               lower[0], upper[0], closes_w[0], m_mid, p)
    else:
        pre = _bridge_prepad(rng, pre_pad, closes_w[0],
                             p['noise_sigma'] * rng.uniform(0.70, 1.20),
                             p['fat_prob'], p['fat_mult'])
    post = closes_b[-1] + np.cumsum(_fat_walk(rng, post_pad,
                                              p['noise_sigma'] * rng.uniform(0.70, 1.20),
                                              p['fat_prob'], p['fat_mult']))
    closes = np.concatenate([pre, closes_w, closes_b, post])

    vols = _volume_profile(rng, pre_pad, n_wedge, n_break, post_pad,
                           fading=True, spike_prob=p['vol_spike_prob'])

    lower_full = np.full(TOTAL_BARS, np.nan)
    upper_full = np.full(TOTAL_BARS, np.nan)
    lower_full[pre_pad:pre_pad + n_wedge] = lower
    upper_full[pre_pad:pre_pad + n_wedge] = upper
    segment = np.zeros(TOTAL_BARS, dtype=np.int8)
    segment[pre_pad:pre_pad + n_wedge] = 1
    segment[pre_pad + n_wedge:pre_pad + n_wedge + n_break] = 2

    gap_pos, closes, lower_full, upper_full = _apply_gap(rng, closes, lower_full, upper_full)
    df = _assemble(rng, closes, p['noise_sigma'], vols,
                   lower_full, upper_full, segment, label=0)
    meta = {
        'dataset_idx': dataset_idx,
        'gap_pos': gap_pos, 'family': 'stale_wedge', 'label': 0,
        'total_bars': TOTAL_BARS, 'n_wedge': n_wedge, 'n_break': n_break,
        'pre_pad': pre_pad, 'post_pad': post_pad,
        'm_mid': round(m_mid, 6), 'break_direction': direction,
        'width_ratio': round(w_end / w0, 4),
        'noise_sigma': round(p['noise_sigma'], 4),
    }
    return df, meta


# =============================================================================
# Random walk (v1-compatible dynamics, same seed offset)
# =============================================================================

def generate_compression_walk(dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    """
    A trending random walk whose VOLATILITY AND DRIFT decay through the window.

    The key negative in v3+. Scanning SPY 2008-2021 showed the v2 detector
    firing on the trading day rather than on geometry: detection rate tracked
    the intraday range-compression curve, peaking at 27.2% where windows are
    most compressed (session minute 210-240, ratio 0.599) and collapsing to
    0.12% at the open and close where windows widen. Per-bar range runs 0.1305
    at the open, troughs at 0.0756 near midday and rises to 0.1218 into the
    close, so an open-to-lunch window has wide bars early and narrow bars
    late, which fit_wedge_lines reads as a converging envelope.

    This family is that shape with NO formation: a walk that simply goes
    quiet, giving a converging fitted envelope while price wanders
    mid-channel instead of riding either boundary.

    Drift decays on the same envelope as volatility. Decaying volatility ALONE
    barely compresses the range because drift keeps carrying price; the real
    open-to-lunch confound loses both at once.
    """
    rng = np.random.RandomState(SEED_OFFSETS['compression_walk'] + dataset_idx)

    drift        = rng.uniform(-0.004, 0.004) * _SCALE
    base_sigma   = rng.uniform(*NOISE_SIGMA_RNG)
    fat_prob     = rng.uniform(0.04, 0.10)
    fat_mult     = rng.uniform(2.5, 5.0)
    momentum_str = rng.uniform(0.05, 0.25)
    vol_persist  = rng.uniform(0.45, 0.80)

    ratio = (rng.uniform(*VOL_DECAY_RNG) if rng.random() < VOL_DECAY_PROB
             else rng.uniform(*VOL_RISE_RNG))
    curve = rng.uniform(*VOL_CURVE_RNG)
    scale = np.linspace(1.0, 0.0, TOTAL_BARS) ** curve
    sigma_t = base_sigma * (ratio + (1.0 - ratio) * scale)
    drift_t = drift * (ratio + (1.0 - ratio) * scale)

    closes    = np.empty(TOTAL_BARS)
    closes[0] = rng.uniform(0.25, 0.75)
    vol_t     = float(sigma_t[0])
    momentum  = 0.0
    for i in range(1, TOTAL_BARS):
        vol_t, step = _garch_step(rng, vol_t, sigma_t[i], vol_persist,
                                  fat_prob, fat_mult)
        momentum = (momentum_str * (closes[i-1] - closes[i-2])
                    + 0.3 * momentum) if i >= 2 else 0.0
        closes[i] = closes[i-1] + drift_t[i] + momentum + step

    abs_moves = np.abs(np.diff(closes, prepend=closes[0]))
    vol_base  = rng.uniform(0.4, 0.9)
    vols      = vol_base + rng.uniform(0.3, 0.7) * (abs_moves / (abs_moves.max() + 1e-10))
    vols     += np.abs(rng.normal(0, vol_base * 0.25, TOTAL_BARS))
    vols      = np.abs(vols)

    gap_pos, closes = _apply_gap(rng, closes)
    df = _assemble(rng, closes, base_sigma, vols,
                   np.full(TOTAL_BARS, np.nan), np.full(TOTAL_BARS, np.nan),
                   np.zeros(TOTAL_BARS, dtype=np.int8), label=0)
    meta = {
        'dataset_idx': dataset_idx,
        'gap_pos': gap_pos, 'family': 'compression_walk', 'label': 0,
        'total_bars': TOTAL_BARS, 'vol_ratio': round(ratio, 4),
        'vol_curve': round(curve, 3), 'drift': round(drift, 6),
        'noise_sigma': round(base_sigma, 4),
    }
    return df, meta


def generate_walk(dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.RandomState(SEED_OFFSETS['walk'] + dataset_idx)

    regime = int(rng.randint(0, 3))
    drift  = [rng.uniform(-0.001, 0.001),
              rng.uniform(0.001, 0.005),
              rng.uniform(-0.005, -0.001)][regime]
    base_sigma   = rng.uniform(*NOISE_SIGMA_RNG)
    fat_prob     = rng.uniform(0.04, 0.10)
    fat_mult     = rng.uniform(2.5, 5.0)
    momentum_str = rng.uniform(0.05, 0.25)
    vol_persist  = rng.uniform(0.45, 0.80)

    sig_t     = _vol_profile(rng, TOTAL_BARS, base_sigma)
    closes    = np.empty(TOTAL_BARS)
    closes[0] = rng.uniform(0.20, 0.80)
    vol_t     = float(sig_t[0])
    momentum  = 0.0
    for i in range(1, TOTAL_BARS):
        vol_t, step = _garch_step(rng, vol_t, sig_t[i], vol_persist,
                                  fat_prob, fat_mult)
        momentum = (momentum_str * (closes[i-1] - closes[i-2])
                    + 0.3 * momentum) if i >= 2 else 0.0
        closes[i] = closes[i-1] + drift + momentum + step

    abs_moves = np.abs(np.diff(closes, prepend=closes[0]))
    vol_base  = rng.uniform(0.4, 0.9)
    vols      = vol_base + rng.uniform(0.3, 0.7) * (abs_moves / (abs_moves.max() + 1e-10))
    vols     += np.abs(rng.normal(0, vol_base * 0.25, TOTAL_BARS))
    vols      = np.abs(vols)

    gap_pos, closes = _apply_gap(rng, closes)
    df = _assemble(rng, closes, base_sigma, vols,
                   np.full(TOTAL_BARS, np.nan), np.full(TOTAL_BARS, np.nan),
                   np.zeros(TOTAL_BARS, dtype=np.int8), label=0)
    meta = {
        'dataset_idx': dataset_idx,
        'gap_pos': gap_pos, 'family': 'walk', 'label': 0,
        'total_bars': TOTAL_BARS,
        'regime': ['sideways', 'uptrend', 'downtrend'][regime],
        'drift': round(drift, 6), 'noise_sigma': round(base_sigma, 4),
    }
    return df, meta


# =============================================================================
# Family dispatch
# =============================================================================

def generate(family: str, dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    if family in ('forming_wedge', 'channel', 'megaphone'):
        return _anchored_pattern(dataset_idx, family)
    if family == 'stale_wedge':
        return generate_stale_wedge(dataset_idx)
    if family == 'walk':
        return generate_walk(dataset_idx)
    if family == 'compression_walk':
        return generate_compression_walk(dataset_idx)
    if family in PEAK_FAMILIES:
        return generate_peak_pattern(dataset_idx, family)
    raise ValueError(f'unknown family {family!r}')


# =============================================================================
# Corpus builder (multiprocess)
# =============================================================================

def _write_one(task: tuple[str, int, str, str]) -> None:
    family, original_idx, out_path, fmt = task
    df, _ = generate(family, original_idx)
    if fmt == 'parquet':
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


def generate_corpus(counts: dict[str, int], output_dir: str,
                    fmt: str = 'parquet', val_fraction: float = 0.10,
                    shuffle_seed: int = 42, workers: int = 4) -> None:
    root  = Path(output_dir)
    t_dir = root / 'training_data'
    v_dir = root / 'validation_data'
    t_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for family, n in counts.items():
        label = 1 if family == POSITIVE_FAMILY else 0
        entries += [{'label': label, 'type': family, 'original_idx': i}
                    for i in range(n)]
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(entries)

    n_total = len(entries)
    n_val   = int(n_total * val_fraction)
    tasks   = []
    for shuffled_idx, e in enumerate(entries):
        split = 'validation' if shuffled_idx < n_val else 'training'
        fname = f'dataset_{shuffled_idx:06d}.{fmt}'
        e.update(shuffled_idx=shuffled_idx, filename=fname, split=split)
        dest  = v_dir if split == 'validation' else t_dir
        tasks.append((e['type'], e['original_idx'], str(dest / fname), fmt))

    n_pos = sum(1 for e in entries if e['label'] == 1)
    print(f'Corpus v2 plan ({TOTAL_BARS}-bar windows, positive class = '
          f'{POSITIVE_FAMILY}): {n_total:,} datasets')
    for fam, n in counts.items():
        print(f'  {fam:<15} {n:>9,}')
    print(f'  positives {n_pos:,} ({n_pos/n_total*100:.1f}%)  |  '
          f'val split {n_val:,}  |  workers {workers}')

    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, _ in enumerate(pool.map(_write_one, tasks, chunksize=250), 1):
            if done % 20_000 == 0:
                rate = done / (time.time() - t0)
                eta  = (n_total - done) / rate / 60
                print(f'  {done:>8,}/{n_total:,}  ({rate:,.0f}/s  ETA {eta:.0f} min)',
                      flush=True)

    with open(root / 'corpus_manifest.json', 'w') as fh:
        json.dump(entries, fh)
    print(f'\nCorpus complete in {(time.time()-t0)/60:.1f} min '
          f'-> {root}  (manifest written)')


# =============================================================================
# Validation plots + parameter stats
# =============================================================================

def plot_family_examples(family: str, indices, save_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n * 2, 1, figsize=(16, 6.0 * n),
                             gridspec_kw={'height_ratios': [3, 1] * n,
                                          'hspace': 0.55})
    fig.patch.set_facecolor('#1a1a2e')

    for row_i, idx in enumerate(indices):
        df, meta = generate(family, idx)
        ax_p, ax_v = axes[row_i * 2], axes[row_i * 2 + 1]
        for ax in (ax_p, ax_v):
            ax.set_facecolor('#13131f')
            ax.tick_params(colors='#8a8aa0', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#333345')
            ax.grid(True, alpha=0.20, color='#2a2a3e')
            ax.set_xlim(-1, TOTAL_BARS)

        for _, row in df.iterrows():
            b, o, c = int(row['bar']), row['open'], row['close']
            col = '#26a69a' if c >= o else '#ef5350'
            ax_p.plot([b, b], [row['low'], row['high']], color=col, lw=0.7, zorder=2)
            ax_p.add_patch(patches.Rectangle(
                (b - 0.38, min(o, c)), 0.76, max(abs(c - o), 0.0015),
                facecolor=col, edgecolor=col, lw=0.3, zorder=3))

        bars = df['bar'].values
        ax_p.plot(bars, df['lower_trendline'].to_numpy(), '--',
                  color='#FF9800', lw=1.8, zorder=4)
        ax_p.plot(bars, df['upper_trendline'].to_numpy(), '--',
                  color='#42A5F5', lw=1.8, zorder=4)

        seg = df['segment'].values
        if (seg == 1).any():
            s1 = np.where(seg == 1)[0]
            ax_p.axvspan(s1[0] - 0.5, s1[-1] + 0.5, alpha=0.06, color='#5c6bc0', zorder=0)
        if (seg == 2).any():
            s2 = np.where(seg == 2)[0]
            ax_p.axvspan(s2[0] - 0.5, s2[-1] + 0.5, alpha=0.10, color='#ef5350', zorder=0)

        detail = ', '.join(f'{k}={v}' for k, v in meta.items()
                           if k not in ('dataset_idx', 'family', 'total_bars'))
        ax_p.set_title(f'{family} #{idx}  [label={meta["label"]}]  {detail}',
                       color='#d0d0e8', fontsize=8, pad=6)
        ax_v.bar(bars, df['volume'].values,
                 color=['#4caf50' if s == 0 else '#5c6bc0' if s == 1 else '#ef5350'
                        for s in seg], width=0.85, alpha=0.8)

    fig.suptitle(f'corpus v2 — {family}  ({TOTAL_BARS}-bar windows)',
                 color='#e8e8ff', fontsize=13, y=1.003)
    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Plot saved: {save_path}')


def print_stats(n_sample: int = 300) -> None:
    """Sample each family and report the achieved width-ratio distributions."""
    print(f'\nParameter check across {n_sample} samples/family '
          f'({TOTAL_BARS}-bar):')
    print(f'{"family":<15} {"width-ratio p5":>14} {"p50":>8} {"p95":>8}   '
          f'{"m_mid p5":>9} {"p95":>9}')
    for family in ('forming_wedge', 'channel', 'megaphone', 'stale_wedge'):
        wr, mm = [], []
        for i in range(n_sample):
            _, m = generate(family, i)
            wr.append(m.get('width_ratio'))
            mm.append(m.get('m_mid', 0.0))
        wr, mm = np.array(wr, dtype=float), np.array(mm, dtype=float)
        print(f'{family:<15} {np.percentile(wr,5):>14.3f} '
              f'{np.percentile(wr,50):>8.3f} {np.percentile(wr,95):>8.3f}   '
              f'{np.percentile(mm,5):>+9.5f} {np.percentile(mm,95):>+9.5f}')


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f'Wedge corpus generator v2 ({TOTAL_BARS}-bar windows)')
    parser.add_argument('--validate', action='store_true',
                        help='Render 3 examples per family + parameter stats')
    parser.add_argument('--corpus', action='store_true',
                        help='Generate the full mixed corpus')
    parser.add_argument('--n-wedge',     type=int, default=200_000)
    parser.add_argument('--n-walk',      type=int, default=150_000)
    parser.add_argument('--n-channel',   type=int, default=150_000)
    parser.add_argument('--n-megaphone', type=int, default=100_000)
    parser.add_argument('--n-stale',     type=int, default=100_000)
    parser.add_argument('--counts', default=None, metavar='FAM=N,FAM=N',
                        help='Explicit family counts, overriding the --n-* '
                             'flags entirely. Required for peak-family '
                             'corpora, which have more families than there '
                             'are flags. Known families: '
                             + ', '.join(sorted(SEED_OFFSETS)))
    parser.add_argument('--output-dir',  default=f'../runs_v2/window_{TOTAL_BARS}bar')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet')
    parser.add_argument('--val-fraction', type=float, default=0.10)
    parser.add_argument('--shuffle-seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plot-dir', default='.',
                        help='Directory for --validate output PNGs')
    args = parser.parse_args()

    if args.validate or not args.corpus:
        for family in ('forming_wedge', 'channel', 'megaphone',
                       'stale_wedge', 'walk'):
            plot_family_examples(
                family, [0, 1, 2],
                str(Path(args.plot_dir) / f'v2_examples_{family}.png'))
        print_stats()

    if args.corpus:
        if args.counts:
            counts = {}
            for spec in args.counts.split(','):
                fam, _, n = spec.partition('=')
                fam = fam.strip()
                if fam not in SEED_OFFSETS:
                    raise SystemExit(f'unknown family {fam!r} in --counts; '
                                     f'known: {", ".join(sorted(SEED_OFFSETS))}')
                counts[fam] = int(n)
            if POSITIVE_FAMILY not in counts:
                raise SystemExit(
                    f'--counts has no entry for the positive family '
                    f'{POSITIVE_FAMILY!r}; the corpus would have no positives')
        else:
            counts = {
                'forming_wedge': args.n_wedge,
                'walk':          args.n_walk,
                'channel':       args.n_channel,
                'megaphone':     args.n_megaphone,
                'stale_wedge':   args.n_stale,
            }
        generate_corpus(
            counts=counts,
            output_dir=args.output_dir,
            fmt=args.format,
            val_fraction=args.val_fraction,
            shuffle_seed=args.shuffle_seed,
            workers=args.workers,
        )


if __name__ == '__main__':
    main()
