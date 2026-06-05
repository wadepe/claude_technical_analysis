"""
generate_rising_wedge.py

Generates two types of simulated OHLCV stock datasets, both exactly
TOTAL_BARS (250) bars long and normalised to [0, 1]:

  label = 1  Rising-wedge pattern embedded at a random position within the
             250 bars; surrounding bars are realistic random-walk noise.

  label = 0  Pure random-walk noise with no embedded pattern.

A full corpus of 100,000 wedge + 400,000 noise datasets can be generated
and shuffled together for binary classification training.

Rising-wedge geometry
---------------------
  Lower trendline : steeper slope (m_lower)
  Upper trendline : shallower slope (m_upper < m_lower)
  Both slopes > 0 -- the entire channel drifts upward
  m_lower > m_upper -- lines converge (channel narrows)
  Breakout         -- close drops below lower trendline (bearish reversal)

Column schema (both dataset types)
------------------------------------
  bar, open, high, low, close   -- OHLC,   normalised [0, 1]
  volume                        -- normalised [0, 1]
  lower_trendline               -- normalised [0,1] inside wedge, NaN elsewhere
  upper_trendline               -- normalised [0,1] inside wedge, NaN elsewhere
  segment                       -- 0=noise padding, 1=wedge, 2=breakout
  label                         -- 1=rising wedge, 0=noise

Usage
-----
  # Plot 3 validation examples (wedge datasets, default indices 0 1 2)
  python generate_rising_wedge.py

  # Plot specific wedge indices
  python generate_rising_wedge.py --validate 0 500 99999

  # Generate 1,000 wedge datasets only
  python generate_rising_wedge.py --generate 1000

  # Generate the full mixed corpus (100k wedge + 400k noise, shuffled)
  python generate_rising_wedge.py --corpus

  # Custom corpus size
  python generate_rising_wedge.py --corpus --n-wedge 50000 --n-noise 200000

Requirements
------------
  pip install numpy pandas matplotlib pyarrow
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- saves to file without needing a display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TOTAL_BARS can be overridden via the WEDGE_TOTAL_BARS environment variable,
# allowing run_pipeline.py to test different time windows without editing this file.
TOTAL_BARS        = int(os.environ.get('WEDGE_TOTAL_BARS', '250'))
NOISE_SEED_OFFSET = 1_000_000  # separates noise seeds from wedge seeds (0-99999)

# Scaling factor relative to the 250-bar baseline.
# Slopes and bar-count ranges scale proportionally so wedge geometry looks
# similar regardless of window size.
_SCALE = 250.0 / TOTAL_BARS


# =============================================================================
# Rising-wedge generator  (label = 1)
# =============================================================================

def generate_rising_wedge(dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    """
    Generate a single rising-wedge dataset padded to TOTAL_BARS (250).

    The wedge formation (wedge + breakout) is placed at a random position
    within the 250 bars.  Surrounding bars are filled with a Brownian-bridge
    pre-noise (pins cleanly to wedge start) and a free random walk post-noise.

    Parameters
    ----------
    dataset_idx : int  -- unique index 0-99,999 (also the random seed)

    Returns
    -------
    df   : pd.DataFrame  (250 rows, see module docstring for columns)
    meta : dict          -- generation parameters
    """
    rng = np.random.RandomState(dataset_idx)

    # ── Pattern length — scaled proportionally to TOTAL_BARS ─────────────────
    # At 250 bars: wedge 40-160, break 10-40, max pattern 200 (80%), min pad 50 (20%)
    # At  50 bars: wedge  8-32,  break  2-8,  max pattern  40 (80%), min pad 10 (20%)
    _w_min    = max(8,  int(TOTAL_BARS * 0.16))
    _w_max    = max(16, int(TOTAL_BARS * 0.64))
    _b_min    = max(2,  int(TOTAL_BARS * 0.04))
    _b_max    = max(4,  int(TOTAL_BARS * 0.16))
    n_wedge   = int(rng.randint(_w_min, _w_max + 1))
    n_break   = int(rng.randint(_b_min, _b_max + 1))
    n_pattern = n_wedge + n_break

    # ── Random placement within 250 bars ─────────────────────────────────────
    max_pre  = TOTAL_BARS - n_pattern        # >= 50
    pre_pad  = int(rng.randint(0, max_pre + 1))
    post_pad = TOTAL_BARS - n_pattern - pre_pad

    wedge_start = pre_pad
    break_start = pre_pad + n_wedge

    # ── Trendline parameters (raw units, normalised at the end) ──────────────
    lower_start   = rng.uniform(0.10, 0.30)
    channel_start = rng.uniform(0.08, 0.25)
    upper_start   = lower_start + channel_start

    # Lower must be STEEPER than upper so the channel converges.
    # Slopes scale inversely with TOTAL_BARS so wedge geometry looks
    # similar across window sizes (same visual angle, same convergence rate).
    m_upper    = rng.uniform(0.0005 * _SCALE, 0.0040 * _SCALE)
    slope_diff = rng.uniform(0.0005 * _SCALE, 0.0030 * _SCALE)
    m_lower    = m_upper + slope_diff

    # Safety: guarantee channel stays open (> min_width) through all wedge bars
    min_end_width = 0.018
    if channel_start - slope_diff * n_wedge < min_end_width:
        slope_diff = (channel_start - min_end_width) / n_wedge
        m_lower    = m_upper + slope_diff

    # ── Noise / dynamics ─────────────────────────────────────────────────────
    noise_sigma    = rng.uniform(0.006, 0.022)
    rev_strength   = rng.uniform(0.08, 0.18)
    breakout_depth = rng.uniform(0.06, 0.18)
    pad_sigma      = noise_sigma * rng.uniform(0.70, 1.20)

    # Realism parameters (vary per dataset)
    fat_tail_prob  = rng.uniform(0.04, 0.10)   # prob of a 3-5x-sigma move each bar
    fat_tail_mult  = rng.uniform(2.5,  5.0)    # size multiplier for fat-tail bars
    violation_prob = rng.uniform(0.05, 0.12)   # prob price briefly breaches trendline
    momentum_str   = rng.uniform(0.05, 0.20)   # bar-to-bar direction persistence
    vol_persist    = rng.uniform(0.40, 0.75)   # volatility autocorrelation (GARCH-like)
    vol_spike_prob = rng.uniform(0.04, 0.10)   # prob of a random volume spike in wedge

    # ── Trendlines for wedge bars ─────────────────────────────────────────────
    t_w   = np.arange(n_wedge, dtype=float)
    lower = lower_start + m_lower * t_w
    upper = upper_start + m_upper * t_w
    mid   = (lower + upper) / 2.0

    # Extrapolated lower line for breakout bars
    t_b      = np.arange(n_wedge, n_pattern, dtype=float)
    lower_br = lower_start + m_lower * t_b

    # ── Core close prices (wedge + breakout) ─────────────────────────────────
    core_closes    = np.empty(n_pattern)
    core_closes[0] = lower_start + channel_start * rng.uniform(0.30, 0.65)
    avg_slope      = (m_lower + m_upper) / 2.0
    vol_t          = noise_sigma   # running volatility state

    for i in range(1, n_wedge):
        # GARCH-like volatility clustering
        vol_shock = rng.normal(0, noise_sigma * 0.20)
        vol_t     = noise_sigma * (1 - vol_persist) + vol_persist * vol_t + vol_shock
        vol_t     = np.clip(vol_t, noise_sigma * 0.25, noise_sigma * 3.5)

        # Fat-tailed step (occasional large moves)
        if rng.random() < fat_tail_prob:
            step = rng.normal(0.0, vol_t * fat_tail_mult)
        else:
            step = rng.normal(0.0, vol_t)

        # Momentum: slight persistence in direction
        momentum = momentum_str * (core_closes[i-1] - core_closes[i-2]) if i >= 2 else 0.0

        rev = rev_strength * (mid[i - 1] - core_closes[i - 1])
        core_closes[i] = core_closes[i - 1] + avg_slope + rev + momentum + step

        # Trendline boundaries: hard reject or allow occasional violation
        hi_bound = upper[i]
        lo_bound = lower[i]
        if core_closes[i] > hi_bound:
            if rng.random() < violation_prob:
                # Brief trendline violation — price pokes above upper line
                core_closes[i] = hi_bound + abs(rng.normal(0, noise_sigma * 0.40))
            else:
                core_closes[i] = hi_bound - abs(rng.normal(0, noise_sigma * 0.15))
        elif core_closes[i] < lo_bound:
            if rng.random() < violation_prob:
                # Brief trendline violation — price pokes below lower line
                core_closes[i] = lo_bound - abs(rng.normal(0, noise_sigma * 0.40))
            else:
                core_closes[i] = lo_bound + abs(rng.normal(0, noise_sigma * 0.15))

    # Breakout: choppy descent below the lower trendline
    for i in range(n_wedge, n_pattern):
        bi     = i - n_wedge
        target = lower_br[bi] - breakout_depth * (bi + 1) / n_break
        # Choppy: mix of mean-pull and random noise, with occasional bounces
        pull  = 0.22 * (target - core_closes[i - 1])
        noise = rng.normal(0.0, noise_sigma * rng.uniform(0.6, 1.1))
        core_closes[i] = core_closes[i - 1] + pull + noise

    # ── Padding noise (fat-tailed random walk for realistic texture) ──────────
    wedge_entry = core_closes[0]
    wedge_exit  = core_closes[-1]

    def _fat_walk(n: int, sigma: float) -> np.ndarray:
        """Random walk with fat tails and mild momentum."""
        if n == 0:
            return np.array([])
        steps = np.where(
            rng.random(n) < fat_tail_prob,
            rng.normal(0, sigma * fat_tail_mult, n),
            rng.normal(0, sigma, n),
        )
        # Mild momentum in padding too
        for k in range(1, n):
            steps[k] += 0.10 * steps[k - 1]
        return steps

    # Pre: Brownian bridge with fat-tailed noise
    if pre_pad > 0:
        start_price = wedge_entry + rng.normal(0, pad_sigma * max(np.sqrt(pre_pad), 1) * 0.4)
        trend       = np.linspace(start_price, wedge_entry, pre_pad)
        raw_walk    = np.cumsum(_fat_walk(pre_pad, pad_sigma))
        bridge      = raw_walk - np.linspace(0, raw_walk[-1], pre_pad)
        pre_closes  = trend + bridge
    else:
        pre_closes = np.array([])

    # Post: free fat-tailed walk from wedge_exit
    if post_pad > 0:
        post_closes = wedge_exit + np.cumsum(_fat_walk(post_pad, pad_sigma))
    else:
        post_closes = np.array([])

    # ── Combine all closes ────────────────────────────────────────────────────
    all_closes = np.concatenate([pre_closes, core_closes, post_closes])

    # ── OHLCV ────────────────────────────────────────────────────────────────
    bar_sigma   = noise_sigma * 0.50
    gap_prob    = rng.uniform(0.02, 0.05)   # prob of gap open vs prior close
    pin_prob    = rng.uniform(0.05, 0.12)   # prob of pin-bar / long-wick candle

    # Opens: mostly near prior close, occasional gap
    opens    = np.empty(TOTAL_BARS)
    opens[0] = all_closes[0] + rng.normal(0, bar_sigma * 0.30)
    gap_flags = rng.random(TOTAL_BARS - 1) < gap_prob
    gap_mult  = np.where(gap_flags, rng.uniform(2.5, 5.0, TOTAL_BARS - 1), 1.0)
    opens[1:] = all_closes[:-1] + rng.normal(0, bar_sigma * 0.22, TOTAL_BARS - 1) * gap_mult

    # Highs and lows: mix of normal candles and pin bars
    highs = np.empty(TOTAL_BARS)
    lows  = np.empty(TOTAL_BARS)
    for i in range(TOTAL_BARS):
        o, c   = opens[i], all_closes[i]
        body   = abs(c - o)
        if rng.random() < pin_prob:
            # Pin bar: one very long wick, short other side
            long_wick  = abs(rng.normal(body * 2.0, body * 0.8 + bar_sigma))
            short_wick = abs(rng.normal(bar_sigma * 0.20, bar_sigma * 0.10))
            if rng.random() < 0.5:   # upper pin
                highs[i] = max(o, c) + long_wick
                lows[i]  = min(o, c) - short_wick
            else:                    # lower pin
                highs[i] = max(o, c) + short_wick
                lows[i]  = min(o, c) - long_wick
        else:
            # Normal candle with variable wick size
            wick_up   = abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
            wick_down = abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
            highs[i]  = max(o, c) + wick_up
            lows[i]   = min(o, c) - wick_down

    # Volume: noisy declining trend in wedge + random spikes + realistic elsewhere
    vols = np.empty(TOTAL_BARS)
    for i in range(pre_pad):
        vols[i] = 0.65 + abs(rng.normal(0, 0.20))
    for i in range(n_wedge):
        # Base declining trend with extra noise and occasional spikes
        fade       = 1.0 - 0.45 * (i / n_wedge)
        vol_noise  = rng.normal(0, 0.18)
        spike_mult = rng.uniform(1.6, 3.5) if rng.random() < vol_spike_prob else 1.0
        vols[wedge_start + i] = max(0.05, (fade + vol_noise) * spike_mult)
    for i in range(n_break):
        # Breakout: elevated volume, choppy (not perfectly smooth)
        spike = 1.45 + abs(rng.normal(0, 0.35))
        vols[break_start + i] = spike * max(0.40, 1.0 - 0.20 * i / n_break)
    post_start = break_start + n_break
    for i in range(post_pad):
        vols[post_start + i] = 0.65 + abs(rng.normal(0, 0.20))

    # ── Normalise ─────────────────────────────────────────────────────────────
    p_all   = np.concatenate([opens, highs, lows, all_closes])
    p_min   = p_all.min()
    p_range = p_all.max() - p_min

    def norm(arr: np.ndarray) -> np.ndarray:
        return (arr - p_min) / p_range

    lower_full = np.full(TOTAL_BARS, np.nan)
    upper_full = np.full(TOTAL_BARS, np.nan)
    lower_full[wedge_start:wedge_start + n_wedge] = lower
    upper_full[wedge_start:wedge_start + n_wedge] = upper
    lower_norm = np.where(np.isnan(lower_full), np.nan, norm(lower_full))
    upper_norm = np.where(np.isnan(upper_full), np.nan, norm(upper_full))

    segment = np.zeros(TOTAL_BARS, dtype=np.int8)
    segment[wedge_start : wedge_start + n_wedge] = 1
    segment[break_start : break_start + n_break] = 2

    df = pd.DataFrame({
        "bar":             np.arange(TOTAL_BARS),
        "open":            norm(opens),
        "high":            norm(highs),
        "low":             norm(lows),
        "close":           norm(all_closes),
        "volume":          vols / vols.max(),
        "lower_trendline": lower_norm,
        "upper_trendline": upper_norm,
        "segment":         segment,
        "label":           np.ones(TOTAL_BARS, dtype=np.int8),
    })

    channel_end = channel_start - slope_diff * n_wedge
    meta = {
        "dataset_idx":     dataset_idx,
        "label":           1,
        "total_bars":      TOTAL_BARS,
        "n_wedge_bars":    n_wedge,
        "n_breakout_bars": n_break,
        "pre_pad_bars":    pre_pad,
        "post_pad_bars":   post_pad,
        "wedge_start_bar": wedge_start,
        "break_start_bar": break_start,
        "m_lower":         round(m_lower, 6),
        "m_upper":         round(m_upper, 6),
        "channel_start":   round(channel_start, 4),
        "channel_end":     round(channel_end, 4),
        "noise_sigma":     round(noise_sigma, 4),
        "breakout_depth":  round(breakout_depth, 4),
    }
    return df, meta


# =============================================================================
# Pure-noise generator  (label = 0)
# =============================================================================

def generate_noise_dataset(dataset_idx: int) -> tuple[pd.DataFrame, dict]:
    """
    Generate a 250-bar pure random-walk dataset with no embedded pattern.

    Uses seed = NOISE_SEED_OFFSET + dataset_idx to guarantee no overlap with
    the wedge seed space.

    Parameters
    ----------
    dataset_idx : int  -- unique index 0-399,999

    Returns
    -------
    df   : pd.DataFrame  (250 rows, same column schema as rising-wedge)
    meta : dict
    """
    rng = np.random.RandomState(NOISE_SEED_OFFSET + dataset_idx)

    # ── Market regime ─────────────────────────────────────────────────────────
    regime = int(rng.randint(0, 3))           # 0=sideways, 1=uptrend, 2=downtrend
    drifts = [
        rng.uniform(-0.001, 0.001),           # sideways
        rng.uniform(0.001, 0.005),            # uptrend
        rng.uniform(-0.005, -0.001),          # downtrend
    ]
    drift      = drifts[regime]
    base_sigma = rng.uniform(0.006, 0.022)

    # Realism parameters
    fat_tail_prob = rng.uniform(0.04, 0.10)
    fat_tail_mult = rng.uniform(2.5, 5.0)
    momentum_str  = rng.uniform(0.05, 0.25)   # persistence of direction
    vol_persist   = rng.uniform(0.45, 0.80)   # GARCH vol autocorrelation

    # ── Simulate closes: fat tails + momentum + vol clustering ───────────────
    closes    = np.empty(TOTAL_BARS)
    closes[0] = rng.uniform(0.20, 0.80)
    vol_t     = base_sigma
    momentum  = 0.0

    for i in range(1, TOTAL_BARS):
        # Volatility clustering
        vol_shock = rng.normal(0, base_sigma * 0.20)
        vol_t     = base_sigma * (1 - vol_persist) + vol_persist * vol_t + vol_shock
        vol_t     = np.clip(vol_t, base_sigma * 0.25, base_sigma * 4.0)

        # Fat-tailed price step
        if rng.random() < fat_tail_prob:
            step = rng.normal(0.0, vol_t * fat_tail_mult)
        else:
            step = rng.normal(0.0, vol_t)

        # Momentum: slight persistence in direction
        momentum = momentum_str * (closes[i-1] - closes[i-2]) + 0.3 * momentum if i >= 2 else 0.0

        closes[i] = closes[i - 1] + drift + momentum + step

    # ── OHLCV: gap opens, pin bars, vol correlated with price moves ───────────
    bar_sigma = base_sigma * 0.50
    gap_prob  = rng.uniform(0.02, 0.05)
    pin_prob  = rng.uniform(0.05, 0.12)

    opens    = np.empty(TOTAL_BARS)
    opens[0] = closes[0] + rng.normal(0, bar_sigma * 0.30)
    gap_flags = rng.random(TOTAL_BARS - 1) < gap_prob
    gap_mult  = np.where(gap_flags, rng.uniform(2.5, 5.0, TOTAL_BARS - 1), 1.0)
    opens[1:] = closes[:-1] + rng.normal(0, bar_sigma * 0.22, TOTAL_BARS - 1) * gap_mult

    highs = np.empty(TOTAL_BARS)
    lows  = np.empty(TOTAL_BARS)
    for i in range(TOTAL_BARS):
        o, c  = opens[i], closes[i]
        body  = abs(c - o)
        if rng.random() < pin_prob:
            long_wick  = abs(rng.normal(body * 2.0, body * 0.8 + bar_sigma))
            short_wick = abs(rng.normal(bar_sigma * 0.20, bar_sigma * 0.10))
            if rng.random() < 0.5:
                highs[i] = max(o, c) + long_wick
                lows[i]  = min(o, c) - short_wick
            else:
                highs[i] = max(o, c) + short_wick
                lows[i]  = min(o, c) - long_wick
        else:
            wick_up   = abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
            wick_down = abs(rng.normal(bar_sigma * 0.55, bar_sigma * 0.35))
            highs[i]  = max(o, c) + wick_up
            lows[i]   = min(o, c) - wick_down

    # Volume: correlated with absolute price moves (realistic market behaviour)
    abs_moves  = np.abs(np.diff(closes, prepend=closes[0]))
    vol_base   = rng.uniform(0.4, 0.9)
    vol_corr   = rng.uniform(0.3, 0.7)   # how much vol tracks price moves
    vols       = vol_base + vol_corr * (abs_moves / (abs_moves.max() + 1e-10))
    vols      += np.abs(rng.normal(0, vol_base * 0.25, TOTAL_BARS))
    vols       = np.abs(vols)

    # ── Normalise ─────────────────────────────────────────────────────────────
    p_all   = np.concatenate([opens, highs, lows, closes])
    p_min   = p_all.min()
    p_range = p_all.max() - p_min
    if p_range == 0:                          # degenerate edge case
        p_range = 1.0

    def norm(arr: np.ndarray) -> np.ndarray:
        return (arr - p_min) / p_range

    df = pd.DataFrame({
        "bar":             np.arange(TOTAL_BARS),
        "open":            norm(opens),
        "high":            norm(highs),
        "low":             norm(lows),
        "close":           norm(closes),
        "volume":          vols / vols.max(),
        "lower_trendline": np.full(TOTAL_BARS, np.nan),
        "upper_trendline": np.full(TOTAL_BARS, np.nan),
        "segment":         np.zeros(TOTAL_BARS, dtype=np.int8),
        "label":           np.zeros(TOTAL_BARS, dtype=np.int8),
    })

    regime_names = {0: "sideways", 1: "uptrend", 2: "downtrend"}
    meta = {
        "dataset_idx": dataset_idx,
        "label":       0,
        "total_bars":  TOTAL_BARS,
        "regime":      regime_names[regime],
        "drift":       round(drift, 6),
        "noise_sigma": round(base_sigma, 4),
    }
    return df, meta


# =============================================================================
# Batch generators
# =============================================================================

def generate_wedge_batch(
    n_datasets: int,
    output_dir: str = "..",
    fmt: str = "parquet",
    val_fraction: float = 0.10,
    start_idx: int = 0,
) -> None:
    """Generate n_datasets rising-wedge datasets and save (wedge-only mode)."""
    n_datasets = min(n_datasets, 100_000)
    root  = Path(output_dir)
    t_dir = root / "training_data"
    v_dir = root / "validation_data"
    t_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)

    n_val     = int(n_datasets * val_fraction)
    meta_list = []

    for i in range(n_datasets):
        idx = start_idx + i
        df, meta = generate_rising_wedge(idx)
        dest  = v_dir if i < n_val else t_dir
        fname = f"rising_wedge_{idx:06d}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(dest / fname, index=False)
        else:
            df.to_csv(dest / fname, index=False)
        meta_list.append(meta)
        if (i + 1) % 1_000 == 0:
            print(f"  {i + 1:>7,} / {n_datasets:,} wedge datasets written ...")

    meta_path = root / "wedge_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta_list, fh, indent=2)

    n_train = n_datasets - n_val
    print(f"\nComplete. {n_datasets:,} wedge datasets saved.")
    print(f"  Training   ({n_train:,}) -> {t_dir}")
    print(f"  Validation ({n_val:,})   -> {v_dir}")
    print(f"  Metadata               -> {meta_path}")


def generate_full_corpus(
    n_wedge: int = 100_000,
    n_noise: int = 400_000,
    output_dir: str = "..",
    fmt: str = "parquet",
    val_fraction: float = 0.10,
    shuffle_seed: int = 42,
) -> None:
    """
    Generate n_wedge rising-wedge datasets and n_noise pure-noise datasets,
    shuffle them together, then write to training_data/ and validation_data/.

    Each saved file contains a 'label' column (1=wedge, 0=noise) and a
    'segment' column (0=noise padding, 1=wedge, 2=breakout).

    A corpus_manifest.json file is written to output_dir recording the label,
    type, original index, and train/val split for every file.

    Parameters
    ----------
    n_wedge      : Rising-wedge datasets to generate (max 100,000).
    n_noise      : Pure-noise datasets to generate (max 400,000).
    output_dir   : Project root directory.
    fmt          : "parquet" or "csv".
    val_fraction : Fraction held out for validation_data/.
    shuffle_seed : Seed used only for shuffling (not for data generation).
    """
    n_wedge = min(n_wedge, 100_000)
    n_noise = min(n_noise, 400_000)
    n_total = n_wedge + n_noise

    root  = Path(output_dir)
    t_dir = root / "training_data"
    v_dir = root / "validation_data"
    t_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)

    # ── Build shuffled manifest ───────────────────────────────────────────────
    entries = (
        [{"label": 1, "type": "wedge", "original_idx": i} for i in range(n_wedge)]
        + [{"label": 0, "type": "noise", "original_idx": i} for i in range(n_noise)]
    )
    rng_shuffle = np.random.RandomState(shuffle_seed)
    rng_shuffle.shuffle(entries)

    n_val = int(n_total * val_fraction)

    print(f"Corpus plan: {n_wedge:,} wedge + {n_noise:,} noise = {n_total:,} total")
    print(f"  Val split : {n_val:,}  |  Train: {n_total - n_val:,}")
    print(f"  Format    : {fmt}")
    print()

    # ── Generate and save ─────────────────────────────────────────────────────
    for shuffled_idx, entry in enumerate(entries):
        if entry["type"] == "wedge":
            df, _ = generate_rising_wedge(entry["original_idx"])
        else:
            df, _ = generate_noise_dataset(entry["original_idx"])

        split = "validation" if shuffled_idx < n_val else "training"
        dest  = v_dir if split == "validation" else t_dir
        fname = f"dataset_{shuffled_idx:06d}.{fmt}"

        if fmt == "parquet":
            df.to_parquet(dest / fname, index=False)
        else:
            df.to_csv(dest / fname, index=False)

        entry["shuffled_idx"] = shuffled_idx
        entry["filename"]     = fname
        entry["split"]        = split

        if (shuffled_idx + 1) % 10_000 == 0:
            pct = (shuffled_idx + 1) / n_total * 100
            print(f"  {shuffled_idx + 1:>7,} / {n_total:,}  ({pct:.1f}%)")

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_path = root / "corpus_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(entries, fh, indent=2)

    wedge_count = sum(1 for e in entries if e["label"] == 1)
    noise_count = n_total - wedge_count
    print(f"\nCorpus complete.")
    print(f"  Total     : {n_total:,}  (wedge={wedge_count:,}, noise={noise_count:,})")
    print(f"  Training  : {n_total - n_val:,}  -> {t_dir}")
    print(f"  Validation: {n_val:,}  -> {v_dir}")
    print(f"  Manifest  : {manifest_path}")


# =============================================================================
# Visualisation
# =============================================================================

_UP_COLOR   = "#26a69a"
_DOWN_COLOR = "#ef5350"
_BG_DARK    = "#13131f"
_FIG_BG     = "#1a1a2e"
_GRID_COLOR = "#2a2a3e"


def _draw_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        b = int(row["bar"])
        o, c = row["open"], row["close"]
        h, l = row["high"], row["low"]
        color = _UP_COLOR if c >= o else _DOWN_COLOR
        ax.plot([b, b], [l, h], color=color, lw=0.75, zorder=2)
        body_h = max(abs(c - o), 0.0015)
        ax.add_patch(patches.Rectangle(
            (b - 0.38, min(o, c)), 0.76, body_h,
            facecolor=color, edgecolor=color, lw=0.3, zorder=3,
        ))


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG_DARK)
    ax.tick_params(colors="#8a8aa0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333345")
    ax.grid(True, alpha=0.20, color=_GRID_COLOR)
    ax.set_xlim(-1, TOTAL_BARS)


def plot_examples(
    indices: list[int] = (0, 1, 2),
    save_path: str = "validation_examples.png",
) -> None:
    """Plot rising-wedge examples (full 250-bar view with annotated regions)."""
    n = len(indices)
    fig, axes = plt.subplots(
        n * 2, 1,
        figsize=(16, 6.5 * n),
        gridspec_kw={"height_ratios": [3, 1] * n, "hspace": 0.60},
    )
    fig.patch.set_facecolor(_FIG_BG)

    for row_i, idx in enumerate(indices):
        df, meta = generate_rising_wedge(idx)
        ws  = meta["wedge_start_bar"]
        bs  = meta["break_start_bar"]
        nw  = meta["n_wedge_bars"]
        nb  = meta["n_breakout_bars"]
        bars = df["bar"].values

        ax_p = axes[row_i * 2]
        ax_v = axes[row_i * 2 + 1]
        _style_ax(ax_p)
        _style_ax(ax_v)

        # Shade noise padding regions
        if ws > 0:
            ax_p.axvspan(-1, ws - 0.5, alpha=0.15, color="#2e7d32",
                         zorder=0, label="Noise padding")
        post_start = bs + nb
        if post_start < TOTAL_BARS:
            ax_p.axvspan(post_start - 0.5, TOTAL_BARS, alpha=0.15,
                         color="#2e7d32", zorder=0)

        ax_p.axvspan(bs - 0.5, bs + nb - 0.5, alpha=0.08,
                     color=_DOWN_COLOR, zorder=0, label="Breakout zone")

        _draw_candles(ax_p, df)

        ax_p.plot(bars, df["lower_trendline"], "--",
                  color="#FF9800", lw=2.0, zorder=4,
                  label=f"Lower (m={meta['m_lower']:.5f})")
        ax_p.plot(bars, df["upper_trendline"], "--",
                  color="#42A5F5", lw=2.0, zorder=4,
                  label=f"Upper (m={meta['m_upper']:.5f})")

        ax_p.axvline(ws - 0.5,        color="#2e7d32",   lw=1.2, ls=":", zorder=5)
        ax_p.axvline(bs - 0.5,        color=_DOWN_COLOR, lw=1.4, ls=":", zorder=5,
                     label="Breakout bar")
        ax_p.axvline(post_start - 0.5, color="#2e7d32",  lw=1.2, ls=":", zorder=5)

        ax_p.set_title(
            f"Dataset #{idx:,}  [WEDGE label=1]   |   "
            f"Pre: {meta['pre_pad_bars']}b  Wedge: {nw}b (bar {ws}-{bs-1})  "
            f"Breakout: {nb}b (bar {bs}-{bs+nb-1})  Post: {meta['post_pad_bars']}b   |   "
            f"Channel: {meta['channel_start']:.3f}->{meta['channel_end']:.3f}   "
            f"Noise: {meta['noise_sigma']:.3f}",
            color="#d0d0e8", fontsize=8.5, pad=6,
        )
        ax_p.set_ylabel("Normalised Price", color="#8a8aa0", fontsize=8)
        ax_p.legend(loc="upper left", fontsize=7.5, facecolor="#222233",
                    labelcolor="#d0d0e8", framealpha=0.75, ncol=3)

        seg = df["segment"].values
        vol = df["volume"].values
        ax_v.bar(bars[seg == 0], vol[seg == 0], color="#4caf50", width=0.85,
                 alpha=0.70, label="Vol (noise)")
        ax_v.bar(bars[seg == 1], vol[seg == 1], color="#5c6bc0", width=0.85,
                 alpha=0.90, label="Vol (wedge)")
        ax_v.bar(bars[seg == 2], vol[seg == 2], color=_DOWN_COLOR, width=0.85,
                 alpha=0.85, label="Vol (breakout)")
        ax_v.set_ylabel("Norm. Volume", color="#8a8aa0", fontsize=8)
        ax_v.set_xlabel("Bar index (0-249)", color="#8a8aa0", fontsize=8)
        ax_v.legend(fontsize=7.5, facecolor="#222233", labelcolor="#d0d0e8",
                    framealpha=0.75, ncol=3)

    fig.suptitle(
        f"Rising Wedge -- Validation Examples  ({TOTAL_BARS}-bar padded datasets)",
        color="#e8e8ff", fontsize=13, y=1.005,
    )
    out = Path(save_path)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved: {out.resolve()}")


def plot_noise_examples(
    indices: list[int] = (0, 1, 2),
    save_path: str = "validation_noise_examples.png",
) -> None:
    """Plot pure-noise examples for visual validation."""
    n = len(indices)
    fig, axes = plt.subplots(
        n * 2, 1,
        figsize=(16, 6.5 * n),
        gridspec_kw={"height_ratios": [3, 1] * n, "hspace": 0.60},
    )
    fig.patch.set_facecolor(_FIG_BG)

    for row_i, idx in enumerate(indices):
        df, meta = generate_noise_dataset(idx)
        bars = df["bar"].values

        ax_p = axes[row_i * 2]
        ax_v = axes[row_i * 2 + 1]
        _style_ax(ax_p)
        _style_ax(ax_v)

        _draw_candles(ax_p, df)
        ax_p.set_title(
            f"Noise Dataset #{idx:,}  [label=0]   |   "
            f"Regime: {meta['regime']}   |   "
            f"Drift: {meta['drift']:+.5f}   |   "
            f"Noise: {meta['noise_sigma']:.4f}",
            color="#d0d0e8", fontsize=9, pad=6,
        )
        ax_p.set_ylabel("Normalised Price", color="#8a8aa0", fontsize=8)

        ax_v.bar(bars, df["volume"].values, color="#78909c", width=0.85, alpha=0.85)
        ax_v.set_ylabel("Norm. Volume", color="#8a8aa0", fontsize=8)
        ax_v.set_xlabel("Bar index (0-249)", color="#8a8aa0", fontsize=8)

    fig.suptitle(
        f"Pure Noise -- Validation Examples  ({TOTAL_BARS}-bar datasets, label=0)",
        color="#e8e8ff", fontsize=13, y=1.005,
    )
    out = Path(save_path)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved: {out.resolve()}")


def print_dataset_summary(indices: list[int], kind: str = "wedge") -> None:
    header = (
        f"{'Idx':>7}  {'Pre':>4}  {'Wdg':>4}  {'Brk':>4}  {'Post':>4}  "
        f"{'WdgBar':>6}  {'BrkBar':>6}  "
        f"{'m_lower':>8}  {'m_upper':>8}  {'Ch.end':>6}  {'Noise':>6}"
    ) if kind == "wedge" else (
        f"{'Idx':>7}  {'Regime':>10}  {'Drift':>8}  {'Noise':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for idx in indices:
        if kind == "wedge":
            _, meta = generate_rising_wedge(idx)
            print(
                f"{meta['dataset_idx']:>7,}  "
                f"{meta['pre_pad_bars']:>4}  "
                f"{meta['n_wedge_bars']:>4}  "
                f"{meta['n_breakout_bars']:>4}  "
                f"{meta['post_pad_bars']:>4}  "
                f"{meta['wedge_start_bar']:>6}  "
                f"{meta['break_start_bar']:>6}  "
                f"{meta['m_lower']:>8.5f}  "
                f"{meta['m_upper']:>8.5f}  "
                f"{meta['channel_end']:>6.4f}  "
                f"{meta['noise_sigma']:>6.4f}"
            )
        else:
            _, meta = generate_noise_dataset(idx)
            print(
                f"{meta['dataset_idx']:>7,}  "
                f"{meta['regime']:>10}  "
                f"{meta['drift']:>+8.5f}  "
                f"{meta['noise_sigma']:>6.4f}"
            )
    print()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Rising Wedge + Noise Data Generator  "
            f"({TOTAL_BARS}-bar normalised OHLCV datasets)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Validation
    parser.add_argument("--validate", nargs="+", type=int, default=[0, 1, 2],
                        metavar="IDX",
                        help="Wedge dataset indices to plot (default: 0 1 2)")
    parser.add_argument("--validate-noise", nargs="+", type=int, default=None,
                        metavar="IDX",
                        help="Noise dataset indices to plot for validation")

    # Wedge-only batch
    parser.add_argument("--generate", type=int, default=0, metavar="N",
                        help="Generate N wedge-only datasets (max 100,000)")

    # Full mixed corpus
    parser.add_argument("--corpus", action="store_true",
                        help="Generate the full mixed corpus (wedge + noise, shuffled)")
    parser.add_argument("--n-wedge", type=int, default=100_000, metavar="N",
                        help="Wedge datasets in corpus (default: 100,000)")
    parser.add_argument("--n-noise", type=int, default=400_000, metavar="N",
                        help="Noise datasets in corpus (default: 400,000)")
    parser.add_argument("--shuffle-seed", type=int, default=42, metavar="S",
                        help="Seed for corpus shuffle (default: 42)")

    # Shared
    parser.add_argument("--output-dir", default="..", metavar="DIR",
                        help="Project root directory (default: parent of source_code/)")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                        help="File format (default: parquet)")
    parser.add_argument("--val-fraction", type=float, default=0.10, metavar="F",
                        help="Validation split fraction (default: 0.10)")
    parser.add_argument("--start-idx", type=int, default=0, metavar="IDX",
                        help="Starting index for wedge-only --generate runs")
    parser.add_argument("--save-plots-to", default="validation_examples.png",
                        metavar="PATH",
                        help="Output path for wedge validation plot")

    args = parser.parse_args()

    # Always show wedge validation plots
    print(f"Wedge validation plots for indices: {args.validate} ...")
    print_dataset_summary(args.validate, kind="wedge")
    plot_examples(indices=args.validate, save_path=args.save_plots_to)

    # Optional noise validation plots
    if args.validate_noise:
        print(f"Noise validation plots for indices: {args.validate_noise} ...")
        print_dataset_summary(args.validate_noise, kind="noise")
        plot_noise_examples(indices=args.validate_noise,
                            save_path="validation_noise_examples.png")

    # Wedge-only batch
    if args.generate > 0:
        n = min(args.generate, 100_000)
        print(f"\nGenerating {n:,} wedge datasets ...")
        generate_wedge_batch(
            n_datasets=n,
            output_dir=args.output_dir,
            fmt=args.format,
            val_fraction=args.val_fraction,
            start_idx=args.start_idx,
        )

    # Full mixed corpus
    if args.corpus:
        print(f"\nGenerating full corpus ...")
        generate_full_corpus(
            n_wedge=args.n_wedge,
            n_noise=args.n_noise,
            output_dir=args.output_dir,
            fmt=args.format,
            val_fraction=args.val_fraction,
            shuffle_seed=args.shuffle_seed,
        )


if __name__ == "__main__":
    main()
