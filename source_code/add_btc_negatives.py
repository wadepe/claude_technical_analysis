"""
add_btc_negatives.py

Extracts real Bitcoin 1-minute OHLCV windows as hard negative (label=0)
training examples and shuffles them into the existing corpus.

Why "confirmed not wedge"
--------------------------
A rising wedge structurally requires ALL of:
  1. Both the high-series and low-series slope upward (ascending channel)
  2. The channel converges  (high-slope > low-slope)
  3. Price stays bounded inside the channel throughout

We confirm a window is NOT a rising wedge by checking the negation of
those conditions.  Any window that fails at least one condition is kept.
In practice >95% of BTC windows fail at least one condition, so very
few genuine wedge signals are discarded.  The small contamination risk
is far outweighed by having realistic market data as negative examples.

What gets added
---------------
  - btc_neg_XXXXXX.parquet  saved to training_data/ (90%) or validation_data/ (10%)
  - Same 10-column schema as the synthetic datasets  (label=0, all segment=0, trendlines=NaN)
  - corpus_manifest.json   updated with new entries, then fully reshuffled
  - numpy_cache/           deleted  (train_cnn.py will rebuild on next run)

Usage
-----
  python add_btc_negatives.py                   # ~41K windows, stride=50
  python add_btc_negatives.py --stride 25       # ~82K windows (more coverage)
  python add_btc_negatives.py --stride 250      # ~8K fully non-overlapping
  python add_btc_negatives.py --no-confirm-check  # skip structural filter
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
BTC_CSV       = _PROJECT_ROOT / 'reference_material' / 'btc_data_bi_min.csv'

N_BARS        = int(os.environ.get('WEDGE_TOTAL_BARS', '250'))
FEATURE_COLS  = ['open', 'high', 'low', 'close', 'volume']
DATE_COL      = 'date'


# =============================================================================
# Rising-wedge exclusion check
# =============================================================================

def _linslope(y: np.ndarray) -> float:
    """Least-squares slope of a 1-D series (fast, no scipy needed)."""
    n = len(y)
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    return float(np.dot(x - xm, y - ym) / (np.dot(x - xm, x - xm) + 1e-12))


def is_confirmed_not_wedge(raw: np.ndarray) -> bool:
    """
    Return True if the window structurally cannot be a rising wedge.

    A window is confirmed NOT a wedge if it fails any of:
      - Both high-series and low-series slope upward
      - High-series slope > low-series slope  (converging)
      - Price stays within ±30% of the linear channel (bounded)

    One failed condition = confirmed non-wedge.
    """
    highs  = raw[:, 1].astype(np.float64)   # column 1 = high
    lows   = raw[:, 2].astype(np.float64)   # column 2 = low
    closes = raw[:, 3].astype(np.float64)   # column 3 = close

    slope_h = _linslope(highs)
    slope_l = _linslope(lows)

    # Condition 1: both trendlines ascending
    if not (slope_h > 0 and slope_l > 0):
        return True

    # Condition 2: channel converging (high rises faster than low)
    if not (slope_h > slope_l):
        return True

    # Condition 3: price bounded within the linear channel
    n = N_BARS
    t = np.arange(n, dtype=np.float64)
    upper_fit = highs[0]  + slope_h * t
    lower_fit = lows[0]   + slope_l * t
    width     = upper_fit - lower_fit + 1e-10
    relative_pos = (closes - lower_fit) / width
    if relative_pos.min() < -0.30 or relative_pos.max() > 1.30:
        return True

    # All conditions met — could be a wedge; exclude from negatives
    return False


# =============================================================================
# Per-window normalisation  (must match training pipeline exactly)
# =============================================================================

def normalise_window(raw: np.ndarray) -> np.ndarray:
    """
    Normalise a single (N_BARS, 5) OHLCV window to [0, 1].
    Prices share one min/max; volume is normalised independently.
    """
    prices  = raw[:, :4]
    p_min   = prices.min()
    p_range = prices.max() - p_min
    if p_range == 0:
        p_range = 1.0
    prices_n = (prices - p_min) / p_range

    vol     = raw[:, 4]
    vol_max = vol.max()
    if vol_max == 0:
        vol_max = 1.0
    vol_n = vol / vol_max

    return np.column_stack([prices_n, vol_n]).astype(np.float32)


# =============================================================================
# Build parquet DataFrame
# =============================================================================

def to_dataframe(norm: np.ndarray) -> pd.DataFrame:
    """Convert a normalised (N_BARS, 5) array to the corpus parquet schema."""
    return pd.DataFrame({
        'bar':             np.arange(N_BARS, dtype=np.int32),
        'open':            norm[:, 0],
        'high':            norm[:, 1],
        'low':             norm[:, 2],
        'close':           norm[:, 3],
        'volume':          norm[:, 4],
        'lower_trendline': np.full(N_BARS, np.nan, dtype=np.float32),
        'upper_trendline': np.full(N_BARS, np.nan, dtype=np.float32),
        'segment':         np.zeros(N_BARS, dtype=np.int8),
        'label':           np.zeros(N_BARS, dtype=np.int8),
    })


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Add real BTC windows as hard negative training examples'
    )
    parser.add_argument('--data-dir',     default=None,
                        help='Run directory containing corpus_manifest.json '
                             '(default: project root)')
    parser.add_argument('--stride',       type=int,   default=50,
                        help='Bars between window starts (default: 50)')
    parser.add_argument('--val-fraction', type=float, default=0.10,
                        help='Fraction saved to validation_data/ (default: 0.10)')
    parser.add_argument('--shuffle-seed', type=int,   default=2025,
                        help='RNG seed for shuffling (default: 2025)')
    parser.add_argument('--no-confirm-check', action='store_true',
                        help='Skip structural non-wedge confirmation filter')
    args = parser.parse_args()

    root = Path(args.data_dir) if args.data_dir else _PROJECT_ROOT
    t_dir = root / 'training_data'
    v_dir = root / 'validation_data'
    t_dir.mkdir(exist_ok=True)
    v_dir.mkdir(exist_ok=True)

    # ── Load BTC data ─────────────────────────────────────────────────────────
    print(f'Loading {BTC_CSV.name} ...')
    df = pd.read_csv(BTC_CSV, usecols=[DATE_COL] + FEATURE_COLS)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df.sort_values(DATE_COL, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f'  {len(df):,} rows  |  {df[DATE_COL].iloc[0]}  to  {df[DATE_COL].iloc[-1]}')

    data  = df[FEATURE_COLS].values.astype(np.float32)
    dates = df[DATE_COL].values
    n_rows = len(data)

    all_starts  = np.arange(0, n_rows - N_BARS + 1, args.stride)
    n_candidate = len(all_starts)
    print(f'  {n_candidate:,} candidate windows at stride={args.stride}')

    # ── Structural filter pass ────────────────────────────────────────────────
    if args.no_confirm_check:
        confirmed_starts = all_starts
        print(f'  Structural filter: OFF  ({len(confirmed_starts):,} windows kept)')
    else:
        print(f'  Applying structural non-wedge confirmation filter ...')
        confirmed_starts = []
        n_excluded = 0
        for i, s in enumerate(all_starts):
            raw = data[s : s + N_BARS]
            if is_confirmed_not_wedge(raw):
                confirmed_starts.append(s)
            else:
                n_excluded += 1
            if (i + 1) % 50_000 == 0:
                print(f'    {i+1:>7,}/{n_candidate:,}  '
                      f'kept: {len(confirmed_starts):,}  excluded: {n_excluded:,}')
        confirmed_starts = np.array(confirmed_starts, dtype=np.int64)
        print(f'  Filter complete: {len(confirmed_starts):,} confirmed non-wedge windows '
              f'({n_excluded:,} excluded as potential wedges)')

    n_windows = len(confirmed_starts)
    n_val     = int(n_windows * args.val_fraction)

    # ── Shuffle window order before saving ────────────────────────────────────
    rng         = np.random.RandomState(args.shuffle_seed)
    perm        = rng.permutation(n_windows)
    save_order  = confirmed_starts[perm]   # window starts in shuffled order

    # ── Save parquet files ────────────────────────────────────────────────────
    print(f'\nSaving {n_windows:,} BTC negative windows to parquet ...')
    new_entries: list[dict] = []
    t0 = time.time()

    for file_i, s in enumerate(save_order):
        s = int(s)
        raw  = data[s : s + N_BARS]
        norm = normalise_window(raw)
        df_w = to_dataframe(norm)

        split    = 'validation' if file_i < n_val else 'training'
        dest_dir = v_dir if split == 'validation' else t_dir
        fname    = f'btc_neg_{file_i:06d}.parquet'
        df_w.to_parquet(dest_dir / fname, index=False)

        new_entries.append({
            'label':        0,
            'type':         'btc_negative',
            'original_idx': int(perm[file_i]),
            'filename':     fname,
            'split':        split,
            'date_start':   str(pd.Timestamp(dates[s])),
            'date_end':     str(pd.Timestamp(dates[s + N_BARS - 1])),
        })

        if (file_i + 1) % 5_000 == 0:
            elapsed = time.time() - t0
            rate    = (file_i + 1) / elapsed
            eta     = (n_windows - file_i - 1) / rate
            print(f'  {file_i+1:>6,}/{n_windows:,}  '
                  f'({rate:.0f} files/s  ETA {eta/60:.1f} min)')

    elapsed = time.time() - t0
    print(f'  Saved {n_windows:,} files in {elapsed/60:.1f} min  '
          f'({n_windows/elapsed:.0f} files/s)')

    # ── Update corpus manifest ─────────────────────────────────────────────────
    manifest_path = root / 'corpus_manifest.json'
    print(f'\nUpdating {manifest_path.name} ...')
    with open(manifest_path) as fh:
        manifest: list[dict] = json.load(fh)

    n_original = len(manifest)
    manifest.extend(new_entries)
    n_combined = len(manifest)

    # Reshuffle ALL entries and reassign shuffled_idx
    rng2  = np.random.RandomState(args.shuffle_seed + 1)
    order = rng2.permutation(n_combined)
    for new_idx, old_pos in enumerate(order):
        manifest[old_pos]['shuffled_idx'] = int(new_idx)
    manifest.sort(key=lambda e: e['shuffled_idx'])

    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    wedge_n   = sum(1 for e in manifest if e.get('label') == 1)
    btc_n     = sum(1 for e in manifest if e.get('type') == 'btc_negative')
    synth_n   = sum(1 for e in manifest if e.get('type') == 'noise')
    total_neg = n_combined - wedge_n

    print(f'\n{"="*56}')
    print(f'CORPUS UPDATE COMPLETE')
    print(f'{"="*56}')
    print(f'  Original entries        : {n_original:,}')
    print(f'  New BTC negatives added : {n_windows:,}')
    print(f'  Combined manifest total : {n_combined:,}')
    print(f'  ---')
    print(f'  Wedge    (label=1)      : {wedge_n:,}')
    print(f'  Noise    (label=0)      : {total_neg:,}')
    print(f'    Synthetic noise       : {synth_n:,}')
    print(f'    Real BTC windows      : {btc_n:,}')
    print(f'  ---')
    print(f'  Wedge fraction          : {wedge_n/n_combined*100:.1f}%')
    print(f'{"="*56}')

    # ── Clear stale numpy cache ───────────────────────────────────────────────
    cache_dir = root / 'numpy_cache'
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f'\nDeleted {cache_dir.name}/  (will rebuild on next training run)')

    print(f'\nNext step — retrain:')
    print(f'  python train_cnn.py --data-dir ..')


if __name__ == '__main__':
    main()
