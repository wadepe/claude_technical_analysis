"""
scan_bitcoin.py

Slides a 250-bar rolling window over the Bitcoin 1-minute OHLCV data,
normalises each window to match the training-data scale, runs inference
with the trained 1D CNN, then prints and plots the 5 highest-confidence
non-overlapping rising-wedge detections.

Normalisation applied per-window
---------------------------------
  Prices (OHLC) : (x - window_min) / (window_max - window_min)
  Volume        : v / window_volume_max
This mirrors exactly what generate_rising_wedge.py does for each dataset.

Usage
-----
  python scan_bitcoin.py
  python scan_bitcoin.py --stride 1      # full scan (slow, ~17 min on CPU)
  python scan_bitcoin.py --stride 5      # ~3 min
  python scan_bitcoin.py --threshold 0.9 # raise bar for detection
  python scan_bitcoin.py --weights ../models/cnn_final.weights.h5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ── path bootstrap so we can import cnn_model from the same folder ────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import build_model, N_BARS

# =============================================================================
# Config
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BTC_CSV      = PROJECT_ROOT / 'reference_material' / 'btc_data_bi_min.csv'
WEIGHTS_PATH = PROJECT_ROOT / 'models' / 'cnn_best.weights.h5'
OUTPUT_DIR   = PROJECT_ROOT / 'models' / 'bitcoin_scan'

PRICE_COLS   = ['open', 'high', 'low', 'close']
VOL_COL      = 'volume'
DATE_COL     = 'date'
FEATURE_COLS = PRICE_COLS + [VOL_COL]   # order must match training


# =============================================================================
# Data loading
# =============================================================================

def load_btc(path: Path) -> pd.DataFrame:
    print(f'Loading {path.name} ...')
    df = pd.read_csv(path, usecols=[DATE_COL] + FEATURE_COLS)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f'  {len(df):,} rows  |  '
          f'{df[DATE_COL].iloc[0]}  to  {df[DATE_COL].iloc[-1]}')
    return df


# =============================================================================
# Per-window normalisation  (matches training pipeline exactly)
# =============================================================================

def normalise_batch(raw: np.ndarray) -> np.ndarray:
    """
    Normalise a batch of raw OHLCV windows.

    Parameters
    ----------
    raw : float32 array  (batch, N_BARS, 5)  — raw OHLCV values

    Returns
    -------
    float32 array  (batch, N_BARS, 5)  — each window independently in [0,1]
    """
    n = raw.shape[0]

    # ── OHLC normalisation ───────────────────────────────────────────────────
    prices = raw[:, :, :4]                               # (n, N_BARS, 4)
    flat   = prices.reshape(n, -1)                       # (n, N_BARS*4)
    p_min  = flat.min(axis=1).reshape(n, 1, 1)
    p_rng  = (flat.max(axis=1) - flat.min(axis=1)).reshape(n, 1, 1)
    p_rng  = np.where(p_rng == 0, 1.0, p_rng)
    prices_n = (prices - p_min) / p_rng

    # ── Volume normalisation ─────────────────────────────────────────────────
    vol     = raw[:, :, 4]                               # (n, N_BARS)
    vol_max = vol.max(axis=1, keepdims=True)
    vol_max = np.where(vol_max == 0, 1.0, vol_max)
    vol_n   = vol / vol_max

    return np.concatenate(
        [prices_n, vol_n[:, :, np.newaxis]], axis=2
    ).astype(np.float32)


# =============================================================================
# Rolling scan
# =============================================================================

def scan(
    df: pd.DataFrame,
    model: keras.Model,
    threshold: float,
    stride: int,
    batch_size: int,
) -> list[dict]:
    """
    Slide a 250-bar window over the data and return every window where
    the model predicts P(wedge) >= threshold.
    """
    data  = df[FEATURE_COLS].values.astype(np.float32)   # (n_rows, 5)
    dates = df[DATE_COL].values

    n_rows    = len(data)
    all_starts = np.arange(0, n_rows - N_BARS + 1, stride)
    n_windows  = len(all_starts)

    print(f'\nScanning {n_windows:,} windows  '
          f'(stride={stride}, ~{N_BARS*stride/60:.0f}-min gaps between windows)')

    positives: list[dict] = []
    t0 = time.time()

    for batch_i in range(0, n_windows, batch_size):
        starts = all_starts[batch_i : batch_i + batch_size]   # (b,)

        # Vectorised window construction — advanced numpy indexing, no Python loop
        idx = starts[:, None] + np.arange(N_BARS)[None, :]    # (b, N_BARS)
        raw = data[idx]                                        # (b, N_BARS, 5)

        # Per-window normalisation
        X = normalise_batch(raw)

        # Inference
        scores = model.predict(X, batch_size=512, verbose=0).squeeze()
        if scores.ndim == 0:          # single-element batch
            scores = scores.reshape(1)

        # Collect hits
        hit_mask = scores >= threshold
        for local_i in np.where(hit_mask)[0]:
            s = int(starts[local_i])
            positives.append({
                'start_idx' : s,
                'end_idx'   : s + N_BARS - 1,
                'score'     : float(scores[local_i]),
                'date_start': pd.Timestamp(dates[s]),
                'date_end'  : pd.Timestamp(dates[s + N_BARS - 1]),
            })

        # Progress every ~10%
        done = batch_i + len(starts)
        if (done // (n_windows // 10 + 1)) != ((done - len(starts)) // (n_windows // 10 + 1)):
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (n_windows - done) / rate
            print(f'  {done:>8,}/{n_windows:,}  '
                  f'({done/n_windows*100:.0f}%)  '
                  f'hits: {len(positives):,}  '
                  f'ETA: {eta/60:.1f} min')

    elapsed = time.time() - t0
    print(f'\nScan complete in {elapsed/60:.1f} min  '
          f'—  {len(positives):,} positive windows found')
    return positives


# =============================================================================
# Non-overlapping selection
# =============================================================================

def pick_nonoverlapping(positives: list[dict], n: int = 5) -> list[dict]:
    """
    Select n non-overlapping windows from the positive detections.
    Strategy: sort by confidence (highest first), greedily accept if no
    bar overlap with any already-accepted window.
    """
    if not positives:
        return []

    ranked = sorted(positives, key=lambda p: p['score'], reverse=True)
    chosen: list[dict] = []

    for p in ranked:
        no_overlap = all(
            p['end_idx'] < s['start_idx'] or p['start_idx'] > s['end_idx']
            for s in chosen
        )
        if no_overlap:
            chosen.append(p)
        if len(chosen) == n:
            break

    # Return in chronological order
    chosen.sort(key=lambda p: p['start_idx'])
    return chosen


# =============================================================================
# Plotting
# =============================================================================

_FIG_BG = '#1a1a2e'
_AX_BG  = '#13131f'
_GRID   = '#2a2a3e'
_SPINE  = '#333345'
_MUTED  = '#8a8aa0'
_TEXT   = '#d0d0e8'
_UP     = '#26a69a'
_DOWN   = '#ef5350'
_BLUE   = '#5c6bc0'


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.20, color=_GRID)


def plot_detections(df: pd.DataFrame, detections: list[dict], out_dir: Path) -> None:
    """
    Plot each detected window as a full candlestick chart with volume panel.
    Uses raw (unnormalised) USD prices so the scale is human-readable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(detections)

    fig, axes = plt.subplots(
        n * 2, 1,
        figsize=(16, 6.5 * n),
        gridspec_kw={'height_ratios': [3, 1] * n, 'hspace': 0.65},
    )
    fig.patch.set_facecolor(_FIG_BG)

    for row_i, win in enumerate(detections):
        seg  = df.iloc[win['start_idx'] : win['end_idx'] + 1].reset_index(drop=True)
        bars = seg.index.values

        ax_p = axes[row_i * 2]
        ax_v = axes[row_i * 2 + 1]
        _style(ax_p)
        _style(ax_v)

        # ── Candlesticks (raw prices) ─────────────────────────────────────────
        for i, row in seg.iterrows():
            o, c = row['open'], row['close']
            h, l = row['high'], row['low']
            col = _UP if c >= o else _DOWN
            ax_p.plot([i, i], [l, h], color=col, lw=0.75, zorder=2)
            body_h = max(abs(c - o), (h - l) * 0.005)
            ax_p.add_patch(patches.Rectangle(
                (i - 0.38, min(o, c)), 0.76, body_h,
                facecolor=col, edgecolor=col, lw=0.3, zorder=3,
            ))

        ax_p.set_title(
            f'Detection #{row_i + 1}  |  '
            f'Score: {win["score"]:.4f}  |  '
            f'{win["date_start"].strftime("%Y-%m-%d %H:%M")}  to  '
            f'{win["date_end"].strftime("%Y-%m-%d %H:%M")}  '
            f'({N_BARS} bars @ 1-min)',
            color=_TEXT, fontsize=9, pad=6,
        )
        ax_p.set_ylabel('BTC/USD', color=_MUTED, fontsize=8)
        ax_p.set_xlim(-1, N_BARS)

        # ── Volume ────────────────────────────────────────────────────────────
        ax_v.bar(bars, seg[VOL_COL].values, color=_BLUE, width=0.85, alpha=0.85)
        ax_v.set_ylabel('Volume (BTC)', color=_MUTED, fontsize=8)
        ax_v.set_xlabel('Bar offset (1 bar = 1 minute)', color=_MUTED, fontsize=8)
        ax_v.set_xlim(-1, N_BARS)
        _style(ax_v)

    fig.suptitle(
        f'Bitcoin 1-min — Rising Wedge Detections  '
        f'(1D CNN, threshold={args_threshold:.2f})',
        color='#e8e8ff', fontsize=13, y=1.005,
    )

    out_path = out_dir / 'btc_rising_wedge_detections.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nChart saved: {out_path.resolve()}')


# module-level var so plot_detections can access threshold for the title
args_threshold: float = 0.5


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    global args_threshold

    parser = argparse.ArgumentParser(
        description='Scan Bitcoin 1-min data for rising wedge patterns'
    )
    parser.add_argument('--stride',    type=int,   default=10,
                        help='Step size between windows in bars (default: 10)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Minimum model score to count as detection (default: 0.5)')
    parser.add_argument('--batch-size',type=int,   default=5_000,
                        help='Windows per inference batch (default: 5,000)')
    parser.add_argument('--top-n',     type=int,   default=5,
                        help='Non-overlapping detections to display (default: 5)')
    parser.add_argument('--weights',    default=str(WEIGHTS_PATH),
                        help='Path to model weights file')
    parser.add_argument('--csv',        default=str(BTC_CSV),
                        help='Path to Bitcoin CSV file')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR),
                        help='Directory for output charts')
    args = parser.parse_args()

    args_threshold = args.threshold

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_btc(Path(args.csv))

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'\nBuilding model and loading weights from:\n  {args.weights}')
    model = build_model(print_summary=False)
    model.load_weights(args.weights)
    print('Ready.\n')

    # ── Scan ──────────────────────────────────────────────────────────────────
    positives = scan(
        df       = df,
        model    = model,
        threshold= args.threshold,
        stride   = args.stride,
        batch_size = args.batch_size,
    )

    if not positives:
        print('No detections above threshold. Try lowering --threshold.')
        return

    # ── Select top-N non-overlapping ──────────────────────────────────────────
    selected = pick_nonoverlapping(positives, n=args.top_n)

    if len(selected) < args.top_n:
        print(f'Warning: only {len(selected)} non-overlapping detections found '
              f'(requested {args.top_n}).')

    # ── Print results ─────────────────────────────────────────────────────────
    print(f'\n{"="*64}')
    print(f'TOP {len(selected)} NON-OVERLAPPING RISING WEDGE DETECTIONS')
    print(f'(selected by highest confidence, displayed chronologically)')
    print(f'{"="*64}')
    for i, win in enumerate(selected, 1):
        duration_h = N_BARS / 60
        print(f'\n  #{i}')
        print(f'    Score      : {win["score"]:.6f}')
        print(f'    Start      : {win["date_start"]}  (row {win["start_idx"]:,})')
        print(f'    End        : {win["date_end"]}  (row {win["end_idx"]:,})')
        print(f'    Duration   : {N_BARS} bars = {duration_h:.1f} hours of 1-min data')
    print(f'\n{"="*64}')
    print(f'\nTotal positives across full scan : {len(positives):,}')
    print(f'Stride used                      : {args.stride} bar(s)')
    print(f'Threshold                        : {args.threshold}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_detections(df, selected, Path(args.output_dir))


if __name__ == '__main__':
    main()
