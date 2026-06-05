"""
compare_windows.py

Runs the BTC scan for multiple trained models (different window sizes),
filters each to non-overlapping detections, backtests both sets, and
prints a side-by-side comparison table.

Usage
-----
  python compare_windows.py
  python compare_windows.py --threshold 0.5 --windows 50 250
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
BTC_CSV      = PROJECT_ROOT / 'reference_material' / 'btc_data_bi_min.csv'

HORIZONS       = [30, 60, 120, 240]
HORIZON_LABELS = ['30m', '1hr', '2hr', '4hr']


# =============================================================================
# Per-window-size helpers
# =============================================================================

def load_model_for_window(window: int):
    """Import cnn_model with WEDGE_TOTAL_BARS set, build and load weights."""
    os.environ['WEDGE_TOTAL_BARS'] = str(window)

    # Force reimport of cnn_model so N_BARS picks up the new env var
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('cnn_model',):
            del sys.modules[mod_name]

    from cnn_model import build_model, N_BARS

    # Prefer runs/window_{N}bar; fall back to legacy models/
    candidates = [
        PROJECT_ROOT / 'runs' / f'window_{window}bar' / 'models' / 'cnn_best.weights.h5',
        PROJECT_ROOT / 'models' / 'cnn_best.weights.h5',
    ]
    weights_path = next((p for p in candidates if p.exists()), None)
    if weights_path is None:
        raise FileNotFoundError(
            f'No weights found for window={window}. '
            f'Run: python run_pipeline.py --window-size {window}'
        )

    model = build_model(print_summary=False)
    model.load_weights(str(weights_path))
    print(f'  [{window}-bar]  N_BARS={N_BARS}  weights={weights_path}')
    return model, N_BARS


def scan_for_window(df: pd.DataFrame, model, n_bars: int,
                    threshold: float, batch_size: int) -> list[dict]:
    """Slide a window of n_bars over df and return all positives."""
    # Re-import normalise_batch with the correct N_BARS in scope
    for mod_name in list(sys.modules.keys()):
        if mod_name in ('scan_bitcoin',):
            del sys.modules[mod_name]

    from scan_bitcoin import scan
    stride = max(1, n_bars // 25)
    print(f'  [{n_bars}-bar]  scanning (stride={stride}) ...')
    positives = scan(df, model, threshold=threshold,
                     stride=stride, batch_size=batch_size)
    return positives


def pick_nonoverlapping(positives: list[dict], window: int) -> list[dict]:
    """Greedy non-overlapping selection sorted by confidence."""
    ranked  = sorted(positives, key=lambda p: p['score'], reverse=True)
    chosen: list[dict] = []
    for p in ranked:
        if all(p['end_idx'] < s['start_idx'] or p['start_idx'] > s['end_idx']
               for s in chosen):
            chosen.append(p)
    chosen.sort(key=lambda p: p['start_idx'])
    return chosen


def backtest(df: pd.DataFrame, detections: list[dict]) -> list[dict]:
    """Compute forward returns at each horizon for a list of detections."""
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n_rows = len(df)
    results = []
    for det in detections:
        we    = det['end_idx']
        entry = float(closes[we])
        row   = dict(det)
        row['entry_price'] = entry
        for h, lbl in zip(HORIZONS, HORIZON_LABELS):
            h_idx = we + h
            if h_idx >= n_rows:
                row[f'return_{lbl}'] = np.nan
                row[f'mfe_{lbl}']    = np.nan
                row[f'win_{lbl}']    = None
            else:
                fc = closes[we + 1 : h_idx + 1]
                fh = highs [we + 1 : h_idx + 1]
                fl = lows  [we + 1 : h_idx + 1]
                row[f'return_{lbl}'] = float((fc[-1] - entry) / entry * 100)
                row[f'mfe_{lbl}']    = float((entry - fl.min()) / entry * 100)
                row[f'win_{lbl}']    = bool(fc[-1] < entry)
        results.append(row)
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_stats(label: str, results: list[dict]) -> dict:
    """Print per-horizon stats and return a summary dict."""
    print(f'\n  {label}  ({len(results)} non-overlapping detections)')
    print(f'  {"Horizon":>6}  {"WinRate":>8}  {"AvgRet":>8}  '
          f'{"MedRet":>8}  {"AvgMFE":>8}  {"n":>5}')
    print(f'  {"-"*55}')

    summary = {'label': label, 'n': len(results)}
    for lbl in HORIZON_LABELS:
        rets = [r[f'return_{lbl}'] for r in results
                if r.get(f'win_{lbl}') is not None
                and not np.isnan(r.get(f'return_{lbl}', np.nan))]
        wins = [r[f'win_{lbl}'] for r in results
                if r.get(f'win_{lbl}') is not None]
        mfes = [r[f'mfe_{lbl}'] for r in results
                if not np.isnan(r.get(f'mfe_{lbl}', np.nan))]
        if not wins:
            continue
        wr  = sum(wins) / len(wins) * 100
        ar  = float(np.mean(rets))
        mr  = float(np.median(rets))
        mfe = float(np.mean(mfes)) if mfes else float('nan')
        print(f'  {lbl:>6}  {wr:>7.1f}%  {ar:>+8.3f}%  '
              f'{mr:>+8.3f}%  {mfe:>+8.3f}%  {len(wins):>5}')
        summary[lbl] = {'win_rate': wr, 'avg_ret': ar,
                         'median_ret': mr, 'avg_mfe': mfe, 'n': len(wins)}
    return summary


def print_comparison(summaries: list[dict]) -> None:
    """Print a side-by-side comparison of all window sizes."""
    print(f'\n{"="*72}')
    print('NON-OVERLAPPING COMPARISON — all window sizes')
    print(f'{"="*72}')
    header = f'{"Horizon":>6}  ' + ''.join(
        f'  {s["label"]:>22}' for s in summaries
    )
    print(header)
    print('-' * 72)
    for lbl in HORIZON_LABELS:
        row = f'{lbl:>6}  '
        for s in summaries:
            h = s.get(lbl, {})
            if h:
                row += f'  {h["win_rate"]:>5.1f}%  {h["avg_ret"]:>+6.2f}%  n={h["n"]:<5}'
            else:
                row += f'  {"--":>22}'
        print(row)
    print(f'{"="*72}\n')


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
_AMBER  = '#FFC107'
_UP_C   = '#80cbc4'
_DN_C   = '#ff8a80'


def _style(ax, xlim):
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.18, color=_GRID)
    ax.set_xlim(-1, xlim + 1)


def _candles(ax, seg, n_window):
    for i, row in seg.iterrows():
        o, c, h, l = row['open'], row['close'], row['high'], row['low']
        in_w = i < n_window
        col  = (_UP if c >= o else _DOWN) if in_w else (_UP_C if c >= o else _DN_C)
        ax.plot([i, i], [l, h], color=col, lw=0.6, zorder=2)
        ax.add_patch(mpatches.Rectangle(
            (i - 0.38, min(o, c)), 0.76, max(abs(c - o), (h-l)*0.005),
            facecolor=col, edgecolor=col, lw=0.3, zorder=3))


def plot_nonoverlapping(df, results, n_bars, cont_bars, out_path, window_label,
                        n_display=5):
    """Plot n_display non-overlapping detections with continuation."""
    top = results[:n_display]   # already sorted chronologically
    if not top:
        return

    n = len(top)
    fig, axes = plt.subplots(
        n * 2, 1, figsize=(16, 6.0 * n),
        gridspec_kw={'height_ratios': [3, 1] * n, 'hspace': 0.65},
    )
    fig.patch.set_facecolor(_FIG_BG)

    for row_i, res in enumerate(top):
        ws     = res['start_idx']
        we     = res['end_idx']
        n_cont = min(cont_bars, len(df) - we - 1)
        seg    = df.iloc[ws : we + 1 + n_cont].reset_index(drop=True)

        ax_p = axes[row_i * 2]
        ax_v = axes[row_i * 2 + 1]
        _style(ax_p, n_bars + n_cont)
        _style(ax_v, n_bars + n_cont)

        ax_p.axvspan(-0.5, n_bars - 0.5,          alpha=0.07, color='#42A5F5', zorder=0)
        ax_p.axvspan(n_bars - 0.5, n_bars + n_cont, alpha=0.07, color='#FF9800', zorder=0)
        for ax in (ax_p, ax_v):
            ax.axvline(n_bars - 0.5, color=_AMBER, lw=1.8, ls='--', zorder=5)

        _candles(ax_p, seg, n_bars)

        ret = res.get('return_4hr', np.nan)
        win = res.get('win_4hr')
        if not np.isnan(ret):
            sign = '+' if ret >= 0 else ''
            ax_p.text(
                n_bars + max(n_cont - 3, 1),
                seg['high'].iloc[n_bars:].max() if n_cont > 0 else seg['high'].max(),
                f"4hr: {sign}{ret:.2f}% ({'WIN' if win else 'LOSS'})",
                color=_UP if win else _DOWN, fontsize=8, ha='right', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='#222233', alpha=0.75, ec='none'),
            )

        ax_p.set_title(
            f'{window_label}  |  Score: {res["score"]:.4f}  |  '
            f'{res["date_start"].strftime("%Y-%m-%d %H:%M")} — '
            f'{res["date_end"].strftime("%Y-%m-%d %H:%M")}  '
            f'+{n_cont}-bar continuation',
            color=_TEXT, fontsize=8.5, pad=5,
        )
        ax_p.set_ylabel('BTC / USD', color=_MUTED, fontsize=7)

        for i, row in seg.iterrows():
            ax_v.bar(i, row['volume'],
                     color='#5c6bc0' if i < n_bars else '#FF9800',
                     width=0.85, alpha=0.85)
        ax_v.set_ylabel('Volume', color=_MUTED, fontsize=7)
        ax_v.set_xlabel('Bar offset (1 bar = 1 min)', color=_MUTED, fontsize=7)

    fig.suptitle(
        f'Non-Overlapping Rising Wedge Detections — {window_label}',
        color='#e8e8ff', fontsize=12, y=1.005,
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Chart saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare non-overlapping detections across window sizes'
    )
    parser.add_argument('--windows',    nargs='+', type=int, default=[50, 250],
                        help='Window sizes to compare (default: 50 250)')
    parser.add_argument('--threshold',  type=float, default=0.5)
    parser.add_argument('--batch-size', type=int,   default=5_000)
    parser.add_argument('--display',    type=int,   default=5,
                        help='Charts per window size (default: 5)')
    args = parser.parse_args()

    # Load BTC data once (shared across all window sizes)
    print('Loading BTC data ...')
    # Import load_btc from scan_bitcoin (N_BARS-agnostic helper)
    from scan_bitcoin import load_btc
    df = load_btc(BTC_CSV)

    out_root = PROJECT_ROOT / 'models' / 'comparison'
    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for window in sorted(args.windows):
        print(f'\n{"="*64}')
        print(f'  Window size: {window} bars')
        print(f'{"="*64}')

        # Load model with correct window size
        model, n_bars = load_model_for_window(window)

        # Scan
        positives = scan_for_window(df, model, n_bars, args.threshold,
                                     args.batch_size)
        print(f'  [{window}-bar]  total positives: {len(positives):,}')

        # Non-overlapping filter
        non_ol = pick_nonoverlapping(positives, window)
        print(f'  [{window}-bar]  non-overlapping: {len(non_ol):,}')

        # Backtest on non-overlapping set
        bt = backtest(df, non_ol)

        # Stats
        label   = f'{window}-bar  (n={len(bt)})'
        summary = print_stats(label, bt)
        all_summaries.append(summary)

        # Charts — pick top 5 by score, display chronologically
        top5 = sorted(
            sorted(bt, key=lambda r: r['score'], reverse=True)[:args.display],
            key=lambda r: r['start_idx'],
        )
        cont = window   # one full window of continuation
        chart_path = out_root / f'nonoverlapping_{window}bar.png'
        plot_nonoverlapping(df, top5, n_bars=n_bars, cont_bars=cont,
                            out_path=chart_path,
                            window_label=f'{window}-bar non-overlapping',
                            n_display=args.display)

    # Side-by-side summary
    print_comparison(all_summaries)
    print(f'All charts saved to: {out_root}')


if __name__ == '__main__':
    main()
