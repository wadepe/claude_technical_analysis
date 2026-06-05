"""
backtest_btc.py

Scans BTC 1-min data for rising-wedge patterns, then for every detection
measures the price action in the hours that follow and plots 5 examples
with continuation data appended to the right of the detection window.

Backtest definition
-------------------
  Entry price  : close of the final bar in the 250-bar detection window
  Horizons     : 30 min / 1 hr / 2 hr / 4 hr  (= 30, 60, 120, 240 bars)
  Win          : close at horizon < entry close  (bearish reversal confirmed)
  MAE          : max adverse excursion  -- how far UP price went before coming back
  MFE          : max favourable excursion -- how far DOWN price got

Usage
-----
  python backtest_btc.py
  python backtest_btc.py --continuation-bars 120   # show 2 hrs of follow-through
  python backtest_btc.py --threshold 0.4            # widen detection net
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import build_model, N_BARS
from scan_bitcoin import load_btc, normalise_batch, scan, pick_nonoverlapping

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
BTC_CSV      = PROJECT_ROOT / 'reference_material' / 'btc_data_bi_min.csv'
WEIGHTS_PATH = PROJECT_ROOT / 'models' / 'cnn_best.weights.h5'
OUTPUT_DIR   = PROJECT_ROOT / 'models' / 'backtest'

FEATURE_COLS   = ['open', 'high', 'low', 'close', 'volume']
HORIZONS       = [30, 60, 120, 240]
HORIZON_LABELS = ['30m', '1hr', '2hr', '4hr']


# =============================================================================
# Backtest computation
# =============================================================================

def compute_backtest(df: pd.DataFrame, detections: list[dict]) -> list[dict]:
    """
    Append forward-return metrics to each detection dict.

    For a rising-wedge SHORT signal:
      - negative return  = price fell  = WIN
      - positive return  = price rose  = LOSS
      - MAE (up from entry) = risk you took on before it worked
      - MFE (down from entry) = profit potential achieved
    """
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n_rows = len(df)

    results = []
    for det in detections:
        we          = det['end_idx']
        entry_price = float(closes[we])
        row         = dict(det)
        row['entry_price'] = entry_price

        for h, lbl in zip(HORIZONS, HORIZON_LABELS):
            h_idx = we + h
            if h_idx >= n_rows:
                row[f'return_{lbl}'] = np.nan
                row[f'mae_{lbl}']    = np.nan
                row[f'mfe_{lbl}']    = np.nan
                row[f'win_{lbl}']    = None
            else:
                future_c = closes[we + 1 : h_idx + 1]
                future_h = highs [we + 1 : h_idx + 1]
                future_l = lows  [we + 1 : h_idx + 1]

                ret = (future_c[-1] - entry_price) / entry_price * 100
                mae = (future_h.max() - entry_price) / entry_price * 100  # up = adverse
                mfe = (entry_price - future_l.min()) / entry_price * 100   # down = favourable

                row[f'return_{lbl}'] = round(float(ret), 3)
                row[f'mae_{lbl}']    = round(float(mae),  3)
                row[f'mfe_{lbl}']    = round(float(mfe),  3)
                row[f'win_{lbl}']    = bool(future_c[-1] < entry_price)

        results.append(row)

    return results


# =============================================================================
# Reporting
# =============================================================================

def print_backtest_table(results: list[dict]) -> None:
    n = len(results)
    sep = '=' * 96

    print(f'\n{sep}')
    print(f'BACKTEST  —  {n} Rising Wedge Detections  (entry = close of detection window)')
    print(f'{sep}')
    print(
        f"{'#':>3}  {'Window end':>20}  {'Score':>6}  "
        f"{'Ret30m':>7}  {'Ret1hr':>7}  {'Ret2hr':>7}  {'Ret4hr':>7}  "
        f"{'MFE4hr':>7}  {'MAE4hr':>7}  "
        f"{'W30m':>5}  {'W1hr':>5}  {'W2hr':>5}  {'W4hr':>5}"
    )
    print('-' * 96)

    for i, r in enumerate(results, 1):
        def fmt_ret(k):
            v = r.get(k, np.nan)
            return f'{v:+7.2f}' if not np.isnan(v) else '    -- '

        def fmt_win(k):
            w = r.get(k)
            return '  Y  ' if w is True else ('  N  ' if w is False else '  -  ')

        print(
            f"{i:>3}  {str(r['date_end']):>20}  {r['score']:>6.4f}  "
            f"{fmt_ret('return_30m')}  {fmt_ret('return_1hr')}  "
            f"{fmt_ret('return_2hr')}  {fmt_ret('return_4hr')}  "
            f"{fmt_ret('mfe_4hr')}  {fmt_ret('mae_4hr')}  "
            f"{fmt_win('win_30m')}{fmt_win('win_1hr')}{fmt_win('win_2hr')}{fmt_win('win_4hr')}"
        )

    print('-' * 96)
    print('Summary (bearish = WIN for a rising-wedge short signal):')

    for lbl in HORIZON_LABELS:
        rets = [r[f'return_{lbl}'] for r in results
                if r.get(f'win_{lbl}') is not None and not np.isnan(r.get(f'return_{lbl}', np.nan))]
        wins = [r[f'win_{lbl}']   for r in results
                if r.get(f'win_{lbl}') is not None]
        if not wins:
            continue
        win_rate  = sum(wins) / len(wins) * 100
        avg_ret   = float(np.mean(rets))
        med_ret   = float(np.median(rets))
        mfes      = [r[f'mfe_{lbl}'] for r in results
                     if r.get(f'win_{lbl}') is not None and not np.isnan(r.get(f'mfe_{lbl}', np.nan))]
        avg_mfe   = float(np.mean(mfes)) if mfes else float('nan')
        print(
            f'  {lbl:>4}:  win_rate={win_rate:5.1f}%  '
            f'avg_return={avg_ret:+.2f}%  median={med_ret:+.2f}%  '
            f'avg_MFE(down)={avg_mfe:+.2f}%  n={len(wins)}'
        )

    print(f'{sep}\n')


# =============================================================================
# Plotting
# =============================================================================

_FIG_BG  = '#1a1a2e'
_AX_BG   = '#13131f'
_GRID    = '#2a2a3e'
_SPINE   = '#333345'
_MUTED   = '#8a8aa0'
_TEXT    = '#d0d0e8'
_UP      = '#26a69a'    # detection window — up candle
_DOWN    = '#ef5350'    # detection window — down candle
_UP_CONT = '#80cbc4'    # continuation — up candle  (lighter)
_DN_CONT = '#ff8a80'    # continuation — down candle (lighter)
_AMBER   = '#FFC107'    # detection boundary marker


def _style_ax(ax: plt.Axes, xlim: int) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.20, color=_GRID)
    ax.set_xlim(-1, xlim + 1)


def _draw_candles(ax: plt.Axes, seg: pd.DataFrame, n_window: int) -> None:
    for i, row in seg.iterrows():
        o, c = row['open'], row['close']
        h, l = row['high'], row['low']
        in_win = i < n_window
        col = (_UP if c >= o else _DOWN) if in_win else (_UP_CONT if c >= o else _DN_CONT)
        ax.plot([i, i], [l, h], color=col, lw=0.75, zorder=2)
        body_h = max(abs(c - o), (h - l) * 0.005)
        ax.add_patch(mpatches.Rectangle(
            (i - 0.38, min(o, c)), 0.76, body_h,
            facecolor=col, edgecolor=col, lw=0.3, zorder=3,
        ))


def plot_backtest(
    df: pd.DataFrame,
    results: list[dict],
    n_display: int,
    cont_bars: int,
    out_dir: Path,
    threshold: float,
) -> None:
    """
    Plot n_display detection windows with continuation data appended.
    Selects the n_display highest-confidence non-overlapping results.
    """
    top = pick_nonoverlapping(results, n=n_display)
    if not top:
        print('No results to plot.')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n     = len(top)
    total = N_BARS + cont_bars

    fig, axes = plt.subplots(
        n * 2, 1,
        figsize=(16, 6.5 * n),
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
        _style_ax(ax_p, N_BARS + n_cont)
        _style_ax(ax_v, N_BARS + n_cont)

        # Background shading
        ax_p.axvspan(-0.5,            N_BARS - 0.5,           alpha=0.07, color='#42A5F5', zorder=0)
        ax_p.axvspan(N_BARS - 0.5,    N_BARS + n_cont + 0.5,  alpha=0.07, color='#FF9800', zorder=0)

        # Window-end boundary
        for ax in (ax_p, ax_v):
            ax.axvline(N_BARS - 0.5, color=_AMBER, lw=2.0, ls='--', zorder=5)

        # Candles
        _draw_candles(ax_p, seg, N_BARS)

        # 4-hr return annotation in top-right of continuation zone
        ret_4hr = res.get('return_4hr', np.nan)
        win_4hr = res.get('win_4hr')
        if not np.isnan(ret_4hr):
            sign    = '+' if ret_4hr >= 0 else ''
            ret_str = f"4hr return: {sign}{ret_4hr:.2f}%  ({'WIN' if win_4hr else 'LOSS'})"
            ret_col = _UP if win_4hr else _DOWN  # green for bearish win (price fell)
            y_pos   = seg['high'].iloc[N_BARS:].max() if n_cont > 0 else seg['high'].max()
            ax_p.text(
                N_BARS + max(n_cont - 5, 1), y_pos,
                ret_str,
                color=ret_col, fontsize=8.5, ha='right', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#222233', alpha=0.75, edgecolor='none'),
            )

        ax_p.set_title(
            f"Detection  |  Score: {res['score']:.4f}  |  "
            f"{res['date_start'].strftime('%Y-%m-%d %H:%M')}  to  "
            f"{res['date_end'].strftime('%Y-%m-%d %H:%M')}"
            f"  +{n_cont} min continuation",
            color=_TEXT, fontsize=9, pad=6,
        )
        ax_p.set_ylabel('BTC / USD', color=_MUTED, fontsize=8)

        legend_items = [
            mpatches.Patch(color='#42A5F5', alpha=0.5, label=f'Detection window ({N_BARS} bars)'),
            mpatches.Patch(color='#FF9800', alpha=0.5, label=f'Continuation ({n_cont} min)'),
            mpatches.Patch(color=_AMBER,    alpha=0.9, label='Window end'),
        ]
        ax_p.legend(handles=legend_items, loc='upper left', fontsize=7.5,
                    facecolor='#222233', labelcolor=_TEXT, framealpha=0.75, ncol=3)

        # Volume: blue = detection window, orange = continuation
        for i, row in seg.iterrows():
            col = '#5c6bc0' if i < N_BARS else '#FF9800'
            ax_v.bar(i, row['volume'], color=col, width=0.85, alpha=0.85)
        ax_v.set_ylabel('Volume (BTC)', color=_MUTED, fontsize=8)
        ax_v.set_xlabel('Bar offset (1 bar = 1 minute)', color=_MUTED, fontsize=8)

    fig.suptitle(
        f"BTC Rising Wedge — Detection Windows + {cont_bars}-bar Continuation  "
        f"(1D CNN, threshold={threshold:.2f})",
        color='#e8e8ff', fontsize=13, y=1.005,
    )

    out_path = out_dir / 'backtest_charts.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Charts saved: {out_path.resolve()}')


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Backtest BTC rising-wedge detections with continuation data'
    )
    parser.add_argument('--threshold',         type=float, default=0.5)
    parser.add_argument('--stride',            type=int,   default=10)
    parser.add_argument('--batch-size',        type=int,   default=5_000)
    parser.add_argument('--continuation-bars', type=int,   default=240,
                        help='Bars of post-window data to display (default: 240 = 4 hr)')
    parser.add_argument('--display',           type=int,   default=5,
                        help='Number of example charts to plot (default: 5)')
    parser.add_argument('--weights',    default=str(WEIGHTS_PATH),
                        help='Path to model weights file')
    parser.add_argument('--output-dir', default=str(OUTPUT_DIR),
                        help='Directory for output charts and reports')
    args = parser.parse_args()

    # ── Load data + model ─────────────────────────────────────────────────────
    df = load_btc(BTC_CSV)
    print(f'\nLoading weights: {args.weights}')
    model = build_model(print_summary=False)
    model.load_weights(args.weights)
    print('Ready.\n')

    # ── Scan ──────────────────────────────────────────────────────────────────
    positives = scan(
        df         = df,
        model      = model,
        threshold  = args.threshold,
        stride     = args.stride,
        batch_size = args.batch_size,
    )

    if not positives:
        print('No detections found. Try lowering --threshold.')
        return

    # ── Backtest ──────────────────────────────────────────────────────────────
    print(f'Computing backtest metrics for {len(positives):,} detections ...')
    results = compute_backtest(df, positives)
    print_backtest_table(results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    print(f'Plotting {args.display} examples with {args.continuation_bars}-bar continuation ...')
    plot_backtest(
        df         = df,
        results    = results,
        n_display  = args.display,
        cont_bars  = args.continuation_bars,
        out_dir    = Path(args.output_dir),
        threshold  = args.threshold,
    )


if __name__ == '__main__':
    main()
