"""
slope_move_study.py

Tests the thesis: does the SHAPE of a detected wedge (upper/lower trendline
slopes, convergence) correlate with the SIZE of the move that follows?

Pipeline
--------
  1. Slide the v2 CNN over a 1-min OHLCV archive (scan_bitcoin.scan)
  2. Collapse overlapping positive windows into distinct EVENTS
     (peak-score window per overlap group, so one wedge = one data point)
  3. Fit upper/lower envelope trendlines on each event window
     (classify_wedge.fit_wedge_lines — raw slopes, no type bucketing)
  4. Measure forward returns at 30m / 1h / 2h / 4h after the window end
  5. Plot forward return vs upper-line slope AND lower-line slope on the
     same axes (one panel per horizon), with per-series correlations
  6. Save the event table (CSV) + correlation summary (txt) + chart (PNG)

Usage
-----
  python slope_move_study.py --csv ../spy_1min_backtest.csv \
      --weights ../runs_v2/window_250bar/models/cnn_best.weights.h5 \
      --output-dir ../runs_v2/window_250bar/models/slope_study
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import build_model, N_BARS
from scan_bitcoin import load_btc, scan
from classify_wedge import fit_wedge_lines

FEATURE_COLS   = ['open', 'high', 'low', 'close', 'volume']
HORIZONS       = [30, 60, 120, 240]
HORIZON_LABELS = ['30m', '1hr', '2hr', '4hr']

# Project dark theme
_FIG_BG, _AX_BG  = '#1a1a2e', '#13131f'
_GRID,  _SPINE   = '#2a2a3e', '#333345'
_MUTED, _TEXT    = '#8a8aa0', '#d0d0e8'
_BLUE,  _ORANGE  = '#42A5F5', '#FF9800'


def cluster_events(positives: list[dict]) -> list[dict]:
    """Collapse overlapping detection windows into one event each (peak score)."""
    if not positives:
        return []
    positives = sorted(positives, key=lambda p: p['start_idx'])
    events, cur = [], positives[0]
    cur_end = cur['end_idx']
    for p in positives[1:]:
        if p['start_idx'] <= cur_end:                 # overlaps current group
            cur_end = max(cur_end, p['end_idx'])
            if p['score'] > cur['score']:
                cur = p
        else:
            events.append(cur)
            cur, cur_end = p, p['end_idx']
    events.append(cur)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description='Wedge slope vs forward-move study')
    parser.add_argument('--csv',        required=True)
    parser.add_argument('--weights',    required=True)
    parser.add_argument('--threshold',  type=float, default=0.8)
    parser.add_argument('--stride',     type=int,   default=10)
    parser.add_argument('--batch-size', type=int,   default=5_000)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Scan ──────────────────────────────────────────────────────────────────
    df = load_btc(Path(args.csv))
    model = build_model(print_summary=False)
    model.load_weights(args.weights)
    positives = scan(df, model, args.threshold, args.stride, args.batch_size)
    print(f'\nRaw positive windows : {len(positives):,}')

    events = cluster_events(positives)
    print(f'Distinct events      : {len(events):,}')
    if len(events) < 10:
        print('Too few events for a correlation study — lower --threshold.')
        return

    # ── Geometry + forward returns per event ─────────────────────────────────
    data   = df[FEATURE_COLS].values.astype(np.float32)
    closes = df['close'].values
    n_rows = len(df)

    rows = []
    for ev in events:
        s, e = ev['start_idx'], ev['end_idx']
        g = fit_wedge_lines(data[s:e + 1])
        entry = float(closes[e])
        row = {
            'date_end':  ev['date_end'], 'score': ev['score'],
            'start_idx': s, 'end_idx': e, 'entry': entry, **g,
        }
        for h, lbl in zip(HORIZONS, HORIZON_LABELS):
            if e + h < n_rows:
                row[f'ret_{lbl}'] = (closes[e + h] - entry) / entry * 100
            else:
                row[f'ret_{lbl}'] = np.nan
        rows.append(row)

    ev_df = pd.DataFrame(rows)
    ev_df.to_csv(out_dir / 'events.csv', index=False)
    print(f'Event table saved: {out_dir / "events.csv"}')

    # ── Correlation summary ───────────────────────────────────────────────────
    lines = [f'SLOPE vs FORWARD MOVE  ({Path(args.csv).name}, '
             f'threshold={args.threshold}, stride={args.stride}, '
             f'{len(ev_df):,} events)', '=' * 74]
    for lbl in HORIZON_LABELS:
        r = ev_df[f'ret_{lbl}'].astype(float)
        ok = r.notna()
        lines.append(f'\n[{lbl}]  n={ok.sum():,}   '
                     f'mean_ret={r[ok].mean():+.3f}%   mean_|ret|={r[ok].abs().mean():.3f}%')
        for feat in ('travel_upper', 'travel_lower', 'convergence', 'score'):
            f_ = ev_df.loc[ok, feat].astype(float)
            r_signed = float(np.corrcoef(f_, r[ok])[0, 1])
            r_size   = float(np.corrcoef(f_, r[ok].abs())[0, 1])
            lines.append(f'  {feat:<13}  r(signed ret)={r_signed:+.3f}   '
                         f'r(|ret| size)={r_size:+.3f}')
    summary = '\n'.join(lines)
    print('\n' + summary)
    (out_dir / 'correlations.txt').write_text(summary)

    # ── Chart: forward return vs both slopes, same axes, per horizon ─────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor(_FIG_BG)

    for ax, h_lbl in zip(axes.ravel(), HORIZON_LABELS):
        ax.set_facecolor(_AX_BG)
        ax.tick_params(colors=_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE)
        ax.grid(True, alpha=0.20, color=_GRID)
        ax.axhline(0, color=_MUTED, lw=0.8, ls=':')
        ax.axvline(0, color=_MUTED, lw=0.8, ls=':')

        ok = ev_df[f'ret_{h_lbl}'].notna()
        y  = ev_df.loc[ok, f'ret_{h_lbl}'].astype(float).values

        for feat, col, name in (('travel_upper', _BLUE,   'upper slope'),
                                ('travel_lower', _ORANGE, 'lower slope')):
            x = ev_df.loc[ok, feat].astype(float).values
            r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
            ax.scatter(x, y, s=14, alpha=0.45, color=col, edgecolors='none',
                       label=f'{name}  (r={r:+.3f})')
            if len(x) > 2:                       # linear trend per series
                b, a = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, a + b * xs, color=col, lw=1.6, alpha=0.9)

        ax.set_title(f'Forward return @ {h_lbl}  (n={ok.sum():,})',
                     color=_TEXT, fontsize=11)
        ax.set_xlabel('Trendline travel across window  (fraction of price range)',
                      color=_MUTED, fontsize=9)
        ax.set_ylabel('Forward return  (%)', color=_MUTED, fontsize=9)
        ax.legend(facecolor='#222233', labelcolor=_TEXT, fontsize=9,
                  framealpha=0.8)

    fig.suptitle(
        f'Wedge shape vs following move — {len(ev_df):,} events, '
        f'{N_BARS}-bar model @ threshold {args.threshold}',
        color='#e8e8ff', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = out_dir / 'slope_vs_move.png'
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nChart saved: {out_png}')


if __name__ == '__main__':
    main()
