"""
slope_return_histograms.py

Raw forward-return distributions for detected wedges split by slope, against a
random-window baseline.

Classical TA reads a rising wedge as bearish and a falling wedge as bullish, so
pooling them cancels opposite drifts in any signed statistic. This keeps them
apart, adds a flat class, and plots the whole distribution rather than a mean --
two groups can share a mean and differ completely in shape, which is exactly
what a "breakout pattern" claim is about.

Returns are RAW PERCENT, deliberately: volatility-normalised z answers "did it
move more than implied", but raw percent is what a position actually earns.

Slope is the midline travel over the fitted span, in units of the window's own
price range, so it is scale-free:  mid = (travel_upper + travel_lower) / 2.
Flat is |mid| < --flat-threshold; the midline is essentially horizontal.

Writes slope_histograms.png plus slope_returns.csv (one row per event, returns
at every horizon) so the distributions can be re-analysed without re-running.

Usage
-----
  python slope_return_histograms.py \
      --csv ../spy_1min_backtest_regular.csv \
      --events /data/runs_v5/window_250bar/models/filtered/events_filtered.csv \
      --output-dir /data/runs_v5/window_250bar/models/slope_returns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATE_COL = 'date'
# 390 bars = one regular session, so 6hr crosses the close for most detections.
HORIZONS = [30, 60, 120, 240, 360]
LABELS   = ['30m', '1hr', '2hr', '4hr', '6hr']

_FIG_BG, _AX_BG = '#1a1a2e', '#13131f'
_GRID, _SPINE   = '#2a2a3e', '#333345'
_MUTED, _TEXT   = '#8a8aa0', '#d0d0e8'
_RISE, _FALL    = '#26A69A', '#EF5350'
_FLAT, _RAND    = '#FF9800', '#8a8aa0'


def forward_pct(close: np.ndarray, ends: np.ndarray) -> dict:
    """Raw forward percentage return at each horizon."""
    out = {}
    for h, lab in zip(HORIZONS, LABELS):
        tgt = ends + h
        ok  = tgt < len(close)
        r = np.full(len(ends), np.nan)
        r[ok] = (close[tgt[ok]] - close[ends[ok]]) / close[ends[ok]] * 100.0
        out[lab] = r
    return out


def plot_separate(groups: dict, out: Path) -> None:
    """
    One panel per (horizon, slope class), each against the random baseline.

    The four-way overlay was unreadable: four step outlines on one axis with
    similar shapes is a spaghetti plot. Splitting them keeps every comparison
    to two series -- the group filled, the baseline outlined -- which is the
    only comparison that matters anyway.
    """
    cols = [('rising', _RISE), ('falling', _FALL), ('flat', _FLAT)]
    fig, axes = plt.subplots(len(LABELS), 3,
                             figsize=(3 * 4.5, len(LABELS) * 2.9),
                             facecolor=_FIG_BG)
    axes = np.atleast_2d(axes)

    for r, lab in enumerate(LABELS):
        rv = groups['random'][0][lab]
        rv = rv[np.isfinite(rv)]
        clip = float(np.percentile(np.abs(rv), 99))
        bins = np.linspace(-clip, clip, 49)
        for c, (nm, col) in enumerate(cols):
            ax = axes[r, c]
            v = groups[nm][0][lab]
            v = v[np.isfinite(v)]
            ax.hist(np.clip(rv, -clip, clip), bins=bins, density=True,
                    histtype='step', color=_MUTED, linewidth=1.5,
                    label=f'random n={len(rv):,}')
            ax.hist(np.clip(v, -clip, clip), bins=bins, density=True,
                    histtype='stepfilled', color=col, alpha=0.5,
                    edgecolor=col, linewidth=1.3, label=f'{nm} n={len(v):,}')
            ax.axvline(0, color=_MUTED, linewidth=0.8, alpha=0.5)
            ax.axvline(rv.mean(), color=_MUTED, linewidth=1.2, linestyle='--')
            ax.axvline(v.mean(), color=col, linewidth=1.4, linestyle='--')
            ax.set_title(f'{nm}  {lab}   mean={v.mean():+.4f}%  '
                         f'(random {rv.mean():+.4f}%)\n'
                         f'win={float((v > 0).mean())*100:.1f}%  '
                         f'|move|={np.abs(v).mean():.4f}%  '
                         f'(random {float((rv > 0).mean())*100:.1f}%, '
                         f'{np.abs(rv).mean():.4f}%)',
                         color=col, fontsize=8.5)
            ax.set_xlabel('return %', color=_MUTED, fontsize=8)
            if c == 0:
                ax.set_ylabel('density', color=_MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=_AX_BG, edgecolor=_SPINE,
                      labelcolor=_TEXT)
            _style(ax)

    fig.suptitle('Raw forward returns: each wedge class against the random '
                 'baseline (dashed = means)', color=_TEXT, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / 'slope_histograms_separate.png', dpi=135,
                facecolor=_FIG_BG)
    plt.close(fig)
    print(f'  saved {out}/slope_histograms_separate.png')

    # One file per horizon as well, for when a single horizon is the question.
    for lab in LABELS:
        rv = groups['random'][0][lab]; rv = rv[np.isfinite(rv)]
        clip = float(np.percentile(np.abs(rv), 99))
        bins = np.linspace(-clip, clip, 49)
        f2, ax2 = plt.subplots(1, 3, figsize=(14, 3.6), facecolor=_FIG_BG)
        for ax, (nm, col) in zip(np.atleast_1d(ax2), cols):
            v = groups[nm][0][lab]; v = v[np.isfinite(v)]
            ax.hist(np.clip(rv, -clip, clip), bins=bins, density=True,
                    histtype='step', color=_MUTED, linewidth=1.5,
                    label=f'random n={len(rv):,}')
            ax.hist(np.clip(v, -clip, clip), bins=bins, density=True,
                    histtype='stepfilled', color=col, alpha=0.5,
                    edgecolor=col, linewidth=1.3, label=f'{nm} n={len(v):,}')
            ax.axvline(0, color=_MUTED, linewidth=0.8, alpha=0.5)
            ax.axvline(v.mean(), color=col, linewidth=1.4, linestyle='--')
            ax.axvline(rv.mean(), color=_MUTED, linewidth=1.2, linestyle='--')
            ax.set_title(f'{nm}   mean={v.mean():+.4f}%  '
                         f'win={float((v > 0).mean())*100:.1f}%  '
                         f'|move|={np.abs(v).mean():.4f}%',
                         color=col, fontsize=9.5)
            ax.set_xlabel('return %', color=_MUTED, fontsize=8)
            ax.legend(fontsize=7, facecolor=_AX_BG, edgecolor=_SPINE,
                      labelcolor=_TEXT)
            _style(ax)
        f2.suptitle(f'{lab} forward return by wedge slope  '
                    f'(random: mean {rv.mean():+.4f}%, '
                    f'win {float((rv > 0).mean())*100:.1f}%, '
                    f'|move| {np.abs(rv).mean():.4f}%)',
                    color=_TEXT, fontsize=11)
        f2.tight_layout(rect=(0, 0, 1, 0.90))
        f2.savefig(out / f'slope_hist_{lab}.png', dpi=135, facecolor=_FIG_BG)
        plt.close(f2)
        print(f'  saved {out}/slope_hist_{lab}.png')


def _style(ax):
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(colors=_MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(_SPINE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--events', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--flat-threshold', type=float, default=0.15,
                   help='|midline travel| below this counts as flat')
    p.add_argument('--category', default='CLEAN',
                   help='which filter category to use, or ALL')
    p.add_argument('--baseline', type=int, default=20_000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, usecols=[DATE_COL, 'close'])
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    close = df['close'].values.astype(float)

    ev = pd.read_csv(args.events)
    if args.category != 'ALL':
        ev = ev[ev['category'] == args.category].copy()
    ev['mid'] = (ev['travel_upper'] + ev['travel_lower']) / 2.0

    t = args.flat_threshold
    ev['slope'] = np.where(ev['mid'] >= t, 'rising',
                  np.where(ev['mid'] <= -t, 'falling', 'flat'))

    rng = np.random.default_rng(args.seed)
    base_ends = rng.choice(
        np.arange(250, len(close) - max(HORIZONS) - 1),
        size=min(args.baseline, len(close) // 4), replace=False)

    groups = {}
    for nm in ('rising', 'falling', 'flat'):
        sub = ev[ev['slope'] == nm]
        groups[nm] = (forward_pct(close, sub['end_idx'].values), len(sub))
    groups['random'] = (forward_pct(close, base_ends), len(base_ends))

    # ── Summary ──────────────────────────────────────────────────────────────
    rep = [f'RAW FORWARD RETURNS BY WEDGE SLOPE  (category={args.category}, '
           f'flat = |mid| < {t})', '=' * 78,
           f'  rising n={groups["rising"][1]:,}   '
           f'falling n={groups["falling"][1]:,}   '
           f'flat n={groups["flat"][1]:,}   random n={groups["random"][1]:,}',
           '',
           f'  {"group":<10}{"horizon":<8}{"mean %":>10}{"median %":>11}'
           f'{"mean|move| %":>14}{"win %":>9}{"p5 %":>9}{"p95 %":>9}']
    rep.append('  ' + '-' * 72)
    for nm in ('rising', 'falling', 'flat', 'random'):
        fr, n = groups[nm]
        for lab in LABELS:
            v = fr[lab]; v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            rep.append(f'  {nm:<10}{lab:<8}{v.mean():>+10.4f}'
                       f'{np.median(v):>+11.4f}{np.abs(v).mean():>14.4f}'
                       f'{float((v > 0).mean())*100:>8.1f}%'
                       f'{np.percentile(v, 5):>+9.3f}{np.percentile(v, 95):>+9.3f}')
        rep.append('')
    text = '\n'.join(rep)
    print(text)
    (out / 'slope_returns.txt').write_text(text)

    # ── Event-level data, for re-analysis without re-running ─────────────────
    rows = ev[['date_start', 'date_end', 'score', 'convergence', 'mid', 'slope',
               'end_idx']].copy()
    fr_all = forward_pct(close, ev['end_idx'].values)
    for lab in LABELS:
        rows[f'ret_{lab}'] = fr_all[lab]
    rows.to_csv(out / 'slope_returns.csv', index=False)
    br = pd.DataFrame({'end_idx': base_ends})
    for lab in LABELS:
        br[f'ret_{lab}'] = groups['random'][0][lab]
    br.to_csv(out / 'baseline_returns.csv', index=False)

    # ── Histograms ───────────────────────────────────────────────────────────
    cols = 3
    rows_n = int(np.ceil(len(LABELS) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.6, rows_n * 3.5),
                             facecolor=_FIG_BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(LABELS):]:
        ax.set_visible(False)

    style = (('random', _RAND, 1.6, '-'), ('rising', _RISE, 1.7, '-'),
             ('falling', _FALL, 1.7, '-'), ('flat', _FLAT, 1.7, '--'))

    for ax, lab in zip(axes, LABELS):
        rv = groups['random'][0][lab]
        rv = rv[np.isfinite(rv)]
        clip = float(np.percentile(np.abs(rv), 99))
        bins = np.linspace(-clip, clip, 55)
        for nm, col, lw, ls in style:
            fr, n = groups[nm]
            v = fr[lab]; v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            ax.hist(np.clip(v, -clip, clip), bins=bins, density=True,
                    histtype='step', color=col, linewidth=lw, linestyle=ls,
                    label=f'{nm} n={len(v):,}  mean={v.mean():+.3f}%')
        ax.axvline(0, color=_MUTED, linewidth=0.8, alpha=0.5)
        ax.set_title(f'{lab} forward return', color=_TEXT, fontsize=10)
        ax.set_xlabel('return %', color=_MUTED, fontsize=8)
        ax.set_ylabel('density', color=_MUTED, fontsize=8)
        ax.legend(fontsize=6.8, facecolor=_AX_BG, edgecolor=_SPINE,
                  labelcolor=_TEXT)
        _style(ax)

    fig.suptitle(f'Raw forward returns by wedge slope  '
                 f'({args.category} detections, flat = |mid| < {t})',
                 color=_TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / 'slope_histograms.png', dpi=135, facecolor=_FIG_BG)
    plt.close(fig)
    plot_separate(groups, out)

    print(f'\nSaved: {out}/slope_histograms.png, slope_returns.csv, '
          f'baseline_returns.csv, slope_returns.txt')


if __name__ == '__main__':
    main()
