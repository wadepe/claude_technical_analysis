"""
nondud_projection.py

Linear model of the 4-hour move size for NON-DUD wedge events.

Non-dud: |4hr forward return| >= the archive's unconditional median |4hr move|
(the same dud definition as vol_control_study.py). The regression therefore
answers: GIVEN a wedge resolves into an above-typical move, how large is that
move as a linear function of the wedge's trendline slopes?

Outputs: scatter + OLS fit per slope series (upper/lower) with equations,
and a printed model summary (OLS coefficients, Pearson/Spearman r, R^2).

Usage
-----
  python nondud_projection.py --csv ../spy_1min_backtest.csv \
      --events ../runs_v2/window_250bar/models/slope_study/events.csv \
      --output-dir ../runs_v2/window_250bar/models/slope_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

H = 240   # 4-hour horizon in bars

_FIG_BG, _AX_BG = '#1a1a2e', '#13131f'
_GRID, _SPINE   = '#2a2a3e', '#333345'
_MUTED, _TEXT   = '#8a8aa0', '#d0d0e8'
_BLUE, _ORANGE  = '#42A5F5', '#FF9800'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',        required=True)
    ap.add_argument('--events',     required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--baseline-stride', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    df      = pd.read_csv(args.csv, usecols=['close'])
    closes  = df['close'].values.astype(float)
    n       = len(closes)
    ev      = pd.read_csv(args.events)

    # ── |4hr move| for every bar; unconditional median = dud threshold ────────
    fwd = np.full(n, np.nan)
    fwd[:n - H] = np.abs(closes[H:] / closes[:-H] - 1.0) * 100
    base_med = float(np.nanmedian(fwd[np.arange(390, n - H, args.baseline_stride)]))

    e_idx = ev['end_idx'].values.astype(int)
    ok    = e_idx < (n - H)
    ev    = ev.loc[ok].copy()
    ev['abs_4hr'] = fwd[e_idx[ok]]

    nondud = ev[ev.abs_4hr >= base_med].copy()
    print(f'events with 4hr data : {len(ev):,}')
    print(f'dud threshold        : |4hr move| >= {base_med:.3f}%  (archive median)')
    print(f'non-dud events       : {len(nondud):,} '
          f'({len(nondud)/len(ev)*100:.1f}% of events)\n')

    # ── OLS per slope series ──────────────────────────────────────────────────
    y = nondud['abs_4hr'].values
    fits = {}
    lines = [f'NON-DUD 4HR MOVE PROJECTION  (n={len(nondud):,}, '
             f'dud threshold {base_med:.3f}%)', '=' * 70]
    for feat, label in (('travel_upper', 'upper slope'),
                        ('travel_lower', 'lower slope'),
                        ('mid', 'mid (avg of both)')):
        x = ((nondud.travel_upper + nondud.travel_lower) / 2).values \
            if feat == 'mid' else nondud[feat].values
        res = stats.linregress(x, y)
        rho = stats.spearmanr(x, y).correlation
        fits[feat] = (res, x)
        lines.append(
            f'\n{label}:\n'
            f'  projected |4hr move|% = {res.intercept:.3f} '
            f'{res.slope:+.3f} * travel\n'
            f'  pearson r={res.rvalue:+.3f}  R^2={res.rvalue**2:.3f}  '
            f'spearman rho={rho:+.3f}  slope_stderr={res.stderr:.3f}  '
            f'p={res.pvalue:.2e}'
        )
    summary = '\n'.join(lines)
    print(summary)
    (out_dir / 'nondud_projection.txt').write_text(summary)

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.20, color=_GRID)
    ax.axvline(0, color=_MUTED, lw=0.8, ls=':')
    ax.axhline(base_med, color=_MUTED, lw=1.0, ls='--',
               label=f'dud threshold ({base_med:.2f}%)')

    for feat, col, name in (('travel_upper', _BLUE, 'upper slope'),
                            ('travel_lower', _ORANGE, 'lower slope')):
        res, x = fits[feat]
        ax.scatter(x, y, s=16, alpha=0.45, color=col, edgecolors='none',
                   label=f'{name}:  y = {res.intercept:.2f} '
                         f'{res.slope:+.2f}x   (r={res.rvalue:+.3f})')
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, res.intercept + res.slope * xs, color=col, lw=2.0)

    ax.set_xlabel('Trendline travel across window  (fraction of price range)',
                  color=_MUTED, fontsize=10)
    ax.set_ylabel('|4hr forward move|  (%)', color=_MUTED, fontsize=10)
    ax.set_title(f'Non-dud wedges (n={len(nondud):,}): 4hr move size vs wedge slopes '
                 f'— linear projection', color=_TEXT, fontsize=12)
    ax.legend(facecolor='#222233', labelcolor=_TEXT, fontsize=9, framealpha=0.85)

    out_png = out_dir / 'nondud_projection.png'
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nChart saved: {out_png}')


if __name__ == '__main__':
    main()
