"""
vol_control_study.py

Follow-up to slope_move_study.py, answering two questions:

1. VOL CONTROL — does "steeper-down wedge -> bigger following move" survive
   normalising each move by trailing realized volatility? If yes, wedge shape
   carries real information; if it flattens, slope was proxying regime vol.
   Normalisation: z = |fwd return| / (trailing 1-day sigma of 1-min returns
   * sqrt(horizon)), i.e. move size in units of "expected move given recent
   vol".

2. DUD RATE — how often does a detected wedge produce a SMALLER following
   move than a random moment in the archive? Baseline = |fwd return| (and z)
   sampled at every stride-th bar. Under "wedges say nothing about what
   follows", ~50% of events would fall below the baseline median.

Usage
-----
  python vol_control_study.py --csv ../spy_1min_backtest.csv \
      --events ../runs_v2/window_250bar/models/slope_study/events.csv \
      --output-dir ../runs_v2/window_250bar/models/slope_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = {'30m': 30, '1hr': 60, '4hr': 240}
VOL_WIN  = 390          # trailing bars for realized vol (~1 RTH day)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',        required=True)
    ap.add_argument('--events',     required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--baseline-stride', type=int, default=10)
    args = ap.parse_args()

    df     = pd.read_csv(args.csv, usecols=['date', 'close'], parse_dates=['date'])
    closes = df['close'].values.astype(float)
    n      = len(closes)
    ev     = pd.read_csv(args.events)

    # ── Trailing realized vol per bar (fraction, per 1-min bar) ──────────────
    r1 = pd.Series(closes).pct_change()
    trail_vol = r1.rolling(VOL_WIN).std().values          # sigma per 1-min bar

    # ── Forward |moves| and vol-normalised z for all bars (baseline) ─────────
    lines = ['VOL-CONTROLLED SLOPE STUDY + DUD RATES', '=' * 74,
             f'events={len(ev):,}   vol window={VOL_WIN} bars   '
             f'baseline stride={args.baseline_stride}']

    ev['mid_travel'] = (ev.travel_upper + ev.travel_lower) / 2
    quint = pd.qcut(ev.mid_travel, 5,
                    labels=['steep down', 'down', 'flat', 'up', 'steep up'])

    for lbl, h in HORIZONS.items():
        fwd = np.full(n, np.nan)
        fwd[:n - h] = np.abs(closes[h:] / closes[:-h] - 1.0) * 100   # |ret| %

        # expected move given trailing vol: sigma*sqrt(h) in %
        expected = trail_vol * np.sqrt(h) * 100
        z_all    = fwd / expected

        # baseline sample: all valid bars at stride
        idx  = np.arange(VOL_WIN, n - h, args.baseline_stride)
        base_raw = fwd[idx]
        base_z   = z_all[idx]
        base_raw = base_raw[~np.isnan(base_raw)]
        base_z   = base_z[~np.isnan(base_z)]

        # events
        e_idx  = ev['end_idx'].values.astype(int)
        ok     = e_idx < (n - h)
        ev_raw = fwd[e_idx[ok]]
        ev_z   = z_all[e_idx[ok]]

        med_raw, med_z = np.nanmedian(base_raw), np.nanmedian(base_z)
        dud_raw = float(np.nanmean(ev_raw < med_raw)) * 100
        dud_z   = float(np.nanmean(ev_z   < med_z))   * 100

        lines.append(f'\n[{lbl}]')
        lines.append(f'  baseline: median |move|={med_raw:.3f}%   median z={med_z:.3f}')
        lines.append(f'  events  : median |move|={np.nanmedian(ev_raw):.3f}%   '
                     f'median z={np.nanmedian(ev_z):.3f}')
        lines.append(f'  DUD RATE (event move below baseline median): '
                     f'raw={dud_raw:.1f}%   vol-adjusted={dud_z:.1f}%   '
                     f'(null: 50%)')

        # vol-controlled quintile table + correlation
        ev.loc[ok, f'z_{lbl}'] = ev_z
        tq = ev.loc[ok].groupby(quint[ok], observed=True)[f'z_{lbl}'].mean()
        travel = ev.loc[ok, 'mid_travel'].astype(float)
        zs     = ev.loc[ok, f'z_{lbl}'].astype(float)
        m      = zs.notna()
        r_z    = float(np.corrcoef(travel[m], zs[m])[0, 1])
        lines.append(f'  vol-adjusted move (z) by slope quintile '
                     f'[r(travel,z)={r_z:+.3f}]:')
        for name, v in tq.items():
            lines.append(f'    {name:<11} {v:.3f}')

    out = Path(args.output_dir) / 'vol_control.txt'
    report = '\n'.join(lines)
    print(report)
    out.write_text(report)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
