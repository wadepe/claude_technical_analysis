"""
analyze_detections.py

Acceptance tests 2 and 3 for a wedge model: does it fire like a pattern
detector, or like a clock?

The held-out corpus eval (evaluate_cnn.py + analyze_v2_families.py) is
acceptance test 1 and it is saturated -- v3 scored ROC AUC 0.9993 there and
still behaved badly on real bars: 3 events/yr, 66% of them inside two
session-minute bins, 29.3% of detections shaped like megaphones. Corpus
metrics cannot see any of that, because the corpus has no clock and no
overnight gaps. This script measures the model on the real archive instead.

  Test 2  Detection rate by SESSION MINUTE, as a rate (detections per window
          scanned in that bin), not a raw count. A pattern detector should be
          roughly flat; v2 swung 228x between its busiest and quietest bin.

  Test 3  N random detections rendered as candlestick charts, with the fitted
          envelope and overnight-gap boundaries drawn in, so they can be
          LOOKED AT. Every real problem in this project was found by looking
          at charts, never by a summary statistic.

Also reports the specific v3 pathologies so v4 can be compared against them
directly: envelope convergence of detections vs the background population,
the diverging (megaphone-shaped) fraction, and the overnight-gap enrichment.

Geometry is fitted over the right-hand --fit-bars only, matching
live_monitor's fit_bars for the v3+ models. With entry context in the corpus,
fitting all 250 bars measures the approach rather than the wedge (r = -0.074
vs +0.510 at 120 bars), so a whole-window fit here would mis-measure the
model on exactly the axis that matters.

Usage
-----
  python analyze_detections.py \
      --csv ../spy_1min_backtest_regular.csv \
      --weights /data/runs_v4/window_250bar/models/cnn_best.weights.h5 \
      --output-dir /data/runs_v4/window_250bar/models/detection_study

Use --stride 1 for the definitive run. The default stride of 7 is coprime
with the 390-bar session so window ends still cycle through every session
minute; a stride sharing a factor with 390 (2, 3, 5, 10, 13 ...) would pin
every window end to a fixed lattice of minutes and leave most bins empty,
manufacturing exactly the concentration this test is looking for.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import gcd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import build_model, N_BARS
from scan_bitcoin import normalise_batch
from slope_move_study import cluster_events
from classify_wedge import fit_wedge_lines

DATE_COL     = 'date'
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']

# Project dark theme (matches slope_move_study / plot_daily)
_FIG_BG, _AX_BG = '#1a1a2e', '#13131f'
_GRID,  _SPINE  = '#2a2a3e', '#333345'
_MUTED, _TEXT   = '#8a8aa0', '#d0d0e8'
_BLUE,  _ORANGE = '#42A5F5', '#FF9800'
_UP,    _DOWN   = '#26A69A', '#EF5350'


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(colors=_MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(_SPINE)


# =============================================================================
# Session minutes
# =============================================================================

def session_minutes(dates: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Minute-of-session for every bar, 0 = first bar of the trading day.

    Derived from the clock, not from position in the file, so half days and
    any missing bars stay correctly aligned. The session anchor is read off
    the data rather than assumed: the archive is written in the dev box's
    LOCAL time (07:30-13:59 = 09:30-15:59 ET), so a hardcoded 9:30 would put
    every bar 120 minutes off.
    """
    ts     = pd.DatetimeIndex(dates)
    tod    = (ts.hour * 60 + ts.minute).to_numpy()
    day    = ts.normalize().to_numpy()
    anchor = int(pd.Series(tod).groupby(day).min().mode().iloc[0])
    return tod - anchor, anchor


# =============================================================================
# Scan
# =============================================================================

def scan_windows(
    df: pd.DataFrame,
    model,
    threshold: float,
    stride: int,
    batch_size: int,
) -> tuple[list[dict], np.ndarray]:
    """
    Slide the model over the archive.

    Returns (positives, all_starts). all_starts is every window scanned --
    the DENOMINATOR for the by-minute rate. Without it a by-minute histogram
    of detections only shows how many windows happened to end in each bin.
    """
    data  = df[FEATURE_COLS].values.astype(np.float32)
    dates = df[DATE_COL].values

    all_starts = np.arange(0, len(data) - N_BARS + 1, stride)
    n_windows  = len(all_starts)
    print(f'\nScanning {n_windows:,} windows  (stride={stride}, '
          f'threshold={threshold})')

    positives: list[dict] = []
    t0 = time.time()

    for batch_i in range(0, n_windows, batch_size):
        starts = all_starts[batch_i: batch_i + batch_size]
        idx    = starts[:, None] + np.arange(N_BARS)[None, :]
        X      = normalise_batch(data[idx])

        scores = model.predict(X, batch_size=512, verbose=0).squeeze()
        if scores.ndim == 0:
            scores = scores.reshape(1)

        for local_i in np.where(scores >= threshold)[0]:
            s = int(starts[local_i])
            positives.append({
                'start_idx' : s,
                'end_idx'   : s + N_BARS - 1,
                'score'     : float(scores[local_i]),
                'date_start': pd.Timestamp(dates[s]),
                'date_end'  : pd.Timestamp(dates[s + N_BARS - 1]),
            })

        done = batch_i + len(starts)
        step = n_windows // 10 + 1
        if (done // step) != ((done - len(starts)) // step):
            el   = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f'  {done:>9,}/{n_windows:,} ({done/n_windows*100:>3.0f}%)  '
                  f'hits: {len(positives):,}  '
                  f'ETA {max(n_windows-done,0)/max(rate,1e-9)/60:.1f} min')

    print(f'\nScan complete in {(time.time()-t0)/60:.1f} min  --  '
          f'{len(positives):,} positive windows')
    return positives, all_starts


# =============================================================================
# Geometry
# =============================================================================

def window_geometry(data: np.ndarray, start: int, fit_bars: int | None) -> dict:
    """Envelope fit over the right-hand fit_bars, matching live_monitor."""
    full = data[start: start + N_BARS]
    arr  = full if not fit_bars else full[-min(int(fit_bars), full.shape[0]):]
    return fit_wedge_lines(arr)


def count_gaps(day_ids: np.ndarray, start: int) -> int:
    """Overnight boundaries inside a window (0 = window sits within one day)."""
    d = day_ids[start: start + N_BARS]
    return int((d[1:] != d[:-1]).sum())


# =============================================================================
# Charts
# =============================================================================

def _draw_candles(ax, o, h, l, c) -> None:
    x  = np.arange(len(o))
    up = c >= o
    dn = ~up
    ax.vlines(x[up], l[up], h[up], color=_UP,   linewidth=0.5)
    ax.vlines(x[dn], l[dn], h[dn], color=_DOWN, linewidth=0.5)
    ax.vlines(x[up], o[up], c[up], color=_UP,   linewidth=1.6)
    ax.vlines(x[dn], o[dn], c[dn], color=_DOWN, linewidth=1.6)


def _draw_window(ax, win: np.ndarray, dates: np.ndarray, geo: dict,
                 fit_bars: int | None, title: str) -> None:
    o, h, l, c = win[:, 0], win[:, 1], win[:, 2], win[:, 3]
    _draw_candles(ax, o, h, l, c)

    # Envelope over the fitted span only -- drawing it across the whole
    # window would imply the fit saw bars it never touched.
    n   = len(o)
    fb  = n if not fit_bars else min(int(fit_bars), n)
    off = n - fb
    xf  = np.arange(fb)
    ax.plot(xf + off, geo['a_upper'] + geo['b_upper'] * xf,
            color=_ORANGE, linewidth=1.1, alpha=0.9)
    ax.plot(xf + off, geo['a_lower'] + geo['b_lower'] * xf,
            color=_BLUE, linewidth=1.1, alpha=0.9)
    if off:
        ax.axvline(off, color=_MUTED, linewidth=0.7, linestyle=':', alpha=0.8)

    # Overnight gaps: a 250-bar window stitched across days spans them, and
    # v3 fired 5x more often on windows that did.
    day = pd.DatetimeIndex(dates).normalize().to_numpy()
    for b in np.where(day[1:] != day[:-1])[0] + 1:
        ax.axvline(b, color=_DOWN, linewidth=0.7, linestyle='--', alpha=0.45)

    ax.set_title(title, color=_TEXT, fontsize=7.5)
    _style(ax)


def render_contact_sheet(events, data, dates, fit_bars, out_path, cols=4):
    rows = int(np.ceil(len(events) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.9),
                             facecolor=_FIG_BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(events):]:
        ax.set_visible(False)

    for ax, ev in zip(axes, events):
        s    = ev['start_idx']
        geo  = ev['geometry']
        conv = geo['convergence']
        tag  = 'DIVERGING' if conv < 0 else 'converging'
        _draw_window(
            ax, data[s: s + N_BARS], dates[s: s + N_BARS], geo, fit_bars,
            f"{ev['date_end']:%Y-%m-%d %H:%M}  score={ev['score']:.3f}\n"
            f"conv={conv:+.3f} ({tag})  sess_min={ev['session_minute']}  "
            f"gaps={ev['n_gaps']}",
        )

    fig.suptitle('Random detections -- look at these', color=_TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, facecolor=_FIG_BG)
    plt.close(fig)
    print(f'  Saved: {out_path}')


def render_single(ev, data, dates, fit_bars, out_path):
    s    = ev['start_idx']
    win  = data[s: s + N_BARS]
    geo  = ev['geometry']
    conv = geo['convergence']
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True, facecolor=_FIG_BG,
        gridspec_kw={'height_ratios': [3, 1]})
    _draw_window(
        ax, win, dates[s: s + N_BARS], geo, fit_bars,
        f"{ev['date_end']:%Y-%m-%d %H:%M}   score={ev['score']:.4f}   "
        f"convergence={conv:+.4f} "
        f"({'DIVERGING' if conv < 0 else 'converging'})   "
        f"session_minute={ev['session_minute']}   "
        f"overnight_gaps={ev['n_gaps']}",
    )
    ax.set_ylabel('price', color=_MUTED, fontsize=8)

    axv.bar(np.arange(N_BARS), win[:, 4], color=_BLUE, alpha=0.65, width=0.9)
    axv.set_ylabel('volume', color=_MUTED, fontsize=8)
    axv.set_xlabel('bar in window', color=_MUTED, fontsize=8)
    _style(axv)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=_FIG_BG)
    plt.close(fig)


# =============================================================================
# Reports
# =============================================================================

def by_minute_report(pos_minutes: np.ndarray, win_minutes: np.ndarray,
                     bin_size: int) -> str:
    """Detection rate per window scanned, bucketed by minute of session."""
    n_bins = int(np.ceil((win_minutes.max() + 1) / bin_size))
    edges  = np.arange(n_bins + 1) * bin_size

    win_hist = np.histogram(win_minutes, bins=edges)[0]
    pos_hist = np.histogram(pos_minutes, bins=edges)[0]

    # Bins with no windows scanned have no rate -- reporting 0 there would
    # read as "the model never fires then" when nothing was ever asked.
    rate = np.divide(pos_hist, win_hist,
                     out=np.full(n_bins, np.nan), where=win_hist > 0)

    lines = ['', 'TEST 2 -- DETECTION RATE BY SESSION MINUTE', '=' * 78,
             f'bin size {bin_size} min   '
             f'(session minute 0 = first bar of the regular session)', '',
             f'  {"bin":<14}{"windows":>12}{"detections":>12}'
             f'{"rate":>12}{"vs mean":>10}']
    lines.append('  ' + '-' * 60)

    overall = pos_hist.sum() / max(win_hist.sum(), 1)
    for i in range(n_bins):
        if win_hist[i] == 0:
            continue
        rel = rate[i] / overall if overall > 0 else float('nan')
        lines.append(f'  {edges[i]:>4}-{edges[i+1]-1:<9}{win_hist[i]:>12,}'
                     f'{pos_hist[i]:>12,}{rate[i]:>12.5f}{rel:>9.2f}x')

    valid = rate[np.isfinite(rate)]
    nz    = valid[valid > 0]
    swing = (valid.max() / nz.min()) if len(nz) else float('inf')
    order = np.argsort(np.nan_to_num(rate, nan=-1.0))[::-1]
    top2  = pos_hist[order[:2]].sum() / max(pos_hist.sum(), 1)

    n_zero = int((valid == 0).sum())
    lines += [
        '',
        f'  overall rate      : {overall:.5f} detections/window',
        f'  busiest/quietest  : {swing:.1f}x'
        f'   (v2 swung 228x, a flat detector is ~1x)',
        f'    ratio is over bins with a NONZERO rate; {n_zero} of '
        f'{len(valid)} bins fired zero times and would divide by zero.'
        f'{"  A model this sparse makes the ratio a weak summary -- read the table." if n_zero > len(valid) // 2 else ""}',
        f'  share in top 2 bins: {top2*100:.1f}%'
        f'   (v3 put 66% in two bins; even coverage would be '
        f'{2.0/max(len(valid),1)*100:.1f}%)',
    ]
    return '\n'.join(lines)


def geometry_report(events: list[dict], pop_conv: np.ndarray,
                    n_years: float, gap_pos: float, gap_pop: float) -> str:
    """
    The v3 pathologies, measured in the SAME units they were recorded in.

    v3 was written up in COMPRESSION (width_end / width_start): detections
    0.284 against a population median of 0.964. fit_wedge_lines returns
    CONVERGENCE, which is 1 - compression, so the two read in opposite
    directions -- printing one under the other's reference numbers inverts
    the conclusion. Both are shown here, and the v3 figures are labelled.
    """
    conv = np.array([e['geometry']['convergence'] for e in events])
    comp     = 1.0 - conv
    pop_comp = 1.0 - pop_conv
    lines = ['', 'V3 PATHOLOGY CHECK (v3 reference numbers labelled inline)',
             '=' * 78,
             f'  events                : {len(events):,}  '
             f'({len(events)/max(n_years,1e-9):.1f}/yr, v3 was 3/yr)']
    if len(conv):
        lines += [
            f'  compression median    : {np.median(comp):.4f}   '
            f'(population {np.median(pop_comp):.4f})',
            f'    width_end/width_start; LOWER = more compressed.',
            f'    v3 read 0.284 against a population median of 0.964 -- it',
            f'    selected for compression even harder than v2, because only',
            f'    compression_walk got the volatility profile and the positives',
            f'    did not, making decay a negative marker. v4 gives every family',
            f'    the same profile, so v4 should sit far closer to the',
            f'    population than 0.284 did.',
            f'  convergence median    : {np.median(conv):+.4f}   '
            f'(population {np.median(pop_conv):+.4f})   [= 1 - compression]',
            f'  DIVERGING (conv < 0)  : {float((conv < 0).mean())*100:.1f}%   '
            f'(population {float((pop_conv < 0).mean())*100:.1f}%,  '
            f'v3 was 29.3%)',
            f'    A wedge detector firing on megaphones is wrong in absolute',
            f'    terms even when it is not enriched over the population.',
        ]
    # Enrichment is capped at 1/base-rate, and the base rate here is high
    # (a 250-bar window is shorter than a 390-bar session, so most windows
    # straddle a boundary). The odds ratio is the scale-free version; the
    # "5x" recorded for v3 cannot be a rate ratio on this quantity at this
    # base rate, so it is reported without a direct comparison.
    odds = ((gap_pos / max(1 - gap_pos, 1e-9)) /
            max(gap_pop / max(1 - gap_pop, 1e-9), 1e-9))
    lines += [
        f'  gap-spanning share    : {gap_pos*100:.1f}%  '
        f'(population {gap_pop*100:.1f}%)',
        f'    rate ratio {gap_pos/max(gap_pop,1e-9):.2f}x  '
        f'(max possible {1/max(gap_pop,1e-9):.2f}x)   '
        f'odds ratio {odds:.2f}x',
    ]
    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description='Wedge model acceptance tests 2 & 3')
    p.add_argument('--csv',        required=True,
                   help='Regular-session archive (spy_1min_backtest_regular.csv)')
    p.add_argument('--weights',    required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--threshold',  type=float, default=0.8)
    p.add_argument('--stride',     type=int,   default=7,
                   help='Must be coprime with 390 (7, 11, 17...). 1 = exact.')
    p.add_argument('--fit-bars',   type=int,   default=120,
                   help='Envelope fit span from the right edge (live: 120)')
    p.add_argument('--batch-size', type=int,   default=5_000)
    p.add_argument('--n-charts',   type=int,   default=20)
    p.add_argument('--bin-size',   type=int,   default=30,
                   help='Session-minute bin width for test 2')
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--pop-sample', type=int,   default=20_000,
                   help='Random windows for the background geometry baseline')
    args = p.parse_args()

    # Coprimality, not divisibility: stride 4 does not divide 390 but shares a
    # factor of 2 with it, so window ends only ever land on even session
    # minutes and half the bins stay empty. gcd is the correct test.
    g = gcd(args.stride, 390)
    if args.stride != 1 and g != 1:
        print(f'WARNING: stride {args.stride} shares factor {g} with the '
              f'390-bar session, so window ends reach only {390 // g} of 390 '
              f'session minutes and the rest will look like bins the model '
              f'never fires in. Use 1, 7, 11, 17, 19, 23 ...')

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading {Path(args.csv).name} ...')
    df = pd.read_csv(args.csv, usecols=[DATE_COL] + FEATURE_COLS)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f'  {len(df):,} rows  |  {df[DATE_COL].iloc[0]} to '
          f'{df[DATE_COL].iloc[-1]}')

    dates   = df[DATE_COL].values
    data    = df[FEATURE_COLS].values.astype(np.float32)
    minutes, anchor = session_minutes(dates)
    day_ids = pd.DatetimeIndex(dates).normalize().to_numpy()
    n_years = (df[DATE_COL].iloc[-1] - df[DATE_COL].iloc[0]).days / 365.25
    print(f'  session anchor {anchor // 60:02d}:{anchor % 60:02d} local  |  '
          f'session minutes 0-{minutes.max()}  |  {n_years:.1f} years')

    print(f'\nLoading weights: {args.weights}')
    model = build_model(print_summary=False)
    model.load_weights(args.weights)

    positives, all_starts = scan_windows(
        df, model, args.threshold, args.stride, args.batch_size)

    all_ends    = all_starts + N_BARS - 1
    win_minutes = minutes[all_ends]

    report = [f'DETECTION STUDY  ({Path(args.weights).parent.parent.name}, '
              f'weights={Path(args.weights).name})',
              f'archive={Path(args.csv).name}  threshold={args.threshold}  '
              f'stride={args.stride}  fit_bars={args.fit_bars}',
              '=' * 78]

    if not positives:
        report.append('\nNo detections at this threshold. Nothing to test.')
        text = '\n'.join(report)
        print('\n' + text)
        (out / 'detection_study.txt').write_text(text)
        return

    pos_minutes = minutes[np.array([p['end_idx'] for p in positives])]
    report.append(by_minute_report(pos_minutes, win_minutes, args.bin_size))

    # Events: one per overlapping detection group, so a single wedge counts
    # once. Raw positive windows over-count events ~8x.
    events = cluster_events(positives)
    print(f'\nClustered {len(positives):,} windows -> {len(events):,} events')
    for e in events:
        e['geometry']        = window_geometry(data, e['start_idx'], args.fit_bars)
        e['session_minute']  = int(minutes[e['end_idx']])
        e['n_gaps']          = count_gaps(day_ids, e['start_idx'])

    # Background geometry baseline from random windows.
    rng     = np.random.default_rng(args.seed)
    sample  = rng.choice(all_starts, size=min(args.pop_sample, len(all_starts)),
                         replace=False)
    print(f'Fitting geometry on {len(sample):,} background windows ...')
    pop_conv = np.array([window_geometry(data, int(s), args.fit_bars)['convergence']
                         for s in sample])
    pop_gap  = float(np.mean([count_gaps(day_ids, int(s)) > 0 for s in sample]))
    ev_gap   = float(np.mean([e['n_gaps'] > 0 for e in events]))

    report.append(geometry_report(events, pop_conv, n_years, ev_gap, pop_gap))

    # ── Test 3: charts ───────────────────────────────────────────────────────
    pick   = rng.choice(len(events), size=min(args.n_charts, len(events)),
                        replace=False)
    chosen = [events[int(i)] for i in sorted(pick)]
    charts = out / 'charts'
    charts.mkdir(exist_ok=True)
    print(f'\nRendering {len(chosen)} random detections ...')
    render_contact_sheet(chosen, data, dates, args.fit_bars,
                         out / 'detections_contact_sheet.png')
    for i, ev in enumerate(chosen):
        render_single(ev, data, dates, args.fit_bars,
                      charts / f'detection_{i:02d}_'
                               f'{ev["date_end"]:%Y%m%d_%H%M}.png')

    report += ['', 'TEST 3 -- RENDERED DETECTIONS', '=' * 78,
               f'  {len(chosen)} random events (seed {args.seed})',
               f'  contact sheet : {out / "detections_contact_sheet.png"}',
               f'  individual    : {charts}',
               '  LOOK AT THEM. Dashed red = overnight gap, dotted grey =',
               '  start of the fitted span, orange/blue = fitted envelope.']

    # Event table for follow-up work.
    pd.DataFrame([{
        'date_start'    : e['date_start'],
        'date_end'      : e['date_end'],
        'score'         : e['score'],
        'session_minute': e['session_minute'],
        'n_gaps'        : e['n_gaps'],
        'convergence'   : e['geometry']['convergence'],
        'travel_upper'  : e['geometry']['travel_upper'],
        'travel_lower'  : e['geometry']['travel_lower'],
        'width_start'   : e['geometry']['width_start'],
        'width_end'     : e['geometry']['width_end'],
    } for e in events]).to_csv(out / 'events.csv', index=False)

    text = '\n'.join(report)
    print('\n' + text)
    (out / 'detection_study.txt').write_text(text)
    (out / 'summary.json').write_text(json.dumps({
        'weights'        : str(args.weights),
        'threshold'      : args.threshold,
        'stride'         : args.stride,
        'fit_bars'       : args.fit_bars,
        'n_windows'      : int(len(all_starts)),
        'n_positives'    : len(positives),
        'n_events'       : len(events),
        'events_per_year': len(events) / max(n_years, 1e-9),
        'diverging_frac' : float(np.mean(
            [e['geometry']['convergence'] < 0 for e in events])),
        # Compression = width_end/width_start = 1 - convergence. Recorded in
        # v3's units so runs stay comparable across model versions.
        'compression_median'     : float(np.median(
            [1.0 - e['geometry']['convergence'] for e in events])),
        'compression_median_pop' : float(np.median(1.0 - pop_conv)),
        'gap_share'      : ev_gap,
        'gap_share_pop'  : pop_gap,
        'gap_rate_ratio' : ev_gap / max(pop_gap, 1e-9),
    }, indent=2))
    print(f'\nSaved: {out / "detection_study.txt"}, events.csv, summary.json')


if __name__ == '__main__':
    main()
