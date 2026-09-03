"""
filter_detections.py

Quality-filter a detection study's events, then measure what survives.

analyze_detections.py answers "when does the model fire". This answers "is what
it fired on actually a wedge", by two tests the eye applies immediately but no
summary statistic in this project has yet measured:

  DIVERGING   the fitted envelope opens out instead of closing. A wedge that
              widens is a megaphone; it is wrong by definition, not by degree.

  ESCAPES     price does not stay inside the lines that were drawn for it. A
              formation you cannot contain is not that formation -- the same
              principle the corpus applies to its own families via touch
              enforcement. Measured as excursion beyond the envelope in units
              of the envelope's own mean width, so it is scale-free.

Both are computed on the SAME 120-bar fit live_monitor would use, so a rejected
event is one live_monitor would also have mis-described.

What survives is then given forward returns at 30m/1h/2h/4h against a random
baseline drawn from the same archive, which is the only way to tell whether
filtering bought anything. Twenty survivors are rendered for inspection.

Usage
-----
  python filter_detections.py --csv ../spy_1min_backtest_regular.csv \
      --events /data/runs_v5/window_250bar/models/detection_study/events.csv \
      --output-dir /data/runs_v5/window_250bar/models/filtered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import N_BARS
from classify_wedge import fit_wedge_lines

DATE_COL = 'date'
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume']
# The archive is stitched regular sessions, so 390 bars = one trading day.
# Horizons past 4hr were added because the 4hr excess turned out to be a
# volatility artefact -- the coil may simply resolve on a longer scale than
# a 250-bar window ever looks at.
HORIZONS = [30, 60, 120, 240, 390, 780, 1950]
LABELS = ['30m', '1hr', '2hr', '4hr', '1day', '2day', '5day']
BUCKET_LABELS = ['30m', '4hr', '1day', '5day']   # keeps the bucket table readable

_FIG_BG, _AX_BG = '#1a1a2e', '#13131f'
_GRID, _SPINE = '#2a2a3e', '#333345'
_MUTED, _TEXT = '#8a8aa0', '#d0d0e8'
_BLUE, _ORANGE = '#42A5F5', '#FF9800'
_UP, _DOWN = '#26A69A', '#EF5350'


def envelope_metrics(win: np.ndarray, fit_bars: int | None) -> dict:
    """
    Refit the envelope over the right-hand fit_bars and measure containment.

    Excursions are normalised by the envelope's own mean width, so a wide
    formation and a tight one are judged on the same scale. `frac_outside`
    counts bars whose high/low leaves the envelope at all; `max_excursion`
    is the worst single breach.
    """
    fb  = win.shape[0] if not fit_bars else min(int(fit_bars), win.shape[0])
    arr = win[-fb:]
    g   = fit_wedge_lines(arr)

    x  = np.arange(fb, dtype=float)
    up = g['a_upper'] + g['b_upper'] * x
    lo = g['a_lower'] + g['b_lower'] * x
    width = up - lo
    mw = float(np.mean(np.abs(width)))
    if mw < 1e-12:
        mw = 1e-12

    over  = np.maximum(arr[:, 1] - up, 0.0)     # high above upper
    under = np.maximum(lo - arr[:, 2], 0.0)     # low below lower
    exc   = np.maximum(over, under) / mw

    # TOUCHES -- the test that actually matters. _envelope_fit regresses through
    # a quantile of the extremes, so a single spike can drag a line up until it
    # floats above the bulk of the range, touching nothing. Excursion cannot see
    # that (the line is ABOVE price, so nothing escapes it); touch count can.
    #
    # Same definition the corpus enforces on its own positives: a visit is a
    # maximal run of bars within CHANNEL_TOUCH_TOL_FRAC (0.18) of the boundary,
    # so hugging a line counts once, and 2 per side are required. Highs are
    # tested against the upper line and lows against the lower, which is how the
    # line reads on a chart.
    tol = 0.18 * np.abs(width)
    near_up = arr[:, 1] >= up - tol
    near_lo = arr[:, 2] <= lo + tol

    def _runs(mask: np.ndarray) -> int:
        m = mask.astype(np.int8)
        return int(np.sum(m - np.concatenate([[0], m[:-1]]) == 1))

    # How much of the total breach sits in the single worst bar. A lone spike
    # anchoring the fit concentrates it; genuine boundary-riding spreads it.
    tot = float(exc.sum())
    spike = float(exc.max() / tot) if tot > 1e-12 else 0.0

    return {
        'convergence'   : g['convergence'],
        'max_excursion' : float(exc.max()),
        'mean_excursion': float(exc.mean()),
        'frac_outside'  : float((exc > 0.02).mean()),
        'touch_up'      : _runs(near_up),
        'touch_lo'      : _runs(near_lo),
        'spike_share'   : spike,
        'a_upper': g['a_upper'], 'b_upper': g['b_upper'],
        'a_lower': g['a_lower'], 'b_lower': g['b_lower'],
    }


def forward_returns(close: np.ndarray, ends: np.ndarray) -> dict:
    """Signed and absolute forward return at each horizon, in percent."""
    out = {}
    for h, lab in zip(HORIZONS, LABELS):
        tgt = ends + h
        ok  = tgt < len(close)
        r = np.full(len(ends), np.nan)
        r[ok] = (close[tgt[ok]] - close[ends[ok]]) / close[ends[ok]] * 100.0
        out[lab] = r
    return out


def _summarise(name: str, fr: dict, n: int) -> list[str]:
    lines = [f'  {name:<22}n={n:>6,}']
    for lab in LABELS:
        v = fr[lab]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        lines.append(f'      {lab:<5} mean={v.mean():+.4f}%  '
                     f'median={np.median(v):+.4f}%  '
                     f'mean|move|={np.abs(v).mean():.4f}%  '
                     f'win={float((v > 0).mean())*100:.1f}%')
    return lines


def convergence_buckets(ev, close, step=0.1, min_n=20):
    """
    Forward returns of the survivors bucketed by convergence.

    The thesis worth testing: a more pinched wedge is nearer its apex, so it
    should be nearer a breakout and precede a larger move. If that holds,
    |move| rises across the buckets. If it does not, convergence carries no
    information about what happens next, whatever it says about shape.

    Thin buckets are flagged: at these effect sizes a handful of events swings
    one completely.
    """
    conv = ev['convergence'].values
    hi = float(np.nanmax(conv))
    edges = np.arange(0.0, np.ceil(hi / step) * step + step, step)

    lines = ['', 'FORWARD RETURNS BY CONVERGENCE (survivors only)', '=' * 78,
             '  convergence = 1 - width_end/width_start; HIGHER = more pinched',
             '',
             f'  {"bucket":<13}{"n":>6}'
             + ''.join(f'{lab + " |mv|":>12}' for lab in BUCKET_LABELS)
             + f'{"4hr win":>10}{"4hr mean":>12}']
    lines.append('  ' + '-' * 92)

    for i in range(len(edges) - 1):
        lo, up = edges[i], edges[i + 1]
        m = (conv >= lo) & (conv < up)
        n = int(m.sum())
        if n == 0:
            continue
        fr = forward_returns(close, ev.loc[m, 'end_idx'].values)
        cells = ''
        for lab in BUCKET_LABELS:
            v = fr[lab]; v = v[np.isfinite(v)]
            cells += f'{np.abs(v).mean():>12.4f}' if len(v) else f'{"-":>12}'
        v4 = fr['4hr']; v4 = v4[np.isfinite(v4)]
        win = f'{float((v4 > 0).mean()) * 100:>9.1f}%' if len(v4) else f'{"-":>10}'
        mean4 = f'{v4.mean():>+11.4f}%' if len(v4) else f'{"-":>12}'
        flag = '  (thin)' if n < min_n else ''
        lines.append(f'  {lo:.1f}-{up:.1f}{"":<5}{n:>6}{cells}{win}{mean4}{flag}')

    fr = forward_returns(close, ev['end_idx'].values)
    for lab in ('2hr', '4hr'):
        v = fr[lab]; ok = np.isfinite(v)
        if ok.sum() > 30:
            r = float(np.corrcoef(conv[ok], np.abs(v[ok]))[0, 1])
            lines.append(f'  corr(convergence, |{lab} move|) = {r:+.4f}'
                         f'   n={int(ok.sum()):,}')
    lines.append('  "Tighter wedge -> bigger break" predicts |move| rising '
                 'across buckets')
    lines.append('  and a clearly positive correlation.')
    return lines



def trailing_vol(close: np.ndarray, ends: np.ndarray,
                 lookback: int = N_BARS) -> np.ndarray:
    """
    Std of 1-bar percentage returns over the `lookback` bars ending at each
    index, in percent. This is the volatility regime the window sat in.
    """
    r = np.zeros_like(close)
    r[1:] = np.diff(close) / close[:-1] * 100.0
    r = r - r.mean()                      # conditioning, std is shift-invariant
    c1 = np.concatenate([[0.0], np.cumsum(r)])
    c2 = np.concatenate([[0.0], np.cumsum(r * r)])
    a = np.maximum(ends - lookback + 1, 0)
    b = ends + 1
    n = np.maximum(b - a, 1)
    s1 = c1[b] - c1[a]
    s2 = c2[b] - c2[a]
    return np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))


def normalised_returns(close: np.ndarray, ends: np.ndarray) -> dict:
    """
    Forward return expressed in units of what the prevailing volatility would
    predict for that horizon: z = ret_h / (sigma_1bar * sqrt(h)).

    A random walk gives E|z| = sqrt(2/pi) ~ 0.798 whatever the regime, so this
    strips out "the market was already moving" and leaves only "did price move
    further than its own volatility implied".
    """
    sig = trailing_vol(close, ends)
    out = {}
    for h, lab in zip(HORIZONS, LABELS):
        tgt = ends + h
        ok = (tgt < len(close)) & (sig > 1e-9)
        z = np.full(len(ends), np.nan)
        z[ok] = ((close[tgt[ok]] - close[ends[ok]]) / close[ends[ok]] * 100.0)                 / (sig[ok] * np.sqrt(h))
        out[lab] = z
    return out


def _summarise_z(name: str, zs: dict, n: int) -> list[str]:
    lines = [f'  {name:<22}n={n:>6,}']
    for lab in LABELS:
        v = zs[lab]; v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        lines.append(f'      {lab:<5} mean|z|={np.abs(v).mean():.4f}  '
                     f'median|z|={np.median(np.abs(v)):.4f}  '
                     f'mean z={v.mean():+.4f}  '
                     f'|z|>1: {float((np.abs(v) > 1).mean())*100:.1f}%')
    return lines


def plot_histograms(groups: dict, base: dict, out_path: Path,
                    unit: str, clip: float):
    """
    Distribution of forward outcomes per horizon, survivors against baseline.

    Summary statistics hide shape: a detector could match the baseline mean
    while having a fatter or thinner tail, which is exactly the question a
    "coiled spring" raises. Densities are drawn so the two are comparable
    despite very different n.
    """
    n = len(LABELS)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.1),
                             facecolor=_FIG_BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, lab in zip(axes, LABELS):
        bins = np.linspace(-clip, clip, 61)
        b = base[lab]; b = np.clip(b[np.isfinite(b)], -clip, clip)
        ax.hist(b, bins=bins, density=True, histtype='step',
                color=_MUTED, linewidth=1.4, label=f'random (n={len(b):,})')
        for (nm, col), g in zip((('CLEAN', _BLUE),), (groups['CLEAN'],)):
            v = g[lab]; v = np.clip(v[np.isfinite(v)], -clip, clip)
            ax.hist(v, bins=bins, density=True, histtype='stepfilled',
                    color=col, alpha=0.45, label=f'{nm} (n={len(v):,})')
            ax.axvline(v.mean(), color=col, linewidth=1.0, linestyle='--')
            ax.set_title(f'{lab}   CLEAN mean|{unit}|={np.abs(v).mean():.3f}   '
                         f'random={np.abs(b).mean():.3f}',
                         color=_TEXT, fontsize=8.5)
        ax.axvline(b.mean(), color=_MUTED, linewidth=1.0, linestyle='--')
        ax.legend(fontsize=6.5, facecolor=_AX_BG, edgecolor=_SPINE,
                  labelcolor=_TEXT)
        ax.set_xlabel(unit, color=_MUTED, fontsize=8)
        _style(ax)

    fig.suptitle(f'Forward outcome distributions by horizon ({unit})',
                 color=_TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130, facecolor=_FIG_BG)
    plt.close(fig)
    print(f'  saved {out_path}')


def _style(ax):
    ax.set_facecolor(_AX_BG)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.tick_params(colors=_MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(_SPINE)


def render(events: pd.DataFrame, data: np.ndarray, dates: np.ndarray,
           fit_bars: int | None, out_path: Path, title: str, cols: int = 4):
    n = len(events)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.9),
                             facecolor=_FIG_BG)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    for ax, (_, e) in zip(axes, events.iterrows()):
        s = int(e['start_idx'])
        win = data[s: s + N_BARS]
        o, h, l, c = win[:, 0], win[:, 1], win[:, 2], win[:, 3]
        x = np.arange(len(o)); up = c >= o; dn = ~up
        ax.vlines(x[up], l[up], h[up], color=_UP, linewidth=0.5)
        ax.vlines(x[dn], l[dn], h[dn], color=_DOWN, linewidth=0.5)
        ax.vlines(x[up], o[up], c[up], color=_UP, linewidth=1.5)
        ax.vlines(x[dn], o[dn], c[dn], color=_DOWN, linewidth=1.5)

        fb = N_BARS if not fit_bars else min(int(fit_bars), N_BARS)
        off = N_BARS - fb
        xf = np.arange(fb)
        ax.plot(xf + off, e['a_upper'] + e['b_upper'] * xf, color=_ORANGE,
                linewidth=1.1)
        ax.plot(xf + off, e['a_lower'] + e['b_lower'] * xf, color=_BLUE,
                linewidth=1.1)
        if off:
            ax.axvline(off, color=_MUTED, linewidth=0.7, linestyle=':', alpha=0.8)
        day = pd.DatetimeIndex(dates[s: s + N_BARS]).normalize().to_numpy()
        for b in np.where(day[1:] != day[:-1])[0] + 1:
            ax.axvline(b, color=_DOWN, linewidth=0.7, linestyle='--', alpha=0.45)

        ax.set_title(f"{pd.Timestamp(e['date_end']):%Y-%m-%d %H:%M}  "
                     f"score={e['score']:.3f}\n"
                     f"conv={e['convergence']:+.3f}  "
                     f"touch {int(e['touch_up'])}up/{int(e['touch_lo'])}lo"
                     + (f"  4hr={e['fwd_4hr']:+.3f}%"
                        if 'fwd_4hr' in e.index and pd.notna(e.get('fwd_4hr'))
                        else f"  maxexc={e['max_excursion']:.2f}"),
                     color=_TEXT, fontsize=7.5)
        _style(ax)

    fig.suptitle(title, color=_TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, facecolor=_FIG_BG)
    plt.close(fig)
    print(f'  saved {out_path}')


def main() -> None:
    p = argparse.ArgumentParser(description='Filter detections and measure survivors')
    p.add_argument('--csv', required=True)
    p.add_argument('--events', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--fit-bars', type=int, default=120)
    # CALIBRATED AGAINST THE CORPUS, not guessed. _envelope_fit is a regression
    # through the upper/lower 45% of residuals, NOT a bounding hull, so ~37% of
    # bars sit outside the line by construction -- measured at 36.7-37.5% for
    # every family including plain walks, which is why frac_outside carries no
    # information and defaults to disabled. Designed forming_wedge positives
    # breach by median 0.789 / p95 1.981 of mean envelope width under this same
    # fit, so 1.98 is "worse contained than 95% of actual designed wedges".
    p.add_argument('--max-excursion', type=float, default=1.98,
                   help='reject if worst breach exceeds this fraction of mean '
                        'width (corpus wedge p95 = 1.98)')
    p.add_argument('--max-frac-outside', type=float, default=1.01,
                   help='disabled by default: ~37%% of bars sit outside for '
                        'EVERY family, so this separates nothing')
    # CALIBRATED, not inherited. The corpus GENERATES with a floor of 2, but
    # designed forming_wedge positives actually achieve 5/6/7/9 at p5/p10/p25/
    # median per side under this same 120-bar fit, while plain walks achieve
    # 2/3/4/5. A detection with 2-3 touches is walk-like, not wedge-like: the
    # line is floating off the price action, anchored by a stray spike rather
    # than drawn on repeated visits. 5 is the corpus p10, so it rejects roughly
    # the worst-touched tenth of genuine wedges.
    # A channel touches its rails beautifully -- riding both is what makes it a
    # channel -- so the touch test structurally cannot catch one. Convergence is
    # the only test that does. The 0.0-0.1 bucket was visually all parallel
    # channels (one at conv=+0.002, literally parallel) and carried the worst
    # forward return of any bucket (-0.184% at 4hr, 50.0% win).
    p.add_argument('--min-convergence', type=float, default=0.5,
                   help='reject near-parallel formations (channel-like)')
    p.add_argument('--min-touches', type=int, default=5,
                   help='required distinct visits to EACH boundary '
                        '(designed-wedge p10 = 5; walks reach only 2-5)')
    p.add_argument('--n-charts', type=int, default=20)
    p.add_argument('--baseline', type=int, default=20_000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, usecols=[DATE_COL] + FEATURE_COLS)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    data  = df[FEATURE_COLS].values.astype(np.float32)
    dates = df[DATE_COL].values
    close = data[:, 3].astype(np.float64)
    pos   = pd.Series(np.arange(len(df)), index=df[DATE_COL])

    ev = pd.read_csv(args.events, parse_dates=['date_start', 'date_end'])
    ev['start_idx'] = ev['date_start'].map(pos).astype('Int64')
    ev = ev.dropna(subset=['start_idx']).copy()
    ev['start_idx'] = ev['start_idx'].astype(int)
    ev['end_idx'] = ev['start_idx'] + N_BARS - 1
    print(f'events: {len(ev):,}')

    print('measuring containment ...')
    m = [envelope_metrics(data[s: s + N_BARS], args.fit_bars)
         for s in ev['start_idx'].values]
    for k in ('convergence', 'max_excursion', 'mean_excursion', 'frac_outside',
              'touch_up', 'touch_lo', 'spike_share',
              'a_upper', 'b_upper', 'a_lower', 'b_lower'):
        ev[k] = [d[k] for d in m]

    diverging = ev['convergence'] < 0
    flat      = (~diverging) & (ev['convergence'] < args.min_convergence)
    untouched = (~diverging) & (~flat) & (
                    (ev['touch_up'] < args.min_touches) |
                    (ev['touch_lo'] < args.min_touches))
    escapes   = (~diverging) & (~flat) & (~untouched) & (
                    (ev['max_excursion'] > args.max_excursion) |
                    (ev['frac_outside'] > args.max_frac_outside))
    clean     = ~(diverging | flat | untouched | escapes)
    n = len(ev)

    rep = [f'DETECTION QUALITY FILTER  ({Path(args.events).parent.name})',
           f'fit_bars={args.fit_bars}  max_excursion={args.max_excursion}  '
           f'max_frac_outside={args.max_frac_outside}', '=' * 78, '',
           f'  {"category":<34}{"events":>9}{"share":>9}',
           '  ' + '-' * 52,
           f'  {"DIVERGING (envelope opens out)":<34}{int(diverging.sum()):>9,}'
           f'{diverging.mean()*100:>8.1f}%',
           f'  {"NEAR-PARALLEL (channel-like)":<34}{int(flat.sum()):>9,}'
           f'{flat.mean()*100:>8.1f}%',
           f'  {"UNTOUCHED (line floats off price)":<34}{int(untouched.sum()):>9,}'
           f'{untouched.mean()*100:>8.1f}%',
           f'  {"ESCAPES (price leaves the lines)":<34}{int(escapes.sum()):>9,}'
           f'{escapes.mean()*100:>8.1f}%',
           f'  {"CLEAN (survives both)":<34}{int(clean.sum()):>9,}'
           f'{clean.mean()*100:>8.1f}%',
           f'  {"TOTAL":<34}{n:>9,}{100.0:>8.1f}%', '',
           '  excursion distribution over ALL events '
           '(worst breach, as a fraction of mean envelope width):']
    q = ev['max_excursion'].quantile([.1, .25, .5, .75, .9, .95]).to_dict()
    rep.append('    ' + '  '.join(f'p{int(k*100)}={v:.3f}' for k, v in q.items()))
    rep.append(f'    frac_outside median={ev["frac_outside"].median()*100:.1f}%'
               f'  p90={ev["frac_outside"].quantile(0.9)*100:.1f}%')

    # ── Forward returns ──────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    base_ends = rng.choice(np.arange(N_BARS, len(close) - max(HORIZONS) - 1),
                           size=min(args.baseline, len(close) // 4),
                           replace=False)
    rep += ['', 'FORWARD RETURNS from the window end', '=' * 78,
            '  signed mean/median, plus mean absolute move and win rate']
    for name, mask in (('CLEAN', clean), ('NEAR-PARALLEL', flat),
                       ('UNTOUCHED', untouched), ('ESCAPES', escapes),
                       ('DIVERGING', diverging)):
        if mask.sum() == 0:
            continue
        fr = forward_returns(close, ev.loc[mask, 'end_idx'].values)
        rep += _summarise(name, fr, int(mask.sum()))
    rep += _summarise('RANDOM BASELINE', forward_returns(close, base_ends),
                      len(base_ends))

    # Direction split on the survivors -- the project's one robust result is
    # that down-sloping wedges precede bigger moves, so keep it visible.
    if clean.sum() > 20:
        mid = (ev.loc[clean, 'travel_upper'] + ev.loc[clean, 'travel_lower']) / 2
        for nm, sub in (('CLEAN down-sloping', mid < 0), ('CLEAN up-sloping', mid > 0)):
            idx = ev.loc[clean].index[sub.values]
            if len(idx) > 5:
                rep += _summarise(nm, forward_returns(close, ev.loc[idx, 'end_idx'].values),
                                  len(idx))

    rep += ['', 'VOLATILITY-NORMALISED FORWARD RETURNS', '=' * 78,
            '  z = return / (trailing 250-bar sigma x sqrt(horizon)).',
            '  A random walk gives mean|z| = 0.798 regardless of regime, so a',
            '  detector that only finds volatile periods scores ~the baseline',
            '  here even though its raw moves look large.']
    for name, mask in (('CLEAN', clean), ('NEAR-PARALLEL', flat),
                       ('UNTOUCHED', untouched), ('DIVERGING', diverging)):
        if mask.sum() > 5:
            rep += _summarise_z(name, normalised_returns(
                close, ev.loc[mask, 'end_idx'].values), int(mask.sum()))
    rep += _summarise_z('RANDOM BASELINE',
                        normalised_returns(close, base_ends), len(base_ends))

    if clean.sum() > 50:
        rep += convergence_buckets(ev.loc[clean], close)

    ce = ev.loc[clean, 'end_idx'].values
    plot_histograms({'CLEAN': normalised_returns(close, ce)},
                    normalised_returns(close, base_ends),
                    out / 'hist_normalised.png', 'z', 4.0)
    plot_histograms({'CLEAN': forward_returns(close, ce)},
                    forward_returns(close, base_ends),
                    out / 'hist_raw.png', '%', 3.0)

    text = '\n'.join(rep)
    print('\n' + text)
    (out / 'filter_report.txt').write_text(text)
    ev.assign(category=np.where(diverging, 'DIVERGING',
                       np.where(flat, 'NEAR-PARALLEL',
                       np.where(untouched, 'UNTOUCHED',
                       np.where(escapes, 'ESCAPES', 'CLEAN'))))
              ).to_csv(out / 'events_filtered.csv', index=False)

    survivors = ev.loc[clean]
    if len(survivors):
        pick = survivors.sample(n=min(args.n_charts, len(survivors)),
                                random_state=args.seed).sort_values('date_end')
        render(pick, data, dates, args.fit_bars,
               out / 'clean_detections.png',
               f'CLEAN detections -- converging and contained '
               f'({int(clean.sum()):,} of {n:,})')
    # The survivors that still went badly wrong -- the honest place to look.
    if clean.sum() > 12:
        sv = ev.loc[clean].copy()
        fr4 = forward_returns(close, sv['end_idx'].values)['4hr']
        sv['fwd_4hr'] = fr4
        worst = sv.dropna(subset=['fwd_4hr']).nsmallest(12, 'fwd_4hr')
        render(worst, data, dates, args.fit_bars, out / 'worst_returns.png',
               f'WORST 12 4hr RETURNS among {int(clean.sum()):,} survivors')

    for nm, mask in (('diverging', diverging), ('near_parallel', flat),
                     ('untouched', untouched), ('escapes', escapes)):
        sub = ev.loc[mask]
        if len(sub) >= 8:
            pick = sub.sample(n=min(8, len(sub)), random_state=args.seed)
            render(pick.sort_values('date_end'), data, dates, args.fit_bars,
                   out / f'rejected_{nm}.png',
                   f'REJECTED -- {nm} ({int(mask.sum()):,} of {n:,})')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
