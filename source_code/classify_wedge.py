"""
classify_wedge.py

Trendline fitting for v2 wedge detections. Fits an upper envelope line to the
highs and a lower envelope line to the lows of a detection window and reports
the raw geometry — slopes, widths, convergence. Deliberately does NOT bucket
detections into rising/falling/flat: downstream analysis works with the
continuous slope values directly.

Fitting: iterative envelope regression. A least-squares line through the highs
is re-fit on only the bars at/above it (2 rounds), converging on the upper
envelope and ignoring interior bars; mirrored for the lows.

Units: slopes are per-bar as a fraction of the window's high-low range, so
values are scale-free and comparable across price regimes. `travel_*` is the
slope integrated across the window (total rise/fall as a fraction of range).

Usage
-----
  from classify_wedge import fit_wedge_lines
  g = fit_wedge_lines(window)        # (n_bars, 5) raw or normalised OHLCV
  g['m_upper'], g['m_lower']         # per-bar slopes (range fraction)
  g['travel_upper'], g['travel_lower']
  g['width_start'], g['width_end'], g['convergence']

Self-test (fitted slopes vs the v2 generator's true trendlines):
  python classify_wedge.py --self-test
"""

from __future__ import annotations

import numpy as np


def _envelope_fit(x: np.ndarray, y: np.ndarray, side: str, rounds: int = 2
                  ) -> tuple[float, float]:
    """Fit y ~ a + b*x, then re-fit on the envelope side only. Returns (a, b)."""
    pts = np.ones_like(y, dtype=bool)
    a = b = 0.0
    for _ in range(rounds + 1):
        if pts.sum() < 3:
            break
        b, a = np.polyfit(x[pts], y[pts], 1)
        resid = y - (a + b * x)
        if side == 'upper':
            pts = resid >= np.quantile(resid, 0.55)
        else:
            pts = resid <= np.quantile(resid, 0.45)
    return a, b


def fit_wedge_lines(window: np.ndarray) -> dict:
    """
    Fit upper/lower trendlines to a (n_bars, 5) OHLCV window and return the
    wedge geometry. Works on raw or normalised prices (slopes are normalised
    by the window's price range internally).
    """
    n     = window.shape[0]
    x     = np.arange(n, dtype=float)
    highs = window[:, 1].astype(float)
    lows  = window[:, 2].astype(float)

    p_range = max(float(highs.max() - lows.min()), 1e-12)

    a_u, b_u = _envelope_fit(x, highs, 'upper')
    a_l, b_l = _envelope_fit(x, lows,  'lower')

    su = b_u / p_range                       # per-bar, range-fraction units
    sl = b_l / p_range
    w_start = (a_u - a_l) / p_range
    w_end   = ((a_u + b_u * (n - 1)) - (a_l + b_l * (n - 1))) / p_range

    return {
        'm_upper':      float(su),
        'm_lower':      float(sl),
        'travel_upper': float(su * n),
        'travel_lower': float(sl * n),
        'width_start':  float(w_start),
        'width_end':    float(w_end),
        'convergence':  float(1.0 - w_end / w_start) if w_start > 1e-9 else 0.0,
        # Raw-unit line parameters (same units as the input prices):
        # upper line = a_upper + b_upper * bar_index, bar_index 0..n-1.
        # Needed for dollar-space projections (apex price/time).
        'a_upper': float(a_u), 'b_upper': float(b_u),
        'a_lower': float(a_l), 'b_lower': float(b_l),
    }


# =============================================================================
# Self-test: fitted slopes vs the generator's true (normalised) trendlines
# =============================================================================

def _self_test(n_samples: int = 400) -> None:
    from generate_wedges import _anchored_pattern

    true_u, true_l, fit_u, fit_l = [], [], [], []
    for idx in range(n_samples):
        df, meta = _anchored_pattern(idx, 'forming_wedge')
        pre = meta['pre_pad']
        win = df[['open', 'high', 'low', 'close', 'volume']].values[pre:]

        # ground truth from the stored (normalised) trendline columns,
        # expressed in the same range-fraction units the fitter reports
        tl = df['lower_trendline'].values[pre:]
        tu = df['upper_trendline'].values[pre:]
        rng = max(float(win[:, 1].max() - win[:, 2].min()), 1e-12)
        n   = len(tl)
        true_u.append((tu[-1] - tu[0]) / (n - 1) / rng)
        true_l.append((tl[-1] - tl[0]) / (n - 1) / rng)

        g = fit_wedge_lines(win)
        fit_u.append(g['m_upper'])
        fit_l.append(g['m_lower'])

    for name, t, f in (('upper', true_u, fit_u), ('lower', true_l, fit_l)):
        t, f = np.array(t), np.array(f)
        r    = float(np.corrcoef(t, f)[0, 1])
        mae  = float(np.mean(np.abs(t - f)))
        scale = float(np.mean(np.abs(t)))
        print(f'{name}: pearson_r={r:.4f}  mae={mae:.5f} '
              f'(mean |true slope|={scale:.5f})')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--n', type=int, default=400)
    args = p.parse_args()
    if args.self_test:
        _self_test(args.n)
