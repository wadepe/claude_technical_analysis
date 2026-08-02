"""
plot_daily.py

Renders a single PNG overlaying the day's SPY price with the live model
output, for quick end-of-day review. Intended to run from push_results.sh
just before the 7 PM ET commit, so the chart is pushed alongside the CSVs.

Reads (from the project root)
-----------------------------
  wedge.db            bars + scores tables (see wedge_db.py); falls back to
                      the legacy spy_data_1min.csv / rising_wedge.csv pair
                      when no database exists yet

Writes
------
  rising_wedge_chart.png   (project root by default)

Layout
------
  Top    : SPY close price, with markers where a 50- or 250-bar signal fired
  Bottom : 50- and 250-bar model scores over time, with the decision threshold

By default only the most recent trading day in the data is plotted (the usual
end-of-day review). Use --all to plot the entire history, or --date to pick a
specific day.

Usage
-----
  python plot_daily.py
  python plot_daily.py --date 2026-06-08      # a specific trading day
  python plot_daily.py --all                  # entire history, not just one day
  python plot_daily.py --threshold 0.65
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # headless — save to file without a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Project dark theme (matches evaluate_cnn.py / scan_bitcoin.py) ─────────────
_FIG_BG = '#1a1a2e'
_AX_BG  = '#13131f'
_GRID   = '#2a2a3e'
_SPINE  = '#333345'
_MUTED  = '#8a8aa0'
_TEXT   = '#d0d0e8'
_BLUE   = '#42A5F5'
_GREEN  = '#26a69a'
_RED    = '#ef5350'


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(_AX_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(True, alpha=0.20, color=_GRID)


# =============================================================================
# Data loading
# =============================================================================

def load_from_db(db_path: Path) -> pd.DataFrame:
    """
    Load bars joined with per-window scores from wedge.db into the same wide
    frame the CSV pair used (score_50bar, signal_50bar, ...), so the plotting
    code below needs no changes.
    """
    import sqlite3
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        spy = pd.read_sql_query(
            'SELECT ts AS timestamp, open, high, low, close, volume '
            'FROM bars ORDER BY ts', con, parse_dates=['timestamp'])
        # The chart is the wedge end-of-day review; channel output lives in
        # the database and the API. Filtering here keeps the pivot below
        # one-column-per-window as before.
        sc = pd.read_sql_query(
            "SELECT ts AS timestamp, window, score, signal FROM scores "
            "WHERE pattern = 'wedge'", con, parse_dates=['timestamp'])
    finally:
        con.close()
    if spy.empty:
        return spy

    df = spy
    if not sc.empty:
        wide = sc.pivot(index='timestamp', columns='window',
                        values=['score', 'signal'])
        wide.columns = [f'{a}_{b}bar' for a, b in wide.columns]
        df = spy.merge(wide.reset_index(), on='timestamp', how='left')
    return df


def load_merged(spy_csv: Path, wedge_csv: Path,
                db_path: Path | None = None) -> pd.DataFrame:
    """
    Load price + score data, preferring wedge.db and falling back to the
    legacy CSV pair.

    Returns an empty DataFrame if there is no price data yet (e.g. the monitor
    has not run), so callers can skip gracefully instead of crashing.
    """
    if db_path is not None and db_path.exists():
        df = load_from_db(db_path)
        if not df.empty:
            for col in ('score_50bar', 'score_250bar',
                        'signal_50bar', 'signal_250bar'):
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.sort_values('timestamp').reset_index(drop=True)

    if not spy_csv.exists():
        return pd.DataFrame()

    spy = pd.read_csv(spy_csv, parse_dates=['timestamp'])
    if spy.empty:
        return spy

    if wedge_csv.exists():
        wdg = pd.read_csv(wedge_csv, parse_dates=['timestamp'])
        df  = spy.merge(wdg, on='timestamp', how='left')
    else:
        df = spy

    # Scores are written as '' until each rolling window fills; coerce to float
    # NaN so they plot as gaps rather than breaking on object dtype.
    for col in ('score_50bar', 'score_250bar', 'signal_50bar', 'signal_250bar'):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.sort_values('timestamp').reset_index(drop=True)


# =============================================================================
# Plot
# =============================================================================

def plot_day(
    df: pd.DataFrame,
    out_path: Path,
    ticker: str,
    threshold: float,
    span_label: str,
) -> None:
    fig, (ax_p, ax_s) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.4], 'hspace': 0.08},
    )
    fig.patch.set_facecolor(_FIG_BG)
    _style(ax_p)
    _style(ax_s)

    # matplotlib/pandas can choke on plotting datetime Series directly across
    # versions — pass plain numpy arrays everywhere.
    x     = df['timestamp'].to_numpy()
    close = df['close'].to_numpy()

    # ── Price panel ───────────────────────────────────────────────────────────
    ax_p.plot(x, close, color=_TEXT, lw=1.1, label=f'{ticker} close', zorder=2)

    # Signal markers, placed on the price line where each window fired.
    sig50  = (df.get('signal_50bar',  pd.Series(dtype=float)).fillna(0) == 1).to_numpy()
    sig250 = (df.get('signal_250bar', pd.Series(dtype=float)).fillna(0) == 1).to_numpy()
    if sig50.any():
        ax_p.scatter(x[sig50], close[sig50], marker='^', s=45,
                     color=_GREEN, edgecolor='white', linewidth=0.4, zorder=5,
                     label=f'50-bar signal ({int(sig50.sum())})')
    if sig250.any():
        ax_p.scatter(x[sig250], close[sig250], marker='v', s=45,
                     color=_RED, edgecolor='white', linewidth=0.4, zorder=5,
                     label=f'250-bar signal ({int(sig250.sum())})')

    ax_p.set_ylabel(f'{ticker} price', color=_MUTED, fontsize=9)
    ax_p.legend(loc='upper left', fontsize=8, facecolor='#222233',
                labelcolor=_TEXT, framealpha=0.75, ncol=3)

    # ── Score panel ───────────────────────────────────────────────────────────
    if 'score_50bar' in df:
        ax_s.plot(x, df['score_50bar'].to_numpy(), color=_GREEN, lw=1.2,
                  label='score 50-bar')
    if 'score_250bar' in df:
        ax_s.plot(x, df['score_250bar'].to_numpy(), color=_BLUE, lw=1.2,
                  label='score 250-bar')
    ax_s.axhline(threshold, color=_RED, lw=1.2, ls='--',
                 label=f'threshold ({threshold})')
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_ylabel('P(rising wedge)', color=_MUTED, fontsize=9)
    ax_s.set_xlabel('Time (ET)', color=_MUTED, fontsize=9)
    ax_s.legend(loc='upper left', fontsize=8, facecolor='#222233',
                labelcolor=_TEXT, framealpha=0.75, ncol=3)

    # ── X axis: time-of-day formatting ─────────────────────────────────────────
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate(rotation=0, ha='center')

    n_sig = int(sig50.sum() + sig250.sum())
    fig.suptitle(
        f'{ticker} — Rising Wedge Monitor   |   {span_label}   |   '
        f'{len(df):,} bars   |   {n_sig} signal bar(s)',
        color='#e8e8ff', fontsize=13, y=0.95,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Chart saved: {out_path.resolve()}')


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot SPY price overlaid with live rising-wedge model output'
    )
    parser.add_argument('--data-dir',  default='..',
                        help='Project root directory (default: parent of source_code/)')
    parser.add_argument('--spy-csv',   default=None,
                        help='Override path to spy_data_1min.csv')
    parser.add_argument('--wedge-csv', default=None,
                        help='Override path to rising_wedge.csv')
    parser.add_argument('--output',    default=None,
                        help='Output PNG path (default: <data-dir>/rising_wedge_chart.png)')
    parser.add_argument('--ticker',    default='SPY',
                        help='Ticker label for titles (default: SPY)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Signal threshold line (default: 0.8, matching '
                             'the live monitor)')
    parser.add_argument('--date',      default=None, metavar='YYYY-MM-DD',
                        help='Plot a specific trading day (default: most recent)')
    parser.add_argument('--all',       action='store_true',
                        help='Plot the entire history instead of a single day')
    args = parser.parse_args()

    root      = Path(args.data_dir)
    spy_csv   = Path(args.spy_csv)   if args.spy_csv   else root / 'spy_data_1min.csv'
    wedge_csv = Path(args.wedge_csv) if args.wedge_csv else root / 'rising_wedge.csv'
    out_path  = Path(args.output)    if args.output    else root / 'rising_wedge_chart.png'

    df = load_merged(spy_csv, wedge_csv, db_path=root / 'wedge.db')
    if df.empty:
        print(f'No price data in {root} yet — skipping chart generation.')
        return   # exit 0 so the caller (push_results.sh) is not aborted

    # ── Select the day to plot ────────────────────────────────────────────────
    if args.all:
        span_label = (f'{df["timestamp"].iloc[0]:%Y-%m-%d} to '
                      f'{df["timestamp"].iloc[-1]:%Y-%m-%d}')
    else:
        if args.date:
            target = datetime.strptime(args.date, '%Y-%m-%d').date()
        else:
            target = df['timestamp'].dt.date.max()
        df = df[df['timestamp'].dt.date == target].reset_index(drop=True)
        if df.empty:
            print(f'No data for {target} — nothing to plot.')
            return
        span_label = f'{target:%Y-%m-%d}'

    plot_day(df, out_path, args.ticker, args.threshold, span_label)


if __name__ == '__main__':
    main()
