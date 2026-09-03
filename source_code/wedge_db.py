"""
wedge_db.py

SQLite storage for the live monitor. Replaces spy_data_1min.csv and
rising_wedge.csv as the output sink (2026-07-30); the CSVs remain readable
by the one-time migration below.

Database:  <project root>/wedge.db      (WAL journal; gitignored)

Schema
------
  bars(ts PK, open, high, low, close, volume)
      Every accepted bar, extended hours included — full price history,
      same coverage the price CSV had.

  scores(ts, pattern, window, score, signal, bars)  PK (ts, pattern, window)
      One row per scored bar per (formation, window size): currently
      wedge@50, wedge@250, channel@250. Only regular-session bars are
      scored (see live_monitor's session gate), so extended-hours bars have
      a bars row and no scores rows. score is NULL while the rolling window
      is still filling.

      Both `pattern` and `window` are DATA, not column names. Adding a
      formation or a window size is new rows, not a schema change — the
      CSV's score_50bar/score_250bar columns are exactly why the v1->v2
      change forced an archive-and-restart.

  signals(ts, pattern, window, proj_move_usd, slope_upper, slope_lower,
          apex_min, apex_price, mid_travel)   PK (ts, pattern, window)
      Fitted geometry, present ONLY where signal=1 — replaces the CSV
      convention of zero-filled stat columns on every non-signal row.

      mid_travel is the formation's SLOPE (midline tilt across the window
      as a fraction of its price range): the wedge's rise/fall, and the
      channel's slope. proj_move_usd is NULL for channels — the PROJ_*
      fit in live_monitor was estimated on wedge events only, and applying
      it to channels would invent a number. apex_* stay 0 for parallel
      fits, which is most channels.

Timestamps are TEXT 'YYYY-MM-DD HH:MM:SS', naive America/New_York — exactly
the strings the CSVs used. ISO text sorts chronologically, so BETWEEN range
scans work, and there is no epoch/timezone conversion layer to get wrong.

Concurrency: WAL mode — one writer (the monitor) plus any number of
readers (API server, plot_daily, ad-hoc sqlite3). synchronous=NORMAL: a
commit cannot corrupt the DB on power loss, though the last moments of
writes may be lost; the next fetch re-covers the day anyway.

CLI
---
  python wedge_db.py --migrate    # import spy_data_1min.csv + rising_wedge.csv
  python wedge_db.py --stats      # row counts and time range
  (--db / --spy-csv / --wedge-csv override the default paths)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / 'wedge.db'

STAT_KEYS = ['proj_move_usd', 'slope_upper', 'slope_lower',
             'apex_min', 'apex_price', 'mid_travel',
             # v5 quality gate inputs. Logged as well as gated on: a signal
             # that stops firing is indistinguishable from a broken monitor
             # unless the reason it was suppressed is recorded alongside it.
             'convergence', 'touch_up', 'touch_lo', 'max_excursion']

DEFAULT_PATTERN = 'wedge'      # what pre-migration rows are attributed to

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS bars (
    ts     TEXT PRIMARY KEY,
    open   REAL, high REAL, low REAL, close REAL, volume REAL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS scores (
    ts      TEXT    NOT NULL,
    pattern TEXT    NOT NULL,
    window  INTEGER NOT NULL,
    score   REAL,
    signal  INTEGER NOT NULL DEFAULT 0,
    bars    INTEGER,
    PRIMARY KEY (ts, pattern, window)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS signals (
    ts      TEXT    NOT NULL,
    pattern TEXT    NOT NULL,
    window  INTEGER NOT NULL,
    {', '.join(f'{k} REAL' for k in STAT_KEYS)},
    PRIMARY KEY (ts, pattern, window)
) WITHOUT ROWID;
"""


def _needs_pattern_migration(con: sqlite3.Connection) -> bool:
    """True if scores/signals exist but predate the `pattern` column."""
    for table in ('scores', 'signals'):
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)).fetchone()
        if row is None:
            continue
        cols = [r[1] for r in con.execute(f'PRAGMA table_info({table})')]
        if 'pattern' not in cols:
            return True
    return False


def _migrate_add_pattern(con: sqlite3.Connection) -> None:
    """
    Rebuild scores/signals with `pattern` in the primary key.

    SQLite cannot ALTER a PRIMARY KEY, so the tables are recreated and
    copied. Existing rows are all wedge output (the only model that ran
    before this change), so they are attributed to DEFAULT_PATTERN. Runs in
    one transaction: either the whole rebuild lands or none of it does.
    """
    with con:
        for table, extra in (('scores', 'score, signal, bars'),
                             ('signals', ', '.join(STAT_KEYS))):
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()
            if row is None:
                continue
            cols = [r[1] for r in con.execute(f'PRAGMA table_info({table})')]
            if 'pattern' in cols:
                continue
            con.execute(f'ALTER TABLE {table} RENAME TO {table}_old')
            con.executescript(_SCHEMA)
            con.execute(
                f'INSERT INTO {table} (ts, pattern, window, {extra}) '
                f'SELECT ts, ?, window, {extra} FROM {table}_old',
                (DEFAULT_PATTERN,))
            con.execute(f'DROP TABLE {table}_old')


def _add_missing_stat_columns(con: sqlite3.Connection) -> None:
    """
    Add any STAT_KEYS the existing signals table predates.

    _SCHEMA is CREATE TABLE IF NOT EXISTS, so extending STAT_KEYS does NOT
    reshape a table that already exists on a running server -- and the insert
    builds its placeholder count from STAT_KEYS, so the next signal would fail
    with a column-count mismatch and take the monitor down. This closes that
    gap. ALTER TABLE ADD COLUMN is O(1) in SQLite and leaves existing rows with
    NULL for the new column, which is the honest value: those signals were
    raised before the measurement existed.
    """
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
    ).fetchone()
    if row is None:
        return
    have = {r[1] for r in con.execute('PRAGMA table_info(signals)')}
    with con:
        for k in STAT_KEYS:
            if k not in have:
                con.execute(f'ALTER TABLE signals ADD COLUMN {k} REAL')


def connect(path: Path | str = DB_PATH, readonly: bool = False
            ) -> sqlite3.Connection:
    """
    Open the database. Writers get WAL + NORMAL sync and the schema is
    ensured; readers get an immutable-safe read-only connection.
    """
    if readonly:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    else:
        con = sqlite3.connect(str(path))
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA synchronous=NORMAL')
        if _needs_pattern_migration(con):
            _migrate_add_pattern(con)
        con.executescript(_SCHEMA)
        _add_missing_stat_columns(con)
    con.row_factory = sqlite3.Row
    return con


# =============================================================================
# Write path (the monitor)
# =============================================================================

def write_bar(con: sqlite3.Connection, bar: dict) -> None:
    """Store one price bar. OR REPLACE: a re-served yfinance bar is an update."""
    with con:
        con.execute('INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?)',
                    (bar['timestamp'], bar['open'], bar['high'],
                     bar['low'], bar['close'], bar['volume']))


def write_minute(con: sqlite3.Connection, bar: dict, scores: dict,
                 sigs: dict, stats: dict, depths: dict) -> None:
    """
    Store one fully-scored minute atomically: the bar, a scores row per
    (pattern, window), and a signals row for each that fired.

    scores/sigs/stats/depths are keyed by the (pattern, window) tuple, so
    adding a formation needs no change here.
    """
    ts = bar['timestamp']
    with con:
        con.execute('INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?)',
                    (ts, bar['open'], bar['high'], bar['low'],
                     bar['close'], bar['volume']))
        for key, score in scores.items():
            pattern, w = key
            con.execute('INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)',
                        (ts, pattern, w, score, sigs.get(key, 0),
                         depths.get(key, 0)))
            if sigs.get(key):
                st = stats[key]
                con.execute(
                    f'INSERT OR REPLACE INTO signals VALUES '
                    f'({",".join("?" * (3 + len(STAT_KEYS)))})',
                    tuple([ts, pattern, w] + [st.get(k) for k in STAT_KEYS]))


def clear_scores(con: sqlite3.Connection) -> None:
    """Drop all model output (bars are kept). Used by a clean replay."""
    with con:
        con.execute('DELETE FROM scores')
        con.execute('DELETE FROM signals')


# =============================================================================
# Read path
# =============================================================================

def last_bar_ts(con: sqlite3.Connection) -> Optional[str]:
    row = con.execute('SELECT max(ts) AS ts FROM bars').fetchone()
    return row['ts'] if row and row['ts'] else None


def bar_count(con: sqlite3.Connection) -> int:
    return con.execute('SELECT count(*) AS n FROM bars').fetchone()['n']


def all_bars(con: sqlite3.Connection):
    """All bars in chronological order as (ts, open, high, low, close, volume)."""
    return con.execute(
        'SELECT ts, open, high, low, close, volume FROM bars ORDER BY ts'
    ).fetchall()


# =============================================================================
# One-time CSV migration
# =============================================================================

def migrate_csvs(con: sqlite3.Connection,
                 spy_csv: Path, wedge_csv: Path) -> dict:
    """
    Import the legacy CSVs. Idempotent: OR REPLACE keys on timestamp, so
    re-running (or running after the monitor has already written rows) only
    overwrites the same keys. Duplicate CSV timestamps collapse to the last
    occurrence, which also dedups the historical files for free.
    """
    import pandas as pd

    counts = {'bars': 0, 'scores': 0, 'signals': 0}
    P = DEFAULT_PATTERN          # the CSVs only ever held wedge output

    if spy_csv.exists():
        spy = pd.read_csv(spy_csv)
        with con:
            con.executemany(
                'INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?)',
                spy[['timestamp', 'open', 'high', 'low',
                     'close', 'volume']].values.tolist())
        counts['bars'] = len(spy)

    if wedge_csv.exists():
        wdg = pd.read_csv(wedge_csv)
        srows, grows = [], []
        for w in (50, 250):
            need = {f'score_{w}bar', f'signal_{w}bar', f'bars_{w}'}
            if not need.issubset(wdg.columns):
                continue   # pre-v2 file; only v2 columns are migrated
            for _, r in wdg.iterrows():
                score = r[f'score_{w}bar']
                score = None if pd.isna(score) else float(score)
                sig   = int(r[f'signal_{w}bar']) if pd.notna(r[f'signal_{w}bar']) else 0
                srows.append((r['timestamp'], P, w, score, sig,
                              int(r[f'bars_{w}'])))
                if sig:
                    grows.append(tuple(
                        [r['timestamp'], P, w]
                        + [float(r[f'{k}_{w}bar']) for k in STAT_KEYS]))
        with con:
            con.executemany(
                'INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?)', srows)
            con.executemany(
                f'INSERT OR REPLACE INTO signals VALUES '
                f'({",".join("?" * (3 + len(STAT_KEYS)))})', grows)
        counts['scores']  = len(srows)
        counts['signals'] = len(grows)

    return counts


def stats(con: sqlite3.Connection) -> str:
    lines = []
    for table in ('bars', 'scores', 'signals'):
        r = con.execute(f'SELECT count(*) AS n, min(ts) AS lo, max(ts) AS hi '
                        f'FROM {table}').fetchone()
        lines.append(f'  {table:8s} {r["n"]:>9,} rows'
                     + (f'   {r["lo"]}  ->  {r["hi"]}' if r['n'] else ''))
        if table in ('scores', 'signals') and r['n']:
            sig_col = 'sum(signal)' if table == 'scores' else 'NULL'
            for br in con.execute(
                    f'SELECT pattern, window, count(*) AS n, '
                    f'{sig_col} AS sig FROM {table} '
                    f'GROUP BY pattern, window ORDER BY pattern, window'):
                extra = ('' if br['sig'] is None
                         else f'   signals {br["sig"]:,}')
                lines.append(f'    {br["pattern"]:<8} @{br["window"]:<4} '
                             f'{br["n"]:>9,}{extra}')
    return '\n'.join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='Wedge monitor SQLite utilities')
    p.add_argument('--db',        default=str(DB_PATH))
    p.add_argument('--spy-csv',   default=str(PROJECT_ROOT / 'spy_data_1min.csv'))
    p.add_argument('--wedge-csv', default=str(PROJECT_ROOT / 'rising_wedge.csv'))
    p.add_argument('--migrate', action='store_true',
                   help='Import the legacy CSVs into the database')
    p.add_argument('--stats',   action='store_true',
                   help='Print row counts and time ranges')
    args = p.parse_args()

    con = connect(args.db)
    if args.migrate:
        c = migrate_csvs(con, Path(args.spy_csv), Path(args.wedge_csv))
        print(f'Migrated: {c["bars"]:,} bar rows, {c["scores"]:,} score rows, '
              f'{c["signals"]:,} signal rows  ->  {args.db}')
    if args.stats or args.migrate:
        print(stats(con))
    if not (args.migrate or args.stats):
        p.print_help()
    con.close()


if __name__ == '__main__':
    main()
