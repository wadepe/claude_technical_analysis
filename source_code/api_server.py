"""
api_server.py

Read-only HTTP/JSON API over wedge.db, for anything that wants the live
data without pulling git: the Android app, Claude on the dev box, curl.

Standard library only (http.server + sqlite3) — no new dependencies on the
server. Every request opens its own read-only SQLite connection (mode=ro,
sub-millisecond), so the monitor stays the sole writer and a wedged reader
can never block it.

Endpoints (all GET, all JSON)
-----------------------------
  /health                  {ok, db, bars, scores, signals, last_bar}
  /bars     ?since=&until=&limit=     price bars, ascending
  /scores   ?since=&until=&limit=&window=&pattern=   scores, ascending
  /signals  ?since=&until=&limit=&window=&pattern=   geometry, ascending
  /latest                  last bar + newest scores row per (pattern, window)

  pattern: 'wedge' or 'channel'. Omit for all formations. In `signals`,
  mid_travel is the formation's slope and proj_move_usd is null for
  channels (its fit was estimated on wedges only).

  since/until: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (ET, inclusive /
  exclusive). limit: rows, default 2000, max 25000 — enough for a full
  extended-hours day of bars in one call. Rows return newest-last; when
  more rows match than `limit`, the NEWEST rows win (the tail is what a
  live viewer wants), still delivered ascending.

Security
--------
  Read-only by construction (SQLite mode=ro + GET only). If the environment
  variable WEDGE_API_TOKEN is set, every request must carry
  'Authorization: Bearer <token>'. There is no TLS here — put it behind
  Tailscale, a LAN, or a reverse proxy; do not expose the bare port to the
  internet.

Usage
-----
  python api_server.py                    # 0.0.0.0:8321, ../wedge.db
  python api_server.py --port 9000 --db /path/to/wedge.db
  WEDGE_API_TOKEN=secret python api_server.py

  Deployed via deploy/wedge-api.service (systemd).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB   = PROJECT_ROOT / 'wedge.db'

MAX_LIMIT     = 25_000
DEFAULT_LIMIT = 2_000

_TABLES = {
    'bars':    'SELECT ts, open, high, low, close, volume FROM bars',
    'scores':  'SELECT ts, pattern, window, score, signal, bars FROM scores',
    'signals': 'SELECT ts, pattern, window, proj_move_usd, slope_upper, '
               'slope_lower, apex_min, apex_price, mid_travel FROM signals',
}


def _rows_to_dicts(cur) -> list:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


class Handler(BaseHTTPRequestHandler):
    server_version = 'WedgeAPI/1.0'
    db_path: str   = str(DEFAULT_DB)
    token: str     = ''

    # ── Plumbing ──────────────────────────────────────────────────────────────

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        # The Android app (and browser-based dev tools) call cross-origin.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):   # stamp like the monitor's logs
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {self.address_string()}  '
              f'{fmt % args}', flush=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True)

    def _query_args(self, qs: dict) -> tuple:
        """Common since/until/limit/window handling -> (where, params, limit)."""
        where, params = [], []
        if 'since' in qs:
            where.append('ts >= ?')
            params.append(qs['since'][0])
        if 'until' in qs:
            where.append('ts < ?')
            params.append(qs['until'][0])
        if 'window' in qs:
            where.append('window = ?')
            params.append(int(qs['window'][0]))
        if 'pattern' in qs:
            where.append('pattern = ?')
            params.append(qs['pattern'][0])
        limit = min(int(qs.get('limit', [DEFAULT_LIMIT])[0]), MAX_LIMIT)
        return (' WHERE ' + ' AND '.join(where) if where else ''), params, limit

    # ── Routes ────────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        if self.token:
            if self.headers.get('Authorization', '') != f'Bearer {self.token}':
                self._send(401, {'error': 'missing or bad bearer token'})
                return

        url = urlparse(self.path)
        qs  = parse_qs(url.query)
        route = url.path.rstrip('/') or '/health'

        try:
            if route == '/health':
                self._health()
            elif route in ('/bars', '/scores', '/signals'):
                self._table(route.lstrip('/'), qs)
            elif route == '/latest':
                self._latest()
            else:
                self._send(404, {'error': f'no such route {route}',
                                 'routes': ['/health', '/bars', '/scores',
                                            '/signals', '/latest']})
        except (ValueError, IndexError) as exc:
            self._send(400, {'error': f'bad parameter: {exc}'})
        except sqlite3.OperationalError as exc:
            self._send(503, {'error': f'database unavailable: {exc}'})

    def _health(self) -> None:
        con = self._connect()
        try:
            counts = {t: con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
                      for t in _TABLES}
            last = con.execute('SELECT max(ts) FROM bars').fetchone()[0]
        finally:
            con.close()
        self._send(200, {'ok': True, 'db': self.db_path, **counts,
                         'last_bar': last})

    def _table(self, table: str, qs: dict) -> None:
        # window/pattern filters are meaningless for bars — reject early so a
        # typo'd query fails loudly instead of silently returning everything.
        if table == 'bars' and ({'window', 'pattern'} & set(qs)):
            raise ValueError('bars has no window or pattern column')
        where, params, limit = self._query_args(qs)
        con = self._connect()
        try:
            # Newest rows win the limit; outer query restores ascending order.
            cur = con.execute(
                f'SELECT * FROM ({_TABLES[table]}{where} '
                f'ORDER BY ts DESC LIMIT ?) ORDER BY ts',
                params + [limit])
            rows = _rows_to_dicts(cur)
        finally:
            con.close()
        self._send(200, {'rows': rows, 'count': len(rows), 'limit': limit})

    def _latest(self) -> None:
        con = self._connect()
        try:
            bar = _rows_to_dicts(con.execute(
                f'{_TABLES["bars"]} ORDER BY ts DESC LIMIT 1'))
            # newest scores row per (pattern, window)
            scores = _rows_to_dicts(con.execute(
                f'''SELECT s.ts, s.pattern, s.window, s.score, s.signal, s.bars
                    FROM scores s
                    JOIN (SELECT pattern, window, max(ts) AS ts FROM scores
                          GROUP BY pattern, window) m
                    ON s.pattern = m.pattern AND s.window = m.window
                       AND s.ts = m.ts
                    ORDER BY s.pattern, s.window'''))
        finally:
            con.close()
        self._send(200, {'bar': bar[0] if bar else None, 'scores': scores})


def main() -> None:
    p = argparse.ArgumentParser(description='Read-only wedge.db HTTP API')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8321)
    p.add_argument('--db',   default=str(DEFAULT_DB))
    args = p.parse_args()

    Handler.db_path = args.db
    Handler.token   = os.environ.get('WEDGE_API_TOKEN', '')

    if not Path(args.db).exists():
        # Start anyway (systemd may launch us before the monitor's first
        # write); /health will report 503 until the file appears.
        print(f'WARNING: {args.db} does not exist yet — serving 503s '
              f'until it does', flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Wedge API on http://{args.host}:{args.port}  db={args.db}  '
          f'auth={"bearer token" if Handler.token else "none"}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
