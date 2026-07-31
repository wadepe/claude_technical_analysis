#!/usr/bin/env bash
# push_results.sh
#
# Commits and pushes the server's LOG files to GitHub for remote visibility.
# Scheduled via cron at 7:00 PM ET on weekdays.
#
# Since the SQLite migration (2026-07-30) the data itself lives in wedge.db
# on the server and is served by the wedge-api service — it is NOT pushed to
# git any more. Only crash.log and the cron logs go up. The daily chart is
# still regenerated locally (from the database) for on-server review, but is
# no longer committed.
#
# Cron entry (added by setup_server.sh):
#   CRON_TZ=America/New_York
#   0 19 * * 1-5 /home/ubuntu/claude_technical_analysis/deploy/push_results.sh

set -euo pipefail

# Derive the repo root from this script's own location (deploy/..) so the
# path is correct regardless of which user/home the repo is cloned under.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_TAG="push_results"
BRANCH="master"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [$LOG_TAG]  $*"; }

cd "$REPO_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
    log "ERROR: $REPO_DIR is not a git repository. Aborting."
    exit 1
fi

# ── Regenerate the daily chart (local review only, not committed) ─────────────
# Use the virtualenv Python if present (matches live-monitor.service), else
# system python3. A chart failure must NOT abort the log push, so we tolerate
# a non-zero exit here despite 'set -e'.
if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PY="$REPO_DIR/.venv/bin/python"
else
    PY="python3"
fi
if "$PY" "$REPO_DIR/source_code/plot_daily.py" --data-dir "$REPO_DIR"; then
    log "Daily chart regenerated (local only)."
else
    log "WARNING: chart generation failed."
fi

# ── Stage only the log files ──────────────────────────────────────────────────
# We intentionally do NOT do 'git add -A' — only logs go to GitHub from the
# server. Code changes flow the other direction (dev -> server), and the data
# lives in wedge.db, served by wedge-api.
LOG_FILES=("crash.log" "logs/push_results.log" "logs/pull_updates.log")
STAGED=0
for f in "${LOG_FILES[@]}"; do
    if [ -f "$REPO_DIR/$f" ]; then
        git add "$REPO_DIR/$f"
        STAGED=$((STAGED + 1))
    else
        log "WARNING: $f not found — skipping"
    fi
done

if [ "$STAGED" -eq 0 ]; then
    log "No log files found. Nothing to push."
    exit 0
fi

# ── Commit only if there are staged changes ───────────────────────────────────
if git diff --cached --quiet; then
    log "No changes since last push. Nothing to commit."
    exit 0
fi

# Row counts from the database, for the commit subject (informational only;
# cwd is REPO_DIR, where wedge.db lives).
ROWS=$("$PY" - <<'PYEOF' 2>/dev/null || echo "db=?"
import sqlite3
try:
    con = sqlite3.connect('file:wedge.db?mode=ro', uri=True)
    b = con.execute('SELECT count(*) FROM bars').fetchone()[0]
    s = con.execute('SELECT count(*) FROM scores').fetchone()[0]
    print(f'bars={b}, scores={s}')
except Exception:
    print('db=?')
PYEOF
)

git commit -m "auto: log update $(date '+%Y-%m-%d')  [${ROWS}]"
log "Committed log update."

# ── Push (retry once on transient network failure) ────────────────────────────
if git push origin "$BRANCH"; then
    log "Pushed to origin/$BRANCH successfully."
else
    log "First push attempt failed — retrying in 30 s ..."
    sleep 30
    git push origin "$BRANCH"
    log "Retry push succeeded."
fi
