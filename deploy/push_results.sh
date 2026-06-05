#!/usr/bin/env bash
# push_results.sh
#
# Commits and pushes the latest SPY data files to GitHub.
# Scheduled via cron at 7:00 PM ET on weekdays.
#
# Cron entry (added by setup_server.sh):
#   CRON_TZ=America/New_York
#   0 19 * * 1-5 /home/ubuntu/claude_technical_analysis/deploy/push_results.sh

set -euo pipefail

REPO_DIR="/home/ubuntu/claude_technical_analysis"
LOG_TAG="push_results"
BRANCH="main"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [$LOG_TAG]  $*"; }

cd "$REPO_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree &>/dev/null; then
    log "ERROR: $REPO_DIR is not a git repository. Aborting."
    exit 1
fi

# ── Stage only the data output files ─────────────────────────────────────────
# We intentionally do NOT do 'git add -A' — only the two result CSVs go to
# GitHub from the server.  Code changes flow the other direction (dev → server).
DATA_FILES=("spy_data_1min.csv" "rising_wedge.csv" "crash.log")
STAGED=0
for f in "${DATA_FILES[@]}"; do
    if [ -f "$REPO_DIR/$f" ]; then
        git add "$REPO_DIR/$f"
        STAGED=$((STAGED + 1))
    else
        log "WARNING: $f not found — skipping"
    fi
done

if [ "$STAGED" -eq 0 ]; then
    log "No data files found. Nothing to push."
    exit 0
fi

# ── Commit only if there are staged changes ───────────────────────────────────
if git diff --cached --quiet; then
    log "No changes since last push. Nothing to commit."
    exit 0
fi

ROWS_SPY=$(wc -l < "$REPO_DIR/spy_data_1min.csv" 2>/dev/null || echo "?")
ROWS_WDG=$(wc -l < "$REPO_DIR/rising_wedge.csv"  2>/dev/null || echo "?")

git commit -m "auto: data update $(date '+%Y-%m-%d')  [spy=${ROWS_SPY}rows, wedge=${ROWS_WDG}rows]"
log "Committed data update."

# ── Push (retry once on transient network failure) ────────────────────────────
if git push origin "$BRANCH"; then
    log "Pushed to origin/$BRANCH successfully."
else
    log "First push attempt failed — retrying in 30 s ..."
    sleep 30
    git push origin "$BRANCH"
    log "Retry push succeeded."
fi
