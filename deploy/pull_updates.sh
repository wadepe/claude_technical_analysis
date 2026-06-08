#!/usr/bin/env bash
# pull_updates.sh
#
# Pulls the latest code from GitHub and restarts the monitor service only
# if Python source files or model weights actually changed.
# Scheduled via cron at 3:00 AM ET on weekdays (one hour before pre-market open).
#
# Why 3 AM? Any code updates are pulled and the monitor is restarted before
# pre-market opens at 4 AM ET, so the updated code is live for the full session.
#
# Cron entry (added by setup_server.sh):
#   CRON_TZ=America/New_York
#   0 3 * * 1-5 /home/ubuntu/claude_technical_analysis/deploy/pull_updates.sh

set -euo pipefail

REPO_DIR="/home/ubuntu/claude_technical_analysis"
SERVICE="live-monitor"
BRANCH="master"
LOG_TAG="pull_updates"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [$LOG_TAG]  $*"; }

cd "$REPO_DIR"

# ── Record current HEAD before pulling ───────────────────────────────────────
BEFORE=$(git rev-parse HEAD)

# ── Fetch and pull ────────────────────────────────────────────────────────────
log "Fetching from origin/$BRANCH ..."
git fetch origin "$BRANCH"

# Check if there is anything new before merging
if git merge-base --is-ancestor "origin/$BRANCH" HEAD 2>/dev/null; then
    log "Already up to date. No restart needed."
    exit 0
fi

git pull origin "$BRANCH"
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
    log "HEAD unchanged after pull. No restart needed."
    exit 0
fi

log "Pulled new commits ($BEFORE -> $AFTER)"

# ── Decide whether to restart the service ────────────────────────────────────
# Only restart if .py source files or model weights changed.
# Pure data file changes (CSVs, logs) don't require a restart.
CHANGED=$(git diff "$BEFORE" "$AFTER" --name-only | grep -E '\.(py|h5)$' || true)

if [ -z "$CHANGED" ]; then
    log "No Python or weight files changed. Skipping restart."
    exit 0
fi

log "Code/model changes detected:"
echo "$CHANGED" | while read -r f; do log "  - $f"; done

# ── Restart the monitor service ───────────────────────────────────────────────
if systemctl is-active --quiet "$SERVICE"; then
    log "Restarting $SERVICE ..."
    sudo systemctl restart "$SERVICE"
    sleep 5
    if systemctl is-active --quiet "$SERVICE"; then
        log "$SERVICE restarted successfully."
    else
        log "ERROR: $SERVICE failed to restart. Check: journalctl -u $SERVICE -n 50"
        exit 1
    fi
else
    log "$SERVICE is not running — starting it now ..."
    sudo systemctl start "$SERVICE"
fi
