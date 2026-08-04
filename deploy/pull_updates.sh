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

# Derive the repo root from this script's own location (deploy/..) so the
# path is correct regardless of which user/home the repo is cloned under.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="master"
LOG_TAG="pull_updates"

# Every service that runs code from this repo and therefore needs restarting
# when Python changes. wedge-api was missed when it was introduced: only
# live-monitor was restarted, so the API silently kept serving a stale module
# for days -- /scores and /signals ignored ?pattern= while the monitor was
# already writing pattern rows. Any future service belongs in this list.
SERVICES="live-monitor wedge-api"

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

# ── Restart every service that runs this repo's code ─────────────────────────
# A service that is not installed is skipped, not treated as an error: the API
# unit may legitimately not exist on an older box.
RC=0
for SERVICE in $SERVICES; do
    if ! systemctl list-unit-files "$SERVICE.service" &>/dev/null \
         || ! systemctl cat "$SERVICE" &>/dev/null; then
        log "$SERVICE is not installed — skipping."
        continue
    fi

    if systemctl is-active --quiet "$SERVICE"; then
        log "Restarting $SERVICE ..."
        if ! sudo -n systemctl restart "$SERVICE"; then
            log "ERROR: could not restart $SERVICE. If this is a sudo problem, run:"
            log "  echo \"$USER ALL=(root) NOPASSWD: /usr/bin/systemctl restart $SERVICE, /usr/bin/systemctl start $SERVICE\" | sudo tee /etc/sudoers.d/${SERVICE}-cron"
            RC=1
            continue
        fi
        sleep 5
        if systemctl is-active --quiet "$SERVICE"; then
            log "$SERVICE restarted successfully."
        else
            log "ERROR: $SERVICE failed to restart. Check: journalctl -u $SERVICE -n 50"
            RC=1
        fi
    else
        log "$SERVICE is not running — starting it now ..."
        sudo -n systemctl start "$SERVICE" || RC=1
    fi
done

# One service failing must not stop the others from being restarted, but the
# job should still exit non-zero so the failure is visible in the cron log.
exit $RC
