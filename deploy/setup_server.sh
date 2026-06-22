#!/usr/bin/env bash
# setup_server.sh
#
# One-time setup script for the Ubuntu deployment laptop.
# Run this once after cloning the repo on the server.
#
# Usage:
#   git clone git@github.com:YOUR_USERNAME/claude_technical_analysis.git
#   cd claude_technical_analysis/deploy
#   chmod +x setup_server.sh
#   ./setup_server.sh
#
# What this does:
#   1. Installs Python dependencies into a virtualenv
#   2. Generates an SSH key for GitHub (if none exists) and prints the
#      public key so you can add it to your GitHub account
#   3. Installs the systemd service
#   4. Installs the crontab entries
#   5. Enables and starts the monitor service

set -euo pipefail

# ── Configuration — edit these if your paths differ ──────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="live-monitor"
GITHUB_USER="YOUR_GITHUB_USERNAME"       # <-- change this
GITHUB_REPO="claude_technical_analysis"  # <-- change this if repo name differs
# Interpreter used to build the venv. Defaults to python3, but can be overridden
# at run time without editing this file (keeps the working tree clean for the
# nightly git pull), e.g. to pick a specific version:
#   PYTHON=python3.12 ./setup_server.sh
PYTHON="${PYTHON:-python3}"
# ─────────────────────────────────────────────────────────────────────────────

DEPLOY_DIR="$REPO_DIR/deploy"
SRC_DIR="$REPO_DIR/source_code"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$HOME/logs"

echo "============================================================"
echo "  Rising Wedge Monitor — Server Setup"
echo "  Repo: $REPO_DIR"
echo "============================================================"
echo ""

# ── 1. Python virtualenv + dependencies ──────────────────────────────────────
echo "[1/5] Setting up Python virtualenv ..."

# A venv left over from a failed run (or created without pip because the
# python3-venv package was missing) must be rebuilt, not reused — otherwise
# the pip calls below fail with a confusing 'no such file' three steps later.
if [ -d "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/pip" ]; then
    echo "  Existing virtualenv at $VENV_DIR has no pip — removing it to rebuild."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    if ! $PYTHON -m venv "$VENV_DIR"; then
        echo "ERROR: '$PYTHON -m venv' failed. On Debian/Ubuntu the venv module"
        echo "ships separately — install it and re-run this script:"
        echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
        exit 1
    fi
    echo "  Created virtualenv at $VENV_DIR"
else
    echo "  Virtualenv already exists."
fi

# venv can exit 0 yet produce no pip when ensurepip is unavailable (the
# python3-venv package is missing). Fail loudly with the fix instead of
# limping into the pip install below.
if [ ! -x "$VENV_DIR/bin/pip" ]; then
    echo "ERROR: $VENV_DIR/bin/pip was not created — the python3-venv package is"
    echo "likely missing. Install it, then re-run this script:"
    echo "  sudo apt update && sudo apt install -y python3-venv python3-pip"
    exit 1
fi

# ── Verify the interpreter version matches the pinned TensorFlow ──────────────
# tensorflow is pinned to 2.15.0 below (see why in the install block), and TF
# 2.15 only ships wheels for Python 3.9–3.11. A venv outside that range fails
# deep in pip with a cryptic "Could not find a version that satisfies the
# requirement tensorflow"; catch it here with actionable guidance instead.
# If the TF pin changes, update this range to match its supported Pythons.
PY_MIN_MINOR=9
PY_MAX_MINOR=11
PY_MINOR=$("$VENV_DIR/bin/python" -c 'import sys; print(sys.version_info[1])')
PY_VER=$("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$PY_MINOR" -lt "$PY_MIN_MINOR" ] || [ "$PY_MAX_MINOR" -lt "$PY_MINOR" ]; then
    echo "ERROR: venv Python is $PY_VER, but the pinned TensorFlow (2.15.0) only"
    echo "has wheels for 3.$PY_MIN_MINOR through 3.$PY_MAX_MINOR. Rebuild the venv"
    echo "against a supported interpreter via the PYTHON override, e.g.:"
    echo "  rm -rf \"$VENV_DIR\""
    echo "  PYTHON=\"\$(uv python find 3.11)\" ./deploy/setup_server.sh"
    exit 1
fi

"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install \
    "tensorflow==2.15.0" \
    yfinance \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    matplotlib \
    pandas_market_calendars \
    --quiet
# tensorflow pinned to 2.15.0 to match the training environment: the saved
# cnn_best.weights.h5 files were written with Keras 2 (TF <= 2.15). TF 2.16+
# bundles Keras 3, whose weights format differs, so model.load_weights() fails
# on those files. Keep server inference on the same major Keras it trained with.
# (TF 2.15 also constrains numpy < 2 automatically — no separate numpy pin needed.)
# pandas_market_calendars: NYSE holiday / half-day calendar for live_monitor.py.

echo "  Python dependencies installed."
echo ""

# ── 2. SSH key for GitHub ─────────────────────────────────────────────────────
echo "[2/5] Checking SSH key for GitHub ..."
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY" ]; then
    ssh-keygen -t ed25519 -C "wedge-monitor@$(hostname)" -f "$SSH_KEY" -N ""
    echo "  Generated new SSH key."
else
    echo "  SSH key already exists."
fi

echo ""
echo "  *** ACTION REQUIRED ***"
echo "  Add this public key to your GitHub account:"
echo "  Settings → SSH and GPG keys → New SSH key"
echo ""
cat "${SSH_KEY}.pub"
echo ""
read -rp "  Press Enter once you have added the key to GitHub ..."

# Test the connection
echo "  Testing GitHub SSH connection ..."
if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "  GitHub SSH authentication OK."
else
    echo "  WARNING: Could not verify GitHub auth — check the key and try again."
    echo "  You can re-test manually with: ssh -T git@github.com"
fi
echo ""

# Ensure the remote uses SSH (not HTTPS)
cd "$REPO_DIR"
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "")
if echo "$CURRENT_REMOTE" | grep -q "https://"; then
    git remote set-url origin "git@github.com:${GITHUB_USER}/${GITHUB_REPO}.git"
    echo "  Switched remote to SSH: $(git remote get-url origin)"
else
    echo "  Remote already uses SSH: $CURRENT_REMOTE"
fi
echo ""

# ── 3. Make deploy scripts executable ────────────────────────────────────────
echo "[3/5] Setting script permissions ..."
chmod +x "$DEPLOY_DIR/push_results.sh"
chmod +x "$DEPLOY_DIR/pull_updates.sh"
mkdir -p "$LOG_DIR"
echo "  Done."
echo ""

# ── 4. Install systemd service ────────────────────────────────────────────────
echo "[4/5] Installing systemd service ..."

# Patch the service file with the actual username and repo path
SERVICE_TMP=$(mktemp)
sed \
    -e "s|User=ubuntu|User=$USER|g" \
    -e "s|/home/ubuntu/claude_technical_analysis|$REPO_DIR|g" \
    "$DEPLOY_DIR/live-monitor.service" > "$SERVICE_TMP"

sudo cp "$SERVICE_TMP" "/etc/systemd/system/${SERVICE_NAME}.service"
rm "$SERVICE_TMP"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "  Service installed and enabled at boot."
echo ""

# ── 5. Install crontab entries ────────────────────────────────────────────────
echo "[5/5] Installing crontab entries ..."

# Build the new crontab block (idempotent — removes old block first)
CRON_MARKER="# rising-wedge-monitor"
CRON_BLOCK="$CRON_MARKER
CRON_TZ=America/New_York
# 7:00 PM ET weekdays — push SPY data to GitHub
0 19 * * 1-5 $DEPLOY_DIR/push_results.sh >> $LOG_DIR/push_results.log 2>&1
# 3:00 AM ET weekdays — pull code updates, restart monitor if code changed
0 3 * * 1-5 $DEPLOY_DIR/pull_updates.sh >> $LOG_DIR/pull_updates.log 2>&1
$CRON_MARKER end"

# Remove any existing block, then append new one
EXISTING_CRON=$(crontab -l 2>/dev/null || true)
CLEAN_CRON=$(echo "$EXISTING_CRON" | \
    awk "/$CRON_MARKER/{found=1} !found{print} /$CRON_MARKER end/{found=0}")
echo -e "${CLEAN_CRON}\n${CRON_BLOCK}" | crontab -

echo "  Crontab installed."
echo ""

# ── Start the service ─────────────────────────────────────────────────────────
echo "Starting $SERVICE_NAME ..."
sudo systemctl start "$SERVICE_NAME"
sleep 3

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo ""
    echo "============================================================"
    echo "  Setup complete!  $SERVICE_NAME is running."
    echo "============================================================"
    echo ""
    echo "  Useful commands:"
    echo "    sudo systemctl status  $SERVICE_NAME"
    echo "    journalctl -u $SERVICE_NAME -f"
    echo "    sudo systemctl restart $SERVICE_NAME"
    echo "    sudo systemctl stop    $SERVICE_NAME"
    echo ""
    echo "  Schedule:"
    echo "    3:00 AM ET (weekdays) — pull code updates from GitHub"
    echo "    7:00 PM ET (weekdays) — push SPY data to GitHub"
    echo ""
    echo "  Deployment workflow:"
    echo "    1. Edit code on dev laptop"
    echo "    2. git push origin master"
    echo "    3. Server auto-pulls at 3 PM and restarts if .py/.h5 changed"
    echo ""
else
    echo "ERROR: $SERVICE_NAME failed to start."
    echo "Check logs: journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi
