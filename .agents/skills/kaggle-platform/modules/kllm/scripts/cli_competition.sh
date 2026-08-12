#!/usr/bin/env bash
# Inspect a competition and download its data using kaggle-cli.
#
# kaggle-cli supports:
#   kaggle competitions list       — list active competitions
#   kaggle competitions files      — list files in a competition
#   kaggle competitions download   — download competition data
#   kaggle competitions submissions — list your submissions
#   kaggle competitions leaderboard — view the leaderboard
#
# NOTE: There is no dedicated CLI command to "join" or "register for" a competition.
#       You must accept the competition rules via the Kaggle website first.
#       After that, repository-approved submission operations work via CLI.
#
# Prerequisites:
#   `uv sync --locked` completed in the repository
#   Credentials configured via `uv run kaggle auth login`, ~/.kaggle/access_token, KAGGLE_API_TOKEN, or legacy ~/.kaggle/kaggle.json
#
# Usage:
#   bash .agents/skills/kaggle-platform/modules/kllm/scripts/cli_competition.sh <competition> [download-dir]
#
# Arguments:
#   competition      — competition slug, e.g., "titanic"
#   download-dir     — directory to save competition data (default: data/raw/<competition>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
KAGGLE=(uv run --project "${REPO_ROOT}" kaggle)

COMPETITION="${1:?Usage: cli_competition.sh <competition> [download-dir]}"
if [[ ! "${COMPETITION}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[FAIL] competition slug '${COMPETITION}' contains unsupported characters" >&2
    exit 2
fi
DOWNLOAD_DIR="${2:-${REPO_ROOT}/data/raw/${COMPETITION}}"

echo "============================================================"
echo "Step 1: List available competitions"
echo "============================================================"

"${KAGGLE[@]}" competitions list --sort-by latestDeadline

echo ""
echo "============================================================"
echo "Step 2: Accept competition rules (MUST be done via UI)"
echo "============================================================"
echo ""
echo "IMPORTANT: Before your first submission, you must accept the"
echo "competition rules at:"
echo "  https://www.kaggle.com/c/${COMPETITION}/rules"
echo ""
echo "Click 'I Understand and Accept' on that page."
echo "This is a one-time step per competition."
echo ""

echo "============================================================"
echo "Step 3: Download competition data"
echo "============================================================"

# List competition files
echo "--- Competition files ---"
"${KAGGLE[@]}" competitions files "${COMPETITION}"

# Download all competition data
echo "--- Downloading competition data ---"
mkdir -p "${DOWNLOAD_DIR}"
"${KAGGLE[@]}" competitions download "${COMPETITION}" \
    --path "${DOWNLOAD_DIR}"

echo "Competition data downloaded to ${DOWNLOAD_DIR}/"
ls -la "${DOWNLOAD_DIR}/"

echo ""
echo "============================================================"
echo "Step 4: Submission handoff"
echo "============================================================"
echo "This helper does not submit. This repository is configured for Notebook-only"
echo "code submissions; validate the notebook output and use task submit-code."

echo ""
echo "============================================================"
echo "Step 5: Check existing submissions"
echo "============================================================"

# List your submissions
"${KAGGLE[@]}" competitions submissions "${COMPETITION}"

echo ""
echo "============================================================"
echo "Step 6: View the leaderboard"
echo "============================================================"

# View the leaderboard (top entries)
"${KAGGLE[@]}" competitions leaderboard "${COMPETITION}" --show
