#!/usr/bin/env bash
# Download datasets and models from Kaggle using the kaggle-cli.
#
# Usage:
#   bash .agents/skills/kaggle-platform/modules/kllm/scripts/cli_download.sh <dataset> [output-dir]
#
# Examples:
#   bash .agents/skills/kaggle-platform/modules/kllm/scripts/cli_download.sh kaggle/meta-kaggle data/external/meta-kaggle
#
# Prerequisites:
#   `uv sync --locked` completed in the repository
#   Credentials configured via `uv run kaggle auth login`, ~/.kaggle/access_token, KAGGLE_API_TOKEN, or legacy ~/.kaggle/kaggle.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
KAGGLE=(uv run --project "${REPO_ROOT}" kaggle)

DATASET="${1:-kaggle/meta-kaggle}"

# Validate the slug — Kaggle slugs are owner/dataset, ASCII-safe characters
# only. Reject anything that could traverse the filesystem when used in
# OUTPUT_DIR or the kaggle-cli `--unzip` step.
if ! printf '%s' "$DATASET" | grep -qE '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$'; then
    echo "[FAIL] dataset slug '$DATASET' is not in the expected owner/name form" >&2
    echo "       allowed chars: A-Z a-z 0-9 . _ - and exactly one '/'" >&2
    exit 2
fi

DATASET_PATH_SLUG="${DATASET//\//-}"
OUTPUT_DIR="${2:-${REPO_ROOT}/data/external/${DATASET_PATH_SLUG}}"

echo "============================================================"
echo "kaggle-cli: Download Dataset"
echo "============================================================"

# List files in the dataset
echo "--- Listing dataset files for ${DATASET} ---"
"${KAGGLE[@]}" datasets files "${DATASET}"

# Download the dataset
echo "--- Downloading dataset to ${OUTPUT_DIR} ---"
mkdir -p "${OUTPUT_DIR}"
"${KAGGLE[@]}" datasets download "${DATASET}" \
    --path "${OUTPUT_DIR}" \
    --unzip

echo "Dataset downloaded to ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}/"
