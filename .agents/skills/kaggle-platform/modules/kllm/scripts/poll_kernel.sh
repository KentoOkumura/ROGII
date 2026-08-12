#!/usr/bin/env bash
# Follow Kaggle kernel live logs and download output.
#
# The filename is retained for compatibility. This script no longer polls
# `kaggle kernels status`, because that endpoint can return transient 500s.
#
# Usage:
#   bash .agents/skills/kaggle-platform/modules/kllm/scripts/poll_kernel.sh <kernel-slug> [output-dir]
#
# Arguments:
#   kernel-slug    — e.g., "username/kernel-name"
#   output-dir     — directory to save output (default: /tmp/kaggle-output/<kernel-slug>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
KAGGLE=(uv run --project "${REPO_ROOT}" kaggle)

KERNEL_SLUG="${1:?Usage: poll_kernel.sh <kernel-slug> [output-dir]}"
if [[ ! "${KERNEL_SLUG}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "[FAIL] kernel slug '${KERNEL_SLUG}' is not in the expected owner/name form" >&2
    exit 2
fi
KERNEL_PATH_SLUG="${KERNEL_SLUG//\//-}"
OUTPUT_DIR="${2:-/tmp/kaggle-output/${KERNEL_PATH_SLUG}}"
if [[ $# -gt 2 ]]; then
    echo "poll-interval is no longer supported; live logs do not use polling intervals" >&2
    exit 2
fi

echo "Following kernel logs: ${KERNEL_SLUG}"
echo "Output dir:     ${OUTPUT_DIR}"
echo ""

"${KAGGLE[@]}" kernels logs -f "${KERNEL_SLUG}"
echo "Live log stream closed. Inspect the final log or Kaggle UI if completion is unclear."
echo "Downloading output..."
mkdir -p "${OUTPUT_DIR}"
"${KAGGLE[@]}" kernels output "${KERNEL_SLUG}" --path "${OUTPUT_DIR}"
echo "Output saved to ${OUTPUT_DIR}/"
ls -la "${OUTPUT_DIR}/"
