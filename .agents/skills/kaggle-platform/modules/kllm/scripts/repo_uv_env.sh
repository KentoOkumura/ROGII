#!/usr/bin/env bash

# Keep repo-local uv execution writable in managed sandboxes while preserving
# explicit caller overrides.
: "${UV_CACHE_DIR:=/tmp/uv-cache}"
: "${PYTHONDONTWRITEBYTECODE:=1}"
export UV_CACHE_DIR PYTHONDONTWRITEBYTECODE
