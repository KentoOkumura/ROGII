#!/usr/bin/env python3
"""Fetch Kaggle notebooks for a competition with metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._/-]+", "-", value)
    return value.strip("-").replace("/", "__")


def list_kernels(competition: str, page_size: int, sort_by: str) -> list[dict[str, str]]:
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "list",
        "--competition",
        competition,
        "--sort-by",
        sort_by,
        "--page-size",
        str(page_size),
        "-v",
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [
        line
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("Warning:")
    ]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def kernel_ref(row: dict[str, str]) -> str | None:
    for key in ("ref", "kernelRef", "id"):
        value = row.get(key)
        if value and "/" in value:
            return value.strip()
    owner = row.get("author") or row.get("owner") or row.get("userName")
    slug = row.get("slug") or row.get("title")
    if owner and slug:
        return f"{owner.strip()}/{slugify(slug)}"
    return None


def validate_pulled_kernel(target: Path) -> str | None:
    metadata_path = target / "kernel-metadata.json"
    if not metadata_path.is_file():
        return "kernel-metadata.json is missing"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"kernel-metadata.json is unreadable: {exc}"
    if not isinstance(metadata, dict):
        return "kernel-metadata.json must contain an object"
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file.strip():
        return "kernel-metadata.json has no code_file"
    target_root = target.resolve()
    code_path = (target / code_file).resolve()
    if not code_path.is_relative_to(target_root):
        return f"code_file resolves outside the target directory: {code_file}"
    if not code_path.is_file():
        return f"code_file is missing: {code_file}"
    if code_path.stat().st_size == 0:
        return f"code_file is empty: {code_file}"
    return None


def pull_kernel(ref: str, output_dir: Path, force: bool, dry_run: bool, retries: int) -> str:
    target = output_dir / slugify(ref)
    if target.exists() and not force:
        validation_error = validate_pulled_kernel(target)
        if validation_error is None:
            return f"skip complete {ref} -> {target}"
    if dry_run:
        return f"would pull {ref} -> {target}"
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "kernels",
        "pull",
        ref,
        "-p",
        str(target),
        "-m",
    ]
    last_error = ""
    for attempt in range(1, retries + 1):
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            validation_error = validate_pulled_kernel(target)
            if validation_error is None:
                return f"pulled {ref} -> {target}"
            last_error = validation_error
        else:
            output = (proc.stderr or proc.stdout).strip().splitlines()
            last_error = output[-1] if output else f"kaggle exited with {proc.returncode}"
        if attempt < retries:
            time.sleep(2 * attempt)
    return f"failed {ref} -> {target}: {last_error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sort-by", default="voteCount")
    parser.add_argument("--listing-name", default="kernel_listing.csv")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"docs/notebooks/{args.competition}")
    rows = list_kernels(args.competition, max(args.limit, 20), args.sort_by)
    refs = []
    for row in rows:
        ref = kernel_ref(row)
        if ref and ref not in refs:
            refs.append(ref)
        if len(refs) >= args.limit:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    listing = output_dir / args.listing_name
    if rows and not args.dry_run:
        with listing.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    failures = 0
    for ref in refs:
        result = pull_kernel(ref, output_dir, args.force, args.dry_run, args.retries)
        print(result)
        failures += result.startswith("failed ")
    print(f"total_refs={len(refs)} failures={failures} output_dir={output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
