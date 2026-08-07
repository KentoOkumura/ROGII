#!/usr/bin/env python3
"""Poll Kaggle submissions and log the latest scored result for an experiment."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PENDING_STATUSES = {"pending", "running", "queued", "submitting"}
COMPLETE_STATUSES = {"complete", "completed", "finished", "scored"}


def read_default_competition() -> str | None:
    env_value = os.environ.get("KAGGLE_COMPETITION")
    if env_value:
        return env_value.strip()
    path = Path(".kaggle_competition")
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def run_submissions(competition: str | None) -> list[dict[str, str]]:
    cmd = ["kaggle", "competitions", "submissions"]
    if competition:
        cmd.append(competition)
    cmd.extend(["-v", "-q"])
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line.startswith("ref,"):
            lines = lines[index:]
            break
    if not lines:
        return []
    return list(csv.DictReader(lines))


def first_value(row: dict[str, str], names: list[str]) -> str:
    lowered = {str(key).lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return ""


def classify_status(row: dict[str, str]) -> str:
    raw = first_value(row, ["status", "Status", "errorDescription"]).strip().lower()
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    if not raw:
        public_score = first_value(row, ["publicScore", "score"])
        return "complete" if public_score else "unknown"
    if raw in COMPLETE_STATUSES:
        return "complete"
    if raw in PENDING_STATUSES:
        return raw
    return raw


def row_summary(name: str, row: dict[str, str], start: float) -> tuple[bool, str]:
    status = classify_status(row)
    elapsed_min = int(round((time.time() - start) / 60))
    public_score = first_value(row, ["publicScore", "score"])
    private_score = first_value(row, ["privateScore"])
    submitted_at = first_value(row, ["date", "submittedDate", "submitted"])
    ref = first_value(row, ["ref", "submissionId", "id"])
    complete = status == "complete"
    line = (
        f"[{name}] run-time: {elapsed_min} min, status: {status}, "
        f"publicScore: {public_score or '-'}, privateScore: {private_score or '-'}, "
        f"submitted: {submitted_at or '-'}, ref: {ref or '-'}"
    )
    return complete, line


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {line}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Experiment name, e.g. exp002_fold0")
    parser.add_argument("--competition", default=read_default_competition())
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-minutes", type=int, default=720)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    args = parser.parse_args()

    log_path = Path(args.log_dir) / f"submission_{args.name}.log"
    start = time.time()
    deadline = start + args.timeout_minutes * 60

    append_log(
        log_path,
        f"[{args.name}] monitor started, competition: {args.competition or '<kaggle default>'}",
    )

    while True:
        try:
            rows = run_submissions(args.competition)
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            if "403" in message or "Forbidden" in message:
                message = (
                    f"{message} | Kaggle denied access to submission history. "
                    "Confirm that the account has joined/accepted rules for this competition, "
                    "or configure the correct Kaggle credentials."
                )
            append_log(log_path, f"[{args.name}] kaggle CLI error: {message}")
            print(message, file=sys.stderr)
            if args.once:
                return 2
        else:
            if not rows:
                line = f"[{args.name}] no submissions found"
            else:
                complete, line = row_summary(args.name, rows[0], start)
                append_log(log_path, line)
                print(line, flush=True)
                if complete:
                    return 0
            if args.once:
                print(line, flush=True)
                return 1

        if time.time() >= deadline:
            append_log(log_path, f"[{args.name}] timeout after {args.timeout_minutes} min")
            return 124
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
