#!/usr/bin/env python3
"""Poll one explicitly identified Kaggle submission and log its scoring status."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

PENDING_STATUSES = {"pending", "running", "queued", "submitting"}
COMPLETE_STATUSES = {"complete", "completed", "finished", "scored"}
REPO_ROOT = Path(__file__).resolve().parents[4]


def read_project_competition(project_path: Path | None = None) -> str:
    path = project_path or REPO_ROOT / "project.yml"
    if not path.is_file():
        raise ValueError(f"project config does not exist: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    competition = config.get("competition")
    slug = competition.get("slug") if isinstance(competition, dict) else None
    if slug is None or str(slug).strip() in {"", "TODO", "TBD", "FIXME"}:
        raise ValueError("competition.slug is not configured in project.yml")
    return str(slug).strip()


def resolve_competition(explicit: str | None, project_path: Path | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return read_project_competition(project_path)


def run_submissions(competition: str) -> list[dict[str, str]]:
    cmd = [sys.executable, "-m", "kaggle", "competitions", "submissions"]
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


def select_submission(
    rows: list[dict[str, str]], submission_ref: str
) -> dict[str, str] | None:
    expected = submission_ref.strip()
    matches = [
        row
        for row in rows
        if first_value(row, ["ref", "submissionId", "id"]).strip() == expected
    ]
    if len(matches) > 1:
        raise ValueError(f"submission ref is ambiguous: {expected}")
    return matches[0] if matches else None


def row_summary(name: str, row: dict[str, str], start: float) -> tuple[bool, str]:
    status = classify_status(row)
    scoring_elapsed_min = int(round((time.time() - start) / 60))
    public_score = first_value(row, ["publicScore", "score"])
    private_score = first_value(row, ["privateScore"])
    submitted_at = first_value(row, ["date", "submittedDate", "submitted"])
    ref = first_value(row, ["ref", "submissionId", "id"])
    complete = status == "complete"
    line = (
        f"[{name}] scoring-elapsed: {scoring_elapsed_min} min, "
        f"submission-status: {status}, "
        f"publicScore: {public_score or '-'}, privateScore: {private_score or '-'}, "
        f"submitted: {submitted_at or '-'}, ref: {ref or '-'}"
    )
    return complete, line


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {line}\n")


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "submission"


def resolve_log_path(name: str, log_file: str | None, log_dir: str | None) -> Path:
    if log_file:
        return Path(log_file)
    if log_dir:
        return Path(log_dir) / f"submission_{safe_name(name)}.log"

    experiments_dir = REPO_ROOT / "experiments"
    exact = experiments_dir / name
    if exact.is_dir():
        return exact / "artifacts" / "submission-monitor.log"

    match = re.match(r"(exp\d+)", name.lower())
    if match:
        candidates = sorted(experiments_dir.glob(f"{match.group(1)}_*"))
        if len(candidates) == 1:
            return candidates[0] / "artifacts" / f"submission-monitor-{safe_name(name)}.log"

    return (
        Path(tempfile.gettempdir())
        / "kaggle-submission-monitor"
        / f"submission_{safe_name(name)}.log"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Experiment name, e.g. exp002_fold0")
    parser.add_argument(
        "--competition",
        default=None,
        help="Override project.yml competition.slug for this monitor run.",
    )
    parser.add_argument(
        "--submission-ref",
        required=True,
        help="Exact Kaggle submission ref returned for the submission being monitored.",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-minutes", type=int, default=720)
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument("--log-file", help="Transient polling log path.")
    log_group.add_argument(
        "--log-dir",
        help="Backward-compatible directory for a transient submission_<name>.log file.",
    )
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    args = parser.parse_args()

    try:
        competition = resolve_competition(args.competition)
    except ValueError as exc:
        parser.error(str(exc))

    log_path = resolve_log_path(args.name, args.log_file, args.log_dir)
    start = time.time()
    deadline = start + args.timeout_minutes * 60

    print(f"monitor log: {log_path}", flush=True)

    append_log(
        log_path,
        (
            f"[{args.name}] monitor started, competition: {competition}, "
            f"submission-ref: {args.submission_ref}"
        ),
    )

    while True:
        try:
            rows = run_submissions(competition)
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
            try:
                selected = select_submission(rows, args.submission_ref)
            except ValueError as exc:
                line = f"[{args.name}] {exc}"
                append_log(log_path, line)
                print(line, file=sys.stderr, flush=True)
                return 2
            if selected is None:
                line = (
                    f"[{args.name}] submission ref {args.submission_ref} not found; "
                    "refusing to monitor the latest submission implicitly"
                )
                append_log(log_path, line)
                print(line, flush=True)
            else:
                complete, line = row_summary(args.name, selected, start)
                append_log(log_path, line)
                print(line, flush=True)
                if complete:
                    return 0
            if args.once:
                return 1

        if time.time() >= deadline:
            append_log(log_path, f"[{args.name}] timeout after {args.timeout_minutes} min")
            return 124
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
