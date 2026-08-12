#!/usr/bin/env python3
"""Collect local Kaggle experiment context for strategy synthesis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CANONICAL_FILES = (
    "KAGGLE_DIRECTION.md",
    "experiment_summary.md",
    "SUBMISSIONS.md",
    "docs/surveys/README.md",
)
EXPERIMENT_RECORDS = ("SESSION_NOTES.md", "metrics.json", "result.md")
SCORE_RE = re.compile(
    r"\b(?:cv|lb|public|private|score|auc|rmse|mae|f1|accuracy)\b[^\n]{0,80}",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)
EXPERIMENT_NUMBER_RE = re.compile(r"^exp(\d+)")
BACKLOG_LINK_RE = re.compile(
    r"^\| (?P<priority>[^|]+) \| `(?:HYP-\d{8}-\d{2}|未整理)` \| "
    r"\[`(?P<name>[a-z0-9_]+)`\]"
    r"\(docs/backlog/(?P=name)\.md\) \|"
)
ACTIVE_PRIORITY_RE = re.compile(r"(?:^|・)(?:最優先|P[0-2])(?:・|$)")


def experiment_sort_key(path: Path) -> tuple[int, int, str]:
    match = EXPERIMENT_NUMBER_RE.match(path.name.lower())
    number = int(match.group(1)) if match else -1
    return (number, path.stat().st_mtime_ns, path.name)


def prioritized_backlog_files(root: Path) -> list[Path]:
    direction = root / "KAGGLE_DIRECTION.md"
    if not direction.is_file():
        return []
    selected: list[Path] = []
    for line in direction.read_text(errors="replace").splitlines():
        match = BACKLOG_LINK_RE.match(line)
        if match and ACTIVE_PRIORITY_RE.search(match.group("priority")):
            selected.append(root / "docs" / "backlog" / f"{match.group('name')}.md")
    return selected


def candidate_files(root: Path, max_files: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path.is_file() and path not in seen and len(selected) < max_files:
            selected.append(path)
            seen.add(path)

    for relative in CANONICAL_FILES:
        add(root / relative)

    for path in prioritized_backlog_files(root):
        add(path)

    experiments_dir = root / "experiments"
    experiment_dirs = []
    if experiments_dir.exists():
        experiment_dirs = sorted(
            (path for path in experiments_dir.iterdir() if path.is_dir()),
            key=experiment_sort_key,
            reverse=True,
        )
    for experiment_dir in experiment_dirs:
        for filename in EXPERIMENT_RECORDS:
            add(experiment_dir / filename)
        if len(selected) >= max_files:
            break
    return selected


def summarize_file(path: Path, root: Path, max_bytes: int) -> dict[str, object]:
    try:
        raw = path.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(path.relative_to(root)), "error": str(exc)}
    headings = HEADING_RE.findall(text)[:12]
    scores = [match.group(0).strip() for match in SCORE_RE.finditer(text)][:12]
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "bytes": stat.st_size,
        "headings": headings,
        "score_lines": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-bytes", type=int, default=8000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = candidate_files(root, args.max_files)
    summaries = [summarize_file(path, root, args.max_bytes) for path in files]

    if args.json:
        print(json.dumps({"root": str(root), "files": summaries}, ensure_ascii=False, indent=2))
        return 0

    print(f"# Kaggle Strategy Context\n\nroot: `{root}`\nfiles: {len(summaries)}\n")
    for item in summaries:
        print(f"## {item['path']}")
        if "error" in item:
            print(f"- error: {item['error']}\n")
            continue
        print(f"- bytes: {item['bytes']}")
        headings = item.get("headings") or []
        scores = item.get("score_lines") or []
        if headings:
            print("- headings: " + " | ".join(str(x) for x in headings[:8]))
        if scores:
            print("- score lines:")
            for score in scores[:8]:
                print(f"  - {score}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
