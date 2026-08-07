#!/usr/bin/env python3
"""Collect local Kaggle experiment context for strategy synthesis."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


TEXT_EXTS = {".md", ".txt", ".yaml", ".yml", ".json"}
IGNORE_PARTS = {".git", ".venv", "__pycache__", "input", "data", ".cache"}
SCORE_RE = re.compile(r"\b(?:cv|lb|public|private|score|auc|rmse|mae|f1|accuracy)\b[^\\n]{0,80}", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


def interesting(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORE_PARTS:
        return False
    name = path.name.lower()
    text = str(path).lower()
    return (
        name in {"kaggle_direction.md", "claudesummary.md", "submissions.md", "session_notes.md", "readme.md"}
        or "daily_report" in text
        or "daily_reports" in text
        or "experiment" in text
        or path.suffix.lower() == ".json" and name == "metrics.json"
    )


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in IGNORE_PARTS]
        base = Path(dirpath)
        for filename in filenames:
            yield base / filename


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
    files = []
    for path in iter_files(root):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTS:
            continue
        if interesting(path):
            files.append(path)
    files = sorted(files, key=lambda p: str(p.relative_to(root)))[: args.max_files]
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
