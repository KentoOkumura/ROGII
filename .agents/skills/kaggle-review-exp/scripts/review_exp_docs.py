#!/usr/bin/env python3
"""Find and score Kaggle experiment documentation for completeness."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path


IGNORE_PARTS = {".git", ".venv", "__pycache__", "input", "data", ".cache"}
CHECKS = {
    "purpose/hypothesis": re.compile(r"目的|仮説|hypothesis|objective|goal", re.IGNORECASE),
    "base/changes": re.compile(r"base|baseline|ベース|変更|changes?|diff", re.IGNORECASE),
    "validation": re.compile(r"cv|fold|validation|oof|split|検証", re.IGNORECASE),
    "results": re.compile(r"result|score|lb|public|private|結果|auc|rmse|mae|f1", re.IGNORECASE),
    "artifacts": re.compile(r"artifact|checkpoint|ckpt|model|submission|成果物", re.IGNORECASE),
    "next action": re.compile(r"next|todo|action|次|課題", re.IGNORECASE),
}


def candidate_files(root: Path, exp: str) -> list[Path]:
    files: list[Path] = []
    lowered = exp.lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in IGNORE_PARTS]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if path.suffix.lower() not in {".md", ".txt", ".json", ".yaml", ".yml"}:
                continue
            text = str(path.relative_to(root)).lower()
            name = path.name.lower()
            if lowered in text or name in {"session_notes.md", "submissions.md"} and lowered in text:
                files.append(path)
    # Include common global logs because they may mention the experiment.
    for pattern in ("submit/SUBMISSIONS.md", "daily_reports/*.md", "KAGGLE_DIRECTION.md", "claudeSummary.md"):
        files.extend(root.glob(pattern))
    return sorted(set(files), key=lambda p: str(p.relative_to(root)))


def review_file(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}
    checks = {name: bool(pattern.search(text)) for name, pattern in CHECKS.items()}
    headings = re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE)[:10]
    score_lines = re.findall(r"(?i).{0,20}(?:cv|lb|public|private|score|auc|rmse|mae|f1).{0,100}", text)[:8]
    return {
        "path": str(path),
        "checks": checks,
        "headings": headings,
        "score_lines": [line.strip() for line in score_lines],
    }


def render(exp: str, root: Path, reviews: list[dict[str, object]]) -> str:
    lines = [
        "# Kaggle Experiment Review",
        "",
        f"- experiment: {exp}",
        f"- root: {root}",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- files: {len(reviews)}",
        "",
    ]
    missing_global = {key: True for key in CHECKS}
    for review in reviews:
        lines.append(f"## {Path(str(review['path'])).relative_to(root)}")
        if "error" in review:
            lines.append(f"- error: {review['error']}")
            lines.append("")
            continue
        checks = review["checks"]
        assert isinstance(checks, dict)
        for key, ok in checks.items():
            if ok:
                missing_global[key] = False
            lines.append(f"- {key}: {'OK' if ok else 'MISSING'}")
        headings = review.get("headings") or []
        if headings:
            lines.append("- headings: " + " | ".join(str(x) for x in headings[:8]))
        scores = review.get("score_lines") or []
        if scores:
            lines.append("- score lines:")
            for score in scores:
                lines.append(f"  - {score}")
        lines.append("")

    lines.append("## Summary")
    missing = [key for key, missing in missing_global.items() if missing]
    if missing:
        lines.append("- Missing evidence across collected files: " + ", ".join(missing))
    else:
        lines.append("- Core evidence categories are present somewhere in collected files.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp")
    parser.add_argument("--root", default=".")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output-dir", default=".log/review")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = candidate_files(root, args.exp)
    reviews = [review_file(path) for path in files]
    output = render(args.exp, root, reviews)
    print(output)
    if args.write:
        out_dir = root / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{datetime.now().strftime('%Y%m%d')}-{args.exp}-review.md"
        out_path.write_text(output, encoding="utf-8")
        print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
