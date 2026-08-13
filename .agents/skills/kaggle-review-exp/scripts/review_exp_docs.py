#!/usr/bin/env python3
"""Find and score Kaggle experiment documentation for completeness."""

from __future__ import annotations

import argparse
import re
import tempfile
from datetime import datetime
from pathlib import Path

IGNORE_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "features",
    "variants",
    "kaggle",
}
GLOBAL_RECORDS = (
    "backlog/KAGGLE_DIRECTION.md",
    "experiment_summary.md",
    "SUBMISSIONS.md",
    "docs/surveys/README.md",
)
CANONICAL_EXPERIMENT_RECORDS = {
    "README.md",
    "requirements.md",
    "SESSION_NOTES.md",
    "result.md",
    "metrics.json",
    "config.yaml",
}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
CHECKS = {
    "purpose/hypothesis": re.compile(r"目的|仮説|hypothesis|objective|goal", re.IGNORECASE),
    "base/changes": re.compile(r"base|baseline|ベース|変更|changes?|diff", re.IGNORECASE),
    "validation": re.compile(r"cv|fold|validation|oof|split|検証", re.IGNORECASE),
    "results": re.compile(r"result|score|lb|public|private|結果|auc|rmse|mae|f1", re.IGNORECASE),
    "artifacts": re.compile(r"artifact|checkpoint|ckpt|model|submission|成果物", re.IGNORECASE),
    "next action": re.compile(r"next|todo|action|次|課題", re.IGNORECASE),
}


def candidate_files(root: Path, exp: str) -> list[Path]:
    files: set[Path] = set()
    lowered = exp.lower()

    for relative in GLOBAL_RECORDS:
        path = root / relative
        if path.is_file():
            files.add(path)

    base_dir = root / "experiments"
    if base_dir.exists():
        for candidate_dir in base_dir.iterdir():
            if not candidate_dir.is_dir() or lowered not in candidate_dir.name.lower():
                continue
            for path in candidate_dir.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in TEXT_SUFFIXES
                    and not (set(path.relative_to(candidate_dir).parts) & IGNORE_PARTS)
                ):
                    files.add(path)

    surveys_dir = root / "docs" / "surveys"
    if surveys_dir.exists():
        for path in surveys_dir.glob("*.md"):
            if path.name == "README.md":
                continue
            if lowered in path.name.lower() or lowered in path.read_text(
                encoding="utf-8", errors="replace"
            ).lower():
                files.add(path)

    return sorted(files, key=lambda path: str(path.relative_to(root)))


def review_file(path: Path) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}
    checks = {name: bool(pattern.search(text)) for name, pattern in CHECKS.items()}
    headings = re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE)[:10]
    score_lines = re.findall(
        r"(?i).{0,20}(?:cv|lb|public|private|score|auc|rmse|mae|f1).{0,100}",
        text,
    )[:8]
    return {
        "path": str(path),
        "checks": checks,
        "headings": headings,
        "score_lines": [line.strip() for line in score_lines],
    }


def evidence_scope(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if (
        len(relative.parts) == 3
        and relative.parts[0] == "experiments"
        and relative.name in CANONICAL_EXPERIMENT_RECORDS
    ):
        return "target evidence"
    if relative.parts and relative.parts[0] == "experiments":
        return "supporting material"
    return "context"


def collect_reviews(root: Path, files: list[Path]) -> list[dict[str, object]]:
    reviews: list[dict[str, object]] = []
    for path in files:
        review = review_file(path)
        review["scope"] = evidence_scope(root, path)
        reviews.append(review)
    return reviews


def target_evidence_summary(
    reviews: list[dict[str, object]],
) -> tuple[bool, list[str]]:
    target_reviews = [
        review
        for review in reviews
        if review.get("scope") == "target evidence" and "error" not in review
    ]
    present = {key: False for key in CHECKS}
    for review in target_reviews:
        checks = review.get("checks")
        if not isinstance(checks, dict):
            continue
        for key in present:
            present[key] = present[key] or checks.get(key) is True
    return bool(target_reviews), [key for key, found in present.items() if not found]


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
    for review in reviews:
        lines.append(f"## {Path(str(review['path'])).relative_to(root)}")
        lines.append(f"- scope: {review.get('scope', 'context')}")
        if "error" in review:
            lines.append(f"- error: {review['error']}")
            lines.append("")
            continue
        checks = review["checks"]
        assert isinstance(checks, dict)
        for key, ok in checks.items():
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
    has_target_evidence, missing = target_evidence_summary(reviews)
    if not has_target_evidence:
        lines.append("- No canonical target experiment records were found.")
    if missing:
        lines.append(
            "- Missing evidence in target experiment records: "
            + ", ".join(missing)
        )
    else:
        lines.append(
            "- Core evidence categories are present in target experiment records."
        )
    lines.append("- Context and supporting material do not satisfy target evidence checks.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp")
    parser.add_argument("--root", default=".")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when target experiment records lack an evidence category.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = candidate_files(root, args.exp)
    reviews = collect_reviews(root, files)
    output = render(args.exp, root, reviews)
    print(output)
    if args.write:
        out_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else Path(tempfile.gettempdir()) / "kaggle-experiment-review"
        )
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{datetime.now().strftime('%Y%m%d')}-{args.exp}-review.md"
        out_path.write_text(output, encoding="utf-8")
        print(f"saved: {out_path}")
    has_target_evidence, missing = target_evidence_summary(reviews)
    return 1 if args.strict and (not has_target_evidence or bool(missing)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
