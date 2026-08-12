from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

try:
    from .config_utils import ROOT, load_project_config, project_path
except ImportError:  # Direct execution: `uv run python scripts/record_submission.py`
    from config_utils import ROOT, load_project_config, project_path

PROJECT_CONFIG = load_project_config()
SUBMISSIONS_PATH = project_path(PROJECT_CONFIG, "paths.submissions_file")
EXPERIMENTS_DIR = project_path(PROJECT_CONFIG, "paths.experiments_dir")
TABLE_HEADER = (
    "| バージョン | 日付 | 実験 | ファイル | 行数 | 列 | SHA256 | "
    "CV | Public LB | Private LB | submission ref | メモ |\n"
)
TABLE_SEPARATOR = (
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update one submission-ref-keyed row in SUBMISSIONS.md."
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument(
        "--submission-ref",
        required=True,
        help="Exact numeric Kaggle submission ref for this history row.",
    )
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--notes",
        default=None,
        help="Notes for a new row, or replacement notes when updating an existing ref",
    )
    parser.add_argument(
        "--allow-missing-file",
        action="store_true",
        help="Record a row even when the submission file is not available locally",
    )
    return parser.parse_args()


def next_version() -> str:
    versions = existing_versions()
    return f"v{max(versions, default=0) + 1:03d}"


def existing_versions() -> list[int]:
    if not SUBMISSIONS_PATH.exists():
        return []
    versions = [
        int(match.group(1))
        for line in SUBMISSIONS_PATH.read_text().splitlines()
        if (match := re.match(r"^\| v(\d{3}) \|", line))
    ]
    duplicates = sorted({version for version in versions if versions.count(version) > 1})
    if duplicates:
        labels = ", ".join(f"v{version:03d}" for version in duplicates)
        raise ValueError(f"duplicate submission versions: {labels}")
    return versions


def validate_new_version(version: str) -> None:
    match = re.fullmatch(r"v(\d{3})", version)
    if match is None:
        raise SystemExit(f"submission version must match vNNN: {version}")
    numeric = int(match.group(1))
    if numeric in existing_versions():
        raise SystemExit(f"submission version already exists: {version}")


def validate_submission_ref(value: str) -> str:
    submission_ref = value.strip()
    if re.fullmatch(r"\d+", submission_ref) is None:
        raise SystemExit(
            f"submission ref must be the exact numeric Kaggle ref: {value!r}"
        )
    return submission_ref


def display_metric(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def experiment_scores(experiment: str) -> tuple[str, str, str]:
    if Path(experiment).name != experiment:
        raise SystemExit(f"invalid experiment name: {experiment!r}")
    metrics_path = EXPERIMENTS_DIR / experiment / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(
            "experiment metrics do not exist; record scores with record-exp first: "
            f"{display_path(metrics_path)}"
        )
    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"invalid metrics JSON: {display_path(metrics_path)}: {exc}"
        ) from exc
    if not isinstance(metrics, dict):
        raise SystemExit(f"{display_path(metrics_path)} must contain a JSON object")
    return tuple(
        display_metric(metrics.get(key))
        for key in ("cv", "public_lb", "private_lb")
    )


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[str, str]:
    with path.open(newline="") as fp:
        reader = csv.reader(fp)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return str(rows), ",".join(header) if header else "-"


def ensure_table() -> None:
    if not SUBMISSIONS_PATH.exists():
        SUBMISSIONS_PATH.write_text("# 提出履歴\n\n" + TABLE_HEADER + TABLE_SEPARATOR)


def parse_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) != 12 or re.fullmatch(r"v\d{3}", cells[0]) is None:
        return None
    return cells


def find_submission_row(
    lines: list[str], submission_ref: str
) -> tuple[int, list[str]] | None:
    matches: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        cells = parse_table_row(line)
        if cells is None:
            continue
        refs = {item.strip() for item in cells[10].split(",")}
        if submission_ref in refs:
            matches.append((index, cells))
    if len(matches) > 1:
        raise SystemExit(f"submission ref appears in multiple rows: {submission_ref}")
    if not matches:
        return None
    index, cells = matches[0]
    if cells[10] != submission_ref:
        raise SystemExit(
            "submission ref is part of a legacy grouped row; split that row before "
            f"updating ref {submission_ref}"
        )
    return index, cells


def render_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def write_lines(lines: list[str]) -> None:
    SUBMISSIONS_PATH.write_text("\n".join(lines).rstrip() + "\n")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    SUBMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ensure_table()

    submission_ref = validate_submission_ref(args.submission_ref)
    cv, public_lb, private_lb = experiment_scores(args.experiment)
    file_path = resolve_path(args.file)
    if not file_path.exists() and not args.allow_missing_file:
        raise SystemExit(f"submission file does not exist: {display_path(file_path)}")

    display_file = display_path(file_path)
    rows, columns, file_hash = "-", "-", "-"
    if file_path.exists():
        rows, columns = csv_shape(file_path)
        file_hash = sha256_file(file_path)

    lines = SUBMISSIONS_PATH.read_text().splitlines()
    existing = find_submission_row(lines, submission_ref)
    if existing is not None:
        index, cells = existing
        version = cells[0]
        if args.version is not None and args.version != version:
            raise SystemExit(
                f"submission ref {submission_ref} already uses {version}, not {args.version}"
            )
        if cells[2] != args.experiment:
            raise SystemExit(
                f"submission ref {submission_ref} already belongs to {cells[2]}"
            )
        if file_path.exists():
            existing_hash = cells[6]
            if existing_hash != "-" and existing_hash != file_hash:
                raise SystemExit(
                    f"submission ref {submission_ref} already has different file evidence"
                )
            if existing_hash == "-":
                cells[3:7] = [display_file, rows, columns, file_hash]
        cells[7:10] = [cv, public_lb, private_lb]
        if args.notes is not None:
            cells[11] = args.notes
        lines[index] = render_table_row(cells)
        write_lines(lines)
        print(f"Updated submission {version} for {args.experiment}")
        return

    version = args.version or next_version()
    validate_new_version(version)
    cells = [
        version,
        date.today().isoformat(),
        args.experiment,
        display_file,
        rows,
        columns,
        file_hash,
        cv,
        public_lb,
        private_lb,
        submission_ref,
        args.notes or "-",
    ]
    lines.append(render_table_row(cells))
    write_lines(lines)
    print(f"Recorded submission {version} for {args.experiment}")


if __name__ == "__main__":
    main()
