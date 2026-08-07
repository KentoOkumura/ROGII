from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import date
from pathlib import Path

from config_utils import ROOT

SUBMISSIONS_PATH = ROOT / "submissions" / "SUBMISSIONS.md"
TABLE_HEADER = (
    "| バージョン | 日付 | 実験 | ファイル | 行数 | 列 | SHA256 | "
    "CV | Public LB | Private LB | メモ |\n"
)
TABLE_SEPARATOR = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one row to submissions/SUBMISSIONS.md.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--cv", default="-")
    parser.add_argument("--public-lb", default="-")
    parser.add_argument("--private-lb", default="-")
    parser.add_argument("--notes", default="-")
    parser.add_argument(
        "--allow-missing-file",
        action="store_true",
        help="Record a row even when the submission file is not available locally",
    )
    return parser.parse_args()


def next_version() -> str:
    if not SUBMISSIONS_PATH.exists():
        return "v001"

    count = 0
    for line in SUBMISSIONS_PATH.read_text().splitlines():
        if line.startswith("| v"):
            count += 1
    return f"v{count + 1:03d}"


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    SUBMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    version = args.version or next_version()
    file_path = resolve_path(args.file)
    if not file_path.exists() and not args.allow_missing_file:
        raise SystemExit(f"submission file does not exist: {file_path.relative_to(ROOT)}")

    display_file = display_path(file_path)
    rows, columns, file_hash = "-", "-", "-"
    if file_path.exists():
        rows, columns = csv_shape(file_path)
        file_hash = sha256_file(file_path)

    row = (
        f"| {version} | {date.today().isoformat()} | {args.experiment} | {display_file} | "
        f"{rows} | {columns} | {file_hash} | "
        f"{args.cv} | {args.public_lb} | {args.private_lb} | {args.notes} |\n"
    )

    ensure_table()

    with SUBMISSIONS_PATH.open("a") as fp:
        fp.write(row)

    print(f"Recorded submission {version} for {args.experiment}")


if __name__ == "__main__":
    main()
