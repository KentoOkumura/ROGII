#!/usr/bin/env python3
"""Check submission bundles while delegating CSV validation to the repository validator."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_CSV_VALIDATOR = REPO_ROOT / "scripts" / "validate_submission.py"


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def print(self) -> None:
        print("# Kaggle Submission Check\n")
        for title, items in (
            ("FAIL", self.failures),
            ("WARN", self.warnings),
            ("PASS", self.passes),
        ):
            print(f"## {title}")
            if not items:
                print("- None")
            for item in items:
                print(f"- {item}")
            print()


def check_csv(path: Path, reporter: Reporter, sample: Path | None = None) -> None:
    command = [
        sys.executable,
        str(CANONICAL_CSV_VALIDATOR),
        "--submission",
        str(path.resolve()),
        "--format",
        "json",
    ]
    if sample is not None:
        command.extend(["--sample", str(sample.resolve())])
    process = subprocess.run(command, text=True, capture_output=True)
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        detail = process.stderr.strip() or process.stdout.strip() or "no output"
        reporter.fail(f"{path}: canonical CSV validator failed: {detail}")
        return

    if result.get("passed") and process.returncode == 0:
        reporter.ok(
            f"{path}: CSV validation passed "
            f"(rows={result.get('row_count')}, duplicate IDs=0, missing=0, infinite=0)"
        )
        return
    errors = result.get("errors") or [f"validator exited with {process.returncode}"]
    for error in errors:
        reporter.fail(f"{path}: {error}")


def check_zip(path: Path, reporter: Reporter, sample: Path | None = None) -> None:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        reporter.fail(f"{path}: invalid zip archive: {exc}")
        return
    with archive:
        names = archive.namelist()
        if not names:
            reporter.fail(f"{path}: zip archive is empty")
            return
        hidden = [
            name
            for name in names
            if name.startswith("__MACOSX/") or Path(name).name.startswith(".")
        ]
        if hidden:
            reporter.warn(f"{path}: hidden/system files in zip: {hidden[:5]}")
        csv_names = [name for name in names if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            reporter.fail(f"{path}: expected exactly one CSV in zip, found {len(csv_names)}")
        else:
            with tempfile.TemporaryDirectory(prefix="kaggle-submit-check-") as temp_dir:
                extracted = Path(temp_dir) / Path(csv_names[0]).name
                with archive.open(csv_names[0]) as source, extracted.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                check_csv(extracted, reporter, sample)
        reporter.ok(f"{path}: zip members={len(names)}")


def check_metadata(path: Path, reporter: Reporter) -> None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reporter.fail(f"{path}: invalid JSON: {exc}")
        return
    for key in ("id", "title"):
        if not meta.get(key):
            reporter.warn(f"{path}: missing `{key}`")
    if meta.get("enable_internet") is True:
        reporter.warn(f"{path}: enable_internet=true; confirm competition rules allow internet")
    if "competition_sources" not in meta and "dataset_sources" not in meta:
        reporter.warn(f"{path}: no competition_sources or dataset_sources found")
    reporter.ok(f"{path}: kernel metadata parsed")


def check_dir(path: Path, reporter: Reporter, sample: Path | None = None) -> None:
    files = [item for item in path.iterdir() if item.is_file()]
    names = {item.name for item in files}
    csvs = sorted(path.glob("*.csv"))
    zips = sorted(path.glob("*.zip"))
    metadata = path / "kernel-metadata.json"
    if "submission.csv" in names:
        check_csv(path / "submission.csv", reporter, sample)
    elif csvs:
        reporter.warn(f"{path}: no submission.csv; checking {csvs[0].name}")
        check_csv(csvs[0], reporter, sample)
    for zip_path in zips[:3]:
        check_zip(zip_path, reporter, sample)
    if metadata.exists():
        check_metadata(metadata, reporter)
    if {"notebook.py", "kernel-metadata.json"} <= names or {
        "code.ipynb",
        "kernel-metadata.json",
    } <= names:
        reporter.ok(f"{path}: looks like a Kaggle code/notebook submission folder")
    if "Dockerfile" in names:
        reporter.warn(
            f"{path}: Docker submission detected; run the platform-specific "
            "local test script if available"
        )
    if not csvs and not zips and not metadata.exists():
        reporter.warn(f"{path}: no CSV, zip, or kernel-metadata.json found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--sample", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    sample = Path(args.sample) if args.sample else None
    reporter = Reporter()

    if not path.exists():
        reporter.fail(f"target does not exist: {path}")
    elif path.is_dir():
        check_dir(path, reporter, sample)
    elif path.suffix.lower() == ".csv":
        check_csv(path, reporter, sample)
    elif path.suffix.lower() == ".zip":
        check_zip(path, reporter, sample)
    elif path.name == "kernel-metadata.json":
        check_metadata(path, reporter)
    else:
        reporter.warn(f"unknown submission target type: {path}")

    reporter.print()
    return 1 if reporter.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
