#!/usr/bin/env python3
"""Local preflight checks for Kaggle submission files and folders."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path


BAD_VALUE_STRINGS = {"", "nan", "none", "null", "inf", "-inf", "infinity", "-infinity"}


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
        for title, items in (("FAIL", self.failures), ("WARN", self.warnings), ("PASS", self.passes)):
            print(f"## {title}")
            if not items:
                print("- None")
            for item in items:
                print(f"- {item}")
            print()


def read_csv_summary(handle: io.TextIOBase, reporter: Reporter, label: str, sample: Path | None = None) -> tuple[list[str], int]:
    reader = csv.reader(handle)
    try:
        header = next(reader)
    except StopIteration:
        reporter.fail(f"{label}: CSV is empty")
        return [], 0
    if not header:
        reporter.fail(f"{label}: CSV header is empty")
    if len(set(header)) != len(header):
        reporter.fail(f"{label}: duplicate column names in header")

    id_values: set[str] = set()
    duplicate_ids = 0
    bad_values = 0
    row_count = 0
    first_col = 0
    for row in reader:
        row_count += 1
        if len(row) != len(header):
            reporter.fail(f"{label}: row {row_count + 1} has {len(row)} columns, expected {len(header)}")
            if row_count > 20:
                break
        if row:
            key = row[first_col]
            if key in id_values:
                duplicate_ids += 1
            id_values.add(key)
        for value in row:
            normalized = value.strip().lower()
            if normalized in BAD_VALUE_STRINGS:
                bad_values += 1
            else:
                try:
                    numeric = float(normalized)
                except ValueError:
                    continue
                if not math.isfinite(numeric):
                    bad_values += 1

    if duplicate_ids:
        reporter.fail(f"{label}: duplicate IDs in first column ({duplicate_ids})")
    else:
        reporter.ok(f"{label}: no duplicate IDs detected in first column")
    if bad_values:
        reporter.warn(f"{label}: found {bad_values} empty/NaN/Inf-like values")
    else:
        reporter.ok(f"{label}: no empty/NaN/Inf-like values detected")
    reporter.ok(f"{label}: rows={row_count}, columns={len(header)}")

    if sample:
        compare_sample(header, row_count, sample, reporter, label)
    return header, row_count


def compare_sample(header: list[str], row_count: int, sample: Path, reporter: Reporter, label: str) -> None:
    if not sample.exists():
        reporter.warn(f"sample not found: {sample}")
        return
    with sample.open(newline="", encoding="utf-8") as handle:
        sample_reader = csv.reader(handle)
        try:
            sample_header = next(sample_reader)
        except StopIteration:
            reporter.fail(f"sample is empty: {sample}")
            return
        sample_rows = sum(1 for _ in sample_reader)
    if header != sample_header:
        reporter.fail(f"{label}: header differs from sample_submission.csv: {header} != {sample_header}")
    else:
        reporter.ok(f"{label}: header matches sample_submission.csv")
    if row_count != sample_rows:
        reporter.fail(f"{label}: row count differs from sample_submission.csv: {row_count} != {sample_rows}")
    else:
        reporter.ok(f"{label}: row count matches sample_submission.csv")


def check_csv(path: Path, reporter: Reporter, sample: Path | None = None) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        read_csv_summary(handle, reporter, str(path), sample)


def check_zip(path: Path, reporter: Reporter, sample: Path | None = None) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if not names:
            reporter.fail(f"{path}: zip archive is empty")
            return
        hidden = [name for name in names if name.startswith("__MACOSX/") or Path(name).name.startswith(".")]
        if hidden:
            reporter.warn(f"{path}: hidden/system files in zip: {hidden[:5]}")
        csv_names = [name for name in names if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            reporter.warn(f"{path}: expected exactly one CSV in zip, found {len(csv_names)}")
        else:
            with archive.open(csv_names[0]) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                read_csv_summary(text, reporter, f"{path}:{csv_names[0]}", sample)
        reporter.ok(f"{path}: zip members={len(names)}")


def check_metadata(path: Path, reporter: Reporter) -> None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
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
    if "submission.csv" in names:
        check_csv(path / "submission.csv", reporter, sample)
    else:
        csvs = sorted(path.glob("*.csv"))
        if csvs:
            reporter.warn(f"{path}: no submission.csv; checking {csvs[0].name}")
            check_csv(csvs[0], reporter, sample)
    zips = sorted(path.glob("*.zip"))
    for zip_path in zips[:3]:
        check_zip(zip_path, reporter, sample)
    metadata = path / "kernel-metadata.json"
    if metadata.exists():
        check_metadata(metadata, reporter)
    if {"notebook.py", "kernel-metadata.json"} <= names or {"code.ipynb", "kernel-metadata.json"} <= names:
        reporter.ok(f"{path}: looks like a Kaggle code/notebook submission folder")
    if "Dockerfile" in names:
        reporter.warn(f"{path}: Docker submission detected; run the platform-specific local test script if available")
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
