from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
OVERLAP_ASSERTION_MESSAGE = "HIDDEN_TRAIN_TEST_WELL_ID_OVERLAP_DETECTED"


def well_id_from_path(path: Path, suffix: str = HORIZONTAL_SUFFIX) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected horizontal well filename: {name}")
    return name[: -len(suffix)]


def collect_well_ids(directory: Path, suffix: str = HORIZONTAL_SUFFIX) -> list[str]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"no horizontal well files found in {directory}")
    return sorted({well_id_from_path(path, suffix=suffix) for path in files})


def classify_test_set(test_wells: set[str], expected_public_test_wells: set[str]) -> str:
    if test_wells == expected_public_test_wells:
        return "public_sample"
    return "hidden_or_private_test"


def build_public_summary(
    *,
    train_wells: set[str],
    test_wells: set[str],
    expected_public_test_wells: set[str],
) -> dict[str, Any]:
    overlap = sorted(train_wells & test_wells)
    expected_overlap = sorted(expected_public_test_wells)
    if sorted(test_wells) != expected_overlap:
        raise AssertionError("PUBLIC_SAMPLE_TEST_WELLS_MISMATCH")
    if overlap != expected_overlap:
        raise AssertionError("PUBLIC_SAMPLE_OVERLAP_MISMATCH")
    return {
        "phase": "public_sample",
        "status": "public_sample_overlap_allowed",
        "train_well_count": len(train_wells),
        "test_well_count": len(test_wells),
        "overlap_well_count": len(overlap),
        "overlap_wells": overlap,
        "assertion": "skipped_for_known_public_sample",
    }


def run_assert_probe(
    *,
    train_dir: Path,
    test_dir: Path,
    expected_public_test_wells: list[str],
    suffix: str = HORIZONTAL_SUFFIX,
) -> dict[str, Any]:
    train_wells = set(collect_well_ids(train_dir, suffix=suffix))
    test_wells = set(collect_well_ids(test_dir, suffix=suffix))
    expected_public = set(expected_public_test_wells)

    phase = classify_test_set(test_wells, expected_public)
    if phase == "public_sample":
        return build_public_summary(
            train_wells=train_wells,
            test_wells=test_wells,
            expected_public_test_wells=expected_public,
        )

    overlap = train_wells & test_wells
    if overlap:
        raise AssertionError(OVERLAP_ASSERTION_MESSAGE)

    return {
        "phase": "hidden_or_private_test",
        "status": "no_train_test_well_id_overlap_detected",
        "assertion": "passed_no_overlap",
        "hidden_test_details_suppressed": True,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_sample_submission(sample_submission_path: Path, submission_path: Path) -> None:
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"sample submission not found: {sample_submission_path}")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample_submission_path, submission_path)
