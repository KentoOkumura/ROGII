from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_submission as VALIDATOR  # noqa: E402


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n")


def validate(tmp_path: Path, submission_rows: list[str]):
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    write_csv(sample, ["id,tvt", "a,0", "b,0", "c,0"])
    write_csv(submission, submission_rows)
    return VALIDATOR.validate_submission_files(
        submission,
        sample,
        id_column="id",
        target_columns=["tvt"],
        allow_extra_columns=False,
    )


def test_valid_submission_checks_ids_and_finite_targets(tmp_path: Path) -> None:
    report = validate(tmp_path, ["id,tvt", "a,1.5", "b,2", "c,-3"])

    assert report.passed is True
    assert report.row_count == 3
    assert report.duplicate_id_count == 0
    assert report.missing_value_count == 0
    assert report.infinite_value_count == 0
    assert report.target_statistics["tvt"]["min"] == -3.0
    assert report.submission_sha256 is not None


def test_submission_rejects_id_order_and_duplicates(tmp_path: Path) -> None:
    report = validate(tmp_path, ["id,tvt", "b,1", "b,2", "c,3"])

    assert report.passed is False
    assert report.duplicate_id_count == 2
    assert any("duplicate IDs" in error for error in report.errors)
    assert any("order/content mismatch" in error for error in report.errors)


def test_submission_rejects_missing_infinite_and_non_numeric_targets(
    tmp_path: Path,
) -> None:
    report = validate(
        tmp_path,
        ["id,tvt", "a,", "b,1e309", "c,not-a-number"],
    )

    assert report.passed is False
    assert report.missing_value_count == 1
    assert report.infinite_value_count == 1
    assert any("non-numeric" in error for error in report.errors)
    assert any("infinite" in error for error in report.errors)


def test_submission_rejects_column_order_and_extra_columns(tmp_path: Path) -> None:
    report = validate(
        tmp_path,
        ["tvt,id,debug", "1,a,x", "2,b,x", "3,c,x"],
    )

    assert report.passed is False
    assert any("extra columns" in error for error in report.errors)
    assert any("column order/content mismatch" in error for error in report.errors)


def test_validation_evidence_is_recorded_in_metrics(tmp_path: Path, monkeypatch) -> None:
    report = validate(tmp_path, ["id,tvt", "a,1", "b,2", "c,3"])
    experiment = tmp_path / "experiments" / "exp999_check"
    experiment.mkdir(parents=True)
    (experiment / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "exp999_check",
                "evidence": {
                    "artifacts": {},
                    "submission_validation": {"fallback_rows": 7},
                },
            }
        )
    )
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)

    VALIDATOR.record_validation("exp999_check", report, regenerate_summary=False)

    metrics = json.loads((experiment / "metrics.json").read_text())
    validation = metrics["evidence"]["submission_validation"]
    assert validation["passed"] is True
    assert validation["duplicate_id_count"] == 0
    assert validation["infinite_value_count"] == 0
    assert validation["fallback_rows"] == 7
    assert metrics["evidence"]["artifacts"]["submission_sha"] == report.submission_sha256


def test_build_report_rejects_string_allow_extra_columns(monkeypatch) -> None:
    config = {
        "submission": {
            "sample_file": "sample_submission.csv",
            "id_column": "id",
            "target_columns": ["tvt"],
            "allow_extra_columns": "false",
        }
    }
    monkeypatch.setattr(VALIDATOR, "load_project_config", lambda: config)
    args = SimpleNamespace(sample=None, submission="submission.csv")

    with pytest.raises(ValueError, match="allow_extra_columns must be true or false"):
        VALIDATOR.build_report(args)
