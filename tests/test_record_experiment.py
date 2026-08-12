from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from record_experiment import apply_evidence_assignments, parse_evidence_value  # noqa: E402


def test_parse_evidence_value_preserves_types() -> None:
    assert parse_evidence_value("true") is True
    assert parse_evidence_value("3600") == 3600
    assert parse_evidence_value('["owner/source"]') == ["owner/source"]
    assert parse_evidence_value("owner/slug") == "owner/slug"


def test_apply_evidence_assignments_updates_nested_schema() -> None:
    metrics = {
        "evidence": {
            "kaggle": {"kernel_id": None, "kernel_version": None},
            "submission_validation": {"passed": None},
        }
    }

    apply_evidence_assignments(
        metrics,
        [
            "kaggle.kernel_id=owner/slug",
            "kaggle.kernel_version=2",
            "submission_validation.passed=true",
        ],
    )

    assert metrics["evidence"]["kaggle"] == {
        "kernel_id": "owner/slug",
        "kernel_version": 2,
    }
    assert metrics["evidence"]["submission_validation"]["passed"] is True


def test_apply_evidence_assignments_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        apply_evidence_assignments({}, ["kaggle.kernel_id"])

    with pytest.raises(ValueError, match="is not an object"):
        apply_evidence_assignments(
            {"evidence": {"kaggle": "invalid"}},
            ["kaggle.kernel_id=owner/slug"],
        )
