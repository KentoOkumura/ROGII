from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from config_utils import ROOT, get_nested, is_todo, load_project_config
from update_experiment_summary import collect_records, render_auto_block, update_summary

MISSING_VALUE_STRINGS = {"", "na", "n/a", "nan", "none", "null"}
INFINITE_VALUE_STRINGS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}


@dataclass
class SubmissionValidationReport:
    passed: bool
    submission: str
    sample: str
    row_count: int | None
    sample_row_count: int | None
    id_column: str | None
    target_columns: list[str]
    duplicate_id_count: int
    missing_value_count: int
    infinite_value_count: int
    submission_sha256: str | None
    target_statistics: dict[str, dict[str, float]]
    errors: list[str]

    def evidence(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "passed": self.passed,
            "checked_at": datetime.now(UTC).isoformat(),
            "row_count": self.row_count,
            "id_column": self.id_column,
            "target_columns": self.target_columns,
            "duplicate_id_count": self.duplicate_id_count,
            "missing_value_count": self.missing_value_count,
            "infinite_value_count": self.infinite_value_count,
            "target_statistics": self.target_statistics,
            "errors": self.errors,
        }
        if len(self.target_columns) == 1:
            statistics = self.target_statistics.get(self.target_columns[0], {})
            value.update(
                {
                    "prediction_min": statistics.get("min"),
                    "prediction_max": statistics.get("max"),
                    "prediction_mean": statistics.get("mean"),
                    "prediction_std": statistics.get("std"),
                }
            )
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission CSV against sample submission."
    )
    parser.add_argument("--submission", required=True, help="Submission CSV path")
    parser.add_argument("--sample", default=None, help="Override sample submission path")
    parser.add_argument(
        "--experiment",
        default="",
        help="Record the validation result in this experiment's metrics.json when set.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. JSON is intended for other repository scripts.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_csv_as_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalized_strings(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.astype(str).str.strip().str.lower())


def validate_submission_files(
    submission_path: Path,
    sample_path: Path,
    *,
    id_column: str | None,
    target_columns: list[str],
    allow_extra_columns: bool,
) -> SubmissionValidationReport:
    errors: list[str] = []
    row_count: int | None = None
    sample_row_count: int | None = None
    duplicate_id_count = 0
    missing_value_count = 0
    infinite_value_count = 0
    submission_sha256: str | None = None
    target_statistics: dict[str, dict[str, float]] = {}

    if not sample_path.is_file():
        errors.append(f"sample submission not found: {display_path(sample_path)}")
    if not submission_path.is_file():
        errors.append(f"submission not found: {display_path(submission_path)}")
    if errors:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            id_column,
            target_columns,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    submission_sha256 = file_sha256(submission_path)
    sample_header = read_header(sample_path)
    submission_header = read_header(submission_path)
    if not sample_header:
        errors.append("sample submission CSV is empty")
    if not submission_header:
        errors.append("submission CSV is empty")
    if len(sample_header) != len(set(sample_header)):
        errors.append("sample submission contains duplicate column names")
    if len(submission_header) != len(set(submission_header)):
        errors.append("submission contains duplicate column names")

    try:
        sample = read_csv_as_text(sample_path)
    except Exception as exc:  # pandas exposes parser/encoding errors through several types
        errors.append(f"could not read sample submission: {exc}")
        sample = None
    try:
        submission = read_csv_as_text(submission_path)
    except Exception as exc:  # pandas exposes parser/encoding errors through several types
        errors.append(f"could not read submission: {exc}")
        submission = None

    if sample is not None:
        sample_row_count = len(sample)
    if submission is not None:
        row_count = len(submission)
    if sample is None or submission is None:
        return SubmissionValidationReport(
            False,
            display_path(submission_path),
            display_path(sample_path),
            row_count,
            sample_row_count,
            id_column,
            target_columns,
            duplicate_id_count,
            missing_value_count,
            infinite_value_count,
            submission_sha256,
            target_statistics,
            errors,
        )

    if row_count != sample_row_count:
        errors.append(f"row count mismatch: sample={sample_row_count}, submission={row_count}")

    sample_columns = list(sample.columns)
    submission_columns = list(submission.columns)
    missing_columns = [column for column in sample_columns if column not in submission_columns]
    extra_columns = [column for column in submission_columns if column not in sample_columns]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if extra_columns and not allow_extra_columns:
        errors.append(f"extra columns: {extra_columns}")
    if not allow_extra_columns and submission_columns != sample_columns:
        errors.append(
            "column order/content mismatch: "
            f"sample={sample_columns}, submission={submission_columns}"
        )
    elif allow_extra_columns:
        projected_columns = [column for column in submission_columns if column in sample_columns]
        if projected_columns != sample_columns:
            errors.append(
                "configured sample columns are not in sample order: "
                f"sample={sample_columns}, submission={submission_columns}"
            )

    if not id_column:
        errors.append("submission.id_column is not configured")
    elif id_column not in sample.columns or id_column not in submission.columns:
        errors.append(f"id column is missing from sample or submission: {id_column}")
    else:
        duplicate_id_count = int(submission[id_column].duplicated(keep=False).sum())
        if duplicate_id_count:
            errors.append(
                f"duplicate IDs found in {id_column}: rows_in_duplicate_groups={duplicate_id_count}"
            )
        sample_duplicate_count = int(sample[id_column].duplicated(keep=False).sum())
        if sample_duplicate_count:
            errors.append(
                f"sample submission contains duplicate IDs in {id_column}: "
                f"rows_in_duplicate_groups={sample_duplicate_count}"
            )
        if sample[id_column].tolist() != submission[id_column].tolist():
            errors.append(f"id column order/content mismatch: {id_column}")

    normalized = normalized_strings(submission)
    missing_mask = normalized.isin(MISSING_VALUE_STRINGS)
    missing_value_count = int(missing_mask.to_numpy().sum())
    if missing_value_count:
        columns = submission.columns[missing_mask.any()].tolist()
        errors.append(f"missing values found: count={missing_value_count}, columns={columns}")

    if not target_columns:
        errors.append("submission.target_columns is not configured")
    for target in target_columns:
        if target not in submission.columns:
            if target not in missing_columns:
                errors.append(f"configured target column is missing: {target}")
            continue
        values = normalized[target]
        numeric = pd.to_numeric(submission[target].str.strip(), errors="coerce")
        infinite_mask = values.isin(INFINITE_VALUE_STRINGS) | numeric.isin(
            [float("inf"), float("-inf")]
        )
        infinite_value_count += int(infinite_mask.sum())
        non_numeric_mask = numeric.isna() & ~values.isin(MISSING_VALUE_STRINGS)
        non_numeric_count = int(non_numeric_mask.sum())
        if non_numeric_count:
            errors.append(
                f"non-numeric values found in target column {target}: count={non_numeric_count}"
            )
        finite_numeric = numeric[~numeric.isna() & ~infinite_mask]
        if not finite_numeric.empty:
            target_statistics[target] = {
                "min": float(finite_numeric.min()),
                "max": float(finite_numeric.max()),
                "mean": float(finite_numeric.mean()),
                "std": float(finite_numeric.std(ddof=0)),
            }
    if infinite_value_count:
        errors.append(f"infinite target values found: count={infinite_value_count}")

    return SubmissionValidationReport(
        not errors,
        display_path(submission_path),
        display_path(sample_path),
        row_count,
        sample_row_count,
        id_column,
        target_columns,
        duplicate_id_count,
        missing_value_count,
        infinite_value_count,
        submission_sha256,
        target_statistics,
        errors,
    )


def regenerate_experiment_summary() -> None:
    summary_path = ROOT / "experiment_summary.md"
    existing = summary_path.read_text() if summary_path.exists() else ""
    summary_path.write_text(update_summary(existing, render_auto_block(collect_records())))


def record_validation(
    experiment: str,
    report: SubmissionValidationReport,
    *,
    regenerate_summary: bool = True,
) -> None:
    if Path(experiment).name != experiment:
        raise ValueError(f"invalid experiment name: {experiment!r}")
    experiment_dir = ROOT / "experiments" / experiment
    if not experiment_dir.is_dir():
        raise ValueError(f"experiment does not exist: experiments/{experiment}")
    metrics_path = experiment_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metrics JSON: {metrics_path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(metrics, dict):
        raise ValueError(f"{metrics_path.relative_to(ROOT)} must contain a JSON object")

    evidence = metrics.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("metrics.json evidence must be a JSON object")
    validation = evidence.setdefault("submission_validation", {})
    if not isinstance(validation, dict):
        raise ValueError("metrics.json evidence.submission_validation must be a JSON object")
    validation.update(report.evidence())
    artifacts = evidence.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("metrics.json evidence.artifacts must be a JSON object")
    artifacts["submission_sha"] = report.submission_sha256
    metrics["updated_at"] = datetime.now(UTC).isoformat()
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    if regenerate_summary:
        regenerate_experiment_summary()


def build_report(args: argparse.Namespace) -> SubmissionValidationReport:
    config = load_project_config()
    sample_value = args.sample or get_nested(config, "submission.sample_file")
    if is_todo(sample_value):
        raise ValueError("submission.sample_file is TODO in project.yml")
    id_value = get_nested(config, "submission.id_column")
    id_column = None if is_todo(id_value) else str(id_value)
    target_value = get_nested(config, "submission.target_columns")
    target_columns = (
        [str(value) for value in target_value] if isinstance(target_value, list) else []
    )
    allow_extra_columns = get_nested(config, "submission.allow_extra_columns")
    if not isinstance(allow_extra_columns, bool):
        raise ValueError("submission.allow_extra_columns must be true or false in project.yml")
    return validate_submission_files(
        resolve_path(args.submission),
        resolve_path(str(sample_value)),
        id_column=id_column,
        target_columns=target_columns,
        allow_extra_columns=allow_extra_columns,
    )


def print_report(report: SubmissionValidationReport, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
        return
    if report.passed:
        print(f"submission validation passed: {report.submission}")
        print(
            f"rows={report.row_count}, duplicate_ids={report.duplicate_id_count}, "
            f"missing={report.missing_value_count}, infinite={report.infinite_value_count}"
        )
    else:
        for error in report.errors:
            print(f"ERROR: {error}")


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except (OSError, ValueError) as exc:
        report = SubmissionValidationReport(
            False,
            args.submission,
            args.sample or "",
            None,
            None,
            None,
            [],
            0,
            0,
            0,
            None,
            {},
            [str(exc)],
        )

    if args.experiment.strip():
        try:
            record_validation(args.experiment.strip(), report)
        except (OSError, ValueError) as exc:
            report.errors.append(f"could not record validation evidence: {exc}")
            report.passed = False

    print_report(report, args.format)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
