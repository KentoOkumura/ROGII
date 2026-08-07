from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from config_utils import ROOT, get_nested, is_todo, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a submission CSV against sample submission."
    )
    parser.add_argument("--submission", required=True, help="Submission CSV path")
    parser.add_argument("--sample", default=None, help="Override sample submission path")
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


def main() -> None:
    args = parse_args()
    config = load_project_config()

    sample_value = args.sample or get_nested(config, "submission.sample_file")
    if is_todo(sample_value):
        raise SystemExit("submission.sample_file is TODO in project.yml")

    sample_path = resolve_path(str(sample_value))
    submission_path = resolve_path(args.submission)

    if not sample_path.exists():
        raise SystemExit(f"sample submission not found: {sample_path}")
    if not submission_path.exists():
        raise SystemExit(f"submission not found: {submission_path}")

    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)

    errors: list[str] = []
    if len(sample) != len(submission):
        errors.append(f"row count mismatch: sample={len(sample)}, submission={len(submission)}")

    allow_extra = bool(get_nested(config, "submission.allow_extra_columns"))
    missing_columns = [column for column in sample.columns if column not in submission.columns]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
    if not allow_extra:
        extra_columns = [column for column in submission.columns if column not in sample.columns]
        if extra_columns:
            errors.append(f"extra columns: {extra_columns}")

    id_column = get_nested(config, "submission.id_column")
    if not is_todo(id_column) and id_column in sample.columns and id_column in submission.columns:
        if sample[id_column].tolist() != submission[id_column].tolist():
            errors.append(f"id column order/content mismatch: {id_column}")

    if submission.isna().any().any():
        columns = submission.columns[submission.isna().any()].tolist()
        errors.append(f"missing values found in columns: {columns}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"submission validation passed: {display_path(submission_path)}")


if __name__ == "__main__":
    main()
