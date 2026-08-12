from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import ROOT, get_nested, is_todo, load_project_config
from validate_experiment import NEW_RESULT_HEADINGS, markdown_headings

REQUIRED_SCHEMA_KEYS = [
    "competition.name",
    "competition.platform",
    "competition.slug",
    "competition.url",
    "competition.is_code_competition",
    "paths.data_dir",
    "paths.experiments_dir",
    "paths.docs_dir",
    "paths.submissions_file",
    "data.raw_dir",
    "data.train_dir",
    "data.test_dir",
    "data.processed_dir",
    "data.target_column",
    "data.group_column",
    "data.score_rows",
    "defaults.seed",
    "defaults.metric",
    "defaults.primary_validation",
    "defaults.n_folds",
    "submission.sample_file",
    "submission.output_file",
    "submission.id_column",
    "submission.target_columns",
    "runtime.kaggle.enable_gpu",
    "runtime.kaggle.enable_internet",
    "runtime.kaggle.time_limit_hours",
]

STRICT_KEYS = [
    "competition.slug",
    "competition.url",
    "defaults.metric",
    "defaults.primary_validation",
    "submission.id_column",
    "submission.sample_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project.yml.")
    parser.add_argument("--strict", action="store_true", help="Fail when TODO values remain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config()

    errors: list[str] = []
    for key in REQUIRED_SCHEMA_KEYS:
        if get_nested(config, key) is None:
            errors.append(f"missing required key: {key}")

    result_template_path = ROOT / "templates" / "experiment" / "result.md"
    if not result_template_path.exists():
        errors.append("missing experiment result template: templates/experiment/result.md")
    else:
        missing_result_headings = NEW_RESULT_HEADINGS - markdown_headings(
            result_template_path.read_text()
        )
        if missing_result_headings:
            errors.append(
                "experiment result template missing current sections: "
                + ", ".join(sorted(missing_result_headings))
            )

    if args.strict:
        for key in STRICT_KEYS:
            if is_todo(get_nested(config, key)):
                errors.append(f"strict value still TODO: {key}")

        target_columns = get_nested(config, "submission.target_columns")
        if not isinstance(target_columns, list) or not target_columns:
            errors.append("submission.target_columns must be a non-empty list")
        else:
            for index, column in enumerate(target_columns):
                if not isinstance(column, str) or is_todo(column):
                    errors.append(f"submission.target_columns[{index}] must be a non-TODO string")

        sample_file = get_nested(config, "submission.sample_file")
        if sample_file and not is_todo(sample_file):
            sample_path = ROOT / Path(str(sample_file))
            if not sample_path.exists():
                errors.append(f"sample submission does not exist: {sample_path.relative_to(ROOT)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    mode = "strict" if args.strict else "template"
    print(f"project.yml validation passed ({mode})")


if __name__ == "__main__":
    main()
