from __future__ import annotations

import argparse
import math
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
    "submission.allow_extra_columns",
    "metadata.owner",
    "metadata.notes",
    "runtime.kaggle.enable_gpu",
    "runtime.kaggle.enable_internet",
    "runtime.kaggle.time_limit_hours",
]

STRICT_KEYS = [key for key in REQUIRED_SCHEMA_KEYS if key != "submission.target_columns"]
BOOLEAN_KEYS = {
    "competition.is_code_competition",
    "submission.allow_extra_columns",
    "runtime.kaggle.enable_gpu",
    "runtime.kaggle.enable_internet",
}
INTEGER_KEYS = {
    "defaults.seed",
    "defaults.n_folds",
}
POSITIVE_NUMBER_KEYS = {"runtime.kaggle.time_limit_hours"}
STRING_KEYS = set(STRICT_KEYS) - BOOLEAN_KEYS - INTEGER_KEYS - POSITIVE_NUMBER_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate project.yml.")
    parser.add_argument("--strict", action="store_true", help="Fail when TODO values remain.")
    parser.add_argument(
        "--expected-competition",
        help="Fail unless competition.slug matches the intended competition.",
    )
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
            value = get_nested(config, key)
            if is_todo(value):
                errors.append(f"strict value still TODO: {key}")

        for key in sorted(STRING_KEYS):
            value = get_nested(config, key)
            if not is_todo(value) and not isinstance(value, str):
                errors.append(f"{key} must be a string")

        for key in sorted(BOOLEAN_KEYS):
            value = get_nested(config, key)
            if not is_todo(value) and not isinstance(value, bool):
                errors.append(f"{key} must be true or false")

        for key in sorted(INTEGER_KEYS):
            value = get_nested(config, key)
            if not is_todo(value) and (not isinstance(value, int) or isinstance(value, bool)):
                errors.append(f"{key} must be an integer")
        n_folds = get_nested(config, "defaults.n_folds")
        if isinstance(n_folds, int) and not isinstance(n_folds, bool) and n_folds < 2:
            errors.append("defaults.n_folds must be at least 2")
        seed = get_nested(config, "defaults.seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed < 0:
            errors.append("defaults.seed must be at least 0")

        for key in sorted(POSITIVE_NUMBER_KEYS):
            value = get_nested(config, key)
            if not is_todo(value) and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                errors.append(f"{key} must be a positive number")

        enable_tpu = get_nested(config, "runtime.kaggle.enable_tpu")
        if enable_tpu is not None:
            if not isinstance(enable_tpu, bool):
                errors.append("runtime.kaggle.enable_tpu must be true or false")
            elif enable_tpu:
                errors.append("runtime.kaggle.enable_tpu=true is unsupported by this repository")

        target_columns = get_nested(config, "submission.target_columns")
        if not isinstance(target_columns, list) or not target_columns:
            errors.append("submission.target_columns must be a non-empty list")
        else:
            for index, column in enumerate(target_columns):
                if not isinstance(column, str) or is_todo(column):
                    errors.append(f"submission.target_columns[{index}] must be a non-TODO string")

        raw_dir = get_nested(config, "data.raw_dir")
        if isinstance(raw_dir, str) and not is_todo(raw_dir):
            raw_path = (ROOT / Path(raw_dir)).resolve()
            if not raw_path.exists():
                errors.append(f"configured path does not exist: data.raw_dir={raw_dir}")
            for key in ("data.train_dir", "data.test_dir", "submission.sample_file"):
                value = get_nested(config, key)
                if not isinstance(value, str) or is_todo(value):
                    continue
                configured_path = (ROOT / Path(value)).resolve()
                if not configured_path.is_relative_to(raw_path):
                    errors.append(f"{key} must be inside data.raw_dir for Kaggle runtime")
                if not configured_path.exists():
                    errors.append(f"configured path does not exist: {key}={value}")
                elif key == "submission.sample_file" and not configured_path.is_file():
                    errors.append(f"submission.sample_file must be a file: {value}")

    competition_slug = get_nested(config, "competition.slug")
    competition_url = get_nested(config, "competition.url")
    if args.expected_competition and competition_slug != args.expected_competition:
        errors.append(
            "competition.slug does not match --expected-competition: "
            f"{competition_slug!r} != {args.expected_competition!r}"
        )
    if not is_todo(competition_slug) and not is_todo(competition_url):
        expected_url = f"https://www.kaggle.com/competitions/{competition_slug}"
        if str(competition_url).rstrip("/") != expected_url:
            errors.append(
                f"competition.url does not match competition.slug: expected {expected_url}"
            )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    mode = "strict" if args.strict else "template"
    print(f"project.yml validation passed ({mode})")


if __name__ == "__main__":
    main()
