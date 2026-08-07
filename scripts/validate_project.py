from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import ROOT, get_nested, is_todo, load_project_config

REQUIRED_SCHEMA_KEYS = [
    "competition.name",
    "competition.platform",
    "competition.slug",
    "competition.url",
    "competition.is_code_competition",
    "paths.data_dir",
    "paths.experiments_dir",
    "paths.docs_dir",
    "paths.submissions_dir",
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

    if args.strict:
        for key in STRICT_KEYS:
            if is_todo(get_nested(config, key)):
                errors.append(f"strict value still TODO: {key}")

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
