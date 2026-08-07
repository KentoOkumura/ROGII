from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from config_utils import (
    ROOT,
    deep_merge,
    get_nested,
    is_todo,
    load_project_config,
    project_experiment_defaults,
)

REQUIRED_FILES = [
    "README.md",
    "config.yaml",
    "settings.py",
    "SESSION_NOTES.md",
    "result.md",
    "metrics.json",
]
REQUIRED_DIRS = ["artifacts", "features", "variants"]
STRICT_CONFIG_KEYS = [
    "experiment.name",
    "experiment.description",
    "lineage.diff_summary",
    "model.name",
]
STRICT_EFFECTIVE_CONFIG_KEYS = [
    "validation.strategy",
    "validation.metric",
    "validation.seed",
    "validation.n_folds",
    "data.target_column",
    "data.id_column",
    "data.sample_submission",
    "data.submission_target_column",
]
PROJECT_BACKED_KEYS = [
    "validation.metric",
    "validation.seed",
    "validation.n_folds",
    "data.id_column",
    "data.sample_submission",
    "data.submission_target_column",
]
ALLOWED_ROUTES = {"ensemble", "ml_model", "pf_beam"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one experiment directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. expXXX_model")
    parser.add_argument(
        "--allow-todo",
        action="store_true",
        help="Only validate structure; allow TODO values in config.yaml",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a YAML mapping")
    return value


def main() -> None:
    args = parse_args()
    experiment_dir = ROOT / "experiments" / args.experiment
    errors: list[str] = []

    if not experiment_dir.exists():
        raise SystemExit(f"experiment does not exist: {experiment_dir.relative_to(ROOT)}")

    for filename in REQUIRED_FILES:
        if not (experiment_dir / filename).exists():
            errors.append(f"missing required file: {filename}")

    for filename in (
        f"{args.experiment}_train.ipynb",
        f"{args.experiment}_inference.ipynb",
    ):
        if not (experiment_dir / filename).exists():
            errors.append(f"missing required notebook: {filename}")

    for dirname in REQUIRED_DIRS:
        if not (experiment_dir / dirname).is_dir():
            errors.append(f"missing required directory: {dirname}")

    readme_path = experiment_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text()
        required_heading_groups = (
            ("## 状態", "## Status"),
            ("## 仮説", "## Hypothesis"),
            ("## 検証方針", "## Validation Strategy"),
            ("## 所見", "## Findings"),
        )
        for headings in required_heading_groups:
            if not any(heading in readme for heading in headings):
                errors.append(f"README.md missing section: {' or '.join(headings)}")

    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        config = read_yaml(config_path)
        project_defaults = project_experiment_defaults(load_project_config())
        effective_config = deep_merge(project_defaults, config)
        if not args.allow_todo:
            for key in STRICT_CONFIG_KEYS:
                value = get_nested(config, key)
                if is_todo(value):
                    errors.append(f"config value still TODO: {key}")

            for key in STRICT_EFFECTIVE_CONFIG_KEYS:
                value = get_nested(effective_config, key)
                if is_todo(value):
                    errors.append(f"effective config value still TODO: {key}")

            project_default_overrides = get_nested(config, "overrides.project_defaults")
            if not isinstance(project_default_overrides, list):
                project_default_overrides = []

            for key in PROJECT_BACKED_KEYS:
                raw_value = get_nested(config, key)
                default_value = get_nested(project_defaults, key)
                if raw_value is None or is_todo(raw_value) or default_value is None:
                    continue
                if raw_value != default_value and key not in project_default_overrides:
                    errors.append(
                        f"config value differs from project.yml without override: {key}"
                    )

        route = get_nested(config, "experiment.route")
        if route is not None:
            if is_todo(route) and not args.allow_todo:
                errors.append("config value still TODO: experiment.route")
            elif not is_todo(route) and route not in ALLOWED_ROUTES:
                allowed_routes = ", ".join(sorted(ALLOWED_ROUTES))
                errors.append(f"invalid experiment.route: {route}. allowed: {allowed_routes}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    mode = "structure" if args.allow_todo else "strict"
    print(f"experiment validation passed ({mode}): {args.experiment}")


if __name__ == "__main__":
    main()
