from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml
from check_strategy_docs import registered_hypothesis_ids
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
REQUIRED_DIRS = ["artifacts"]
STRICT_CONFIG_KEYS = [
    "experiment.name",
    "experiment.description",
    "lineage.diff_summary",
    "model.name",
]
OPTIONAL_STRICT_CONFIG_KEYS = [
    "lineage.hypothesis_id",
    "lineage.backlog_candidate",
]
HYPOTHESIS_ID_RE = re.compile(r"HYP-\d{8}-\d{2}")
BACKLOG_CANDIDATE_RE = re.compile(r"[a-z0-9_]+")
STRICT_EFFECTIVE_CONFIG_KEYS = [
    "validation.strategy",
    "validation.metric",
    "validation.seed",
    "validation.n_folds",
    "reproducibility.seed",
    "data.target_column",
    "data.id_column",
    "data.sample_submission",
    "data.submission_target_column",
]
PROJECT_BACKED_KEYS = [
    "validation.metric",
    "validation.seed",
    "validation.n_folds",
    "reproducibility.seed",
    "data.id_column",
    "data.sample_submission",
    "data.submission_target_column",
]
ALLOWED_ROUTES = {"ensemble", "ml_model", "pf_beam"}
NEW_README_REQUIRED_HEADINGS = {"## 正の記録", "## 実行入口"}
NEW_README_OVERVIEW_HEADINGS = {"## 概要", "## 状態概要"}
LEGACY_README_HEADINGS = {
    "## 状態",
    "## 仮説",
    "## 検証方針",
    "## 所見",
}
NEW_RESULT_HEADINGS = {
    "## 仮説",
    "## 実行証拠",
    "## 解釈",
    "## ユーザー判断",
}


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


def markdown_headings(text: str) -> set[str]:
    return set(re.findall(r"^#{1,6}\s+.+$", text, flags=re.MULTILINE))


def first_h1(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def experiment_id(value: str) -> str:
    match = re.match(r"exp\d+", value.lower())
    return match.group(0) if match else value


def validate_required_directories(
    experiment_dir: Path,
    *,
    legacy_layout: bool,
    errors: list[str],
) -> None:
    missing_dirs = [dirname for dirname in REQUIRED_DIRS if not (experiment_dir / dirname).is_dir()]
    if legacy_layout and missing_dirs:
        print(
            "WARNING: legacy experiment is missing generated directories; they will be "
            "created when the experiment is next migrated: " + ", ".join(missing_dirs)
        )
        return
    for dirname in missing_dirs:
        errors.append(f"missing required directory: {dirname}")


def validate_lineage(
    config: dict[str, Any],
    errors: list[str],
    *,
    registered_ids: set[str] | None = None,
) -> None:
    hypothesis_id = get_nested(config, "lineage.hypothesis_id")
    backlog_candidate = get_nested(config, "lineage.backlog_candidate")

    if (
        hypothesis_id is not None
        and not is_todo(hypothesis_id)
        and hypothesis_id != "N/A"
        and (not isinstance(hypothesis_id, str) or not HYPOTHESIS_ID_RE.fullmatch(hypothesis_id))
    ):
        errors.append("invalid lineage.hypothesis_id: expected HYP-YYYYMMDD-NN or N/A")

    if (
        backlog_candidate is not None
        and not is_todo(backlog_candidate)
        and backlog_candidate != "N/A"
        and (
            not isinstance(backlog_candidate, str)
            or not BACKLOG_CANDIDATE_RE.fullmatch(backlog_candidate)
        )
    ):
        errors.append(
            "invalid lineage.backlog_candidate: expected a docs/backlog candidate name or N/A"
        )

    if (
        hypothesis_id is not None
        and backlog_candidate is not None
        and not is_todo(hypothesis_id)
        and not is_todo(backlog_candidate)
        and (hypothesis_id == "N/A") != (backlog_candidate == "N/A")
    ):
        errors.append(
            "lineage.hypothesis_id and lineage.backlog_candidate must both be tracked values "
            "or both be N/A"
        )

    if (
        registered_ids is not None
        and isinstance(hypothesis_id, str)
        and HYPOTHESIS_ID_RE.fullmatch(hypothesis_id)
        and hypothesis_id not in registered_ids
    ):
        errors.append(
            "unregistered lineage.hypothesis_id: register it in KAGGLE_DIRECTION.md or a "
            "final/superseded survey before validating the experiment"
        )


def validate_config_contract(
    config: dict[str, Any],
    *,
    allow_todo: bool,
    project_defaults: dict[str, Any],
    registered_ids: set[str],
    errors: list[str],
) -> None:
    """Validate config values while always checking completed lineage values."""
    effective_config = deep_merge(project_defaults, config)
    validate_lineage(config, errors, registered_ids=registered_ids)

    if not allow_todo:
        for key in STRICT_CONFIG_KEYS:
            value = get_nested(config, key)
            if is_todo(value):
                errors.append(f"config value still TODO: {key}")

        for key in OPTIONAL_STRICT_CONFIG_KEYS:
            value = get_nested(config, key)
            if value is not None and is_todo(value):
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
                errors.append(f"config value differs from project.yml without override: {key}")

    route = get_nested(config, "experiment.route")
    if route is not None:
        if is_todo(route) and not allow_todo:
            errors.append("config value still TODO: experiment.route")
        elif not is_todo(route) and route not in ALLOWED_ROUTES:
            allowed_routes = ", ".join(sorted(ALLOWED_ROUTES))
            errors.append(f"invalid experiment.route: {route}. allowed: {allowed_routes}")


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
        notebook_path = experiment_dir / filename
        if not notebook_path.exists():
            errors.append(f"missing required notebook: {filename}")
            continue
        try:
            notebook = json.loads(notebook_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"notebook is invalid JSON: {filename}: {exc}")
        else:
            if not isinstance(notebook, dict):
                errors.append(f"notebook must contain a JSON object: {filename}")

    readme_path = experiment_dir / "README.md"
    new_layout = False
    legacy_layout = False
    if readme_path.exists():
        readme = readme_path.read_text()
        headings = markdown_headings(readme)
        new_layout = NEW_README_REQUIRED_HEADINGS <= headings and bool(
            NEW_README_OVERVIEW_HEADINGS & headings
        )
        legacy_layout = LEGACY_README_HEADINGS <= headings
        if not new_layout and not legacy_layout:
            errors.append(
                "README.md must use the current overview layout or the complete legacy layout"
            )
        elif legacy_layout:
            print(
                "WARNING: legacy README layout detected; migrate it to the overview-and-links "
                "layout when this experiment is next updated"
            )
        heading = first_h1(readme)
        heading_matches = heading is not None and (
            args.experiment in heading
            if new_layout
            else experiment_id(args.experiment) in heading.lower()
        )
        if not heading_matches:
            errors.append("README.md H1 does not match the experiment directory")

        result_path = experiment_dir / "result.md"
        if new_layout and result_path.exists():
            result = result_path.read_text()
            missing_result_headings = NEW_RESULT_HEADINGS - markdown_headings(result)
            if missing_result_headings:
                errors.append(
                    "result.md missing current sections: "
                    + ", ".join(sorted(missing_result_headings))
                )
            result_heading = first_h1(result)
            if result_heading is None or args.experiment not in result_heading:
                errors.append("result.md H1 does not match the experiment directory")

    validate_required_directories(
        experiment_dir,
        legacy_layout=legacy_layout,
        errors=errors,
    )

    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        config = read_yaml(config_path)
        config_experiment_name = get_nested(config, "experiment.name")
        if config_experiment_name != args.experiment:
            errors.append(
                "config experiment.name does not match the experiment directory: "
                f"{config_experiment_name!r}"
            )
        validate_config_contract(
            config,
            allow_todo=args.allow_todo,
            project_defaults=project_experiment_defaults(load_project_config()),
            registered_ids=registered_hypothesis_ids(ROOT),
            errors=errors,
        )

    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"metrics.json is invalid JSON: {exc}")
        else:
            if not isinstance(metrics, dict):
                errors.append("metrics.json must contain a JSON object")
            elif metrics.get("experiment") is None and legacy_layout:
                print(
                    "WARNING: legacy metrics.json has no experiment identity; add it when this "
                    "experiment is next updated"
                )
            elif metrics.get("experiment") != args.experiment:
                errors.append(
                    "metrics.json experiment does not match the experiment directory: "
                    f"{metrics.get('experiment')!r}"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    mode = "structure" if args.allow_todo else "strict"
    print(f"experiment validation passed ({mode}): {args.experiment}")


if __name__ == "__main__":
    main()
