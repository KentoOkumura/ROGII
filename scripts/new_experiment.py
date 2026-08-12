from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = ("artifacts",)
LEGACY_GENERATED_DIRS = ("features", "variants")
IGNORED_DIRS = (*GENERATED_DIRS, *LEGACY_GENERATED_DIRS, "kaggle")
STEERING_DIR = ROOT / ".steering"
RESET_RECORD_FILES = ("README.md", "SESSION_NOTES.md", "result.md", "metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new experiment from a template.")
    parser.add_argument("--name", required=True, help="Experiment name, e.g. expXXX_next_idea")
    parser.add_argument(
        "--source",
        default="templates/experiment",
        help="Template directory or existing experiment directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing experiment directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the destination without writing files",
    )
    parser.add_argument(
        "--skip-steering-check",
        action="store_true",
        help="Allow creating an experiment without a matching .steering plan",
    )
    parser.add_argument(
        "--copy-tests",
        action="store_true",
        help="Copy tests from a source experiment; review copied paths and contracts manually",
    )
    return parser.parse_args()


def copy_tree(source: Path, destination: Path, force: bool, copy_tests: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists. Use --force to overwrite.")
        shutil.rmtree(destination)

    ignored_names = [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        *IGNORED_DIRS,
    ]
    if source.parent == ROOT / "experiments" and not copy_tests:
        ignored_names.append("tests")
    ignore = shutil.ignore_patterns(*ignored_names)
    shutil.copytree(source, destination, ignore=ignore)
    for dirname in GENERATED_DIRS:
        generated_dir = destination / dirname
        generated_dir.mkdir(parents=True, exist_ok=True)
        (generated_dir / ".gitkeep").touch()


def replace_text(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            file_path.write_text(updated)


def rename_paths(path: Path, replacements: tuple[tuple[str, str], ...]) -> None:
    for file_path in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        new_name = file_path.name
        for old, new in replacements:
            new_name = new_name.replace(old, new)
        if new_name != file_path.name:
            file_path.rename(file_path.with_name(new_name))


def replace_tokens(path: Path, experiment_name: str) -> None:
    today = date.today().isoformat()
    replacements = (
        ("{{ EXPERIMENT_NAME }}", experiment_name),
        ("{{EXPERIMENT_NAME}}", experiment_name),
        ("{{ TODAY }}", today),
        ("{{TODAY}}", today),
    )
    replace_text(path, replacements)
    rename_paths(path, replacements)


def replace_parent_experiment_identity(
    destination: Path,
    source_experiment: str,
    experiment_name: str,
) -> None:
    replacements = ((source_experiment, experiment_name),)
    replace_text(destination, replacements)
    rename_paths(destination, replacements)


def reset_parent_records(
    destination: Path,
    experiment_name: str,
    source_experiment: str,
) -> None:
    template_dir = ROOT / "templates" / "experiment"
    missing = [name for name in RESET_RECORD_FILES if not (template_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"experiment templates missing: {', '.join(missing)}")

    for filename in RESET_RECORD_FILES:
        text = (template_dir / filename).read_text()
        text = text.replace("{{ EXPERIMENT_NAME }}", experiment_name)
        text = text.replace("{{EXPERIMENT_NAME}}", experiment_name)
        text = text.replace("{{ TODAY }}", date.today().isoformat())
        text = text.replace("{{TODAY}}", date.today().isoformat())
        (destination / filename).write_text(text)

    config_path = destination / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError("source experiment is missing config.yaml")
    config = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError("source experiment config.yaml must contain a mapping")
    experiment = config.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("source experiment config.yaml experiment must contain a mapping")
    experiment["name"] = experiment_name
    experiment["description"] = "TODO"
    experiment["created_at"] = date.today().isoformat()
    experiment.pop("updated_at", None)
    experiment.pop("status", None)
    lineage = config.setdefault("lineage", {})
    if not isinstance(lineage, dict):
        raise ValueError("source experiment config.yaml lineage must contain a mapping")
    lineage["parent"] = source_experiment
    lineage["diff_summary"] = "TODO"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    metrics = json.loads((destination / "metrics.json").read_text())
    if metrics.get("experiment") != experiment_name or metrics.get("status") != "planned":
        raise RuntimeError("reset metrics.json did not produce a planned child experiment")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def experiment_id(experiment_name: str) -> str:
    match = re.search(r"exp\d+", experiment_name.lower())
    return match.group(0) if match else experiment_name


def matching_steering_dirs(experiment_name: str) -> list[Path]:
    if not STEERING_DIR.exists():
        return []

    normalized_experiment = normalize_name(experiment_name)
    exp_id = experiment_id(experiment_name)
    matches: list[Path] = []
    for path in STEERING_DIR.iterdir():
        if not path.is_dir():
            continue
        normalized_path = normalize_name(path.name)
        if normalized_experiment in normalized_path or exp_id in normalized_path:
            matches.append(path)
    return matches


def require_steering_plan(experiment_name: str) -> None:
    if matching_steering_dirs(experiment_name):
        return

    suggested = f".steering/{date.today().strftime('%Y%m%d')}-{normalize_name(experiment_name)}/"
    raise SystemExit(
        "missing matching .steering plan. "
        f"Create one first, for example: {suggested} "
        "or pass --skip-steering-check for exceptional scaffolding."
    )


def main() -> None:
    args = parse_args()
    source = (ROOT / args.source).resolve()
    destination = ROOT / "experiments" / args.name

    if not source.exists():
        raise FileNotFoundError(f"Template/source does not exist: {source}")

    if not args.skip_steering_check:
        require_steering_plan(args.name)

    if args.dry_run:
        print(f"Source: {source.relative_to(ROOT)}")
        print(f"Destination: {destination.relative_to(ROOT)}")
        print(f"Exists: {destination.exists()}")
        return

    copy_tree(source, destination, args.force, args.copy_tests)
    if source.parent == ROOT / "experiments":
        replace_parent_experiment_identity(destination, source.name, args.name)
        reset_parent_records(destination, args.name, source.name)
    else:
        replace_tokens(destination, args.name)

    print(f"Created {destination.relative_to(ROOT)}")
    if source.parent == ROOT / "experiments" and not args.copy_tests:
        print("Source experiment tests were not copied. Add tests for the new experiment contract.")
    elif source.parent == ROOT / "experiments" and args.copy_tests:
        print("Source experiment tests were copied; review old experiment paths and expectations.")
    if source.parent == ROOT / "experiments":
        print(
            "Parent experiment identity was replaced and execution records were reset to planned. "
            "Review copied input paths and contracts before implementation."
        )
    print(
        "Next: fill the matching .steering plan, then implement and validate the experiment. "
        "Shared competition defaults are inherited from project.yml."
    )


if __name__ == "__main__":
    main()
