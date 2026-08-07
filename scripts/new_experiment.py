from __future__ import annotations

import argparse
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = ("artifacts", "features", "variants")
IGNORED_DIRS = (*GENERATED_DIRS, "kaggle")
STEERING_DIR = ROOT / ".steering"


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
    return parser.parse_args()


def copy_tree(source: Path, destination: Path, force: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(f"{destination} already exists. Use --force to overwrite.")
        shutil.rmtree(destination)

    ignore = shutil.ignore_patterns(
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        *IGNORED_DIRS,
    )
    shutil.copytree(source, destination, ignore=ignore)
    for dirname in GENERATED_DIRS:
        (destination / dirname).mkdir(parents=True, exist_ok=True)


def replace_tokens(path: Path, experiment_name: str) -> None:
    today = date.today().isoformat()
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        text = file_path.read_text()
        text = text.replace("{{ EXPERIMENT_NAME }}", experiment_name)
        text = text.replace("{{EXPERIMENT_NAME}}", experiment_name)
        text = text.replace("{{ TODAY }}", today)
        text = text.replace("{{TODAY}}", today)
        file_path.write_text(text)

    for file_path in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        new_name = file_path.name
        new_name = new_name.replace("{{ EXPERIMENT_NAME }}", experiment_name)
        new_name = new_name.replace("{{EXPERIMENT_NAME}}", experiment_name)
        new_name = new_name.replace("{{ TODAY }}", today)
        new_name = new_name.replace("{{TODAY}}", today)
        if new_name != file_path.name:
            file_path.rename(file_path.with_name(new_name))


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

    copy_tree(source, destination, args.force)
    replace_tokens(destination, args.name)

    print(f"Created {destination.relative_to(ROOT)}")
    print(
        "Next: fill the matching .steering plan, then implement and validate the experiment. "
        "Shared competition defaults are inherited from project.yml."
    )


if __name__ == "__main__":
    main()
