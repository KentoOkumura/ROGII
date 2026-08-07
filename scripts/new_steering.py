from __future__ import annotations

import argparse
import re
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "templates" / "steering"
STEERING_DIR = ROOT / ".steering"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a steering document folder.")
    parser.add_argument("--experiment", required=True, help="Experiment name, e.g. expXXX_model")
    parser.add_argument(
        "--title", default=None, help="Human-readable title for the steering folder"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing steering folder"
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def destination_name(experiment: str, title: str | None) -> str:
    exp_slug = slugify(experiment)
    title_slug = slugify(title or re.sub(r"^exp\d+[-_]?", "", experiment) or experiment)
    if title_slug and title_slug not in exp_slug:
        exp_slug = f"{exp_slug}-{title_slug}"
    return f"{date.today().strftime('%Y%m%d')}-{exp_slug}"


def replace_tokens(path: Path, experiment: str) -> None:
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        text = file_path.read_text()
        text = text.replace("{{ EXPERIMENT_NAME }}", experiment)
        text = text.replace("{{ TODAY }}", date.today().isoformat())
        file_path.write_text(text)


def main() -> None:
    args = parse_args()
    destination = STEERING_DIR / destination_name(args.experiment, args.title)

    if destination.exists():
        if not args.force:
            raise FileExistsError(f"{destination.relative_to(ROOT)} already exists")
        shutil.rmtree(destination)

    shutil.copytree(TEMPLATE_DIR, destination)
    replace_tokens(destination, args.experiment)
    print(f"Created {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
