from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .config_utils import get_nested, load_project_config
    from .prepare_kaggle_notebooks import metadata_validation_errors
except ImportError:  # Direct execution: `uv run python scripts/validate_kaggle_metadata.py ...`
    from config_utils import get_nested, load_project_config
    from prepare_kaggle_notebooks import metadata_validation_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one prepared Kaggle notebook package before push."
    )
    parser.add_argument("--package-dir", required=True, type=Path)
    return parser.parse_args()


def validate_package(package_dir: Path) -> dict[str, Any]:
    metadata_path = package_dir / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing Kaggle metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Kaggle metadata must be a JSON object: {metadata_path}")

    project_config = load_project_config()
    expected_enable_internet = get_nested(
        project_config,
        "runtime.kaggle.enable_internet",
    )
    if not isinstance(expected_enable_internet, bool):
        raise ValueError(
            "project.yml runtime.kaggle.enable_internet must be true or false"
        )
    errors = metadata_validation_errors(
        metadata,
        expected_enable_internet=expected_enable_internet,
    )
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file:
        errors.append("code_file is empty")
    elif not (package_dir / code_file).is_file():
        errors.append(f"code_file does not exist in package: {code_file}")
    if errors:
        raise ValueError("; ".join(errors))
    return metadata


def main() -> None:
    args = parse_args()
    try:
        metadata = validate_package(args.package_dir)
    except ValueError as exc:
        raise SystemExit(f"Kaggle metadata validation failed: {exc}") from exc
    print(f"Kaggle metadata is push-ready: {metadata['id']}")


if __name__ == "__main__":
    main()
