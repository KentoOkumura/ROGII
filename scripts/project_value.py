from __future__ import annotations

import argparse
from typing import Any

try:
    from .config_utils import get_nested, is_todo, load_project_config
except ImportError:  # Direct execution: `uv run python scripts/project_value.py ...`
    from config_utils import get_nested, is_todo, load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print one configured scalar value from project.yml."
    )
    parser.add_argument("key", help="Dotted key, for example competition.slug")
    return parser.parse_args()


def configured_scalar(config: dict[str, Any], dotted_key: str) -> str:
    value = get_nested(config, dotted_key)
    if is_todo(value):
        raise ValueError(f"project.yml value is not configured: {dotted_key}")
    if isinstance(value, (dict, list)):
        raise ValueError(f"project.yml value is not a scalar: {dotted_key}")
    return str(value)


def main() -> None:
    args = parse_args()
    print(configured_scalar(load_project_config(), args.key))


if __name__ == "__main__":
    main()
