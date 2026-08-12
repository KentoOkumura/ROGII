from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".agents" / "skills"
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MIN_SHORT_DESCRIPTION_LENGTH = 25
MAX_SHORT_DESCRIPTION_LENGTH = 64
REQUIRED_FRONT_MATTER = {"name", "description"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate repository SKILL.md files and optional agents/openai.yaml metadata."
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=SKILLS_DIR,
        help="Directory whose immediate child directories are repository skills.",
    )
    return parser.parse_args()


def _front_matter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text()
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("YAML front matter must start on the first line")
    try:
        end_index = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("YAML front matter is not closed") from error

    try:
        metadata = yaml.safe_load("\n".join(lines[1:end_index]))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML front matter: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("YAML front matter must be a mapping")
    body = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, body


def _validate_skill_md(skill_dir: Path) -> list[str]:
    path = skill_dir / "SKILL.md"
    if not path.is_file():
        return ["SKILL.md is required"]

    try:
        metadata, body = _front_matter(path)
    except (OSError, ValueError) as error:
        return [str(error)]

    errors: list[str] = []
    non_string_keys = [repr(key) for key in metadata if not isinstance(key, str)]
    keys = {key for key in metadata if isinstance(key, str)}
    missing = sorted(REQUIRED_FRONT_MATTER - keys)
    unexpected = sorted(keys - REQUIRED_FRONT_MATTER)
    if non_string_keys:
        errors.append("front matter field names must be strings: " + ", ".join(non_string_keys))
    if missing:
        errors.append("missing front matter field(s): " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected front matter field(s): " + ", ".join(unexpected))

    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    else:
        name = name.strip()
        if len(name) > MAX_NAME_LENGTH:
            errors.append(f"name must be at most {MAX_NAME_LENGTH} characters")
        if not NAME_PATTERN.fullmatch(name):
            errors.append("name must use lowercase letters, digits, and single hyphens")
        if name != skill_dir.name:
            errors.append(f"name must match directory name {skill_dir.name!r}")

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("description must be a non-empty string")
    else:
        description = description.strip()
        if len(description) > MAX_DESCRIPTION_LENGTH:
            errors.append(f"description must be at most {MAX_DESCRIPTION_LENGTH} characters")
        if "<" in description or ">" in description:
            errors.append("description must not contain angle brackets")

    if not body:
        errors.append("Markdown instructions are required after front matter")
    return errors


def _required_interface_text(
    interface: dict[str, object], key: str, path: Path, errors: list[str]
) -> str | None:
    value = interface.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: interface.{key} must be a non-empty string")
        return None
    return value.strip()


def _validate_openai_yaml(skill_dir: Path) -> list[str]:
    path = skill_dir / "agents" / "openai.yaml"
    if not path.exists():
        return []
    if not path.is_file():
        return [f"{path}: must be a file"]

    try:
        metadata = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        return [f"{path}: invalid YAML: {error}"]
    if not isinstance(metadata, dict):
        return [f"{path}: top level must be a mapping"]

    interface = metadata.get("interface")
    if not isinstance(interface, dict):
        return [f"{path}: interface must be a mapping"]

    errors: list[str] = []
    _required_interface_text(interface, "display_name", path, errors)
    short_description = _required_interface_text(interface, "short_description", path, errors)
    default_prompt = _required_interface_text(interface, "default_prompt", path, errors)

    if short_description is not None and not (
        MIN_SHORT_DESCRIPTION_LENGTH <= len(short_description) <= MAX_SHORT_DESCRIPTION_LENGTH
    ):
        errors.append(
            f"{path}: interface.short_description must be "
            f"{MIN_SHORT_DESCRIPTION_LENGTH}-{MAX_SHORT_DESCRIPTION_LENGTH} characters"
        )
    if default_prompt is not None and f"${skill_dir.name}" not in default_prompt:
        errors.append(f"{path}: interface.default_prompt must mention ${skill_dir.name}")

    for key in ("icon_small", "icon_large"):
        value = interface.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{path}: interface.{key} must be a non-empty string")
            continue
        asset_path = (skill_dir / value).resolve()
        try:
            asset_path.relative_to(skill_dir.resolve())
        except ValueError:
            errors.append(f"{path}: interface.{key} must stay inside the skill directory")
            continue
        if not asset_path.is_file():
            errors.append(f"{path}: interface.{key} does not exist: {value}")
    return errors


def validate_skills(skills_dir: Path = SKILLS_DIR) -> tuple[list[str], int, int]:
    if not skills_dir.is_dir():
        return [f"{skills_dir}: skills directory does not exist"], 0, 0

    skill_dirs = sorted(
        path
        for path in skills_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
    )
    if not skill_dirs:
        return [f"{skills_dir}: no skill directories found"], 0, 0

    errors: list[str] = []
    metadata_count = 0
    for skill_dir in skill_dirs:
        for error in _validate_skill_md(skill_dir):
            errors.append(f"{skill_dir / 'SKILL.md'}: {error}")
        metadata_path = skill_dir / "agents" / "openai.yaml"
        if metadata_path.exists():
            metadata_count += 1
        errors.extend(_validate_openai_yaml(skill_dir))
    return errors, len(skill_dirs), metadata_count


def main() -> None:
    args = parse_args()
    errors, skill_count, metadata_count = validate_skills(args.skills_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"validated {skill_count} skills ({metadata_count} agents/openai.yaml files)")


if __name__ == "__main__":
    main()
