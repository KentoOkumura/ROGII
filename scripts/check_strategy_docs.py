from __future__ import annotations

import re
from pathlib import Path

try:
    from .config_utils import ROOT
except ImportError:  # Direct execution: `uv run python scripts/check_strategy_docs.py`
    from config_utils import ROOT


DIRECTION_PATH = Path("KAGGLE_DIRECTION.md")
BACKLOG_DIR = Path("docs/backlog")
MAX_DIRECTION_BYTES = 50_000
MAX_DIRECTION_LINES = 220
MAX_LINE_LENGTH = 800
ALLOWED_STATES = {"検討メモ・設計不可", "設計可能・実験化未承認"}
DETAIL_LINK_RE = re.compile(r"^\[`(?P<name>[a-z0-9_]+)`\]\(docs/backlog/(?P=name)\.md\)$")


def backlog_rows(text: str) -> list[tuple[int, list[str]]]:
    heading = "### 未着手バックログ"
    start = text.find(heading)
    if start < 0:
        return []
    rows: list[tuple[int, list[str]]] = []
    in_table = False
    first_line = text[:start].count("\n") + 1
    for offset, line in enumerate(text[start:].splitlines(), start=first_line):
        if line.startswith("| 優先度 | アイデア |"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("| ---"):
            continue
        if not line.startswith("| "):
            if rows:
                break
            continue
        cells = line[2:-2].split(" | ")
        rows.append((offset, cells))
    return rows


def validate_strategy_docs(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    direction = root / DIRECTION_PATH
    if not direction.is_file():
        return [f"missing {DIRECTION_PATH}"]
    text = direction.read_text()
    encoded_size = len(text.encode("utf-8"))
    lines = text.splitlines()
    if encoded_size > MAX_DIRECTION_BYTES:
        errors.append(f"{DIRECTION_PATH} is {encoded_size} bytes; limit is {MAX_DIRECTION_BYTES}")
    if len(lines) > MAX_DIRECTION_LINES:
        errors.append(f"{DIRECTION_PATH} has {len(lines)} lines; limit is {MAX_DIRECTION_LINES}")
    for line_number, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE_LENGTH:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has {len(line)} characters; "
                f"limit is {MAX_LINE_LENGTH}"
            )

    rows = backlog_rows(text)
    if not rows:
        errors.append(f"{DIRECTION_PATH} has no rows under '未着手バックログ'")
        return errors

    referenced: dict[str, int] = {}
    for line_number, cells in rows:
        if len(cells) != 5:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} backlog row has {len(cells)} cells; expected 5"
            )
            continue
        _priority, idea_link, _summary, _dependency, state_cell = cells
        match = DETAIL_LINK_RE.fullmatch(idea_link)
        if not match:
            errors.append(f"{DIRECTION_PATH}:{line_number} has an invalid candidate detail link")
            continue
        name = match.group("name")
        if name in referenced:
            errors.append(
                f"{DIRECTION_PATH}:{line_number} duplicates candidate {name} "
                f"from line {referenced[name]}"
            )
        referenced[name] = line_number
        state = state_cell.strip("`")
        if state not in ALLOWED_STATES or state_cell != f"`{state}`":
            errors.append(
                f"{DIRECTION_PATH}:{line_number} has invalid candidate state {state_cell}"
            )
            continue
        detail_path = root / BACKLOG_DIR / f"{name}.md"
        if not detail_path.is_file():
            errors.append(f"{DIRECTION_PATH}:{line_number} missing {detail_path.relative_to(root)}")
            continue
        detail = detail_path.read_text()
        if f"- 候補名: `{name}`" not in detail:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched candidate name")
        if f"- 状態: `{state}`" not in detail:
            errors.append(f"{detail_path.relative_to(root)} has a mismatched candidate state")

    backlog_dir = root / BACKLOG_DIR
    detail_names = {
        path.stem
        for path in backlog_dir.glob("*.md")
        if path.name not in {"_TEMPLATE.md", "README.md"}
    }
    orphaned = sorted(detail_names - referenced.keys())
    if orphaned:
        errors.append("unreferenced backlog detail files: " + ", ".join(orphaned))
    return errors


def main() -> None:
    errors = validate_strategy_docs()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Strategy documents passed")


if __name__ == "__main__":
    main()
