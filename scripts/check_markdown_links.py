from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

try:
    from .config_utils import ROOT
except ImportError:  # Direct execution: `uv run python scripts/check_markdown_links.py`
    from config_utils import ROOT


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv"}
SKIP_PREFIXES = ("#", "/", "http://", "https://", "mailto:", "data:")


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def normalized_target(raw_target: str) -> str | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(SKIP_PREFIXES):
        return None
    if "{{" in target or "}}" in target or any(char in target for char in "*?"):
        return None
    target = unquote(target.split("#", 1)[0])
    return target or None


def broken_links() -> list[str]:
    offenders: list[str] = []
    for path in markdown_files():
        text = path.read_text(errors="replace")
        for match in LINK_PATTERN.finditer(text):
            target = normalized_target(match.group(1))
            if target is None:
                continue
            destination = (path.parent / target).resolve()
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line}: missing local link target {target}"
                )
    return offenders


def main() -> None:
    offenders = broken_links()
    if offenders:
        raise SystemExit("\n".join(offenders))
    print(f"Markdown local links passed ({len(markdown_files())} files)")


if __name__ == "__main__":
    main()
