from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SURVEYS_DIR = ROOT / "docs" / "surveys"
README_PATH = SURVEYS_DIR / "README.md"
EXCLUDED_REPORTS = {"README.md", "summary.md"}
BEGIN_MARKER = "<!-- BEGIN AUTO SURVEY INDEX -->"
END_MARKER = "<!-- END AUTO SURVEY INDEX -->"
ALLOWED_STATUSES = {"draft", "final", "superseded"}
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
EXPERIMENT_PATTERN = re.compile(r"^exp\d+$")


@dataclass(frozen=True)
class SurveyReport:
    path: Path
    title: str
    report_date: str
    types: tuple[str, ...]
    experiments: tuple[str, ...]
    topics: tuple[str, ...]
    status: str
    summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate docs/surveys report metadata and regenerate README.md indexes."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if metadata is invalid or README.md is not up to date.",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Allow structurally valid draft reports while checking index freshness.",
    )
    return parser.parse_args()


def _front_matter(path: Path) -> dict[str, object]:
    text = path.read_text()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: YAML front matter is required")

    try:
        end_index = next(index for index, line in enumerate(lines[1:], start=1) if line == "---")
    except StopIteration as error:
        raise ValueError(f"{path}: YAML front matter is not closed") from error

    metadata = yaml.safe_load("\n".join(lines[1:end_index]))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: YAML front matter must be a mapping")
    return metadata


def _required_text(metadata: dict[str, object], key: str, path: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return value.strip()


def _string_list(
    metadata: dict[str, object],
    key: str,
    path: Path,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path}: {key} must be a list of strings")
    normalized = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
    if not allow_empty and not normalized:
        raise ValueError(f"{path}: {key} must contain at least one value")
    return normalized


def _iso_date(value: object, path: Path) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise ValueError(f"{path}: date must use YYYY-MM-DD") from error
    raise ValueError(f"{path}: date must use YYYY-MM-DD")


def load_report(path: Path) -> SurveyReport:
    metadata = _front_matter(path)
    title = _required_text(metadata, "title", path)
    report_date = _iso_date(metadata.get("date"), path)
    types = _string_list(metadata, "types", path, allow_empty=False)
    experiments = _string_list(metadata, "experiments", path, allow_empty=True)
    topics = _string_list(metadata, "topics", path, allow_empty=False)
    status = _required_text(metadata, "status", path)
    summary = _required_text(metadata, "summary", path)

    invalid_types = [value for value in types if not TAG_PATTERN.fullmatch(value)]
    invalid_topics = [value for value in topics if not TAG_PATTERN.fullmatch(value)]
    invalid_experiments = [
        value for value in experiments if not EXPERIMENT_PATTERN.fullmatch(value)
    ]
    if invalid_types:
        raise ValueError(f"{path}: invalid types: {', '.join(invalid_types)}")
    if invalid_topics:
        raise ValueError(f"{path}: invalid topics: {', '.join(invalid_topics)}")
    if invalid_experiments:
        raise ValueError(
            f"{path}: experiments must be short ids such as exp238: "
            f"{', '.join(invalid_experiments)}"
        )
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"{path}: status must be one of {', '.join(sorted(ALLOWED_STATUSES))}")
    if status != "draft" and "TODO" in summary.upper():
        raise ValueError(f"{path}: non-draft summary must not contain TODO")

    return SurveyReport(
        path=path,
        title=title,
        report_date=report_date,
        types=types,
        experiments=experiments,
        topics=topics,
        status=status,
        summary=summary,
    )


def load_reports(surveys_dir: Path = SURVEYS_DIR) -> list[SurveyReport]:
    paths = sorted(path for path in surveys_dir.glob("*.md") if path.name not in EXCLUDED_REPORTS)
    return sorted(
        (load_report(path) for path in paths),
        key=lambda report: (report.report_date, report.title),
        reverse=True,
    )


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _code_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "-"


def _report_link(report: SurveyReport) -> str:
    return f"[{report.title}]({report.path.name})"


def _grouped_index(reports: list[SurveyReport], field: str, heading: str) -> list[str]:
    grouped: dict[str, list[SurveyReport]] = {}
    for report in reports:
        for value in getattr(report, field):
            grouped.setdefault(value, []).append(report)

    lines = [f"## {heading}", "", "| キー | レポート |", "| --- | --- |"]
    if not grouped:
        lines.append("| - | - |")
        return lines

    for value in sorted(grouped):
        links = "<br>".join(_report_link(report) for report in grouped[value])
        lines.append(f"| `{value}` | {links} |")
    return lines


def render_generated_index(reports: list[SurveyReport]) -> str:
    lines = [
        BEGIN_MARKER,
        "## レポート一覧",
        "",
        "| 日付 | レポート | 種類 | 実験 | トピック | 状態 | 一行要約 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if reports:
        for report in reports:
            lines.append(
                "| "
                f"{report.report_date} | {_report_link(report)} | {_code_list(report.types)} | "
                f"{_code_list(report.experiments)} | {_code_list(report.topics)} | "
                f"`{report.status}` | {_escape_cell(report.summary)} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")

    lines.extend([""] + _grouped_index(reports, "experiments", "実験番号別"))
    lines.extend([""] + _grouped_index(reports, "types", "種類別"))
    lines.extend([""] + _grouped_index(reports, "topics", "トピック別"))
    lines.extend([END_MARKER, ""])
    return "\n".join(lines)


def expected_readme(readme_path: Path, reports: list[SurveyReport]) -> str:
    text = readme_path.read_text()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise ValueError(f"{readme_path}: generated index markers are required")
    prefix, remainder = text.split(BEGIN_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    return f"{prefix}{render_generated_index(reports)}{suffix.lstrip()}"


def update_index(
    *,
    surveys_dir: Path = SURVEYS_DIR,
    readme_path: Path = README_PATH,
    check: bool = False,
    allow_draft: bool = False,
) -> bool:
    reports = load_reports(surveys_dir)
    if check and not allow_draft:
        drafts = [report.path.name for report in reports if report.status == "draft"]
        if drafts:
            raise SystemExit("draft survey reports are not complete: " + ", ".join(drafts))
    expected = expected_readme(readme_path, reports)
    current = readme_path.read_text()
    if current == expected:
        print(f"survey index is up to date ({len(reports)} reports)")
        return False
    if check:
        raise SystemExit("docs/surveys/README.md is out of date; run task update-survey-index")
    readme_path.write_text(expected)
    try:
        display_path = readme_path.relative_to(ROOT)
    except ValueError:
        display_path = readme_path
    print(f"updated {display_path} ({len(reports)} reports)")
    return True


def main() -> None:
    args = parse_args()
    update_index(check=args.check, allow_draft=args.allow_draft)


if __name__ == "__main__":
    main()
