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
HYPOTHESIS_PATTERN = re.compile(r"^HYP-\d{8}-\d{2}$")
HYPOTHESIS_FIND_PATTERN = re.compile(r"HYP-\d{8}-\d{2}")
PLACEHOLDER_PATTERN = re.compile(
    r"(?:TODO|TBD|FIXME)|\{\{[^{}\n]+\}\}",
    flags=re.IGNORECASE,
)
BODY_HYPOTHESIS_PATTERN = re.compile(r"(?m)^- 対応する上位仮説:\s*(?P<value>.+?)\s*$")


@dataclass(frozen=True)
class SurveyReport:
    path: Path
    title: str
    report_date: str
    types: tuple[str, ...]
    hypotheses: tuple[str, ...]
    experiments: tuple[str, ...]
    topics: tuple[str, ...]
    status: str
    summary: str
    superseded_by: str | None


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


def _document_parts(path: Path) -> tuple[dict[str, object], str]:
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
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return metadata, body


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
    allow_missing: bool = False,
) -> tuple[str, ...]:
    value = metadata.get(key, [] if allow_missing else None)
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


def _superseded_by_filename(value: object, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: superseded reports must set superseded_by")
    filename = value.strip()
    if Path(filename).name != filename or not filename.endswith(".md"):
        raise ValueError(f"{path}: superseded_by must be a Markdown filename in docs/surveys")
    return filename


def _validate_supersession_chain(path: Path, successor: str) -> None:
    origin = path.resolve()
    seen = {origin}
    current_path = path
    current_successor = successor

    while True:
        successor_path = current_path.parent / current_successor
        resolved_successor = successor_path.resolve()
        if resolved_successor in seen:
            raise ValueError(f"{path}: superseded_by chain contains a cycle")
        if not successor_path.is_file():
            raise ValueError(
                f"{current_path}: superseded_by must reference an existing replacement report"
            )
        seen.add(resolved_successor)

        successor_metadata, _ = _document_parts(successor_path)
        successor_status = successor_metadata.get("status")
        if successor_status == "final":
            return
        if successor_status != "superseded":
            raise ValueError(f"{path}: superseded_by chain must terminate at a final report")

        current_path = successor_path
        current_successor = _superseded_by_filename(
            successor_metadata.get("superseded_by"),
            current_path,
        )


def load_report(path: Path) -> SurveyReport:
    metadata, body = _document_parts(path)
    title = _required_text(metadata, "title", path)
    report_date = _iso_date(metadata.get("date"), path)
    types = _string_list(metadata, "types", path, allow_empty=False)
    hypotheses = _string_list(
        metadata,
        "hypotheses",
        path,
        allow_empty=True,
        allow_missing=True,
    )
    experiments = _string_list(metadata, "experiments", path, allow_empty=True)
    topics = _string_list(metadata, "topics", path, allow_empty=False)
    status = _required_text(metadata, "status", path)
    summary = _required_text(metadata, "summary", path)
    superseded_by_value = metadata.get("superseded_by")

    invalid_types = [value for value in types if not TAG_PATTERN.fullmatch(value)]
    invalid_topics = [value for value in topics if not TAG_PATTERN.fullmatch(value)]
    invalid_experiments = [
        value for value in experiments if not EXPERIMENT_PATTERN.fullmatch(value)
    ]
    invalid_hypotheses = [value for value in hypotheses if not HYPOTHESIS_PATTERN.fullmatch(value)]
    if invalid_types:
        raise ValueError(f"{path}: invalid types: {', '.join(invalid_types)}")
    if invalid_topics:
        raise ValueError(f"{path}: invalid topics: {', '.join(invalid_topics)}")
    if invalid_experiments:
        raise ValueError(
            f"{path}: experiments must be short ids such as exp238: "
            f"{', '.join(invalid_experiments)}"
        )
    if invalid_hypotheses:
        raise ValueError(
            f"{path}: hypotheses must use HYP-YYYYMMDD-NN: {', '.join(invalid_hypotheses)}"
        )
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"{path}: status must be one of {', '.join(sorted(ALLOWED_STATUSES))}")
    superseded_by: str | None = None
    if status == "superseded":
        superseded_by = _superseded_by_filename(superseded_by_value, path)
        _validate_supersession_chain(path, superseded_by)
    elif superseded_by_value is not None and superseded_by_value != "":
        raise ValueError(f"{path}: superseded_by is only valid when status is superseded")
    if status != "draft" and PLACEHOLDER_PATTERN.search(summary):
        raise ValueError(f"{path}: non-draft summary must not contain placeholders")
    if status != "draft" and PLACEHOLDER_PATTERN.search(body):
        raise ValueError(f"{path}: non-draft body must not contain placeholders")

    body_hypothesis_match = BODY_HYPOTHESIS_PATTERN.search(body)
    if status != "draft" and body_hypothesis_match is None:
        raise ValueError(f"{path}: non-draft reports must declare corresponding hypotheses")
    if hypotheses and body_hypothesis_match is None:
        raise ValueError(f"{path}: reports with hypotheses metadata must declare them in the body")
    if body_hypothesis_match is not None:
        body_value = body_hypothesis_match.group("value").strip()
        body_hypotheses = tuple(dict.fromkeys(HYPOTHESIS_FIND_PATTERN.findall(body_value)))
        if set(body_hypotheses) != set(hypotheses):
            raise ValueError(f"{path}: body hypothesis declaration must match hypotheses metadata")
        if status != "draft" and not hypotheses and body_value != "なし":
            raise ValueError(
                f"{path}: a non-hypothesis report must declare '対応する上位仮説: なし'"
            )

    return SurveyReport(
        path=path,
        title=title,
        report_date=report_date,
        types=types,
        hypotheses=hypotheses,
        experiments=experiments,
        topics=topics,
        status=status,
        summary=summary,
        superseded_by=superseded_by,
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


def _replacement_link(report: SurveyReport) -> str:
    return f"[後継]({report.superseded_by})" if report.superseded_by else "-"


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
        "| 日付 | レポート | 種類 | 上位仮説 | 実験 | トピック | 状態 | 後継 | 一行要約 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if reports:
        for report in reports:
            lines.append(
                "| "
                f"{report.report_date} | {_report_link(report)} | {_code_list(report.types)} | "
                f"{_code_list(report.hypotheses)} | "
                f"{_code_list(report.experiments)} | {_code_list(report.topics)} | "
                f"`{report.status}` | {_replacement_link(report)} | "
                f"{_escape_cell(report.summary)} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - |")

    lines.extend([""] + _grouped_index(reports, "hypotheses", "上位仮説別"))
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
