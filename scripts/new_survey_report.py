from __future__ import annotations

import argparse
import re
from datetime import date

import yaml
from update_survey_index import (
    EXPERIMENT_PATTERN,
    HYPOTHESIS_PATTERN,
    ROOT,
    SURVEYS_DIR,
    TAG_PATTERN,
    update_index,
)

TEMPLATE_PATH = ROOT / "templates" / "survey" / "report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a metadata-indexed draft investigation report in docs/surveys."
    )
    parser.add_argument("--title", required=True, help="Human-readable report title")
    parser.add_argument("--slug", required=True, help="Filename slug without date or extension")
    parser.add_argument("--type", action="append", dest="types", default=[])
    parser.add_argument("--hypothesis", action="append", dest="hypotheses", default=[])
    parser.add_argument("--experiment", action="append", dest="experiments", default=[])
    parser.add_argument("--topic", action="append", dest="topics", default=[])
    parser.add_argument("--summary", default="TODO", help="One-line conclusion for the index")
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD")
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def _validate(values: list[str], pattern: re.Pattern[str], label: str) -> list[str]:
    normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    invalid = [value for value in normalized if not pattern.fullmatch(value)]
    if invalid:
        raise SystemExit(f"invalid {label}: {', '.join(invalid)}")
    return normalized


def render_body(title: str, report_date: str, hypotheses: list[str]) -> str:
    body = TEMPLATE_PATH.read_text()
    body = body.replace("{{ TITLE }}", title)
    body = body.replace("{{ DATE }}", report_date)
    hypothesis_text = ", ".join(f"`{value}`" for value in hypotheses) if hypotheses else "なし"
    return body.replace("{{ HYPOTHESES }}", hypothesis_text)


def main() -> None:
    args = parse_args()
    report_date = date.fromisoformat(args.date).isoformat()
    types = _validate(args.types, TAG_PATTERN, "type")
    hypotheses = _validate(args.hypotheses, HYPOTHESIS_PATTERN, "hypothesis")
    experiments = _validate(args.experiments, EXPERIMENT_PATTERN, "experiment")
    topics = _validate(args.topics, TAG_PATTERN, "topic")
    if not types:
        raise SystemExit("at least one --type is required")
    if not topics:
        raise SystemExit("at least one --topic is required")
    slug = slugify(args.slug)
    if not slug:
        raise SystemExit("--slug must contain an ASCII letter or number")
    compact_date = report_date.replace("-", "")
    destination = SURVEYS_DIR / f"{slug}_{compact_date}.md"
    if destination.exists():
        raise FileExistsError(f"{destination.relative_to(ROOT)} already exists")

    metadata = {
        "title": args.title.strip(),
        "date": report_date,
        "types": types,
        "hypotheses": hypotheses,
        "experiments": experiments,
        "topics": topics,
        "status": "draft",
        "summary": args.summary.strip(),
    }
    front_matter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    body = render_body(args.title.strip(), report_date, hypotheses)
    destination.write_text(f"---\n{front_matter}\n---\n\n{body}")
    update_index()
    print(f"created {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
