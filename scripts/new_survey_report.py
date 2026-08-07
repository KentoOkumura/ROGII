from __future__ import annotations

import argparse
import re
from datetime import date

import yaml
from update_survey_index import EXPERIMENT_PATTERN, ROOT, SURVEYS_DIR, TAG_PATTERN, update_index

TEMPLATE_PATH = ROOT / "templates" / "survey" / "report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a metadata-indexed completed-investigation report in docs/surveys."
    )
    parser.add_argument("--title", required=True, help="Human-readable report title")
    parser.add_argument("--slug", required=True, help="Filename slug without date or extension")
    parser.add_argument("--type", action="append", dest="types", default=[])
    parser.add_argument("--experiment", action="append", dest="experiments", default=[])
    parser.add_argument("--topic", action="append", dest="topics", default=[])
    parser.add_argument("--summary", default="TODO", help="One-line conclusion for the index")
    parser.add_argument(
        "--status",
        choices=("draft", "final", "superseded"),
        default="draft",
    )
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


def main() -> None:
    args = parse_args()
    report_date = date.fromisoformat(args.date).isoformat()
    types = _validate(args.types, TAG_PATTERN, "type")
    experiments = _validate(args.experiments, EXPERIMENT_PATTERN, "experiment")
    topics = _validate(args.topics, TAG_PATTERN, "topic")
    if not types:
        raise SystemExit("at least one --type is required")
    if not topics:
        raise SystemExit("at least one --topic is required")
    if args.status != "draft" and "TODO" in args.summary.upper():
        raise SystemExit("--summary must be completed before using a non-draft status")

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
        "experiments": experiments,
        "topics": topics,
        "status": args.status,
        "summary": args.summary.strip(),
    }
    front_matter = yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()
    body = TEMPLATE_PATH.read_text()
    body = body.replace("{{ TITLE }}", args.title.strip())
    body = body.replace("{{ DATE }}", report_date)
    destination.write_text(f"---\n{front_matter}\n---\n\n{body}")
    update_index()
    print(f"created {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
