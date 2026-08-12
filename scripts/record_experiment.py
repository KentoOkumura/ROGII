from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config_utils import ROOT
from update_experiment_summary import collect_records, render_auto_block, update_summary

EXPERIMENTS_DIR = ROOT / "experiments"
SUMMARY_PATH = ROOT / "experiment_summary.md"
ALLOWED_STATUSES = {
    "planned",
    "running",
    "usable",
    "completed",
    "failed",
    "discarded",
    "deprecated",
    "leak-risk",
    "debug_completed",
    "scaffold_completed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one experiment result in metrics.json.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--status", default="")
    parser.add_argument("--cv", default="")
    parser.add_argument("--public-lb", default="")
    parser.add_argument("--private-lb", default="")
    parser.add_argument("--metric", default="")
    parser.add_argument("--key-idea", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Set a dotted key below metrics.json evidence. Repeat as needed; "
            "VALUE is parsed as JSON when possible."
        ),
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not regenerate experiment_summary.md after writing metrics.json.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def parse_score(value: str) -> str | float | None:
    cleaned = value.strip()
    if cleaned in {"", "-", "None", "null"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def set_if_provided(metrics: dict[str, Any], key: str, value: str) -> None:
    if value.strip() == "":
        return
    metrics[key] = value


def set_score_if_provided(metrics: dict[str, Any], key: str, value: str) -> None:
    if value.strip() == "":
        return
    metrics[key] = parse_score(value)


def parse_evidence_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_nested_value(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid evidence key: {dotted_key!r}")
    current = mapping
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(
                f"cannot set evidence key {dotted_key!r}: {part!r} is not an object"
            )
        current = child
    current[parts[-1]] = value


def apply_evidence_assignments(metrics: dict[str, Any], assignments: list[str]) -> None:
    if not assignments:
        return
    evidence = metrics.setdefault("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("metrics.json evidence must be a JSON object")
    for assignment in assignments:
        key, separator, raw_value = assignment.partition("=")
        if not separator:
            raise ValueError(
                f"invalid --evidence value {assignment!r}; expected KEY=VALUE"
            )
        set_nested_value(evidence, key.strip(), parse_evidence_value(raw_value))


def regenerate_summary() -> None:
    records = collect_records()
    auto_block = render_auto_block(records)
    existing = SUMMARY_PATH.read_text() if SUMMARY_PATH.exists() else ""
    SUMMARY_PATH.write_text(update_summary(existing, auto_block))


def main() -> None:
    args = parse_args()
    experiment_dir = EXPERIMENTS_DIR / args.experiment
    if not experiment_dir.is_dir():
        raise SystemExit(f"experiment does not exist: {experiment_dir.relative_to(ROOT)}")

    if args.status and args.status not in ALLOWED_STATUSES:
        allowed = ", ".join(sorted(ALLOWED_STATUSES))
        raise SystemExit(f"invalid status: {args.status}. allowed: {allowed}")

    metrics_path = experiment_dir / "metrics.json"
    metrics = read_json(metrics_path)
    metrics.setdefault("experiment", args.experiment)

    set_if_provided(metrics, "status", args.status)
    set_score_if_provided(metrics, "cv", args.cv)
    set_score_if_provided(metrics, "public_lb", args.public_lb)
    set_score_if_provided(metrics, "private_lb", args.private_lb)
    set_if_provided(metrics, "metric", args.metric)
    set_if_provided(metrics, "key_idea", args.key_idea)
    set_if_provided(metrics, "notes", args.notes)
    try:
        apply_evidence_assignments(metrics, args.evidence)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    metrics["updated_at"] = datetime.now(UTC).isoformat()

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    if not args.no_summary:
        regenerate_summary()

    print(f"Recorded experiment metrics: {metrics_path.relative_to(ROOT)}")
    if not args.no_summary:
        print(f"Updated {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
