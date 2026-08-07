from __future__ import annotations

import argparse

from update_experiment_summary import ExperimentRecord, collect_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact experiment comparison table.")
    parser.add_argument(
        "--status",
        default="",
        help="Filter by rendered status label or raw status text.",
    )
    parser.add_argument(
        "--route",
        default="",
        help="Filter by rendered route label or raw route text.",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "updated", "cv", "public-lb", "private-lb"],
        default="name",
    )
    parser.add_argument("--desc", action="store_true")
    return parser.parse_args()


def score_key(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("-inf")


def sort_records(records: list[ExperimentRecord], key: str, desc: bool) -> list[ExperimentRecord]:
    if key == "cv":
        return sorted(records, key=lambda record: score_key(record.cv), reverse=desc)
    if key == "public-lb":
        return sorted(records, key=lambda record: score_key(record.public_lb), reverse=desc)
    if key == "private-lb":
        return sorted(records, key=lambda record: score_key(record.private_lb), reverse=desc)
    if key == "updated":
        return sorted(records, key=lambda record: record.updated, reverse=desc)
    return sorted(records, key=lambda record: record.name, reverse=desc)


def render_table(records: list[ExperimentRecord]) -> str:
    lines = [
        "| Experiment | Route | Status | CV | Public LB | Private LB | Key Idea | Updated |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.name} | {record.route} | {record.status} | "
            f"{record.cv} | {record.public_lb} | "
            f"{record.private_lb} | {record.summary} | {record.updated} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = collect_records()
    if args.status:
        needle = args.status.lower()
        records = [record for record in records if needle in record.status.lower()]
    if args.route:
        needle = args.route.lower()
        records = [record for record in records if needle in record.route.lower()]
    records = sort_records(records, args.sort, args.desc)

    if not records:
        print("No experiments found.")
        return

    print(render_table(records))


if __name__ == "__main__":
    main()
