from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "discussion"


def run_csv(cmd: list[str]) -> list[dict[str, str]]:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines))


def list_topics(competition: str, sort_by: str, max_pages: int) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        rows = run_csv(
            [
                "kaggle",
                "competitions",
                "topics",
                "list",
                competition,
                "-s",
                sort_by,
                "-p",
                str(page),
                "-v",
            ]
        )
        new_rows = [row for row in rows if row.get("id") and row["id"] not in seen]
        if not new_rows:
            break
        for row in new_rows:
            seen.add(row["id"])
            row["source_page"] = str(page)
            row["sort_by"] = sort_by
        topics.extend(new_rows)
    return topics


def archive_topic(
    competition: str,
    topic: dict[str, str],
    output_dir: Path,
    converter: Path,
    force: bool,
) -> str:
    topic_id = topic["id"]
    title = topic.get("title", topic_id)
    slug = f"{competition}-{topic_id}-{slugify(title)[:80]}"
    out_path = output_dir / f"{slug}.md"
    if out_path.exists() and not force:
        return f"skip existing {topic_id} -> {out_path}"

    proc = subprocess.run(
        [
            "kaggle",
            "competitions",
            "topics",
            "show",
            competition,
            topic_id,
            "--page-size",
            "200",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(proc.stdout)
        tmp_path = Path(handle.name)

    try:
        subprocess.run(
            [
                "python3",
                str(converter),
                str(tmp_path),
                "--title",
                title,
                "--url",
                f"https://www.kaggle.com/competitions/{competition}/discussion/{topic_id}",
                "--slug",
                slug,
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return f"archived {topic_id} -> {out_path}"


def write_listing(topics: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "title", "authorName", "commentCount", "votes", "postDate", "source_page", "sort_by"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(topics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", required=True)
    parser.add_argument("--sort-by", default="recent")
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/discussions"))
    parser.add_argument("--listing", type=Path, default=None)
    parser.add_argument("--converter", type=Path, default=Path(".agents/skills/kaggle-discussion-archive/scripts/html_to_discussion_md.py"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    topics = list_topics(args.competition, args.sort_by, args.max_pages)
    listing = args.listing or args.output_dir / f"{args.competition}_topics_{args.sort_by}.csv"
    write_listing(topics, listing)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for topic in topics:
        print(archive_topic(args.competition, topic, args.output_dir, args.converter, args.force))
    print(f"total_topics={len(topics)} listing={listing} output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
