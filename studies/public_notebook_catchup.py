from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

COMPETITION = "rogii-wellbore-geology-prediction"
NOTEBOOK_ROOT = ROOT / "docs" / "notebooks" / COMPETITION
DEFAULT_LISTING_DIRS = ("score_ascending_latest", "vote_top", "date_run_recent")
DEFAULT_TARGET_REFS = (
    "needless090/lb-8-860-rogii-sel15-256seeds",
    "safar1/lb-score-8-863",
    "aidensong123/rogii-sel15-rerun",
    "nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based",
    "svanikkolli/aeroridge-engine-v2",
    "kojimar/rogii-physical-pf-signal-meets-artifact-stack",
    "kojimar/rogii-inference-stack-with-pf-beam-and-tabicl",
)

SIGNAL_PATTERNS = {
    "particle_filter": r"\bparticle\b|\bpf\b|particlefilter",
    "beam": r"\bbeam\b",
    "dwt": r"\bdwt\b|wavelet|pywt",
    "tabicl": r"tabicl",
    "aeroridge": r"aeroridge",
    "savitzky_golay": r"savgol|savitzky",
    "scale_selector": r"scale[_ -]?selector|pf_scale|scale\s*=",
    "seed_ensemble": r"n_seeds|num_seeds|256\s*seeds|128\s*seeds|for\s+seed",
    "gr_interpolation": r"interpolat.{0,24}\bgr\b|\bgr\b.{0,24}interpolat",
    "artifact_stack": r"tabicl|artifact|model_sources|dataset_sources|fresh[-_ ]?artifact",
    "formation_or_geology": r"\bformation\b|\bgeology\b",
    "static_submission": r"read_csv\([^)]*submission\.csv|submission\.csv[^\\n]{0,80}read",
    "visible_branch": r"000d7d20|visible|public\s+test|public_test",
}

METHOD_FAMILY_RULES = (
    ("aeroridge", ("aeroridge",)),
    ("dwt_alignment", ("dwt",)),
    ("pf_physical_sel15", ("particle_filter", "scale_selector", "seed_ensemble")),
    ("pf_beam", ("particle_filter", "beam")),
    ("postprocess", ("savitzky_golay",)),
    ("artifact_stack", ("tabicl", "artifact_stack")),
)


@dataclass
class NotebookRecord:
    ref: str
    title: str = ""
    author: str = ""
    first_listing: str = ""
    source_lists: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)
    last_run_time: str = ""
    total_votes: int | None = None
    metadata_path: Path | None = None
    notebook_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    known_score: str = ""
    signals: list[str] = field(default_factory=list)
    method_family: str = "unknown"
    risk_flags: list[str] = field(default_factory=list)
    replay_priority: str = "low"
    replay_notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory public Kaggle notebooks for replay/catch-up planning."
    )
    parser.add_argument("--notebook-root", type=Path, default=NOTEBOOK_ROOT)
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument(
        "--listing-dir",
        action="append",
        default=[],
        help="Listing directory under notebook-root. Defaults to score/vote/date archives.",
    )
    parser.add_argument(
        "--target-ref",
        action="append",
        default=[],
        help="Notebook ref that should be audited even if it is not in the first ranks.",
    )
    parser.add_argument(
        "--self-anchor",
        default="exp026_pseudo_tail_bucket_shrink_inference_submit Public LB 12.102",
    )
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_listing(path: Path, listing_name: str) -> list[tuple[int, dict[str, str]]]:
    if not path.exists():
        return []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        return [(idx, row) for idx, row in enumerate(reader, start=1)]


def discover_metadata(notebook_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for metadata_path in notebook_root.rglob("kernel-metadata.json"):
        metadata = read_json(metadata_path)
        ref = str(metadata.get("id", "")).strip()
        if ref and ref not in result:
            result[ref] = metadata_path
    return result


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open() as fp:
        value = json.load(fp)
    return value if isinstance(value, dict) else {}


def notebook_code_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    if path.suffix == ".py":
        return path.read_text(errors="replace")
    if path.suffix != ".ipynb":
        return ""
    with path.open() as fp:
        notebook = json.load(fp)
    cells = notebook.get("cells", [])
    chunks: list[str] = []
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                chunks.append("".join(str(item) for item in source))
            elif isinstance(source, str):
                chunks.append(source)
    return "\n".join(chunks)


def extract_known_score(title: str) -> str:
    lowered = title.lower()
    if not any(token in lowered for token in ("lb", "score", "cv", "best")):
        return ""
    match = re.search(r"(?<!\d)(\d{1,2}\.\d{2,6})(?!\d)", title)
    return match.group(1) if match else ""


def find_notebook_path(metadata_path: Path | None, metadata: dict[str, Any]) -> Path | None:
    if metadata_path is None:
        return None
    code_file = str(metadata.get("code_file", "")).strip()
    if code_file:
        candidate = metadata_path.parent / code_file
        if candidate.exists():
            return candidate
    for suffix in ("*.ipynb", "*.py"):
        candidates = sorted(path for path in metadata_path.parent.glob(suffix) if path.is_file())
        if candidates:
            return candidates[0]
    return None


def detect_signals(text: str) -> list[str]:
    lowered = text.lower()
    signals = [
        name
        for name, pattern in SIGNAL_PATTERNS.items()
        if re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL)
    ]
    return sorted(signals)


def classify_family(signals: list[str], title: str) -> str:
    signal_set = set(signals)
    title_lower = title.lower()
    if "sel15" in title_lower:
        return "pf_physical_sel15"
    if "tabicl" in title_lower or "artifact" in title_lower:
        return "artifact_stack"
    if "aeroridge" in title_lower:
        return "aeroridge"
    if "dwt" in title_lower:
        return "dwt_alignment"
    for family, required_any in METHOD_FAMILY_RULES:
        if any(signal in signal_set for signal in required_any):
            return family
    if "lgb" in title_lower or "cat" in title_lower:
        return "tabular_lgb_cat"
    return "unknown"


def source_count(metadata: dict[str, Any]) -> int:
    total = 0
    for key in ("dataset_sources", "kernel_sources", "model_sources"):
        value = metadata.get(key, [])
        if isinstance(value, list):
            total += len(value)
    return total


def detect_risks(metadata: dict[str, Any], signals: list[str]) -> list[str]:
    risks: list[str] = []
    if metadata.get("enable_internet"):
        risks.append("internet_enabled")
    if metadata.get("enable_gpu"):
        risks.append("gpu_required_or_enabled")
    if source_count(metadata):
        risks.append("external_artifact_dependency")
    if "formation_or_geology" in signals:
        risks.append("formation_or_geology_boundary_check")
    if "visible_branch" in signals:
        risks.append("public_visible_branch_check")
    if "static_submission" in signals:
        risks.append("static_submission_or_blend_check")
    return sorted(set(risks))


def priority_for(record: NotebookRecord, target_refs: set[str]) -> str:
    if record.ref in target_refs:
        return "target"
    score_rank = record.ranks.get("score_ascending_latest")
    if score_rank is not None and score_rank <= 5:
        return "high"
    if record.known_score:
        try:
            if float(record.known_score) < 9.5:
                return "high"
        except ValueError:
            pass
    if record.total_votes is not None and record.total_votes >= 80:
        return "medium"
    return "low"


def replay_notes(record: NotebookRecord) -> str:
    if "external_artifact_dependency" in record.risk_flags:
        return "Inventory external inputs before replay."
    if "internet_enabled" in record.risk_flags:
        return "Replay only after removing internet dependency."
    if record.metadata and not source_count(record.metadata):
        return "Replay candidate: no metadata external dataset/model/kernel sources."
    if not record.metadata:
        return "Metadata missing; refetch before replay."
    return "Review code and metadata before replay."


def score_value(record: NotebookRecord) -> float:
    try:
        return float(record.known_score)
    except ValueError:
        return 99.0


def select_first_replay_candidate(records: list[NotebookRecord]) -> NotebookRecord | None:
    candidates = [
        record
        for record in records
        if record.method_family == "pf_physical_sel15"
        and not source_count(record.metadata)
        and "internet_enabled" not in record.risk_flags
        and record.replay_priority in {"target", "high"}
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            score_value(item),
            item.ranks.get("score_ascending_latest", 999),
            item.ranks.get("vote_top", 999),
            item.ref,
        ),
    )[0]


def suggested_exp_name(record: NotebookRecord | None) -> str:
    if record is None:
        return "exp027_public_replay_selected_public_notebook"
    owner, _, slug = record.ref.partition("/")
    slug = slug or owner
    slug = re.sub(r"^lb[-_]?\d+(?:[-_.]\d+)?[-_]?", "", slug)
    slug = re.sub(r"^rogii[-_]?", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    owner = re.sub(r"[^a-z0-9]+", "_", owner.lower()).strip("_")
    parts = [part for part in (owner, slug) if part]
    suffix = "_".join(parts) or "selected_public_notebook"
    return f"exp027_public_replay_{suffix}"[:80].rstrip("_")


def collect_records(
    notebook_root: Path, listing_dirs: tuple[str, ...], target_refs: tuple[str, ...]
) -> list[NotebookRecord]:
    records: dict[str, NotebookRecord] = {}
    for listing_name in listing_dirs:
        listing_path = notebook_root / listing_name / "kernel_listing.csv"
        for rank, row in read_listing(listing_path, listing_name):
            ref = row.get("ref", "").strip()
            if not ref:
                continue
            record = records.setdefault(ref, NotebookRecord(ref=ref))
            if not record.first_listing:
                record.first_listing = listing_name
            record.source_lists.add(listing_name)
            record.ranks[listing_name] = min(rank, record.ranks.get(listing_name, rank))
            record.title = record.title or row.get("title", "").strip()
            record.author = record.author or row.get("author", "").strip()
            record.last_run_time = record.last_run_time or row.get("lastRunTime", "").strip()
            record.total_votes = record.total_votes or safe_int(row.get("totalVotes", ""))

    metadata_by_ref = discover_metadata(notebook_root)
    for ref in target_refs:
        if ref not in records:
            records[ref] = NotebookRecord(ref=ref, first_listing="target_ref")

    target_set = set(target_refs)
    for record in records.values():
        record.metadata_path = metadata_by_ref.get(record.ref)
        record.metadata = read_json(record.metadata_path)
        if record.metadata:
            record.title = record.title or str(record.metadata.get("title", ""))
            record.notebook_path = find_notebook_path(record.metadata_path, record.metadata)
        text = "\n".join([record.title, notebook_code_text(record.notebook_path)])
        record.known_score = extract_known_score(record.title)
        record.signals = detect_signals(text)
        record.method_family = classify_family(record.signals, record.title)
        record.risk_flags = detect_risks(record.metadata, record.signals)
        record.replay_priority = priority_for(record, target_set)
        record.replay_notes = replay_notes(record)

    return sorted(
        records.values(),
        key=lambda item: (
            {"target": 0, "high": 1, "medium": 2, "low": 3}.get(item.replay_priority, 4),
            item.ranks.get("score_ascending_latest", 999),
            item.ranks.get("vote_top", 999),
            item.ref,
        ),
    )


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact_list(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return ""


def write_csv(records: list[NotebookRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref",
        "title",
        "author",
        "last_run_time",
        "total_votes",
        "source_lists",
        "score_rank",
        "vote_rank",
        "date_rank",
        "known_score",
        "method_family",
        "replay_priority",
        "enable_gpu",
        "enable_internet",
        "machine_shape",
        "dataset_sources",
        "kernel_sources",
        "model_sources",
        "signals",
        "risk_flags",
        "replay_notes",
        "metadata_path",
        "notebook_path",
    ]
    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "ref": record.ref,
                    "title": record.title,
                    "author": record.author,
                    "last_run_time": record.last_run_time,
                    "total_votes": record.total_votes if record.total_votes is not None else "",
                    "source_lists": ";".join(sorted(record.source_lists)),
                    "score_rank": record.ranks.get("score_ascending_latest", ""),
                    "vote_rank": record.ranks.get("vote_top", ""),
                    "date_rank": record.ranks.get("date_run_recent", ""),
                    "known_score": record.known_score,
                    "method_family": record.method_family,
                    "replay_priority": record.replay_priority,
                    "enable_gpu": record.metadata.get("enable_gpu", ""),
                    "enable_internet": record.metadata.get("enable_internet", ""),
                    "machine_shape": record.metadata.get("machine_shape", ""),
                    "dataset_sources": compact_list(record.metadata.get("dataset_sources", [])),
                    "kernel_sources": compact_list(record.metadata.get("kernel_sources", [])),
                    "model_sources": compact_list(record.metadata.get("model_sources", [])),
                    "signals": ";".join(record.signals),
                    "risk_flags": ";".join(record.risk_flags),
                    "replay_notes": record.replay_notes,
                    "metadata_path": rel(record.metadata_path),
                    "notebook_path": rel(record.notebook_path),
                }
            )


def markdown_table(records: list[NotebookRecord], limit: int = 12) -> str:
    lines = [
        "| Priority | Ref | Score | Family | Risks | Replay note |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for record in records[:limit]:
        risks = ", ".join(record.risk_flags) if record.risk_flags else "none"
        score = record.known_score or "-"
        lines.append(
            f"| {record.replay_priority} | `{record.ref}` | {score} | "
            f"{record.method_family} | {risks} | {record.replay_notes} |"
        )
    return "\n".join(lines)


def render_markdown(
    records: list[NotebookRecord],
    as_of: str,
    listing_dirs: tuple[str, ...],
    target_refs: tuple[str, ...],
    self_anchor: str,
    output_csv: Path,
) -> str:
    family_counts = Counter(record.method_family for record in records)
    risk_counts: Counter[str] = Counter()
    for record in records:
        risk_counts.update(record.risk_flags or ["none"])

    target_records = [record for record in records if record.ref in set(target_refs)]
    replay_queue = [
        record
        for record in records
        if record.replay_priority in {"target", "high"}
        and "internet_enabled" not in record.risk_flags
    ]
    first_replay = select_first_replay_candidate(records)
    first_replay_ref = f"`{first_replay.ref}`" if first_replay else "the selected candidate"
    first_replay_exp = suggested_exp_name(first_replay)

    lines = [
        "# Public notebook catch-up after self improvements",
        "",
        f"調査日: {as_of}",
        "",
        "## Context",
        "",
        f"- Self anchor: {self_anchor}",
        "- Trigger: high-priority self improvements have produced a new public LB anchor, "
        "so public top-notebook replay can start as a separate, audited route.",
        f"- Listings scanned: {', '.join(listing_dirs)}",
        f"- Inventory CSV: `{rel(output_csv)}`",
        "",
        "## Refresh commands",
        "",
        "Run these before regenerating this report when network/Kaggle credentials are available:",
        "",
        "```bash",
        "task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction "
        'EXTRA_ARGS="--limit 20 '
        "--output-dir docs/notebooks/rogii-wellbore-geology-prediction/vote_top "
        '--sort-by voteCount --force"',
        "task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction "
        'EXTRA_ARGS="--limit 20 '
        "--output-dir docs/notebooks/rogii-wellbore-geology-prediction/score_ascending_latest "
        '--sort-by scoreAscending --force"',
        "task fetch-kaggle-notebooks COMPETITION=rogii-wellbore-geology-prediction "
        'EXTRA_ARGS="--limit 20 '
        "--output-dir docs/notebooks/rogii-wellbore-geology-prediction/date_run_recent "
        '--sort-by dateRun --force --retries 3"',
        f"uv run python studies/public_notebook_catchup.py --as-of {as_of}",
        "```",
        "",
        "`task`が利用できない環境では、同じ変数と引数で`make fetch-kaggle-notebooks`を使う。",
        "",
        "## Replay Queue",
        "",
        markdown_table(replay_queue, limit=16),
        "",
        "## Target Notebook Inventory",
        "",
        markdown_table(target_records, limit=len(target_records)),
        "",
        "## Family Counts",
        "",
    ]
    for family, count in family_counts.most_common():
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Risk Counts", ""])
    for risk, count in risk_counts.most_common():
        lines.append(f"- `{risk}`: {count}")
    lines.extend(
        [
            "",
            "## Implementation Handoff",
            "",
            f"1. Start with {first_replay_ref} as `{first_replay_exp}` if refreshed metadata "
            "still shows no external dataset/model/kernel dependency.",
            "2. Replay the selected public notebook on Kaggle without code changes first; "
            "record kernel version, output hash, runtime, `submission.csv` checks, and LB.",
            "3. Keep replay output separate from self CV until dependency and hidden-safety "
            "checks pass.",
            "4. Treat `kojimar/*TabICL*` and AeroRidge routes as artifact-stack audits first, "
            "because they require dataset/model input inventory before replay.",
            "",
            "## Notes",
            "",
            "- Kaggle listing metadata does not expose public score directly; `known_score` "
            "is parsed "
            "only from notebook titles.",
            "- `formation_or_geology_boundary_check` is a review flag, not an automatic rejection. "
            "The replay audit must distinguish hidden-safe runtime inputs from train-only leakage.",
            "- Static public CSV blends remain unsafe until rerun on hidden test inside the "
            "submitted "
            "Kaggle notebook.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    listing_dirs = tuple(args.listing_dir) if args.listing_dir else DEFAULT_LISTING_DIRS
    target_refs = tuple(args.target_ref) if args.target_ref else DEFAULT_TARGET_REFS
    output_md = args.output_md or (
        args.notebook_root / f"public_notebook_catchup_after_self_improvements_{args.as_of}.md"
    )
    output_csv = args.output_csv or (
        args.notebook_root / f"public_notebook_catchup_inventory_{args.as_of}.csv"
    )

    records = collect_records(args.notebook_root, listing_dirs, target_refs)
    write_csv(records, output_csv)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        render_markdown(
            records=records,
            as_of=args.as_of,
            listing_dirs=listing_dirs,
            target_refs=target_refs,
            self_anchor=args.self_anchor,
            output_csv=output_csv,
        )
    )
    print(f"Wrote {output_md.relative_to(ROOT)}")
    print(f"Wrote {output_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
