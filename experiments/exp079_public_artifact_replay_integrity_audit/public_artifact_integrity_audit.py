from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
SUBMISSION_NAME_PATTERNS = (
    "submission",
    "projected",
    "ridge",
    "lgbm",
    "modelpkg",
    "gate",
)
RISK_PATTERNS = {
    "writes_submission_csv": r"to_csv\([^)]*submission\.csv",
    "reads_submission_csv": r"read_csv\([^)]*submission\.csv",
    "mentions_sample_submission": r"sample_submission",
    "mentions_public_or_visible": r"\b(public|visible)\b",
    "hardcoded_working_submission": r"/kaggle/working/submission\.csv",
    "hardcoded_input_submission": r"/kaggle/input/[^\"']*submission\.csv",
    "exact_match_or_override": r"\b(exact|override|guarded overlap)\b",
}


@dataclass(frozen=True)
class CsvCandidate:
    label: str
    path: Path
    source_name: str


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def sha256_file(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else open
    with opener(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(value, fp, indent=2, sort_keys=True)
        fp.write("\n")


def resolve_path(value: str | Path, root: Path) -> Path:
    path_text = str(value)
    if path_text.startswith("kaggle://"):
        relative = path_text.removeprefix("kaggle://").strip("/")
        return resolve_kaggle_slug_path(relative)
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def resolve_kaggle_slug_path(slug: str) -> Path:
    normalized = slug.strip("/")
    direct = KAGGLE_INPUT_ROOT / normalized
    if direct.exists():
        return direct
    if "/" in normalized:
        dataset_style = KAGGLE_INPUT_ROOT / "datasets" / normalized
        if dataset_style.exists():
            return dataset_style
        kernel_style = KAGGLE_INPUT_ROOT / "kernels" / normalized
        if kernel_style.exists():
            return kernel_style
    if KAGGLE_INPUT_ROOT.exists():
        leaf = Path(normalized).name
        matches = sorted(
            path
            for path in KAGGLE_INPUT_ROOT.rglob(leaf)
            if path.is_dir() and path.name == leaf
        )
        if matches:
            return matches[0]
    return direct


def list_kaggle_input_tree(max_dirs: int = 200) -> list[str]:
    if not KAGGLE_INPUT_ROOT.exists():
        return []
    dirs = [KAGGLE_INPUT_ROOT]
    dirs.extend(path for path in sorted(KAGGLE_INPUT_ROOT.rglob("*")) if path.is_dir())
    return [str(path) for path in dirs[:max_dirs]]


def resolve_existing_sample_submission(configured_path: Path) -> Path:
    if configured_path.exists():
        return configured_path
    if KAGGLE_INPUT_ROOT.exists():
        candidates = sorted(KAGGLE_INPUT_ROOT.rglob("sample_submission.csv"))
        if candidates:
            competition_candidates = [
                path
                for path in candidates
                if "rogii-wellbore-geology-prediction" in str(path)
            ]
            if competition_candidates:
                return competition_candidates[0]
            return candidates[0]
    return configured_path


def slug_path(slug: str) -> Path:
    return resolve_kaggle_slug_path(slug)


def find_existing_paths(values: list[str], root: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = resolve_path(value, root)
        if path.exists():
            paths.append(path)
    return paths


def inventory_path(path: Path, max_files: int = 5000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    rows: list[dict[str, Any]] = []
    for file_path in files[:max_files]:
        relative = str(file_path.relative_to(path)) if path.is_dir() else file_path.name
        raw_sha = sha256_file(file_path)
        row: dict[str, Any] = {
            "path": str(file_path),
            "relative_path": relative,
            "suffix": file_path.suffix.lower(),
            "size_bytes": int(file_path.stat().st_size),
            "sha256": raw_sha,
        }
        if file_path.suffix.lower() == ".gz":
            row["decompressed_sha256"] = sha256_file(file_path, decompress_gzip=True)
        rows.append(row)
    return rows


def notebook_text_and_metadata(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        with path.open() as fp:
            notebook = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        return "", {"parse_error": str(exc)}

    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.extend(str(part) for part in source)
        else:
            chunks.append(str(source))
    metadata = notebook.get("metadata", {})
    return "\n".join(chunks), metadata if isinstance(metadata, dict) else {}


def inspect_notebooks(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for base_path in paths:
        notebook_paths = (
            [base_path] if base_path.suffix == ".ipynb" else sorted(base_path.rglob("*.ipynb"))
        )
        for notebook_path in notebook_paths:
            text, metadata = notebook_text_and_metadata(notebook_path)
            lower_text = text.lower()
            risk_hits = {
                name: len(re.findall(pattern, lower_text, flags=re.IGNORECASE))
                for name, pattern in RISK_PATTERNS.items()
            }
            input_refs = sorted(set(re.findall(r"/kaggle/input/([A-Za-z0-9_.\-/]+)", text)))
            csv_writes = sorted(set(re.findall(r"to_csv\(([^)]*\.csv[^)]*)\)", text)))
            records.append(
                {
                    "path": str(notebook_path),
                    "sha256": sha256_file(notebook_path),
                    "metadata_keys": sorted(metadata.keys()),
                    "kaggle_metadata": metadata.get("kaggle", {}),
                    "input_refs": input_refs,
                    "csv_writes": csv_writes,
                    "risk_hits": risk_hits,
                    "line_count": int(text.count("\n") + 1) if text else 0,
                }
            )
    return records


def looks_like_submission_csv(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in {".csv", ".gz"} and not name.endswith(".csv.gz"):
        return False
    return any(pattern in name for pattern in SUBMISSION_NAME_PATTERNS)


def find_candidate_csvs(source_specs: list[dict[str, Any]], root: Path) -> list[CsvCandidate]:
    candidates: list[CsvCandidate] = []
    seen: set[Path] = set()
    for source in source_specs:
        source_name = str(source.get("name", "source"))
        for file_value in source.get("branch_files", []) or []:
            for base in source_paths(source, root):
                candidate = base / str(file_value)
                if candidate.exists() and candidate not in seen:
                    candidates.append(CsvCandidate(str(file_value), candidate, source_name))
                    seen.add(candidate)
        for base in source_paths(source, root):
            if not base.exists():
                continue
            for file_path in sorted(base.rglob("*")):
                if (
                    file_path.is_file()
                    and looks_like_submission_csv(file_path)
                    and file_path not in seen
                ):
                    candidates.append(CsvCandidate(file_path.name, file_path, source_name))
                    seen.add(file_path)
    return candidates


def source_paths(source: dict[str, Any], root: Path) -> list[Path]:
    values = list(source.get("source_paths", []) or [])
    values.extend(f"kaggle://{slug}" for slug in source.get("kernel_output_slugs", []) or [])
    values.extend(f"kaggle://{slug}" for slug in source.get("external_input_slugs", []) or [])
    return [resolve_path(value, root) for value in values]


def read_submission_like(path: Path, id_column: str, target_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise ValueError(f"{path} does not contain id column {id_column!r}")
    if target_column not in frame.columns:
        non_id = [column for column in frame.columns if column != id_column]
        if len(non_id) != 1:
            raise ValueError(f"{path} does not contain target column {target_column!r}")
        frame = frame.rename(columns={non_id[0]: target_column})
    return frame[[id_column, target_column]].copy()


def summarize_submission(
    candidate: CsvCandidate,
    sample: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    try:
        frame = read_submission_like(candidate.path, id_column, target_column)
    except Exception as exc:  # noqa: BLE001
        return (
            {
                "label": candidate.label,
                "source_name": candidate.source_name,
                "path": str(candidate.path),
                "read_error": str(exc),
            },
            None,
        )

    sample_ids = set(sample[id_column].astype(str))
    ids = frame[id_column].astype(str)
    values = pd.to_numeric(frame[target_column], errors="coerce")
    missing_ids = sample_ids - set(ids)
    extra_ids = set(ids) - sample_ids
    duplicate_rows = int(ids.duplicated().sum())
    record = {
        "label": candidate.label,
        "source_name": candidate.source_name,
        "path": str(candidate.path),
        "rows": int(len(frame)),
        "expected_rows": int(len(sample)),
        "duplicate_id_rows": duplicate_rows,
        "missing_id_count": int(len(missing_ids)),
        "extra_id_count": int(len(extra_ids)),
        "null_prediction_count": int(values.isna().sum()),
        "prediction_mean": float(values.mean()) if len(values) else None,
        "prediction_std": float(values.std()) if len(values) else None,
        "prediction_min": float(values.min()) if len(values) else None,
        "prediction_max": float(values.max()) if len(values) else None,
        "prediction_p01": float(values.quantile(0.01)) if len(values) else None,
        "prediction_p99": float(values.quantile(0.99)) if len(values) else None,
        "sha256": sha256_file(candidate.path),
    }
    if str(candidate.path).endswith(".gz"):
        record["decompressed_sha256"] = sha256_file(candidate.path, decompress_gzip=True)
    return record, frame


def pairwise_distance(
    left: pd.DataFrame,
    right: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> dict[str, Any]:
    merged = left.merge(right, on=id_column, suffixes=("_left", "_right"))
    if merged.empty:
        return {"aligned_rows": 0}
    diff = (
        pd.to_numeric(merged[f"{target_column}_left"], errors="coerce")
        - pd.to_numeric(merged[f"{target_column}_right"], errors="coerce")
    )
    return {
        "aligned_rows": int(len(merged)),
        "rmse": float(np.sqrt(np.nanmean(np.square(diff)))),
        "mae": float(np.nanmean(np.abs(diff))),
        "max_abs": float(np.nanmax(np.abs(diff))),
        "mean_diff": float(np.nanmean(diff)),
        "std_diff": float(np.nanstd(diff)),
    }


def load_anchor_predictions(
    anchor_specs: list[dict[str, Any]],
    root: Path,
    id_column: str,
    target_column: str,
) -> dict[str, pd.DataFrame]:
    anchors: dict[str, pd.DataFrame] = {}
    for spec in anchor_specs:
        name = str(spec.get("name", spec.get("path", "anchor")))
        path = resolve_path(str(spec.get("path")), root)
        if not path.exists():
            continue
        anchors[name] = read_submission_like(path, id_column, target_column)
    return anchors


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Public Artifact Replay Integrity Audit",
        "",
        f"- Status: {summary['status']}",
        f"- Candidate CSVs: {len(summary['submission_summaries'])}",
        f"- Notebook files inspected: {len(summary['notebook_inspections'])}",
        f"- Missing required sources: {len(summary['missing_required_sources'])}",
        "",
        "## Missing Required Sources",
        "",
    ]
    if summary["missing_required_sources"]:
        for item in summary["missing_required_sources"]:
            lines.append(f"- {item['source_name']}: {item['slug']} -> {item['path']}")
    else:
        lines.append("- none")

    lines.extend(["", "## Candidate Submissions", ""])
    for record in summary["submission_summaries"]:
        if record.get("read_error"):
            lines.append(f"- {record['label']}: read error: {record['read_error']}")
            continue
        lines.append(
            "- "
            f"{record['label']}: rows={record['rows']} "
            f"missing_ids={record['missing_id_count']} "
            f"extra_ids={record['extra_id_count']} "
            f"range=[{record['prediction_min']:.6f}, {record['prediction_max']:.6f}]"
        )

    lines.extend(["", "## Pairwise Distances", ""])
    if summary["pairwise_distances"]:
        for record in summary["pairwise_distances"]:
            distance = record["distance"]
            lines.append(
                "- "
                f"{record['left']} vs {record['right']}: "
                f"rows={distance.get('aligned_rows', 0)} "
                f"rmse={distance.get('rmse')}"
            )
    else:
        lines.append("- none")

    path.write_text("\n".join(lines) + "\n")


def run_integrity_audit(
    *,
    config: dict[str, Any],
    root: Path,
    artifacts_dir: Path,
) -> dict[str, Any]:
    audit_config = config.get("audit", {})
    data_config = config.get("data", {})
    output_prefix = str(audit_config.get("output_prefix", "public_artifact_integrity_audit"))
    id_column = str(data_config.get("id_column", "id"))
    target_column = str(data_config.get("submission_target_column", "tvt"))
    sample_path = resolve_path(
        str(data_config.get("sample_submission", "data/raw/sample_submission.csv")),
        root,
    )
    sample_path = resolve_existing_sample_submission(sample_path)
    sample = pd.read_csv(sample_path)
    if id_column not in sample.columns:
        raise ValueError(f"sample submission does not contain {id_column!r}: {sample_path}")

    source_specs = audit_config.get("source_specs", []) or []
    missing_required_sources: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    for source in source_specs:
        source_name = str(source.get("name", "source"))
        for slug in source.get("required_input_slugs", []) or []:
            path = slug_path(str(slug))
            if not path.exists():
                missing_required_sources.append(
                    {"source_name": source_name, "slug": str(slug), "path": str(path)}
                )
        for path in source_paths(source, root):
            inventories.append(
                {
                    "source_name": source_name,
                    "path": str(path),
                    "exists": path.exists(),
                    "files": inventory_path(
                        path, max_files=int(audit_config.get("max_inventory_files", 5000))
                    ),
                }
            )

    notebook_paths: list[Path] = []
    for source in source_specs:
        notebook_paths.extend(find_existing_paths(source.get("notebook_paths", []) or [], root))
        notebook_paths.extend(path for path in source_paths(source, root) if path.exists())

    notebook_inspections = inspect_notebooks(sorted(set(notebook_paths)))
    candidates = find_candidate_csvs(source_specs, root)
    submission_summaries: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        record, frame = summarize_submission(candidate, sample, id_column, target_column)
        submission_summaries.append(record)
        if frame is not None and not record.get("read_error"):
            frames[f"{candidate.source_name}::{candidate.label}"] = frame

    anchor_frames = load_anchor_predictions(
        audit_config.get("anchor_submission_paths", []) or [],
        root,
        id_column,
        target_column,
    )
    pairwise_distances: list[dict[str, Any]] = []
    for label, frame in frames.items():
        for anchor_label, anchor_frame in anchor_frames.items():
            pairwise_distances.append(
                {
                    "left": label,
                    "right": anchor_label,
                    "distance": pairwise_distance(frame, anchor_frame, id_column, target_column),
                }
            )

    labels = list(frames)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            pairwise_distances.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "distance": pairwise_distance(
                        frames[left_label], frames[right_label], id_column, target_column
                    ),
                }
            )

    status = "audit_completed"
    if missing_required_sources:
        status = "blocked_missing_required_sources"
    elif not submission_summaries:
        status = "blocked_no_candidate_submissions"

    summary = {
        "experiment": get_nested(config, "experiment.name", output_prefix),
        "status": status,
        "sample_submission": str(sample_path),
        "kaggle_input_tree": list_kaggle_input_tree(),
        "source_specs": source_specs,
        "missing_required_sources": missing_required_sources,
        "input_inventories": inventories,
        "notebook_inspections": notebook_inspections,
        "submission_summaries": submission_summaries,
        "pairwise_distances": pairwise_distances,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stable_json_dump(summary, artifacts_dir / f"{output_prefix}_summary.json")
    pd.DataFrame(submission_summaries).to_csv(
        artifacts_dir / f"{output_prefix}_submission_summary.csv", index=False
    )
    pd.DataFrame(pairwise_distances).to_json(
        artifacts_dir / f"{output_prefix}_pairwise_distances.jsonl",
        orient="records",
        lines=True,
    )
    write_markdown_report(summary, artifacts_dir / f"{output_prefix}_README.md")
    return summary
