from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import EXPERIMENT_NAME, ExperimentPaths, load_config

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PREFIX_DEFAULT = "tabicl_artifact_diversity_audit"


@dataclass(frozen=True)
class SubmissionSpec:
    name: str
    path: Path
    role: str
    family: str
    source_name: str
    required: bool = False


def stable_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(value, fp, indent=2, sort_keys=True)
        fp.write("\n")


def sha256_file(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else open
    with opener(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_kaggle_path(value: str) -> Path:
    normalized = value.removeprefix("kaggle://").strip("/")
    candidates = [KAGGLE_INPUT_ROOT / normalized]
    if normalized.startswith(("datasets/", "kernels/", "models/")):
        candidates.append(KAGGLE_INPUT_ROOT / normalized.split("/", 1)[1])
    elif "/" in normalized:
        candidates.extend(
            [
                KAGGLE_INPUT_ROOT / "datasets" / normalized,
                KAGGLE_INPUT_ROOT / "kernels" / normalized,
                KAGGLE_INPUT_ROOT / "models" / normalized,
                KAGGLE_INPUT_ROOT / Path(normalized).name,
            ]
        )
    if KAGGLE_INPUT_ROOT.exists():
        leaf = Path(normalized).name
        candidates.extend(
            path
            for path in sorted(KAGGLE_INPUT_ROOT.rglob(leaf))
            if path.name == leaf
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_path(value: str | Path, root: Path) -> Path:
    text = str(value)
    if text.startswith("kaggle://"):
        return resolve_kaggle_path(text)
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def has_candidate_keyword(path: Path, keywords: list[str]) -> bool:
    lower = str(path.name).lower()
    return any(keyword.lower() in lower for keyword in keywords)


def looks_like_csv(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") or name.endswith(".csv.gz")


def discover_source_specs(
    config: dict[str, Any],
    root: Path,
    *,
    artifact_rows: list[dict[str, Any]],
) -> list[SubmissionSpec]:
    audit_config = config.get("audit", {})
    keywords = list(audit_config.get("candidate_name_keywords") or [])
    max_files = int(audit_config.get("max_files_per_source") or 3000)
    max_candidates = int(audit_config.get("max_candidates_per_source") or 80)

    specs: list[SubmissionSpec] = []
    for source in audit_config.get("source_roots", []) or []:
        source_name = str(source.get("name", "source"))
        role = str(source.get("role", "candidate_source"))
        family = str(source.get("family", "unknown"))
        seen_paths: set[Path] = set()
        found_count = 0
        source_paths = [resolve_path(value, root) for value in source.get("paths", []) or []]
        if not source_paths:
            artifact_rows.append(
                {
                    "record_type": "source_root",
                    "source_name": source_name,
                    "family": family,
                    "role": role,
                    "path": "",
                    "status": "missing_no_paths_configured",
                }
            )
            continue
        for source_path in source_paths:
            artifact_rows.append(
                {
                    "record_type": "source_root",
                    "source_name": source_name,
                    "family": family,
                    "role": role,
                    "path": str(source_path),
                    "status": "found" if source_path.exists() else "missing",
                }
            )
            if not source_path.exists():
                continue
            files = [source_path] if source_path.is_file() else sorted(source_path.rglob("*"))
            for file_path in (path for path in files if path.is_file()):
                if found_count >= max_files:
                    break
                found_count += 1
                if not looks_like_csv(file_path):
                    continue
                if keywords and not has_candidate_keyword(file_path, keywords):
                    continue
                if file_path in seen_paths:
                    continue
                seen_paths.add(file_path)
                label = f"{source_name}__{len(specs):03d}__{file_path.stem.replace('.', '_')}"
                specs.append(
                    SubmissionSpec(
                        name=label,
                        path=file_path,
                        role="candidate" if role != "reference_source" else "reference",
                        family=family,
                        source_name=source_name,
                        required=False,
                    )
                )
                if len(seen_paths) >= max_candidates:
                    break
    return specs


def explicit_submission_specs(config: dict[str, Any], root: Path) -> list[SubmissionSpec]:
    specs: list[SubmissionSpec] = []
    for item in config.get("audit", {}).get("explicit_submissions", []) or []:
        specs.append(
            SubmissionSpec(
                name=str(item["name"]),
                path=resolve_path(item["path"], root),
                role=str(item.get("role", "anchor")),
                family=str(item.get("family", "unknown")),
                source_name=str(item.get("source_name", item["name"])),
                required=bool(item.get("required", False)),
            )
        )
    return specs


def read_submission(
    spec: SubmissionSpec,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    row: dict[str, Any] = {
        "record_type": "submission",
        "name": spec.name,
        "source_name": spec.source_name,
        "family": spec.family,
        "role": spec.role,
        "path": str(spec.path),
        "required": bool(spec.required),
        "status": "missing",
    }
    if not spec.path.exists():
        if spec.required:
            row["status"] = "missing_required"
        return None, row

    row["size_bytes"] = int(spec.path.stat().st_size)
    row["sha256"] = sha256_file(spec.path)
    if spec.path.name.lower().endswith(".gz"):
        row["decompressed_sha256"] = sha256_file(spec.path, decompress_gzip=True)
    try:
        frame = pd.read_csv(spec.path)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "read_error"
        row["error"] = str(exc)
        return None, row

    row["columns"] = list(frame.columns)
    row["rows"] = int(len(frame))
    if "id" not in frame.columns:
        row["status"] = "invalid_missing_id"
        return None, row
    value_col = "tvt" if "tvt" in frame.columns else "TVT" if "TVT" in frame.columns else None
    if value_col is None:
        row["status"] = "invalid_missing_tvt"
        return None, row

    frame = frame[["id", value_col]].rename(columns={value_col: "tvt"}).copy()
    frame["tvt"] = pd.to_numeric(frame["tvt"], errors="coerce")
    if frame["id"].duplicated().any():
        row["status"] = "invalid_duplicate_id"
        row["duplicate_ids"] = int(frame["id"].duplicated().sum())
        return None, row
    if frame["tvt"].isna().any() or not np.isfinite(frame["tvt"].to_numpy(dtype=float)).all():
        row["status"] = "invalid_nonfinite_tvt"
        row["nonfinite_tvt"] = int(frame["tvt"].isna().sum())
        return None, row
    sample_ids = sample["id"].astype(str)
    frame["id"] = frame["id"].astype(str)
    row["id_order_matches_sample"] = bool(frame["id"].tolist() == sample_ids.tolist())
    if len(frame) != len(sample):
        row["status"] = "invalid_row_count"
        return None, row
    if set(frame["id"]) != set(sample_ids):
        row["status"] = "invalid_id_set"
        return None, row
    if not row["id_order_matches_sample"]:
        frame = sample[["id"]].merge(frame, on="id", how="left")
        if frame["tvt"].isna().any():
            row["status"] = "invalid_id_alignment"
            return None, row

    values = frame["tvt"].to_numpy(dtype=float)
    row.update(
        {
            "status": "valid_submission",
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p50": float(np.quantile(values, 0.50)),
            "p99": float(np.quantile(values, 0.99)),
        }
    )
    return frame, row


def distance_metrics(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    diff = left["tvt"].to_numpy(dtype=float) - right["tvt"].to_numpy(dtype=float)
    abs_diff = np.abs(diff)
    return {
        "rows": int(len(diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mae": float(np.mean(abs_diff)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "max_abs": float(np.max(abs_diff)),
        "p50_abs": float(np.quantile(abs_diff, 0.50)),
        "p90_abs": float(np.quantile(abs_diff, 0.90)),
        "p95_abs": float(np.quantile(abs_diff, 0.95)),
        "p99_abs": float(np.quantile(abs_diff, 0.99)),
        "count_abs_gt_1": int(np.sum(abs_diff > 1.0)),
        "count_abs_gt_2": int(np.sum(abs_diff > 2.0)),
        "count_abs_gt_5": int(np.sum(abs_diff > 5.0)),
        "count_abs_gt_10": int(np.sum(abs_diff > 10.0)),
    }


def flatten_distance(
    left: str,
    right: str,
    pair_role: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"left": left, "right": right, "pair_role": pair_role, **metrics}


def build_pairwise(
    frames: dict[str, pd.DataFrame],
    specs: dict[str, SubmissionSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_names = sorted(
        name for name, spec in specs.items() if spec.role in {"candidate", "reference"}
    )
    anchor_names = sorted(name for name, spec in specs.items() if spec.role == "anchor")

    pairwise_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    sample_frame = next(iter(frames.values())) if frames else pd.DataFrame()
    well_ids = (
        sample_frame["id"].astype(str).str.split("_", n=1).str[0]
        if not sample_frame.empty
        else []
    )

    for left_name in candidate_names:
        for right_name in anchor_names:
            pairwise_rows.append(
                flatten_distance(
                    left_name,
                    right_name,
                    "candidate_vs_anchor",
                    distance_metrics(frames[left_name], frames[right_name]),
                )
            )
            by_well_rows.extend(
                by_well_distance_rows(
                    left_name,
                    right_name,
                    frames[left_name],
                    frames[right_name],
                    well_ids,
                )
            )

    for left_index, left_name in enumerate(candidate_names):
        for right_name in candidate_names[left_index + 1 :]:
            pairwise_rows.append(
                flatten_distance(
                    left_name,
                    right_name,
                    "candidate_vs_candidate",
                    distance_metrics(frames[left_name], frames[right_name]),
                )
            )
            by_well_rows.extend(
                by_well_distance_rows(
                    left_name,
                    right_name,
                    frames[left_name],
                    frames[right_name],
                    well_ids,
                )
            )

    return pd.DataFrame(pairwise_rows), pd.DataFrame(by_well_rows)


def by_well_distance_rows(
    left_name: str,
    right_name: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    well_ids: pd.Series,
) -> list[dict[str, Any]]:
    diff = left["tvt"].to_numpy(dtype=float) - right["tvt"].to_numpy(dtype=float)
    frame = pd.DataFrame({"well_id": well_ids, "diff": diff})
    rows: list[dict[str, Any]] = []
    for well_id, group in frame.groupby("well_id", sort=True):
        values = group["diff"].to_numpy(dtype=float)
        abs_values = np.abs(values)
        rows.append(
            {
                "left": left_name,
                "right": right_name,
                "well_id": well_id,
                "rows": int(len(values)),
                "rmse": float(np.sqrt(np.mean(values * values))),
                "mae": float(np.mean(abs_values)),
                "mean_diff": float(np.mean(values)),
                "max_abs": float(np.max(abs_values)),
                "p95_abs": float(np.quantile(abs_values, 0.95)),
            }
        )
    return rows


def write_report(
    summary: dict[str, Any],
    inventory: pd.DataFrame,
    pairwise: pd.DataFrame,
    path: Path,
) -> None:
    valid_inventory = inventory[inventory.get("status", "") == "valid_submission"]
    source_inventory = inventory[inventory.get("record_type", "") == "source_root"]
    lines = [
        "# TabICL Artifact Diversity Audit",
        "",
        f"- Status: {summary['status']}",
        f"- Valid submissions: {summary['valid_submission_count']}",
        f"- Candidate/reference submissions: {summary['candidate_count']}",
        f"- Anchor submissions: {summary['anchor_count']}",
        f"- Pairwise rows: {summary['pairwise_count']}",
        "",
        "## Source Roots",
        "",
    ]
    if source_inventory.empty:
        lines.append("- No source roots configured.")
    else:
        for row in source_inventory.to_dict("records"):
            lines.append(
                f"- {row['source_name']}: {row['status']} `{row['path']}`"
            )

    lines.extend(["", "## Valid Submissions", ""])
    if valid_inventory.empty:
        lines.append("- None. Mount candidate or anchor CSVs and rerun the audit.")
    else:
        for row in valid_inventory.to_dict("records"):
            lines.append(
                "- "
                f"{row['name']} ({row['role']}, {row['family']}): "
                f"rows={row['rows']} sha={row.get('sha256', '')} "
                f"range=[{row['min']:.6f}, {row['max']:.6f}]"
            )

    lines.extend(["", "## Closest Candidate-Anchors", ""])
    candidate_anchor = (
        pairwise[pairwise["pair_role"] == "candidate_vs_anchor"]
        if "pair_role" in pairwise.columns
        else pd.DataFrame()
    )
    if candidate_anchor.empty:
        lines.append("- No candidate-vs-anchor distances were available.")
    else:
        for row in candidate_anchor.sort_values("rmse").head(20).to_dict("records"):
            lines.append(
                "- "
                f"{row['left']} vs {row['right']}: "
                f"rmse={row['rmse']:.9f} p95_abs={row['p95_abs']:.9f} "
                f"max_abs={row['max_abs']:.9f}"
            )
    path.write_text("\n".join(lines) + "\n")


def run_audit() -> dict[str, Any]:
    config = load_config()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    audit_config = config.get("audit", {})
    output_prefix = str(audit_config.get("output_prefix") or OUTPUT_PREFIX_DEFAULT)
    sample = pd.read_csv(paths.sample_submission_path)
    if list(sample.columns)[:2] != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")
    sample = sample[["id", "tvt"]].copy()
    sample["id"] = sample["id"].astype(str)

    inventory_rows: list[dict[str, Any]] = []
    specs = [
        *explicit_submission_specs(config, paths.root),
        *discover_source_specs(config, paths.root, artifact_rows=inventory_rows),
    ]

    frames: dict[str, pd.DataFrame] = {}
    valid_specs: dict[str, SubmissionSpec] = {}
    for spec in specs:
        frame, row = read_submission(spec, sample)
        inventory_rows.append(row)
        if frame is not None:
            frames[spec.name] = frame
            valid_specs[spec.name] = spec

    inventory = pd.DataFrame(inventory_rows)
    pairwise, by_well = build_pairwise(frames, valid_specs)

    anchor_count = sum(1 for spec in valid_specs.values() if spec.role == "anchor")
    candidate_count = sum(
        1 for spec in valid_specs.values() if spec.role in {"candidate", "reference"}
    )
    status = (
        "audit_completed"
        if anchor_count and candidate_count
        else "audit_completed_no_candidates"
        if anchor_count
        else "audit_completed_no_anchors"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "sample_submission": str(paths.sample_submission_path),
        "sample_rows": int(len(sample)),
        "valid_submission_count": int(len(valid_specs)),
        "candidate_count": int(candidate_count),
        "anchor_count": int(anchor_count),
        "pairwise_count": int(len(pairwise)),
        "by_well_pairwise_count": int(len(by_well)),
        "output_prefix": output_prefix,
        "gpu_required": False,
        "tabicl_rerun_performed": False,
        "submission_candidate_created": False,
    }

    inventory_path = paths.artifacts_dir / f"{output_prefix}_inventory.csv"
    pairwise_path = paths.artifacts_dir / f"{output_prefix}_pairwise.csv"
    by_well_path = paths.artifacts_dir / f"{output_prefix}_by_well_distance.csv"
    oof_path = paths.artifacts_dir / f"{output_prefix}_oof_error_correlation.csv"
    summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"
    readme_path = paths.artifacts_dir / f"{output_prefix}_README.md"

    inventory.to_csv(inventory_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    pd.DataFrame(
        [
            {
                "status": "skipped_no_fold_safe_oof_candidates_configured",
                "note": "This CPU audit found submission-style test predictions only.",
            }
        ]
    ).to_csv(oof_path, index=False)
    stable_json_dump(summary, summary_path)
    write_report(summary, inventory, pairwise, readme_path)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "target_free_pairwise_rmse",
        "valid_submission_count": summary["valid_submission_count"],
        "candidate_count": summary["candidate_count"],
        "anchor_count": summary["anchor_count"],
        "pairwise_count": summary["pairwise_count"],
        "gpu_required": False,
        "tabicl_rerun_performed": False,
        "submission_candidate_created": False,
        "artifacts": {
            "inventory": str(inventory_path),
            "pairwise": str(pairwise_path),
            "by_well_distance": str(by_well_path),
            "oof_error_correlation": str(oof_path),
            "summary": str(summary_path),
            "readme": str(readme_path),
        },
    }
    stable_json_dump(metrics, paths.metrics_path)
    return metrics


def main() -> None:
    metrics = run_audit()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
