from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from settings import ExperimentPaths, get_nested, load_config

RISK_PATTERNS = {
    "writes_submission_csv": r"to_csv\([^)]*submission\.csv",
    "reads_submission_csv": r"read_csv\([^)]*submission\.csv",
    "mentions_sample_submission": r"sample_submission",
    "mentions_public_or_visible": r"\b(public|visible)\b",
    "hardcoded_working_submission": r"/kaggle/working/submission\.csv",
    "hardcoded_input_submission": r"/kaggle/input/[^\"']*submission\.csv",
    "exact_match_or_override": r"\b(exact|override|guarded overlap)\b",
}

ASSIGNMENT_KEYS = (
    "SUBMISSION_PROFILE",
    "RUN_GUARDED_OVERLAP_OVERRIDE",
    "GUARDED_OVERRIDE_REF_COL",
    "GUARDED_OVERRIDE_MIN_VALID_PHYS_ROWS",
    "GUARDED_OVERRIDE_MIN_KNOWN_PREFIX_ROWS",
    "GUARDED_OVERRIDE_PREFIX_RMSE_LIMIT",
    "PF_SELECTOR_USE_SAME_WELL_PHYSICAL",
)

GUARD_OUTPUT_NAMES = (
    "exact_match_recovery_summary.csv",
    "exact_match_recovery_report.csv",
    "submission_before_exact_match_recovery.csv",
    "submission_exact_match_recovery.csv",
    "guarded_overlap_override_summary.csv",
    "guarded_overlap_override_report.csv",
    "submission_before_guarded_overlap_override.csv",
    "submission_guarded_overlap_override.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(errors="replace")


def notebook_source(path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = read_text(path)
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError:
        return raw, []

    cells = notebook.get("cells") if isinstance(notebook, dict) else None
    if not isinstance(cells, list):
        return raw, []

    parts: list[str] = []
    simplified_cells: list[dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source_text = "".join(str(part) for part in source)
        else:
            source_text = str(source)
        parts.append(source_text)
        simplified_cells.append(
            {
                "idx": idx,
                "cell_type": cell.get("cell_type"),
                "source": source_text,
            }
        )
    return "\n".join(parts), simplified_cells


def count_risk_patterns(source: str) -> dict[str, int]:
    return {
        key: len(re.findall(pattern, source, flags=re.IGNORECASE))
        for key, pattern in RISK_PATTERNS.items()
    }


def parse_literal_token(value: str) -> Any:
    token = value.strip().split("#", 1)[0].strip()
    if token in {"True", "False"}:
        return token == "True"
    if token in {"None", "null"}:
        return None
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        return token[1:-1]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def parse_assignments(source: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in ASSIGNMENT_KEYS:
        pattern = rf"(?m)^\s*{re.escape(key)}\s*=\s*([^\n]+)"
        matches = re.findall(pattern, source)
        if matches:
            values[key] = parse_literal_token(matches[-1])
    return values


def csv_writes_for_guard_layers(source: str) -> list[str]:
    rows: list[str] = []
    for line in source.splitlines():
        lower = line.lower()
        if "to_csv" in lower and ("guarded" in lower or "exact_match" in lower):
            rows.append(line.strip()[:240])
    return rows


def audit_notebook(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return item
    source, cells = notebook_source(path)
    item.update(
        {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "cell_count": len(cells) if cells else None,
            "line_count": source.count("\n") + 1,
            "risk_hits": count_risk_patterns(source),
            "assignments": parse_assignments(source),
            "guard_csv_writes": csv_writes_for_guard_layers(source),
        }
    )
    return item


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def configured_paths(root: Path, values: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in values:
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        out.append(path)
    return out


def exp079_summary_candidates(root: Path, config: dict[str, Any]) -> list[Path]:
    raw = get_nested(config, "audit.exp079_summary_json_candidates") or []
    return configured_paths(root, list(raw))


def exp079_submission_summary_candidates(root: Path, config: dict[str, Any]) -> list[Path]:
    raw = get_nested(config, "audit.exp079_submission_summary_candidates") or []
    return configured_paths(root, list(raw))


def exp079_pairwise_candidates(root: Path, config: dict[str, Any]) -> list[Path]:
    raw = get_nested(config, "audit.exp079_pairwise_jsonl_candidates") or []
    return configured_paths(root, list(raw))


def exp064_metrics_candidates(root: Path, config: dict[str, Any]) -> list[Path]:
    raw = get_nested(config, "audit.exp064_metrics_candidates") or []
    return configured_paths(root, list(raw))


def notebook_candidates(root: Path, config: dict[str, Any]) -> list[Path]:
    raw = get_nested(config, "audit.notebook_candidates") or []
    return configured_paths(root, list(raw))


def guard_output_roots(root: Path, paths: ExperimentPaths, config: dict[str, Any]) -> list[Path]:
    raw = list(get_nested(config, "audit.guard_output_roots") or [])
    roots = configured_paths(root, raw)
    roots.append(paths.artifacts_dir)
    roots.append(paths.experiment_dir)
    return roots


def inspect_exp079_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False}
    obj = load_json_if_exists(path)
    if obj is None:
        return {"path": str(path), "exists": False}

    source_specs = obj.get("source_specs")
    notebook_inspections = obj.get("notebook_inspections")
    inventories = obj.get("input_inventories")

    expected_checks: list[str] = []
    if isinstance(source_specs, list):
        for spec in source_specs:
            if (
                isinstance(spec, dict)
                and spec.get("name") == "pilkwang_target_free_tvt_geosteering"
            ):
                checks = spec.get("expected_checks") or []
                if isinstance(checks, list):
                    expected_checks.extend(str(check) for check in checks)

    risk_hits: dict[str, int] = {}
    if isinstance(notebook_inspections, list):
        for item in notebook_inspections:
            if not isinstance(item, dict):
                continue
            path_text = str(item.get("path") or "").lower()
            if "pilkwang" not in path_text and "target-free-tvt" not in path_text:
                continue
            raw_hits = item.get("risk_hits") or {}
            if isinstance(raw_hits, dict):
                risk_hits = {str(k): int(v) for k, v in raw_hits.items()}
                break

    inventory_guard_outputs: list[dict[str, Any]] = []
    if isinstance(inventories, list):
        for inventory in inventories:
            if not isinstance(inventory, dict):
                continue
            files = inventory.get("files") or []
            if not isinstance(files, list):
                continue
            for file_info in files:
                if not isinstance(file_info, dict):
                    continue
                rel = str(file_info.get("relative_path") or file_info.get("path") or "")
                if any(name in rel for name in GUARD_OUTPUT_NAMES):
                    inventory_guard_outputs.append(
                        {
                            "relative_path": rel,
                            "sha256": file_info.get("sha256"),
                            "size_bytes": file_info.get("size_bytes"),
                        }
                    )

    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "expected_checks": sorted(set(expected_checks)),
        "expected_exact_disabled": "exact_match_recovery_disabled" in expected_checks,
        "expected_guarded_disabled": "guarded_overlap_override_disabled" in expected_checks,
        "pilkwang_risk_hits": risk_hits,
        "inventory_guard_outputs": inventory_guard_outputs,
    }


def inspect_submission_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None}
    rows = read_csv_rows(path)
    pilkwang_rows = [
        row
        for row in rows
        if row.get("source_name") == "pilkwang_target_free_tvt_geosteering"
    ]
    if pilkwang_rows:
        rows = pilkwang_rows
    by_label = {row.get("label", ""): row for row in rows}
    final = by_label.get("submission.csv")
    base = by_label.get("submission_projected_ridge_pf_pretrained_lgbm_base.csv")
    w055 = by_label.get("submission_projected_ridge_pf_pretrained_lgbm_w0.55.csv")
    final_sha = final.get("sha256") if final else None
    base_sha = base.get("sha256") if base else None
    w055_sha = w055.get("sha256") if w055 else None
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "final_sha256": final_sha,
        "base_sha256": base_sha,
        "w055_sha256": w055_sha,
        "final_equals_base": bool(final_sha and base_sha and final_sha == base_sha),
        "final_equals_w055": bool(final_sha and w055_sha and final_sha == w055_sha),
    }


def inspect_pairwise_jsonl(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None}
    matched: list[dict[str, Any]] = []
    with path.open() as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            labels = {
                str(row.get("left_label") or row.get("left") or "").split("::")[-1],
                str(row.get("right_label") or row.get("right") or "").split("::")[-1],
            }
            if "submission.csv" in labels and any(
                label in labels
                for label in (
                    "submission_projected_ridge_pf_pretrained_lgbm_base.csv",
                    "submission_projected_ridge_pf_pretrained_lgbm_w0.55.csv",
                    "projected_ridge_pf_projection_submission.csv",
                    "pretrained_lgbm_pretrained_submission.csv",
                    "submission_model_package_only.csv",
                )
            ):
                matched.append(row)
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "matched_final_pairwise": matched,
    }


def inspect_exp064_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False}
    obj = load_json_if_exists(path)
    if obj is None:
        return {"path": str(path), "exists": False}
    hidden = obj.get("hidden_code_submission") if isinstance(obj, dict) else {}
    if not isinstance(hidden, dict):
        hidden = {}
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "hidden_status": hidden.get("status"),
        "hidden_ref": hidden.get("ref"),
        "hidden_public_lb": hidden.get("public_lb"),
        "hidden_interpretation": hidden.get("interpretation"),
        "assertion_not_triggered": hidden.get("status") == "complete",
        "probe_summary": get_nested(obj, "kaggle_inference.probe_summary"),
    }


def key_value_csv(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if len(rows) == 1:
        return dict(rows[0])
    if rows and set(rows[0].keys()) == {"", "0"}:
        return {row.get("", ""): row.get("0") for row in rows}
    if rows and len(rows[0]) >= 2:
        keys = list(rows[0].keys())
        if keys[0] and keys[1]:
            return {row.get(keys[0], ""): row.get(keys[1]) for row in rows}
    return {"row_count": len(rows)}


def parse_boolish(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_intish(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def read_submission(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if "id" not in (reader.fieldnames or []) or "tvt" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain id,tvt columns")
        for row in reader:
            values[str(row["id"])] = float(row["tvt"])
    return values


def compare_submissions(before: Path, after: Path) -> dict[str, Any]:
    left = read_submission(before)
    right = read_submission(after)
    common = sorted(set(left) & set(right))
    if not common:
        return {"common_rows": 0}
    diffs = [right[key] - left[key] for key in common]
    abs_diffs = sorted(abs(x) for x in diffs)
    changed = [x for x in abs_diffs if x > 1e-12]
    rmse = math.sqrt(sum(x * x for x in diffs) / len(diffs))
    p95_idx = min(len(abs_diffs) - 1, int(round(0.95 * (len(abs_diffs) - 1))))
    return {
        "common_rows": len(common),
        "missing_left_rows": len(set(right) - set(left)),
        "missing_right_rows": len(set(left) - set(right)),
        "changed_rows": len(changed),
        "rmse_diff": rmse,
        "max_abs_diff": max(abs_diffs),
        "p95_abs_diff": abs_diffs[p95_idx],
        "before_sha256": sha256_file(before),
        "after_sha256": sha256_file(after),
    }


def inspect_guard_outputs(root_paths: list[Path]) -> dict[str, Any]:
    found: dict[str, list[dict[str, Any]]] = {name: [] for name in GUARD_OUTPUT_NAMES}
    for root in root_paths:
        if not root.exists():
            continue
        for name in GUARD_OUTPUT_NAMES:
            for path in root.rglob(name):
                item: dict[str, Any] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                if name.endswith("_summary.csv"):
                    try:
                        item["values"] = key_value_csv(path)
                    except Exception as exc:  # noqa: BLE001
                        item["read_error"] = str(exc)
                found[name].append(item)

    comparisons: dict[str, Any] = {}
    pairs = {
        "exact_match_recovery": (
            "submission_before_exact_match_recovery.csv",
            "submission_exact_match_recovery.csv",
        ),
        "guarded_overlap_override": (
            "submission_before_guarded_overlap_override.csv",
            "submission_guarded_overlap_override.csv",
        ),
    }
    for label, (before_name, after_name) in pairs.items():
        before_items = found.get(before_name) or []
        after_items = found.get(after_name) or []
        if not before_items or not after_items:
            comparisons[label] = {"available": False}
            continue
        before = Path(before_items[0]["path"])
        after = Path(after_items[0]["path"])
        try:
            comparisons[label] = {"available": True, **compare_submissions(before, after)}
        except Exception as exc:  # noqa: BLE001
            comparisons[label] = {"available": True, "read_error": str(exc)}

    compact_found = {name: values for name, values in found.items() if values}
    return {
        "found": compact_found,
        "comparisons": comparisons,
    }


def summarize_guard_firing(guard_outputs: dict[str, Any]) -> dict[str, Any]:
    changed_any = False
    rows_overridden = 0
    summary_rows: list[dict[str, Any]] = []
    for name, items in (guard_outputs.get("found") or {}).items():
        if not name.endswith("_summary.csv"):
            continue
        for item in items:
            values = item.get("values") or {}
            changed = parse_boolish(values.get("changed"))
            if changed is True:
                changed_any = True
            for key in ("rows_overridden", "rows_recovered", "rows_changed"):
                count = parse_intish(values.get(key))
                if count:
                    rows_overridden += count
            summary_rows.append({"name": name, "path": item.get("path"), "values": values})

    for comparison in (guard_outputs.get("comparisons") or {}).values():
        if not isinstance(comparison, dict) or not comparison.get("available"):
            continue
        if int(comparison.get("changed_rows") or 0) > 0:
            changed_any = True

    return {
        "summary_count": len(summary_rows),
        "changed_any": changed_any,
        "rows_overridden_or_recovered": rows_overridden,
        "summaries": summary_rows,
    }


def make_decision(
    notebook_audits: list[dict[str, Any]],
    exp079_summary: dict[str, Any],
    submission_summary: dict[str, Any],
    exp064: dict[str, Any],
    guard_firing: dict[str, Any],
) -> dict[str, Any]:
    source_flags_enabled = any(
        (audit.get("assignments") or {}).get("RUN_GUARDED_OVERLAP_OVERRIDE") is True
        or (audit.get("assignments") or {}).get("PF_SELECTOR_USE_SAME_WELL_PHYSICAL") is True
        for audit in notebook_audits
        if audit.get("exists")
    )
    exp079_expected_disabled = bool(
        exp079_summary.get("expected_exact_disabled")
        or exp079_summary.get("expected_guarded_disabled")
    )
    final_equals_base = bool(submission_summary.get("final_equals_base"))
    hidden_assertion_not_triggered = bool(exp064.get("assertion_not_triggered"))
    guard_changed = bool(guard_firing.get("changed_any"))
    guard_rows = int(guard_firing.get("rows_overridden_or_recovered") or 0)

    if guard_changed or guard_rows > 0:
        status = "diagnostic_fired_do_not_adopt"
        confidence = "medium"
        reason = (
            "Optional exact/override output changed predictions. Treat it as diagnostic "
            "only and exclude same-well override from hidden-safe improvement claims."
        )
    elif final_equals_base and hidden_assertion_not_triggered:
        status = "negative_control_passed_current_evidence"
        confidence = "medium"
        reason = (
            "Pilkwang final equals the archived base branch, and exp064 hidden code "
            "submission did not trigger the exposed well-id overlap assertion."
        )
    elif exp079_expected_disabled and hidden_assertion_not_triggered:
        status = "negative_control_supported_but_incomplete"
        confidence = "low"
        reason = (
            "exp079 expected disabled checks and exp064 support no exposed well-id "
            "overlap, but final/base equality or guard output evidence is incomplete."
        )
    else:
        status = "inconclusive_exclude_from_improvement_claims"
        confidence = "low"
        reason = "Evidence is incomplete, so same-well override remains excluded."

    caveats = [
        (
            "This audit does not rule out the same physical well appearing under a "
            "different anonymized id."
        ),
        "Any optional exact/override layer is diagnostic only and must not be a submit candidate.",
    ]
    if source_flags_enabled and exp079_expected_disabled:
        caveats.append(
            "The archived notebook source contains enabled same-well shortcut flags, "
            "while exp079 source specs expected exact/override-disabled checks; this "
            "conflict is recorded as risk rather than improvement evidence."
        )

    return {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "adoption": "exclude_same_well_exact_or_guarded_override",
        "source_flags_enabled": source_flags_enabled,
        "exp079_expected_disabled": exp079_expected_disabled,
        "final_equals_base": final_equals_base,
        "hidden_assertion_not_triggered": hidden_assertion_not_triggered,
        "guard_changed": guard_changed,
        "guard_rows": guard_rows,
        "caveats": caveats,
    }


def write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def run_audit(
    paths: ExperimentPaths | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = paths or ExperimentPaths()
    config = config or load_config()
    paths.ensure_output_dirs()
    root = paths.root

    notebooks = [audit_notebook(path) for path in notebook_candidates(root, config)]
    exp079_summary = inspect_exp079_summary(first_existing(exp079_summary_candidates(root, config)))
    submission_summary = inspect_submission_summary(
        first_existing(exp079_submission_summary_candidates(root, config))
    )
    pairwise_summary = inspect_pairwise_jsonl(
        first_existing(exp079_pairwise_candidates(root, config))
    )
    exp064 = inspect_exp064_metrics(first_existing(exp064_metrics_candidates(root, config)))
    guard_outputs = inspect_guard_outputs(guard_output_roots(root, paths, config))
    guard_firing = summarize_guard_firing(guard_outputs)
    decision = make_decision(
        notebook_audits=notebooks,
        exp079_summary=exp079_summary,
        submission_summary=submission_summary,
        exp064=exp064,
        guard_firing=guard_firing,
    )

    summary = {
        "experiment": paths.experiment_name,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "route": get_nested(config, "experiment.route"),
        "status": decision["status"],
        "decision": decision,
        "notebooks": notebooks,
        "exp079_summary": exp079_summary,
        "exp079_submission_summary": submission_summary,
        "exp079_pairwise_summary": pairwise_summary,
        "exp064_probe": exp064,
        "guard_outputs": guard_outputs,
        "guard_firing": guard_firing,
    }

    summary_path = paths.artifacts_dir / "exact_override_negative_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    notebook_rows = []
    for item in notebooks:
        notebook_rows.append(
            {
                "path": item.get("path"),
                "exists": item.get("exists"),
                "sha256": item.get("sha256"),
                "exact_match_or_override": (item.get("risk_hits") or {}).get(
                    "exact_match_or_override"
                ),
                "writes_submission_csv": (item.get("risk_hits") or {}).get("writes_submission_csv"),
                "RUN_GUARDED_OVERLAP_OVERRIDE": (item.get("assignments") or {}).get(
                    "RUN_GUARDED_OVERLAP_OVERRIDE"
                ),
                "PF_SELECTOR_USE_SAME_WELL_PHYSICAL": (item.get("assignments") or {}).get(
                    "PF_SELECTOR_USE_SAME_WELL_PHYSICAL"
                ),
            }
        )
    write_csv_dicts(paths.artifacts_dir / "notebook_risk_summary.csv", notebook_rows)

    guard_rows = []
    for name, items in (guard_outputs.get("found") or {}).items():
        for item in items:
            guard_rows.append({"name": name, **item})
    write_csv_dicts(paths.artifacts_dir / "guard_output_inventory.csv", guard_rows)

    metrics = {
        "experiment": paths.experiment_name,
        "status": decision["status"],
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "diagnostic_status",
        "route": get_nested(config, "experiment.route"),
        "key_idea": "exact-match / guarded overlap override negative control",
        "decision": decision,
        "artifacts": {
            "summary": str(summary_path),
            "notebook_risk_summary": str(paths.artifacts_dir / "notebook_risk_summary.csv"),
            "guard_output_inventory": str(paths.artifacts_dir / "guard_output_inventory.csv"),
        },
        "updated_at": summary["created_at"],
    }
    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact/override negative-control audit.")
    parser.add_argument("--summary", action="store_true", help="Print compact JSON summary")
    args = parser.parse_args()
    summary = run_audit()
    if args.summary:
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "decision": summary["decision"],
                    "summary_path": str(
                        ExperimentPaths().artifacts_dir
                        / "exact_override_negative_control_summary.json"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
