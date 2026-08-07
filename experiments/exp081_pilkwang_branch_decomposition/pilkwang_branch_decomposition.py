from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from settings import ExperimentPaths, load_config
from settings import get_nested as settings_get_nested

DISTANCE_COLUMNS = ["aligned_rows", "rmse", "mae", "max_abs", "mean_diff", "std_diff"]


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value = settings_get_nested(config, dotted_key)
    return default if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose Pilkwang public replay branches from exp079 audit outputs."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to this experiment's artifacts directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(value, fp, indent=2, sort_keys=True)
        fp.write("\n")


def resolve_input_paths(config: dict[str, Any]) -> dict[str, Path]:
    audit = get_nested(config, "audit", {}) or {}
    base_dir = Path(str(audit["exp079_output_dir"]))
    return {
        "summary": base_dir / str(audit["exp079_summary_file"]),
        "submission_summary": base_dir / str(audit["exp079_submission_summary_file"]),
        "pairwise": base_dir / str(audit["exp079_pairwise_file"]),
    }


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            distance = item.get("distance", {})
            row = {"left": item.get("left"), "right": item.get("right")}
            row.update({key: distance.get(key) for key in DISTANCE_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def candidate_id(source_name: str, label: str) -> str:
    return f"{source_name}::{label}"


def short_label(value: str) -> str:
    return value.split("::", 1)[1] if "::" in value else value


def normalize_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "rows",
        "expected_rows",
        "duplicate_id_rows",
        "missing_id_count",
        "extra_id_count",
        "null_prediction_count",
        "prediction_mean",
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "prediction_p01",
        "prediction_p99",
    ]
    result = frame.copy()
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def build_role_map(config: dict[str, Any]) -> dict[str, str]:
    role_map: dict[str, str] = {}
    branch_roles = get_nested(config, "audit.branch_roles", {}) or {}
    for role, labels in branch_roles.items():
        for label in labels or []:
            role_map[str(label)] = str(role)
    return role_map


def infer_role(label: str, role_map: dict[str, str]) -> str:
    if label in role_map:
        return role_map[label]
    if "modelpkg_gated" in label:
        return "modelpkg_tiny_gate"
    if re.search(r"_w0\.\d+\.csv$", label):
        return "blend_weight"
    if "model_package_only" in label:
        return "model_package_only"
    if "pretrained_lgbm" in label:
        return "pretrained_lgbm"
    if "projection" in label or "ridge_pf" in label:
        return "projected_ridge_pf"
    if label == "submission.csv":
        return "final_or_external_submission"
    return "support_or_unclassified"


def parse_blend_weight(label: str) -> float | None:
    match = re.search(r"_w(\d+\.\d+)\.csv$", label)
    return None if match is None else float(match.group(1))


def parse_gate_gmax(label: str) -> float | None:
    match = re.search(r"_gated_(\d{3})\.csv$", label)
    return None if match is None else float(match.group(1)) / 1000.0


def distance_lookup(pairwise: pd.DataFrame) -> dict[frozenset[str], dict[str, Any]]:
    lookup: dict[frozenset[str], dict[str, Any]] = {}
    for row in pairwise.to_dict(orient="records"):
        left = str(row["left"])
        right = str(row["right"])
        lookup[frozenset((left, right))] = {key: row.get(key) for key in DISTANCE_COLUMNS}
    return lookup


def get_distance(
    lookup: dict[frozenset[str], dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    if left == right:
        return {
            "aligned_rows": None,
            "rmse": 0.0,
            "mae": 0.0,
            "max_abs": 0.0,
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }
    return lookup.get(frozenset((left, right)), {})


def load_summary(path: Path) -> dict[str, Any]:
    with path.open() as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def risk_hits_for_source(summary: dict[str, Any], source_name: str) -> dict[str, int]:
    inspections = summary.get("notebook_inspections", [])
    combined: dict[str, int] = {}
    if not isinstance(inspections, list):
        return combined
    source_hint = source_name.split("_", 1)[0]
    for inspection in inspections:
        if not isinstance(inspection, dict):
            continue
        path = str(inspection.get("path", ""))
        if source_hint not in path and source_name not in path:
            continue
        risk_hits = inspection.get("risk_hits", {})
        if not isinstance(risk_hits, dict):
            continue
        for name, value in risk_hits.items():
            combined[name] = combined.get(name, 0) + int(value or 0)
    return combined


def prepare_candidate_table(
    submission_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    audit = get_nested(config, "audit", {}) or {}
    source_name = str(audit["source_name"])
    final = candidate_id(source_name, str(audit["final_label"]))
    role_map = build_role_map(config)
    lookup = distance_lookup(pairwise)

    candidates = normalize_numeric_columns(submission_summary)
    candidates = candidates[candidates["read_error"].fillna("").eq("")]
    candidates = candidates[candidates["source_name"].eq(source_name)].copy()
    candidates["candidate"] = [
        candidate_id(str(source), str(label))
        for source, label in zip(candidates["source_name"], candidates["label"], strict=False)
    ]
    candidates["role"] = [infer_role(str(label), role_map) for label in candidates["label"]]
    candidates["blend_weight"] = [parse_blend_weight(str(label)) for label in candidates["label"]]
    candidates["gate_gmax"] = [parse_gate_gmax(str(label)) for label in candidates["label"]]
    candidates["valid_submission_contract"] = (
        candidates["rows"].eq(candidates["expected_rows"])
        & candidates["duplicate_id_rows"].fillna(1).eq(0)
        & candidates["missing_id_count"].fillna(1).eq(0)
        & candidates["extra_id_count"].fillna(1).eq(0)
        & candidates["null_prediction_count"].fillna(1).eq(0)
    )

    final_distances = [
        get_distance(lookup, final, candidate) for candidate in candidates["candidate"]
    ]
    for column in DISTANCE_COLUMNS:
        candidates[f"vs_final_{column}"] = [row.get(column) for row in final_distances]

    ridge_sp = "ridge_sp_lb_7776::submission.csv"
    ridge_distances = [
        get_distance(lookup, ridge_sp, candidate) for candidate in candidates["candidate"]
    ]
    for column in DISTANCE_COLUMNS:
        candidates[f"vs_ridge_sp_{column}"] = [row.get(column) for row in ridge_distances]

    display_columns = [
        "label",
        "role",
        "valid_submission_contract",
        "rows",
        "prediction_mean",
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "prediction_p01",
        "prediction_p99",
        "sha256",
        "blend_weight",
        "gate_gmax",
        "vs_final_rmse",
        "vs_final_mae",
        "vs_final_max_abs",
        "vs_final_mean_diff",
        "vs_ridge_sp_rmse",
        "vs_ridge_sp_mae",
        "vs_ridge_sp_max_abs",
    ]
    return candidates[display_columns].sort_values(
        ["role", "vs_final_rmse", "label"], na_position="last"
    )


def prepare_anchor_table(
    candidates: pd.DataFrame,
    pairwise: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    audit = get_nested(config, "audit", {}) or {}
    source_name = str(audit["source_name"])
    final = candidate_id(source_name, str(audit["final_label"]))
    lookup = distance_lookup(pairwise)
    rows: list[dict[str, Any]] = []
    labels = [str(value) for value in audit.get("anchor_labels", [])]
    for anchor in labels:
        distance = get_distance(lookup, final, anchor)
        rows.append(
            {
                "anchor": anchor,
                "comparison": f"{final} vs {anchor}",
                "status": "available" if distance else "missing_pairwise",
                **{key: distance.get(key) for key in DISTANCE_COLUMNS},
            }
        )

    available_candidates = set(candidates["label"])
    for label in sorted(available_candidates):
        if label == str(audit["final_label"]):
            continue
        full = candidate_id(source_name, label)
        distance = get_distance(lookup, full, "ridge_sp_lb_7776::submission.csv")
        if distance:
            rows.append(
                {
                    "anchor": "ridge_sp_lb_7776::submission.csv",
                    "comparison": f"{full} vs ridge_sp_lb_7776::submission.csv",
                    "status": "available",
                    **{key: distance.get(key) for key in DISTANCE_COLUMNS},
                }
            )
    return pd.DataFrame(rows)


def prepare_role_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    grouped = candidates.groupby("role", dropna=False)
    return grouped.agg(
        candidate_count=("label", "count"),
        valid_count=("valid_submission_contract", "sum"),
        min_vs_final_rmse=("vs_final_rmse", "min"),
        max_vs_final_rmse=("vs_final_rmse", "max"),
        min_vs_ridge_sp_rmse=("vs_ridge_sp_rmse", "min"),
        max_vs_ridge_sp_rmse=("vs_ridge_sp_rmse", "max"),
        prediction_mean_min=("prediction_mean", "min"),
        prediction_mean_max=("prediction_mean", "max"),
    ).reset_index()


def decision_for_row(row: pd.Series) -> tuple[str, str]:
    label = str(row["label"])
    role = str(row["role"])
    valid = bool(row["valid_submission_contract"])
    vs_final = row.get("vs_final_rmse")
    vs_ridge = row.get("vs_ridge_sp_rmse")
    if not valid:
        return "reject", "invalid submission contract in exp079 summary"
    if label in {"submission.csv", "submission_projected_ridge_pf_pretrained_lgbm_w0.55.csv"}:
        return "hold", "identical or equivalent to Pilkwang final; submit only after risk review"
    if role == "model_package_only":
        return "reject", "large distance from final; diagnostic only"
    if role == "modelpkg_tiny_gate":
        return (
            "hold",
            "tiny movement from final; possible 1-candidate submit only if final is selected",
        )
    if role == "blend_weight" and pd.notna(vs_ridge) and float(vs_ridge) < 2.0:
        return "shortlist", "moves toward ridge-sp while staying close to final"
    if role == "projected_ridge_pf" and pd.notna(vs_ridge) and float(vs_ridge) < 1.7:
        return "shortlist", "closest Pilkwang branch family to ridge-sp in exp079 pairwise"
    if pd.notna(vs_final) and float(vs_final) < 0.15:
        return "hold", "near-duplicate of final; low independent value"
    return "hold", "keep as diagnostic; not enough evidence for direct submit"


def prepare_decision_table(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        decision, reason = decision_for_row(row)
        rows.append(
            {
                "label": row["label"],
                "role": row["role"],
                "decision": decision,
                "reason": reason,
                "vs_final_rmse": row.get("vs_final_rmse"),
                "vs_ridge_sp_rmse": row.get("vs_ridge_sp_rmse"),
                "sha256": row.get("sha256"),
            }
        )
    order = {"shortlist": 0, "hold": 1, "reject": 2}
    decisions = pd.DataFrame(rows)
    decisions["_order"] = decisions["decision"].map(order).fillna(9)
    decisions = decisions.sort_values(
        ["_order", "vs_ridge_sp_rmse", "vs_final_rmse", "label"]
    ).drop(columns=["_order"])

    max_submit = int(get_nested(config, "audit.candidate_policy.max_submit_candidates", 0) or 0)
    decisions["submit_candidate"] = False
    decisions["submit_rank"] = pd.NA
    if max_submit <= 0:
        return decisions

    shortlist = decisions[decisions["decision"] == "shortlist"]
    selected_indices: list[int] = []
    selected_roles: set[str] = set()
    for index, row in shortlist.iterrows():
        role = str(row["role"])
        if role in selected_roles:
            continue
        selected_indices.append(index)
        selected_roles.add(role)
        if len(selected_indices) >= max_submit:
            break
    if len(selected_indices) < max_submit:
        for index in shortlist.index:
            if index in selected_indices:
                continue
            selected_indices.append(index)
            if len(selected_indices) >= max_submit:
                break
    for rank, index in enumerate(selected_indices, start=1):
        decisions.loc[index, "submit_candidate"] = True
        decisions.loc[index, "submit_rank"] = rank
    return decisions


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# exp081 Pilkwang branch decomposition",
        "",
        f"- status: {summary['status']}",
        f"- candidates: {summary['candidate_count']}",
        f"- shortlisted: {summary['shortlist_count']}",
        f"- input exp079 output dir: `{summary['input_dir']}`",
        "",
        "This audit is target-free. It uses exp079 v4 summary and pairwise distance outputs only.",
        (
            "Row-level segment guards are unavailable unless candidate CSV files are "
            "re-mounted or copied."
        ),
        "",
    ]
    (output_dir / "exp081_pilkwang_branch_decomposition_README.md").write_text("\n".join(lines))


def run(output_dir: Path | None = None) -> dict[str, Any]:
    paths = ExperimentPaths()
    config = load_config()
    output_dir = output_dir or paths.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_input_paths(config)
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        summary = {
            "status": "blocked_missing_exp079_outputs",
            "missing_inputs": missing,
            "input_dir": str(Path(str(get_nested(config, "audit.exp079_output_dir")))),
        }
        stable_json_dump(summary, output_dir / "exp081_pilkwang_branch_decomposition_summary.json")
        stable_json_dump(summary, paths.metrics_path)
        write_readme(output_dir, {**summary, "candidate_count": 0, "shortlist_count": 0})
        return summary

    exp079_summary = load_summary(input_paths["summary"])
    submission_summary = pd.read_csv(input_paths["submission_summary"])
    pairwise = read_jsonl(input_paths["pairwise"])

    candidates = prepare_candidate_table(submission_summary, pairwise, config)
    role_summary = prepare_role_summary(candidates)
    anchors = prepare_anchor_table(candidates, pairwise, config)
    decisions = prepare_decision_table(candidates, config)

    prefix = str(get_nested(config, "audit.output_prefix", "exp081_pilkwang_branch_decomposition"))
    candidates.to_csv(output_dir / f"{prefix}_branch_summary.csv", index=False)
    role_summary.to_csv(output_dir / f"{prefix}_role_summary.csv", index=False)
    anchors.to_csv(output_dir / f"{prefix}_anchor_comparison.csv", index=False)
    decisions.to_csv(output_dir / f"{prefix}_candidate_decisions.csv", index=False)

    source_name = str(get_nested(config, "audit.source_name"))
    risk_hits = risk_hits_for_source(exp079_summary, source_name)
    summary = {
        "experiment": "exp081_pilkwang_branch_decomposition",
        "status": "decomposition_completed",
        "input_dir": str(Path(str(get_nested(config, "audit.exp079_output_dir")))),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "candidate_count": int(len(candidates)),
        "valid_candidate_count": int(candidates["valid_submission_contract"].sum()),
        "shortlist_count": int((decisions["decision"] == "shortlist").sum()),
        "submit_candidate_count": int(decisions["submit_candidate"].sum()),
        "held_count": int((decisions["decision"] == "hold").sum()),
        "rejected_count": int((decisions["decision"] == "reject").sum()),
        "risk_hits": risk_hits,
        "missing_required_sources": exp079_summary.get("missing_required_sources", []),
        "row_level_diff_available": False,
        "row_level_diff_note": (
            "exp079 v4 local output contains summary and pairwise distance files, "
            "not the candidate CSV bodies."
        ),
        "top_submit_candidates": decisions[decisions["submit_candidate"]]
        .sort_values("submit_rank")
        .to_dict(orient="records"),
        "top_shortlist": decisions[decisions["decision"] == "shortlist"]
        .head(5)
        .to_dict(orient="records"),
        "outputs": {
            "branch_summary": f"{prefix}_branch_summary.csv",
            "role_summary": f"{prefix}_role_summary.csv",
            "anchor_comparison": f"{prefix}_anchor_comparison.csv",
            "candidate_decisions": f"{prefix}_candidate_decisions.csv",
        },
    }
    stable_json_dump(summary, output_dir / f"{prefix}_summary.json")
    stable_json_dump(
        {
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": "target_free_pairwise_rmse",
            "status": summary["status"],
            "candidate_count": summary["candidate_count"],
            "shortlist_count": summary["shortlist_count"],
            "submit_candidate_count": summary["submit_candidate_count"],
            "input_sha256": summary["input_sha256"],
        },
        paths.metrics_path,
    )
    write_readme(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = None if args.output_dir is None else Path(args.output_dir)
    summary = run(output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
