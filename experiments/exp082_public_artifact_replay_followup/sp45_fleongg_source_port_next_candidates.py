from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    output_path: Path
    source_path: Path
    role: str
    source_notebook: str
    required_inputs: tuple[str, ...]
    source_port_plan: str


@dataclass(frozen=True)
class AnchorSpec:
    name: str
    path: Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
OUTPUT_PREFIX = "sp45_fleongg_source_port_next_candidates"

SOURCE_CHECK_ROOT = Path("/tmp/kaggle-output/source-check")

FLE3N_SOURCE = (
    ROOT
    / "docs"
    / "notebooks"
    / "rogii-wellbore-geology-prediction"
    / "score_ascending_20260611"
    / "fleongg__fle3n-rogii-v4"
    / "fle3n-rogii-v4.ipynb"
)
JAEMIN_SOURCE = (
    ROOT
    / "docs"
    / "notebooks"
    / "rogii-wellbore-geology-prediction"
    / "score_ascending_20260611"
    / "jaemin3404__rogii-sp45-fleongg-blend-v2"
    / "rogii-sp45-fleongg-blend-v2.py"
)
PILKWANG_SOURCE = (
    ROOT
    / "docs"
    / "notebooks"
    / "rogii-wellbore-geology-prediction"
    / "score_ascending_20260611"
    / "pilkwang__rogii-target-free-tvt-geosteering"
    / "rogii-target-free-tvt-geosteering.ipynb"
)

CANDIDATES = [
    CandidateSpec(
        name="fle3n_final_blend",
        output_path=SOURCE_CHECK_ROOT / "fleongg-fle3n-rogii-v4" / "submission.csv",
        source_path=FLE3N_SOURCE,
        role="next_source_port_candidate",
        source_notebook="fleongg/fle3n-rogii-v4",
        required_inputs=(
            "phongnguyn23021656/koolbox-offline",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ),
        source_port_plan=(
            "Extend the existing fle3n source-port past the SP45 projection save point, "
            "regenerate fleongg_pretrained_submission.csv from mounted model artifacts, "
            "then write the final blend."
        ),
    ),
    CandidateSpec(
        name="jaemin_sp45_fleongg_final",
        output_path=SOURCE_CHECK_ROOT / "jaemin3404-rogii-sp45-fleongg-blend-v2" / "submission.csv",
        source_path=JAEMIN_SOURCE,
        role="next_source_port_candidate",
        source_notebook="jaemin3404/rogii-sp45-fleongg-blend-v2",
        required_inputs=(
            "phongnguyn23021656/koolbox-offline",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ),
        source_port_plan=(
            "Port the script source directly, keeping generation of both SP45 and "
            "fleongg branches in the submit notebook. Do not read mounted notebook outputs."
        ),
    ),
    CandidateSpec(
        name="pilkwang_raw_projection",
        output_path=(
            SOURCE_CHECK_ROOT
            / "pilkwang-rogii-target-free-tvt-geosteering"
            / "submission_projected_ridge_pf_projection_d4_b075_raw.csv"
        ),
        source_path=PILKWANG_SOURCE,
        role="branch_shortlist_reference",
        source_notebook="pilkwang/rogii-target-free-tvt-geosteering",
        required_inputs=(
            "pilkwang/rogii-model-package",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ),
        source_port_plan=(
            "Reuse only the projected ridge/PF branch generation. Keep pretrained LGBM "
            "and final late blend disabled unless their inputs are regenerated in-source."
        ),
    ),
    CandidateSpec(
        name="pilkwang_w0_60_blend",
        output_path=(
            SOURCE_CHECK_ROOT
            / "pilkwang-rogii-target-free-tvt-geosteering"
            / "submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv"
        ),
        source_path=PILKWANG_SOURCE,
        role="branch_shortlist_reference",
        source_notebook="pilkwang/rogii-target-free-tvt-geosteering",
        required_inputs=(
            "pilkwang/rogii-model-package",
            "fleongg/rogii-claude-models-pub",
            "ravaghi/wellbore-geology-prediction-artifacts",
        ),
        source_port_plan=(
            "Port only after raw projection and pretrained-LGBM branches are both "
            "confirmed to regenerate without public-output CSV reads."
        ),
    ),
]

ANCHORS = [
    AnchorSpec(
        "exp082_submitted_source_port",
        Path(
            "/tmp/kaggle-output/exp082_public_artifact_replay_followup/"
            "source_inference_v1/submission.csv"
        ),
    ),
    AnchorSpec(
        "fle3n_public_sp45_projection",
        SOURCE_CHECK_ROOT / "fleongg-fle3n-rogii-v4" / "sp45_projection_submission.csv",
    ),
    AnchorSpec(
        "ridge_sp_public_anchor",
        SOURCE_CHECK_ROOT / "lightningv08-lb-7-776-rogii-ridge-sp" / "submission.csv",
    ),
    AnchorSpec(
        "pilkwang_raw_projection",
        SOURCE_CHECK_ROOT
        / "pilkwang-rogii-target-free-tvt-geosteering"
        / "submission_projected_ridge_pf_projection_d4_b075_raw.csv",
    ),
]

RISK_PATTERNS = {
    "input_notebook_refs": r"/kaggle/input/notebooks/",
    "hardcoded_input_submission": r"/kaggle/input/[^\"'\s)]*submission[^\"'\s)]*\.csv",
    "read_input_submission_csv": (
        r"read_csv\([^)]*/kaggle/input/[^)]*submission[^)]*\.csv"
    ),
    "read_working_submission_csv": (
        r"read_csv\([^)]*/kaggle/working/[^)]*submission[^)]*\.csv"
    ),
    "writes_submission_csv": r"to_csv\([^)]*submission[^)]*\.csv",
    "mentions_public_or_visible": r"\b(public|visible)\b",
}

BLOCKING_RISK_KEYS = (
    "input_notebook_refs",
    "hardcoded_input_submission",
    "read_input_submission_csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_text(path: Path) -> str:
    if not path.exists():
        return ""
    if path.suffix == ".py":
        return path.read_text(errors="replace")
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(errors="replace"))
        chunks: list[str] = []
        for cell in notebook.get("cells", []):
            source = cell.get("source", "")
            if isinstance(source, list):
                chunks.extend(str(part) for part in source)
            else:
                chunks.append(str(source))
        return "\n".join(chunks)
    return path.read_text(errors="replace")


def inspect_source(spec: CandidateSpec) -> dict[str, Any]:
    text = source_text(spec.source_path)
    risk_hits = {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in RISK_PATTERNS.items()
    }
    blocking_hits = {
        name: count for name, count in risk_hits.items() if name in BLOCKING_RISK_KEYS and count
    }
    if not spec.source_path.exists():
        blocking_hits["missing_archived_source"] = 1
    return {
        "name": spec.name,
        "source_notebook": spec.source_notebook,
        "source_path": str(spec.source_path),
        "source_exists": spec.source_path.exists(),
        "source_sha256": sha256_file(spec.source_path) if spec.source_path.exists() else None,
        "line_count": int(text.count("\n") + 1) if text else 0,
        "risk_hits": risk_hits,
        "blocking_risks": blocking_hits,
        "source_port_blocked": bool(blocking_hits) or not spec.source_path.exists(),
        "required_inputs": list(spec.required_inputs),
        "source_port_plan": spec.source_port_plan,
    }


def read_submission(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "tvt"]:
        raise ValueError(f"{path}: expected columns ['id', 'tvt'], got {list(frame.columns)}")
    if frame["id"].tolist() != sample["id"].tolist():
        raise ValueError(f"{path}: id order/content differs from sample_submission")
    values = pd.to_numeric(frame["tvt"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite tvt values")
    return frame.assign(tvt=values.astype(float))


def summarize_submission(spec: CandidateSpec, frame: pd.DataFrame) -> dict[str, Any]:
    values = frame["tvt"].to_numpy(dtype=float)
    return {
        "name": spec.name,
        "role": spec.role,
        "source_notebook": spec.source_notebook,
        "path": str(spec.output_path),
        "rows": int(len(frame)),
        "sha256": sha256_file(spec.output_path),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
    }


def distance(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
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


def decision_for(
    spec: CandidateSpec,
    submission_error: str | None,
    source_record: dict[str, Any],
) -> str:
    if submission_error:
        return "blocked_missing_or_invalid_public_sample_output"
    if "missing_archived_source" in source_record["blocking_risks"]:
        return "blocked_missing_archived_source"
    if source_record["source_port_blocked"]:
        return "blocked_public_output_or_static_input_dependency"
    if spec.role == "branch_shortlist_reference":
        return "reference_only_until_branch_source_port_is_isolated"
    return "ready_for_one_hidden_compatible_source_port_run"


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# SP45/Fleongg Source-Port Next Candidates",
        "",
        f"- Status: {summary['status']}",
        f"- Candidate count: {len(summary['candidate_decisions'])}",
        f"- Recommended next action: {summary['recommended_next_action']}",
        "",
        "## Candidate Decisions",
        "",
    ]
    for item in summary["candidate_decisions"]:
        lines.append(
            "- "
            f"{item['name']}: decision={item['decision']} "
            f"sha={item.get('sha256', '-')} "
            f"blocking_risks={item['blocking_risks']}"
        )
    lines.extend(["", "## Key Distances", ""])
    for item in summary["key_distances"]:
        d = item["distance"]
        lines.append(
            "- "
            f"{item['left']} vs {item['right']}: "
            f"rmse={d['rmse']:.9f} p95_abs={d['p95_abs']:.9f} "
            f"max_abs={d['max_abs']:.9f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    sample = pd.read_csv(SAMPLE_PATH)
    frames: dict[str, pd.DataFrame] = {}
    submission_summaries: list[dict[str, Any]] = []
    source_risks = [inspect_source(spec) for spec in CANDIDATES]
    source_by_name = {record["name"]: record for record in source_risks}
    errors: dict[str, str] = {}

    for spec in CANDIDATES:
        try:
            frame = read_submission(spec.output_path, sample)
        except Exception as exc:  # noqa: BLE001
            errors[spec.name] = str(exc)
            continue
        frames[spec.name] = frame
        submission_summaries.append(summarize_submission(spec, frame))

    anchor_frames: dict[str, pd.DataFrame] = {}
    for anchor in ANCHORS:
        if anchor.path.exists():
            anchor_frames[anchor.name] = read_submission(anchor.path, sample)

    pairwise_rows: list[dict[str, Any]] = []
    for candidate_name, frame in frames.items():
        for anchor_name, anchor_frame in anchor_frames.items():
            pairwise_rows.append(
                {
                    "left": candidate_name,
                    "right": anchor_name,
                    "distance": distance(frame, anchor_frame),
                }
            )
    candidate_names = list(frames)
    for left_index, left_name in enumerate(candidate_names):
        for right_name in candidate_names[left_index + 1 :]:
            pairwise_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "distance": distance(frames[left_name], frames[right_name]),
                }
            )

    summary_by_name = {item["name"]: item for item in submission_summaries}
    candidate_decisions: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        source_record = source_by_name[spec.name]
        submission_record = summary_by_name.get(spec.name, {})
        decision = decision_for(spec, errors.get(spec.name), source_record)
        candidate_decisions.append(
            {
                "name": spec.name,
                "role": spec.role,
                "decision": decision,
                "submission_error": errors.get(spec.name),
                "sha256": submission_record.get("sha256"),
                "source_notebook": spec.source_notebook,
                "blocking_risks": source_record["blocking_risks"],
                "required_inputs": list(spec.required_inputs),
                "source_port_plan": spec.source_port_plan,
            }
        )

    ready = [
        item
        for item in candidate_decisions
        if item["decision"] == "ready_for_one_hidden_compatible_source_port_run"
    ]
    if ready:
        recommended_next_action = (
            f"source-port {ready[0]['name']} once, then submit only after Kaggle commit "
            "output passes submit-check and no /kaggle/input/notebooks dependency is present"
        )
    else:
        recommended_next_action = "no next candidate is ready for hidden-compatible source-port"

    key_pairs = {
        ("fle3n_final_blend", "jaemin_sp45_fleongg_final"),
        ("fle3n_final_blend", "exp082_submitted_source_port"),
        ("jaemin_sp45_fleongg_final", "exp082_submitted_source_port"),
        ("pilkwang_raw_projection", "exp082_submitted_source_port"),
        ("pilkwang_w0_60_blend", "exp082_submitted_source_port"),
        ("pilkwang_raw_projection", "pilkwang_w0_60_blend"),
    }
    key_distances = [
        row for row in pairwise_rows if (row["left"], row["right"]) in key_pairs
    ]

    summary = {
        "status": "next_candidate_guard_completed",
        "sample_path": str(SAMPLE_PATH),
        "candidate_decisions": candidate_decisions,
        "source_risks": source_risks,
        "submission_summaries": submission_summaries,
        "pairwise_distances": pairwise_rows,
        "key_distances": key_distances,
        "recommended_next_action": recommended_next_action,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(submission_summaries).to_csv(
        ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_submission_summary.csv", index=False
    )
    pd.DataFrame(source_risks).to_csv(
        ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_source_risk.csv", index=False
    )
    pd.DataFrame(
        {
            "left": row["left"],
            "right": row["right"],
            **row["distance"],
        }
        for row in pairwise_rows
    ).to_csv(ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_pairwise.csv", index=False)
    write_report(summary, ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_README.md")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
