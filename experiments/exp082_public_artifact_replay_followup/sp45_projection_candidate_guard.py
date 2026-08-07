from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubmissionSpec:
    name: str
    path: Path
    role: str


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_PREFIX = "sp45_projection_candidate_guard"

SPECS = [
    SubmissionSpec(
        "candidate_fle3n_sp45_projection",
        Path("/tmp/kaggle-output/source-check/fleongg-fle3n-rogii-v4/sp45_projection_submission.csv"),
        "candidate",
    ),
    SubmissionSpec(
        "candidate_jaemin_sp45_projection",
        Path(
            "/tmp/kaggle-output/source-check/jaemin3404-rogii-sp45-fleongg-blend-v2/"
            "sp45_projection_submission.csv"
        ),
        "candidate",
    ),
    SubmissionSpec(
        "reference_rauff_sp45_projection_direct_output",
        Path(
            "/tmp/kaggle-output/source-check/rauffauzanrambe-rogii-sp45-wellbore-for-blend-prediction/"
            "sp45_projection_submission.csv"
        ),
        "direct_output_reference",
    ),
    SubmissionSpec(
        "anchor_ridge_sp",
        Path("/tmp/kaggle-output/source-check/lightningv08-lb-7-776-rogii-ridge-sp/submission.csv"),
        "anchor",
    ),
    SubmissionSpec(
        "anchor_pilkwang_final",
        Path("/tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering/submission.csv"),
        "anchor",
    ),
    SubmissionSpec(
        "anchor_pilkwang_raw_projection",
        Path(
            "/tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering/"
            "submission_projected_ridge_pf_projection_d4_b075_raw.csv"
        ),
        "anchor",
    ),
    SubmissionSpec(
        "anchor_pilkwang_w0_60",
        Path(
            "/tmp/kaggle-output/source-check/pilkwang-rogii-target-free-tvt-geosteering/"
            "submission_projected_ridge_pf_pretrained_lgbm_w0.60.csv"
        ),
        "anchor",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_submission(spec: SubmissionSpec, sample: pd.DataFrame) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"{spec.name}: missing {spec.path}")
    frame = pd.read_csv(spec.path)
    if list(frame.columns) != ["id", "tvt"]:
        raise ValueError(f"{spec.name}: expected columns ['id', 'tvt'], got {list(frame.columns)}")
    if frame["id"].tolist() != sample["id"].tolist():
        raise ValueError(f"{spec.name}: id order/content differs from sample_submission")
    values = pd.to_numeric(frame["tvt"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{spec.name}: non-finite tvt values")
    return frame.assign(tvt=values.astype(float))


def summarize_frame(spec: SubmissionSpec, frame: pd.DataFrame) -> dict[str, Any]:
    values = frame["tvt"].to_numpy(dtype=float)
    return {
        "name": spec.name,
        "role": spec.role,
        "path": str(spec.path),
        "rows": int(len(frame)),
        "sha256": sha256_file(spec.path),
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


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# SP45 Projection Candidate Guard",
        "",
        f"- Status: {summary['status']}",
        f"- Candidates: {len(summary['candidate_names'])}",
        f"- Anchors: {len(summary['anchor_names'])}",
        "",
        "## Candidates",
        "",
    ]
    for item in summary["submission_summaries"]:
        if item["role"] != "candidate" and item["role"] != "direct_output_reference":
            continue
        lines.append(
            "- "
            f"{item['name']}: rows={item['rows']} sha={item['sha256']} "
            f"range=[{item['min']:.6f}, {item['max']:.6f}]"
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
    frames = {spec.name: read_submission(spec, sample) for spec in SPECS}
    submission_summaries = [
        summarize_frame(spec, frames[spec.name]) for spec in SPECS
    ]

    candidate_names = [
        spec.name for spec in SPECS if spec.role in {"candidate", "direct_output_reference"}
    ]
    anchor_names = [spec.name for spec in SPECS if spec.role == "anchor"]

    pairwise_rows: list[dict[str, Any]] = []
    for left_name in candidate_names:
        for right_name in anchor_names:
            pairwise_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "distance": distance(frames[left_name], frames[right_name]),
                }
            )
    for left_index, left_name in enumerate(candidate_names):
        for right_name in candidate_names[left_index + 1 :]:
            pairwise_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "distance": distance(frames[left_name], frames[right_name]),
                }
            )

    key_pairs = {
        ("candidate_fle3n_sp45_projection", "anchor_ridge_sp"),
        ("candidate_jaemin_sp45_projection", "anchor_ridge_sp"),
        ("reference_rauff_sp45_projection_direct_output", "anchor_ridge_sp"),
        ("candidate_fle3n_sp45_projection", "candidate_jaemin_sp45_projection"),
        ("candidate_fle3n_sp45_projection", "reference_rauff_sp45_projection_direct_output"),
        ("candidate_jaemin_sp45_projection", "reference_rauff_sp45_projection_direct_output"),
        ("candidate_fle3n_sp45_projection", "anchor_pilkwang_raw_projection"),
        ("candidate_jaemin_sp45_projection", "anchor_pilkwang_raw_projection"),
        ("reference_rauff_sp45_projection_direct_output", "anchor_pilkwang_raw_projection"),
    }
    key_distances = [
        row for row in pairwise_rows if (row["left"], row["right"]) in key_pairs
    ]

    summary = {
        "status": "guard_completed",
        "sample_path": str(SAMPLE_PATH),
        "candidate_names": candidate_names,
        "anchor_names": anchor_names,
        "submission_summaries": submission_summaries,
        "pairwise_distances": pairwise_rows,
        "key_distances": key_distances,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(submission_summaries).to_csv(
        ARTIFACTS_DIR / f"{OUTPUT_PREFIX}_submission_summary.csv", index=False
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
