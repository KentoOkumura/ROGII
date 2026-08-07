#!/usr/bin/env python3
"""Audit saved exp270 Viterbi latent-rate stickiness.

exp270 did not persist row-level latent rate paths, but it did persist the
top-1 rate-state switch fraction and SHA256 hashes for each joint top-5 rate
path.  This readout tests whether persistent offsets require rate-state
switching or top-K rate-mode diversity.  No HMM or prediction is rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ARTIFACT_ROOT = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts"
)
DEFAULT_PATH_DIAGNOSTICS = ARTIFACT_ROOT / (
    "exp270_exact_hmm_posterior_mode_candidate_audit_path_diagnostics.csv"
)
DEFAULT_BY_WELL = ARTIFACT_ROOT / (
    "exp270_exact_hmm_posterior_mode_candidate_audit_by_well.csv"
)
DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_onset_missingness_timing_20260726/"
    "episode_onset_missingness_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_viterbi_rate_state_stickiness_20260726"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--path-diagnostics",
        type=Path,
        default=DEFAULT_PATH_DIAGNOSTICS,
    )
    parser.add_argument("--by-well", type=Path, default=DEFAULT_BY_WELL)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def summarize_wells(group: pd.DataFrame) -> dict[str, Any]:
    zero_switch = group["top1_rate_switch_count"].eq(0)
    one_rate_path = group["top5_unique_rate_path_hashes"].eq(1)
    both = zero_switch & one_rate_path
    return {
        "wells": int(len(group)),
        "top1_zero_rate_switch_fraction": float(zero_switch.mean()),
        "top1_rate_switch_count_mean": float(
            group["top1_rate_switch_count"].mean()
        ),
        "top1_rate_switch_count_median": float(
            group["top1_rate_switch_count"].median()
        ),
        "top1_rate_switch_count_p90": float(
            group["top1_rate_switch_count"].quantile(0.90)
        ),
        "top1_rate_switch_count_p99": float(
            group["top1_rate_switch_count"].quantile(0.99)
        ),
        "top1_rate_switch_rate_mean": float(
            group["top1_rate_switch_rate"].mean()
        ),
        "top1_rate_switch_rate_p90": float(
            group["top1_rate_switch_rate"].quantile(0.90)
        ),
        "top5_one_unique_rate_path_fraction": float(
            one_rate_path.mean()
        ),
        "top1_zero_switch_and_top5_one_rate_path_fraction": float(
            both.mean()
        ),
        "top5_unique_rate_path_hashes_mean": float(
            group["top5_unique_rate_path_hashes"].mean()
        ),
        "top5_unique_rate_path_hashes_median": float(
            group["top5_unique_rate_path_hashes"].median()
        ),
    }


def summarize_episodes(
    group: pd.DataFrame,
    total_sse: float,
) -> dict[str, Any]:
    zero_switch = group["top1_rate_switch_count"].eq(0)
    one_rate_path = group["top5_unique_rate_path_hashes"].eq(1)
    denominator = float(group["episode_sse"].sum())
    return {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(denominator / total_sse),
        "top1_zero_rate_switch_episode_fraction": float(
            zero_switch.mean()
        ),
        "top1_zero_rate_switch_sse_fraction": (
            float(
                group.loc[zero_switch, "episode_sse"].sum()
                / denominator
            )
            if denominator > 0.0
            else None
        ),
        "top5_one_unique_rate_path_episode_fraction": float(
            one_rate_path.mean()
        ),
        "top5_one_unique_rate_path_sse_fraction": (
            float(
                group.loc[one_rate_path, "episode_sse"].sum()
                / denominator
            )
            if denominator > 0.0
            else None
        ),
        "rate_switch_rate_vs_rmse_spearman": safe_spearman(
            group["top1_rate_switch_rate"],
            group["rmse_ft"],
        ),
        "rate_switch_rate_vs_mean_rate_error_growth_spearman": (
            safe_spearman(
                group["top1_rate_switch_rate"],
                group[
                    "posterior_mean_near_minus_far_rate_error_abs"
                ],
            )
        ),
        "rate_switch_rate_vs_viterbi_rate_error_growth_spearman": (
            safe_spearman(
                group["top1_rate_switch_rate"],
                group[
                    "global_viterbi_near_minus_far_rate_error_abs"
                ],
            )
        ),
        "rate_switch_rate_vs_transition_crescendo_spearman": (
            safe_spearman(
                group["top1_rate_switch_rate"],
                group[
                    "truth_pre_crescendo_near_minus_far_nll"
                ],
            )
        ),
        "rate_switch_rate_vs_pre128_error_slope_spearman": (
            safe_spearman(
                group["top1_rate_switch_rate"],
                group["pre128_error_slope_ft_per_row"].abs(),
            )
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    path_diagnostics_path = resolve(root, args.path_diagnostics)
    by_well_path = resolve(root, args.by_well)
    episodes_path = resolve(root, args.episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    path_diagnostics = pd.read_csv(path_diagnostics_path)
    path_diagnostics["well"] = path_diagnostics["well"].astype(str)
    by_well = pd.read_csv(by_well_path)
    by_well["well"] = by_well["well"].astype(str)
    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)

    top1 = path_diagnostics.loc[
        path_diagnostics["joint_rank"].eq(1)
    ].copy()
    if len(top1) != 773 or top1["well"].nunique() != 773:
        raise ValueError("top-1 path diagnostics must cover 773 wells")
    if not top1["status"].eq("unique").all():
        raise ValueError("top-1 path must be unique for every well")
    rows = (
        by_well.loc[
            by_well["candidate"].eq("posterior_mean"),
            ["well", "rows"],
        ]
        .drop_duplicates("well")
        .copy()
    )
    if len(rows) != 773:
        raise ValueError("by-well row counts must cover 773 wells")
    rate_hash_counts = (
        path_diagnostics.groupby("well", sort=False)[
            "rate_path_sha256"
        ]
        .nunique()
        .rename("top5_unique_rate_path_hashes")
        .reset_index()
    )
    tvt_hash_counts = (
        path_diagnostics.groupby("well", sort=False)[
            "tvt_path_sha256"
        ]
        .nunique()
        .rename("top5_unique_tvt_path_hashes")
        .reset_index()
    )
    well = (
        top1[
            [
                "well",
                "rate_switch_rate",
                "grid_edge_rate",
                "rate_path_sha256",
                "tvt_path_sha256",
            ]
        ]
        .rename(
            columns={
                "rate_switch_rate": "top1_rate_switch_rate",
                "grid_edge_rate": "top1_grid_edge_rate",
                "rate_path_sha256": "top1_rate_path_sha256",
                "tvt_path_sha256": "top1_tvt_path_sha256",
            }
        )
        .merge(rows, on="well", how="left", validate="one_to_one")
        .merge(
            rate_hash_counts,
            on="well",
            how="left",
            validate="one_to_one",
        )
        .merge(
            tvt_hash_counts,
            on="well",
            how="left",
            validate="one_to_one",
        )
    )
    raw_switch_count = (
        well["top1_rate_switch_rate"] * (well["rows"] - 1)
    )
    rounded_switch_count = np.rint(raw_switch_count).astype(np.int64)
    maximum_switch_count_rounding_error = float(
        np.max(np.abs(raw_switch_count - rounded_switch_count))
    )
    if maximum_switch_count_rounding_error > 1e-9:
        raise ValueError(
            "rate switch fraction does not map to integer count: "
            f"{maximum_switch_count_rounding_error}"
        )
    well["top1_rate_switch_count"] = rounded_switch_count
    persistent_wells = set(episodes["well"])
    well["has_persistent_offset_episode"] = well["well"].isin(
        persistent_wells
    )
    well.to_csv(
        output / "well_viterbi_rate_state_metrics.csv",
        index=False,
    )

    episode = episodes.merge(
        well[
            [
                "well",
                "top1_rate_switch_rate",
                "top1_rate_switch_count",
                "top1_grid_edge_rate",
                "top5_unique_rate_path_hashes",
                "top5_unique_tvt_path_hashes",
            ]
        ],
        on="well",
        how="left",
        validate="many_to_one",
    )
    if episode[
        [
            "top1_rate_switch_rate",
            "top1_rate_switch_count",
            "top5_unique_rate_path_hashes",
        ]
    ].isna().any().any():
        raise ValueError("episode-to-well rate diagnostic join failed")
    episode.to_csv(
        output / "episode_viterbi_rate_state_metrics.csv",
        index=False,
    )
    total_sse = float(episode["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in episode.groupby(
        "cause_bucket",
        sort=True,
    ):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_episodes(group, total_sse),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_viterbi_rate_state_summary.csv",
        index=False,
    )

    persistent_summary = []
    for flag, group in well.groupby(
        "has_persistent_offset_episode",
        sort=True,
    ):
        persistent_summary.append(
            {
                "has_persistent_offset_episode": bool(flag),
                **summarize_wells(group),
            }
        )
    summary = {
        "scope": {
            "wells": int(len(well)),
            "persistent_wells": int(
                well["has_persistent_offset_episode"].sum()
            ),
            "episodes": int(len(episode)),
            "episode_rows": int(episode["rows"].sum()),
        },
        "source_sha256": {
            str(path_diagnostics_path.relative_to(root)): sha256(
                path_diagnostics_path
            ),
            str(by_well_path.relative_to(root)): sha256(by_well_path),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "all_wells": summarize_wells(well),
        "persistent_well_split": persistent_summary,
        "all_episodes": summarize_episodes(episode, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "maximum_rate_switch_count_rounding_abs": (
                maximum_switch_count_rounding_error
            ),
            "top1_path_unique_for_every_well": True,
            "top1_position_edge_rate_max": float(
                well["top1_grid_edge_rate"].max()
            ),
            "row_level_rate_paths_persisted": False,
            "interpretation": (
                "The saved switch fraction and rate-path hashes establish "
                "well-level max-product stickiness. They do not expose "
                "row-level rate values or sum-product rate mass."
            ),
            "prediction_generation": False,
            "hmm_rerun": False,
            "model_or_booster": False,
        },
    }
    text = json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    (output / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
