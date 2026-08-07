#!/usr/bin/env python3
"""Audit whether exp270 exact joint top-K paths span distinct offset basins.

The exp270 candidates and exact joint-path scores were frozen without
unknown-suffix truth.  This truth-late readout measures, inside every exp209
persistent-offset episode, whether the five highest-scoring legal joint paths
actually provide different absolute-datum hypotheses or only near-duplicate
trajectories inside the same basin.  It does not rerun or modify the HMM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_PATH_DIAGNOSTICS = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_path_diagnostics.csv"
)
DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_truth_path_grammar_audit_20260726/"
    "episode_grammar_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_topk_mode_basin_audit_20260726"
)
PATH_COLUMNS = tuple(f"topk_path_{rank}" for rank in range(1, 6))
USECOLS = (
    "well",
    "row_idx",
    "true_tvt_readout_only",
    "posterior_mean",
    *PATH_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--path-diagnostics",
        type=Path,
        default=DEFAULT_PATH_DIAGNOSTICS,
    )
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
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


def iter_well_frames(
    path: Path,
    chunksize: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    pending: pd.DataFrame | None = None
    last_well: str | None = None
    for chunk in pd.read_csv(
        path,
        usecols=list(USECOLS),
        chunksize=chunksize,
        dtype={"well": str},
    ):
        if pending is not None:
            chunk = pd.concat([pending, chunk], ignore_index=True)
            pending = None
        final_well = str(chunk["well"].iloc[-1])
        pending = chunk.loc[chunk["well"] == final_well].copy()
        complete = chunk.loc[chunk["well"] != final_well]
        for well, frame in complete.groupby("well", sort=False):
            well = str(well)
            if last_well is not None and well <= last_well:
                raise ValueError("candidate well order is not strictly increasing")
            last_well = well
            yield well, frame.reset_index(drop=True)
    if pending is not None:
        well = str(pending["well"].iloc[0])
        if last_well is not None and well <= last_well:
            raise ValueError("candidate final well order is invalid")
        yield well, pending.reset_index(drop=True)


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def weighted_fraction(mask: pd.Series, weights: pd.Series) -> float | None:
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return None
    return float(weights.loc[mask].sum() / denominator)


def pooled_rmse(
    group: pd.DataFrame,
    column: str,
) -> float | None:
    finite = group[column].notna()
    rows = float(group.loc[finite, "rows"].sum())
    if rows <= 0.0:
        return None
    sse = float(
        (
            group.loc[finite, column] ** 2
            * group.loc[finite, "rows"]
        ).sum()
    )
    return float(np.sqrt(sse / rows))


def weighted_row_mean(
    group: pd.DataFrame,
    value_column: str,
) -> float | None:
    finite = group[value_column].notna()
    rows = float(group.loc[finite, "rows"].sum())
    if rows <= 0.0:
        return None
    return float(
        (
            group.loc[finite, value_column]
            * group.loc[finite, "rows"]
        ).sum()
        / rows
    )


def summarize_group(
    group: pd.DataFrame,
    total_sse: float,
) -> dict[str, Any]:
    full_top5 = group["available_topk_paths"] == 5
    same_sign = group["topk_all_mean_errors_same_sign_as_posterior_mean"]
    any_within5 = group["topk_any_episode_mean_error_within5"]
    any_opposite = group[
        "topk_any_mean_error_opposite_sign_to_posterior_mean"
    ]
    any_better = group["topk_any_episode_rmse_better_than_posterior_mean"]
    result: dict[str, Any] = {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
        "full_top5_episode_fraction": float(full_top5.mean()),
        "full_top5_episode_sse_fraction": weighted_fraction(
            full_top5,
            group["episode_sse"],
        ),
        "available_topk_paths_median": float(
            group["available_topk_paths"].median()
        ),
        "path_score_gap_rank2_median": float(
            group["path_score_gap_rank2"].median()
        ),
        "path_score_gap_max_available_median": float(
            group["path_score_gap_max_available"].median()
        ),
        "topk_mean_tvt_span_ft_median": float(
            group["topk_mean_tvt_span_ft"].median()
        ),
        "topk_mean_tvt_span_ft_p90": float(
            group["topk_mean_tvt_span_ft"].quantile(0.9)
        ),
        "topk_row_tvt_span_mean_ft_pooled": weighted_row_mean(
            group,
            "topk_row_tvt_span_mean_ft",
        ),
        "topk_row_tvt_span_p90_ft_episode_median": float(
            group["topk_row_tvt_span_p90_ft"].median()
        ),
        "topk_row_tvt_span_max_ft_p90": float(
            group["topk_row_tvt_span_max_ft"].quantile(0.9)
        ),
        "topk_pairwise_rmse_max_ft_median": float(
            group["topk_pairwise_rmse_max_ft"].median()
        ),
        "topk_pairwise_rmse_max_ft_p90": float(
            group["topk_pairwise_rmse_max_ft"].quantile(0.9)
        ),
        "topk_alt_mean_separation_gt1_episode_fraction": float(
            (group["topk_alt_mean_separation_gt1_count"] > 0).mean()
        ),
        "topk_alt_mean_separation_gt5_episode_fraction": float(
            (group["topk_alt_mean_separation_gt5_count"] > 0).mean()
        ),
        "topk_alt_mean_separation_gt10_episode_fraction": float(
            (group["topk_alt_mean_separation_gt10_count"] > 0).mean()
        ),
        "topk_all_mean_errors_same_sign_fraction": float(
            same_sign.mean()
        ),
        "topk_all_mean_errors_same_sign_sse_fraction": weighted_fraction(
            same_sign,
            group["episode_sse"],
        ),
        "topk_any_episode_mean_error_within5_fraction": float(
            any_within5.mean()
        ),
        "topk_any_episode_mean_error_within5_sse_fraction": (
            weighted_fraction(
                any_within5,
                group["episode_sse"],
            )
        ),
        "topk_any_mean_error_opposite_sign_fraction": float(
            any_opposite.mean()
        ),
        "topk_any_mean_error_opposite_sign_sse_fraction": (
            weighted_fraction(
                any_opposite,
                group["episode_sse"],
            )
        ),
        "topk_any_episode_rmse_better_than_posterior_mean_fraction": (
            float(any_better.mean())
        ),
        "topk_any_episode_rmse_better_than_posterior_mean_sse_fraction": (
            weighted_fraction(
                any_better,
                group["episode_sse"],
            )
        ),
        "posterior_mean_episode_rmse_ft_pooled": pooled_rmse(
            group,
            "rmse_ft",
        ),
        "top1_episode_rmse_ft_pooled": pooled_rmse(
            group,
            "top1_episode_rmse_ft",
        ),
        "topk_best_episode_rmse_ft_pooled": pooled_rmse(
            group,
            "topk_best_episode_rmse_ft",
        ),
        "topk_rowwise_oracle_rmse_ft_pooled": pooled_rmse(
            group,
            "topk_rowwise_oracle_rmse_ft",
        ),
        "topk_truth_bracket_row_fraction_pooled": weighted_row_mean(
            group,
            "topk_truth_bracket_row_fraction",
        ),
        "topk_truth_within5_row_fraction_pooled": weighted_row_mean(
            group,
            "topk_truth_within5_row_fraction",
        ),
        "topk_mean_span_vs_rmse_spearman": safe_spearman(
            group["topk_mean_tvt_span_ft"],
            group["rmse_ft"],
        ),
        "topk_mean_span_vs_viterbi_gain_spearman": safe_spearman(
            group["topk_mean_tvt_span_ft"],
            group["viterbi_rmse_gain_ft"],
        ),
        "rank2_score_gap_vs_topk_mean_span_spearman": safe_spearman(
            group["path_score_gap_rank2"],
            group["topk_mean_tvt_span_ft"],
        ),
    }
    return result


def summarize_wells(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "wells": int(len(group)),
        "persistent_wells": int(group["has_persistent_episode"].sum()),
        "rows_median": float(group["suffix_rows"].median()),
        "posterior_mean_rmse_ft_median": float(
            group["posterior_mean_rmse_ft"].median()
        ),
        "top1_joint_path_log10_probability_median": float(
            group["top1_joint_path_log10_probability"].median()
        ),
        "top1_joint_path_log10_probability_p10": float(
            group["top1_joint_path_log10_probability"].quantile(0.1)
        ),
        "top1_joint_path_log10_probability_p90": float(
            group["top1_joint_path_log10_probability"].quantile(0.9)
        ),
        "top5_joint_path_log10_mass_median": float(
            group["top5_joint_path_log10_mass"].median()
        ),
        "top1_surprisal_nats_per_row_median": float(
            group["top1_surprisal_nats_per_row"].median()
        ),
        "top1_surprisal_nats_per_row_p10": float(
            group["top1_surprisal_nats_per_row"].quantile(0.1)
        ),
        "top1_surprisal_nats_per_row_p90": float(
            group["top1_surprisal_nats_per_row"].quantile(0.9)
        ),
        "top5_surprisal_nats_per_row_median": float(
            group["top5_surprisal_nats_per_row"].median()
        ),
        "top5_vs_top1_log_mass_gain_median": float(
            group["top5_vs_top1_log_mass_gain"].median()
        ),
        "top1_surprisal_vs_rmse_spearman": safe_spearman(
            group["top1_surprisal_nats_per_row"],
            group["posterior_mean_rmse_ft"],
        ),
        "top1_surprisal_vs_persistent_row_fraction_spearman": (
            safe_spearman(
                group["top1_surprisal_nats_per_row"],
                group["persistent_row_fraction"],
            )
        ),
    }


def episode_metrics(
    spec: dict[str, Any],
    frame: pd.DataFrame,
    path_score_gaps: dict[int, float],
) -> dict[str, Any]:
    row_index = pd.to_numeric(
        frame["row_idx"],
        errors="raise",
    ).to_numpy(np.int64)
    start = int(
        np.searchsorted(
            row_index,
            int(spec["start_row_idx"]),
        )
    )
    end = int(
        np.searchsorted(
            row_index,
            int(spec["end_row_idx_exclusive"]),
        )
    )
    if end - start != int(spec["rows"]):
        raise ValueError(f"{spec['episode_id']}: episode row mismatch")
    part = frame.iloc[start:end]
    truth = pd.to_numeric(
        part["true_tvt_readout_only"],
        errors="raise",
    ).to_numpy(np.float64)
    posterior_mean = pd.to_numeric(
        part["posterior_mean"],
        errors="raise",
    ).to_numpy(np.float64)
    paths = []
    available_ranks = []
    for rank, column in enumerate(PATH_COLUMNS, start=1):
        values = pd.to_numeric(
            part[column],
            errors="coerce",
        ).to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.all():
            paths.append(values)
            available_ranks.append(rank)
        elif finite.any():
            raise ValueError(
                f"{spec['episode_id']}: partial availability for {column}"
            )
    if not paths or available_ranks[0] != 1:
        raise ValueError(f"{spec['episode_id']}: top-1 path unavailable")
    path_matrix = np.vstack(paths)
    errors = path_matrix - truth[None, :]
    mean_errors = errors.mean(axis=1)
    path_rmse = np.sqrt(np.mean(errors**2, axis=1))
    posterior_mean_error = posterior_mean - truth
    posterior_mean_rmse = float(
        np.sqrt(np.mean(posterior_mean_error**2))
    )
    row_min = path_matrix.min(axis=0)
    row_max = path_matrix.max(axis=0)
    row_span = row_max - row_min
    pairwise_rmse = [
        float(np.sqrt(np.mean((left - right) ** 2)))
        for left, right in combinations(path_matrix, 2)
    ]
    top1 = path_matrix[0]
    alt_mean_separation = np.abs(
        path_matrix[1:].mean(axis=1) - top1.mean()
    )
    posterior_sign = float(np.sign(posterior_mean_error.mean()))
    path_sign = np.sign(mean_errors)
    score_gaps = [
        float(path_score_gaps[rank])
        for rank in available_ranks
        if rank in path_score_gaps
    ]
    if len(score_gaps) != len(available_ranks):
        raise ValueError(
            f"{spec['episode_id']}: path score diagnostics missing"
        )
    record = {
        **spec,
        "available_topk_paths": int(len(paths)),
        "available_topk_ranks": "|".join(map(str, available_ranks)),
        "path_score_gap_rank2": path_score_gaps.get(2),
        "path_score_gap_max_available": float(max(score_gaps)),
        "topk_mean_tvt_span_ft": float(
            path_matrix.mean(axis=1).max()
            - path_matrix.mean(axis=1).min()
        ),
        "topk_row_tvt_span_mean_ft": float(row_span.mean()),
        "topk_row_tvt_span_p90_ft": float(np.quantile(row_span, 0.9)),
        "topk_row_tvt_span_max_ft": float(row_span.max()),
        "topk_pairwise_rmse_max_ft": (
            float(max(pairwise_rmse)) if pairwise_rmse else 0.0
        ),
        "topk_alt_mean_separation_gt1_count": int(
            (alt_mean_separation > 1.0).sum()
        ),
        "topk_alt_mean_separation_gt5_count": int(
            (alt_mean_separation > 5.0).sum()
        ),
        "topk_alt_mean_separation_gt10_count": int(
            (alt_mean_separation > 10.0).sum()
        ),
        "topk_all_mean_errors_same_sign_as_posterior_mean": bool(
            posterior_sign != 0.0
            and np.all(path_sign == posterior_sign)
        ),
        "topk_any_episode_mean_error_within5": bool(
            np.any(np.abs(mean_errors) <= 5.0)
        ),
        "topk_any_mean_error_opposite_sign_to_posterior_mean": bool(
            posterior_sign != 0.0
            and np.any(path_sign == -posterior_sign)
        ),
        "topk_any_episode_rmse_better_than_posterior_mean": bool(
            np.any(path_rmse < posterior_mean_rmse)
        ),
        "top1_episode_mean_error_ft": float(mean_errors[0]),
        "top1_episode_rmse_ft": float(path_rmse[0]),
        "topk_best_episode_mean_abs_error_ft": float(
            np.min(np.abs(mean_errors))
        ),
        "topk_best_episode_rmse_ft": float(np.min(path_rmse)),
        "topk_rowwise_oracle_rmse_ft": float(
            np.sqrt(np.mean(np.min(errors**2, axis=0)))
        ),
        "topk_truth_bracket_row_fraction": float(
            ((truth >= row_min) & (truth <= row_max)).mean()
        ),
        "topk_truth_within5_row_fraction": float(
            (np.min(np.abs(errors), axis=0) <= 5.0).mean()
        ),
    }
    return record


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    candidates_path = resolve(root, args.candidates)
    path_diagnostics_path = resolve(root, args.path_diagnostics)
    episodes_path = resolve(root, args.episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)
    episode_lookup = {
        well: group.to_dict("records")
        for well, group in episodes.groupby("well", sort=False)
    }
    diagnostics = pd.read_csv(
        path_diagnostics_path,
        dtype={"well": str},
    )
    unique = diagnostics.loc[
        diagnostics["status"].eq("unique")
        & diagnostics["unique_rank"].notna()
    ].copy()
    unique["unique_rank"] = unique["unique_rank"].astype(int)
    score_gap_lookup = {
        well: dict(
            zip(
                group["unique_rank"],
                group["score_gap_vs_top1"],
                strict=True,
            )
        )
        for well, group in unique.groupby("well", sort=False)
    }
    joint_path_lookup = {
        well: group.sort_values("joint_rank")[
            "path_log_posterior"
        ].to_numpy(np.float64)
        for well, group in diagnostics.groupby("well", sort=False)
    }

    records: list[dict[str, Any]] = []
    well_records: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    for well, frame in iter_well_frames(candidates_path, args.chunksize):
        specs = episode_lookup.get(well, [])
        if well not in score_gap_lookup or well not in joint_path_lookup:
            raise ValueError(f"{well}: path diagnostics unavailable")
        truth = pd.to_numeric(
            frame["true_tvt_readout_only"],
            errors="raise",
        ).to_numpy(np.float64)
        posterior_mean = pd.to_numeric(
            frame["posterior_mean"],
            errors="raise",
        ).to_numpy(np.float64)
        joint_log_probability = joint_path_lookup[well]
        if len(joint_log_probability) != 5:
            raise ValueError(f"{well}: expected five exact joint paths")
        top5_log_mass = float(logsumexp(joint_log_probability))
        if top5_log_mass > 1e-8:
            raise ValueError(f"{well}: top-5 posterior mass exceeds one")
        persistent_rows = int(sum(int(spec["rows"]) for spec in specs))
        well_records.append(
            {
                "well": well,
                "suffix_rows": int(len(frame)),
                "posterior_mean_rmse_ft": float(
                    np.sqrt(np.mean((posterior_mean - truth) ** 2))
                ),
                "persistent_episodes": int(len(specs)),
                "persistent_rows": persistent_rows,
                "persistent_row_fraction": float(
                    persistent_rows / len(frame)
                ),
                "has_persistent_episode": bool(specs),
                "top1_joint_path_log_probability": float(
                    joint_log_probability[0]
                ),
                "top5_joint_path_log_mass": top5_log_mass,
                "top1_joint_path_log10_probability": float(
                    joint_log_probability[0] / np.log(10.0)
                ),
                "top5_joint_path_log10_mass": float(
                    top5_log_mass / np.log(10.0)
                ),
                "top1_surprisal_nats_per_row": float(
                    -joint_log_probability[0] / len(frame)
                ),
                "top5_surprisal_nats_per_row": float(
                    -top5_log_mass / len(frame)
                ),
                "top5_vs_top1_log_mass_gain": float(
                    top5_log_mass - joint_log_probability[0]
                ),
            }
        )
        if not specs:
            continue
        seen_wells.add(well)
        for spec in specs:
            records.append(
                episode_metrics(
                    spec,
                    frame,
                    score_gap_lookup[well],
                )
            )
    if seen_wells != set(episode_lookup):
        missing = sorted(set(episode_lookup) - seen_wells)
        raise ValueError(f"episode wells missing: {missing[:5]}")

    result = pd.DataFrame(records)
    if len(result) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(result)} != {len(episodes)}"
        )
    if result[["episode_id", "well"]].duplicated().any():
        raise ValueError("duplicate episode key")
    result.to_csv(output / "episode_topk_basin_metrics.csv", index=False)
    well_frame = pd.DataFrame(well_records)
    if len(well_frame) != 773:
        raise ValueError(f"well count mismatch: {len(well_frame)} != 773")
    well_frame.to_csv(
        output / "well_joint_path_mass_metrics.csv",
        index=False,
    )

    total_sse = float(result["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in result.groupby("cause_bucket", sort=True):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_group(group, total_sse),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_topk_basin_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(result)),
            "wells": int(result["well"].nunique()),
            "episode_rows": int(result["rows"].sum()),
            "maximum_joint_paths": len(PATH_COLUMNS),
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(candidates_path),
            str(path_diagnostics_path.relative_to(root)): sha256(
                path_diagnostics_path
            ),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(result, total_sse),
        "cause_bucket_summary": cause_rows,
        "joint_path_mass_summary": {
            "all": summarize_wells(well_frame),
            "persistent": summarize_wells(
                well_frame.loc[well_frame["has_persistent_episode"]]
            ),
            "nonpersistent": summarize_wells(
                well_frame.loc[~well_frame["has_persistent_episode"]]
            ),
        },
        "guards": {
            "decoder": (
                "Inputs are exp270 exact joint top-5 paths, ranked before "
                "truth and deduplicated only by full TVT-path identity."
            ),
            "interpretation": (
                "Top-K max-product paths are not sum-product basin masses. "
                "Near-duplicate top paths cannot rule out a lower-ranked "
                "macro basin with large aggregate path multiplicity."
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
