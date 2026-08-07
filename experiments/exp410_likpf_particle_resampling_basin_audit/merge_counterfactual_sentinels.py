"""Strictly merge the four exp410 PF counterfactual sentinel shards."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = "exp410_likpf_particle_resampling_basin_audit"
PREFIX = f"{EXPERIMENT}_counterfactual"
DEFAULT_SENTINELS = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_counterfactual_sentinel_wells.csv"
)
DEFAULT_FULL_EPISODES = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/"
    "full_merged/exp410_likpf_particle_resampling_basin_audit_episode_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/"
    "counterfactual_merged"
)
EXPECTED_WELLS = 12
EXPECTED_VARIANTS = (
    "baseline",
    "momentum_one",
    "zero_process_noise",
    "process_noise_x3",
    "init_spread_x3",
    "gr_sigma_x1p3",
    "gr_sigma_x3",
    "gr_emission_near_disabled",
    "resample_threshold_0p1",
    "resampling_disabled",
    "roughening_x10",
    "clamp_margin_x2",
)
EXPECTED_READOUTS = tuple(
    f"{variant}_arithmetic_seed_mean" for variant in EXPECTED_VARIANTS
) + (
    "baseline_seed_median",
    "baseline_best_loglik_seed",
    "baseline_loglik_weighted_seed_mean",
    "baseline_oracle_best_seed_per_row",
    "baseline_seed_block0_mean",
    "baseline_seed_block1_mean",
    "baseline_seed_block2_mean",
    "baseline_seed_block3_mean",
    "baseline_particle_mode_seed_mean",
    "baseline_particle_mode_seed_median",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--sentinels", type=Path, default=DEFAULT_SENTINELS)
    parser.add_argument("--full-episodes", type=Path, default=DEFAULT_FULL_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def require_artifact(root: Path, filename: str) -> Path:
    for candidate in (root / "artifacts" / filename, root / filename):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    matches = sorted(root.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"{root}: {filename}")


def add_pooled_metrics(grouped: pd.DataFrame) -> pd.DataFrame:
    result = grouped.copy()
    result["rmse_ft"] = np.sqrt(result["sse"] / result["rows"])
    result["baseline_rmse_ft"] = np.sqrt(
        result["baseline_sse"] / result["rows"]
    )
    result["sse_delta_vs_baseline"] = result["sse"] - result["baseline_sse"]
    result["rmse_delta_vs_baseline_ft"] = (
        result["rmse_ft"] - result["baseline_rmse_ft"]
    )
    result["sse_relative_vs_baseline"] = result["sse"] / np.maximum(
        result["baseline_sse"], 1.0e-12
    )
    return result


def aggregate_metrics(
    metrics: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    grouped = (
        metrics.groupby(
            group_columns
            + [
                "scope",
                "readout",
                "pf_variant",
                "target_free",
                "oracle",
                "offline_suffix_readout",
            ],
            observed=False,
            sort=True,
            dropna=False,
        )
        .agg(
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            sse=("sse", "sum"),
            baseline_sse=("baseline_sse", "sum"),
        )
        .reset_index()
    )
    return add_pooled_metrics(grouped).sort_values(
        group_columns + ["scope", "rmse_ft", "readout"],
        kind="stable",
    )


def paired_episode_summary(episode_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["cause", "readout", "pf_variant"]
    for key, group in episode_metrics.groupby(
        group_columns, observed=False, sort=True, dropna=False
    ):
        cause, readout, pf_variant = key
        delta = group["sse_delta_vs_baseline"].to_numpy(np.float64)
        rmse_delta = group["rmse_delta_vs_baseline_ft"].to_numpy(np.float64)
        rows.append(
            {
                "cause": cause,
                "readout": readout,
                "pf_variant": pf_variant,
                "episodes": int(group["episode_id"].nunique()),
                "wells": int(group["well"].nunique()),
                "rows": int(group["rows"].sum()),
                "sse": float(group["sse"].sum()),
                "baseline_sse": float(group["baseline_sse"].sum()),
                "sse_delta_vs_baseline": float(delta.sum()),
                "sse_relative_vs_baseline": float(
                    group["sse"].sum()
                    / max(float(group["baseline_sse"].sum()), 1.0e-12)
                ),
                "episodes_improved": int(np.sum(delta < 0.0)),
                "episodes_worsened": int(np.sum(delta > 0.0)),
                "episode_improved_fraction": float(np.mean(delta < 0.0)),
                "median_episode_rmse_delta_ft": float(np.median(rmse_delta)),
                "worst_episode_rmse_delta_ft": float(np.max(rmse_delta)),
                "best_episode_rmse_delta_ft": float(np.min(rmse_delta)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cause", "sse_delta_vs_baseline", "readout"], kind="stable"
    )


def exact_two_sided_sign_p(improved: int, worsened: int) -> float:
    non_tied = improved + worsened
    if non_tied == 0:
        return float("nan")
    lower_tail = min(improved, worsened)
    probability = 2.0 * sum(
        math.comb(non_tied, index) for index in range(lower_tail + 1)
    ) / (2**non_tied)
    return float(min(1.0, probability))


def paired_overall_summary(episode_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (readout, pf_variant), group in episode_metrics.groupby(
        ["readout", "pf_variant"],
        observed=False,
        sort=True,
        dropna=False,
    ):
        delta = group["sse_delta_vs_baseline"].to_numpy(np.float64)
        rmse_delta = group["rmse_delta_vs_baseline_ft"].to_numpy(np.float64)
        improved = int(np.sum(delta < 0.0))
        worsened = int(np.sum(delta > 0.0))
        tied = int(np.sum(delta == 0.0))
        baseline_sse = float(group["baseline_sse"].sum())
        sse = float(group["sse"].sum())
        rows.append(
            {
                "readout": readout,
                "pf_variant": pf_variant,
                "target_free": bool(group["target_free"].iloc[0]),
                "oracle": bool(group["oracle"].iloc[0]),
                "offline_suffix_readout": bool(
                    group["offline_suffix_readout"].iloc[0]
                ),
                "episodes": int(group["episode_id"].nunique()),
                "wells": int(group["well"].nunique()),
                "rows": int(group["rows"].sum()),
                "sse": sse,
                "baseline_sse": baseline_sse,
                "sse_delta_vs_baseline": sse - baseline_sse,
                "sse_relative_vs_baseline": sse
                / max(baseline_sse, 1.0e-12),
                "episodes_improved": improved,
                "episodes_worsened": worsened,
                "episodes_tied": tied,
                "episode_improved_fraction": float(
                    improved / max(improved + worsened, 1)
                ),
                "episode_sign_test_two_sided_p": exact_two_sided_sign_p(
                    improved, worsened
                ),
                "median_episode_rmse_delta_ft": float(
                    np.median(rmse_delta)
                ),
                "worst_episode_rmse_delta_ft": float(np.max(rmse_delta)),
                "best_episode_rmse_delta_ft": float(np.min(rmse_delta)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sse_relative_vs_baseline", "readout"], kind="stable"
    )


def well_stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    fixed = metrics.loc[metrics["scope"] == "fixed_episode_rows"].copy()
    rows: list[dict[str, Any]] = []
    for (readout, pf_variant), group in fixed.groupby(
        ["readout", "pf_variant"],
        observed=False,
        sort=True,
        dropna=False,
    ):
        delta = group["sse_delta_vs_baseline"].to_numpy(np.float64)
        relative = group["sse_relative_vs_baseline"].to_numpy(np.float64)
        improved = int(np.sum(delta < 0.0))
        worsened = int(np.sum(delta > 0.0))
        tied = int(np.sum(delta == 0.0))
        total_sse = float(group["sse"].sum())
        total_baseline_sse = float(group["baseline_sse"].sum())
        leave_one_out = np.asarray(
            [
                (total_sse - float(row.sse))
                / max(
                    total_baseline_sse - float(row.baseline_sse),
                    1.0e-12,
                )
                for row in group.itertuples(index=False)
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "readout": readout,
                "pf_variant": pf_variant,
                "wells": int(group["well"].nunique()),
                "pooled_sse_relative_vs_baseline": total_sse
                / max(total_baseline_sse, 1.0e-12),
                "wells_improved": improved,
                "wells_worsened": worsened,
                "wells_tied": tied,
                "well_improved_fraction": float(
                    improved / max(improved + worsened, 1)
                ),
                "well_sign_test_two_sided_p": exact_two_sided_sign_p(
                    improved, worsened
                ),
                "well_sse_relative_p25": float(np.quantile(relative, 0.25)),
                "well_sse_relative_median": float(
                    np.quantile(relative, 0.50)
                ),
                "well_sse_relative_p75": float(np.quantile(relative, 0.75)),
                "leave_one_well_out_relative_min": float(
                    np.min(leave_one_out)
                ),
                "leave_one_well_out_relative_max": float(
                    np.max(leave_one_out)
                ),
                "leave_one_well_out_all_improve": bool(
                    np.all(leave_one_out < 1.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pooled_sse_relative_vs_baseline", "readout"], kind="stable"
    )


def seed_block_stability(
    paired_by_cause: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block = paired_by_cause.loc[
        paired_by_cause["readout"].str.match(
            r"baseline_seed_block[0-3]_mean"
        )
    ].copy()
    block["block_index"] = (
        block["readout"].str.extract(r"block([0-3])", expand=False).astype(int)
    )
    rows: list[dict[str, Any]] = []
    for cause, group in block.groupby("cause", observed=False, sort=True):
        relative = group["sse_relative_vs_baseline"].to_numpy(np.float64)
        rows.append(
            {
                "cause": cause,
                "seed_blocks": int(group["block_index"].nunique()),
                "all_four_blocks_present": bool(
                    group["block_index"].nunique() == 4
                ),
                "blocks_improved": int(
                    np.sum(
                        group["sse_delta_vs_baseline"].to_numpy(np.float64)
                        < 0.0
                    )
                ),
                "min_sse_relative_vs_baseline": float(np.min(relative)),
                "max_sse_relative_vs_baseline": float(np.max(relative)),
                "range_sse_relative_vs_baseline": float(
                    np.max(relative) - np.min(relative)
                ),
                "median_sse_relative_vs_baseline": float(
                    np.median(relative)
                ),
            }
        )
    return block, pd.DataFrame(rows)


def mechanism_target_for_readout(readout: str) -> str:
    if readout == "init_spread_x3_arithmetic_seed_mean":
        return "initial_condition_support_miss"
    if readout.startswith(
        ("momentum_one_", "zero_process_noise_", "process_noise_x3_")
    ):
        return "transition_propagation_escape"
    if readout.startswith(("gr_sigma_", "gr_emission_")):
        return "gr_emission"
    if readout.startswith(
        (
            "resample_threshold_",
            "resampling_disabled_",
            "roughening_x10_",
        )
    ):
        return "resampling_particle_extinction"
    if readout.startswith("baseline_particle_mode_"):
        return "within_and_across_seed_multiplicity"
    if readout in {
        "baseline_seed_median",
        "baseline_best_loglik_seed",
        "baseline_loglik_weighted_seed_mean",
        "baseline_seed_block0_mean",
        "baseline_seed_block1_mean",
        "baseline_seed_block2_mean",
        "baseline_seed_block3_mean",
    }:
        return "across_seed_aggregation_multiplicity"
    if readout == "clamp_margin_x2_arithmetic_seed_mean":
        return "support_or_clamp_shortage"
    if readout == "baseline_oracle_best_seed_per_row":
        return "oracle_upper_bound"
    if readout == "baseline_arithmetic_seed_mean":
        return "control"
    return "unmapped"


def matched_intervention_summary(
    paired_by_cause: pd.DataFrame,
) -> pd.DataFrame:
    result = paired_by_cause.copy()
    result["mechanism_target"] = result["readout"].map(
        mechanism_target_for_readout
    )
    emission_causes = {
        "gr_emission_alias_observed",
        "gr_emission_imputation",
    }
    result["matched_to_exclusive_cause"] = (
        (result["mechanism_target"] == result["cause"])
        | (
            result["cause"].isin(emission_causes)
            & (result["mechanism_target"] == "gr_emission")
        )
        | (
            result["cause"]
            == "within_seed_particle_mean_multiplicity"
        )
        & (
            result["mechanism_target"]
            == "within_and_across_seed_multiplicity"
        )
        | (
            result["cause"]
            == "across_seed_aggregation_multiplicity"
        )
        & (
            result["mechanism_target"].isin(
                {
                    "within_and_across_seed_multiplicity",
                    "across_seed_aggregation_multiplicity",
                }
            )
        )
    )
    return result.loc[result["matched_to_exclusive_cause"]].sort_values(
        ["cause", "sse_delta_vs_baseline", "readout"], kind="stable"
    )


def main() -> None:
    args = parse_args()
    if len(args.shard) != 4:
        raise RuntimeError("exactly four counterfactual shards are required")
    sentinels = pd.read_csv(args.sentinels, dtype={"well": str})
    full_episodes = pd.read_csv(
        args.full_episodes, dtype={"episode_id": str, "well": str}
    )
    if (
        len(sentinels) != EXPECTED_WELLS
        or sentinels["well"].duplicated().any()
    ):
        raise RuntimeError("fixed counterfactual sentinel asset changed")

    summaries: list[dict[str, Any]] = []
    metrics_parts: list[pd.DataFrame] = []
    episode_metric_parts: list[pd.DataFrame] = []
    row_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    shard_artifacts: list[dict[str, Any]] = []
    for root in args.shard:
        paths = {
            "summary": require_artifact(root, f"{PREFIX}_summary.json"),
            "variant_metrics": require_artifact(
                root, f"{PREFIX}_variant_metrics.csv"
            ),
            "episode_metrics": require_artifact(
                root, f"{PREFIX}_episode_metrics.csv"
            ),
            "episode_row_predictions": require_artifact(
                root, f"{PREFIX}_episode_row_predictions.csv.gz"
            ),
            "run_manifest": require_artifact(
                root, f"{PREFIX}_run_manifest.csv"
            ),
        }
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        summaries.append(summary)
        metrics_parts.append(
            pd.read_csv(paths["variant_metrics"], dtype={"well": str})
        )
        episode_metric_parts.append(
            pd.read_csv(
                paths["episode_metrics"],
                dtype={"well": str, "episode_id": str},
            )
        )
        row_parts.append(
            pd.read_csv(
                paths["episode_row_predictions"],
                dtype={"well": str, "id": str},
            )
        )
        manifest_parts.append(
            pd.read_csv(paths["run_manifest"], dtype={"well": str})
        )
        shard_artifacts.append(
            {
                "root": str(root),
                "shard_index": int(summary["shard_index"]),
                "files": {
                    name: {
                        "path": str(path),
                        "sha256": sha256_path(path),
                        "decompressed_sha256": (
                            sha256_path(path, decompressed=True)
                            if path.suffix == ".gz"
                            else None
                        ),
                    }
                    for name, path in paths.items()
                },
            }
        )
    if sorted(int(summary["shard_index"]) for summary in summaries) != [
        0,
        1,
        2,
        3,
    ]:
        raise RuntimeError("counterfactual shard indices incomplete or duplicated")
    implementation_sha_fields = (
        "baseline_kernel_sha256",
        "config_sha256",
        "counterfactual_runner_sha256",
        "variant_contract_sha256",
    )
    implementation_sha = {
        field: str(summaries[0]["implementation"][field])
        for field in implementation_sha_fields
    }
    for summary in summaries:
        if (
            summary["status"] != "complete"
            or summary["stage"] != "counterfactual"
        ):
            raise RuntimeError("a counterfactual shard is not a complete full run")
        if not all(bool(value) for value in summary["guards"].values()):
            raise RuntimeError("a counterfactual shard guard failed")
        if (
            float(summary["parity"]["baseline_max_abs_ft"]) != 0.0
            or int(summary["parity"]["failed_wells"]) != 0
        ):
            raise RuntimeError("a counterfactual shard parity guard failed")
        for field, expected in implementation_sha.items():
            if str(summary["implementation"][field]) != expected:
                raise RuntimeError(
                    f"counterfactual shard implementation mismatch: {field}"
                )
        if (
            str(summary["fixed_assets"]["sentinels_sha256"])
            != sha256_path(args.sentinels)
        ):
            raise RuntimeError("counterfactual sentinel SHA changed")

    metrics = pd.concat(metrics_parts, ignore_index=True)
    episode_metrics = pd.concat(episode_metric_parts, ignore_index=True)
    row_predictions = pd.concat(row_parts, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="stable"
    )
    run_manifest = pd.concat(manifest_parts, ignore_index=True).sort_values(
        ["well", "variant"], kind="stable"
    )
    expected_wells = set(sentinels["well"])
    if set(metrics["well"]) != expected_wells:
        raise RuntimeError("counterfactual metrics well coverage changed")
    if set(run_manifest["well"]) != expected_wells:
        raise RuntimeError("counterfactual run-manifest well coverage changed")
    if (
        run_manifest[["well", "variant"]].duplicated().any()
        or len(run_manifest) != EXPECTED_WELLS * len(EXPECTED_VARIANTS)
        or set(run_manifest["variant"]) != set(EXPECTED_VARIANTS)
    ):
        raise RuntimeError("counterfactual well/variant coverage changed")
    variants_per_well = run_manifest.groupby("well")["variant"].nunique()
    if not (variants_per_well == len(EXPECTED_VARIANTS)).all():
        raise RuntimeError("a sentinel well is missing a counterfactual variant")
    if (
        set(metrics["readout"]) != set(EXPECTED_READOUTS)
        or metrics[["well", "scope", "readout"]].duplicated().any()
        or len(metrics) != EXPECTED_WELLS * 2 * len(EXPECTED_READOUTS)
    ):
        raise RuntimeError("counterfactual metric readout coverage changed")
    if float(
        run_manifest.loc[
            run_manifest["variant"] == "baseline",
            "parity_max_abs_ft_vs_fixed",
        ].max()
    ) != 0.0:
        raise RuntimeError("counterfactual baseline parity is nonzero")
    if row_predictions[["well", "row_idx"]].duplicated().any():
        raise RuntimeError("duplicate counterfactual episode prediction rows")

    selected_full_episodes = full_episodes.loc[
        full_episodes["well"].isin(expected_wells)
    ].copy()
    if (
        set(episode_metrics["episode_id"])
        != set(selected_full_episodes["episode_id"])
        or episode_metrics[["episode_id", "readout"]].duplicated().any()
        or len(episode_metrics)
        != len(selected_full_episodes) * len(EXPECTED_READOUTS)
    ):
        raise RuntimeError("counterfactual episode/readout coverage changed")
    expected_episode_rows = int(selected_full_episodes["rows"].sum())
    if len(row_predictions) != expected_episode_rows:
        raise RuntimeError("counterfactual episode-row coverage changed")

    metrics = metrics.merge(
        sentinels[
            [
                "well",
                "selection_order",
                "selection_phase",
                "representative_episode_id",
                "representative_cause",
            ]
        ],
        on="well",
        how="left",
        validate="many_to_one",
    )
    episode_metrics = episode_metrics.merge(
        selected_full_episodes[
            [
                "episode_id",
                "cause",
                "episode_sse",
                "rows",
                "mean_error_ft",
            ]
        ].rename(
            columns={
                "episode_sse": "fixed_asset_episode_sse",
                "rows": "fixed_asset_episode_rows",
            }
        ),
        on="episode_id",
        how="left",
        validate="many_to_one",
    )

    aggregate = aggregate_metrics(metrics, [])
    by_representative_cause = aggregate_metrics(
        metrics, ["representative_cause"]
    )
    paired_by_cause = paired_episode_summary(episode_metrics)
    paired_overall = paired_overall_summary(episode_metrics)
    well_stability = well_stability_summary(metrics)
    seed_block_by_cause, seed_block_summary = seed_block_stability(
        paired_by_cause
    )
    matched_interventions = matched_intervention_summary(paired_by_cause)
    run_summary = (
        run_manifest.groupby("variant", sort=True)
        .agg(
            wells=("well", "nunique"),
            suffix_rows=("suffix_rows", "sum"),
            elapsed_seconds=("elapsed_seconds", "sum"),
            mean_ess=("ess_mean", "mean"),
            mean_resampling_rate=("resampling_rate", "mean"),
            mean_max_likelihood_seed_weight=(
                "max_likelihood_seed_weight",
                "mean",
            ),
            mean_likelihood_effective_seed_count=(
                "likelihood_effective_seed_count",
                "mean",
            ),
            max_peak_rss_gb=("peak_rss_gb_after", "max"),
        )
        .reset_index()
    )

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "variant_metrics": output / f"{PREFIX}_merged_variant_metrics.csv",
        "episode_metrics": output / f"{PREFIX}_merged_episode_metrics.csv",
        "episode_row_predictions": output
        / f"{PREFIX}_merged_episode_row_predictions.csv.gz",
        "run_manifest": output / f"{PREFIX}_merged_run_manifest.csv",
        "aggregate_summary": output / f"{PREFIX}_merged_aggregate_summary.csv",
        "by_representative_cause": output
        / f"{PREFIX}_merged_by_representative_cause.csv",
        "paired_by_cause": output / f"{PREFIX}_merged_paired_by_cause.csv",
        "paired_overall": output / f"{PREFIX}_merged_paired_overall.csv",
        "well_stability": output / f"{PREFIX}_merged_well_stability.csv",
        "seed_block_by_cause": output
        / f"{PREFIX}_merged_seed_block_by_cause.csv",
        "seed_block_stability": output
        / f"{PREFIX}_merged_seed_block_stability.csv",
        "matched_interventions": output
        / f"{PREFIX}_merged_matched_interventions.csv",
        "run_summary": output / f"{PREFIX}_merged_run_summary.csv",
    }
    frames = {
        "variant_metrics": metrics,
        "episode_metrics": episode_metrics,
        "episode_row_predictions": row_predictions,
        "run_manifest": run_manifest,
        "aggregate_summary": aggregate,
        "by_representative_cause": by_representative_cause,
        "paired_by_cause": paired_by_cause,
        "paired_overall": paired_overall,
        "well_stability": well_stability,
        "seed_block_by_cause": seed_block_by_cause,
        "seed_block_stability": seed_block_summary,
        "matched_interventions": matched_interventions,
        "run_summary": run_summary,
    }
    for name, path in outputs.items():
        frames[name].to_csv(
            path,
            index=False,
            compression="gzip" if path.suffix == ".gz" else None,
        )

    summary = {
        "experiment": EXPERIMENT,
        "stage": "counterfactual_strict_merge",
        "status": "complete",
        "counts": {
            "wells": EXPECTED_WELLS,
            "episodes": int(len(selected_full_episodes)),
            "episode_rows": expected_episode_rows,
            "pf_variants": len(EXPECTED_VARIANTS),
            "pf_well_runs": int(len(run_manifest)),
            "readouts": len(EXPECTED_READOUTS),
        },
        "parity": {
            "baseline_max_abs_ft": 0.0,
            "failed_wells": 0,
        },
        "runtime": {
            "sum_variant_well_seconds": float(
                run_manifest["elapsed_seconds"].sum()
            ),
            "max_peak_rss_gb": float(
                run_manifest["peak_rss_gb_after"].max()
            ),
            "shard_elapsed_seconds": {
                str(summary["shard_index"]): float(
                    summary["runtime"]["elapsed_seconds"]
                )
                for summary in summaries
            },
        },
        "implementation": implementation_sha,
        "fixed_assets": {
            "sentinels": str(args.sentinels),
            "sentinels_sha256": sha256_path(args.sentinels),
            "full_episode_summary": str(args.full_episodes),
            "full_episode_summary_sha256": sha256_path(args.full_episodes),
        },
        "shards": shard_artifacts,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
                "decompressed_sha256": (
                    sha256_path(path, decompressed=True)
                    if path.suffix == ".gz"
                    else None
                ),
            }
            for name, path in outputs.items()
        },
        "guards": {
            "strict_four_shard_coverage": True,
            "strict_twelve_well_coverage": True,
            "strict_twelve_variant_per_well_coverage": True,
            "strict_episode_and_row_coverage": True,
            "baseline_persisted_prediction_parity_zero": True,
            "identical_implementation_sha_across_shards": True,
            "all_shard_guards_passed": True,
            "no_duplicate_well_variant_episode_readout_or_row": True,
        },
    }
    summary_path = output / f"{PREFIX}_merged_summary.json"
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
