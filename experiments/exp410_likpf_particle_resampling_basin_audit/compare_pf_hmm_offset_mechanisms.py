"""Compare exp410 PF offset episodes with exp408 HMM mechanism episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PF_EXPERIMENT = "exp410_likpf_particle_resampling_basin_audit"
HMM_EXPERIMENT = "exp408_hmm_message_rate_basin_audit"
DEFAULT_PF_EPISODES = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/"
    "full_merged/exp410_likpf_particle_resampling_basin_audit_episode_summary.csv"
)
DEFAULT_PF_WELLS = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_target_wells.csv"
)
DEFAULT_HMM_EPISODES = Path(
    "experiments/exp408_hmm_message_rate_basin_audit/artifacts/kaggle_v3/"
    "exp408_hmm_message_rate_basin_audit_episode_summary.csv"
)
DEFAULT_HMM_EPISODE_ASSET = Path(
    "experiments/exp408_hmm_message_rate_basin_audit/assets/"
    "persistent_offset_episodes.csv"
)
DEFAULT_HMM_WELLS = Path(
    "experiments/exp408_hmm_message_rate_basin_audit/assets/target_wells.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/"
    "pf_hmm_cross_mechanism"
)
EXPECTED_PF_EPISODES = 839
EXPECTED_PF_WELLS = 496
EXPECTED_HMM_EPISODES = 638
EXPECTED_HMM_WELLS = 450


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pf-episodes", type=Path, default=DEFAULT_PF_EPISODES)
    parser.add_argument("--pf-wells", type=Path, default=DEFAULT_PF_WELLS)
    parser.add_argument("--hmm-episodes", type=Path, default=DEFAULT_HMM_EPISODES)
    parser.add_argument(
        "--hmm-episode-asset",
        type=Path,
        default=DEFAULT_HMM_EPISODE_ASSET,
    )
    parser.add_argument("--hmm-wells", type=Path, default=DEFAULT_HMM_WELLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mechanism_family(model: str, cause: str) -> str:
    if model == "pf":
        if cause in {
            "initial_condition_support_miss",
            "transition_propagation_escape",
        }:
            return "initial_or_forward_propagation"
        if cause.startswith("gr_emission"):
            return "gr_emission"
        if cause == "resampling_particle_extinction":
            return "resampling_extinction"
        if cause in {
            "within_seed_particle_mean_multiplicity",
            "across_seed_aggregation_multiplicity",
        }:
            return "path_or_mean_multiplicity"
        if cause == "support_or_clamp_shortage":
            return "state_support_shortage"
        return "mixed_or_unresolved"
    if cause == "forward_transition_prior_hysteresis":
        return "initial_or_forward_propagation"
    if cause == "backward_smoothing_reversal":
        return "backward_only_smoothing"
    if cause == "sum_product_path_multiplicity":
        return "path_or_mean_multiplicity"
    if cause == "state_support_shortage":
        return "state_support_shortage"
    return "mixed_or_unresolved"


def interval_pairs(
    pf_episodes: pd.DataFrame,
    hmm_episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hmm_by_well = {
        str(well): group.sort_values("start_row_idx", kind="stable")
        for well, group in hmm_episodes.groupby("well", sort=False)
    }
    pair_rows: list[dict[str, Any]] = []
    pf_rows: list[dict[str, Any]] = []
    for pf in pf_episodes.itertuples(index=False):
        candidates = hmm_by_well.get(str(pf.well))
        overlaps: list[dict[str, Any]] = []
        if candidates is not None:
            for hmm in candidates.itertuples(index=False):
                start = max(int(pf.start_row_idx), int(hmm.start_row_idx))
                end = min(
                    int(pf.end_row_idx_exclusive),
                    int(hmm.end_row_idx_exclusive),
                )
                intersection = max(end - start, 0)
                if intersection <= 0:
                    continue
                union = int(pf.rows) + int(hmm.rows) - intersection
                record = {
                    "pf_episode_id": str(pf.episode_id),
                    "hmm_episode_id": str(hmm.episode_id),
                    "well": str(pf.well),
                    "pf_cause": str(pf.cause),
                    "hmm_cause": str(hmm.cause),
                    "pf_family": mechanism_family("pf", str(pf.cause)),
                    "hmm_family": mechanism_family("hmm", str(hmm.cause)),
                    "pf_rows": int(pf.rows),
                    "hmm_rows": int(hmm.rows),
                    "intersection_rows": intersection,
                    "pf_overlap_fraction": intersection / int(pf.rows),
                    "hmm_overlap_fraction": intersection / int(hmm.rows),
                    "interval_jaccard": intersection / max(union, 1),
                    "pf_episode_sse": float(pf.episode_sse),
                    "hmm_episode_sse": float(hmm.episode_sse),
                    "pf_mean_error_ft": float(pf.mean_error_ft),
                    "hmm_mean_error_ft": float(hmm.mean_error_ft),
                    "same_error_sign": bool(
                        float(pf.mean_error_ft) * float(hmm.mean_error_ft)
                        > 0.0
                    ),
                }
                overlaps.append(record)
                pair_rows.append(record)
        if overlaps:
            best = max(
                overlaps,
                key=lambda row: (
                    row["intersection_rows"],
                    row["interval_jaccard"],
                    row["hmm_episode_id"],
                ),
            )
            pf_rows.append(
                {
                    **best,
                    "overlapping_hmm_episodes": len(overlaps),
                    "total_intersection_rows": int(
                        sum(row["intersection_rows"] for row in overlaps)
                    ),
                    "any_hmm_overlap": True,
                    "mechanism_family_match": bool(
                        best["pf_family"] == best["hmm_family"]
                    ),
                }
            )
        else:
            pf_rows.append(
                {
                    "pf_episode_id": str(pf.episode_id),
                    "hmm_episode_id": "",
                    "well": str(pf.well),
                    "pf_cause": str(pf.cause),
                    "hmm_cause": "no_hmm_offset_overlap",
                    "pf_family": mechanism_family("pf", str(pf.cause)),
                    "hmm_family": "no_hmm_offset_overlap",
                    "pf_rows": int(pf.rows),
                    "hmm_rows": 0,
                    "intersection_rows": 0,
                    "pf_overlap_fraction": 0.0,
                    "hmm_overlap_fraction": 0.0,
                    "interval_jaccard": 0.0,
                    "pf_episode_sse": float(pf.episode_sse),
                    "hmm_episode_sse": 0.0,
                    "pf_mean_error_ft": float(pf.mean_error_ft),
                    "hmm_mean_error_ft": np.nan,
                    "same_error_sign": False,
                    "overlapping_hmm_episodes": 0,
                    "total_intersection_rows": 0,
                    "any_hmm_overlap": False,
                    "mechanism_family_match": False,
                }
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(pf_rows)


def cross_cause_summary(pf_best: pd.DataFrame) -> pd.DataFrame:
    total_pf_sse = float(pf_best["pf_episode_sse"].sum())
    result = (
        pf_best.groupby(
            ["pf_cause", "hmm_cause", "pf_family", "hmm_family"],
            observed=False,
            sort=True,
        )
        .agg(
            pf_episodes=("pf_episode_id", "nunique"),
            wells=("well", "nunique"),
            pf_rows=("pf_rows", "sum"),
            intersection_rows=("total_intersection_rows", "sum"),
            pf_episode_sse=("pf_episode_sse", "sum"),
            mean_pf_overlap_fraction=("pf_overlap_fraction", "mean"),
            same_error_sign_fraction=("same_error_sign", "mean"),
            mechanism_family_match_fraction=(
                "mechanism_family_match",
                "mean",
            ),
        )
        .reset_index()
    )
    result["pf_episode_fraction"] = result["pf_episodes"] / len(pf_best)
    result["pf_sse_fraction"] = result["pf_episode_sse"] / total_pf_sse
    return result.sort_values(
        ["pf_episode_sse", "pf_cause", "hmm_cause"],
        ascending=[False, True, True],
        kind="stable",
    )


def main() -> None:
    args = parse_args()
    pf_episodes = pd.read_csv(
        args.pf_episodes, dtype={"episode_id": str, "well": str}
    )
    pf_wells = pd.read_csv(args.pf_wells, dtype={"well": str})
    hmm_episodes = pd.read_csv(
        args.hmm_episodes, dtype={"episode_id": str, "well": str}
    )
    hmm_asset = pd.read_csv(
        args.hmm_episode_asset, dtype={"episode_id": str, "well": str}
    )
    hmm_wells = pd.read_csv(args.hmm_wells, dtype={"well": str})
    if (
        len(pf_episodes) != EXPECTED_PF_EPISODES
        or len(pf_wells) != EXPECTED_PF_WELLS
        or len(hmm_episodes) != EXPECTED_HMM_EPISODES
        or len(hmm_wells) != EXPECTED_HMM_WELLS
    ):
        raise RuntimeError("PF/HMM strict episode or well inventory changed")
    if set(pf_episodes["well"]) != set(pf_wells["well"]):
        raise RuntimeError("PF episode/well inventory mismatch")
    if set(hmm_episodes["well"]) != set(hmm_wells["well"]):
        raise RuntimeError("HMM episode/well inventory mismatch")
    hmm_episodes = hmm_episodes.merge(
        hmm_asset[["episode_id", "mean_error_ft"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    if hmm_episodes["mean_error_ft"].isna().any():
        raise RuntimeError("HMM mean error missing after fixed-asset join")

    pairs, pf_best = interval_pairs(pf_episodes, hmm_episodes)
    if (
        len(pf_best) != EXPECTED_PF_EPISODES
        or pf_best["pf_episode_id"].duplicated().any()
    ):
        raise RuntimeError("PF/HMM best-overlap coverage changed")
    cross = cross_cause_summary(pf_best)
    pf_set = set(pf_wells["well"])
    hmm_set = set(hmm_wells["well"])
    overlapping = pf_best["any_hmm_overlap"].to_numpy(bool)
    family_match = pf_best["mechanism_family_match"].to_numpy(bool)
    same_sign = pf_best["same_error_sign"].to_numpy(bool)
    pf_sse = pf_best["pf_episode_sse"].to_numpy(np.float64)
    total_pf_sse = float(pf_sse.sum())

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "all_interval_pairs": output
        / "exp410_exp408_all_interval_overlap_pairs.csv",
        "pf_episode_best_overlap": output
        / "exp410_exp408_pf_episode_best_overlap.csv",
        "cross_cause_summary": output
        / "exp410_exp408_cross_cause_summary.csv",
    }
    pairs.to_csv(outputs["all_interval_pairs"], index=False)
    pf_best.to_csv(outputs["pf_episode_best_overlap"], index=False)
    cross.to_csv(outputs["cross_cause_summary"], index=False)

    summary = {
        "experiments": {
            "pf": PF_EXPERIMENT,
            "hmm": HMM_EXPERIMENT,
        },
        "inventory": {
            "pf_wells": len(pf_set),
            "hmm_wells": len(hmm_set),
            "both_offset_wells": len(pf_set & hmm_set),
            "pf_only_offset_wells": len(pf_set - hmm_set),
            "hmm_only_offset_wells": len(hmm_set - pf_set),
            "union_offset_wells": len(pf_set | hmm_set),
            "pf_episodes": len(pf_episodes),
            "hmm_episodes": len(hmm_episodes),
            "overlap_pairs": len(pairs),
        },
        "pf_episode_overlap": {
            "any_hmm_overlap_episodes": int(overlapping.sum()),
            "any_hmm_overlap_episode_fraction": float(overlapping.mean()),
            "any_hmm_overlap_sse_fraction": float(
                pf_sse[overlapping].sum() / total_pf_sse
            ),
            "at_least_half_interval_overlap_episodes": int(
                (
                    pf_best["pf_overlap_fraction"].to_numpy(np.float64)
                    >= 0.5
                ).sum()
            ),
            "same_error_sign_given_overlap_fraction": float(
                same_sign[overlapping].mean()
            )
            if overlapping.any()
            else None,
            "mechanism_family_match_given_overlap_fraction": float(
                family_match[overlapping].mean()
            )
            if overlapping.any()
            else None,
            "mechanism_family_match_sse_fraction": float(
                pf_sse[family_match].sum() / total_pf_sse
            ),
            "total_interval_intersection_rows": int(
                pf_best["total_intersection_rows"].sum()
            ),
        },
        "sources": {
            "pf_episodes": {
                "path": str(args.pf_episodes),
                "sha256": sha256_path(args.pf_episodes),
            },
            "pf_wells": {
                "path": str(args.pf_wells),
                "sha256": sha256_path(args.pf_wells),
            },
            "hmm_episodes": {
                "path": str(args.hmm_episodes),
                "sha256": sha256_path(args.hmm_episodes),
            },
            "hmm_episode_asset": {
                "path": str(args.hmm_episode_asset),
                "sha256": sha256_path(args.hmm_episode_asset),
            },
            "hmm_wells": {
                "path": str(args.hmm_wells),
                "sha256": sha256_path(args.hmm_wells),
            },
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
            }
            for name, path in outputs.items()
        },
        "guards": {
            "strict_pf_hmm_inventory": True,
            "one_best_overlap_record_per_pf_episode": True,
            "interval_overlap_not_used_as_mechanism_proof": True,
        },
    }
    summary_path = output / "exp410_exp408_cross_mechanism_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
