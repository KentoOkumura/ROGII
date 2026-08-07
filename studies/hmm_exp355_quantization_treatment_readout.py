#!/usr/bin/env python3
"""Read out exp355 on exp209 persistent-offset and transition-bias scopes.

This joins saved, already-frozen OOF predictions only. It asks whether the
exp355 geometry rate-mean treatment repairs rows that were persistent offsets
under exp209, and whether its global safety failure comes from creating error
outside those original episodes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EXP209 = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_EXP355 = Path(
    "/tmp/exp355_quantization_audit/artifacts/"
    "exp355_exp226_dip_rate_prior_on_exp209_stage1_oof_predictions.csv.gz"
)
DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_KERNEL_EPISODES = Path(
    "studies/hmm_exp209_transition_kernel_audit_20260725/"
    "episode_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp355_quantization_treatment_readout_20260725"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--exp209", type=Path, default=DEFAULT_EXP209)
    parser.add_argument("--exp355", type=Path, default=DEFAULT_EXP355)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--kernel-episodes",
        type=Path,
        default=DEFAULT_KERNEL_EPISODES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def iter_well_frames(
    path: Path,
    *,
    well_column: str,
    usecols: list[str],
    chunksize: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    carry: pd.DataFrame | None = None
    previous_well: str | None = None
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        dtype={well_column: str},
    ):
        chunk = chunk.rename(columns={well_column: "well"})
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        if not chunk["well"].is_monotonic_increasing:
            raise ValueError(f"{path} is not sorted by well")
        final_well = str(chunk["well"].iloc[-1])
        complete = chunk.loc[chunk["well"] != final_well]
        carry = chunk.loc[chunk["well"] == final_well].copy()
        for well, group in complete.groupby("well", sort=False):
            well = str(well)
            if previous_well is not None and well <= previous_well:
                raise ValueError(f"{path} well order is not strictly increasing")
            previous_well = well
            yield well, group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        well = str(carry["well"].iloc[0])
        if previous_well is not None and well <= previous_well:
            raise ValueError(f"{path} final well order is invalid")
        yield well, carry.reset_index(drop=True)


def rmse_from_sse(sse: float, rows: int) -> float:
    return float(np.sqrt(sse / rows))


def weighted_fraction(mask: pd.Series, weights: pd.Series) -> float:
    return float(
        np.average(
            mask.to_numpy(bool),
            weights=weights.to_numpy(np.float64),
        )
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    exp209_path = resolved(root, args.exp209)
    exp355_path = resolved(root, args.exp355)
    episodes_path = resolved(root, args.episodes)
    kernel_episodes_path = resolved(root, args.kernel_episodes)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)
    kernel_episodes = pd.read_csv(kernel_episodes_path)[
        [
            "episode_id",
            "pre128_kernel_bias_sum_ft",
        ]
    ]
    episodes = episodes.merge(
        kernel_episodes,
        on="episode_id",
        validate="one_to_one",
    )
    episodes_by_well = {
        str(well): frame.reset_index(drop=True)
        for well, frame in episodes.groupby("well", sort=False)
    }

    exp209_iterator = iter_well_frames(
        exp209_path,
        well_column="well",
        usecols=[
            "well",
            "row_idx",
            "true_tvt_readout_only",
            "posterior_mean",
        ],
        chunksize=args.chunksize,
    )
    exp355_iterator = iter_well_frames(
        exp355_path,
        well_column="well_id",
        usecols=[
            "well_id",
            "row_idx",
            "candidate_tvt",
            "mu_rate",
            "prefix_rate",
        ],
        chunksize=args.chunksize,
    )

    total_rows = 0
    exp209_total_sse = 0.0
    exp355_total_sse = 0.0
    original_episode_rows = 0
    exp209_episode_sse = 0.0
    exp355_episode_sse = 0.0
    well_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for (well, base), (candidate_well, candidate) in zip(
        exp209_iterator,
        exp355_iterator,
        strict=True,
    ):
        if well != candidate_well:
            raise ValueError(f"well mismatch: {well} != {candidate_well}")
        if not np.array_equal(
            base["row_idx"].to_numpy(np.int64),
            candidate["row_idx"].to_numpy(np.int64),
        ):
            raise ValueError(f"{well}: row_idx mismatch")
        truth = base["true_tvt_readout_only"].to_numpy(np.float64)
        exp209_prediction = base["posterior_mean"].to_numpy(np.float64)
        exp355_prediction = candidate["candidate_tvt"].to_numpy(np.float64)
        if not np.isfinite(
            np.column_stack([truth, exp209_prediction, exp355_prediction])
        ).all():
            raise ValueError(f"{well}: non-finite score input")
        exp209_error = exp209_prediction - truth
        exp355_error = exp355_prediction - truth
        exp209_sse = float(np.sum(exp209_error**2))
        exp355_sse = float(np.sum(exp355_error**2))
        original_episode_mask = np.zeros(len(base), dtype=bool)

        well_episodes = episodes_by_well.get(well)
        if well_episodes is not None:
            for episode in well_episodes.itertuples(index=False):
                start = int(episode.start_suffix_offset)
                end = start + int(episode.rows)
                original_episode_mask[start:end] = True
                base_error = exp209_error[start:end]
                treatment_error = exp355_error[start:end]
                base_sse = float(np.sum(base_error**2))
                treatment_sse = float(np.sum(treatment_error**2))
                pre128_bias = float(episode.pre128_kernel_bias_sum_ft)
                treatment_mean_error = float(np.mean(treatment_error))
                episode_rows.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "well": well,
                        "rows": int(episode.rows),
                        "episode_sse_weight": float(episode.episode_sse),
                        "observed_emission_evidence_class": str(
                            episode.observed_emission_evidence_class
                        ),
                        "viterbi_recovery_class": str(
                            episode.viterbi_recovery_class
                        ),
                        "pre128_kernel_bias_sum_ft": pre128_bias,
                        "pre128_true_minus_init_rate_median": float(
                            episode.pre128_true_minus_init_rate_median
                        ),
                        "exp209_mean_error_ft": float(
                            np.mean(base_error)
                        ),
                        "exp355_mean_error_ft": treatment_mean_error,
                        "exp209_rmse_ft": rmse_from_sse(
                            base_sse,
                            int(episode.rows),
                        ),
                        "exp355_rmse_ft": rmse_from_sse(
                            treatment_sse,
                            int(episode.rows),
                        ),
                        "rmse_gain_ft": (
                            rmse_from_sse(base_sse, int(episode.rows))
                            - rmse_from_sse(
                                treatment_sse,
                                int(episode.rows),
                            )
                        ),
                        "exp355_sse_reduction_fraction": float(
                            1.0 - treatment_sse / base_sse
                        ),
                        "pre128_bias_sign_matches_exp209_error": bool(
                            np.sign(pre128_bias)
                            == np.sign(float(np.mean(base_error)))
                        ),
                        "pre128_bias_sign_matches_exp355_error": bool(
                            np.sign(pre128_bias)
                            == np.sign(treatment_mean_error)
                        ),
                        "exp355_mu_adjustment_abs_mean": float(
                            np.mean(
                                np.abs(
                                    candidate["mu_rate"].to_numpy(
                                        np.float64
                                    )[start:end]
                                    - candidate["prefix_rate"].to_numpy(
                                        np.float64
                                    )[start:end]
                                )
                            )
                        ),
                    }
                )

        episode_base_sse = float(
            np.sum(exp209_error[original_episode_mask] ** 2)
        )
        episode_treatment_sse = float(
            np.sum(exp355_error[original_episode_mask] ** 2)
        )
        outside_rows = int((~original_episode_mask).sum())
        well_rows.append(
            {
                "well": well,
                "rows": int(len(base)),
                "original_episode_rows": int(
                    original_episode_mask.sum()
                ),
                "exp209_rmse_ft": rmse_from_sse(
                    exp209_sse,
                    len(base),
                ),
                "exp355_rmse_ft": rmse_from_sse(
                    exp355_sse,
                    len(base),
                ),
                "rmse_gain_ft": (
                    rmse_from_sse(exp209_sse, len(base))
                    - rmse_from_sse(exp355_sse, len(base))
                ),
                "original_episode_exp209_rmse_ft": (
                    rmse_from_sse(
                        episode_base_sse,
                        int(original_episode_mask.sum()),
                    )
                    if original_episode_mask.any()
                    else np.nan
                ),
                "original_episode_exp355_rmse_ft": (
                    rmse_from_sse(
                        episode_treatment_sse,
                        int(original_episode_mask.sum()),
                    )
                    if original_episode_mask.any()
                    else np.nan
                ),
                "outside_exp209_rmse_ft": rmse_from_sse(
                    exp209_sse - episode_base_sse,
                    outside_rows,
                ),
                "outside_exp355_rmse_ft": rmse_from_sse(
                    exp355_sse - episode_treatment_sse,
                    outside_rows,
                ),
            }
        )
        total_rows += len(base)
        exp209_total_sse += exp209_sse
        exp355_total_sse += exp355_sse
        original_episode_rows += int(original_episode_mask.sum())
        exp209_episode_sse += episode_base_sse
        exp355_episode_sse += episode_treatment_sse

    by_well = pd.DataFrame(well_rows).sort_values("well").reset_index(drop=True)
    by_episode = (
        pd.DataFrame(episode_rows)
        .sort_values(["well", "episode_id"])
        .reset_index(drop=True)
    )
    outside_rows = total_rows - original_episode_rows
    exp209_outside_sse = exp209_total_sse - exp209_episode_sse
    exp355_outside_sse = exp355_total_sse - exp355_episode_sse

    class_rows: list[dict[str, Any]] = []
    for label, group in by_episode.groupby(
        "observed_emission_evidence_class",
        sort=True,
    ):
        class_rows_total = int(group["rows"].sum())
        class_exp209_sse = float(
            np.sum(group["exp209_rmse_ft"] ** 2 * group["rows"])
        )
        class_exp355_sse = float(
            np.sum(group["exp355_rmse_ft"] ** 2 * group["rows"])
        )
        class_rows.append(
            {
                "observed_emission_evidence_class": str(label),
                "episodes": int(len(group)),
                "rows": class_rows_total,
                "exp209_pooled_rmse_ft": rmse_from_sse(
                    class_exp209_sse,
                    class_rows_total,
                ),
                "exp355_pooled_rmse_ft": rmse_from_sse(
                    class_exp355_sse,
                    class_rows_total,
                ),
                "sse_reduction_fraction": float(
                    1.0 - class_exp355_sse / class_exp209_sse
                ),
                "mean_rmse_gain_ft": float(group["rmse_gain_ft"].mean()),
                "median_rmse_gain_ft": float(
                    group["rmse_gain_ft"].median()
                ),
                "improved_episode_fraction": float(
                    (group["rmse_gain_ft"] > 0).mean()
                ),
                "kernel_bias_vs_exp209_mean_error_spearman": float(
                    group["pre128_kernel_bias_sum_ft"].corr(
                        group["exp209_mean_error_ft"],
                        method="spearman",
                    )
                ),
                "kernel_bias_vs_exp355_mean_error_spearman": float(
                    group["pre128_kernel_bias_sum_ft"].corr(
                        group["exp355_mean_error_ft"],
                        method="spearman",
                    )
                ),
                "exp209_bias_sign_match_fraction": float(
                    group[
                        "pre128_bias_sign_matches_exp209_error"
                    ].mean()
                ),
                "exp355_bias_sign_match_fraction": float(
                    group[
                        "pre128_bias_sign_matches_exp355_error"
                    ].mean()
                ),
            }
        )
    class_summary = pd.DataFrame(class_rows)

    summary = {
        "scope": {
            "rows": int(total_rows),
            "wells": int(len(by_well)),
            "original_persistent_episode_rows": int(
                original_episode_rows
            ),
            "original_persistent_episodes": int(len(by_episode)),
        },
        "overall": {
            "exp209_rmse_ft": rmse_from_sse(
                exp209_total_sse,
                total_rows,
            ),
            "exp355_rmse_ft": rmse_from_sse(
                exp355_total_sse,
                total_rows,
            ),
            "rmse_gain_ft": (
                rmse_from_sse(exp209_total_sse, total_rows)
                - rmse_from_sse(exp355_total_sse, total_rows)
            ),
            "sse_reduction": float(
                exp209_total_sse - exp355_total_sse
            ),
        },
        "original_exp209_persistent_rows": {
            "exp209_rmse_ft": rmse_from_sse(
                exp209_episode_sse,
                original_episode_rows,
            ),
            "exp355_rmse_ft": rmse_from_sse(
                exp355_episode_sse,
                original_episode_rows,
            ),
            "sse_reduction": float(
                exp209_episode_sse - exp355_episode_sse
            ),
            "sse_reduction_fraction": float(
                1.0 - exp355_episode_sse / exp209_episode_sse
            ),
        },
        "outside_original_exp209_persistent_rows": {
            "rows": int(outside_rows),
            "exp209_rmse_ft": rmse_from_sse(
                exp209_outside_sse,
                outside_rows,
            ),
            "exp355_rmse_ft": rmse_from_sse(
                exp355_outside_sse,
                outside_rows,
            ),
            "sse_increase": float(
                exp355_outside_sse - exp209_outside_sse
            ),
        },
        "episode_association": {
            "mean_rmse_gain_ft": float(
                by_episode["rmse_gain_ft"].mean()
            ),
            "median_rmse_gain_ft": float(
                by_episode["rmse_gain_ft"].median()
            ),
            "improved_episodes": int(
                (by_episode["rmse_gain_ft"] > 0).sum()
            ),
            "kernel_bias_vs_exp209_mean_error_spearman": float(
                by_episode["pre128_kernel_bias_sum_ft"].corr(
                    by_episode["exp209_mean_error_ft"],
                    method="spearman",
                )
            ),
            "kernel_bias_vs_exp355_mean_error_spearman": float(
                by_episode["pre128_kernel_bias_sum_ft"].corr(
                    by_episode["exp355_mean_error_ft"],
                    method="spearman",
                )
            ),
            "exp209_bias_sign_match_fraction": float(
                by_episode[
                    "pre128_bias_sign_matches_exp209_error"
                ].mean()
            ),
            "exp355_bias_sign_match_fraction": float(
                by_episode[
                    "pre128_bias_sign_matches_exp355_error"
                ].mean()
            ),
            "exp209_bias_sign_match_sse_weighted": weighted_fraction(
                by_episode[
                    "pre128_bias_sign_matches_exp209_error"
                ],
                by_episode["episode_sse_weight"],
            ),
            "exp355_bias_sign_match_sse_weighted": weighted_fraction(
                by_episode[
                    "pre128_bias_sign_matches_exp355_error"
                ],
                by_episode["episode_sse_weight"],
            ),
        },
        "interpretation_guard": (
            "The original episode mask is defined only from saved exp209. "
            "This treatment readout supports transition-mean causality but "
            "does not isolate position sigma, momentum, or backward messages."
        ),
    }

    by_well.to_csv(output / "by_well_metrics.csv", index=False)
    by_episode.to_csv(output / "episode_metrics.csv", index=False)
    class_summary.to_csv(output / "emission_class_summary.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
