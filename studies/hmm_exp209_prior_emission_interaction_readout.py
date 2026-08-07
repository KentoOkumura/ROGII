#!/usr/bin/env python3
"""Separate prefix-prior direction, GR evidence, and position shrinkage.

This is a truth-late diagnostic over already saved exp209 audit artifacts.  It
does not rerun an HMM or create a prediction candidate.  The readout asks:

1. Does the transition-only prior point in the same direction as the actual
   persistent HMM offset?
2. Is that relation stable across observed-GR evidence classes?
3. Does the current-minus-exact-mean position effect reinforce or oppose the
   actual offset once the prefix-prior direction is considered?

The result is used to specify the minimum internal messages required by a
future current-control HMM diagnostic.
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

DEFAULT_PRIOR_EPISODES = Path(
    "studies/hmm_exp209_actual_geometry_transition_prior_20260726/"
    "episode_metrics.csv"
)
DEFAULT_CAUSE_EPISODES = Path(
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_KERNEL_EPISODES = Path(
    "studies/hmm_exp209_transition_kernel_audit_20260725/"
    "episode_metrics.csv"
)
DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_prior_emission_interaction_readout_20260726"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--prior-episodes", type=Path, default=DEFAULT_PRIOR_EPISODES)
    parser.add_argument("--cause-episodes", type=Path, default=DEFAULT_CAUSE_EPISODES)
    parser.add_argument("--kernel-episodes", type=Path, default=DEFAULT_KERNEL_EPISODES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
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


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(finite)) < 3:
        return float("nan")
    left_values = left.loc[finite].to_numpy(np.float64)
    right_values = right.loc[finite].to_numpy(np.float64)
    if np.ptp(left_values) == 0.0 or np.ptp(right_values) == 0.0:
        return float("nan")
    return float(spearmanr(left_values, right_values).statistic)


def sign_match(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(
        np.sign(left.to_numpy(np.float64))
        == np.sign(right.to_numpy(np.float64)),
        index=left.index,
    )


def weighted_fraction(mask: pd.Series, weight: pd.Series) -> float:
    weights = weight.to_numpy(np.float64)
    return float(np.sum(weights[mask.to_numpy(bool)]) / np.sum(weights))


def weighted_class_fraction(
    frame: pd.DataFrame,
    column: str,
    label: str,
) -> float:
    return weighted_fraction(
        frame[column] == label,
        frame["episode_sse"],
    )


def partial_rank_corr(
    left: pd.Series,
    right: pd.Series,
    controls: list[pd.Series],
) -> float:
    columns = [left.rename("left"), right.rename("right")]
    columns.extend(
        control.rename(f"control_{index}")
        for index, control in enumerate(controls)
    )
    ranked = pd.concat(columns, axis=1).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna().rank()
    design = np.column_stack(
        [
            np.ones(len(ranked), dtype=np.float64),
            *[
                ranked[f"control_{index}"].to_numpy(np.float64)
                for index in range(len(controls))
            ],
        ]
    )
    left_values = ranked["left"].to_numpy(np.float64)
    right_values = ranked["right"].to_numpy(np.float64)
    left_residual = left_values - design @ np.linalg.lstsq(
        design,
        left_values,
        rcond=None,
    )[0]
    right_residual = right_values - design @ np.linalg.lstsq(
        design,
        right_values,
        rcond=None,
    )[0]
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def summarize_group(group: pd.DataFrame, total_sse: float) -> dict[str, Any]:
    actual = group["actual_mean_error_ft"]
    current = group["current_mean_error_ft"]
    position_effect = group[
        "position_quantization_episode_mean_effect_ft"
    ]
    prior_match = sign_match(current, actual)
    position_match = sign_match(position_effect, actual)
    actual_abs_better = actual.abs() < current.abs()
    viterbi_recoverable = group["viterbi_recovery_class"] != "not_better"
    viterbi_error = group["viterbi_mean_error_ft"]
    viterbi_sign_match = sign_match(viterbi_error, actual)
    viterbi_abs_mean_better = viterbi_error.abs() < actual.abs()
    pre128_slope = group["pre128_error_slope_ft_per_row"]
    episode_slope = group["error_slope_ft_per_row"]
    pre128_change = (
        group["pre128_error_end_ft"] - group["pre128_error_start_ft"]
    )
    return {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(group["episode_sse"].sum() / total_sse),
        "current_prior_vs_actual_spearman": safe_spearman(current, actual),
        "current_prior_sign_match_fraction": float(prior_match.mean()),
        "current_prior_sign_match_sse_fraction_within_group": weighted_fraction(
            prior_match,
            group["episode_sse"],
        ),
        "position_effect_vs_actual_spearman": safe_spearman(
            position_effect,
            actual,
        ),
        "position_effect_sign_match_fraction": float(position_match.mean()),
        "actual_abs_better_than_prior_fraction": float(
            actual_abs_better.mean()
        ),
        "viterbi_recoverable_fraction": float(viterbi_recoverable.mean()),
        "viterbi_recoverable_sse_fraction_within_group": weighted_fraction(
            viterbi_recoverable,
            group["episode_sse"],
        ),
        "viterbi_mean_error_sign_matches_actual_fraction": float(
            viterbi_sign_match.mean()
        ),
        "viterbi_abs_mean_error_better_fraction": float(
            viterbi_abs_mean_better.mean()
        ),
        "viterbi_abs_mean_error_better_sse_fraction_within_group": (
            weighted_fraction(
                viterbi_abs_mean_better,
                group["episode_sse"],
            )
        ),
        "viterbi_abs_mean_error_within5_fraction": float(
            (viterbi_error.abs() <= 5.0).mean()
        ),
        "all_row_emission_truth_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "emission_evidence_class",
                "truth_strong",
            )
        ),
        "all_row_emission_candidate_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "emission_evidence_class",
                "candidate_strong",
            )
        ),
        "imputed_emission_truth_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "imputed_emission_evidence_class",
                "truth_strong",
            )
        ),
        "imputed_emission_candidate_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "imputed_emission_evidence_class",
                "candidate_strong",
            )
        ),
        "affine_observed_truth_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "affine_observed_emission_evidence_class",
                "truth_strong",
            )
        ),
        "affine_observed_candidate_strong_sse_fraction_within_group": (
            weighted_class_fraction(
                group,
                "affine_observed_emission_evidence_class",
                "candidate_strong",
            )
        ),
        "pre128_error_slope_abs_median_ft_per_row": float(
            pre128_slope.abs().median()
        ),
        "episode_error_slope_abs_median_ft_per_row": float(
            episode_slope.abs().median()
        ),
        "pre128_error_change_abs_median_ft": float(
            pre128_change.abs().median()
        ),
        "pre128_slope_sign_matches_actual_fraction": float(
            sign_match(pre128_slope, actual).mean()
        ),
        "current_prior_abs_error_median_ft": float(current.abs().median()),
        "actual_abs_error_median_ft": float(actual.abs().median()),
        "actual_to_current_prior_abs_ratio_median": float(
            (actual.abs() / current.abs().clip(lower=1e-9)).median()
        ),
    }


def cause_bucket(
    evidence_class: str,
    prior_matches_actual: bool,
) -> str:
    mapping = {
        (True, "candidate_strong"): "prior_and_wrong_gr_reinforce",
        (True, "near_tie"): "prior_aligned_gr_ambiguous",
        (True, "truth_strong"): "prior_persists_against_truth_gr",
        (False, "candidate_strong"): "candidate_gr_with_opposed_prior",
        (False, "near_tie"): "opposed_prior_gr_ambiguous",
        (False, "truth_strong"): "neither_prior_nor_observed_gr",
    }
    return mapping[(prior_matches_actual, evidence_class)]


def decoder_signed_episode_metrics(
    candidates_path: Path,
    episodes: pd.DataFrame,
    chunksize: int,
) -> pd.DataFrame:
    episode_specs = {
        str(well): group[
            [
                "episode_id",
                "start_row_idx",
                "end_row_idx_exclusive",
                "rows",
            ]
        ].to_dict("records")
        for well, group in episodes.groupby("well", sort=False)
    }
    accumulator = {
        str(episode_id): {
            "well": str(well),
            "rows": 0,
            "posterior_mean_error_sum": 0.0,
            "marginal_map_error_sum": 0.0,
            "viterbi_error_sum": 0.0,
        }
        for episode_id, well in episodes[
            ["episode_id", "well"]
        ].itertuples(index=False, name=None)
    }
    usecols = [
        "well",
        "row_idx",
        "true_tvt_readout_only",
        "posterior_mean",
        "marginal_map",
        "topk_path_1",
    ]
    for chunk in pd.read_csv(
        candidates_path,
        usecols=usecols,
        chunksize=chunksize,
    ):
        chunk["well"] = chunk["well"].astype(str)
        for well, group in chunk.groupby("well", sort=False):
            specs = episode_specs.get(str(well))
            if specs is None:
                continue
            row_index = pd.to_numeric(
                group["row_idx"],
                errors="raise",
            ).to_numpy(np.int64)
            truth = pd.to_numeric(
                group["true_tvt_readout_only"],
                errors="raise",
            ).to_numpy(np.float64)
            for spec in specs:
                mask = (
                    (row_index >= int(spec["start_row_idx"]))
                    & (row_index < int(spec["end_row_idx_exclusive"]))
                )
                if not np.any(mask):
                    continue
                record = accumulator[str(spec["episode_id"])]
                record["rows"] += int(np.sum(mask))
                record["posterior_mean_error_sum"] += float(
                    np.sum(
                        pd.to_numeric(
                            group.loc[mask, "posterior_mean"],
                            errors="raise",
                        ).to_numpy(np.float64)
                        - truth[mask]
                    )
                )
                record["marginal_map_error_sum"] += float(
                    np.sum(
                        pd.to_numeric(
                            group.loc[mask, "marginal_map"],
                            errors="raise",
                        ).to_numpy(np.float64)
                        - truth[mask]
                    )
                )
                record["viterbi_error_sum"] += float(
                    np.sum(
                        pd.to_numeric(
                            group.loc[mask, "topk_path_1"],
                            errors="raise",
                        ).to_numpy(np.float64)
                        - truth[mask]
                    )
                )
    rows: list[dict[str, Any]] = []
    expected_rows = episodes.set_index("episode_id")["rows"].to_dict()
    for episode_id, record in accumulator.items():
        count = int(record["rows"])
        if count != int(expected_rows[episode_id]):
            raise ValueError(
                f"{episode_id}: decoder rows {count} != "
                f"{int(expected_rows[episode_id])}"
            )
        rows.append(
            {
                "episode_id": episode_id,
                "well": record["well"],
                "decoder_rows": count,
                "posterior_mean_error_recomputed_ft": (
                    record["posterior_mean_error_sum"] / count
                ),
                "marginal_map_mean_error_ft": (
                    record["marginal_map_error_sum"] / count
                ),
                "viterbi_mean_error_ft": (
                    record["viterbi_error_sum"] / count
                ),
            }
        )
    return pd.DataFrame(rows)


def quartile_summary(
    frame: pd.DataFrame,
    column: str,
    total_sse: float,
) -> list[dict[str, Any]]:
    bucket = pd.qcut(frame[column], 4, duplicates="drop")
    rows: list[dict[str, Any]] = []
    for interval, group in frame.groupby(bucket, observed=False):
        summary = summarize_group(group, total_sse)
        rows.append(
            {
                "stratifier": column,
                "bucket": str(interval),
                "value_min": float(group[column].min()),
                "value_max": float(group[column].max()),
                **summary,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    prior_path = resolve(root, args.prior_episodes)
    cause_path = resolve(root, args.cause_episodes)
    kernel_path = resolve(root, args.kernel_episodes)
    candidates_path = resolve(root, args.candidates)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    prior = pd.read_csv(prior_path)
    cause = pd.read_csv(cause_path)
    kernel = pd.read_csv(kernel_path)
    decoder_signed = decoder_signed_episode_metrics(
        candidates_path,
        cause,
        int(args.chunksize),
    )
    key = ["episode_id", "well"]
    for name, frame in (
        ("prior", prior),
        ("cause", cause),
        ("kernel", kernel),
    ):
        if frame.duplicated(key).any():
            raise ValueError(f"{name} contains duplicate episode keys")
    joined = (
        prior.merge(
            cause,
            on=key,
            suffixes=("", "_cause"),
            validate="one_to_one",
        )
        .merge(
            kernel,
            on=key,
            suffixes=("", "_kernel"),
            validate="one_to_one",
        )
        .merge(
            decoder_signed,
            on=key,
            validate="one_to_one",
        )
        .copy()
    )
    expected_keys = set(map(tuple, prior[key].to_numpy()))
    if set(map(tuple, joined[key].to_numpy())) != expected_keys:
        raise ValueError("episode join changed the prior episode key set")
    if not len(prior) == len(cause) == len(kernel) == len(joined):
        raise ValueError("episode input row counts differ")
    decoder_mean_parity = float(
        np.max(
            np.abs(
                joined["posterior_mean_error_recomputed_ft"]
                - joined["actual_mean_error_ft"]
            )
        )
    )
    if decoder_mean_parity > 1e-9:
        raise ValueError(
            "recomputed posterior mean episode error parity failed: "
            f"{decoder_mean_parity}"
        )
    decoder_signed.to_csv(
        output / "decoder_signed_episode_metrics.csv",
        index=False,
    )

    actual = joined["actual_mean_error_ft"]
    current = joined["current_mean_error_ft"]
    exact = joined["exact_mean_position_mean_error_ft"]
    position_effect = joined[
        "position_quantization_episode_mean_effect_ft"
    ]
    truth_kernel = joined["pre128_kernel_bias_sum_ft"]
    rate_mismatch = joined["pre128_true_minus_init_rate_median"]
    observed_nll = joined["observed_total_truth_minus_candidate_nll"]
    total_sse = float(joined["episode_sse"].sum())

    joined["current_prior_sign_matches_actual"] = sign_match(
        current,
        actual,
    )
    joined["position_effect_sign_matches_actual_recomputed"] = sign_match(
        position_effect,
        actual,
    )
    joined["actual_abs_better_than_current_prior"] = (
        actual.abs() < current.abs()
    )
    joined["imputed_emission_evidence_class"] = np.select(
        [
            joined["imputed_total_truth_minus_candidate_nll"] > 5.0,
            joined["imputed_total_truth_minus_candidate_nll"] < -5.0,
        ],
        ["candidate_strong", "truth_strong"],
        default="near_tie",
    )

    regime_rows: list[dict[str, Any]] = []
    for (evidence_class, prior_match), group in joined.groupby(
        [
            "observed_emission_evidence_class",
            "current_prior_sign_matches_actual",
        ],
        sort=True,
    ):
        regime_rows.append(
            {
                "cause_bucket": cause_bucket(
                    str(evidence_class),
                    bool(prior_match),
                ),
                "observed_emission_evidence_class": evidence_class,
                "current_prior_sign_matches_actual": bool(prior_match),
                **summarize_group(group, total_sse),
            }
        )
    regime_frame = pd.DataFrame(regime_rows)
    regime_frame.to_csv(output / "prior_gr_regime_summary.csv", index=False)

    evidence_rows = []
    for evidence_class, group in joined.groupby(
        "observed_emission_evidence_class",
        sort=True,
    ):
        evidence_rows.append(
            {
                "observed_emission_evidence_class": evidence_class,
                **summarize_group(group, total_sse),
            }
        )
    evidence_frame = pd.DataFrame(evidence_rows)
    evidence_frame.to_csv(
        output / "observed_gr_class_summary.csv",
        index=False,
    )

    prior_aligned = joined.loc[
        joined["current_prior_sign_matches_actual"]
    ]
    prior_opposed = joined.loc[
        ~joined["current_prior_sign_matches_actual"]
    ]
    stratifier_rows: list[dict[str, Any]] = []
    for column in (
        "rows_from_last_within5_to_episode_start",
        "start_suffix_fraction",
        "episode_suffix_fraction",
        "raw_gr_missing_fraction",
        "prefix_sigma_gr",
    ):
        stratifier_rows.extend(
            quartile_summary(joined, column, total_sse)
        )
    pd.DataFrame(stratifier_rows).to_csv(
        output / "prior_alignment_quartile_summary.csv",
        index=False,
    )

    actual_better = joined["actual_abs_better_than_current_prior"]
    current_match = joined["current_prior_sign_matches_actual"]
    exact_match = sign_match(exact, actual)
    position_match = joined[
        "position_effect_sign_matches_actual_recomputed"
    ]
    decoder_signed_summary: dict[str, Any] = {}
    for label, column in (
        ("marginal_map", "marginal_map_mean_error_ft"),
        ("global_viterbi", "viterbi_mean_error_ft"),
    ):
        decoder_error = joined[column]
        decoder_sign_match = sign_match(decoder_error, actual)
        decoder_abs_better = decoder_error.abs() < actual.abs()
        decoder_within5 = decoder_error.abs() <= 5.0
        decoder_signed_summary[label] = {
            "mean_error_sign_matches_posterior_mean_fraction": float(
                decoder_sign_match.mean()
            ),
            "mean_error_sign_matches_posterior_mean_sse_fraction": (
                weighted_fraction(
                    decoder_sign_match,
                    joined["episode_sse"],
                )
            ),
            "abs_mean_error_better_fraction": float(
                decoder_abs_better.mean()
            ),
            "abs_mean_error_better_sse_fraction": weighted_fraction(
                decoder_abs_better,
                joined["episode_sse"],
            ),
            "abs_mean_error_within5_fraction": float(
                decoder_within5.mean()
            ),
            "abs_mean_error_within5_sse_fraction": weighted_fraction(
                decoder_within5,
                joined["episode_sse"],
            ),
        }
    transition_variant_rows: list[dict[str, Any]] = []
    for variant, column in (
        ("current", "current_mean_error_ft"),
        ("exact_mean_position", "exact_mean_position_mean_error_ft"),
        ("boundary_normalized", "boundary_normalized_mean_error_ft"),
        ("momentum_one", "momentum_one_mean_error_ft"),
        (
            "exact_mean_plus_momentum_one",
            "exact_mean_plus_momentum_one_mean_error_ft",
        ),
        (
            "boundary_plus_momentum_one",
            "boundary_plus_momentum_one_mean_error_ft",
        ),
        ("all_three_corrections", "all_three_corrections_mean_error_ft"),
    ):
        variant_error = joined[column]
        variant_match = sign_match(variant_error, actual)
        closer_than_current = variant_error.abs() < current.abs()
        transition_variant_rows.append(
            {
                "variant": variant,
                "abs_episode_mean_error_median_ft": float(
                    variant_error.abs().median()
                ),
                "abs_episode_mean_error_p90_ft": float(
                    variant_error.abs().quantile(0.9)
                ),
                "variant_error_vs_actual_spearman": safe_spearman(
                    variant_error,
                    actual,
                ),
                "variant_error_sign_matches_actual_fraction": float(
                    variant_match.mean()
                ),
                "variant_closer_to_truth_than_current_fraction": float(
                    closer_than_current.mean()
                ),
                "variant_closer_to_truth_than_current_sse_fraction": (
                    weighted_fraction(
                        closer_than_current,
                        joined["episode_sse"],
                    )
                ),
            }
        )
    pd.DataFrame(transition_variant_rows).to_csv(
        output / "transition_variant_episode_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(joined)),
            "wells": int(joined["well"].nunique()),
            "episode_sse": total_sse,
            "input_rows": {
                "prior": int(len(prior)),
                "cause": int(len(cause)),
                "kernel": int(len(kernel)),
            },
            "key_sets_identical": True,
            "decoder_mean_error_parity_max_abs_ft": decoder_mean_parity,
        },
        "source_sha256": {
            str(prior_path.relative_to(root)): sha256(prior_path),
            str(cause_path.relative_to(root)): sha256(cause_path),
            str(kernel_path.relative_to(root)): sha256(kernel_path),
            str(candidates_path.relative_to(root)): sha256(candidates_path),
        },
        "transition_only_prior_alignment": {
            "current_prior_vs_actual_spearman": safe_spearman(
                current,
                actual,
            ),
            "current_prior_sign_match_fraction": float(
                current_match.mean()
            ),
            "current_prior_sign_match_sse_fraction": weighted_fraction(
                current_match,
                joined["episode_sse"],
            ),
            "current_prior_abs_error_quantiles_ft": {
                "median": float(current.abs().median()),
                "p90": float(current.abs().quantile(0.9)),
            },
            "actual_abs_error_quantiles_ft": {
                "median": float(actual.abs().median()),
                "p90": float(actual.abs().quantile(0.9)),
            },
            "actual_abs_better_than_prior_fraction": float(
                actual_better.mean()
            ),
            "actual_abs_better_than_prior_sse_fraction": weighted_fraction(
                actual_better,
                joined["episode_sse"],
            ),
            "exact_mean_prior_vs_actual_spearman": safe_spearman(
                exact,
                actual,
            ),
            "exact_mean_prior_sign_match_fraction": float(
                exact_match.mean()
            ),
            "exact_mean_prior_abs_error_quantiles_ft": {
                "median": float(exact.abs().median()),
                "p90": float(exact.abs().quantile(0.9)),
            },
        },
        "position_shrinkage_effect": {
            "definition": (
                "current transition-only prior minus exact-mean "
                "transition-only prior"
            ),
            "effect_vs_actual_spearman": safe_spearman(
                position_effect,
                actual,
            ),
            "effect_sign_match_fraction": float(position_match.mean()),
            "effect_sign_match_sse_fraction": weighted_fraction(
                position_match,
                joined["episode_sse"],
            ),
            "effect_vs_current_prior_spearman": safe_spearman(
                position_effect,
                current,
            ),
            "prior_aligned_group": {
                **summarize_group(prior_aligned, total_sse),
                "position_effect_sign_match_sse_fraction_within_group": (
                    weighted_fraction(
                        sign_match(
                            prior_aligned[
                                "position_quantization_episode_mean_effect_ft"
                            ],
                            prior_aligned["actual_mean_error_ft"],
                        ),
                        prior_aligned["episode_sse"],
                    )
                ),
            },
            "prior_opposed_group": {
                **summarize_group(prior_opposed, total_sse),
                "position_effect_sign_match_sse_fraction_within_group": (
                    weighted_fraction(
                        sign_match(
                            prior_opposed[
                                "position_quantization_episode_mean_effect_ft"
                            ],
                            prior_opposed["actual_mean_error_ft"],
                        ),
                        prior_opposed["episode_sse"],
                    )
                ),
            },
        },
        "rate_dynamics_joint_effect": {
            "definition": (
                "current prior minus boundary-normalized momentum-one prior, "
                "with the current position kernel retained"
            ),
            "effect_abs_median_ft": float(
                (
                    current
                    - joined["boundary_plus_momentum_one_mean_error_ft"]
                ).abs().median()
            ),
            "effect_vs_actual_spearman": safe_spearman(
                current
                - joined["boundary_plus_momentum_one_mean_error_ft"],
                actual,
            ),
            "effect_sign_match_fraction": float(
                sign_match(
                    current
                    - joined["boundary_plus_momentum_one_mean_error_ft"],
                    actual,
                ).mean()
            ),
        },
        "ramp_then_parallel_lock": {
            "pre128_error_slope_abs_median_ft_per_row": float(
                joined["pre128_error_slope_ft_per_row"].abs().median()
            ),
            "episode_error_slope_abs_median_ft_per_row": float(
                joined["error_slope_ft_per_row"].abs().median()
            ),
            "pre128_to_episode_abs_slope_ratio": float(
                joined["pre128_error_slope_ft_per_row"].abs().median()
                / joined["error_slope_ft_per_row"].abs().median()
            ),
            "pre128_error_change_abs_median_ft": float(
                (
                    joined["pre128_error_end_ft"]
                    - joined["pre128_error_start_ft"]
                ).abs().median()
            ),
            "pre128_slope_vs_actual_spearman": safe_spearman(
                joined["pre128_error_slope_ft_per_row"],
                actual,
            ),
            "pre128_slope_sign_match_fraction": float(
                sign_match(
                    joined["pre128_error_slope_ft_per_row"],
                    actual,
                ).mean()
            ),
            "pre128_slope_sign_match_sse_fraction": weighted_fraction(
                sign_match(
                    joined["pre128_error_slope_ft_per_row"],
                    actual,
                ),
                joined["episode_sse"],
            ),
            "selection_warning": (
                "Persistent episodes are defined after crossing an absolute "
                "error threshold, so pre-onset sign agreement is partly "
                "selection-induced. The slope collapse after onset is the "
                "independent temporal observation."
            ),
        },
        "truth_centered_vs_actual_prefix_confounding": {
            "truth_centered_kernel_bias_vs_actual_spearman": safe_spearman(
                truth_kernel,
                actual,
            ),
            "truth_centered_partial_controlling_current_prior_rate_mismatch_and_gr_nll": (
                partial_rank_corr(
                    truth_kernel,
                    actual,
                    [current, rate_mismatch, observed_nll],
                )
            ),
            "current_prior_partial_with_other_diagnostics": (
                partial_rank_corr(
                    current,
                    actual,
                    [
                        truth_kernel,
                        rate_mismatch,
                        observed_nll,
                        position_effect,
                    ],
                )
            ),
            "position_effect_partial_with_other_diagnostics": (
                partial_rank_corr(
                    position_effect,
                    actual,
                    [current, truth_kernel, rate_mismatch, observed_nll],
                )
            ),
            "warning": (
                "These rank partials are descriptive under strong "
                "collinearity; the frozen source-rate one-step transition "
                "moment in an actual HMM pass is the required causal readout."
            ),
        },
        "observed_gr_class_summary": evidence_rows,
        "exclusive_cause_taxonomy": regime_rows,
        "decoder_signed_attribution": decoder_signed_summary,
        "transition_variant_episode_summary": transition_variant_rows,
        "required_actual_hmm_diagnostic": {
            "priority": (
                "Run the current transition only before selecting a "
                "position or momentum intervention."
            ),
            "per_row_before_truth_join": [
                "predictive position mean/std and candidate/Viterbi basin mass",
                "filtered position mean/std and candidate/Viterbi basin mass",
                "smoothed position mean/std and candidate/Viterbi basin mass",
                "predictive/filtered/smoothed rate mean/std and edge mass",
                "predictive/filtered/smoothed rate mass near candidate path rates",
                "GR-emission change in each basin log-odds",
                "backward-message change in each basin log-odds",
                (
                    "current and exact-mean expected position increment "
                    "under the same actual filtered source-rate mass"
                ),
                "position-rate covariance and conditional rate mean per basin",
            ],
            "late_truth_attribution_after_message_freeze": [
                "truth-neighborhood predictive/filtered/smoothed position mass",
                "predictive/filtered/smoothed rate mass near the truth path rate",
                "episode/error metrics joined only to the frozen message ledger",
            ],
            "key_identification_rule": (
                "Freeze row keys, prediction, messages, and target-free "
                "candidate/Viterbi basin definitions before loading suffix "
                "truth/error; define truth neighborhoods only in the late "
                "attribution pass."
            ),
            "why_frozen_source_rate_is_required": (
                "It isolates the one-step position-kernel moment from rate "
                "posterior feedback, which the truth-centered and "
                "prefix-only counterfactuals cannot do."
            ),
        },
        "interpretation": {
            "main": (
                "The current prefix transition-only prior already points in "
                "the actual offset direction for the dominant SSE regime, "
                "and the HMM usually reduces rather than creates its "
                "magnitude."
            ),
            "position": (
                "Current position shrinkage opposes the prefix-prior error in "
                "the dominant regime; exact-mean correction is therefore not "
                "a justified unconditional fix."
            ),
            "gr": (
                "Prior alignment is present in candidate-strong, truth-strong, "
                "and near-tie GR groups, so GR evidence changes persistence "
                "and correction but does not define the initial direction by "
                "itself."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
