"""Analyze exp408 episode and row ledgers without loading the wide CSV at once."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EFFECT = math.log(3.0)

ROW_USECOLS = [
    "episode_id",
    "mean_error_ft",
    "viterbi_error_ft",
    "posterior_mean",
    "global_viterbi",
    "raw_gr_missing",
    "true_rate",
    "mean_path_rate",
    "truth_rate_support_inside",
    "predictive__position_std",
    "predictive__rate_mean",
    "predictive__rate_std",
    "predictive__position_rate_covariance",
    "predictive__position_edge_mass",
    "predictive__rate_edge_mass",
    "predictive__truth_position_mass",
    "predictive__truth_rate_near_mass",
    "filtered__position_std",
    "filtered__rate_mean",
    "filtered__rate_std",
    "filtered__position_rate_covariance",
    "filtered__position_edge_mass",
    "filtered__rate_edge_mass",
    "filtered__logsum_minus_max",
    "filtered__truth_position_mass",
    "filtered__truth_rate_near_mass",
    "smoothed__position_std",
    "smoothed__rate_mean",
    "smoothed__rate_std",
    "smoothed__position_rate_covariance",
    "smoothed__position_edge_mass",
    "smoothed__rate_edge_mass",
    "smoothed__truth_position_mass",
    "smoothed__truth_rate_near_mass",
    "predictive__truth_vs_mean_position_logit",
    "filtered__truth_vs_mean_position_logit",
    "smoothed__truth_vs_mean_position_logit",
    "emission__truth_vs_mean_logit_delta",
    "beta__truth_vs_mean_logit_delta",
    "current_expected_displacement_ft",
    "exact_mean_expected_displacement_ft",
    "current_minus_exact_mean_ft",
    "current_expected_variance_ft2",
    "rate_transition_survival_mass",
    "destination_rate_mean",
    "destination_rate_variance",
    "emission_ll__truth_minus_mean",
    "truth_position_support_inside",
    "compensating_rate_distance_from_filtered_mean",
    "compensating_rate_is_edge",
    "true_displacement_ft",
]

CONTINUOUS_COLUMNS = [
    "predictive__truth_position_mass",
    "filtered__truth_position_mass",
    "smoothed__truth_position_mass",
    "predictive__truth_rate_near_mass",
    "filtered__truth_rate_near_mass",
    "smoothed__truth_rate_near_mass",
    "predictive__truth_vs_mean_position_logit",
    "filtered__truth_vs_mean_position_logit",
    "smoothed__truth_vs_mean_position_logit",
    "emission__truth_vs_mean_logit_delta",
    "beta__truth_vs_mean_logit_delta",
    "emission_ll__truth_minus_mean",
    "filtered__logsum_minus_max",
    "predictive__position_std",
    "filtered__position_std",
    "smoothed__position_std",
    "predictive__rate_std",
    "filtered__rate_std",
    "smoothed__rate_std",
    "predictive__position_rate_covariance",
    "filtered__position_rate_covariance",
    "smoothed__position_rate_covariance",
    "current_minus_exact_mean_ft",
    "current_expected_variance_ft2",
    "rate_transition_survival_mass",
    "destination_rate_variance",
    "compensating_rate_distance_from_filtered_mean",
    "filtered_rate_error",
    "mean_path_rate_error",
    "current_displacement_error",
    "exact_mean_displacement_error",
    "backward_truth_position_mass_delta",
    "backward_truth_rate_mass_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-summary", required=True, type=Path)
    parser.add_argument("--row-ledger", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=20_000)
    return parser.parse_args()


def sse_fraction(frame: pd.DataFrame, mask: pd.Series) -> float:
    total = float(frame["episode_sse"].sum())
    return float(frame.loc[mask, "episode_sse"].sum() / total)


def episode_readout(episodes: pd.DataFrame) -> dict[str, Any]:
    flags = {
        "state_support_shortage": episodes["state_support_shortage_fraction"] >= 0.10,
        "backward_smoothing_reversal": episodes["backward_reversal_fraction"]
        >= 0.50,
        "raw_gr_alias": episodes["raw_gr_alias_fraction_observed"] >= 0.50,
        "imputation_alias": (
            (episodes["imputation_alias_fraction_missing"] >= 0.50)
            & (episodes["raw_gr_missing_fraction"] > 0.0)
        ),
        "forward_transition_prior_hysteresis": episodes[
            "forward_hysteresis_fraction"
        ]
        >= 0.50,
        "sum_product_path_multiplicity": (
            (episodes["viterbi_rmse_gain_ft"] >= 2.0)
            & (episodes["mean_viterbi_abs_gap_ft"] >= 6.0)
            & (episodes["filtered_logsum_minus_max_median"] >= 2.0)
        ),
    }
    independent = []
    for name, mask in flags.items():
        independent.append(
            {
                "condition": name,
                "episodes": int(mask.sum()),
                "episode_fraction": float(mask.mean()),
                "wells": int(episodes.loc[mask, "well"].nunique()),
                "rows": int(episodes.loc[mask, "rows"].sum()),
                "sse_fraction": sse_fraction(episodes, mask),
                "folds": sorted(
                    int(value) for value in episodes.loc[mask, "fold"].unique()
                ),
            }
        )
    overlap = pd.DataFrame(
        {
            left: {
                right: int((left_mask & right_mask).sum())
                for right, right_mask in flags.items()
            }
            for left, left_mask in flags.items()
        }
    )

    evidence_columns = [
        "prior_audit__emission_evidence_class",
        "prior_audit__observed_emission_evidence_class",
        "prior_audit__affine_observed_emission_evidence_class",
        "prior_audit__viterbi_recovery_class",
    ]
    evidence: dict[str, list[dict[str, Any]]] = {}
    total_sse = float(episodes["episode_sse"].sum())
    for column in evidence_columns:
        grouped = (
            episodes.groupby(column, dropna=False)
            .agg(
                episodes=("episode_id", "size"),
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                episode_sse=("episode_sse", "sum"),
            )
            .reset_index()
        )
        grouped["episode_fraction"] = grouped["episodes"] / len(episodes)
        grouped["sse_fraction"] = grouped["episode_sse"] / total_sse
        evidence[column] = grouped.to_dict(orient="records")

    escape_at_start = (
        episodes["truth_basin_escape_after_episode_start_rows"] == 0.0
    )
    recaptured = episodes["truth_basin_first_recapture_suffix_offset"].notna()
    result = {
        "independent_conditions": independent,
        "condition_overlap_episode_counts": overlap.to_dict(),
        "prior_evidence": evidence,
        "escape": {
            "escaped_at_episode_start_fraction": float(escape_at_start.mean()),
            "escaped_at_episode_start_sse_fraction": sse_fraction(
                episodes, escape_at_start
            ),
            "ever_recaptured_fraction": float(recaptured.mean()),
            "ever_recaptured_sse_fraction": sse_fraction(episodes, recaptured),
        },
        "episode_mean_direction": {
            "emission_hurts_truth_fraction": float(
                (episodes["emission_truth_vs_mean_logit_delta_mean"] < 0.0).mean()
            ),
            "emission_hurts_truth_sse_fraction": sse_fraction(
                episodes,
                episodes["emission_truth_vs_mean_logit_delta_mean"] < 0.0,
            ),
            "beta_hurts_truth_fraction": float(
                (episodes["beta_truth_vs_mean_logit_delta_mean"] < 0.0).mean()
            ),
            "beta_hurts_truth_sse_fraction": sse_fraction(
                episodes,
                episodes["beta_truth_vs_mean_logit_delta_mean"] < 0.0,
            ),
        },
    }
    return result


def add_row_derived_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["row_sse"] = chunk["mean_error_ft"] ** 2
    chunk["filtered_rate_error"] = chunk["filtered__rate_mean"] - chunk["true_rate"]
    chunk["mean_path_rate_error"] = chunk["mean_path_rate"] - chunk["true_rate"]
    chunk["current_displacement_error"] = (
        chunk["current_expected_displacement_ft"] - chunk["true_displacement_ft"]
    )
    chunk["exact_mean_displacement_error"] = (
        chunk["exact_mean_expected_displacement_ft"] - chunk["true_displacement_ft"]
    )
    chunk["backward_truth_position_mass_delta"] = (
        chunk["smoothed__truth_position_mass"]
        - chunk["filtered__truth_position_mass"]
    )
    chunk["backward_truth_rate_mass_delta"] = (
        chunk["smoothed__truth_rate_near_mass"]
        - chunk["filtered__truth_rate_near_mass"]
    )
    return chunk


def condition_columns(chunk: pd.DataFrame) -> dict[str, np.ndarray]:
    pred = chunk["predictive__truth_vs_mean_position_logit"].to_numpy()
    filt = chunk["filtered__truth_vs_mean_position_logit"].to_numpy()
    smooth = chunk["smoothed__truth_vs_mean_position_logit"].to_numpy()
    emission = chunk["emission__truth_vs_mean_logit_delta"].to_numpy()
    beta = chunk["beta__truth_vs_mean_logit_delta"].to_numpy()
    direct = chunk["emission_ll__truth_minus_mean"].to_numpy()
    missing = chunk["raw_gr_missing"].to_numpy(bool)
    error = chunk["mean_error_ft"].to_numpy()
    bias = chunk["current_minus_exact_mean_ft"].to_numpy()
    current_displacement_error = chunk["current_displacement_error"].to_numpy()
    exact_displacement_error = chunk["exact_mean_displacement_error"].to_numpy()
    truth_rate = chunk["true_rate"].to_numpy()
    filtered_rate = chunk["filtered__rate_mean"].to_numpy()
    mean_path_rate = chunk["mean_path_rate"].to_numpy()
    backward_position_mass_delta = chunk[
        "backward_truth_position_mass_delta"
    ].to_numpy()
    backward_rate_mass_delta = chunk["backward_truth_rate_mass_delta"].to_numpy()
    viterbi_gap = np.abs(
        chunk["posterior_mean"].to_numpy() - chunk["global_viterbi"].to_numpy()
    )
    logsum_gap = chunk["filtered__logsum_minus_max"].to_numpy()
    result = {
        "predictive_wrong_strong": pred < -EFFECT,
        "filtered_wrong_strong": filt < -EFFECT,
        "smoothed_wrong_strong": smooth < -EFFECT,
        "emission_hurts_truth_strong": emission < -EFFECT,
        "emission_helps_truth_strong": emission > EFFECT,
        "emission_hurts_truth_any": emission < 0.0,
        "emission_helps_truth_any": emission > 0.0,
        "beta_hurts_truth_strong": beta < -EFFECT,
        "beta_helps_truth_strong": beta > EFFECT,
        "beta_hurts_truth_any": beta < 0.0,
        "beta_helps_truth_any": beta > 0.0,
        "emission_creates_wrong_basin": (
            (pred > 0.0) & (filt < 0.0) & (emission < -EFFECT)
        ),
        "emission_rescues_wrong_prior": (
            (pred < 0.0) & (filt > 0.0) & (emission > EFFECT)
        ),
        "backward_creates_wrong_basin": (
            (filt > 0.0) & (smooth < 0.0) & (beta < -EFFECT)
        ),
        "backward_rescues_wrong_filter": (
            (filt < 0.0) & (smooth > 0.0) & (beta > EFFECT)
        ),
        "backward_rate_recovers_while_position_mass_hurts": (
            (backward_rate_mass_delta > 0.0)
            & (backward_position_mass_delta < 0.0)
        ),
        "backward_rate_recovers_while_truth_odds_hurt_strong": (
            (backward_rate_mass_delta > 0.0) & (beta < -EFFECT)
        ),
        "direct_observed_gr_alias_strong": (~missing) & (direct < -EFFECT),
        "direct_observed_gr_truth_strong": (~missing) & (direct > EFFECT),
        "direct_observed_gr_alias_any": (~missing) & (direct < 0.0),
        "direct_observed_gr_truth_any": (~missing) & (direct > 0.0),
        "direct_truth_but_posterior_emission_hurts": (
            (~missing) & (direct > 0.0) & (emission < 0.0)
        ),
        "direct_alias_but_posterior_emission_helps": (
            (~missing) & (direct < 0.0) & (emission > 0.0)
        ),
        "posterior_emission_alias_strong_observed": (
            (~missing) & (emission < -EFFECT)
        ),
        "posterior_emission_alias_strong_missing": missing & (emission < -EFFECT),
        "state_support_shortage": (
            (~chunk["truth_position_support_inside"].to_numpy(bool))
            | (~chunk["truth_rate_support_inside"].to_numpy(bool))
        ),
        "sum_product_multiplicity_row": (viterbi_gap >= 6.0) & (logsum_gap >= 2.0),
        "quantization_bias_same_error_direction": (
            (np.abs(bias) > 1e-12) & (np.sign(bias) == np.sign(error))
        ),
        "current_transition_error_same_offset_direction": (
            (np.abs(current_displacement_error) > 1e-12)
            & (np.sign(current_displacement_error) == np.sign(error))
        ),
        "exact_mean_transition_error_same_offset_direction": (
            (np.abs(exact_displacement_error) > 1e-12)
            & (np.sign(exact_displacement_error) == np.sign(error))
        ),
        "filtered_rate_zero_directed_underresponse": (
            (np.sign(filtered_rate) == np.sign(truth_rate))
            & (np.abs(filtered_rate) < np.abs(truth_rate))
        ),
        "filtered_rate_same_direction_overshoot": (
            (np.sign(filtered_rate) == np.sign(truth_rate))
            & (np.abs(filtered_rate) > np.abs(truth_rate))
        ),
        "mean_path_rate_zero_directed_underresponse": (
            (np.sign(mean_path_rate) == np.sign(truth_rate))
            & (np.abs(mean_path_rate) < np.abs(truth_rate))
        ),
        "mean_path_rate_same_direction_overshoot": (
            (np.sign(mean_path_rate) == np.sign(truth_rate))
            & (np.abs(mean_path_rate) > np.abs(truth_rate))
        ),
    }
    sensitivity_effects = {
        "e0p100": 0.100,
        "e0p405": math.log(1.5),
        "e0p693": math.log(2.0),
        "e1p099": math.log(3.0),
    }
    for tag, effect in sensitivity_effects.items():
        result[f"sensitivity__forward_wrong__{tag}"] = pred < -effect
        result[f"sensitivity__emission_hurts__{tag}"] = emission < -effect
        result[f"sensitivity__imputation_hurts__{tag}"] = (
            missing & (emission < -effect)
        )
        result[f"sensitivity__direct_observed_alias__{tag}"] = (
            (~missing) & (direct < -effect)
        )
        result[f"sensitivity__backward_reversal__{tag}"] = (
            (filt > 0.0) & (smooth < 0.0) & (beta < -effect)
        )
    return result


def row_readout(
    ledger: Path,
    *,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_parts: dict[str, list[np.ndarray]] = {
        column: [] for column in CONTINUOUS_COLUMNS
    }
    value_sums = {column: 0.0 for column in CONTINUOUS_COLUMNS}
    value_counts = {column: 0 for column in CONTINUOUS_COLUMNS}
    weighted_sums = {column: 0.0 for column in CONTINUOUS_COLUMNS}
    condition_rows: dict[str, int] = {}
    condition_sse: dict[str, float] = {}
    episode_accumulator: pd.DataFrame | None = None
    total_rows = 0
    total_sse = 0.0

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            ledger,
            usecols=ROW_USECOLS,
            chunksize=chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = add_row_derived_columns(chunk)
        conditions = condition_columns(chunk)
        row_sse = chunk["row_sse"].to_numpy(np.float64)
        total_rows += len(chunk)
        total_sse += float(np.sum(row_sse))

        for column in CONTINUOUS_COLUMNS:
            values = chunk[column].to_numpy(np.float64)
            finite = np.isfinite(values)
            value_parts[column].append(values[finite])
            value_sums[column] += float(np.sum(values[finite]))
            value_counts[column] += int(finite.sum())
            weighted_sums[column] += float(np.sum(values[finite] * row_sse[finite]))

        episode_piece = pd.DataFrame(
            {
                "episode_id": chunk["episode_id"].astype(str),
                "rows": 1,
                "row_sse": row_sse,
                "mean_error_ft_sum": chunk["mean_error_ft"].to_numpy(np.float64),
                "quantization_bias_sum": chunk[
                    "current_minus_exact_mean_ft"
                ].to_numpy(np.float64),
                "current_displacement_error_sum": chunk[
                    "current_displacement_error"
                ].to_numpy(np.float64),
                "exact_mean_displacement_error_sum": chunk[
                    "exact_mean_displacement_error"
                ].to_numpy(np.float64),
            }
        )
        for name, mask in conditions.items():
            mask = np.asarray(mask, dtype=bool)
            condition_rows[name] = condition_rows.get(name, 0) + int(mask.sum())
            condition_sse[name] = condition_sse.get(name, 0.0) + float(
                np.sum(row_sse[mask])
            )
            episode_piece[name] = mask.astype(np.int64)

        grouped = episode_piece.groupby("episode_id", sort=False).sum()
        if episode_accumulator is None:
            episode_accumulator = grouped
        else:
            episode_accumulator = episode_accumulator.add(grouped, fill_value=0.0)
        if chunk_index % 10 == 0:
            print(
                f"chunk={chunk_index} rows={total_rows} "
                f"sse={total_sse:.6f}",
                flush=True,
            )

    if episode_accumulator is None:
        raise RuntimeError("row ledger was empty")
    if total_rows != 807_710:
        raise RuntimeError(f"row count mismatch: {total_rows}")

    continuous_rows = []
    for column in CONTINUOUS_COLUMNS:
        values = np.concatenate(value_parts[column])
        quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99])
        continuous_rows.append(
            {
                "metric": column,
                "finite_rows": value_counts[column],
                "mean": value_sums[column] / value_counts[column],
                "sse_weighted_mean": weighted_sums[column] / total_sse,
                "p01": float(quantiles[0]),
                "p10": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p90": float(quantiles[3]),
                "p99": float(quantiles[4]),
            }
        )

    episode_accumulator["mean_error_ft_mean"] = (
        episode_accumulator["mean_error_ft_sum"] / episode_accumulator["rows"]
    )
    episode_accumulator["quantization_bias_mean"] = (
        episode_accumulator["quantization_bias_sum"] / episode_accumulator["rows"]
    )
    episode_accumulator["current_displacement_error_mean"] = (
        episode_accumulator["current_displacement_error_sum"]
        / episode_accumulator["rows"]
    )
    episode_accumulator["exact_mean_displacement_error_mean"] = (
        episode_accumulator["exact_mean_displacement_error_sum"]
        / episode_accumulator["rows"]
    )

    condition_summary = []
    fraction_columns = {}
    for name in sorted(condition_rows):
        fractions = episode_accumulator[name] / episode_accumulator["rows"]
        condition_summary.append(
            {
                "condition": name,
                "rows": condition_rows[name],
                "row_fraction": condition_rows[name] / total_rows,
                "row_sse": condition_sse[name],
                "sse_fraction": condition_sse[name] / total_sse,
                "episodes_any": int((fractions > 0.0).sum()),
                "episodes_dominant": int((fractions >= 0.50).sum()),
            }
        )
        fraction_columns[f"{name}_fraction"] = fractions
    episode_accumulator = pd.concat(
        [
            episode_accumulator,
            pd.DataFrame(fraction_columns, index=episode_accumulator.index),
        ],
        axis=1,
    )

    return (
        pd.DataFrame(continuous_rows),
        pd.DataFrame(condition_summary),
        episode_accumulator.reset_index(),
    )


def correlation_readout(
    episode_rows: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, float]:
    merged = episodes[["episode_id", "episode_sse", "cause"]].merge(
        episode_rows, on="episode_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(episodes):
        raise RuntimeError("episode aggregation join was incomplete")
    pairs = [
        ("quantization_bias_mean", "mean_error_ft_mean"),
        ("current_displacement_error_mean", "mean_error_ft_mean"),
        ("exact_mean_displacement_error_mean", "mean_error_ft_mean"),
    ]
    result: dict[str, float] = {}
    for left, right in pairs:
        result[f"pearson__{left}__{right}"] = float(
            merged[left].corr(merged[right], method="pearson")
        )
        result[f"spearman__{left}__{right}"] = float(
            merged[left].corr(merged[right], method="spearman")
        )
        same_sign = np.sign(merged[left].to_numpy()) == np.sign(
            merged[right].to_numpy()
        )
        result[f"same_sign_fraction__{left}__{right}"] = float(same_sign.mean())
        result[f"sse_weighted_same_sign__{left}__{right}"] = float(
            merged.loc[same_sign, "episode_sse"].sum()
            / merged["episode_sse"].sum()
        )
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(args.episode_summary)
    if len(episodes) != 638:
        raise RuntimeError(f"episode count mismatch: {len(episodes)}")

    episode_result = episode_readout(episodes)
    continuous, conditions, episode_rows = row_readout(
        args.row_ledger,
        chunksize=args.chunksize,
    )
    correlations = correlation_readout(episode_rows, episodes)

    continuous.to_csv(args.output_dir / "row_continuous_summary.csv", index=False)
    conditions.to_csv(args.output_dir / "row_condition_summary.csv", index=False)
    episode_rows.to_csv(
        args.output_dir / "episode_row_condition_summary.csv", index=False
    )
    cause_fold = (
        episodes.pivot_table(
            index="cause",
            columns="fold",
            values="episode_sse",
            aggfunc="sum",
            fill_value=0.0,
        )
        .div(episodes.groupby("fold")["episode_sse"].sum(), axis=1)
        .reset_index()
    )
    cause_fold.to_csv(args.output_dir / "cause_fold_sse_fraction.csv", index=False)
    sensitivity_rows = []
    total_episode_sse = float(episode_rows["row_sse"].sum())
    for fraction_column in sorted(
        column
        for column in episode_rows.columns
        if column.startswith("sensitivity__") and column.endswith("_fraction")
    ):
        condition = fraction_column.removesuffix("_fraction")
        for dominant_fraction in (0.25, 0.50, 0.75):
            mask = episode_rows[fraction_column] >= dominant_fraction
            sensitivity_rows.append(
                {
                    "condition": condition,
                    "dominant_fraction": dominant_fraction,
                    "episodes": int(mask.sum()),
                    "episode_fraction": float(mask.mean()),
                    "sse_fraction": float(
                        episode_rows.loc[mask, "row_sse"].sum()
                        / total_episode_sse
                    ),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        args.output_dir / "threshold_sensitivity.csv", index=False
    )
    result = {
        "episode_readout": episode_result,
        "episode_correlations": correlations,
        "rows": int(episode_rows["rows"].sum()),
        "episodes": len(episodes),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
