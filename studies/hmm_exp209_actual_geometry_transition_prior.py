#!/usr/bin/env python3
"""Propagate exp209's transition-only prior on every actual well geometry.

The trajectory starts from each well's real prefix-derived rate prior and uses
the real suffix dMD/dZ sequence, but no GR emission and no suffix truth.  Eight
fixed transition variants separate:

- current versus exact-mean position transport;
- sub-stochastic versus source-normalized rate boundaries;
- momentum 0.998 versus momentum 1.

Truth and the saved exp209 posterior are joined only after all transition-only
trajectories for a well have been generated.  The result is a causal forward-
prior mechanism diagnostic, not a deployable predictor or an OOF candidate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmm_exp209_transition_kernel_audit import (
    DEFAULT_CANDIDATES,
    DEFAULT_EPISODES,
    iter_well_frames,
    position_kernel_mean,
    quantiles,
    require_path,
    safe_corr,
    weighted_fraction,
)

DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_actual_geometry_transition_prior_20260726"
)
POSITION_STEP_FT = 0.35
POSITION_SIGMA_FT = 0.1225
RATE_COUNT = 41
RATE_SPAN_FLOOR = 0.10
RATE_SIGMA_PER_SQRT_MD = 0.002
CURRENT_MOMENTUM = 0.998
INITIAL_RATE_SIGMA = 0.01
ANCHOR_SAFE_DISTANCE_FT = 90.0
VARIANTS = (
    "current",
    "exact_mean_position",
    "boundary_normalized",
    "exact_mean_plus_boundary",
    "momentum_one",
    "exact_mean_plus_momentum_one",
    "boundary_plus_momentum_one",
    "all_three_corrections",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def rate_transition_probabilities(
    rates: np.ndarray,
    dm: float,
    momentum: float,
    normalize_boundary_rows: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate_step = rates[1] - rates[0]
    rate_var_cells = (
        RATE_SIGMA_PER_SQRT_MD * np.sqrt(dm) / rate_step
    ) ** 2
    p_minus_values = np.empty(len(rates), dtype=np.float64)
    p_stay_values = np.empty(len(rates), dtype=np.float64)
    p_plus_values = np.empty(len(rates), dtype=np.float64)
    for source in range(len(rates)):
        mean_move = (
            -(1.0 - momentum)
            * rates[source]
            * dm
            / rate_step
        )
        p_plus = max(
            0.5 * (rate_var_cells + mean_move),
            1e-12,
        )
        p_minus = max(
            0.5 * (rate_var_cells - mean_move),
            1e-12,
        )
        probability_total = p_plus + p_minus
        if probability_total > 0.9:
            p_plus *= 0.9 / probability_total
            p_minus *= 0.9 / probability_total
        probabilities = np.array(
            (p_minus, 1.0 - p_plus - p_minus, p_plus)
        )
        if normalize_boundary_rows:
            valid_total = 0.0
            for offset in range(-1, 2):
                destination = source + offset
                if 0 <= destination < len(rates):
                    valid_total += probabilities[offset + 1]
            probabilities /= valid_total
        p_minus_values[source] = probabilities[0]
        p_stay_values[source] = probabilities[1]
        p_plus_values[source] = probabilities[2]
    return p_minus_values, p_stay_values, p_plus_values


def rate_distribution_sequences(
    initial_distribution: np.ndarray,
    rates: np.ndarray,
    dm: np.ndarray,
) -> np.ndarray:
    if not np.all(dm == dm[0]):
        raise ValueError("vectorized audit requires constant dMD")
    dynamics = (
        (CURRENT_MOMENTUM, False),
        (CURRENT_MOMENTUM, True),
        (1.0, False),
        (1.0, True),
    )
    p_minus = np.empty((4, len(rates)), dtype=np.float64)
    p_stay = np.empty((4, len(rates)), dtype=np.float64)
    p_plus = np.empty((4, len(rates)), dtype=np.float64)
    for dynamics_index, (
        momentum,
        normalize_boundary_rows,
    ) in enumerate(dynamics):
        (
            p_minus[dynamics_index],
            p_stay[dynamics_index],
            p_plus[dynamics_index],
        ) = rate_transition_probabilities(
            rates,
            float(dm[0]),
            momentum,
            normalize_boundary_rows,
        )
    sequence = np.empty(
        (len(dm), 4, len(rates)),
        dtype=np.float64,
    )
    distribution = np.tile(initial_distribution, (4, 1))
    for row_index in range(len(dm)):
        next_distribution = distribution * p_stay
        next_distribution[:, 1:] += (
            distribution[:, :-1] * p_plus[:, :-1]
        )
        next_distribution[:, :-1] += (
            distribution[:, 1:] * p_minus[:, 1:]
        )
        next_distribution /= np.sum(
            next_distribution,
            axis=1,
            keepdims=True,
        )
        distribution = next_distribution
        sequence[row_index] = distribution
    return sequence


def transition_only_trajectories(
    rates: np.ndarray,
    initial_distribution: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
    last_tvt: float,
) -> np.ndarray:
    distributions = rate_distribution_sequences(
        initial_distribution,
        rates,
        dm,
    )
    transition_means = (
        dm[:, None] * rates[None, :] - dz[:, None]
    )
    current_kernel_means = position_kernel_mean(
        transition_means.reshape(-1),
        POSITION_SIGMA_FT,
    ).reshape(transition_means.shape)
    trajectories = np.empty((len(dm), 8), dtype=np.float64)
    for dynamics_index in range(4):
        distribution = distributions[:, dynamics_index, :]
        current_increment = np.sum(
            distribution * current_kernel_means,
            axis=1,
        )
        exact_increment = np.sum(
            distribution * transition_means,
            axis=1,
        )
        current_variant = 2 * dynamics_index
        trajectories[:, current_variant] = (
            last_tvt + np.cumsum(current_increment)
        )
        trajectories[:, current_variant + 1] = (
            last_tvt + np.cumsum(exact_increment)
        )
    return trajectories


def prefix_initial_rate(horizontal: pd.DataFrame) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    tail = known.tail(30)
    dmd = np.diff(
        pd.to_numeric(tail["MD"], errors="raise").to_numpy(
            np.float64
        )
    )
    dtvt = np.diff(
        pd.to_numeric(
            tail["TVT_input"],
            errors="raise",
        ).to_numpy(np.float64)
    )
    dz = np.diff(
        pd.to_numeric(tail["Z"], errors="raise").to_numpy(
            np.float64
        )
    )
    valid = dmd > 0.0
    if int(valid.sum()) < 3:
        return 0.0
    return float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))


def initial_rate_distribution(
    rates: np.ndarray,
    initial_rate: float,
) -> np.ndarray:
    log_weight = -0.5 * (
        (rates - initial_rate) / INITIAL_RATE_SIGMA
    ) ** 2
    weight = np.exp(log_weight - float(np.max(log_weight)))
    return weight / np.sum(weight)


def pooled_rmse(sse: float, rows: int) -> float:
    return float(np.sqrt(sse / rows))


def mean_sign_match(
    left: pd.Series,
    right: pd.Series,
) -> float:
    finite = left.notna() & right.notna()
    return float(
        np.mean(
            np.sign(left.loc[finite].to_numpy(np.float64))
            == np.sign(right.loc[finite].to_numpy(np.float64))
        )
    )


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    root = args.root.resolve()
    candidates = require_path(root / args.candidates)
    episodes_path = require_path(root / args.episodes)
    train_dir = require_path(root / "data/raw/train")
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)
    episodes_by_well = {
        str(well): frame.reset_index(drop=True)
        for well, frame in episodes.groupby("well", sort=False)
    }

    variant_sse = {variant: 0.0 for variant in VARIANTS}
    variant_max_error = {variant: 0.0 for variant in VARIANTS}
    variant_max_anchor_distance = {
        variant: 0.0 for variant in VARIANTS
    }
    variant_anchor_safe_rows = {
        variant: 0 for variant in VARIANTS
    }
    actual_sse = 0.0
    total_rows = 0
    well_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    maximum_transition_only_distance_from_anchor = 0.0
    processed_wells = 0

    for well, frame in iter_well_frames(
        candidates,
        args.chunksize,
    ):
        processed_wells += 1
        horizontal = pd.read_csv(
            train_dir / f"{well}__horizontal_well.csv"
        )
        known = horizontal.loc[horizontal["TVT_input"].notna()]
        last_known = known.iloc[-1]
        row_index = frame["row_idx"].to_numpy(np.int64)
        md = pd.to_numeric(
            horizontal.loc[row_index, "MD"],
            errors="raise",
        ).to_numpy(np.float64)
        z = pd.to_numeric(
            horizontal.loc[row_index, "Z"],
            errors="raise",
        ).to_numpy(np.float64)
        raw_dm = np.diff(
            np.concatenate([[float(last_known["MD"])], md])
        )
        if not np.all(raw_dm == 1.0):
            raise ValueError(f"{well}: expected exact 1 ft dMD")
        dm = np.maximum(raw_dm, 1.0)
        dz = np.diff(
            np.concatenate([[float(last_known["Z"])], z])
        )
        initial_rate = prefix_initial_rate(horizontal)
        rate_span = max(
            RATE_SPAN_FLOOR,
            abs(initial_rate) + 0.04,
        )
        rates = np.linspace(
            -rate_span,
            rate_span,
            RATE_COUNT,
            dtype=np.float64,
        )
        initial_distribution = initial_rate_distribution(
            rates,
            initial_rate,
        )

        trajectories = transition_only_trajectories(
            rates,
            initial_distribution,
            dm,
            dz,
            float(last_known["TVT_input"]),
        )
        if not np.isfinite(trajectories).all():
            raise ValueError(f"{well}: non-finite trajectory")
        maximum_transition_only_distance_from_anchor = max(
            maximum_transition_only_distance_from_anchor,
            float(
                np.max(
                    np.abs(
                        trajectories
                        - float(last_known["TVT_input"])
                    )
                )
            ),
        )

        # Truth and the saved posterior enter only after every transition-only
        # trajectory for this well has been generated.
        truth = frame["true_tvt_readout_only"].to_numpy(np.float64)
        actual = frame["posterior_mean"].to_numpy(np.float64)
        actual_error = actual - truth
        actual_sse += float(np.sum(actual_error**2))
        total_rows += len(frame)
        well_row: dict[str, Any] = {
            "well": well,
            "rows": int(len(frame)),
            "initial_rate": initial_rate,
            "rate_span": rate_span,
            "actual_hmm_rmse_ft": float(
                np.sqrt(np.mean(actual_error**2))
            ),
        }
        for variant_index, variant in enumerate(VARIANTS):
            anchor_distance = np.abs(
                trajectories[:, variant_index]
                - float(last_known["TVT_input"])
            )
            variant_max_anchor_distance[variant] = max(
                variant_max_anchor_distance[variant],
                float(np.max(anchor_distance)),
            )
            variant_anchor_safe_rows[variant] += int(
                np.count_nonzero(
                    anchor_distance <= ANCHOR_SAFE_DISTANCE_FT
                )
            )
            error = trajectories[:, variant_index] - truth
            sse = float(np.sum(error**2))
            variant_sse[variant] += sse
            variant_max_error[variant] = max(
                variant_max_error[variant],
                float(np.max(np.abs(error))),
            )
            well_row[f"{variant}_rmse_ft"] = float(
                np.sqrt(np.mean(error**2))
            )
            well_row[
                f"{variant}_error_vs_actual_hmm_error_corr"
            ] = safe_corr(error, actual_error)
        well_rows.append(well_row)

        well_episodes = episodes_by_well.get(well)
        if processed_wells % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed_wells": processed_wells,
                        "elapsed_seconds": (
                            time.perf_counter() - started_at
                        ),
                    }
                ),
                flush=True,
            )
        if well_episodes is None:
            continue
        for episode in well_episodes.itertuples(index=False):
            start = int(episode.start_suffix_offset)
            end = start + int(episode.rows)
            pre128_start = max(0, start - 128)
            row: dict[str, Any] = {
                "episode_id": str(episode.episode_id),
                "well": well,
                "rows": int(episode.rows),
                "episode_sse": float(episode.episode_sse),
                "actual_mean_error_ft": float(
                    np.mean(actual_error[start:end])
                ),
                "actual_rmse_ft": float(
                    np.sqrt(np.mean(actual_error[start:end] ** 2))
                ),
            }
            for variant_index, variant in enumerate(VARIANTS):
                error = (
                    trajectories[start:end, variant_index]
                    - truth[start:end]
                )
                row[f"{variant}_mean_error_ft"] = float(
                    np.mean(error)
                )
                row[f"{variant}_rmse_ft"] = float(
                    np.sqrt(np.mean(error**2))
                )
            effects = {
                "position_quantization": (
                    trajectories[:, 0] - trajectories[:, 1]
                ),
                "rate_boundary": (
                    trajectories[:, 0] - trajectories[:, 2]
                ),
                "rate_mean_reversion": (
                    trajectories[:, 0] - trajectories[:, 4]
                ),
                "all_transition_components": (
                    trajectories[:, 0] - trajectories[:, 7]
                ),
            }
            effect_variant_pairs = {
                "position_quantization": (0, 1),
                "rate_boundary": (0, 2),
                "rate_mean_reversion": (0, 4),
                "all_transition_components": (0, 7),
            }
            for effect_name, effect in effects.items():
                left_variant, right_variant = (
                    effect_variant_pairs[effect_name]
                )
                anchor_safe = (
                    np.abs(
                        trajectories[:, left_variant]
                        - float(last_known["TVT_input"])
                    )
                    <= ANCHOR_SAFE_DISTANCE_FT
                ) & (
                    np.abs(
                        trajectories[:, right_variant]
                        - float(last_known["TVT_input"])
                    )
                    <= ANCHOR_SAFE_DISTANCE_FT
                )
                episode_effect = float(np.mean(effect[start:end]))
                onset_effect = float(effect[start])
                preceding_effect_change = float(
                    effect[start]
                    - (
                        effect[pre128_start]
                        if pre128_start < start
                        else 0.0
                    )
                )
                row[f"{effect_name}_episode_mean_effect_ft"] = (
                    episode_effect
                )
                row[f"{effect_name}_onset_effect_ft"] = onset_effect
                row[
                    f"{effect_name}_preceding128_effect_change_ft"
                ] = preceding_effect_change
                row[
                    f"{effect_name}_episode_sign_matches_actual"
                ] = bool(
                    np.sign(episode_effect)
                    == np.sign(row["actual_mean_error_ft"])
                )
                row[
                    f"{effect_name}_onset_sign_matches_actual"
                ] = bool(
                    np.sign(onset_effect)
                    == np.sign(row["actual_mean_error_ft"])
                )
                row[
                    f"{effect_name}_episode_anchor_safe_fraction"
                ] = float(np.mean(anchor_safe[start:end]))
                row[f"{effect_name}_onset_anchor_safe"] = bool(
                    anchor_safe[start]
                )
            episode_rows.append(row)
    by_well = (
        pd.DataFrame(well_rows)
        .sort_values("well")
        .reset_index(drop=True)
    )
    by_episode = (
        pd.DataFrame(episode_rows)
        .sort_values(["well", "episode_id"])
        .reset_index(drop=True)
    )

    variant_summary = {
        variant: {
            "transition_only_prior_rmse_ft": pooled_rmse(
                variant_sse[variant],
                total_rows,
            ),
            "maximum_abs_error_ft": variant_max_error[variant],
            "well_error_vs_actual_hmm_error_corr_median": float(
                by_well[
                    f"{variant}_error_vs_actual_hmm_error_corr"
                ].median()
            ),
            "maximum_anchor_distance_ft": (
                variant_max_anchor_distance[variant]
            ),
            "anchor_safe_row_fraction": float(
                variant_anchor_safe_rows[variant] / total_rows
            ),
        }
        for variant in VARIANTS
    }
    effect_summary: dict[str, Any] = {}
    for effect_name in (
        "position_quantization",
        "rate_boundary",
        "rate_mean_reversion",
        "all_transition_components",
    ):
        episode_effect = by_episode[
            f"{effect_name}_episode_mean_effect_ft"
        ]
        onset_effect = by_episode[
            f"{effect_name}_onset_effect_ft"
        ]
        preceding_effect = by_episode[
            f"{effect_name}_preceding128_effect_change_ft"
        ]
        sign_match = by_episode[
            f"{effect_name}_episode_sign_matches_actual"
        ]
        onset_sign_match = by_episode[
            f"{effect_name}_onset_sign_matches_actual"
        ]
        safe_episode = (
            by_episode[
                f"{effect_name}_episode_anchor_safe_fraction"
            ]
            >= 0.90
        ) & by_episode[f"{effect_name}_onset_anchor_safe"]
        safe_group = by_episode.loc[safe_episode]
        safe_effect = safe_group[
            f"{effect_name}_episode_mean_effect_ft"
        ]
        safe_sign_match = safe_group[
            f"{effect_name}_episode_sign_matches_actual"
        ]
        effect_summary[effect_name] = {
            "episode_effect_abs_quantiles_ft": quantiles(
                np.abs(episode_effect.to_numpy(np.float64))
            ),
            "episode_effect_vs_actual_mean_error_spearman": float(
                episode_effect.corr(
                    by_episode["actual_mean_error_ft"],
                    method="spearman",
                )
            ),
            "episode_sign_match_fraction": float(sign_match.mean()),
            "episode_sign_match_sse_fraction": weighted_fraction(
                sign_match,
                by_episode["episode_sse"],
            ),
            "onset_effect_vs_actual_mean_error_spearman": float(
                onset_effect.corr(
                    by_episode["actual_mean_error_ft"],
                    method="spearman",
                )
            ),
            "onset_sign_match_fraction": float(
                onset_sign_match.mean()
            ),
            "preceding128_effect_change_vs_actual_mean_error_spearman": float(
                preceding_effect.corr(
                    by_episode["actual_mean_error_ft"],
                    method="spearman",
                )
            ),
            "episode_effect_sign_match_recomputed": mean_sign_match(
                episode_effect,
                by_episode["actual_mean_error_ft"],
            ),
            "anchor_safe_subset": {
                "definition": (
                    "Both compared transition-only trajectories remain "
                    "within 90 ft of the prefix anchor for at least 90% "
                    "of the episode and at onset."
                ),
                "episodes": int(len(safe_group)),
                "episode_sse_fraction": float(
                    safe_group["episode_sse"].sum()
                    / by_episode["episode_sse"].sum()
                ),
                "effect_vs_actual_mean_error_spearman": float(
                    safe_effect.corr(
                        safe_group["actual_mean_error_ft"],
                        method="spearman",
                    )
                ),
                "sign_match_fraction": float(
                    safe_sign_match.mean()
                ),
                "sign_match_sse_fraction": weighted_fraction(
                    safe_sign_match,
                    safe_group["episode_sse"],
                ),
            },
        }

    summary = {
        "scope": {
            "rows": int(total_rows),
            "wells": int(len(by_well)),
            "persistent_episodes": int(len(by_episode)),
        },
        "saved_exp209_posterior_rmse_ft": pooled_rmse(
            actual_sse,
            total_rows,
        ),
        "transition_only_variants": variant_summary,
        "persistent_episode_component_association": effect_summary,
        "maximum_transition_only_distance_from_anchor_ft": (
            maximum_transition_only_distance_from_anchor
        ),
        "elapsed_seconds": float(time.perf_counter() - started_at),
        "guards": {
            "truth_usage": (
                "Per-well transition-only trajectories are generated from "
                "prefix rate, dMD, and dZ before truth/posterior evaluation."
            ),
            "position_boundary": (
                "The moment propagation is translation-invariant and omits "
                "the finite +/-100 ft position grid boundary. The saved HMM "
                "has zero posterior edge rows. Per-variant maximum distance "
                "and <=90 ft row coverage are reported, and episode "
                "associations include a conservative anchor-safe subset."
            ),
            "interpretation": (
                "Transition-only prior RMSE is not a prediction score. "
                "Differences between fixed variants isolate forward-prior "
                "component effects on actual geometry, while GR emissions "
                "and backward smoothing are intentionally absent."
            ),
        },
    }

    by_well.to_csv(output / "by_well_metrics.csv", index=False)
    by_episode.to_csv(output / "episode_metrics.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
