#!/usr/bin/env python3
"""Audit exp209's discrete position-transition kernel against true TVT motion.

This is a truth-late diagnostic. It does not generate or select predictions.
For each suffix row, it measures:

1. the expected displacement of exp209's normalized five-cell position kernel
   when centered at the true TVT displacement;
2. the best expected displacement representable by any of exp209's 41 rate
   states at that row;
3. the rate-state shift required to compensate for position-grid shrinkage;
4. the sampled initial position/rate prior moments on each actual grid;
5. the first-step rate-transition moments and boundary mass loss.

The first quantity isolates position-grid discretization. The second separates
that approximation from rate support. Episode summaries are joined only after
the saved exp209/exp270 predictions have been generated.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_OUTPUT = Path("studies/hmm_exp209_transition_kernel_audit_20260725")
USECOLS = [
    "well",
    "row_idx",
    "true_tvt_readout_only",
    "last_known_tvt",
    "posterior_mean",
]
POSITION_STEP_FT = 0.35
POSITION_SIGMA_FLOOR_FT = 0.35 * POSITION_STEP_FT
POSITION_PHASE_MINIMAX_SIGMA_FT = 0.2325
RATE_COUNT = 41
RATE_SPAN_FLOOR = 0.10
RATE_SIGMA_PER_SQRT_MD = 0.002
RATE_MOMENTUM = 0.998
START_POSITION_SIGMA_FT = 0.75
INITIAL_RATE_SIGMA = 0.01
BAND_PAD_FT = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def iter_well_frames(
    path: Path,
    chunksize: int,
) -> Iterator[tuple[str, pd.DataFrame]]:
    carry: pd.DataFrame | None = None
    previous_well: str | None = None
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize):
        chunk["well"] = chunk["well"].astype(str)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        if chunk.empty:
            continue
        if not chunk["well"].is_monotonic_increasing:
            raise ValueError("candidate artifact must be sorted by well")
        final_well = str(chunk["well"].iloc[-1])
        complete = chunk.loc[chunk["well"] != final_well]
        carry = chunk.loc[chunk["well"] == final_well].copy()
        for well, group in complete.groupby("well", sort=False):
            well = str(well)
            if previous_well is not None and well <= previous_well:
                raise ValueError("candidate well order must be strictly increasing")
            previous_well = well
            yield well, group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        well = str(carry["well"].iloc[0])
        if previous_well is not None and well <= previous_well:
            raise ValueError("candidate final well order is invalid")
        yield well, carry.reset_index(drop=True)


def position_kernel_mean(
    displacement: np.ndarray,
    sigma_ft: float = POSITION_SIGMA_FLOOR_FT,
) -> np.ndarray:
    """Match exp209's normalized five-cell Gaussian displacement kernel."""
    displacement = np.asarray(displacement, dtype=np.float64)
    center = np.floor(displacement / POSITION_STEP_FT + 0.5).astype(np.int64)
    numerator = np.zeros_like(displacement)
    denominator = np.zeros_like(displacement)
    for cell_offset in range(-2, 3):
        grid_displacement = (center + cell_offset) * POSITION_STEP_FT
        weight = np.exp(
            -0.5
            * (
                (grid_displacement - displacement)
                / sigma_ft
            )
            ** 2
        )
        numerator += weight * grid_displacement
        denominator += weight
    return numerator / denominator


def position_kernel_mean_and_variance(
    displacement: np.ndarray,
    sigma_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first two moments of the normalized five-cell kernel."""
    displacement = np.asarray(displacement, dtype=np.float64)
    center = np.floor(
        displacement / POSITION_STEP_FT + 0.5
    ).astype(np.int64)
    numerator = np.zeros_like(displacement)
    second_numerator = np.zeros_like(displacement)
    denominator = np.zeros_like(displacement)
    for cell_offset in range(-2, 3):
        grid_displacement = (
            center + cell_offset
        ) * POSITION_STEP_FT
        weight = np.exp(
            -0.5
            * (
                (grid_displacement - displacement)
                / sigma_ft
            )
            ** 2
        )
        numerator += weight * grid_displacement
        second_numerator += weight * grid_displacement**2
        denominator += weight
    mean = numerator / denominator
    variance = np.maximum(
        second_numerator / denominator - mean**2,
        0.0,
    )
    return mean, variance


def five_cell_phase_bias_audit() -> dict[str, Any]:
    phase = np.linspace(
        -0.5 * POSITION_STEP_FT,
        0.5 * POSITION_STEP_FT,
        20_001,
    )
    fixed_rows: list[dict[str, float]] = []
    for sigma_ft in (0.1225, 0.15, 0.175, 0.20, 0.245, 0.35):
        bias = position_kernel_mean(phase, sigma_ft) - phase
        fixed_rows.append(
            {
                "sigma_ft": sigma_ft,
                "sigma_in_grid_steps": sigma_ft / POSITION_STEP_FT,
                "maximum_abs_bias_ft": float(np.max(np.abs(bias))),
                "phase_abs_mean_bias_ft": float(np.mean(np.abs(bias))),
            }
        )
    scan_rows: list[tuple[float, float, float]] = []
    for sigma_ft in np.linspace(0.1225, 0.35, 456):
        bias = position_kernel_mean(phase, float(sigma_ft)) - phase
        scan_rows.append(
            (
                float(np.max(np.abs(bias))),
                float(np.mean(np.abs(bias))),
                float(sigma_ft),
            )
        )
    minimum_maximum = min(scan_rows, key=lambda item: item[0])
    minimum_mean = min(scan_rows, key=lambda item: item[1])
    return {
        "fixed_sigma_readout": fixed_rows,
        "minimum_maximum_abs_bias": {
            "sigma_ft": minimum_maximum[2],
            "maximum_abs_bias_ft": minimum_maximum[0],
            "phase_abs_mean_bias_ft": minimum_maximum[1],
        },
        "minimum_phase_abs_mean_bias": {
            "sigma_ft": minimum_mean[2],
            "maximum_abs_bias_ft": minimum_mean[0],
            "phase_abs_mean_bias_ft": minimum_mean[1],
        },
        "guard": (
            "This is a deterministic five-cell numerical phase audit, not "
            "an HMM CV parameter selection."
        ),
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size != left.size:
        return np.nan
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "median", "p90", "p95", "p99", "max"),
            np.quantile(values, [0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0]),
            strict=True,
        )
    }


def weighted_fraction(mask: pd.Series, weight: pd.Series) -> float:
    finite = mask.notna() & weight.notna()
    if not finite.any():
        return np.nan
    kept_weight = weight.loc[finite].to_numpy(np.float64)
    return float(
        np.sum(kept_weight * mask.loc[finite].to_numpy(bool))
        / np.sum(kept_weight)
    )


def weighted_mean(values: pd.Series, weight: pd.Series) -> float:
    finite = values.notna() & weight.notna()
    if not finite.any():
        return np.nan
    return float(
        np.average(
            values.loc[finite].to_numpy(np.float64),
            weights=weight.loc[finite].to_numpy(np.float64),
        )
    )


def standardized_rank(values: pd.Series) -> np.ndarray:
    ranks = values.rank(method="average").to_numpy(np.float64)
    return (ranks - ranks.mean()) / ranks.std()


def partial_spearman(
    left: pd.Series,
    right: pd.Series,
    control: pd.Series,
) -> float:
    return partial_spearman_controls(left, right, [control])


def partial_spearman_controls(
    left: pd.Series,
    right: pd.Series,
    controls: list[pd.Series],
) -> float:
    frame = pd.DataFrame(
        {
            "left": left,
            "right": right,
            **{
                f"control_{index}": control
                for index, control in enumerate(controls)
            },
        }
    ).dropna()
    left_rank = standardized_rank(frame["left"])
    right_rank = standardized_rank(frame["right"])
    control_ranks = np.column_stack(
        [
            standardized_rank(frame[f"control_{index}"])
            for index in range(len(controls))
        ]
    )
    design = np.column_stack([np.ones(len(frame)), control_ranks])
    left_residual = left_rank - design @ np.linalg.lstsq(
        design,
        left_rank,
        rcond=None,
    )[0]
    right_residual = right_rank - design @ np.linalg.lstsq(
        design,
        right_rank,
        rcond=None,
    )[0]
    return float(np.corrcoef(left_residual, right_residual)[0, 1])


def rank_regression_r2(target: pd.Series, features: list[pd.Series]) -> float:
    frame = pd.concat([target, *features], axis=1).dropna()
    target_rank = standardized_rank(frame.iloc[:, 0])
    feature_rank = np.column_stack(
        [
            standardized_rank(frame.iloc[:, index])
            for index in range(1, frame.shape[1])
        ]
    )
    design = np.column_stack([np.ones(len(frame)), feature_rank])
    fitted = design @ np.linalg.lstsq(
        design,
        target_rank,
        rcond=None,
    )[0]
    residual_sse = float(np.sum((target_rank - fitted) ** 2))
    total_sse = float(
        np.sum((target_rank - target_rank.mean()) ** 2)
    )
    return float(1.0 - residual_sse / total_sse)


def main() -> None:
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

    well_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    all_kernel_bias: list[np.ndarray] = []
    all_rate_mean_reversion_position_bias: list[np.ndarray] = []
    all_representation_error: list[np.ndarray] = []
    all_rate_shift_supported: list[np.ndarray] = []
    all_diffusion_rows_proxy_supported: list[np.ndarray] = []
    all_truth_nearest_rate_edge: list[np.ndarray] = []
    all_compensating_rate_edge: list[np.ndarray] = []
    all_compensating_edge_outward_probability: list[np.ndarray] = []
    total_rows = 0
    raw_dm_min = np.inf
    raw_dm_max = -np.inf
    raw_dm_not_one_rows = 0
    raw_dm_below_one_rows = 0
    raw_dm_nonpositive_rows = 0
    current_kernel_variance_sum = 0.0
    phase_minimax_kernel_abs_bias_sum = 0.0
    phase_minimax_kernel_abs_bias_max = 0.0
    phase_minimax_kernel_variance_sum = 0.0
    exact_mean_minimum_variance_sum = 0.0
    exact_mean_minimum_variance_max = 0.0
    initial_position_prior_biases: list[float] = []
    initial_position_prior_variances: list[float] = []
    initial_rate_prior_biases: list[float] = []
    initial_rate_prior_variances: list[float] = []
    initial_rate_prior_edge_masses: list[float] = []
    initial_rate_transition_mass_losses: list[float] = []
    initial_rate_transition_conditional_mean_errors: list[float] = []
    rate_transition_interior_mean_error_max = 0.0
    rate_transition_probability_cap_rows = 0
    rate_transition_probability_floor_rows = 0
    rate_transition_source_rows = 0

    for well, frame in iter_well_frames(candidates, args.chunksize):
        horizontal = pd.read_csv(train_dir / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(train_dir / f"{well}__typewell.csv")
        row_index = frame["row_idx"].to_numpy(np.int64)
        md = pd.to_numeric(
            horizontal.loc[row_index, "MD"], errors="raise"
        ).to_numpy(np.float64)
        z = pd.to_numeric(
            horizontal.loc[row_index, "Z"], errors="raise"
        ).to_numpy(np.float64)
        known = horizontal.loc[horizontal["TVT_input"].notna()]
        last_known = known.iloc[-1]
        raw_dm = np.diff(
            np.concatenate([[float(last_known["MD"])], md])
        )
        raw_dm_min = min(raw_dm_min, float(np.min(raw_dm)))
        raw_dm_max = max(raw_dm_max, float(np.max(raw_dm)))
        raw_dm_not_one_rows += int(np.count_nonzero(raw_dm != 1.0))
        raw_dm_below_one_rows += int(np.count_nonzero(raw_dm < 1.0))
        raw_dm_nonpositive_rows += int(
            np.count_nonzero(raw_dm <= 0.0)
        )
        dm = np.maximum(raw_dm, 1.0)
        dz = np.diff(np.concatenate([[float(last_known["Z"])], z]))
        truth = frame["true_tvt_readout_only"].to_numpy(np.float64)
        true_displacement = np.diff(
            np.concatenate([[float(frame["last_known_tvt"].iloc[0])], truth])
        )
        true_rate = (true_displacement + dz) / dm
        rate_mean_reversion_position_bias = (
            -(1.0 - RATE_MOMENTUM) * true_rate * dm**2
        )

        tail = known.tail(30)
        prefix_dm = np.diff(
            pd.to_numeric(tail["MD"], errors="raise").to_numpy(np.float64)
        )
        prefix_rate = (
            np.diff(
                pd.to_numeric(
                    tail["TVT_input"], errors="raise"
                ).to_numpy(np.float64)
            )
            + np.diff(
                pd.to_numeric(tail["Z"], errors="raise").to_numpy(np.float64)
            )
        ) / prefix_dm
        prefix_valid = np.isfinite(prefix_rate) & (prefix_dm > 0)
        init_rate = (
            float(np.median(prefix_rate[prefix_valid]))
            if int(prefix_valid.sum()) >= 3
            else 0.0
        )
        rate_span = max(RATE_SPAN_FLOOR, abs(init_rate) + 0.04)
        rates = np.linspace(-rate_span, rate_span, RATE_COUNT)
        rate_supported = np.abs(true_rate) <= rate_span
        rate_step = float(rates[1] - rates[0])

        typewell_tvt = pd.to_numeric(
            typewell["TVT"], errors="raise"
        ).to_numpy(np.float64)
        last_tvt = float(last_known["TVT_input"])
        grid_min = max(
            float(np.min(typewell_tvt)) - 40.0,
            last_tvt - BAND_PAD_FT,
        )
        grid_max = min(
            float(np.max(typewell_tvt)) + 40.0,
            last_tvt + BAND_PAD_FT,
        )
        position_grid = np.arange(
            grid_min,
            grid_max + POSITION_STEP_FT,
            POSITION_STEP_FT,
            dtype=np.float64,
        )
        initial_position_displacement = position_grid - last_tvt
        position_log_weight = -0.5 * (
            initial_position_displacement / START_POSITION_SIGMA_FT
        ) ** 2
        position_weight = np.where(
            position_log_weight >= -60.0,
            np.exp(position_log_weight),
            0.0,
        )
        position_weight /= np.sum(position_weight)
        initial_position_prior_bias = float(
            np.sum(position_weight * initial_position_displacement)
        )
        initial_position_prior_variance = float(
            np.sum(
                position_weight
                * (
                    initial_position_displacement
                    - initial_position_prior_bias
                )
                ** 2
            )
        )

        initial_rate_log_weight = -0.5 * (
            (rates - init_rate) / INITIAL_RATE_SIGMA
        ) ** 2
        initial_rate_weight = np.exp(
            initial_rate_log_weight
            - float(np.max(initial_rate_log_weight))
        )
        initial_rate_weight /= np.sum(initial_rate_weight)
        initial_rate_prior_mean = float(
            np.sum(initial_rate_weight * rates)
        )
        initial_rate_prior_bias = initial_rate_prior_mean - init_rate
        initial_rate_prior_variance = float(
            np.sum(
                initial_rate_weight
                * (rates - initial_rate_prior_mean) ** 2
            )
        )
        initial_rate_prior_edge_mass = float(
            initial_rate_weight[0] + initial_rate_weight[-1]
        )

        first_dm = float(dm[0])
        rate_var_cells_first = (
            RATE_SIGMA_PER_SQRT_MD * np.sqrt(first_dm) / rate_step
        ) ** 2
        rate_mean_move_first = (
            -(1.0 - RATE_MOMENTUM)
            * rates
            * first_dm
            / rate_step
        )
        raw_p_plus = 0.5 * (
            rate_var_cells_first + rate_mean_move_first
        )
        raw_p_minus = 0.5 * (
            rate_var_cells_first - rate_mean_move_first
        )
        rate_transition_probability_floor_rows += int(
            np.count_nonzero(raw_p_plus < 1e-12)
            + np.count_nonzero(raw_p_minus < 1e-12)
        )
        p_plus = np.maximum(raw_p_plus, 1e-12)
        p_minus = np.maximum(raw_p_minus, 1e-12)
        probability_total = p_plus + p_minus
        capped = probability_total > 0.9
        rate_transition_probability_cap_rows += int(
            np.count_nonzero(capped)
        )
        p_plus[capped] *= 0.9 / probability_total[capped]
        p_minus[capped] *= 0.9 / probability_total[capped]
        rate_transition_source_rows += RATE_COUNT
        intended_next_rate = (
            rates
            - (1.0 - RATE_MOMENTUM) * rates * first_dm
        )
        represented_next_rate = rates + rate_step * (p_plus - p_minus)
        rate_transition_interior_mean_error_max = max(
            rate_transition_interior_mean_error_max,
            float(
                np.max(
                    np.abs(
                        represented_next_rate[1:-1]
                        - intended_next_rate[1:-1]
                    )
                )
            ),
        )
        transition_row_sum = np.ones(RATE_COUNT, dtype=np.float64)
        transition_row_sum[0] -= p_minus[0]
        transition_row_sum[-1] -= p_plus[-1]
        transition_rate_numerator = represented_next_rate.copy()
        transition_rate_numerator[0] -= p_minus[0] * (
            rates[0] - rate_step
        )
        transition_rate_numerator[-1] -= p_plus[-1] * (
            rates[-1] + rate_step
        )
        surviving_mass = float(
            np.sum(initial_rate_weight * transition_row_sum)
        )
        conditional_next_rate = float(
            np.sum(
                initial_rate_weight * transition_rate_numerator
            )
            / surviving_mass
        )
        intended_prior_next_rate = float(
            np.sum(initial_rate_weight * intended_next_rate)
        )
        initial_rate_transition_mass_loss = 1.0 - surviving_mass
        initial_rate_transition_conditional_mean_error = (
            conditional_next_rate - intended_prior_next_rate
        )

        initial_position_prior_biases.append(
            initial_position_prior_bias
        )
        initial_position_prior_variances.append(
            initial_position_prior_variance
        )
        initial_rate_prior_biases.append(initial_rate_prior_bias)
        initial_rate_prior_variances.append(initial_rate_prior_variance)
        initial_rate_prior_edge_masses.append(
            initial_rate_prior_edge_mass
        )
        initial_rate_transition_mass_losses.append(
            initial_rate_transition_mass_loss
        )
        initial_rate_transition_conditional_mean_errors.append(
            initial_rate_transition_conditional_mean_error
        )

        true_centered_kernel_bias = (
            position_kernel_mean(true_displacement) - true_displacement
        )
        _, current_kernel_variance = position_kernel_mean_and_variance(
            true_displacement,
            POSITION_SIGMA_FLOOR_FT,
        )
        (
            phase_minimax_kernel_mean,
            phase_minimax_kernel_variance,
        ) = position_kernel_mean_and_variance(
            true_displacement,
            POSITION_PHASE_MINIMAX_SIGMA_FT,
        )
        phase_minimax_kernel_abs_bias = np.abs(
            phase_minimax_kernel_mean - true_displacement
        )
        nearest_grid_displacement = (
            np.floor(
                true_displacement / POSITION_STEP_FT + 0.5
            )
            * POSITION_STEP_FT
        )
        absolute_phase = np.abs(
            true_displacement - nearest_grid_displacement
        )
        exact_mean_minimum_variance = (
            absolute_phase
            * (POSITION_STEP_FT - absolute_phase)
        )
        current_kernel_variance_sum += float(
            np.sum(current_kernel_variance)
        )
        phase_minimax_kernel_abs_bias_sum += float(
            np.sum(phase_minimax_kernel_abs_bias)
        )
        phase_minimax_kernel_abs_bias_max = max(
            phase_minimax_kernel_abs_bias_max,
            float(np.max(phase_minimax_kernel_abs_bias)),
        )
        phase_minimax_kernel_variance_sum += float(
            np.sum(phase_minimax_kernel_variance)
        )
        exact_mean_minimum_variance_sum += float(
            np.sum(exact_mean_minimum_variance)
        )
        exact_mean_minimum_variance_max = max(
            exact_mean_minimum_variance_max,
            float(np.max(exact_mean_minimum_variance)),
        )
        candidate_mu = (
            rates[:, None] * dm[None, :] - dz[None, :]
        )
        candidate_kernel_mean = position_kernel_mean(candidate_mu)
        absolute_error = np.abs(
            candidate_kernel_mean - true_displacement[None, :]
        )
        best_index = np.argmin(absolute_error, axis=0)
        true_nearest_rate_index = np.argmin(
            np.abs(rates[:, None] - true_rate[None, :]),
            axis=0,
        )
        true_nearest_rate_edge = (
            (true_nearest_rate_index == 0)
            | (true_nearest_rate_index == RATE_COUNT - 1)
        )
        compensating_rate_edge = (
            (best_index == 0)
            | (best_index == RATE_COUNT - 1)
        )
        rate_var_cells = (
            RATE_SIGMA_PER_SQRT_MD * np.sqrt(dm) / rate_step
        ) ** 2
        best_rate = rates[best_index]
        mean_rate_move = (
            -(1.0 - RATE_MOMENTUM)
            * best_rate
            * dm
            / rate_step
        )
        p_plus = np.maximum(
            0.5 * (rate_var_cells + mean_rate_move),
            1e-12,
        )
        p_minus = np.maximum(
            0.5 * (rate_var_cells - mean_rate_move),
            1e-12,
        )
        probability_total = p_plus + p_minus
        capped = probability_total > 0.9
        p_plus[capped] *= 0.9 / probability_total[capped]
        p_minus[capped] *= 0.9 / probability_total[capped]
        compensating_edge_outward_probability = np.where(
            best_index == 0,
            p_minus,
            np.where(best_index == RATE_COUNT - 1, p_plus, np.nan),
        )
        compensating_edge_log_mass_loss = np.where(
            compensating_rate_edge,
            -np.log1p(-compensating_edge_outward_probability),
            0.0,
        )
        row_number = np.arange(len(frame))
        representation_error = (
            candidate_kernel_mean[best_index, row_number] - true_displacement
        )
        compensating_rate_shift = rates[best_index] - true_rate
        compensating_diffusion_rows_proxy = (
            compensating_rate_shift**2
            / (RATE_SIGMA_PER_SQRT_MD**2 * dm)
        )

        prediction_error = (
            frame["posterior_mean"].to_numpy(np.float64) - truth
        )
        cumulative_kernel_bias = np.cumsum(true_centered_kernel_bias)
        well_rows.append(
            {
                "well": well,
                "rows": int(len(frame)),
                "initial_position_prior_mean_bias_ft": (
                    initial_position_prior_bias
                ),
                "initial_position_prior_variance_ft2": (
                    initial_position_prior_variance
                ),
                "initial_rate": init_rate,
                "rate_span": rate_span,
                "rate_step": rate_step,
                "initial_rate_prior_mean_bias": initial_rate_prior_bias,
                "initial_rate_prior_variance": (
                    initial_rate_prior_variance
                ),
                "initial_rate_prior_edge_mass": (
                    initial_rate_prior_edge_mass
                ),
                "initial_rate_transition_mass_loss": (
                    initial_rate_transition_mass_loss
                ),
                "initial_rate_transition_conditional_mean_error": (
                    initial_rate_transition_conditional_mean_error
                ),
                "kernel_bias_mean_ft_per_row": float(
                    np.mean(true_centered_kernel_bias)
                ),
                "kernel_bias_abs_mean_ft_per_row": float(
                    np.mean(np.abs(true_centered_kernel_bias))
                ),
                "kernel_bias_sum_ft": float(
                    np.sum(true_centered_kernel_bias)
                ),
                "rate_mean_reversion_position_bias_abs_mean_ft_per_row": float(
                    np.mean(
                        np.abs(rate_mean_reversion_position_bias)
                    )
                ),
                "rate_mean_reversion_position_bias_sum_ft": float(
                    np.sum(rate_mean_reversion_position_bias)
                ),
                "representation_error_abs_mean_ft_per_row": float(
                    np.mean(np.abs(representation_error))
                ),
                "supported_representation_error_abs_mean_ft_per_row": float(
                    np.mean(np.abs(representation_error[rate_supported]))
                )
                if rate_supported.any()
                else np.nan,
                "supported_compensating_rate_shift_abs_mean": float(
                    np.mean(
                        np.abs(compensating_rate_shift[rate_supported])
                    )
                )
                if rate_supported.any()
                else np.nan,
                "supported_compensating_diffusion_rows_proxy_median": float(
                    np.median(
                        compensating_diffusion_rows_proxy[rate_supported]
                    )
                )
                if rate_supported.any()
                else np.nan,
                "true_rate_outside_fraction": float(
                    np.mean(~rate_supported)
                ),
                "true_nearest_rate_edge_fraction": float(
                    np.mean(true_nearest_rate_edge)
                ),
                "best_compensating_rate_edge_fraction": float(
                    np.mean(compensating_rate_edge)
                ),
                "prediction_error_vs_cumulative_kernel_bias_corr": safe_corr(
                    prediction_error,
                    cumulative_kernel_bias,
                ),
                "end_prediction_error_ft": float(prediction_error[-1]),
                "end_cumulative_kernel_bias_ft": float(
                    cumulative_kernel_bias[-1]
                ),
            }
        )

        well_episodes = episodes_by_well.get(well)
        if well_episodes is not None:
            for episode in well_episodes.itertuples(index=False):
                start = int(episode.start_suffix_offset)
                end = start + int(episode.rows)
                pre128_start = max(0, start - 128)
                onset_rows = (
                    episode.rows_from_last_within5_to_episode_start
                )
                onset_start = (
                    max(0, start - int(onset_rows))
                    if pd.notna(onset_rows)
                    else pre128_start
                )
                pre128_bias = float(
                    np.sum(true_centered_kernel_bias[pre128_start:start])
                )
                pre128_rate_mean_reversion_bias = float(
                    np.sum(
                        rate_mean_reversion_position_bias[
                            pre128_start:start
                        ]
                    )
                )
                onset_bias = float(
                    np.sum(true_centered_kernel_bias[onset_start:start])
                )
                episode_supported = rate_supported[start:end]
                episode_rows.append(
                    {
                        "episode_id": str(episode.episode_id),
                        "well": well,
                        "rows": int(episode.rows),
                        "episode_sse": float(episode.episode_sse),
                        "mean_error_ft": float(episode.mean_error_ft),
                        "rmse_ft": float(episode.rmse_ft),
                        "pre128_true_minus_init_rate_median": float(
                            episode.pre128_true_minus_init_rate_median
                        ),
                        "observed_emission_evidence_class": str(
                            episode.observed_emission_evidence_class
                        ),
                        "viterbi_recovery_class": str(
                            episode.viterbi_recovery_class
                        ),
                        "pre128_kernel_bias_sum_ft": pre128_bias,
                        "pre128_rate_mean_reversion_position_bias_sum_ft": (
                            pre128_rate_mean_reversion_bias
                        ),
                        "onset_kernel_bias_sum_ft": onset_bias,
                        "pre128_bias_sign_matches_episode_error": bool(
                            np.sign(pre128_bias)
                            == np.sign(float(episode.mean_error_ft))
                        ),
                        "pre128_rate_mean_reversion_bias_sign_matches_episode_error": bool(
                            np.sign(pre128_rate_mean_reversion_bias)
                            == np.sign(float(episode.mean_error_ft))
                        ),
                        "onset_bias_sign_matches_episode_error": bool(
                            np.sign(onset_bias)
                            == np.sign(float(episode.mean_error_ft))
                        ),
                        "episode_kernel_bias_mean_ft_per_row": float(
                            np.mean(true_centered_kernel_bias[start:end])
                        ),
                        "episode_representation_error_abs_mean_ft_per_row": float(
                            np.mean(np.abs(representation_error[start:end]))
                        ),
                        "episode_supported_representation_error_abs_mean_ft_per_row": float(
                            np.mean(
                                np.abs(
                                    representation_error[start:end][
                                        episode_supported
                                    ]
                                )
                            )
                        )
                        if episode_supported.any()
                        else np.nan,
                        "episode_supported_compensating_rate_shift_abs_mean": float(
                            np.mean(
                                np.abs(
                                    compensating_rate_shift[start:end][
                                        episode_supported
                                    ]
                                )
                            )
                        )
                        if episode_supported.any()
                        else np.nan,
                        "episode_supported_compensating_diffusion_rows_proxy_median": float(
                            np.median(
                                compensating_diffusion_rows_proxy[start:end][
                                    episode_supported
                                ]
                            )
                        )
                        if episode_supported.any()
                        else np.nan,
                        "episode_true_rate_outside_fraction": float(
                            np.mean(~episode_supported)
                        ),
                        "episode_true_nearest_rate_edge_fraction": float(
                            np.mean(true_nearest_rate_edge[start:end])
                        ),
                        "episode_best_compensating_rate_edge_fraction": float(
                            np.mean(compensating_rate_edge[start:end])
                        ),
                        "episode_best_compensating_rate_edge_log_mass_loss_sum": float(
                            np.sum(
                                compensating_edge_log_mass_loss[start:end]
                            )
                        ),
                    }
                )

        all_kernel_bias.append(true_centered_kernel_bias)
        all_rate_mean_reversion_position_bias.append(
            rate_mean_reversion_position_bias
        )
        all_representation_error.append(representation_error)
        all_rate_shift_supported.append(
            compensating_rate_shift[rate_supported]
        )
        all_diffusion_rows_proxy_supported.append(
            compensating_diffusion_rows_proxy[rate_supported]
        )
        all_truth_nearest_rate_edge.append(true_nearest_rate_edge)
        all_compensating_rate_edge.append(compensating_rate_edge)
        all_compensating_edge_outward_probability.append(
            compensating_edge_outward_probability[
                compensating_rate_edge
            ]
        )
        total_rows += len(frame)

    by_well = pd.DataFrame(well_rows).sort_values("well").reset_index(drop=True)
    by_episode = (
        pd.DataFrame(episode_rows)
        .sort_values(["well", "episode_id"])
        .reset_index(drop=True)
    )
    kernel_bias = np.concatenate(all_kernel_bias)
    rate_mean_reversion_position_bias = np.concatenate(
        all_rate_mean_reversion_position_bias
    )
    representation_error = np.concatenate(all_representation_error)
    supported_rate_shift = np.concatenate(all_rate_shift_supported)
    supported_diffusion_rows_proxy = np.concatenate(
        all_diffusion_rows_proxy_supported
    )
    truth_nearest_rate_edge = np.concatenate(
        all_truth_nearest_rate_edge
    )
    compensating_rate_edge = np.concatenate(
        all_compensating_rate_edge
    )
    compensating_edge_outward_probability = np.concatenate(
        all_compensating_edge_outward_probability
    )
    initial_position_prior_bias = np.asarray(
        initial_position_prior_biases,
        dtype=np.float64,
    )
    initial_position_prior_variance = np.asarray(
        initial_position_prior_variances,
        dtype=np.float64,
    )
    initial_rate_prior_bias = np.asarray(
        initial_rate_prior_biases,
        dtype=np.float64,
    )
    initial_rate_prior_variance = np.asarray(
        initial_rate_prior_variances,
        dtype=np.float64,
    )
    initial_rate_prior_edge_mass = np.asarray(
        initial_rate_prior_edge_masses,
        dtype=np.float64,
    )
    initial_rate_transition_mass_loss = np.asarray(
        initial_rate_transition_mass_losses,
        dtype=np.float64,
    )
    initial_rate_transition_conditional_mean_error = np.asarray(
        initial_rate_transition_conditional_mean_errors,
        dtype=np.float64,
    )

    class_rows: list[dict[str, Any]] = []
    for column in (
        "observed_emission_evidence_class",
        "viterbi_recovery_class",
    ):
        for label, group in by_episode.groupby(column, sort=True):
            class_rows.append(
                {
                    "class_axis": column,
                    "class": str(label),
                    "episodes": int(len(group)),
                    "episode_sse_fraction": float(
                        group["episode_sse"].sum()
                        / by_episode["episode_sse"].sum()
                    ),
                    "pre128_sign_match_fraction": float(
                        group[
                            "pre128_bias_sign_matches_episode_error"
                        ].mean()
                    ),
                    "pre128_sign_match_sse_fraction": weighted_fraction(
                        group[
                            "pre128_bias_sign_matches_episode_error"
                        ],
                        group["episode_sse"],
                    ),
                    "pre128_bias_vs_mean_error_spearman": float(
                        group["pre128_kernel_bias_sum_ft"].corr(
                            group["mean_error_ft"],
                            method="spearman",
                        )
                    ),
                    (
                        "pre128_bias_vs_mean_error_partial_spearman_"
                        "controlling_initial_rate"
                    ): partial_spearman(
                        group["pre128_kernel_bias_sum_ft"],
                        group["mean_error_ft"],
                        group["pre128_true_minus_init_rate_median"],
                    ),
                    "supported_compensating_rate_shift_abs_sse_weighted_mean": weighted_mean(
                        group[
                            "episode_supported_compensating_rate_shift_abs_mean"
                        ],
                        group["episode_sse"],
                    ),
                }
            )
    class_summary = pd.DataFrame(class_rows)

    summary = {
        "scope": {
            "rows": int(total_rows),
            "wells": int(len(by_well)),
            "persistent_episodes": int(len(by_episode)),
        },
        "kernel": {
            "position_step_ft": POSITION_STEP_FT,
            "configured_sig_p_ft": 0.02,
            "effective_sigma_floor_ft": POSITION_SIGMA_FLOOR_FT,
            "cells": 5,
            "raw_md_step": {
                "rows": int(total_rows),
                "minimum_ft": float(raw_dm_min),
                "maximum_ft": float(raw_dm_max),
                "rows_not_exactly_1ft": int(raw_dm_not_one_rows),
                "rows_below_1ft": int(raw_dm_below_one_rows),
                "rows_nonpositive": int(raw_dm_nonpositive_rows),
                "guard": (
                    "Computed before exp209's max(raw dMD, 1.0) clamp."
                ),
            },
            "actual_motion_moment_tradeoff": {
                "configured_sig_p_ft": 0.02,
                "configured_variance_ft2": 0.02**2,
                "current_effective_sigma_ft": (
                    POSITION_SIGMA_FLOOR_FT
                ),
                "current_kernel_variance_mean_ft2": (
                    current_kernel_variance_sum / total_rows
                ),
                "phase_minimax_sigma_ft": (
                    POSITION_PHASE_MINIMAX_SIGMA_FT
                ),
                "phase_minimax_abs_bias_mean_ft": (
                    phase_minimax_kernel_abs_bias_sum / total_rows
                ),
                "phase_minimax_abs_bias_max_ft": (
                    phase_minimax_kernel_abs_bias_max
                ),
                "phase_minimax_kernel_variance_mean_ft2": (
                    phase_minimax_kernel_variance_sum / total_rows
                ),
                "phase_minimax_variance_ratio_vs_current": (
                    phase_minimax_kernel_variance_sum
                    / current_kernel_variance_sum
                ),
                "exact_mean_minimum_variance_mean_ft2": (
                    exact_mean_minimum_variance_sum / total_rows
                ),
                "exact_mean_minimum_variance_max_ft2": (
                    exact_mean_minimum_variance_max
                ),
                "exact_mean_minimum_variance_ratio_vs_configured": (
                    exact_mean_minimum_variance_sum
                    / total_rows
                    / (0.02**2)
                ),
                "guard": (
                    "Truth-late position-moment diagnostic. Exact-mean "
                    "minimum variance is the two-adjacent-cell lower bound; "
                    "it is not an HMM intervention."
                ),
            },
            "initial_state_prior": {
                "start_position_sigma_ft": START_POSITION_SIGMA_FT,
                "initial_rate_sigma": INITIAL_RATE_SIGMA,
                "position_mean_bias_abs_max_ft": float(
                    np.max(np.abs(initial_position_prior_bias))
                ),
                "position_mean_bias_abs_quantiles_ft": quantiles(
                    np.abs(initial_position_prior_bias)
                ),
                "position_variance_ft2_quantiles": quantiles(
                    initial_position_prior_variance
                ),
                "rate_mean_bias_abs_max": float(
                    np.max(np.abs(initial_rate_prior_bias))
                ),
                "rate_mean_bias_abs_quantiles": quantiles(
                    np.abs(initial_rate_prior_bias)
                ),
                "rate_variance_quantiles": quantiles(
                    initial_rate_prior_variance
                ),
                "rate_edge_mass_quantiles": quantiles(
                    initial_rate_prior_edge_mass
                ),
                "guard": (
                    "Reconstructs exp209's sampled initial position/rate "
                    "priors on each well's actual grids."
                ),
            },
            "rate_transition_first_step": {
                "source_rows": int(rate_transition_source_rows),
                "probability_floor_rows": int(
                    rate_transition_probability_floor_rows
                ),
                "probability_cap_rows": int(
                    rate_transition_probability_cap_rows
                ),
                "interior_mean_error_abs_max": float(
                    rate_transition_interior_mean_error_max
                ),
                "initial_prior_weighted_boundary_mass_loss_quantiles": (
                    quantiles(initial_rate_transition_mass_loss)
                ),
                (
                    "initial_prior_weighted_boundary_conditional_"
                    "mean_error_abs_quantiles"
                ): quantiles(
                    np.abs(
                        initial_rate_transition_conditional_mean_error
                    )
                ),
                "guard": (
                    "All raw suffix dMD values are exactly 1 ft here, so "
                    "the first-step kernel represents every row's local "
                    "rate-transition formula. Boundary metrics are weighted "
                    "by the initial prior only and do not measure later "
                    "decoded edge occupancy."
                ),
            },
            "true_centered_bias_ft_per_row": {
                "mean": float(np.mean(kernel_bias)),
                "abs_mean": float(np.mean(np.abs(kernel_bias))),
                "quantiles": quantiles(kernel_bias),
                "fraction_abs_gt_0p02": float(
                    np.mean(np.abs(kernel_bias) > 0.02)
                ),
                "fraction_abs_gt_0p04": float(
                    np.mean(np.abs(kernel_bias) > 0.04)
                ),
            },
            "truth_centered_rate_mean_reversion_position_bias": {
                "momentum": RATE_MOMENTUM,
                "definition": (
                    "-(1 - momentum) * true_rate * dMD^2"
                ),
                "abs_mean_ft_per_row": float(
                    np.mean(
                        np.abs(rate_mean_reversion_position_bias)
                    )
                ),
                "quantiles_ft_per_row": quantiles(
                    rate_mean_reversion_position_bias
                ),
                "abs_mean_ratio_vs_position_kernel_bias": float(
                    np.mean(
                        np.abs(rate_mean_reversion_position_bias)
                    )
                    / np.mean(np.abs(kernel_bias))
                ),
                "guard": (
                    "Interior truth-centered local expectation only; "
                    "this does not include rate-grid boundaries, emissions, "
                    "or the decoded rate distribution."
                ),
            },
            "best_rate_grid_representation_error_ft_per_row": {
                "abs_mean": float(
                    np.mean(np.abs(representation_error))
                ),
                "abs_quantiles": quantiles(
                    np.abs(representation_error)
                ),
            },
            "supported_compensating_rate_shift": {
                "abs_mean": float(
                    np.mean(np.abs(supported_rate_shift))
                ),
                "abs_quantiles": quantiles(
                    np.abs(supported_rate_shift)
                ),
            },
            "supported_compensating_rate_diffusion_rows_proxy": {
                "sig_r_per_sqrt_md": RATE_SIGMA_PER_SQRT_MD,
                "definition": "compensating_rate_shift^2 / (sig_r^2 * dMD)",
                "quantiles": quantiles(
                    supported_diffusion_rows_proxy
                ),
                "guard": (
                    "This is a local variance-timescale proxy, not an exact "
                    "first-passage time; it omits mean reversion, changing "
                    "targets, emissions, and smoothing."
                ),
            },
            "rate_grid_edge": {
                "truth_nearest_rate_edge_fraction": float(
                    np.mean(truth_nearest_rate_edge)
                ),
                "best_compensating_rate_edge_fraction": float(
                    np.mean(compensating_rate_edge)
                ),
                "persistent_episodes_true_edge_ge_0p10": int(
                    (
                        by_episode[
                            "episode_true_nearest_rate_edge_fraction"
                        ]
                        >= 0.10
                    ).sum()
                ),
                "persistent_episode_sse_fraction_true_edge_ge_0p10": weighted_fraction(
                    (
                        by_episode[
                            "episode_true_nearest_rate_edge_fraction"
                        ]
                        >= 0.10
                    ),
                    by_episode["episode_sse"],
                ),
                "persistent_episodes_compensating_edge_ge_0p10": int(
                    (
                        by_episode[
                            "episode_best_compensating_rate_edge_fraction"
                        ]
                        >= 0.10
                    ).sum()
                ),
                "persistent_episode_sse_fraction_compensating_edge_ge_0p10": weighted_fraction(
                    (
                        by_episode[
                            "episode_best_compensating_rate_edge_fraction"
                        ]
                        >= 0.10
                    ),
                    by_episode["episode_sse"],
                ),
                "compensating_edge_outward_probability_quantiles": quantiles(
                    compensating_edge_outward_probability
                ),
                "episode_compensating_edge_log_mass_loss_vs_rmse_spearman": float(
                    by_episode[
                        "episode_best_compensating_rate_edge_log_mass_loss_sum"
                    ].corr(
                        by_episode["rmse_ft"],
                        method="spearman",
                    )
                ),
                (
                    "episode_compensating_edge_fraction_vs_rmse_"
                    "partial_spearman"
                ): partial_spearman_controls(
                    by_episode[
                        "episode_best_compensating_rate_edge_fraction"
                    ],
                    by_episode["rmse_ft"],
                    [
                        by_episode["rows"],
                        by_episode["episode_true_rate_outside_fraction"],
                        by_episode[
                            "episode_supported_compensating_rate_shift_abs_mean"
                        ],
                        by_episode[
                            "episode_supported_compensating_diffusion_rows_proxy_median"
                        ],
                    ],
                ),
                "guard": (
                    "Truth-nearest and oracle-compensating states are "
                    "truth-late local diagnostics, not decoded posterior "
                    "rate-edge occupancy."
                ),
            },
            "five_cell_phase_bias": five_cell_phase_bias_audit(),
        },
        "association": {
            "well_prediction_error_vs_cumulative_kernel_bias_corr_median": float(
                by_well[
                    "prediction_error_vs_cumulative_kernel_bias_corr"
                ].median()
            ),
            "wells_with_positive_error_vs_cumulative_bias_corr": int(
                (
                    by_well[
                        "prediction_error_vs_cumulative_kernel_bias_corr"
                    ]
                    > 0
                ).sum()
            ),
            "well_end_bias_vs_end_error_spearman": float(
                by_well["end_cumulative_kernel_bias_ft"].corr(
                    by_well["end_prediction_error_ft"],
                    method="spearman",
                )
            ),
            "episode_pre128_bias_vs_mean_error_spearman": float(
                by_episode["pre128_kernel_bias_sum_ft"].corr(
                    by_episode["mean_error_ft"],
                    method="spearman",
                )
            ),
            "episode_pre128_rate_mean_reversion_bias_vs_mean_error_spearman": float(
                by_episode[
                    "pre128_rate_mean_reversion_position_bias_sum_ft"
                ].corr(
                    by_episode["mean_error_ft"],
                    method="spearman",
                )
            ),
            "episode_pre128_rate_mean_reversion_bias_sign_match_fraction": float(
                by_episode[
                    "pre128_rate_mean_reversion_bias_sign_matches_episode_error"
                ].mean()
            ),
            (
                "episode_pre128_rate_mean_reversion_bias_vs_mean_error_"
                "partial_spearman_controlling_kernel_bias"
            ): partial_spearman(
                by_episode[
                    "pre128_rate_mean_reversion_position_bias_sum_ft"
                ],
                by_episode["mean_error_ft"],
                by_episode["pre128_kernel_bias_sum_ft"],
            ),
            "episode_pre128_bias_vs_rmse_spearman": float(
                by_episode["pre128_kernel_bias_sum_ft"].corr(
                    by_episode["rmse_ft"],
                    method="spearman",
                )
            ),
            "episode_abs_pre128_bias_vs_rmse_spearman": float(
                by_episode["pre128_kernel_bias_sum_ft"].abs().corr(
                    by_episode["rmse_ft"],
                    method="spearman",
                )
            ),
            (
                "episode_pre128_bias_vs_mean_error_partial_spearman_"
                "controlling_initial_rate"
            ): partial_spearman(
                by_episode["pre128_kernel_bias_sum_ft"],
                by_episode["mean_error_ft"],
                by_episode["pre128_true_minus_init_rate_median"],
            ),
            "episode_signed_offset_rank_r2": {
                "kernel_bias_only": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [by_episode["pre128_kernel_bias_sum_ft"]],
                ),
                "rate_mean_reversion_bias_only": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [
                        by_episode[
                            "pre128_rate_mean_reversion_position_bias_sum_ft"
                        ]
                    ],
                ),
                "initial_rate_mismatch_only": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [
                        by_episode[
                            "pre128_true_minus_init_rate_median"
                        ]
                    ],
                ),
                "kernel_bias_plus_initial_rate_mismatch": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [
                        by_episode["pre128_kernel_bias_sum_ft"],
                        by_episode[
                            "pre128_true_minus_init_rate_median"
                        ],
                    ],
                ),
                "kernel_plus_rate_mean_reversion_bias": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [
                        by_episode["pre128_kernel_bias_sum_ft"],
                        by_episode[
                            "pre128_rate_mean_reversion_position_bias_sum_ft"
                        ],
                    ],
                ),
                "kernel_plus_rate_mean_reversion_plus_initial_rate": rank_regression_r2(
                    by_episode["mean_error_ft"],
                    [
                        by_episode["pre128_kernel_bias_sum_ft"],
                        by_episode[
                            "pre128_rate_mean_reversion_position_bias_sum_ft"
                        ],
                        by_episode[
                            "pre128_true_minus_init_rate_median"
                        ],
                    ],
                ),
            },
            "episode_pre128_sign_match_fraction": float(
                by_episode[
                    "pre128_bias_sign_matches_episode_error"
                ].mean()
            ),
            "episode_pre128_sign_match_sse_fraction": weighted_fraction(
                by_episode[
                    "pre128_bias_sign_matches_episode_error"
                ],
                by_episode["episode_sse"],
            ),
            "episode_onset_sign_match_fraction": float(
                by_episode[
                    "onset_bias_sign_matches_episode_error"
                ].mean()
            ),
            "episode_onset_sign_match_sse_fraction": weighted_fraction(
                by_episode[
                    "onset_bias_sign_matches_episode_error"
                ],
                by_episode["episode_sse"],
            ),
            "episode_representation_abs_error_vs_rmse_spearman": float(
                by_episode[
                    "episode_representation_error_abs_mean_ft_per_row"
                ].corr(
                    by_episode["rmse_ft"],
                    method="spearman",
                )
            ),
            "episode_supported_rate_shift_abs_vs_rmse_spearman": float(
                by_episode[
                    "episode_supported_compensating_rate_shift_abs_mean"
                ].corr(
                    by_episode["rmse_ft"],
                    method="spearman",
                )
            ),
            "episode_compensating_diffusion_rows_proxy_vs_rmse_spearman": float(
                by_episode[
                    "episode_supported_compensating_diffusion_rows_proxy_median"
                ].corr(
                    by_episode["rmse_ft"],
                    method="spearman",
                )
            ),
        },
        "interpretation_guard": (
            "Truth-centered kernel bias is a counterfactual transition diagnostic, "
            "not a deployable feature. Association establishes a directional "
            "mechanism but does not alone separate forward, backward, and emission "
            "contributions or prove that changing sig_p improves CV."
        ),
    }

    by_well.to_csv(output / "by_well_metrics.csv", index=False)
    by_episode.to_csv(output / "episode_metrics.csv", index=False)
    class_summary.to_csv(output / "episode_class_summary.csv", index=False)
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
