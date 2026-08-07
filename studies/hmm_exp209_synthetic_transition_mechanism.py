#!/usr/bin/env python3
"""Controlled synthetic audit of exp209 transition-induced TVT offsets.

This study removes geological and GR-reference ambiguity.  A synthetic path has
constant U-rate and receives a correctly centered Gaussian position emission.
The forward-backward recursion uses exp209's 41-state rate transition and
normalized five-cell position transition.  One-at-a-time counterfactuals test:

- the current 0.1225 ft effective position sigma;
- the deterministic phase-minimax 0.2325 ft position sigma;
- minimum-variance adjacent-cell transport with an exact position mean;
- source-row normalization at rate-grid boundaries;
- removal of rate mean reversion by setting momentum to one.

Eight Gaussian variants form a full 2x2x2 factorial over position sigma, rate
boundary normalization, and rate momentum.  Four additional variants apply
exact-mean position transport across the boundary/momentum combinations.  The
result is a mechanism diagnostic, not an OOF parameter selection or a
deployable prediction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POSITION_STEP_FT = 0.35
CURRENT_POSITION_SIGMA_FT = 0.1225
PHASE_MINIMAX_POSITION_SIGMA_FT = 0.2325
RATE_SIGMA_PER_SQRT_MD = 0.002
CURRENT_MOMENTUM = 0.998
RATE_SPAN = 0.10
RATE_COUNT = 41
POSITION_HALF_CELLS = 210
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_synthetic_transition_mechanism_20260726"
)


@dataclass(frozen=True)
class Variant:
    name: str
    position_sigma_ft: float
    momentum: float
    normalize_rate_boundary_rows: bool
    position_kernel_kind: str = "gaussian"


VARIANTS = (
    Variant(
        "current",
        CURRENT_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        False,
    ),
    Variant(
        "position_sigma_phase_minimax",
        PHASE_MINIMAX_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        False,
    ),
    Variant(
        "rate_boundary_row_normalized",
        CURRENT_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        True,
    ),
    Variant(
        "momentum_one",
        CURRENT_POSITION_SIGMA_FT,
        1.0,
        False,
    ),
    Variant(
        "position_sigma_plus_boundary_normalized",
        PHASE_MINIMAX_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        True,
    ),
    Variant(
        "position_sigma_plus_momentum_one",
        PHASE_MINIMAX_POSITION_SIGMA_FT,
        1.0,
        False,
    ),
    Variant(
        "boundary_normalized_plus_momentum_one",
        CURRENT_POSITION_SIGMA_FT,
        1.0,
        True,
    ),
    Variant(
        "all_three_transition_corrections",
        PHASE_MINIMAX_POSITION_SIGMA_FT,
        1.0,
        True,
    ),
    Variant(
        "exact_mean_transport",
        CURRENT_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        False,
        "exact_mean",
    ),
    Variant(
        "exact_mean_transport_plus_boundary_normalized",
        CURRENT_POSITION_SIGMA_FT,
        CURRENT_MOMENTUM,
        True,
        "exact_mean",
    ),
    Variant(
        "exact_mean_transport_plus_momentum_one",
        CURRENT_POSITION_SIGMA_FT,
        1.0,
        False,
        "exact_mean",
    ),
    Variant(
        "exact_mean_all_transition_corrections",
        CURRENT_POSITION_SIGMA_FT,
        1.0,
        True,
        "exact_mean",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=512)
    return parser.parse_args()


def rate_transition_matrix(
    rates: np.ndarray,
    momentum: float,
    normalize_boundary_rows: bool,
) -> np.ndarray:
    rate_step = float(rates[1] - rates[0])
    rate_var_cells = (RATE_SIGMA_PER_SQRT_MD / rate_step) ** 2
    transition = np.zeros((len(rates), len(rates)), dtype=np.float64)
    for source, rate in enumerate(rates):
        mean_rate_move = -(1.0 - momentum) * rate / rate_step
        p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
        p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
        total = p_plus + p_minus
        if total > 0.9:
            p_plus *= 0.9 / total
            p_minus *= 0.9 / total
        for delta, probability in (
            (-1, p_minus),
            (0, 1.0 - p_plus - p_minus),
            (1, p_plus),
        ):
            destination = source + delta
            if 0 <= destination < len(rates):
                transition[source, destination] = probability
        if normalize_boundary_rows:
            transition[source] /= transition[source].sum()
    return transition


def position_kernels(
    rates: np.ndarray,
    sigma_ft: float,
) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.empty((len(rates), 5), dtype=np.int64)
    weights = np.empty((len(rates), 5), dtype=np.float64)
    for rate_index, rate in enumerate(rates):
        center = int(np.floor(rate / POSITION_STEP_FT + 0.5))
        for kernel_index in range(5):
            shift = center - 2 + kernel_index
            shifts[rate_index, kernel_index] = shift
            delta = shift * POSITION_STEP_FT - rate
            weights[rate_index, kernel_index] = np.exp(
                -0.5 * (delta / sigma_ft) ** 2
            )
        weights[rate_index] /= weights[rate_index].sum()
    return shifts, weights


def exact_mean_position_kernels(
    rates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-variance adjacent-cell transport with exact rate mean."""
    shared_shifts = np.arange(-2, 3, dtype=np.int64)
    shifts = np.tile(shared_shifts, (len(rates), 1))
    weights = np.zeros((len(rates), 5), dtype=np.float64)
    for rate_index, rate in enumerate(rates):
        lower_shift = int(np.floor(rate / POSITION_STEP_FT))
        upper_shift = lower_shift + 1
        if lower_shift < -2 or upper_shift > 2:
            raise ValueError("exact-mean support exceeds five cells")
        fraction = (
            rate - lower_shift * POSITION_STEP_FT
        ) / POSITION_STEP_FT
        weights[rate_index, lower_shift + 2] = 1.0 - fraction
        weights[rate_index, upper_shift + 2] = fraction
    if not np.allclose(
        np.sum(weights, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("exact-mean position weights are not normalized")
    represented_mean = weights @ (
        shared_shifts.astype(np.float64) * POSITION_STEP_FT
    )
    if not np.allclose(
        represented_mean,
        rates,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("exact-mean position transport missed its mean")
    return shifts, weights


def transition_forward(
    previous: np.ndarray,
    rate_transition: np.ndarray,
    position_shifts: np.ndarray,
    position_weights: np.ndarray,
) -> np.ndarray:
    after_rate = (
        previous
        * np.diag(rate_transition)[None, :]
    )
    after_rate[:, 1:] += (
        previous[:, :-1]
        * np.diag(rate_transition, k=1)[None, :]
    )
    after_rate[:, :-1] += (
        previous[:, 1:]
        * np.diag(rate_transition, k=-1)[None, :]
    )
    current = np.zeros_like(after_rate)
    for kernel_index in range(position_shifts.shape[1]):
        shifts = position_shifts[:, kernel_index]
        if not np.all(shifts == shifts[0]):
            raise ValueError(
                "synthetic vectorization requires a shared shift support"
            )
        shift = int(shifts[0])
        weight = position_weights[:, kernel_index][None, :]
        if shift > 0:
            current[shift:] += weight * after_rate[:-shift]
        elif shift < 0:
            current[:shift] += weight * after_rate[-shift:]
        else:
            current += weight * after_rate
    return current


def transition_backward(
    destination_value: np.ndarray,
    rate_transition: np.ndarray,
    position_shifts: np.ndarray,
    position_weights: np.ndarray,
) -> np.ndarray:
    before_position = np.zeros_like(destination_value)
    for kernel_index in range(position_shifts.shape[1]):
        shifts = position_shifts[:, kernel_index]
        if not np.all(shifts == shifts[0]):
            raise ValueError(
                "synthetic vectorization requires a shared shift support"
            )
        shift = int(shifts[0])
        weight = position_weights[:, kernel_index][None, :]
        if shift > 0:
            before_position[:-shift] += (
                weight * destination_value[shift:]
            )
        elif shift < 0:
            before_position[-shift:] += (
                weight * destination_value[:shift]
            )
        else:
            before_position += weight * destination_value
    source_value = (
        before_position
        * np.diag(rate_transition)[None, :]
    )
    source_value[:, :-1] += (
        before_position[:, 1:]
        * np.diag(rate_transition, k=1)[None, :]
    )
    source_value[:, 1:] += (
        before_position[:, :-1]
        * np.diag(rate_transition, k=-1)[None, :]
    )
    return source_value


def transition_forward_reference(
    previous: np.ndarray,
    rate_transition: np.ndarray,
    position_shifts: np.ndarray,
    position_weights: np.ndarray,
) -> np.ndarray:
    """Literal exp270-style source/destination enumeration for parity."""
    current = np.zeros_like(previous)
    position_count, rate_count = previous.shape
    for source_position in range(position_count):
        for source_rate in range(rate_count):
            for destination_rate in range(
                max(0, source_rate - 1),
                min(rate_count, source_rate + 2),
            ):
                rate_probability = rate_transition[
                    source_rate,
                    destination_rate,
                ]
                for kernel_index in range(
                    position_shifts.shape[1]
                ):
                    destination_position = (
                        source_position
                        + position_shifts[
                            destination_rate,
                            kernel_index,
                        ]
                    )
                    if 0 <= destination_position < position_count:
                        current[
                            destination_position,
                            destination_rate,
                        ] += (
                            previous[source_position, source_rate]
                            * rate_probability
                            * position_weights[
                                destination_rate,
                                kernel_index,
                            ]
                        )
    return current


def transition_backward_reference(
    destination_value: np.ndarray,
    rate_transition: np.ndarray,
    position_shifts: np.ndarray,
    position_weights: np.ndarray,
) -> np.ndarray:
    """Literal exp270-style backward enumeration for parity."""
    source_value = np.zeros_like(destination_value)
    position_count, rate_count = destination_value.shape
    for source_position in range(position_count):
        for source_rate in range(rate_count):
            for destination_rate in range(
                max(0, source_rate - 1),
                min(rate_count, source_rate + 2),
            ):
                rate_probability = rate_transition[
                    source_rate,
                    destination_rate,
                ]
                for kernel_index in range(
                    position_shifts.shape[1]
                ):
                    destination_position = (
                        source_position
                        + position_shifts[
                            destination_rate,
                            kernel_index,
                        ]
                    )
                    if 0 <= destination_position < position_count:
                        source_value[
                            source_position,
                            source_rate,
                        ] += (
                            rate_probability
                            * position_weights[
                                destination_rate,
                                kernel_index,
                            ]
                            * destination_value[
                                destination_position,
                                destination_rate,
                            ]
                        )
    return source_value


def normalized(values: np.ndarray, label: str) -> np.ndarray:
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label}: invalid probability total {total}")
    result = values / total
    if not np.isfinite(result).all():
        raise ValueError(f"{label}: non-finite normalized probability")
    return result


def position_mean(
    state_probability: np.ndarray,
    positions: np.ndarray,
) -> float:
    return float(
        np.dot(state_probability.sum(axis=1), positions)
    )


def position_mean_and_std(
    state_probability: np.ndarray,
    positions: np.ndarray,
) -> tuple[float, float]:
    position_probability = state_probability.sum(axis=1)
    mean = float(np.dot(position_probability, positions))
    variance = max(
        float(np.dot(position_probability, positions**2) - mean**2),
        0.0,
    )
    return mean, float(np.sqrt(variance))


def edge_rate_mass(state_probability: np.ndarray) -> float:
    return float(
        state_probability[:, 0].sum()
        + state_probability[:, -1].sum()
    )


def run_scenario(
    steps: int,
    true_rate: float,
    emission_sigma_ft: float | None,
    variant: Variant,
) -> dict[str, Any]:
    rates = np.linspace(-RATE_SPAN, RATE_SPAN, RATE_COUNT)
    positions = (
        np.arange(
            -POSITION_HALF_CELLS,
            POSITION_HALF_CELLS + 1,
            dtype=np.float64,
        )
        * POSITION_STEP_FT
    )
    rate_transition = rate_transition_matrix(
        rates,
        variant.momentum,
        variant.normalize_rate_boundary_rows,
    )
    if variant.position_kernel_kind == "gaussian":
        position_shifts, position_weights = position_kernels(
            rates,
            variant.position_sigma_ft,
        )
    elif variant.position_kernel_kind == "exact_mean":
        position_shifts, position_weights = (
            exact_mean_position_kernels(rates)
        )
    else:
        raise ValueError(
            f"unknown position kernel {variant.position_kernel_kind}"
        )

    start_rate_index = int(np.argmin(np.abs(rates - true_rate)))
    if not np.isclose(rates[start_rate_index], true_rate, atol=1e-12):
        raise ValueError(f"true rate {true_rate} is not on the rate grid")
    start_position_index = int(np.flatnonzero(positions == 0.0)[0])
    previous = np.zeros((len(positions), len(rates)), dtype=np.float64)
    previous[start_position_index, start_rate_index] = 1.0

    alpha = np.empty(
        (steps, len(positions), len(rates)),
        dtype=np.float32,
    )
    predictive_mean = np.empty(steps, dtype=np.float64)
    filtered_mean = np.empty(steps, dtype=np.float64)
    filtered_position_std = np.empty(steps, dtype=np.float64)
    filtered_rate_edge_mass = np.empty(steps, dtype=np.float64)
    emissions = np.empty((steps, len(positions)), dtype=np.float64)

    for time_index in range(steps):
        predictive = normalized(
            transition_forward(
                previous,
                rate_transition,
                position_shifts,
                position_weights,
            ),
            f"predictive[{time_index}]",
        )
        predictive_mean[time_index] = position_mean(
            predictive,
            positions,
        )
        truth = true_rate * (time_index + 1)
        if emission_sigma_ft is None:
            emission = np.ones(len(positions), dtype=np.float64)
        else:
            emission = np.exp(
                -0.5
                * ((positions - truth) / emission_sigma_ft) ** 2
            )
        emissions[time_index] = emission
        filtered = normalized(
            predictive * emission[:, None],
            f"filtered[{time_index}]",
        )
        alpha[time_index] = filtered.astype(np.float32)
        (
            filtered_mean[time_index],
            filtered_position_std[time_index],
        ) = position_mean_and_std(
            filtered,
            positions,
        )
        filtered_rate_edge_mass[time_index] = edge_rate_mass(
            filtered
        )
        previous = filtered

    smoothed_mean = np.empty(steps, dtype=np.float64)
    smoothed_position_std = np.empty(steps, dtype=np.float64)
    smoothed_rate_edge_mass = np.empty(steps, dtype=np.float64)
    beta_next = np.ones(
        (len(positions), len(rates)),
        dtype=np.float64,
    )
    for time_index in range(steps - 1, -1, -1):
        smoothed = normalized(
            alpha[time_index].astype(np.float64) * beta_next,
            f"smoothed[{time_index}]",
        )
        (
            smoothed_mean[time_index],
            smoothed_position_std[time_index],
        ) = position_mean_and_std(
            smoothed,
            positions,
        )
        smoothed_rate_edge_mass[time_index] = edge_rate_mass(
            smoothed
        )
        if time_index > 0:
            beta_next = normalized(
                transition_backward(
                    emissions[time_index, :, None] * beta_next,
                    rate_transition,
                    position_shifts,
                    position_weights,
                ),
                f"beta[{time_index - 1}]",
            )

    truth_path = true_rate * np.arange(
        1,
        steps + 1,
        dtype=np.float64,
    )
    predictive_error = predictive_mean - truth_path
    filtered_error = filtered_mean - truth_path
    smoothed_error = smoothed_mean - truth_path
    position_edge_mass = float(
        max(
            alpha[:, 0, :].sum(axis=1).max(),
            alpha[:, -1, :].sum(axis=1).max(),
        )
    )
    return {
        "variant": variant.name,
        "steps": steps,
        "true_rate": true_rate,
        "emission_sigma_ft": (
            emission_sigma_ft
            if emission_sigma_ft is not None
            else np.nan
        ),
        "emission": (
            f"gaussian_{emission_sigma_ft:g}ft"
            if emission_sigma_ft is not None
            else "neutral"
        ),
        "position_kernel_kind": variant.position_kernel_kind,
        "predictive_rmse_ft": float(
            np.sqrt(np.mean(predictive_error**2))
        ),
        "filtered_rmse_ft": float(
            np.sqrt(np.mean(filtered_error**2))
        ),
        "smoothed_rmse_ft": float(
            np.sqrt(np.mean(smoothed_error**2))
        ),
        "predictive_end_error_ft": float(predictive_error[-1]),
        "filtered_end_error_ft": float(filtered_error[-1]),
        "smoothed_end_error_ft": float(smoothed_error[-1]),
        "filtered_position_std_mean_ft": float(
            filtered_position_std.mean()
        ),
        "filtered_position_std_end_ft": float(
            filtered_position_std[-1]
        ),
        "smoothed_position_std_mean_ft": float(
            smoothed_position_std.mean()
        ),
        "smoothed_position_std_end_ft": float(
            smoothed_position_std[-1]
        ),
        "filtered_rate_edge_mass_mean": float(
            filtered_rate_edge_mass.mean()
        ),
        "filtered_rate_edge_mass_max": float(
            filtered_rate_edge_mass.max()
        ),
        "smoothed_rate_edge_mass_mean": float(
            smoothed_rate_edge_mass.mean()
        ),
        "smoothed_rate_edge_mass_max": float(
            smoothed_rate_edge_mass.max()
        ),
        "maximum_position_edge_mass": position_edge_mass,
        "rate_transition_min_row_sum": float(
            rate_transition.sum(axis=1).min()
        ),
        "rate_transition_max_row_sum": float(
            rate_transition.sum(axis=1).max()
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    parity_rates = np.linspace(-RATE_SPAN, RATE_SPAN, RATE_COUNT)
    parity_transition = rate_transition_matrix(
        parity_rates,
        CURRENT_MOMENTUM,
        False,
    )
    parity_shifts, parity_weights = position_kernels(
        parity_rates,
        CURRENT_POSITION_SIGMA_FT,
    )
    parity_rng = np.random.default_rng(20260726)
    parity_shape = (
        2 * POSITION_HALF_CELLS + 1,
        RATE_COUNT,
    )
    parity_source = normalized(
        parity_rng.random(parity_shape),
        "parity_source",
    )
    parity_destination = normalized(
        parity_rng.random(parity_shape),
        "parity_destination",
    )
    forward_parity_max_abs = float(
        np.max(
            np.abs(
                transition_forward(
                    parity_source,
                    parity_transition,
                    parity_shifts,
                    parity_weights,
                )
                - transition_forward_reference(
                    parity_source,
                    parity_transition,
                    parity_shifts,
                    parity_weights,
                )
            )
        )
    )
    backward_parity_max_abs = float(
        np.max(
            np.abs(
                transition_backward(
                    parity_destination,
                    parity_transition,
                    parity_shifts,
                    parity_weights,
                )
                - transition_backward_reference(
                    parity_destination,
                    parity_transition,
                    parity_shifts,
                    parity_weights,
                )
            )
        )
    )
    if max(
        forward_parity_max_abs,
        backward_parity_max_abs,
    ) > 1e-15:
        raise ValueError("optimized transition failed literal parity")

    scenarios = (
        (0.025, None),
        (0.025, 10.0),
        (0.025, 2.0),
        (0.050, None),
        (0.050, 10.0),
        (0.050, 2.0),
        (0.100, 2.0),
        (-0.050, 2.0),
    )
    rows = [
        run_scenario(
            args.steps,
            true_rate,
            emission_sigma,
            variant,
        )
        for true_rate, emission_sigma in scenarios
        for variant in VARIANTS
    ]
    results = pd.DataFrame(rows)
    if float(results["maximum_position_edge_mass"].max()) > 1e-8:
        raise ValueError("synthetic position grid is too narrow")
    normalized_variants = results[
        results["variant"].isin(
            [
                variant.name
                for variant in VARIANTS
                if variant.normalize_rate_boundary_rows
            ]
        )
    ]
    if not np.allclose(
        normalized_variants["rate_transition_min_row_sum"],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("normalized rate-boundary variants are not stochastic")

    neutral = results[results["emission"] == "neutral"]
    if not np.allclose(
        neutral["predictive_rmse_ft"],
        neutral["filtered_rmse_ft"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("neutral emissions changed the filtered distribution")
    normalized_neutral = normalized_variants[
        normalized_variants["emission"] == "neutral"
    ]
    if not np.allclose(
        normalized_neutral["filtered_rmse_ft"],
        normalized_neutral["smoothed_rmse_ft"],
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "stochastic neutral transition changed under smoothing"
        )

    symmetry = results[
        (results["emission"] == "gaussian_2ft")
        & results["true_rate"].isin([-0.05, 0.05])
    ].pivot(index="variant", columns="true_rate")
    for metric in (
        "predictive_end_error_ft",
        "filtered_end_error_ft",
        "smoothed_end_error_ft",
    ):
        signed_sum = (
            symmetry[metric][-0.05]
            + symmetry[metric][0.05]
        )
        if float(np.abs(signed_sum).max()) > 1e-5:
            raise ValueError(f"{metric}: sign symmetry failed")

    current = results[results["variant"] == "current"].copy()
    summary = {
        "steps": int(args.steps),
        "scenarios": int(len(scenarios)),
        "variants": [variant.name for variant in VARIANTS],
        "rows": int(len(results)),
        "current_transition_boundary_row_sum": {
            "minimum": float(
                current["rate_transition_min_row_sum"].min()
            ),
            "maximum": float(
                current["rate_transition_max_row_sum"].max()
            ),
        },
        "maximum_position_edge_mass": float(
            results["maximum_position_edge_mass"].max()
        ),
        "sign_symmetry_max_error_ft": float(
            max(
                np.abs(
                    symmetry[metric][-0.05]
                    + symmetry[metric][0.05]
                ).max()
                for metric in (
                    "predictive_end_error_ft",
                    "filtered_end_error_ft",
                    "smoothed_end_error_ft",
                )
            )
        ),
        "literal_transition_parity_max_abs": {
            "forward": forward_parity_max_abs,
            "backward": backward_parity_max_abs,
        },
        "interpretation_guard": (
            "Synthetic constant-rate Gaussian-position evidence isolates "
            "transition mechanics. It is not an exp209 OOF intervention, "
            "does not reproduce typewell GR alias, and does not select a "
            "production parameter."
        ),
    }
    results.to_csv(output / "scenario_metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
