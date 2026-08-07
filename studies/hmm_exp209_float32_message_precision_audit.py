#!/usr/bin/env python3
"""Stress-test exp209-style unnormalized float32 log messages.

This controlled diagnostic uses the same 41-state rate transition and
five-cell position transition as exp209.  It compares:

1. normalized float64 log messages;
2. normalized float32 log messages;
3. unnormalized float32 log messages, matching exp209's scale handling.

An additive emission-log offset is also applied uniformly to every state.
It cannot change an exact posterior, but it increases the absolute magnitude
of unnormalized messages and exposes precision loss.  This is a numerical
mechanism audit, not an OOF prediction or parameter selection.
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
POSITION_SIGMA_FT = 0.1225
RATE_SIGMA_PER_SQRT_MD = 0.002
RATE_MOMENTUM = 0.998
RATE_SPAN = 0.10
RATE_COUNT = 41
POSITION_HALF_CELLS = 210
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_float32_message_precision_audit_20260726"
)


@dataclass(frozen=True)
class PrecisionMode:
    name: str
    dtype: type[np.float32] | type[np.float64]
    normalize_messages: bool


PRECISION_MODES = (
    PrecisionMode("normalized_float64", np.float64, True),
    PrecisionMode("normalized_float32", np.float32, True),
    PrecisionMode("unnormalized_float32", np.float32, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=512)
    return parser.parse_args()


def rate_transition_matrix(rates: np.ndarray) -> np.ndarray:
    rate_step = float(rates[1] - rates[0])
    rate_var_cells = (RATE_SIGMA_PER_SQRT_MD / rate_step) ** 2
    transition = np.zeros((len(rates), len(rates)), dtype=np.float64)
    for source, rate in enumerate(rates):
        mean_move = (
            -(1.0 - RATE_MOMENTUM) * rate / rate_step
        )
        p_plus = max(0.5 * (rate_var_cells + mean_move), 1e-12)
        p_minus = max(0.5 * (rate_var_cells - mean_move), 1e-12)
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
    return transition


def position_weights(rates: np.ndarray) -> np.ndarray:
    weights = np.empty((len(rates), 5), dtype=np.float64)
    for rate_index, rate in enumerate(rates):
        center = int(np.floor(rate / POSITION_STEP_FT + 0.5))
        if center != 0:
            raise ValueError("shared five-cell support assumption failed")
        shifts = np.arange(-2, 3, dtype=np.float64)
        delta = shifts * POSITION_STEP_FT - rate
        weights[rate_index] = np.exp(
            -0.5 * (delta / POSITION_SIGMA_FT) ** 2
        )
        weights[rate_index] /= weights[rate_index].sum()
    return weights


def log_probabilities(
    probabilities: np.ndarray,
    dtype: type[np.float32] | type[np.float64],
    negative: float,
) -> np.ndarray:
    result = np.full(probabilities.shape, negative, dtype=dtype)
    positive = probabilities > 0.0
    result[positive] = np.log(probabilities[positive]).astype(dtype)
    return result


def normalize_log_message(
    message: np.ndarray,
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray:
    maximum = float(np.max(message))
    log_total = maximum + float(
        np.log(
            np.sum(
                np.exp(message.astype(np.float64) - maximum),
                dtype=np.float64,
            )
        )
    )
    return (message - dtype(log_total)).astype(dtype)


def logadd(
    left: np.ndarray,
    right: np.ndarray,
    dtype: type[np.float32] | type[np.float64],
) -> np.ndarray:
    return np.logaddexp(left, right).astype(dtype)


def transition_forward_log(
    previous: np.ndarray,
    rate_log: np.ndarray,
    position_log_weight: np.ndarray,
    dtype: type[np.float32] | type[np.float64],
    negative: float,
) -> np.ndarray:
    after_rate = (
        previous + np.diag(rate_log)[None, :]
    ).astype(dtype)
    after_rate[:, 1:] = logadd(
        after_rate[:, 1:],
        previous[:, :-1] + np.diag(rate_log, k=1)[None, :],
        dtype,
    )
    after_rate[:, :-1] = logadd(
        after_rate[:, :-1],
        previous[:, 1:] + np.diag(rate_log, k=-1)[None, :],
        dtype,
    )

    current = np.full(after_rate.shape, negative, dtype=dtype)
    for kernel_index, shift in enumerate(range(-2, 3)):
        weight = position_log_weight[:, kernel_index][None, :]
        if shift > 0:
            current[shift:] = logadd(
                current[shift:],
                after_rate[:-shift] + weight,
                dtype,
            )
        elif shift < 0:
            current[:shift] = logadd(
                current[:shift],
                after_rate[-shift:] + weight,
                dtype,
            )
        else:
            current = logadd(current, after_rate + weight, dtype)
    return current


def transition_backward_log(
    destination_value: np.ndarray,
    rate_log: np.ndarray,
    position_log_weight: np.ndarray,
    dtype: type[np.float32] | type[np.float64],
    negative: float,
) -> np.ndarray:
    before_position = np.full(
        destination_value.shape,
        negative,
        dtype=dtype,
    )
    for kernel_index, shift in enumerate(range(-2, 3)):
        weight = position_log_weight[:, kernel_index][None, :]
        if shift > 0:
            before_position[:-shift] = logadd(
                before_position[:-shift],
                destination_value[shift:] + weight,
                dtype,
            )
        elif shift < 0:
            before_position[-shift:] = logadd(
                before_position[-shift:],
                destination_value[:shift] + weight,
                dtype,
            )
        else:
            before_position = logadd(
                before_position,
                destination_value + weight,
                dtype,
            )

    source = (
        before_position + np.diag(rate_log)[None, :]
    ).astype(dtype)
    source[:, :-1] = logadd(
        source[:, :-1],
        before_position[:, 1:]
        + np.diag(rate_log, k=1)[None, :],
        dtype,
    )
    source[:, 1:] = logadd(
        source[:, 1:],
        before_position[:, :-1]
        + np.diag(rate_log, k=-1)[None, :],
        dtype,
    )
    return source


def position_mean_from_log(
    log_state: np.ndarray,
    positions: np.ndarray,
) -> float:
    maximum = float(np.max(log_state))
    state = np.exp(log_state.astype(np.float64) - maximum)
    state /= np.sum(state, dtype=np.float64)
    return float(
        np.dot(state.sum(axis=1), positions)
    )


def run_scenario(
    *,
    steps: int,
    true_rate: float,
    emission_sigma_ft: float,
    additive_emission_log_offset: float,
    mode: PrecisionMode,
) -> dict[str, Any]:
    dtype = mode.dtype
    negative = float(dtype(-1e18))
    rates = np.linspace(-RATE_SPAN, RATE_SPAN, RATE_COUNT)
    positions = (
        np.arange(
            -POSITION_HALF_CELLS,
            POSITION_HALF_CELLS + 1,
            dtype=np.float64,
        )
        * POSITION_STEP_FT
    )
    rate_log = log_probabilities(
        rate_transition_matrix(rates),
        dtype,
        negative,
    )
    position_log_weight = np.log(
        position_weights(rates)
    ).astype(dtype)

    start_rate = int(np.argmin(np.abs(rates - true_rate)))
    if not np.isclose(rates[start_rate], true_rate, atol=1e-12):
        raise ValueError("true rate must be on the rate grid")
    start_position = int(np.flatnonzero(positions == 0.0)[0])
    previous = np.full(
        (len(positions), len(rates)),
        negative,
        dtype=dtype,
    )
    previous[start_position, start_rate] = dtype(0.0)

    alpha = np.empty(
        (steps, len(positions), len(rates)),
        dtype=dtype,
    )
    emissions = np.empty(
        (steps, len(positions)),
        dtype=dtype,
    )
    filtered_mean = np.empty(steps, dtype=np.float64)
    alpha_abs_max = 0.0
    for time_index in range(steps):
        predictive = transition_forward_log(
            previous,
            rate_log,
            position_log_weight,
            dtype,
            negative,
        )
        truth = true_rate * (time_index + 1)
        emission = (
            -0.5
            * ((positions - truth) / emission_sigma_ft) ** 2
            + additive_emission_log_offset
        ).astype(dtype)
        emissions[time_index] = emission
        current = (predictive + emission[:, None]).astype(dtype)
        if mode.normalize_messages:
            current = normalize_log_message(current, dtype)
        alpha[time_index] = current
        filtered_mean[time_index] = position_mean_from_log(
            current,
            positions,
        )
        alpha_abs_max = max(
            alpha_abs_max,
            abs(float(np.max(current))),
        )
        previous = current

    beta_next = np.zeros(
        (len(positions), len(rates)),
        dtype=dtype,
    )
    smoothed_mean = np.empty(steps, dtype=np.float64)
    beta_abs_max = 0.0
    for time_index in range(steps - 1, -1, -1):
        smoothed_mean[time_index] = position_mean_from_log(
            alpha[time_index] + beta_next,
            positions,
        )
        if time_index > 0:
            destination_value = (
                emissions[time_index, :, None] + beta_next
            ).astype(dtype)
            beta_next = transition_backward_log(
                destination_value,
                rate_log,
                position_log_weight,
                dtype,
                negative,
            )
            if mode.normalize_messages:
                beta_next = normalize_log_message(beta_next, dtype)
            beta_abs_max = max(
                beta_abs_max,
                abs(float(np.max(beta_next))),
            )

    truth_path = true_rate * np.arange(
        1,
        steps + 1,
        dtype=np.float64,
    )
    filtered_error = filtered_mean - truth_path
    smoothed_error = smoothed_mean - truth_path
    return {
        "precision_mode": mode.name,
        "steps": steps,
        "true_rate": true_rate,
        "emission_sigma_ft": emission_sigma_ft,
        "additive_emission_log_offset": (
            additive_emission_log_offset
        ),
        "filtered_rmse_ft": float(
            np.sqrt(np.mean(filtered_error**2))
        ),
        "smoothed_rmse_ft": float(
            np.sqrt(np.mean(smoothed_error**2))
        ),
        "filtered_end_error_ft": float(filtered_error[-1]),
        "smoothed_end_error_ft": float(smoothed_error[-1]),
        "alpha_abs_max": alpha_abs_max,
        "beta_abs_max": beta_abs_max,
        "filtered_mean": filtered_mean,
        "smoothed_mean": smoothed_mean,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_results: list[dict[str, Any]] = []
    for true_rate in (-0.05, 0.05):
        for additive_offset in (0.0, -3.5):
            for mode in PRECISION_MODES:
                raw_results.append(
                    run_scenario(
                        steps=args.steps,
                        true_rate=true_rate,
                        emission_sigma_ft=2.0,
                        additive_emission_log_offset=additive_offset,
                        mode=mode,
                    )
                )

    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    for result in raw_results:
        key = (
            float(result["true_rate"]),
            float(result["additive_emission_log_offset"]),
        )
        grouped.setdefault(key, {})[
            str(result["precision_mode"])
        ] = result
        for time_index, (filtered, smoothed) in enumerate(
            zip(
                result["filtered_mean"],
                result["smoothed_mean"],
                strict=True,
            )
        ):
            trajectory_rows.append(
                {
                    "precision_mode": result["precision_mode"],
                    "true_rate": result["true_rate"],
                    "additive_emission_log_offset": result[
                        "additive_emission_log_offset"
                    ],
                    "time_index": time_index,
                    "filtered_mean": filtered,
                    "smoothed_mean": smoothed,
                }
            )

    maximum_filtered_difference = 0.0
    maximum_smoothed_difference = 0.0
    for _key, by_mode in grouped.items():
        reference = by_mode["normalized_float64"]
        for _mode_name, result in by_mode.items():
            filtered_difference = np.asarray(
                result["filtered_mean"]
            ) - np.asarray(reference["filtered_mean"])
            smoothed_difference = np.asarray(
                result["smoothed_mean"]
            ) - np.asarray(reference["smoothed_mean"])
            filtered_max = float(
                np.max(np.abs(filtered_difference))
            )
            smoothed_max = float(
                np.max(np.abs(smoothed_difference))
            )
            maximum_filtered_difference = max(
                maximum_filtered_difference,
                filtered_max,
            )
            maximum_smoothed_difference = max(
                maximum_smoothed_difference,
                smoothed_max,
            )
            summary_rows.append(
                {
                    key_name: value
                    for key_name, value in result.items()
                    if key_name
                    not in {"filtered_mean", "smoothed_mean"}
                }
                | {
                    "filtered_mean_max_abs_difference_vs_float64_ft": (
                        filtered_max
                    ),
                    "smoothed_mean_max_abs_difference_vs_float64_ft": (
                        smoothed_max
                    ),
                }
            )

    sign_symmetry_max = 0.0
    summary_frame = pd.DataFrame(summary_rows)
    for additive_offset in (0.0, -3.5):
        for mode_name in summary_frame["precision_mode"].unique():
            positive = grouped[(0.05, additive_offset)][mode_name]
            negative = grouped[(-0.05, additive_offset)][mode_name]
            sign_symmetry_max = max(
                sign_symmetry_max,
                float(
                    np.max(
                        np.abs(
                            np.asarray(positive["smoothed_mean"])
                            + np.asarray(negative["smoothed_mean"])
                        )
                    )
                ),
            )

    summary = {
        "scope": {
            "steps": int(args.steps),
            "scenarios": int(len(grouped)),
            "precision_runs": int(len(raw_results)),
        },
        "maximum_filtered_mean_abs_difference_vs_normalized_float64_ft": (
            maximum_filtered_difference
        ),
        "maximum_smoothed_mean_abs_difference_vs_normalized_float64_ft": (
            maximum_smoothed_difference
        ),
        "maximum_smoothed_sign_symmetry_error_ft": sign_symmetry_max,
        "stress": {
            "additive_emission_log_offsets": [0.0, -3.5],
            "guard": (
                "A uniform additive log offset is posterior-invariant in "
                "exact arithmetic; -3.5 per row drives a 512-step "
                "unnormalized message to roughly the worst exp270 saved "
                "log-likelihood scale."
            ),
        },
        "interpretation_guard": (
            "This vectorized sparse log recurrence is a controlled numerical "
            "stress test, not an actual-well float64 redecode. A material "
            "difference would establish a precision mechanism; a negligible "
            "difference strongly rejects float32 scale handling as the root "
            "cause but cannot prove bitwise actual-well parity."
        ),
    }

    summary_frame.to_csv(output / "precision_summary.csv", index=False)
    pd.DataFrame(trajectory_rows).to_csv(
        output / "message_trajectories.csv.gz",
        index=False,
        compression="gzip",
    )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
