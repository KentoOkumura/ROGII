#!/usr/bin/env python3
"""Audit position-grid bias versus transition-variance trade-offs.

The exp209 process requests a narrow continuous position transition but
represents it on a 0.35 ft grid.  This deterministic study measures the first
two moments of the sampled five-cell Gaussian and the minimum variance required
by any grid-supported distribution that exactly matches a sub-grid mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_position_resolution_tradeoff_20260726"
)
CURRENT_GRID_STEP_FT = 0.35
CONFIGURED_SIGMA_FT = 0.02
CURRENT_EFFECTIVE_SIGMA_FT = 0.1225
PHASE_MINIMAX_SIGMA_FT = 0.2325
PHASE_POINTS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def gaussian_moments(
    grid_step_ft: float,
    sigma_ft: float,
) -> dict[str, float]:
    phase = (
        np.arange(PHASE_POINTS, dtype=np.float64) + 0.5
    ) / PHASE_POINTS * grid_step_ft - 0.5 * grid_step_ft
    positions = np.arange(-2, 3, dtype=np.float64) * grid_step_ft
    weights = np.exp(
        -0.5
        * (
            (
                positions[:, None]
                - phase[None, :]
            )
            / sigma_ft
        )
        ** 2
    )
    weights /= weights.sum(axis=0)
    mean = np.sum(weights * positions[:, None], axis=0)
    bias = mean - phase
    variance = np.sum(
        weights * (positions[:, None] - mean[None, :]) ** 2,
        axis=0,
    )
    return {
        "grid_step_ft": grid_step_ft,
        "sigma_ft": sigma_ft,
        "sigma_in_grid_cells": sigma_ft / grid_step_ft,
        "phase_abs_bias_mean_ft": float(np.mean(np.abs(bias))),
        "phase_abs_bias_max_ft": float(np.max(np.abs(bias))),
        "phase_variance_mean_ft2": float(np.mean(variance)),
        "phase_variance_max_ft2": float(np.max(variance)),
        "phase_sd_mean_ft": float(np.mean(np.sqrt(variance))),
    }


def main() -> None:
    args = parse_args()
    output = args.root.resolve() / args.output
    output.mkdir(parents=True, exist_ok=True)

    kernel_rows = []
    for label, sigma_ft in (
        ("configured_without_floor", CONFIGURED_SIGMA_FT),
        ("current_effective_floor", CURRENT_EFFECTIVE_SIGMA_FT),
        ("phase_minimax_bias", PHASE_MINIMAX_SIGMA_FT),
    ):
        row = gaussian_moments(CURRENT_GRID_STEP_FT, sigma_ft)
        row["kernel"] = label
        kernel_rows.append(row)

    exact_mean_phase_variance = CURRENT_GRID_STEP_FT**2 / 6.0
    exact_mean_max_variance = CURRENT_GRID_STEP_FT**2 / 4.0
    kernel_rows.append(
        {
            "kernel": "minimum_variance_exact_mean_grid_transport",
            "grid_step_ft": CURRENT_GRID_STEP_FT,
            "sigma_ft": np.nan,
            "sigma_in_grid_cells": np.nan,
            "phase_abs_bias_mean_ft": 0.0,
            "phase_abs_bias_max_ft": 0.0,
            "phase_variance_mean_ft2": exact_mean_phase_variance,
            "phase_variance_max_ft2": exact_mean_max_variance,
            "phase_sd_mean_ft": float(
                np.pi * CURRENT_GRID_STEP_FT / 8.0
            ),
        }
    )
    kernel_metrics = pd.DataFrame(kernel_rows)

    refinement_rows = []
    for grid_step_ft in (
        0.35,
        0.175,
        0.0875,
        0.04375,
        0.035,
    ):
        effective_sigma_ft = max(
            CONFIGURED_SIGMA_FT,
            0.35 * grid_step_ft,
        )
        row = gaussian_moments(
            grid_step_ft,
            effective_sigma_ft,
        )
        row["state_count_multiplier_vs_current"] = (
            CURRENT_GRID_STEP_FT / grid_step_ft
        )
        row["effective_sigma_floor_ft"] = effective_sigma_ft
        refinement_rows.append(row)
    refinement = pd.DataFrame(refinement_rows)

    configured_variance = CONFIGURED_SIGMA_FT**2
    current = kernel_metrics[
        kernel_metrics["kernel"] == "current_effective_floor"
    ].iloc[0]
    corrected = kernel_metrics[
        kernel_metrics["kernel"] == "phase_minimax_bias"
    ].iloc[0]
    summary = {
        "current_grid_step_ft": CURRENT_GRID_STEP_FT,
        "configured_sigma_ft": CONFIGURED_SIGMA_FT,
        "configured_variance_ft2": configured_variance,
        "current_effective_sigma_ft": CURRENT_EFFECTIVE_SIGMA_FT,
        "phase_minimax_sigma_ft": PHASE_MINIMAX_SIGMA_FT,
        "phase_minimax_variance_increase_vs_current": float(
            corrected["phase_variance_mean_ft2"]
            / current["phase_variance_mean_ft2"]
        ),
        "exact_mean_grid_transport": {
            "minimum_phase_mean_variance_ft2": (
                exact_mean_phase_variance
            ),
            "minimum_maximum_variance_ft2": exact_mean_max_variance,
            "phase_mean_variance_ratio_vs_configured": (
                exact_mean_phase_variance / configured_variance
            ),
            "maximum_variance_ratio_vs_configured": (
                exact_mean_max_variance / configured_variance
            ),
        },
        "interpretation_guard": (
            "Deterministic moment audit only. A finer grid multiplies "
            "state count and runtime; a wider sigma changes process variance. "
            "Neither is selected here as an OOF production parameter."
        ),
    }

    kernel_metrics.to_csv(output / "kernel_tradeoff.csv", index=False)
    refinement.to_csv(output / "grid_refinement.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(kernel_metrics.to_string(index=False))
    print(refinement.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
