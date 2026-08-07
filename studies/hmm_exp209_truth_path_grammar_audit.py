#!/usr/bin/env python3
"""Audit whether exp209's fixed transition grammar can represent truth paths.

The saved exp270 artifact freezes exp209 posterior-mean, marginal-MAP, and
global-Viterbi paths before truth is used.  This diagnostic quantizes those
paths and the readout-only truth to the unchanged exp209 0.35-ft grid, then
checks every transition against:

- the five allowed position offsets around each destination-rate mean; and
- the adjacent-three-state rate transition support.

The audit distinguishes a locally impossible position jump from a sequence
that is locally feasible row by row but requires a rate-state jump larger than
one cell.  After an infeasible row, the dynamic support is restarted only for
descriptive counting; this is not an HMM decode or a prediction candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmm_exp209_offset_cause_readout import (
    DEFAULT_CANDIDATES,
    iter_well_frames,
    load_well_inputs,
)
from scipy.special import logsumexp
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_CAUSES = Path(
    "studies/hmm_exp209_gr_rigid_shift_barrier_20260726/"
    "episode_landscape.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_truth_path_grammar_audit_20260726"
)
STEP = 0.35
N_RATES = 41
SIG_R = 0.002
SIG_P = 0.02
MOMENTUM = 0.998
BAND_PAD = 100.0
PATH_COLUMNS = {
    "truth": "true_tvt_readout_only",
    "posterior_mean": "posterior_mean",
    "marginal_map": "marginal_map",
    "global_viterbi": "topk_path_1",
}
SOFT_PATHS = ("truth", "global_viterbi")
PRE_WINDOWS = (16, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--causes", type=Path, default=DEFAULT_CAUSES)
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


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def weighted_fraction(mask: pd.Series, weight: pd.Series) -> float | None:
    total = float(weight.sum())
    if total <= 0.0:
        return None
    return float(weight.loc[mask].sum() / total)


def quantize_path(
    path: np.ndarray,
    grid_min: float,
) -> np.ndarray:
    return np.floor((path - grid_min) / STEP + 0.5).astype(np.int64)


def grammar_geometry(
    rates: np.ndarray,
    dm: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the path-independent position kernel for rows after row 0."""
    mu = dm[1:, None] * rates[None, :] - dz[1:, None]
    base = np.floor(mu / STEP + 0.5).astype(np.int64)
    offsets = base[:, :, None] + np.arange(-2, 3, dtype=np.int64)
    delta = offsets * STEP - mu[:, :, None]
    sigma_position = max(SIG_P, 0.35 * STEP)
    log_kernel = -0.5 * (delta / sigma_position) ** 2
    log_kernel -= logsumexp(log_kernel, axis=2, keepdims=True)
    return base, log_kernel


def conditioned_rate_path_diagnostics(
    position_log_by_rate: np.ndarray,
    rates: np.ndarray,
    dm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score consecutive path shifts through the exact one-step rate kernel."""
    row_count, rate_count = position_log_by_rate.shape
    transition_nll = np.full(row_count, np.nan, dtype=np.float64)
    conditioned_rate_mean = np.full(row_count, np.nan, dtype=np.float64)
    conditioned_rate_edge_mass = np.full(
        row_count,
        np.nan,
        dtype=np.float64,
    )
    if row_count <= 2:
        return (
            transition_nll,
            conditioned_rate_mean,
            conditioned_rate_edge_mass,
        )
    rate_step = rates[1] - rates[0]
    previous_log = position_log_by_rate[1:-1]
    previous_norm = logsumexp(
        previous_log,
        axis=1,
        keepdims=True,
    )
    previous_valid = np.isfinite(previous_norm[:, 0])
    previous_weight = np.zeros_like(previous_log)
    previous_weight[previous_valid] = np.exp(
        previous_log[previous_valid]
        - previous_norm[previous_valid]
    )

    row_dm = dm[2:, None]
    rate_var_cells = (
        SIG_R * np.sqrt(row_dm) / rate_step
    ) ** 2
    mean_move = (
        -(1.0 - MOMENTUM)
        * rates[None, :]
        * row_dm
        / rate_step
    )
    p_plus = np.maximum(
        0.5 * (rate_var_cells + mean_move),
        1e-12,
    )
    p_minus = np.maximum(
        0.5 * (rate_var_cells - mean_move),
        1e-12,
    )
    total = p_plus + p_minus
    scale = np.ones_like(total)
    capped = total > 0.9
    scale[capped] = 0.9 / total[capped]
    p_plus *= scale
    p_minus *= scale
    p_stay = 1.0 - p_plus - p_minus

    predictive = previous_weight * p_stay
    predictive[:, 1:] += previous_weight[:, :-1] * p_plus[:, :-1]
    predictive[:, :-1] += previous_weight[:, 1:] * p_minus[:, 1:]
    likelihood = np.exp(position_log_by_rate[2:])
    joint = predictive * likelihood
    probability = np.sum(joint, axis=1)
    valid = previous_valid & np.isfinite(probability) & (probability > 0.0)
    transition_target = transition_nll[2:]
    transition_target[valid] = -np.log(probability[valid])
    posterior = np.zeros_like(joint)
    posterior[valid] = joint[valid] / probability[valid, None]
    rate_mean_target = conditioned_rate_mean[2:]
    rate_mean_target[valid] = posterior[valid] @ rates
    edge_mass_target = conditioned_rate_edge_mass[2:]
    edge_mass_target[valid] = (
        posterior[valid, 0] + posterior[valid, -1]
    )
    return (
        transition_nll,
        conditioned_rate_mean,
        conditioned_rate_edge_mass,
    )


def grammar_arrays(
    path: np.ndarray,
    grid_min: float,
    grid_count: int,
    base: np.ndarray,
    log_kernel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return row-aligned hard-support and local position-cost diagnostics."""
    row_count = len(path)
    if len(base) != max(0, row_count - 1):
        raise ValueError("path and geometry lengths differ")
    q = quantize_path(path, grid_min)
    in_grid = (q >= 0) & (q < grid_count)
    local_illegal = np.zeros(row_count, dtype=bool)
    dynamic_break = np.zeros(row_count, dtype=bool)
    best_position_nll = np.full(row_count, np.nan, dtype=np.float64)
    support_rate_count = np.zeros(row_count, dtype=np.int16)
    excess_shift_cells = np.full(row_count, np.nan, dtype=np.float64)
    position_log_by_rate = np.full(
        (row_count, log_kernel.shape[1]),
        -np.inf,
        dtype=np.float64,
    )
    if row_count <= 1:
        return {
            "grid_index": q,
            "in_grid": in_grid,
            "local_illegal": local_illegal,
            "dynamic_break": dynamic_break,
            "best_position_nll": best_position_nll,
            "support_rate_count": support_rate_count,
            "excess_shift_cells": excess_shift_cells,
            "position_log_by_rate": position_log_by_rate,
        }

    shift = np.diff(q)
    allowed = np.abs(shift[:, None] - base) <= 2
    endpoint_valid = in_grid[1:] & in_grid[:-1]
    allowed &= endpoint_valid[:, None]
    local_illegal[1:] = ~np.any(allowed, axis=1)
    support_rate_count[1:] = np.sum(allowed, axis=1).astype(np.int16)

    required_code = shift[:, None] - base + 2
    clipped_code = np.clip(required_code, 0, 4)
    selected = np.take_along_axis(
        log_kernel,
        clipped_code[:, :, None],
        axis=2,
    )[:, :, 0]
    selected[~allowed] = -np.inf
    position_log_by_rate[1:] = selected
    best = np.max(selected, axis=1)
    feasible = np.isfinite(best)
    best_position_target = best_position_nll[1:]
    best_position_target[feasible] = -best[feasible]

    minimum_shift = np.min(base - 2, axis=1)
    maximum_shift = np.max(base + 2, axis=1)
    excess = np.zeros(row_count - 1, dtype=np.float64)
    below = shift < minimum_shift
    above = shift > maximum_shift
    excess[below] = shift[below] - minimum_shift[below]
    excess[above] = shift[above] - maximum_shift[above]
    excess_shift_cells[1:] = excess

    previous_support: np.ndarray | None = None
    for index, current_support in enumerate(allowed, start=1):
        if not current_support.any():
            dynamic_break[index] = True
            previous_support = None
            continue
        if previous_support is None:
            previous_support = current_support.copy()
            continue
        reachable = previous_support.copy()
        reachable[:-1] |= previous_support[1:]
        reachable[1:] |= previous_support[:-1]
        feasible_support = current_support & reachable
        if not feasible_support.any():
            dynamic_break[index] = True
            previous_support = current_support.copy()
        else:
            previous_support = feasible_support

    return {
        "grid_index": q,
        "in_grid": in_grid,
        "local_illegal": local_illegal,
        "dynamic_break": dynamic_break,
        "best_position_nll": best_position_nll,
        "support_rate_count": support_rate_count,
        "excess_shift_cells": excess_shift_cells,
        "position_log_by_rate": position_log_by_rate,
    }


def interval_metrics(
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    prefix: str,
) -> dict[str, Any]:
    rows = max(0, end - start)
    if rows == 0:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_out_of_grid_fraction": None,
            f"{prefix}_local_illegal_fraction": None,
            f"{prefix}_dynamic_break_fraction": None,
            f"{prefix}_best_position_nll_median": None,
            f"{prefix}_best_position_nll_p90": None,
            f"{prefix}_support_rate_count_median": None,
            f"{prefix}_conditioned_transition_nll_mean": None,
            f"{prefix}_conditioned_transition_nll_median": None,
            f"{prefix}_conditioned_transition_nll_p90": None,
            f"{prefix}_conditioned_rate_edge_mass_mean": None,
        }
    in_grid = arrays["in_grid"][start:end]
    illegal = arrays["local_illegal"][start:end]
    dynamic = arrays["dynamic_break"][start:end]
    position_nll = arrays["best_position_nll"][start:end]
    support_count = arrays["support_rate_count"][start:end]
    transition_nll = arrays["conditioned_transition_nll"][start:end]
    rate_edge_mass = arrays["conditioned_rate_edge_mass"][start:end]
    finite_nll = position_nll[np.isfinite(position_nll)]
    valid_support = support_count[support_count > 0]
    finite_transition = transition_nll[np.isfinite(transition_nll)]
    finite_edge_mass = rate_edge_mass[np.isfinite(rate_edge_mass)]
    return {
        f"{prefix}_rows": int(rows),
        f"{prefix}_out_of_grid_fraction": float((~in_grid).mean()),
        f"{prefix}_local_illegal_fraction": float(illegal.mean()),
        f"{prefix}_dynamic_break_fraction": float(dynamic.mean()),
        f"{prefix}_best_position_nll_median": (
            float(np.median(finite_nll)) if len(finite_nll) else None
        ),
        f"{prefix}_best_position_nll_p90": (
            float(np.quantile(finite_nll, 0.9))
            if len(finite_nll)
            else None
        ),
        f"{prefix}_support_rate_count_median": (
            float(np.median(valid_support)) if len(valid_support) else None
        ),
        f"{prefix}_conditioned_transition_nll_mean": (
            float(np.mean(finite_transition))
            if len(finite_transition)
            else None
        ),
        f"{prefix}_conditioned_transition_nll_median": (
            float(np.median(finite_transition))
            if len(finite_transition)
            else None
        ),
        f"{prefix}_conditioned_transition_nll_p90": (
            float(np.quantile(finite_transition, 0.9))
            if len(finite_transition)
            else None
        ),
        f"{prefix}_conditioned_rate_edge_mass_mean": (
            float(np.mean(finite_edge_mass))
            if len(finite_edge_mass)
            else None
        ),
    }


def summarize_episode_group(
    group: pd.DataFrame,
    total_sse: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
    }
    for path_name in PATH_COLUMNS:
        local_any = group[f"{path_name}_episode_local_illegal_fraction"] > 0.0
        dynamic_any = (
            group[f"{path_name}_episode_dynamic_break_fraction"] > 0.0
        )
        result.update(
            {
                f"{path_name}_episode_any_local_illegal_fraction": float(
                    local_any.mean()
                ),
                (
                    f"{path_name}_episode_any_local_illegal_"
                    "sse_fraction"
                ): weighted_fraction(local_any, group["episode_sse"]),
                f"{path_name}_episode_any_dynamic_break_fraction": float(
                    dynamic_any.mean()
                ),
                (
                    f"{path_name}_episode_any_dynamic_break_"
                    "sse_fraction"
                ): weighted_fraction(dynamic_any, group["episode_sse"]),
                f"{path_name}_episode_local_illegal_row_fraction": float(
                    group[
                        f"{path_name}_episode_local_illegal_fraction"
                    ].mean()
                ),
                f"{path_name}_episode_dynamic_break_row_fraction": float(
                    group[
                        f"{path_name}_episode_dynamic_break_fraction"
                    ].mean()
                ),
                f"{path_name}_episode_best_position_nll_median": float(
                    group[
                        f"{path_name}_episode_best_position_nll_median"
                    ].median()
                ),
            }
        )
        if path_name in SOFT_PATHS:
            result.update(
                {
                    (
                    f"{path_name}_episode_conditioned_transition_"
                    "nll_mean"
                    ): float(
                        group[
                            f"{path_name}_episode_conditioned_"
                            "transition_nll_mean"
                        ].mean()
                    ),
                    (
                    f"{path_name}_episode_conditioned_transition_"
                    "nll_median"
                    ): float(
                        group[
                            f"{path_name}_episode_conditioned_"
                            "transition_nll_mean"
                        ].median()
                    ),
                    (
                    f"{path_name}_episode_conditioned_rate_"
                    "edge_mass_mean"
                    ): float(
                        group[
                            f"{path_name}_episode_conditioned_"
                            "rate_edge_mass_mean"
                        ].mean()
                    ),
                }
            )
    for comparator in ("global_viterbi",):
        difference = (
            group["truth_episode_conditioned_transition_nll_mean"]
            - group[
                f"{comparator}_episode_conditioned_transition_nll_mean"
            ]
        )
        truth_penalized = difference > 0.0
        result.update(
            {
                (
                    f"truth_minus_{comparator}_conditioned_"
                    "transition_nll_mean"
                ): float(difference.mean()),
                (
                    f"truth_minus_{comparator}_conditioned_"
                    "transition_nll_median"
                ): float(difference.median()),
                (
                    f"truth_more_penalized_than_{comparator}_fraction"
                ): float(truth_penalized.mean()),
                (
                    f"truth_more_penalized_than_{comparator}_"
                    "sse_fraction"
                ): weighted_fraction(
                    truth_penalized,
                    group["episode_sse"],
                ),
                (
                    f"truth_minus_{comparator}_transition_nll_"
                    "vs_rmse_spearman"
                ): safe_spearman(difference, group["rmse_ft"]),
            }
        )
    for window in PRE_WINDOWS:
        column = f"truth_pre{window}_any_dynamic_break"
        result[f"{column}_fraction"] = float(group[column].mean())
        result[f"{column}_sse_fraction"] = weighted_fraction(
            group[column],
            group["episode_sse"],
        )
        pre_nll_column = (
            f"truth_pre{window}_conditioned_transition_nll_mean"
        )
        delta_column = (
            f"truth_episode_minus_pre{window}_conditioned_"
            "transition_nll"
        )
        result[f"{pre_nll_column}_mean"] = float(
            group[pre_nll_column].mean()
        )
        result[f"{pre_nll_column}_median"] = float(
            group[pre_nll_column].median()
        )
        result[f"{delta_column}_mean"] = float(
            group[delta_column].mean()
        )
        result[f"{delta_column}_median"] = float(
            group[delta_column].median()
        )
        result[f"{pre_nll_column}_vs_rmse_spearman"] = safe_spearman(
            group[pre_nll_column],
            group["rmse_ft"],
        )
    recent = group["truth_rows_since_last_dynamic_break"].notna()
    result["truth_prior_dynamic_break_observed_fraction"] = float(
        recent.mean()
    )
    for window in PRE_WINDOWS:
        mask = recent & (
            group["truth_rows_since_last_dynamic_break"] <= window
        )
        result[
            f"truth_last_dynamic_break_within_{window}_fraction"
        ] = float(mask.mean())
        result[
            f"truth_last_dynamic_break_within_{window}_sse_fraction"
        ] = weighted_fraction(mask, group["episode_sse"])
    result["truth_dynamic_break_sign_matches_offset_fraction"] = float(
        group["truth_dynamic_break_sign_matches_offset"].mean()
    )
    result["truth_dynamic_break_sign_matches_offset_sse_fraction"] = (
        weighted_fraction(
            group["truth_dynamic_break_sign_matches_offset"],
            group["episode_sse"],
        )
    )
    result["truth_episode_illegal_vs_rmse_spearman"] = safe_spearman(
        group["truth_episode_local_illegal_fraction"],
        group["rmse_ft"],
    )
    result["truth_episode_position_nll_vs_rmse_spearman"] = (
        safe_spearman(
            group["truth_episode_best_position_nll_median"],
            group["rmse_ft"],
        )
    )
    result["truth_episode_transition_nll_vs_rmse_spearman"] = (
        safe_spearman(
            group["truth_episode_conditioned_transition_nll_mean"],
            group["rmse_ft"],
        )
    )
    result["truth_episode_transition_nll_vs_viterbi_gain_spearman"] = (
        safe_spearman(
            group["truth_episode_conditioned_transition_nll_mean"],
            group["viterbi_rmse_gain_ft"],
        )
    )
    result["truth_episode_illegal_vs_viterbi_gain_spearman"] = (
        safe_spearman(
            group["truth_episode_local_illegal_fraction"],
            group["viterbi_rmse_gain_ft"],
        )
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    candidates_path = resolve(root, args.candidates)
    episodes_path = resolve(root, args.episodes)
    causes_path = resolve(root, args.causes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    causes = pd.read_csv(
        causes_path,
        usecols=["episode_id", "well", "cause_bucket"],
    )
    episodes["well"] = episodes["well"].astype(str)
    causes["well"] = causes["well"].astype(str)
    episode_key = ["episode_id", "well"]
    episodes = episodes.merge(
        causes,
        on=episode_key,
        validate="one_to_one",
    )
    episode_lookup = {
        str(well): group.sort_values("start_row_idx").to_dict("records")
        for well, group in episodes.groupby("well", sort=False)
    }

    row_totals = {
        path_name: {
            scope: {
                "rows": 0,
                "out_of_grid": 0,
                "local_illegal": 0,
                "dynamic_break": 0,
                "conditioned_transition_nll_sum": 0.0,
                "conditioned_transition_nll_rows": 0,
                "conditioned_rate_edge_mass_sum": 0.0,
                "conditioned_rate_edge_mass_rows": 0,
            }
            for scope in ("persistent", "nonpersistent")
        }
        for path_name in PATH_COLUMNS
    }
    well_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    train_dir = root / "data/raw/train"
    processed_wells = 0
    for well, frame in iter_well_frames(
        candidates_path,
        int(args.chunksize),
    ):
        (
            horizontal,
            typewell_tvt,
            _,
            _,
            _,
            rate_span,
            _,
            _,
        ) = load_well_inputs(train_dir, well)
        known = horizontal.loc[horizontal["TVT_input"].notna()]
        eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
        if len(known) == 0 or len(eval_rows) != len(frame):
            raise ValueError(f"{well}: raw/candidate suffix mismatch")
        last = known.iloc[-1]
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        if not np.array_equal(
            row_index,
            eval_rows.index.to_numpy(np.int64),
        ):
            raise ValueError(f"{well}: row index mismatch")
        grid_min = max(
            float(typewell_tvt.min()) - 40.0,
            float(last["TVT_input"]) - BAND_PAD,
        )
        grid_max = min(
            float(typewell_tvt.max()) + 40.0,
            float(last["TVT_input"]) + BAND_PAD,
        )
        grid = np.arange(
            grid_min,
            grid_max + STEP,
            STEP,
            dtype=np.float64,
        )
        rates = np.linspace(
            -float(rate_span),
            float(rate_span),
            N_RATES,
            dtype=np.float64,
        )
        md = pd.to_numeric(
            eval_rows["MD"],
            errors="raise",
        ).to_numpy(np.float64)
        z = pd.to_numeric(
            eval_rows["Z"],
            errors="raise",
        ).to_numpy(np.float64)
        dm = np.maximum(
            np.diff(np.concatenate([[float(last["MD"])], md])),
            1.0,
        )
        dz = np.diff(np.concatenate([[float(last["Z"])], z]))
        base, log_kernel = grammar_geometry(rates, dm, dz)
        specs = episode_lookup.get(well, [])
        arrays_by_path = {
            path_name: grammar_arrays(
                pd.to_numeric(
                    frame[column],
                    errors="raise",
                ).to_numpy(np.float64),
                grid_min,
                len(grid),
                base,
                log_kernel,
            )
            for path_name, column in PATH_COLUMNS.items()
        }
        for arrays in arrays_by_path.values():
            arrays["conditioned_transition_nll"] = np.full(
                len(frame),
                np.nan,
                dtype=np.float64,
            )
            arrays["conditioned_rate_mean"] = np.full(
                len(frame),
                np.nan,
                dtype=np.float64,
            )
            arrays["conditioned_rate_edge_mass"] = np.full(
                len(frame),
                np.nan,
                dtype=np.float64,
            )
        for path_name in SOFT_PATHS:
            arrays = arrays_by_path[path_name]
            (
                transition_nll,
                conditioned_rate_mean,
                conditioned_rate_edge_mass,
            ) = conditioned_rate_path_diagnostics(
                arrays["position_log_by_rate"],
                rates,
                dm,
            )
            arrays["conditioned_transition_nll"] = transition_nll
            arrays["conditioned_rate_mean"] = conditioned_rate_mean
            arrays[
                "conditioned_rate_edge_mass"
            ] = conditioned_rate_edge_mass
        for arrays in arrays_by_path.values():
            del arrays["position_log_by_rate"]

        persistent_mask = np.zeros(len(frame), dtype=bool)
        for spec in specs:
            start = int(np.searchsorted(row_index, int(spec["start_row_idx"])))
            end = int(
                np.searchsorted(
                    row_index,
                    int(spec["end_row_idx_exclusive"]),
                )
            )
            if end - start != int(spec["rows"]):
                raise ValueError(f"{spec['episode_id']}: row count mismatch")
            persistent_mask[start:end] = True
            record: dict[str, Any] = {
                **spec,
                "episode_start_suffix_index": start,
                "episode_end_suffix_index_exclusive": end,
            }
            for path_name, arrays in arrays_by_path.items():
                record.update(
                    interval_metrics(
                        arrays,
                        start,
                        end,
                        f"{path_name}_episode",
                    )
                )
            truth_arrays = arrays_by_path["truth"]
            for window in PRE_WINDOWS:
                pre_start = max(0, start - window)
                pre_transition = truth_arrays[
                    "conditioned_transition_nll"
                ][pre_start:start]
                pre_transition = pre_transition[
                    np.isfinite(pre_transition)
                ]
                record[f"truth_pre{window}_rows"] = int(
                    start - pre_start
                )
                record[f"truth_pre{window}_any_local_illegal"] = bool(
                    truth_arrays["local_illegal"][pre_start:start].any()
                )
                record[f"truth_pre{window}_any_dynamic_break"] = bool(
                    truth_arrays["dynamic_break"][pre_start:start].any()
                )
                record[
                    f"truth_pre{window}_conditioned_transition_nll_mean"
                ] = (
                    float(np.mean(pre_transition))
                    if len(pre_transition)
                    else None
                )
                episode_transition = record[
                    "truth_episode_conditioned_transition_nll_mean"
                ]
                record[
                    f"truth_episode_minus_pre{window}_conditioned_"
                    "transition_nll"
                ] = (
                    float(episode_transition)
                    - float(np.mean(pre_transition))
                    if len(pre_transition)
                    and episode_transition is not None
                    else None
                )
            prior_breaks = np.flatnonzero(
                truth_arrays["dynamic_break"][:start]
            )
            if len(prior_breaks):
                last_break = int(prior_breaks[-1])
                excess = float(
                    truth_arrays["excess_shift_cells"][last_break]
                )
                expected_error_sign = -float(np.sign(excess))
                actual_error_sign = float(
                    np.sign(float(spec["mean_error_ft"]))
                )
                record["truth_rows_since_last_dynamic_break"] = int(
                    start - last_break
                )
                record["truth_last_dynamic_break_excess_cells"] = excess
                record[
                    "truth_dynamic_break_sign_matches_offset"
                ] = bool(
                    expected_error_sign != 0.0
                    and expected_error_sign == actual_error_sign
                )
            else:
                record["truth_rows_since_last_dynamic_break"] = None
                record["truth_last_dynamic_break_excess_cells"] = None
                record[
                    "truth_dynamic_break_sign_matches_offset"
                ] = False
            episode_rows.append(record)

        well_record: dict[str, Any] = {
            "well": well,
            "rows": int(len(frame)),
            "persistent_rows": int(persistent_mask.sum()),
            "episodes": int(len(specs)),
            "grid_count": int(len(grid)),
            "rate_span": float(rate_span),
        }
        for path_name, arrays in arrays_by_path.items():
            well_record.update(
                interval_metrics(
                    arrays,
                    0,
                    len(frame),
                    path_name,
                )
            )
            for scope, mask in (
                ("persistent", persistent_mask),
                ("nonpersistent", ~persistent_mask),
            ):
                totals = row_totals[path_name][scope]
                totals["rows"] += int(mask.sum())
                totals["out_of_grid"] += int(
                    ((~arrays["in_grid"]) & mask).sum()
                )
                totals["local_illegal"] += int(
                    (arrays["local_illegal"] & mask).sum()
                )
                totals["dynamic_break"] += int(
                    (arrays["dynamic_break"] & mask).sum()
                )
                transition_values = arrays[
                    "conditioned_transition_nll"
                ][mask]
                transition_values = transition_values[
                    np.isfinite(transition_values)
                ]
                totals["conditioned_transition_nll_sum"] += float(
                    transition_values.sum()
                )
                totals["conditioned_transition_nll_rows"] += int(
                    len(transition_values)
                )
                edge_values = arrays[
                    "conditioned_rate_edge_mass"
                ][mask]
                edge_values = edge_values[np.isfinite(edge_values)]
                totals["conditioned_rate_edge_mass_sum"] += float(
                    edge_values.sum()
                )
                totals["conditioned_rate_edge_mass_rows"] += int(
                    len(edge_values)
                )
        well_rows.append(well_record)
        processed_wells += 1
        if processed_wells % 100 == 0:
            print(f"processed_wells={processed_wells}", flush=True)

    episode_frame = pd.DataFrame(episode_rows)
    if len(episode_frame) != len(episodes):
        raise ValueError(
            f"episode output {len(episode_frame)} != {len(episodes)}"
        )
    if episode_frame.duplicated(episode_key).any():
        raise ValueError("duplicate episode output keys")
    well_frame = pd.DataFrame(well_rows)
    episode_frame.to_csv(
        output / "episode_grammar_metrics.csv",
        index=False,
    )
    well_frame.to_csv(
        output / "well_grammar_metrics.csv",
        index=False,
    )

    total_sse = float(episode_frame["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in episode_frame.groupby(
        "cause_bucket",
        sort=True,
    ):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_episode_group(group, total_sse),
            }
        )
    cause_frame = pd.DataFrame(cause_rows)
    cause_frame.to_csv(
        output / "cause_bucket_grammar_summary.csv",
        index=False,
    )

    row_summary: dict[str, Any] = {}
    for path_name, scopes in row_totals.items():
        row_summary[path_name] = {}
        for scope, totals in scopes.items():
            rows = int(totals["rows"])
            row_summary[path_name][scope] = {
                **totals,
                "out_of_grid_fraction": (
                    float(totals["out_of_grid"] / rows) if rows else None
                ),
                "local_illegal_fraction": (
                    float(totals["local_illegal"] / rows) if rows else None
                ),
                "dynamic_break_fraction": (
                    float(totals["dynamic_break"] / rows) if rows else None
                ),
                "conditioned_transition_nll_mean": (
                    float(
                        totals["conditioned_transition_nll_sum"]
                        / totals["conditioned_transition_nll_rows"]
                    )
                    if totals["conditioned_transition_nll_rows"]
                    else None
                ),
                "conditioned_rate_edge_mass_mean": (
                    float(
                        totals["conditioned_rate_edge_mass_sum"]
                        / totals["conditioned_rate_edge_mass_rows"]
                    )
                    if totals["conditioned_rate_edge_mass_rows"]
                    else None
                ),
            }
    overall_episode = summarize_episode_group(
        episode_frame,
        total_sse,
    )
    summary = {
        "scope": {
            "wells": int(len(well_frame)),
            "rows": int(well_frame["rows"].sum()),
            "episodes": int(len(episode_frame)),
            "episode_wells": int(episode_frame["well"].nunique()),
            "episode_rows": int(episode_frame["rows"].sum()),
            "grid_step_ft": STEP,
            "rate_states": N_RATES,
            "position_offsets": [-2, -1, 0, 1, 2],
            "rate_predecessor_offsets": [-1, 0, 1],
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(
                candidates_path
            ),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
            str(causes_path.relative_to(root)): sha256(causes_path),
        },
        "row_summary": row_summary,
        "overall_episode_summary": overall_episode,
        "cause_bucket_summary": cause_rows,
        "guards": {
            "truth_usage": (
                "Saved exp270 candidates are frozen before readout-only "
                "truth is quantized for this diagnostic."
            ),
            "interpretation": (
                "Hard-support checks are not posterior probabilities. "
                "Dynamic support is restarted after each infeasibility only "
                "to count later independent breaks. Conditioned transition "
                "NLL uses the previous row's local position-conditioned rate "
                "distribution, propagates it through the exact one-step rate "
                "kernel, and scores the next fixed truth or global-Viterbi "
                "position shift. It is a two-step local diagnostic, not a "
                "long-history filtered rate posterior, and excludes all GR "
                "emissions."
            ),
            "prediction_generation": False,
            "hmm_rerun": False,
            "model_or_booster": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
