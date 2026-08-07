#!/usr/bin/env python3
"""Decompose exp209 persistent offsets into emission and dynamics evidence.

The saved exp270 artifact contains exact-exp209 posterior mean, marginal MAP,
global Viterbi, posterior spread, and truth for readout only.  This script
reconstructs exp209's raw-GR Gaussian emission without changing the decoder,
then compares emission NLL at the true TVT and at each decoded TVT.

Interpretation:

- positive ``truth_minus_candidate_nll`` means raw GR locally favors the
  decoded candidate over the true TVT (GR alias / misleading observation);
- negative values mean raw GR locally favors truth while the HMM remains
  elsewhere (transition, prior, smoothing, or state-support pressure);
- decoder error differences separate posterior averaging from a shared
  wrong-state problem.

The study is diagnostic only.  It does not create or select a prediction.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_OUTPUT = Path("studies/hmm_exp209_offset_cause_readout_20260725")
USECOLS = [
    "well",
    "row_idx",
    "true_tvt_readout_only",
    "last_known_tvt",
    "md_since",
    "posterior_std",
    "marginal_mode_mass",
    "marginal_mode_gap",
    "posterior_mean",
    "marginal_map",
    "topk_path_1",
]
ERROR_BUCKETS = (
    ("abs_error_000_005", 0.0, 5.0),
    ("abs_error_005_010", 5.0, 10.0),
    ("abs_error_010_025", 10.0, 25.0),
    ("abs_error_025_plus", 25.0, np.inf),
)
PERSISTENT_ERROR_FT = 10.0
PERSISTENT_MIN_ROWS = 128
RATE_SPAN_FLOOR = 0.10
STRONG_SEQUENCE_NLL = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def iter_well_frames(path: Path, chunksize: int) -> Iterator[tuple[str, pd.DataFrame]]:
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
                raise ValueError("candidate well order is not strictly increasing")
            previous_well = well
            yield well, group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        well = str(carry["well"].iloc[0])
        if previous_well is not None and well <= previous_well:
            raise ValueError("candidate final well order is invalid")
        yield well, carry.reset_index(drop=True)


def load_well_inputs(
    train_dir: Path,
    well: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, float, float, float]:
    horizontal = pd.read_csv(train_dir / f"{well}__horizontal_well.csv")
    typewell = (
        pd.read_csv(train_dir / f"{well}__typewell.csv")
        .sort_values("TVT")
        .reset_index(drop=True)
    )
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce")
        .ffill()
        .bfill()
        .to_numpy(np.float64)
    )
    known = horizontal[horizontal["TVT_input"].notna()]
    typewell_at_known = np.interp(
        pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64),
        typewell_tvt,
        typewell_gr,
    )
    known_gr_zero = (
        pd.to_numeric(known["GR"], errors="coerce").fillna(0).to_numpy(np.float64)
    )
    sigma = float(np.clip(np.nanstd(known_gr_zero - typewell_at_known), 10.0, 60.0))
    known_gr_finite = pd.to_numeric(known["GR"], errors="coerce").to_numpy(
        np.float64
    )
    calibration_valid = np.isfinite(known_gr_finite) & np.isfinite(
        typewell_at_known
    )
    if (
        int(calibration_valid.sum()) >= 20
        and float(np.std(typewell_at_known[calibration_valid])) > 1e-6
    ):
        cal_a, cal_b = np.polyfit(
            typewell_at_known[calibration_valid],
            known_gr_finite[calibration_valid],
            1,
        )
    elif calibration_valid.any():
        cal_a = 1.0
        cal_b = float(
            np.mean(known_gr_finite[calibration_valid])
            - np.mean(typewell_at_known[calibration_valid])
        )
    else:
        cal_a, cal_b = 1.0, 0.0

    tail = known.tail(30)
    dmd = np.diff(pd.to_numeric(tail["MD"], errors="raise").to_numpy(np.float64))
    rate = (
        np.diff(
            pd.to_numeric(tail["TVT_input"], errors="raise").to_numpy(np.float64)
        )
        + np.diff(pd.to_numeric(tail["Z"], errors="raise").to_numpy(np.float64))
    ) / dmd
    valid = np.isfinite(rate) & (dmd > 0)
    init_rate = float(np.median(rate[valid])) if int(valid.sum()) >= 3 else 0.0
    rate_span = max(RATE_SPAN_FLOOR, abs(init_rate) + 0.04)
    return (
        horizontal,
        typewell_tvt,
        typewell_gr,
        sigma,
        init_rate,
        rate_span,
        float(cal_a),
        float(cal_b),
    )


def emission_nll(
    observed_gr: np.ndarray,
    tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma: float,
) -> np.ndarray:
    expected_gr = np.interp(tvt, typewell_tvt, typewell_gr)
    z2 = ((observed_gr - expected_gr) / sigma) ** 2
    return 0.5 * np.minimum(z2, 600.0)


def persistent_runs(mask: np.ndarray, minimum_rows: int) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if int(end - start) >= minimum_rows
    ]


def linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x -= x.mean()
    centered = values - values.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, centered) / denom) if denom > 0.0 else 0.0


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or right.size != left.size:
        return np.nan
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def longest_true_run(mask: np.ndarray) -> int:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def adjacent_sign_persistence(values: np.ndarray, observed: np.ndarray) -> float:
    nonzero = values != 0.0
    valid_pairs = observed[:-1] & observed[1:] & nonzero[:-1] & nonzero[1:]
    if not valid_pairs.any():
        return np.nan
    return float(
        (
            np.sign(values[:-1][valid_pairs])
            == np.sign(values[1:][valid_pairs])
        ).mean()
    )


def contiguous_lag_correlation(
    values: np.ndarray,
    observed: np.ndarray,
    lag: int,
) -> float:
    if lag <= 0 or values.size <= lag:
        return np.nan
    valid_pairs = observed[:-lag] & observed[lag:]
    if int(valid_pairs.sum()) < 3:
        return np.nan
    return safe_pearson(
        values[:-lag][valid_pairs],
        values[lag:][valid_pairs],
    )


def positive_sequence_iat(
    values: np.ndarray,
    observed: np.ndarray,
    max_lag: int = 20,
) -> float:
    """Return a descriptive positive-sequence integrated autocorrelation time."""
    tau = 1.0
    found = False
    for lag in range(1, min(max_lag, values.size - 1) + 1):
        corr = contiguous_lag_correlation(values, observed, lag)
        if not np.isfinite(corr):
            continue
        found = True
        if corr <= 0.0:
            break
        tau += 2.0 * corr
    return float(tau) if found else np.nan


@dataclass
class ScopeAccumulator:
    rows: int = 0
    wells: set[str] = field(default_factory=set)
    mean_sse: float = 0.0
    map_sse: float = 0.0
    viterbi_sse: float = 0.0
    rowwise_oracle_decoder_sse: float = 0.0
    mean_to_map_squared_distance_sum: float = 0.0
    mean_to_viterbi_squared_distance_sum: float = 0.0
    mean_error_sum: float = 0.0
    posterior_std_sum: float = 0.0
    advantage_sum: float = 0.0
    advantage_parts: list[np.ndarray] = field(default_factory=list)
    candidate_favored: int = 0
    truth_favored: int = 0
    map_improves: int = 0
    viterbi_improves: int = 0

    def update(self, well: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        truth = frame["truth"].to_numpy(np.float64)
        mean_error = frame["posterior_mean"].to_numpy(np.float64) - truth
        map_error = frame["marginal_map"].to_numpy(np.float64) - truth
        vit_error = frame["global_viterbi"].to_numpy(np.float64) - truth
        advantage = frame["mean_nll_advantage"].to_numpy(np.float64)
        self.rows += int(len(frame))
        self.wells.add(well)
        self.mean_sse += float(np.dot(mean_error, mean_error))
        self.map_sse += float(np.dot(map_error, map_error))
        self.viterbi_sse += float(np.dot(vit_error, vit_error))
        self.rowwise_oracle_decoder_sse += float(
            np.minimum.reduce(
                [mean_error**2, map_error**2, vit_error**2]
            ).sum()
        )
        self.mean_to_map_squared_distance_sum += float(
            np.sum(
                (
                    frame["posterior_mean"].to_numpy(np.float64)
                    - frame["marginal_map"].to_numpy(np.float64)
                )
                ** 2
            )
        )
        self.mean_to_viterbi_squared_distance_sum += float(
            np.sum(
                (
                    frame["posterior_mean"].to_numpy(np.float64)
                    - frame["global_viterbi"].to_numpy(np.float64)
                )
                ** 2
            )
        )
        self.mean_error_sum += float(mean_error.sum())
        self.posterior_std_sum += float(frame["posterior_std"].sum())
        self.advantage_sum += float(advantage.sum())
        self.advantage_parts.append(advantage.copy())
        self.candidate_favored += int((advantage > 0.0).sum())
        self.truth_favored += int((advantage < 0.0).sum())
        self.map_improves += int((np.abs(map_error) < np.abs(mean_error)).sum())
        self.viterbi_improves += int(
            (np.abs(vit_error) < np.abs(mean_error)).sum()
        )

    def to_row(self, scope: str) -> dict[str, Any]:
        if self.rows == 0:
            raise ValueError(f"{scope}: empty scope")
        advantage = np.concatenate(self.advantage_parts)
        return {
            "scope": scope,
            "rows": self.rows,
            "wells": len(self.wells),
            "mean_rmse_ft": float(np.sqrt(self.mean_sse / self.rows)),
            "map_rmse_ft": float(np.sqrt(self.map_sse / self.rows)),
            "viterbi_rmse_ft": float(np.sqrt(self.viterbi_sse / self.rows)),
            "rowwise_oracle_decoder_rmse_ft": float(
                np.sqrt(self.rowwise_oracle_decoder_sse / self.rows)
            ),
            "posterior_mean_to_map_rmse_ft": float(
                np.sqrt(
                    self.mean_to_map_squared_distance_sum / self.rows
                )
            ),
            "posterior_mean_to_viterbi_rmse_ft": float(
                np.sqrt(
                    self.mean_to_viterbi_squared_distance_sum / self.rows
                )
            ),
            "mean_bias_ft": self.mean_error_sum / self.rows,
            "posterior_std_mean": self.posterior_std_sum / self.rows,
            "mean_truth_minus_candidate_nll": self.advantage_sum / self.rows,
            "median_truth_minus_candidate_nll": float(np.median(advantage)),
            "fraction_gr_favors_candidate": self.candidate_favored / self.rows,
            "fraction_gr_favors_truth": self.truth_favored / self.rows,
            "fraction_map_improves_mean": self.map_improves / self.rows,
            "fraction_viterbi_improves_mean": self.viterbi_improves / self.rows,
        }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = require_file(root / "data/raw/train")
    candidates = require_file(root / args.candidates)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    scope_accumulators = {
        "all": ScopeAccumulator(),
        **{label: ScopeAccumulator() for label, _, _ in ERROR_BUCKETS},
        "persistent_offset_rows": ScopeAccumulator(),
    }
    seen_rows = 0

    for well, frame in iter_well_frames(candidates, args.chunksize):
        (
            horizontal,
            typewell_tvt,
            typewell_gr,
            sigma,
            init_rate,
            rate_span,
            cal_a,
            cal_b,
        ) = load_well_inputs(train_dir, well)
        row_idx = pd.to_numeric(frame["row_idx"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(row_idx, np.sort(row_idx)):
            raise ValueError(f"{well}: row_idx not sorted")
        raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
        gr_fill = float(np.nanmean(typewell_gr))
        observed_gr = (
            raw_gr.interpolate(limit_direction="both")
            .fillna(gr_fill)
            .to_numpy(np.float64)[row_idx]
        )
        truth = pd.to_numeric(
            frame["true_tvt_readout_only"], errors="raise"
        ).to_numpy(np.float64)
        mean = pd.to_numeric(frame["posterior_mean"], errors="raise").to_numpy(
            np.float64
        )
        marginal_map = pd.to_numeric(
            frame["marginal_map"], errors="raise"
        ).to_numpy(np.float64)
        viterbi = pd.to_numeric(frame["topk_path_1"], errors="raise").to_numpy(
            np.float64
        )
        nll_truth = emission_nll(
            observed_gr, truth, typewell_tvt, typewell_gr, sigma
        )
        nll_mean = emission_nll(observed_gr, mean, typewell_tvt, typewell_gr, sigma)
        nll_map = emission_nll(
            observed_gr, marginal_map, typewell_tvt, typewell_gr, sigma
        )
        nll_viterbi = emission_nll(
            observed_gr, viterbi, typewell_tvt, typewell_gr, sigma
        )
        truth_reference_gr = np.interp(truth, typewell_tvt, typewell_gr)
        mean_reference_gr = np.interp(mean, typewell_tvt, typewell_gr)
        affine_truth_reference_gr = cal_a * truth_reference_gr + cal_b
        affine_mean_reference_gr = cal_a * mean_reference_gr + cal_b
        affine_truth_nll = 0.5 * np.minimum(
            ((observed_gr - affine_truth_reference_gr) / sigma) ** 2,
            600.0,
        )
        affine_mean_nll = 0.5 * np.minimum(
            ((observed_gr - affine_mean_reference_gr) / sigma) ** 2,
            600.0,
        )
        affine_advantage = affine_truth_nll - affine_mean_nll

        last_known_row = int(row_idx[0]) - 1
        last_row = horizontal.iloc[last_known_row]
        eval_md = pd.to_numeric(horizontal.iloc[row_idx]["MD"], errors="raise").to_numpy(
            np.float64
        )
        eval_z = pd.to_numeric(horizontal.iloc[row_idx]["Z"], errors="raise").to_numpy(
            np.float64
        )
        eval_dmd = np.diff(np.concatenate([[float(last_row["MD"])], eval_md]))
        true_rate = np.diff(
            np.concatenate(
                [
                    [float(last_row["TVT_input"]) + float(last_row["Z"])],
                    truth + eval_z,
                ]
            )
        ) / eval_dmd
        mean_rate = np.diff(
            np.concatenate(
                [
                    [float(last_row["TVT_input"]) + float(last_row["Z"])],
                    mean + eval_z,
                ]
            )
        ) / eval_dmd

        work = pd.DataFrame(
            {
                "well": well,
                "row_idx": row_idx,
                "truth": truth,
                "posterior_mean": mean,
                "marginal_map": marginal_map,
                "global_viterbi": viterbi,
                "posterior_std": pd.to_numeric(
                    frame["posterior_std"], errors="raise"
                ).to_numpy(np.float64),
                "marginal_mode_mass": pd.to_numeric(
                    frame["marginal_mode_mass"], errors="raise"
                ).to_numpy(np.float64),
                "marginal_mode_gap": pd.to_numeric(
                    frame["marginal_mode_gap"], errors="raise"
                ).to_numpy(np.float64),
                "mean_nll_advantage": nll_truth - nll_mean,
                "map_nll_advantage": nll_truth - nll_map,
                "viterbi_nll_advantage": nll_truth - nll_viterbi,
                "true_rate_outside": np.abs(true_rate) > rate_span,
                "raw_gr_missing": raw_gr.isna().to_numpy()[row_idx],
            }
        )

        mean_error = mean - truth
        runs = persistent_runs(
            np.abs(mean_error) > PERSISTENT_ERROR_FT,
            PERSISTENT_MIN_ROWS,
        )
        work["persistent_offset_episode"] = False
        for episode_index, (start, end) in enumerate(runs):
            work.loc[start : end - 1, "persistent_offset_episode"] = True
            part = work.iloc[start:end]
            error = mean_error[start:end]
            map_error = marginal_map[start:end] - truth[start:end]
            vit_error = viterbi[start:end] - truth[start:end]
            advantage = part["mean_nll_advantage"].to_numpy(np.float64)
            raw_missing = part["raw_gr_missing"].to_numpy(bool)
            raw_observed = ~raw_missing
            observed_advantage = advantage[~raw_missing]
            imputed_advantage = advantage[raw_missing]
            affine_observed_advantage = affine_advantage[start:end][~raw_missing]
            observed_truth_nll = nll_truth[start:end][~raw_missing]
            observed_mean_nll = nll_mean[start:end][~raw_missing]
            raw_observed_gr = observed_gr[start:end][~raw_missing]
            truth_observed_gr = truth_reference_gr[start:end][~raw_missing]
            mean_observed_gr = mean_reference_gr[start:end][~raw_missing]
            preceding_within5 = np.flatnonzero(np.abs(mean_error[:start]) <= 5.0)
            following_within5 = np.flatnonzero(np.abs(mean_error[end:]) <= 5.0)
            last_within5 = (
                int(preceding_within5[-1]) if preceding_within5.size else None
            )
            next_within5 = (
                int(end + following_within5[0])
                if following_within5.size
                else None
            )
            pre_start = max(0, start - 128)
            pre_error = mean_error[pre_start:start]
            episode_rows.append(
                {
                    "episode_id": f"{well}:{episode_index:03d}",
                    "well": well,
                    "start_row_idx": int(row_idx[start]),
                    "end_row_idx_exclusive": int(row_idx[end - 1] + 1),
                    "start_suffix_offset": int(start),
                    "rows": int(end - start),
                    "suffix_rows": int(len(work)),
                    "start_suffix_fraction": float(start / len(work)),
                    "episode_suffix_fraction": float((end - start) / len(work)),
                    "rows_from_last_within5_to_episode_start": (
                        int(start - last_within5) if last_within5 is not None else np.nan
                    ),
                    "rows_from_episode_end_to_next_within5": (
                        int(next_within5 - end) if next_within5 is not None else np.nan
                    ),
                    "pre128_error_slope_ft_per_row": (
                        linear_slope(pre_error) if pre_error.size else np.nan
                    ),
                    "pre128_error_start_ft": (
                        float(pre_error[0]) if pre_error.size else np.nan
                    ),
                    "pre128_error_end_ft": (
                        float(pre_error[-1]) if pre_error.size else np.nan
                    ),
                    "prefix_init_rate": init_rate,
                    "episode_true_rate_median": float(
                        np.median(true_rate[start:end])
                    ),
                    "episode_candidate_rate_median": float(
                        np.median(mean_rate[start:end])
                    ),
                    "episode_true_minus_init_rate_median": float(
                        np.median(true_rate[start:end] - init_rate)
                    ),
                    "pre128_true_minus_init_rate_median": (
                        float(np.median(true_rate[pre_start:start] - init_rate))
                        if start > pre_start
                        else np.nan
                    ),
                    "mean_error_ft": float(error.mean()),
                    "rmse_ft": float(np.sqrt(np.mean(error**2))),
                    "error_std_ft": float(error.std()),
                    "error_slope_ft_per_row": linear_slope(error),
                    "error_sign_consistency": float(
                        np.mean(np.sign(error) == np.sign(np.median(error)))
                    ),
                    "map_rmse_ft": float(np.sqrt(np.mean(map_error**2))),
                    "viterbi_rmse_ft": float(np.sqrt(np.mean(vit_error**2))),
                    "rowwise_oracle_decoder_rmse_ft": float(
                        np.sqrt(
                            np.mean(
                                np.minimum.reduce(
                                    [
                                        error**2,
                                        map_error**2,
                                        vit_error**2,
                                    ]
                                )
                            )
                        )
                    ),
                    "posterior_mean_to_map_rmse_ft": float(
                        np.sqrt(
                            np.mean(
                                (
                                    mean[start:end]
                                    - marginal_map[start:end]
                                )
                                ** 2
                            )
                        )
                    ),
                    "posterior_mean_to_viterbi_rmse_ft": float(
                        np.sqrt(
                            np.mean(
                                (
                                    mean[start:end]
                                    - viterbi[start:end]
                                )
                                ** 2
                            )
                        )
                    ),
                    "fraction_map_improves_mean": float(
                        (np.abs(map_error) < np.abs(error)).mean()
                    ),
                    "fraction_viterbi_improves_mean": float(
                        (np.abs(vit_error) < np.abs(error)).mean()
                    ),
                    "mean_truth_minus_candidate_nll": float(advantage.mean()),
                    "total_truth_minus_candidate_nll": float(advantage.sum()),
                    "observed_total_truth_minus_candidate_nll": float(
                        observed_advantage.sum()
                    ),
                    "imputed_total_truth_minus_candidate_nll": float(
                        imputed_advantage.sum()
                    ),
                    "observed_mean_truth_minus_candidate_nll": (
                        float(observed_advantage.mean())
                        if observed_advantage.size
                        else np.nan
                    ),
                    "observed_longest_candidate_favoring_run": longest_true_run(
                        raw_observed & (advantage > 0.0)
                    ),
                    "observed_longest_truth_favoring_run": longest_true_run(
                        raw_observed & (advantage < 0.0)
                    ),
                    "observed_adjacent_sign_persistence": (
                        adjacent_sign_persistence(advantage, raw_observed)
                    ),
                    "observed_advantage_lag1_correlation": (
                        contiguous_lag_correlation(
                            advantage,
                            raw_observed,
                            lag=1,
                        )
                    ),
                    "observed_advantage_positive_sequence_iat_lag20": (
                        positive_sequence_iat(
                            advantage,
                            raw_observed,
                            max_lag=20,
                        )
                    ),
                    "affine_observed_total_truth_minus_candidate_nll": float(
                        affine_observed_advantage.sum()
                    ),
                    "affine_observed_mean_truth_minus_candidate_nll": (
                        float(affine_observed_advantage.mean())
                        if affine_observed_advantage.size
                        else np.nan
                    ),
                    "observed_truth_emission_cap_fraction": (
                        float((observed_truth_nll >= 300.0).mean())
                        if observed_truth_nll.size
                        else np.nan
                    ),
                    "observed_candidate_emission_cap_fraction": (
                        float((observed_mean_nll >= 300.0).mean())
                        if observed_mean_nll.size
                        else np.nan
                    ),
                    "imputed_mean_truth_minus_candidate_nll": (
                        float(imputed_advantage.mean())
                        if imputed_advantage.size
                        else np.nan
                    ),
                    "reference_candidate_truth_gr_corr_observed": safe_pearson(
                        mean_observed_gr,
                        truth_observed_gr,
                    ),
                    "raw_candidate_gr_corr_observed": safe_pearson(
                        raw_observed_gr,
                        mean_observed_gr,
                    ),
                    "raw_truth_gr_corr_observed": safe_pearson(
                        raw_observed_gr,
                        truth_observed_gr,
                    ),
                    "raw_candidate_gr_rmse_observed": (
                        float(
                            np.sqrt(
                                np.mean((raw_observed_gr - mean_observed_gr) ** 2)
                            )
                        )
                        if raw_observed_gr.size
                        else np.nan
                    ),
                    "raw_truth_gr_rmse_observed": (
                        float(
                            np.sqrt(
                                np.mean((raw_observed_gr - truth_observed_gr) ** 2)
                            )
                        )
                        if raw_observed_gr.size
                        else np.nan
                    ),
                    "reference_candidate_truth_gr_rmse_observed": (
                        float(
                            np.sqrt(
                                np.mean(
                                    (mean_observed_gr - truth_observed_gr) ** 2
                                )
                            )
                        )
                        if raw_observed_gr.size
                        else np.nan
                    ),
                    "median_truth_minus_candidate_nll": float(np.median(advantage)),
                    "fraction_gr_favors_candidate": float((advantage > 0.0).mean()),
                    "fraction_gr_favors_truth": float((advantage < 0.0).mean()),
                    "mean_map_nll_advantage": float(
                        part["map_nll_advantage"].mean()
                    ),
                    "mean_viterbi_nll_advantage": float(
                        part["viterbi_nll_advantage"].mean()
                    ),
                    "posterior_std_mean": float(part["posterior_std"].mean()),
                    "marginal_mode_mass_mean": float(
                        part["marginal_mode_mass"].mean()
                    ),
                    "marginal_mode_gap_mean": float(
                        part["marginal_mode_gap"].mean()
                    ),
                    "true_rate_outside_fraction": float(
                        part["true_rate_outside"].mean()
                    ),
                    "raw_gr_missing_fraction": float(part["raw_gr_missing"].mean()),
                    "prefix_sigma_gr": sigma,
                    "prefix_cal_a": cal_a,
                    "prefix_cal_b": cal_b,
                    "rate_span": rate_span,
                }
            )

        scope_accumulators["all"].update(well, work)
        abs_error = np.abs(mean_error)
        for label, lower, upper in ERROR_BUCKETS:
            mask = (abs_error >= lower) & (abs_error < upper)
            scope_accumulators[label].update(well, work.loc[mask])
        scope_accumulators["persistent_offset_rows"].update(
            well,
            work.loc[work["persistent_offset_episode"]],
        )
        seen_rows += len(work)
        well_rows.append(
            {
                "well": well,
                "rows": int(len(work)),
                "rmse_ft": float(np.sqrt(np.mean(mean_error**2))),
                "bias_ft": float(mean_error.mean()),
                "posterior_std_mean": float(work["posterior_std"].mean()),
                "mean_truth_minus_candidate_nll": float(
                    work["mean_nll_advantage"].mean()
                ),
                "fraction_gr_favors_candidate": float(
                    (work["mean_nll_advantage"] > 0.0).mean()
                ),
                "fraction_gr_favors_truth": float(
                    (work["mean_nll_advantage"] < 0.0).mean()
                ),
                "persistent_episode_count": int(len(runs)),
                "persistent_episode_rows": int(sum(end - start for start, end in runs)),
                "true_rate_outside_fraction": float(
                    work["true_rate_outside"].mean()
                ),
                "raw_gr_missing_fraction": float(work["raw_gr_missing"].mean()),
                "prefix_sigma_gr": sigma,
                "rate_span": rate_span,
            }
        )

    if seen_rows != 3_783_989:
        raise ValueError(f"expected 3,783,989 rows, got {seen_rows}")
    episodes = pd.DataFrame(episode_rows)
    by_well = pd.DataFrame(well_rows)
    if int(episodes["rows"].sum()) != scope_accumulators[
        "persistent_offset_rows"
    ].rows:
        raise ValueError("persistent episode row accounting mismatch")
    split_advantage = (
        episodes["observed_total_truth_minus_candidate_nll"]
        + episodes["imputed_total_truth_minus_candidate_nll"]
    )
    if not np.allclose(
        split_advantage,
        episodes["total_truth_minus_candidate_nll"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("observed/imputed emission evidence does not reconcile")
    if len(episodes):
        total_advantage = episodes["total_truth_minus_candidate_nll"]
        episodes["emission_evidence_class"] = np.select(
            [
                total_advantage >= STRONG_SEQUENCE_NLL,
                total_advantage <= -STRONG_SEQUENCE_NLL,
            ],
            [
                "candidate_strong",
                "truth_strong",
            ],
            default="near_tie",
        )
        observed_total_advantage = episodes[
            "observed_total_truth_minus_candidate_nll"
        ]
        episodes["observed_emission_evidence_class"] = np.select(
            [
                observed_total_advantage >= STRONG_SEQUENCE_NLL,
                observed_total_advantage <= -STRONG_SEQUENCE_NLL,
            ],
            [
                "candidate_strong",
                "truth_strong",
            ],
            default="near_tie",
        )
        affine_observed_total_advantage = episodes[
            "affine_observed_total_truth_minus_candidate_nll"
        ]
        episodes["affine_observed_emission_evidence_class"] = np.select(
            [
                affine_observed_total_advantage >= STRONG_SEQUENCE_NLL,
                affine_observed_total_advantage <= -STRONG_SEQUENCE_NLL,
            ],
            [
                "candidate_strong",
                "truth_strong",
            ],
            default="near_tie",
        )
        viterbi_ratio = episodes["viterbi_rmse_ft"] / episodes["rmse_ft"]
        episodes["viterbi_recovery_class"] = np.select(
            [
                viterbi_ratio <= 0.5,
                viterbi_ratio < 1.0,
            ],
            [
                "large_recovery",
                "partial_recovery",
            ],
            default="not_better",
        )
        episodes["episode_sse"] = episodes["rmse_ft"] ** 2 * episodes["rows"]
        episodes["viterbi_rmse_gain_ft"] = (
            episodes["rmse_ft"] - episodes["viterbi_rmse_ft"]
        )

    scope_rows = [
        accumulator.to_row(scope)
        for scope, accumulator in scope_accumulators.items()
    ]
    scope_metrics = pd.DataFrame(scope_rows)

    episode_summary = {
        "episodes": int(len(episodes)),
        "episode_wells": int(episodes["well"].nunique()) if len(episodes) else 0,
        "episode_rows": int(episodes["rows"].sum()) if len(episodes) else 0,
        "episode_row_fraction": (
            float(episodes["rows"].sum() / seen_rows) if len(episodes) else 0.0
        ),
        "episodes_majority_gr_favors_candidate": (
            int((episodes["fraction_gr_favors_candidate"] > 0.5).sum())
            if len(episodes)
            else 0
        ),
        "episodes_majority_gr_favors_truth": (
            int((episodes["fraction_gr_favors_truth"] > 0.5).sum())
            if len(episodes)
            else 0
        ),
        "episodes_map_rmse_better_than_mean": (
            int((episodes["map_rmse_ft"] < episodes["rmse_ft"]).sum())
            if len(episodes)
            else 0
        ),
        "episodes_viterbi_rmse_better_than_mean": (
            int((episodes["viterbi_rmse_ft"] < episodes["rmse_ft"]).sum())
            if len(episodes)
            else 0
        ),
        "episodes_true_rate_outside_ge_0p10": (
            int((episodes["true_rate_outside_fraction"] >= 0.10).sum())
            if len(episodes)
            else 0
        ),
        "strong_sequence_nll_threshold": STRONG_SEQUENCE_NLL,
    }
    overall = scope_metrics.loc[scope_metrics["scope"] == "all"].iloc[0]
    summary = {
        "rows": seen_rows,
        "wells": int(len(by_well)),
        "mean_rmse_ft": float(overall["mean_rmse_ft"]),
        "map_rmse_ft": float(overall["map_rmse_ft"]),
        "viterbi_rmse_ft": float(overall["viterbi_rmse_ft"]),
        "rowwise_oracle_decoder_rmse_ft": float(
            overall["rowwise_oracle_decoder_rmse_ft"]
        ),
        "posterior_mean_to_map_rmse_ft": float(
            overall["posterior_mean_to_map_rmse_ft"]
        ),
        "posterior_mean_to_viterbi_rmse_ft": float(
            overall["posterior_mean_to_viterbi_rmse_ft"]
        ),
        "persistent_error_ft": PERSISTENT_ERROR_FT,
        "persistent_min_rows": PERSISTENT_MIN_ROWS,
        **episode_summary,
    }

    scope_metrics.to_csv(output / "scope_metrics.csv", index=False)
    by_well.sort_values("rmse_ft", ascending=False).to_csv(
        output / "by_well_metrics.csv", index=False
    )
    episodes.sort_values(["rmse_ft", "rows"], ascending=False).to_csv(
        output / "persistent_offset_episodes.csv", index=False
    )
    if len(episodes):
        total_episode_sse = float(episodes["episode_sse"].sum())
        class_summary = (
            episodes.groupby(
                ["emission_evidence_class", "viterbi_recovery_class"],
                observed=True,
            )
            .agg(
                episodes=("episode_id", "size"),
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                episode_sse=("episode_sse", "sum"),
                rmse_ft_mean=("rmse_ft", "mean"),
                rmse_ft_median=("rmse_ft", "median"),
                raw_gr_missing_fraction_mean=("raw_gr_missing_fraction", "mean"),
                true_rate_outside_fraction_mean=(
                    "true_rate_outside_fraction",
                    "mean",
                ),
                posterior_std_mean=("posterior_std_mean", "mean"),
                reference_candidate_truth_gr_corr_median=(
                    "reference_candidate_truth_gr_corr_observed",
                    "median",
                ),
            )
            .reset_index()
        )
        class_summary["episode_sse_fraction"] = (
            class_summary["episode_sse"] / total_episode_sse
        )
        class_summary.sort_values("episode_sse", ascending=False).to_csv(
            output / "episode_class_summary.csv",
            index=False,
        )
        observed_class_summary = (
            episodes.groupby(
                ["observed_emission_evidence_class", "viterbi_recovery_class"],
                observed=True,
            )
            .agg(
                episodes=("episode_id", "size"),
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                episode_sse=("episode_sse", "sum"),
                rmse_ft_mean=("rmse_ft", "mean"),
                rmse_ft_median=("rmse_ft", "median"),
                raw_gr_missing_fraction_mean=("raw_gr_missing_fraction", "mean"),
                posterior_std_mean=("posterior_std_mean", "mean"),
                reference_candidate_truth_gr_corr_median=(
                    "reference_candidate_truth_gr_corr_observed",
                    "median",
                ),
            )
            .reset_index()
        )
        observed_class_summary["episode_sse_fraction"] = (
            observed_class_summary["episode_sse"] / total_episode_sse
        )
        observed_class_summary.sort_values(
            "episode_sse",
            ascending=False,
        ).to_csv(
            output / "episode_observed_class_summary.csv",
            index=False,
        )
        evidence_persistence_summary = (
            episodes.groupby(
                "observed_emission_evidence_class",
                observed=True,
            )
            .agg(
                episodes=("episode_id", "size"),
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                episode_sse=("episode_sse", "sum"),
                candidate_run_median=(
                    "observed_longest_candidate_favoring_run",
                    "median",
                ),
                candidate_run_p90=(
                    "observed_longest_candidate_favoring_run",
                    lambda values: float(np.quantile(values, 0.90)),
                ),
                truth_run_median=(
                    "observed_longest_truth_favoring_run",
                    "median",
                ),
                truth_run_p90=(
                    "observed_longest_truth_favoring_run",
                    lambda values: float(np.quantile(values, 0.90)),
                ),
                adjacent_sign_persistence_median=(
                    "observed_adjacent_sign_persistence",
                    "median",
                ),
                lag1_correlation_median=(
                    "observed_advantage_lag1_correlation",
                    "median",
                ),
                positive_sequence_iat_lag20_median=(
                    "observed_advantage_positive_sequence_iat_lag20",
                    "median",
                ),
            )
            .reset_index()
        )
        evidence_persistence_summary["episode_sse_fraction"] = (
            evidence_persistence_summary["episode_sse"] / total_episode_sse
        )
        evidence_persistence_summary.to_csv(
            output / "episode_evidence_persistence_summary.csv",
            index=False,
        )
        summary["observed_advantage_lag1_correlation_median"] = float(
            episodes["observed_advantage_lag1_correlation"].median()
        )
        summary["observed_adjacent_sign_persistence_median"] = float(
            episodes["observed_adjacent_sign_persistence"].median()
        )
        summary["observed_advantage_positive_sequence_iat_lag20_median"] = float(
            episodes[
                "observed_advantage_positive_sequence_iat_lag20"
            ].median()
        )
        summary[
            "candidate_run_vs_observed_nll_total_spearman"
        ] = float(
            episodes[
                [
                    "observed_longest_candidate_favoring_run",
                    "observed_total_truth_minus_candidate_nll",
                ]
            ].corr(method="spearman").iloc[0, 1]
        )
        summary["candidate_run_vs_episode_rmse_spearman"] = float(
            episodes[
                [
                    "observed_longest_candidate_favoring_run",
                    "rmse_ft",
                ]
            ].corr(method="spearman").iloc[0, 1]
        )
        summary["posterior_std_vs_viterbi_rmse_gain_spearman"] = float(
            episodes[
                ["posterior_std_mean", "viterbi_rmse_gain_ft"]
            ].corr(method="spearman").iloc[0, 1]
        )
        affine_observed_class_summary = (
            episodes.groupby(
                [
                    "affine_observed_emission_evidence_class",
                    "viterbi_recovery_class",
                ],
                observed=True,
            )
            .agg(
                episodes=("episode_id", "size"),
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                episode_sse=("episode_sse", "sum"),
                rmse_ft_mean=("rmse_ft", "mean"),
                raw_gr_missing_fraction_mean=("raw_gr_missing_fraction", "mean"),
                posterior_std_mean=("posterior_std_mean", "mean"),
            )
            .reset_index()
        )
        affine_observed_class_summary["episode_sse_fraction"] = (
            affine_observed_class_summary["episode_sse"] / total_episode_sse
        )
        affine_observed_class_summary.sort_values(
            "episode_sse",
            ascending=False,
        ).to_csv(
            output / "episode_affine_observed_class_summary.csv",
            index=False,
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
