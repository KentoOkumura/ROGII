#!/usr/bin/env python3
"""Decompose pre-offset transition pressure into physical rate and rate lag.

Using frozen exp270 posterior-mean TVT and readout-only truth, this script
reconstructs true and decoded U-space rates on the original exp209 geometry.
It measures true rate magnitude, true rate acceleration, and decoded-minus-
true rate error in non-overlapping rings before each persistent-offset onset.
No HMM, model, or prediction is rerun.
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
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_posterior_geometry_timing_20260726/"
    "episode_posterior_geometry.csv"
)
DEFAULT_OUTPUT = Path("studies/hmm_exp209_rate_lag_timing_20260726")
RINGS = (
    ("000_016", 0, 16),
    ("016_064", 16, 64),
    ("064_128", 64, 128),
    ("128_256", 128, 256),
    ("256_512", 256, 512),
)
RATE_METRICS = (
    "true_rate_abs",
    "true_rate_acceleration_abs",
    "decoded_rate_abs",
    "rate_error_signed",
    "rate_error_abs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
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


def fraction_within(mask: pd.Series, eligible: pd.Series) -> float | None:
    count = int(eligible.sum())
    if count == 0:
        return None
    return float((mask & eligible).sum() / count)


def weighted_fraction_within(
    mask: pd.Series,
    eligible: pd.Series,
    weights: pd.Series,
) -> float | None:
    denominator = float(weights.loc[eligible].sum())
    if denominator <= 0.0:
        return None
    return float(weights.loc[mask & eligible].sum() / denominator)


def add_interval(
    record: dict[str, Any],
    label: str,
    start: int,
    end: int,
    arrays: dict[str, np.ndarray],
) -> None:
    rows = end - start
    record[f"rate_{label}_rows"] = int(rows)
    for name, values in arrays.items():
        part = values[start:end]
        finite = part[np.isfinite(part)]
        record[f"rate_{label}_{name}_mean"] = (
            float(finite.mean()) if len(finite) else None
        )
        record[f"rate_{label}_{name}_median"] = (
            float(np.median(finite)) if len(finite) else None
        )


def add_changes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in RATE_METRICS:
        result[f"rate_near_minus_far_{name}"] = (
            result[f"rate_ring_000_016_{name}_mean"]
            - result[f"rate_ring_256_512_{name}_mean"]
        )
    near_signed = result[
        "rate_ring_000_016_rate_error_signed_mean"
    ]
    result["near_rate_error_sign_matches_episode_offset"] = (
        np.sign(near_signed) == np.sign(result["mean_error_ft"])
    ) & near_signed.notna()
    return result


def pooled_interval_mean(
    group: pd.DataFrame,
    label: str,
    name: str,
) -> float | None:
    value = f"rate_{label}_{name}_mean"
    rows = f"rate_{label}_rows"
    finite = group[value].notna() & (group[rows] > 0)
    denominator = float(group.loc[finite, rows].sum())
    if denominator <= 0.0:
        return None
    return float(
        (
            group.loc[finite, value]
            * group.loc[finite, rows]
        ).sum()
        / denominator
    )


def summarize_group(
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
    labels = ("ring_256_512", "ring_000_016", "episode")
    for label in labels:
        for name in RATE_METRICS:
            column = f"rate_{label}_{name}_mean"
            result[f"{column}_episode_mean"] = float(
                group[column].mean()
            )
            result[f"{column}_episode_median"] = float(
                group[column].median()
            )
            result[f"{column}_pooled"] = pooled_interval_mean(
                group,
                label,
                name,
            )
    for name in RATE_METRICS:
        change = f"rate_near_minus_far_{name}"
        finite = group[change].notna()
        positive = finite & (group[change] > 0.0)
        result[f"{change}_eligible_episodes"] = int(finite.sum())
        result[f"{change}_mean"] = float(group[change].mean())
        result[f"{change}_median"] = float(group[change].median())
        result[f"{change}_positive_fraction_eligible"] = fraction_within(
            positive,
            finite,
        )
        result[f"{change}_positive_sse_fraction_eligible"] = (
            weighted_fraction_within(
                positive,
                finite,
                group["episode_sse"],
            )
        )
        result[f"{change}_vs_rmse_spearman"] = safe_spearman(
            group[change],
            group["rmse_ft"],
        )
        result[f"{change}_vs_transition_crescendo_spearman"] = (
            safe_spearman(
                group[change],
                group["truth_pre_crescendo_near_minus_far_nll"],
            )
        )
        result[f"{change}_vs_pre128_error_slope_spearman"] = (
            safe_spearman(
                group[change],
                group["pre128_error_slope_ft_per_row"].abs(),
            )
        )
    sign_eligible = group[
        "rate_ring_000_016_rate_error_signed_mean"
    ].notna()
    sign_match = group[
        "near_rate_error_sign_matches_episode_offset"
    ]
    result.update(
        {
            "near_rate_error_sign_matches_episode_offset_fraction": (
                fraction_within(sign_match, sign_eligible)
            ),
            "near_rate_error_sign_matches_episode_offset_sse_fraction": (
                weighted_fraction_within(
                    sign_match,
                    sign_eligible,
                    group["episode_sse"],
                )
            ),
            "near_abs_rate_error_vs_pre128_error_slope_spearman": (
                safe_spearman(
                    group["rate_ring_000_016_rate_error_abs_mean"],
                    group["pre128_error_slope_ft_per_row"].abs(),
                )
            ),
            "near_true_rate_abs_vs_transition_nll_spearman": (
                safe_spearman(
                    group["rate_ring_000_016_true_rate_abs_mean"],
                    group[
                        "truth_ring_000_016_transition_nll_mean"
                    ],
                )
            ),
        }
    )
    far_abs_error = pooled_interval_mean(
        group,
        "ring_256_512",
        "rate_error_abs",
    )
    near_abs_error = pooled_interval_mean(
        group,
        "ring_000_016",
        "rate_error_abs",
    )
    result["near_vs_far_abs_rate_error_ratio_pooled"] = (
        float(near_abs_error / far_abs_error)
        if far_abs_error is not None
        and far_abs_error > 0.0
        and near_abs_error is not None
        else None
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = resolve(root, Path("data/raw/train"))
    candidates_path = resolve(root, args.candidates)
    episodes_path = resolve(root, args.episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)
    episode_lookup = {
        well: group.to_dict("records")
        for well, group in episodes.groupby("well", sort=False)
    }
    records: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    maximum_derivative_identity_error = 0.0
    for well, frame in iter_well_frames(candidates_path, args.chunksize):
        specs = episode_lookup.get(well, [])
        if not specs:
            continue
        seen_wells.add(well)
        (
            horizontal,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = load_well_inputs(train_dir, well)
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        previous_index = int(row_index[0]) - 1
        md = pd.to_numeric(
            horizontal["MD"],
            errors="raise",
        ).to_numpy(np.float64)
        z = pd.to_numeric(
            horizontal["Z"],
            errors="raise",
        ).to_numpy(np.float64)
        dmd = np.maximum(
            np.diff(np.concatenate([[md[previous_index]], md[row_index]])),
            1.0,
        )
        truth = pd.to_numeric(
            frame["true_tvt_readout_only"],
            errors="raise",
        ).to_numpy(np.float64)
        posterior_mean = pd.to_numeric(
            frame["posterior_mean"],
            errors="raise",
        ).to_numpy(np.float64)
        previous_tvt = float(horizontal.iloc[previous_index]["TVT_input"])
        true_rate = np.diff(
            np.concatenate(
                [[previous_tvt + z[previous_index]], truth + z[row_index]]
            )
        ) / dmd
        decoded_rate = np.diff(
            np.concatenate(
                [
                    [previous_tvt + z[previous_index]],
                    posterior_mean + z[row_index],
                ]
            )
        ) / dmd
        rate_error = decoded_rate - true_rate
        tvt_error = posterior_mean - truth
        derivative_error = np.diff(
            np.concatenate([[0.0], tvt_error])
        ) / dmd
        maximum_derivative_identity_error = max(
            maximum_derivative_identity_error,
            float(np.max(np.abs(rate_error - derivative_error))),
        )
        true_acceleration = np.full(len(true_rate), np.nan, np.float64)
        true_acceleration[1:] = np.abs(np.diff(true_rate))
        arrays = {
            "true_rate_abs": np.abs(true_rate),
            "true_rate_acceleration_abs": true_acceleration,
            "decoded_rate_abs": np.abs(decoded_rate),
            "rate_error_signed": rate_error,
            "rate_error_abs": np.abs(rate_error),
        }
        for spec in specs:
            record = dict(spec)
            start = int(
                np.searchsorted(
                    row_index,
                    int(spec["start_row_idx"]),
                )
            )
            end = int(
                np.searchsorted(
                    row_index,
                    int(spec["end_row_idx_exclusive"]),
                )
            )
            if end - start != int(spec["rows"]):
                raise ValueError(
                    f"{spec['episode_id']}: episode row mismatch"
                )
            add_interval(record, "episode", start, end, arrays)
            for label, inner, outer in RINGS:
                ring_start = max(0, start - outer)
                ring_end = max(0, start - inner)
                expected_rows = int(spec[f"truth_ring_{label}_rows"])
                if ring_end - ring_start != expected_rows:
                    raise ValueError(
                        f"{spec['episode_id']}: {label} row mismatch"
                    )
                add_interval(
                    record,
                    f"ring_{label}",
                    ring_start,
                    ring_end,
                    arrays,
                )
            records.append(record)
    if seen_wells != set(episode_lookup):
        missing = sorted(set(episode_lookup) - seen_wells)
        raise ValueError(f"episode wells missing: {missing[:5]}")
    if maximum_derivative_identity_error > 1e-10:
        raise ValueError(
            "TVT-error derivative and rate error do not reconcile: "
            f"{maximum_derivative_identity_error}"
        )

    result = add_changes(pd.DataFrame(records))
    if len(result) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(result)} != {len(episodes)}"
        )
    result.to_csv(output / "episode_rate_lag_metrics.csv", index=False)
    total_sse = float(result["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in result.groupby("cause_bucket", sort=True):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_group(group, total_sse),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_rate_lag_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(result)),
            "wells": int(result["well"].nunique()),
            "episode_rows": int(result["rows"].sum()),
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(candidates_path),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(result, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "maximum_rate_error_vs_tvt_error_derivative_abs": (
                maximum_derivative_identity_error
            ),
            "interpretation": (
                "Rates use readout-only truth and the frozen smoothed "
                "posterior mean around error-defined onset. They establish "
                "kinematic timing, not the hidden filtered rate posterior."
            ),
            "prediction_generation": False,
            "hmm_rerun": False,
            "model_or_booster": False,
        },
    }
    text = json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    (output / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
