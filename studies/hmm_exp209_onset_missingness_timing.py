#!/usr/bin/env python3
"""Audit whether GR missingness rises before persistent-offset onset.

The input contains frozen, truth-late episode diagnostics and raw-observed GR
row counts for non-overlapping onset rings.  This script only aggregates those
counts; it does not rerun the HMM, generate predictions, or fit a model.
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

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_rate_directional_underresponse_20260726/"
    "episode_directional_rate_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_onset_missingness_timing_20260726"
)
RING_LABELS = (
    "256_512",
    "128_256",
    "064_128",
    "016_064",
    "000_016",
)
THRESHOLDS = (0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def fraction(mask: pd.Series, eligible: pd.Series) -> float | None:
    count = int(eligible.sum())
    if count == 0:
        return None
    return float((mask & eligible).sum() / count)


def weighted_fraction(
    mask: pd.Series,
    eligible: pd.Series,
    weights: pd.Series,
) -> float | None:
    denominator = float(weights.loc[eligible].sum())
    if denominator <= 0.0:
        return None
    return float(weights.loc[mask & eligible].sum() / denominator)


def add_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for label in RING_LABELS:
        total = pd.to_numeric(
            frame[f"truth_ring_{label}_rows"],
            errors="raise",
        )
        observed = pd.to_numeric(
            frame[f"observed_gr_ring_{label}_rows"],
            errors="raise",
        )
        if ((total < 0) | (observed < 0) | (observed > total)).any():
            raise ValueError(f"{label}: invalid observed/total row counts")
        columns[f"missing_ring_{label}_fraction"] = pd.Series(
            np.where(total > 0, 1.0 - observed / total, np.nan),
            index=frame.index,
        )
        columns[f"missing_ring_{label}_rows"] = total - observed
    result = pd.concat(
        [frame.copy(), pd.DataFrame(columns, index=frame.index)],
        axis=1,
    )
    result["missing_near_minus_far_fraction"] = (
        result["missing_ring_000_016_fraction"]
        - result["missing_ring_256_512_fraction"]
    )
    return result


def summarize_group(
    group: pd.DataFrame,
    total_sse: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
    }
    for label in RING_LABELS:
        total = group[f"truth_ring_{label}_rows"]
        missing = group[f"missing_ring_{label}_rows"]
        values = group[f"missing_ring_{label}_fraction"]
        eligible = values.notna()
        all_missing = eligible & np.isclose(values, 1.0)
        no_missing = eligible & np.isclose(values, 0.0)
        denominator = float(total.sum())
        summary[f"missing_ring_{label}_fraction_episode_mean"] = float(
            values.mean()
        )
        summary[f"missing_ring_{label}_fraction_episode_median"] = float(
            values.median()
        )
        summary[f"missing_ring_{label}_fraction_pooled"] = (
            float(missing.sum() / denominator)
            if denominator > 0.0
            else None
        )
        summary[f"missing_ring_{label}_eligible_episodes"] = int(
            eligible.sum()
        )
        summary[f"missing_ring_{label}_all_missing_fraction"] = fraction(
            all_missing,
            eligible,
        )
        summary[
            f"missing_ring_{label}_all_missing_sse_fraction"
        ] = weighted_fraction(
            all_missing,
            eligible,
            group["episode_sse"],
        )
        summary[f"missing_ring_{label}_no_missing_fraction"] = fraction(
            no_missing,
            eligible,
        )
        summary[
            f"missing_ring_{label}_no_missing_sse_fraction"
        ] = weighted_fraction(
            no_missing,
            eligible,
            group["episode_sse"],
        )
    change = group["missing_near_minus_far_fraction"]
    eligible = change.notna()
    positive = eligible & (change > 0.0)
    negative = eligible & (change < 0.0)
    unchanged = eligible & np.isclose(change, 0.0)
    summary.update(
        {
            "missing_near_minus_far_eligible_episodes": int(
                eligible.sum()
            ),
            "missing_near_minus_far_mean": float(change.mean()),
            "missing_near_minus_far_median": float(change.median()),
            "missing_near_minus_far_positive_fraction": fraction(
                positive,
                eligible,
            ),
            "missing_near_minus_far_positive_sse_fraction": (
                weighted_fraction(
                    positive,
                    eligible,
                    group["episode_sse"],
                )
            ),
            "missing_near_minus_far_negative_fraction": fraction(
                negative,
                eligible,
            ),
            "missing_near_minus_far_negative_sse_fraction": (
                weighted_fraction(
                    negative,
                    eligible,
                    group["episode_sse"],
                )
            ),
            "missing_near_minus_far_unchanged_fraction": fraction(
                unchanged,
                eligible,
            ),
            "missing_near_minus_far_unchanged_sse_fraction": (
                weighted_fraction(
                    unchanged,
                    eligible,
                    group["episode_sse"],
                )
            ),
            "missing_change_vs_transition_crescendo_spearman": (
                safe_spearman(
                    change,
                    group[
                        "truth_pre_crescendo_near_minus_far_nll"
                    ],
                )
            ),
            "missing_change_vs_posterior_mean_rate_error_growth_spearman": (
                safe_spearman(
                    change,
                    group[
                        "posterior_mean_near_minus_far_rate_error_abs"
                    ],
                )
            ),
            "missing_change_vs_viterbi_rate_error_growth_spearman": (
                safe_spearman(
                    change,
                    group[
                        "global_viterbi_near_minus_far_rate_error_abs"
                    ],
                )
            ),
            "missing_change_vs_pre128_error_slope_spearman": (
                safe_spearman(
                    change,
                    group["pre128_error_slope_ft_per_row"].abs(),
                )
            ),
            "missing_change_vs_rmse_spearman": safe_spearman(
                change,
                group["rmse_ft"],
            ),
        }
    )
    for threshold in THRESHOLDS:
        key = str(threshold).replace(".", "p")
        strong = eligible & (change >= threshold)
        summary[
            f"missing_near_minus_far_ge_{key}_fraction"
        ] = fraction(strong, eligible)
        summary[
            f"missing_near_minus_far_ge_{key}_sse_fraction"
        ] = weighted_fraction(
            strong,
            eligible,
            group["episode_sse"],
        )
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    episodes_path = resolve(root, args.episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    result = add_missingness(episodes)
    if len(result) != 638:
        raise ValueError(f"unexpected episode count: {len(result)}")
    if result["well"].nunique() != 450:
        raise ValueError(
            f"unexpected well count: {result['well'].nunique()}"
        )
    if int(result["rows"].sum()) != 807_710:
        raise ValueError(
            f"unexpected episode rows: {int(result['rows'].sum())}"
        )
    result.to_csv(
        output / "episode_onset_missingness_metrics.csv",
        index=False,
    )
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
        output / "cause_onset_missingness_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(result)),
            "wells": int(result["well"].nunique()),
            "episode_rows": int(result["rows"].sum()),
        },
        "source_sha256": {
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(result, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "observed_rows_within_total_rows": True,
            "rings_are_non_overlapping": True,
            "missingness_uses_raw_observed_gr_counts": True,
            "interpretation": (
                "This readout tests temporal association of raw-GR "
                "missingness with a prediction-error-defined onset. It "
                "does not isolate imputation quality or hidden messages."
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
