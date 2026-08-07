#!/usr/bin/env python3
"""Read out posterior geometry before persistent-offset onset.

The exp270 posterior summaries and paths are frozen before truth is used.
For each exp209 persistent-offset episode, this script measures posterior
width, marginal concentration, and decoder disagreement in the same
non-overlapping pre-onset rings as the transition/GR timing audits.  It does
not rerun the HMM or generate a prediction.
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
)
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_onset_transition_gr_timing_20260726/"
    "episode_onset_timing.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_posterior_geometry_timing_20260726"
)
RINGS = (
    ("000_016", 0, 16),
    ("016_064", 16, 64),
    ("064_128", 64, 128),
    ("128_256", 128, 256),
    ("256_512", 256, 512),
)
GEOMETRY_NAMES = (
    "posterior_std",
    "marginal_mode_mass",
    "marginal_mode_gap",
    "mean_to_map_abs",
    "mean_to_viterbi_abs",
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
    record[f"posterior_{label}_rows"] = int(rows)
    for name, values in arrays.items():
        part = values[start:end]
        record[f"posterior_{label}_{name}_mean"] = (
            float(part.mean()) if rows else None
        )
        record[f"posterior_{label}_{name}_p90"] = (
            float(np.quantile(part, 0.9)) if rows else None
        )


def add_changes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in GEOMETRY_NAMES:
        result[f"posterior_near_minus_far_{name}"] = (
            result[f"posterior_ring_000_016_{name}_mean"]
            - result[f"posterior_ring_256_512_{name}_mean"]
        )
        result[f"posterior_episode_minus_near_{name}"] = (
            result[f"posterior_episode_{name}_mean"]
            - result[f"posterior_ring_000_016_{name}_mean"]
        )
    result["posterior_std_broadens_near"] = (
        result["posterior_near_minus_far_posterior_std"] > 0.0
    )
    result["posterior_mode_mass_drops_near"] = (
        result["posterior_near_minus_far_marginal_mode_mass"] < 0.0
    )
    return result


def pooled_interval_mean(
    group: pd.DataFrame,
    label: str,
    name: str,
) -> float | None:
    value = f"posterior_{label}_{name}_mean"
    rows = f"posterior_{label}_rows"
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
        for name in GEOMETRY_NAMES:
            column = f"posterior_{label}_{name}_mean"
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
    for name in GEOMETRY_NAMES:
        change = f"posterior_near_minus_far_{name}"
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
        result[f"{change}_vs_gr_crescendo_spearman"] = safe_spearman(
            group[change],
            group[
                "observed_gr_near_minus_far_mean_"
                "candidate_advantage_nll"
            ],
        )
    std_change = "posterior_near_minus_far_posterior_std"
    result_column = std_change
    eligible = group[result_column].notna()
    broadens = eligible & group["posterior_std_broadens_near"]
    near_gr_truth = (
        group["observed_gr_near_far_eligible"]
        & ~group["observed_gr_near_favors_candidate"]
    )
    transition_up = group["transition_pressure_increases"]
    result.update(
        {
            "posterior_std_broadens_near_fraction_eligible": (
                fraction_within(broadens, eligible)
            ),
            "posterior_std_broadens_near_sse_fraction_eligible": (
                weighted_fraction_within(
                    broadens,
                    eligible,
                    group["episode_sse"],
                )
            ),
            "posterior_std_broadens_and_near_gr_truth_fraction_eligible": (
                fraction_within(
                    broadens & near_gr_truth,
                    eligible,
                )
            ),
            "posterior_std_broadens_and_near_gr_truth_"
            "sse_fraction_eligible": weighted_fraction_within(
                broadens & near_gr_truth,
                eligible,
                group["episode_sse"],
            ),
            "posterior_std_broadens_and_transition_up_fraction_eligible": (
                fraction_within(
                    broadens & transition_up,
                    eligible,
                )
            ),
            "posterior_std_broadens_and_transition_up_"
            "sse_fraction_eligible": weighted_fraction_within(
                broadens & transition_up,
                eligible,
                group["episode_sse"],
            ),
            "posterior_std_change_vs_pre128_error_slope_spearman": (
                safe_spearman(
                    group[std_change],
                    group["pre128_error_slope_ft_per_row"].abs(),
                )
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
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
    for well, frame in iter_well_frames(candidates_path, args.chunksize):
        specs = episode_lookup.get(well, [])
        if not specs:
            continue
        seen_wells.add(well)
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        posterior_mean = pd.to_numeric(
            frame["posterior_mean"],
            errors="raise",
        ).to_numpy(np.float64)
        arrays = {
            "posterior_std": pd.to_numeric(
                frame["posterior_std"],
                errors="raise",
            ).to_numpy(np.float64),
            "marginal_mode_mass": pd.to_numeric(
                frame["marginal_mode_mass"],
                errors="raise",
            ).to_numpy(np.float64),
            "marginal_mode_gap": pd.to_numeric(
                frame["marginal_mode_gap"],
                errors="raise",
            ).to_numpy(np.float64),
            "mean_to_map_abs": np.abs(
                posterior_mean
                - pd.to_numeric(
                    frame["marginal_map"],
                    errors="raise",
                ).to_numpy(np.float64)
            ),
            "mean_to_viterbi_abs": np.abs(
                posterior_mean
                - pd.to_numeric(
                    frame["topk_path_1"],
                    errors="raise",
                ).to_numpy(np.float64)
            ),
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
            add_interval(
                record,
                "episode",
                start,
                end,
                arrays,
            )
            for label, inner, outer in RINGS:
                ring_start = max(0, start - outer)
                ring_end = max(0, start - inner)
                expected_rows = int(
                    spec[f"truth_ring_{label}_rows"]
                )
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

    result = add_changes(pd.DataFrame(records))
    if len(result) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(result)} != {len(episodes)}"
        )
    result.to_csv(
        output / "episode_posterior_geometry.csv",
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
        output / "cause_posterior_geometry_summary.csv",
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
            "posterior": (
                "All geometry values are frozen exp270 summaries of the "
                "exp209 smoothed position posterior and legal Viterbi path."
            ),
            "interpretation": (
                "Error-defined onset and truth/cause labels are attached "
                "only for this readout. Smoothed geometry cannot identify "
                "whether broadening entered through alpha or beta."
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
