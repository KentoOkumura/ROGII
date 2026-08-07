#!/usr/bin/env python3
"""Compare transition pressure and raw-GR evidence before offset onset.

This is a truth-late readout over frozen exp209/exp270 paths.  It does not
rerun the HMM or generate predictions.  For the same non-overlapping
pre-onset rings used by the transition-grammar audit, it reconstructs the
exp209 raw-observation emission NLL at truth, posterior mean, and global
Viterbi TVT.  Positive truth-minus-candidate NLL means raw GR favors the
candidate path.
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
    emission_nll,
    iter_well_frames,
    load_well_inputs,
)
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_truth_grammar_temporal_readout_20260726/"
    "episode_temporal_grammar_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_onset_transition_gr_timing_20260726"
)
RINGS = (
    ("000_016", 0, 16),
    ("016_064", 16, 64),
    ("064_128", 64, 128),
    ("128_256", 128, 256),
    ("256_512", 256, 512),
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


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


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


def add_emission_ring(
    record: dict[str, Any],
    label: str,
    start: int,
    end: int,
    observed: np.ndarray,
    mean_advantage: np.ndarray,
    viterbi_advantage: np.ndarray,
) -> None:
    observed_ring = observed[start:end]
    observed_rows = int(observed_ring.sum())
    mean_ring = mean_advantage[start:end][observed_ring]
    viterbi_ring = viterbi_advantage[start:end][observed_ring]
    prefix = f"observed_gr_ring_{label}"
    record[f"{prefix}_rows"] = observed_rows
    record[f"{prefix}_mean_candidate_advantage_nll"] = (
        float(mean_ring.mean()) if observed_rows else None
    )
    record[f"{prefix}_total_candidate_advantage_nll"] = (
        float(mean_ring.sum()) if observed_rows else None
    )
    record[f"{prefix}_candidate_favored_row_fraction"] = (
        float((mean_ring > 0.0).mean()) if observed_rows else None
    )
    record[f"{prefix}_mean_viterbi_advantage_nll"] = (
        float(viterbi_ring.mean()) if observed_rows else None
    )
    record[f"{prefix}_total_viterbi_advantage_nll"] = (
        float(viterbi_ring.sum()) if observed_rows else None
    )
    record[f"{prefix}_viterbi_favored_row_fraction"] = (
        float((viterbi_ring > 0.0).mean()) if observed_rows else None
    )


def add_timing_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    near = "observed_gr_ring_000_016"
    far = "observed_gr_ring_256_512"
    result["observed_gr_near_minus_far_mean_candidate_advantage_nll"] = (
        result[f"{near}_mean_candidate_advantage_nll"]
        - result[f"{far}_mean_candidate_advantage_nll"]
    )
    near_total = result[f"{near}_total_candidate_advantage_nll"]
    far_total = result[f"{far}_total_candidate_advantage_nll"]
    result["observed_gr_near_favors_candidate"] = near_total > 0.0
    result["observed_gr_far_favors_candidate"] = far_total > 0.0
    result["observed_gr_near_far_eligible"] = (
        near_total.notna() & far_total.notna()
    )
    grammar_change = result[
        "truth_pre_crescendo_near_minus_far_nll"
    ]
    result["transition_pressure_increases"] = grammar_change > 0.0
    result["timing_transition_up_gr_truth_both"] = (
        result["observed_gr_near_far_eligible"]
        & result["transition_pressure_increases"]
        & ~result["observed_gr_far_favors_candidate"]
        & ~result["observed_gr_near_favors_candidate"]
    )
    result["timing_transition_up_gr_switches_to_truth"] = (
        result["observed_gr_near_far_eligible"]
        & result["transition_pressure_increases"]
        & result["observed_gr_far_favors_candidate"]
        & ~result["observed_gr_near_favors_candidate"]
    )
    result["timing_transition_up_gr_switches_to_candidate"] = (
        result["observed_gr_near_far_eligible"]
        & result["transition_pressure_increases"]
        & ~result["observed_gr_far_favors_candidate"]
        & result["observed_gr_near_favors_candidate"]
    )
    result["timing_transition_up_gr_candidate_both"] = (
        result["observed_gr_near_far_eligible"]
        & result["transition_pressure_increases"]
        & result["observed_gr_far_favors_candidate"]
        & result["observed_gr_near_favors_candidate"]
    )
    return result


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
    for label, _, _ in RINGS:
        grammar = f"truth_ring_{label}_transition_nll_mean"
        grammar_rows = f"truth_ring_{label}_rows"
        gr = f"observed_gr_ring_{label}"
        observed_rows = group[f"{gr}_rows"]
        grammar_valid = group[grammar].notna() & (group[grammar_rows] > 0)
        observed_valid = (
            group[f"{gr}_mean_candidate_advantage_nll"].notna()
            & (observed_rows > 0)
        )
        grammar_sum = float(
            (
                group.loc[grammar_valid, grammar]
                * group.loc[grammar_valid, grammar_rows]
            ).sum()
        )
        grammar_count = int(
            group.loc[grammar_valid, grammar_rows].sum()
        )
        mean_advantage_sum = float(
            group.loc[
                observed_valid,
                f"{gr}_total_candidate_advantage_nll",
            ].sum()
        )
        viterbi_advantage_sum = float(
            group.loc[
                observed_valid,
                f"{gr}_total_viterbi_advantage_nll",
            ].sum()
        )
        observed_count = int(observed_rows.loc[observed_valid].sum())
        candidate_favored_rows = float(
            (
                group.loc[
                    observed_valid,
                    f"{gr}_candidate_favored_row_fraction",
                ]
                * observed_rows.loc[observed_valid]
            ).sum()
        )
        candidate_episode = (
            group[f"{gr}_total_candidate_advantage_nll"] > 0.0
        )
        result.update(
            {
                f"{grammar}_pooled": (
                    grammar_sum / grammar_count
                    if grammar_count
                    else None
                ),
                f"{gr}_observed_rows": observed_count,
                f"{gr}_mean_candidate_advantage_nll_pooled": (
                    mean_advantage_sum / observed_count
                    if observed_count
                    else None
                ),
                f"{gr}_mean_viterbi_advantage_nll_pooled": (
                    viterbi_advantage_sum / observed_count
                    if observed_count
                    else None
                ),
                f"{gr}_candidate_favored_row_fraction_pooled": (
                    candidate_favored_rows / observed_count
                    if observed_count
                    else None
                ),
                f"{gr}_candidate_favored_episode_fraction": (
                    fraction_within(
                        candidate_episode,
                        observed_valid,
                    )
                ),
                f"{gr}_candidate_favored_episode_sse_fraction_eligible": (
                    weighted_fraction_within(
                        candidate_episode,
                        observed_valid,
                        group["episode_sse"],
                    )
                ),
                f"{grammar}_vs_gr_advantage_spearman": safe_spearman(
                    group[grammar],
                    group[f"{gr}_mean_candidate_advantage_nll"],
                ),
            }
        )
    timing_eligible = group["observed_gr_near_far_eligible"]
    timing_masks = (
        "timing_transition_up_gr_truth_both",
        "timing_transition_up_gr_switches_to_truth",
        "timing_transition_up_gr_switches_to_candidate",
        "timing_transition_up_gr_candidate_both",
    )
    result["timing_eligible_episodes"] = int(timing_eligible.sum())
    for column in timing_masks:
        result[f"{column}_fraction_eligible"] = fraction_within(
            group[column],
            timing_eligible,
        )
        result[f"{column}_sse_fraction_eligible"] = (
            weighted_fraction_within(
                group[column],
                timing_eligible,
                group["episode_sse"],
            )
        )
    result.update(
        {
            "transition_crescendo_vs_gr_crescendo_spearman": (
                safe_spearman(
                    group[
                        "truth_pre_crescendo_near_minus_far_nll"
                    ],
                    group[
                        "observed_gr_near_minus_far_mean_"
                        "candidate_advantage_nll"
                    ],
                )
            ),
            "gr_crescendo_vs_rmse_spearman": safe_spearman(
                group[
                    "observed_gr_near_minus_far_mean_"
                    "candidate_advantage_nll"
                ],
                group["rmse_ft"],
            ),
            "gr_crescendo_vs_pre128_error_slope_spearman": (
                safe_spearman(
                    group[
                        "observed_gr_near_minus_far_mean_"
                        "candidate_advantage_nll"
                    ],
                    group["pre128_error_slope_ft_per_row"].abs(),
                )
            ),
            "near_gr_advantage_vs_rmse_spearman": safe_spearman(
                group[
                    "observed_gr_ring_000_016_"
                    "mean_candidate_advantage_nll"
                ],
                group["rmse_ft"],
            ),
        }
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
    output_records: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    for well, frame in iter_well_frames(candidates_path, args.chunksize):
        specs = episode_lookup.get(well, [])
        if not specs:
            continue
        seen_wells.add(well)
        (
            horizontal,
            typewell_tvt,
            typewell_gr,
            sigma,
            _,
            _,
            _,
            _,
        ) = load_well_inputs(train_dir, well)
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        raw_gr = pd.to_numeric(
            horizontal["GR"],
            errors="coerce",
        ).to_numpy(np.float64)[row_index]
        observed = np.isfinite(raw_gr)
        truth = pd.to_numeric(
            frame["true_tvt_readout_only"],
            errors="raise",
        ).to_numpy(np.float64)
        mean = pd.to_numeric(
            frame["posterior_mean"],
            errors="raise",
        ).to_numpy(np.float64)
        viterbi = pd.to_numeric(
            frame["topk_path_1"],
            errors="raise",
        ).to_numpy(np.float64)
        truth_nll = emission_nll(
            raw_gr,
            truth,
            typewell_tvt,
            typewell_gr,
            sigma,
        )
        mean_advantage = truth_nll - emission_nll(
            raw_gr,
            mean,
            typewell_tvt,
            typewell_gr,
            sigma,
        )
        viterbi_advantage = truth_nll - emission_nll(
            raw_gr,
            viterbi,
            typewell_tvt,
            typewell_gr,
            sigma,
        )
        for spec in specs:
            record = dict(spec)
            start = int(record["episode_start_suffix_index"])
            expected_start = int(
                np.searchsorted(
                    row_index,
                    int(record["start_row_idx"]),
                )
            )
            if start != expected_start:
                raise ValueError(
                    f"{record['episode_id']}: episode start mismatch"
                )
            for label, inner, outer in RINGS:
                ring_start = max(0, start - outer)
                ring_end = max(0, start - inner)
                expected_rows = int(
                    record[f"truth_ring_{label}_rows"]
                )
                if ring_end - ring_start != expected_rows:
                    raise ValueError(
                        f"{record['episode_id']}: {label} row mismatch"
                    )
                add_emission_ring(
                    record,
                    label,
                    ring_start,
                    ring_end,
                    observed,
                    mean_advantage,
                    viterbi_advantage,
                )
            output_records.append(record)

    if seen_wells != set(episode_lookup):
        missing = sorted(set(episode_lookup) - seen_wells)
        raise ValueError(f"episode wells missing from candidates: {missing[:5]}")
    timing = add_timing_metrics(pd.DataFrame(output_records))
    if len(timing) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(timing)} != {len(episodes)}"
        )
    timing.to_csv(output / "episode_onset_timing.csv", index=False)

    total_sse = float(timing["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in timing.groupby("cause_bucket", sort=True):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_group(group, total_sse),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_onset_timing_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(timing)),
            "wells": int(timing["well"].nunique()),
            "rings_rows_before_onset": [
                {"label": label, "inner": inner, "outer": outer}
                for label, inner, outer in RINGS
            ],
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(candidates_path),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(timing, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "emission": (
                "Only raw observed GR rows are scored with the frozen exp209 "
                "Gaussian emission and cap."
            ),
            "interpretation": (
                "The posterior-mean and Viterbi paths and error-defined onset "
                "are frozen before this truth-late readout. Temporal ordering "
                "is descriptive and is not an intervention on alpha or beta."
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
