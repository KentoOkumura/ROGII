#!/usr/bin/env python3
"""Sensitivity audit for exp209 persistent-offset episode definitions.

This truth-late readout changes only the absolute-error threshold and minimum
run length used to describe persistent offsets.  It does not generate or
select predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_offset_definition_sensitivity_20260726"
)
USECOLS = [
    "well",
    "row_idx",
    "true_tvt_readout_only",
    "posterior_mean",
]
ERROR_THRESHOLDS_FT = (5.0, 10.0, 15.0, 20.0)
MINIMUM_RUN_ROWS = (64, 128, 256)
CANONICAL_EPISODES = 638
CANONICAL_EPISODE_ROWS = 807_710
CANONICAL_SSE_FRACTION = 0.919880


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    if len(mask) == 0:
        return []
    changes = np.flatnonzero(
        np.concatenate([[True], mask[1:] != mask[:-1]])
    )
    ends = np.concatenate([changes[1:], [len(mask)]])
    return [
        (int(start), int(end))
        for start, end in zip(changes, ends, strict=True)
        if bool(mask[start])
    ]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    candidates = root / args.candidates
    if not candidates.exists():
        raise FileNotFoundError(candidates)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(candidates, usecols=USECOLS)
    frame["well"] = frame["well"].astype(str)
    if not frame["well"].is_monotonic_increasing:
        raise ValueError("candidate artifact must be sorted by well")
    error = (
        frame["posterior_mean"].to_numpy(np.float64)
        - frame["true_tvt_readout_only"].to_numpy(np.float64)
    )
    if not np.isfinite(error).all():
        raise ValueError("candidate error contains non-finite values")
    total_sse = float(np.sum(error**2))

    groups = [
        (
            str(well),
            positions.to_numpy(np.int64),
        )
        for well, positions in frame.groupby(
            "well",
            sort=False,
        ).groups.items()
    ]
    rows: list[dict[str, Any]] = []
    for threshold in ERROR_THRESHOLDS_FT:
        for minimum_rows in MINIMUM_RUN_ROWS:
            episode_count = 0
            episode_wells: set[str] = set()
            episode_rows = 0
            episode_sse = 0.0
            sign_consistency: list[float] = []
            for well, positions in groups:
                well_error = error[positions]
                mask = np.abs(well_error) > threshold
                for start, end in true_runs(mask):
                    run_rows = end - start
                    if run_rows < minimum_rows:
                        continue
                    run_error = well_error[start:end]
                    episode_count += 1
                    episode_wells.add(well)
                    episode_rows += run_rows
                    episode_sse += float(np.sum(run_error**2))
                    positive_fraction = float(
                        np.mean(run_error > 0.0)
                    )
                    sign_consistency.append(
                        max(
                            positive_fraction,
                            1.0 - positive_fraction,
                        )
                    )
            non_episode_rows = len(frame) - episode_rows
            non_episode_sse = total_sse - episode_sse
            rows.append(
                {
                    "error_threshold_ft": threshold,
                    "minimum_run_rows": minimum_rows,
                    "episodes": episode_count,
                    "episode_wells": len(episode_wells),
                    "episode_rows": episode_rows,
                    "episode_row_fraction": (
                        episode_rows / len(frame)
                    ),
                    "episode_sse_fraction": (
                        episode_sse / total_sse
                    ),
                    "episode_pooled_rmse_ft": float(
                        np.sqrt(episode_sse / episode_rows)
                    ),
                    "non_episode_pooled_rmse_ft": float(
                        np.sqrt(
                            non_episode_sse / non_episode_rows
                        )
                    ),
                    "episode_sign_consistency_median": float(
                        np.median(sign_consistency)
                    ),
                }
            )

    results = pd.DataFrame(rows)
    canonical = results[
        (results["error_threshold_ft"] == 10.0)
        & (results["minimum_run_rows"] == 128)
    ].iloc[0]
    if int(canonical["episodes"]) != CANONICAL_EPISODES:
        raise ValueError("canonical episode-count guard failed")
    if int(canonical["episode_rows"]) != CANONICAL_EPISODE_ROWS:
        raise ValueError("canonical episode-row guard failed")
    if not np.isclose(
        canonical["episode_sse_fraction"],
        CANONICAL_SSE_FRACTION,
        rtol=0.0,
        atol=5e-7,
    ):
        raise ValueError("canonical episode-SSE guard failed")

    summary = {
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "thresholds_ft": list(ERROR_THRESHOLDS_FT),
        "minimum_run_rows": list(MINIMUM_RUN_ROWS),
        "configurations": int(len(results)),
        "canonical": canonical.to_dict(),
        "interpretation_guard": (
            "Truth-late definition sensitivity only; no prediction, "
            "parameter selection, or target-free routing is produced."
        ),
    }
    results.to_csv(output / "definition_sensitivity.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
