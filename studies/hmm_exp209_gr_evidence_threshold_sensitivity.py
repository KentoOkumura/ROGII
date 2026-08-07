#!/usr/bin/env python3
"""Sensitivity of exp209 episode GR attribution to the NLL threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_gr_evidence_threshold_sensitivity_20260726"
)
THRESHOLDS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    episodes_path = root / args.episodes
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(
        episodes_path,
        usecols=[
            "episode_id",
            "episode_sse",
            "observed_total_truth_minus_candidate_nll",
        ],
    )
    if len(episodes) != 638:
        raise ValueError("persistent episode-count guard failed")
    evidence = episodes[
        "observed_total_truth_minus_candidate_nll"
    ].to_numpy(np.float64)
    sse = episodes["episode_sse"].to_numpy(np.float64)
    if not np.isfinite(evidence).all() or not np.isfinite(sse).all():
        raise ValueError("episode evidence contains non-finite values")
    total_sse = float(np.sum(sse))

    rows = []
    for threshold in THRESHOLDS:
        candidate = evidence > threshold
        truth = evidence < -threshold
        tie = ~(candidate | truth)
        candidate_fraction = float(np.sum(sse[candidate]) / total_sse)
        truth_fraction = float(np.sum(sse[truth]) / total_sse)
        rows.append(
            {
                "absolute_nll_threshold": threshold,
                "candidate_strong_episodes": int(candidate.sum()),
                "candidate_strong_episode_sse_fraction": candidate_fraction,
                "truth_strong_episodes": int(truth.sum()),
                "truth_strong_episode_sse_fraction": truth_fraction,
                "near_tie_episodes": int(tie.sum()),
                "near_tie_episode_sse_fraction": float(
                    np.sum(sse[tie]) / total_sse
                ),
                "truth_minus_candidate_sse_fraction": (
                    truth_fraction - candidate_fraction
                ),
            }
        )
    results = pd.DataFrame(rows)
    if not (
        results["truth_minus_candidate_sse_fraction"] > 0.0
    ).all():
        raise ValueError("truth/candidate ordering changed with threshold")

    canonical = results[
        results["absolute_nll_threshold"] == 5.0
    ].iloc[0]
    if (
        int(canonical["candidate_strong_episodes"]) != 180
        or int(canonical["truth_strong_episodes"]) != 278
        or int(canonical["near_tie_episodes"]) != 180
    ):
        raise ValueError("canonical evidence-class guard failed")
    if not np.isclose(
        canonical["candidate_strong_episode_sse_fraction"],
        0.33415808436656397,
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        canonical["truth_strong_episode_sse_fraction"],
        0.5735360405342573,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("canonical evidence-SSE guard failed")

    summary = {
        "episodes": int(len(episodes)),
        "thresholds": list(THRESHOLDS),
        "truth_exceeds_candidate_at_every_threshold": True,
        "truth_minus_candidate_sse_fraction_range": {
            "minimum": float(
                results["truth_minus_candidate_sse_fraction"].min()
            ),
            "maximum": float(
                results["truth_minus_candidate_sse_fraction"].max()
            ),
        },
        "interpretation_guard": (
            "Truth-late observed-GR NLL attribution sensitivity only; "
            "this does not rerun the HMM or select an emission threshold."
        ),
    }
    results.to_csv(output / "threshold_sensitivity.csv", index=False)
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
