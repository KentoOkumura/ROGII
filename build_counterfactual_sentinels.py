"""Freeze exp410 counterfactual sentinel wells from the strict full merge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT = "exp410_likpf_particle_resampling_basin_audit"
DEFAULT_EPISODES = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/"
    "full_merged/exp410_likpf_particle_resampling_basin_audit_episode_summary.csv"
)
DEFAULT_TARGET_WELLS = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_target_wells.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_counterfactual_sentinel_wells.csv"
)
EXPECTED_EPISODES = 839
EXPECTED_WELLS = 496
MAX_SENTINELS = 12
SHARD_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--target-wells", type=Path, default=DEFAULT_TARGET_WELLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def frame_content_sha(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values("selection_order", kind="stable")
    payload = normalized.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def select_sentinels(episodes: pd.DataFrame) -> pd.DataFrame:
    cause_totals = (
        episodes.groupby("cause", sort=True)["episode_sse"]
        .sum()
        .sort_values(ascending=False, kind="stable")
    )
    selected: list[dict[str, Any]] = []
    selected_wells: set[str] = set()

    # First cover every exclusive cause with its largest-SSE still-unused well.
    for cause in cause_totals.index:
        candidates = episodes.loc[episodes["cause"] == cause].sort_values(
            ["episode_sse", "episode_id"],
            ascending=[False, True],
            kind="stable",
        )
        candidate = next(
            (
                row
                for row in candidates.itertuples(index=False)
                if str(row.well) not in selected_wells
            ),
            None,
        )
        if candidate is None:
            continue
        selected_wells.add(str(candidate.well))
        selected.append(
            {
                "well": str(candidate.well),
                "representative_episode_id": str(candidate.episode_id),
                "representative_cause": str(candidate.cause),
                "representative_episode_sse": float(candidate.episode_sse),
                "selection_phase": "exclusive_cause_max_sse",
            }
        )
        if len(selected) == MAX_SENTINELS:
            break

    # Then fill the fixed budget by global episode SSE without duplicate wells.
    if len(selected) < MAX_SENTINELS:
        candidates = episodes.sort_values(
            ["episode_sse", "episode_id"],
            ascending=[False, True],
            kind="stable",
        )
        for candidate in candidates.itertuples(index=False):
            well = str(candidate.well)
            if well in selected_wells:
                continue
            selected_wells.add(well)
            selected.append(
                {
                    "well": well,
                    "representative_episode_id": str(candidate.episode_id),
                    "representative_cause": str(candidate.cause),
                    "representative_episode_sse": float(candidate.episode_sse),
                    "selection_phase": "global_episode_sse_fill",
                }
            )
            if len(selected) == MAX_SENTINELS:
                break

    result = pd.DataFrame(selected)
    result.insert(0, "selection_order", range(len(result)))
    return result


def assign_lpt_shards(frame: pd.DataFrame) -> pd.DataFrame:
    loads = [0] * SHARD_COUNT
    assignments: dict[str, int] = {}
    ordered = frame.sort_values(
        ["suffix_rows", "well"],
        ascending=[False, True],
        kind="stable",
    )
    for row in ordered.itertuples(index=False):
        shard = min(range(SHARD_COUNT), key=lambda index: (loads[index], index))
        assignments[str(row.well)] = shard
        loads[shard] += int(row.suffix_rows)
    result = frame.copy()
    result["counterfactual_shard_index"] = (
        result["well"].map(assignments).astype("int8")
    )
    return result


def main() -> None:
    args = parse_args()
    episodes = pd.read_csv(
        args.episodes, dtype={"episode_id": str, "well": str}
    )
    target_wells = pd.read_csv(args.target_wells, dtype={"well": str})
    if len(episodes) != EXPECTED_EPISODES:
        raise RuntimeError(f"episode count changed: {len(episodes)}")
    if len(target_wells) != EXPECTED_WELLS:
        raise RuntimeError(f"target well count changed: {len(target_wells)}")
    required_episode_columns = {
        "episode_id",
        "well",
        "cause",
        "episode_sse",
    }
    if not required_episode_columns.issubset(episodes.columns):
        raise RuntimeError("strict merged episode columns missing")
    required_well_columns = (
        "well",
        "suffix_rows",
        "episodes",
        "episode_rows",
        "episode_sse",
    )
    if not set(required_well_columns).issubset(target_wells.columns):
        raise RuntimeError("target-well asset columns missing")

    selected = select_sentinels(episodes)
    if len(selected) != MAX_SENTINELS or selected["well"].duplicated().any():
        raise RuntimeError("sentinel selection did not produce 12 unique wells")
    selected = selected.merge(
        target_wells[list(required_well_columns)],
        on="well",
        how="left",
        validate="one_to_one",
    )
    if selected["suffix_rows"].isna().any():
        raise RuntimeError("sentinel well missing from fixed target asset")
    selected = assign_lpt_shards(selected).sort_values(
        "selection_order", kind="stable"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    manifest = {
        "experiment": EXPERIMENT,
        "selection": (
            "top unused well by episode SSE for each exclusive cause ordered by "
            "cause SSE, then unique wells by global episode SSE"
        ),
        "max_sentinels": MAX_SENTINELS,
        "shard_count": SHARD_COUNT,
        "counts": {
            "wells": int(len(selected)),
            "suffix_rows": int(selected["suffix_rows"].sum()),
            "episodes": int(selected["episodes"].sum()),
            "episode_rows": int(selected["episode_rows"].sum()),
        },
        "source": {
            "merged_episode_summary": str(args.episodes),
            "merged_episode_summary_sha256": sha256_path(args.episodes),
            "target_wells": str(args.target_wells),
            "target_wells_sha256": sha256_path(args.target_wells),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256_path(args.output),
            "content_sha256": frame_content_sha(selected),
        },
        "shard_suffix_rows": {
            str(shard): int(
                selected.loc[
                    selected["counterfactual_shard_index"] == shard,
                    "suffix_rows",
                ].sum()
            )
            for shard in range(SHARD_COUNT)
        },
    }
    manifest_path = args.output.with_name(
        "pf_counterfactual_sentinel_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
