"""Freeze PF-specific persistent-offset episodes from the saved exp072 prediction."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "experiments/exp235_fixed_lag_particle_smoother_pf/artifacts/"
    "lag64_merged_v3/exp235_fixed_lag_particle_smoother_pf_merged_row_candidates.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets"
)
USECOLS = [
    "well",
    "row_idx",
    "id",
    "true_tvt",
    "last_known_tvt",
    "last_known_md",
    "md_since",
    "exp072_likpf_mean",
]
ERROR_THRESHOLD_FT = 10.0
MIN_RUN_ROWS = 128
SHARD_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def sha256_file(path: Path, *, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_frame_content(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for well, group in frame.groupby("well", sort=False):
        well_bytes = str(well).encode("utf-8")
        digest.update(len(well_bytes).to_bytes(4, "little"))
        digest.update(well_bytes)
        digest.update(
            np.ascontiguousarray(group["row_idx"].to_numpy(np.int64)).tobytes()
        )
        digest.update(
            np.ascontiguousarray(
                group["exp072_likpf_mean"].to_numpy(np.float32)
            ).tobytes()
        )
    return digest.hexdigest()


def iter_well_frames(path: Path, chunksize: int) -> Iterator[tuple[str, pd.DataFrame]]:
    carry: pd.DataFrame | None = None
    completed_wells: set[str] = set()
    for chunk in pd.read_csv(path, usecols=USECOLS, chunksize=chunksize):
        chunk["well"] = chunk["well"].astype(str)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        if chunk.empty:
            continue
        final_well = str(chunk["well"].iloc[-1])
        complete = chunk.loc[chunk["well"] != final_well]
        carry = chunk.loc[chunk["well"] == final_well].copy()
        for well, group in complete.groupby("well", sort=False):
            well = str(well)
            if well in completed_wells:
                raise ValueError(f"well is not contiguous in source artifact: {well}")
            completed_wells.add(well)
            yield well, group.reset_index(drop=True)
    if carry is not None and not carry.empty:
        well = str(carry["well"].iloc[0])
        if well in completed_wells:
            raise ValueError(f"well is not contiguous in source artifact: {well}")
        yield well, carry.reset_index(drop=True)


def contiguous_runs(mask: np.ndarray, min_rows: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in changes.reshape(-1, 2)
        if int(end - start) >= int(min_rows)
    ]


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=np.float64)
    x -= x.mean()
    centered = values.astype(np.float64) - float(np.mean(values))
    denominator = float(np.sum(x * x))
    return float(np.sum(x * centered) / denominator) if denominator > 0.0 else 0.0


def build_assets(source: Path, output: Path, chunksize: int) -> dict[str, object]:
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(source)

    episodes: list[dict[str, object]] = []
    wells: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []
    total_rows = 0
    total_wells = 0

    for well, frame in iter_well_frames(source, chunksize):
        total_wells += 1
        total_rows += len(frame)
        frame = frame.sort_values("row_idx", kind="stable").reset_index(drop=True)
        row_idx = frame["row_idx"].to_numpy(np.int64)
        if len(np.unique(row_idx)) != len(row_idx):
            raise ValueError(f"{well}: duplicate row_idx")
        if len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1):
            raise ValueError(f"{well}: evaluation row_idx is not contiguous")
        truth = (
            pd.to_numeric(frame["true_tvt"], errors="raise")
            .to_numpy(np.float32)
            .astype(np.float64)
        )
        frame["true_tvt"] = truth.astype(np.float32)
        prediction = pd.to_numeric(
            frame["exp072_likpf_mean"], errors="raise"
        ).to_numpy(np.float32).astype(np.float64)
        frame["exp072_likpf_mean"] = prediction.astype(np.float32)
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError(f"{well}: non-finite truth or prediction")
        error = prediction - truth
        runs = contiguous_runs(np.abs(error) > ERROR_THRESHOLD_FT, MIN_RUN_ROWS)

        if runs:
            all_frames.append(
                frame[["well", "row_idx", "exp072_likpf_mean"]].copy()
            )
        well_episode_rows = 0
        well_episode_sse = 0.0
        for episode_index, (start, end) in enumerate(runs):
            episode_error = error[start:end]
            pre_start = max(0, start - 128)
            pre_error = error[pre_start:start]
            preceding_within5 = np.flatnonzero(np.abs(error[:start]) <= 5.0)
            following_within5 = np.flatnonzero(np.abs(error[end:]) <= 5.0)
            last_within5 = (
                int(preceding_within5[-1]) if preceding_within5.size else None
            )
            next_within5 = (
                int(end + following_within5[0])
                if following_within5.size
                else None
            )
            episode_sse = float(np.sum(episode_error * episode_error))
            well_episode_rows += end - start
            well_episode_sse += episode_sse
            episodes.append(
                {
                    "episode_id": f"{well}:{episode_index:03d}",
                    "well": well,
                    "start_row_idx": int(row_idx[start]),
                    "end_row_idx_exclusive": int(row_idx[end - 1] + 1),
                    "start_suffix_offset": int(start),
                    "rows": int(end - start),
                    "suffix_rows": int(len(frame)),
                    "audit_start_row_idx": int(row_idx[max(0, start - 128)]),
                    "audit_start_suffix_offset": int(max(0, start - 128)),
                    "start_suffix_fraction": float(start / len(frame)),
                    "episode_suffix_fraction": float((end - start) / len(frame)),
                    "rows_from_last_within5_to_episode_start": (
                        int(start - last_within5)
                        if last_within5 is not None
                        else np.nan
                    ),
                    "rows_from_episode_end_to_next_within5": (
                        int(next_within5 - end)
                        if next_within5 is not None
                        else np.nan
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
                    "mean_error_ft": float(np.mean(episode_error)),
                    "rmse_ft": float(np.sqrt(np.mean(episode_error * episode_error))),
                    "error_std_ft": float(np.std(episode_error)),
                    "error_slope_ft_per_row": linear_slope(episode_error),
                    "error_sign_consistency": float(
                        max(np.mean(episode_error > 0.0), np.mean(episode_error < 0.0))
                    ),
                    "max_abs_error_ft": float(np.max(np.abs(episode_error))),
                    "episode_sse": episode_sse,
                    "last_known_tvt": float(frame["last_known_tvt"].iloc[0]),
                    "last_known_md": float(frame["last_known_md"].iloc[0]),
                    "episode_start_md_since": float(frame["md_since"].iloc[start]),
                    "episode_end_md_since": float(frame["md_since"].iloc[end - 1]),
                }
            )
        if runs:
            wells.append(
                {
                    "well": well,
                    "episodes": len(runs),
                    "episode_rows": int(well_episode_rows),
                    "suffix_rows": int(len(frame)),
                    "episode_sse": float(well_episode_sse),
                    "suffix_rmse_ft": float(np.sqrt(np.mean(error * error))),
                    "suffix_max_abs_error_ft": float(np.max(np.abs(error))),
                    "last_known_tvt": float(frame["last_known_tvt"].iloc[0]),
                    "last_known_md": float(frame["last_known_md"].iloc[0]),
                }
            )

    episode_frame = pd.DataFrame(episodes).sort_values(
        ["well", "start_row_idx"], kind="stable"
    )
    well_frame = pd.DataFrame(wells).sort_values("well", kind="stable")
    if episode_frame.empty or well_frame.empty:
        raise RuntimeError("no persistent PF episodes found")
    shard_loads = np.zeros(SHARD_COUNT, dtype=np.int64)
    shard_assignment: dict[str, int] = {}
    for row in well_frame.sort_values(
        ["suffix_rows", "well"], ascending=[False, True], kind="stable"
    ).itertuples(index=False):
        shard_index = int(np.argmin(shard_loads))
        shard_assignment[str(row.well)] = shard_index
        shard_loads[shard_index] += int(row.suffix_rows)
    well_frame["shard_index"] = well_frame["well"].map(shard_assignment).astype(np.int8)
    episode_frame["shard_index"] = (
        episode_frame["well"].map(shard_assignment).astype(np.int8)
    )

    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "pf_persistent_offset_episodes.csv"
    well_path = output / "pf_target_wells.csv"
    manifest_path = output / "pf_persistent_asset_manifest.json"
    episode_frame.to_csv(episode_path, index=False)
    well_frame.to_csv(well_path, index=False)

    fixed_prediction_rows = pd.concat(all_frames, ignore_index=True)
    fixed_prediction_rows = fixed_prediction_rows.sort_values(
        ["well", "row_idx"], kind="stable"
    )
    manifest: dict[str, object] = {
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "source_decompressed_sha256": sha256_file(source, decompress_gzip=True),
        "fixed_prediction_subset_content_sha256": sha256_frame_content(
            fixed_prediction_rows
        ),
        "error_threshold_ft": ERROR_THRESHOLD_FT,
        "min_run_rows": MIN_RUN_ROWS,
        "source_rows": total_rows,
        "source_wells": total_wells,
        "target_wells": int(len(well_frame)),
        "shard_count": SHARD_COUNT,
        "shard_target_wells": {
            str(index): int((well_frame["shard_index"] == index).sum())
            for index in range(SHARD_COUNT)
        },
        "shard_suffix_rows": {
            str(index): int(
                well_frame.loc[well_frame["shard_index"] == index, "suffix_rows"].sum()
            )
            for index in range(SHARD_COUNT)
        },
        "episodes": int(len(episode_frame)),
        "episode_rows": int(episode_frame["rows"].sum()),
        "audit_rows_with_pre128_sum_nonunique": int(
            (
                episode_frame["end_row_idx_exclusive"]
                - episode_frame["audit_start_row_idx"]
            ).sum()
        ),
        "episode_sse": float(episode_frame["episode_sse"].sum()),
        "episode_csv_sha256": sha256_file(episode_path),
        "target_well_csv_sha256": sha256_file(well_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_assets(args.source, args.output, args.chunksize)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
