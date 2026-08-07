"""Strictly merge exact fixed-lag PF shard outputs after Kaggle completion."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PREFIX = "exp235_fixed_lag_particle_smoother_pf"
CHUNK_SIZE = 10_000


def decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(root: Path, suffix: str) -> Path:
    matches = sorted(root.rglob(f"{OUTPUT_PREFIX}_{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {suffix} under {root}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    shard_wells: list[pd.DataFrame] = []
    full_manifests: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    row_paths: list[Path] = []
    for root in args.input:
        row_path = one_file(root, "row_candidates.csv.gz")
        shard_path = one_file(root, "target_wells.csv")
        full_path = one_file(root, "target_wells_full.csv")
        row_paths.append(row_path)
        shard_wells.append(pd.read_csv(shard_path))
        full_manifests.append(pd.read_csv(full_path))
        sources.append(
            {
                "root": str(root),
                "row_candidates": str(row_path),
                "row_candidates_decompressed_sha256": decompressed_sha256(row_path),
            }
        )

    full_well_sets = [tuple(sorted(frame["well"].astype(str).unique())) for frame in full_manifests]
    if len(set(full_well_sets)) != 1:
        raise ValueError("Shard target_wells_full manifests differ; refusing to merge")
    selected_wells = pd.concat(shard_wells, ignore_index=True)
    if selected_wells["well"].astype(str).duplicated().any():
        raise ValueError("A well was assigned to more than one shard")
    expected_wells = set(full_well_sets[0])
    if set(selected_wells["well"].astype(str)) != expected_wells:
        missing = sorted(expected_wells - set(selected_wells["well"].astype(str)))
        extra = sorted(set(selected_wells["well"].astype(str)) - expected_wells)
        raise ValueError(
            f"Shard wells do not cover full surface: missing={missing[:5]}, extra={extra[:5]}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / f"{OUTPUT_PREFIX}_merged_row_candidates.csv.gz"
    partial_path = args.output_dir / f".{OUTPUT_PREFIX}_merged_row_candidates.partial.csv.gz"
    sqlite_path = args.output_dir / f".{OUTPUT_PREFIX}_seen_ids.sqlite"
    if partial_path.exists() or sqlite_path.exists():
        raise FileExistsError(
            "Found stale partial merge state; inspect and remove it before retrying"
        )
    if merged_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing merged candidates: {merged_path}")

    required = {"id", "well", "row_idx", "true_tvt"}
    candidates: list[str] | None = None
    stats: dict[str, dict[str, float | int]] = {}
    seen_wells: set[str] = set()
    rows = 0
    chunks_read = 0
    write_header = True
    try:
        with sqlite3.connect(sqlite_path) as connection, gzip.open(partial_path, "wt") as handle:
            connection.execute("CREATE TABLE seen_ids (id TEXT PRIMARY KEY)")
            for row_path in row_paths:
                for chunk in pd.read_csv(row_path, compression="gzip", chunksize=CHUNK_SIZE):
                    missing = sorted(required - set(chunk.columns))
                    if missing:
                        raise ValueError(f"{row_path} missing required row columns: {missing}")
                    chunk_candidates = [
                        column
                        for column in chunk.columns
                        if column == "exp072_likpf_mean"
                        or (column.startswith("pf_lag") and column.endswith("_mean"))
                    ]
                    if candidates is None:
                        candidates = chunk_candidates
                        stats = {
                            candidate: {
                                "n": 0,
                                "squared_error": 0.0,
                                "absolute_error": 0.0,
                                "within10": 0,
                            }
                            for candidate in candidates
                        }
                    elif chunk_candidates != candidates:
                        raise ValueError(f"Candidate columns differ in {row_path}")

                    ids = chunk["id"].astype(str).tolist()
                    try:
                        connection.executemany(
                            "INSERT INTO seen_ids(id) VALUES (?)",
                            ((value,) for value in ids),
                        )
                    except sqlite3.IntegrityError as exc:
                        message = f"Duplicate ids encountered while reading {row_path}"
                        raise ValueError(message) from exc
                    connection.commit()

                    target = pd.to_numeric(chunk["true_tvt"], errors="coerce").to_numpy(np.float64)
                    for candidate in candidates:
                        values = pd.to_numeric(chunk[candidate], errors="coerce").to_numpy(
                            np.float64
                        )
                        error = values - target
                        finite = np.isfinite(error)
                        if finite.any():
                            finite_error = error[finite]
                            stats[candidate]["n"] += int(finite.sum())
                            stats[candidate]["squared_error"] += float(
                                np.sum(finite_error * finite_error)
                            )
                            stats[candidate]["absolute_error"] += float(
                                np.sum(np.abs(finite_error))
                            )
                            stats[candidate]["within10"] += int(
                                np.sum(np.abs(finite_error) <= 10.0)
                            )

                    seen_wells.update(chunk["well"].dropna().astype(str))
                    rows += len(chunk)
                    chunks_read += 1
                    chunk.to_csv(handle, index=False, header=write_header)
                    write_header = False
                    if chunks_read % 10 == 0:
                        print(f"Merged {rows:,} rows so far", flush=True)

        if seen_wells != expected_wells:
            missing = sorted(expected_wells - seen_wells)
            extra = sorted(seen_wells - expected_wells)
            raise ValueError(
                "Merged row candidates do not cover expected wells: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if candidates is None:
            raise ValueError("No candidate rows were read")
        partial_path.replace(merged_path)
    finally:
        if partial_path.exists():
            partial_path.unlink()
        if sqlite_path.exists():
            sqlite_path.unlink()

    metrics = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "rmse": (
                    float(np.sqrt(values["squared_error"] / values["n"]))
                    if values["n"]
                    else float("nan")
                ),
                "mae": (
                    float(values["absolute_error"] / values["n"]) if values["n"] else float("nan")
                ),
                "within10": (
                    float(values["within10"] / values["n"]) if values["n"] else float("nan")
                ),
            }
            for candidate, values in stats.items()
        ]
    )
    metrics_path = args.output_dir / f"{OUTPUT_PREFIX}_merged_candidate_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    manifest = {
        "shard_count": len(args.input),
        "rows": rows,
        "wells": len(seen_wells),
        "merged_row_candidates": str(merged_path),
        "merged_row_candidates_decompressed_sha256": decompressed_sha256(merged_path),
        "sources": sources,
    }
    (args.output_dir / f"{OUTPUT_PREFIX}_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
