from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from direct_hmm_comparison import run_direct_comparison
from exact_hmm_smoother import (
    resolve_existing_file,
    sha256_gzip_decompressed,
    sha256_path,
    to_jsonable,
)
from settings import ExperimentPaths, get_nested, load_config


EXPERIMENT_NAME = "exp234_crossfitted_residual_scale_emission_hmm_on_exp218"
OUTPUT_PREFIX = "exp234_crossfitted_residual_scale_emission_hmm_on_exp218"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def resolve_or_materialize_hmm_cache(
    *,
    paths: ExperimentPaths,
    comparison: dict[str, Any],
    reuse: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    candidates = list(comparison.get("hmm_feature_cache") or [])
    try:
        return resolve_existing_file(paths.root, candidates), {"mode": "existing_file"}
    except FileNotFoundError as direct_error:
        archive_candidates = list(reuse.get("cache_archive_candidates") or [])
        member = str(reuse.get("cache_archive_member") or "")
        if not archive_candidates or not member:
            raise direct_error
        archive_path = resolve_existing_file(paths.root, archive_candidates)
        destination = paths.artifacts_dir / Path(member).name
        with zipfile.ZipFile(archive_path) as archive:
            if member not in archive.namelist():
                raise FileNotFoundError(
                    f"HMM cache member {member!r} is absent from dataset archive {archive_path}"
                )
            with archive.open(member) as source, destination.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"materialized HMM cache is empty: {destination}")
        return destination, {
            "mode": "dataset_archive_extract",
            "archive": str(archive_path),
            "member": member,
        }


def run_comparison_only_readout() -> dict[str, Any]:
    """Compare the completed v1 HMM cache without regenerating scale or HMM rows."""
    started = time.time()
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    comparison = dict(get_nested(config, "comparison") or {})
    reuse = dict(comparison.get("comparison_only") or {})
    if bool(reuse.get("hmm_recomputation_allowed", True)):
        raise ValueError("comparison-only notebook must keep hmm_recomputation_allowed=false")

    cache_path, cache_resolution = resolve_or_materialize_hmm_cache(
        paths=paths,
        comparison=comparison,
        reuse=reuse,
    )
    header = pd.read_csv(cache_path, nrows=0)
    required = {"id", "well", "target", "last_known_tvt", "md_since"}
    hmm_columns = [column for column in header.columns if column.endswith("_mean_tvt")]
    missing = sorted(required.difference(header.columns))
    if missing or len(hmm_columns) != 1:
        raise ValueError(
            "v1 HMM cache does not meet comparison-only contract: "
            f"missing={missing}, hmm_mean_columns={hmm_columns}"
        )

    comparison_summary = run_direct_comparison()
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_comparison_only_summary.json"
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "comparison_only_readout_completed",
        "mode": "reuse_v1_hmm_cache_direct_comparison_only",
        "source_hmm_cache": {
            "path": str(cache_path),
            "raw_sha256": sha256_path(cache_path),
            "content_sha256": (
                sha256_gzip_decompressed(cache_path)
                if cache_path.suffix == ".gz"
                else sha256_path(cache_path)
            ),
            "hmm_mean_column": hmm_columns[0],
            "resolution": cache_resolution,
        },
        "reuse_contract": reuse,
        "comparison_summary": comparison_summary,
        "notes": [
            "This notebook never calls residual-scale fitting or exact HMM cache generation.",
            "The v1 HMM cache is reused solely for direct train-side readout.",
            "No inference or submission output is generated.",
        ],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    summary["sha256"] = {"summary": sha256_path(summary_path)}
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_comparison_only_readout()
