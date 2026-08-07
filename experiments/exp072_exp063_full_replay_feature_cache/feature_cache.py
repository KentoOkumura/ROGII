from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from public_notebook_replay_audit import (
    build_replay_train_frames,
    configure_public_runtime,
    feature_columns_for_variant,
)

OUTPUT_PREFIX = "exp063_full_replay_feature_cache"
VARIANT = "pixiux_likpf_public_replay"


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _numeric_frame(frame: pd.DataFrame, meta_columns: set[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in frame.columns:
        if col in meta_columns:
            if col in {"id", "well"}:
                frame[col] = frame[col].astype(str)
            continue
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    return frame


def run_train_feature_cache(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    max_wells: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu="cpu",
        n_train_wells=max_wells,
    )

    t0 = time.time()
    _, pixiux_df, feature_meta = build_replay_train_frames(max_wells=max_wells)
    feature_columns = feature_columns_for_variant(pixiux_df, VARIANT)
    meta_columns = ["id", "well", "target"]
    missing = [col for col in [*meta_columns, *feature_columns] if col not in pixiux_df.columns]
    if missing:
        raise ValueError(f"Generated train frame is missing columns: {missing[:20]}")

    train_frame = _numeric_frame(
        pixiux_df[[*meta_columns, *feature_columns]],
        meta_columns=set(meta_columns),
    )
    numeric_values = train_frame[["target", *feature_columns]].to_numpy(np.float32)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Generated train feature cache contains non-finite numeric values")

    train_path = output_dir / f"{OUTPUT_PREFIX}_{VARIANT}_train_features.csv.gz"
    schema_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    train_frame.to_csv(train_path, index=False, compression="gzip")

    schema = pd.DataFrame(
        {
            "variant": VARIANT,
            "feature_index": np.arange(len(feature_columns), dtype=np.int32),
            "feature": feature_columns,
        }
    )
    schema.to_csv(schema_path, index=False)
    train_sha = sha256_file(train_path)

    summary = {
        "experiment": "exp072_exp063_full_replay_feature_cache",
        "status": "train_feature_cache_completed",
        "mode": "exp063_full_public_replay_train_feature_cache",
        "variant": VARIANT,
        "rows": int(len(train_frame)),
        "wells": int(train_frame["well"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "feature_meta": feature_meta,
        "outputs": {
            "train_features": train_path.name,
            "feature_schema": schema_path.name,
            "summary": summary_path.name,
        },
        "sha256": {
            "train_features": train_sha,
        },
        "notes": [
            "This notebook intentionally generates train features only.",
            "Current test features must be regenerated inside each downstream inference notebook.",
            "No LightGBM, CatBoost, Ridge, prediction, or submission step is executed.",
        ],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
