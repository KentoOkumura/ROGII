# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp228_direct_residual_correction_on_exp226 train_aggregate
#
# Aggregate the OOF residual predictions from `train_lgb0`, `train_lgb1`, and `train_lgb2`. This notebook does not train boosters.

# %% [markdown]
# ## Contents
#
# 1. Setup and split output discovery
# 2. OOF aggregation
# 3. Metrics and generated artifacts

# %% [markdown]
# ## 1. Setup and split output discovery

# %%
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from direct_residual_correction_on_exp226 import OUTPUT_PREFIX, prediction_sha256, rmse


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


def sha256_gzip_decompressed(path: Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

filename = f"{OUTPUT_PREFIX}_predictions.csv.gz"
candidates: list[Path] = []
for root in [Path("/kaggle/input"), Path("/tmp/kaggle-output"), paths.artifacts_dir]:
    if root.exists():
        candidates.extend(root.glob(f"**/{filename}"))

prediction_paths: list[Path] = []
seen = set()
for path in sorted(candidates, key=lambda value: str(value)):
    key = str(path.resolve())
    if key not in seen and path.stat().st_size > 0:
        prediction_paths.append(path)
        seen.add(key)

print("Experiment:", EXPERIMENT_NAME)
print("Prediction files:", [str(path) for path in prediction_paths])
if len(prediction_paths) < 3:
    raise FileNotFoundError(f"Expected at least 3 split prediction files, found {len(prediction_paths)}")

# %% [markdown]
# ## 2. OOF aggregation

# %%
frames = []
source_rows = []
for index, path in enumerate(prediction_paths):
    frame = pd.read_csv(path, dtype={"id": str, "well": str})
    frame = frame[
        (frame["variant"].astype(str) == cfg_get(config, "inference.selected_variant"))
        & (frame["mode"].astype(str) == cfg_get(config, "inference.selected_mode"))
        & (frame["model"].astype(str) == "lgb_mean")
    ].copy()
    if frame.empty:
        continue
    frame = frame[
        ["id", "well", "target_tvt", "exp226_oof_pred", "pred_target", "pred_tvt"]
    ].rename(
        columns={
            "pred_target": f"pred_residual_split{index}",
            "pred_tvt": f"pred_tvt_split{index}",
        }
    )
    frames.append(frame)
    source_rows.append(
        {
            "path": str(path),
            "rows": int(len(frame)),
            "decompressed_sha256": sha256_gzip_decompressed(path),
        }
    )

if len(frames) < 3:
    raise ValueError(f"Expected 3 non-empty split prediction frames, got {len(frames)}")

merged = frames[0]
for frame in frames[1:]:
    merged = merged.merge(
        frame,
        on=["id", "well", "target_tvt", "exp226_oof_pred"],
        how="inner",
        validate="one_to_one",
    )

residual_columns = [column for column in merged.columns if column.startswith("pred_residual_split")]
merged["pred_residual_mean"] = merged[residual_columns].mean(axis=1).astype(np.float32)
merged["pred_tvt_mean"] = (
    merged["exp226_oof_pred"].to_numpy(np.float32)
    + merged["pred_residual_mean"].to_numpy(np.float32)
).astype(np.float32)
merged["error_tvt"] = merged["pred_tvt_mean"] - merged["target_tvt"]

# %% [markdown]
# ## 3. Metrics and generated artifacts

# %%
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": "split_oof_aggregate_completed",
    "variant": cfg_get(config, "inference.selected_variant"),
    "mode": cfg_get(config, "inference.selected_mode"),
    "model": "lgb_mean_from_split_lgb0_lgb1_lgb2",
    "rows": int(len(merged)),
    "wells": int(merged["well"].nunique()),
    "rmse_tvt": rmse(merged["target_tvt"].to_numpy(), merged["pred_tvt_mean"].to_numpy()),
    "rmse_residual": rmse(
        (merged["target_tvt"] - merged["exp226_oof_pred"]).to_numpy(),
        merged["pred_residual_mean"].to_numpy(),
    ),
    "prediction_sha256": prediction_sha256(
        merged["id"],
        merged["pred_tvt_mean"].to_numpy(np.float32),
        label=f"{OUTPUT_PREFIX}/split_lgb_mean/oof",
    ),
}

metrics_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_split_aggregate_metrics.csv"
predictions_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_split_aggregate_predictions.csv.gz"
summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_split_aggregate_summary.json"

pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
merged.to_csv(predictions_path, index=False, compression="gzip")
summary = {
    **metrics,
    "prediction_sources": source_rows,
    "artifacts": {
        "metrics": metrics_path.name,
        "predictions": predictions_path.name,
        "summary": summary_path.name,
    },
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
paths.metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

display(pd.DataFrame([metrics]))
display(merged.groupby("well", as_index=False).agg(rows=("id", "size"), rmse_tvt=("error_tvt", lambda x: float(np.sqrt(np.mean(np.square(x)))))).sort_values("rmse_tvt", ascending=False).head(30))
print("Aggregate summary:", summary_path)
