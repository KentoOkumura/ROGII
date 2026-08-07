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
# # exp165_coordinate_frame_normalization_features_on_exp148 train_aggregate
#
# Aggregate the three CPU split train outputs on Kaggle and compute the final
# 3-model `lgb_mean` OOF score without downloading large prediction archives
# locally.

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# %% [markdown]
# ## 2. Configuration

# %%
EXPERIMENT = "exp165_coordinate_frame_normalization_features_on_exp148"
VARIANT = "coordinate_frame_addonly"
MODE = "cpu_deterministic_threads8"
SPLITS = ["lgb0", "lgb1", "lgb2"]
BASELINE_EXP148_LGB_MEAN = 8.50128118189582
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT,
            "variant": VARIANT,
            "mode": MODE,
            "splits": SPLITS,
            "baseline_exp148_lgb_mean": BASELINE_EXP148_LGB_MEAN,
        },
        indent=2,
    )
)


# %% [markdown]
# ## 3. Input Discovery

# %%
def find_one(filename: str) -> Path:
    roots = [Path("/kaggle/input"), Path.cwd()]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob(f"**/{filename}"))
    candidates = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one non-empty {filename}, got {len(candidates)}: "
            + ", ".join(str(path) for path in candidates[:20])
        )
    return candidates[0]


prediction_paths = {
    split: find_one(f"{EXPERIMENT}_{split}_predictions.csv.gz") for split in SPLITS
}
metrics_paths = {split: find_one(f"{EXPERIMENT}_{split}_metrics.csv") for split in SPLITS}
print(json.dumps({split: str(path) for split, path in prediction_paths.items()}, indent=2))


# %% [markdown]
# ## 4. Aggregate Predictions

# %%
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))


def prediction_sha_proxy(ids: pd.Series, values: np.ndarray) -> str:
    frame = pd.DataFrame(
        {
            "id": ids.astype(str).to_numpy(),
            "pred_tvt": np.round(np.asarray(values, dtype=np.float64), 8),
        }
    ).sort_values("id")
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


split_frames: list[pd.DataFrame] = []
single_metrics: dict[str, float] = {}
single_metric_rows: list[dict[str, object]] = []

for split in SPLITS:
    predictions = pd.read_csv(prediction_paths[split])
    selected = predictions[
        predictions["variant"].eq(VARIANT)
        & predictions["mode"].eq(MODE)
        & predictions["model"].eq("lgb_mean")
    ].copy()
    if selected.empty:
        raise ValueError(f"{split}: no selected lgb_mean predictions")
    selected = selected.sort_values("id").reset_index(drop=True)
    selected_rmse = rmse(
        selected["target_tvt"].to_numpy(np.float64),
        selected["pred_tvt"].to_numpy(np.float64),
    )
    single_metrics[split] = selected_rmse

    metrics = pd.read_csv(metrics_paths[split])
    metric_row = metrics[
        metrics["variant"].eq(VARIANT)
        & metrics["mode"].eq(MODE)
        & metrics["model"].eq("lgb_mean")
        & metrics["fold"].astype(str).eq("pooled")
    ].copy()
    if not metric_row.empty:
        single_metric_rows.append(metric_row.iloc[0].to_dict())

    keep = selected[["id", "well", "last_known_tvt", "target_tvt", "pred_tvt"]].copy()
    keep = keep.rename(columns={"pred_tvt": f"pred_tvt_{split}"})
    split_frames.append(keep)

merged = split_frames[0]
for frame in split_frames[1:]:
    merged = merged.merge(
        frame,
        on=["id", "well", "last_known_tvt", "target_tvt"],
        how="inner",
        validate="one_to_one",
    )

if len(merged) != len(split_frames[0]):
    raise ValueError(f"merge dropped rows: {len(merged)} vs {len(split_frames[0])}")

pred_cols = [f"pred_tvt_{split}" for split in SPLITS]
merged["pred_tvt"] = merged[pred_cols].mean(axis=1)
merged["error_tvt"] = merged["pred_tvt"] - merged["target_tvt"]

final_rmse = rmse(
    merged["target_tvt"].to_numpy(np.float64),
    merged["pred_tvt"].to_numpy(np.float64),
)
sha_proxy = prediction_sha_proxy(merged["id"], merged["pred_tvt"].to_numpy(np.float64))

by_well = (
    merged.groupby("well", as_index=False)
    .agg(
        rows=("id", "size"),
        rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
        error_mean=("error_tvt", "mean"),
        error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
    )
    .sort_values("rmse_tvt", ascending=False)
)

summary = {
    "experiment": EXPERIMENT,
    "variant": VARIANT,
    "mode": MODE,
    "rows": int(len(merged)),
    "split_rmse_tvt": single_metrics,
    "lgb_mean_rmse_tvt": final_rmse,
    "baseline_exp148_lgb_mean_rmse_tvt": BASELINE_EXP148_LGB_MEAN,
    "delta_vs_exp148_lgb_mean": final_rmse - BASELINE_EXP148_LGB_MEAN,
    "prediction_sha256_id_pred8": sha_proxy,
    "prediction_paths": {split: str(path) for split, path in prediction_paths.items()},
    "metrics_paths": {split: str(path) for split, path in metrics_paths.items()},
    "single_metric_rows": single_metric_rows,
    "worst_wells": by_well.head(20).to_dict(orient="records"),
}

print(json.dumps(summary, indent=2, ensure_ascii=False))


# %% [markdown]
# ## 5. Artifacts

# %%
summary_path = OUTPUT_DIR / f"{EXPERIMENT}_split_lgb_mean_summary.json"
by_well_path = OUTPUT_DIR / f"{EXPERIMENT}_split_lgb_mean_by_well.csv"
metrics_path = OUTPUT_DIR / f"{EXPERIMENT}_split_lgb_mean_metrics.csv"

summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
by_well.to_csv(by_well_path, index=False)
pd.DataFrame(
    [
        {
            "variant": VARIANT,
            "mode": MODE,
            "model": "split_lgb_mean",
            "fold": "pooled",
            "rows": int(len(merged)),
            "rmse_tvt": final_rmse,
            "delta_vs_exp148_lgb_mean": final_rmse - BASELINE_EXP148_LGB_MEAN,
            "prediction_sha256_id_pred8": sha_proxy,
        }
    ]
).to_csv(metrics_path, index=False)

print(
    json.dumps(
        {
            "summary": str(summary_path),
            "by_well": str(by_well_path),
            "metrics": str(metrics_path),
        },
        indent=2,
    )
)
