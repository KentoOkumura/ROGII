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
# # exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe train
#
# Train-side diagnostic that regenerates dense stride heatmap MDN local paths
# from exp202 saved fold models, then reruns the exp207 target-free stitch
# readout against the exp099 PF/Beam candidate union.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime setup
# 2. Configuration and input contract
# 3. Run dense path regeneration and stitch diagnostic
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime setup

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from IPython.display import display

from heatmap_mdn_dense_stride_window_path_regeneration_probe import (
    exp202_manifest_path,
    load_candidate_cache,
    run_stitch_probe,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
MAX_WELLS_ENV = os.environ.get("EXPERIMENT_MAX_WELLS")
MAX_WELLS = int(MAX_WELLS_ENV) if MAX_WELLS_ENV else None

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Status:", get_nested(config, "experiment.status"))
print("Root:", paths.root)
print("Artifacts:", paths.artifacts_dir)
print("Debug:", DEBUG, "Max wells:", MAX_WELLS)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# %% [markdown]
# ## 2. Configuration and input contract

# %%
print("Parent:", get_nested(config, "lineage.parent"))
print("Stitch parent:", get_nested(config, "lineage.stitch_parent"))
print("Comparison parent:", get_nested(config, "lineage.comparison_parent"))
print("Validation strategy:", get_nested(config, "validation.strategy"))
print("Dense row-center stride:", get_nested(config, "path_generation.row_center_stride"))
print("Dense topK generated:", get_nested(config, "path_generation.topk"))
print("Stitch local topK values:", get_nested(config, "stitching.local_topk_values"))
print("Primary local topK:", get_nested(config, "stitching.primary_local_topk"))
print("Beam width:", get_nested(config, "stitching.beam_width"))
print("Output topN:", get_nested(config, "stitching.output_topn"))
print("Score weights:")
print(json.dumps(get_nested(config, "stitching.score_weights"), indent=2, sort_keys=True))

print("Existing candidate columns:")
print(json.dumps(get_nested(config, "candidate_union.existing_candidates"), indent=2))

# %%
manifest_path = exp202_manifest_path(config)
cache, existing_candidates, cache_meta = load_candidate_cache(config)

print("exp202 model manifest:", manifest_path)
print("exp099 candidate cache:")
print(json.dumps(cache_meta, indent=2, sort_keys=True))
print("Available existing candidates:", existing_candidates)
print("Rows in exp099 cache:", len(cache), "wells:", cache["well"].nunique())
print("Kaggle GPU enabled:", get_nested(config, "runtime.kaggle.enable_gpu"))
print("LightGBM configs / boosters: 0 / 0")
print(
    "Parent/control retraining:",
    get_nested(config, "model.training.control_or_parent_retraining"),
)

# %% [markdown]
# ## 3. Run dense path regeneration and stitch diagnostic

# %%
summary = run_stitch_probe(
    config=config,
    paths=paths,
    max_wells=MAX_WELLS,
    debug=DEBUG,
)

print("Summary path:", summary["output_paths"]["summary"])
print("Dense path generation:")
print(json.dumps(summary["dense_path_generation"], indent=2, sort_keys=True))
print("Primary local topK:", summary["primary_local_topk"])

for topk_summary in summary["topk_summaries"]:
    print("====", topk_summary["label"], "====")
    print("Evaluation:")
    print(json.dumps(topk_summary["evaluation"], indent=2, sort_keys=True))
    print("Physicality:")
    print(json.dumps(topk_summary["physicality"], indent=2, sort_keys=True))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
primary = next(
    item
    for item in summary["topk_summaries"]
    if item["local_topk"] == summary["primary_local_topk"]
)
metrics_path = Path(primary["output_paths"]["candidate_union_metrics"])
distance_path = Path(primary["output_paths"]["candidate_union_distance_bucket_metrics"])
by_well_path = Path(primary["output_paths"]["candidate_union_by_well"])
coverage_path = Path(primary["output_paths"]["stitched_coverage_by_well"])

metrics_df = pd.read_csv(metrics_path)
distance_df = pd.read_csv(distance_path) if distance_path.exists() else pd.DataFrame()
by_well_df = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

print("Primary candidate union metrics:")
display(metrics_df)
print("Primary distance bucket metrics:")
display(distance_df)
print("Best / worst by-well RMSE deltas:")
if not by_well_df.empty:
    display(by_well_df.head(10))
    display(by_well_df.tail(10))
else:
    display(by_well_df)
print("Coverage by well:")
display(coverage_df.head(10))

comparison_rows = []
for item in summary["topk_summaries"]:
    for metric_row in item["metrics"]:
        row = dict(metric_row)
        row["local_topk"] = item["local_topk"]
        comparison_rows.append(row)
print("TopK comparison:")
display(pd.DataFrame(comparison_rows))

print("Generated artifacts:")
print(json.dumps(summary["output_paths"], indent=2, sort_keys=True))
print("Output SHA:")
print(json.dumps(summary["output_sha256"], indent=2, sort_keys=True))
