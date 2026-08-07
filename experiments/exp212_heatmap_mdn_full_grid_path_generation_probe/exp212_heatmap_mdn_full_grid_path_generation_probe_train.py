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
# # exp212_heatmap_mdn_full_grid_path_generation_probe train
#
# Train-side diagnostic that turns exp208 dense heatmap MDN local paths into an
# explicit full-grid candidate path artifact for downstream selector work.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime setup
# 2. Configuration and input contract
# 3. Run full-grid path generation diagnostic
# 4. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime setup

# %%
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import display

from heatmap_mdn_full_grid_path_generation_probe import (
    load_candidate_cache,
    load_candidate_path_inputs,
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

# %% [markdown]
# ## 2. Configuration and input contract

# %%
print("Backlog:", get_nested(config, "lineage.backlog"))
print("Parent:", get_nested(config, "lineage.parent"))
print("Dense parent:", get_nested(config, "lineage.dense_parent"))
print("Stitch parent:", get_nested(config, "lineage.stitch_parent"))
print("Downstream candidate:", get_nested(config, "lineage.downstream_candidate"))
print("Validation strategy:", get_nested(config, "validation.strategy"))
print("Path input mode:", get_nested(config, "path_generation.source_mode"))
print("Stitch local topK values:", get_nested(config, "stitching.local_topk_values"))
print("Primary local topK:", get_nested(config, "stitching.primary_local_topk"))
print("Beam width:", get_nested(config, "stitching.beam_width"))
print("Output topN full-grid paths:", get_nested(config, "stitching.output_topn"))
print("Contract output ranks:", get_nested(config, "full_path_contract.output_ranks"))
print("Score weights:")
print(json.dumps(get_nested(config, "stitching.score_weights"), indent=2, sort_keys=True))

print("Existing candidate columns:")
print(json.dumps(get_nested(config, "candidate_union.existing_candidates"), indent=2))
print("Required contract columns:")
print(json.dumps(get_nested(config, "full_path_contract.required_columns"), indent=2))

# %%
arrays, samples, path_meta = load_candidate_path_inputs(config)
cache, existing_candidates, cache_meta = load_candidate_cache(config)

print("exp208 dense path inputs:")
print(json.dumps(path_meta, indent=2, sort_keys=True))
print("exp099 candidate cache:")
print(json.dumps(cache_meta, indent=2, sort_keys=True))
print("Available existing candidates:", existing_candidates)
print("Rows in exp099 cache:", len(cache), "wells:", cache["well"].nunique())
print("Rows in exp208 dense path samples:", len(samples), "wells:", samples["well"].nunique())
print("Dense path tensor shape:", tuple(arrays["pred_tvt_path"].shape))
print("Kaggle GPU enabled:", get_nested(config, "runtime.kaggle.enable_gpu"))
print("CNN models / LightGBM configs / boosters: 0 / 0 / 0")
print(
    "Parent/control retraining:",
    get_nested(config, "model.training.control_or_parent_retraining"),
)

# %% [markdown]
# ## 3. Run full-grid path generation diagnostic

# %%
summary = run_stitch_probe(
    config=config,
    paths=paths,
    max_wells=MAX_WELLS,
    debug=DEBUG,
)

print("Summary path:", summary["output_paths"]["summary"])
print("Path inputs:")
print(json.dumps(summary["path_inputs"], indent=2, sort_keys=True))
print("Primary local topK:", summary["primary_local_topk"])

for topk_summary in summary["topk_summaries"]:
    print("====", topk_summary["label"], "====")
    print("Full-grid contract:")
    print(json.dumps(topk_summary["full_grid_contract"], indent=2, sort_keys=True))
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
full_path_path = Path(primary["output_paths"]["full_grid_candidate_paths"])
schema_path = Path(primary["output_paths"]["full_grid_path_schema"])
contract_path = Path(primary["output_paths"]["full_grid_contract_metrics"])

metrics_df = pd.read_csv(metrics_path)
distance_df = pd.read_csv(distance_path) if distance_path.exists() else pd.DataFrame()
by_well_df = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
full_path_head = pd.read_csv(full_path_path, nrows=20)
schema_df = pd.read_csv(schema_path)
contract_df = pd.read_csv(contract_path)

print("Primary full-grid path contract metrics:")
display(contract_df)
print("Primary full-grid path schema:")
display(schema_df)
print("Primary full-grid candidate path preview:")
display(full_path_head)
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
