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
# # exp207_heatmap_mdn_overlapping_window_path_stitch_probe train
#
# Train-side diagnostic that stitches exp202 heatmap MDN local topK window paths
# into target-free full-well candidate paths and audits oracle headroom against
# the existing exp099 PF/Beam candidate union.

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime setup
# 2. Configuration and input contract
# 3. Run overlapping-window path stitch diagnostic
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

from heatmap_mdn_overlapping_window_path_stitch_probe import (
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
print("Parent:", get_nested(config, "lineage.parent"))
print("Comparison parent:", get_nested(config, "lineage.comparison_parent"))
print("Validation strategy:", get_nested(config, "validation.strategy"))
print("Local topK:", get_nested(config, "stitching.local_topk"))
print("Beam width:", get_nested(config, "stitching.beam_width"))
print("Output topN:", get_nested(config, "stitching.output_topn"))
print("Score weights:")
print(json.dumps(get_nested(config, "stitching.score_weights"), indent=2, sort_keys=True))

print("Existing candidate columns:")
print(json.dumps(get_nested(config, "candidate_union.existing_candidates"), indent=2))

# %%
arrays, samples, path_meta = load_candidate_path_inputs(config)
cache, existing_candidates, cache_meta = load_candidate_cache(config)

print("exp202 path artifact:")
print(json.dumps(path_meta, indent=2, sort_keys=True))
print("exp099 candidate cache:")
print(json.dumps(cache_meta, indent=2, sort_keys=True))
print("Available existing candidates:", existing_candidates)

sample_spacing = (
    samples.sort_values(["well", "row_center"])
    .groupby("well", observed=True)["row_center"]
    .diff()
    .dropna()
)
print("Source samples:", len(samples), "wells:", samples["well"].nunique())
print("Sample center gap summary:")
print(sample_spacing.describe())
print("Rows in exp099 cache:", len(cache), "wells:", cache["well"].nunique())

for forbidden in ["true_center_tvt", "target_in_grid", "center_abs_error"]:
    if forbidden in samples.columns:
        raise ValueError(f"Forbidden target-derived stitch input column loaded: {forbidden}")

# %% [markdown]
# ## 3. Run overlapping-window path stitch diagnostic

# %%
summary = run_stitch_probe(
    config=config,
    paths=paths,
    max_wells=MAX_WELLS,
    debug=DEBUG,
)

print("Summary path:", summary["output_paths"]["summary"])
print("Evaluation summary:")
print(json.dumps(summary["evaluation"], indent=2, sort_keys=True))
print("Physicality summary:")
print(json.dumps(summary["physicality"], indent=2, sort_keys=True))

# %% [markdown]
# ## 4. Metrics and generated artifacts

# %%
metrics_path = Path(summary["output_paths"]["candidate_union_metrics"])
distance_path = Path(summary["output_paths"]["candidate_union_distance_bucket_metrics"])
by_well_path = Path(summary["output_paths"]["candidate_union_by_well"])
coverage_path = Path(summary["output_paths"]["stitched_coverage_by_well"])

metrics_df = pd.read_csv(metrics_path)
distance_df = pd.read_csv(distance_path) if distance_path.exists() else pd.DataFrame()
by_well_df = pd.read_csv(by_well_path) if by_well_path.exists() else pd.DataFrame()
coverage_df = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

print("Candidate union metrics:")
display(metrics_df)
print("Distance bucket metrics:")
display(distance_df)
print("Best / worst by-well RMSE deltas:")
if not by_well_df.empty:
    display(by_well_df.head(10))
    display(by_well_df.tail(10))
else:
    display(by_well_df)
print("Coverage by well:")
display(coverage_df.head(10))

print("Generated artifacts:")
print(json.dumps(summary["output_paths"], indent=2, sort_keys=True))
print("Output SHA:")
print(json.dumps(summary["output_sha256"], indent=2, sort_keys=True))
