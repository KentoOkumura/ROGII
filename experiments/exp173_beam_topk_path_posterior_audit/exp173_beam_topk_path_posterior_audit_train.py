# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp173_beam_topk_path_posterior_audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and replay cache checks
# 4. Beam top-K path posterior helpers
# 5. Setup and configuration
# 6. Run train-side audit
# 7. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from beam_topk_path_posterior_audit import (
    OUTPUT_PREFIX,
    parse_beam_variants,
    parse_candidate_specs,
    read_feature_cache,
    run_audit,
)
from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
paths.ensure_output_dirs()

print("experiment:", get_nested(config, "experiment.name"))
print("route:", get_nested(config, "experiment.route"))
print("status:", get_nested(config, "experiment.status"))
print("parent:", get_nested(config, "lineage.parent"))
print("output_prefix:", OUTPUT_PREFIX)
print("kaggle_required:", True)

# %% [markdown]
# ## 3. Input and replay cache checks

# %%
candidate_specs = parse_candidate_specs(config)
frame_preview, feature_meta, resolved_sources = read_feature_cache(config, candidate_specs)
print("feature cache rows:", feature_meta["rows"])
print("feature cache wells:", feature_meta["wells"])
print("resolved sources:", resolved_sources)
print("source:", feature_meta["source"])
print("source decompressed sha:", feature_meta["source_decompressed_sha256"])
display(frame_preview.head())

# %% [markdown]
# ## 4. Beam top-K path posterior helpers

# %%
variants = parse_beam_variants(config)
variant_table = pd.DataFrame(
    [
        {
            "name": variant.name,
            "beam_size": variant.beam_size,
            "top_k": variant.top_k,
            "move_cost": variant.move_cost,
            "error_scale": variant.error_scale,
            "smooth_radius": variant.smooth_radius,
            "posterior_temperatures": list(variant.posterior_temperatures),
            "weighted_mean_temperature": variant.weighted_mean_temperature,
        }
        for variant in variants
    ]
)
display(variant_table)

print("LightGBM config count: 0")
print("fold count: 0")
print("total boosters: 0")
print("control retraining: false")

# %% [markdown]
# ## 5. Setup and configuration

# %%
print("primary baseline:", get_nested(config, "audit.primary_baseline"))
print("comparison baselines:", get_nested(config, "audit.comparison_baselines"))
print("top_k:", get_nested(config, "audit.top_k"))
print("posterior_temperatures:", get_nested(config, "audit.posterior_temperatures"))
print("max_rows:", get_nested(config, "audit.max_rows"))
print("max_wells:", get_nested(config, "audit.max_wells"))

# %% [markdown]
# ## 6. Run train-side audit

# %%
summary = run_audit(config=config, paths=paths)
print(json.dumps(summary["best_beam_topk_posterior_candidate"], indent=2, sort_keys=True))
print(json.dumps(summary["best_topk_oracle_headroom"], indent=2, sort_keys=True))

# %% [markdown]
# ## 7. Metrics, diagnostics, and generated artifacts

# %%
artifacts = summary["artifacts"]
candidate_metrics = pd.read_csv(artifacts["candidate_metrics"])
group_metrics = pd.read_csv(artifacts["group_metrics"])
beam_quality = pd.read_csv(artifacts["beam_quality"])

display(candidate_metrics.head(30))
display(group_metrics.head(30))
display(beam_quality.head(30))

# %%
summary_path = Path(artifacts["summary"])
print("summary:", summary_path)
print("summary exists:", summary_path.exists())
print("artifact sha256:")
print(json.dumps(summary["artifact_sha256"], indent=2, sort_keys=True))
