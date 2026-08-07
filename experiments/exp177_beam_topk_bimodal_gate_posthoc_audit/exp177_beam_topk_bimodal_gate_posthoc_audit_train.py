# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp177_beam_topk_bimodal_gate_posthoc_audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Parent artifact checks
# 4. Gate policy definition
# 5. Setup and compute budget
# 6. Run posthoc gate audit
# 7. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from beam_topk_bimodal_gate_posthoc_audit import (
    OUTPUT_PREFIX,
    find_artifact,
    read_csv_header,
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
# ## 3. Parent artifact checks

# %%
artifact_rows = []
for name in ["topk_diagnostics", "topk_paths", "candidate_wide", "candidate_metrics"]:
    artifact_path = find_artifact(config, name)
    artifact_rows.append(
        {
            "name": name,
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "columns_preview": read_csv_header(artifact_path)[:12],
        }
    )

display(pd.DataFrame(artifact_rows))

# %% [markdown]
# ## 4. Gate policy definition

# %%
gate_config = get_nested(config, "gate") or {}
beam_variants = gate_config.get("beam_variants", [])
replacement_suffixes = gate_config.get("replacement_suffixes", [])
and_policies = gate_config.get("and_policies", [])

display(
    pd.DataFrame(
        {
            "beam_variant": beam_variants,
        }
    )
)
display(pd.DataFrame({"replacement_suffix": replacement_suffixes}))
display(pd.DataFrame({"and_policy": [", ".join(policy) for policy in and_policies]}))

print("quantiles:", json.dumps(gate_config.get("quantiles", {}), indent=2, sort_keys=True))

# %% [markdown]
# ## 5. Setup and compute budget

# %%
print("primary baseline:", get_nested(config, "audit.primary_baseline"))
print("beam baseline:", get_nested(config, "audit.beam_baseline"))
print("max_rows:", get_nested(config, "audit.max_rows"))
print("max_wells:", get_nested(config, "audit.max_wells"))
print("active Beam regeneration variants: 0")
print("LightGBM config count: 0")
print("fold count: 0")
print("total boosters: 0")
print("control / parent retraining: false")
print("GPU:", get_nested(config, "runtime.kaggle.enable_gpu"))

# %% [markdown]
# ## 6. Run posthoc gate audit

# %%
summary = run_audit(config=config, paths=paths)
print(json.dumps(summary["primary_baseline"], indent=2, sort_keys=True))
print(json.dumps(summary["best_policy"], indent=2, sort_keys=True))
print(json.dumps(summary["decision"], indent=2, sort_keys=True))

# %% [markdown]
# ## 7. Metrics, diagnostics, and generated artifacts

# %%
artifacts = summary["artifacts"]
policy_metrics = pd.read_csv(artifacts["policy_metrics"])
gate_thresholds = pd.read_csv(artifacts["gate_thresholds"])
group_metrics = pd.read_csv(artifacts["group_metrics"])
by_well = pd.read_csv(artifacts["by_well"])

display(policy_metrics.head(30))
display(gate_thresholds.head(30))
display(group_metrics.head(30))
display(by_well.sort_values("delta_rmse_vs_baseline", ascending=False).head(30))

# %%
summary_path = Path(artifacts["summary"])
print("summary:", summary_path)
print("summary exists:", summary_path.exists())
print("artifact sha256:")
print(json.dumps(summary["artifact_sha256"], indent=2, sort_keys=True))
