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
# # exp201_typewell_spatial_tvt_error_readout train
#
# Diagnostic-only OOF readout for exp148 residual patterns by common typewell, XY neighborhood,
# sharp TVT step, and whole-well offset.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Input contract
# 4. Execute readout
# 5. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path

from typewell_spatial_tvt_error_readout import (
    EXP_DIR,
    load_config,
    main as run_readout,
    resolve_first_existing,
)

# %% [markdown]
# ## 2. Runtime and configuration

# %%
config = load_config()

print("Experiment:", config["experiment"]["name"])
print("Route:", config["experiment"]["route"])
print("Parent:", config["lineage"]["parent"])
print("Status:", config["experiment"]["status"])
print("Diagnostic only:", True)
print("Experiment directory:", EXP_DIR)

# %% [markdown]
# ## 3. Input contract

# %%
prediction_path = resolve_first_existing(config["data"]["exp148_prediction_candidates"])
typewell_path = resolve_first_existing(config["data"]["typewell_summary_candidates"])

print("Prediction source:", prediction_path)
print("Typewell summary:", typewell_path)
print("Model filter:", config["validation"])
print("Readout config:", json.dumps(config["readout"], indent=2))

# %% [markdown]
# ## 4. Execute readout

# %%
run_readout()

# %% [markdown]
# ## 5. Metrics and generated artifacts

# %%
metrics_path = EXP_DIR / "metrics.json"
summary_path = EXP_DIR / "artifacts" / "readout_summary.json"

metrics = json.loads(metrics_path.read_text())
summary = json.loads(summary_path.read_text())

print("Metrics path:", metrics_path)
print("Summary path:", summary_path)
print("Overall:", json.dumps(metrics["overall"], indent=2))
print("XY neighbor:", json.dumps(metrics["xy_neighbor"], indent=2))
print("Shape similarity:", json.dumps(metrics["shape_similarity"], indent=2))
print("Sharp TVT steps:", json.dumps(metrics["sharp_tvt_steps"], indent=2))
print("Offset wells:", metrics["offset_wells"])

print("Generated artifacts:")
for name, path in metrics["artifacts"].items():
    resolved = EXP_DIR / path
    print(f"- {name}: {resolved} ({resolved.stat().st_size} bytes)")
