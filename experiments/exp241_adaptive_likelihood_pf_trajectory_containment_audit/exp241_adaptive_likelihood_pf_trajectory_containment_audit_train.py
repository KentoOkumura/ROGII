# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp241 adaptive likelihood-PF trajectory containment audit — train
#
# exp072-compatible T=1 control と gated T=2 treatment を同一 seed base で paired replayし、
# 最初の target-free gate 後に path divergence が resampling / seed aggregation で増幅するかを診断する。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Input and replay-cache checks
# 4. Paired PF and target-free event contract
# 5. Full train-side audit
# 6. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import os

from IPython.display import display

from adaptive_robust_likelihood_pf import (
    read_exp072_eval_cache,
    run_trajectory_containment_audit,
    select_target_wells,
)
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
paths = ExperimentPaths()
config = load_config()
active_shard_index = int(
    os.environ.get(
        "EXP241_ACTIVE_WELL_SHARD_INDEX",
        str(get_nested(config, "execution.active_well_shard_index") or 0),
    )
)
config.setdefault("execution", {})["active_well_shard_index"] = active_shard_index
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

runtime = get_nested(config, "model.runtime") or {}
paired = get_nested(config, "model.paired_replay") or {}
gate = get_nested(config, "model.gate") or {}
audit = get_nested(config, "audit") or {}
execution = get_nested(config, "execution") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "status": get_nested(config, "experiment.status"),
        "paired_replay": paired,
        "particles": runtime.get("particles"),
        "seed_count": runtime.get("seed_count"),
        "horizons": audit.get("horizons_rows"),
        "event_policy": audit.get("event_policy"),
        "execution_cost": execution,
        "well_shard": {
            "index": execution.get("active_well_shard_index"),
            "count": execution.get("well_shard_count"),
            "policy": execution.get("shard_policy"),
        },
        "gpu": kaggle_runtime.get("enable_gpu"),
        "internet": kaggle_runtime.get("enable_internet"),
        "kernel_sources": kaggle_runtime.get("kernel_sources"),
    }
)

# %% [markdown]
# ## 3. Input and replay-cache checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))
if not horizontal_files or not typewell_files:
    raise FileNotFoundError(f"Missing raw train horizontal/typewell files under {train_dir}")

validation_frame, validation_meta = read_exp072_eval_cache(config)
target_wells = select_target_wells(validation_frame, train_dir, config)
display(
    {
        "horizontal_wells": len(horizontal_files),
        "typewells": len(typewell_files),
        "validation_rows": len(validation_frame),
        "validation_wells": validation_frame["well"].nunique(),
        "eligible_wells": len(target_wells),
        "input_source": validation_meta.get("source"),
        "input_decompressed_sha": validation_meta.get("source_decompressed_sha256"),
        "reference_candidates": validation_meta.get("reference_candidates_present"),
    }
)
display(target_wells.head(20))

if target_wells.empty:
    raise RuntimeError("No eligible exp072 pseudo-tail wells were selected")
if get_nested(config, "model.validation_surface.max_target_wells") is not None:
    raise RuntimeError("Canonical exp241 must use all eligible wells")

# %% [markdown]
# ## 4. Paired PF and target-free event contract

# %%
if paired.get("control_temperature") != 1.0:
    raise RuntimeError("The paired control must remain T=1")
if paired.get("treatment_temperature") != 2.0:
    raise RuntimeError("The only treatment must remain gated T=2")
if paired.get("active_treatments") != ["gated_t2"]:
    raise RuntimeError("Exactly one treatment must be active")
if runtime.get("particles") != 500 or runtime.get("seed_count") != 128:
    raise RuntimeError("The PF contract is fixed to 500 particles x 128 seeds")
if runtime.get("seed_aggregation") != "mean":
    raise RuntimeError("Seed aggregation must remain mean")
if not gate.get("enabled"):
    raise RuntimeError("The frozen target-free gate must be enabled")
if audit.get("horizons_rows") != [8, 32, 64, 128, 256, 512, 1024]:
    raise RuntimeError("The event horizon grid is frozen")
if execution.get("lightgbm_config_count") != 0 or execution.get("total_boosters") != 0:
    raise RuntimeError("exp241 must not train ML models")
if execution.get("well_shard_count") != 4:
    raise RuntimeError("The full exp241 audit is fixed to four deterministic well shards")
if execution.get("active_well_shard_index") not in [0, 1, 2, 3]:
    raise RuntimeError("active_well_shard_index must be one of 0/1/2/3")
if kaggle_runtime.get("enable_gpu") or kaggle_runtime.get("enable_internet"):
    raise RuntimeError("exp241 is an offline CPU-only audit")

display(
    {
        "event_selection": "first treatment gate per well x seed",
        "gate_inputs": [
            "pre-update normalized innovation",
            "raw-GR change point and novelty",
            "pre-update ESS ratio and maximum particle weight",
        ],
        "target_usage": "post-event scoring only",
        "shared_seed": paired.get("shared_seed_namespace"),
        "conditional_rng_caveat": get_nested(config, "reproducibility.conditional_rng_caveat"),
        "forbidden": [
            "true TVT / error / oracle event selection",
            "temperature or mixture grid",
            "raw-test inference",
            "submission",
        ],
    }
)

# %% [markdown]
# ## 5. Full train-side audit

# %%
result = run_trajectory_containment_audit(
    config=config,
    paths=paths,
    validation_frame=validation_frame,
    validation_meta=validation_meta,
)

# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
display(result["candidate_metrics"])
display(result["event_summary"])
display(result["hidden_like_metrics"])
display(result["by_well"].sort_values("delta_rmse_vs_primary_baseline", ascending=False).head(30))
display(result["event_manifest"].head(30))
display(result["well_status"].head(30))
display(result["summary"])

print("generated artifacts")
for artifact_name, artifact_path in result["summary"]["artifacts"].items():
    print(f"- {artifact_name}: {artifact_path}")
