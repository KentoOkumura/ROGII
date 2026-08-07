# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp243 PF seed medoids — train-side candidate audit
#
# exp072互換likelihood-PFの128 seed trajectoryを平均前にクラスタリングし、
# 実在するseed trajectoryであるmedoidを候補として保持する。true TVTは候補生成後の
# RMSE / oracle診断だけに使い、selector・raw-test inference・submissionは実行しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Configuration and fixed experiment contract
# 3. Input cache and raw-data checks
# 4. PF replay and deterministic K-medoids contract
# 5. Full train-side candidate generation
# 6. Metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import os

from IPython.display import display

from pf_seed_medoids import (
    read_exp072_eval_cache,
    run_pf_seed_medoids,
    select_target_wells,
)
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# Numba PF kernel、raw train path解決、pseudo-tail復元は重いPF helperに置く。
# notebook側では入力契約、固定K-medoids仕様、実行対象、評価、生成物を確認する。

# %% [markdown]
# ## 2. Configuration and fixed experiment contract

# %%
paths = ExperimentPaths()
config = load_config()
active_shard_index = int(
    os.environ.get(
        "EXP243_ACTIVE_WELL_SHARD_INDEX",
        str(get_nested(config, "execution.active_well_shard_index") or 0),
    )
)
config.setdefault("execution", {})["active_well_shard_index"] = active_shard_index
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

runtime = get_nested(config, "model.runtime") or {}
replay = get_nested(config, "model.replay") or {}
clustering = get_nested(config, "model.clustering") or {}
distance = clustering.get("distance") or {}
execution = get_nested(config, "execution") or {}
parity_probe = bool(execution.get("parity_probe", False))
audit = get_nested(config, "audit") or {}
kaggle_runtime = get_nested(config, "runtime.kaggle") or {}

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "status": get_nested(config, "experiment.status"),
        "parent": get_nested(config, "lineage.parent"),
        "implementation_source": get_nested(config, "lineage.implementation_source"),
        "replay": replay,
        "runtime": runtime,
        "clustering": clustering,
        "oracle_block_rows": audit.get("oracle_block_rows"),
        "execution_cost": execution,
        "parity_probe": parity_probe,
        "active_well_shard": active_shard_index,
        "gpu_enabled": kaggle_runtime.get("enable_gpu"),
        "internet_enabled": kaggle_runtime.get("enable_internet"),
        "kernel_sources": kaggle_runtime.get("kernel_sources"),
    }
)

if get_nested(config, "experiment.route") != "pf_beam":
    raise RuntimeError("exp243 must remain on the pf_beam route")
if runtime.get("particles") != 500 or runtime.get("seed_count") != 128:
    raise RuntimeError("The PF replay contract is fixed to 500 particles x 128 seeds")
if replay.get("replay_count") != 1:
    raise RuntimeError("Exactly one PF replay per well is allowed")
if clustering.get("algorithm") != "deterministic_build_pam":
    raise RuntimeError("The clustering algorithm must remain deterministic BUILD+PAM")
if clustering.get("k_values") != [3, 5, 8]:
    raise RuntimeError("K is predeclared as 3/5/8 and must not be target-selected")
if distance != {
    "name": "tail_half_weighted_trajectory_rmse",
    "tail_half_start_fraction": 0.5,
    "first_half_weight": 1.0,
    "second_half_weight": 1.5,
}:
    raise RuntimeError("The approved trajectory-distance contract changed")
count_keys = ["lightgbm_config_count", "fold_count", "total_boosters"]
if any(execution.get(key) != 0 for key in count_keys):
    raise RuntimeError("LightGBM configs, folds, and boosters must all remain zero")
if execution.get("control_or_parent_retraining"):
    raise RuntimeError("Parent/control retraining is forbidden")
if execution.get("well_shard_count") != 1 or active_shard_index != 0:
    raise RuntimeError("exp243 v3 must run as one unsharded CPU notebook")
if kaggle_runtime.get("enable_gpu") or kaggle_runtime.get("enable_internet"):
    raise RuntimeError("exp243 is an offline CPU-only audit")

# %% [markdown]
# ## 3. Input cache and raw-data checks

# %%
train_dir = paths.train_data_dir
horizontal_files = sorted(train_dir.glob("*__horizontal_well.csv"))
typewell_files = sorted(train_dir.glob("*__typewell.csv"))
if not horizontal_files or not typewell_files:
    raise FileNotFoundError(f"Missing raw horizontal/typewell train files under {train_dir}")

validation_frame, validation_meta = read_exp072_eval_cache(config)
target_wells = select_target_wells(validation_frame, train_dir, config)
required_references = set(get_nested(config, "data.exp072_reference_candidates") or [])
present_references = set(validation_meta.get("reference_candidates_present") or [])

display(
    {
        "horizontal_files": len(horizontal_files),
        "typewell_files": len(typewell_files),
        "validation_rows": len(validation_frame),
        "validation_wells": validation_frame["well"].nunique(),
        "eligible_target_wells": len(target_wells),
        "exp072_cache": validation_meta.get("source"),
        "exp072_cache_decompressed_sha": validation_meta.get("source_decompressed_sha256"),
        "likpf_parity_control": validation_meta.get("likpf_parity_control"),
        "reference_candidates": sorted(present_references),
    }
)
display(target_wells.head(20))

if target_wells.empty:
    raise RuntimeError("No eligible exp072 pseudo-tail wells were selected")
if required_references - present_references:
    raise RuntimeError(
        f"Missing exp237 base8 reference columns: {sorted(required_references - present_references)}"
    )
if parity_probe and len(target_wells) != 1:
    raise RuntimeError("The parity probe must select exactly one well")
if not parity_probe and get_nested(config, "model.validation_surface.max_target_wells") is not None:
    raise RuntimeError("Canonical exp243 must cover all eligible wells in one notebook")

# %% [markdown]
# ## 4. PF replay and deterministic K-medoids contract

# %%
display(
    {
        "PF": "exp072 raw-GR Gaussian surface-state likelihood-PF",
        "seed_policy": get_nested(config, "reproducibility.seed_policy"),
        "trajectory_matrix": "128 seeds x evaluation rows, transient per well",
        "distance": "weighted trajectory RMSE; first half 1.0, second half 1.5",
        "medoid": "real PF seed trajectory minimizing within-cluster total distance",
        "candidate_order": clustering.get("candidate_order"),
        "fallback": "saved canonical exp072 v2 likpf_mean",
        "target_usage": "candidate generation後のdiagnostic scoringのみ",
        "forbidden": [
            "true TVT/error/oracle in clustering, K, or medoid selection",
            "centroid or medoid averaging as direct prediction",
            "target-based K selection",
            "selector training",
            "raw-test inference",
            "submission",
        ],
    }
)

# %% [markdown]
# ## 5. Full train-side candidate generation

# %%
result = run_pf_seed_medoids(
    config=config,
    paths=paths,
    validation_frame=validation_frame,
    validation_meta=validation_meta,
)

# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
print("Summary")
display(result["summary"])

print("Direct candidate metrics")
display(result["candidate_metrics"].head(80))

print("Row / block / whole-well oracle readout")
display(result["oracle_scope_metrics"])

print("Unique-best readout")
display(
    result["unique_best"].sort_values(
        ["bank", "unique_best_rate"], ascending=[True, False]
    )
)

print("Cluster summaries")
display(result["cluster_summary"].head(60))

print("1000+ and other distance buckets")
display(
    result["bucket_metrics"].sort_values(["distance_bucket", "rmse"]).head(120)
)

print("Hidden-like groups")
if result["hidden_like_metrics"].empty:
    print("Hidden-like fold assignments were not available")
else:
    display(result["hidden_like_metrics"].sort_values(["subgroup", "rmse"]).head(120))

print("Worst-well regressions")
display(
    result["by_well"].sort_values(
        "delta_rmse_vs_primary_baseline", ascending=False
    ).head(100)
)

print("Generated artifacts")
for artifact_name, artifact_path in result["summary"]["artifacts"].items():
    print(f"- {artifact_name}: {artifact_path}")
