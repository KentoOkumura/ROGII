# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # exp231 same-typewell horizontal GR atlas gated HMM emission — train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and result helpers
# 3. Setup and cost guard
# 4. Input and fold-safe atlas contract
# 5. Peer-atlas HMM generation
# 6. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exact_hmm_smoother import list_well_ids, to_jsonable
from feature_cache import resolve_cluster_assignment
from joint_cache_generation import run_joint_generation
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Runtime and result helpers

# %%
def print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"\n[{label}]")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def write_metrics(paths: ExperimentPaths, summary: dict[str, Any]) -> None:
    comparison = summary.get("comparison") or {}
    best = comparison.get("best_candidate") or {}
    hmm = summary.get("hmm") or {}
    is_preflight = bool(((hmm.get("well_generation") or {}).get("is_target_subset")))
    metrics = {
        "experiment": paths.experiment_name,
        "status": (
            "kaggle_preflight_completed_pending_review"
            if is_preflight
            else "kaggle_train_completed_pending_review"
        ),
        "metric": "train_side_rmse",
        "cv": best.get("rmse"),
        "public_lb": None,
        "private_lb": None,
        "best_candidate": best,
        "best_hmm_candidate": comparison.get("best_hmm_candidate"),
        "rows": summary.get("rows"),
        "wells": summary.get("wells"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "hmm_feature_content_sha256": ((hmm.get("sha256") or {}).get("train_features_decompressed")),
        "cluster_assignment": hmm.get("cluster_assignment"),
        "fold_assignment": hmm.get("fold_assignment"),
        "true_state_rank_metrics": comparison.get("true_state_rank_metrics"),
        "persistent_offset_onset": comparison.get("persistent_offset_onset"),
        "hidden_like_metrics_available": comparison.get("hidden_like_metrics_available"),
        "kernel": None,
        "kernel_version": None,
    }
    paths.metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    print_json("metrics.json", metrics)


# %% [markdown]
# ## 3. Setup and cost guard

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()
experiment = get_nested(config, "experiment") or {}
lineage = get_nested(config, "lineage") or {}
validation = get_nested(config, "validation") or {}
execution = get_nested(config, "execution") or {}
runtime = get_nested(config, "runtime") or {}
hmm_cache = get_nested(config, "feature_cache.hmm") or {}
peer_atlas = get_nested(config, "model.peer_atlas_emission") or {}

print_json(
    "experiment",
    {
        "name": experiment.get("name"),
        "route": experiment.get("route"),
        "status": experiment.get("status"),
        "parent": lineage.get("parent"),
        "references": lineage.get("references"),
        "train_dir": str(paths.train_data_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    },
)
print_json(
    "cost guard",
    {
        "active_generation_stages": {
            "exp072_full_cache": bool(execution.get("run_exp072_full_cache", False)),
            "peer_atlas_hmm_cache": bool(execution.get("run_hmm_cache", True)),
            "direct_comparison": bool(execution.get("run_direct_comparison", True)),
        },
        "peer_atlas_hmm_variants": [
            {"name": item.get("name"), "alpha": item.get("alpha")}
            for item in (peer_atlas.get("active_variants") or [])
        ],
        "preflight_target_wells": hmm_cache.get("preflight_target_wells") or None,
        "preflight_target_well_count": len(hmm_cache.get("preflight_target_wells") or []),
        "lightgbm_config_count": 0,
        "well_group_folds": validation.get("n_folds"),
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "saved_exp072_baseline_reused": True,
        "inference_or_submit": False,
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "hmm_outer_workers": hmm_cache.get("outer_workers"),
        "numba_num_threads": runtime.get("numba_num_threads"),
    },
)
print_json("hmm", get_nested(config, "model.hmm") or {})
print_json("peer atlas emission", peer_atlas)


# %% [markdown]
# ## 4. Input and fold-safe atlas contract

# %%
train_dir = paths.train_data_dir
if not train_dir.exists():
    raise FileNotFoundError(f"train data directory not found: {train_dir}")
wells = list_well_ids(train_dir)
if not wells:
    raise ValueError(f"no train wells with horizontal/typewell pairs found under {train_dir}")
cluster_assignment = resolve_cluster_assignment(paths, config)
if int(validation.get("n_folds", 0)) < 2:
    raise ValueError("peer-atlas validation requires at least two well folds")

print_json(
    "input and leakage contract",
    {
        "well_pairs": len(wells),
        "first_wells": wells[:5],
        "cluster_assignment": str(cluster_assignment),
        "cluster_assignment_exists": cluster_assignment.exists(),
        "group_method": peer_atlas.get("group_method"),
        "group_threshold": peer_atlas.get("group_threshold"),
        "validation_folds": validation.get("n_folds"),
        "target_well_policy": (
            "fixed preflight subset with full-fold atlas"
            if hmm_cache.get("preflight_target_wells")
            else "all wells"
        ),
        "fold_seed": validation.get("seed"),
        "source_policy": "same-typewell training-fold horizontal rows only",
        "validation_tvt_policy": "target and diagnostics only; never atlas construction or gating",
        "test_peer_joint_inference": False,
    },
)


# %% [markdown]
# ## 5. Peer-atlas HMM generation

# %%
summary = run_joint_generation()
print_json("joint generation summary", summary)


# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
write_metrics(paths, summary)
print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("*")):
    print(f"- {path.name}")
