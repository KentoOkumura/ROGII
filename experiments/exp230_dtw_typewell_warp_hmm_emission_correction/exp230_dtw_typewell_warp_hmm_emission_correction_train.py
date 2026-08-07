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
# # exp230 dtw typewell warp hmm emission correction train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and configuration
# 4. Input and feature contract checks
# 5. DTW-emission HMM cache generation
# 6. Metrics, diagnostics, and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exact_hmm_smoother import list_well_ids, to_jsonable
from joint_cache_generation import run_joint_generation
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def write_metrics(paths: ExperimentPaths, summary: dict[str, Any]) -> None:
    comparison = summary.get("comparison") or {}
    best = comparison.get("best_candidate") or {}
    exp072 = summary.get("exp072") or {}
    hmm = summary.get("hmm") or {}
    exp072_sha = ((exp072.get("sha256") or {}).get("train_features_decompressed"))
    hmm_sha = ((hmm.get("sha256") or {}).get("train_features_decompressed"))
    metrics = {
        "experiment": paths.experiment_name,
        "status": "kaggle_train_completed",
        "metric": "train_side_rmse",
        "cv": best.get("rmse"),
        "public_lb": None,
        "private_lb": None,
        "best_candidate": best,
        "rows": summary.get("rows"),
        "wells": summary.get("wells"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "exp072_feature_content_sha256": exp072_sha,
        "hmm_feature_content_sha256": hmm_sha,
        "comparison_load_mode": (comparison.get("baseline_load_mode"), comparison.get("hmm_load_mode")),
        "best_hmm_candidate": comparison.get("best_hmm_candidate"),
        "hidden_like_metrics_available": comparison.get("hidden_like_metrics_available"),
        "joint_summary": (summary.get("outputs") or {}).get("joint_summary"),
        "kernel": None,
        "kernel_version": None,
    }
    paths.metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    print_json("metrics.json", metrics)


# %% [markdown]
# ## 3. Setup and configuration

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()

experiment = get_nested(config, "experiment") or {}
lineage = get_nested(config, "lineage") or {}
exp072_cache = get_nested(config, "feature_cache.exp072") or {}
hmm_cache = get_nested(config, "feature_cache.hmm") or {}
execution = get_nested(config, "execution") or {}
runtime = get_nested(config, "runtime") or {}
dtw_emission = get_nested(config, "model.dtw_emission") or {}

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
            "exp072_full_cache": bool(execution.get("run_exp072_full_cache", True)),
            "hmm_cache": bool(execution.get("run_hmm_cache", True)),
            "direct_comparison": bool(execution.get("run_direct_comparison", True)),
        },
        "lightgbm_config_count": 0,
        "fold_count": 0,
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "inference_or_submit": False,
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "dtw_hmm_variant_count": len(dtw_emission.get("active_variants") or []),
        "hmm_outer_workers": hmm_cache.get("outer_workers"),
        "numba_num_threads": runtime.get("numba_num_threads"),
    },
)
print_json("execution", execution)
print_json("hmm", get_nested(config, "model.hmm") or {})
print_json("dtw emission", dtw_emission)


# %% [markdown]
# ## 4. Input and feature contract checks

# %%
train_dir = paths.train_data_dir
if not train_dir.exists():
    raise FileNotFoundError(f"train data directory not found: {train_dir}")
wells = list_well_ids(train_dir)
if not wells:
    raise ValueError(f"no train wells with horizontal/typewell pairs found under {train_dir}")

print_json(
    "input contract",
    {
        "well_pairs": len(wells),
        "first_wells": wells[:5],
        "expected_exp072_feature_count": exp072_cache.get("expected_feature_count"),
        "expected_hmm_feature_count": hmm_cache.get("expected_feature_count"),
        "exp072_variant": exp072_cache.get("variant"),
        "hmm_variant": hmm_cache.get("variant"),
        "comparison_baseline_mode": (
            "in_memory"
            if bool(execution.get("direct_comparison_use_in_memory_exp072", True))
            else "csv_gzip"
        ),
    },
)


# %% [markdown]
# ## 5. DTW-emission HMM cache generation

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
