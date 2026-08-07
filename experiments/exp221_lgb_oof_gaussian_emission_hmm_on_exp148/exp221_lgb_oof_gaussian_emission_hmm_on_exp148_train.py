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
# # exp221 lgb_oof_gaussian_emission_hmm_on_exp148 train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and cost guard
# 4. Input and OOF source checks
# 5. LGB Gaussian-emission HMM generation
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


def configured_lgb_variant_count(config: dict[str, Any]) -> int:
    lgb = get_nested(config, "lgb_emission") or {}
    if not bool(lgb.get("enabled", False)):
        return 0
    active_sources = list(lgb.get("active_sources") or [])
    sigma_grid = list(lgb.get("sigma_grid") or [])
    lambda_grid = list(lgb.get("lambda_grid") or [])
    count = len(active_sources) * len(sigma_grid) * len(lambda_grid)
    max_variants = lgb.get("max_variants")
    return min(count, int(max_variants)) if max_variants is not None else count


def write_metrics(paths: ExperimentPaths, summary: dict[str, Any]) -> None:
    comparison = summary.get("comparison") or {}
    hmm = summary.get("hmm") or {}
    best = comparison.get("best_candidate") or {}
    best_hmm = comparison.get("best_hmm_lgb_candidate") or {}
    metrics = {
        "experiment": paths.experiment_name,
        "status": "implemented_pending_kaggle_review",
        "route": "ensemble",
        "metric": "train_side_oof_rmse_tvt",
        "cv": best_hmm.get("rmse"),
        "public_lb": None,
        "private_lb": None,
        "best_candidate": best,
        "best_hmm_lgb_candidate": best_hmm,
        "rows": summary.get("rows"),
        "wells": summary.get("wells"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "hmm_feature_content_sha256": (hmm.get("sha256") or {}).get("train_features_decompressed"),
        "lgb_emission": (hmm.get("lgb_emission") or {}),
        "comparison_load_mode": (comparison.get("baseline_load_mode"), comparison.get("hmm_load_mode")),
        "hidden_like_metrics_available": comparison.get("hidden_like_metrics_available"),
        "joint_summary": (summary.get("outputs") or {}).get("joint_summary"),
        "kernel": None,
        "kernel_version": None,
        "submitted": False,
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
execution = get_nested(config, "execution") or {}
hmm_cache = get_nested(config, "feature_cache.hmm") or {}
lgb_emission = get_nested(config, "lgb_emission") or {}
comparison = get_nested(config, "comparison") or {}
runtime = get_nested(config, "runtime") or {}

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
        "active_hmm_lgb_variants": configured_lgb_variant_count(config),
        "lightgbm_config_count": 0,
        "fold_count": 0,
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "inference_or_submit": False,
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "run_exp072_full_cache": bool(execution.get("run_exp072_full_cache", False)),
        "hmm_outer_workers": hmm_cache.get("outer_workers"),
        "numba_num_threads": runtime.get("numba_num_threads"),
    },
)
print_json("execution", execution)
print_json("hmm", get_nested(config, "model.hmm") or {})
print_json(
    "lgb emission grid",
    {
        "active_sources": lgb_emission.get("active_sources"),
        "sigma_grid": lgb_emission.get("sigma_grid"),
        "lambda_grid": lgb_emission.get("lambda_grid"),
        "max_variants": lgb_emission.get("max_variants"),
        "deferred_full_grid": lgb_emission.get("deferred_full_grid"),
    },
)


# %% [markdown]
# ## 4. Input and OOF source checks

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
        "hmm_variant": hmm_cache.get("variant"),
        "expected_hmm_feature_count": hmm_cache.get("expected_feature_count"),
        "baseline_feature_cache_candidates": comparison.get("baseline_feature_cache"),
        "hmm_feature_cache_candidates": comparison.get("hmm_feature_cache"),
        "lgb_baseline_sources": comparison.get("lgb_baseline_sources"),
        "hidden_like_enabled": (comparison.get("hidden_like") or {}).get("enabled"),
    },
)


# %% [markdown]
# ## 5. LGB Gaussian-emission HMM generation

# %%
summary = run_joint_generation()
print_json("generation and readout summary", summary)


# %% [markdown]
# ## 6. Metrics, diagnostics, and generated artifacts

# %%
write_metrics(paths, summary)
print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("*")):
    print(f"- {path.name}")
