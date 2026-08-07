# %% [markdown]
# # exp205 exact HMM smoother exp072 compatible cache audit train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and configuration
# 4. Input and feature contract checks
# 5. Run HMM train feature cache
# 6. Run exp072 direct comparison
# 7. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from direct_hmm_comparison import run_direct_comparison
from exact_hmm_smoother import list_well_ids, to_jsonable
from feature_cache import main as run_hmm_feature_cache
from settings import ExperimentPaths, get_nested, load_config


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def write_metrics(paths: ExperimentPaths, feature_summary: dict[str, Any], comparison_summary: dict[str, Any]) -> None:
    best = comparison_summary.get("best_candidate") or {}
    metrics = {
        "experiment": paths.experiment_name,
        "status": "kaggle_train_completed_pending_review",
        "metric": "rmse_tvt",
        "cv": best.get("rmse"),
        "public_lb": None,
        "private_lb": None,
        "best_candidate": best,
        "rows": feature_summary.get("rows"),
        "wells": feature_summary.get("wells"),
        "feature_content_sha256": (feature_summary.get("sha256") or {}).get("train_features_decompressed"),
        "comparison_summary": comparison_summary.get("artifacts", {}).get("summary"),
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
model = get_nested(config, "model") or {}
feature_cache = get_nested(config, "feature_cache") or {}
comparison = get_nested(config, "comparison") or {}
runtime = get_nested(config, "runtime") or {}

print_json(
    "experiment",
    {
        "name": experiment.get("name"),
        "route": experiment.get("route"),
        "status": experiment.get("status"),
        "parent": get_nested(config, "lineage.parent"),
        "reference": "exp072_exp063_full_replay_feature_cache",
        "train_dir": str(paths.train_data_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    },
)
print_json(
    "cost guard",
    {
        "active_feature_cache_variants": [feature_cache.get("variant")],
        "lightgbm_config_count": 0,
        "fold_count": 0,
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "inference_or_submit": False,
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "numba_num_threads": runtime.get("numba_num_threads"),
    },
)
print_json("hmm", model.get("hmm") or {})


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
        "expected_hmm_feature_count": feature_cache.get("expected_feature_count"),
        "baseline_feature_cache_candidates": comparison.get("baseline_feature_cache"),
        "hmm_feature_cache_candidates": comparison.get("hmm_feature_cache"),
    },
)


# %% [markdown]
# ## 5. Run HMM train feature cache

# %%
feature_summary = run_hmm_feature_cache()
print_json("feature summary", feature_summary)


# %% [markdown]
# ## 6. Run exp072 direct comparison

# %%
comparison_summary = run_direct_comparison()
print_json("comparison summary", comparison_summary)


# %% [markdown]
# ## 7. Metrics and generated artifacts

# %%
write_metrics(paths, feature_summary, comparison_summary)
print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("exp205*")):
    print(f"- {path.name}")
