# %% [markdown]
# # exp229_lgb_quantile_band_emission_hmm_on_exp148 train_aggregate
#
# Read the q16/q50/q84 OOF band from the train notebook, convert band width to
# row-wise HMM emission sigma, and run the exp221-style train-side HMM audit.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and audit guard
# 4. Input and HMM contract
# 5. Quantile-band HMM execution
# 6. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from quantile_band_hmm_audit import run_quantile_band_hmm_audit
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def audit_guard(config: dict[str, Any]) -> dict[str, Any]:
    lgb_emission = dict(get_nested(config, "lgb_emission") or {})
    active_sources = list(lgb_emission.get("active_sources") or [])
    lambda_grid = list(lgb_emission.get("lambda_grid") or [])
    floor_grid = list(lgb_emission.get("sigma_floor_grid") or [])
    cap_grid = list(lgb_emission.get("sigma_cap_grid") or [])
    variant_count = len(active_sources) * len(lambda_grid) * max(1, len(floor_grid)) * max(1, len(cap_grid))
    return {
        "experiment": EXPERIMENT_NAME,
        "active_sources": active_sources,
        "lambda_grid": lambda_grid,
        "sigma_floor_grid": floor_grid,
        "sigma_cap_grid": cap_grid,
        "planned_hmm_variant_count": variant_count,
        "lightgbm_config_count": 0,
        "fold_count": 0,
        "planned_booster_count": 0,
        "parent_control_retraining": False,
    }


# %% [markdown]
# ## 3. Setup and audit guard

# %%
DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
MAX_WELLS = env_int("EXPERIMENT_MAX_WELLS")

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Root:", paths.root)
print("Train data:", paths.train_data_dir)
print("Artifacts:", paths.artifacts_dir)
print("Debug:", DEBUG, "MAX_WELLS:", MAX_WELLS)
print(json.dumps(audit_guard(config), indent=2, sort_keys=True))

# %% [markdown]
# ## 4. Input and HMM contract

# %%
hmm_config = dict(get_nested(config, "model.hmm") or {})
lgb_emission = dict(get_nested(config, "lgb_emission") or {})
feature_cache = dict(get_nested(config, "feature_cache.hmm") or {})
comparison = dict(get_nested(config, "comparison") or {})

if not bool(lgb_emission.get("enabled", False)):
    raise ValueError("lgb_emission.enabled must be true")
if not lgb_emission.get("active_sources"):
    raise ValueError("lgb_emission.active_sources is empty")

print("HMM config:", json.dumps(hmm_config, indent=2, sort_keys=True))
print("LGB emission:", json.dumps({k: v for k, v in lgb_emission.items() if k != "sources"}, indent=2, sort_keys=True))
print("HMM feature cache:", json.dumps(feature_cache, indent=2, sort_keys=True))
print("Comparison output prefix:", comparison.get("output_prefix"))

# %% [markdown]
# ## 5. Quantile-band HMM execution

# %%
summary = run_quantile_band_hmm_audit(
    max_wells=MAX_WELLS,
    fast=DEBUG,
)

# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
comparison_summary = summary["comparison_summary"]
best_hmm = comparison_summary.get("best_hmm_lgb_candidate") or {}
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "updated_at": datetime.now(UTC).isoformat(),
    "cv": best_hmm.get("rmse"),
    "public_lb": None,
    "private_lb": None,
    "metric": "rmse",
    "seed": get_nested(config, "validation.seed"),
    "route": get_nested(config, "experiment.route"),
    "planned_booster_count": audit_guard(config)["planned_booster_count"],
    "best_hmm_lgb_candidate": best_hmm,
    "best_candidate": comparison_summary.get("best_candidate"),
    "primary_baselines": comparison_summary.get("primary_baselines"),
    "artifacts": comparison_summary.get("artifacts"),
    "sha256": summary.get("sha256"),
}
paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
print("Metrics written:", paths.metrics_path)
