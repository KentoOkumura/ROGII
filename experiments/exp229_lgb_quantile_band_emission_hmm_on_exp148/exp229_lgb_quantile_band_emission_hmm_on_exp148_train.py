# %% [markdown]
# # exp229_lgb_quantile_band_emission_hmm_on_exp148 train
#
# Train q16/q50/q84 LightGBM quantile models on the exp148 feature surface and
# write an OOF quantile band for the downstream HMM audit notebook.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and push guard
# 4. Input and feature contract
# 5. Quantile LightGBM execution
# 6. Metrics and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quantile_lgb_train import run_quantile_lgb_train
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def env_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def enabled_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = list(get_nested(config, "model.feature_ablation.active_variants") or [])
    return [variant for variant in variants if variant.get("enabled", True)]


def active_mode_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mode_map = dict(get_nested(config, "model.training.modes") or {})
    active_modes = list(get_nested(config, "model.training.active_modes") or mode_map)
    return {name: dict(mode_map[name]) for name in active_modes}


def notebook_push_guard(config: dict[str, Any]) -> dict[str, Any]:
    variants = enabled_variants(config)
    modes = active_mode_configs(config)
    alphas = list(get_nested(config, "model.quantile_lgb.alphas") or [])
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    config_count = 0
    mode_details: dict[str, Any] = {}
    for mode_name, mode_config in modes.items():
        selected = list(mode_config.get("selected_config_indices") or [1])
        config_count += len(selected)
        mode_details[mode_name] = {
            "selected_config_indices": selected,
            "use_gpu": bool(mode_config.get("use_gpu", False)),
        }
    booster_count = len(variants) * max(1, config_count) * len(alphas) * n_folds
    return {
        "experiment": EXPERIMENT_NAME,
        "active_variant_count": len(variants),
        "active_modes": list(modes),
        "lightgbm_config_count": config_count,
        "quantile_alpha_count": len(alphas),
        "fold_count": n_folds,
        "planned_booster_count": booster_count,
        "parent_control_retraining": False,
        "mode_details": mode_details,
        "alphas": alphas,
    }


# %% [markdown]
# ## 3. Setup and push guard

# %%
DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
MAX_ROWS = env_int("EXPERIMENT_MAX_ROWS")
MAX_TRAIN_ROWS = env_int("EXPERIMENT_MAX_TRAIN_ROWS")

paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

print("Experiment:", EXPERIMENT_NAME)
print("Root:", paths.root)
print("Train data:", paths.train_data_dir)
print("Artifacts:", paths.artifacts_dir)
print("Debug:", DEBUG, "MAX_ROWS:", MAX_ROWS, "MAX_TRAIN_ROWS:", MAX_TRAIN_ROWS)
print(json.dumps(notebook_push_guard(config), indent=2, sort_keys=True))

# %% [markdown]
# ## 4. Input and feature contract

# %%
variants = enabled_variants(config)
if len(variants) != 1:
    raise ValueError(f"expected exactly one active quantile variant, got {len(variants)}")
variant = variants[0]

modes = active_mode_configs(config)
if len(modes) != 1:
    raise ValueError(f"expected exactly one active quantile training mode, got {list(modes)}")
mode_name, mode_config = next(iter(modes.items()))

quantile_config = dict(get_nested(config, "model.quantile_lgb") or {})
projection_config = dict(get_nested(config, "model.u_projection") or {})
learned_feature_config = dict(get_nested(config, "model.learned_likelihood_features") or {})
data_config = dict(config.get("data") or {})

print("Variant:", json.dumps(variant, indent=2, sort_keys=True))
print("Mode:", mode_name, json.dumps(mode_config, indent=2, sort_keys=True))
print("Quantile config:", json.dumps(quantile_config, indent=2, sort_keys=True))

for path_label, path_value in {
    "train_data_dir": paths.train_data_dir,
    "raw_data_dir": paths.raw_data_dir,
}.items():
    print(path_label, path_value, "exists=", Path(path_value).exists())

# %% [markdown]
# ## 5. Quantile LightGBM execution

# %%
summary = run_quantile_lgb_train(
    output_dir=paths.artifacts_dir,
    train_dir=paths.train_data_dir,
    cache_path=None,
    learned_feature_path=None,
    learned_schema_path=data_config.get("learned_likelihood_rawtest_feature_schema_local"),
    learned_summary_path=data_config.get("learned_likelihood_rawtest_summary_local"),
    projection_config=projection_config,
    learned_feature_config=learned_feature_config,
    variant=variant,
    mode_name=mode_name,
    mode_config=mode_config,
    quantile_config=quantile_config,
    n_splits=int(get_nested(config, "validation.n_folds") or 5),
    fast=DEBUG,
    max_rows=MAX_ROWS,
    max_train_rows=MAX_TRAIN_ROWS or quantile_config.get("max_train_rows"),
    save_models=bool(quantile_config.get("save_models", True)),
    save_predictions=bool(quantile_config.get("save_predictions", True)),
    top_n_importance=int(quantile_config.get("top_n_importance", 50)),
)

# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
metrics = {
    "experiment": EXPERIMENT_NAME,
    "status": summary["status"],
    "updated_at": datetime.now(UTC).isoformat(),
    "cv": summary["band_summary"]["corrected_q_mid_rmse"],
    "public_lb": None,
    "private_lb": None,
    "metric": "rmse",
    "seed": get_nested(config, "validation.seed"),
    "route": get_nested(config, "experiment.route"),
    "planned_booster_count": notebook_push_guard(config)["planned_booster_count"],
    "actual_booster_count": summary["booster_count"],
    "band_summary": summary["band_summary"],
    "artifacts": summary["artifacts"],
    "sha256": summary["sha256"],
}
paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
print("Metrics written:", paths.metrics_path)
