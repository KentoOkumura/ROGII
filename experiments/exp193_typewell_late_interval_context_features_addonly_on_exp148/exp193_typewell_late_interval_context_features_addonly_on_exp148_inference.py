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
# # exp193_typewell_late_interval_context_features_addonly_on_exp148 inference
#
# Saved-booster inference for the exp193 add-only typewell late-interval context model.
# The notebook rebuilds raw-test replay features, learned-likelihood features, and the
# 19 target-free typewell context features before writing `submission.csv`.

# %% [markdown]
# ## Contents
#
# 1. Runtime and selected inference contract
# 2. Train manifest checks
# 3. Saved-booster inference and submission generation

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from typewell_late_interval_context_features_addonly_on_exp148 import (
    find_model_manifest,
    run_saved_model_inference,
)


def cfg_get(config, dotted_key, default=None):
    value = get_nested(config, dotted_key)
    return default if value is None else value


paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()

selected_variant = cfg_get(
    config,
    "inference.selected_variant",
    "typewell_late_interval_context_addonly",
)
selected_mode = cfg_get(config, "inference.selected_mode", "gpu_repro_guard_dp_threads8")
selected_model = cfg_get(config, "inference.selected_model", "lgb_mean")

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Inference mode:", cfg_get(config, "inference.mode"))
print("Selected variant:", selected_variant)
print("Selected mode:", selected_mode)
print("Selected model:", selected_model)
print("Raw-test learned likelihood features:", "generated from current test frame if cache is absent")
print("Raw-test typewell context features:", "generated from current test typewell files")
print("Submission path:", paths.submission_path)

# %% [markdown]
# ## 2. Train manifest checks

# %%
manifest_path = find_model_manifest(cfg_get(config, "inference.model_manifest_path"))
manifest = json.loads(manifest_path.read_text())
selected_model_rows = [
    item
    for item in manifest.get("models", [])
    if str(item.get("variant")) == selected_variant
    and str(item.get("mode")) == selected_mode
    and (selected_model == "lgb_mean" or str(item.get("model")) == selected_model)
]
typewell_groups = manifest.get("typewell_context_feature_groups") or {}
print("Train manifest:", manifest_path)
print("Selected saved booster rows:", len(selected_model_rows))
print("Typewell context feature count:", sum(len(value) for value in typewell_groups.values()))
print("Enabled variants:", [item.get("name") for item in manifest.get("variants", []) if item.get("enabled", True)])
if not selected_model_rows:
    raise ValueError("No saved booster rows match the selected inference contract.")

# %% [markdown]
# ## 3. Saved-booster inference and submission generation

# %%
summary = run_saved_model_inference(
    output_dir=paths.artifacts_dir,
    submission_path=paths.submission_path,
    sample_submission_path=paths.sample_submission_path,
    data_dir=paths.raw_data_dir,
    test_dir=paths.test_data_dir,
    model_manifest_path=manifest_path,
    learned_feature_path=None,
    learned_schema_path=cfg_get(config, "data.learned_likelihood_rawtest_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_rawtest_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    typewell_context_feature_config=cfg_get(
        config,
        "model.typewell_late_interval_context_features",
        {},
    ),
    variant_name=selected_variant,
    mode_name=selected_mode,
    model_name=selected_model,
    submission_target_column=cfg_get(config, "data.submission_target_column", "tvt"),
    n_jobs=cfg_get(
        config,
        "generator.rawtest_replay.n_jobs",
        cfg_get(config, "runtime.num_workers", 8),
    ),
    pf_seeds=cfg_get(config, "generator.rawtest_replay.pf_seeds", 128),
    pf_particles=cfg_get(config, "generator.rawtest_replay.pf_particles", 500),
    fast=bool(cfg_get(config, "audit.fast", False)),
)
paths.metrics_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
)
print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
print("Submission:", paths.submission_path)
