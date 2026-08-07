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
# # exp224_well_scaled_z_dz_features_on_exp218 inference
#
# Saved-booster inference for the exp218 add-only LightGBM surface with
# well-scaled z/dz/dzdmd/slp_z features. This regenerates current-test replay,
# learned likelihood, GRWR, and well-scaled features, then loads the exp224
# train boosters.

# %% [markdown]
# ## Contents
#
# 1. Setup and selected inference contract
# 2. Saved-booster inference and submission generation
# 3. Metrics and generated artifacts

# %%
from __future__ import annotations

import json

from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config
from well_scaled_z_dz_features_on_exp218 import run_saved_model_inference


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
    "well_scaled_z_dz_addonly",
)
selected_mode = cfg_get(config, "inference.selected_mode", "cpu_deterministic_threads8")
selected_model = cfg_get(config, "inference.selected_model", "lgb_mean")

print("Experiment:", EXPERIMENT_NAME)
print("Route:", cfg_get(config, "experiment.route"))
print("Selected variant:", selected_variant)
print("Selected mode:", selected_mode)
print("Selected model:", selected_model)
print("Parent:", cfg_get(config, "lineage.parent"))
print("Train kernel sources:", cfg_get(config, "runtime.kaggle.inference_kernel_sources"))
print("Split-train note:", "requires aggregate manifest or explicit lgb0/lgb1/lgb2 handling")
print("Raw-test learned likelihood features:", "generated from current test if cache keys differ")
print("GRWR current-test features:", "generated from raw test horizontal/typewell GR")
print("Well-scaled z/dz current-test features:", "generated from raw-test feature frame")

# %% [markdown]
# ## 2. Saved-booster inference and submission generation

# %%
summary = run_saved_model_inference(
    output_dir=paths.artifacts_dir,
    submission_path=paths.submission_path,
    sample_submission_path=paths.sample_submission_path,
    data_dir=paths.raw_data_dir,
    test_dir=paths.test_data_dir,
    learned_feature_path=None,
    learned_schema_path=cfg_get(config, "data.learned_likelihood_rawtest_feature_schema_local"),
    learned_summary_path=cfg_get(config, "data.learned_likelihood_rawtest_summary_local"),
    projection_config=cfg_get(config, "model.u_projection", {}),
    learned_feature_config=cfg_get(config, "model.learned_likelihood_features", {}),
    grwr_feature_config=cfg_get(config, "model.gr_wavelet_rotation_confidence_features", {}),
    well_scaled_feature_config=cfg_get(config, "model.well_scaled_z_dz_features", {}),
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

# %% [markdown]
# ## 3. Metrics and generated artifacts

# %%
paths.metrics_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
)
print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
print("Submission:", paths.submission_path)
