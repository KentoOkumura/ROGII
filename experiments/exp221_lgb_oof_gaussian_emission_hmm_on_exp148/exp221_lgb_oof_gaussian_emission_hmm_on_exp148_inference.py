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
# # exp221 lgb_oof_gaussian_emission_hmm_on_exp148 inference

# %% [markdown]
# ## Contents
# 1. Imports and runtime checks
# 2. Configuration
# 3. Generate exp148 current-test LightGBM predictions
# 4. Run LGB-emission HMM inference
# 5. Save metrics

# %%
from __future__ import annotations

import json
import time
from typing import Any

from exact_hmm_smoother import run_lgb_emission_hmm_inference, to_jsonable
from learned_likelihood_fulltrain_addonly_on_exp092 import (
    run_saved_model_inference as run_exp148_saved_model_inference,
)
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config


def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
inference = get_nested(config, "inference") or {}
runtime = get_nested(config, "runtime") or {}

print_json(
    "runtime",
    {
        "experiment": paths.experiment_name,
        "route": get_nested(config, "experiment.route"),
        "mode": inference.get("mode"),
        "selected_candidate": inference.get("selected_candidate"),
        "data_dir": str(paths.raw_data_dir),
        "test_dir": str(paths.test_data_dir),
        "sample_submission": str(paths.sample_submission_path),
        "artifacts_dir": str(paths.artifacts_dir),
        "submission_path": str(paths.submission_path),
    },
)

# %% [markdown]
# ## 2. Configuration

# %%
selected_candidate = str(
    inference.get("selected_candidate") or "hmm_lgb_exp148_lgb_mean_s2000_l0500"
)
selected_source = str(inference.get("selected_source") or "exp148_lgb_mean")
exp148_variant = str(inference.get("exp148_variant") or "learned_likelihood_confidence_addonly")
exp148_mode = str(inference.get("exp148_mode") or "gpu_repro_guard_dp_threads8")
exp148_model = str(inference.get("exp148_model") or "lgb_mean")
exp148_use_gpu = str(inference.get("exp148_use_gpu") or "auto")
strict_sample_ids = bool(inference.get("strict_sample_ids", True))
output_prefix = str(get_nested(config, "feature_cache.hmm.output_prefix") or EXPERIMENT_NAME)
outer_workers = int(get_nested(config, "feature_cache.hmm.outer_workers") or 1)
numba_num_threads = get_nested(config, "runtime.numba_num_threads")
numba_num_threads = int(numba_num_threads) if numba_num_threads else None

print_json(
    "selected settings",
    {
        "selected_candidate": selected_candidate,
        "selected_source": selected_source,
        "exp148_variant": exp148_variant,
        "exp148_mode": exp148_mode,
        "exp148_model": exp148_model,
        "exp148_use_gpu": exp148_use_gpu,
        "outer_workers": outer_workers,
        "numba_num_threads": numba_num_threads,
        "strict_sample_ids": strict_sample_ids,
    },
)

# %% [markdown]
# ## 3. Generate exp148 current-test LightGBM predictions

# %%
t0 = time.time()
exp148_proxy_submission = paths.artifacts_dir / "exp148_proxy_submission.csv"
exp148_summary = run_exp148_saved_model_inference(
    output_dir=paths.artifacts_dir,
    submission_path=exp148_proxy_submission,
    sample_submission_path=paths.sample_submission_path,
    data_dir=paths.raw_data_dir,
    test_dir=paths.test_data_dir,
    model_manifest_path=None,
    learned_feature_path=None,
    learned_schema_path=get_nested(config, "data.learned_likelihood_rawtest_feature_schema_local"),
    learned_summary_path=get_nested(config, "data.learned_likelihood_rawtest_summary_local"),
    projection_config=get_nested(config, "model.u_projection") or {},
    learned_feature_config=get_nested(config, "model.learned_likelihood_features") or {},
    variant_name=exp148_variant,
    mode_name=exp148_mode,
    model_name=exp148_model,
    submission_target_column=str(get_nested(config, "data.submission_target_column") or "tvt"),
    n_jobs=int(get_nested(config, "generator.rawtest_replay.n_jobs") or 8),
    pf_seeds=int(get_nested(config, "generator.rawtest_replay.pf_seeds") or 128),
    pf_particles=int(get_nested(config, "generator.rawtest_replay.pf_particles") or 500),
    fast=bool(get_nested(config, "generator.rawtest_replay.fast") or False),
    use_gpu=exp148_use_gpu,
)
exp148_prediction_name = str((exp148_summary.get("artifacts") or {}).get("predictions"))
if not exp148_prediction_name:
    raise ValueError("exp148 saved-model inference did not report a predictions artifact")
exp148_prediction_path = paths.artifacts_dir / exp148_prediction_name
if not exp148_prediction_path.exists():
    raise FileNotFoundError(f"exp148 predictions not found: {exp148_prediction_path}")

print_json(
    "exp148 proxy inference",
    {
        "prediction_path": str(exp148_prediction_path),
        "metrics": exp148_summary.get("metrics"),
        "elapsed_seconds": exp148_summary.get("elapsed_seconds"),
    },
)

# %% [markdown]
# ## 4. Run LGB-emission HMM inference

# %%
lgb_emission_config = dict(get_nested(config, "lgb_emission") or {})
lgb_emission_config["active_sources"] = [selected_source]
lgb_sources = dict(lgb_emission_config.get("sources") or {})
lgb_sources[selected_source] = {
    "description": "Current-test exp148 lgb_mean predictions generated inside exp221 inference.",
    "id_column": "id",
    "prediction_column": "pred_tvt",
    "model_filter": exp148_model,
    "candidates": [str(exp148_prediction_path)],
}
lgb_emission_config["sources"] = lgb_sources

hmm_summary = run_lgb_emission_hmm_inference(
    root=paths.root,
    data_dir=paths.test_data_dir,
    output_dir=paths.artifacts_dir,
    submission_path=paths.submission_path,
    sample_submission_path=paths.sample_submission_path,
    hmm_config=get_nested(config, "model.hmm") or {},
    lgb_emission_config=lgb_emission_config,
    output_prefix=output_prefix,
    selected_candidate=selected_candidate,
    strict_sample_ids=strict_sample_ids,
    submission_target_column=str(get_nested(config, "data.submission_target_column") or "tvt"),
    max_wells=get_nested(config, "feature_cache.hmm.max_wells"),
    fast=bool(get_nested(config, "feature_cache.hmm.fast") or False),
    numba_num_threads=numba_num_threads,
    outer_workers=outer_workers,
)

# %% [markdown]
# ## 5. Save metrics

# %%
combined_summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "inference_completed",
    "mode": inference.get("mode"),
    "selected_candidate": selected_candidate,
    "exp148_proxy": {
        "variant": exp148_variant,
        "mode": exp148_mode,
        "model": exp148_model,
        "prediction_path": str(exp148_prediction_path),
        "metrics": exp148_summary.get("metrics"),
        "sha256": exp148_summary.get("sha256"),
    },
    "hmm_inference": {
        "metrics": hmm_summary.get("metrics"),
        "outputs": hmm_summary.get("outputs"),
        "sha256": hmm_summary.get("sha256"),
    },
    "artifacts": {
        "metrics_json": str(paths.metrics_path),
        "submission": str(paths.submission_path),
    },
    "elapsed_seconds": round(time.time() - t0, 3),
}
paths.metrics_path.write_text(json.dumps(to_jsonable(combined_summary), indent=2, sort_keys=True))
print_json("combined inference summary", combined_summary)
