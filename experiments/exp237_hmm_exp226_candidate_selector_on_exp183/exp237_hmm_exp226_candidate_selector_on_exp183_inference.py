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
# # exp237_hmm_exp226_candidate_selector_on_exp183 inference
#
# User-authorized raw-test inference for the fixed exp237 candidate-error ranker
# and its one pre-fixed Viterbi continuity rule. This notebook writes an inference
# artifact only; it never calls the Kaggle competition submission API.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime, approval, and source contracts
# 3. Raw-test candidate and ranker inference
# 4. Generated artifact checks

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json

import pandas as pd
from IPython.display import display
from rawtest_inference import run_rawtest_inference
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime, approval, and source contracts

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

inference = get_nested(config, "inference") or {}
runtime = get_nested(config, "runtime.kaggle") or {}
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(config, "experiment.route"))
print("Experiment status:", get_nested(config, "experiment.status"))
print("Inference mode:", inference.get("mode"))
print("Selected fixed Viterbi:", inference.get("selected_variant"))
print("GPU enabled:", runtime.get("enable_gpu"))
print("Internet enabled:", runtime.get("enable_internet"))
print("Output artifacts:", paths.artifacts_dir)
display(
    {
        "approval": inference.get("notes"),
        "rawtest_feature_policy": inference.get("rawtest_feature_policy"),
        "source_contract": inference.get("source_contract"),
        "candidate_names": [
            item.get("name") for item in get_nested(config, "ranker.candidates") or []
        ],
        "kernel_sources": runtime.get("inference_kernel_sources"),
    }
)

# %% [markdown]
# ## 3. Raw-test candidate and ranker inference
#
# The notebook reuses the fixed train-side models without retraining. It reads the
# deterministic exp073 raw-test PF/Beam cache and exp226 K16 prediction, regenerates
# the two HMM paths and exp099 scorer target-free, reconstructs each saved fold's
# training medians, then applies the single fixed Viterbi rule.

# %%
summary = run_rawtest_inference(config=config, paths=paths)
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

# %% [markdown]
# ## 4. Generated artifact checks

# %%
artifacts = summary["artifacts"]
selected = pd.read_csv(paths.artifacts_dir / artifacts["selected_predictions"], compression="gzip")
submission = pd.read_csv(paths.submission_path)

print("selected prediction rows:", len(selected))
print("submission rows:", len(submission))
print("candidate distribution")
display(selected["selected_candidate"].value_counts(dropna=False).rename("rows").to_frame())
print("feature coverage")
display(summary["feature_coverage"])
print("prediction preview")
display(selected.head(20))
print("submission preview")
display(submission.head(20))
