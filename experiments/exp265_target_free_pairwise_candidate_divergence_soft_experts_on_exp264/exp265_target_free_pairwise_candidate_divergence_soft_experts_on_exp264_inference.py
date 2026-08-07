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
# # exp265 target-free pairwise candidate divergence soft experts — inference
#
# exp265はStage 0 separability audit専用であり、current-test inferenceやsubmissionを
# 生成しない。Stage 0の全guard通過とconditional Stage 1の別承認後にのみ実装対象とする。

# %%
from settings import ExperimentPaths, load_config

paths = ExperimentPaths()
paths.require_kaggle_runtime()
config = load_config()

assert config["experiment"]["route"] == "ensemble"
assert config["execution"]["inference_enabled"] is False
assert config["model"]["conditional_stage1"]["enabled"] is False

raise RuntimeError(
    "exp265 inference is intentionally disabled: Stage 0 has no trained expert models or submission."
)
