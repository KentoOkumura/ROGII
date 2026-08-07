# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp303 inference is intentionally disabled
#
# exp303 is an OOF selectability readout. It must not create a gate,
# corrected prediction, current-test inference, or submission.

# %%
from pathlib import Path

import yaml

config_path = Path.cwd() / "config.yaml"
if not config_path.exists():
    config_path = (
        Path.cwd()
        / "experiments"
        / "exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264"
        / "config.yaml"
    )
with config_path.open() as handle:
    config = yaml.safe_load(handle)

assert config["execution"]["inference"] is False
assert config["execution"]["submission"] is False
assert config["runtime"]["kaggle"]["inference_run_on_push"] is False
print(
    "exp303 inference disabled: train-side OOF readout only; "
    "no prediction or submission is generated."
)
