# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp235_fixed_lag_particle_smoother_pf inference
#
# This experiment intentionally has no inference path. A raw-test port may be
# designed only after exp232 temperature comparison, train-side RMSE, sampled
# coverage, hidden-like, and worst-well guards have been recorded and reviewed.

# %% [markdown]
# ## 1. Configuration guard

# %%
from IPython.display import display

from settings import ExperimentPaths, get_nested, load_config

paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()

display(
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "inference_mode": get_nested(config, "inference.mode"),
        "selected_candidate": get_nested(config, "inference.selected_candidate"),
        "required_train_side_guards": [
            "RMSE versus exp072 Gaussian control",
            "sampled particle coverage versus exp232 temperature variants",
            "1000_plus and hidden-like readouts",
            "worst-well regression",
            "completed exp232 artifact comparison",
        ],
    }
)

raise RuntimeError(
    "exp235 is train-side only. Do not generate raw-test predictions or submission.csv before a separate approval."
)
