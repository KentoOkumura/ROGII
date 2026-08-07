# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp171_bimodal_posterior_pfbeam_candidate_audit inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Scope statement

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from IPython.display import display
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration

# %%
paths = ExperimentPaths()
config = load_config()

print(f"experiment={get_nested(config, 'experiment.name')}")
print(f"route={get_nested(config, 'experiment.route')}")
print(f"inference_mode={get_nested(config, 'inference.mode')}")

# %% [markdown]
# ## 3. Scope statement

# %%
display(
    {
        "status": "not_applicable_train_side_audit_only",
        "reason": (
            "This experiment audits train-side fixed top2 GR modes and posterior mean "
            "candidates. It does not generate test predictions or submission.csv."
        ),
        "train_notebook": "exp171_bimodal_posterior_pfbeam_candidate_audit_train.ipynb",
        "artifacts_dir": str(paths.artifacts_dir),
    }
)
