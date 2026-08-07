# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # exp169_tvt_input_pfbeam_offset_calibration all-interval visualization

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration
# 3. Generate all-interval PF/Beam replay plots
# 4. Generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import pandas as pd
from IPython.display import HTML, display

from pfbeam_all_interval_visualization import run_visualization
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration

# %%
paths = ExperimentPaths()
config = load_config()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()

visualization_config = get_nested(config, "visualization") or {}
print(f"experiment={get_nested(config, 'experiment.name')}")
print(f"route={get_nested(config, 'experiment.route')}")
print(f"mode={visualization_config.get('mode')}")
print(f"train_dir={paths.train_data_dir}")
print(f"artifacts_dir={paths.artifacts_dir}")
display(
    {
        "well_ids": visualization_config.get("well_ids"),
        "modes": visualization_config.get("modes"),
        "replay_runtime": visualization_config.get("replay_runtime"),
        "full_known_anchor_rows": visualization_config.get("full_known_anchor_rows"),
    }
)

# %% [markdown]
# ## 3. Generate all-interval PF/Beam replay plots

# %%
summary = run_visualization(config=config, paths=paths)
display(summary)

# %% [markdown]
# ## 4. Generated artifacts

# %%
manifest_path = summary["artifacts"]["manifest"]
html_path = summary["artifacts"]["html_index"]
manifest = pd.read_csv(manifest_path)

print("plot manifest")
display(manifest)

print("html index")
display(HTML(f"<a href='{html_path}' target='_blank'>{html_path}</a>"))

print("generated artifacts")
for key, value in summary["artifacts"].items():
    print(f"{key}: {value}")
