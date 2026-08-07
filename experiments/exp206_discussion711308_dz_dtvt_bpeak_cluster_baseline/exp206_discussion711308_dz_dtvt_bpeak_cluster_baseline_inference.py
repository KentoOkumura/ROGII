# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     main_language: python
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp206 discussion711308 dz/dTVT b-peak cluster baseline inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and input contract
# 4. Generate direct test prediction and submission
# 5. Submission summary

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from dz_dtvt_bpeak_cluster_baseline import list_wells, run_inference, to_jsonable
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Setup and input contract

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()
params = get_nested(config, "model.params") or {}

train_wells = list_wells(paths.train_data_dir)
test_wells = list_wells(paths.test_data_dir)
sample = pd.read_csv(paths.sample_submission_path)

print_json(
    "inference contract",
    {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "selected_variant": get_nested(config, "inference.selected_variant"),
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "sample_rows": len(sample),
        "submission_path": str(paths.submission_path),
        "uses_test_tail_true_tvt": False,
        "uses_test_known_tvt_direct_fit": (
            get_nested(config, "inference.selected_variant") == "known_tvt_fit_full"
        ),
        "variants": params.get("variants"),
    },
)

if not train_wells:
    raise FileNotFoundError(f"No train wells found under {paths.train_data_dir}")
if not test_wells:
    raise FileNotFoundError(f"No test wells found under {paths.test_data_dir}")


# %% [markdown]
# ## 4. Generate direct test prediction and submission

# %%
summary = run_inference(paths, config)
print_json("inference summary", summary)


# %% [markdown]
# ## 5. Submission summary

# %%
submission = pd.read_csv(paths.submission_path)
print("Submission:", paths.submission_path)
print(submission.head())
print(submission.describe(include="all"))

print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("exp206*")):
    print(f"- {path.name} ({path.stat().st_size} bytes)")
