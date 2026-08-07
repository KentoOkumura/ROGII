# %% [markdown]
# # exp226 connortynan K16 spline kernel kNN adaptive kappa reproduction inference

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and inference contract
# 4. Input and submission contract
# 5. Run full-train K16 source-port inference
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from connortynan_k16_reproduction import list_wells, run_inference, to_jsonable
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Setup and inference contract

# %%
paths = ExperimentPaths()
paths.ensure_output_dirs()
config = load_config()

experiment = get_nested(config, "experiment") or {}
params = get_nested(config, "model.params") or {}

print_json(
    "experiment",
    {
        "name": experiment.get("name"),
        "route": experiment.get("route"),
        "status": experiment.get("status"),
        "parent": get_nested(config, "lineage.parent"),
        "source_notebook": get_nested(config, "lineage.source_notebook"),
        "train_dir": str(paths.train_data_dir),
        "test_dir": str(paths.test_data_dir),
        "sample_submission": str(paths.sample_submission_path),
        "submission_path": str(paths.submission_path),
    },
)

print_json(
    "inference profile",
    {
        "selected_variant": get_nested(config, "inference.selected_variant"),
        "external_weights_enabled": False,
        "uses_v7_neural_committee": False,
        "uses_v8_gbm_meta_layer": False,
        "kappa_regimes": params.get("kappa_regimes"),
        "enable_gr_correction": params.get("enable_gr_correction"),
        "enable_u_projection": params.get("enable_u_projection"),
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "internet": bool(get_nested(config, "runtime.kaggle.enable_internet")),
    },
)


# %% [markdown]
# ## 4. Input and submission contract

# %%
train_wells = list_wells(paths.train_data_dir)
test_wells = list_wells(paths.test_data_dir)
sample = pd.read_csv(paths.sample_submission_path)

if not train_wells:
    raise FileNotFoundError(f"No train wells found under {paths.train_data_dir}")
if not test_wells:
    raise FileNotFoundError(f"No test wells found under {paths.test_data_dir}")
if list(sample.columns) != ["id", "tvt"]:
    raise RuntimeError(
        f"sample_submission columns must be ['id', 'tvt'], got {list(sample.columns)}"
    )

print_json(
    "input contract",
    {
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "sample_rows": len(sample),
        "first_sample_ids": sample["id"].head(5).tolist(),
        "output_columns": ["id", "tvt"],
    },
)


# %% [markdown]
# ## 5. Run full-train K16 source-port inference

# %%
summary = run_inference(paths, config)
print_json("inference summary", summary)


# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
submission = pd.read_csv(paths.submission_path)
if list(submission.columns) != ["id", "tvt"]:
    raise RuntimeError(f"submission columns must be ['id', 'tvt'], got {list(submission.columns)}")
if len(submission) != len(sample):
    raise RuntimeError(f"submission row mismatch: got {len(submission)}, expected {len(sample)}")
if not submission["id"].equals(sample["id"]):
    raise RuntimeError("submission ids do not match sample_submission order")
if submission["tvt"].isna().any():
    raise RuntimeError("submission contains NaN tvt")

metrics = json.loads(paths.metrics_path.read_text())
print_json("metrics.json", metrics)

print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("exp226*")):
    print(f"- {path.name} ({path.stat().st_size} bytes)")
