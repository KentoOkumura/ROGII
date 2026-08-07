# %% [markdown]
# # exp226 connortynan K16 spline kernel kNN adaptive kappa reproduction train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and cost guard
# 4. Input and leakage contract
# 5. Run group-safe K16 source-port CV
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from connortynan_k16_reproduction import list_wells, run_train_audit, to_jsonable
from settings import ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime and configuration helpers


# %%
def print_json(title: str, payload: dict[str, Any]) -> None:
    print(f"\n## {title}")
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


# %% [markdown]
# ## 3. Setup and cost guard

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
        "artifacts_dir": str(paths.artifacts_dir),
    },
)

print_json(
    "cost guard",
    {
        "active_rule_variants": ["v6_k16_geometry_gr_u_projection"],
        "kappa_regimes": params.get("kappa_regimes"),
        "k_segments": params.get("k_segments"),
        "lightgbm_config_count": 0,
        "fold_count": get_nested(config, "validation.n_folds"),
        "total_boosters": 0,
        "parent_or_control_retraining": False,
        "gpu": bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "internet": bool(get_nested(config, "runtime.kaggle.enable_internet")),
    },
)


# %% [markdown]
# ## 4. Input and leakage contract

# %%
train_wells = list_wells(paths.train_data_dir)
if not train_wells:
    raise FileNotFoundError(f"No train wells found under {paths.train_data_dir}")

print_json(
    "input contract",
    {
        "train_wells": len(train_wells),
        "first_train_wells": train_wells[:5],
        "score_rows": get_nested(config, "validation.score_rows"),
        "validation_strategy": get_nested(config, "validation.strategy"),
        "leakage_policy": get_nested(config, "validation.leakage_policy"),
        "external_weights": get_nested(config, "external_weights"),
    },
)


# %% [markdown]
# ## 5. Run group-safe K16 source-port CV

# %%
summary = run_train_audit(paths, config)
print_json("train audit summary", summary)


# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
metrics = json.loads(paths.metrics_path.read_text())
print_json("metrics.json", metrics)

print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("exp226*")):
    print(f"- {path.name} ({path.stat().st_size} bytes)")
