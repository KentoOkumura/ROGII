# %% [markdown]
# # exp206 discussion711308 dz/dTVT b-peak cluster baseline train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. Setup and cost guard
# 4. Input and leakage contract
# 5. Run full-fit diagnostics and target-free pseudo-tail audit
# 6. Metrics and generated artifacts

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dz_dtvt_bpeak_cluster_baseline import list_wells, run_train_audit, to_jsonable
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
variants = list(params.get("variants") or [])

print_json(
    "experiment",
    {
        "name": experiment.get("name"),
        "route": experiment.get("route"),
        "status": experiment.get("status"),
        "parent": get_nested(config, "lineage.parent"),
        "train_dir": str(paths.train_data_dir),
        "artifacts_dir": str(paths.artifacts_dir),
        "selected_variant": params.get("selected_variant"),
    },
)
print_json(
    "cost guard",
    {
        "active_rule_variants": variants,
        "lightgbm_config_count": 0,
        "fold_count": 0,
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
        "fit_equation": get_nested(config, "model.fit_equation"),
        "prefix_tail_rows": params.get("prefix_tail_rows"),
        "min_fit_steps": params.get("min_fit_steps"),
        "leakage_policy": get_nested(config, "validation.leakage_policy"),
    },
)


# %% [markdown]
# ## 5. Run full-fit diagnostics and target-free pseudo-tail audit

# %%
summary = run_train_audit(paths, config)
print_json("train audit summary", summary)


# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
metrics = json.loads(paths.metrics_path.read_text())
print_json("metrics.json", metrics)

print("Generated artifacts:")
for path in sorted(Path(paths.artifacts_dir).glob("exp206*")):
    print(f"- {path.name} ({path.stat().st_size} bytes)")
