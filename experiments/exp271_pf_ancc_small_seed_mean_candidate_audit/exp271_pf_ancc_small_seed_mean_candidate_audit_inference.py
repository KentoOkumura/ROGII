# %% [markdown]
# # exp268 PF ANCC small-seed mean candidate audit — inference disabled

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Configuration check
# 3. Disabled inference guard

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp271_pf_ancc_small_seed_mean_candidate_audit"


# %% [markdown]
# ## 2. Configuration check

# %%
def nested(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def find_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        Path.cwd() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"config.yaml for {EXPERIMENT_NAME} was not found")


CONFIG_PATH = find_config_path()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text()) or {}
print(
    {
        "experiment": nested(CONFIG, "experiment.name"),
        "route": nested(CONFIG, "experiment.route"),
        "parent": nested(CONFIG, "lineage.parent"),
        "candidate_variants": nested(CONFIG, "model.active_variants"),
        "inference_enabled": nested(CONFIG, "inference.enabled"),
        "submission_enabled": nested(CONFIG, "execution.submission_enabled"),
    }
)


# %% [markdown]
# ## 3. Disabled inference guard
#
# 本実験はtrain-sideの0-booster candidate auditに限定する。guard通過後のraw-test再生成、
# selector/add-only feature、submissionは別実験・別判断で扱う。

# %%
if nested(CONFIG, "inference.enabled") is not False:
    raise RuntimeError("exp268 inference must remain disabled")
if nested(CONFIG, "execution.submission_enabled") is not False:
    raise RuntimeError("exp268 submission must remain disabled")

print("Inference is intentionally disabled; submission.csv was not created.")
