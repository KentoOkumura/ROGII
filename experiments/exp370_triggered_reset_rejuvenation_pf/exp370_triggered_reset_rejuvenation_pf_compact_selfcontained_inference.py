# %% [markdown]
# # exp370 triggered reset rejuvenation PF — inference policy
#
# Inference is intentionally unavailable. Stage 1 particle reinjection has not
# been implemented and requires both a Stage 0 PASS and separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe configuration lookup
# 3. Fail-closed inference contract

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from IPython.display import display
except ImportError:

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp370_triggered_reset_rejuvenation_pf"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe configuration lookup

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def locate_config() -> Path:
    relative = Path("experiments") / EXPERIMENT_NAME / "config.yaml"
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / relative,
        Path.cwd() / "config.yaml",
        Path.cwd() / relative,
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        value = yaml.safe_load(candidate.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"Could not locate config.yaml for {EXPERIMENT_NAME}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or locate_config()
    if source.is_dir():
        source = source / "config.yaml"
    value = yaml.safe_load(source.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


# %% [markdown]
# ## 3. Fail-closed inference contract
#
# No sample submission is copied and no prediction file is created.

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    status = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "experiment_status": get_nested(config, "experiment.status"),
        "implementation_scope": get_nested(config, "execution.implementation_scope"),
        "run_stage_1": bool(get_nested(config, "execution.run_stage_1")),
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
        "stage0_particles": int(
            get_nested(config, "validation.stage_0.particles")
        ),
        "stage0_seed_count": int(
            get_nested(config, "validation.stage_0.diagnostic_seed_count")
        ),
        "stage1_seed_well_runs": int(
            get_nested(config, "execution.stage_1_counts.seed_well_runs")
        ),
    }
    if status["experiment"] != EXPERIMENT_NAME or status["route"] != "pf_beam":
        raise ValueError(f"Unexpected exp370 inference contract: {status}")
    if status["implementation_scope"] != "stage0_only":
        raise ValueError("exp370 implementation scope must remain stage0_only")
    enabled = [
        key
        for key in ("run_stage_1", "run_inference", "create_submission")
        if status[key]
    ]
    if enabled:
        raise ValueError(
            "Inference flags cannot be enabled before Stage 1 implementation: "
            f"{enabled}"
        )
    return status


CONFIG = load_config()
STATUS = validate_disabled_inference(CONFIG)
display(pd.DataFrame([STATUS]))

# %%
if __name__ == "__main__":
    print(json.dumps(STATUS, indent=2))
    raise RuntimeError(
        "exp370 inference is fail-closed: Stage 1 rejuvenation PF is not "
        "implemented. A Stage 0 PASS and separate approval are required."
    )
