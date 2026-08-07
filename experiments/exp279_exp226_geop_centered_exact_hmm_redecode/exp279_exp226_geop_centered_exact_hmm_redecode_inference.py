# %% [markdown]
# # exp279 exp226 geop-centered exact HMM re-decode inference
#
# Inference is deliberately disabled. The train-side candidate must pass every
# fixed promotion guard before a hidden-safe exp226 geometry regeneration path
# is designed in this same experiment.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Load and display the disabled inference contract
# 3. Refuse prediction and submission generation

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp279_exp226_geop_centered_exact_hmm_redecode"


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if path.exists():
            config = read_yaml(path)
            if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
                return config
    raise FileNotFoundError("exp279 config was not found")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


# %% [markdown]
# ## 2. Load and display the disabled inference contract


# %%
if in_notebook_runtime():
    CONFIG = load_config()
    CONTRACT = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(CONFIG, "experiment.route"),
        "inference_enabled": get_nested(CONFIG, "inference.enabled"),
        "mode": get_nested(CONFIG, "inference.mode"),
        "create_submission": get_nested(CONFIG, "inference.create_submission"),
        "selected_candidate": get_nested(CONFIG, "inference.selected_candidate"),
    }
    print(json.dumps(CONTRACT, indent=2, sort_keys=True), flush=True)


# %% [markdown]
# ## 3. Refuse prediction and submission generation


# %%
if in_notebook_runtime():
    if bool(get_nested(CONFIG, "inference.enabled")) or bool(
        get_nested(CONFIG, "inference.create_submission")
    ):
        raise RuntimeError("exp279 inference contract changed without train guard review")
    print(
        "Inference is disabled until all exp279 train guards pass and the user approves a "
        "hidden-safe raw-test regeneration design.",
        flush=True,
    )
