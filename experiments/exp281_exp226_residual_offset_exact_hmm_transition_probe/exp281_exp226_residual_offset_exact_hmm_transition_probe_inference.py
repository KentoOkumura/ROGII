# %% [markdown]
# # exp281 exp226 residual-offset exact HMM transition probe inference
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

EXPERIMENT_NAME = "exp281_exp226_residual_offset_exact_hmm_transition_probe"


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
    raise FileNotFoundError("exp281 config was not found")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


def validate_disabled_inference(config: dict[str, Any]) -> dict[str, Any]:
    inference = get_nested(config, "inference") or {}
    execution = get_nested(config, "execution") or {}
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "inference_enabled": bool(inference.get("enabled")),
        "mode": str(inference.get("mode") or ""),
        "create_submission": bool(inference.get("create_submission")),
        "selected_candidate": inference.get("selected_candidate"),
        "execution_inference": bool(execution.get("inference")),
        "execution_submission": bool(execution.get("submission")),
    }
    if any(
        contract[key]
        for key in (
            "inference_enabled",
            "create_submission",
            "execution_inference",
            "execution_submission",
        )
    ):
        raise ValueError("exp281 inference and submission must remain disabled")
    if contract["selected_candidate"] is not None:
        raise ValueError("exp281 cannot select an inference candidate before train guard review")
    return contract


# %% [markdown]
# ## 2. Load and display the disabled inference contract


# %%
if in_notebook_runtime():
    CONFIG = load_config()
    CONTRACT = validate_disabled_inference(CONFIG)
    print(json.dumps(CONTRACT, indent=2, sort_keys=True), flush=True)


# %% [markdown]
# ## 3. Refuse prediction and submission generation


# %%
if in_notebook_runtime():
    print(
        "Inference is disabled until all exp281 train guards pass and the user approves a "
        "hidden-safe raw-test regeneration design.",
        flush=True,
    )
