# %% [markdown]
# # exp343 ACF effective-sample likelihood tempering audit — inference
#
# exp343 は Stage 0 target-free ACF stability diagnostic だけを実装している。
# Stage 1 HMM、raw-test inference、submission は別承認前なので fail-closed とする。

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe configuration
# 2. Disabled inference contract
# 3. Setup and explicit stop

# %%
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover
    get_ipython = lambda: None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp343_acf_effective_sample_likelihood_tempering_audit"
PACKAGE_DIR = Path.cwd()
IMPORT_ONLY = os.environ.get("EXP343_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def load_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"Could not locate exp343 config; checked={candidates}")


CONFIG = load_config()

# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp343 route must remain pf_beam")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("Stage 0 implementation must remain enabled")
    if bool(get_nested(config, "implementation.stage_1_implemented")):
        raise ValueError("Stage 1 must remain unimplemented")
    if bool(get_nested(config, "execution.run_stage_1")):
        raise ValueError("Stage 1 execution is forbidden")
    if bool(get_nested(config, "execution.run_inference")):
        raise ValueError("inference execution is forbidden")
    if bool(get_nested(config, "execution.create_submission")):
        raise ValueError("submission creation is forbidden")
    if bool(get_nested(config, "inference.enabled")):
        raise ValueError("inference.enabled must remain false")
    if bool(get_nested(config, "inference.create_submission")):
        raise ValueError("inference.create_submission must remain false")
    return dict(get_nested(config, "execution_contract.stage_0") or {})


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp343 implements only the target-free Stage 0 ACF audit. Stage 1, "
        "inference, and submission require a separate user approval."
    )


# %% [markdown]
# ## 3. Setup and explicit stop

# %%
if EXECUTE_NOTEBOOK:
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "stage_0_execution_contract": validate_disabled_inference(CONFIG),
            "stage_1_implemented": False,
            "inference_enabled": False,
            "submission_enabled": False,
        }
    )
    stop_disabled_inference(CONFIG)
