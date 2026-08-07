# %% [markdown]
# # exp364 signed-curvature exact HMM — inference
#
# exp364 currently implements only the train-side Stage 0 signed-path and
# resource preflight. Stage 1 exact-HMM decoding, raw-test inference, and
# submission are fail-closed.

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe configuration
# 2. Disabled inference contract
# 3. Setup and explicit stop

# %%
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp364_signed_curvature_exact_hmm"
PACKAGE_DIR = Path.cwd()


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
    raise FileNotFoundError(f"Could not locate exp364 config; checked={candidates}")


# %% [markdown]
# ## 2. Disabled inference contract

# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in checks.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(
                f"exp364 disabled inference contract changed: "
                f"{key}={actual!r}, expected {expected!r}"
            )
    return dict(get_nested(config, "execution.stage_0_counts") or {})


def stop_disabled_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp364 implements only Stage 0. Stage 1 exact HMM, inference, and "
        "submission require a Stage 0 PASS and separate user approval."
    )


# %% [markdown]
# ## 3. Setup and explicit stop

# %%
if __name__ == "__main__":
    CONFIG = load_config()
    print(
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
