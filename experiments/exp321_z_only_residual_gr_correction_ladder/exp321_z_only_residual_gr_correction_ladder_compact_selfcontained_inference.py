# %% [markdown]
# # exp321 z-only residual / GR correction ladder — inference disabled
#
# exp321 currently implements only the train-side Stage A/B diagnostic. It has
# no promoted candidate, saved inference model, test prediction, or submission.

# %% [markdown]
# ## Contents
# 1. Imports and experiment identity
# 2. Notebook-safe configuration loading
# 3. Fail-closed inference contract
# 4. Setup and explicit stop

# %%
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPERIMENT_NAME = "exp321_z_only_residual_gr_correction_ladder"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP321_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Notebook-safe configuration loading


# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    for path in (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ):
        if not path.is_file():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp321 config.yaml was not found")


# %% [markdown]
# ## 3. Fail-closed inference contract


# %%
def validate_disabled_inference(config: Mapping[str, Any]) -> dict[str, bool]:
    checks = {
        "implementation_inference_disabled": not bool(
            get_nested(config, "implementation.inference_enabled")
        ),
        "execution_inference_disabled": not bool(
            get_nested(config, "execution_contract.inference")
        ),
        "inference_section_disabled": not bool(get_nested(config, "inference.enabled")),
        "submission_creation_disabled": not bool(
            get_nested(config, "inference.create_submission")
        ),
        "stage_c_disabled": not bool(
            get_nested(config, "stage_c_window_gr_correction.enabled")
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"exp321 inference must remain disabled: {failed}")
    return checks


def stop_without_inference(config: Mapping[str, Any]) -> None:
    validate_disabled_inference(config)
    raise RuntimeError(
        "exp321 inference is intentionally disabled until Stage A/B PASS, "
        "separate Stage C approval/run, and candidate promotion"
    )


# %% [markdown]
# ## 4. Setup and explicit stop


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    CHECKS = validate_disabled_inference(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "inference_contract": CHECKS,
                "message": "No prediction or submission is generated.",
            },
            indent=2,
        )
    )
    stop_without_inference(CONFIG)
