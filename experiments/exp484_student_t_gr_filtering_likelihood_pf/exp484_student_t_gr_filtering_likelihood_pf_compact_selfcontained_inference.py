# %% [markdown]
# # exp484 Student-t GR filtering likelihood-PF — inference guard
#
# Exp484 currently implements only the train-side fixed32 Stage 0 technical
# preflight. Raw-test regeneration, `submission.csv`, and competition
# submission remain outside the approved scope.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe config loading
# 3. Fail-closed inference contract
# 4. Contract preview

# %%
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp484_student_t_gr_filtering_likelihood_pf"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe config loading

# %%
def get_nested(
    config: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_config(package_dir: Path | None = None) -> dict[str, Any]:
    root = project_root()
    candidates = [
        package_dir,
        Path.cwd(),
        root / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError("exp484 config.yaml was not found")


# %% [markdown]
# ## 3. Fail-closed inference contract

# %%
def validate_inference_is_disabled(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "experiment": get_nested(config, "experiment.name") == EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route") == "pf_beam",
        "implementation_enabled": bool(
            get_nested(config, "implementation.enabled", False)
        ),
        "canonical_inference_not_adopted": not bool(
            get_nested(
                config,
                "implementation.canonical_inference_notebook_adopted",
                True,
            )
        ),
        "inference_disabled": not bool(
            get_nested(config, "implementation.inference_enabled", True)
        ),
        "submission_disabled": not bool(
            get_nested(config, "implementation.submission_enabled", True)
        ),
        "run_inference_false": not bool(
            get_nested(config, "execution.run_inference", True)
        ),
        "create_submission_false": not bool(
            get_nested(config, "execution.create_submission", True)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"exp484 inference guard mismatch: {checks}")
    return {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_disabled_pending_separate_approval",
        "checks": checks,
        "submission_created": False,
    }


# %% [markdown]
# ## 4. Contract preview

# %%
CONFIG = load_config()
INFERENCE_GUARD = validate_inference_is_disabled(CONFIG)
print(json.dumps(INFERENCE_GUARD, indent=2, sort_keys=True))
