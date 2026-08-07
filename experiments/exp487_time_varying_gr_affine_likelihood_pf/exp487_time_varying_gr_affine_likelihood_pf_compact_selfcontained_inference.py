# %% [markdown]
# # exp487 time-varying GR affine likelihood-PF — inference guard
#
# Train-side Stage 0/1 code is implemented, but current-test exp209 path
# regeneration and deployment process-noise generation are intentionally absent.
# Inference, candidate selection, blending, submission creation, and submission
# remain fail-closed pending train-side promotion and separate approval.

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe config loading
# 2. Fail-closed inference contract
# 3. Configuration preview

# %%
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT_NAME = "exp487_time_varying_gr_affine_likelihood_pf"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP487_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Fail-closed inference contract


# %%
def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
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
        config = yaml.safe_load(path.read_text()) or {}
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError("exp487 config.yaml was not found")


def validate_inference_is_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    status = {
        "implementation_scope": "train_side_stage0_and_stage1_only",
        "canonical_inference_notebook_adopted": bool(
            get_nested(
                config,
                "implementation.canonical_inference_notebook_adopted",
                False,
            )
        ),
        "inference_enabled": bool(
            get_nested(config, "implementation.inference_enabled", False)
        ),
        "run_inference": bool(get_nested(config, "execution.run_inference", False)),
        "create_submission": bool(
            get_nested(config, "execution.create_submission", False)
        ),
        "submit_to_kaggle": bool(
            get_nested(config, "execution.submit_to_kaggle", False)
        ),
        "test_exp209_path_regeneration_implemented": bool(
            get_nested(
                config,
                "implementation.test_exp209_path_regeneration_implemented",
                False,
            )
        ),
        "test_deployment_process_noise_implemented": bool(
            get_nested(
                config,
                "implementation.test_deployment_process_noise_implemented",
                False,
            )
        ),
    }
    if any(
        (
            status["canonical_inference_notebook_adopted"],
            status["inference_enabled"],
            status["run_inference"],
            status["create_submission"],
            status["submit_to_kaggle"],
            status["test_exp209_path_regeneration_implemented"],
            status["test_deployment_process_noise_implemented"],
        )
    ):
        raise RuntimeError("exp487 inference contract was enabled without approval")
    return status


def run_inference(_: Mapping[str, Any]) -> None:
    raise RuntimeError(
        "exp487 inference is not implemented or approved; current-test exp209 "
        "path regeneration and all-train deployment process-noise generation "
        "require a separate contract and approval"
    )


# %% [markdown]
# ## 3. Configuration preview


# %%
CONFIG = load_config()
INFERENCE_STATUS = validate_inference_is_disabled(CONFIG)
print(json.dumps(INFERENCE_STATUS, indent=2, sort_keys=True))

if EXECUTE_NOTEBOOK:
    run_inference(CONFIG)
