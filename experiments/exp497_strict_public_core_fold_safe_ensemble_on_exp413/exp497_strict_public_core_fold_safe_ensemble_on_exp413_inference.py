# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp497 saved-model hidden-safe inference candidate
#
# Stage I version 4で保存したLightGBM 24本、CatBoost 16本、Ridge 2セットを
# SHA検証して読み込む。raw hidden testからstrict public-core特徴とexp413を
# 動的再生成し、固定weight `0.13716473330712417`だけでblendする。
# booster学習、weight再fit、Public LB固有overlay、外部competition submitは行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Authorization and zero-training inventory
# 3. Saved-model and dynamic input contracts
# 4. Dynamic hidden-safe exp413 generation
# 5. Dynamic strict public-core feature generation
# 6. Saved-model inference and fixed blend
# 7. Submission and reproducibility outputs

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from exp510_exp413_hidden_safe_runtime import (
    PARENT_SOURCE_SHA256 as EXP413_PARENT_SOURCE_SHA256,
    generate_dynamic_exp413_prediction,
)
from src.strict_public_core import (
    STAGE_I_ARTIFACT_FILES,
    build_stage_i_test_feature_frame,
    load_stage_i_saved_inference_artifacts,
    run_stage_i_saved_model_inference,
    sha256_file,
)

EXPERIMENT = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").is_file():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
if not KAGGLE_INPUT_ROOT.is_dir() or not KAGGLE_WORKING.is_dir():
    raise RuntimeError("exp497 inference must run in Kaggle Notebook")
OUTPUT_DIR = KAGGLE_WORKING / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_unique_input_file(filename: str, path_token: str) -> Path:
    matches = [
        path
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if path_token in str(path)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one {filename} under input token {path_token}, got {matches}"
        )
    return matches[0]


def find_competition_data_dir() -> Path:
    candidates = [
        KAGGLE_INPUT_ROOT / "competitions/rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    ]
    candidates.extend(path.parent for path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv"))
    matches = []
    for candidate in candidates:
        if (
            (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
            and (candidate / "sample_submission.csv").is_file()
        ):
            matches.append(candidate.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one competition data directory, got {unique}")
    return unique[0]


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reload_exp413_serialized_boundary(
    path: Path,
    in_memory_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    if not path.is_file():
        raise FileNotFoundError(f"dynamic exp413 prediction artifact is missing: {path}")
    serialized = pd.read_csv(path, dtype={"id": str, "well": str})
    required = {"id", "well", "pred_tvt"}
    for label, frame in (("serialized", serialized), ("memory", in_memory_frame)):
        missing = sorted(required - set(frame.columns))
        if missing or frame["id"].astype(str).duplicated().any():
            raise RuntimeError(f"{label} exp413 prediction contract failed: {missing}")
    comparison = serialized[["id", "pred_tvt"]].merge(
        in_memory_frame[["id", "pred_tvt"]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_serialized", "_memory"),
    )
    if comparison["pred_tvt_memory"].isna().any() or len(comparison) != len(in_memory_frame):
        raise RuntimeError("serialized exp413 prediction IDs differ from memory")
    max_abs = float(
        np.max(
            np.abs(
                comparison["pred_tvt_serialized"].to_numpy(np.float64)
                - comparison["pred_tvt_memory"].to_numpy(np.float64)
            )
        )
    )
    if max_abs > 1e-3:
        raise RuntimeError(f"dynamic exp413 serialization drift is too large: {max_abs}")
    return serialized, max_abs


def isolate_exp413_intermediate_submission(
    source_path: Path,
    destination_path: Path,
    sample_frame: pd.DataFrame,
    exp413_frame: pd.DataFrame,
) -> dict[str, object]:
    if not source_path.is_file() or destination_path.exists():
        raise RuntimeError(
            "exp413 intermediate submission isolation path contract failed: "
            f"source={source_path} destination={destination_path}"
        )
    intermediate = pd.read_csv(source_path, dtype={"id": str})
    if (
        list(intermediate.columns) != ["id", "tvt"]
        or len(intermediate) != len(sample_frame)
        or intermediate["id"].duplicated().any()
        or not intermediate["id"].equals(sample_frame["id"])
        or not np.isfinite(intermediate["tvt"].to_numpy(np.float64)).all()
    ):
        raise RuntimeError("exp413 intermediate submission identity contract failed")
    expected = intermediate[["id", "tvt"]].merge(
        exp413_frame[["id", "pred_tvt"]],
        on="id",
        validate="one_to_one",
    )
    max_abs = float(
        np.max(
            np.abs(
                expected["tvt"].to_numpy(np.float64)
                - expected["pred_tvt"].to_numpy(np.float64)
            )
        )
    )
    if max_abs > 1e-3:
        raise RuntimeError(f"exp413 intermediate submission drift is too large: {max_abs}")
    source_path.replace(destination_path)
    if source_path.exists() or not destination_path.is_file():
        raise RuntimeError("exp413 intermediate submission was not isolated")
    return {
        "path": str(destination_path),
        "rows": len(intermediate),
        "max_abs_vs_serialized_exp413": max_abs,
        "sha256": sha256_file(destination_path),
    }


# %% [markdown]
# ## 2. Authorization and zero-training inventory

# %%
deployment = CONFIG["inference"]["saved_model_deployment"]
stage = CONFIG["stages"]["stage_i_saved_model_inference"]
inventory = {
    "exp497_fitted_boosters": int(stage["fitted_boosters"]),
    "exp497_loaded_boosters": int(stage["loaded_exp497_boosters"]),
    "exp497_loaded_ridge_models": int(stage["loaded_exp497_ridge_models"]),
    "exp413_fitted_boosters": int(stage["exp413_retraining"]),
    "exp413_loaded_boosters": int(stage["loaded_exp413_boosters"]),
}
if (
    CONFIG["experiment"]["route"] != "ensemble"
    or not deployment["approved"]
    or deployment["mode"] != "hidden_safe_saved_model_inference"
    or inventory
    != {
        "exp497_fitted_boosters": 0,
        "exp497_loaded_boosters": 40,
        "exp497_loaded_ridge_models": 2,
        "exp413_fitted_boosters": 0,
        "exp413_loaded_boosters": 75,
    }
    or deployment["public_test_exp413_sidecar_allowed"]
    or not deployment["generate_submission_csv"]
    or deployment["external_submission"]
    or not stage["submission_enabled"]
    or stage["external_submission_enabled"]
):
    raise RuntimeError("exp497 saved-model inference authorization contract failed")
print("experiment", EXPERIMENT)
print("route/mode", CONFIG["experiment"]["route"], deployment["mode"])
display(pd.DataFrame([inventory]))

# %% [markdown]
# ## 3. Saved-model and dynamic input contracts

# %%
competition_data_dir = find_competition_data_dir()
sample_submission_path = competition_data_dir / "sample_submission.csv"
sample = pd.read_csv(sample_submission_path, dtype={"id": str})
if list(sample.columns) != ["id", "tvt"] or sample.empty or sample["id"].duplicated().any():
    raise RuntimeError("dynamic sample submission identity contract failed")

model_manifest_path = resolve_unique_input_file(
    STAGE_I_ARTIFACT_FILES["model_manifest"],
    "exp497-strict-public-core-current-test-inference",
)
model_artifact_dir = model_manifest_path.parent
public_runtime_path = resolve_unique_input_file(
    "public_notebook_replay_audit.py",
    "exp072-exp063-full-replay-feature-cache-train",
)
exp413_runtime_path = PACKAGE_DIR / str(deployment["dynamic_exp413_runtime_source"])
if (
    not exp413_runtime_path.is_file()
    or sha256_file(exp413_runtime_path) != deployment["dynamic_exp413_runtime_sha256"]
    or EXP413_PARENT_SOURCE_SHA256 != deployment["dynamic_exp413_parent_source_sha256"]
):
    raise RuntimeError("dynamic exp413 runtime source SHA contract failed")

saved_artifacts = load_stage_i_saved_inference_artifacts(
    model_artifact_dir,
    expected_sha256=deployment["artifact_sha256"],
    expected_model_set_sha256=deployment["serialized_model_set_sha256"],
)
artifact_rows = [
    {
        "role": role,
        "path": str(saved_artifacts["paths"][role]),
        "sha256": saved_artifacts["sha256"][role],
    }
    for role in STAGE_I_ARTIFACT_FILES
]
display(pd.DataFrame(artifact_rows))
display(sample.head())
print("dynamic sample rows", len(sample))
print("saved exp497 model bytes", saved_artifacts["serialization"]["serialized_model_bytes"])

# %% [markdown]
# ## 4. Dynamic hidden-safe exp413 generation

# %%
exp413_memory, exp413_runtime_metrics, exp413_prediction_path = (
    generate_dynamic_exp413_prediction()
)
exp413_frame, exp413_serialization_roundtrip_max_abs = reload_exp413_serialized_boundary(
    exp413_prediction_path,
    exp413_memory,
)
exp413_intermediate_submission_audit = isolate_exp413_intermediate_submission(
    KAGGLE_WORKING / "submission.csv",
    OUTPUT_DIR / "exp413_intermediate_submission.csv",
    sample,
    exp413_frame,
)
if (
    int(exp413_runtime_metrics.get("rows", -1)) != len(sample)
    or int(exp413_runtime_metrics.get("booster_training_count", -1)) != 0
    or bool(exp413_runtime_metrics.get("external_submission_performed", True))
):
    raise RuntimeError("dynamic hidden-safe exp413 runtime contract failed")
print(
    json.dumps(
        {
            "exp413_runtime_seconds": exp413_runtime_metrics.get("runtime_seconds"),
            "exp413_rows": exp413_runtime_metrics.get("rows"),
            "exp413_wells": exp413_runtime_metrics.get("wells"),
            "exp413_loaded_boosters": 75,
            "exp413_fitted_boosters": 0,
            "serialization_roundtrip_max_abs": exp413_serialization_roundtrip_max_abs,
            "prediction_gzip_content_sha256": sha256_gzip_content(exp413_prediction_path),
            "intermediate_submission": exp413_intermediate_submission_audit,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 5. Dynamic strict public-core feature generation

# %%
test_frame, test_well_metadata, test_feature_summary = build_stage_i_test_feature_frame(
    competition_data_dir=competition_data_dir,
    output_dir=OUTPUT_DIR,
    public_runtime_path=public_runtime_path,
    particles=int(CONFIG["public_core"]["pf"]["particles"]),
    seeds=int(CONFIG["public_core"]["pf"]["seeds"]),
    n_jobs=4,
)
if (
    len(test_frame) != len(sample)
    or test_frame["id"].astype(str).duplicated().any()
    or set(test_frame["id"].astype(str)) != set(sample["id"])
    or test_frame["well"].nunique() <= 0
):
    raise RuntimeError("dynamic strict public-core feature identity contract failed")
display(test_well_metadata)
display(test_frame[["id", "well", "md_since", "last_known_tvt"]].head())
print(json.dumps(test_feature_summary, indent=2))

# %% [markdown]
# ## 6. Saved-model inference and fixed blend

# %%
exp413_runtime_metrics = dict(exp413_runtime_metrics)
exp413_runtime_metrics["serialization_roundtrip_max_abs"] = (
    exp413_serialization_roundtrip_max_abs
)
exp413_runtime_metrics["intermediate_submission"] = exp413_intermediate_submission_audit
summary = run_stage_i_saved_model_inference(
    output_dir=OUTPUT_DIR,
    artifact_dir=model_artifact_dir,
    expected_artifact_sha256=deployment["artifact_sha256"],
    expected_model_set_sha256=deployment["serialized_model_set_sha256"],
    test_frame=test_frame,
    test_well_metadata=test_well_metadata,
    sample_submission_path=sample_submission_path,
    exp413_prediction_frame=exp413_frame,
    exp413_runtime_metrics=exp413_runtime_metrics,
    test_feature_summary=test_feature_summary,
    submission_output_path=KAGGLE_WORKING / "submission.csv",
    prediction_chunk_size=int(deployment["prediction_chunk_size"]),
    visible_strict_public_core_max_abs=float(
        deployment["visible_strict_public_core_max_abs"]
    ),
    visible_blend_max_abs=float(deployment["visible_blend_max_abs"]),
)
if (
    summary["status"] != "complete"
    or summary["fitted_boosters"] != 0
    or summary["loaded_exp497_boosters"] != 40
    or summary["loaded_exp497_ridge_models"] != 2
    or summary["loaded_exp413_boosters"] != 75
    or summary["exp413_retraining"] != 0
    or summary["exp497_retraining"] != 0
    or not summary["submission_generated"]
    or summary["external_submission_performed"]
):
    raise RuntimeError("exp497 saved-model inference completion contract failed")

# %% [markdown]
# ## 7. Submission and reproducibility outputs

# %%
submission_path = KAGGLE_WORKING / "submission.csv"
submission = pd.read_csv(submission_path, dtype={"id": str})
if (
    list(submission.columns) != list(sample.columns)
    or len(submission) != len(sample)
    or not submission["id"].equals(sample["id"])
    or submission["id"].duplicated().any()
    or not np.isfinite(submission["tvt"].to_numpy(np.float64)).all()
):
    raise RuntimeError("exp497 submission output contract failed")
display(submission.head())
display(pd.DataFrame([summary["prediction_stats"]]))
display(pd.DataFrame([summary["visible_parity"]]))
print(json.dumps(summary, indent=2))
print(
    "exp497 saved-model inference complete: 40 exp497 + 75 exp413 boosters loaded, "
    "0 boosters fitted, submission.csv generated, no external submit performed."
)
