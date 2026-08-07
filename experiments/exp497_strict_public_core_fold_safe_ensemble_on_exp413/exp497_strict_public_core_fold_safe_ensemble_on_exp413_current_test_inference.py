# %% [markdown]
# # exp497 Stage I — current-test prediction-only diagnostic inference
#
# Stage E promotion gateはFAILのままとし、selected train anchorもexp413から変更しない。
# 保存済みexp413 current-test予測をSHA固定入力として再利用し、exp497 strict
# public-coreだけをfull-train inner-4で学習して、Stage E meta-fold weightの中央値で
# 定数blendする。`submission.csv`の生成とcompetition submitは行わない。

# %% [markdown]
# ## Contents
# 1. Imports, runtime, and authorization
# 2. Stage P / Stage E / exp413 input contracts
# 3. Dynamic current-test feature generation
# 4. Full-train inner-4 model inference
# 5. Prediction-only artifacts and stop

# %% [markdown]
# ## 1. Imports, runtime, and authorization

# %%
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from src.strict_public_core import (
    build_stage_i_test_feature_frame,
    find_artifact,
    run_stage_i_current_test,
    sha256_file,
    sha256_gzip_decompressed,
)

EXPERIMENT = "exp497_strict_public_core_fold_safe_ensemble_on_exp413"
ROOT = Path.cwd()
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
if not EXPERIMENT_DIR.is_dir():
    EXPERIMENT_DIR = ROOT
CONFIG = yaml.safe_load((EXPERIMENT_DIR / "config.yaml").read_text())
OUTPUT_DIR = Path("/kaggle/working/artifacts")
if not Path("/kaggle/working").is_dir():
    raise RuntimeError("Stage I must run in Kaggle Notebook; local execution is disabled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

inference_config = CONFIG["inference"]
stage_i_config = CONFIG["stages"]["stage_i"]
expected_inventory = {
    "lightgbm_boosters": 24,
    "catboost_boosters": 16,
    "total_boosters": 40,
    "ridge_models": 2,
    "exp413_retraining": 0,
    "exp413_reinference": 0,
}
observed_inventory = {name: stage_i_config[name] for name in expected_inventory}
if observed_inventory != expected_inventory:
    raise RuntimeError(f"Stage I inventory changed: {observed_inventory}")
if (
    CONFIG["experiment"]["route"] != "ensemble"
    or not CONFIG["experiment"]["inference_enabled"]
    or inference_config["mode"] != "post_gate_failure_prediction_only_diagnostic_override"
    or inference_config["selected_train_anchor_unchanged"] != "exp413"
    or inference_config["generate_submission_csv"]
    or inference_config["external_submission"]
    or stage_i_config["submission_enabled"]
):
    raise RuntimeError("Stage I authorization / prediction-only contract failed")
print("experiment", EXPERIMENT)
print("route/mode", CONFIG["experiment"]["route"], inference_config["mode"])
display(pd.DataFrame([observed_inventory]))

# %% [markdown]
# ## 2. Stage P / Stage E / exp413 input contracts

# %%
stage_p_feature_paths = [
    find_artifact(f"stage_p_fold{fold}_physical_features.parquet") for fold in range(5)
]
stage_p_summary_paths = [
    find_artifact(f"stage_p_fold{fold}_summary.json") for fold in range(5)
]
stage_e_summary_path = find_artifact("stage_e_summary.json")
promotion_gate_path = find_artifact("promotion_gate.json")
meta_fold_weights_path = find_artifact("meta_fold_weights.csv")
exp413_prediction_path = find_artifact(
    inference_config["exp413_current_test_prediction_file"]
)
public_runtime_path = find_artifact("public_notebook_replay_audit.py")

stage_e_summary = json.loads(stage_e_summary_path.read_text())
promotion_gate = json.loads(promotion_gate_path.read_text())
if (
    stage_e_summary["status"] != "complete_gate_failed"
    or stage_e_summary["selected_prediction"] != "exp413_oof"
    or promotion_gate["passed"] is not False
    or sha256_file(promotion_gate_path)
    != stage_e_summary["sha256"]["promotion_gate"]
):
    raise RuntimeError("Stage E FAIL / selected-exp413 contract changed")
if (
    sha256_file(meta_fold_weights_path)
    != inference_config["stage_e_meta_fold_weights_sha256"]
    or sha256_file(meta_fold_weights_path)
    != stage_e_summary["sha256"]["meta_fold_weights"]
):
    raise RuntimeError("Stage E meta-fold weight SHA changed")

weights_frame = pd.read_csv(meta_fold_weights_path).sort_values("meta_fold")
observed_weights = weights_frame["public_core_weight"].to_numpy(np.float64)
configured_weights = np.asarray(inference_config["meta_fold_weights"], dtype=np.float64)
if (
    weights_frame["meta_fold"].tolist() != list(range(5))
    or not np.allclose(observed_weights, configured_weights, rtol=0.0, atol=1e-15)
):
    raise RuntimeError("Stage E meta-fold weight values changed")
deployment_weight = float(inference_config["deployment_weight"])
if abs(float(np.median(observed_weights)) - deployment_weight) > 1e-15:
    raise RuntimeError("Deployment weight is not the Stage E five-weight median")

if (
    sha256_file(exp413_prediction_path)
    != inference_config["exp413_current_test_prediction_sha256"]
    or sha256_gzip_decompressed(exp413_prediction_path)
    != inference_config["exp413_current_test_prediction_decompressed_sha256"]
):
    raise RuntimeError("Frozen exp413 current-test prediction SHA changed")

competition_candidates = [
    Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
    Path("/kaggle/input/rogii-wellbore-geology-prediction"),
]
competition_data_dir = next(
    (
        path
        for path in competition_candidates
        if (path / "train").is_dir()
        and (path / "test").is_dir()
        and (path / "sample_submission.csv").is_file()
    ),
    None,
)
if competition_data_dir is None:
    raise FileNotFoundError(f"Competition data not found: {competition_candidates}")
sample_submission_path = competition_data_dir / "sample_submission.csv"
sample = pd.read_csv(sample_submission_path, dtype={"id": str})
if list(sample.columns) != ["id", "tvt"] or sample.empty or sample["id"].duplicated().any():
    raise RuntimeError("Current sample submission identity contract failed")

input_rows = [
    *(
        {
            "role": f"stage_p_fold{fold}",
            "path": str(stage_p_feature_paths[fold]),
            "sha256": sha256_file(stage_p_feature_paths[fold]),
        }
        for fold in range(5)
    ),
    {
        "role": "stage_e_meta_weights",
        "path": str(meta_fold_weights_path),
        "sha256": sha256_file(meta_fold_weights_path),
    },
    {
        "role": "exp413_current_test_prediction",
        "path": str(exp413_prediction_path),
        "sha256": sha256_file(exp413_prediction_path),
    },
    {
        "role": "public_replay_runtime",
        "path": str(public_runtime_path),
        "sha256": sha256_file(public_runtime_path),
    },
]
display(pd.DataFrame(input_rows))
display(weights_frame)
display(sample.head())
print("dynamic current-test sample rows", len(sample))
print("deployment public-core weight", deployment_weight)
print("exp413 retraining/reinference", 0, 0)

# %% [markdown]
# ## 3. Dynamic current-test feature generation

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
    or test_frame["id"].duplicated().any()
    or set(test_frame["id"].astype(str)) != set(sample["id"])
    or test_frame["well"].nunique() <= 0
):
    raise RuntimeError("Dynamic current-test feature identity contract failed")
display(test_well_metadata.head())
display(test_frame[["id", "well", "md_since", "last_known_tvt"]].head())
print(json.dumps(test_feature_summary, indent=2))

# %% [markdown]
# ## 4. Full-train inner-4 model inference

# %%
summary = run_stage_i_current_test(
    output_dir=OUTPUT_DIR,
    stage_p_feature_paths=stage_p_feature_paths,
    stage_p_summary_paths=stage_p_summary_paths,
    test_frame=test_frame,
    test_well_metadata=test_well_metadata,
    sample_submission_path=sample_submission_path,
    exp413_prediction_path=exp413_prediction_path,
    exp413_prediction_sha256=inference_config["exp413_current_test_prediction_sha256"],
    exp413_prediction_decompressed_sha256=inference_config[
        "exp413_current_test_prediction_decompressed_sha256"
    ],
    meta_fold_weights=configured_weights,
    deployment_weight=deployment_weight,
    test_feature_summary=test_feature_summary,
)
if (
    summary["status"] != "complete"
    or summary["fitted_boosters"] != 40
    or summary["fitted_ridge_models"] != 2
    or summary["serialized_model_count"] != 40
    or summary["serialized_lightgbm_count"] != 24
    or summary["serialized_catboost_count"] != 16
    or summary["serialized_model_bytes"] <= 0
    or summary["exp413_retraining"] != 0
    or summary["exp413_reinference"] != 0
    or summary["submission_generated"]
    or summary["external_submission_performed"]
):
    raise RuntimeError("Stage I completion contract failed")

# %% [markdown]
# ## 5. Prediction-only artifacts and stop

# %%
prediction_path = OUTPUT_DIR / summary["outputs"]["predictions"]
predictions = pd.read_csv(prediction_path, dtype={"id": str, "well": str})
model_manifest_path = OUTPUT_DIR / summary["outputs"]["model_manifest"]
model_manifest = json.loads(model_manifest_path.read_text())
serialized_models = [
    *model_manifest["sp45_models"],
    *model_manifest["learned_models"],
]
if len(serialized_models) != 40 or any(
    not (OUTPUT_DIR / row["model_file"]).is_file() for row in serialized_models
):
    raise RuntimeError("Stage I serialized model output contract failed")
display(
    predictions[
        [
            "id",
            "well",
            "exp413_pred_tvt",
            "strict_public_core_pred_tvt",
            "exp497_blend_pred_tvt",
        ]
    ].head()
)
display(pd.DataFrame([summary["prediction_stats"]]))
display(pd.DataFrame([summary["internal_weights"]]))
display(
    pd.DataFrame(serialized_models)[
        [
            "branch",
            "config",
            "inner_fold",
            "kind",
            "model_file",
            "model_sha256",
            "model_bytes",
            "serialization_max_abs",
        ]
    ]
)
print(json.dumps(summary, indent=2))
for forbidden_output in (OUTPUT_DIR / "submission.csv", Path("/kaggle/working/submission.csv")):
    if forbidden_output.exists():
        raise RuntimeError(f"Forbidden submission artifact exists: {forbidden_output}")
print(
    "Stage I complete: prediction-only diagnostic and 40 serialized boosters generated. "
    "exp413 remains the selected train anchor; no submission was generated or submitted."
)
