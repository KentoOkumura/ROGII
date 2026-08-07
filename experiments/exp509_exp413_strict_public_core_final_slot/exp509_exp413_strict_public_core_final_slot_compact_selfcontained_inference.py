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
# # exp509 strict public-core final-slot inference candidate
#
# 保存済みexp497 Stage I modelとdynamic hidden-safe exp413 runtimeだけを読み、
# strict public-coreを再生成して固定weight `0.13716473330712417`で混合する。
# exp497の科学的promotion FAILは変更せず、最終提出第1枠のreference overrideだけを作る。
# booster学習、weight再fit、Public LB固有overlay、外部competition submitは行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Authorization and zero-training inventory
# 3. Saved-model and dynamic input contracts
# 4. Dynamic hidden-safe exp413 generation
# 5. Dynamic strict public-core feature generation
# 6. Saved-model component inference
# 7. Fixed float64 final-slot blend
# 8. Technical audit and reproducibility outputs

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

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

EXPERIMENT = "exp509_exp413_strict_public_core_final_slot"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").is_file():
    PACKAGE_DIR = Path("experiments") / EXPERIMENT
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
if not KAGGLE_INPUT_ROOT.is_dir() or not KAGGLE_WORKING.is_dir():
    raise RuntimeError("exp509 inference must run in Kaggle Notebook")
OUTPUT_DIR = KAGGLE_WORKING / "artifacts"
CORE_OUTPUT_DIR = OUTPUT_DIR / "strict_public_core_runtime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CORE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def prediction_float64_sha256(
    ids: pd.Series,
    values: np.ndarray,
    label: str,
) -> str:
    digest = hashlib.sha256(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        digest.update(str(raw_id).encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    tolerance = float(CONFIG["technical_gate"]["exp413_serialization_max_abs"])
    if max_abs > tolerance:
        raise RuntimeError(f"dynamic exp413 serialization drift is too large: {max_abs}")
    return serialized, max_abs


def isolate_exp413_intermediate_submission(
    source_path: Path,
    destination_path: Path,
    sample_frame: pd.DataFrame,
    exp413_frame: pd.DataFrame,
) -> dict[str, Any]:
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
    comparison = intermediate.merge(
        exp413_frame[["id", "pred_tvt"]],
        on="id",
        validate="one_to_one",
    )
    max_abs = float(
        np.max(
            np.abs(
                comparison["tvt"].to_numpy(np.float64)
                - comparison["pred_tvt"].to_numpy(np.float64)
            )
        )
    )
    if max_abs > float(CONFIG["technical_gate"]["exp413_serialization_max_abs"]):
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


def movement_summary(frame: pd.DataFrame, group_column: str | None = None) -> Any:
    def summarize(group: pd.DataFrame) -> dict[str, Any]:
        delta = group["final_tvt"].to_numpy(np.float64) - group["exp413_tvt"].to_numpy(
            np.float64
        )
        absolute = np.abs(delta)
        return {
            "rows": int(len(group)),
            "movement_rms_ft": float(np.sqrt(np.mean(np.square(delta)))),
            "movement_mae_ft": float(np.mean(absolute)),
            "movement_p95_abs_ft": float(np.quantile(absolute, 0.95)),
            "movement_max_abs_ft": float(np.max(absolute)),
            "movement_mean_signed_ft": float(np.mean(delta)),
        }

    if group_column is None:
        return summarize(frame)
    rows = []
    for key, group in frame.groupby(group_column, sort=True, observed=True):
        rows.append({group_column: str(key), **summarize(group)})
    return rows


# %% [markdown]
# ## 2. Authorization and zero-training inventory

# %%
deployment = CONFIG["inference"]["saved_model_deployment"]
stage = CONFIG["stages"]["stage_i_saved_model_inference"]
ensemble = CONFIG["ensemble"]
inventory = {
    "scientific_variants": int(CONFIG["execution_contract"]["scientific_variants"]),
    "exp497_fitted_boosters": int(stage["fitted_boosters"]),
    "exp497_loaded_boosters": int(stage["loaded_exp497_boosters"]),
    "exp497_loaded_ridge_models": int(stage["loaded_exp497_ridge_models"]),
    "exp413_fitted_boosters": int(stage["exp413_retraining"]),
    "exp413_loaded_boosters": int(stage["loaded_exp413_boosters"]),
    "weight_refit": int(stage["weight_refit"]),
}
expected_inventory = {
    "scientific_variants": 1,
    "exp497_fitted_boosters": 0,
    "exp497_loaded_boosters": 40,
    "exp497_loaded_ridge_models": 2,
    "exp413_fitted_boosters": 0,
    "exp413_loaded_boosters": 75,
    "weight_refit": 0,
}
if (
    CONFIG["experiment"]["route"] != "ensemble"
    or not CONFIG["implementation"]["approved"]
    or deployment["mode"] != "exp509_hidden_safe_saved_model_final_slot"
    or inventory != expected_inventory
    or deployment["public_test_exp413_sidecar_allowed"]
    or deployment["external_submission"]
    or stage["external_submission_enabled"]
    or ensemble["weight_refit"]
    or ensemble["row_gate"]
    or ensemble["well_gate"]
    or ensemble["conditional_router"]
    or ensemble["final_postprocess"] != "none"
):
    raise RuntimeError("exp509 authorization and zero-training contract failed")

EXP413_WEIGHT = float(ensemble["exp413_weight"])
STRICT_PUBLIC_CORE_WEIGHT = float(ensemble["strict_public_core_weight"])
if (
    EXP413_WEIGHT + STRICT_PUBLIC_CORE_WEIGHT != 1.0
    or STRICT_PUBLIC_CORE_WEIGHT != float(deployment["deployment_weight"])
    or STRICT_PUBLIC_CORE_WEIGHT != float(np.median(deployment["meta_fold_weights"]))
):
    raise RuntimeError("exp509 fixed weight contract failed")
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
    str(deployment["model_source_path_token"]),
)
model_artifact_dir = model_manifest_path.parent
public_runtime_path = resolve_unique_input_file(
    "public_notebook_replay_audit.py",
    str(deployment["public_runtime_path_token"]),
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
            "runtime_seconds": exp413_runtime_metrics.get("runtime_seconds"),
            "rows": exp413_runtime_metrics.get("rows"),
            "wells": exp413_runtime_metrics.get("wells"),
            "loaded_boosters": 75,
            "fitted_boosters": 0,
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
    output_dir=CORE_OUTPUT_DIR,
    public_runtime_path=public_runtime_path,
    particles=int(CONFIG["public_core"]["pf"]["particles"]),
    seeds=int(CONFIG["public_core"]["pf"]["seeds"]),
    n_jobs=int(CONFIG["runtime"]["num_workers"]),
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
# ## 6. Saved-model component inference

# %%
exp413_runtime_metrics = dict(exp413_runtime_metrics)
exp413_runtime_metrics["serialization_roundtrip_max_abs"] = (
    exp413_serialization_roundtrip_max_abs
)
exp413_runtime_metrics["intermediate_submission"] = exp413_intermediate_submission_audit
core_summary = run_stage_i_saved_model_inference(
    output_dir=CORE_OUTPUT_DIR,
    artifact_dir=model_artifact_dir,
    expected_artifact_sha256=deployment["artifact_sha256"],
    expected_model_set_sha256=deployment["serialized_model_set_sha256"],
    test_frame=test_frame,
    test_well_metadata=test_well_metadata,
    sample_submission_path=sample_submission_path,
    exp413_prediction_frame=exp413_frame,
    exp413_runtime_metrics=exp413_runtime_metrics,
    test_feature_summary=test_feature_summary,
    submission_output_path=CORE_OUTPUT_DIR / "exp497_intermediate_submission.csv",
    prediction_chunk_size=int(deployment["prediction_chunk_size"]),
    visible_strict_public_core_max_abs=float(
        deployment["visible_strict_public_core_max_abs"]
    ),
    visible_blend_max_abs=float(deployment["visible_blend_max_abs"]),
)
visible_parity = core_summary["visible_parity"]
if (
    core_summary["status"] != "complete"
    or core_summary["fitted_boosters"] != 0
    or core_summary["loaded_exp497_boosters"] != 40
    or core_summary["loaded_exp497_ridge_models"] != 2
    or core_summary["loaded_exp413_boosters"] != 75
    or core_summary["exp413_retraining"] != 0
    or core_summary["exp497_retraining"] != 0
    or core_summary["external_submission_performed"]
    or (
        visible_parity.get("applicable")
        and float(visible_parity.get("exp413_max_abs", np.inf))
        > float(deployment["visible_exp413_max_abs"])
    )
):
    raise RuntimeError("exp509 component inference completion contract failed")

# %% [markdown]
# ## 7. Fixed float64 final-slot blend

# %%
core_prediction_path = CORE_OUTPUT_DIR / "exp497_saved_model_predictions.csv.gz"
core_predictions = pd.read_csv(core_prediction_path, dtype={"id": str, "well": str})
required_core_columns = {
    "id",
    "well",
    "exp413_pred_tvt",
    "strict_public_core_pred_tvt",
}
if (
    sorted(required_core_columns - set(core_predictions.columns))
    or core_predictions["id"].duplicated().any()
    or len(core_predictions) != len(sample)
    or set(core_predictions["id"]) != set(sample["id"])
):
    raise RuntimeError("exp509 component prediction identity contract failed")

components = sample[["id"]].merge(
    core_predictions[
        ["id", "well", "exp413_pred_tvt", "strict_public_core_pred_tvt"]
    ],
    on="id",
    how="left",
    validate="one_to_one",
).merge(
    test_frame[["id", "md_since"]],
    on="id",
    how="left",
    validate="one_to_one",
)
components = components.rename(
    columns={
        "exp413_pred_tvt": "exp413_tvt",
        "strict_public_core_pred_tvt": "strict_public_core_tvt",
    }
)
exp413_values = components["exp413_tvt"].to_numpy(np.float64)
strict_values = components["strict_public_core_tvt"].to_numpy(np.float64)
final_values = EXP413_WEIGHT * exp413_values + STRICT_PUBLIC_CORE_WEIGHT * strict_values
components["final_tvt"] = final_values
numeric_columns = ["md_since", "exp413_tvt", "strict_public_core_tvt", "final_tvt"]
if (
    components["well"].isna().any()
    or components["well"].astype(str).str.len().eq(0).any()
    or components[numeric_columns].isna().any().any()
    or not np.isfinite(components[numeric_columns].to_numpy(np.float64)).all()
    or not components["id"].equals(sample["id"])
):
    raise RuntimeError("exp509 fixed blend output contract failed")

formula_reference = np.add(
    np.multiply(EXP413_WEIGHT, exp413_values, dtype=np.float64),
    np.multiply(STRICT_PUBLIC_CORE_WEIGHT, strict_values, dtype=np.float64),
    dtype=np.float64,
)
formula_parity_max_abs = float(np.max(np.abs(final_values - formula_reference)))
if formula_parity_max_abs > float(CONFIG["technical_gate"]["formula_parity_max_abs"]):
    raise RuntimeError(f"exp509 fixed formula parity failed: {formula_parity_max_abs}")

component_path = OUTPUT_DIR / "exp509_component_predictions.csv.gz"
components.to_csv(component_path, index=False, compression="gzip", float_format="%.17g")
component_reload = pd.read_csv(component_path, dtype={"id": str, "well": str})
component_reload_max_abs = float(
    np.max(
        np.abs(
            component_reload[["exp413_tvt", "strict_public_core_tvt", "final_tvt"]].to_numpy(
                np.float64
            )
            - components[["exp413_tvt", "strict_public_core_tvt", "final_tvt"]].to_numpy(
                np.float64
            )
        )
    )
)
if (
    not component_reload["id"].equals(sample["id"])
    or component_reload_max_abs > float(CONFIG["technical_gate"]["formula_parity_max_abs"])
):
    raise RuntimeError("exp509 component serialization parity failed")

# %% [markdown]
# ## 8. Technical audit and reproducibility outputs

# %%
components["horizon_bucket"] = pd.cut(
    components["md_since"],
    bins=[-np.inf, 250.0, 1000.0, np.inf],
    labels=["000_250", "250_1000", "1000_plus"],
    right=False,
)
start_rows = (
    components.sort_values(["well", "md_since", "id"])
    .groupby("well", sort=True, observed=True)
    .head(1)
)
difference_summary = {
    "pooled": movement_summary(components),
    "by_well": movement_summary(components, "well"),
    "by_horizon": movement_summary(components, "horizon_bucket"),
    "start_rows": movement_summary(start_rows),
    "prediction_stats": {
        column: {
            "mean": float(components[column].mean()),
            "std": float(components[column].std()),
            "min": float(components[column].min()),
            "max": float(components[column].max()),
        }
        for column in ["exp413_tvt", "strict_public_core_tvt", "final_tvt"]
    },
}
difference_summary_path = OUTPUT_DIR / "exp509_prediction_difference_summary.json"
write_json(difference_summary_path, difference_summary)

submission = component_reload[["id", "final_tvt"]].rename(columns={"final_tvt": "tvt"})
submission_path = KAGGLE_WORKING / "submission.csv"
submission.to_csv(submission_path, index=False, float_format="%.17g")
submission_reload = pd.read_csv(submission_path, dtype={"id": str})
if (
    list(submission_reload.columns) != ["id", "tvt"]
    or len(submission_reload) != len(sample)
    or not submission_reload["id"].equals(sample["id"])
    or submission_reload["id"].duplicated().any()
    or not np.isfinite(submission_reload["tvt"].to_numpy(np.float64)).all()
):
    raise RuntimeError("exp509 submission output contract failed")

technical_checks = {
    "artifact_sha_exact": saved_artifacts["sha256"] == deployment["artifact_sha256"],
    "model_set_sha_exact": (
        saved_artifacts["serialization"]["serialized_model_set_sha256"]
        == deployment["serialized_model_set_sha256"]
    ),
    "id_one_to_one": not component_reload["id"].duplicated().any(),
    "sample_id_order_exact": component_reload["id"].equals(sample["id"]),
    "duplicate_ids_zero": int(component_reload["id"].duplicated().sum()) == 0,
    "nonfinite_predictions_zero": bool(
        np.isfinite(
            component_reload[["exp413_tvt", "strict_public_core_tvt", "final_tvt"]].to_numpy(
                np.float64
            )
        ).all()
    ),
    "nonempty_wells": bool(component_reload["well"].astype(str).str.len().gt(0).all()),
    "weights_constant_all_rows": True,
    "weights_sum_exactly_one": EXP413_WEIGHT + STRICT_PUBLIC_CORE_WEIGHT == 1.0,
    "formula_parity": (
        formula_parity_max_abs <= float(CONFIG["technical_gate"]["formula_parity_max_abs"])
    ),
    "component_serialization_parity": (
        component_reload_max_abs <= float(CONFIG["technical_gate"]["formula_parity_max_abs"])
    ),
    "zero_new_training": inventory == expected_inventory,
    "external_submission_false": not deployment["external_submission"],
}
if not all(technical_checks.values()):
    failed = sorted(name for name, passed in technical_checks.items() if not passed)
    raise RuntimeError(f"exp509 technical gate failed: {failed}")

input_manifest = {
    "competition_sample": {
        "path": str(sample_submission_path),
        "sha256": sha256_file(sample_submission_path),
        "rows": len(sample),
    },
    "exp413_dynamic": {
        "runtime_source": str(exp413_runtime_path),
        "runtime_source_sha256": sha256_file(exp413_runtime_path),
        "parent_source_sha256": EXP413_PARENT_SOURCE_SHA256,
        "prediction_path": str(exp413_prediction_path),
        "prediction_gzip_content_sha256": sha256_gzip_content(exp413_prediction_path),
        "serialization_roundtrip_max_abs": exp413_serialization_roundtrip_max_abs,
    },
    "strict_public_core_saved_model": {
        "model_source_kernel": deployment["model_source_kernel"],
        "model_source_kernel_version": deployment["model_source_kernel_version"],
        "serialized_model_set_sha256": deployment["serialized_model_set_sha256"],
        "artifact_sha256": saved_artifacts["sha256"],
        "public_runtime_path": str(public_runtime_path),
        "public_runtime_sha256": sha256_file(public_runtime_path),
        "test_feature_summary": test_feature_summary,
        "visible_parity": visible_parity,
    },
}
input_manifest_path = OUTPUT_DIR / "exp509_input_manifest.json"
write_json(input_manifest_path, input_manifest)

prediction_sha = {
    "exp413_float64": prediction_float64_sha256(
        component_reload["id"],
        component_reload["exp413_tvt"].to_numpy(np.float64),
        "exp509:exp413",
    ),
    "strict_public_core_float64": prediction_float64_sha256(
        component_reload["id"],
        component_reload["strict_public_core_tvt"].to_numpy(np.float64),
        "exp509:strict_public_core",
    ),
    "final_float64": prediction_float64_sha256(
        component_reload["id"],
        component_reload["final_tvt"].to_numpy(np.float64),
        "exp509:final",
    ),
}
reproducibility = {
    "experiment": EXPERIMENT,
    "status": "technical_gate_pass",
    "route": "ensemble",
    "scientific_promotion": False,
    "scientific_decision_source": "exp497_preregistered_gate_fail_preserved",
    "seed_policy": CONFIG["reproducibility"]["seed_policy"],
    "weights": {
        "exp413": EXP413_WEIGHT,
        "strict_public_core": STRICT_PUBLIC_CORE_WEIGHT,
    },
    "inventory": inventory,
    "technical_checks": technical_checks,
    "formula_parity_max_abs": formula_parity_max_abs,
    "component_reload_max_abs": component_reload_max_abs,
    "prediction_sha256": prediction_sha,
    "file_sha256": {
        "config": sha256_file(PACKAGE_DIR / "config.yaml"),
        "component_predictions_raw_gzip": sha256_file(component_path),
        "component_predictions_decompressed": sha256_gzip_content(component_path),
        "difference_summary": sha256_file(difference_summary_path),
        "input_manifest": sha256_file(input_manifest_path),
        "submission": sha256_file(submission_path),
    },
    "upstream_core_summary": core_summary,
    "deterministic_anchor": False,
    "deterministic_anchor_note": "requires same-source rerun final/submission SHA match",
    "external_submission_performed": False,
}
reproducibility_path = OUTPUT_DIR / "exp509_reproducibility_manifest.json"
write_json(reproducibility_path, reproducibility)

display(submission_reload.head())
display(pd.DataFrame([difference_summary["pooled"]]))
display(pd.DataFrame([visible_parity]))
print(json.dumps(reproducibility, indent=2))
print(
    "exp509 candidate complete: 40 exp497 + 75 exp413 saved boosters loaded, "
    "0 boosters fitted, fixed float64 blend written, no external submit performed."
)
