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
# # exp402 fold-safe GRWR-5 add-only on exp287 — Stage 1 train
#
# Stage 0で固定したouter-fold別GRWR-5 10 partitionを、exp287と同じ
# clean-273 + nested compact-74 + fold-safe formation-74へadd-onlyする。
# 学習対象は1 variant / 3 LightGBM configs / 5 folds = 15 GPU boostersだけで、
# exp287 / exp264 controlは保存済みOOFを比較に使い、再学習しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Stage 1 execution and GPU cost contract
# 3. Stage 0 and frozen parent evidence
# 4. Clean, compact, formation, and GRWR-5 input assembly
# 5. LightGBM family and 426-feature contract
# 6. Promotion metrics and well-tail guards
# 7. Fifteen-booster GPU training
# 8. Metrics, model manifests, and reproducibility evidence
# 9. Execution orchestration and generated files

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    build_stage_d_exp218_surface,
    load_stage_d_compact_fold,
    resolve_stage_c_artifact_root,
    verify_stage_c_artifact_root,
)
from src.fold_safe_formation_pipeline import (
    logical_feature_content_sha256,
    select_unique_columns,
)


EXPERIMENT_NAME = "exp402_fold_safe_grwr_5_addonly_on_exp287"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP402_STAGE1_IMPORT_ONLY", "0") == "1"
GRWR5_FEATURES = [
    "grwr_candidate_tvt_std",
    "grwr_candidate_tvt_range",
    "grwr_dwt_energy_ratio_w065_x_candidate_std",
    "grwr_fft_rotation_ratio_x_candidate_range",
    "grwr_dwt_minus_raw_ncc_gap_x_candidate_range",
]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(dotted_key)
        value = value[part]
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(Path(path), "rb") as handle:  # type: ignore[arg-type]
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_identity_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = ("id", "well"),
) -> str:
    ordered = frame.sort_values(list(columns), kind="stable").reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(
        ordered[list(columns)], index=False, categorize=True
    ).to_numpy(dtype="<u8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"pandas_hash_pandas_object_categorize_v1\0")
    digest.update(hashes.tobytes(order="C"))
    return digest.hexdigest()


def logical_float_frame_sha256(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
) -> str:
    columns = list(map(str, value_columns))
    if len(columns) != len(set(columns)):
        raise ValueError("logical float SHA received duplicate feature names")
    required = {"id", "well", *columns}
    if missing := required - set(frame.columns):
        raise ValueError(f"logical float SHA columns missing: {sorted(missing)}")
    ordered = frame[["id", "well", *columns]].copy()
    ordered["id"] = ordered["id"].astype(str)
    ordered["well"] = ordered["well"].astype(str)
    if ordered["id"].duplicated().any():
        raise ValueError("logical float SHA received duplicate IDs")
    ordered = ordered.sort_values("id", kind="stable").reset_index(drop=True)
    values = ordered[columns].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("logical float SHA received non-finite values")
    digest = hashlib.sha256(b"exp402-id-sorted-float32-v1\n")
    digest.update(
        json.dumps(columns, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    for identifier, well in ordered[["id", "well"]].itertuples(
        index=False, name=None
    ):
        for value in (identifier, well):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(dict(payload)),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    return sha256_file(path)


def current_peak_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.exists()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp402 config resolution is ambiguous: {matches}")
    return matches[0]


def find_competition_input_root() -> Path:
    preferred = [
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    ]
    candidates = [
        path.resolve()
        for path in preferred
        if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
    ]
    if not candidates and KAGGLE_INPUT_ROOT.exists():
        candidates = [
            path.resolve()
            for path in KAGGLE_INPUT_ROOT.glob("*/*")
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        ]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"competition input with train/test directories was not unique: {candidates}"
        )
    return candidates[0]


def resolve_existing_path(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    label: str,
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.exists():
            candidates.append(direct.resolve())
        if not direct.is_absolute() and (PACKAGE_DIR / direct).exists():
            candidates.append((PACKAGE_DIR / direct).resolve())
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(
                    path.resolve() for path in root.glob(raw) if path.exists()
                )
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(f"{label} not found for patterns={list(patterns)}")
    return unique[0]


def resolve_artifact_root(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    required_files: Sequence[str],
    required_file_sha256: Mapping[str, str] | None = None,
    label: str,
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.is_dir():
            candidates.append(direct.resolve())
        if not direct.is_absolute() and (PACKAGE_DIR / direct).is_dir():
            candidates.append((PACKAGE_DIR / direct).resolve())
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(
                    path.resolve() for path in root.glob(raw) if path.is_dir()
                )
        if required_files:
            candidates.extend(
                path.parent.resolve()
                for path in root.rglob(str(required_files[0]))
                if path.is_file()
            )
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        checked.append(str(candidate))
        if not all((candidate / name).is_file() for name in required_files):
            continue
        expected = dict(required_file_sha256 or {})
        if any(
            sha256_file(candidate / name) != str(expected[name])
            for name in required_files
            if name in expected
        ):
            continue
        return candidate
    raise FileNotFoundError(
        f"complete {label} root not found; checked={json.dumps(checked[:60])}"
    )


def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def verify_t4_runtime() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("exp402 Stage 1 requires a T4 but nvidia-smi is absent")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not rows or not any("T4" in row for row in rows):
        raise RuntimeError(f"exp402 Stage 1 requires Nvidia T4; detected={rows}")
    evidence = {
        "nvidia_smi": executable,
        "gpu_rows": rows,
        "t4_detected": True,
    }
    print("GPU runtime verified", json.dumps(evidence, sort_keys=True), flush=True)
    return evidence


# %% [markdown]
# ## 2. Stage 1 execution and GPU cost contract
#
# 実行時はtraining implementation / run / Kaggle pushの承認flagをすべて要求する。
# active surfaceはGRWR-5 add-only 1 variantだけで、control boosterは0。

# %%
def validate_stage1_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("exp402 experiment name changed")
    if nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp402 route must remain ml_model")
    if nested(config, "lineage.parent") != (
        "exp287_fold_safe_formation_74_addonly_on_exp264"
    ):
        raise ValueError("exp402 parent must remain exp287")
    if not bool(nested(config, "implementation.training_implemented")):
        raise PermissionError("exp402 Stage 1 training is not implemented")
    if str(nested(config, "execution.current_stage")) != (
        "fold_safe_grwr_5_addonly_train"
    ):
        raise PermissionError("exp402 Stage 1 train stage is not selected")
    approval_keys = [
        "execution.training_implementation_approved",
        "execution.training_run_approved",
        "execution.kaggle_push_approved",
        "execution.run_train",
        "execution.run_approved",
    ]
    if not all(bool(nested(config, key)) for key in approval_keys):
        raise PermissionError(f"exp402 Stage 1 approval is incomplete: {approval_keys}")
    if bool(nested(config, "execution.run_inference")) or bool(
        nested(config, "execution.create_submission")
    ):
        raise PermissionError("exp402 inference and submission remain disabled")

    training = dict(nested(config, "model.training"))
    variants = [str(value) for value in training["active_variants"]]
    config_indices = [
        int(value) for value in training["lightgbm_config_indices"]
    ]
    folds = int(training["folds"])
    boosters = len(variants) * len(config_indices) * folds
    if variants != ["fold_safe_grwr_5_addonly"]:
        raise ValueError("exp402 active variant changed")
    if config_indices != [0, 1, 2] or folds != 5 or boosters != 15:
        raise ValueError("exp402 GPU cost must remain 1 x 3 x 5 = 15")
    if int(training["planned_gpu_boosters"]) != 15:
        raise ValueError("exp402 planned booster count changed")
    if bool(training["control_retraining"]):
        raise ValueError("exp287/exp264 control retraining is forbidden")
    if not bool(training["enabled"]):
        raise ValueError("exp402 Stage 1 training is disabled")

    runtime = dict(nested(config, "runtime.kaggle"))
    if (
        not bool(runtime["enable_gpu"])
        or str(runtime["machine_shape"]) != "NvidiaTeslaT4"
        or bool(runtime["enable_internet"])
    ):
        raise ValueError("exp402 Stage 1 must use private T4 with internet off")
    expected_kernel_sources = {
        "kentookumura/exp072-exp063-full-replay-feature-cache-train",
        "kentookumura/exp145-train",
        "kentookumura/exp264-exp263-confidence-dual-selector-train",
        "kentookumura/exp264-exp263-confidence-dual-selector-tvt-train",
        "kentookumura/exp287-foldsafe-form74-addonly-exp264-train",
        "kentookumura/exp402-foldsafe-grwr5-stage0-aggregate",
        "kentookumura/exp402-foldsafe-grwr5-train-fold0",
        "kentookumura/exp402-foldsafe-grwr5-train-fold1",
        "kentookumura/exp402-foldsafe-grwr5-train-fold2",
        "kentookumura/exp402-foldsafe-grwr5-train-fold3",
        "kentookumura/exp402-foldsafe-grwr5-train-fold4-v2",
    }
    kernel_sources = [
        str(value) for value in runtime.get("train_kernel_sources", [])
    ]
    if len(kernel_sources) != 11 or len(set(kernel_sources)) != 11:
        raise ValueError("exp402 Stage 1 must attach exactly 11 unique kernel inputs")
    if set(kernel_sources) != expected_kernel_sources:
        raise ValueError("exp402 Stage 1 kernel input contract changed")
    stage1 = dict(nested(config, "runtime.stage_1"))
    expected_stage1 = {
        "accelerator": "gpu",
        "machine_shape": "NvidiaTeslaT4",
        "gpu_use_dp": True,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 8,
    }
    if stage1 != expected_stage1:
        raise ValueError("exp402 Stage 1 reproducibility flags changed")
    if list(nested(config, "grwr5.ordered_features")) != [
        {
            "name": name,
            "formula": formula,
        }
        for name, formula in [
            (
                "grwr_candidate_tvt_std",
                "numpy_std_float32_over_eight_candidate_tvt_values_ddof0",
            ),
            (
                "grwr_candidate_tvt_range",
                "numpy_max_minus_min_float32_over_eight_candidate_tvt_values",
            ),
            (
                "grwr_dwt_energy_ratio_w065_x_candidate_std",
                "grwr_dwt_detail_energy_ratio_w065_times_grwr_candidate_tvt_std_float32",
            ),
            (
                "grwr_fft_rotation_ratio_x_candidate_range",
                "grwr_fft_rotation_energy_ratio_times_grwr_candidate_tvt_range_float32",
            ),
            (
                "grwr_dwt_minus_raw_ncc_gap_x_candidate_range",
                "grwr_dwt_approx_minus_raw_default_candidate_ncc_times_grwr_candidate_tvt_range_float32",
            ),
        ]
    ]:
        raise ValueError("exp402 GRWR-5 formula contract changed")
    if sha256_json(GRWR5_FEATURES) != str(
        nested(config, "grwr5.feature_schema_sha256")
    ):
        raise ValueError("exp402 GRWR-5 schema SHA changed")
    return {
        "active_variants": 1,
        "variant": variants[0],
        "lightgbm_configs": len(config_indices),
        "config_indices": config_indices,
        "folds": folds,
        "planned_gpu_boosters": boosters,
        "control_retraining_boosters": 0,
        "inference": False,
        "submission": False,
    }


# %% [markdown]
# ## 3. Stage 0 and frozen parent evidence
#
# aggregate v2のpreflight / reproducibility / 10-role manifestをfile SHAで固定する。
# 各fold shardのmanifestもaggregateが記録したSHAと照合し、実GRWR parquetは
# 学習foldでreadした時にphysical / logical content SHAを再確認する。

# %%
def verify_stage0_evidence(
    config: Mapping[str, Any],
    *,
    search_roots: Sequence[Path],
) -> tuple[pd.DataFrame, dict[int, Path], dict[str, Any]]:
    stage0 = dict(nested(config, "data.stage_0_aggregate"))
    names = {
        "partition": f"{OUTPUT_PREFIX}_partition_manifest.csv",
        "preflight": f"{OUTPUT_PREFIX}_preflight_manifest.json",
        "reproducibility": f"{OUTPUT_PREFIX}_reproducibility_manifest.json",
    }
    aggregate_root = resolve_artifact_root(
        [str(value) for value in stage0["artifact_root_patterns"]],
        search_roots,
        required_files=list(names.values()),
        required_file_sha256={
            filename: str(stage0[f"{key}_sha256"])
            for key, filename in names.items()
        },
        label="exp402 Stage 0 aggregate",
    )
    file_sha = {
        key: verify_file_sha(
            aggregate_root / filename,
            str(stage0[f"{key}_sha256"]),
            f"exp402 Stage 0 aggregate {key}",
        )
        for key, filename in names.items()
    }
    preflight = json.loads((aggregate_root / names["preflight"]).read_text())
    reproducibility = json.loads(
        (aggregate_root / names["reproducibility"]).read_text()
    )
    if (
        not bool(preflight.get("passed"))
        or preflight.get("status") != "zero_booster_preflight_passed"
        or not all(bool(value) for value in dict(preflight["checks"]).values())
        or len(preflight["checks"]) != 18
    ):
        raise ValueError("exp402 Stage 0 aggregate is not the fixed 18/18 PASS")
    if not bool(reproducibility.get("passed")):
        raise ValueError("exp402 Stage 0 reproducibility manifest did not pass")
    if str(reproducibility.get("preflight_manifest_sha256")) != file_sha[
        "preflight"
    ]:
        raise ValueError("Stage 0 preflight/reproducibility SHA link changed")
    if dict(preflight["cost_contract"]["future_training"]) != {
        "control_boosters": 0,
        "folds": 5,
        "gpu_boosters": 15,
        "lightgbm_configs": 3,
        "variants": 1,
    }:
        raise ValueError("Stage 0 future training contract changed")
    partitions = pd.read_csv(aggregate_root / names["partition"])
    if len(partitions) != 10 or partitions.duplicated(
        ["downstream_outer_fold", "role"]
    ).any():
        raise ValueError("Stage 0 aggregate must contain 10 unique roles")
    if set(partitions["downstream_outer_fold"].astype(int)) != set(range(5)):
        raise ValueError("Stage 0 aggregate fold coverage changed")
    if set(partitions["role"].astype(str)) != {"train", "valid"}:
        raise ValueError("Stage 0 aggregate role coverage changed")
    if not partitions["grwr5_schema_sha256"].astype(str).eq(
        sha256_json(GRWR5_FEATURES)
    ).all():
        raise ValueError("Stage 0 GRWR-5 schema changed")
    if partitions["formation_target_columns_read"].astype(bool).any():
        raise ValueError("Stage 0 role read target formation columns")
    if int(partitions["historical_grwr5_values_loaded"].sum()) != 0 or int(
        partitions["entropy_or_score_features_loaded"].sum()
    ) != 0:
        raise ValueError("Stage 0 role loaded forbidden historical values")

    fold_roots: dict[int, Path] = {}
    fold_config = dict(nested(config, "data.stage_0_train_folds"))
    preflight_folds = dict(preflight["input"]["train_folds"])
    fold_manifest_name = f"{OUTPUT_PREFIX}_stage0_train_fold_manifest.json"
    partition_manifest_name = f"{OUTPUT_PREFIX}_partition_manifest.csv"
    for outer_fold in range(5):
        expected = dict(preflight_folds[str(outer_fold)])
        fold_root = resolve_artifact_root(
            [
                str(value)
                for value in fold_config[str(outer_fold)]["artifact_root_patterns"]
            ],
            search_roots,
            required_files=[fold_manifest_name, partition_manifest_name],
            required_file_sha256={
                fold_manifest_name: str(expected["manifest_file_sha256"]),
                partition_manifest_name: str(
                    expected["partition_manifest_file_sha256"]
                ),
            },
            label=f"exp402 Stage 0 outer-fold {outer_fold}",
        )
        verify_file_sha(
            fold_root / fold_manifest_name,
            str(expected["manifest_file_sha256"]),
            f"exp402 outer-fold {outer_fold} manifest",
        )
        verify_file_sha(
            fold_root / partition_manifest_name,
            str(expected["partition_manifest_file_sha256"]),
            f"exp402 outer-fold {outer_fold} partition manifest",
        )
        fold_manifest = json.loads((fold_root / fold_manifest_name).read_text())
        if (
            not bool(fold_manifest.get("passed"))
            or int(fold_manifest.get("outer_fold", -1)) != outer_fold
        ):
            raise ValueError(f"exp402 Stage 0 fold {outer_fold} did not pass")
        fold_roots[outer_fold] = fold_root
    return partitions, fold_roots, {
        "root": str(aggregate_root),
        "file_sha256": file_sha,
        "status": str(preflight["status"]),
        "checks_passed": 18,
        "partition_count": len(partitions),
        "stage_0_config_sha256": str(
            preflight["execution_identity"]["config_sha256"]
        ),
        "stage_0_implementation_source_sha256": str(
            preflight["execution_identity"]["implementation_source_sha256"]
        ),
        "scientific_contract_sha256": str(
            preflight["execution_identity"]["scientific_contract_sha256"]
        ),
        "fold_roots": {
            str(key): str(value) for key, value in fold_roots.items()
        },
    }


def verify_parent_artifacts(
    config: Mapping[str, Any],
    *,
    search_roots: Sequence[Path],
) -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parent = dict(nested(config, "data.parent_exp287"))
    fixed = {
        "oof": (
            "fold_safe_formation_oof_predictions.parquet",
            parent["oof_sha256"],
        ),
        "model_manifest": ("model_manifest.json", parent["model_manifest_sha256"]),
        "metrics": ("metrics.json", parent["metrics_sha256"]),
        "fold_metrics": ("fold_metrics.csv", parent["fold_metrics_sha256"]),
        "by_well": ("by_well_metrics.csv", parent["by_well_sha256"]),
        "formation_manifest": (
            "formation_fold_manifest.json",
            parent["formation_fold_manifest_sha256"],
        ),
        "raw_schema": (
            "raw_train_current_test_schema_audit.csv",
            parent["raw_schema_audit_sha256"],
        ),
    }
    parent_root = resolve_artifact_root(
        [str(value) for value in parent["artifact_root_patterns"]],
        search_roots,
        required_files=[value[0] for value in fixed.values()],
        required_file_sha256={
            name: str(expected) for name, expected in fixed.values()
        },
        label="saved exp287 train artifacts",
    )
    file_sha = {
        key: verify_file_sha(
            parent_root / name, str(expected), f"saved exp287 {key}"
        )
        for key, (name, expected) in fixed.items()
    }
    model_manifest = json.loads((parent_root / "model_manifest.json").read_text())
    if int(model_manifest.get("model_count", -1)) != 15:
        raise ValueError("saved exp287 must contain exactly 15 models")
    groups = dict(model_manifest.get("feature_groups") or {})
    if {
        key: len(groups.get(key, []))
        for key in ["clean_base", "nested_compact", "fold_safe_formation"]
    } != {
        "clean_base": 273,
        "nested_compact": 74,
        "fold_safe_formation": 74,
    }:
        raise ValueError("saved exp287 feature groups changed")
    parent_features = [
        str(feature)
        for group in ["clean_base", "nested_compact", "fold_safe_formation"]
        for feature in groups[group]
    ]
    if len(parent_features) != 421 or len(set(parent_features)) != 421:
        raise ValueError("saved exp287 421-feature schema changed")
    if sha256_json(parent_features) != str(parent["feature_schema_sha256"]):
        raise ValueError("saved exp287 logical schema SHA changed")
    if set(parent_features).intersection(GRWR5_FEATURES):
        raise ValueError("saved exp287 already contains GRWR-5")

    oof = pd.read_parquet(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        columns=[
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            "fold_safe_formation_74_addonly__lgb_mean__pred_tvt",
        ],
    )
    for column in ["id", "well"]:
        oof[column] = oof[column].astype(str)
    oof["outer_fold"] = pd.to_numeric(
        oof["outer_fold"], errors="raise"
    ).astype(np.int8)
    if oof["id"].duplicated().any() or set(oof["outer_fold"]) != set(range(5)):
        raise ValueError("saved exp287 OOF identity/fold contract changed")

    formation_manifest = json.loads(
        (parent_root / "formation_fold_manifest.json").read_text()
    )
    formation_partitions = pd.DataFrame(
        formation_manifest.get("partitions") or []
    )
    if len(formation_partitions) != 10:
        raise ValueError("saved exp287 formation roles changed")
    if str(formation_manifest.get("feature_schema_sha256")) != sha256_json(
        list(groups["fold_safe_formation"])
    ):
        raise ValueError("saved exp287 formation schema changed")
    return parent_root, model_manifest, oof, formation_partitions, {
        "root": str(parent_root),
        "file_sha256": file_sha,
        "model_count": 15,
        "feature_count": 421,
        "feature_schema_sha256": sha256_json(parent_features),
    }


# %% [markdown]
# ## 4. Clean, compact, formation, and GRWR-5 input assembly
#
# clean-273は親exp287と同じexp218 reconstruction + allowlist、
# compact-74はcorrected exp264 Stage C、formation-74はexp287保存role、
# GRWR-5はStage 0 fold shardから読む。targetは全featureを固定した後の学習だけに使う。

# %%
def align_oof_to_base(
    path: Path,
    *,
    expected_sha256: str,
    columns: Sequence[str],
    base_frame: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual_sha = verify_file_sha(path, expected_sha256, label)
    frame = pd.read_parquet(path, columns=list(columns))
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    indexed = frame.set_index("id", drop=False)
    base_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(base_ids):
        raise ValueError(f"{label} IDs differ from clean base")
    aligned = indexed.loc[base_ids].reset_index(drop=True)
    if not aligned["well"].equals(
        base_frame["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"{label} wells differ from clean base")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    actual = aligned["actual_tvt"].to_numpy(np.float32)
    if float(np.max(np.abs(truth - actual), initial=0.0)) > 1.0e-4:
        raise ValueError(f"{label} truth differs from clean base")
    return aligned, {
        "path": str(path),
        "sha256": actual_sha,
        "rows": len(aligned),
        "wells": int(aligned["well"].nunique()),
    }


def load_saved_formation_role(
    *,
    parent_root: Path,
    partition_manifest: pd.DataFrame,
    outer_fold: int,
    role: str,
    compact: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = partition_manifest[
        partition_manifest["downstream_outer_fold"].astype(int).eq(outer_fold)
        & partition_manifest["role"].astype(str).eq(role)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"formation role is not unique: outer={outer_fold}, role={role}"
        )
    item = rows.iloc[0]
    path = parent_root / str(item["path"])
    frame = pd.read_parquet(path, columns=["id", "well", *feature_names])
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if len(frame) != int(item["rows"]) or frame["id"].duplicated().any():
        raise ValueError("saved formation role row/ID contract changed")
    logical_sha = logical_feature_content_sha256(frame, feature_names)
    if logical_sha != str(item["logical_content_sha256"]):
        raise ValueError("saved formation logical content SHA changed")
    compact_ids = compact["id"].astype(str).reset_index(drop=True)
    if not frame["id"].reset_index(drop=True).equals(compact_ids):
        indexed = frame.set_index("id", drop=False)
        if set(indexed.index) != set(compact_ids):
            raise ValueError("saved formation IDs differ from compact role")
        frame = indexed.loc[compact_ids].reset_index(drop=True)
    if not frame["well"].equals(compact["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved formation wells differ from compact role")
    if not np.isfinite(frame[list(feature_names)].to_numpy(np.float32)).all():
        raise ValueError("saved formation role contains non-finite values")
    return frame, {
        "component": "fold_safe_formation",
        "path": str(path),
        "file_sha256": str(item["file_sha256"]),
        "logical_content_sha256": logical_sha,
        "rows": len(frame),
        "outer_fold": outer_fold,
        "role": role,
    }


def load_saved_grwr5_role(
    *,
    fold_root: Path,
    partition_manifest: pd.DataFrame,
    outer_fold: int,
    role: str,
    compact: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = partition_manifest[
        partition_manifest["downstream_outer_fold"].astype(int).eq(outer_fold)
        & partition_manifest["role"].astype(str).eq(role)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"GRWR-5 role is not unique: outer={outer_fold}, role={role}"
        )
    item = rows.iloc[0]
    path = fold_root / str(item["path"])
    physical_sha = verify_file_sha(
        path,
        str(item["file_sha256"]),
        f"exp402 GRWR-5 outer{outer_fold} {role}",
    )
    frame = pd.read_parquet(path, columns=["id", "well", *GRWR5_FEATURES])
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if len(frame) != int(item["rows"]) or frame["id"].duplicated().any():
        raise ValueError("saved GRWR-5 role row/ID contract changed")
    logical_sha = logical_float_frame_sha256(
        frame,
        value_columns=GRWR5_FEATURES,
    )
    if logical_sha != str(item["grwr5_logical_content_sha256"]):
        raise ValueError("saved GRWR-5 logical content SHA changed")
    compact_ids = compact["id"].astype(str).reset_index(drop=True)
    if not frame["id"].reset_index(drop=True).equals(compact_ids):
        indexed = frame.set_index("id", drop=False)
        if set(indexed.index) != set(compact_ids):
            raise ValueError("saved GRWR-5 IDs differ from compact role")
        frame = indexed.loc[compact_ids].reset_index(drop=True)
    if not frame["well"].equals(compact["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved GRWR-5 wells differ from compact role")
    if not np.isfinite(frame[GRWR5_FEATURES].to_numpy(np.float32)).all():
        raise ValueError("saved GRWR-5 role contains non-finite values")
    return frame, {
        "component": "fold_safe_grwr5",
        "path": str(path),
        "file_sha256": physical_sha,
        "logical_content_sha256": logical_sha,
        "rows": len(frame),
        "outer_fold": outer_fold,
        "role": role,
    }


# %% [markdown]
# ## 5. LightGBM family and 426-feature contract
#
# exp287 model manifestの3 configと、同じexp218 source/configから復元した
# `gpu_repro_guard_dp_threads8` parameter familyをJSON等価で照合する。

# %%
def prepare_stage1_inputs(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    cost = validate_stage1_contract(config)
    search_roots = [KAGGLE_INPUT_ROOT, Path("/tmp"), PACKAGE_DIR]
    grwr_partitions, fold_roots, stage0_evidence = verify_stage0_evidence(
        config, search_roots=search_roots
    )
    (
        parent_root,
        parent_manifest,
        _parent_oof_raw,
        formation_partitions,
        parent_evidence,
    ) = verify_parent_artifacts(config, search_roots=search_roots)

    stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
    stage_c_evidence = verify_stage_c_artifact_root(
        stage_c_root,
        config,
        verify_partition_sha256=False,
        expected_compact_feature_count=74,
    )
    historical = dict(nested(config, "data.historical_formula"))
    exp218_source = resolve_existing_path(
        [str(value) for value in historical["patterns"]],
        search_roots,
        label="exp218 source",
    )
    exp218_config_path = resolve_existing_path(
        [str(value) for value in historical["config_patterns"]],
        search_roots,
        label="exp218 config",
    )
    verify_file_sha(
        exp218_source, str(historical["source_sha256"]), "exp218 source"
    )
    verify_file_sha(
        exp218_config_path,
        str(historical["source_config_sha256"]),
        "exp218 config",
    )
    exp145_config = dict(nested(config, "data.exp145_learned_likelihood"))
    exp145_files = [str(value) for value in exp145_config["required_files"]]
    exp145_root = resolve_artifact_root(
        [str(value) for value in exp145_config["artifact_root_patterns"]],
        search_roots,
        required_files=exp145_files,
        label="exp145 learned-likelihood train artifacts",
    )
    exp145_evidence = {
        "root": str(exp145_root),
        "kernel_source": str(exp145_config["kernel_source"]),
        "files": {
            name: {
                "sha256": sha256_file(exp145_root / name),
                "bytes": int((exp145_root / name).stat().st_size),
            }
            for name in exp145_files
        },
    }
    allowlist_cfg = dict(nested(config, "data.availability_audit"))
    clean_allowlist = resolve_existing_path(
        [str(value) for value in allowlist_cfg["clean_allowlist_patterns"]],
        search_roots,
        label="clean-273 allowlist",
    )
    hidden_cfg = dict(nested(config, "data.hidden_like_assignment"))
    hidden_assignment = resolve_existing_path(
        [str(value) for value in hidden_cfg["patterns"]],
        search_roots,
        label="hidden-like assignment",
    )
    hidden_sha = verify_file_sha(
        hidden_assignment,
        str(hidden_cfg["sha256"]),
        "hidden-like assignment",
    )
    raw_train_dir = find_competition_input_root() / "train"
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=clean_allowlist,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    base_frame = select_unique_columns(
        base_frame,
        [
            "id",
            "well",
            "target",
            "last_known_tvt",
            "md_since",
            *base_features,
        ],
        context="exp402 Stage 1 clean base",
    )
    groups = dict(parent_manifest["feature_groups"])
    expected_base = [str(value) for value in groups["clean_base"]]
    compact_features = [str(value) for value in groups["nested_compact"]]
    formation_features = [str(value) for value in groups["fold_safe_formation"]]
    if base_features != expected_base:
        raise ValueError("exp402 clean-273 schema differs from saved exp287")
    if list(stage_c_evidence["compact_features"]) != compact_features:
        raise ValueError("exp402 compact-74 schema differs from saved exp287")
    parent_features = [*base_features, *compact_features, *formation_features]
    final_features = [*parent_features, *GRWR5_FEATURES]
    if len(parent_features) != 421 or len(final_features) != 426:
        raise ValueError("exp402 final feature count must be 421+5=426")
    if len(set(final_features)) != 426:
        raise ValueError("exp402 final feature schema contains duplicates")
    if sha256_json(parent_features) != str(
        nested(config, "data.parent_exp287.feature_schema_sha256")
    ):
        raise ValueError("exp402 parent schema differs from exp287")

    parent_oof, parent_oof_evidence = align_oof_to_base(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(nested(config, "data.parent_exp287.oof_sha256")),
        columns=[
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            "fold_safe_formation_74_addonly__lgb_mean__pred_tvt",
        ],
        base_frame=base_frame,
        label="saved exp287 OOF",
    )
    parent_rmse = rmse(
        parent_oof["actual_tvt"],
        parent_oof["fold_safe_formation_74_addonly__lgb_mean__pred_tvt"],
    )
    if abs(parent_rmse - float(nested(config, "validation.primary_control.rmse"))) > 1.0e-9:
        raise ValueError("saved exp287 RMSE differs from fixed control")
    exp264_cfg = dict(nested(config, "data.clean_control_exp264"))
    exp264_path = resolve_existing_path(
        [str(value) for value in exp264_cfg["oof_patterns"]],
        search_roots,
        label="corrected exp264 OOF",
    )
    exp264_oof, exp264_evidence = align_oof_to_base(
        exp264_path,
        expected_sha256=str(exp264_cfg["oof_sha256"]),
        columns=[
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            "selector_compact_addonly__lgb_mean__pred_tvt",
        ],
        base_frame=base_frame,
        label="corrected exp264 OOF",
    )
    if not parent_oof["outer_fold"].equals(exp264_oof["outer_fold"]):
        raise ValueError("exp287 and exp264 outer folds differ")

    training = dict(nested(config, "model.training"))
    mode_name = str(training["mode"])
    mode_config = dict(exp218_config["model"]["training"]["modes"][mode_name])
    params_all = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False),
        mode_config,
    )
    config_indices = [int(value) for value in training["lightgbm_config_indices"]]
    params_family = [params_all[index] for index in config_indices]
    parent_params = {
        int(item["config_index"]): item["params"]
        for item in parent_manifest["models"]
        if int(item["outer_fold"]) == 0
    }
    for config_index, params in zip(config_indices, params_family, strict=True):
        if to_jsonable(params) != to_jsonable(parent_params[config_index]):
            raise ValueError(
                f"exp402 LightGBM config {config_index} differs from exp287"
            )

    checks = {
        "stage_0_aggregate_18_of_18_passed": stage0_evidence["checks_passed"] == 18,
        "stage_0_grwr_partitions_10": len(grwr_partitions) == 10,
        "exp287_models_15_saved_not_retrained": parent_evidence["model_count"] == 15,
        "exp287_formation_partitions_10": len(formation_partitions) == 10,
        "stage_c_compact_partitions_25": stage_c_evidence["partition_count"] == 25,
        "exp145_required_files_3": len(exp145_evidence["files"]) == 3,
        "kernel_inputs_11": len(
            nested(config, "runtime.kaggle.train_kernel_sources")
        )
        == 11,
        "clean_features_273": len(base_features) == 273,
        "nested_compact_features_74": len(compact_features) == 74,
        "fold_safe_formation_features_74": len(formation_features) == 74,
        "added_grwr5_features_5": len(GRWR5_FEATURES) == 5,
        "final_features_426_unique": len(set(final_features)) == 426,
        "gpu_boosters_15": cost["planned_gpu_boosters"] == 15,
        "control_retraining_zero": cost["control_retraining_boosters"] == 0,
        "inference_submission_disabled": not cost["inference"]
        and not cost["submission"],
        "parent_oof_rmse_matches": abs(
            parent_rmse - float(nested(config, "validation.primary_control.rmse"))
        )
        <= 1.0e-9,
        "historical_grwr5_absent_from_final": not set(
            GRWR5_FEATURES
        ).intersection(parent_features),
    }
    preflight = {
        "schema_version": "1.0.0",
        "status": (
            "stage_1_preflight_passed"
            if all(checks.values())
            else "stage_1_preflight_failed"
        ),
        "experiment": EXPERIMENT_NAME,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "cost_contract": cost,
        "feature_counts": {
            "clean_base": len(base_features),
            "nested_compact": len(compact_features),
            "fold_safe_formation": len(formation_features),
            "grwr5": len(GRWR5_FEATURES),
            "parent": len(parent_features),
            "final": len(final_features),
        },
        "feature_schema_sha256": {
            "parent_421": sha256_json(parent_features),
            "grwr5": sha256_json(GRWR5_FEATURES),
            "final_426": sha256_json(final_features),
        },
        "input": {
            "stage_0": stage0_evidence,
            "exp287": parent_evidence,
            "exp145": exp145_evidence,
            "stage_c": {
                "root": stage_c_evidence["root"],
                "sha256": stage_c_evidence["sha256"],
                "partition_count": stage_c_evidence["partition_count"],
                "compact_meta_schema_sha256": stage_c_evidence[
                    "compact_meta_schema_sha256"
                ],
            },
            "clean_base": base_evidence,
            "parent_oof": {**parent_oof_evidence, "rmse": parent_rmse},
            "exp264_oof": exp264_evidence,
            "hidden_assignment": {
                "path": str(hidden_assignment),
                "sha256": hidden_sha,
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "boosters_trained": 0,
        "control_retraining_boosters": 0,
        "prediction_generated": False,
        "submission_generated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / f"{OUTPUT_PREFIX}_stage_1_preflight_manifest.json"
    write_json(preflight_path, preflight)
    if not preflight["passed"]:
        raise RuntimeError("exp402 Stage 1 preflight failed")
    return {
        "preflight": preflight,
        "preflight_path": preflight_path,
        "parent_root": parent_root,
        "parent_manifest": parent_manifest,
        "parent_oof": parent_oof,
        "exp264_oof": exp264_oof,
        "formation_partitions": formation_partitions,
        "grwr_partitions": grwr_partitions,
        "grwr_fold_roots": fold_roots,
        "stage_c_root": stage_c_root,
        "stage_c_evidence": stage_c_evidence,
        "base_frame": base_frame,
        "base_features": base_features,
        "compact_features": compact_features,
        "formation_features": formation_features,
        "parent_features": parent_features,
        "final_features": final_features,
        "params_family": params_family,
        "config_indices": config_indices,
        "hidden_assignment": hidden_assignment,
    }


# %% [markdown]
# ## 6. Promotion metrics and well-tail guards

# %%
def rmse(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def evaluate_promotion(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    parent_oof: pd.DataFrame,
    exp264_oof: pd.DataFrame,
    new_prediction: np.ndarray,
    hidden_assignment_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = parent_oof["actual_tvt"].to_numpy(np.float32)
    parent = parent_oof[
        "fold_safe_formation_74_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    clean_tail = exp264_oof[
        "selector_compact_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    folds = parent_oof["outer_fold"].to_numpy(np.int8)
    new = np.asarray(new_prediction, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        mask = folds == outer_fold
        parent_value = rmse(truth[mask], parent[mask])
        new_value = rmse(truth[mask], new[mask])
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "rows": int(mask.sum()),
                "exp287_rmse": parent_value,
                "exp402_rmse": new_value,
                "delta_rmse_exp402_minus_exp287": new_value - parent_value,
                "nonworse_vs_exp287": new_value <= parent_value,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base_frame["md_since"].to_numpy(np.float32)
    scope_masks: dict[str, np.ndarray] = {
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    assignment = pd.read_csv(
        hidden_assignment_path, dtype={"well_id": str}
    ).set_index("well_id")
    wells = base_frame["well"].astype(str)
    scope_masks["hidden_like_spatial"] = wells.map(
        assignment["verification_like_spatial_role"]
    ).eq("valid").to_numpy()
    scope_masks["hidden_like_typewell_purged"] = wells.map(
        assignment["verification_like_typewell_purged_role"]
    ).eq("valid").to_numpy()
    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        parent_value = rmse(truth[mask], parent[mask])
        new_value = rmse(truth[mask], new[mask])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(wells[mask].nunique()),
                "exp287_rmse": parent_value,
                "exp402_rmse": new_value,
                "delta_rmse_exp402_minus_exp287": new_value - parent_value,
            }
        )
    scope_metrics = pd.DataFrame(scope_rows)
    well_frame = pd.DataFrame(
        {
            "well": wells,
            "actual_tvt": truth,
            "exp264_pred_tvt": clean_tail,
            "exp287_pred_tvt": parent,
            "exp402_pred_tvt": new,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in well_frame.groupby("well", sort=True):
        exp264_value = rmse(group["actual_tvt"], group["exp264_pred_tvt"])
        exp287_value = rmse(group["actual_tvt"], group["exp287_pred_tvt"])
        exp402_value = rmse(group["actual_tvt"], group["exp402_pred_tvt"])
        well_rows.append(
            {
                "well": well,
                "rows": len(group),
                "exp264_rmse": exp264_value,
                "exp287_rmse": exp287_value,
                "exp402_rmse": exp402_value,
                "exp287_minus_exp264_delta": exp287_value - exp264_value,
                "exp402_minus_exp287_delta": exp402_value - exp287_value,
                "exp402_minus_exp264_delta": exp402_value - exp264_value,
            }
        )
    by_well = pd.DataFrame(well_rows)
    guard_config = dict(nested(config, "guards.promotion"))
    pooled_exp264 = rmse(truth, clean_tail)
    pooled_parent = rmse(truth, parent)
    pooled_new = rmse(truth, new)
    pooled_delta = pooled_new - pooled_parent
    nonworse_folds = int(fold_metrics["nonworse_vs_exp287"].sum())
    scope_max_delta = float(
        scope_metrics["delta_rmse_exp402_minus_exp287"].max()
    )
    by_well_p95 = float(
        np.quantile(by_well["exp402_minus_exp287_delta"], 0.95)
    )
    worst_vs_exp264 = float(by_well["exp402_minus_exp264_delta"].max())
    threshold_limits = dict(
        guard_config["maximum_worsened_well_counts_vs_exp264"]
    )
    worsened_counts: dict[str, dict[str, Any]] = {}
    worsened_checks: list[bool] = []
    for threshold in [1.0, 3.0, 5.0]:
        key = f"plus_{int(threshold)}ft"
        count = int(
            (by_well["exp402_minus_exp264_delta"] > threshold).sum()
        )
        maximum = int(threshold_limits[key])
        passed = count <= maximum
        worsened_counts[key] = {
            "threshold_ft": threshold,
            "exp402_count": count,
            "maximum": maximum,
            "passed": passed,
        }
        worsened_checks.append(passed)
    checks = {
        "pooled_delta_rmse_vs_exp287": pooled_delta
        <= float(guard_config["maximum_pooled_delta_rmse_vs_exp287"]),
        "minimum_nonworse_folds_vs_exp287": nonworse_folds
        >= int(guard_config["minimum_nonworse_folds_vs_exp287"]),
        "all_scope_non_regression_vs_exp287": scope_max_delta
        <= float(guard_config["maximum_scope_delta_rmse_vs_exp287"]),
        "by_well_delta_p95_vs_exp287": by_well_p95
        <= float(guard_config["maximum_by_well_p95_delta_rmse_vs_exp287"]),
        "worst_well_delta_rmse_vs_exp264": worst_vs_exp264
        <= float(guard_config["maximum_worst_well_delta_rmse_vs_exp264"]),
        "worsened_well_counts_vs_exp264": all(worsened_checks),
    }
    guard = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "exp264_rmse": pooled_exp264,
        "exp287_rmse": pooled_parent,
        "exp402_rmse": pooled_new,
        "delta_rmse_exp402_minus_exp287": pooled_delta,
        "nonworse_folds_vs_exp287": nonworse_folds,
        "maximum_scope_delta_rmse_vs_exp287": scope_max_delta,
        "by_well_delta_p95_vs_exp287": by_well_p95,
        "worst_well_delta_rmse_vs_exp264": worst_vs_exp264,
        "worsened_well_counts_vs_exp264": worsened_counts,
    }
    return guard, fold_metrics, scope_metrics, by_well


# %% [markdown]
# ## 7. Fifteen-booster GPU training

# %%
def run_gpu_train(
    *,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    started = time.perf_counter()
    base_frame = inputs["base_frame"]
    base_features = list(inputs["base_features"])
    compact_features = list(inputs["compact_features"])
    formation_features = list(inputs["formation_features"])
    final_features = list(inputs["final_features"])
    parent_oof = inputs["parent_oof"]
    exp264_oof = inputs["exp264_oof"]
    params_family = list(inputs["params_family"])
    config_indices = list(inputs["config_indices"])
    training = dict(nested(config, "model.training"))
    n_rows = len(base_frame)
    base_index = pd.Index(base_frame["id"].astype(str))
    if not base_index.is_unique:
        raise ValueError("exp402 clean base IDs are duplicated")
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    oof_by_config = [
        np.full(n_rows, np.nan, dtype=np.float32) for _ in params_family
    ]
    model_dir = output_dir / "stage_1_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    chunk_columns = int(training["matrix_copy_chunk_columns"])

    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=inputs["stage_c_root"],
            stage_c_evidence=inputs["stage_c_evidence"],
            downstream_outer_fold=outer_fold,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("exp402 compact IDs are absent from clean base")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("exp402 outer train/valid rows overlap")
        if not np.all(
            parent_oof.iloc[valid_indices]["outer_fold"].to_numpy(np.int8)
            == outer_fold
        ):
            raise ValueError("exp402 valid role differs from exp287 OOF fold")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("exp402 OOF valid rows were assigned twice")
        oof_fold[valid_indices] = np.int8(outer_fold)

        formation_train, formation_train_evidence = load_saved_formation_role(
            parent_root=inputs["parent_root"],
            partition_manifest=inputs["formation_partitions"],
            outer_fold=outer_fold,
            role="train",
            compact=compact_train,
            feature_names=formation_features,
        )
        formation_valid, formation_valid_evidence = load_saved_formation_role(
            parent_root=inputs["parent_root"],
            partition_manifest=inputs["formation_partitions"],
            outer_fold=outer_fold,
            role="valid",
            compact=compact_valid,
            feature_names=formation_features,
        )
        grwr_train, grwr_train_evidence = load_saved_grwr5_role(
            fold_root=inputs["grwr_fold_roots"][outer_fold],
            partition_manifest=inputs["grwr_partitions"],
            outer_fold=outer_fold,
            role="train",
            compact=compact_train,
        )
        grwr_valid, grwr_valid_evidence = load_saved_grwr5_role(
            fold_root=inputs["grwr_fold_roots"][outer_fold],
            partition_manifest=inputs["grwr_partitions"],
            outer_fold=outer_fold,
            role="valid",
            compact=compact_valid,
        )
        component_rows.extend(
            [
                formation_train_evidence,
                formation_valid_evidence,
                grwr_train_evidence,
                grwr_valid_evidence,
            ]
        )

        x_train_values = np.empty(
            (len(train_indices), len(final_features)), dtype=np.float32
        )
        x_valid_values = np.empty(
            (len(valid_indices), len(final_features)), dtype=np.float32
        )
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            source = base_frame[columns]
            x_train_values[:, start:stop] = source.iloc[
                train_indices
            ].to_numpy(np.float32, copy=True)
            x_valid_values[:, start:stop] = source.iloc[
                valid_indices
            ].to_numpy(np.float32, copy=True)
        compact_start = len(base_features)
        formation_start = compact_start + len(compact_features)
        grwr_start = formation_start + len(formation_features)
        x_train_values[:, compact_start:formation_start] = compact_train[
            compact_features
        ].to_numpy(np.float32, copy=False)
        x_valid_values[:, compact_start:formation_start] = compact_valid[
            compact_features
        ].to_numpy(np.float32, copy=False)
        x_train_values[:, formation_start:grwr_start] = formation_train[
            formation_features
        ].to_numpy(np.float32, copy=False)
        x_valid_values[:, formation_start:grwr_start] = formation_valid[
            formation_features
        ].to_numpy(np.float32, copy=False)
        x_train_values[:, grwr_start:] = grwr_train[GRWR5_FEATURES].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, grwr_start:] = grwr_valid[GRWR5_FEATURES].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(x_train_values).all() or not np.isfinite(
            x_valid_values
        ).all():
            raise ValueError("exp402 426-feature matrix contains non-finite values")
        x_train = pd.DataFrame(
            x_train_values, columns=final_features, copy=False
        )
        x_valid = pd.DataFrame(
            x_valid_values, columns=final_features, copy=False
        )
        fold_predictions: list[np.ndarray] = []
        for family_position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                target[train_indices],
                eval_set=[(x_valid, target[valid_indices])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(
                        int(training["early_stopping_rounds"]), verbose=False
                    ),
                    log_evaluation(int(training["log_evaluation_period"])),
                ],
            )
            best_iteration = int(
                model.best_iteration_ or params["n_estimators"]
            )
            prediction = model.predict(
                x_valid, num_iteration=best_iteration
            ).astype(np.float32)
            oof_by_config[family_position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(
                str(model_path), num_iteration=best_iteration
            )
            rmse_value = rmse(
                truth[valid_indices], anchor[valid_indices] + prediction
            )
            model_row = {
                "variant": "fold_safe_grwr_5_addonly",
                "model": f"lgb{config_index}",
                "config_index": config_index,
                "outer_fold": outer_fold,
                "feature_count": len(final_features),
                "feature_schema_sha256": sha256_json(final_features),
                "best_iteration": best_iteration,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_file(model_path),
                "params": to_jsonable(params),
            }
            model_rows.append(model_row)
            fold_model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_indices),
                    "rmse_tvt": rmse_value,
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ["gain", "split"]:
                importance = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                for feature, value in zip(
                    final_features, importance, strict=True
                ):
                    importance_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "model": f"lgb{config_index}",
                            "importance_type": importance_type,
                            "feature": feature,
                            "feature_group": (
                                "fold_safe_grwr5"
                                if feature in GRWR5_FEATURES
                                else "fold_safe_formation"
                                if feature in formation_features
                                else "nested_compact"
                                if feature in compact_features
                                else "clean_base"
                            ),
                            "importance": float(value),
                        }
                    )
            progress = {
                "schema_version": "1.0.0",
                "status": "stage_1_training_in_progress",
                "completed_gpu_boosters": len(model_rows),
                "planned_gpu_boosters": 15,
                "last_model": model_row,
                "peak_rss_gb": current_peak_rss_gb(),
            }
            write_json(
                output_dir / f"{OUTPUT_PREFIX}_stage_1_progress.json",
                progress,
            )
            print(
                json.dumps(
                    {
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": rmse_value,
                        "completed_gpu_boosters": len(model_rows),
                        "planned_gpu_boosters": 15,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, prediction
            gc.collect()
        fold_mean_residual = np.mean(
            np.vstack(fold_predictions), axis=0
        ).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_indices),
                "rmse_tvt": rmse(
                    truth[valid_indices],
                    anchor[valid_indices] + fold_mean_residual,
                ),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            formation_train,
            formation_valid,
            grwr_train,
            grwr_valid,
            x_train,
            x_valid,
            x_train_values,
            x_valid_values,
            fold_predictions,
            fold_mean_residual,
        )
        gc.collect()

    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("exp402 15-model OOF contract is incomplete")
    if not np.array_equal(
        oof_fold, parent_oof["outer_fold"].to_numpy(np.int8)
    ):
        raise AssertionError("exp402 OOF fold assignment differs from exp287")
    if any(not np.isfinite(prediction).all() for prediction in oof_by_config):
        raise AssertionError("exp402 OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    guard, fold_metrics, scope_metrics, by_well = evaluate_promotion(
        config=config,
        base_frame=base_frame,
        parent_oof=parent_oof,
        exp264_oof=exp264_oof,
        new_prediction=mean_prediction,
        hidden_assignment_path=inputs["hidden_assignment"],
    )

    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    prediction_frame["exp287_pred_tvt"] = parent_oof[
        "fold_safe_formation_74_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    prediction_frame["exp264_pred_tvt"] = exp264_oof[
        "selector_compact_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    for config_index, residual in zip(
        config_indices, oof_by_config, strict=True
    ):
        prediction_frame[
            f"fold_safe_grwr_5_addonly__lgb{config_index}__pred_tvt"
        ] = (anchor + residual).astype(np.float32)
    prediction_frame[
        "fold_safe_grwr_5_addonly__lgb_mean__pred_tvt"
    ] = mean_prediction

    paths = {
        "oof": output_dir / f"{OUTPUT_PREFIX}_stage_1_oof_predictions.parquet",
        "fold_metrics": output_dir
        / f"{OUTPUT_PREFIX}_stage_1_fold_metrics.csv",
        "scope_metrics": output_dir
        / f"{OUTPUT_PREFIX}_stage_1_scope_metrics.csv",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_stage_1_by_well_metrics.csv",
        "importance": output_dir
        / f"{OUTPUT_PREFIX}_stage_1_feature_importance.csv",
        "component_manifest": output_dir
        / f"{OUTPUT_PREFIX}_stage_1_component_manifest.csv",
        "model_manifest": output_dir
        / f"{OUTPUT_PREFIX}_stage_1_model_manifest.json",
        "metrics": output_dir / f"{OUTPUT_PREFIX}_stage_1_metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(
        fold_metrics, on="outer_fold", how="left"
    ).to_csv(paths["fold_metrics"], index=False)
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    pd.DataFrame(importance_rows).to_csv(paths["importance"], index=False)
    pd.DataFrame(component_rows).to_csv(
        paths["component_manifest"], index=False
    )
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_1_15_gpu_boosters_completed",
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "feature_groups": {
            "clean_base": base_features,
            "nested_compact": compact_features,
            "fold_safe_formation": formation_features,
            "fold_safe_grwr5": GRWR5_FEATURES,
        },
        "control_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": (
            "stage_1_complete_all_promotion_gates_passed_inference_approval_pending"
            if guard["passed"]
            else "stage_1_complete_promotion_gate_failed_closed"
        ),
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": {
            "parent": 421,
            "added_grwr5": 5,
            "final": 426,
        },
        "completed_gpu_boosters": len(model_rows),
        "control_retraining_boosters": 0,
        "gpu_runtime": inputs["gpu_runtime"],
        "guard": guard,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "prediction_generated": True,
        "inference_generated": False,
        "submission_generated": False,
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {
        name: sha256_file(path) for name, path in paths.items()
    }
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "deterministic_anchor": False,
        "rerun_parity_checked": False,
        "gpu_policy": dict(nested(config, "runtime.stage_1")),
        "gpu_runtime": inputs["gpu_runtime"],
        "stage_1_preflight_manifest_sha256": sha256_file(
            inputs["preflight_path"]
        ),
        "feature_schema_sha256": sha256_json(final_features),
        "component_manifest_sha256": artifact_sha["component_manifest"],
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "oof_prediction_sha256": artifact_sha["oof"],
        "artifact_sha256": artifact_sha,
        "guard": guard,
        "model_count": len(model_rows),
        "control_retraining_boosters": 0,
        "inference_generated": False,
        "submission_sha256": None,
    }
    reproducibility_path = (
        output_dir / f"{OUTPUT_PREFIX}_stage_1_reproducibility_manifest.json"
    )
    write_json(reproducibility_path, reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        reproducibility_path
    )
    print(
        "Stage 1 GPU train finished",
        json.dumps(
            {
                "status": metrics["status"],
                "passed": guard["passed"],
                "completed_gpu_boosters": len(model_rows),
                "control_retraining_boosters": 0,
                "exp264_rmse": guard["exp264_rmse"],
                "exp287_rmse": guard["exp287_rmse"],
                "exp402_rmse": guard["exp402_rmse"],
                "delta_rmse_exp402_minus_exp287": guard[
                    "delta_rmse_exp402_minus_exp287"
                ],
                "runtime_seconds": metrics["runtime_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return metrics


# %% [markdown]
# ## 8. Metrics, model manifests, and reproducibility evidence
#
# OOF / fold / scope / by-well / feature importance / component / model manifestを保存し、
# GRWR-5 gain importanceを表とplotで表示する。GPU bitwise determinismは主張せず、
# input / feature / model / OOF SHAを記録する。

# %%
def display_stage1_diagnostics(
    result: Mapping[str, Any],
    *,
    output_dir: Path,
) -> None:
    display(dict(result["guard"]))
    fold_metrics = pd.read_csv(
        output_dir / f"{OUTPUT_PREFIX}_stage_1_fold_metrics.csv"
    )
    display(fold_metrics[fold_metrics["model"].eq("lgb_mean")])
    scope_metrics = pd.read_csv(
        output_dir / f"{OUTPUT_PREFIX}_stage_1_scope_metrics.csv"
    )
    display(scope_metrics)
    by_well = pd.read_csv(
        output_dir / f"{OUTPUT_PREFIX}_stage_1_by_well_metrics.csv"
    )
    display(
        by_well.sort_values(
            "exp402_minus_exp287_delta", ascending=False
        ).head(80)
    )
    importance = pd.read_csv(
        output_dir / f"{OUTPUT_PREFIX}_stage_1_feature_importance.csv"
    )
    mean_gain = (
        importance[importance["importance_type"].eq("gain")]
        .groupby(["feature_group", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["feature_group", "importance"], ascending=[True, False])
    )
    display(mean_gain.groupby("feature_group", group_keys=False).head(30))
    grwr_gain = mean_gain[mean_gain["feature_group"].eq("fold_safe_grwr5")]
    display(grwr_gain)
    try:
        import matplotlib.pyplot as plt

        plot_frame = grwr_gain.sort_values("importance")
        axis = plot_frame.plot.barh(
            x="feature",
            y="importance",
            figsize=(10, 4),
            legend=False,
            title="exp402 GRWR-5 mean gain importance across 15 models",
        )
        axis.set_xlabel("mean gain importance")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{OUTPUT_PREFIX}_stage_1_grwr5_importance.png",
            dpi=140,
        )
        plt.show()
    except ImportError:
        print("matplotlib is unavailable; importance CSV was still saved.")


# %% [markdown]
# ## 9. Execution orchestration and generated files

# %%
def run_experiment() -> dict[str, Any]:
    config = read_yaml(find_config_path())
    validate_stage1_contract(config)
    if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("Kaggle Notebook execution is authoritative for exp402")
    gpu_runtime = verify_t4_runtime()
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = prepare_stage1_inputs(config=config, output_dir=output_dir)
    inputs["gpu_runtime"] = gpu_runtime
    return run_gpu_train(config=config, inputs=inputs, output_dir=output_dir)


if not IMPORT_ONLY:
    CONFIG = read_yaml(find_config_path())
    COST_CONTRACT = validate_stage1_contract(CONFIG)
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "stage": CONFIG["execution"]["current_stage"],
            **COST_CONTRACT,
        }
    )
    RUN_RESULT = run_experiment()
    display(RUN_RESULT)
    OUTPUT_DIR = KAGGLE_WORKING_ROOT / "artifacts"
    display_stage1_diagnostics(RUN_RESULT, output_dir=OUTPUT_DIR)
    print("Generated files")
    for generated in sorted(OUTPUT_DIR.rglob("*")):
        if generated.is_file():
            print(generated.relative_to(OUTPUT_DIR), generated.stat().st_size)
