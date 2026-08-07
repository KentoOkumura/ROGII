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
# # exp502 exp501 fixed13 selector replacement on exp413 — train
#
# exp413 Stage Dの`clean273 + nested74 + signed23 = final370`から、Stage C
# nested selector blockだけを除外し、保存済みexp501 fixed13 compact77へ置換する。
# treatmentは`clean273 + exp501 compact77 + signed23 = final373`の1 variantだけ。
# selector、signed selector、exp413 control、HMM、PF、Beamは再学習・再生成しない。
#
# このsourceは実装候補であり、configのpackage/train承認とrun flagがすべてtrueに
# ならない限り学習しない。current-test inferenceとsubmissionは実装しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and notebook-safe runtime helpers
# 2. Frozen replacement and GPU cost contract
# 3. Saved artifact resolvers and SHA verification
# 4. Replacement-only feature-surface helpers
# 5. Saved exp413 control and evaluation helpers
# 6. Setup, configuration, and input checks
# 7. Final373 fold assembly and 15-booster training
# 8. Metrics, feature importance, and generated artifacts
# 9. Reproducibility evidence and fixed stop

# %% [markdown]
# ## 1. Imports and notebook-safe runtime helpers

# %%
from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    compact_feature_names,
    load_stage_d_compact_fold,
    logical_frame_sha256,
    sha256_file,
    sha256_json,
    verify_stage_c_artifact_root,
    write_json,
)
from src.likpf_full_replacement import (
    build_replacement_clean273_surface,
    resolve_by_patterns,
)
from src.signed_residual_meta import (
    load_signed_compact_fold,
    verify_signed_stage_s_root,
)

EXPERIMENT_NAME = "exp502_exp501_fixed13_selector_replacement_on_exp413"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_project_root(start: Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start)
    for candidate in (current, *current.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return current


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_notebook_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp502 is Kaggle-first. Local execution requires an explicitly approved "
        "EXPERIMENT_ALLOW_LOCAL=1 smoke run."
    )


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def resolve_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp502 config.yaml")


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def resolve_file(spec: Mapping[str, Any], *, sha_key: str = "sha256") -> Path:
    return resolve_by_patterns(
        [str(item) for item in spec["patterns"]],
        search_roots(),
        marker_sha256=str(spec.get(sha_key) or ""),
    )


def resolve_artifact_root(
    patterns: Sequence[str],
    *,
    marker: str,
    expected_marker_sha256: str,
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.is_dir():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots():
            if root.exists():
                candidates.extend(path for path in root.glob(raw) if path.is_dir())
    for root in search_roots():
        if root.exists():
            candidates.extend(path.parent for path in root.rglob(marker))
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        marker_path = candidate / marker
        checked.append(str(candidate))
        if not marker_path.is_file():
            continue
        if sha256_file(marker_path) == str(expected_marker_sha256):
            return candidate
    raise FileNotFoundError(
        f"artifact root with frozen {marker} was not found; checked={checked[:80]}"
    )


def competition_data_root() -> Path:
    local = ROOT / "data" / "raw"
    if not is_kaggle_runtime():
        return local
    project_path = ROOT / "project.yml"
    project = load_yaml(project_path) if project_path.exists() else {}
    slug = str(project.get("competition", {}).get("slug", ""))
    candidates = [KAGGLE_INPUT_ROOT / slug]
    if slug:
        candidates.append(KAGGLE_INPUT_ROOT / "competitions" / slug)
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    for candidate in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError("competition train/test root was not found")


# %% [markdown]
# ## 2. Frozen replacement and GPU cost contract
#
# 変更可能な面はselector block 1個だけ。実行量はtreatment 1 × config 3 ×
# outer fold 5 = 15 GPU boostersに固定し、保存controlやselectorを再学習しない。

# %%
def validate_static_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = dict(config["experiment"])
    if experiment["route"] != "ml_model":
        raise ValueError("exp502 route must remain ml_model")
    if not bool(config["authorization"]["implementation_approved"]):
        raise ValueError("exp502 implementation approval is missing")

    surface = dict(config["feature_surface"])
    ordered = [dict(item) for item in surface["ordered_blocks"]]
    observed = [
        (item["name"], item["source"], int(item["feature_count"]), item["action"])
        for item in ordered
    ]
    expected = [
        ("exp413_clean_base", "exp413", 273, "retain"),
        ("nested_selector_compact", "exp501", 77, "replace_exp413_nested74"),
        ("exp413_signed_selector_compact", "exp413", 23, "retain"),
    ]
    if observed != expected:
        raise ValueError(f"replacement block order changed: {observed}")
    if int(surface["expected_final_feature_count"]) != 373:
        raise ValueError("exp502 final feature count must remain 373")
    if int(surface["expected_old_selector_columns_in_final"]) != 0:
        raise ValueError("the exp413 nested74 block must not survive")
    if int(surface["expected_new_selector_columns_in_final"]) != 77:
        raise ValueError("the exp501 compact77 block must be inserted exactly once")

    stage = dict(config["model"]["downstream_tvt"])
    count = dict(config["model"]["execution_count"])
    expected_count = {
        "treatment_variants": 1,
        "lightgbm_configs": 3,
        "outer_folds": 5,
        "planned_gpu_downstream_boosters": 15,
        "planned_total_new_boosters": 15,
        "exp413_control_retraining_boosters": 0,
        "exp501_selector_retraining_boosters": 0,
        "exp413_signed_selector_retraining_boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
    }
    observed_count = {key: int(count[key]) for key in expected_count}
    if observed_count != expected_count:
        raise ValueError(f"exp502 compute contract changed: {observed_count}")
    if list(stage["lightgbm_config_indices"]) != [0, 1, 2]:
        raise ValueError("exp502 must inherit exp413 LightGBM configs 0/1/2")
    if int(stage["folds"]) != 5 or int(stage["planned_gpu_boosters"]) != 15:
        raise ValueError("exp502 fold/booster count changed")
    if list(config["model"]["active_variants"]) != [
        "exp501_fixed13_selector_replacement"
    ]:
        raise ValueError("exp502 must train exactly one treatment variant")
    return {"ordered_blocks": observed, "cost": observed_count}


def require_train_authorization(config: Mapping[str, Any]) -> None:
    authorization = dict(config["authorization"])
    execution = dict(config["execution"])
    checks = {
        "implementation_approved": bool(authorization["implementation_approved"]),
        "kaggle_package_approved": bool(authorization["kaggle_package_approved"]),
        "kaggle_train_approved": bool(authorization["kaggle_train_approved"]),
        "execution_run_approved": bool(execution["run_approved"]),
        "train_run_flag": bool(execution["run_flags"]["train"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"exp502 train remains disabled: {checks}")


# %% [markdown]
# ## 3. Saved artifact resolvers and SHA verification
#
# exp501 compact、exp413 removed compact、exp413 signed compact、exp413 saved
# controlを別々のrootとして解決する。同名markerの誤選択を防ぐため、root解決時点で
# marker SHAを一致させる。fold manifestはexp413 C/Sとexp501でbyte SHA一致を要求する。

# %%
def _stage_c_verify_config(
    *,
    metrics_sha256: str,
    model_manifest_sha256: str,
    compact_manifest_sha256: str,
    compact_schema_file_sha256: str = "",
    compact_schema_logical_sha256: str = "",
) -> dict[str, Any]:
    return {
        "data": {
            "stage_c_expected_nested_selector_metrics_sha256": metrics_sha256,
            "stage_c_expected_nested_selector_model_manifest_sha256": (
                model_manifest_sha256
            ),
            "stage_c_expected_nested_compact_manifest_sha256": (
                compact_manifest_sha256
            ),
            "stage_c_expected_compact_meta_schema_file_sha256": (
                compact_schema_file_sha256
            ),
            "stage_c_expected_compact_meta_schema_logical_sha256": (
                compact_schema_logical_sha256
            ),
        }
    }


def verify_saved_feature_sources(
    *,
    config: Mapping[str, Any],
    exp501_contract: Mapping[str, Any],
    exp501_root: Path,
    exp413_stage_c_root: Path,
    exp413_stage_s_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    exp501_spec = dict(config["data"]["exp501_selector_source"])
    exp413_spec = dict(config["data"]["exp413_source"])
    exp501_evidence = verify_stage_c_artifact_root(
        exp501_root,
        _stage_c_verify_config(
            metrics_sha256=str(exp501_spec["nested_selector_metrics_sha256"]),
            model_manifest_sha256=str(exp501_spec["selector_model_manifest_sha256"]),
            compact_manifest_sha256=str(exp501_spec["compact_manifest_sha256"]),
        ),
        verify_partition_sha256=True,
        expected_compact_feature_count=77,
        require_score_guard=True,
    )
    expected_exp501_features = compact_feature_names(exp501_contract)
    if exp501_evidence["compact_features"] != expected_exp501_features:
        raise ValueError("exp501 compact77 schema/order differs from its frozen contract")

    removed_evidence = verify_stage_c_artifact_root(
        exp413_stage_c_root,
        _stage_c_verify_config(
            metrics_sha256=str(exp413_spec["stage_c_metrics_sha256"]),
            model_manifest_sha256=str(exp413_spec["stage_c_model_manifest_sha256"]),
            compact_manifest_sha256=str(
                exp413_spec["removed_nested_compact_manifest_sha256"]
            ),
            compact_schema_file_sha256=str(
                exp413_spec["stage_c_compact_schema_file_sha256"]
            ),
            compact_schema_logical_sha256=str(
                exp413_spec["stage_c_compact_schema_logical_sha256"]
            ),
        ),
        verify_partition_sha256=True,
        expected_compact_feature_count=74,
        require_score_guard=False,
    )
    stage_c_lineage = exp413_stage_c_root / "replacement_stage_c_lineage.json"
    if sha256_file(stage_c_lineage) != str(exp413_spec["stage_c_lineage_sha256"]):
        raise ValueError("exp413 Stage C lineage SHA mismatch")

    signed_config = {
        "data": {
            "stage_s_signed_selector_metrics_sha256": exp413_spec[
                "stage_s_metrics_sha256"
            ],
            "stage_s_model_manifest_sha256": exp413_spec[
                "stage_s_model_manifest_sha256"
            ],
            "stage_s_compact_manifest_sha256": exp413_spec[
                "retained_signed_compact_manifest_sha256"
            ],
            "stage_s_compact_schema_file_sha256": exp413_spec[
                "stage_s_compact_schema_file_sha256"
            ],
            "stage_s_compact_schema_logical_sha256": exp413_spec[
                "stage_s_compact_schema_logical_sha256"
            ],
            "stage_s_reproducibility_manifest_sha256": exp413_spec[
                "stage_s_reproducibility_manifest_sha256"
            ],
        }
    }
    signed_evidence = verify_signed_stage_s_root(
        exp413_stage_s_root,
        signed_config,
        verify_partition_sha=True,
        verify_model_sha=True,
        require_score_gate=False,
    )
    stage_s_lineage = exp413_stage_s_root / "replacement_stage_s_lineage.json"
    if sha256_file(stage_s_lineage) != str(exp413_spec["stage_s_lineage_sha256"]):
        raise ValueError("exp413 Stage S lineage SHA mismatch")

    expected_fold_sha = str(config["data"]["fold_contract"]["expected_manifest_sha256"])
    fold_paths = {
        "exp501_nested": exp501_root / "nested_fold_manifest.csv",
        "exp413_nested_removed": exp413_stage_c_root / "nested_fold_manifest.csv",
        "exp413_signed_retained": exp413_stage_s_root
        / "signed_nested_fold_manifest.csv",
    }
    fold_sha = {name: sha256_file(path) for name, path in fold_paths.items()}
    if set(fold_sha.values()) != {expected_fold_sha}:
        raise ValueError(f"exp413/exp501 nested fold manifest mismatch: {fold_sha}")
    return exp501_evidence, removed_evidence, signed_evidence, fold_sha


# %% [markdown]
# ## 4. Replacement-only feature-surface helpers
#
# old74とnew77には同名の意味slotが多い。この実験の「old block 0」は文字列名の
# 差集合ではなく、final matrixへ値を供給するblock provenanceで判定する。
# final順は必ずexp413 clean273、exp501 compact77、exp413 signed23とする。

# %%
def build_feature_surface_contract(
    *,
    base_features: Sequence[str],
    replacement_features: Sequence[str],
    signed_features: Sequence[str],
    removed_features: Sequence[str],
) -> dict[str, Any]:
    base = [str(item) for item in base_features]
    replacement = [str(item) for item in replacement_features]
    signed = [str(item) for item in signed_features]
    removed = [str(item) for item in removed_features]
    final = [*base, *replacement, *signed]
    blocks = [
        {
            "name": "exp413_clean_base",
            "source": "exp413",
            "action": "retain",
            "feature_count": len(base),
            "features": base,
        },
        {
            "name": "nested_selector_compact",
            "source": "exp501",
            "action": "replace_exp413_nested74",
            "feature_count": len(replacement),
            "features": replacement,
        },
        {
            "name": "exp413_signed_selector_compact",
            "source": "exp413",
            "action": "retain",
            "feature_count": len(signed),
            "features": signed,
        },
    ]
    if (len(base), len(replacement), len(signed), len(removed)) != (273, 77, 23, 74):
        raise ValueError("exp502 component feature count changed")
    if len(final) != 373 or len(set(final)) != 373:
        raise ValueError("exp502 final373 schema is not exact and unique")
    old_block_instances = sum(
        int(block["source"] == "exp413" and block["name"] == "nested_selector_compact")
        for block in blocks
    )
    new_block_instances = sum(
        int(block["source"] == "exp501" and block["name"] == "nested_selector_compact")
        for block in blocks
    )
    if old_block_instances != 0 or new_block_instances != 1:
        raise ValueError("replacement-only block provenance contract failed")
    return {
        "policy": "replace_only_no_addonly",
        "blocks": blocks,
        "removed_block": {
            "name": "exp413_nested_selector_compact",
            "source": "exp413",
            "feature_count": len(removed),
            "features": removed,
        },
        "old_selector_block_instances_in_final": old_block_instances,
        "new_selector_block_instances_in_final": new_block_instances,
        "old_feature_name_overlap_with_replacement": len(set(removed) & set(replacement)),
        "feature_count": len(final),
        "feature_schema_sha256": sha256_json(final),
        "features": final,
    }


def validate_fold_alignment(
    *,
    base: pd.DataFrame,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    downstream_outer_fold: int,
    role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    compact_keys = compact[KEY_COLUMNS].reset_index(drop=True)
    signed_keys = signed[KEY_COLUMNS].reset_index(drop=True)
    if not compact_keys.equals(signed_keys):
        raise ValueError(f"exp501 compact/exp413 signed key mismatch: fold={downstream_outer_fold} role={role}")
    if compact["id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate compact ids: fold={downstream_outer_fold} role={role}")
    base_index = pd.Index(base["id"].astype(str))
    positions = base_index.get_indexer(compact["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(positions):
        raise ValueError(f"compact/base join is not one-to-one: fold={downstream_outer_fold} role={role}")
    aligned = base.iloc[positions]
    if not aligned["well"].astype(str).reset_index(drop=True).equals(
        compact["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"compact/base well mismatch: fold={downstream_outer_fold} role={role}")
    anchor_delta = np.abs(
        aligned["last_known_tvt"].to_numpy(np.float32)
        - compact["last_known_tvt"].to_numpy(np.float32)
    )
    if float(anchor_delta.max(initial=0.0)) > 1.0e-4:
        raise ValueError(f"compact/base anchor mismatch: fold={downstream_outer_fold} role={role}")
    return positions, {
        "downstream_outer_fold": int(downstream_outer_fold),
        "role": role,
        "rows": len(compact),
        "wells": int(compact["well"].nunique()),
        "key_sha256": logical_frame_sha256(compact_keys),
        "missing_base_rows": int(np.sum(positions < 0)),
        "duplicate_ids": int(compact["id"].astype(str).duplicated().sum()),
        "anchor_max_abs_error": float(anchor_delta.max(initial=0.0)),
    }


def assemble_matrix(
    *,
    base: pd.DataFrame,
    positions: np.ndarray,
    compact: pd.DataFrame,
    signed: pd.DataFrame,
    base_features: Sequence[str],
    compact_features: Sequence[str],
    signed_features: Sequence[str],
    chunk_columns: int,
) -> np.ndarray:
    width = len(base_features) + len(compact_features) + len(signed_features)
    matrix = np.empty((len(positions), width), dtype=np.float32)
    for start in range(0, len(base_features), int(chunk_columns)):
        stop = min(start + int(chunk_columns), len(base_features))
        columns = list(base_features[start:stop])
        matrix[:, start:stop] = base[columns].iloc[positions].to_numpy(
            np.float32, copy=True
        )
    compact_start = len(base_features)
    signed_start = compact_start + len(compact_features)
    matrix[:, compact_start:signed_start] = compact[list(compact_features)].to_numpy(
        np.float32, copy=False
    )
    matrix[:, signed_start:] = signed[list(signed_features)].to_numpy(
        np.float32, copy=False
    )
    if width != 373 or not np.isfinite(matrix).all():
        raise ValueError("exp502 final373 matrix width or finite contract failed")
    return matrix


def matrix_content_sha256(matrix: np.ndarray, features: Sequence[str]) -> str:
    values = np.ascontiguousarray(matrix, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(values.shape), "dtype": str(values.dtype), "features": list(features)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


# %% [markdown]
# ## 5. Saved exp413 control and evaluation helpers
#
# controlはexp413 Stage D version 2のsaved OOF / metrics / 15-model manifestを
# SHA固定で読む。再学習は0。promotion gateはexp413 late-stage ML gateと同じ
# pooled、fold、3 distance scope、2 hidden-like scopeを使い、tailはreport-only。

# %%
def _rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def load_saved_exp413_control(
    *,
    root: Path,
    base: pd.DataFrame,
    base_features: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    spec = dict(config["data"]["exp413_source"])
    paths = {
        "oof": Path(root) / "stage_d_oof_predictions.parquet",
        "metrics": Path(root) / "stage_d_metrics.json",
        "model_manifest": Path(root) / "stage_d_model_manifest.json",
    }
    expected_sha = {
        "oof": str(config["validation"]["primary_control"]["oof_sha256"]),
        "metrics": str(spec["stage_d_metrics_sha256"]),
        "model_manifest": str(spec["stage_d_model_manifest_sha256"]),
    }
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    if actual_sha != expected_sha:
        raise ValueError(f"saved exp413 control SHA mismatch: {actual_sha}")
    metrics = json.loads(paths["metrics"].read_text())
    manifest = json.loads(paths["model_manifest"].read_text())
    groups = dict(manifest["feature_groups"])
    old_features = [str(item) for item in groups["nested_compact"]]
    signed_features = [str(item) for item in groups["signed_compact"]]
    if [str(item) for item in groups["clean_base"]] != [str(item) for item in base_features]:
        raise ValueError("rebuilt exp413 clean273 schema/order differs from saved Stage D")
    if (
        int(manifest["model_count"]) != 15
        or int(manifest["feature_count"]) != 370
        or len(old_features) != 74
        or len(signed_features) != 23
        or int(metrics["model_count"]) != 15
    ):
        raise ValueError("saved exp413 Stage D model/feature contract changed")

    prediction_column = str(spec["stage_d_prediction_column"])
    columns = [
        "id",
        "well",
        "md_since",
        "last_known_tvt",
        "target",
        "outer_fold",
        "actual_tvt",
        prediction_column,
    ]
    frame = pd.read_parquet(paths["oof"], columns=columns)
    if (
        len(frame) != int(config["validation"]["expected_rows"])
        or int(frame["well"].nunique()) != int(config["validation"]["expected_wells"])
        or frame["id"].astype(str).duplicated().any()
    ):
        raise ValueError("saved exp413 OOF identity or coverage mismatch")
    index = pd.Index(frame["id"].astype(str))
    positions = index.get_indexer(base["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(base):
        raise ValueError("saved exp413 OOF does not align one-to-one with clean273")
    frame = frame.iloc[positions].reset_index(drop=True)
    if not frame["well"].astype(str).equals(base["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved exp413 OOF well alignment mismatch")
    truth = (
        base["last_known_tvt"].to_numpy(np.float32)
        + base["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if float(np.abs(frame["actual_tvt"].to_numpy(np.float32) - truth).max(initial=0.0)) > 1.0e-4:
        raise ValueError("saved exp413 OOF truth differs from clean273")
    observed_rmse = _rmse(truth, frame[prediction_column])
    expected_rmse = float(config["validation"]["primary_control"]["rmse"])
    if abs(observed_rmse - expected_rmse) > 1.0e-9:
        raise ValueError(f"saved exp413 OOF RMSE mismatch: {observed_rmse}")
    return frame, {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "feature_count": 370,
        "model_count": 15,
        "rmse": observed_rmse,
        "models_retrained": 0,
        "prediction_column": prediction_column,
    }, old_features, signed_features


def evaluate_exp502_gate(
    *,
    config: Mapping[str, Any],
    base: pd.DataFrame,
    saved_control: pd.DataFrame,
    oof_fold: np.ndarray,
    prediction: np.ndarray,
    hidden_like_assignment_path: Path,
    technical_checks: Mapping[str, bool],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = (
        base["last_known_tvt"].to_numpy(np.float32)
        + base["target"].to_numpy(np.float32)
    ).astype(np.float32)
    parent_column = str(
        config["data"]["exp413_source"]["stage_d_prediction_column"]
    )
    parent = saved_control[parent_column].to_numpy(np.float32)
    candidate = np.asarray(prediction, dtype=np.float32)
    if not np.isfinite(candidate).all():
        raise ValueError("exp502 OOF prediction contains non-finite values")

    fold_rows: list[dict[str, Any]] = []
    for fold in [int(item) for item in config["validation"]["expected_folds"]]:
        mask = np.asarray(oof_fold) == fold
        if not np.any(mask):
            raise ValueError(f"exp502 OOF fold {fold} has no rows")
        control_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "saved_exp413_rmse": control_rmse,
                "exp502_rmse": candidate_rmse,
                "delta_rmse_exp502_minus_exp413": candidate_rmse - control_rmse,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base["md_since"].to_numpy(np.float32)
    masks = {
        "md_since_0_250": md_since <= 250.0,
        "md_since_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "md_since_1000_plus": md_since >= 1000.0,
    }
    scope_rows: list[dict[str, Any]] = []
    for scope, mask in masks.items():
        control_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(base.loc[mask, "well"].nunique()),
                "saved_exp413_rmse": control_rmse,
                "exp502_rmse": candidate_rmse,
                "delta_rmse_exp502_minus_exp413": candidate_rmse - control_rmse,
            }
        )
    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str}).set_index(
        "well_id"
    )
    hidden_columns = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    hidden_rows: list[dict[str, Any]] = []
    for scope, column in hidden_columns.items():
        mask = base["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        if not np.any(mask):
            raise ValueError(f"hidden-like assignment has no valid rows for {scope}")
        control_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        hidden_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(base.loc[mask, "well"].nunique()),
                "saved_exp413_rmse": control_rmse,
                "exp502_rmse": candidate_rmse,
                "delta_rmse_exp502_minus_exp413": candidate_rmse - control_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)

    by_well_source = pd.DataFrame(
        {
            "well": base["well"].astype(str),
            "actual_tvt": truth,
            "saved_exp413": parent,
            "exp502": candidate,
        }
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in by_well_source.groupby("well", sort=True):
        control_rmse = _rmse(group["actual_tvt"], group["saved_exp413"])
        candidate_rmse = _rmse(group["actual_tvt"], group["exp502"])
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "saved_exp413_rmse": control_rmse,
                "exp502_rmse": candidate_rmse,
                "delta_rmse_exp502_minus_exp413": candidate_rmse - control_rmse,
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    pooled_control = _rmse(truth, parent)
    pooled_candidate = _rmse(truth, candidate)
    gain = pooled_control - pooled_candidate
    nonworse_folds = int((fold_metrics["delta_rmse_exp502_minus_exp413"] <= 0.0).sum())
    scope_table = pd.concat([pd.DataFrame(scope_rows), hidden_metrics], ignore_index=True)
    promotion = dict(config["validation"]["promotion"])
    if set(scope_table["scope"]) != set(str(item) for item in promotion["required_scopes"]):
        raise ValueError("exp502 scope inventory differs from preregistered gate")
    maximum_scope_delta = float(scope_table["delta_rmse_exp502_minus_exp413"].max())
    checks = {
        "minimum_pooled_rmse_gain": gain
        >= float(promotion["minimum_pooled_rmse_gain_ft"]),
        "minimum_nonworse_folds": nonworse_folds
        >= int(promotion["minimum_nonworse_folds"]),
        "maximum_scope_delta": maximum_scope_delta
        <= float(promotion["maximum_scope_delta_rmse_ft"]),
        "all_technical_checks": bool(technical_checks)
        and all(bool(item) for item in technical_checks.values()),
    }
    delta = by_well["delta_rmse_exp502_minus_exp413"]
    tail = {
        "by_well_delta_p95": float(delta.quantile(0.95)),
        "worst_well": str(by_well.loc[delta.idxmax(), "well"]),
        "worst_well_delta_rmse": float(delta.max()),
        "worsened_well_count_plus_1ft": int((delta > 1.0).sum()),
        "worsened_well_count_plus_3ft": int((delta > 3.0).sum()),
        "worsened_well_count_plus_5ft": int((delta > 5.0).sum()),
        "policy": "report_only_not_automatic_stop",
    }
    gate = {
        "saved_exp413_rmse": pooled_control,
        "exp502_rmse": pooled_candidate,
        "gain_ft": gain,
        "delta_rmse_exp502_minus_exp413": pooled_candidate - pooled_control,
        "nonworse_folds": nonworse_folds,
        "maximum_scope_delta_rmse": maximum_scope_delta,
        "technical_checks": dict(technical_checks),
        "checks": checks,
        "tail_readout": tail,
        "passed": bool(all(checks.values())),
        "pass_action": promotion["pass_action"],
        "fail_action": promotion["fail_action"],
    }
    return gate, fold_metrics, pd.DataFrame(scope_rows), hidden_metrics, by_well


# %% [markdown]
# ## 6. Setup, configuration, and input checks
#
# 実行時は親config、exp501 contract、4 saved artifact root、exp404 prediction、
# exp072/099/111 cache、exp218/145 source、clean allowlist、hidden-like assignmentを
# SHA検証する。ここまででselector / signed selector / control再学習は0。

# %% [markdown]
# ## 7. Final373 fold assembly and 15-booster training
#
# 下のrunnerは入力freeze後、各outer foldでexp501 compactとexp413 signed compactを
# key/role照合し、final373を構築して3 configsだけを学習する。内部の処理順も
# MarkdownコメントでStage 7/8/9に分け、生成物保存とfixed stopまで追跡可能にする。

# %%
def run_exp502_train() -> dict[str, Any]:
    import matplotlib.pyplot as plt

    require_notebook_runtime()
    config = load_yaml(resolve_config_path())
    static_contract = validate_static_contract(config)
    require_train_authorization(config)
    data_root = competition_data_root()
    raw_train_dir = data_root / "train"
    output_dir = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if is_kaggle_runtime()
        else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_exp413_path = resolve_file(config["data"]["parent_configs"]["exp413"])
    parent_exp501_path = resolve_file(config["data"]["parent_configs"]["exp501"])
    exp501_contract_path = resolve_file(config["data"]["exp501_candidate_contract"])
    parent_exp413 = load_yaml(parent_exp413_path)
    parent_exp501 = load_yaml(parent_exp501_path)
    exp501_contract = load_yaml(exp501_contract_path)
    if parent_exp501["results"]["decision"] != (
        "FAIL_CLOSE_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR"
    ):
        raise ValueError("exp501 terminal selector decision changed")

    exp413_spec = dict(config["data"]["exp413_source"])
    exp501_spec = dict(config["data"]["exp501_selector_source"])
    exp501_root = resolve_artifact_root(
        exp501_spec["root_patterns"],
        marker="nested_compact_manifest.json",
        expected_marker_sha256=str(exp501_spec["compact_manifest_sha256"]),
    )
    exp413_stage_c_root = resolve_artifact_root(
        exp413_spec["stage_c_root_patterns"],
        marker="nested_compact_manifest.json",
        expected_marker_sha256=str(exp413_spec["removed_nested_compact_manifest_sha256"]),
    )
    exp413_stage_s_root = resolve_artifact_root(
        exp413_spec["stage_s_root_patterns"],
        marker="signed_compact_manifest.json",
        expected_marker_sha256=str(exp413_spec["retained_signed_compact_manifest_sha256"]),
    )
    exp413_stage_d_root = resolve_artifact_root(
        exp413_spec["stage_d_root_patterns"],
        marker="stage_d_model_manifest.json",
        expected_marker_sha256=str(exp413_spec["stage_d_model_manifest_sha256"]),
    )
    exp501_evidence, removed_evidence, signed_evidence, fold_manifest_sha = (
        verify_saved_feature_sources(
            config=config,
            exp501_contract=exp501_contract,
            exp501_root=exp501_root,
            exp413_stage_c_root=exp413_stage_c_root,
            exp413_stage_s_root=exp413_stage_s_root,
        )
    )

    frozen_prediction_path = resolve_by_patterns(
        parent_exp413["data"]["exp404_scale5_train_prediction"]["patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp404_scale5_train_prediction"][
            "expected_raw_sha256"
        ],
    )
    exp218_source_path = resolve_by_patterns(
        parent_exp413["data"]["exp218_source"]["script_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp218_source"]["script_sha256"],
    )
    exp218_config_path = resolve_by_patterns(
        parent_exp413["data"]["exp218_source"]["config_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp218_source"]["config_sha256"],
    )
    exp145_source_path = resolve_by_patterns(
        parent_exp413["data"]["exp145_source"]["script_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp145_source"]["script_sha256"],
    )
    exp145_config_path = resolve_by_patterns(
        parent_exp413["data"]["exp145_source"]["config_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp145_source"]["config_sha256"],
    )
    multiobs_source_path = resolve_by_patterns(
        parent_exp413["data"]["exp145_source"]["multiobs_script_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp145_source"][
            "multiobs_script_sha256"
        ],
    )
    exp099_source_path = resolve_by_patterns(
        parent_exp413["data"]["exp099_train_feature_cache"]["patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp099_train_feature_cache"][
            "expected_raw_sha256"
        ],
    )
    exp111_schema_path = resolve_by_patterns(
        parent_exp413["data"]["exp111_saved_models"]["schema_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp111_saved_models"]["schema_sha256"],
    )
    exp111_manifest_path = resolve_by_patterns(
        parent_exp413["data"]["exp111_saved_models"]["manifest_patterns"],
        search_roots(),
        marker_sha256=parent_exp413["data"]["exp111_saved_models"][
            "manifest_sha256"
        ],
    )
    clean_allowlist_path = resolve_file(parent_exp413["data"]["clean_base_allowlist"])
    hidden_like_assignment_path = resolve_file(
        parent_exp413["data"]["hidden_like_assignment"]
    )

    input_contract = {
        "status": "inputs_frozen_before_exp502_feature_assembly",
        "static_contract": static_contract,
        "parent_configs": {
            "exp413": {
                "path": str(parent_exp413_path),
                "sha256": sha256_file(parent_exp413_path),
            },
            "exp501": {
                "path": str(parent_exp501_path),
                "sha256": sha256_file(parent_exp501_path),
                "terminal_decision": parent_exp501["results"]["decision"],
            },
        },
        "exp501_candidate_contract_sha256": sha256_file(exp501_contract_path),
        "fold_manifest_sha256": fold_manifest_sha,
        "exp501_compact": exp501_evidence,
        "exp413_removed_compact": removed_evidence,
        "exp413_retained_signed": signed_evidence,
        "control_retraining_boosters": 0,
        "selector_retraining_boosters": 0,
        "signed_selector_retraining_boosters": 0,
        "hmm_pf_beam_reruns": 0,
    }
    write_json(output_dir / "exp502_input_contract.json", input_contract)

    base, base_features, base_evidence, exp218, exp218_config = (
        build_replacement_clean273_surface(
            config=parent_exp413,
            frozen_prediction_path=frozen_prediction_path,
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            exp099_source_path=exp099_source_path,
            exp145_source_path=exp145_source_path,
            exp145_config_path=exp145_config_path,
            multiobs_source_path=multiobs_source_path,
            exp111_schema_path=exp111_schema_path,
            exp111_manifest_path=exp111_manifest_path,
            clean_allowlist_path=clean_allowlist_path,
            raw_train_dir=raw_train_dir,
        )
    )
    required_base = list(
        dict.fromkeys(
            ["id", "well", "target", "last_known_tvt", "md_since", *base_features]
        )
    )
    base = base.loc[:, ~base.columns.duplicated()].loc[:, required_base].copy()
    if len(base) != int(config["validation"]["expected_rows"]):
        raise ValueError("exp413 clean273 base row count changed")

    saved_control, control_evidence, removed_features, control_signed_features = (
        load_saved_exp413_control(
            root=exp413_stage_d_root,
            base=base,
            base_features=base_features,
            config=config,
        )
    )
    replacement_features = [str(item) for item in exp501_evidence["compact_features"]]
    signed_features = [str(item) for item in signed_evidence["features"]]
    if signed_features != control_signed_features:
        raise ValueError("retained exp413 signed23 differs from saved Stage D schema")
    surface = build_feature_surface_contract(
        base_features=base_features,
        replacement_features=replacement_features,
        signed_features=signed_features,
        removed_features=removed_features,
    )
    final_features = [str(item) for item in surface["features"]]
    write_json(output_dir / "exp502_selector_block_replacement_manifest.json", surface)
    pd.DataFrame(
        [
            {
                "position": position,
                "feature": feature,
                "group": "clean_base"
                if position < 273
                else "exp501_nested_compact"
                if position < 350
                else "exp413_signed_compact",
                "source": "exp501" if 273 <= position < 350 else "exp413",
            }
            for position, feature in enumerate(final_features)
        ]
    ).to_csv(output_dir / "exp502_final_feature_schema.csv", index=False)

    # Stage 7: Final373 fold assembly and 15-booster training.
    # 各outer foldでexp501 compact train=4 inner-OOF partitions、valid=4-model
    # ensemble partitionを読む。同じfoldのexp413 signed23と全KEY_COLUMNSを照合し、
    # clean273へid joinする。各fold 3 configsだけを学習する。

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    stage_cfg = dict(config["model"]["downstream_tvt"])
    mode = dict(exp218_config["model"]["training"]["modes"][str(stage_cfg["mode"])])
    if not bool(mode.get("use_gpu", False)):
        raise ValueError("exp502 must inherit the exp413 GPU mode")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode
    )
    config_indices = [int(item) for item in stage_cfg["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    base_index = pd.Index(base["id"].astype(str))
    target = base["target"].to_numpy(np.float32)
    anchor = base["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    n_rows = len(base)
    oof_by_config = [np.full(n_rows, np.nan, dtype=np.float32) for _ in config_indices]
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    model_dir = output_dir / "exp502_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_surface_rows: list[dict[str, Any]] = []

    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=exp501_root,
            stage_c_evidence=exp501_evidence,
            downstream_outer_fold=outer_fold,
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=signed_evidence,
            downstream_outer_fold=outer_fold,
        )
        train_positions, train_alignment = validate_fold_alignment(
            base=base,
            compact=compact_train,
            signed=signed_train,
            downstream_outer_fold=outer_fold,
            role="train",
        )
        valid_positions, valid_alignment = validate_fold_alignment(
            base=base,
            compact=compact_valid,
            signed=signed_valid,
            downstream_outer_fold=outer_fold,
            role="valid",
        )
        if np.intersect1d(train_positions, valid_positions).size:
            raise ValueError("exp502 train/valid rows overlap")
        if len(np.unique(np.concatenate([train_positions, valid_positions]))) != n_rows:
            raise ValueError("exp502 fold does not cover all clean273 rows exactly once")
        if np.any(oof_fold[valid_positions] >= 0):
            raise ValueError("exp502 OOF row assigned twice")
        if not saved_control.iloc[valid_positions]["outer_fold"].eq(outer_fold).all():
            raise ValueError("exp502 valid fold differs from saved exp413 control")
        oof_fold[valid_positions] = np.int8(outer_fold)

        x_train = assemble_matrix(
            base=base,
            positions=train_positions,
            compact=compact_train,
            signed=signed_train,
            base_features=base_features,
            compact_features=replacement_features,
            signed_features=signed_features,
            chunk_columns=int(stage_cfg["matrix_copy_chunk_columns"]),
        )
        x_valid = assemble_matrix(
            base=base,
            positions=valid_positions,
            compact=compact_valid,
            signed=signed_valid,
            base_features=base_features,
            compact_features=replacement_features,
            signed_features=signed_features,
            chunk_columns=int(stage_cfg["matrix_copy_chunk_columns"]),
        )
        fold_surface_rows.extend(
            [
                {
                    **train_alignment,
                    "matrix_content_sha256": matrix_content_sha256(
                        x_train, final_features
                    ),
                },
                {
                    **valid_alignment,
                    "matrix_content_sha256": matrix_content_sha256(
                        x_valid, final_features
                    ),
                },
            ]
        )
        x_train_frame = pd.DataFrame(x_train, columns=final_features, copy=False)
        x_valid_frame = pd.DataFrame(x_valid, columns=final_features, copy=False)
        fold_predictions: list[np.ndarray] = []
        for config_index, params in zip(config_indices, params_family, strict=True):
            model = LGBMRegressor(**params)
            model.fit(
                x_train_frame,
                target[train_positions],
                eval_set=[(x_valid_frame, target[valid_positions])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(stage_cfg["early_stopping_rounds"]), verbose=False),
                    log_evaluation(int(stage_cfg["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            residual = model.predict(
                x_valid_frame, num_iteration=best_iteration
            ).astype(np.float32)
            prediction = (anchor[valid_positions] + residual).astype(np.float32)
            oof_by_config[config_indices.index(config_index)][valid_positions] = residual
            fold_predictions.append(prediction)
            model_path = (
                model_dir
                / f"exp501_fixed13_selector_replacement__lgb{config_index}__outer{outer_fold}.txt"
            )
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            model_rows.append(
                {
                    "variant": "exp501_fixed13_selector_replacement",
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": outer_fold,
                    "feature_count": 373,
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                }
            )
            fold_model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_positions),
                    "rmse_tvt": _rmse(truth[valid_positions], prediction),
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                for position, (feature, value) in enumerate(
                    zip(final_features, importance, strict=True)
                ):
                    group = (
                        "clean_base"
                        if position < 273
                        else "exp501_nested_compact"
                        if position < 350
                        else "exp413_signed_compact"
                    )
                    importance_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "model": f"lgb{config_index}",
                            "importance_type": importance_type,
                            "feature": feature,
                            "feature_group": group,
                            "importance": float(value),
                        }
                    )
            print(
                json.dumps(
                    {
                        "stage": "exp502_train",
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": fold_model_rows[-1]["rmse_tvt"],
                        "completed_boosters": len(model_rows),
                        "planned_boosters": 15,
                        "saved_exp413_control_retraining": 0,
                        "selector_retraining": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, residual
            gc.collect()
        fold_mean = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_positions),
                "rmse_tvt": _rmse(truth[valid_positions], fold_mean),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            x_train,
            x_valid,
            x_train_frame,
            x_valid_frame,
            fold_predictions,
        )
        gc.collect()

    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("exp502 model/OOF contract is incomplete")
    if not np.array_equal(oof_fold, saved_control["outer_fold"].to_numpy(np.int8)):
        raise AssertionError("exp502 fold assignment differs from saved exp413")
    if any(not np.isfinite(item).all() for item in oof_by_config):
        raise AssertionError("exp502 OOF residual is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)

    technical_checks = {
        "input_sha_and_fold_manifest_parity": len(set(fold_manifest_sha.values())) == 1,
        "clean273_schema": len(base_features) == 273,
        "removed_exp413_nested_block_instances_zero": surface[
            "old_selector_block_instances_in_final"
        ]
        == 0,
        "inserted_exp501_compact77_once": surface[
            "new_selector_block_instances_in_final"
        ]
        == 1,
        "signed23_schema": len(signed_features) == 23,
        "final373_schema_unique": len(final_features) == len(set(final_features)) == 373,
        "model_count_15": len(model_rows) == 15,
        "saved_exp413_control_retraining_zero": control_evidence["models_retrained"] == 0,
        "selector_signed_hmm_pf_beam_retraining_zero": True,
        "row_fold_role_alignment": all(
            int(item["missing_base_rows"]) == 0 and int(item["duplicate_ids"]) == 0
            for item in fold_surface_rows
        ),
    }
    gate, fold_metrics, scope_metrics, hidden_metrics, by_well = evaluate_exp502_gate(
        config=config,
        base=base,
        saved_control=saved_control,
        oof_fold=oof_fold,
        prediction=mean_prediction,
        hidden_like_assignment_path=hidden_like_assignment_path,
        technical_checks=technical_checks,
    )

    prediction_frame = base[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    prediction_frame["saved_exp413__lgb_mean__pred_tvt"] = saved_control[
        str(exp413_spec["stage_d_prediction_column"])
    ].to_numpy(np.float32)
    for config_index, residual in zip(config_indices, oof_by_config, strict=True):
        prediction_frame[
            f"exp501_fixed13_selector_replacement__lgb{config_index}__pred_tvt"
        ] = (anchor + residual).astype(np.float32)
    prediction_frame[
        "exp501_fixed13_selector_replacement__lgb_mean__pred_tvt"
    ] = mean_prediction

    # Stage 8: Metrics, feature importance, and generated artifacts.
    paths = {
        "oof": output_dir / "exp502_oof_predictions.parquet",
        "fold_metrics": output_dir / "exp502_fold_metrics.csv",
        "scope_metrics": output_dir / "exp502_scope_metrics.csv",
        "hidden_metrics": output_dir / "exp502_hidden_scope_metrics.csv",
        "by_well": output_dir / "exp502_by_well_metrics.csv",
        "importance": output_dir / "exp502_feature_importance.csv",
        "model_manifest": output_dir / "exp502_model_manifest.json",
        "feature_manifest": output_dir / "exp502_final_feature_manifest.json",
        "metrics": output_dir / "exp502_metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(
        fold_metrics, on="outer_fold", how="left"
    ).to_csv(paths["fold_metrics"], index=False)
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    pd.DataFrame(importance_rows).to_csv(paths["importance"], index=False)
    feature_manifest = {
        "status": "exp502_final373_assembled",
        "surface": surface,
        "clean273": base_evidence,
        "fold_surfaces": fold_surface_rows,
        "fold_manifest_sha256": fold_manifest_sha,
    }
    write_json(paths["feature_manifest"], feature_manifest)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "exp502_train_complete",
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": 373,
        "feature_schema_sha256": surface["feature_schema_sha256"],
        "feature_groups": {
            "exp413_clean_base": list(base_features),
            "exp501_nested_compact": replacement_features,
            "exp413_signed_compact": signed_features,
        },
        "removed_exp413_nested_compact": removed_features,
        "saved_exp413_control_retraining_boosters": 0,
        "selector_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items() if path.exists()}
    metrics = {
        "schema_version": "1.0.0",
        "status": "train_complete_gate_passed"
        if gate["passed"]
        else "train_complete_gate_failed_closed",
        "rows": n_rows,
        "wells": int(base["well"].nunique()),
        "feature_counts": {
            "clean_base": 273,
            "removed_exp413_nested": 74,
            "inserted_exp501_nested": 77,
            "signed_compact": 23,
            "final": 373,
        },
        "model_count": len(model_rows),
        "cost_contract": static_contract["cost"],
        "saved_exp413_control": control_evidence,
        "primary_gate": gate,
        "artifact_sha256": artifact_sha,
    }
    write_json(paths["metrics"], metrics)
    artifact_sha["metrics"] = sha256_file(paths["metrics"])

    importance = pd.DataFrame(importance_rows)
    importance_mean = (
        importance[importance["importance_type"].eq("gain")]
        .groupby(["feature", "feature_group"], as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    display(fold_metrics)
    display(scope_metrics)
    display(hidden_metrics)
    display(by_well.sort_values("delta_rmse_exp502_minus_exp413", ascending=False).head(80))
    display(importance_mean.head(100))
    top = importance_mean.head(30).sort_values("importance")
    if len(top):
        ax = top.plot.barh(
            x="feature",
            y="importance",
            figsize=(11, 11),
            legend=False,
            title="exp502 final373 mean gain importance",
        )
        ax.set_xlabel("mean gain across 15 treatment models")
        plt.tight_layout()
        plt.savefig(output_dir / "exp502_feature_importance_top30.png", dpi=140)
        plt.show()

    # Stage 9: Reproducibility evidence and fixed stop.
    # GPU bitwise一致は主張しない。15 model、final373 fold matrix、OOF、入力rootの
    # SHAをrun単位の証拠にする。gate PASSでもinference実装資格を得るだけで停止する。

    reproducibility = {
        "status": "exp502_train_reproducibility_recorded",
        "deterministic_anchor": False,
        "seed": int(config["reproducibility"]["seed"]),
        "runtime": "kaggle_gpu" if is_kaggle_runtime() else "approved_local_smoke",
        "gpu_bitwise_determinism_claimed": False,
        "input_contract_sha256": sha256_file(output_dir / "exp502_input_contract.json"),
        "feature_manifest_sha256": sha256_file(paths["feature_manifest"]),
        "model_manifest_sha256": sha256_file(paths["model_manifest"]),
        "oof_prediction_sha256": sha256_file(paths["oof"]),
        "artifact_sha256": artifact_sha,
        "model_count": len(model_rows),
        "control_retraining_boosters": 0,
        "selector_retraining_boosters": 0,
        "signed_selector_retraining_boosters": 0,
        "hmm_pf_beam_reruns": 0,
        "inference_executed": False,
        "submission_generated": False,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if gate["passed"]:
        print(
            "exp502 train gate PASS. Stop here; inference implementation/run, "
            "submission generation, and external "
            "submission require separate approval."
        )
    else:
        print(
            "exp502 train gate FAIL. Close without same-OOF feature subset, blend, "
            "weight, threshold, or gate rescue."
        )
    print("Inference executed: False")
    print("Submission generated or submitted: False")
    return metrics


# %% [markdown]
# ## 8. Metrics, feature importance, and generated artifacts
#
# runnerはfold/scope/hidden-like/by-well、15-model manifest、final373 schema/content、
# OOF SHA、feature importanceを保存・表示する。

# %% [markdown]
# ## 9. Reproducibility evidence and fixed stop
#
# import/test時はrunnerを呼ばない。Kaggle Notebook実行時だけauthorization gateを通し、
# train gate判定後はinferenceやsubmissionへ進まず停止する。

# %%
if os.environ.get("EXP502_IMPORT_ONLY", "0") != "1":
    run_exp502_train()
