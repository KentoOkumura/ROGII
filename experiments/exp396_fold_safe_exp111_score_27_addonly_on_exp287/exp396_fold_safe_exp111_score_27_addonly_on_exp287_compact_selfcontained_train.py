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
# # exp396 fold-safe exp111 score 27 add-only on exp287 — train
#
# exp111 の保存済み fold-0 scorer は使わず、exp287 の downstream outer 5 folds の
# 各 outer-train 内で4個の well-group inner foldsを作る。outer-train行にはinner OOF、
# outer-valid行にはouter-trainだけで学習した4 model平均を与え、固定27列を生成する。
# この実装ターンはStage Aコードと0-booster preflightまでを対象とし、CPU scorer実行、
# Stage Bの15 GPU boosters、inference、submissionは別承認なしに開始しない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Scientific, execution, and cost contract
# 3. Frozen input and parent-artifact verification
# 4. exp111 target-free 48-feature contract
# 5. Stable nested folds, sampling, and imputation
# 6. Fixed 10-core to 27-feature derivation
# 7. Stage A zero-booster preflight
# 8. Strict nested scorer training
# 9. Stage A quality and resource gates
# 10. Orchestration, diagnostics, and generated artifacts

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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.model_selection import GroupKFold

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

EXPERIMENT_NAME = "exp396_fold_safe_exp111_score_27_addonly_on_exp287"
PARENT_EXPERIMENT = "exp287_fold_safe_formation_74_addonly_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP396_IMPORT_ONLY", "0") == "1"

CANDIDATE_ORDER = ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]
LONG_CANDIDATE_COLUMNS = [
    "candidate_index",
    "candidate_tvt",
    "candidate_minus_last",
    "candidate_abs_minus_likpf",
    "candidate_abs_minus_row_mean",
    "candidate_z_within_row",
    "candidate_multiobs_score",
    "candidate_multiobs_mae",
    "candidate_multiobs_ncc",
    "candidate_score_gap_from_best",
    "candidate_score_centered",
    "candidate_mae_gap_from_best",
    "candidate_ncc_gap_from_best",
    "candidate_score_rank",
    "candidate_mae_rank",
    "candidate_ncc_rank",
]
FEATURE_COLUMNS_27 = [
    "ll_learned_prob_top1_index",
    "ll_learned_error_top1_index",
    "ll_learned_prob_top1_value",
    "ll_learned_prob_top2_value",
    "ll_learned_prob_margin_top1_top2",
    "ll_learned_prob_entropy",
    "ll_learned_error_top1_value",
    "ll_learned_error_top2_value",
    "ll_learned_error_margin_top2_top1",
    "ll_learned_prob_likpf_rank",
    "ll_learned_error_likpf_rank",
    "ll_learned_prob_top3_contains_likpf",
    "ll_learned_error_top3_contains_likpf",
    "ll_learned_prob_pf_ancc",
    "ll_learned_pred_abs_error_pf_ancc",
    "ll_learned_prob_beam_mean",
    "ll_learned_pred_abs_error_beam_mean",
    "ll_learned_prob_likpf_mean",
    "ll_learned_pred_abs_error_likpf_mean",
    "ll_learned_prob_sc_ens",
    "ll_learned_pred_abs_error_sc_ens",
    "ll_learned_prob_hyb",
    "ll_learned_pred_abs_error_hyb",
    "ll_learned_prob_weighted_tvt_minus_last_known_tvt",
    "ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt",
    "ll_learned_error_weighted_tvt_minus_last_known_tvt",
    "ll_learned_error_weighted_tvt_minus_likpf_mean_tvt",
]
SCORE_CORE_COLUMNS = [
    value
    for candidate in CANDIDATE_ORDER
    for value in (
        f"ll_learned_prob_{candidate}",
        f"ll_learned_pred_abs_error_{candidate}",
    )
]
DEPENDENT_GRWR_SIX = [
    "grwr_candidate_tvt_std",
    "grwr_candidate_tvt_range",
    "grwr_dwt_energy_ratio_w065_x_candidate_std",
    "grwr_fft_rotation_ratio_x_candidate_range",
    "grwr_dwt_minus_raw_ncc_gap_x_candidate_range",
    "grwr_ll_entropy_x_dwt_energy_ratio_w065",
]
PROTECTED_LABEL_COLUMNS = {
    "target",
    "TVT",
    "true_tvt",
    "actual_tvt",
    "abs_error",
    "within_10ft",
    "oracle_label",
    "oracle_candidate",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )


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


def logical_identity_sha256(*columns: pd.Series | np.ndarray | Sequence[Any]) -> str:
    frame = pd.DataFrame(
        {f"column_{index}": np.asarray(column) for index, column in enumerate(columns)}
    )
    hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy(
        dtype="<u8", copy=False
    )
    digest = hashlib.sha256()
    digest.update(b"pandas_hash_pandas_object_categorize_v1\0")
    digest.update(hashes.tobytes(order="C"))
    return digest.hexdigest()


def logical_float_frame_sha256(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    value_columns: Sequence[str],
) -> str:
    ordered = frame.sort_values(list(identity_columns), kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(sha256_json(list(identity_columns) + list(value_columns)).encode("ascii"))
    digest.update(
        logical_identity_sha256(*(ordered[column] for column in identity_columns)).encode("ascii")
    )
    values = ordered[list(value_columns)].to_numpy(np.float32, copy=True).astype(
        "<f4", copy=False
    )
    if not np.isfinite(values).all():
        raise ValueError("logical float frame contains non-finite values")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def stable_seed_from_material(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "little")


def current_peak_rss_gb() -> float:
    # Linux ru_maxrss is KiB. Kaggle's supported runtime is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(PACKAGE_DIR.rglob(f"{EXPERIMENT_NAME}/config.yaml"))
    matches = sorted({path.resolve() for path in candidates if path.exists()})
    if len(matches) != 1:
        raise FileNotFoundError(f"exp396 config resolution is ambiguous: {matches}")
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
        if not direct.is_absolute():
            package_relative = PACKAGE_DIR / direct
            if package_relative.exists():
                candidates.append(package_relative.resolve())
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path.resolve() for path in root.glob(raw) if path.exists())
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(f"{label} not found for patterns={list(patterns)}")
    return unique[0]


def resolve_artifact_root(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    required_files: Sequence[str],
    label: str,
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        path = Path(raw)
        if path.is_dir():
            candidates.append(path.resolve())
        if not path.is_absolute() and (PACKAGE_DIR / path).is_dir():
            candidates.append((PACKAGE_DIR / path).resolve())
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path.resolve() for path in root.glob(raw) if path.is_dir())
    checked: list[str] = []
    for candidate in dict.fromkeys(candidates):
        checked.append(str(candidate))
        if all((candidate / filename).is_file() for filename in required_files):
            return candidate
    raise FileNotFoundError(
        f"complete {label} root not found; checked={json.dumps(checked[:40])}"
    )


def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


# %% [markdown]
# ## 2. Scientific, execution, and cost contract
#
# Stage A は `5 outer × 4 inner × 2 objectives = 40` CPU boosters。
# `implementation_only`ではfitもKaggle input readも行わない。preflight/trainはそれぞれ
# approval flagを必要とする。Stage Bは実装も未承認で、15 GPU boostersとcontrol再学習0を
# 別途再提示するまで選択できない。

# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_execution_approval: bool = False,
) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name contract changed")
    if nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp396 route must remain ml_model")
    if nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp396 parent contract changed")
    if not bool(nested(config, "execution.implementation_approved")):
        raise RuntimeError("exp396 Stage A implementation is not approved")

    stage = str(nested(config, "execution.stage"))
    allowed = {str(value) for value in nested(config, "execution.allowed_stages")}
    expected_allowed = {
        "implementation_only",
        "stage_a_preflight_only",
        "stage_a_cpu_scorer_train",
        "stage_b_preflight_only",
        "stage_b_gpu_tvt_train",
    }
    if allowed != expected_allowed or stage not in allowed:
        raise ValueError(f"unexpected exp396 execution-stage contract: {stage}, {allowed}")

    candidates = [
        str(item["name"]) for item in nested(config, "scorer.candidates")
    ]
    candidate_columns = [
        str(item["column"]) for item in nested(config, "scorer.candidates")
    ]
    if candidates != CANDIDATE_ORDER or candidate_columns != CANDIDATE_ORDER:
        raise ValueError("fixed exp111 candidate order changed")
    if list(nested(config, "scorer.feature_columns")) != FEATURE_COLUMNS_27:
        raise ValueError("fixed 27-column schema/order changed")
    if int(nested(config, "scorer.feature_count")) != 27:
        raise ValueError("exp396 must derive exactly 27 score columns")
    if int(nested(config, "scorer.input_feature_count")) != 48:
        raise ValueError("exp396 scorer input must remain 48 features")

    outer = int(nested(config, "scorer.outer_folds"))
    inner = int(nested(config, "scorer.inner_folds"))
    objectives = int(nested(config, "scorer.objectives_per_inner"))
    planned_cpu = int(nested(config, "scorer.planned_cpu_boosters"))
    calculated_cpu = outer * inner * objectives
    if (outer, inner, objectives, planned_cpu, calculated_cpu) != (5, 4, 2, 40, 40):
        raise ValueError("Stage A cost must remain outer5 x inner4 x 2 = 40 CPU boosters")

    stage_b = dict(nested(config, "model.execution_count"))
    expected_stage_b = {
        "active_variants": 1,
        "variant": "fold_safe_exp111_score_27_addonly",
        "lightgbm_configs": 3,
        "folds": 5,
        "planned_gpu_boosters": 15,
        "control_retraining_boosters": 0,
    }
    if stage_b != expected_stage_b:
        raise ValueError(f"Stage B cost contract changed: {stage_b}")
    if int(nested(config, "model.source_surface.parent_feature_count")) != 421:
        raise ValueError("exp396 parent surface must remain 421 features")
    if int(nested(config, "model.source_surface.final_feature_count")) != 448:
        raise ValueError("exp396 final surface must remain 448 features")

    forbidden_true = {
        "run_inference": bool(nested(config, "execution.run_inference")),
        "create_submission": bool(nested(config, "execution.create_submission")),
        "submit_to_kaggle": bool(nested(config, "execution.submit_to_kaggle")),
        "control_retraining": bool(nested(config, "execution.control_retraining")),
    }
    if any(forbidden_true.values()):
        raise ValueError(f"forbidden exp396 execution flag enabled: {forbidden_true}")
    if require_execution_approval:
        if stage == "stage_a_preflight_only":
            if not bool(nested(config, "execution.stage_a_preflight_run_approved")):
                raise RuntimeError("Stage A preflight requires separate user approval")
            if not bool(nested(config, "execution.kaggle_push_approved")):
                raise RuntimeError("Stage A preflight requires separate Kaggle push approval")
        elif stage == "stage_a_cpu_scorer_train":
            if not bool(nested(config, "execution.stage_a_cpu_run_approved")):
                raise RuntimeError("40 CPU boosters require separate user approval")
            if not bool(nested(config, "execution.kaggle_push_approved")):
                raise RuntimeError("40 CPU boosters require separate Kaggle push approval")
            if not bool(nested(config, "execution.run_train")):
                raise RuntimeError("40 CPU boosters require execution.run_train=true")
        elif stage in {"stage_b_preflight_only", "stage_b_gpu_tvt_train"}:
            if not bool(nested(config, "execution.stage_b_implementation_approved")):
                raise RuntimeError("Stage B implementation requires separate user approval")
            if not bool(nested(config, "execution.kaggle_push_approved")):
                raise RuntimeError("Stage B requires separate Kaggle push approval")
            if stage == "stage_b_gpu_tvt_train":
                if not bool(nested(config, "execution.stage_b_gpu_run_approved")):
                    raise RuntimeError("15 GPU boosters require separate user approval")
                if not bool(nested(config, "execution.run_train")):
                    raise RuntimeError("15 GPU boosters require execution.run_train=true")
        else:
            raise RuntimeError(f"selected stage does not execute an approved stage: {stage}")

    if bool(nested(config, "runtime.stage_a.enable_internet")):
        raise ValueError("Stage A Kaggle internet must remain disabled")
    if bool(nested(config, "runtime.stage_b.enable_internet")):
        raise ValueError("Stage B Kaggle internet must remain disabled")
    if stage.startswith("stage_b_"):
        if not bool(nested(config, "execution.stage_b_implementation_approved")):
            raise RuntimeError("Stage B is not approved")
        if not bool(nested(config, "outcome.stage_a_gate_passed")):
            raise RuntimeError("Stage B requires the recorded Stage A all-gates PASS")
    return {
        "stage": stage,
        "stage_a_active_variants": 1,
        "outer_folds": outer,
        "inner_folds": inner,
        "objectives": objectives,
        "planned_cpu_boosters": calculated_cpu,
        "stage_b_planned_gpu_boosters": 15,
        "control_retraining_boosters": 0,
        "inference": False,
        "submission": False,
    }


# %% [markdown]
# ## 3. Frozen input and parent-artifact verification
#
# exp099 wide cache、exp111 source/config、exp287 OOF/model/metrics/fold/by-well/
# formation manifest/raw schema、corrected exp264 OOFをSHA固定する。exp287 OOFの
# `outer_fold`をdownstream foldの唯一の正とし、score cacheとID/well/actual TVTを照合する。

# %%
def verify_parent_artifacts(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = dict(config["data"])
    fixed_files = {
        "oof": (
            root / "fold_safe_formation_oof_predictions.parquet",
            data["expected_exp287_oof_sha256"],
        ),
        "model_manifest": (
            root / "model_manifest.json",
            data["expected_exp287_model_manifest_sha256"],
        ),
        "metrics": (root / "metrics.json", data["expected_exp287_metrics_sha256"]),
        "fold_metrics": (
            root / "fold_metrics.csv",
            data["expected_exp287_fold_metrics_sha256"],
        ),
        "by_well": (
            root / "by_well_metrics.csv",
            data["expected_exp287_by_well_sha256"],
        ),
        "formation_manifest": (
            root / "formation_fold_manifest.json",
            data["expected_exp287_formation_fold_manifest_sha256"],
        ),
        "raw_schema": (
            root / "raw_train_current_test_schema_audit.csv",
            data["expected_exp287_raw_schema_audit_sha256"],
        ),
    }
    actual = {
        name: verify_file_sha(path, expected, f"saved exp287 {name}")
        for name, (path, expected) in fixed_files.items()
    }
    manifest = json.loads(fixed_files["model_manifest"][0].read_text())
    if int(manifest.get("model_count", -1)) != 15:
        raise ValueError("saved exp287 model manifest must contain 15 models")
    if int(manifest.get("feature_count", -1)) != 421:
        raise ValueError("saved exp287 model manifest must contain 421 features")
    if str(manifest.get("feature_schema_sha256")) != str(
        data["expected_exp287_feature_schema_sha256"]
    ):
        raise ValueError("saved exp287 logical feature schema SHA mismatch")
    groups = dict(manifest.get("feature_groups") or {})
    expected_counts = {
        "clean_base": 273,
        "nested_compact": 74,
        "fold_safe_formation": 74,
    }
    if {key: len(groups.get(key, [])) for key in expected_counts} != expected_counts:
        raise ValueError("saved exp287 feature-group counts changed")
    parent_features = [
        str(feature)
        for group in ["clean_base", "nested_compact", "fold_safe_formation"]
        for feature in groups[group]
    ]
    if len(parent_features) != 421 or len(set(parent_features)) != 421:
        raise ValueError("saved exp287 parent feature schema is not 421 unique columns")
    forbidden_overlap = sorted(
        set(parent_features).intersection(FEATURE_COLUMNS_27 + DEPENDENT_GRWR_SIX)
    )
    if forbidden_overlap:
        raise ValueError(f"forbidden historical columns survived exp287: {forbidden_overlap}")
    formation_manifest = json.loads(fixed_files["formation_manifest"][0].read_text())
    if int(formation_manifest.get("partition_count", -1)) != 10:
        raise ValueError("saved exp287 formation manifest must contain 10 fold-role caches")
    return manifest, {
        "root": str(root),
        "file_sha256": actual,
        "parent_feature_count": len(parent_features),
        "parent_feature_schema_sha256": str(manifest["feature_schema_sha256"]),
        "historical_27_overlap": [],
        "dependent_grwr_six_overlap": [],
    }


def load_exp287_fold_contract(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    actual_sha = verify_file_sha(path, expected_sha256, "saved exp287 OOF")
    columns = ["id", "well", "outer_fold", "actual_tvt"]
    frame = pd.read_parquet(path, columns=columns)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError("saved exp287 OOF ids are duplicated")
    folds = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy(np.int8)
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("saved exp287 OOF fold assignment is incomplete")
    if not np.isfinite(frame["actual_tvt"].to_numpy(np.float32)).all():
        raise ValueError("saved exp287 actual TVT contains non-finite values")
    return frame, {
        "path": str(path),
        "sha256": actual_sha,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "fold_assignment_sha256": logical_identity_sha256(
            frame["id"], frame["well"], frame["outer_fold"]
        ),
    }


def candidate_specs(config: Mapping[str, Any]) -> list[CandidateSpec]:
    specs = [
        CandidateSpec(str(item["name"]), str(item["column"]))
        for item in nested(config, "scorer.candidates")
    ]
    if [spec.name for spec in specs] != CANDIDATE_ORDER:
        raise ValueError("candidate contract differs from the fixed order")
    return specs


def exp111_required_source_columns(config: Mapping[str, Any]) -> list[str]:
    specs = candidate_specs(config)
    columns = ["id", "well", "target"]
    columns.extend(str(value) for value in nested(config, "scorer.exp111_row_context_columns"))
    columns.extend(str(value) for value in nested(config, "scorer.exp111_multiobs_global_columns"))
    columns.extend(spec.column for spec in specs)
    for spec in specs:
        columns.extend(
            [
                f"multiobs_score_{spec.name}",
                f"multiobs_mae_{spec.name}",
                f"multiobs_ncc_{spec.name}",
            ]
        )
    ordered = list(dict.fromkeys(columns))
    if len(ordered) != 37:
        raise ValueError(f"fixed exp111 source projection must contain 37 columns, got {len(ordered)}")
    return ordered


def verify_exp111_contract_files(
    source_path: Path,
    config_path: Path,
    feature_schema_path: Path,
    model_manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data = dict(config["data"])
    source_sha = verify_file_sha(
        source_path,
        str(data["exp111_contract_source_sha256"]),
        "exp111 contract source",
    )
    config_sha = verify_file_sha(
        config_path,
        str(data["exp111_contract_config_sha256"]),
        "exp111 contract config",
    )
    feature_schema_sha = verify_file_sha(
        feature_schema_path,
        str(data["exp111_contract_feature_schema_sha256"]),
        "exp111 row-feature schema",
    )
    model_manifest_sha = verify_file_sha(
        model_manifest_path,
        str(data["exp111_reference_model_manifest_sha256"]),
        "exp111 reference model manifest",
    )
    reference = read_yaml(config_path)
    reference_candidates = [
        (str(item["name"]), str(item["column"]))
        for item in nested(reference, "likelihood.candidates")
    ]
    current_candidates = [
        (str(item["name"]), str(item["column"]))
        for item in nested(config, "scorer.candidates")
    ]
    if reference_candidates != current_candidates:
        raise ValueError("exp111 five-candidate contract changed")
    if list(nested(reference, "likelihood.row_context_columns")) != list(
        nested(config, "scorer.exp111_row_context_columns")
    ):
        raise ValueError("exp111 row-context contract changed")
    if list(nested(reference, "likelihood.multiobs_global_columns")) != list(
        nested(config, "scorer.exp111_multiobs_global_columns")
    ):
        raise ValueError("exp111 multi-observation global contract changed")

    def scientific_params(mapping: Mapping[str, Any]) -> dict[str, Any]:
        excluded = {
            "n_jobs",
            "random_state",
            "verbose",
            "deterministic",
            "force_col_wise",
        }
        return {str(key): value for key, value in mapping.items() if key not in excluded}

    for objective, reference_key, current_key in [
        ("within10", "likelihood.classifier_lgbm.params", "scorer.classifier_lgbm.params"),
        ("expected_error", "likelihood.error_lgbm.params", "scorer.error_lgbm.params"),
    ]:
        if scientific_params(dict(nested(reference, reference_key))) != scientific_params(
            dict(nested(config, current_key))
        ):
            raise ValueError(f"exp111 {objective} scientific hyperparameters changed")
    reference_schema = pd.read_csv(feature_schema_path)
    if list(reference_schema.columns) != ["feature_index", "feature"]:
        raise ValueError("exp111 reference row-feature schema columns changed")
    reference_schema = reference_schema.sort_values("feature_index", kind="stable")
    if reference_schema["feature_index"].tolist() != list(range(32)):
        raise ValueError("exp111 reference row-feature indices changed")
    if reference_schema["feature"].astype(str).tolist() != fixed_row_feature_columns(
        config
    ):
        raise ValueError("exp111 fixed 32-row-feature schema/order changed")
    reference_models = json.loads(model_manifest_path.read_text())
    variants = [
        str(item["variant"]) for item in reference_models.get("models", [])
    ]
    if variants != ["within10_classifier", "expected_error_regressor"]:
        raise ValueError("exp111 reference dual-objective manifest changed")
    return {
        "source": str(source_path),
        "source_sha256": source_sha,
        "config": str(config_path),
        "config_sha256": config_sha,
        "feature_schema": str(feature_schema_path),
        "feature_schema_sha256": feature_schema_sha,
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": model_manifest_sha,
        "saved_model_prediction_use": False,
    }


def validate_source_header(
    cache_path: Path,
    schema_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    header = pd.read_csv(cache_path, nrows=0).columns.astype(str).tolist()
    required = exp111_required_source_columns(config)
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"exp099 cache is missing fixed exp111 columns: {missing}")
    forbidden_present = sorted(PROTECTED_LABEL_COLUMNS.intersection(header) - {"target"})
    if forbidden_present:
        raise ValueError(f"unexpected protected label columns in exp099 cache: {forbidden_present}")
    schema_sha = verify_file_sha(
        schema_path,
        str(nested(config, "data.exp111_reference_feature_schema_sha256")),
        "exp099 feature schema",
    )
    decompressed_sha = sha256_file(cache_path, decompressed=cache_path.suffix == ".gz")
    expected_decompressed = str(
        nested(config, "data.exp111_reference_input_decompressed_sha256")
    )
    if decompressed_sha != expected_decompressed:
        raise ValueError(
            f"exp099 decompressed content SHA mismatch: {decompressed_sha} != "
            f"{expected_decompressed}"
        )
    return {
        "path": str(cache_path),
        "schema_path": str(schema_path),
        "schema_sha256": schema_sha,
        "decompressed_content_sha256": decompressed_sha,
        "source_column_count": len(header),
        "required_column_count": len(required),
        "historical_model_prediction_columns_read": False,
    }


# %% [markdown]
# ## 4. exp111 target-free 48-feature contract
#
# 32 row featuresと16 candidate-long featuresをexp111順で再構築する。
# target-free long builderはtarget/truth/errorを引数に取らない。labelはfit/evaluation直前に
# `candidate_labels`で別生成し、sample、imputation median、outer-valid predictionには使わない。

# %%
def fixed_row_feature_columns(config: Mapping[str, Any]) -> list[str]:
    engineered = [
        "candidate_mean",
        "candidate_std",
        "candidate_range",
        *(f"{candidate}_minus_last" for candidate in CANDIDATE_ORDER),
        *(
            f"{left}_vs_{right}_abs"
            for left_index, left in enumerate(CANDIDATE_ORDER)
            for right in CANDIDATE_ORDER[left_index + 1 :]
        ),
    ]
    columns = [
        *map(str, nested(config, "scorer.exp111_row_context_columns")),
        *map(str, nested(config, "scorer.exp111_multiobs_global_columns")),
        *engineered,
    ]
    if len(columns) != 32 or len(set(columns)) != 32:
        raise ValueError("fixed exp111 row-feature schema must contain 32 unique columns")
    return columns


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def _rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def prepare_target_free_wide_features(
    source: pd.DataFrame,
    specs: Sequence[CandidateSpec],
    *,
    configured_row_columns: Sequence[str],
    configured_global_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    protected_present = sorted(PROTECTED_LABEL_COLUMNS.intersection(source.columns))
    if protected_present:
        raise ValueError(
            "target-free feature builder received protected labels: "
            f"{protected_present}"
        )
    required = [
        "id",
        "well",
        *configured_row_columns,
        *configured_global_columns,
        *(spec.column for spec in specs),
    ]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"target-free source projection is missing: {missing}")
    frame = source.copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in specs
        ]
    )
    if not np.isfinite(candidate_values).all():
        raise ValueError("candidate TVT matrix contains non-finite values")
    value_columns = [spec.column for spec in specs]
    frame["candidate_mean"] = np.mean(candidate_values, axis=1, dtype=np.float32)
    # Match exp111 exactly: pandas row-wise std uses the sample standard
    # deviation (ddof=1), while candidate_z_within_row below uses NumPy ddof=0.
    frame["candidate_std"] = np.std(
        candidate_values,
        axis=1,
        ddof=1,
        dtype=np.float32,
    )
    frame["candidate_range"] = (
        np.max(candidate_values, axis=1) - np.min(candidate_values, axis=1)
    ).astype(np.float32)
    engineered = ["candidate_mean", "candidate_std", "candidate_range"]
    last_known = pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    for spec in specs:
        name = f"{spec.name}_minus_last"
        frame[name] = (
            pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(np.float32)
            - last_known
        ).astype(np.float32)
        engineered.append(name)
    for left_index, left in enumerate(specs):
        for right in specs[left_index + 1 :]:
            name = f"{left.name}_vs_{right.name}_abs"
            frame[name] = np.abs(
                pd.to_numeric(frame[left.column], errors="coerce").to_numpy(np.float32)
                - pd.to_numeric(frame[right.column], errors="coerce").to_numpy(np.float32)
            ).astype(np.float32)
            engineered.append(name)
    row_features = list(
        dict.fromkeys(
            [
                *map(str, configured_row_columns),
                *map(str, configured_global_columns),
                *engineered,
            ]
        )
    )
    if len(row_features) != 32 or len(set(row_features)) != 32:
        raise ValueError(f"exp111 row feature contract must be 32 unique columns: {row_features}")
    if set(row_features).intersection(PROTECTED_LABEL_COLUMNS):
        raise ValueError("protected label leaked into exp111 row feature contract")
    numeric = frame[row_features].replace([np.inf, -np.inf], np.nan)
    if numeric.notna().sum().eq(0).any():
        empty = numeric.columns[numeric.notna().sum().eq(0)].tolist()
        raise ValueError(f"exp111 row feature columns are entirely missing: {empty}")
    # Keep the source candidate columns for candidate-long generation.
    if value_columns != CANDIDATE_ORDER:
        raise ValueError("candidate value columns changed")
    return frame, row_features, candidate_values


def model_feature_columns(row_feature_columns: Sequence[str]) -> list[str]:
    columns = [*map(str, row_feature_columns), *LONG_CANDIDATE_COLUMNS]
    if len(columns) != 48 or len(set(columns)) != 48:
        raise ValueError("exp396 scorer schema must contain 48 unique features")
    if set(columns).intersection(PROTECTED_LABEL_COLUMNS):
        raise ValueError("protected label leaked into 48-feature model schema")
    return columns


def build_target_free_candidate_long(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    specs: Sequence[CandidateSpec],
    *,
    row_feature_columns: Sequence[str],
    candidate_values: np.ndarray,
) -> pd.DataFrame:
    indices = np.asarray(row_indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("candidate-long projection cannot be empty")
    names = [spec.name for spec in specs]
    row_mean = candidate_values.mean(axis=1).astype(np.float32)
    row_std = candidate_values.std(axis=1).astype(np.float32)
    row_std_safe = np.where(row_std > 1.0e-6, row_std, 1.0).astype(np.float32)
    last_known = pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    likpf = pd.to_numeric(frame["likpf_mean"], errors="coerce").to_numpy(np.float32)
    score_matrix = (
        frame[[f"multiobs_score_{name}" for name in names]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1.0e9)
        .to_numpy(np.float32)
    )
    mae_matrix = (
        frame[[f"multiobs_mae_{name}" for name in names]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0e9)
        .to_numpy(np.float32)
    )
    ncc_matrix = (
        frame[[f"multiobs_ncc_{name}" for name in names]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    score_max = score_matrix.max(axis=1)
    score_mean = score_matrix.mean(axis=1)
    mae_min = mae_matrix.min(axis=1)
    ncc_max = ncc_matrix.max(axis=1)
    score_rank = _rank_desc(score_matrix)
    mae_rank = _rank_asc(mae_matrix)
    ncc_rank = _rank_desc(ncc_matrix)
    chunks: list[pd.DataFrame] = []
    for candidate_index, spec in enumerate(specs):
        part = frame.iloc[indices][["id", "well", *row_feature_columns]].copy()
        candidate = candidate_values[indices, candidate_index].astype(np.float32)
        part["candidate_index"] = np.int16(candidate_index)
        part["candidate_name"] = spec.name
        part["candidate_tvt"] = candidate
        part["candidate_minus_last"] = (candidate - last_known[indices]).astype(np.float32)
        part["candidate_abs_minus_likpf"] = np.abs(
            candidate - likpf[indices]
        ).astype(np.float32)
        part["candidate_abs_minus_row_mean"] = np.abs(
            candidate - row_mean[indices]
        ).astype(np.float32)
        part["candidate_z_within_row"] = (
            (candidate - row_mean[indices]) / row_std_safe[indices]
        ).astype(np.float32)
        part["candidate_multiobs_score"] = score_matrix[indices, candidate_index]
        part["candidate_multiobs_mae"] = mae_matrix[indices, candidate_index]
        part["candidate_multiobs_ncc"] = ncc_matrix[indices, candidate_index]
        part["candidate_score_gap_from_best"] = (
            score_max[indices] - score_matrix[indices, candidate_index]
        ).astype(np.float32)
        part["candidate_score_centered"] = (
            score_matrix[indices, candidate_index] - score_mean[indices]
        ).astype(np.float32)
        part["candidate_mae_gap_from_best"] = (
            mae_matrix[indices, candidate_index] - mae_min[indices]
        ).astype(np.float32)
        part["candidate_ncc_gap_from_best"] = (
            ncc_max[indices] - ncc_matrix[indices, candidate_index]
        ).astype(np.float32)
        part["candidate_score_rank"] = score_rank[indices, candidate_index]
        part["candidate_mae_rank"] = mae_rank[indices, candidate_index]
        part["candidate_ncc_rank"] = ncc_rank[indices, candidate_index]
        chunks.append(part)
    long_frame = pd.concat(chunks, ignore_index=True)
    features = model_feature_columns(row_feature_columns)
    if list(long_frame[features].columns) != features:
        raise AssertionError("candidate-long 48-feature order changed")
    return long_frame


def candidate_labels(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(row_indices, dtype=np.int64)
    errors = np.abs(
        candidate_values[indices].astype(np.float32)
        - np.asarray(true_tvt, dtype=np.float32)[indices, None]
    ).astype(np.float32)
    # Candidate-long rows are candidate-major, matching build_target_free_candidate_long.
    flattened_error = errors.T.reshape(-1).astype(np.float32)
    within10 = (flattened_error <= 10.0).astype(np.int8)
    return flattened_error, within10


# %% [markdown]
# ## 5. Stable nested folds, sampling, and imputation
#
# inner foldはouter-trainだけをwell-groupで4分割する。sample前に`well,id`でstable sortし、
# SHA256由来local RNGで最大350,000 wide rowsを選ぶ。同じsampleを2目的で共有する。
# 48-column medianは各inner-trainだけでfitし、objectiveごとに独立した保存file/SHAを持つ。

# %%
def build_inner_fold_assignment(
    ids: pd.Series | Sequence[str],
    wells: pd.Series | Sequence[str],
    *,
    n_splits: int = 4,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame = pd.DataFrame(
        {
            "id": pd.Series(ids, dtype=str).to_numpy(),
            "well": pd.Series(wells, dtype=str).to_numpy(),
        }
    )
    if frame["id"].duplicated().any():
        raise ValueError("inner fold input ids must be unique")
    order = frame.sort_values(["well", "id"], kind="stable").index.to_numpy(np.int64)
    sorted_wells = frame.loc[order, "well"].to_numpy()
    assignment = np.full(len(frame), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=int(n_splits))
    manifest: list[dict[str, Any]] = []
    for inner_fold, (train_position, valid_position) in enumerate(
        splitter.split(order, groups=sorted_wells)
    ):
        train_indices = order[train_position]
        valid_indices = order[valid_position]
        train_wells = set(frame.iloc[train_indices]["well"])
        valid_wells = set(frame.iloc[valid_indices]["well"])
        if train_wells.intersection(valid_wells):
            raise AssertionError("inner train/valid well overlap")
        assignment[valid_indices] = np.int8(inner_fold)
        manifest.append(
            {
                "inner_fold": inner_fold,
                "inner_train_rows": int(len(train_indices)),
                "inner_valid_rows": int(len(valid_indices)),
                "inner_train_wells": int(len(train_wells)),
                "inner_valid_wells": int(len(valid_wells)),
                "well_overlap": 0,
                "inner_train_well_sha256": sha256_json(sorted(train_wells)),
                "inner_valid_well_sha256": sha256_json(sorted(valid_wells)),
                "inner_train_row_sha256": logical_identity_sha256(
                    frame.iloc[train_indices]["id"], frame.iloc[train_indices]["well"]
                ),
                "inner_valid_row_sha256": logical_identity_sha256(
                    frame.iloc[valid_indices]["id"], frame.iloc[valid_indices]["well"]
                ),
            }
        )
    if np.any(assignment < 0) or set(np.unique(assignment).tolist()) != set(range(n_splits)):
        raise AssertionError("inner fold assignment is incomplete")
    return assignment, manifest


def stable_sample_row_indices(
    ids: pd.Series | Sequence[str],
    wells: pd.Series | Sequence[str],
    row_indices: np.ndarray,
    *,
    outer_fold: int,
    inner_fold: int,
    maximum_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    indices = np.asarray(row_indices, dtype=np.int64)
    identity = pd.DataFrame(
        {
            "id": pd.Series(ids, dtype=str).iloc[indices].to_numpy(),
            "well": pd.Series(wells, dtype=str).iloc[indices].to_numpy(),
            "row_index": indices,
        }
    ).sort_values(["well", "id"], kind="stable")
    material = (
        f"exp396|outer={int(outer_fold)}|inner={int(inner_fold)}|candidate_long"
    )
    seed = stable_seed_from_material(material)
    if len(identity) > int(maximum_rows):
        rng = np.random.default_rng(seed)
        chosen_positions = np.sort(
            rng.choice(len(identity), size=int(maximum_rows), replace=False)
        )
        selected = identity.iloc[chosen_positions].copy()
    else:
        selected = identity.copy()
    selected = selected.sort_values(["well", "id"], kind="stable")
    sampled_indices = selected["row_index"].to_numpy(np.int64)
    return sampled_indices, {
        "seed_material": material,
        "seed_uint32": int(seed),
        "source_rows": int(len(identity)),
        "sampled_rows": int(len(sampled_indices)),
        "sample_row_identity_sha256": logical_identity_sha256(
            selected["id"], selected["well"]
        ),
        "sample_shared_between_objectives": True,
        "global_rng_used": False,
    }


def fit_inner_train_medians(
    train_long: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    values = (
        train_long[list(feature_columns)]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(np.float32)
    )
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    if len(medians) != 48 or not np.isfinite(medians).all():
        raise ValueError("model-specific imputation median must be 48 finite values")
    return medians


def apply_fixed_medians(
    long_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    medians: np.ndarray,
) -> np.ndarray:
    values = (
        long_frame[list(feature_columns)]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32, copy=True)
    )
    median_values = np.asarray(medians, dtype=np.float32)
    if median_values.shape != (len(feature_columns),):
        raise ValueError("imputation median shape differs from feature schema")
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(median_values, np.where(bad)[1])
    if not np.isfinite(values).all():
        raise ValueError("imputed model matrix contains non-finite values")
    return values


# %% [markdown]
# ## 6. Fixed 10-core to 27-feature derivation
#
# tieは固定candidate順で解く。probabilityは降順、expected errorは0以上へclip後の昇順。
# entropy、likPF rank、candidate別10列、probability/inverse-error weighted TVT差分を
# 固定順でfloat32化する。weighted TVTは特徴であり、予測やblendには使わない。

# %%
def derive_fixed_27_features(
    *,
    ids: pd.Series | Sequence[str],
    wells: pd.Series | Sequence[str],
    last_known_tvt: np.ndarray,
    likpf_mean_d: np.ndarray,
    candidate_tvt: np.ndarray,
    probability: np.ndarray,
    predicted_error: np.ndarray,
) -> pd.DataFrame:
    probability_values = np.asarray(probability, dtype=np.float32)
    error_values = np.maximum(
        np.asarray(predicted_error, dtype=np.float32), 0.0
    ).astype(np.float32)
    candidate_values = np.asarray(candidate_tvt, dtype=np.float32)
    n_rows = len(probability_values)
    expected_shape = (n_rows, len(CANDIDATE_ORDER))
    if (
        probability_values.shape != expected_shape
        or error_values.shape != expected_shape
        or candidate_values.shape != expected_shape
    ):
        raise ValueError("score/candidate matrices must have shape (rows, 5)")
    if (
        not np.isfinite(probability_values).all()
        or not np.isfinite(error_values).all()
        or not np.isfinite(candidate_values).all()
    ):
        raise ValueError("10-core derivation input contains non-finite values")
    if np.any((probability_values < -1.0e-6) | (probability_values > 1.0 + 1.0e-6)):
        raise ValueError("within10 probability is outside [0, 1]")
    probability_values = np.clip(probability_values, 0.0, 1.0).astype(np.float32)
    last_known = np.asarray(last_known_tvt, dtype=np.float32)
    likpf_delta = np.asarray(likpf_mean_d, dtype=np.float32)
    if last_known.shape != (n_rows,) or likpf_delta.shape != (n_rows,):
        raise ValueError("anchor vectors must have one value per row")

    prob_order = np.argsort(-probability_values, axis=1, kind="stable")
    error_order = np.argsort(error_values, axis=1, kind="stable")
    prob_sorted = np.take_along_axis(probability_values, prob_order, axis=1)
    error_sorted = np.take_along_axis(error_values, error_order, axis=1)
    prob_ranks = _rank_desc(probability_values)
    error_ranks = _rank_asc(error_values)
    likpf_index = CANDIDATE_ORDER.index("likpf_mean")
    clipped_probability = np.clip(probability_values, 1.0e-6, 1.0)
    entropy = -np.sum(
        clipped_probability * np.log(clipped_probability), axis=1
    ).astype(np.float32)

    result = pd.DataFrame(
        {
            "id": pd.Series(ids, dtype=str).reset_index(drop=True),
            "well": pd.Series(wells, dtype=str).reset_index(drop=True),
        }
    )
    if len(result) != n_rows or result["id"].duplicated().any():
        raise ValueError("derived feature identities must be unique and row-aligned")
    result["ll_learned_prob_top1_index"] = prob_order[:, 0].astype(np.float32)
    result["ll_learned_error_top1_index"] = error_order[:, 0].astype(np.float32)
    result["ll_learned_prob_top1_value"] = prob_sorted[:, 0]
    result["ll_learned_prob_top2_value"] = prob_sorted[:, 1]
    result["ll_learned_prob_margin_top1_top2"] = (
        prob_sorted[:, 0] - prob_sorted[:, 1]
    ).astype(np.float32)
    result["ll_learned_prob_entropy"] = entropy
    result["ll_learned_error_top1_value"] = error_sorted[:, 0]
    result["ll_learned_error_top2_value"] = error_sorted[:, 1]
    result["ll_learned_error_margin_top2_top1"] = (
        error_sorted[:, 1] - error_sorted[:, 0]
    ).astype(np.float32)
    result["ll_learned_prob_likpf_rank"] = prob_ranks[:, likpf_index].astype(
        np.float32
    )
    result["ll_learned_error_likpf_rank"] = error_ranks[:, likpf_index].astype(
        np.float32
    )
    result["ll_learned_prob_top3_contains_likpf"] = (
        prob_ranks[:, likpf_index] < 3
    ).astype(np.float32)
    result["ll_learned_error_top3_contains_likpf"] = (
        error_ranks[:, likpf_index] < 3
    ).astype(np.float32)
    for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        result[f"ll_learned_prob_{candidate}"] = probability_values[
            :, candidate_index
        ]
        result[f"ll_learned_pred_abs_error_{candidate}"] = error_values[
            :, candidate_index
        ]

    probability_sum = probability_values.sum(axis=1)
    probability_denominator = np.where(
        probability_sum > 1.0e-6, probability_sum, 1.0
    ).astype(np.float32)
    probability_weighted_tvt = (
        np.sum(candidate_values * probability_values, axis=1)
        / probability_denominator
    ).astype(np.float32)
    inverse_error_weight = 1.0 / np.maximum(error_values, 1.0e-3)
    inverse_error_sum = inverse_error_weight.sum(axis=1)
    error_weighted_tvt = (
        np.sum(candidate_values * inverse_error_weight, axis=1)
        / inverse_error_sum
    ).astype(np.float32)
    likpf_mean_tvt = (last_known + likpf_delta).astype(np.float32)
    result["ll_learned_prob_weighted_tvt_minus_last_known_tvt"] = (
        probability_weighted_tvt - last_known
    ).astype(np.float32)
    result["ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt"] = (
        probability_weighted_tvt - likpf_mean_tvt
    ).astype(np.float32)
    result["ll_learned_error_weighted_tvt_minus_last_known_tvt"] = (
        error_weighted_tvt - last_known
    ).astype(np.float32)
    result["ll_learned_error_weighted_tvt_minus_likpf_mean_tvt"] = (
        error_weighted_tvt - likpf_mean_tvt
    ).astype(np.float32)
    if list(result.columns) != ["id", "well", *FEATURE_COLUMNS_27]:
        raise AssertionError("fixed 27-feature output order changed")
    result[FEATURE_COLUMNS_27] = result[FEATURE_COLUMNS_27].astype(np.float32)
    if not np.isfinite(result[FEATURE_COLUMNS_27].to_numpy(np.float32)).all():
        raise ValueError("derived fixed 27 features contain non-finite values")
    return result


def score_core_frame(
    *,
    ids: pd.Series | Sequence[str],
    wells: pd.Series | Sequence[str],
    probability: np.ndarray,
    predicted_error: np.ndarray,
    downstream_outer_fold: int,
    role: str,
) -> pd.DataFrame:
    if role not in {"train", "valid"}:
        raise ValueError(f"unexpected score role: {role}")
    probability_values = np.asarray(probability, dtype=np.float32)
    error_values = np.maximum(
        np.asarray(predicted_error, dtype=np.float32), 0.0
    ).astype(np.float32)
    if probability_values.shape != error_values.shape or probability_values.shape[1] != 5:
        raise ValueError("score core matrices must be row-aligned and five-candidate")
    output = pd.DataFrame(
        {
            "id": pd.Series(ids, dtype=str).reset_index(drop=True),
            "well": pd.Series(wells, dtype=str).reset_index(drop=True),
            "downstream_outer_fold": np.int8(downstream_outer_fold),
            "role": role,
        }
    )
    if output["id"].duplicated().any():
        raise ValueError("score core partition ids are duplicated")
    for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
        output[f"ll_learned_prob_{candidate}"] = probability_values[
            :, candidate_index
        ]
        output[f"ll_learned_pred_abs_error_{candidate}"] = error_values[
            :, candidate_index
        ]
    if not np.isfinite(output[SCORE_CORE_COLUMNS].to_numpy(np.float32)).all():
        raise ValueError("score core contains non-finite values")
    return output


def score_core_to_matrices(core: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = np.column_stack(
        [
            core[f"ll_learned_prob_{candidate}"].to_numpy(np.float32)
            for candidate in CANDIDATE_ORDER
        ]
    )
    predicted_error = np.column_stack(
        [
            core[f"ll_learned_pred_abs_error_{candidate}"].to_numpy(np.float32)
            for candidate in CANDIDATE_ORDER
        ]
    )
    return probability, predicted_error


def validate_derived_schema_hashes(
    row_feature_columns: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, str]:
    model_schema = model_feature_columns(row_feature_columns)
    hashes = {
        "model_feature_schema_sha256": sha256_json(model_schema),
        "score_core_schema_sha256": sha256_json(SCORE_CORE_COLUMNS),
        "derived_27_schema_sha256": sha256_json(FEATURE_COLUMNS_27),
    }
    expected = {
        "model_feature_schema_sha256": str(
            nested(config, "scorer.model_feature_schema_sha256")
        ),
        "score_core_schema_sha256": str(
            nested(config, "scorer.score_core.schema_sha256")
        ),
        "derived_27_schema_sha256": str(
            nested(config, "scorer.derivation.feature_schema_sha256")
        ),
    }
    if hashes != expected:
        raise ValueError(f"fixed scorer schema SHA mismatch: {hashes} != {expected}")
    return hashes


# %% [markdown]
# ## 7. Stage A zero-booster preflight
#
# preflightは入力SHA、3,783,989行/773 wells、exp287 outer-fold alignment、
# inner well非重複、48/10/27 schema SHA、旧27/GRWR6非混入、実行量40/0-controlを確認する。
# LightGBM fit、model load、prediction、Stage B、submissionは0。

# %%
def align_score_cache_to_parent(
    score_identity: pd.DataFrame,
    parent_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    score = score_identity.copy()
    score["id"] = score["id"].astype(str)
    score["well"] = score["well"].astype(str)
    if score["id"].duplicated().any():
        raise ValueError("exp099 score cache ids are duplicated")
    parent = parent_fold.copy()
    parent["id"] = parent["id"].astype(str)
    parent["well"] = parent["well"].astype(str)
    indexed = parent.set_index("id", drop=False)
    if set(score["id"]) != set(indexed.index):
        raise ValueError("exp099 score-cache ids differ from saved exp287 OOF ids")
    aligned = indexed.loc[score["id"]].reset_index(drop=True)
    if not aligned["well"].equals(score["well"].reset_index(drop=True)):
        raise ValueError("exp099 score-cache wells differ from saved exp287 OOF wells")
    if {"last_known_tvt", "target"}.issubset(score.columns):
        truth = (
            pd.to_numeric(score["last_known_tvt"], errors="coerce").to_numpy(np.float32)
            + pd.to_numeric(score["target"], errors="coerce").to_numpy(np.float32)
        )
        delta = np.abs(aligned["actual_tvt"].to_numpy(np.float32) - truth)
        if not np.isfinite(delta).all() or float(delta.max(initial=0.0)) > 1.0e-4:
            raise ValueError("exp099 target/anchor truth differs from saved exp287 actual TVT")
    result = score.copy()
    result["outer_fold"] = aligned["outer_fold"].to_numpy(np.int8)
    return result, {
        "rows": int(len(result)),
        "wells": int(result["well"].nunique()),
        "duplicate_ids": int(result["id"].duplicated().sum()),
        "row_coverage": 1.0,
        "well_coverage": 1.0,
        "full_coverage": 1.0,
        "fold_assignment_sha256": logical_identity_sha256(
            result["id"], result["well"], result["outer_fold"]
        ),
    }


def build_all_nested_fold_manifests(identity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        outer_train = identity[identity["outer_fold"].ne(outer_fold)].copy()
        outer_valid = identity[identity["outer_fold"].eq(outer_fold)].copy()
        train_wells = set(outer_train["well"].astype(str))
        valid_wells = set(outer_valid["well"].astype(str))
        if train_wells.intersection(valid_wells):
            raise ValueError("exp287 outer train/valid wells overlap")
        inner_assignment, inner_manifest = build_inner_fold_assignment(
            outer_train["id"], outer_train["well"], n_splits=4
        )
        for inner in inner_manifest:
            rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "outer_train_rows": int(len(outer_train)),
                    "outer_valid_rows": int(len(outer_valid)),
                    "outer_train_wells": int(len(train_wells)),
                    "outer_valid_wells": int(len(valid_wells)),
                    "outer_well_overlap": 0,
                    "outer_train_well_sha256": sha256_json(sorted(train_wells)),
                    "outer_valid_well_sha256": sha256_json(sorted(valid_wells)),
                    "inner_assignment_sha256": logical_identity_sha256(
                        outer_train["id"],
                        outer_train["well"],
                        inner_assignment,
                    ),
                    **inner,
                }
            )
    manifest = pd.DataFrame(rows)
    if len(manifest) != 20:
        raise AssertionError("nested fold manifest must contain 5 x 4 = 20 rows")
    if int(manifest[["outer_well_overlap", "well_overlap"]].to_numpy().sum()) != 0:
        raise AssertionError("nested fold manifest contains well overlap")
    return manifest


def synthetic_derivation_contract() -> dict[str, Any]:
    probability = np.asarray(
        [
            [0.8, 0.8, 0.2, 0.1, 0.05],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    error = np.asarray(
        [
            [2.0, 2.0, 5.0, 8.0, 10.0],
            [-1.0, 1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [[100.0, 101.0, 102.0, 103.0, 104.0], [200.0, 201.0, 202.0, 203.0, 204.0]],
        dtype=np.float32,
    )
    result = derive_fixed_27_features(
        ids=["row_0", "row_1"],
        wells=["well_0", "well_1"],
        last_known_tvt=np.asarray([90.0, 190.0], dtype=np.float32),
        likpf_mean_d=np.asarray([12.0, 12.0], dtype=np.float32),
        candidate_tvt=candidate,
        probability=probability,
        predicted_error=error,
    )
    if result.loc[0, "ll_learned_prob_top1_index"] != 0.0:
        raise AssertionError("probability tie did not preserve fixed candidate order")
    if result.loc[0, "ll_learned_error_top1_index"] != 0.0:
        raise AssertionError("error tie did not preserve fixed candidate order")
    return {
        "rows": int(len(result)),
        "features": len(FEATURE_COLUMNS_27),
        "finite": bool(np.isfinite(result[FEATURE_COLUMNS_27].to_numpy()).all()),
        "tie_break": "fixed_candidate_order",
        "logical_content_sha256": logical_float_frame_sha256(
            result,
            identity_columns=["id", "well"],
            value_columns=FEATURE_COLUMNS_27,
        ),
    }


def run_stage_a_preflight(
    *,
    config: Mapping[str, Any],
    search_roots: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = dict(config["data"])
    parent_required = [
        "fold_safe_formation_oof_predictions.parquet",
        "model_manifest.json",
        "metrics.json",
        "fold_metrics.csv",
        "by_well_metrics.csv",
        "formation_fold_manifest.json",
        "raw_train_current_test_schema_audit.csv",
    ]
    parent_root = resolve_artifact_root(
        [str(value) for value in data["saved_exp287_artifact_patterns"]],
        search_roots,
        required_files=parent_required,
        label="saved exp287 artifact",
    )
    parent_manifest, parent_evidence = verify_parent_artifacts(parent_root, config)
    parent_fold, fold_evidence = load_exp287_fold_contract(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(data["expected_exp287_oof_sha256"]),
    )
    exp264_oof = resolve_existing_path(
        [str(value) for value in data["saved_exp264_oof_patterns"]],
        search_roots,
        label="corrected exp264 OOF",
    )
    exp264_sha = verify_file_sha(
        exp264_oof,
        str(data["expected_exp264_oof_sha256"]),
        "corrected exp264 OOF",
    )
    exp111_source = resolve_existing_path(
        [str(value) for value in data["exp111_contract_source_patterns"]],
        search_roots,
        label="exp111 contract source",
    )
    exp111_config = resolve_existing_path(
        [str(value) for value in data["exp111_contract_config_patterns"]],
        search_roots,
        label="exp111 contract config",
    )
    exp111_feature_schema = resolve_existing_path(
        [str(value) for value in data["exp111_contract_feature_schema_patterns"]],
        search_roots,
        label="exp111 row-feature schema",
    )
    exp111_model_manifest = resolve_existing_path(
        [str(value) for value in data["exp111_contract_model_manifest_patterns"]],
        search_roots,
        label="exp111 reference model manifest",
    )
    exp111_evidence = verify_exp111_contract_files(
        exp111_source,
        exp111_config,
        exp111_feature_schema,
        exp111_model_manifest,
        config,
    )
    cache_path = resolve_existing_path(
        [str(value) for value in data["exp099_train_feature_cache_patterns"]],
        search_roots,
        label="exp099 train feature cache",
    )
    schema_path = resolve_existing_path(
        [str(value) for value in data["exp099_train_feature_schema_patterns"]],
        search_roots,
        label="exp099 train feature schema",
    )
    cache_evidence = validate_source_header(cache_path, schema_path, config)
    identity = pd.read_csv(
        cache_path,
        usecols=["id", "well", "target", "last_known_tvt"],
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    aligned_identity, alignment = align_score_cache_to_parent(identity, parent_fold)
    if int(len(aligned_identity)) != int(nested(config, "validation.expected_rows")):
        raise ValueError("Stage A score row count differs from the fixed contract")
    if int(aligned_identity["well"].nunique()) != int(
        nested(config, "validation.expected_wells")
    ):
        raise ValueError("Stage A well count differs from the fixed contract")
    nested_manifest = build_all_nested_fold_manifests(aligned_identity)
    output_dir.mkdir(parents=True, exist_ok=True)
    nested_manifest_path = output_dir / f"{OUTPUT_PREFIX}_preflight_nested_fold_manifest.csv"
    nested_manifest.to_csv(nested_manifest_path, index=False)

    row_features = fixed_row_feature_columns(config)
    if len(row_features) != 32:
        raise AssertionError("preflight row-feature schema must contain 32 columns")
    schema_hashes = validate_derived_schema_hashes(row_features, config)
    derivation_evidence = synthetic_derivation_contract()
    cost = validate_scientific_contract(config, require_execution_approval=False)
    checks = {
        "exp099_exp287_row_alignment": alignment["full_coverage"] == 1.0,
        "expected_rows": alignment["rows"]
        == int(nested(config, "validation.expected_rows")),
        "expected_wells": alignment["wells"]
        == int(nested(config, "validation.expected_wells")),
        "duplicate_ids_zero": alignment["duplicate_ids"] == 0,
        "outer_inner_well_overlap_zero": int(
            nested_manifest[["outer_well_overlap", "well_overlap"]].to_numpy().sum()
        )
        == 0,
        "nested_manifest_20_rows": len(nested_manifest) == 20,
        "input_features_48": len(model_feature_columns(row_features)) == 48,
        "score_core_10": len(SCORE_CORE_COLUMNS) == 10,
        "derived_features_27": len(FEATURE_COLUMNS_27) == 27,
        "historical_exp111_model_predictions_unused": bool(
            exp111_evidence["saved_model_prediction_use"] is False
        ),
        "historical_27_absent_from_parent": not set(FEATURE_COLUMNS_27).intersection(
            feature
            for group in parent_manifest["feature_groups"].values()
            for feature in group
        ),
        "dependent_grwr_six_absent_from_parent": not set(
            DEPENDENT_GRWR_SIX
        ).intersection(
            feature
            for group in parent_manifest["feature_groups"].values()
            for feature in group
        ),
        "stage_a_40_cpu_boosters": cost["planned_cpu_boosters"] == 40,
        "stage_b_15_gpu_boosters": cost["stage_b_planned_gpu_boosters"] == 15,
        "control_retraining_zero": cost["control_retraining_boosters"] == 0,
        "inference_submission_disabled": not cost["inference"]
        and not cost["submission"],
    }
    manifest = {
        "schema_version": "1.0.0",
        "status": "stage_a_zero_booster_preflight_passed"
        if all(checks.values())
        else "stage_a_zero_booster_preflight_failed",
        "experiment": EXPERIMENT_NAME,
        "cost_contract": cost,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "boosters_trained": 0,
        "predictions_generated": False,
        "submission_generated": False,
        "input": {
            "exp099": cache_evidence,
            "exp111": exp111_evidence,
            "exp287": parent_evidence,
            "exp287_fold": fold_evidence,
            "exp264_oof": {"path": str(exp264_oof), "sha256": exp264_sha},
        },
        "alignment": alignment,
        "schema_sha256": schema_hashes,
        "synthetic_derivation": derivation_evidence,
        "nested_fold_manifest": {
            "path": nested_manifest_path.name,
            "sha256": sha256_file(nested_manifest_path),
            "logical_sha256": logical_identity_sha256(
                nested_manifest["downstream_outer_fold"],
                nested_manifest["inner_fold"],
                nested_manifest["inner_assignment_sha256"],
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "resource_gate": "pending_full_stage_a_run",
    }
    write_json(output_dir / f"{OUTPUT_PREFIX}_preflight_manifest.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError("exp396 Stage A zero-booster preflight failed")
    return manifest


# %% [markdown]
# ## 8. Strict nested scorer training
#
# 各 `(outer, inner)` でsampleとmedianをtarget/error前に固定し、binary/L1を1本ずつfitする。
# inner-valid predictionでouter-train coreを一度だけ埋め、4 modelのouter-valid predictionを
# 単純平均する。model、objective別median、48-feature schema、well/row/sample SHA、
# best iteration、importanceを保存する。

# %%
def load_full_score_cache(
    cache_path: Path,
    *,
    config: Mapping[str, Any],
    parent_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    required = exp111_required_source_columns(config)
    source = pd.read_csv(
        cache_path,
        usecols=required,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    for column in source.columns:
        if column not in {"id", "well"}:
            source[column] = pd.to_numeric(source[column], errors="coerce").astype(
                np.float32
            )
    aligned, alignment = align_score_cache_to_parent(source, parent_fold)
    true_tvt = (
        aligned["last_known_tvt"].to_numpy(np.float32)
        + aligned.pop("target").to_numpy(np.float32)
    ).astype(np.float32)
    if not np.isfinite(true_tvt).all():
        raise ValueError("isolated scorer labels contain non-finite true TVT")
    specs = candidate_specs(config)
    frame, row_features, candidate_values = prepare_target_free_wide_features(
        aligned,
        specs,
        configured_row_columns=list(
            nested(config, "scorer.exp111_row_context_columns")
        ),
        configured_global_columns=list(
            nested(config, "scorer.exp111_multiobs_global_columns")
        ),
    )
    validate_derived_schema_hashes(row_features, config)
    if not np.isfinite(
        frame[["last_known_tvt", "likpf_mean_d"]].to_numpy(np.float32)
    ).all():
        raise ValueError("anchor/likpf delta contains non-finite values")
    if PROTECTED_LABEL_COLUMNS.intersection(frame.columns):
        raise AssertionError("protected labels remain in target-free scorer frame")
    likpf_formula = (
        frame["last_known_tvt"].to_numpy(np.float32)
        + frame["likpf_mean_d"].to_numpy(np.float32)
    )
    likpf_delta = np.abs(
        likpf_formula - frame["likpf_mean"].to_numpy(np.float32)
    )
    if float(likpf_delta.max(initial=0.0)) > 1.0e-4:
        raise ValueError("likpf_mean differs from last_known_tvt + likpf_mean_d")
    return frame, row_features, candidate_values, true_tvt, {
        **alignment,
        "row_feature_count": len(row_features),
        "model_feature_count": len(model_feature_columns(row_features)),
        "likpf_formula_max_abs_error": float(likpf_delta.max(initial=0.0)),
        "target_free_feature_frame": True,
        "labels_isolated_before_feature_build": True,
    }


def save_objective_median(
    model_dir: Path,
    *,
    outer_fold: int,
    inner_fold: int,
    objective: str,
    medians: np.ndarray,
) -> tuple[Path, str]:
    path = (
        model_dir
        / f"{objective}__outer{int(outer_fold)}__inner{int(inner_fold)}__median.npy"
    )
    np.save(path, np.asarray(medians, dtype=np.float32), allow_pickle=False)
    return path, sha256_file(path)


def candidate_prior_from_outer_train(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.abs(
        candidate_values[train_indices].astype(np.float64)
        - np.asarray(true_tvt, dtype=np.float64)[train_indices, None]
    )
    return (
        np.mean(error, axis=0).astype(np.float64),
        np.mean(error <= 10.0, axis=0).astype(np.float64),
    )


def score_quality_sums(
    *,
    actual_error: np.ndarray,
    actual_within10: np.ndarray,
    predicted_error: np.ndarray,
    predicted_probability: np.ndarray,
    prior_error: np.ndarray,
    prior_probability: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(actual_error, dtype=np.float64).reshape(-1)
    within = np.asarray(actual_within10, dtype=np.float64).reshape(-1)
    predicted_e = np.maximum(
        np.asarray(predicted_error, dtype=np.float64).reshape(-1), 0.0
    )
    predicted_p = np.clip(
        np.asarray(predicted_probability, dtype=np.float64).reshape(-1),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    prior_e = np.asarray(prior_error, dtype=np.float64).reshape(-1)
    prior_p = np.clip(
        np.asarray(prior_probability, dtype=np.float64).reshape(-1),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    sizes = {len(value) for value in [error, within, predicted_e, predicted_p, prior_e, prior_p]}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("quality arrays must be nonempty and row-aligned")
    return {
        "candidate_rows": float(len(error)),
        "learned_expected_error_abs_sum": float(np.abs(predicted_e - error).sum()),
        "prior_expected_error_abs_sum": float(np.abs(prior_e - error).sum()),
        "learned_within10_logloss_sum": float(
            (
                -within * np.log(predicted_p)
                - (1.0 - within) * np.log(1.0 - predicted_p)
            ).sum()
        ),
        "prior_within10_logloss_sum": float(
            (
                -within * np.log(prior_p)
                - (1.0 - within) * np.log(1.0 - prior_p)
            ).sum()
        ),
        "learned_within10_brier_sum": float(
            np.square(predicted_p - within).sum()
        ),
        "prior_within10_brier_sum": float(np.square(prior_p - within).sum()),
    }


def quality_row_from_sums(
    sums: Mapping[str, float],
    *,
    outer_fold: int | str,
) -> dict[str, Any]:
    rows = float(sums["candidate_rows"])
    if rows <= 0:
        raise ValueError("quality summary has no candidate rows")
    learned_mae = float(sums["learned_expected_error_abs_sum"]) / rows
    prior_mae = float(sums["prior_expected_error_abs_sum"]) / rows
    learned_logloss = float(sums["learned_within10_logloss_sum"]) / rows
    prior_logloss = float(sums["prior_within10_logloss_sum"]) / rows
    learned_brier = float(sums["learned_within10_brier_sum"]) / rows
    prior_brier = float(sums["prior_within10_brier_sum"]) / rows
    return {
        "outer_fold": outer_fold,
        "candidate_rows": int(rows),
        "learned_expected_error_mae": learned_mae,
        "prior_expected_error_mae": prior_mae,
        "expected_error_mae_delta": learned_mae - prior_mae,
        "expected_error_mae_improved": learned_mae < prior_mae,
        "learned_within10_logloss": learned_logloss,
        "prior_within10_logloss": prior_logloss,
        "within10_logloss_delta": learned_logloss - prior_logloss,
        "within10_logloss_improved": learned_logloss < prior_logloss,
        "learned_within10_brier": learned_brier,
        "prior_within10_brier": prior_brier,
        "within10_brier_delta": learned_brier - prior_brier,
        "within10_brier_improved": learned_brier < prior_brier,
    }


def add_quality_sums(
    accumulator: dict[str, float],
    update: Mapping[str, float],
) -> None:
    for key, value in update.items():
        accumulator[key] = float(accumulator.get(key, 0.0) + float(value))


def fit_one_inner_model_pair(
    *,
    config: Mapping[str, Any],
    outer_fold: int,
    inner_fold: int,
    frame: pd.DataFrame,
    row_features: Sequence[str],
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    fit_indices: np.ndarray,
    inner_valid_indices: np.ndarray,
    outer_valid_long: pd.DataFrame,
    model_dir: Path,
    train_wells: set[str],
    inner_valid_wells: set[str],
    outer_valid_wells: set[str],
    sample_evidence: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    feature_columns = model_feature_columns(row_features)
    train_long = build_target_free_candidate_long(
        frame,
        fit_indices,
        candidate_specs(config),
        row_feature_columns=row_features,
        candidate_values=candidate_values,
    )
    medians = fit_inner_train_medians(train_long, feature_columns)
    x_train = apply_fixed_medians(train_long, feature_columns, medians)
    y_train_error, y_train_within10 = candidate_labels(
        true_tvt, candidate_values, fit_indices
    )
    if len(x_train) != len(y_train_error):
        raise AssertionError("inner train feature/label alignment changed")

    inner_valid_long = build_target_free_candidate_long(
        frame,
        inner_valid_indices,
        candidate_specs(config),
        row_feature_columns=row_features,
        candidate_values=candidate_values,
    )
    x_inner_valid = apply_fixed_medians(
        inner_valid_long, feature_columns, medians
    )
    y_valid_error, y_valid_within10 = candidate_labels(
        true_tvt, candidate_values, inner_valid_indices
    )
    x_outer_valid = apply_fixed_medians(
        outer_valid_long, feature_columns, medians
    )
    if train_wells.intersection(inner_valid_wells):
        raise AssertionError("inner train/valid wells overlap before fit")
    if train_wells.intersection(outer_valid_wells):
        raise AssertionError("outer-valid well leaked into scorer fit")

    threads = int(nested(config, "runtime.stage_a.threads"))
    log_period = int(nested(config, "scorer.log_evaluation_period"))
    classifier_cfg = dict(nested(config, "scorer.classifier_lgbm"))
    classifier_params = dict(classifier_cfg["params"])
    classifier_params.update(
        {
            "n_jobs": threads,
            "random_state": int(nested(config, "validation.seed")),
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    classifier = LGBMClassifier(
        objective=str(classifier_cfg["objective"]),
        **classifier_params,
    )
    classifier.fit(
        x_train,
        y_train_within10,
        eval_set=[(x_inner_valid, y_valid_within10)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(int(classifier_cfg["early_stopping_rounds"])),
            log_evaluation(log_period),
        ],
    )
    inner_probability = classifier.predict_proba(x_inner_valid)[:, 1].astype(
        np.float32
    )
    outer_probability = classifier.predict_proba(x_outer_valid)[:, 1].astype(
        np.float32
    )
    classifier_path = (
        model_dir / f"within10__outer{outer_fold}__inner{inner_fold}.txt"
    )
    classifier.booster_.save_model(str(classifier_path))
    classifier_median_path, classifier_median_sha = save_objective_median(
        model_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        objective="within10",
        medians=medians,
    )

    error_cfg = dict(nested(config, "scorer.error_lgbm"))
    error_params = dict(error_cfg["params"])
    error_params.update(
        {
            "n_jobs": threads,
            "random_state": int(nested(config, "validation.seed")),
            "deterministic": True,
            "force_col_wise": True,
        }
    )
    error_model = LGBMRegressor(
        objective=str(error_cfg["objective"]),
        **error_params,
    )
    error_model.fit(
        x_train,
        y_train_error,
        eval_set=[(x_inner_valid, y_valid_error)],
        eval_metric="l1",
        callbacks=[
            early_stopping(int(error_cfg["early_stopping_rounds"])),
            log_evaluation(log_period),
        ],
    )
    inner_error = np.maximum(
        error_model.predict(x_inner_valid).astype(np.float32), 0.0
    )
    outer_error = np.maximum(
        error_model.predict(x_outer_valid).astype(np.float32), 0.0
    )
    error_path = (
        model_dir / f"expected_error__outer{outer_fold}__inner{inner_fold}.txt"
    )
    error_model.booster_.save_model(str(error_path))
    error_median_path, error_median_sha = save_objective_median(
        model_dir,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        objective="expected_error",
        medians=medians,
    )

    schema_sha = sha256_json(feature_columns)
    common_manifest = {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "input_feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_schema_sha256": schema_sha,
        "inner_train_well_count": len(train_wells),
        "inner_valid_well_count": len(inner_valid_wells),
        "outer_valid_well_count": len(outer_valid_wells),
        "inner_train_well_sha256": sha256_json(sorted(train_wells)),
        "inner_valid_well_sha256": sha256_json(sorted(inner_valid_wells)),
        "outer_valid_well_sha256": sha256_json(sorted(outer_valid_wells)),
        "inner_train_inner_valid_well_overlap": 0,
        "inner_train_outer_valid_well_overlap": 0,
        "sample_row_identity_sha256": str(
            sample_evidence["sample_row_identity_sha256"]
        ),
        "sample_seed_material": str(sample_evidence["seed_material"]),
        "sample_seed_uint32": int(sample_evidence["seed_uint32"]),
        "sampled_wide_rows": int(sample_evidence["sampled_rows"]),
    }
    model_rows = [
        {
            **common_manifest,
            "objective": "within10_classifier",
            "path": str(classifier_path.relative_to(model_dir.parent)),
            "sha256": sha256_file(classifier_path),
            "best_iteration": int(
                classifier.best_iteration_ or classifier.n_estimators
            ),
            "median_path": str(
                classifier_median_path.relative_to(model_dir.parent)
            ),
            "median_sha256": classifier_median_sha,
            "median_feature_count": len(medians),
        },
        {
            **common_manifest,
            "objective": "expected_error_regressor",
            "path": str(error_path.relative_to(model_dir.parent)),
            "sha256": sha256_file(error_path),
            "best_iteration": int(
                error_model.best_iteration_ or error_model.n_estimators
            ),
            "median_path": str(error_median_path.relative_to(model_dir.parent)),
            "median_sha256": error_median_sha,
            "median_feature_count": len(medians),
        },
    ]
    importance_rows: list[dict[str, Any]] = []
    for objective, model in [
        ("within10_classifier", classifier),
        ("expected_error_regressor", error_model),
    ]:
        for importance_type in ["gain", "split"]:
            values = model.booster_.feature_importance(
                importance_type=importance_type
            )
            for feature, importance in zip(feature_columns, values, strict=True):
                importance_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "objective": objective,
                        "importance_type": importance_type,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    del (
        train_long,
        inner_valid_long,
        x_train,
        x_inner_valid,
        x_outer_valid,
        classifier,
        error_model,
    )
    gc.collect()
    return (
        inner_probability,
        inner_error,
        outer_probability,
        outer_error,
        model_rows,
        importance_rows,
    )


def run_strict_nested_stage_a(
    *,
    config: Mapping[str, Any],
    search_roots: Sequence[Path],
    output_dir: Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    data = dict(config["data"])
    parent_root = Path(str(preflight["input"]["exp287"]["root"]))
    parent_fold, parent_fold_evidence = load_exp287_fold_contract(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(data["expected_exp287_oof_sha256"]),
    )
    cache_path = Path(str(preflight["input"]["exp099"]["path"]))
    (
        frame,
        row_features,
        candidate_values,
        true_tvt,
        source_evidence,
    ) = load_full_score_cache(cache_path, config=config, parent_fold=parent_fold)
    feature_columns = model_feature_columns(row_features)
    model_dir = output_dir / "stage_a_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    score_root = output_dir / "stage_a_score_core"
    score_root.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(config)
    maximum_sample_rows = int(
        nested(config, "scorer.sampling.max_train_rows_per_inner_fit")
    )

    model_manifest_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_manifest_rows: list[dict[str, Any]] = []
    partition_manifest_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    pooled_quality_sums: dict[str, float] = {}

    for outer_fold in range(5):
        outer_train_indices = np.flatnonzero(
            frame["outer_fold"].to_numpy(np.int8) != outer_fold
        )
        outer_valid_indices = np.flatnonzero(
            frame["outer_fold"].to_numpy(np.int8) == outer_fold
        )
        outer_train_wells = set(
            frame.iloc[outer_train_indices]["well"].astype(str)
        )
        outer_valid_wells = set(
            frame.iloc[outer_valid_indices]["well"].astype(str)
        )
        if outer_train_wells.intersection(outer_valid_wells):
            raise AssertionError("outer train/valid well overlap before Stage A fit")
        assignment, inner_manifest = build_inner_fold_assignment(
            frame.iloc[outer_train_indices]["id"],
            frame.iloc[outer_train_indices]["well"],
            n_splits=4,
        )
        outer_train_probability = np.full(
            (len(outer_train_indices), 5), np.nan, dtype=np.float32
        )
        outer_train_error = np.full(
            (len(outer_train_indices), 5), np.nan, dtype=np.float32
        )
        outer_valid_probability_sum = np.zeros(
            (len(outer_valid_indices), 5), dtype=np.float64
        )
        outer_valid_error_sum = np.zeros(
            (len(outer_valid_indices), 5), dtype=np.float64
        )
        outer_valid_long = build_target_free_candidate_long(
            frame,
            outer_valid_indices,
            specs,
            row_feature_columns=row_features,
            candidate_values=candidate_values,
        )
        for inner_fold in range(4):
            inner_train_local = np.flatnonzero(assignment != inner_fold)
            inner_valid_local = np.flatnonzero(assignment == inner_fold)
            inner_train_indices = outer_train_indices[inner_train_local]
            inner_valid_indices = outer_train_indices[inner_valid_local]
            inner_train_wells = set(
                frame.iloc[inner_train_indices]["well"].astype(str)
            )
            inner_valid_wells = set(
                frame.iloc[inner_valid_indices]["well"].astype(str)
            )
            sampled_indices, sample_evidence = stable_sample_row_indices(
                frame["id"],
                frame["well"],
                inner_train_indices,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                maximum_rows=maximum_sample_rows,
            )
            (
                inner_probability_flat,
                inner_error_flat,
                outer_probability_flat,
                outer_error_flat,
                pair_model_rows,
                pair_importance_rows,
            ) = fit_one_inner_model_pair(
                config=config,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                frame=frame,
                row_features=row_features,
                candidate_values=candidate_values,
                true_tvt=true_tvt,
                fit_indices=sampled_indices,
                inner_valid_indices=inner_valid_indices,
                outer_valid_long=outer_valid_long,
                model_dir=model_dir,
                train_wells=inner_train_wells,
                inner_valid_wells=inner_valid_wells,
                outer_valid_wells=outer_valid_wells,
                sample_evidence=sample_evidence,
            )
            inner_probability = inner_probability_flat.reshape(
                5, len(inner_valid_indices)
            ).T
            inner_error = inner_error_flat.reshape(5, len(inner_valid_indices)).T
            outer_probability = outer_probability_flat.reshape(
                5, len(outer_valid_indices)
            ).T
            outer_error = outer_error_flat.reshape(
                5, len(outer_valid_indices)
            ).T
            outer_train_probability[inner_valid_local] = inner_probability
            outer_train_error[inner_valid_local] = inner_error
            outer_valid_probability_sum += outer_probability
            outer_valid_error_sum += outer_error
            model_manifest_rows.extend(pair_model_rows)
            importance_rows.extend(pair_importance_rows)
            inner_row = dict(inner_manifest[inner_fold])
            fold_manifest_rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "outer_train_rows": len(outer_train_indices),
                    "outer_valid_rows": len(outer_valid_indices),
                    "outer_train_wells": len(outer_train_wells),
                    "outer_valid_wells": len(outer_valid_wells),
                    "outer_well_overlap": 0,
                    **inner_row,
                    **sample_evidence,
                }
            )
        del outer_valid_long
        gc.collect()
        outer_valid_probability = (
            outer_valid_probability_sum / 4.0
        ).astype(np.float32)
        outer_valid_error = np.maximum(
            outer_valid_error_sum / 4.0, 0.0
        ).astype(np.float32)
        if not np.isfinite(outer_train_probability).all() or not np.isfinite(
            outer_train_error
        ).all():
            raise ValueError("outer-train inner OOF score coverage is incomplete")
        if not np.isfinite(outer_valid_probability).all() or not np.isfinite(
            outer_valid_error
        ).all():
            raise ValueError("outer-valid four-model score coverage is incomplete")

        role_values = [
            (
                "train",
                outer_train_indices,
                outer_train_probability,
                outer_train_error,
            ),
            (
                "valid",
                outer_valid_indices,
                outer_valid_probability,
                outer_valid_error,
            ),
        ]
        for role, indices, probability, predicted_error in role_values:
            core = score_core_frame(
                ids=frame.iloc[indices]["id"],
                wells=frame.iloc[indices]["well"],
                probability=probability,
                predicted_error=predicted_error,
                downstream_outer_fold=outer_fold,
                role=role,
            )
            role_dir = score_root / f"downstream_outer_fold={outer_fold}"
            role_dir.mkdir(parents=True, exist_ok=True)
            path = role_dir / f"role={role}.parquet"
            core.to_parquet(path, index=False)
            derived = derive_fixed_27_features(
                ids=core["id"],
                wells=core["well"],
                last_known_tvt=frame.iloc[indices][
                    "last_known_tvt"
                ].to_numpy(np.float32),
                likpf_mean_d=frame.iloc[indices]["likpf_mean_d"].to_numpy(
                    np.float32
                ),
                candidate_tvt=candidate_values[indices],
                probability=probability,
                predicted_error=predicted_error,
            )
            partition_manifest_rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "role": role,
                    "path": str(path.relative_to(output_dir)),
                    "rows": len(core),
                    "wells": int(core["well"].nunique()),
                    "duplicate_ids": int(core["id"].duplicated().sum()),
                    "score_core_count": len(SCORE_CORE_COLUMNS),
                    "derived_feature_count": len(FEATURE_COLUMNS_27),
                    "file_sha256": sha256_file(path),
                    "row_identity_sha256": logical_identity_sha256(
                        core["id"], core["well"]
                    ),
                    "score_core_logical_sha256": logical_float_frame_sha256(
                        core,
                        identity_columns=["id", "well"],
                        value_columns=SCORE_CORE_COLUMNS,
                    ),
                    "derived_27_logical_sha256": logical_float_frame_sha256(
                        derived,
                        identity_columns=["id", "well"],
                        value_columns=FEATURE_COLUMNS_27,
                    ),
                }
            )
            del core, derived
            gc.collect()

        prior_error_by_candidate, prior_probability_by_candidate = (
            candidate_prior_from_outer_train(
                true_tvt, candidate_values, outer_train_indices
            )
        )
        valid_actual_error = np.abs(
            candidate_values[outer_valid_indices].astype(np.float64)
            - true_tvt[outer_valid_indices, None].astype(np.float64)
        )
        valid_within10 = (valid_actual_error <= 10.0).astype(np.float64)
        prior_error = np.broadcast_to(
            prior_error_by_candidate[None, :], valid_actual_error.shape
        )
        prior_probability = np.broadcast_to(
            prior_probability_by_candidate[None, :], valid_actual_error.shape
        )
        quality_sums = score_quality_sums(
            actual_error=valid_actual_error,
            actual_within10=valid_within10,
            predicted_error=outer_valid_error,
            predicted_probability=outer_valid_probability,
            prior_error=prior_error,
            prior_probability=prior_probability,
        )
        quality_rows.append(
            quality_row_from_sums(quality_sums, outer_fold=outer_fold)
        )
        add_quality_sums(pooled_quality_sums, quality_sums)
        del (
            outer_train_probability,
            outer_train_error,
            outer_valid_probability,
            outer_valid_error,
            outer_valid_probability_sum,
            outer_valid_error_sum,
            valid_actual_error,
            valid_within10,
        )
        gc.collect()

    quality_rows.append(
        quality_row_from_sums(pooled_quality_sums, outer_fold="pooled")
    )
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_a_40_cpu_boosters_completed",
        "model_count": len(model_manifest_rows),
        "median_vector_count": len(model_manifest_rows),
        "schema_record_count": len(model_manifest_rows),
        "input_feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_schema_sha256": sha256_json(feature_columns),
        "objectives": ["within10_classifier", "expected_error_regressor"],
        "models": model_manifest_rows,
        "saved_exp111_model_reuse_count": 0,
        "control_retraining_boosters": 0,
    }
    model_manifest_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_model_manifest.json"
    write_json(model_manifest_path, model_manifest)
    fold_manifest = pd.DataFrame(fold_manifest_rows)
    fold_manifest_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_fold_manifest.csv"
    fold_manifest.to_csv(fold_manifest_path, index=False)
    partition_manifest = pd.DataFrame(partition_manifest_rows)
    partition_manifest_path = (
        output_dir / f"{OUTPUT_PREFIX}_stage_a_score_partition_manifest.csv"
    )
    partition_manifest.to_csv(partition_manifest_path, index=False)
    importance = pd.DataFrame(importance_rows)
    importance_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    quality = pd.DataFrame(quality_rows)
    quality_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_score_quality.csv"
    quality.to_csv(quality_path, index=False)
    return {
        "model_manifest": model_manifest,
        "model_manifest_path": model_manifest_path,
        "fold_manifest": fold_manifest,
        "fold_manifest_path": fold_manifest_path,
        "partition_manifest": partition_manifest,
        "partition_manifest_path": partition_manifest_path,
        "importance": importance,
        "importance_path": importance_path,
        "quality": quality,
        "quality_path": quality_path,
        "source_evidence": source_evidence,
        "parent_fold_evidence": parent_fold_evidence,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
    }


# %% [markdown]
# ## 9. Stage A quality and resource gates
#
# technical/leakage/resourceと3 scorer-quality指標を全ANDする。qualityはouter-valid 5 foldsを
# 一度ずつ連結し、各candidateのouter-train priorと比較する。閾値、candidate、model、
# feature subsetは結果後に変更しない。PASSしてもStage B実装/15 GPU boostersへ自動進行しない。

# %%
def evaluate_stage_a_gate(
    result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_manifest = dict(result["model_manifest"])
    fold_manifest = pd.DataFrame(result["fold_manifest"])
    partitions = pd.DataFrame(result["partition_manifest"])
    quality = pd.DataFrame(result["quality"])
    expected_rows = int(nested(config, "validation.expected_rows"))
    expected_wells = int(nested(config, "validation.expected_wells"))
    per_outer_rows = partitions.groupby("downstream_outer_fold")["rows"].sum()
    per_outer_wells = partitions.groupby("downstream_outer_fold")["wells"].sum()
    technical_checks = {
        "outer_inner_well_overlap_zero": int(
            fold_manifest[["outer_well_overlap", "well_overlap"]]
            .to_numpy(np.int64)
            .sum()
        )
        == 0,
        "fold_manifest_20_rows": len(fold_manifest) == 20,
        "all_outer_row_coverage": len(per_outer_rows) == 5
        and bool(per_outer_rows.eq(expected_rows).all()),
        "all_outer_well_coverage": len(per_outer_wells) == 5
        and bool(per_outer_wells.eq(expected_wells).all()),
        "score_partition_count_10": len(partitions) == 10,
        "score_partition_duplicate_ids_zero": bool(
            partitions["duplicate_ids"].eq(0).all()
        ),
        "score_core_count_10": bool(partitions["score_core_count"].eq(10).all()),
        "derived_feature_count_27": bool(
            partitions["derived_feature_count"].eq(27).all()
        ),
        "model_count_40": int(model_manifest["model_count"]) == 40,
        "median_vector_count_40": int(model_manifest["median_vector_count"]) == 40,
        "schema_record_count_40": int(model_manifest["schema_record_count"]) == 40,
        "input_feature_count_48": int(model_manifest["input_feature_count"]) == 48,
        "every_model_schema_48": all(
            int(item["input_feature_count"]) == 48
            and len(item["feature_columns"]) == 48
            for item in model_manifest["models"]
        ),
        "every_outer_valid_model_excludes_outer_valid_wells": all(
            int(item["inner_train_outer_valid_well_overlap"]) == 0
            for item in model_manifest["models"]
        ),
        "sample_shared_between_objectives": bool(
            fold_manifest["sample_shared_between_objectives"].all()
        ),
        "global_rng_unused_for_sample": bool(
            (~fold_manifest["global_rng_used"]).all()
        ),
        "feature_imputation_sample_target_free_before_label_join": bool(
            result["source_evidence"]["target_free_feature_frame"]
            and result["source_evidence"]["labels_isolated_before_feature_build"]
        ),
        "score_and_derived_content_sha_recorded": bool(
            partitions[
                ["score_core_logical_sha256", "derived_27_logical_sha256"]
            ]
            .apply(lambda column: column.astype(str).str.len().eq(64).all())
            .all()
        ),
        "measured_runtime_within_30600_seconds": float(result["runtime_seconds"])
        <= float(nested(config, "runtime.stage_a.maximum_projected_runtime_seconds")),
        "peak_rss_within_25_gb": float(result["peak_rss_gb"])
        <= float(nested(config, "runtime.stage_a.maximum_peak_rss_gb")),
        "saved_exp111_model_reuse_zero": int(
            model_manifest["saved_exp111_model_reuse_count"]
        )
        == 0,
        "control_retraining_zero": int(
            model_manifest["control_retraining_boosters"]
        )
        == 0,
    }
    fold_quality = quality[quality["outer_fold"].astype(str).ne("pooled")].copy()
    pooled = quality[quality["outer_fold"].astype(str).eq("pooled")]
    if len(fold_quality) != 5 or len(pooled) != 1:
        raise ValueError("Stage A quality table must contain five folds plus pooled")
    pooled_row = pooled.iloc[0]
    quality_checks = {
        "expected_error_mae_pooled_improved": bool(
            pooled_row["expected_error_mae_improved"]
        ),
        "expected_error_mae_at_least_4_of_5": int(
            fold_quality["expected_error_mae_improved"].sum()
        )
        >= 4,
        "within10_logloss_pooled_improved": bool(
            pooled_row["within10_logloss_improved"]
        ),
        "within10_logloss_at_least_4_of_5": int(
            fold_quality["within10_logloss_improved"].sum()
        )
        >= 4,
        "within10_brier_pooled_improved": bool(
            pooled_row["within10_brier_improved"]
        ),
        "within10_brier_at_least_4_of_5": int(
            fold_quality["within10_brier_improved"].sum()
        )
        >= 4,
    }
    return {
        "technical": {
            "checks": technical_checks,
            "passed": bool(all(technical_checks.values())),
        },
        "score_quality": {
            "checks": quality_checks,
            "passed": bool(all(quality_checks.values())),
            "pooled": to_jsonable(pooled_row.to_dict()),
            "improved_fold_counts": {
                "expected_error_mae": int(
                    fold_quality["expected_error_mae_improved"].sum()
                ),
                "within10_logloss": int(
                    fold_quality["within10_logloss_improved"].sum()
                ),
                "within10_brier": int(
                    fold_quality["within10_brier_improved"].sum()
                ),
            },
        },
        "passed": bool(
            all(technical_checks.values()) and all(quality_checks.values())
        ),
        "stage_b_implementation_authorized": False,
        "stage_b_gpu_run_authorized": False,
        "inference_authorized": False,
        "submission_authorized": False,
    }


def finalize_stage_a_outputs(
    result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    gate = evaluate_stage_a_gate(result, config=config)
    gate_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_gate.json"
    write_json(gate_path, gate)
    paths = {
        "preflight_manifest": output_dir
        / f"{OUTPUT_PREFIX}_preflight_manifest.json",
        "preflight_nested_fold_manifest": output_dir
        / f"{OUTPUT_PREFIX}_preflight_nested_fold_manifest.csv",
        "stage_a_model_manifest": Path(result["model_manifest_path"]),
        "stage_a_fold_manifest": Path(result["fold_manifest_path"]),
        "stage_a_score_partition_manifest": Path(
            result["partition_manifest_path"]
        ),
        "stage_a_feature_importance": Path(result["importance_path"]),
        "stage_a_score_quality": Path(result["quality_path"]),
        "stage_a_gate": gate_path,
    }
    artifact_sha = {
        name: sha256_file(path) for name, path in paths.items() if path.is_file()
    }
    summary = {
        "schema_version": "1.0.0",
        "status": "stage_a_complete_all_gates_passed_stage_b_approval_pending"
        if gate["passed"]
        else "stage_a_complete_gate_failed_closed",
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "rows": int(nested(config, "validation.expected_rows")),
        "wells": int(nested(config, "validation.expected_wells")),
        "cost_contract": validate_scientific_contract(
            config, require_execution_approval=False
        ),
        "completed_cpu_boosters": int(result["model_manifest"]["model_count"]),
        "saved_exp111_model_reuse_count": 0,
        "control_retraining_boosters": 0,
        "score_core_partitions": len(result["partition_manifest"]),
        "score_core_count": len(SCORE_CORE_COLUMNS),
        "derived_feature_count": len(FEATURE_COLUMNS_27),
        "runtime_seconds": float(result["runtime_seconds"]),
        "peak_rss_gb": float(result["peak_rss_gb"]),
        "gate": gate,
        "artifact_sha256": artifact_sha,
        "prediction_generated": False,
        "submission_generated": False,
        "stage_b_started": False,
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_stage_a_summary.json"
    write_json(summary_path, summary)
    reproducibility = {
        "schema_version": "1.0.0",
        "status": summary["status"],
        "input": preflight["input"],
        "preflight_sha256": artifact_sha["preflight_manifest"],
        "parent_fold": result["parent_fold_evidence"],
        "score_source": result["source_evidence"],
        "model_manifest_sha256": artifact_sha["stage_a_model_manifest"],
        "fold_manifest_sha256": artifact_sha["stage_a_fold_manifest"],
        "score_partition_manifest_sha256": artifact_sha[
            "stage_a_score_partition_manifest"
        ],
        "feature_importance_sha256": artifact_sha["stage_a_feature_importance"],
        "score_quality_sha256": artifact_sha["stage_a_score_quality"],
        "gate_sha256": artifact_sha["stage_a_gate"],
        "model_count": int(result["model_manifest"]["model_count"]),
        "median_vector_count": int(
            result["model_manifest"]["median_vector_count"]
        ),
        "schema_record_count": int(result["model_manifest"]["schema_record_count"]),
        "score_partition_file_sha256": {
            f"outer{int(row.downstream_outer_fold)}_{row.role}": str(
                row.file_sha256
            )
            for row in result["partition_manifest"].itertuples(index=False)
        },
        "score_partition_logical_sha256": {
            f"outer{int(row.downstream_outer_fold)}_{row.role}": str(
                row.score_core_logical_sha256
            )
            for row in result["partition_manifest"].itertuples(index=False)
        },
        "derived_27_partition_logical_sha256": {
            f"outer{int(row.downstream_outer_fold)}_{row.role}": str(
                row.derived_27_logical_sha256
            )
            for row in result["partition_manifest"].itertuples(index=False)
        },
        "deterministic_anchor": False,
        "rerun_parity_checked": False,
        "gpu_model_count": 0,
        "submission_sha256": None,
    }
    reproducibility_path = (
        output_dir / f"{OUTPUT_PREFIX}_stage_a_reproducibility_manifest.json"
    )
    write_json(reproducibility_path, reproducibility)
    summary["stage_a_summary_sha256"] = sha256_file(summary_path)
    summary["reproducibility_manifest_sha256"] = sha256_file(
        reproducibility_path
    )
    return summary


# %% [markdown]
# ## 10. Stage B frozen-input preflight
#
# Stage A version 2のall-gates PASS、10 score-core partitions、exp287の421列schemaと
# 10 formation partitions、corrected exp264 OOF、Stage C compact、exp218 clean baseを
# fit前にSHA/ID/well/foldで検証する。formationとscore coreは保存済み物理cacheを再利用し、
# 本実験では再生成しない。

# %%
def rmse(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    return float(np.sqrt(np.mean(delta * delta)))


def verify_stage_a_artifacts(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = dict(config["data"])
    names = {
        "summary": (
            f"{OUTPUT_PREFIX}_stage_a_summary.json",
            data["expected_stage_a_summary_sha256"],
        ),
        "gate": (
            f"{OUTPUT_PREFIX}_stage_a_gate.json",
            data["expected_stage_a_gate_sha256"],
        ),
        "model_manifest": (
            f"{OUTPUT_PREFIX}_stage_a_model_manifest.json",
            data["expected_stage_a_model_manifest_sha256"],
        ),
        "score_partition_manifest": (
            f"{OUTPUT_PREFIX}_stage_a_score_partition_manifest.csv",
            data["expected_stage_a_score_partition_manifest_sha256"],
        ),
        "reproducibility_manifest": (
            f"{OUTPUT_PREFIX}_stage_a_reproducibility_manifest.json",
            data["expected_stage_a_reproducibility_manifest_sha256"],
        ),
    }
    actual = {
        name: verify_file_sha(root / filename, expected, f"Stage A {name}")
        for name, (filename, expected) in names.items()
    }
    summary = json.loads((root / names["summary"][0]).read_text())
    gate = json.loads((root / names["gate"][0]).read_text())
    model_manifest = json.loads((root / names["model_manifest"][0]).read_text())
    reproducibility = json.loads(
        (root / names["reproducibility_manifest"][0]).read_text()
    )
    if summary.get("status") != (
        "stage_a_complete_all_gates_passed_stage_b_approval_pending"
    ):
        raise ValueError("Stage A summary is not the fixed all-gates PASS")
    if not bool(gate.get("passed")):
        raise ValueError("Stage A fixed gate did not pass")
    if int(summary.get("completed_cpu_boosters", -1)) != 40:
        raise ValueError("Stage A must contain exactly 40 completed CPU boosters")
    if int(model_manifest.get("model_count", -1)) != 40:
        raise ValueError("Stage A model manifest must contain exactly 40 models")
    if int(model_manifest.get("median_vector_count", -1)) != 40:
        raise ValueError("Stage A model manifest must contain exactly 40 medians")
    if str(reproducibility.get("status")) != str(summary["status"]):
        raise ValueError("Stage A reproducibility status differs from summary")

    manifest_path = root / names["score_partition_manifest"][0]
    partitions = pd.read_csv(manifest_path)
    expected_columns = {
        "downstream_outer_fold",
        "role",
        "path",
        "rows",
        "wells",
        "duplicate_ids",
        "score_core_count",
        "derived_feature_count",
        "file_sha256",
        "row_identity_sha256",
        "score_core_logical_sha256",
        "derived_27_logical_sha256",
    }
    if not expected_columns.issubset(partitions.columns):
        raise ValueError("Stage A score partition manifest schema changed")
    if len(partitions) != 10:
        raise ValueError("Stage A score partition manifest must contain 10 rows")
    if set(partitions["downstream_outer_fold"].astype(int)) != set(range(5)):
        raise ValueError("Stage A score partition outer folds are incomplete")
    if set(partitions["role"].astype(str)) != {"train", "valid"}:
        raise ValueError("Stage A score partition roles are incomplete")
    if int(partitions["duplicate_ids"].sum()) != 0:
        raise ValueError("Stage A score partitions contain duplicate IDs")
    if not partitions["score_core_count"].eq(10).all():
        raise ValueError("Stage A score partitions must contain 10 score columns")
    if not partitions["derived_feature_count"].eq(27).all():
        raise ValueError("Stage A score partitions must derive 27 features")
    physical_sha: dict[str, str] = {}
    for row in partitions.itertuples(index=False):
        path = root / str(row.path)
        key = f"outer{int(row.downstream_outer_fold)}_{row.role}"
        physical_sha[key] = verify_file_sha(
            path, str(row.file_sha256), f"Stage A score partition {key}"
        )
    return partitions, {
        "root": str(root),
        "file_sha256": actual,
        "score_partition_count": len(partitions),
        "score_partition_file_sha256": physical_sha,
        "stage_a_gate_passed": True,
        "completed_cpu_boosters": 40,
        "saved_exp111_model_reuse_count": 0,
    }


def verify_parent_formation_partitions(
    parent_root: Path,
    parent_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = parent_root / "formation_fold_manifest.json"
    formation_manifest = json.loads(path.read_text())
    partitions = pd.DataFrame(formation_manifest.get("partitions") or [])
    if len(partitions) != 10:
        raise ValueError("saved exp287 formation manifest must contain 10 partitions")
    if int(formation_manifest.get("feature_count", -1)) != 74:
        raise ValueError("saved exp287 formation manifest must contain 74 features")
    expected_schema = sha256_json(
        list(parent_manifest["feature_groups"]["fold_safe_formation"])
    )
    if str(formation_manifest.get("feature_schema_sha256")) != expected_schema:
        raise ValueError("saved exp287 formation schema differs from model manifest")
    physical_sha: dict[str, str] = {}
    for row in partitions.itertuples(index=False):
        cache_path = parent_root / str(row.path)
        key = f"outer{int(row.downstream_outer_fold)}_{row.role}"
        physical_sha[key] = verify_file_sha(
            cache_path,
            str(row.file_sha256),
            f"saved exp287 formation partition {key}",
        )
        if bool(row.target_formation_columns_read):
            raise ValueError("saved exp287 formation cache read target formation columns")
        if int(row.correlation_pruned_count) != 0:
            raise ValueError("saved exp287 formation cache used correlation pruning")
    return partitions, {
        "manifest_sha256": sha256_file(path),
        "partition_count": len(partitions),
        "partition_file_sha256": physical_sha,
        "feature_schema_sha256": expected_schema,
        "formation_regenerated": False,
    }


def load_stage_b_candidate_context(
    path: Path,
    *,
    config: Mapping[str, Any],
    parent_fold: pd.DataFrame,
    base_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_decompressed_sha = str(
        nested(config, "data.exp111_reference_input_decompressed_sha256")
    )
    actual_decompressed_sha = sha256_file(path, decompressed=True)
    if actual_decompressed_sha != expected_decompressed_sha:
        raise ValueError(
            "exp099 decompressed content SHA mismatch for Stage B context: "
            f"{actual_decompressed_sha} != {expected_decompressed_sha}"
        )
    columns = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "likpf_mean_d",
        *CANDIDATE_ORDER,
    ]
    source = pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    for column in columns:
        if column not in {"id", "well"}:
            source[column] = pd.to_numeric(
                source[column], errors="coerce"
            ).astype(np.float32)
    aligned, alignment = align_score_cache_to_parent(source, parent_fold)
    indexed = aligned.set_index("id", drop=False)
    base_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(base_ids):
        raise ValueError("Stage B candidate context IDs differ from clean base")
    context = indexed.loc[base_ids].reset_index(drop=True)
    if not context["well"].astype(str).equals(
        base_frame["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("Stage B candidate context wells differ from clean base")
    base_truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    context_truth = (
        context["last_known_tvt"].to_numpy(np.float32)
        + context.pop("target").to_numpy(np.float32)
    ).astype(np.float32)
    if float(np.max(np.abs(base_truth - context_truth), initial=0.0)) > 1.0e-4:
        raise ValueError("Stage B candidate context truth differs from clean base")
    numeric = context[
        ["last_known_tvt", "likpf_mean_d", *CANDIDATE_ORDER]
    ].to_numpy(np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError("Stage B candidate context contains non-finite values")
    likpf_formula = (
        context["last_known_tvt"].to_numpy(np.float32)
        + context["likpf_mean_d"].to_numpy(np.float32)
    )
    formula_error = float(
        np.max(
            np.abs(
                likpf_formula
                - context["likpf_mean"].to_numpy(np.float32)
            ),
            initial=0.0,
        )
    )
    if formula_error > 1.0e-4:
        raise ValueError("Stage B likpf candidate formula changed")
    return context, {
        **alignment,
        "path": str(path),
        "decompressed_content_sha256": actual_decompressed_sha,
        "required_column_count": len(columns),
        "likpf_formula_max_abs_error": formula_error,
        "target_removed_before_27_feature_derivation": True,
    }


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
    if not aligned["well"].astype(str).equals(
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
        raise ValueError(f"formation role is not unique: outer={outer_fold}, role={role}")
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
            raise ValueError("saved formation IDs differ from Stage C compact role")
        frame = indexed.loc[compact_ids].reset_index(drop=True)
    if not frame["well"].astype(str).equals(
        compact["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("saved formation wells differ from Stage C compact role")
    values = frame[list(feature_names)].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("saved formation role contains non-finite values")
    return frame, {
        "path": str(path),
        "file_sha256": str(item["file_sha256"]),
        "logical_content_sha256": logical_sha,
        "rows": len(frame),
        "role": role,
        "outer_fold": outer_fold,
    }


def load_saved_score_role(
    *,
    stage_a_root: Path,
    partition_manifest: pd.DataFrame,
    outer_fold: int,
    role: str,
    compact: pd.DataFrame,
    candidate_context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = partition_manifest[
        partition_manifest["downstream_outer_fold"].astype(int).eq(outer_fold)
        & partition_manifest["role"].astype(str).eq(role)
    ]
    if len(rows) != 1:
        raise ValueError(f"score role is not unique: outer={outer_fold}, role={role}")
    item = rows.iloc[0]
    path = stage_a_root / str(item["path"])
    core = pd.read_parquet(
        path,
        columns=[
            "id",
            "well",
            "downstream_outer_fold",
            "role",
            *SCORE_CORE_COLUMNS,
        ],
    )
    core["id"] = core["id"].astype(str)
    core["well"] = core["well"].astype(str)
    if len(core) != int(item["rows"]) or core["id"].duplicated().any():
        raise ValueError("Stage A score role row/ID contract changed")
    if not core["downstream_outer_fold"].eq(outer_fold).all():
        raise ValueError("Stage A score role outer fold changed")
    if not core["role"].astype(str).eq(role).all():
        raise ValueError("Stage A score role label changed")
    score_logical = logical_float_frame_sha256(
        core,
        identity_columns=["id", "well"],
        value_columns=SCORE_CORE_COLUMNS,
    )
    if score_logical != str(item["score_core_logical_sha256"]):
        raise ValueError("Stage A score-core logical SHA changed")
    context_index = candidate_context
    if context_index.index.name != "id" or not context_index.index.is_unique:
        raise ValueError("Stage B candidate context must use a unique ID index")
    positions = context_index.index.get_indexer(core["id"])
    if np.any(positions < 0):
        raise ValueError("Stage A score IDs are absent from candidate context")
    context = context_index.iloc[positions].reset_index(drop=True)
    if not context["well"].astype(str).equals(core["well"].reset_index(drop=True)):
        raise ValueError("Stage A score wells differ from candidate context")
    probability, predicted_error = score_core_to_matrices(core)
    candidate_tvt = context[CANDIDATE_ORDER].to_numpy(np.float32, copy=False)
    derived = derive_fixed_27_features(
        ids=core["id"],
        wells=core["well"],
        last_known_tvt=context["last_known_tvt"].to_numpy(np.float32),
        likpf_mean_d=context["likpf_mean_d"].to_numpy(np.float32),
        candidate_tvt=candidate_tvt,
        probability=probability,
        predicted_error=predicted_error,
    )
    derived_logical = logical_float_frame_sha256(
        derived,
        identity_columns=["id", "well"],
        value_columns=FEATURE_COLUMNS_27,
    )
    if derived_logical != str(item["derived_27_logical_sha256"]):
        raise ValueError("Stage A derived-27 logical SHA changed")
    compact_ids = compact["id"].astype(str).reset_index(drop=True)
    if not derived["id"].reset_index(drop=True).equals(compact_ids):
        indexed = derived.set_index("id", drop=False)
        if set(indexed.index) != set(compact_ids):
            raise ValueError("Stage A score IDs differ from Stage C compact role")
        derived = indexed.loc[compact_ids].reset_index(drop=True)
    if not derived["well"].astype(str).equals(
        compact["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("Stage A score wells differ from Stage C compact role")
    return derived, {
        "path": str(path),
        "file_sha256": str(item["file_sha256"]),
        "score_core_logical_sha256": score_logical,
        "derived_27_logical_sha256": derived_logical,
        "rows": len(derived),
        "role": role,
        "outer_fold": outer_fold,
    }


def prepare_stage_b_inputs(
    *,
    config: Mapping[str, Any],
    search_roots: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    data = dict(config["data"])
    parent_required = [
        "fold_safe_formation_oof_predictions.parquet",
        "model_manifest.json",
        "metrics.json",
        "fold_metrics.csv",
        "by_well_metrics.csv",
        "formation_fold_manifest.json",
        "raw_train_current_test_schema_audit.csv",
    ]
    parent_root = resolve_artifact_root(
        [str(value) for value in data["saved_exp287_artifact_patterns"]],
        search_roots,
        required_files=parent_required,
        label="saved exp287 artifact",
    )
    parent_manifest, parent_evidence = verify_parent_artifacts(parent_root, config)
    parent_fold, parent_fold_evidence = load_exp287_fold_contract(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(data["expected_exp287_oof_sha256"]),
    )
    formation_partitions, formation_evidence = verify_parent_formation_partitions(
        parent_root, parent_manifest
    )
    stage_a_required = [
        f"{OUTPUT_PREFIX}_stage_a_summary.json",
        f"{OUTPUT_PREFIX}_stage_a_gate.json",
        f"{OUTPUT_PREFIX}_stage_a_model_manifest.json",
        f"{OUTPUT_PREFIX}_stage_a_score_partition_manifest.csv",
        f"{OUTPUT_PREFIX}_stage_a_reproducibility_manifest.json",
    ]
    stage_a_root = resolve_artifact_root(
        [str(value) for value in data["saved_stage_a_artifact_patterns"]],
        search_roots,
        required_files=stage_a_required,
        label="saved exp396 Stage A artifact",
    )
    score_partitions, stage_a_evidence = verify_stage_a_artifacts(
        stage_a_root, config
    )

    stage_c_root = resolve_stage_c_artifact_root(config, search_roots)
    stage_c_evidence = verify_stage_c_artifact_root(stage_c_root, config)
    exp218_source = resolve_existing_path(
        [str(value) for value in data["exp218_source_patterns"]],
        search_roots,
        label="exp218 source",
    )
    exp218_config_path = resolve_existing_path(
        [str(value) for value in data["exp218_config_patterns"]],
        search_roots,
        label="exp218 config",
    )
    clean_allowlist = resolve_existing_path(
        [str(value) for value in data["clean_273_allowlist_patterns"]],
        search_roots,
        label="clean 273 allowlist",
    )
    hidden_assignment = resolve_existing_path(
        [str(value) for value in data["hidden_like_assignment_patterns"]],
        search_roots,
        label="hidden-like assignment",
    )
    hidden_sha = verify_file_sha(
        hidden_assignment,
        str(data["hidden_like_assignment_sha256"]),
        "hidden-like assignment",
    )
    competition_root = find_competition_input_root()
    raw_train_dir = competition_root / "train"
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=clean_allowlist,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    retained = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        *base_features,
    ]
    base_frame = select_unique_columns(
        base_frame,
        retained,
        context="exp396 Stage B clean base",
    )
    parent_groups = parent_manifest["feature_groups"]
    expected_base = [str(value) for value in parent_groups["clean_base"]]
    compact_features = [str(value) for value in parent_groups["nested_compact"]]
    formation_features = [
        str(value) for value in parent_groups["fold_safe_formation"]
    ]
    if base_features != expected_base:
        raise ValueError("Stage B clean 273 schema differs from saved exp287")
    if list(stage_c_evidence["compact_features"]) != compact_features:
        raise ValueError("Stage B compact 74 schema differs from saved exp287")
    parent_features = [*base_features, *compact_features, *formation_features]
    final_features = [*parent_features, *FEATURE_COLUMNS_27]
    if len(parent_features) != 421 or len(final_features) != 448:
        raise ValueError("Stage B 421+27=448 feature count contract changed")
    if len(set(final_features)) != 448:
        raise ValueError("Stage B final feature schema contains duplicates")
    if sha256_json(parent_features) != str(
        data["expected_exp287_feature_schema_sha256"]
    ):
        raise ValueError("Stage B parent logical feature schema changed")
    if sha256_json(final_features) != str(
        nested(config, "model.source_surface.final_feature_schema_sha256")
    ):
        raise ValueError("Stage B final logical feature schema changed")

    parent_oof, parent_oof_evidence = align_oof_to_base(
        parent_root / "fold_safe_formation_oof_predictions.parquet",
        expected_sha256=str(data["expected_exp287_oof_sha256"]),
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
        raise ValueError("saved exp287 OOF RMSE differs from fixed control")
    exp264_path = resolve_existing_path(
        [str(value) for value in data["saved_exp264_oof_patterns"]],
        search_roots,
        label="corrected exp264 OOF",
    )
    exp264_oof, exp264_evidence = align_oof_to_base(
        exp264_path,
        expected_sha256=str(data["expected_exp264_oof_sha256"]),
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
    if not parent_oof["outer_fold"].astype(int).equals(
        exp264_oof["outer_fold"].astype(int)
    ):
        raise ValueError("exp287 and corrected exp264 outer folds differ")
    if not parent_oof["outer_fold"].astype(int).equals(
        parent_fold.set_index("id")
        .loc[base_frame["id"].astype(str), "outer_fold"]
        .reset_index(drop=True)
        .astype(int)
    ):
        raise ValueError("exp287 fold contract changed during Stage B alignment")

    score_cache_path = resolve_existing_path(
        [str(value) for value in data["exp099_train_feature_cache_patterns"]],
        search_roots,
        label="exp099 Stage B candidate context",
    )
    candidate_context, candidate_evidence = load_stage_b_candidate_context(
        score_cache_path,
        config=config,
        parent_fold=parent_fold,
        base_frame=base_frame,
    )
    candidate_context.set_index("id", drop=False, inplace=True)
    mode_name = str(nested(config, "model.source_surface.mode"))
    mode_config = dict(exp218_config["model"]["training"]["modes"][mode_name])
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False),
        mode_config,
    )
    config_indices = [
        int(value)
        for value in nested(config, "model.source_surface.lightgbm_config_indices")
    ]
    params_family = [params_family[index] for index in config_indices]
    parent_params = {
        int(item["config_index"]): item["params"]
        for item in parent_manifest["models"]
        if int(item["outer_fold"]) == 0
    }
    for config_index, params in zip(config_indices, params_family, strict=True):
        if to_jsonable(params) != to_jsonable(parent_params[config_index]):
            raise ValueError(
                f"Stage B LightGBM config {config_index} differs from exp287"
            )

    cost = validate_scientific_contract(
        config, require_execution_approval=False
    )
    checks = {
        "stage_a_all_gates_passed": bool(stage_a_evidence["stage_a_gate_passed"]),
        "stage_a_40_cpu_boosters_complete": int(
            stage_a_evidence["completed_cpu_boosters"]
        )
        == 40,
        "stage_a_score_partitions_10": len(score_partitions) == 10,
        "exp287_formation_partitions_10": len(formation_partitions) == 10,
        "parent_features_421": len(parent_features) == 421,
        "added_features_27": len(FEATURE_COLUMNS_27) == 27,
        "final_features_448": len(final_features) == 448,
        "final_features_unique": len(set(final_features)) == 448,
        "stage_b_gpu_boosters_15": cost["stage_b_planned_gpu_boosters"] == 15,
        "control_retraining_zero": cost["control_retraining_boosters"] == 0,
        "inference_submission_disabled": not cost["inference"]
        and not cost["submission"],
        "saved_exp111_model_reuse_zero": int(
            stage_a_evidence["saved_exp111_model_reuse_count"]
        )
        == 0,
        "formation_regeneration_zero": not bool(
            formation_evidence["formation_regenerated"]
        ),
        "historical_27_absent_from_parent": not set(
            FEATURE_COLUMNS_27
        ).intersection(parent_features),
        "dependent_grwr_six_absent": not set(
            DEPENDENT_GRWR_SIX
        ).intersection(final_features),
    }
    public_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_b_preflight_passed"
        if all(checks.values())
        else "stage_b_preflight_failed",
        "experiment": EXPERIMENT_NAME,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "cost_contract": cost,
        "feature_counts": {
            "clean_base": len(base_features),
            "nested_compact": len(compact_features),
            "fold_safe_formation": len(formation_features),
            "strict_nested_score": len(FEATURE_COLUMNS_27),
            "parent": len(parent_features),
            "final": len(final_features),
        },
        "feature_schema_sha256": {
            "parent_421": sha256_json(parent_features),
            "added_27": sha256_json(FEATURE_COLUMNS_27),
            "final_448": sha256_json(final_features),
        },
        "input": {
            "stage_a": stage_a_evidence,
            "exp287": parent_evidence,
            "exp287_fold": parent_fold_evidence,
            "exp287_formation": formation_evidence,
            "stage_c": {
                "root": stage_c_evidence["root"],
                "sha256": stage_c_evidence["sha256"],
                "partition_count": stage_c_evidence["partition_count"],
                "compact_feature_count": stage_c_evidence[
                    "compact_feature_count"
                ],
                "compact_meta_schema_sha256": stage_c_evidence[
                    "compact_meta_schema_sha256"
                ],
            },
            "clean_base": base_evidence,
            "candidate_context": candidate_evidence,
            "parent_oof": {
                **parent_oof_evidence,
                "rmse": parent_rmse,
            },
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
    preflight_path = output_dir / f"{OUTPUT_PREFIX}_stage_b_preflight_manifest.json"
    write_json(preflight_path, public_manifest)
    if not public_manifest["passed"]:
        raise RuntimeError("exp396 Stage B preflight failed")
    return {
        "manifest": public_manifest,
        "manifest_path": preflight_path,
        "parent_root": parent_root,
        "parent_manifest": parent_manifest,
        "parent_oof": parent_oof,
        "exp264_oof": exp264_oof,
        "stage_a_root": stage_a_root,
        "score_partitions": score_partitions,
        "formation_partitions": formation_partitions,
        "stage_c_root": stage_c_root,
        "stage_c_evidence": stage_c_evidence,
        "base_frame": base_frame,
        "base_features": base_features,
        "compact_features": compact_features,
        "formation_features": formation_features,
        "parent_features": parent_features,
        "final_features": final_features,
        "candidate_context": candidate_context,
        "params_family": params_family,
        "config_indices": config_indices,
        "hidden_assignment": hidden_assignment,
    }


# %% [markdown]
# ## 11. Stage B 448-feature / 15-booster training
#
# 各outer foldでclean 273、Stage C compact 74、保存済みformation 74、
# 保存済みStage A coreから再導出した27列を固定順に連結する。3 configs × 5 foldsだけを
# T4で学習し、exp287/exp264 controlは保存済みOOFを比較に使うだけで再学習しない。

# %%
def evaluate_stage_b_promotion(
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
    for fold in range(5):
        mask = folds == fold
        parent_value = rmse(truth[mask], parent[mask])
        new_value = rmse(truth[mask], new[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "exp287_rmse": parent_value,
                "exp396_rmse": new_value,
                "delta_rmse_exp396_minus_exp287": new_value - parent_value,
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
                "exp396_rmse": new_value,
                "delta_rmse_exp396_minus_exp287": new_value - parent_value,
            }
        )
    scope_metrics = pd.DataFrame(scope_rows)
    well_frame = pd.DataFrame(
        {
            "well": wells,
            "actual_tvt": truth,
            "exp264_pred_tvt": clean_tail,
            "exp287_pred_tvt": parent,
            "exp396_pred_tvt": new,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in well_frame.groupby("well", sort=True):
        exp264_value = rmse(group["actual_tvt"], group["exp264_pred_tvt"])
        exp287_value = rmse(group["actual_tvt"], group["exp287_pred_tvt"])
        exp396_value = rmse(group["actual_tvt"], group["exp396_pred_tvt"])
        well_rows.append(
            {
                "well": well,
                "rows": len(group),
                "exp264_rmse": exp264_value,
                "exp287_rmse": exp287_value,
                "exp396_rmse": exp396_value,
                "exp287_minus_exp264_delta": exp287_value - exp264_value,
                "exp396_minus_exp287_delta": exp396_value - exp287_value,
                "exp396_minus_exp264_delta": exp396_value - exp264_value,
            }
        )
    by_well = pd.DataFrame(well_rows)
    guard_config = dict(nested(config, "guards.stage_b_promotion"))
    pooled_parent = rmse(truth, parent)
    pooled_new = rmse(truth, new)
    pooled_exp264 = rmse(truth, clean_tail)
    pooled_delta = pooled_new - pooled_parent
    nonworse_folds = int(fold_metrics["nonworse_vs_exp287"].sum())
    scope_max_delta = float(
        scope_metrics["delta_rmse_exp396_minus_exp287"].max()
    )
    by_well_p95 = float(
        np.quantile(by_well["exp396_minus_exp287_delta"], 0.95)
    )
    worst_vs_exp264 = float(by_well["exp396_minus_exp264_delta"].max())
    thresholds = dict(
        guard_config["maximum_worsened_well_counts_vs_exp264"]
    )
    worsened_counts: dict[str, dict[str, Any]] = {}
    worsened_checks: list[bool] = []
    for threshold in [1.0, 3.0, 5.0]:
        key = f"plus_{int(threshold)}ft"
        count = int(
            (by_well["exp396_minus_exp264_delta"] > threshold).sum()
        )
        maximum = int(thresholds[key])
        passed = count <= maximum
        worsened_counts[key] = {
            "threshold_ft": threshold,
            "exp396_count": count,
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
        <= float(guard_config["maximum_by_well_delta_p95_vs_exp287"]),
        "worst_well_delta_rmse_vs_exp264": worst_vs_exp264
        <= float(guard_config["maximum_worst_well_delta_rmse_vs_exp264"]),
        "worsened_well_counts_vs_exp264": all(worsened_checks),
    }
    guard = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "exp264_rmse": pooled_exp264,
        "exp287_rmse": pooled_parent,
        "exp396_rmse": pooled_new,
        "delta_rmse_exp396_minus_exp287": pooled_delta,
        "nonworse_folds_vs_exp287": nonworse_folds,
        "maximum_scope_delta_rmse_vs_exp287": scope_max_delta,
        "by_well_delta_p95_vs_exp287": by_well_p95,
        "worst_well_delta_rmse_vs_exp264": worst_vs_exp264,
        "worsened_well_counts_vs_exp264": worsened_counts,
    }
    return guard, fold_metrics, scope_metrics, by_well


def run_stage_b_gpu_train(
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
    candidate_context = inputs["candidate_context"]
    parent_oof = inputs["parent_oof"]
    exp264_oof = inputs["exp264_oof"]
    params_family = list(inputs["params_family"])
    config_indices = list(inputs["config_indices"])
    stage = dict(nested(config, "model.source_surface"))
    n_rows = len(base_frame)
    base_index = pd.Index(base_frame["id"].astype(str))
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    oof_by_config = [
        np.full(n_rows, np.nan, dtype=np.float32) for _ in params_family
    ]
    model_dir = output_dir / "stage_b_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    chunk_columns = int(stage["matrix_copy_chunk_columns"])
    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=inputs["stage_c_root"],
            stage_c_evidence=inputs["stage_c_evidence"],
            downstream_outer_fold=outer_fold,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("Stage B compact IDs are absent from clean base")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Stage B outer train/valid row indices overlap")
        expected_valid_fold = parent_oof.iloc[valid_indices][
            "outer_fold"
        ].to_numpy(np.int8)
        if not np.all(expected_valid_fold == outer_fold):
            raise ValueError("Stage B compact valid role differs from exp287 OOF fold")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("Stage B OOF valid rows were assigned twice")
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
        score_train, score_train_evidence = load_saved_score_role(
            stage_a_root=inputs["stage_a_root"],
            partition_manifest=inputs["score_partitions"],
            outer_fold=outer_fold,
            role="train",
            compact=compact_train,
            candidate_context=candidate_context,
        )
        score_valid, score_valid_evidence = load_saved_score_role(
            stage_a_root=inputs["stage_a_root"],
            partition_manifest=inputs["score_partitions"],
            outer_fold=outer_fold,
            role="valid",
            compact=compact_valid,
            candidate_context=candidate_context,
        )
        for evidence in [
            formation_train_evidence,
            formation_valid_evidence,
            score_train_evidence,
            score_valid_evidence,
        ]:
            component_rows.append(evidence)
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
        score_start = formation_start + len(formation_features)
        x_train_values[:, compact_start:formation_start] = compact_train[
            compact_features
        ].to_numpy(np.float32, copy=False)
        x_valid_values[:, compact_start:formation_start] = compact_valid[
            compact_features
        ].to_numpy(np.float32, copy=False)
        x_train_values[:, formation_start:score_start] = formation_train[
            formation_features
        ].to_numpy(np.float32, copy=False)
        x_valid_values[:, formation_start:score_start] = formation_valid[
            formation_features
        ].to_numpy(np.float32, copy=False)
        x_train_values[:, score_start:] = score_train[
            FEATURE_COLUMNS_27
        ].to_numpy(np.float32, copy=False)
        x_valid_values[:, score_start:] = score_valid[
            FEATURE_COLUMNS_27
        ].to_numpy(np.float32, copy=False)
        if not np.isfinite(x_train_values).all() or not np.isfinite(
            x_valid_values
        ).all():
            raise ValueError("Stage B 448-feature matrix contains non-finite values")
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
                        int(stage["early_stopping_rounds"]), verbose=False
                    ),
                    log_evaluation(int(stage["log_evaluation_period"])),
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
            model_path = (
                model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            )
            model.booster_.save_model(
                str(model_path), num_iteration=best_iteration
            )
            rmse_value = rmse(
                truth[valid_indices], anchor[valid_indices] + prediction
            )
            model_row = {
                "variant": "fold_safe_exp111_score_27_addonly",
                "model": f"lgb{config_index}",
                "config_index": config_index,
                "outer_fold": outer_fold,
                "feature_count": len(final_features),
                "feature_schema_sha256": sha256_json(final_features),
                "best_iteration": best_iteration,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_file(model_path),
                "params": params,
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
                                "strict_nested_score_27"
                                if feature in FEATURE_COLUMNS_27
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
                "status": "stage_b_training_in_progress",
                "completed_gpu_boosters": len(model_rows),
                "planned_gpu_boosters": 15,
                "last_model": model_row,
                "peak_rss_gb": current_peak_rss_gb(),
            }
            write_json(
                output_dir / f"{OUTPUT_PREFIX}_stage_b_progress.json",
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
        mean_prediction = np.mean(
            np.vstack(fold_predictions), axis=0
        ).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_indices),
                "rmse_tvt": rmse(
                    truth[valid_indices],
                    anchor[valid_indices] + mean_prediction,
                ),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            formation_train,
            formation_valid,
            score_train,
            score_valid,
            x_train,
            x_valid,
            x_train_values,
            x_valid_values,
            fold_predictions,
            mean_prediction,
        )
        gc.collect()
    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("Stage B 15-model OOF contract is incomplete")
    if not np.array_equal(
        oof_fold, parent_oof["outer_fold"].to_numpy(np.int8)
    ):
        raise AssertionError("Stage B OOF fold assignment differs from exp287")
    for prediction in oof_by_config:
        if not np.isfinite(prediction).all():
            raise AssertionError("Stage B OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(
        np.float32
    )
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    guard, fold_metrics, scope_metrics, by_well = (
        evaluate_stage_b_promotion(
            config=config,
            base_frame=base_frame,
            parent_oof=parent_oof,
            exp264_oof=exp264_oof,
            new_prediction=mean_prediction,
            hidden_assignment_path=inputs["hidden_assignment"],
        )
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
            f"fold_safe_exp111_score_27_addonly__lgb{config_index}__pred_tvt"
        ] = (anchor + residual).astype(np.float32)
    prediction_frame[
        "fold_safe_exp111_score_27_addonly__lgb_mean__pred_tvt"
    ] = mean_prediction
    paths = {
        "oof": output_dir / f"{OUTPUT_PREFIX}_stage_b_oof_predictions.parquet",
        "fold_metrics": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_fold_metrics.csv",
        "scope_metrics": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_scope_metrics.csv",
        "by_well": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_by_well_metrics.csv",
        "importance": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_feature_importance.csv",
        "component_manifest": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_component_manifest.csv",
        "model_manifest": output_dir
        / f"{OUTPUT_PREFIX}_stage_b_model_manifest.json",
        "metrics": output_dir / f"{OUTPUT_PREFIX}_stage_b_metrics.json",
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
        "status": "stage_b_15_gpu_boosters_completed",
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "feature_groups": {
            "clean_base": base_features,
            "nested_compact": list(inputs["compact_features"]),
            "fold_safe_formation": list(inputs["formation_features"]),
            "strict_nested_score_27": FEATURE_COLUMNS_27,
        },
        "control_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": (
            "stage_b_complete_all_promotion_gates_passed_inference_approval_pending"
            if guard["passed"]
            else "stage_b_complete_promotion_gate_failed_closed"
        ),
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": {
            "parent": 421,
            "added_strict_nested_score": 27,
            "final": 448,
        },
        "completed_gpu_boosters": len(model_rows),
        "control_retraining_boosters": 0,
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
        "gpu_policy": dict(nested(config, "runtime.stage_b")),
        "preflight_manifest_sha256": sha256_file(
            inputs["manifest_path"]
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
        output_dir
        / f"{OUTPUT_PREFIX}_stage_b_reproducibility_manifest.json"
    )
    write_json(reproducibility_path, reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        reproducibility_path
    )
    return metrics


# %% [markdown]
# ## 12. Orchestration, diagnostics, and generated artifacts
#
# `implementation_only`は契約表示だけで終了する。preflight/trainはKaggle private CPU、
# Stage Bはprivate T4、internet offを正とする。Stage A/Stage B train完了時はそれぞれの
# feature importanceと固定gateを表示する。Stage B結果にかかわらずinference/submissionは
# 開始しない。

# %%
def run_experiment() -> dict[str, Any]:
    config = read_yaml(find_config_path())
    stage = str(nested(config, "execution.stage"))
    cost = validate_scientific_contract(
        config,
        require_execution_approval=stage
        in {
            "stage_a_preflight_only",
            "stage_a_cpu_scorer_train",
            "stage_b_preflight_only",
            "stage_b_gpu_tvt_train",
        },
    )
    if stage == "implementation_only":
        return {
            "status": "stage_a_implementation_complete_no_execution",
            "cost_contract": cost,
            "boosters_trained": 0,
            "kaggle_input_read": False,
            "stage_b_started": False,
            "inference_started": False,
            "submission_generated": False,
        }
    if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("Kaggle Notebook execution is authoritative for exp396")
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    search_roots = [KAGGLE_INPUT_ROOT, Path("/tmp"), PACKAGE_DIR]
    if stage.startswith("stage_b_"):
        stage_b_inputs = prepare_stage_b_inputs(
            config=config,
            search_roots=search_roots,
            output_dir=output_dir,
        )
        if stage == "stage_b_preflight_only":
            return stage_b_inputs["manifest"]
        if stage != "stage_b_gpu_tvt_train":
            raise RuntimeError(
                f"unsupported or unauthorized exp396 Stage B stage: {stage}"
            )
        return run_stage_b_gpu_train(
            config=config,
            inputs=stage_b_inputs,
            output_dir=output_dir,
        )
    preflight = run_stage_a_preflight(
        config=config,
        search_roots=search_roots,
        output_dir=output_dir,
    )
    if stage == "stage_a_preflight_only":
        return preflight
    if stage != "stage_a_cpu_scorer_train":
        raise RuntimeError(f"unsupported or unauthorized exp396 stage: {stage}")
    result = run_strict_nested_stage_a(
        config=config,
        search_roots=search_roots,
        output_dir=output_dir,
        preflight=preflight,
    )
    return finalize_stage_a_outputs(
        result,
        config=config,
        preflight=preflight,
        output_dir=output_dir,
    )


# %%
if not IMPORT_ONLY:
    RUN_RESULT = run_experiment()
    display(RUN_RESULT)
    if str(RUN_RESULT.get("status", "")).startswith("stage_a_complete"):
        output_dir = KAGGLE_WORKING_ROOT / "artifacts"
        importance_path = (
            output_dir / f"{OUTPUT_PREFIX}_stage_a_feature_importance.csv"
        )
        importance = pd.read_csv(importance_path)
        mean_gain = (
            importance[importance["importance_type"].eq("gain")]
            .groupby(["objective", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["objective", "importance"], ascending=[True, False])
        )
        display(mean_gain.groupby("objective", group_keys=False).head(30))
        try:
            import matplotlib.pyplot as plt

            plot_frame = (
                mean_gain.groupby("feature", as_index=False)["importance"]
                .mean()
                .sort_values("importance", ascending=False)
                .head(30)
                .sort_values("importance")
            )
            axis = plot_frame.plot.barh(
                x="feature",
                y="importance",
                figsize=(10, 11),
                legend=False,
                title="exp396 Stage A mean gain importance across 40 scorers",
            )
            axis.set_xlabel("mean gain importance")
            plt.tight_layout()
            plt.savefig(
                output_dir / f"{OUTPUT_PREFIX}_stage_a_feature_importance_top30.png",
                dpi=140,
            )
            plt.show()
        except ImportError:
            print("matplotlib is unavailable; feature-importance table was still saved.")
    if str(RUN_RESULT.get("status", "")).startswith("stage_b_complete"):
        output_dir = KAGGLE_WORKING_ROOT / "artifacts"
        importance = pd.read_csv(
            output_dir / f"{OUTPUT_PREFIX}_stage_b_feature_importance.csv"
        )
        mean_gain = (
            importance[importance["importance_type"].eq("gain")]
            .groupby(["feature_group", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(
                ["feature_group", "importance"], ascending=[True, False]
            )
        )
        display(mean_gain.groupby("feature_group", group_keys=False).head(30))
        display(RUN_RESULT["guard"])
    if KAGGLE_WORKING_ROOT.exists():
        artifact_root = KAGGLE_WORKING_ROOT / "artifacts"
        if artifact_root.exists():
            print("Generated files")
            for generated in sorted(artifact_root.rglob("*")):
                if generated.is_file():
                    print(generated.relative_to(artifact_root), generated.stat().st_size)
