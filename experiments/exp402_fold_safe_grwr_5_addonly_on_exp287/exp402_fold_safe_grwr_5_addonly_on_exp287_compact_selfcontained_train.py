# %% [markdown]
# # exp402 fold-safe GRWR-5 add-only on exp287 — train
#
# exp287 の保存済み outer-role formation 74列と、exp072 / raw GR から再生成する
# target-free 成分だけを使い、formation 依存 GRWR 5列を決定論的に作る。
# version 1 と train-roles 一括 retry の長時間終了を受け、Stage 0 は train source、
# outer fold 5 shard、current test、aggregate の CPU run に分割する。
# 科学式・候補・fold・生成物の論理内容は変更しない。
# 15 GPU booster 学習、推論、提出は引き続き別承認とする。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. Notebook-safe configuration and path resolution
# 3. Reproducibility and fixed scientific contract
# 4. Parent artifacts, folds, and availability audit
# 5. Minimal GR/DWT/FFT source-component regeneration
# 6. Deterministic GRWR-5 derivation
# 7. Saved outer-role formation verification
# 8. Raw current-test regeneration
# 9. Zero-booster preflight orchestration
# 10. Setup and configuration
# 11. Execution boundary and generated evidence

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

from src.fold_safe_formation_pipeline import (
    build_current_test_formation_surface,
    canonical_formation_feature_names,
    logical_feature_content_sha256,
)


EXPERIMENT_NAME = "exp402_fold_safe_grwr_5_addonly_on_exp287"
PARENT_EXPERIMENT = "exp287_fold_safe_formation_74_addonly_on_exp264"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP402_IMPORT_ONLY", "0") == "1"
SPLIT_STAGE_PHASES = ("train_source", "train_fold", "current_test", "aggregate")

CANDIDATE_ORDER = [
    "pf_ancc",
    "beam_mean",
    "likpf_mean",
    "sc_ens",
    "hyb",
    "tvt_dense",
    "tvt_densew",
    "tvt_dense50",
]
CANDIDATE_SPECS = [
    ("pf_ancc", "pf_ancc", "absolute"),
    ("beam_mean", "beam_mean_d", "delta_from_last_known_tvt"),
    ("likpf_mean", "likpf_mean_d", "delta_from_last_known_tvt"),
    ("sc_ens", "sc_ens_d", "delta_from_last_known_tvt"),
    ("hyb", "hyb_d", "delta_from_last_known_tvt"),
    ("tvt_dense", "tvt_dense_d", "delta_from_last_known_tvt"),
    ("tvt_densew", "tvt_densew_d", "delta_from_last_known_tvt"),
    ("tvt_dense50", "tvt_dense50_d", "delta_from_last_known_tvt"),
]
CLEAN_CANDIDATE_COLUMNS = [
    "last_known_tvt",
    "pf_ancc",
    "beam_mean_d",
    "likpf_mean_d",
    "sc_ens_d",
    "hyb_d",
]
FORMATION_CANDIDATE_COLUMNS = [
    "tvt_dense_d",
    "tvt_densew_d",
    "tvt_dense50_d",
]
SOURCE_COMPONENT_COLUMNS = [
    "grwr_dwt_detail_energy_ratio_w065",
    "grwr_fft_rotation_energy_ratio",
    "grwr_dwt_approx_minus_raw_default_candidate_ncc",
]
GRWR5_FEATURES = [
    "grwr_candidate_tvt_std",
    "grwr_candidate_tvt_range",
    "grwr_dwt_energy_ratio_w065_x_candidate_std",
    "grwr_fft_rotation_ratio_x_candidate_range",
    "grwr_dwt_minus_raw_ncc_gap_x_candidate_range",
]
FORBIDDEN_INPUT_FEATURES = [
    *GRWR5_FEATURES,
    "grwr_ll_entropy_x_dwt_energy_ratio_w065",
]
EXP111_SCORE_PREFIXES = (
    "ll_learned_",
    "learned_prob_",
    "learned_pred_abs_error_",
    "multiobs_",
)


def current_peak_rss_gb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0**2
        return rss / divisor / 1024.0
    except (ImportError, OSError):
        return float("nan")


# %% [markdown]
# ## 2. Notebook-safe configuration and path resolution
#
# `Path.cwd()` を起点にし、Notebook cell で未定義になる file-relative path は使わない。
# Kaggle input は固定 filename と path token を併用し、曖昧な候補を採用しない。

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    candidates.extend(
        path
        for path in PACKAGE_DIR.rglob("config.yaml")
        if path.parent.name == EXPERIMENT_NAME
    )
    matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        loaded = read_yaml(candidate)
        if get_nested(loaded, "experiment.name") == EXPERIMENT_NAME:
            matches.append(candidate.resolve())
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"exp402 config resolution must be unique, got {matches}"
        )
    return matches[0]


def find_competition_input_root() -> Path:
    local = PACKAGE_DIR / "data" / "raw"
    if (local / "train").is_dir() and (local / "test").is_dir():
        return local
    if KAGGLE_INPUT_ROOT.exists():
        direct = [
            path
            for path in KAGGLE_INPUT_ROOT.iterdir()
            if path.is_dir()
            and (path / "train").is_dir()
            and (path / "test").is_dir()
            and (path / "sample_submission.csv").is_file()
        ]
        if len(direct) == 1:
            return direct[0]
        nested = [
            path.parent
            for path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")
            if (path.parent / "train").is_dir()
            and (path.parent / "test").is_dir()
        ]
        nested = sorted(set(nested))
        if len(nested) == 1:
            return nested[0]
    raise FileNotFoundError("competition raw root could not be resolved uniquely")


def resolve_existing_path(
    patterns: Sequence[str],
    *,
    search_roots: Sequence[Path],
    label: str,
) -> Path:
    for pattern in patterns:
        raw = Path(str(pattern))
        direct = [raw] if raw.is_absolute() else [
            PACKAGE_DIR / raw,
            *[Path(root) / raw for root in search_roots],
        ]
        existing = sorted(
            {
                candidate.resolve()
                for candidate in direct
                if candidate.is_file()
            }
        )
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise FileNotFoundError(f"{label} is ambiguous: {existing}")
    filenames = [Path(str(pattern)).name for pattern in patterns]
    matches: list[Path] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for filename in filenames:
            matches.extend(path.resolve() for path in root.rglob(filename))
    matches = sorted(set(path for path in matches if path.is_file()))
    if len(matches) != 1:
        raise FileNotFoundError(f"{label} resolution expected one file, got {matches}")
    return matches[0]


def resolve_artifact_root(
    patterns: Sequence[str],
    *,
    search_roots: Sequence[Path],
    required_files: Sequence[str],
    label: str,
) -> Path:
    for pattern in patterns:
        raw = Path(str(pattern))
        direct = [raw] if raw.is_absolute() else [
            PACKAGE_DIR / raw,
            *[Path(root) / raw for root in search_roots],
        ]
        for candidate in direct:
            if candidate.is_dir() and all(
                (candidate / filename).is_file() for filename in required_files
            ):
                return candidate.resolve()
    candidates: list[Path] = []
    sentinel = str(required_files[0])
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob(sentinel):
            candidate = path.parent
            if all((candidate / filename).is_file() for filename in required_files):
                candidates.append(candidate.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{label} resolution expected one artifact root, got {candidates}"
        )
    return candidates[0]


# %% [markdown]
# ## 3. Reproducibility and fixed scientific contract
#
# GRWR-5 は id 昇順・宣言列順・little-endian float32 で logical SHA を計算する。
# 実装状態では input を読まず、preflight 実行を要求された場合だけ別承認 flag を確認する。

# %%
def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener: Callable[..., Any]
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_identity_sha256(frame: pd.DataFrame) -> str:
    required = ["id", "well"]
    if not set(required).issubset(frame.columns):
        raise ValueError("identity SHA requires id and well")
    ordered = frame[required].copy()
    ordered["id"] = ordered["id"].astype(str)
    ordered["well"] = ordered["well"].astype(str)
    ordered = ordered.sort_values("id", kind="stable").reset_index(drop=True)
    digest = hashlib.sha256(b"exp402-identity-v1\n")
    for row in ordered.itertuples(index=False):
        for value in (row.id, row.well):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
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


def write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=lambda item: item.item()
            if isinstance(item, np.generic)
            else str(item),
        )
        + "\n"
    )
    return sha256_file(path)


def verify_file_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("exp402 experiment name changed")
    if get_nested(config, "experiment.route") != "ml_model":
        raise ValueError("exp402 route must remain ml_model")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp402 parent must remain exp287")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("exp402 Stage 0 implementation flag is disabled")
    if get_nested(config, "implementation.scope") != (
        "stage_0_zero_booster_preflight_implementation_only"
    ):
        raise ValueError("exp402 implementation scope changed")
    if bool(get_nested(config, "implementation.training_implemented")):
        raise ValueError("exp402 Stage 1 training must remain unimplemented")
    if bool(get_nested(config, "implementation.inference_implemented")):
        raise ValueError("exp402 inference must remain unimplemented")
    if bool(get_nested(config, "implementation.submission_enabled")):
        raise ValueError("exp402 submission must remain disabled")
    canonical_adopted = bool(
        get_nested(config, "implementation.canonical_notebook_adopted")
    )
    package_created = bool(
        get_nested(config, "implementation.kaggle_package_created")
    )
    stage_0_approved = bool(
        get_nested(config, "execution.zero_booster_preflight_run_approved")
    ) and bool(get_nested(config, "execution.run_approved"))
    if canonical_adopted and not stage_0_approved:
        raise ValueError("exp402 canonical adoption lacks Stage 0 approval")
    if package_created and (not canonical_adopted or not stage_0_approved):
        raise ValueError("exp402 Kaggle package lacks canonical/run approval")

    candidate_order = list(get_nested(config, "grwr5.candidate_order") or [])
    specs = [
        (str(item["name"]), str(item["column"]), str(item["kind"]))
        for item in get_nested(config, "grwr5.candidate_tvt_specs")
    ]
    ordered_features = [
        str(item["name"]) for item in get_nested(config, "grwr5.ordered_features")
    ]
    if candidate_order != CANDIDATE_ORDER or specs != CANDIDATE_SPECS:
        raise ValueError("exp402 fixed eight-candidate contract changed")
    if ordered_features != GRWR5_FEATURES:
        raise ValueError("exp402 fixed GRWR-5 schema changed")
    if str(get_nested(config, "grwr5.output_dtype")) != "float32":
        raise ValueError("exp402 output dtype must remain float32")
    if str(get_nested(config, "grwr5.candidate_stack_dtype")) != "float32":
        raise ValueError("exp402 candidate stack dtype must remain float32")
    if int(get_nested(config, "grwr5.std_ddof")) != 0:
        raise ValueError("exp402 standard deviation must use ddof=0")
    if sha256_json(GRWR5_FEATURES) != str(
        get_nested(config, "grwr5.feature_schema_sha256")
    ):
        raise ValueError("exp402 GRWR-5 schema SHA changed")

    surface = dict(get_nested(config, "model.feature_surface"))
    if surface != {
        "clean_273": 273,
        "nested_compact_74": 74,
        "fold_safe_formation_74": 74,
        "fold_safe_grwr_5": 5,
        "final": 426,
    }:
        raise ValueError("exp402 421+5=426 feature surface changed")
    training = dict(get_nested(config, "model.training"))
    variants = list(training["active_variants"])
    configs = [int(value) for value in training["lightgbm_config_indices"]]
    folds = int(training["folds"])
    planned = len(variants) * len(configs) * folds
    if variants != ["fold_safe_grwr_5_addonly"]:
        raise ValueError("exp402 must retain exactly one scientific variant")
    if configs != [0, 1, 2] or folds != 5 or planned != 15:
        raise ValueError("future cost must remain 1 x 3 x 5 = 15 GPU boosters")
    if int(training["planned_gpu_boosters"]) != 15:
        raise ValueError("planned GPU booster count changed")
    if bool(training["control_retraining"]):
        raise ValueError("exp287 and exp264 controls must not be retrained")

    preflight_counts = {
        key: int(get_nested(config, f"preflight.{key}"))
        for key in [
            "booster_count",
            "model_count",
            "prediction_count",
            "submission_count",
        ]
    }
    if set(preflight_counts.values()) != {0}:
        raise ValueError("exp402 Stage 0 must remain zero model/output")
    stage_0_runtime = dict(get_nested(config, "runtime.stage_0"))
    if stage_0_runtime != {
        "accelerator": "cpu",
        "model_count": 0,
        "booster_count": 0,
        "prediction_count": 0,
        "submission_count": 0,
    }:
        raise ValueError("exp402 Stage 0 runtime contract changed")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("exp402 Stage 0 package must not enable a GPU")

    rawtest_runtime = dict(
        get_nested(config, "data.exp072_candidate_context.rawtest_runtime")
    )
    if rawtest_runtime != {
        "n_jobs": 8,
        "pf_seeds": 128,
        "pf_particles": 500,
        "fast": False,
        "use_gpu": "cpu",
    }:
        raise ValueError("exp402 raw current-test replay runtime changed")
    formation = dict(get_nested(config, "formation_generator"))
    expected_formation = {
        "family": "exp072_public_replay_formation_only",
        "formations": ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"],
        "plane_k": 10,
        "dense_samples_per_well": 60,
        "dense_k": 20,
        "dense_nfetch": 5000,
        "query_workers": 1,
        "n_jobs": 8,
        "current_test_reference_policy": "all_train_wells",
        "current_test_target_formation_columns_read": False,
    }
    if formation != expected_formation:
        raise ValueError("exp402 current-test formation generator changed")

    current_stage = str(get_nested(config, "execution.current_stage"))
    if current_stage not in {"implementation_only", "zero_booster_preflight"}:
        raise ValueError(f"unauthorized exp402 stage: {current_stage}")
    default_phase = str(get_nested(config, "execution.stage_0_default_phase"))
    configured_phases = tuple(
        str(value)
        for value in (
            get_nested(config, "execution.stage_0_split_phases") or []
        )
    )
    if default_phase != "train_source":
        raise ValueError("exp402 split Stage 0 default phase must be train_source")
    if configured_phases != SPLIT_STAGE_PHASES:
        raise ValueError("exp402 split Stage 0 phases changed")
    if bool(get_nested(config, "execution.run_train")):
        raise ValueError("exp402 Stage 1 train is not implemented or approved")
    if bool(get_nested(config, "execution.run_inference")) or bool(
        get_nested(config, "execution.create_submission")
    ):
        raise ValueError("exp402 inference and output prediction are disabled")
    if require_run_approval:
        if current_stage != "zero_booster_preflight":
            raise PermissionError("exp402 preflight stage is not selected")
        if not bool(get_nested(config, "preflight.enabled")):
            raise PermissionError("exp402 zero-booster preflight is disabled")
        if not bool(
            get_nested(config, "execution.zero_booster_preflight_run_approved")
        ):
            raise PermissionError("exp402 zero-booster preflight run is not approved")
        if not bool(get_nested(config, "execution.run_preflight")):
            raise PermissionError("exp402 execution.run_preflight must be true")
        if not bool(get_nested(config, "execution.run_approved")):
            raise PermissionError("exp402 execution.run_approved must be true")

    return {
        "stage": current_stage,
        "stage_0_default_phase": default_phase,
        "stage_0_split_phases": list(SPLIT_STAGE_PHASES),
        "candidate_order": CANDIDATE_ORDER,
        "grwr5_features": GRWR5_FEATURES,
        "grwr5_schema_sha256": sha256_json(GRWR5_FEATURES),
        "parent_features": 421,
        "added_features": 5,
        "final_features": 426,
        "current_execution": {
            "models": 0,
            "boosters": 0,
            "predictions": 0,
            "submissions": 0,
        },
        "future_training": {
            "variants": 1,
            "lightgbm_configs": 3,
            "folds": 5,
            "gpu_boosters": 15,
            "control_boosters": 0,
        },
        "stage_0_current_test_regeneration": {
            "test_wells": 3,
            "pf_ancc_well_runs": 3,
            "pf_z_well_runs": 3,
            "beam_paths": 21,
            "likelihood_pf_well_runs": 3,
            "likelihood_pf_seed_well_trajectories": 384,
            "likelihood_pf_particle_starts": 192000,
        },
    }


# %% [markdown]
# ## 4. Parent artifacts, folds, and availability audit
#
# exp287 の OOF / model / metrics / formation manifest と corrected exp264 OOF を
# SHA 固定する。parent 421列の値は読み直さず、model manifest の列順と保存 OOF の
# row / well / fold / actual TVT を正とする。exp072 cache は必要な候補成分だけを読む。

# %%
def verify_parent_artifacts(
    parent_root: Path,
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    formation_roles_to_verify: Sequence[tuple[int, str]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parent_cfg = dict(get_nested(config, "data.parent_exp287"))
    fixed = {
        "oof": (
            parent_root / "fold_safe_formation_oof_predictions.parquet",
            parent_cfg["oof_sha256"],
        ),
        "model_manifest": (
            parent_root / "model_manifest.json",
            parent_cfg["model_manifest_sha256"],
        ),
        "metrics": (
            parent_root / "metrics.json",
            parent_cfg["metrics_sha256"],
        ),
        "fold_metrics": (
            parent_root / "fold_metrics.csv",
            parent_cfg["fold_metrics_sha256"],
        ),
        "by_well": (
            parent_root / "by_well_metrics.csv",
            parent_cfg["by_well_sha256"],
        ),
        "formation_manifest": (
            parent_root / "formation_fold_manifest.json",
            parent_cfg["formation_fold_manifest_sha256"],
        ),
        "raw_schema": (
            parent_root / "raw_train_current_test_schema_audit.csv",
            parent_cfg["raw_schema_audit_sha256"],
        ),
    }
    file_sha = {
        name: verify_file_sha(path, str(expected), f"saved exp287 {name}")
        for name, (path, expected) in fixed.items()
    }
    model_manifest = json.loads(fixed["model_manifest"][0].read_text())
    if int(model_manifest.get("model_count", -1)) != 15:
        raise ValueError("saved exp287 model manifest must contain 15 models")
    if int(model_manifest.get("feature_count", -1)) != 421:
        raise ValueError("saved exp287 model manifest must contain 421 features")
    groups = dict(model_manifest.get("feature_groups") or {})
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
        raise ValueError("saved exp287 parent schema must contain 421 unique columns")
    if sha256_json(parent_features) != str(parent_cfg["feature_schema_sha256"]):
        raise ValueError("saved exp287 logical feature schema changed")
    forbidden_parent = sorted(
        set(parent_features).intersection(FORBIDDEN_INPUT_FEATURES)
    )
    forbidden_parent.extend(
        feature
        for feature in parent_features
        if feature.startswith(EXP111_SCORE_PREFIXES)
    )
    if forbidden_parent:
        raise ValueError(
            f"forbidden GRWR/score columns survived parent 421: {forbidden_parent}"
        )
    schema_frame = pd.DataFrame(
        {
            "feature_index": np.arange(421, dtype=np.int32),
            "feature": parent_features,
            "feature_group": [
                "clean_base"
                if index < 273
                else "nested_compact"
                if index < 347
                else "fold_safe_formation"
                for index in range(421)
            ],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / f"{OUTPUT_PREFIX}_parent_421_schema.csv"
    schema_frame.to_csv(schema_path, index=False)
    expected_schema_file_sha = str(
        get_nested(
            config,
            "data.current_test_reference.reference_feature_schema_sha256",
        )
    )
    schema_file_sha = sha256_file(schema_path)
    if schema_file_sha != expected_schema_file_sha:
        raise ValueError(
            "reconstructed exp287 inference schema file SHA changed: "
            f"{schema_file_sha} != {expected_schema_file_sha}"
        )

    oof = pd.read_parquet(
        fixed["oof"][0],
        columns=["id", "well", "outer_fold", "actual_tvt"],
    )
    oof["id"] = oof["id"].astype(str)
    oof["well"] = oof["well"].astype(str)
    oof["outer_fold"] = pd.to_numeric(
        oof["outer_fold"], errors="raise"
    ).astype(np.int8)
    oof["actual_tvt"] = pd.to_numeric(
        oof["actual_tvt"], errors="raise"
    ).astype(np.float32)
    if oof["id"].duplicated().any():
        raise ValueError("saved exp287 OOF IDs are duplicated")
    if set(oof["outer_fold"].unique().tolist()) != set(range(5)):
        raise ValueError("saved exp287 OOF folds are incomplete")
    if not np.isfinite(oof["actual_tvt"]).all():
        raise ValueError("saved exp287 actual TVT is non-finite")

    formation_manifest = json.loads(fixed["formation_manifest"][0].read_text())
    partitions = pd.DataFrame(formation_manifest.get("partitions") or [])
    if len(partitions) != 10:
        raise ValueError("saved exp287 formation manifest must contain 10 roles")
    if int(formation_manifest.get("feature_count", -1)) != 74:
        raise ValueError("saved exp287 formation manifest feature count changed")
    formation_features = [
        str(value) for value in groups["fold_safe_formation"]
    ]
    if formation_features != canonical_formation_feature_names():
        raise ValueError("saved exp287 formation schema is not canonical")
    if str(formation_manifest.get("feature_schema_sha256")) != sha256_json(
        formation_features
    ):
        raise ValueError("saved exp287 formation manifest schema changed")
    required_manifest_columns = {
        "downstream_outer_fold",
        "role",
        "rows",
        "wells",
        "reference_wells",
        "target_wells_inside_reference",
        "target_wells_self_excluded_from_reference_query",
        "reference_well_sha256",
        "target_well_sha256",
        "path",
        "file_sha256",
        "logical_content_sha256",
        "feature_schema_sha256",
        "correlation_pruned_count",
        "target_formation_columns_read",
    }
    if missing := required_manifest_columns - set(partitions.columns):
        raise ValueError(
            f"saved formation manifest columns missing: {sorted(missing)}"
        )
    if set(partitions["downstream_outer_fold"].astype(int)) != set(range(5)):
        raise ValueError("saved formation manifest folds are incomplete")
    if set(partitions["role"].astype(str)) != {"train", "valid"}:
        raise ValueError("saved formation manifest roles are incomplete")
    if partitions.duplicated(["downstream_outer_fold", "role"]).any():
        raise ValueError("saved formation fold/roles are duplicated")
    if partitions["target_formation_columns_read"].astype(bool).any():
        raise ValueError("saved formation role read target formation columns")
    if not partitions["correlation_pruned_count"].astype(int).eq(0).all():
        raise ValueError("saved formation roles used correlation pruning")
    if formation_roles_to_verify is None:
        selected_roles = {
            (int(row.downstream_outer_fold), str(row.role))
            for row in partitions.itertuples(index=False)
        }
    else:
        selected_roles = {
            (int(outer_fold), str(role))
            for outer_fold, role in formation_roles_to_verify
        }
    valid_roles = {
        (int(row.downstream_outer_fold), str(row.role))
        for row in partitions.itertuples(index=False)
    }
    if not selected_roles.issubset(valid_roles):
        raise ValueError(
            f"requested formation roles are invalid: {sorted(selected_roles - valid_roles)}"
        )
    partition_sha: dict[str, str] = {}
    for row in partitions.itertuples(index=False):
        role_identity = (int(row.downstream_outer_fold), str(row.role))
        if role_identity not in selected_roles:
            continue
        role_path = parent_root / str(row.path)
        key = f"outer{int(row.downstream_outer_fold)}_{row.role}"
        partition_sha[key] = verify_file_sha(
            role_path,
            str(row.file_sha256),
            f"saved exp287 formation role {key}",
        )
    return model_manifest, oof, partitions, {
        "root": str(parent_root),
        "file_sha256": file_sha,
        "partition_file_sha256": partition_sha,
        "partition_sha_scope": sorted(
            f"outer{outer_fold}_{role}" for outer_fold, role in selected_roles
        ),
        "parent_feature_count": 421,
        "parent_feature_schema_sha256": sha256_json(parent_features),
        "parent_schema_file_sha256": schema_file_sha,
        "parent_schema_path": str(schema_path),
        "historical_grwr5_loaded": 0,
        "exp111_score_features_loaded": 0,
    }


def verify_availability_contract(
    audit_path: Path,
    allowlist_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    audit_cfg = dict(get_nested(config, "data.availability_audit"))
    audit_sha = verify_file_sha(
        audit_path,
        str(audit_cfg["file_sha256"]),
        "exp218 availability audit",
    )
    allowlist_sha = verify_file_sha(
        allowlist_path,
        str(audit_cfg["clean_allowlist_sha256"]),
        "clean-273 allowlist",
    )
    audit = pd.read_csv(audit_path, dtype=str).fillna("")
    indexed = audit.set_index("feature")
    for feature in GRWR5_FEATURES:
        row = indexed.loc[feature]
        if str(row["status"]) != "fail" or str(row["dependency"]) != (
            "transitive_dense_formation_candidate"
        ):
            raise ValueError(f"availability reason changed for {feature}")
    entropy = indexed.loc["grwr_ll_entropy_x_dwt_energy_ratio_w065"]
    if str(entropy["dependency"]) != "transitive_exp111_fold0_score":
        raise ValueError("entropy interaction availability reason changed")
    clean = pd.read_csv(allowlist_path, dtype=str).fillna("")
    clean_features = clean["feature"].astype(str).tolist()
    required_clean = [*CLEAN_CANDIDATE_COLUMNS, *SOURCE_COMPONENT_COLUMNS]
    if not set(required_clean).issubset(clean_features):
        raise ValueError("clean-273 allowlist lacks required exp402 components")
    if set(FORBIDDEN_INPUT_FEATURES).intersection(clean_features):
        raise ValueError("clean-273 allowlist contains forbidden GRWR outputs")
    return {
        "audit_path": str(audit_path),
        "audit_sha256": audit_sha,
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": allowlist_sha,
        "required_clean_components": required_clean,
        "historical_grwr5_values_selected": 0,
        "entropy_interaction_selected": 0,
    }


def load_exp072_candidate_context(
    path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_cfg = dict(get_nested(config, "data.exp072_candidate_context"))
    raw_sha = verify_file_sha(
        path,
        str(source_cfg["raw_gzip_sha256"]),
        "exp072 deterministic train cache",
    )
    decompressed_sha = sha256_file(path, decompressed=True)
    columns = ["id", "well", "target", *CLEAN_CANDIDATE_COLUMNS]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    if frame["id"].duplicated().any():
        raise ValueError("exp072 context IDs are duplicated")
    for column in columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype(np.float32)
    numeric = frame[["target", *CLEAN_CANDIDATE_COLUMNS]].to_numpy(
        np.float32, copy=False
    )
    if not np.isfinite(numeric).all():
        raise ValueError("exp072 candidate context contains non-finite values")
    return frame, {
        "path": str(path),
        "raw_gzip_sha256": raw_sha,
        "decompressed_content_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "required_columns": columns,
        "historical_grwr5_columns_read": 0,
        "exp111_score_columns_read": 0,
    }


def align_oof_contracts(
    context: pd.DataFrame,
    parent_oof: pd.DataFrame,
    exp264_oof_path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    control_cfg = dict(get_nested(config, "data.clean_control_exp264"))
    control_sha = verify_file_sha(
        exp264_oof_path,
        str(control_cfg["oof_sha256"]),
        "corrected exp264 OOF",
    )
    exp264 = pd.read_parquet(
        exp264_oof_path,
        columns=["id", "well", "outer_fold", "actual_tvt"],
    )
    for frame in (parent_oof, exp264):
        frame["id"] = frame["id"].astype(str)
        frame["well"] = frame["well"].astype(str)
        frame["outer_fold"] = pd.to_numeric(
            frame["outer_fold"], errors="raise"
        ).astype(np.int8)
        frame["actual_tvt"] = pd.to_numeric(
            frame["actual_tvt"], errors="raise"
        ).astype(np.float32)
        if frame["id"].duplicated().any():
            raise ValueError("control OOF IDs are duplicated")
    parent_index = parent_oof.set_index("id", drop=False)
    context_ids = context["id"].astype(str)
    if set(parent_index.index) != set(context_ids):
        raise ValueError("exp072 and exp287 row identities differ")
    parent = parent_index.loc[context_ids].reset_index(drop=True)
    if not parent["well"].equals(context["well"].reset_index(drop=True)):
        raise ValueError("exp072 and exp287 well identities differ")
    truth = (
        context["last_known_tvt"].to_numpy(np.float32)
        + context["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if float(
        np.max(
            np.abs(truth - parent["actual_tvt"].to_numpy(np.float32)),
            initial=0.0,
        )
    ) > 1.0e-4:
        raise ValueError("exp072 target and exp287 actual TVT differ")

    control_index = exp264.set_index("id", drop=False)
    if set(control_index.index) != set(context_ids):
        raise ValueError("corrected exp264 and exp287 row identities differ")
    control = control_index.loc[context_ids].reset_index(drop=True)
    if not control[["well", "outer_fold"]].equals(
        parent[["well", "outer_fold"]]
    ):
        raise ValueError("corrected exp264 and exp287 well/fold contracts differ")
    if float(
        np.max(
            np.abs(
                control["actual_tvt"].to_numpy(np.float32)
                - parent["actual_tvt"].to_numpy(np.float32)
            ),
            initial=0.0,
        )
    ) > 1.0e-4:
        raise ValueError("corrected exp264 and exp287 actual TVT differ")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(parent) != expected_rows or parent["well"].nunique() != expected_wells:
        raise ValueError("exp402 parent row/well count changed")
    aligned_context = context.drop(columns=["target"]).copy()
    aligned_context["outer_fold"] = parent["outer_fold"].to_numpy(np.int8)
    return aligned_context, control, {
        "rows": len(parent),
        "wells": int(parent["well"].nunique()),
        "folds": sorted(parent["outer_fold"].unique().astype(int).tolist()),
        "row_well_identity_sha256": logical_identity_sha256(parent),
        "exp264_oof_sha256": control_sha,
        "target_removed_before_feature_derivation": True,
        "score_row_coverage": 1.0,
    }


# %% [markdown]
# ## 5. Minimal GR/DWT/FFT source-component regeneration
#
# exp218 の全 GRWR generator は呼ばない。固定された三成分だけを、同じ float32 化、
# db4 approximation、FFT band、likpf default-candidate NCC の演算順で再生成する。
# 候補 spread と entropy interaction はこの段階で一切作らない。

# %%
def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    suffix = ids.astype(str).str.rsplit("_", n=1).str[-1]
    values = pd.to_numeric(suffix, errors="coerce").fillna(-1).to_numpy(np.int32)
    if np.any(values < 0):
        raise ValueError("row IDs must end in a non-negative integer index")
    return values


def _fill_numeric(
    values: pd.Series | np.ndarray,
    fallback: float = 0.0,
) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = (
        series.interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(fallback)
    )
    return filled.to_numpy(np.float32)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray | float,
) -> np.ndarray:
    denom = np.asarray(denominator, dtype=np.float32)
    return (
        numerator / np.maximum(np.abs(denom), np.float32(1.0))
    ).astype(np.float32)


def _wavelet_approximation(
    values: np.ndarray,
    *,
    wavelet_name: str,
    level: int,
    fallback_window: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values64 = np.asarray(values, dtype=np.float64)
    try:
        import pywt

        wavelet = pywt.Wavelet(str(wavelet_name))
        max_level = int(pywt.dwt_max_level(len(values64), wavelet.dec_len))
        used_level = int(min(max(int(level), 1), max_level))
        if used_level < 1:
            raise ValueError("series too short for configured wavelet")
        coeffs = pywt.wavedec(
            values64, wavelet, mode="symmetric", level=used_level
        )
        approx_coeffs = [
            coeffs[0],
            *[np.zeros_like(item) for item in coeffs[1:]],
        ]
        approx = pywt.waverec(
            approx_coeffs, wavelet, mode="symmetric"
        )[: len(values64)]
        return approx.astype(np.float32), {
            "effective_kind": "pywt_dwt_approximation",
            "wavelet": str(wavelet_name),
            "level": used_level,
            "max_level": max_level,
        }
    except Exception as exc:
        window = int(fallback_window)
        return _rolling_mean(values, window), {
            "effective_kind": "rolling_mean_wavelet_fallback",
            "wavelet": str(wavelet_name),
            "level": int(level),
            "window": window,
            "fallback_reason": type(exc).__name__,
        }


def _fft_rotation_summary(
    values: np.ndarray,
    md: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, float]:
    values64 = np.asarray(values, dtype=np.float64)
    md64 = np.asarray(md, dtype=np.float64)
    if len(values64) < 8:
        return {
            "fft_dominant_frequency_norm": 0.0,
            "fft_dominant_energy_ratio": 0.0,
            "fft_rotation_energy_ratio": 0.0,
            "fft_high_frequency_ratio": 0.0,
            "fft_notch_residual_energy_ratio": 0.0,
        }
    x = np.arange(len(values64), dtype=np.float64)
    centered = values64 - float(np.nanmean(values64))
    if (
        str(config.get("detrend", "linear")) == "linear"
        and float(np.nanstd(centered)) > 1.0e-9
    ):
        slope, intercept = np.polyfit(x, values64, deg=1)
        centered = values64 - (slope * x + intercept)
    spacing = (
        float(np.nanmedian(np.diff(md64))) if len(md64) > 1 else 1.0
    )
    if not np.isfinite(spacing) or abs(spacing) <= 1.0e-9:
        spacing = 1.0
    power = np.abs(np.fft.rfft(centered)) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=abs(spacing))
    if len(power) <= 1:
        return {
            "fft_dominant_frequency_norm": 0.0,
            "fft_dominant_energy_ratio": 0.0,
            "fft_rotation_energy_ratio": 0.0,
            "fft_high_frequency_ratio": 0.0,
            "fft_notch_residual_energy_ratio": 0.0,
        }
    valid_power = power[1:]
    total = float(np.sum(valid_power))
    if not np.isfinite(total) or total <= 1.0e-12:
        total = 1.0e-12
    valid_freqs = freqs[1:]
    nyquist = float(np.max(valid_freqs)) if len(valid_freqs) else 1.0
    if not np.isfinite(nyquist) or nyquist <= 1.0e-12:
        nyquist = 1.0
    norm_freqs = valid_freqs / nyquist
    dominant_pos = int(np.argmax(valid_power))
    band = config.get("rotation_band_norm", [0.06, 0.35])
    lo, hi = float(band[0]), float(band[1])
    rotation_mask = (norm_freqs >= lo) & (norm_freqs <= hi)
    high_threshold = float(config.get("high_frequency_norm", 0.35))
    high_mask = norm_freqs >= high_threshold
    top_k = max(int(config.get("top_k_notch", 3)), 1)
    top_energy = float(np.sum(np.sort(valid_power)[-top_k:]))
    return {
        "fft_dominant_frequency_norm": float(norm_freqs[dominant_pos]),
        "fft_dominant_energy_ratio": float(valid_power[dominant_pos] / total),
        "fft_rotation_energy_ratio": float(
            np.sum(valid_power[rotation_mask]) / total
        ),
        "fft_high_frequency_ratio": float(
            np.sum(valid_power[high_mask]) / total
        ),
        "fft_notch_residual_energy_ratio": float(
            max(total - top_energy, 0.0) / total
        ),
    }


def _prefix_slope_prior(
    *,
    md: np.ndarray,
    tvt_input: np.ndarray,
    known_end: int,
    slope_window_rows: int,
    slope_clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    if known_end <= 1:
        raise ValueError("known_end must include at least two prefix rows")
    fit_start = max(0, int(known_end) - int(slope_window_rows))
    fit_md = md[fit_start:known_end].astype(np.float64)
    fit_tvt = tvt_input[fit_start:known_end].astype(np.float64)
    finite = np.isfinite(fit_md) & np.isfinite(fit_tvt)
    if finite.sum() >= 2 and float(np.nanstd(fit_md[finite])) > 1.0e-6:
        slope, intercept = np.polyfit(
            fit_md[finite], fit_tvt[finite], deg=1
        )
    else:
        slope = 1.0
        intercept = float(
            tvt_input[known_end - 1] - md[known_end - 1]
        )
    unclipped_slope = float(slope)
    lo, hi = slope_clip
    slope = float(np.clip(slope, lo, hi))
    last_md = float(md[known_end - 1])
    last_tvt = float(tvt_input[known_end - 1])
    prior = last_tvt + slope * (md.astype(np.float64) - last_md)
    return prior.astype(np.float32), {
        "known_end": int(known_end),
        "fit_start": int(fit_start),
        "fit_rows": int(finite.sum()),
        "unclipped_slope": unclipped_slope,
        "slope": slope,
        "intercept": float(intercept),
        "last_md": last_md,
        "last_tvt": last_tvt,
    }


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1.0e-6
    return centered / scale


def _gather_horizontal(
    series: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    indices = np.clip(
        centers[:, None] + offsets[None, :],
        0,
        len(series) - 1,
    )
    return series[indices].astype(np.float32)


def _interpolate_typewell(
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    candidate_tvt: np.ndarray,
) -> np.ndarray:
    flat = np.interp(
        candidate_tvt.reshape(-1),
        type_tvt.astype(np.float64),
        type_gr.astype(np.float64),
        left=float(type_gr[0]),
        right=float(type_gr[-1]),
    )
    return flat.reshape(candidate_tvt.shape).astype(np.float32)


def _candidate_match_scores(
    *,
    row_idx: np.ndarray,
    md: np.ndarray,
    horizontal_gr: np.ndarray,
    type_tvt: np.ndarray,
    type_gr: np.ndarray,
    candidate_tvt: np.ndarray,
    slope: float,
    offsets: np.ndarray,
    ncc_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_idx = np.asarray(row_idx, dtype=np.int32)
    if row_idx.size == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    eval_gr = _gather_horizontal(horizontal_gr, row_idx, offsets)
    local_rows = np.clip(
        row_idx[:, None] + offsets[None, :],
        0,
        len(md) - 1,
    )
    local_md = md[local_rows].astype(np.float32)
    center_md = md[row_idx].astype(np.float32)[:, None]
    local_tvt = (
        candidate_tvt[:, None].astype(np.float32)
        + float(slope) * (local_md - center_md)
    )
    candidate_gr = _interpolate_typewell(type_tvt, type_gr, local_tvt)
    mae = np.mean(np.abs(candidate_gr - eval_gr), axis=1).astype(np.float32)
    ncc = np.mean(
        _standardize_rows(candidate_gr) * _standardize_rows(eval_gr),
        axis=1,
    ).astype(np.float32)
    cost = (mae - float(ncc_weight) * ncc).astype(np.float32)
    return mae, ncc, cost


def validate_formula_source_contract(
    source_path: Path,
    source_config_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    formula_cfg = dict(get_nested(config, "data.historical_formula"))
    source_sha = verify_file_sha(
        source_path,
        str(formula_cfg["source_sha256"]),
        "exp218 formula source",
    )
    config_sha = verify_file_sha(
        source_config_path,
        str(formula_cfg["source_config_sha256"]),
        "exp218 formula config",
    )
    source_config = read_yaml(source_config_path)
    gr = dict(
        get_nested(
            source_config,
            "model.gr_wavelet_rotation_confidence_features",
        )
        or {}
    )
    expected = {
        "prefix": "grwr_",
        "prefix_slope_window_rows": 80,
        "slope_clip": [-3.0, 3.0],
        "local_windows_rows": [33, 65, 129],
        "candidate_match_offsets_rows": [-32, -24, -16, -8, 0, 8, 16, 24, 32],
        "typewell_smooth_window": 5,
        "ncc_weight": 8.0,
        "default_candidate": "likpf_mean",
    }
    for key, value in expected.items():
        if gr.get(key) != value:
            raise ValueError(f"exp218 fixed formula config changed: {key}")
    if dict(gr["wavelet"]) != {
        "name": "db4",
        "level": 3,
        "fallback_window": 65,
    }:
        raise ValueError("exp218 wavelet config changed")
    fft = dict(gr["fft"])
    if fft.get("rotation_band_norm") != [0.06, 0.35]:
        raise ValueError("exp218 FFT rotation band changed")
    if float(fft.get("high_frequency_norm")) != 0.35:
        raise ValueError("exp218 FFT high-frequency threshold changed")
    return gr, {
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "config_path": str(source_config_path),
        "config_sha256": config_sha,
        "whole_grwr_generator_called": False,
        "selected_source_components": SOURCE_COMPONENT_COLUMNS,
    }


def build_grwr_source_components(
    base_frame: pd.DataFrame,
    *,
    raw_dir: Path,
    gr_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"id", "well", "last_known_tvt", "likpf_mean_d"}
    if missing := required - set(base_frame.columns):
        raise ValueError(f"GR source input columns missing: {sorted(missing)}")
    if set(FORBIDDEN_INPUT_FEATURES).intersection(base_frame.columns):
        raise ValueError("historical GRWR output values were passed as source input")
    if any(
        str(column).startswith(EXP111_SCORE_PREFIXES)
        for column in base_frame.columns
    ):
        raise ValueError("exp111 score columns were passed as source input")
    row_indices = _row_indices_from_ids(base_frame["id"])
    output_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    wavelet_cfg = dict(gr_config["wavelet"])
    fft_cfg = dict(gr_config["fft"])
    offsets = np.asarray(
        gr_config["candidate_match_offsets_rows"], dtype=np.int32
    )
    slope_clip = tuple(float(value) for value in gr_config["slope_clip"])
    for well, positions_raw in base_frame.groupby(
        "well", sort=False
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        query_rows = row_indices[positions].astype(np.int32)
        horizontal_path = Path(raw_dir) / f"{well}__horizontal_well.csv"
        typewell_path = Path(raw_dir) / f"{well}__typewell.csv"
        horizontal = pd.read_csv(
            horizontal_path, usecols=["MD", "GR", "TVT_input"]
        )
        typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
        if np.any(query_rows >= len(horizontal)):
            raise ValueError(f"row identity exceeds raw horizontal length: {well}")
        md = _fill_numeric(horizontal["MD"])
        gr = _fill_numeric(horizontal["GR"])
        tvt_input_raw = pd.to_numeric(
            horizontal["TVT_input"], errors="coerce"
        )
        known = tvt_input_raw.notna().to_numpy()
        if not known.any():
            raise ValueError(f"no finite TVT_input prefix for {well}")
        known_end = int(np.flatnonzero(known)[-1] + 1)
        tvt_input = _fill_numeric(tvt_input_raw)
        _prior, prior_meta = _prefix_slope_prior(
            md=md,
            tvt_input=tvt_input,
            known_end=known_end,
            slope_window_rows=int(gr_config["prefix_slope_window_rows"]),
            slope_clip=(slope_clip[0], slope_clip[1]),
        )
        type_sorted = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
        type_tvt = pd.to_numeric(
            type_sorted["TVT"], errors="coerce"
        ).to_numpy(np.float32)
        type_gr = _fill_numeric(type_sorted["GR"])
        type_gr = _rolling_mean(
            type_gr, int(gr_config["typewell_smooth_window"])
        )
        if len(type_tvt) < 4:
            raise ValueError(f"typewell GR too short for {well}")

        dwt_approx, wavelet_meta = _wavelet_approximation(
            gr,
            wavelet_name=str(wavelet_cfg["name"]),
            level=int(wavelet_cfg["level"]),
            fallback_window=int(wavelet_cfg["fallback_window"]),
        )
        dwt_detail = (gr - dwt_approx).astype(np.float32)
        detail_energy = _rolling_mean(
            np.square(dwt_detail).astype(np.float32), 65
        )
        raw_energy = _rolling_mean(
            np.square(gr - _rolling_mean(gr, 65)).astype(np.float32),
            65,
        )
        dwt_ratio = _safe_divide(
            detail_energy, raw_energy + detail_energy
        )
        fft_meta = _fft_rotation_summary(gr, md, fft_cfg)
        likpf = (
            base_frame["last_known_tvt"].iloc[positions].to_numpy(np.float32)
            + base_frame["likpf_mean_d"].iloc[positions].to_numpy(np.float32)
        ).astype(np.float32)
        _raw_mae, raw_ncc, _raw_cost = _candidate_match_scores(
            row_idx=query_rows,
            md=md,
            horizontal_gr=gr,
            type_tvt=type_tvt,
            type_gr=type_gr,
            candidate_tvt=likpf,
            slope=float(prior_meta["slope"]),
            offsets=offsets,
            ncc_weight=float(gr_config["ncc_weight"]),
        )
        _dwt_mae, dwt_ncc, _dwt_cost = _candidate_match_scores(
            row_idx=query_rows,
            md=md,
            horizontal_gr=dwt_approx,
            type_tvt=type_tvt,
            type_gr=type_gr,
            candidate_tvt=likpf,
            slope=float(prior_meta["slope"]),
            offsets=offsets,
            ncc_weight=float(gr_config["ncc_weight"]),
        )
        part = pd.DataFrame(
            {
                "id": base_frame["id"].iloc[positions].astype(str).to_numpy(),
                "well": np.full(len(positions), str(well), dtype=object),
                SOURCE_COMPONENT_COLUMNS[0]: dwt_ratio[query_rows],
                SOURCE_COMPONENT_COLUMNS[1]: np.full(
                    len(positions),
                    np.float32(fft_meta["fft_rotation_energy_ratio"]),
                    dtype=np.float32,
                ),
                SOURCE_COMPONENT_COLUMNS[2]: (
                    dwt_ncc - raw_ncc
                ).astype(np.float32),
            }
        )
        output_parts.append(part)
        summary_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "known_prefix_rows": known_end,
                "wavelet_metadata": json.dumps(
                    wavelet_meta, sort_keys=True
                ),
                "fft_rotation_energy_ratio": float(
                    fft_meta["fft_rotation_energy_ratio"]
                ),
                "historical_grwr5_values_generated": 0,
                "entropy_interaction_generated": 0,
            }
        )
    output = pd.concat(output_parts, ignore_index=True)
    indexed = output.set_index("id", drop=False)
    expected_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(expected_ids):
        raise ValueError("GR source component IDs differ from base input")
    output = indexed.loc[expected_ids].reset_index(drop=True)
    if not output["well"].astype(str).equals(
        base_frame["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("GR source component wells differ from base input")
    values = output[SOURCE_COMPONENT_COLUMNS].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("GR source components contain non-finite values")
    return output, pd.DataFrame(summary_rows)


# %% [markdown]
# ## 6. Deterministic GRWR-5 derivation
#
# 候補順は8本固定、stack は float32、標準偏差は `ddof=0`、range は max-min。
# 三つの interaction も source component と spread の float32 積だけで作る。

# %%
def build_grwr5_features(
    clean: pd.DataFrame,
    formation: pd.DataFrame,
    source_components: pd.DataFrame,
) -> pd.DataFrame:
    for label, frame, columns in [
        ("clean", clean, CLEAN_CANDIDATE_COLUMNS),
        ("formation", formation, FORMATION_CANDIDATE_COLUMNS),
        ("source", source_components, SOURCE_COMPONENT_COLUMNS),
    ]:
        required = {"id", "well", *columns}
        if missing := required - set(frame.columns):
            raise ValueError(
                f"{label} GRWR input columns missing: {sorted(missing)}"
            )
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"{label} GRWR input IDs are duplicated")
    identity = clean[["id", "well"]].astype(str).reset_index(drop=True)
    for label, frame in [
        ("formation", formation),
        ("source", source_components),
    ]:
        if not identity.equals(
            frame[["id", "well"]].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"{label} GRWR input is not row aligned")
    if set(FORBIDDEN_INPUT_FEATURES).intersection(clean.columns):
        raise ValueError("historical GRWR-5 values entered the clean input")
    if any(
        str(column).startswith(EXP111_SCORE_PREFIXES)
        for column in clean.columns
    ):
        raise ValueError("exp111 score feature entered the clean input")

    last = clean["last_known_tvt"].to_numpy(np.float32)
    candidates = np.vstack(
        [
            clean["pf_ancc"].to_numpy(np.float32),
            (last + clean["beam_mean_d"].to_numpy(np.float32)).astype(
                np.float32
            ),
            (last + clean["likpf_mean_d"].to_numpy(np.float32)).astype(
                np.float32
            ),
            (last + clean["sc_ens_d"].to_numpy(np.float32)).astype(
                np.float32
            ),
            (last + clean["hyb_d"].to_numpy(np.float32)).astype(
                np.float32
            ),
            (
                last
                + formation["tvt_dense_d"].to_numpy(np.float32)
            ).astype(np.float32),
            (
                last
                + formation["tvt_densew_d"].to_numpy(np.float32)
            ).astype(np.float32),
            (
                last
                + formation["tvt_dense50_d"].to_numpy(np.float32)
            ).astype(np.float32),
        ]
    ).astype(np.float32, copy=False)
    if candidates.shape != (8, len(clean)):
        raise ValueError("GRWR candidate stack shape changed")
    if not np.isfinite(candidates).all():
        raise ValueError("GRWR candidate stack contains non-finite values")
    candidate_std = np.std(candidates, axis=0, ddof=0).astype(np.float32)
    candidate_range = (
        np.max(candidates, axis=0) - np.min(candidates, axis=0)
    ).astype(np.float32)
    dwt_ratio = source_components[SOURCE_COMPONENT_COLUMNS[0]].to_numpy(
        np.float32
    )
    fft_ratio = source_components[SOURCE_COMPONENT_COLUMNS[1]].to_numpy(
        np.float32
    )
    ncc_gap = source_components[SOURCE_COMPONENT_COLUMNS[2]].to_numpy(
        np.float32
    )
    result = identity.copy()
    result[GRWR5_FEATURES[0]] = candidate_std
    result[GRWR5_FEATURES[1]] = candidate_range
    result[GRWR5_FEATURES[2]] = (
        dwt_ratio * candidate_std
    ).astype(np.float32)
    result[GRWR5_FEATURES[3]] = (
        fft_ratio * candidate_range
    ).astype(np.float32)
    result[GRWR5_FEATURES[4]] = (
        ncc_gap * candidate_range
    ).astype(np.float32)
    if list(result.columns) != ["id", "well", *GRWR5_FEATURES]:
        raise ValueError("GRWR-5 output order changed")
    for column in GRWR5_FEATURES:
        if result[column].dtype != np.dtype("float32"):
            raise ValueError(f"GRWR-5 dtype changed for {column}")
    if not np.isfinite(result[GRWR5_FEATURES].to_numpy(np.float32)).all():
        raise ValueError("GRWR-5 output contains non-finite values")
    if result[GRWR5_FEATURES].T.duplicated().any():
        raise ValueError("GRWR-5 output contains exact duplicate columns")
    return result


# %% [markdown]
# ## 7. Saved outer-role formation verification
#
# outer-train role は同じ fold の outer-train wells を reference とし、target well を
# query ごとに self-exclude する。outer-valid role は同じ outer-train reference のみを
# 使い、valid well を reference に含めない。保存 manifest と row identity の両方で確認する。

# %%
def verify_fold_role_boundary(
    item: Mapping[str, Any],
    *,
    expected: pd.DataFrame,
    outer_train: pd.DataFrame,
    role: str,
) -> dict[str, Any]:
    reference_wells = sorted(outer_train["well"].astype(str).unique())
    target_wells = sorted(expected["well"].astype(str).unique())
    reference_set = set(reference_wells)
    target_set = set(target_wells)
    overlap = len(reference_set.intersection(target_set))
    if int(item["reference_wells"]) != len(reference_wells):
        raise ValueError("formation reference-well count changed")
    if str(item["reference_well_sha256"]) != sha256_json(reference_wells):
        raise ValueError("formation reference-well SHA changed")
    if str(item["target_well_sha256"]) != sha256_json(target_wells):
        raise ValueError("formation target-well SHA changed")
    if int(item["target_wells_inside_reference"]) != overlap:
        raise ValueError("formation target/reference overlap changed")
    if role == "train":
        if overlap != len(target_wells):
            raise ValueError("outer-train targets must be inside the reference set")
        if int(
            item["target_wells_self_excluded_from_reference_query"]
        ) != len(target_wells):
            raise ValueError("outer-train self-exclusion count changed")
    elif role == "valid":
        if overlap != 0:
            raise ValueError("outer-valid well leaked into formation reference")
        if int(
            item["target_wells_self_excluded_from_reference_query"]
        ) != 0:
            raise ValueError("outer-valid self-exclusion count must be zero")
    else:
        raise ValueError(f"unexpected formation role: {role}")
    return {
        "reference_wells": len(reference_wells),
        "target_wells": len(target_wells),
        "target_wells_inside_reference": overlap,
        "target_formation_columns_read": False,
    }


def load_saved_formation_role(
    *,
    parent_root: Path,
    partitions: pd.DataFrame,
    outer_fold: int,
    role: str,
    expected: pd.DataFrame,
    outer_train: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = partitions[
        partitions["downstream_outer_fold"].astype(int).eq(outer_fold)
        & partitions["role"].astype(str).eq(role)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"formation role is not unique: outer={outer_fold}, role={role}"
        )
    item = rows.iloc[0]
    boundary = verify_fold_role_boundary(
        item,
        expected=expected,
        outer_train=outer_train,
        role=role,
    )
    path = parent_root / str(item["path"])
    frame = pd.read_parquet(
        path, columns=["id", "well", *feature_names]
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if len(frame) != int(item["rows"]) or frame["id"].duplicated().any():
        raise ValueError("saved formation role row/ID contract changed")
    logical_sha = logical_feature_content_sha256(frame, feature_names)
    if logical_sha != str(item["logical_content_sha256"]):
        raise ValueError("saved formation role logical SHA changed")
    expected_ids = expected["id"].astype(str)
    indexed = frame.set_index("id", drop=False)
    if set(indexed.index) != set(expected_ids):
        raise ValueError("saved formation role IDs differ from outer role")
    frame = indexed.loc[expected_ids].reset_index(drop=True)
    if not frame["well"].equals(
        expected["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("saved formation role wells differ from outer role")
    if not np.isfinite(
        frame[list(feature_names)].to_numpy(np.float32)
    ).all():
        raise ValueError("saved formation role contains non-finite values")
    return frame, {
        "path": str(path),
        "file_sha256": str(item["file_sha256"]),
        "logical_content_sha256": logical_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "outer_fold": outer_fold,
        "role": role,
        **boundary,
    }


def generate_outer_role_grwr5(
    *,
    context: pd.DataFrame,
    source_components: pd.DataFrame,
    parent_root: Path,
    partitions: pd.DataFrame,
    formation_features: Sequence[str],
    output_dir: Path,
    outer_folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    context_index = context.set_index("id", drop=False)
    source_index = source_components.set_index("id", drop=False)
    rows: list[dict[str, Any]] = []
    selected_folds = (
        list(range(5))
        if outer_folds is None
        else sorted({int(value) for value in outer_folds})
    )
    if not selected_folds or not set(selected_folds).issubset(set(range(5))):
        raise ValueError(f"invalid outer folds: {selected_folds}")
    for outer_fold in selected_folds:
        outer_train = context[
            context["outer_fold"].astype(int).ne(outer_fold)
        ].reset_index(drop=True)
        outer_valid = context[
            context["outer_fold"].astype(int).eq(outer_fold)
        ].reset_index(drop=True)
        if set(outer_train["well"]).intersection(set(outer_valid["well"])):
            raise ValueError("outer train/valid wells overlap")
        for role, expected in [
            ("train", outer_train),
            ("valid", outer_valid),
        ]:
            formation, formation_evidence = load_saved_formation_role(
                parent_root=parent_root,
                partitions=partitions,
                outer_fold=outer_fold,
                role=role,
                expected=expected,
                outer_train=outer_train,
                feature_names=formation_features,
            )
            ids = expected["id"].astype(str)
            clean = context_index.loc[ids][
                ["id", "well", *CLEAN_CANDIDATE_COLUMNS]
            ].reset_index(drop=True)
            source = source_index.loc[ids][
                ["id", "well", *SOURCE_COMPONENT_COLUMNS]
            ].reset_index(drop=True)
            grwr5 = build_grwr5_features(clean, formation, source)
            path = (
                output_dir
                / "fold_safe_grwr5"
                / f"downstream_outer_fold={outer_fold}"
                / f"role={role}"
                / "part-00000.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            grwr5.to_parquet(path, index=False, compression="zstd")
            rows.append(
                {
                    "downstream_outer_fold": outer_fold,
                    "role": role,
                    "rows": len(grwr5),
                    "wells": int(grwr5["well"].nunique()),
                    "path": str(path.relative_to(output_dir)),
                    "file_sha256": sha256_file(path),
                    "row_identity_sha256": logical_identity_sha256(grwr5),
                    "grwr5_schema_sha256": sha256_json(GRWR5_FEATURES),
                    "grwr5_logical_content_sha256": logical_float_frame_sha256(
                        grwr5, value_columns=GRWR5_FEATURES
                    ),
                    "formation_logical_content_sha256": formation_evidence[
                        "logical_content_sha256"
                    ],
                    "formation_reference_wells": formation_evidence[
                        "reference_wells"
                    ],
                    "formation_target_wells_inside_reference": (
                        formation_evidence["target_wells_inside_reference"]
                    ),
                    "formation_target_columns_read": False,
                    "historical_grwr5_values_loaded": 0,
                    "entropy_or_score_features_loaded": 0,
                }
            )
            del formation, clean, source, grwr5
            gc.collect()
    manifest = pd.DataFrame(rows)
    expected_partitions = 2 * len(selected_folds)
    if len(manifest) != expected_partitions:
        raise ValueError(
            f"GRWR-5 outer-role manifest must contain {expected_partitions} partitions"
        )
    if int(manifest["rows"].sum()) != len(context) * len(selected_folds):
        raise ValueError("GRWR-5 outer-role row coverage is incomplete")
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_partition_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


# %% [markdown]
# ## 8. Raw current-test regeneration
#
# exp072 の deterministic raw-test replay を現在の raw test に再実行し、三つの GR source
# component と all-train-reference formation 74列を同じ run 内で作る。保存済み public-test
# feature row artifact、target well formation、model、prediction は使わない。

# %%
def build_current_test_grwr5_from_frames(
    *,
    replay_frame: pd.DataFrame,
    source_components: pd.DataFrame,
    formation_surface: pd.DataFrame,
    formation_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_rows = int(get_nested(config, "validation.expected_current_test_rows"))
    expected_wells = int(
        get_nested(config, "validation.expected_current_test_wells")
    )
    if len(replay_frame) != expected_rows:
        raise ValueError("current-test replay row count changed")
    if int(replay_frame["well"].nunique()) != expected_wells:
        raise ValueError("current-test replay well count changed")
    if bool(formation_evidence.get("target_formation_columns_read", True)):
        raise ValueError("current-test formation read target formation columns")
    clean = replay_frame[["id", "well", *CLEAN_CANDIDATE_COLUMNS]].copy()
    formation = formation_surface[
        ["id", "well", *FORMATION_CANDIDATE_COLUMNS]
    ].copy()
    grwr5 = build_grwr5_features(clean, formation, source_components)
    return grwr5, {
        "rows": len(grwr5),
        "wells": int(grwr5["well"].nunique()),
        "feature_count": len(GRWR5_FEATURES),
        "schema_sha256": sha256_json(GRWR5_FEATURES),
        "logical_content_sha256": logical_float_frame_sha256(
            grwr5, value_columns=GRWR5_FEATURES
        ),
        "target_formation_columns_read": False,
        "static_public_test_feature_artifact_loaded": False,
        "historical_grwr5_values_loaded": 0,
        "model_count": 0,
        "prediction_count": 0,
    }


def generate_raw_current_test_grwr5(
    *,
    config: Mapping[str, Any],
    replay_source_path: Path,
    raw_root: Path,
    gr_config: Mapping[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_cfg = dict(get_nested(config, "data.exp072_candidate_context"))
    verify_file_sha(
        replay_source_path,
        str(replay_cfg["rawtest_source_sha256"]),
        "exp072 raw-test replay source",
    )
    from experiments.exp072_exp063_full_replay_feature_cache import (
        public_notebook_replay_audit as replay_module,
    )
    runtime = dict(replay_cfg["rawtest_runtime"])
    replay_module.configure_public_runtime(
        data_dir=raw_root,
        output_dir=output_dir / "rawtest_replay",
        n_jobs=int(runtime["n_jobs"]),
        pf_seeds=int(runtime["pf_seeds"]),
        pf_particles=int(runtime["pf_particles"]),
        fast=bool(runtime["fast"]),
        use_gpu=str(runtime["use_gpu"]),
    )
    replay, replay_meta = replay_module.build_replay_test_frame()
    replay["id"] = replay["id"].astype(str)
    replay["well"] = replay["well"].astype(str)
    required_for_formation = {
        "id",
        "well",
        "last_known_tvt",
        "pf_ancc",
        "beam_cons_d",
        "beam_loose_d",
        "beam_vcons_d",
        "beam_sm5_d",
        "beam_vloose_d",
        "beam_mid_d",
        "beam_stiff_d",
        "sc8_d",
        "sc15_d",
        "sc25_d",
        "sc_ens_d",
    }
    required = {
        *required_for_formation,
        *CLEAN_CANDIDATE_COLUMNS,
    }
    if missing := required - set(replay.columns):
        raise ValueError(
            f"raw current-test replay columns missing: {sorted(missing)}"
        )
    if set(FORBIDDEN_INPUT_FEATURES).intersection(replay.columns):
        raise ValueError("raw replay unexpectedly contains historical GRWR-5")
    source_components, source_summary = build_grwr_source_components(
        replay[["id", "well", "last_known_tvt", "likpf_mean_d"]],
        raw_dir=raw_root / "test",
        gr_config=gr_config,
    )
    train_reference_wells = sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in (raw_root / "train").glob("*__horizontal_well.csv")
    )
    if len(train_reference_wells) != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("current-test train-reference well count changed")
    formation_features = canonical_formation_feature_names()
    formation_config = dict(get_nested(config, "formation_generator"))
    if formation_config.get("current_test_reference_policy") != "all_train_wells":
        raise ValueError("current-test formation reference policy changed")
    if bool(
        formation_config.get("current_test_target_formation_columns_read", True)
    ):
        raise ValueError("current-test formation target-read policy changed")
    generator_config = {
        key: formation_config[key]
        for key in [
            "plane_k",
            "dense_samples_per_well",
            "dense_k",
            "dense_nfetch",
            "query_workers",
            "n_jobs",
        ]
    }
    formation, formation_evidence = build_current_test_formation_surface(
        base_frame=replay,
        raw_train_dir=raw_root / "train",
        raw_test_dir=raw_root / "test",
        reference_wells=train_reference_wells,
        feature_names=formation_features,
        generator_config=generator_config,
    )
    expected_formation_logical = str(
        get_nested(
            config,
            "data.current_test_reference.reference_result_logical_sha256",
        )
    )
    actual_formation_logical = logical_feature_content_sha256(
        formation, formation_features
    )
    if actual_formation_logical != expected_formation_logical:
        raise ValueError(
            "raw current-test formation logical SHA differs from exp287: "
            f"{actual_formation_logical} != {expected_formation_logical}"
        )
    grwr5, evidence = build_current_test_grwr5_from_frames(
        replay_frame=replay,
        source_components=source_components,
        formation_surface=formation,
        formation_evidence=formation_evidence,
        config=config,
    )
    path = output_dir / f"{OUTPUT_PREFIX}_current_test_grwr5.parquet"
    grwr5.to_parquet(path, index=False, compression="zstd")
    evidence.update(
        {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "replay_source_path": str(replay_source_path),
            "replay_source_sha256": sha256_file(replay_source_path),
            "replay_meta": replay_meta,
            "source_component_summary_rows": len(source_summary),
            "formation_feature_count": len(formation_features),
            "formation_logical_content_sha256": actual_formation_logical,
            "formation_reference_wells": len(train_reference_wells),
        }
    )
    return grwr5, evidence


def resolve_implementation_source_path() -> Path:
    filename = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"
    candidates = [
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / filename,
    ]
    matches = sorted(
        {candidate.resolve() for candidate in candidates if candidate.is_file()}
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"exp402 implementation source resolution expected one file, got {matches}"
        )
    return matches[0]


def split_execution_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    source_path = resolve_implementation_source_path()
    config_path = find_config_path()
    return {
        "implementation_source": str(source_path),
        "implementation_source_sha256": sha256_file(source_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "scientific_contract_sha256": sha256_json(
            validate_scientific_contract(config, require_run_approval=True)
        ),
        "split_phases": list(SPLIT_STAGE_PHASES),
    }


def read_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def run_stage0_current_test(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = validate_scientific_contract(
        config, require_run_approval=require_run_approval
    )
    search_roots = [PACKAGE_DIR, KAGGLE_INPUT_ROOT, Path("/tmp")]
    formula_cfg = dict(get_nested(config, "data.historical_formula"))
    formula_source_path = resolve_existing_path(
        formula_cfg["patterns"],
        search_roots=search_roots,
        label="exp218 formula source",
    )
    formula_config_path = resolve_existing_path(
        formula_cfg["config_patterns"],
        search_roots=search_roots,
        label="exp218 formula config",
    )
    gr_config, formula_evidence = validate_formula_source_contract(
        formula_source_path, formula_config_path, config
    )
    replay_cfg = dict(get_nested(config, "data.exp072_candidate_context"))
    replay_source_path = resolve_existing_path(
        replay_cfg["rawtest_source_patterns"],
        search_roots=search_roots,
        label="exp072 raw-test replay source",
    )
    raw_root = find_competition_input_root()
    print("Stage 0B current-test replay started", flush=True)
    _current_test, current_test_evidence = generate_raw_current_test_grwr5(
        config=config,
        replay_source_path=replay_source_path,
        raw_root=raw_root,
        gr_config=gr_config,
        output_dir=output_dir,
    )
    current_test_path = (
        output_dir / f"{OUTPUT_PREFIX}_current_test_grwr5.parquet"
    )
    checks = {
        "current_test_rows_and_wells_match": (
            current_test_evidence["rows"]
            == int(get_nested(config, "validation.expected_current_test_rows"))
            and current_test_evidence["wells"]
            == int(get_nested(config, "validation.expected_current_test_wells"))
        ),
        "current_test_grwr5_finite_and_hashed": (
            len(str(current_test_evidence["logical_content_sha256"])) == 64
            and len(str(current_test_evidence["file_sha256"])) == 64
        ),
        "current_test_grwr5_schema_matches": (
            current_test_evidence["schema_sha256"]
            == sha256_json(GRWR5_FEATURES)
        ),
        "current_test_formation_matches_exp287": (
            current_test_evidence["formation_logical_content_sha256"]
            == str(
                get_nested(
                    config,
                    "data.current_test_reference.reference_result_logical_sha256",
                )
            )
        ),
        "current_test_target_formation_read_zero": not bool(
            current_test_evidence["target_formation_columns_read"]
        ),
        "historical_grwr5_load_zero": int(
            current_test_evidence["historical_grwr5_values_loaded"]
        )
        == 0,
        "model_prediction_submission_zero": (
            contract["current_execution"]
            == {
                "models": 0,
                "boosters": 0,
                "predictions": 0,
                "submissions": 0,
            }
            and int(current_test_evidence["model_count"]) == 0
            and int(current_test_evidence["prediction_count"]) == 0
        ),
    }
    portable_current_test_evidence = {
        **current_test_evidence,
        "path": str(current_test_path.relative_to(output_dir)),
    }
    manifest = {
        "schema_version": "2.0.0",
        "experiment": EXPERIMENT_NAME,
        "phase": "current_test",
        "status": (
            "stage_0_current_test_passed"
            if all(checks.values())
            else "stage_0_current_test_failed"
        ),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "execution_identity": split_execution_identity(config),
        "cost_contract": contract,
        "input": {
            "formula": formula_evidence,
            "raw_root": str(raw_root),
        },
        "generated": {
            "current_test": portable_current_test_evidence,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "models_trained": 0,
        "boosters_trained": 0,
        "prediction_rows_generated": 0,
        "submission_rows_generated": 0,
        "control_boosters_trained": 0,
    }
    manifest_path = (
        output_dir / f"{OUTPUT_PREFIX}_stage0_current_test_manifest.json"
    )
    manifest["manifest_sha256"] = write_json(manifest_path, manifest)
    print(
        "Stage 0B current-test replay finished",
        json.dumps(
            {
                "passed": manifest["passed"],
                "runtime_seconds": manifest["runtime_seconds"],
                "rows": current_test_evidence["rows"],
                "wells": current_test_evidence["wells"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not manifest["passed"]:
        raise RuntimeError("exp402 Stage 0B current-test replay failed")
    return manifest


def run_stage0_train_fold(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    output_dir: Path,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    if outer_fold not in range(5):
        raise ValueError(f"exp402 train fold must be 0..4, got {outer_fold}")
    contract = validate_scientific_contract(
        config, require_run_approval=require_run_approval
    )
    print(f"Stage 0F outer-fold {outer_fold} generation started", flush=True)
    search_roots = [PACKAGE_DIR, KAGGLE_INPUT_ROOT, Path("/tmp")]
    parent_cfg = dict(get_nested(config, "data.parent_exp287"))
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
        parent_cfg["artifact_root_patterns"],
        search_roots=search_roots,
        required_files=parent_required,
        label="saved exp287 train artifacts",
    )
    model_manifest, parent_oof, partitions, parent_evidence = (
        verify_parent_artifacts(
            parent_root,
            config,
            output_dir=output_dir,
            formation_roles_to_verify=[
                (outer_fold, "train"),
                (outer_fold, "valid"),
            ],
        )
    )
    formation_features = [
        str(value)
        for value in model_manifest["feature_groups"]["fold_safe_formation"]
    ]

    source_manifest_name = (
        f"{OUTPUT_PREFIX}_stage0_train_source_manifest.json"
    )
    source_data_name = f"{OUTPUT_PREFIX}_train_context_source.parquet"
    source_root = resolve_split_artifact_root(
        config,
        phase="train_source",
        required_files=[source_manifest_name, source_data_name],
    )
    source_manifest_path = source_root / source_manifest_name
    source_manifest = read_json_mapping(source_manifest_path)
    source_generated = dict(
        get_nested(source_manifest, "generated.train_context_source") or {}
    )
    source_path = source_root / str(source_generated.get("path", ""))
    source_file_sha = verify_file_sha(
        source_path,
        str(source_generated.get("file_sha256")),
        "exp402 train-context source",
    )
    source_columns = [
        "id",
        "well",
        "outer_fold",
        *CLEAN_CANDIDATE_COLUMNS,
        *SOURCE_COMPONENT_COLUMNS,
    ]
    context_source = pd.read_parquet(source_path, columns=source_columns)
    context_source["id"] = context_source["id"].astype(str)
    context_source["well"] = context_source["well"].astype(str)
    context_source["outer_fold"] = pd.to_numeric(
        context_source["outer_fold"], errors="raise"
    ).astype(np.int8)
    if context_source["id"].duplicated().any():
        raise ValueError("train-context source IDs are duplicated")
    source_logical_sha = logical_float_frame_sha256(
        context_source,
        value_columns=SOURCE_COMPONENT_COLUMNS,
    )
    context = context_source[
        ["id", "well", "outer_fold", *CLEAN_CANDIDATE_COLUMNS]
    ].copy()
    source_components = context_source[
        ["id", "well", *SOURCE_COMPONENT_COLUMNS]
    ].copy()
    parent_index = parent_oof.set_index("id", drop=False)
    context_index = context.set_index("id", drop=False)
    if set(context_index.index) != set(parent_index.index):
        raise ValueError("train-context source IDs differ from exp287 OOF")
    expected_parent = parent_index.loc[context["id"]].reset_index(drop=True)
    if not context["well"].equals(expected_parent["well"]):
        raise ValueError("train-context source wells differ from exp287 OOF")
    if not context["outer_fold"].equals(expected_parent["outer_fold"]):
        raise ValueError("train-context source folds differ from exp287 OOF")

    partition_manifest = generate_outer_role_grwr5(
        context=context,
        source_components=source_components,
        parent_root=parent_root,
        partitions=partitions,
        formation_features=formation_features,
        output_dir=output_dir,
        outer_folds=[outer_fold],
    )
    partition_manifest_path = (
        output_dir / f"{OUTPUT_PREFIX}_partition_manifest.csv"
    )
    local_identity = split_execution_identity(config)
    source_identity = dict(source_manifest.get("execution_identity") or {})
    checks = {
        "train_source_phase_passed": bool(source_manifest.get("passed")),
        "split_execution_identity_matches": (
            source_identity.get("implementation_source_sha256")
            == local_identity["implementation_source_sha256"]
            and source_identity.get("config_sha256")
            == local_identity["config_sha256"]
            and source_identity.get("scientific_contract_sha256")
            == local_identity["scientific_contract_sha256"]
        ),
        "train_context_source_file_matches": (
            source_file_sha == str(source_generated.get("file_sha256"))
        ),
        "train_context_source_logical_sha_matches": (
            source_logical_sha
            == str(source_generated.get("logical_content_sha256"))
        ),
        "rows_and_wells_match": (
            len(context) == int(get_nested(config, "validation.expected_rows"))
            and int(context["well"].nunique())
            == int(get_nested(config, "validation.expected_wells"))
        ),
        "parent_fixed_artifacts_match": (
            len(parent_evidence["file_sha256"]) == 7
        ),
        "parent_role_physical_sha_match_2": (
            set(parent_evidence["partition_file_sha256"])
            == {
                f"outer{outer_fold}_train",
                f"outer{outer_fold}_valid",
            }
        ),
        "outer_role_partitions_match_2": len(partition_manifest) == 2,
        "outer_role_grwr5_finite_and_hashed": bool(
            partition_manifest["grwr5_logical_content_sha256"]
            .astype(str)
            .str.len()
            .eq(64)
            .all()
        ),
        "grwr5_schema_matches": bool(
            partition_manifest["grwr5_schema_sha256"]
            .astype(str)
            .eq(sha256_json(GRWR5_FEATURES))
            .all()
        ),
        "formation_boundaries_match": bool(
            partition_manifest["formation_target_columns_read"].eq(False).all()
        ),
        "historical_entropy_score_load_zero": (
            int(partition_manifest["historical_grwr5_values_loaded"].sum()) == 0
            and int(
                partition_manifest["entropy_or_score_features_loaded"].sum()
            )
            == 0
        ),
        "planned_training_is_15_and_control_zero": (
            contract["future_training"]["gpu_boosters"] == 15
            and contract["future_training"]["control_boosters"] == 0
        ),
        "model_prediction_submission_zero": contract["current_execution"]
        == {
            "models": 0,
            "boosters": 0,
            "predictions": 0,
            "submissions": 0,
        },
    }
    manifest = {
        "schema_version": "3.0.0",
        "experiment": EXPERIMENT_NAME,
        "phase": "train_fold",
        "outer_fold": outer_fold,
        "status": (
            "stage_0_train_fold_passed"
            if all(checks.values())
            else "stage_0_train_fold_failed"
        ),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "execution_identity": local_identity,
        "cost_contract": contract,
        "feature_schema_sha256": {
            "parent_421": parent_evidence["parent_feature_schema_sha256"],
            "grwr5": sha256_json(GRWR5_FEATURES),
        },
        "input": {
            "parent": parent_evidence,
            "train_source": {
                "root": str(source_root),
                "manifest_path": str(source_manifest_path),
                "manifest_file_sha256": sha256_file(source_manifest_path),
                "data_path": str(source_path),
                "data_file_sha256": source_file_sha,
            },
        },
        "generated": {
            "outer_role_partition_manifest": {
                "path": str(partition_manifest_path.relative_to(output_dir)),
                "sha256": sha256_file(partition_manifest_path),
                "partitions": len(partition_manifest),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "models_trained": 0,
        "boosters_trained": 0,
        "prediction_rows_generated": 0,
        "submission_rows_generated": 0,
        "control_boosters_trained": 0,
    }
    manifest_path = (
        output_dir
        / f"{OUTPUT_PREFIX}_stage0_train_fold_manifest.json"
    )
    write_json(manifest_path, manifest)
    print(
        f"Stage 0F outer-fold {outer_fold} generation finished",
        json.dumps(
            {
                "passed": manifest["passed"],
                "runtime_seconds": manifest["runtime_seconds"],
                "partitions": len(partition_manifest),
                "partition_rows": int(partition_manifest["rows"].sum()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not manifest["passed"]:
        raise RuntimeError(f"exp402 Stage 0F outer-fold {outer_fold} failed")
    return manifest


def resolve_split_artifact_root(
    config: Mapping[str, Any],
    *,
    phase: str,
    required_files: Sequence[str],
) -> Path:
    patterns = list(
        get_nested(
            config,
            f"runtime.kaggle.split_stage_0.{phase}.artifact_root_patterns",
        )
        or []
    )
    if not patterns:
        raise ValueError(f"exp402 split artifact patterns missing for {phase}")
    return resolve_artifact_root(
        patterns,
        search_roots=[PACKAGE_DIR, KAGGLE_INPUT_ROOT, Path("/tmp")],
        required_files=required_files,
        label=f"exp402 split Stage 0 {phase}",
    )


def resolve_train_fold_artifact_root(
    config: Mapping[str, Any],
    *,
    outer_fold: int,
    required_files: Sequence[str],
) -> Path:
    patterns = list(
        get_nested(
            config,
            f"runtime.kaggle.split_stage_0.train_folds.{outer_fold}.artifact_root_patterns",
        )
        or []
    )
    if not patterns:
        raise ValueError(
            f"exp402 split artifact patterns missing for train fold {outer_fold}"
        )
    return resolve_artifact_root(
        patterns,
        search_roots=[PACKAGE_DIR, KAGGLE_INPUT_ROOT, Path("/tmp")],
        required_files=required_files,
        label=f"exp402 split Stage 0 train fold {outer_fold}",
    )


def run_stage0_aggregate(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = validate_scientific_contract(
        config, require_run_approval=require_run_approval
    )
    source_manifest_name = (
        f"{OUTPUT_PREFIX}_stage0_train_source_manifest.json"
    )
    source_data_name = f"{OUTPUT_PREFIX}_train_context_source.parquet"
    fold_manifest_name = f"{OUTPUT_PREFIX}_stage0_train_fold_manifest.json"
    partition_manifest_name = f"{OUTPUT_PREFIX}_partition_manifest.csv"
    current_manifest_name = f"{OUTPUT_PREFIX}_stage0_current_test_manifest.json"
    current_test_name = f"{OUTPUT_PREFIX}_current_test_grwr5.parquet"

    source_root = resolve_split_artifact_root(
        config,
        phase="train_source",
        required_files=[source_manifest_name, source_data_name],
    )
    current_root = resolve_split_artifact_root(
        config,
        phase="current_test",
        required_files=[current_manifest_name, current_test_name],
    )
    source_manifest_path = source_root / source_manifest_name
    current_manifest_path = current_root / current_manifest_name
    source_manifest = read_json_mapping(source_manifest_path)
    current_manifest = read_json_mapping(current_manifest_path)
    source_generated = dict(
        get_nested(source_manifest, "generated.train_context_source") or {}
    )
    source_path = source_root / str(source_generated.get("path", ""))
    current_generated = dict(
        get_nested(current_manifest, "generated.current_test") or {}
    )
    current_path = current_root / str(current_generated.get("path", ""))

    fold_manifests: list[dict[str, Any]] = []
    fold_inputs: dict[str, dict[str, Any]] = {}
    partition_frames: list[pd.DataFrame] = []
    partition_file_checks: dict[str, bool] = {}
    fold_manifest_sha_checks: dict[str, bool] = {}
    for outer_fold in range(5):
        fold_root = resolve_train_fold_artifact_root(
            config,
            outer_fold=outer_fold,
            required_files=[fold_manifest_name, partition_manifest_name],
        )
        fold_manifest_path = fold_root / fold_manifest_name
        partition_path = fold_root / partition_manifest_name
        fold_manifest = read_json_mapping(fold_manifest_path)
        partition_frame = pd.read_csv(partition_path)
        if (
            int(fold_manifest.get("outer_fold", -1)) != outer_fold
            or len(partition_frame) != 2
            or set(partition_frame["downstream_outer_fold"].astype(int))
            != {outer_fold}
            or set(partition_frame["role"].astype(str)) != {"train", "valid"}
        ):
            raise ValueError(
                f"split Stage 0 train fold {outer_fold} coverage changed"
            )
        reported_partition_sha = str(
            get_nested(
                fold_manifest,
                "generated.outer_role_partition_manifest.sha256",
            )
        )
        fold_manifest_sha_checks[str(outer_fold)] = (
            sha256_file(partition_path) == reported_partition_sha
        )
        for row in partition_frame.itertuples(index=False):
            key = f"outer{int(row.downstream_outer_fold)}_{row.role}"
            path = fold_root / str(row.path)
            partition_file_checks[key] = (
                path.is_file() and sha256_file(path) == str(row.file_sha256)
            )
        partition_frame["source_outer_fold"] = outer_fold
        partition_frames.append(partition_frame)
        fold_manifests.append(fold_manifest)
        fold_inputs[str(outer_fold)] = {
            "root": str(fold_root),
            "manifest_path": str(fold_manifest_path),
            "manifest_file_sha256": sha256_file(fold_manifest_path),
            "partition_manifest_path": str(partition_path),
            "partition_manifest_file_sha256": sha256_file(partition_path),
        }

    partition_manifest = pd.concat(partition_frames, ignore_index=True)
    if (
        len(partition_manifest) != 10
        or partition_manifest.duplicated(
            ["downstream_outer_fold", "role"]
        ).any()
    ):
        raise ValueError("split Stage 0 fold coverage must contain 10 unique roles")
    combined_partition_path = (
        output_dir / f"{OUTPUT_PREFIX}_partition_manifest.csv"
    )
    partition_manifest.to_csv(combined_partition_path, index=False)

    local_identity = split_execution_identity(config)
    upstream_manifests = [source_manifest, *fold_manifests, current_manifest]
    identity_checks = []
    for upstream in upstream_manifests:
        identity = dict(upstream.get("execution_identity") or {})
        identity_checks.append(
            identity.get("implementation_source_sha256")
            == local_identity["implementation_source_sha256"]
            and identity.get("config_sha256") == local_identity["config_sha256"]
            and identity.get("scientific_contract_sha256")
            == local_identity["scientific_contract_sha256"]
        )
    upstream_zero = all(
        int(manifest.get("models_trained", -1)) == 0
        and int(manifest.get("boosters_trained", -1)) == 0
        for manifest in upstream_manifests
    )
    checks = {
        "train_source_phase_passed": bool(source_manifest.get("passed")),
        "all_five_train_fold_phases_passed": all(
            bool(manifest.get("passed")) for manifest in fold_manifests
        ),
        "current_test_phase_passed": bool(current_manifest.get("passed")),
        "split_execution_identity_matches": all(identity_checks),
        "fold_partition_manifest_sha_matches": all(
            fold_manifest_sha_checks.values()
        ),
        "formation_roles_match_10": (
            len(partition_manifest) == 10
            and set(partition_manifest["downstream_outer_fold"].astype(int))
            == set(range(5))
            and set(partition_manifest["role"].astype(str))
            == {"train", "valid"}
        ),
        "outer_role_files_match_declared_sha": (
            len(partition_file_checks) == 10
            and all(partition_file_checks.values())
        ),
        "outer_role_rows_cover_five_full_train_copies": (
            int(partition_manifest["rows"].sum())
            == int(get_nested(config, "validation.expected_rows")) * 5
        ),
        "outer_role_grwr5_logical_sha_present": bool(
            partition_manifest["grwr5_logical_content_sha256"]
            .astype(str)
            .str.len()
            .eq(64)
            .all()
        ),
        "outer_role_schema_matches": bool(
            partition_manifest["grwr5_schema_sha256"]
            .astype(str)
            .eq(sha256_json(GRWR5_FEATURES))
            .all()
        ),
        "formation_boundaries_match": bool(
            partition_manifest["formation_target_columns_read"].eq(False).all()
        ),
        "historical_entropy_score_load_zero": (
            int(partition_manifest["historical_grwr5_values_loaded"].sum()) == 0
            and int(
                partition_manifest["entropy_or_score_features_loaded"].sum()
            )
            == 0
        ),
        "train_context_source_file_matches": (
            source_path.is_file()
            and sha256_file(source_path)
            == str(source_generated.get("file_sha256"))
        ),
        "current_test_file_matches": (
            current_path.is_file()
            and sha256_file(current_path)
            == str(current_generated.get("file_sha256"))
        ),
        "current_test_rows_and_wells_match": (
            int(current_generated.get("rows", -1))
            == int(get_nested(config, "validation.expected_current_test_rows"))
            and int(current_generated.get("wells", -1))
            == int(get_nested(config, "validation.expected_current_test_wells"))
        ),
        "current_test_target_formation_read_zero": not bool(
            current_generated.get("target_formation_columns_read", True)
        ),
        "planned_training_is_15_and_control_zero": (
            contract["future_training"]["gpu_boosters"] == 15
            and contract["future_training"]["control_boosters"] == 0
        ),
        "model_prediction_submission_zero": (
            contract["current_execution"]
            == {
                "models": 0,
                "boosters": 0,
                "predictions": 0,
                "submissions": 0,
            }
            and upstream_zero
        ),
    }
    manifest = {
        "schema_version": "3.0.0",
        "experiment": EXPERIMENT_NAME,
        "phase": "aggregate",
        "status": (
            "zero_booster_preflight_passed"
            if all(checks.values())
            else "zero_booster_preflight_failed"
        ),
        "passed": bool(all(checks.values())),
        "checks": checks,
        "partition_file_checks": partition_file_checks,
        "fold_manifest_sha_checks": fold_manifest_sha_checks,
        "execution_identity": local_identity,
        "cost_contract": contract,
        "feature_counts": {
            "clean": 273,
            "nested_compact": 74,
            "formation": 74,
            "parent": 421,
            "grwr5": 5,
            "final": 426,
        },
        "feature_schema_sha256": {
            "parent_421": get_nested(
                fold_manifests[0], "feature_schema_sha256.parent_421"
            ),
            "grwr5": sha256_json(GRWR5_FEATURES),
        },
        "input": {
            "train_source": {
                "root": str(source_root),
                "manifest_path": str(source_manifest_path),
                "manifest_file_sha256": sha256_file(source_manifest_path),
            },
            "train_folds": fold_inputs,
            "current_test": {
                "root": str(current_root),
                "manifest_path": str(current_manifest_path),
                "manifest_file_sha256": sha256_file(current_manifest_path),
            },
        },
        "generated": {
            "train_context_source": source_generated,
            "outer_role_partition_manifest": {
                "path": str(combined_partition_path.relative_to(output_dir)),
                "sha256": sha256_file(combined_partition_path),
                "partitions": len(partition_manifest),
            },
            "current_test": current_generated,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "models_trained": 0,
        "boosters_trained": 0,
        "prediction_rows_generated": 0,
        "submission_rows_generated": 0,
        "control_boosters_trained": 0,
    }
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_preflight_manifest.json"
    manifest_sha = write_json(manifest_path, manifest)
    reproducibility = {
        **manifest,
        "preflight_manifest_sha256": manifest_sha,
        "partition_file_sha256": {
            f"outer{int(row.downstream_outer_fold)}_{row.role}": str(
                row.file_sha256
            )
            for row in partition_manifest.itertuples(index=False)
        },
        "partition_logical_content_sha256": {
            f"outer{int(row.downstream_outer_fold)}_{row.role}": str(
                row.grwr5_logical_content_sha256
            )
            for row in partition_manifest.itertuples(index=False)
        },
        "train_context_source_file_sha256": source_generated.get(
            "file_sha256"
        ),
        "train_context_source_logical_content_sha256": source_generated.get(
            "logical_content_sha256"
        ),
        "current_test_file_sha256": current_generated.get("file_sha256"),
        "current_test_logical_content_sha256": current_generated.get(
            "logical_content_sha256"
        ),
    }
    write_json(
        output_dir / f"{OUTPUT_PREFIX}_reproducibility_manifest.json",
        reproducibility,
    )
    print(
        "Stage 0C aggregate finished",
        json.dumps(
            {
                "passed": manifest["passed"],
                "runtime_seconds": manifest["runtime_seconds"],
                "checks_passed": int(sum(checks.values())),
                "checks_total": len(checks),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not manifest["passed"]:
        raise RuntimeError("exp402 split zero-booster preflight failed")
    return manifest


# %% [markdown]
# ## 9. Zero-booster preflight orchestration
#
# Stage 0 は 10 outer-role cache と raw current-test GRWR-5、schema/content SHA、
# leakage/cost manifest だけを保存する。LightGBM、保存 model inference、TVT prediction、
# competition output はこの関数に存在しない。

# %%
def run_zero_booster_preflight(
    config: Mapping[str, Any],
    *,
    output_dir: Path,
    require_run_approval: bool = True,
    phase: str | None = None,
    outer_fold: int | None = None,
) -> dict[str, Any]:
    selected_phase = str(
        phase or get_nested(config, "execution.stage_0_default_phase")
    )
    if selected_phase not in SPLIT_STAGE_PHASES:
        raise ValueError(
            f"exp402 Stage 0 requires one split phase, got {selected_phase}"
        )
    if selected_phase == "current_test":
        return run_stage0_current_test(
            config,
            output_dir=output_dir,
            require_run_approval=require_run_approval,
        )
    if selected_phase == "aggregate":
        return run_stage0_aggregate(
            config,
            output_dir=output_dir,
            require_run_approval=require_run_approval,
        )
    if selected_phase == "train_fold":
        if outer_fold is None:
            raise ValueError("exp402 train_fold phase requires outer_fold")
        return run_stage0_train_fold(
            config,
            outer_fold=int(outer_fold),
            output_dir=output_dir,
            require_run_approval=require_run_approval,
        )

    started = time.perf_counter()
    contract = validate_scientific_contract(
        config, require_run_approval=require_run_approval
    )
    print("Stage 0A train-source generation started", flush=True)
    search_roots = [PACKAGE_DIR, KAGGLE_INPUT_ROOT, Path("/tmp")]
    parent_cfg = dict(get_nested(config, "data.parent_exp287"))
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
        parent_cfg["artifact_root_patterns"],
        search_roots=search_roots,
        required_files=parent_required,
        label="saved exp287 train artifacts",
    )
    model_manifest, parent_oof, partitions, parent_evidence = (
        verify_parent_artifacts(
            parent_root,
            config,
            output_dir=output_dir,
            formation_roles_to_verify=[],
        )
    )
    formation_features = [
        str(value)
        for value in model_manifest["feature_groups"]["fold_safe_formation"]
    ]

    availability_cfg = dict(get_nested(config, "data.availability_audit"))
    audit_path = resolve_existing_path(
        availability_cfg["patterns"],
        search_roots=search_roots,
        label="exp218 availability audit",
    )
    allowlist_path = resolve_existing_path(
        availability_cfg["clean_allowlist_patterns"],
        search_roots=search_roots,
        label="clean-273 allowlist",
    )
    availability = verify_availability_contract(
        audit_path, allowlist_path, config
    )
    formula_cfg = dict(get_nested(config, "data.historical_formula"))
    formula_source_path = resolve_existing_path(
        formula_cfg["patterns"],
        search_roots=search_roots,
        label="exp218 formula source",
    )
    formula_config_path = resolve_existing_path(
        formula_cfg["config_patterns"],
        search_roots=search_roots,
        label="exp218 formula config",
    )
    gr_config, formula_evidence = validate_formula_source_contract(
        formula_source_path, formula_config_path, config
    )

    replay_cfg = dict(get_nested(config, "data.exp072_candidate_context"))
    train_cache_path = resolve_existing_path(
        replay_cfg["cache_patterns"],
        search_roots=search_roots,
        label="exp072 deterministic train cache",
    )
    context_with_target, context_evidence = load_exp072_candidate_context(
        train_cache_path, config
    )
    control_cfg = dict(get_nested(config, "data.clean_control_exp264"))
    exp264_oof_path = resolve_existing_path(
        control_cfg["oof_patterns"],
        search_roots=search_roots,
        label="corrected exp264 OOF",
    )
    context, _exp264_oof, alignment = align_oof_contracts(
        context_with_target, parent_oof, exp264_oof_path, config
    )
    raw_root = find_competition_input_root()
    source_components, source_summary = build_grwr_source_components(
        context[["id", "well", "last_known_tvt", "likpf_mean_d"]],
        raw_dir=raw_root / "train",
        gr_config=gr_config,
    )
    train_context_source = context[
        ["id", "well", "outer_fold", *CLEAN_CANDIDATE_COLUMNS]
    ].copy()
    for column in SOURCE_COMPONENT_COLUMNS:
        train_context_source[column] = source_components[column].to_numpy(
            np.float32
        )
    source_component_path = (
        output_dir / f"{OUTPUT_PREFIX}_train_context_source.parquet"
    )
    train_context_source.to_parquet(
        source_component_path, index=False, compression="zstd"
    )
    checks = {
        "pinned_exp287_artifacts_match": len(
            parent_evidence["file_sha256"]
        )
        == 7,
        "pinned_exp264_oof_matches": alignment["exp264_oof_sha256"]
        == str(get_nested(config, "data.clean_control_exp264.oof_sha256")),
        "rows_match_3783989": len(context)
        == int(get_nested(config, "validation.expected_rows")),
        "wells_match_773": context["well"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "folds_match_5": set(context["outer_fold"].unique()) == set(range(5)),
        "parent_features_match_421": parent_evidence["parent_feature_count"]
        == 421,
        "formation_partition_physical_reads_zero": (
            parent_evidence["partition_file_sha256"] == {}
        ),
        "train_context_source_finite": bool(
            np.isfinite(
                train_context_source[
                    [*CLEAN_CANDIDATE_COLUMNS, *SOURCE_COMPONENT_COLUMNS]
                ].to_numpy(np.float32)
            ).all()
        ),
        "historical_grwr5_load_zero": (
            int(parent_evidence["historical_grwr5_loaded"]) == 0
        ),
        "entropy_and_score_load_zero": (
            int(parent_evidence["exp111_score_features_loaded"]) == 0
            and int(availability["entropy_interaction_selected"]) == 0
        ),
        "planned_training_is_15_and_control_zero": (
            contract["future_training"]["gpu_boosters"] == 15
            and contract["future_training"]["control_boosters"] == 0
        ),
        "model_prediction_submission_zero": contract["current_execution"]
        == {
            "models": 0,
            "boosters": 0,
            "predictions": 0,
            "submissions": 0,
        },
    }
    manifest = {
        "schema_version": "2.0.0",
        "experiment": EXPERIMENT_NAME,
        "phase": "train_source",
        "status": "stage_0_train_source_passed"
        if all(checks.values())
        else "stage_0_train_source_failed",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "execution_identity": split_execution_identity(config),
        "cost_contract": contract,
        "feature_counts": {
            "clean": 273,
            "nested_compact": 74,
            "formation": 74,
            "parent": 421,
            "grwr5": 5,
            "final": 426,
        },
        "feature_schema_sha256": {
            "parent_421": parent_evidence[
                "parent_feature_schema_sha256"
            ],
            "grwr5": sha256_json(GRWR5_FEATURES),
        },
        "input": {
            "parent": parent_evidence,
            "alignment": alignment,
            "availability": availability,
            "formula": formula_evidence,
            "exp072_context": context_evidence,
        },
        "generated": {
            "train_context_source": {
                "path": str(source_component_path.relative_to(output_dir)),
                "file_sha256": sha256_file(source_component_path),
                "logical_content_sha256": logical_float_frame_sha256(
                    train_context_source,
                    value_columns=SOURCE_COMPONENT_COLUMNS,
                ),
                "rows": len(train_context_source),
                "wells": int(train_context_source["well"].nunique()),
                "summary_rows": len(source_summary),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_gb": current_peak_rss_gb(),
        "models_trained": 0,
        "boosters_trained": 0,
        "prediction_rows_generated": 0,
        "submission_rows_generated": 0,
        "control_boosters_trained": 0,
    }
    manifest_path = (
        output_dir / f"{OUTPUT_PREFIX}_stage0_train_source_manifest.json"
    )
    write_json(manifest_path, manifest)
    print(
        "Stage 0A train-source generation finished",
        json.dumps(
            {
                "passed": manifest["passed"],
                "runtime_seconds": manifest["runtime_seconds"],
                "rows": len(train_context_source),
                "wells": int(train_context_source["well"].nunique()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not manifest["passed"]:
        raise RuntimeError("exp402 Stage 0A train-source generation failed")
    return manifest


# %% [markdown]
# ## 10. Setup and configuration

# %%
if not IMPORT_ONLY:
    CONFIG = read_yaml(find_config_path())
    CONTRACT = validate_scientific_contract(CONFIG)
    display(CONTRACT)
    print(
        "Stage 0 split execution is approved. Default phase: "
        f"{CONTRACT['stage_0_default_phase']}. Stage 1 remains unapproved."
    )


# %% [markdown]
# ## 11. Execution boundary and generated evidence
#
# `implementation_only` は設定と cost contract の表示だけで終了し、Kaggle input を読まない。
# `zero_booster_preflight` は三つの承認 flag がすべて true の場合だけ実行できる。
# 重い生成は`train_roles`と`current_test`へ分け、`aggregate`は両outputのSHAを統合する。

# %%
def run_experiment(
    config: Mapping[str, Any],
    *,
    phase: str | None = None,
    outer_fold: int | None = None,
) -> dict[str, Any]:
    stage = str(get_nested(config, "execution.current_stage"))
    if stage == "implementation_only":
        return {
            "status": "stage_0_implementation_complete_no_execution",
            "contract": validate_scientific_contract(config),
            "kaggle_input_read": False,
            "models_trained": 0,
            "boosters_trained": 0,
            "prediction_rows_generated": 0,
            "submission_rows_generated": 0,
        }
    validate_scientific_contract(config, require_run_approval=True)
    if not KAGGLE_INPUT_ROOT.exists() or not KAGGLE_WORKING_ROOT.exists():
        raise RuntimeError("Kaggle Notebook execution is authoritative for exp402")
    output_dir = KAGGLE_WORKING_ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_zero_booster_preflight(
        config,
        output_dir=output_dir,
        require_run_approval=True,
        phase=phase,
        outer_fold=outer_fold,
    )


if not IMPORT_ONLY:
    RUN_RESULT = run_experiment(CONFIG)
    display(RUN_RESULT)
    if KAGGLE_WORKING_ROOT.exists():
        artifact_root = KAGGLE_WORKING_ROOT / "artifacts"
        if artifact_root.exists():
            print("Generated files")
            for generated in sorted(artifact_root.rglob("*")):
                if generated.is_file():
                    print(
                        generated.relative_to(artifact_root),
                        generated.stat().st_size,
                    )
