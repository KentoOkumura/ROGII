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
# # exp372 exp287 + exp335 feature union CPU inference
#
# exp263の12候補と21 confidence列をraw competition testから同じrunで再生成する。保存済み
# corrected exp264 Stage C v6の40本からouter別74 compact特徴、exp335 Stage S v3の20本から
# outer別signed-residual 23特徴を作る。さらにall-train-reference formation74を生成し、
# exp372 train version 2の15本へ`clean273 -> saved74 -> formation74 -> signed23`で渡す。
# model fit、hard selector、Viterbi、competition submitは行わない。

# %% [markdown]
# ## Contents
#
# 1. Imports and runtime helpers
# 2. User authorization and saved-model contracts
# 3. Exp263 hidden-safe 12-candidate regeneration
# 4. Candidate-long context, parent compact, and signed compact features
# 5. Exp218 current-test 380-feature surface
# 6. Current-test all-train-reference formation features
# 7. Exp372 union saved-booster inference
# 8. Prediction artifacts and reproducibility manifest

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, load_config

from src.candidate_selector_pipeline import (
    ShapeState,
    build_candidate_long_features,
    build_compact_meta,
    build_raw_context,
    candidate_contract_sha,
    candidate_ids,
    current_test_bundle_from_wide,
    fill_current_test_anchor,
    load_feature_schema,
    read_yaml,
    resolve_existing_path,
    sha256_file,
    validate_current_test_native_confidence,
    validate_inference_feature_missingness,
    write_json,
)
from src.fold_safe_formation_pipeline import (
    build_current_test_formation_surface,
    canonical_formation_feature_names,
    logical_feature_content_sha256,
)
from src.signed_residual_meta import (
    build_signed_compact_meta,
    signed_compact_feature_names,
)

STARTED_AT = time.time()
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(
        "experiments/exp372_exp287_exp335_feature_union_on_exp264"
    )
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
output_dir = paths.artifacts_dir
output_dir.mkdir(parents=True, exist_ok=True)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def get_nested(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def import_file(name: str, candidates: list[Path]) -> Any:
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve_unique_source(filename: str, path_token: str) -> Path:
    matches = [
        path
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if path_token in str(path)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} under source token {path_token}, got {matches}"
        )
    return matches[0]


def copy_trusted_source(source: Path, target_dir: Path, module_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{module_name}.py"
    shutil.copy2(source, target)
    return target


def parse_identity(frame: pd.DataFrame) -> pd.DataFrame:
    ids = frame["id"].astype(str)
    split = ids.str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("candidate id must use <well>_<row_idx>")
    return pd.DataFrame(
        {
            "id": ids,
            "well": split[0].astype(str),
            "well_row_idx": pd.to_numeric(split[1], errors="raise").astype(np.int32),
        }
    )


def finalize_primitive_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    excluded = {"id", "well", "well_row_idx", "candidate_tvt", "confidence_valid"}
    native_fields = [column for column in result if column not in excluded]
    available: list[np.ndarray] = []
    for field in native_fields:
        values = pd.to_numeric(result[field], errors="coerce").to_numpy(np.float32)
        result[field] = values
        available.append(np.isfinite(values))
    candidate_finite = np.isfinite(result["candidate_tvt"].to_numpy(np.float32))
    result["confidence_valid"] = (
        candidate_finite & np.logical_or.reduce(available)
        if available
        else np.zeros(len(result), dtype=bool)
    )
    return result


def standard_primitive(
    frame: pd.DataFrame,
    value: Any,
    *,
    confidence: dict[str, Any] | None = None,
) -> pd.DataFrame:
    result = parse_identity(frame)
    result["candidate_tvt"] = np.asarray(value, dtype=np.float32)
    for field, field_value in (confidence or {}).items():
        result[field] = np.asarray(field_value, dtype=np.float32)
    return finalize_primitive_confidence(result)


def generate_hmm_primitive(
    *,
    list_well_ids: Callable[[str | Path], list[str]],
    load_well: Callable[[str, str | Path], tuple[pd.DataFrame, pd.DataFrame]],
    run_hmm2: Callable[..., dict[str, Any]],
    test_dir: Path,
    hmm_params: dict[str, Any],
    self_gr: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in list_well_ids(test_dir):
        horizontal, typewell = load_well(well, test_dir)
        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
        if not known.any():
            raise ValueError(f"raw test well {well} has no finite TVT_input prefix")
        expected_eval = np.flatnonzero(~known).astype(np.int64)
        if len(expected_eval) == 0:
            continue
        kwargs = dict(hmm_params)
        if self_gr is not None:
            kwargs.update(
                {
                    "self_gr_config": dict(self_gr["surface"]),
                    "self_gr_alpha": float(self_gr["alpha"]),
                    "self_gr_clip": float(self_gr["clip"]),
                    "self_gr_mode": str(self_gr["mode"]),
                }
            )
        inference = run_hmm2(horizontal, typewell, **kwargs)
        actual_eval = np.asarray(inference["ev_index"], dtype=np.int64)
        if not np.array_equal(actual_eval, expected_eval):
            raise ValueError(f"HMM eval identity mismatch for well {well}")
        row = pd.DataFrame(
            {
                "id": [f"{well}_{int(index)}" for index in actual_eval],
                "well": str(well),
                "well_row_idx": actual_eval.astype(np.int32),
                "candidate_tvt": np.asarray(inference["mean_eval"], dtype=np.float32),
                "sigma_tvt": np.asarray(inference["std_eval"], dtype=np.float32),
                "source_loglik": np.full(
                    len(actual_eval), np.float32(inference["loglik"]), dtype=np.float32
                ),
                "loglik_per_row": np.full(
                    len(actual_eval),
                    np.float32(float(inference["loglik"]) / len(actual_eval)),
                    dtype=np.float32,
                ),
            }
        )
        if self_gr is not None:
            row["candidate_finite_source"] = np.isfinite(
                np.asarray(inference["mean_eval"], dtype=np.float32)
            ).astype(np.float32)
            row["selfgr_quality"] = np.asarray(inference["self_gr_quality"], dtype=np.float32)
            row["selfgr_peak_tvt"] = np.asarray(
                inference["self_gr_peak_tvt"], dtype=np.float32
            )
            row["score_margin"] = np.asarray(
                inference["self_gr_peak_gap"], dtype=np.float32
            )
            row["selfgr_typewell_agreement"] = np.asarray(
                inference["self_gr_typewell_agreement"], dtype=np.float32
            )
            row["selfgr_valid"] = np.asarray(inference["self_gr_valid"], dtype=np.float32)
        rows.append(row)
    if not rows:
        raise ValueError("HMM raw-test generation produced no rows")
    result = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
    if result.duplicated("id").any() or not np.isfinite(result["candidate_tvt"]).all():
        raise ValueError("HMM raw-test output violates duplicate/finite contract")
    return result


def generate_k16_primitive(
    module: Any,
    *,
    train_dir: Path,
    test_dir: Path,
    source_config: dict[str, Any],
    frame_content_sha256: Callable[[pd.DataFrame], str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = module.params_from_config(source_config)
    max_train = get_nested(source_config, "inference.max_train_wells")
    max_test = get_nested(source_config, "inference.max_test_wells")
    train_wells = module.load_train_wells(
        train_dir,
        params,
        max_wells=int(max_train) if max_train is not None else None,
    )
    test_wells = module.load_test_wells(
        test_dir,
        params,
        max_wells=int(max_test) if max_test is not None else None,
    )
    if not train_wells or not test_wells:
        raise FileNotFoundError("exp226 K16 requires non-empty train and test wells")
    fields = module.build_fields(train_wells, params)
    kappa = module.fit_kappa(train_wells, fields, params)
    print("exp226 kappa:", np.round(kappa, 3), flush=True)
    rows: list[pd.DataFrame] = []
    well_summaries: list[dict[str, Any]] = []
    for order, well in enumerate(test_wells, start=1):
        inference = module.predict_well(well, fields, kappa, params)
        row_idx = np.arange(well.s + 1, well.s + well.n + 1, dtype=np.int32)
        if len(row_idx) != len(inference.pred) or len(inference.pred) != len(inference.delta):
            raise ValueError(f"exp226 K16 row contract mismatch for well={well.wid}")
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{well.wid}_{int(index)}" for index in row_idx],
                    "well": str(well.wid),
                    "well_row_idx": row_idx,
                    "candidate_tvt": np.asarray(inference.pred, dtype=np.float32),
                    "geometry_gr_delta": np.asarray(inference.delta, dtype=np.float32),
                }
            )
        )
        summary = dict(inference.summary)
        summary["order"] = order
        well_summaries.append(summary)
    result = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
    if result.duplicated("id").any() or not np.isfinite(
        result[["candidate_tvt", "geometry_gr_delta"]].to_numpy()
    ).all():
        raise ValueError("exp226 K16 output violates duplicate/finite confidence contract")
    return result, {
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "rows": len(result),
        "kappa": [float(value) for value in np.asarray(kappa).ravel()],
        "well_summaries": well_summaries,
        "prediction_and_confidence_content_sha256": frame_content_sha256(result),
    }


def safe_extract_tar(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination.resolve() not in (target, *target.parents):
                raise RuntimeError(f"unsafe Stage C bundle member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"links are forbidden in Stage C bundle: {member.name}")
        handle.extractall(destination, members=members)


# %% [markdown]
# ## 2. User authorization and saved-model contracts
#
# exp372のincremental/tail/promotion gate FAILは保持する。2026-07-25のユーザー指示を、
# 保存modelによるCPU推論とsubmission file生成だけの例外承認として扱う。
# 再学習、same-OOF rescue、外部competition submitへは拡張しない。

# %%
inference_cfg = dict(config["inference"])
if not bool(config["execution"]["inference_approved"]):
    raise RuntimeError("exp372 inference is not approved")
if not bool(config["execution"]["run_inference"]):
    raise RuntimeError("exp372 inference run flag is disabled")
if inference_cfg.get("status") != (
    "user_authorized_2026_07_25_cpu_after_guard_failure"
):
    raise RuntimeError("exp372 CPU inference lacks the fixed user authorization")
if not bool(inference_cfg.get("technical_gate_passed")):
    raise RuntimeError("exp372 technical gate PASS must remain recorded")
if bool(inference_cfg.get("incremental_utility_gate_passed")):
    raise RuntimeError("exp372 incremental gate FAIL must remain explicit")
if bool(inference_cfg.get("tail_promotion_gate_passed")):
    raise RuntimeError("exp372 tail gate FAIL must remain explicit")
if bool(inference_cfg.get("promotion_gate_passed")):
    raise RuntimeError("exp372 promotion FAIL must remain explicit")
if str(inference_cfg.get("runtime")) != "kaggle_cpu":
    raise RuntimeError("exp372 inference must use Kaggle CPU")
if bool(inference_cfg.get("submit_to_kaggle")):
    raise RuntimeError("the inference notebook must not call the Kaggle submit API")
if bool(inference_cfg.get("competition_submit_authorized")):
    raise RuntimeError("external Kaggle competition submit must remain unauthorized")
if int(inference_cfg.get("booster_training_count", -1)) != 0:
    raise RuntimeError("inference must train zero boosters")

candidate_contract_path = PACKAGE_DIR / "inputs/exp264_candidate_contract.yaml"
candidate_contract = read_yaml(candidate_contract_path)
names = candidate_ids(candidate_contract)
if len(names) != int(inference_cfg["candidate_count"]):
    raise ValueError("exp372 inference requires exactly 12 candidates")
sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})

configured_bundle_path = Path(str(inference_cfg["parent_selector_model_bundle"]))
bundle_candidates = [PACKAGE_DIR / configured_bundle_path]
if KAGGLE_INPUT_ROOT.exists():
    bundle_candidates.extend(KAGGLE_INPUT_ROOT.rglob(configured_bundle_path.name))
stage_c_bundle_matches = [path for path in bundle_candidates if path.exists()]
if len(stage_c_bundle_matches) != 1:
    raise FileNotFoundError(
        "expected exactly one Stage C inference bundle, got "
        f"{stage_c_bundle_matches}"
    )
stage_c_bundle = stage_c_bundle_matches[0]
if sha256_file(stage_c_bundle) != str(
    inference_cfg["parent_selector_model_bundle_sha256"]
):
    raise ValueError("Stage C inference bundle SHA mismatch")
stage_c_root = Path("/tmp/exp264_stage_c_inference_models")
safe_extract_tar(stage_c_bundle, stage_c_root)
stage_c_manifest_path = stage_c_root / "nested_selector_model_manifest.json"
selector_schema_path = stage_c_root / "feature_schema.json"
compact_schema_path = stage_c_root / "compact_meta_schema.json"
selector_catalog_path = PACKAGE_DIR / str(inference_cfg["selector_feature_catalog"])
expected_stage_c_files = {
    stage_c_manifest_path: inference_cfg["parent_selector_model_manifest_sha256"],
    selector_schema_path: inference_cfg["selector_feature_schema_file_sha256"],
    compact_schema_path: inference_cfg["parent_compact_schema_file_sha256"],
}
for artifact_path, expected_sha in expected_stage_c_files.items():
    if sha256_file(artifact_path) != str(expected_sha):
        raise ValueError(f"Stage C contract SHA mismatch: {artifact_path.name}")
if sha256_file(selector_catalog_path) != str(
    inference_cfg["selector_feature_catalog_sha256"]
):
    raise ValueError("Stage A selector feature catalog SHA mismatch")

stage_c_manifest = json.loads(stage_c_manifest_path.read_text())
selector_schema = load_feature_schema(selector_schema_path)
selector_features = [str(item) for item in selector_schema["features"]]
compact_schema = json.loads(compact_schema_path.read_text())
parent_compact_features = [str(item) for item in compact_schema["features"]]
if stage_c_manifest.get("candidate_order") != names:
    raise ValueError("Stage C candidate order differs from exp264 contract")
if int(stage_c_manifest.get("model_count", -1)) != int(
    inference_cfg["parent_selector_model_count"]
):
    raise ValueError("Stage C manifest must contain 40 selector models")
if stage_c_manifest.get("feature_schema_sha256") != selector_schema.get(
    "feature_schema_sha256"
):
    raise ValueError("Stage C selector schema logical SHA mismatch")
if compact_schema.get("compact_meta_schema_sha256") != str(
    inference_cfg["parent_compact_schema_logical_sha256"]
):
    raise ValueError("Stage C compact schema logical SHA mismatch")
if len(selector_features) != int(inference_cfg["expected_selector_feature_count"]):
    raise ValueError("Stage C selector feature count mismatch")
if len(parent_compact_features) != int(
    inference_cfg["expected_parent_compact_feature_count"]
):
    raise ValueError("Stage C compact feature count mismatch")

selector_catalog = pd.read_csv(selector_catalog_path)
selected_mask = selector_catalog["selected"].astype(str).str.lower().eq("true")
selected_catalog = selector_catalog.loc[selected_mask].copy()
catalog_features = selected_catalog["feature"].astype(str).tolist()
if catalog_features != selector_features or selected_catalog["feature"].duplicated().any():
    raise ValueError("Stage A selected feature catalog differs from Stage C schema")
selected_catalog["missing_rate"] = pd.to_numeric(
    selected_catalog["missing_rate"], errors="raise"
)
training_missing_rate_by_feature = dict(
    zip(
        catalog_features,
        selected_catalog["missing_rate"].astype(float),
        strict=True,
    )
)
training_sparse_feature_count = int(
    (selected_catalog["missing_rate"] > 0.0).sum()
)
if training_sparse_feature_count != int(
    inference_cfg["selector_training_sparse_feature_count"]
):
    raise ValueError("Stage A selector sparse-feature count mismatch")

selector_models: dict[int, dict[str, list[lgb.Booster]]] = {
    outer: {"pred_abs_error": [], "p_within10": []} for outer in range(5)
}
selector_model_audit: list[dict[str, Any]] = []
for item in stage_c_manifest["models"]:
    outer = int(item["downstream_outer_fold"])
    objective = str(item["objective"])
    model_path = stage_c_root / str(item["path"])
    if sha256_file(model_path) != str(item["sha256"]):
        raise ValueError(f"Stage C selector model SHA mismatch: {model_path.name}")
    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != selector_features:
        raise ValueError(f"Stage C selector feature order mismatch: {model_path.name}")
    selector_models[outer][objective].append(booster)
    selector_model_audit.append(
        {
            "outer_fold": outer,
            "inner_fold": int(item["inner_fold"]),
            "objective": objective,
            "file": model_path.name,
            "sha256": str(item["sha256"]),
            "best_iteration": int(item["best_iteration"]),
        }
    )
for outer, by_objective in selector_models.items():
    for objective, models in by_objective.items():
        if len(models) != int(
            inference_cfg["parent_selector_models_per_outer_objective"]
        ):
            raise ValueError(f"Stage C model coverage mismatch: outer={outer} {objective}")

search_roots = [KAGGLE_INPUT_ROOT, Path("/tmp"), paths.root]
signed_manifest_path = resolve_existing_path(
    [
        str(item)
        for item in inference_cfg["signed_selector_model_manifest_patterns"]
    ],
    search_roots,
)
if sha256_file(signed_manifest_path) != str(
    inference_cfg["signed_selector_model_manifest_sha256"]
):
    raise ValueError("Stage S signed-selector model manifest SHA mismatch")
signed_schema_path = resolve_existing_path(
    [str(item) for item in inference_cfg["signed_compact_schema_patterns"]],
    search_roots,
)
if sha256_file(signed_schema_path) != str(
    inference_cfg["signed_compact_schema_file_sha256"]
):
    raise ValueError("Stage S signed compact schema file SHA mismatch")
signed_manifest = json.loads(signed_manifest_path.read_text())
signed_schema = json.loads(signed_schema_path.read_text())
signed_compact_features = [str(item) for item in signed_schema["features"]]
if signed_manifest.get("candidate_order") != names:
    raise ValueError("Stage S candidate order differs from exp335 contract")
if int(signed_manifest.get("model_count", -1)) != int(
    inference_cfg["signed_selector_model_count"]
):
    raise ValueError("Stage S manifest must contain exactly 20 signed selectors")
if signed_manifest.get("feature_schema_sha256") != selector_schema.get(
    "feature_schema_sha256"
):
    raise ValueError("Stage S selector feature schema differs from corrected Stage A")
if signed_schema.get("signed_compact_schema_sha256") != str(
    inference_cfg["signed_compact_schema_logical_sha256"]
):
    raise ValueError("Stage S signed compact logical schema SHA mismatch")
if signed_compact_features != signed_compact_feature_names(candidate_contract):
    raise ValueError("Stage S signed compact schema differs from source contract")
if len(signed_compact_features) != int(
    inference_cfg["expected_signed_compact_feature_count"]
):
    raise ValueError("Stage S signed compact feature count mismatch")

signed_selector_models: dict[int, list[lgb.Booster]] = {
    outer: [] for outer in range(5)
}
signed_selector_model_audit: list[dict[str, Any]] = []
for item in signed_manifest["models"]:
    outer = int(item["downstream_outer_fold"])
    model_path = signed_manifest_path.parent / str(item["path"])
    if sha256_file(model_path) != str(item["sha256"]):
        raise ValueError(f"Stage S signed-selector model SHA mismatch: {model_path.name}")
    booster = lgb.Booster(model_file=str(model_path))
    if list(booster.feature_name()) != selector_features:
        raise ValueError(
            f"Stage S signed-selector feature order mismatch: {model_path.name}"
        )
    signed_selector_models[outer].append(booster)
    signed_selector_model_audit.append(
        {
            "outer_fold": outer,
            "inner_fold": int(item["inner_fold"]),
            "objective": str(item["objective"]),
            "file": model_path.name,
            "sha256": str(item["sha256"]),
            "best_iteration": int(item["best_iteration"]),
        }
    )
for outer, models in signed_selector_models.items():
    if len(models) != int(inference_cfg["signed_selector_models_per_outer"]):
        raise ValueError(f"Stage S signed model coverage mismatch: outer={outer}")

stage_d_manifest_path = resolve_existing_path(
    [str(item) for item in inference_cfg["tvt_model_manifest_patterns"]], search_roots
)
if sha256_file(stage_d_manifest_path) != str(inference_cfg["tvt_model_manifest_sha256"]):
    raise ValueError("exp372 union model manifest SHA mismatch")
stage_d_manifest = json.loads(stage_d_manifest_path.read_text())
if stage_d_manifest.get("status") != str(
    inference_cfg["tvt_model_manifest_status"]
):
    raise ValueError("exp372 union model manifest completion status mismatch")
if int(stage_d_manifest.get("feature_count", -1)) != int(
    inference_cfg["expected_final_feature_count"]
):
    raise ValueError("exp372 model manifest feature count mismatch")
if stage_d_manifest.get("feature_schema_sha256") != str(
    inference_cfg["tvt_model_feature_schema_sha256"]
):
    raise ValueError("exp372 model manifest logical feature SHA mismatch")
stage_d_rows = [
    dict(item)
    for item in stage_d_manifest["models"]
    if str(item["variant"]) == str(inference_cfg["tvt_model_variant"])
]
if len(stage_d_rows) != int(inference_cfg["tvt_model_count"]):
    raise ValueError("exp372 inference requires exactly 15 union models")
resolved_tvt_models: list[tuple[dict[str, Any], Path]] = []
for item in stage_d_rows:
    model_path = stage_d_manifest_path.parent / str(item["path"])
    if sha256_file(model_path) != str(item["sha256"]):
        raise ValueError(f"exp372 TVT model SHA mismatch: {model_path.name}")
    resolved_tvt_models.append((item, model_path))

schema_probe = lgb.Booster(model_file=str(resolved_tvt_models[0][1]))
final_feature_columns = list(schema_probe.feature_name())
del schema_probe
base_feature_count = int(inference_cfg["expected_base_feature_count"])
base_feature_columns = final_feature_columns[:base_feature_count]
parent_feature_stop = base_feature_count + int(
    inference_cfg["expected_parent_compact_feature_count"]
)
formation_feature_stop = parent_feature_stop + int(
    inference_cfg["expected_formation_feature_count"]
)
model_parent_compact_features = final_feature_columns[
    base_feature_count:parent_feature_stop
]
model_formation_features = final_feature_columns[
    parent_feature_stop:formation_feature_stop
]
model_signed_compact_features = final_feature_columns[formation_feature_stop:]
source_base_catalog_path = PACKAGE_DIR / str(
    inference_cfg["source_base_feature_catalog"]
)
if sha256_file(source_base_catalog_path) != str(
    inference_cfg["source_base_feature_catalog_sha256"]
):
    raise ValueError("exp218 source 380 feature catalog SHA mismatch")
source_base_catalog = pd.read_csv(source_base_catalog_path)
source_base_columns = source_base_catalog["feature"].astype(str).tolist()
if len(source_base_columns) != int(inference_cfg["expected_source_base_feature_count"]):
    raise ValueError("exp218 source feature count mismatch")
if len(source_base_columns) != len(set(source_base_columns)):
    raise ValueError("exp218 source feature catalog contains duplicates")
base_allowlist_path = PACKAGE_DIR / str(inference_cfg["base_feature_allowlist"])
if sha256_file(base_allowlist_path) != str(
    inference_cfg["base_feature_allowlist_sha256"]
):
    raise ValueError("clean 273 base feature allowlist SHA mismatch")
base_allowlist = pd.read_csv(base_allowlist_path)["feature"].astype(str).tolist()
if len(base_allowlist) != base_feature_count or len(base_allowlist) != len(
    set(base_allowlist)
):
    raise ValueError("clean 273 base feature allowlist count/uniqueness mismatch")
if base_feature_columns != base_allowlist:
    raise ValueError("Stage D model base feature order differs from clean 273 allowlist")
safe_catalog = source_base_catalog.loc[
    source_base_catalog["fold_safe"].astype(str).str.lower().eq("true")
    & source_base_catalog["hidden_safe"].astype(str).str.lower().eq("true"),
    "feature",
].astype(str).tolist()
if safe_catalog != base_allowlist:
    raise ValueError("clean 273 allowlist differs from source feature safety catalog")
formation_features = canonical_formation_feature_names()
if len(formation_features) != int(inference_cfg["expected_formation_feature_count"]):
    raise ValueError("canonical formation feature count mismatch")
if model_parent_compact_features != parent_compact_features:
    raise ValueError("exp372 model parent compact order differs from saved74 schema")
if model_formation_features != formation_features:
    raise ValueError("exp372 model formation order differs from canonical formation74")
if model_signed_compact_features != signed_compact_features:
    raise ValueError(
        "exp372 model signed compact order differs from signed23 schema"
    )
if len(final_feature_columns) != int(inference_cfg["expected_final_feature_count"]):
    raise ValueError("exp372 final feature count mismatch")

display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": config["experiment"]["route"],
        "authorization": inference_cfg["authorization_scope"],
        "technical_gate_passed": True,
        "incremental_utility_gate_passed": False,
        "tail_promotion_gate_passed": False,
        "promotion_gate_passed": False,
        "retained_guard_failure": inference_cfg["retained_guard_failure"],
        "candidate_count": len(names),
        "parent_selector_models": len(selector_model_audit),
        "signed_selector_models": len(signed_selector_model_audit),
        "tvt_models": len(resolved_tvt_models),
        "base_features": len(base_feature_columns),
        "parent_compact_features": len(parent_compact_features),
        "formation_features": len(formation_features),
        "signed_compact_features": len(signed_compact_features),
        "final_features": len(final_feature_columns),
        "booster_training_count": 0,
        "runtime": "kaggle_cpu",
        "competition_submit_authorized": False,
    }
)

# %% [markdown]
# ## 3. Exp263 hidden-safe 12-candidate regeneration
#
# 保存済みpublic-test row artifactは使わない。exp263が固定したsource file名・Kaggle source token・
# parameterを用い、PF/Beam/likPF、exact/self-GR HMM、K16をraw testから再生成する。

# %%
exp263_source_dir = PACKAGE_DIR / "inputs/exp263_source"
sys.path.insert(0, str(exp263_source_dir))
from candidate_cache_builder import (  # noqa: E402
    assemble_stage1_current_test_parity,
    attach_stage1_current_test_confidence,
)
from candidate_cache_contract import (  # noqa: E402
    PAIR_SHORTLIST,
    RAWTEST_CORE_CANDIDATE_IDS,
    STAGE1_NATIVE_CONFIDENCE_FIELDS,
    validate_contract,
)
from candidate_cache_loader import frame_content_sha256  # noqa: E402

exp263_config = yaml.safe_load((exp263_source_dir / "config.yaml").read_text())
stage1 = dict(exp263_config["stage1"])
generation = dict(stage1["raw_test_generation"])
validate_contract()
rawtest_pairs = [pair for pair in PAIR_SHORTLIST if pair.tier == "raw-test"]
if len(RAWTEST_CORE_CANDIDATE_IDS) != 6 or len(rawtest_pairs) != 5:
    raise ValueError("exp263 Stage 1 deployability tier count mismatch")
expected_confidence_contract = {
    candidate_id: ["confidence_valid", *fields]
    for candidate_id, fields in STAGE1_NATIVE_CONFIDENCE_FIELDS.items()
}
if stage1["confidence_output"]["required_fields_by_primitive"] != (
    expected_confidence_contract
):
    raise ValueError("exp263 Stage 1 native-confidence contract mismatch")

source_work = Path("/tmp/exp264_trusted_upstream_sources")
if source_work.exists():
    shutil.rmtree(source_work)
source_specs = {
    "exp263_public_replay_source": generation["pf_replay"],
    "exp263_exact_hmm_source": generation["exact_hmm"],
    "exp263_selfgr_hmm_source": generation["selfgr_hmm_a070"],
    "exp263_k16_source": generation["exp226_k16"],
}
resolved_sources: dict[str, Path] = {}
for module_name, source_spec in source_specs.items():
    source = resolve_unique_source(
        str(source_spec["source_filename"]), str(source_spec["source_path_token"])
    )
    resolved_sources[module_name] = source
    copy_trusted_source(source, source_work, module_name)
sys.path.insert(0, str(source_work))

import exp263_k16_source as k16_module  # noqa: E402
from exp263_exact_hmm_source import list_well_ids as exact_list_well_ids  # noqa: E402
from exp263_exact_hmm_source import load_well as exact_load_well  # noqa: E402
from exp263_exact_hmm_source import run_hmm2 as exact_run_hmm2  # noqa: E402
from exp263_public_replay_source import (  # noqa: E402
    build_replay_test_frame,
    configure_public_runtime,
)
from exp263_selfgr_hmm_source import (  # noqa: E402
    list_well_ids as selfgr_list_well_ids,
)
from exp263_selfgr_hmm_source import load_well as selfgr_load_well  # noqa: E402
from exp263_selfgr_hmm_source import run_hmm2 as selfgr_run_hmm2  # noqa: E402

stage0_manifest_path = resolve_unique_source(
    "cache_manifest.json", "exp263-last-anchor-pair-cache-train"
)
if sha256_file(stage0_manifest_path) != str(
    stage1["stage0_manifest"]["expected_manifest_sha256"]
):
    raise ValueError("exp263 Stage 0 manifest SHA mismatch")

pf_config = generation["pf_replay"]
configure_public_runtime(
    data_dir=paths.raw_data_dir,
    output_dir=output_dir / "pf_replay",
    n_jobs=int(pf_config["n_jobs"]),
    pf_seeds=int(pf_config["pf_seeds"]),
    pf_particles=int(pf_config["pf_particles"]),
    fast=bool(pf_config["fast"]),
    use_gpu=str(pf_config["use_gpu"]),
)
pf_frame, pf_meta = build_replay_test_frame()
pf_frame["id"] = pf_frame["id"].astype(str)
pf_frame["well"] = pf_frame["well"].astype(str)
required_pf = {
    "id",
    "well",
    "last_known_tvt",
    "likpf_mean_d",
    "pf_ancc",
    "pf_ancc_std",
    "beam_mean_d",
    "beam_std_d",
}
if missing_pf := required_pf - set(pf_frame.columns):
    raise ValueError(f"exp073 raw-test replay columns missing: {sorted(missing_pf)}")

k16_source_config = resolved_sources["exp263_k16_source"].parent / str(
    generation["exp226_k16"]["source_config_filename"]
)
if not k16_source_config.exists():
    raise FileNotFoundError(f"exp226 source config missing: {k16_source_config}")
k16_frame, k16_summary = generate_k16_primitive(
    k16_module,
    train_dir=paths.train_data_dir,
    test_dir=paths.test_data_dir,
    source_config=yaml.safe_load(k16_source_config.read_text()),
    frame_content_sha256=frame_content_sha256,
)
exact_config = generation["exact_hmm"]
exact_frame = generate_hmm_primitive(
    list_well_ids=exact_list_well_ids,
    load_well=exact_load_well,
    run_hmm2=exact_run_hmm2,
    test_dir=paths.test_data_dir,
    hmm_params=dict(exact_config["params"]),
)
selfgr_config = generation["selfgr_hmm_a070"]
selfgr_frame = generate_hmm_primitive(
    list_well_ids=selfgr_list_well_ids,
    load_well=selfgr_load_well,
    run_hmm2=selfgr_run_hmm2,
    test_dir=paths.test_data_dir,
    hmm_params=dict(exact_config["params"]),
    self_gr=dict(selfgr_config),
)
primitive_frames = {
    "exp226_k16": k16_frame,
    "selfgr_hmm_a070": selfgr_frame,
    "likpf_mean": standard_primitive(
        pf_frame,
        pf_frame["last_known_tvt"].to_numpy(np.float32)
        + pf_frame["likpf_mean_d"].to_numpy(np.float32),
    ),
    "exact_hmm": exact_frame,
    "pf_ancc": standard_primitive(
        pf_frame,
        pf_frame["pf_ancc"],
        confidence={"sigma_tvt": pf_frame["pf_ancc_std"]},
    ),
    "beam_mean": standard_primitive(
        pf_frame,
        pf_frame["last_known_tvt"].to_numpy(np.float32)
        + pf_frame["beam_mean_d"].to_numpy(np.float32),
        confidence={"beam_family_std": pf_frame["beam_std_d"]},
    ),
}
formula_frame, max_abs_formula = assemble_stage1_current_test_parity(primitive_frames)
formula_frame = attach_stage1_current_test_confidence(formula_frame, primitive_frames)
formula_frame = formula_frame.sort_values(["well", "well_row_idx"], kind="stable").reset_index(
    drop=True
)
formula_path = output_dir / "current_test_formula_parity.parquet"
formula_frame.to_parquet(formula_path, index=False, compression="zstd")
confidence_parity = validate_current_test_native_confidence(
    formula_frame, candidate_contract
)
if int(confidence_parity["required_column_count"]) != int(
    inference_cfg["required_namespaced_confidence_column_count"]
):
    raise ValueError("exp263 current-test confidence column count mismatch")
if sum(column.startswith("confidence__") for column in formula_frame) != 21:
    raise ValueError("exp263 Stage 1 must export exactly 21 confidence columns")
if set(formula_frame["id"].astype(str)) != set(sample["id"]):
    raise ValueError("generated exp263 candidate IDs differ from sample submission")

source_audit = {name: source_record(path) for name, path in resolved_sources.items()}
source_audit["exp263_stage0_manifest"] = source_record(stage0_manifest_path)
primitive_content_sha = {
    candidate_id: frame_content_sha256(frame)
    for candidate_id, frame in primitive_frames.items()
}
display(
    {
        "rows": len(formula_frame),
        "wells": int(formula_frame["well"].nunique()),
        "primitive_count": len(primitive_frames),
        "pair_count": len(rawtest_pairs),
        "candidate_count": len(names),
        "confidence_columns": 21,
        "formula_max_abs_error": float(max_abs_formula),
    }
)
display(formula_frame[["id", "well", *names]].head())

# %% [markdown]
# ## 4. Candidate-long context, parent compact, and signed compact features
#
# candidate-long matrixはchunkごとに一度だけ作る。同じmatrixへouter-fold別8 parent selectorと
# 4 signed selectorを適用し、saved74とsigned23を同じcandidate/order/outer契約で生成する。

# %%
bundle = current_test_bundle_from_wide(formula_frame, candidate_contract)
fill_current_test_anchor(bundle, paths.test_data_dir)
feature_cfg = dict(config["features"])
feature_cfg["primary_domain"] = candidate_contract["legal_domains"][
    "primitive_pair_bank"
]["candidates"]
feature_cfg["fixed_domain"] = candidate_contract["legal_domains"][
    "primitive_fixed_bank"
]["candidates"]
raw_context, truth = build_raw_context(
    bundle.base, paths.test_data_dir, feature_cfg, require_truth=False
)
if truth is not None:
    raise RuntimeError("current-test selector context unexpectedly contains truth")

shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
chunk_size = int(inference_cfg["selector_predict_base_row_chunk_size"])
compact_parts: dict[int, list[pd.DataFrame]] = {outer: [] for outer in range(5)}
signed_compact_parts: dict[int, list[pd.DataFrame]] = {
    outer: [] for outer in range(5)
}
signed_top1_parity_max = 0.0
score_sample: pd.DataFrame | None = None
selector_missing_count_by_feature = np.zeros(len(selector_features), dtype=np.int64)
selector_missing_count_by_candidate = np.zeros(len(names), dtype=np.int64)
selector_long_row_count = 0
for start in range(0, len(bundle.base), chunk_size):
    stop = min(start + chunk_size, len(bundle.base))
    indices = np.arange(start, stop, dtype=np.int64)
    long_frame, metadata = build_candidate_long_features(
        bundle,
        raw_context,
        indices,
        feature_cfg,
        shape_state=shape_state,
        expected_features=selector_features,
    )
    matrix = long_frame.to_numpy(np.float32, copy=False)
    missingness_chunk = validate_inference_feature_missingness(
        long_frame,
        training_missing_rate_by_feature,
        context=f"current-test selector rows {start}:{stop}",
    )
    selector_missing_count_by_feature += missingness_chunk["missing_count"].to_numpy(
        np.int64
    )
    missing_tensor = np.isnan(matrix).reshape(
        len(indices), len(names), len(selector_features)
    )
    selector_missing_count_by_candidate += missing_tensor.sum(axis=(0, 2)).astype(
        np.int64
    )
    selector_long_row_count += len(long_frame)
    confidence_valid = metadata["confidence_valid"].to_numpy(bool).reshape(
        len(indices), len(names)
    )
    for outer in range(5):
        p = np.mean(
            [
                model.predict(matrix, num_iteration=model.best_iteration)
                for model in selector_models[outer]["p_within10"]
            ],
            axis=0,
        ).reshape(len(indices), len(names)).astype(np.float32)
        e = np.mean(
            [
                model.predict(matrix, num_iteration=model.best_iteration)
                for model in selector_models[outer]["pred_abs_error"]
            ],
            axis=0,
        ).reshape(len(indices), len(names)).astype(np.float32)
        e = np.maximum(e, 0.0)
        if not np.isfinite(e).all() or not np.isfinite(p).all():
            raise ValueError(f"Stage C selector scores are non-finite for outer fold {outer}")
        parent_compact = build_compact_meta(
            bundle.base.iloc[indices].reset_index(drop=True),
            bundle.values[indices],
            e,
            p,
            bundle.available[indices],
            confidence_valid,
            candidate_contract,
        )
        compact_parts[outer].append(parent_compact)
        signed_prediction = np.mean(
            [
                model.predict(matrix, num_iteration=model.best_iteration)
                for model in signed_selector_models[outer]
            ],
            axis=0,
        ).reshape(len(indices), len(names)).astype(np.float32)
        if not np.isfinite(signed_prediction).all():
            raise ValueError(
                f"Stage S signed selector scores are non-finite for outer fold {outer}"
            )
        signed_compact, signed_evidence = build_signed_compact_meta(
            bundle.base.iloc[indices].reset_index(drop=True),
            bundle.values[indices],
            signed_prediction,
            parent_compact,
            candidate_contract,
            top1_value_atol=float(inference_cfg["signed_top1_value_parity_atol"]),
        )
        signed_compact_parts[outer].append(signed_compact)
        signed_top1_parity_max = max(
            signed_top1_parity_max,
            float(signed_evidence["top1_value_parity_max_abs_error"]),
        )
        if outer == 0 and score_sample is None:
            take = min(
                len(metadata), int(inference_cfg["score_sample_rows"])
            )
            score_sample = metadata.iloc[:take].copy()
            score_sample["pred_abs_error"] = e.reshape(-1)[:take]
            score_sample["p_within10"] = p.reshape(-1)[:take]
            score_sample["pred_signed_residual"] = signed_prediction.reshape(-1)[:take]
            score_sample["downstream_outer_fold"] = np.int8(outer)
        del parent_compact, signed_prediction, signed_compact, signed_evidence
    del long_frame, metadata, matrix, confidence_valid, missingness_chunk, missing_tensor
    gc.collect()

selector_missingness = selected_catalog[
    ["feature", "group", "missing_rate"]
].rename(columns={"missing_rate": "training_missing_rate"})
selector_missingness["current_missing_count"] = selector_missing_count_by_feature
selector_missingness["current_missing_rate"] = (
    selector_missing_count_by_feature.astype(np.float64) / float(selector_long_row_count)
)
selector_missingness["structural_missingness"] = selector_missingness[
    "feature"
].str.startswith(("conf__", "formula__"))
all_missing_current = selector_missingness.loc[
    selector_missingness["current_missing_rate"].ge(1.0), "feature"
].tolist()
if all_missing_current:
    raise ValueError(
        f"current-test selector features became all-missing: {all_missing_current[:20]}"
    )
selector_missingness_path = output_dir / "selector_missingness_current_test.csv"
selector_missingness.to_csv(selector_missingness_path, index=False)

selector_candidate_missingness = pd.DataFrame(
    {
        "candidate_id": names,
        "missing_count": selector_missing_count_by_candidate,
        "missing_rate": selector_missing_count_by_candidate.astype(np.float64)
        / float(len(bundle.base) * len(selector_features)),
    }
)
selector_candidate_missingness_path = (
    output_dir / "selector_missingness_by_candidate_current_test.csv"
)
selector_candidate_missingness.to_csv(selector_candidate_missingness_path, index=False)
display(
    {
        "selector_training_sparse_features": training_sparse_feature_count,
        "selector_current_sparse_features": int(
            selector_missingness["current_missing_count"].gt(0).sum()
        ),
        "selector_current_missing_cells": int(selector_missing_count_by_feature.sum()),
        "selector_infinite_cells": 0,
    }
)
display(
    selector_missingness.sort_values(
        ["current_missing_rate", "feature"], ascending=[False, True]
    ).head(40)
)
display(selector_candidate_missingness)

compact_by_outer: dict[int, pd.DataFrame] = {}
compact_sha: dict[str, str] = {}
signed_compact_by_outer: dict[int, pd.DataFrame] = {}
signed_compact_sha: dict[str, str] = {}
for outer in range(5):
    compact = pd.concat(compact_parts[outer], ignore_index=True)
    if len(compact) != len(bundle.base):
        raise ValueError(f"compact row coverage mismatch for outer fold {outer}")
    if [
        column for column in compact if column.startswith("selector__")
    ] != parent_compact_features:
        raise ValueError(f"compact schema mismatch for outer fold {outer}")
    if not np.isfinite(
        compact[parent_compact_features].to_numpy(np.float32)
    ).all():
        raise ValueError(f"compact features are non-finite for outer fold {outer}")
    compact_path = output_dir / f"parent_compact_current_test_outer{outer}.parquet"
    compact.to_parquet(compact_path, index=False, compression="zstd")
    compact_sha[str(outer)] = sha256_file(compact_path)
    compact_by_outer[outer] = compact
    signed_compact = pd.concat(signed_compact_parts[outer], ignore_index=True)
    if len(signed_compact) != len(bundle.base):
        raise ValueError(f"signed compact row coverage mismatch for outer fold {outer}")
    if [
        column for column in signed_compact if column.startswith("selector__")
    ] != signed_compact_features:
        raise ValueError(f"signed compact schema mismatch for outer fold {outer}")
    if not np.isfinite(
        signed_compact[signed_compact_features].to_numpy(np.float32)
    ).all():
        raise ValueError(
            f"signed compact features are non-finite for outer fold {outer}"
        )
    signed_path = output_dir / f"signed_compact_current_test_outer{outer}.parquet"
    signed_compact.to_parquet(signed_path, index=False, compression="zstd")
    signed_compact_sha[str(outer)] = sha256_file(signed_path)
    signed_compact_by_outer[outer] = signed_compact
if score_sample is None:
    raise RuntimeError("selector score sample was not generated")
score_sample_path = output_dir / "candidate_score_sample_outer0.parquet"
score_sample.to_parquet(score_sample_path, index=False, compression="zstd")
del compact_parts, signed_compact_parts, score_sample
gc.collect()
display(compact_by_outer[0].head())
display(signed_compact_by_outer[0].head())

# %% [markdown]
# ## 5. Exp218 current-test clean 273-feature surface
#
# exp263 replay frameを共通baseとし、anchor、U projection、exp145 learned likelihood、GRWRを
# current testから再計算する。保存済みpublic-test feature artifactは入力に使わず、Stage Dモデルの
# clean 273 allowlistと列順が一致する特徴だけを使う。

# %%
exp218 = import_file(
    "exp264_inference_exp218",
    [
        PACKAGE_DIR / "inputs/exp218_source/gr_wavelet_rotation_confidence_features_on_exp148.py",
        Path(
            "experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
            "gr_wavelet_rotation_confidence_features_on_exp148.py"
        ),
    ],
)
exp218_config = yaml.safe_load(
    (PACKAGE_DIR / "inputs/exp218_source/config.yaml").read_text()
)
exp145_source_dir = PACKAGE_DIR / "inputs/exp145_source"
exp145_settings = import_file(
    "exp264_inference_exp145_settings",
    [exp145_source_dir / "settings.py"],
)
original_settings_module = sys.modules.get("settings")
sys.modules["settings"] = exp145_settings
exp145 = import_file(
    "exp264_inference_exp145",
    [exp145_source_dir / "learned_likelihood_rawtest_feature_generator_parity.py"],
)

exp145_config = exp145.load_config()
exp145_candidates = exp145.candidate_specs_from_config(exp145_config)
learned_source_frame = exp145.ensure_candidate_value_columns(
    pf_frame.copy(), exp145_candidates
)
learned_cache_path = output_dir / "exp263_replay_for_exp145.csv.gz"
learned_source_frame.to_csv(learned_cache_path, index=False, compression="gzip")
learned_output_dir = output_dir / "exp145_current_test"
sys.modules["settings"] = exp145_settings
learned_generator_summary = exp145.run_generator(
    output_dir=learned_output_dir,
    mode="rawtest",
    train_cache_path=None,
    rawtest_cache_path=learned_cache_path,
    exp111_schema_path=None,
    exp111_manifest_path=None,
    exp112_schema_path=None,
    max_rows=None,
)
if original_settings_module is not None:
    sys.modules["settings"] = original_settings_module
else:
    sys.modules.pop("settings", None)
if not bool(learned_generator_summary["generated_schema"]["schema_parity_pass"]):
    raise ValueError("exp145 current-test learned feature schema parity failed")
learned_feature_path = Path(
    learned_generator_summary["outputs"]["rawtest_ml_features"]["path"]
)
learned_source = pd.read_csv(learned_feature_path, dtype={"id": str, "well": str})

test_frame, anchor_meta = exp218.add_inference_anchor_columns(
    pf_frame.copy(), paths.test_data_dir
)
projection_cfg = get_nested(exp218_config, "model.u_projection", {}) or {}
projection, _, _ = exp218.build_u_projection_features(
    test_frame,
    source_specs=dict(projection_cfg.get("sources") or {}),
    degree=int(projection_cfg.get("degree", 3)),
    robust_iters=int(projection_cfg.get("robust_iters", 3)),
    clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
)
projection_columns = [column for column in projection if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    test_frame, projection.reset_index(drop=True), projection_columns
)
if not exp218.learned_feature_keys_match(learned_source, test_frame):
    raise ValueError("dynamic exp145 learned-feature keys differ from exp263 replay test")
learned, _, _ = exp218.build_learned_likelihood_features(
    learned_source,
    test_frame,
    get_nested(exp218_config, "model.learned_likelihood_features", {}) or {},
)
learned_columns = [column for column in learned if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    test_frame, learned.reset_index(drop=True), learned_columns
)
grwr, _, _, grwr_meta = exp218.build_gr_wavelet_rotation_confidence_features(
    test_frame,
    train_dir=paths.test_data_dir,
    config=get_nested(exp218_config, "model.gr_wavelet_rotation_confidence_features", {})
    or {},
)
grwr_columns = [column for column in grwr if column not in {"id", "well"}]
exp218._assign_aligned_float32_columns(
    test_frame, grwr.reset_index(drop=True), grwr_columns
)
missing_source_base = [column for column in source_base_columns if column not in test_frame]
if missing_source_base:
    raise ValueError(
        "raw-test exp218 surface missing source features: "
        f"{missing_source_base[:40]}"
    )
for start in range(0, len(source_base_columns), 32):
    columns = source_base_columns[start : start + 32]
    if not np.isfinite(test_frame[columns].to_numpy(np.float32, copy=False)).all():
        raise ValueError(f"raw-test exp218 features contain non-finite values: {columns}")
if set(test_frame["id"].astype(str)) != set(formula_frame["id"].astype(str)):
    raise ValueError("exp218 and exp263 current-test ID sets differ")
display(
    {
        "rows": len(test_frame),
        "wells": int(test_frame["well"].nunique()),
        "source_base_feature_count": len(source_base_columns),
        "base_feature_count": len(base_feature_columns),
        "learned_schema_parity": True,
    }
)
del projection, learned_source, learned, grwr, learned_source_frame
gc.collect()

# %% [markdown]
# ## 6. Current-test all-train-reference formation features
#
# 全773 train wellsからFormationPlaneKNN / DenseANCCImputer referenceを構築する。
# target側はraw testのMD/X/Y/Z/TVT_inputだけを読み、formation列は読まない。
# public testでtrainとwell名が重なる場合は、そのwellのtrain referenceだけをself-excludeする。

# %%
train_reference_wells = sorted(
    path.name.removesuffix("__horizontal_well.csv")
    for path in paths.train_data_dir.glob("*__horizontal_well.csv")
)
if len(train_reference_wells) != int(config["validation"]["expected_wells"]):
    raise ValueError("exp372 inference train-reference well count mismatch")
formation_surface, formation_generation = build_current_test_formation_surface(
    base_frame=test_frame,
    raw_train_dir=paths.train_data_dir,
    raw_test_dir=paths.test_data_dir,
    reference_wells=train_reference_wells,
    feature_names=formation_features,
    generator_config=config["formation_generator"],
)
if not formation_surface[["id", "well"]].equals(
    test_frame[["id", "well"]].reset_index(drop=True)
):
    raise ValueError("exp372 current-test formation alignment mismatch")
if not np.isfinite(
    formation_surface[formation_features].to_numpy(np.float32, copy=False)
).all():
    raise ValueError("exp372 current-test formation features are non-finite")
formation_path = output_dir / "fold_safe_formation_current_test.parquet"
formation_surface.to_parquet(formation_path, index=False, compression="zstd")
formation_generation["parquet_sha256"] = sha256_file(formation_path)
formation_generation["logical_content_sha256"] = logical_feature_content_sha256(
    formation_surface, formation_features
)
display(formation_generation)
display(formation_surface.head())

# %% [markdown]
# ## 7. Exp372 union saved-booster CPU inference
#
# 各TVT modelへ、学習時と同じouter foldのsaved74 + signed23と、共通の
# all-train-reference formation74を渡す。GPU学習済みtext modelをCPU predictorで読み、
# 3 config × 5 foldの15 residual predictionを等重み平均する。

# %%
pred_delta = np.zeros(len(test_frame), dtype=np.float32)
component_predictions: dict[str, np.ndarray] = {}
tvt_model_audit: list[dict[str, Any]] = []
for outer in range(5):
    compact = compact_by_outer[outer]
    signed_compact = signed_compact_by_outer[outer]
    aligned_compact = test_frame[["id"]].merge(
        compact[["id", *parent_compact_features]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    aligned_signed = test_frame[["id"]].merge(
        signed_compact[["id", *signed_compact_features]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if aligned_compact[parent_compact_features].isna().any().any():
        raise ValueError(
            f"parent compact alignment introduced missing values for outer fold {outer}"
        )
    if aligned_signed[signed_compact_features].isna().any().any():
        raise ValueError(
            f"signed compact alignment introduced missing values for outer fold {outer}"
        )
    matrix_frame = pd.concat(
        [
            test_frame[base_feature_columns].reset_index(drop=True),
            aligned_compact[parent_compact_features].reset_index(drop=True),
            formation_surface[formation_features].reset_index(drop=True),
            aligned_signed[signed_compact_features].reset_index(drop=True),
        ],
        axis=1,
    )
    if list(matrix_frame.columns) != final_feature_columns:
        raise ValueError(f"exp372 feature order mismatch for outer fold {outer}")
    matrix = matrix_frame.to_numpy(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"exp372 feature matrix is non-finite for outer fold {outer}")
    fold_models = [
        (item, model_path)
        for item, model_path in resolved_tvt_models
        if int(item["outer_fold"]) == outer
    ]
    if len(fold_models) != 3:
        raise ValueError(f"exp372 outer fold {outer} must have three union models")
    for item, model_path in fold_models:
        booster = lgb.Booster(model_file=str(model_path))
        if list(booster.feature_name()) != final_feature_columns:
            raise ValueError(f"exp372 model feature schema mismatch: {model_path.name}")
        prediction = booster.predict(
            matrix, num_iteration=int(item["best_iteration"])
        ).astype(np.float32)
        if not np.isfinite(prediction).all():
            raise ValueError(f"exp372 model prediction is non-finite: {model_path.name}")
        key = f"pred_delta__{item['model']}__outer{outer}"
        component_predictions[key] = prediction
        pred_delta += prediction / np.float32(len(resolved_tvt_models))
        tvt_model_audit.append(
            {
                "model": str(item["model"]),
                "config_index": int(item["config_index"]),
                "outer_fold": outer,
                "selector_score_outer_fold": outer,
                "file": model_path.name,
                "sha256": str(item["sha256"]),
                "best_iteration": int(item["best_iteration"]),
            }
        )
        del booster, prediction
        gc.collect()
    del compact, signed_compact, aligned_compact, aligned_signed, matrix_frame, matrix
    gc.collect()
if len(tvt_model_audit) != 15:
    raise ValueError("exp372 inference did not use all 15 union models")
pred_tvt = test_frame["last_known_tvt"].to_numpy(np.float32) + pred_delta
if not np.isfinite(pred_tvt).all():
    raise ValueError("final exp372 TVT prediction contains non-finite values")
predictions = pd.DataFrame(
    {
        "id": test_frame["id"].astype(str),
        "well": test_frame["well"].astype(str),
        "last_known_tvt": test_frame["last_known_tvt"].to_numpy(np.float32),
        "pred_delta": pred_delta,
        "pred_tvt": pred_tvt,
        **component_predictions,
    }
)

# %% [markdown]
# ## 8. Prediction artifacts and reproducibility manifest
#
# `submission.csv`は推論成果物として生成するが、このnotebookはcompetition submit APIを呼ばない。

# %%
submission = sample[["id"]].merge(
    predictions[["id", "pred_tvt"]], on="id", how="left", validate="one_to_one"
).rename(columns={"pred_tvt": "tvt"})
if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise ValueError("submission row/order contract failed")
if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
    raise ValueError("submission finite contract failed")

prediction_path = output_dir / "exp372_current_test_predictions.csv.gz"
feature_schema_path = output_dir / "exp372_inference_feature_schema.csv"
predictions.to_csv(prediction_path, index=False, compression="gzip")
pd.DataFrame(
    {
        "feature_index": np.arange(len(final_feature_columns), dtype=np.int32),
        "feature": final_feature_columns,
        "feature_group": [
            (
                "exp218_base"
                if index < len(base_feature_columns)
                else (
                    "saved_exp264_compact"
                    if index
                    < len(base_feature_columns) + len(parent_compact_features)
                    else (
                        "fold_safe_formation"
                        if index
                        < (
                            len(base_feature_columns)
                            + len(parent_compact_features)
                            + len(formation_features)
                        )
                        else "signed_residual_compact"
                    )
                )
            )
            for index in range(len(final_feature_columns))
        ],
    }
).to_csv(feature_schema_path, index=False)
submission.to_csv(paths.submission_path, index=False)

metrics = {
    "experiment": EXPERIMENT_NAME,
    "route": config["experiment"]["route"],
    "status": "cpu_inference_completed_pending_submit_check",
    "authorization": {
        "status": inference_cfg["status"],
        "scope": inference_cfg["authorization_scope"],
        "explicit_user_override_after_guard_failure": True,
        "technical_gate_passed": True,
        "incremental_utility_gate_passed": False,
        "tail_promotion_gate_passed": False,
        "promotion_gate_passed": False,
        "retained_guard_failure": inference_cfg["retained_guard_failure"],
        "generate_submission_file": True,
        "competition_submit_performed": False,
        "competition_submit_authorized": False,
    },
    "runtime": "kaggle_cpu",
    "runtime_seconds": round(time.time() - STARTED_AT, 3),
    "rows": int(len(predictions)),
    "wells": int(predictions["well"].nunique()),
    "candidate_count": len(names),
    "namespaced_confidence_column_count": 21,
    "selector_feature_count": len(selector_features),
    "selector_missingness": {
        "training_sparse_feature_count": training_sparse_feature_count,
        "current_sparse_feature_count": int(
            selector_missingness["current_missing_count"].gt(0).sum()
        ),
        "current_missing_cell_count": int(selector_missing_count_by_feature.sum()),
        "infinite_cell_count": 0,
        "zero_imputation_performed": False,
    },
    "parent_compact_feature_count": len(parent_compact_features),
    "formation_feature_count": len(formation_features),
    "signed_compact_feature_count": len(signed_compact_features),
    "base_feature_count": len(base_feature_columns),
    "source_base_feature_count": len(source_base_columns),
    "final_feature_count": len(final_feature_columns),
    "parent_selector_model_count": len(selector_model_audit),
    "signed_selector_model_count": len(signed_selector_model_audit),
    "tvt_model_count": len(tvt_model_audit),
    "booster_training_count": 0,
    "max_abs_formula_parity": float(max_abs_formula),
    "signed_top1_value_parity_max_abs_error": float(signed_top1_parity_max),
    "confidence_parity": confidence_parity,
    "prediction_stats": {
        "min": float(pred_tvt.min()),
        "max": float(pred_tvt.max()),
        "mean": float(pred_tvt.mean()),
        "std": float(pred_tvt.std()),
    },
    "train_guard_evidence": {
        "union_rmse": float(inference_cfg["train_union_rmse"]),
        "best_standalone_rmse": float(
            inference_cfg["train_best_standalone_rmse"]
        ),
        "by_well_delta_p95_vs_exp264": float(
            inference_cfg["train_by_well_delta_p95_vs_exp264"]
        ),
        "worst_well_delta_vs_exp264": float(
            inference_cfg["train_worst_well_delta_vs_exp264"]
        ),
        "worst_well": str(inference_cfg["train_worst_well"]),
        "incremental_utility_passed": False,
        "tail_promotion_passed": False,
        "promotion_passed": False,
    },
    "source_audit": source_audit,
    "pf_generation": pf_meta,
    "exp226_generation": k16_summary,
    "exp145_generation": learned_generator_summary,
    "exp218_anchor": anchor_meta,
    "exp218_grwr": exp218._jsonable(grwr_meta),
    "formation_generation": formation_generation,
    "primitive_content_sha256": primitive_content_sha,
    "parent_selector_models": selector_model_audit,
    "signed_selector_models": signed_selector_model_audit,
    "tvt_models": tvt_model_audit,
    "sha256": {
        "candidate_contract": candidate_contract_sha(candidate_contract),
        "exp263_formula_parquet": sha256_file(formula_path),
        "stage_c_bundle": sha256_file(stage_c_bundle),
        "stage_c_model_manifest": sha256_file(stage_c_manifest_path),
        "stage_s_signed_model_manifest": sha256_file(signed_manifest_path),
        "stage_s_signed_compact_schema": sha256_file(signed_schema_path),
        "selector_feature_schema": sha256_file(selector_schema_path),
        "selector_feature_catalog": sha256_file(selector_catalog_path),
        "source_base_feature_catalog": sha256_file(source_base_catalog_path),
        "base_feature_allowlist": sha256_file(base_allowlist_path),
        "selector_missingness_current_test": sha256_file(selector_missingness_path),
        "selector_missingness_by_candidate_current_test": sha256_file(
            selector_candidate_missingness_path
        ),
        "parent_compact_meta_schema": sha256_file(compact_schema_path),
        "union_model_manifest": sha256_file(stage_d_manifest_path),
        "parent_compact_parquet_by_outer": compact_sha,
        "signed_compact_parquet_by_outer": signed_compact_sha,
        "formation_parquet": sha256_file(formation_path),
        "formation_logical_content": formation_generation[
            "logical_content_sha256"
        ],
        "candidate_score_sample": sha256_file(score_sample_path),
        "exp145_replay_cache_decompressed": sha256_gzip_decompressed(learned_cache_path),
        "predictions_decompressed": sha256_gzip_decompressed(prediction_path),
        "feature_schema": sha256_file(feature_schema_path),
        "submission": sha256_file(paths.submission_path),
    },
    "notes": [
        (
            "All 12 exp263 candidates and 21 native-confidence columns are "
            "regenerated from raw test in this run."
        ),
        "Selector NaN values are preserved exactly as trained; no zero imputation is performed.",
        (
            "The Stage A catalog guards training-dense features and structural "
            "confidence/formula missing rates."
        ),
        (
            "Each union model receives saved74 and signed23 from its matching "
            "outer fold plus one all-train-reference formation74 surface."
        ),
        "The exact model order is clean273, saved74, formation74, signed23 for 444 unique columns.",
        (
            "All 40 parent selectors, 20 signed selectors, and 15 union TVT "
            "models are SHA-verified; no model is fitted."
        ),
        (
            "No public-test row artifact, saved selector score CSV, hard "
            "selector, Viterbi, or candidate softmax average participates in "
            "prediction."
        ),
        (
            "submission.csv is generated for submit-check; external "
            "competition submission remains unauthorized."
        ),
        (
            "The pre-registered incremental and tail failures remain recorded "
            "and are not reclassified as passes."
        ),
    ],
}
write_json(output_dir / "inference_metrics.json", metrics)
write_json(output_dir / "reproducibility_manifest_inference.json", metrics)
write_json(paths.metrics_path, metrics)
display(submission.head(20))
display(submission["tvt"].describe())
display(metrics)
print("Generated artifacts:")
for artifact_path in [
    formula_path,
    formation_path,
    prediction_path,
    feature_schema_path,
    selector_missingness_path,
    selector_candidate_missingness_path,
    paths.submission_path,
    output_dir / "inference_metrics.json",
]:
    print(f"- {artifact_path} ({artifact_path.stat().st_size} bytes)")
