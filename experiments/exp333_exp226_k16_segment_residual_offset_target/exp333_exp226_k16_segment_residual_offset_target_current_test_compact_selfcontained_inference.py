# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # exp333 current-test K16 segment residual candidate inference
#
# exp361でadd-one候補として支持されたexp333を、保存済みStage 1 outer-fold
# 5 modelだけでcurrent testへ展開する。exp226 inference v1をbaseとし、train時と
# 同じtarget-free replay / U projection / GRWR / K16 aggregationを再生成する。
# 出力は候補artifactだけであり、selector、blend、`submission.csv`、competition
# submitはこのNotebookの範囲外とする。

# %% [markdown]
# ## Contents
# 1. Imports and fixed contract
# 2. Path, hashing, and serialization helpers
# 3. Authorization and saved-train parity
# 4. Raw current-test replay and target-free feature surface
# 5. Exact K16 aggregation
# 6. Saved five-model inference
# 7. Candidate artifacts and reproducibility manifest

# %% [markdown]
# ## 1. Imports and fixed contract

# %%
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp333_exp226_k16_segment_residual_offset_target"
OUTPUT_PREFIX = EXPERIMENT_NAME
K_SEGMENTS = 16
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KEY_COLUMNS = ("well_id", "row_idx")
ALLOWED_STAGE1_GROUPS = (
    "projection_correction",
    "u_disagreement",
    "gr_wavelet_rotation_confidence",
)
STRUCTURAL_FEATURE_COLUMNS = (
    "segment_id",
    "segment_position",
    "segment_row_count",
    "segment_md_span",
    "exp226_pred_mean",
    "exp226_pred_start",
    "exp226_pred_end_minus_start",
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP333_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


# %% [markdown]
# ## 2. Path, hashing, and serialization helpers

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    return project_root() / "experiments" / EXPERIMENT_NAME


def load_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            return value
    raise FileNotFoundError("exp333 config.yaml was not found")


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hashed_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, list(columns)]
    digest = hashlib.sha256(canonical_json_bytes({"columns": list(columns)}))
    digest.update(
        pd.util.hash_pandas_object(selected, index=False)
        .to_numpy(np.uint64)
        .tobytes()
    )
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(payload),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def artifact_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "file_sha256": sha256_file(path),
    }
    if path.suffix == ".gz":
        evidence["decompressed_sha256"] = sha256_gzip_decompressed(path)
    return evidence


def resolve_named_input(filename: str, *, kernel_token: str | None = None) -> Path:
    local_candidates = (
        Path.cwd() / filename,
        Path.cwd() / "artifacts" / filename,
        experiment_dir() / "artifacts" / filename,
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(
        path
        for path in KAGGLE_INPUT_ROOT.rglob(filename)
        if path.is_file()
    )
    if kernel_token is not None:
        preferred = [path for path in matches if kernel_token in str(path)]
        if len(preferred) == 1:
            return preferred[0]
        if len(preferred) > 1:
            matches = preferred
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{filename} was not unique for token={kernel_token!r}: {matches}"
        )
    return matches[0]


def resolve_bootstrap_source(relative_path: str) -> Path:
    candidates = (
        Path.cwd() / relative_path,
        project_root() / relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"bootstrap source was not found: {relative_path}")


def import_file(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_raw_data_root() -> Path:
    local = project_root() / "data" / "raw"
    if (
        (local / "sample_submission.csv").is_file()
        and (local / "train").is_dir()
        and (local / "test").is_dir()
    ):
        return local
    preferred = (
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
    )
    for candidate in preferred:
        if (
            (candidate / "sample_submission.csv").is_file()
            and (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
        ):
            return candidate
    matches = sorted(
        path.parent
        for path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")
        if (path.parent / "train").is_dir() and (path.parent / "test").is_dir()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"competition raw root was not unique: {matches}")
    return matches[0]


def output_artifacts_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise ValueError(f"{label} parity failed: {actual} != {expected}")


# %% [markdown]
# ## 3. Authorization and saved-train parity

# %%
def validate_candidate_inference_contract(
    config: Mapping[str, Any], *, require_execution_authorization: bool = False
) -> dict[str, Any]:
    exact = {
        "experiment.route": "ensemble",
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "downstream_candidate_path_evidence.submission_approved": False,
        "execution_contract.submission_approved": False,
        "inference.enabled": False,
        "inference.create_submission": False,
        "candidate_inference.enabled": True,
        "candidate_inference.create_submission": False,
        "candidate_inference.authorization_scope": "current_test_candidate_artifact_only",
        "candidate_inference.selected_variant": "exp333_k16_segment_residual_offset",
        "candidate_inference.base_prediction": "exp226_inference_v1",
        "candidate_inference.base_submission_rows": 14151,
        "candidate_inference.expected_test_wells": 3,
        "candidate_inference.k_segments": 16,
        "candidate_inference.expected_segments": 48,
        "candidate_inference.saved_model_count": 5,
        "candidate_inference.fold_ensemble": "float64_arithmetic_mean",
        "candidate_inference.offset_clipping": "none",
        "candidate_inference.offset_shrinkage": "none",
        "candidate_inference.offset_taper": "none",
        "candidate_inference.offset_interpolation": "none",
        "candidate_inference.offset_slope": "disabled",
        "candidate_inference.selector": "disabled",
        "candidate_inference.blend": "disabled",
        "candidate_inference.fixed12_average": "disabled",
        "candidate_inference.new_training_variants": 0,
        "candidate_inference.model_configs_trained": 0,
        "candidate_inference.folds_trained": 0,
        "candidate_inference.boosters_trained": 0,
        "candidate_inference.parent_control_retraining": False,
        "candidate_inference.runtime.gpu": False,
        "candidate_inference.runtime.n_jobs": 8,
        "candidate_inference.runtime.pf_seeds": 128,
        "candidate_inference.runtime.pf_particles": 500,
    }
    changed = {
        key: {"expected": expected, "actual": get_nested(config, key)}
        for key, expected in exact.items()
        if get_nested(config, key) != expected
    }
    if changed:
        raise ValueError(f"exp333 candidate inference contract changed: {changed}")
    if tuple(get_nested(config, "candidate_inference.saved_model_folds", ())) != (
        0,
        1,
        2,
        3,
        4,
    ):
        raise ValueError("candidate inference must use saved outer folds 0..4")
    if tuple(get_nested(config, "features.allowed_groups", ())) != ALLOWED_STAGE1_GROUPS:
        raise ValueError("candidate inference Stage 1 feature groups changed")
    if tuple(get_nested(config, "features.structural_columns", ())) != (
        STRUCTURAL_FEATURE_COLUMNS
    ):
        raise ValueError("candidate inference structural columns changed")
    if require_execution_authorization and not (
        get_nested(config, "execution_contract.candidate_inference_approved") is True
        and get_nested(
            config, "execution_contract.candidate_inference_authorization_consumed"
        )
        is False
        and get_nested(
            config,
            "downstream_candidate_path_evidence.current_test_candidate_inference_approved",
        )
        is True
    ):
        raise RuntimeError("exp333 candidate inference execution is not authorized")
    return {
        "authorization_scope": get_nested(
            config, "candidate_inference.authorization_scope"
        ),
        "variant_count": 1,
        "saved_model_inference_count": 5,
        "trained_model_configs": 0,
        "trained_folds": 0,
        "trained_boosters": 0,
        "parent_control_retraining": False,
        "submission_created": False,
    }


@dataclass(frozen=True)
class SavedTrainContract:
    manifest_path: Path
    summary_path: Path
    feature_schema_path: Path
    sha_manifest_path: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    feature_schema: pd.DataFrame
    model_rows: tuple[dict[str, Any], ...]
    parity: dict[str, Any]


def load_saved_train_contract(config: Mapping[str, Any]) -> SavedTrainContract:
    spec = get_nested(config, "candidate_inference.inputs.exp333_stage1_train")
    token = "exp333-k16-segment-residual-stage1-train"
    manifest_path = resolve_named_input(
        str(spec["model_manifest_filename"]), kernel_token=token
    )
    summary_path = resolve_named_input(str(spec["summary_filename"]), kernel_token=token)
    feature_schema_path = resolve_named_input(
        str(spec["feature_schema_filename"]), kernel_token=token
    )
    sha_manifest_path = resolve_named_input(
        str(spec["sha_manifest_filename"]), kernel_token=token
    )
    fixed_files = {
        manifest_path: str(spec["model_manifest_sha256"]),
        summary_path: str(spec["summary_sha256"]),
        feature_schema_path: str(spec["feature_schema_file_sha256"]),
        sha_manifest_path: str(spec["sha_manifest_sha256"]),
    }
    for path, expected_sha in fixed_files.items():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"saved train artifact SHA mismatch: {path.name}")

    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    feature_schema = pd.read_csv(feature_schema_path)
    sha_manifest = pd.read_csv(sha_manifest_path, dtype=str).fillna("")
    schema_content_sha = hashed_frame_sha256(
        feature_schema, tuple(feature_schema.columns)
    )
    if schema_content_sha != str(spec["feature_schema_content_sha256"]):
        raise ValueError("saved train feature schema content SHA mismatch")
    if manifest["feature_schema_sha256"] != schema_content_sha:
        raise ValueError("model manifest and saved feature schema differ")
    if manifest["feature_freeze_sha256"] != str(spec["feature_freeze_sha256"]):
        raise ValueError("saved train feature freeze SHA mismatch")
    if int(manifest["boosters"]) != int(spec["expected_boosters"]):
        raise ValueError("saved train booster count mismatch")
    if len(manifest["feature_columns"]) != int(spec["expected_feature_count"]):
        raise ValueError("saved train model feature count mismatch")

    row_features = feature_schema.loc[
        ~feature_schema["source_group"].eq("structural"), "feature_name"
    ].astype(str).tolist()
    structural = feature_schema.loc[
        feature_schema["source_group"].eq("structural"), "feature_name"
    ].astype(str).tolist()
    if structural != list(STRUCTURAL_FEATURE_COLUMNS):
        raise ValueError("saved train structural feature order mismatch")
    if list(manifest["feature_columns"]) != [*structural, *row_features]:
        raise ValueError("saved train manifest feature order mismatch")

    expected_summary = {
        "boosters": int(spec["expected_boosters"]),
        "decision": str(spec["expected_decision"]),
        "model_manifest_sha256": str(spec["model_manifest_sha256"]),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"saved train summary parity failed for {key}: "
                f"{summary.get(key)!r} != {expected!r}"
            )
    assert_close(
        summary["pooled"]["exp226_rmse"],
        spec["expected_exp226_rmse"],
        "saved train exp226 RMSE",
    )
    assert_close(
        summary["pooled"]["stage1_rmse"],
        spec["expected_stage1_rmse"],
        "saved train Stage 1 RMSE",
    )
    if int(summary["pooled"]["rows"]) != int(spec["expected_rows"]):
        raise ValueError("saved train pooled row parity failed")
    if int(summary["pooled"]["wells"]) != int(spec["expected_wells"]):
        raise ValueError("saved train pooled well parity failed")
    if int(summary["feature_freeze"]["model_feature_count"]) != int(
        spec["expected_feature_count"]
    ):
        raise ValueError("saved train summary feature-count parity failed")
    for item in manifest["models"]:
        if int(item["train_segments"]) + int(item["valid_segments"]) != int(
            spec["expected_segments"]
        ):
            raise ValueError("saved train segment-count parity failed")

    sha_by_name = sha_manifest.set_index("filename")
    oof_name = (
        f"{OUTPUT_PREFIX}_stage1_oof_predictions.csv.gz"
    )
    if sha_by_name.loc[oof_name, "file_sha256"] != str(
        spec["saved_oof_file_sha256"]
    ):
        raise ValueError("saved train OOF file SHA manifest mismatch")
    if sha_by_name.loc[oof_name, "decompressed_sha256"] != str(
        spec["saved_oof_decompressed_sha256"]
    ):
        raise ValueError("saved train OOF decompressed SHA manifest mismatch")

    expected_model_sha = {
        int(key): str(value) for key, value in spec["model_sha256"].items()
    }
    model_rows: list[dict[str, Any]] = []
    seen_folds: list[int] = []
    for item in manifest["models"]:
        fold = int(item["outer_fold"])
        filename = Path(str(item["model_path"])).name
        model_path = resolve_named_input(filename, kernel_token=token)
        actual_sha = sha256_file(model_path)
        if actual_sha != expected_model_sha[fold]:
            raise ValueError(f"saved model SHA mismatch for outer fold {fold}")
        if actual_sha != str(item["model_sha256"]):
            raise ValueError(f"model manifest SHA mismatch for outer fold {fold}")
        if sha_by_name.loc[filename, "file_sha256"] != actual_sha:
            raise ValueError(f"train SHA manifest mismatch for outer fold {fold}")
        model_rows.append(
            {
                **item,
                "resolved_model_path": str(model_path),
                "actual_model_sha256": actual_sha,
            }
        )
        seen_folds.append(fold)
    if sorted(seen_folds) != [0, 1, 2, 3, 4]:
        raise ValueError(f"saved model fold set changed: {seen_folds}")

    parity = {
        "status": "saved_train_contract_parity_pass",
        "train_decision_retained": summary["decision"],
        "train_rows": summary["pooled"]["rows"],
        "train_wells": summary["pooled"]["wells"],
        "train_segments": int(spec["expected_segments"]),
        "train_exp226_rmse": summary["pooled"]["exp226_rmse"],
        "train_stage1_rmse": summary["pooled"]["stage1_rmse"],
        "feature_count": len(manifest["feature_columns"]),
        "feature_schema_content_sha256": schema_content_sha,
        "feature_freeze_sha256": manifest["feature_freeze_sha256"],
        "model_manifest_sha256": sha256_file(manifest_path),
        "model_sha256": {
            str(row["outer_fold"]): row["actual_model_sha256"]
            for row in model_rows
        },
        "saved_oof_file_sha256": str(spec["saved_oof_file_sha256"]),
        "saved_oof_decompressed_sha256": str(
            spec["saved_oof_decompressed_sha256"]
        ),
    }
    return SavedTrainContract(
        manifest_path=manifest_path,
        summary_path=summary_path,
        feature_schema_path=feature_schema_path,
        sha_manifest_path=sha_manifest_path,
        manifest=manifest,
        summary=summary,
        feature_schema=feature_schema,
        model_rows=tuple(model_rows),
        parity=parity,
    )


def validate_bootstrap_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    specs = get_nested(config, "candidate_inference.inputs")
    paths = {
        "exp072_replay_source": resolve_bootstrap_source(
            "inputs/exp072_source/public_notebook_replay_audit.py"
        ),
        "exp228_target_free_source": resolve_bootstrap_source(
            "inputs/exp228_source/direct_residual_correction_on_exp226.py"
        ),
    }
    records: dict[str, Any] = {}
    for name, path in paths.items():
        actual_sha = sha256_file(path)
        expected_sha = str(specs[name]["sha256"])
        if actual_sha != expected_sha:
            raise ValueError(f"bootstrap source SHA mismatch: {name}")
        records[name] = {"path": str(path), "sha256": actual_sha}
    return records


# %% [markdown]
# ## 4. Raw current-test replay and target-free feature surface

# %%
@dataclass(frozen=True)
class CurrentFeatureSurface:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    schema: pd.DataFrame
    projection_summary: pd.DataFrame
    grwr_summary: pd.DataFrame
    metadata: dict[str, Any]


def _row_index_from_id(ids: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        ids.astype(str).str.rsplit("_", n=1).str[-1], errors="raise"
    ).to_numpy(np.float64)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("current-test id suffix is not an integer row index")
    return values.astype(np.int64)


def build_current_feature_surface(
    config: Mapping[str, Any],
    *,
    raw_root: Path,
    replay_source: Any,
    feature_source: Any,
    train_contract: SavedTrainContract,
    artifacts_dir: Path,
) -> CurrentFeatureSurface:
    inference = get_nested(config, "candidate_inference")
    replay_source.configure_public_runtime(
        data_dir=raw_root,
        output_dir=artifacts_dir / "raw_test_replay",
        n_jobs=int(inference["runtime"]["n_jobs"]),
        pf_seeds=int(inference["runtime"]["pf_seeds"]),
        pf_particles=int(inference["runtime"]["pf_particles"]),
        fast=False,
        use_gpu="false",
    )
    replay_started = time.perf_counter()
    base, replay_meta = replay_source.build_replay_test_frame()
    replay_seconds = time.perf_counter() - replay_started
    base = base.reset_index(drop=True)
    base["id"] = base["id"].astype(str)
    base["well"] = base["well"].astype(str)
    if "target" in base.columns:
        raise ValueError("raw current-test replay unexpectedly contains target")
    if base.duplicated(["id", "well"]).any():
        raise ValueError("raw current-test replay keys are not unique")

    schema_spec = inference["inputs"]["exp072_feature_schema"]
    exp072_schema_path = resolve_named_input(
        str(schema_spec["filename"]),
        kernel_token="exp072-exp063-full-replay-feature-cache-train",
    )
    if sha256_file(exp072_schema_path) != str(schema_spec["sha256"]):
        raise ValueError("exp072 feature schema SHA mismatch")
    exp072_schema = pd.read_csv(exp072_schema_path).sort_values(
        "feature_index", kind="mergesort"
    )
    expected_base_columns = exp072_schema["feature"].astype(str).tolist()
    current_base_columns = [
        str(column)
        for column in replay_source.feature_columns_for_variant(
            base, "pixiux_likpf_public_replay"
        )
    ]
    replay_returned_columns = [
        str(column) for column in base.columns if column not in {"id", "well"}
    ]
    excluded_replay_columns = [
        column for column in replay_returned_columns if column not in current_base_columns
    ]
    if len(current_base_columns) != int(schema_spec["expected_feature_count"]):
        raise ValueError(
            f"expected 196 raw-test replay features, found {len(current_base_columns)}"
        )
    if current_base_columns != expected_base_columns:
        raise ValueError("raw current-test replay schema differs from exp072 train")
    # build_replay_test_frame also returns nine likelihood-PF diagnostic columns
    # that exp072 intentionally excludes from its reusable 196-feature cache.
    # Remove them before downstream generation so current test has the exact
    # target-free train-cache surface rather than a permissive superset.
    base = base[["id", "well", *current_base_columns]].copy()

    anchored, anchor_meta = feature_source.add_inference_anchor_columns(
        base, raw_root / "test"
    )
    projection_config = dict(get_nested(config, "features.u_projection"))
    projection, projection_groups, projection_summary = (
        feature_source.build_u_projection_features(
            anchored,
            source_specs=dict(projection_config["sources"]),
            degree=int(projection_config["degree"]),
            robust_iters=int(projection_config["robust_iters"]),
            clip_sigma=float(projection_config["clip_sigma"]),
        )
    )
    if not anchored[["id", "well"]].reset_index(drop=True).equals(
        projection[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("current-test U projection row identity changed")
    grwr, grwr_groups, grwr_summary, grwr_meta = (
        feature_source.build_gr_wavelet_rotation_confidence_features(
            anchored,
            train_dir=raw_root / "test",
            config=dict(
                get_nested(config, "features.gr_wavelet_rotation_confidence")
            ),
        )
    )
    if not anchored[["id", "well"]].reset_index(drop=True).equals(
        grwr[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError("current-test GRWR row identity changed")

    group_columns = {
        "projection_correction": list(
            projection_groups["projection_correction"]
        ),
        "u_disagreement": list(projection_groups["u_disagreement"]),
        "gr_wavelet_rotation_confidence": list(
            grwr_groups["gr_wavelet_rotation_confidence"]
        ),
    }
    selected: list[str] = []
    column_group: dict[str, str] = {}
    for group in ALLOWED_STAGE1_GROUPS:
        for column in group_columns[group]:
            if column not in selected:
                selected.append(column)
                column_group[column] = group
    saved_row_features = train_contract.feature_schema.loc[
        ~train_contract.feature_schema["source_group"].eq("structural"),
        "feature_name",
    ].astype(str).tolist()
    if selected != saved_row_features:
        raise ValueError("current-test row feature order differs from saved train")

    surface = pd.DataFrame(
        {
            "id": anchored["id"].astype(str),
            "well_id": anchored["well"].astype(str),
            "row_idx": _row_index_from_id(anchored["id"]),
            "md_since": pd.to_numeric(
                anchored["md_since"], errors="raise"
            ).astype(np.float64),
        }
    )
    projection_selected = [
        column
        for column in selected
        if column_group[column] in {"projection_correction", "u_disagreement"}
    ]
    for column in projection_selected:
        surface[column] = projection[column].to_numpy(np.float32, copy=False)
    for column in group_columns["gr_wavelet_rotation_confidence"]:
        surface[column] = grwr[column].to_numpy(np.float32, copy=False)
    if surface.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("current-test feature row keys are not unique")
    if not np.isfinite(surface["md_since"].to_numpy(np.float64)).all():
        raise ValueError("current-test md_since contains non-finite values")

    schema = pd.DataFrame(
        [
            {
                "feature_name": column,
                "source_group": column_group[column],
                "row_to_segment_aggregation": "finite_float64_mean",
                "all_nonfinite_policy": "preserve_nan",
            }
            for column in selected
        ]
        + [
            {
                "feature_name": column,
                "source_group": "structural",
                "row_to_segment_aggregation": "fixed_definition",
                "all_nonfinite_policy": "not_applicable",
            }
            for column in STRUCTURAL_FEATURE_COLUMNS
        ]
    )
    schema_sha = hashed_frame_sha256(schema, tuple(schema.columns))
    expected_schema_sha = str(
        inference["inputs"]["exp333_stage1_train"][
            "feature_schema_content_sha256"
        ]
    )
    if schema_sha != expected_schema_sha:
        raise ValueError("current-test feature schema content SHA differs from train")
    ordered = surface.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    metadata = {
        "raw_test_replay": replay_meta,
        "raw_test_replay_seconds": replay_seconds,
        "raw_test_rows": len(base),
        "raw_test_wells": int(base["well"].nunique()),
        "raw_test_replay_returned_feature_count": len(replay_returned_columns),
        "raw_test_replay_excluded_diagnostic_columns": excluded_replay_columns,
        "exp072_feature_count": len(current_base_columns),
        "exp072_schema_path": str(exp072_schema_path),
        "exp072_schema_sha256": sha256_file(exp072_schema_path),
        "anchor": anchor_meta,
        "grwr": grwr_meta,
        "row_feature_count": len(selected),
        "model_feature_count": len(selected) + len(STRUCTURAL_FEATURE_COLUMNS),
        "row_feature_schema_sha256": schema_sha,
        "row_feature_content_sha256": hashed_frame_sha256(
            ordered,
            ("well_id", "row_idx", "md_since", *selected),
        ),
        "row_feature_nonfinite_cells": int(
            (~np.isfinite(ordered[selected].to_numpy(np.float32))).sum()
        ),
        "target_or_error_columns_loaded": 0,
    }
    return CurrentFeatureSurface(
        frame=ordered,
        feature_columns=tuple(selected),
        schema=schema,
        projection_summary=projection_summary,
        grwr_summary=grwr_summary,
        metadata=metadata,
    )


def load_exp226_current_test(
    config: Mapping[str, Any], sample: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "candidate_inference.inputs.exp226_inference")
    token = "exp226-k16-kappa-repro-inference"
    submission_path = resolve_named_input(
        str(spec["submission_filename"]), kernel_token=token
    )
    summary_path = resolve_named_input(
        str(spec["summary_filename"]), kernel_token=token
    )
    if sha256_file(submission_path) != str(spec["submission_sha256"]):
        raise ValueError("exp226 current-test submission SHA mismatch")
    if sha256_file(summary_path) != str(spec["summary_sha256"]):
        raise ValueError("exp226 current-test summary SHA mismatch")
    summary = json.loads(summary_path.read_text())
    if (
        int(summary["submission_rows"]) != len(sample)
        or int(summary["test_wells"]) != 3
        or str(summary["submission_sha256"]) != str(spec["submission_sha256"])
    ):
        raise ValueError("exp226 current-test summary parity failed")
    base = pd.read_csv(submission_path, dtype={"id": str})
    if list(base.columns) != ["id", "tvt"]:
        raise ValueError(f"exp226 current-test columns changed: {base.columns.tolist()}")
    if not base["id"].equals(sample["id"]):
        raise ValueError("exp226 current-test ID/order differs from sample submission")
    values = pd.to_numeric(base["tvt"], errors="raise").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("exp226 current-test prediction contains non-finite values")
    base = base.rename(columns={"tvt": "exp226_tvt"})
    return base, {
        "submission_path": str(submission_path),
        "submission_sha256": sha256_file(submission_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "rows": len(base),
        "tvt_min": float(values.min()),
        "tvt_max": float(values.max()),
        "tvt_mean": float(values.mean()),
    }


# %% [markdown]
# ## 5. Exact K16 aggregation

# %%
def exact_k16_segment_ids(
    length: int, k_segments: int = K_SEGMENTS
) -> np.ndarray:
    if length <= 0:
        raise ValueError("unknown suffix length must be positive")
    if k_segments != K_SEGMENTS:
        raise ValueError("exp333 current-test inference is fixed to K16")
    edges = np.linspace(0.0, float(length), k_segments + 1)
    one_based_step = np.arange(1, length + 1, dtype=np.float64)
    return np.clip(
        np.searchsorted(edges[1:], one_based_step, side="left"),
        0,
        k_segments - 1,
    ).astype(np.int16)


def assign_current_test_k16(
    surface: CurrentFeatureSurface,
    exp226: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    raw_test_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = surface.frame.merge(
        exp226,
        on="id",
        how="left",
        validate="one_to_one",
    )
    if rows["exp226_tvt"].isna().any() or len(rows) != len(sample):
        raise ValueError("exp226 base does not cover current-test feature rows")
    if set(rows["id"]) != set(sample["id"]):
        raise ValueError("current-test feature IDs differ from sample submission")
    rows = rows.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    rows["suffix_offset"] = rows.groupby(
        "well_id", sort=False, observed=True
    ).cumcount().astype(np.int64)
    segment_id = np.empty(len(rows), dtype=np.int16)
    for positions in rows.groupby(
        "well_id", sort=False, observed=True
    ).indices.values():
        index = np.asarray(positions, dtype=np.int64)
        row_index = rows.loc[index, "row_idx"].to_numpy(np.int64)
        if len(row_index) > 1 and not np.all(np.diff(row_index) == 1):
            raise ValueError("current-test suffix row_idx is not contiguous")
        segment_id[index] = exact_k16_segment_ids(len(index))
    rows["segment_id"] = segment_id

    boundary_records: list[dict[str, Any]] = []
    for well_id, group in rows.groupby("well_id", sort=True, observed=True):
        raw_path = raw_test_dir / f"{well_id}__horizontal_well.csv"
        raw = pd.read_csv(raw_path, usecols=["TVT_input"])
        unknown_index = np.flatnonzero(
            pd.to_numeric(raw["TVT_input"], errors="coerce").isna().to_numpy()
        )
        if len(unknown_index) == 0:
            raise ValueError(f"raw test has no unknown suffix: {well_id}")
        expected_unknown = np.arange(
            int(unknown_index[0]), len(raw), dtype=np.int64
        )
        if not np.array_equal(unknown_index, expected_unknown):
            raise ValueError(f"raw test unknown rows are not a suffix: {well_id}")
        actual_index = group["row_idx"].to_numpy(np.int64)
        if not np.array_equal(actual_index, expected_unknown):
            raise ValueError(f"raw/current feature row boundary mismatch: {well_id}")
        expected_ids = np.asarray(
            [f"{well_id}_{int(index)}" for index in expected_unknown],
            dtype=object,
        )
        if not np.array_equal(group["id"].to_numpy(dtype=object), expected_ids):
            raise ValueError(f"raw/current feature ID boundary mismatch: {well_id}")
        per_segment = (
            group.groupby("segment_id", sort=True, observed=True)
            .agg(
                segment_row_count=("id", "size"),
                suffix_offset_min=("suffix_offset", "min"),
                suffix_offset_max=("suffix_offset", "max"),
                row_idx_min=("row_idx", "min"),
                row_idx_max=("row_idx", "max"),
            )
            .reset_index()
        )
        if per_segment["segment_id"].astype(int).tolist() != list(range(16)):
            raise ValueError(f"K16 coverage changed for {well_id}")
        for record in per_segment.to_dict(orient="records"):
            boundary_records.append(
                {
                    "well_id": str(well_id),
                    "raw_rows": len(raw),
                    "known_prefix_rows": int(unknown_index[0]),
                    "unknown_suffix_rows": len(unknown_index),
                    **record,
                }
            )
    boundary = pd.DataFrame(boundary_records)
    if len(boundary) != 48:
        raise ValueError(f"expected 48 current-test K16 segments, found {len(boundary)}")
    return rows, boundary


def aggregate_current_test_segments(
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    context = rows.sort_values(
        ["well_id", "suffix_offset"], kind="mergesort"
    ).reset_index(drop=True)
    group_keys = ["well_id", "segment_id"]
    group_index = context.groupby(
        group_keys, sort=True, observed=True
    ).ngroup().to_numpy(np.int64)
    group_meta = (
        context.groupby(group_keys, sort=True, observed=True)
        .agg(
            segment_row_count=("row_idx", "size"),
            segment_md_min=("md_since", "min"),
            segment_md_max=("md_since", "max"),
            exp226_pred_start=("exp226_tvt", "first"),
            exp226_pred_end=("exp226_tvt", "last"),
        )
        .reset_index()
    )
    n_groups = len(group_meta)

    def finite_mean(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        sums = np.bincount(
            group_index[finite], weights=array[finite], minlength=n_groups
        )
        counts = np.bincount(group_index[finite], minlength=n_groups)
        result = np.full(n_groups, np.nan, dtype=np.float64)
        np.divide(sums, counts, out=result, where=counts > 0)
        return result

    segment = group_meta.drop(
        columns=["segment_md_min", "segment_md_max", "exp226_pred_end"]
    )
    segment["segment_position"] = (
        segment["segment_id"].to_numpy(np.float64) + 0.5
    ) / K_SEGMENTS
    segment["segment_md_span"] = (
        group_meta["segment_md_max"].to_numpy(np.float64)
        - group_meta["segment_md_min"].to_numpy(np.float64)
    )
    segment["exp226_pred_mean"] = finite_mean(
        context["exp226_tvt"].to_numpy()
    )
    segment["exp226_pred_end_minus_start"] = (
        group_meta["exp226_pred_end"].to_numpy(np.float64)
        - group_meta["exp226_pred_start"].to_numpy(np.float64)
    )
    for column in feature_columns:
        segment[column] = finite_mean(context[column].to_numpy())
    ordered = [
        "well_id",
        "segment_id",
        *STRUCTURAL_FEATURE_COLUMNS[1:],
        *feature_columns,
    ]
    result = segment.loc[:, ordered].sort_values(
        ["well_id", "segment_id"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != 48:
        raise ValueError(f"expected 48 aggregated current-test segments, found {len(result)}")
    matrix = result[
        [*STRUCTURAL_FEATURE_COLUMNS, *feature_columns]
    ].to_numpy(np.float64)
    if np.isinf(matrix).any():
        raise ValueError("aggregated current-test model features contain infinity")
    return result


# %% [markdown]
# ## 6. Saved five-model inference

# %%
def predict_saved_fold_ensemble(
    segments: pd.DataFrame,
    train_contract: SavedTrainContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    feature_columns = [str(item) for item in train_contract.manifest["feature_columns"]]
    matrix_frame = segments.loc[:, feature_columns]
    matrix = matrix_frame.to_numpy(np.float32, copy=False)
    component_values: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    result = segments.copy()
    for item in sorted(train_contract.model_rows, key=lambda row: int(row["outer_fold"])):
        fold = int(item["outer_fold"])
        model_path = Path(str(item["resolved_model_path"]))
        booster = lgb.Booster(model_file=str(model_path))
        if booster.feature_name() != feature_columns:
            raise ValueError(f"saved model feature names differ for outer fold {fold}")
        if int(booster.num_feature()) != len(feature_columns):
            raise ValueError(f"saved model feature count differs for outer fold {fold}")
        best_iteration = int(item["best_iteration"])
        prediction = booster.predict(
            matrix,
            num_iteration=best_iteration,
        ).astype(np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"saved model prediction is non-finite for outer fold {fold}")
        column = f"segment_offset_outer_fold_{fold}"
        result[column] = prediction
        component_values.append(prediction)
        audit_rows.append(
            {
                "outer_fold": fold,
                "model_filename": model_path.name,
                "model_sha256": item["actual_model_sha256"],
                "best_iteration": best_iteration,
                "booster_current_iteration": int(booster.current_iteration()),
                "feature_count": len(feature_columns),
                "feature_order_match": True,
                "segment_rows": len(prediction),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
                "prediction_mean": float(prediction.mean()),
                "prediction_sha256": hashed_frame_sha256(
                    pd.DataFrame(
                        {
                            "well_id": result["well_id"],
                            "segment_id": result["segment_id"],
                            column: prediction,
                        }
                    ),
                    ("well_id", "segment_id", column),
                ),
            }
        )
    if len(component_values) != 5:
        raise ValueError("candidate inference did not use exactly five saved models")
    component_matrix = np.column_stack(component_values).astype(np.float64)
    result["segment_offset_pred"] = component_matrix.mean(axis=1, dtype=np.float64)
    if not np.isfinite(result["segment_offset_pred"].to_numpy(np.float64)).all():
        raise ValueError("five-model ensemble segment offset is non-finite")
    return result, pd.DataFrame(audit_rows)


def broadcast_candidate_rows(
    rows: pd.DataFrame,
    segment_predictions: pd.DataFrame,
    sample: pd.DataFrame,
) -> pd.DataFrame:
    component_columns = [
        f"segment_offset_outer_fold_{fold}" for fold in range(5)
    ]
    lookup_columns = [
        "well_id",
        "segment_id",
        "segment_offset_pred",
        *component_columns,
    ]
    predicted = rows.merge(
        segment_predictions[lookup_columns],
        on=["well_id", "segment_id"],
        how="left",
        validate="many_to_one",
    )
    if predicted[["segment_offset_pred", *component_columns]].isna().any().any():
        raise ValueError("segment prediction broadcast did not cover current-test rows")
    predicted["exp333_candidate_tvt"] = (
        predicted["exp226_tvt"].to_numpy(np.float64)
        + predicted["segment_offset_pred"].to_numpy(np.float64)
    )
    finite_columns = [
        "exp226_tvt",
        "segment_offset_pred",
        "exp333_candidate_tvt",
        *component_columns,
    ]
    if not np.isfinite(predicted[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("current-test candidate output contains non-finite values")
    candidate = sample[["id"]].merge(
        predicted[
            [
                "id",
                "well_id",
                "row_idx",
                "suffix_offset",
                "segment_id",
                "exp226_tvt",
                "segment_offset_pred",
                "exp333_candidate_tvt",
                *component_columns,
            ]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if len(candidate) != len(sample) or not candidate["id"].equals(sample["id"]):
        raise ValueError("candidate output row/order contract failed")
    if candidate.isna().any().any() or candidate["id"].duplicated().any():
        raise ValueError("candidate output coverage/uniqueness contract failed")
    return candidate


# %% [markdown]
# ## 7. Candidate artifacts and reproducibility manifest

# %%
def run_candidate_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    contract = validate_candidate_inference_contract(
        config, require_execution_authorization=True
    )
    artifacts = output_artifacts_dir()
    submission_path = KAGGLE_WORKING_ROOT / "submission.csv"
    if submission_path.exists():
        raise RuntimeError("submission.csv already exists before candidate-only inference")

    bootstrap = validate_bootstrap_sources(config)
    replay_source = import_file(
        "exp333_public_replay_source",
        Path(bootstrap["exp072_replay_source"]["path"]),
    )
    feature_source = import_file(
        "exp333_target_free_feature_source",
        Path(bootstrap["exp228_target_free_source"]["path"]),
    )
    train_contract = load_saved_train_contract(config)
    print(json.dumps(train_contract.parity, indent=2, sort_keys=True), flush=True)

    raw_root = resolve_raw_data_root()
    sample = pd.read_csv(raw_root / "sample_submission.csv", dtype={"id": str})
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"sample submission columns changed: {sample.columns.tolist()}")
    expected_rows = int(get_nested(config, "candidate_inference.base_submission_rows"))
    if len(sample) != expected_rows or sample["id"].duplicated().any():
        raise ValueError("sample submission row/ID contract failed")
    exp226, exp226_meta = load_exp226_current_test(config, sample)
    print(
        f"exp226 base: rows={len(exp226):,} sha={exp226_meta['submission_sha256']}",
        flush=True,
    )

    feature_started = time.perf_counter()
    surface = build_current_feature_surface(
        config,
        raw_root=raw_root,
        replay_source=replay_source,
        feature_source=feature_source,
        train_contract=train_contract,
        artifacts_dir=artifacts,
    )
    feature_seconds = time.perf_counter() - feature_started
    if len(surface.frame) != expected_rows:
        raise ValueError("current-test feature row count differs from sample submission")
    if int(surface.frame["well_id"].nunique()) != int(
        get_nested(config, "candidate_inference.expected_test_wells")
    ):
        raise ValueError("current-test feature well count changed")
    print(
        f"target-free surface: rows={len(surface.frame):,} "
        f"row_features={len(surface.feature_columns)} elapsed={feature_seconds:.1f}s",
        flush=True,
    )

    rows, boundary = assign_current_test_k16(
        surface,
        exp226,
        sample,
        raw_test_dir=raw_root / "test",
    )
    segments = aggregate_current_test_segments(rows, surface.feature_columns)
    model_feature_columns = [
        str(item) for item in train_contract.manifest["feature_columns"]
    ]
    if model_feature_columns != [
        *STRUCTURAL_FEATURE_COLUMNS,
        *surface.feature_columns,
    ]:
        raise ValueError("current-test model feature order differs from manifest")
    segment_feature_sha = hashed_frame_sha256(
        segments,
        ("well_id", "segment_id", *model_feature_columns),
    )

    inference_started = time.perf_counter()
    segment_predictions, model_audit = predict_saved_fold_ensemble(
        segments, train_contract
    )
    candidate = broadcast_candidate_rows(rows, segment_predictions, sample)
    inference_seconds = time.perf_counter() - inference_started
    print(
        f"saved-model inference: models=5 segments={len(segments)} "
        f"rows={len(candidate):,} elapsed={inference_seconds:.1f}s",
        flush=True,
    )

    candidate_path = artifacts / f"{OUTPUT_PREFIX}_current_test_candidate.csv.gz"
    segment_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_segment_predictions.csv"
    )
    feature_schema_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_feature_schema.csv"
    )
    projection_summary_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_projection_feature_summary.csv"
    )
    grwr_summary_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_grwr_feature_summary.csv"
    )
    model_audit_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_model_audit.csv"
    )
    boundary_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_boundary_audit.csv"
    )
    input_manifest_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_input_manifest.csv"
    )
    summary_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_summary.json"
    )
    sha_manifest_path = (
        artifacts / f"{OUTPUT_PREFIX}_current_test_sha_manifest.csv"
    )

    write_csv_gzip(candidate_path, candidate)
    segment_predictions.to_csv(segment_path, index=False)
    surface.schema.to_csv(feature_schema_path, index=False)
    surface.projection_summary.to_csv(projection_summary_path, index=False)
    surface.grwr_summary.to_csv(grwr_summary_path, index=False)
    model_audit.to_csv(model_audit_path, index=False)
    boundary.to_csv(boundary_path, index=False)
    input_manifest = pd.DataFrame(
        [
            {
                "name": "exp072_replay_source",
                **bootstrap["exp072_replay_source"],
            },
            {
                "name": "exp228_target_free_source",
                **bootstrap["exp228_target_free_source"],
            },
            {
                "name": "exp072_feature_schema",
                "path": surface.metadata["exp072_schema_path"],
                "sha256": surface.metadata["exp072_schema_sha256"],
            },
            {
                "name": "exp226_current_test",
                "path": exp226_meta["submission_path"],
                "sha256": exp226_meta["submission_sha256"],
            },
            {
                "name": "exp333_stage1_model_manifest",
                "path": str(train_contract.manifest_path),
                "sha256": train_contract.parity["model_manifest_sha256"],
            },
            *[
                {
                    "name": f"exp333_outer_fold_{row['outer_fold']}",
                    "path": row["resolved_model_path"],
                    "sha256": row["actual_model_sha256"],
                }
                for row in train_contract.model_rows
            ],
        ]
    )
    input_manifest.to_csv(input_manifest_path, index=False)

    component_columns = [
        f"segment_offset_outer_fold_{fold}" for fold in range(5)
    ]
    candidate_content_columns = (
        "id",
        "well_id",
        "row_idx",
        "suffix_offset",
        "segment_id",
        "exp226_tvt",
        "segment_offset_pred",
        "exp333_candidate_tvt",
        *component_columns,
    )
    prediction_content_sha = hashed_frame_sha256(
        candidate, candidate_content_columns
    )
    segment_prediction_sha = hashed_frame_sha256(
        segment_predictions,
        (
            "well_id",
            "segment_id",
            *component_columns,
            "segment_offset_pred",
        ),
    )
    candidate_evidence = artifact_evidence(candidate_path)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "status": "current_test_candidate_inference_completed",
        "authorization": contract,
        "original_direct_promotion_decision": train_contract.summary["decision"],
        "downstream_evidence": get_nested(
            config, "downstream_candidate_path_evidence"
        ),
        "rows": len(candidate),
        "wells": int(candidate["well_id"].nunique()),
        "segments": len(segment_predictions),
        "row_feature_count": len(surface.feature_columns),
        "model_feature_count": len(model_feature_columns),
        "saved_model_count": len(train_contract.model_rows),
        "trained_models_this_run": 0,
        "trained_boosters_this_run": 0,
        "parent_control_retraining": False,
        "submission_created": False,
        "competition_submit_performed": False,
        "exp226": exp226_meta,
        "saved_train_parity": train_contract.parity,
        "feature_surface": surface.metadata,
        "segment_feature_content_sha256": segment_feature_sha,
        "segment_prediction_content_sha256": segment_prediction_sha,
        "candidate_prediction_content_sha256": prediction_content_sha,
        "candidate_artifact": candidate_evidence,
        "prediction_distribution": {
            "exp226_min": float(candidate["exp226_tvt"].min()),
            "exp226_max": float(candidate["exp226_tvt"].max()),
            "exp226_mean": float(candidate["exp226_tvt"].mean()),
            "offset_min": float(candidate["segment_offset_pred"].min()),
            "offset_max": float(candidate["segment_offset_pred"].max()),
            "offset_mean": float(candidate["segment_offset_pred"].mean()),
            "offset_std": float(candidate["segment_offset_pred"].std()),
            "candidate_min": float(candidate["exp333_candidate_tvt"].min()),
            "candidate_max": float(candidate["exp333_candidate_tvt"].max()),
            "candidate_mean": float(candidate["exp333_candidate_tvt"].mean()),
            "candidate_std": float(candidate["exp333_candidate_tvt"].std()),
        },
        "technical_guards": {
            "sample_order_match": True,
            "id_unique": True,
            "row_coverage": 1.0,
            "well_coverage": 1.0,
            "k16_segments_per_well": True,
            "raw_suffix_boundary_match": True,
            "feature_schema_match": True,
            "model_feature_order_match": True,
            "model_sha_match": True,
            "saved_train_summary_parity": True,
            "five_fold_ensemble": True,
            "finite_base_offset_candidate": True,
            "submission_absent": True,
        },
        "timing_seconds": {
            "feature_generation": feature_seconds,
            "saved_model_inference": inference_seconds,
            "total": time.perf_counter() - started,
        },
        "artifacts": {
            "candidate": candidate_path.name,
            "segment_predictions": segment_path.name,
            "feature_schema": feature_schema_path.name,
            "projection_feature_summary": projection_summary_path.name,
            "grwr_feature_summary": grwr_summary_path.name,
            "model_audit": model_audit_path.name,
            "boundary_audit": boundary_path.name,
            "input_manifest": input_manifest_path.name,
            "summary": summary_path.name,
            "sha_manifest": sha_manifest_path.name,
        },
    }
    if submission_path.exists():
        raise RuntimeError("candidate-only inference unexpectedly created submission.csv")
    write_json(summary_path, summary)
    evidence_paths = [
        candidate_path,
        segment_path,
        feature_schema_path,
        projection_summary_path,
        grwr_summary_path,
        model_audit_path,
        boundary_path,
        input_manifest_path,
        summary_path,
    ]
    sha_manifest = pd.DataFrame(artifact_evidence(path) for path in evidence_paths)
    sha_manifest.to_csv(sha_manifest_path, index=False)
    if KAGGLE_WORKING_ROOT.is_dir():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %%
CONFIG = load_config()
CONTRACT_PREVIEW = validate_candidate_inference_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "selected_stage": get_nested(
                CONFIG, "execution_contract.selected_downstream_stage"
            ),
            "authorization": CONTRACT_PREVIEW,
            "submission_created": False,
        },
        indent=2,
        sort_keys=True,
    )
)
if EXECUTE_NOTEBOOK:
    SUMMARY = run_candidate_inference(CONFIG)
else:
    SUMMARY = {
        "status": "import_only",
        "authorization": CONTRACT_PREVIEW,
    }
