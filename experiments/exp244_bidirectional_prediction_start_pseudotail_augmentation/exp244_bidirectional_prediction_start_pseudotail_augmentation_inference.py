# %% [markdown]
# # exp244 bidirectional prediction-start pseudo-tail augmentation — inference audit
#
# This notebook creates current-test-compatible rolling-prefix calibration requests.
# It never reads TVT after the actual prediction start and never creates a submission.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and current-test input checks
# 3. Known-prefix calibration request generation
# 4. Prefix-only backtest feature materialization
# 5. Leakage and no-submission guards
# 6. Audit orchestration and generated files

# %%
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp244_bidirectional_prediction_start_pseudotail_augmentation"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Configuration and current-test input checks


# %%
def find_repo_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


ROOT = find_repo_root()


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        ROOT / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if value.get("experiment", {}).get("name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError(f"Could not resolve config.yaml for {EXPERIMENT_NAME}")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(find_config_path().read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def resolve_test_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(nested(config, "data.test_dir", "data/raw/test")))
    local = configured if configured.is_absolute() else ROOT / configured
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    if local.exists() and any(local.glob(pattern)):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        for source in sorted(KAGGLE_INPUT_ROOT.iterdir()):
            candidate = source / "test"
            if candidate.is_dir() and any(candidate.glob(pattern)):
                return candidate
        for match in sorted(KAGGLE_INPUT_ROOT.rglob(pattern)):
            if match.parent.name == "test":
                return match.parent
    raise FileNotFoundError("Could not resolve current-test directory")


def output_dir() -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT
    else:
        path = ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def stable_key(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


class HashWriter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        self.digest.update(value.encode())
        return len(value)

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def canonical_csv_sha256(frame: pd.DataFrame) -> str:
    writer = HashWriter()
    frame.to_csv(writer, index=False, lineterminator="\n")
    return writer.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_test_well(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        str(nested(config, "data.md_column", "MD")),
        *list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"])),
        str(nested(config, "data.gr_column", "GR")),
        str(nested(config, "data.input_target_column", "TVT_input")),
    ]
    frame = pd.read_csv(path, usecols=lambda value: value in set(columns))
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def actual_start_index(frame: pd.DataFrame, config: dict[str, Any], label: str) -> int:
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    values = pd.to_numeric(frame[input_column], errors="coerce").to_numpy(float)
    known = np.flatnonzero(np.isfinite(values))
    if known.size == 0:
        raise ValueError(f"{label} has no known TVT_input prefix")
    actual = int(known[-1])
    if not np.array_equal(known, np.arange(actual + 1)):
        raise ValueError(f"{label} TVT_input is not one contiguous prefix")
    if actual >= len(frame) - 1:
        raise ValueError(f"{label} has no unknown evaluation tail")
    return actual


# %% [markdown]
# ## 3. Known-prefix calibration request generation


# %%
def build_calibration_requests(
    test_files: list[Path], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    offsets = sorted(
        int(value)
        for value in nested(config, "model.view_generation.start_offsets_rows", [])
        if int(value) < 0
    )
    min_prefix = int(nested(config, "model.view_generation.min_prefix_rows", 200))
    min_eval = int(nested(config, "model.calibration_backtest.min_known_evaluation_rows", 50))
    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(test_files, key=well_id_from_path):
        well = well_id_from_path(path)
        frame = read_test_well(path, config)
        frames[well] = frame
        actual = actual_start_index(frame, config, path.name)
        for offset in offsets:
            pseudo_start = actual + offset
            known_eval_rows = actual - pseudo_start
            if pseudo_start + 1 < min_prefix or known_eval_rows < min_eval:
                continue
            request_id = stable_key(
                EXPERIMENT_NAME, "current_test_calibration", well, actual, offset
            )
            rows.append(
                {
                    "request_id": request_id,
                    "source_well": well,
                    "source_path": str(path),
                    "request_purpose": "known_prefix_calibration_backtest",
                    "start_kind": "early",
                    "start_offset_rows": offset,
                    "pseudo_start_index": pseudo_start,
                    "actual_prediction_start_index": actual,
                    "calibration_end_index": actual,
                    "known_calibration_rows": known_eval_rows,
                    "unknown_tail_rows": len(frame) - actual - 1,
                    "current_test_compatible": True,
                    "actual_prediction_start_may_be_exceeded": False,
                    "target_usage": "known_prefix_calibration_only",
                    "full_model_finetune_allowed": False,
                    "submission_prediction_allowed": False,
                }
            )
    requests = (
        pd.DataFrame(rows).sort_values(["source_well", "start_offset_rows"]).reset_index(drop=True)
    )
    if requests.empty:
        raise AssertionError("No current-test calibration requests are feasible")
    return requests, frames


def assert_calibration_contract(requests: pd.DataFrame) -> dict[str, Any]:
    if requests["request_id"].duplicated().any():
        raise AssertionError("Duplicate calibration request ID")
    if not requests["start_kind"].eq("early").all():
        raise AssertionError("Current-test calibration requests must be early-start")
    if not (requests["pseudo_start_index"] < requests["actual_prediction_start_index"]).all():
        raise AssertionError("A calibration request reaches or exceeds actual start")
    if not (requests["calibration_end_index"] == requests["actual_prediction_start_index"]).all():
        raise AssertionError("Calibration evaluation must stop at actual start")
    if requests["actual_prediction_start_may_be_exceeded"].any():
        raise AssertionError("Unknown current-test tail access is enabled")
    if requests["full_model_finetune_allowed"].any():
        raise AssertionError("Full-model test-time fine-tuning is enabled")
    if requests["submission_prediction_allowed"].any():
        raise AssertionError("Calibration audit is allowed to create submission predictions")
    return {
        "request_ids_unique": True,
        "only_early_known_prefix_requests": True,
        "actual_prediction_start_not_exceeded": True,
        "unknown_tail_tvt_forbidden": True,
        "full_model_finetune_forbidden": True,
        "submission_prediction_forbidden": True,
    }


# %% [markdown]
# ## 4. Prefix-only backtest feature materialization


# %%
def evenly_spaced_indices(indices: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if count <= 0 or values.size == 0:
        return np.empty(0, dtype=int)
    if values.size <= count:
        return values
    positions = np.linspace(0, values.size - 1, count).round().astype(int)
    return values[np.unique(positions)]


def robust_rate(values: np.ndarray, md: np.ndarray) -> float:
    delta_md = np.diff(md)
    delta_value = np.diff(values)
    valid = np.isfinite(delta_md) & np.isfinite(delta_value) & (np.abs(delta_md) > 1e-12)
    if not np.any(valid):
        return 0.0
    return float(np.median(delta_value[valid] / delta_md[valid]))


def materialize_calibration_backtests(
    requests: pd.DataFrame, frames: dict[str, pd.DataFrame], config: dict[str, Any]
) -> pd.DataFrame:
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    md_column = str(nested(config, "data.md_column", "MD"))
    gr_column = str(nested(config, "data.gr_column", "GR"))
    coordinates = list(nested(config, "data.coordinate_columns", ["X", "Y", "Z"]))
    max_rows = int(nested(config, "model.materialization.max_rows_per_view", 1000))
    parts: list[pd.DataFrame] = []
    for request in requests.sort_values("request_id").itertuples(index=False):
        original = frames[str(request.source_well)]
        known_tvt = pd.to_numeric(original[input_column], errors="coerce").to_numpy(float)
        pseudo_start = int(request.pseudo_start_index)
        actual_start = int(request.actual_prediction_start_index)
        feature_frame = original.copy()
        feature_frame[input_column] = np.nan
        feature_frame.loc[:pseudo_start, input_column] = known_tvt[: pseudo_start + 1]
        if feature_frame.loc[pseudo_start + 1 :, input_column].notna().any():
            raise AssertionError("Pseudo-start feature frame exposes later known TVT")
        calibration_rows = np.arange(pseudo_start + 1, actual_start + 1, dtype=int)
        selected = evenly_spaced_indices(calibration_rows, max_rows)
        if selected.size == 0 or np.any(selected > actual_start):
            raise AssertionError("Invalid known-prefix calibration row selection")
        md = pd.to_numeric(feature_frame[md_column], errors="coerce").to_numpy(float)
        gr = pd.to_numeric(feature_frame[gr_column], errors="coerce").to_numpy(float)
        xyz = feature_frame[coordinates].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pseudo_tvt = pd.to_numeric(feature_frame[input_column], errors="coerce").to_numpy(float)
        prefix_gr = gr[: pseudo_start + 1]
        prefix_md = md[: pseudo_start + 1]
        prefix_tvt = pseudo_tvt[: pseudo_start + 1]
        anchor_xyz = xyz[pseudo_start]
        part = pd.DataFrame(
            {
                "request_id": str(request.request_id),
                "source_well": str(request.source_well),
                "request_purpose": str(request.request_purpose),
                "start_offset_rows": int(request.start_offset_rows),
                "pseudo_start_index": pseudo_start,
                "actual_prediction_start_index": actual_start,
                "row_index": selected,
                "eval_step": selected - pseudo_start - 1,
                "anchor_tvt_input": float(pseudo_tvt[pseudo_start]),
                "anchor_md": float(md[pseudo_start]),
                "delta_md": md[selected] - md[pseudo_start],
                "row_gr": gr[selected],
                "delta_x": xyz[selected, 0] - anchor_xyz[0],
                "delta_y": xyz[selected, 1] - anchor_xyz[1],
                "delta_z": xyz[selected, 2] - anchor_xyz[2],
                "prefix_gr_mean": float(np.nanmean(prefix_gr))
                if np.any(np.isfinite(prefix_gr))
                else 0.0,
                "prefix_gr_std": float(np.nanstd(prefix_gr))
                if np.any(np.isfinite(prefix_gr))
                else 0.0,
                "prefix_gr_missing_rate": float(np.mean(~np.isfinite(prefix_gr))),
                "prefix_tvt_md_rate": robust_rate(prefix_tvt, prefix_md),
                "known_prefix_target_tvt": known_tvt[selected],
            }
        )
        if not np.all(np.isfinite(part["known_prefix_target_tvt"])):
            raise AssertionError("Calibration labels must come from finite known prefix TVT_input")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["request_id", "row_index"]).reset_index(drop=True)


# %% [markdown]
# ## 5. Leakage and no-submission guards


# %%
def write_outputs(
    out: Path,
    requests: pd.DataFrame,
    features: pd.DataFrame,
    guards: dict[str, Any],
    elapsed_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    request_path = out / f"{OUTPUT_PREFIX}_calibration_request_manifest.csv"
    feature_path = out / f"{OUTPUT_PREFIX}_calibration_backtest_features.csv.gz"
    schema_path = out / f"{OUTPUT_PREFIX}_calibration_feature_schema.csv"
    requests.to_csv(request_path, index=False)
    features.to_csv(feature_path, index=False, compression={"method": "gzip", "mtime": 0})
    schema = pd.DataFrame(
        {
            "column": features.columns,
            "dtype": [str(features[column].dtype) for column in features],
        }
    )
    schema.to_csv(schema_path, index=False)
    forbidden_outputs = [out / "submission.csv"]
    if any(path.exists() for path in forbidden_outputs):
        raise AssertionError("Calibration audit must not create submission.csv")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "current_test_calibration_request_audit_complete",
        "runtime": "kaggle_cpu" if is_kaggle_runtime() else "local_debug",
        "elapsed_seconds": elapsed_seconds,
        "test_wells": int(requests["source_well"].nunique()),
        "request_count": int(len(requests)),
        "materialized_known_prefix_rows": int(len(features)),
        "guards": guards,
        "content_sha256": {
            "request_manifest": canonical_csv_sha256(requests),
            "feature_content_decompressed": canonical_csv_sha256(features),
            "feature_schema": canonical_csv_sha256(schema),
        },
        "raw_gzip_file_sha256_non_primary": sha256_file(feature_path),
        "allowed_parameters": nested(config, "model.calibration_backtest.allowed_parameters", []),
        "full_model_finetune_performed": False,
        "inference_prediction_performed": False,
        "submission_created": False,
        "generated_files": {
            "request_manifest": str(request_path),
            "features": str(feature_path),
            "schema": str(schema_path),
        },
    }
    summary_path = out / f"{OUTPUT_PREFIX}_inference_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 6. Audit orchestration and generated files


# %%
def run_inference_audit() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    if bool(nested(config, "model.execution.inference_prediction_enabled", False)):
        raise AssertionError("inference prediction must stay disabled")
    if bool(nested(config, "model.execution.submission_enabled", False)):
        raise AssertionError("submission must stay disabled")
    test_dir = resolve_test_dir(config)
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    test_files = sorted(test_dir.glob(pattern), key=well_id_from_path)
    if not test_files:
        raise FileNotFoundError(f"No current-test wells found in {test_dir}")
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": nested(config, "experiment.route"),
                "test_dir": str(test_dir),
                "test_wells": len(test_files),
                "calibration_offsets": [
                    value
                    for value in nested(config, "model.view_generation.start_offsets_rows", [])
                    if int(value) < 0
                ],
                "inference_prediction_enabled": False,
                "submission_enabled": False,
            },
            indent=2,
        )
    )
    requests, frames = build_calibration_requests(test_files, config)
    guards = assert_calibration_contract(requests)
    features = materialize_calibration_backtests(requests, frames, config)
    guards["all_materialized_rows_within_known_prefix"] = bool(
        (features["row_index"] <= features["actual_prediction_start_index"]).all()
    )
    if not all(guards.values()):
        raise AssertionError(f"One or more calibration guards failed: {guards}")
    summary = write_outputs(
        output_dir(),
        requests,
        features,
        guards,
        time.perf_counter() - started,
        config,
    )
    print(requests.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %%
if not is_kaggle_runtime() and os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError(
        "This notebook is Kaggle-first. Set EXPERIMENT_ALLOW_LOCAL=1 only for "
        "an approved local smoke debug."
    )

INFERENCE_AUDIT_SUMMARY = run_inference_audit()
