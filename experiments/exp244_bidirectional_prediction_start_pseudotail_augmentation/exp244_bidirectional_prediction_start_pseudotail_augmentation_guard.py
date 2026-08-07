# %% [markdown]
# # exp244 frozen-anchor parity preflight — guard
#
# This CPU-only notebook verifies that exp244 can evaluate future calibration on the
# exact exp218 official-start OOF surface. It does not train a model or predict test rows.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input resolution
# 3. Raw official-start surface
# 4. Frozen exp218 OOF and model-manifest audit
# 5. exp218-compatible fold reconstruction
# 6. Parity guards and generated files

# %%
from __future__ import annotations

import gzip
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
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_frozen_anchor_parity"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Configuration and input resolution


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


def output_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT
        if is_kaggle_runtime()
        else ROOT / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(nested(config, "data.train_dir", "data/raw/train")))
    local = configured if configured.is_absolute() else ROOT / configured
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    if local.exists() and any(local.glob(pattern)):
        return local
    if KAGGLE_INPUT_ROOT.exists():
        for source in sorted(KAGGLE_INPUT_ROOT.iterdir()):
            candidate = source / "train"
            if candidate.is_dir() and any(candidate.glob(pattern)):
                return candidate
        for match in sorted(KAGGLE_INPUT_ROOT.rglob(pattern)):
            if match.parent.name == "train":
                return match.parent
    raise FileNotFoundError(f"Could not resolve raw train directory with {pattern}")


def resolve_artifact(configured: Any, filename: str, label: str) -> Path:
    candidates: list[Path] = []
    if configured:
        path = Path(str(configured))
        candidates.extend([path, ROOT / path] if not path.is_absolute() else [path])
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.rglob(filename)))
    for path in candidates:
        if path.exists() and path.is_file() and path.name == filename:
            return path
    raise FileNotFoundError(f"Could not resolve {label}: {filename}")


def sha256_file(path: Path, *, decompressed_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed_gzip else Path.open
    if decompressed_gzip:
        handle = opener(path, "rb")
    else:
        handle = opener(path, "rb")
    with handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class HashWriter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: str) -> int:
        encoded = value.encode()
        self.digest.update(encoded)
        return len(value)

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def canonical_csv_sha256(frame: pd.DataFrame) -> str:
    writer = HashWriter()
    frame.to_csv(writer, index=False, lineterminator="\n")
    return writer.hexdigest()


# %% [markdown]
# ## 3. Raw official-start surface


# %%
def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def build_raw_surface(train_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    rows: list[dict[str, Any]] = []
    for path in sorted(train_dir.glob(pattern), key=well_id_from_path):
        frame = pd.read_csv(path, usecols=[input_column])
        values = pd.to_numeric(frame[input_column], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(values))
        if known.size == 0 or not np.array_equal(known, np.arange(int(known[-1]) + 1)):
            raise AssertionError(f"Non-contiguous known prefix: {path.name}")
        official_start = int(known[-1])
        tail_rows = int(len(frame) - official_start - 1)
        if tail_rows <= 0:
            raise AssertionError(f"No official tail: {path.name}")
        rows.append(
            {
                "well_id": well_id_from_path(path),
                "n_rows": int(len(frame)),
                "official_start_index": official_start,
                "official_tail_rows": tail_rows,
                "expected_first_oof_index": official_start + 1,
                "expected_last_oof_index": len(frame) - 1,
            }
        )
    surface = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    if surface.empty or surface["well_id"].duplicated().any():
        raise AssertionError("Raw official-start surface is empty or duplicated")
    return surface


# %% [markdown]
# ## 4. Frozen exp218 OOF and model-manifest audit


# %%
def audit_exp218_oof(path: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target",
        "target_tvt",
        "pred_tvt",
    ]
    expected_variant = str(nested(config, "frozen_anchor_parity.variant"))
    expected_mode = str(nested(config, "frozen_anchor_parity.mode"))
    expected_model = str(nested(config, "frozen_anchor_parity.model"))
    aggregates: dict[str, dict[str, int]] = {}
    rows = 0
    squared_error = 0.0
    previous_well: str | None = None
    previous_index: int | None = None
    for chunk in pd.read_csv(path, chunksize=200_000):
        missing = [column for column in required if column not in chunk]
        if missing:
            raise ValueError(f"exp218 OOF is missing columns: {missing}")
        if set(chunk["variant"].astype(str)) != {expected_variant}:
            raise AssertionError("Unexpected exp218 OOF variant")
        if set(chunk["mode"].astype(str)) != {expected_mode}:
            raise AssertionError("Unexpected exp218 OOF mode")
        if set(chunk["model"].astype(str)) != {expected_model}:
            raise AssertionError("Unexpected exp218 OOF model")
        numeric = chunk[["last_known_tvt", "target", "target_tvt", "pred_tvt"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.all(np.isfinite(numeric.to_numpy(float))):
            raise AssertionError("exp218 OOF contains non-finite numeric values")
        if not np.allclose(
            numeric["last_known_tvt"] + numeric["target"],
            numeric["target_tvt"],
            atol=0.002,
            rtol=0.0,
        ):
            raise AssertionError("exp218 OOF target_tvt does not match anchor plus target")
        error = numeric["pred_tvt"].to_numpy(float) - numeric["target_tvt"].to_numpy(float)
        squared_error += float(error @ error)
        rows += len(chunk)
        for record in chunk[["id", "well"]].itertuples(index=False):
            well = str(record.well)
            row_id = str(record.id)
            prefix, separator, suffix = row_id.rpartition("_")
            if separator != "_" or prefix != well:
                raise AssertionError(f"Malformed exp218 OOF id: {row_id}")
            index = int(suffix)
            if previous_well == well and previous_index is not None and index != previous_index + 1:
                raise AssertionError(f"Non-contiguous exp218 OOF rows for {well}")
            if previous_well is not None and well < previous_well:
                raise AssertionError("exp218 OOF wells are not stably sorted")
            stats = aggregates.setdefault(
                well,
                {"oof_rows": 0, "oof_first_index": index, "oof_last_index": index},
            )
            stats["oof_rows"] += 1
            stats["oof_first_index"] = min(stats["oof_first_index"], index)
            stats["oof_last_index"] = max(stats["oof_last_index"], index)
            previous_well = well
            previous_index = index
    by_well = (
        pd.DataFrame([{"well_id": well, **values} for well, values in aggregates.items()])
        .sort_values("well_id")
        .reset_index(drop=True)
    )
    summary = {
        "rows": int(rows),
        "wells": int(len(by_well)),
        "rmse": float(np.sqrt(squared_error / rows)),
        "decompressed_sha256": sha256_file(path, decompressed_gzip=True),
        "variant": expected_variant,
        "mode": expected_mode,
        "model": expected_model,
    }
    return by_well, summary


def audit_model_manifest(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    expected_experiment = str(nested(config, "frozen_anchor_parity.experiment"))
    expected_variant = str(nested(config, "frozen_anchor_parity.variant"))
    expected_mode = str(nested(config, "frozen_anchor_parity.mode"))
    expected_count = int(nested(config, "frozen_anchor_parity.expected_model_count"))
    models = manifest.get("models", [])
    if manifest.get("experiment") != expected_experiment:
        raise AssertionError("Unexpected exp218 model-manifest experiment")
    if int(manifest.get("model_count", -1)) != expected_count or len(models) != expected_count:
        raise AssertionError("Unexpected exp218 model count")
    if {str(item.get("variant")) for item in models} != {expected_variant}:
        raise AssertionError("Unexpected model-manifest variant")
    if {str(item.get("mode")) for item in models} != {expected_mode}:
        raise AssertionError("Unexpected model-manifest mode")
    if {int(item.get("fold")) for item in models} != set(range(5)):
        raise AssertionError("exp218 model manifest does not cover folds 0..4")
    missing_files = [
        str(item.get("file")) for item in models if not (path.parent / item["file"]).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing exp218 saved boosters: {missing_files[:3]}")
    return {
        "experiment": expected_experiment,
        "model_count": len(models),
        "folds": sorted({int(item["fold"]) for item in models}),
        "model_families": sorted({str(item["model"]) for item in models}),
        "sha256": sha256_file(path),
        "all_model_files_present": True,
    }


# %% [markdown]
# ## 5. exp218-compatible fold reconstruction


# %%
def weighted_groupkfold(wells: np.ndarray, weights: np.ndarray, n_folds: int) -> np.ndarray:
    if len(wells) < n_folds or np.any(weights <= 0):
        raise ValueError("Invalid groups or weights for GroupKFold reconstruction")
    order = np.argsort(weights)[::-1]
    fold_loads = np.zeros(n_folds, dtype=np.int64)
    assignments = np.full(len(wells), -1, dtype=int)
    for index in order:
        fold = int(np.argmin(fold_loads))
        assignments[index] = fold
        fold_loads[fold] += int(weights[index])
    return assignments


def build_parity_manifest(
    raw_surface: pd.DataFrame,
    oof_by_well: pd.DataFrame,
    v1_folds: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    manifest = raw_surface.merge(
        oof_by_well, on="well_id", how="outer", validate="1:1", indicator=True
    )
    if not manifest["_merge"].eq("both").all():
        raise AssertionError("Raw train wells and exp218 OOF wells do not match")
    manifest = manifest.drop(columns="_merge").sort_values("well_id").reset_index(drop=True)
    if not np.array_equal(
        manifest["official_tail_rows"].to_numpy(np.int64),
        manifest["oof_rows"].to_numpy(np.int64),
    ):
        raise AssertionError("Raw official-tail rows and exp218 OOF rows differ")
    if not np.array_equal(
        manifest["expected_first_oof_index"].to_numpy(np.int64),
        manifest["oof_first_index"].to_numpy(np.int64),
    ) or not np.array_equal(
        manifest["expected_last_oof_index"].to_numpy(np.int64),
        manifest["oof_last_index"].to_numpy(np.int64),
    ):
        raise AssertionError("Raw official-tail ID range and exp218 OOF ID range differ")
    wells = manifest["well_id"].to_numpy(str)
    weights = manifest["official_tail_rows"].to_numpy(np.int64)
    n_folds = int(nested(config, "validation.n_folds", 5))
    required_v1 = {"well_id", "fold"}
    if not required_v1.issubset(v1_folds.columns):
        raise ValueError(
            f"v1 fold manifest is missing columns: {sorted(required_v1 - set(v1_folds))}"
        )
    legacy = v1_folds[["well_id", "fold"]].rename(columns={"fold": "fold_v1_unique_well"})
    if legacy["well_id"].duplicated().any():
        raise AssertionError("v1 fold manifest has duplicated wells")
    manifest = manifest.merge(legacy, on="well_id", how="left", validate="1:1")
    if manifest["fold_v1_unique_well"].isna().any():
        raise AssertionError("v1 fold manifest does not cover all exp218 wells")
    manifest["fold_v1_unique_well"] = manifest["fold_v1_unique_well"].astype(int)
    manifest["fold_v2_exp218_weighted"] = weighted_groupkfold(wells, weights, n_folds)
    manifest["fold_changed_from_v1"] = (
        manifest["fold_v1_unique_well"] != manifest["fold_v2_exp218_weighted"]
    )
    manifest["fold_assignment"] = str(nested(config, "validation.fold_assignment"))
    return manifest


# %% [markdown]
# ## 6. Parity guards and generated files


# %%
def run_guard() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    oof_filename = "exp218_gr_wavelet_rotation_confidence_features_on_exp148_predictions.csv.gz"
    manifest_filename = "manifest.json"
    v1_fold_filename = f"{EXPERIMENT_NAME}_fold_manifest.csv"
    oof_path = resolve_artifact(
        nested(config, "data.exp218_oof_predictions_local"), oof_filename, "exp218 OOF"
    )
    model_manifest_path = resolve_artifact(
        nested(config, "data.exp218_model_manifest_local"),
        manifest_filename,
        "exp218 model manifest",
    )
    v1_fold_path = resolve_artifact(
        nested(config, "data.exp244_v1_fold_manifest_local"),
        v1_fold_filename,
        "exp244 v1 fold manifest",
    )
    if "exp218_gr_wavelet_rotation_confidence_features_on_exp148_lgb_models" not in str(
        model_manifest_path.parent
    ):
        candidates = [
            path
            for path in KAGGLE_INPUT_ROOT.rglob(manifest_filename)
            if "exp218_gr_wavelet_rotation_confidence_features_on_exp148_lgb_models"
            in str(path.parent)
        ]
        if candidates:
            model_manifest_path = sorted(candidates)[0]
    raw_surface = build_raw_surface(resolve_train_dir(config), config)
    oof_by_well, oof_summary = audit_exp218_oof(oof_path, config)
    model_summary = audit_model_manifest(model_manifest_path, config)
    v1_folds = pd.read_csv(v1_fold_path, dtype={"well_id": str})
    v1_fold_sha = canonical_csv_sha256(v1_folds)
    parity = build_parity_manifest(raw_surface, oof_by_well, v1_folds, config)
    expected_rows = int(nested(config, "frozen_anchor_parity.expected_rows"))
    expected_wells = int(nested(config, "frozen_anchor_parity.expected_wells"))
    expected_rmse = float(nested(config, "frozen_anchor_parity.expected_rmse"))
    tolerance = float(nested(config, "frozen_anchor_parity.rmse_tolerance", 1e-6))
    expected_oof_sha = str(nested(config, "frozen_anchor_parity.expected_oof_decompressed_sha256"))
    expected_manifest_sha = str(
        nested(config, "frozen_anchor_parity.expected_model_manifest_sha256")
    )
    expected_v1_fold_sha = str(
        nested(config, "frozen_anchor_parity.expected_v1_fold_manifest_sha256")
    )
    guards = {
        "oof_rows_match": oof_summary["rows"] == expected_rows,
        "oof_wells_match": oof_summary["wells"] == expected_wells,
        "oof_rmse_match": abs(oof_summary["rmse"] - expected_rmse) <= tolerance,
        "oof_decompressed_sha_match": oof_summary["decompressed_sha256"] == expected_oof_sha,
        "model_manifest_sha_match": model_summary["sha256"] == expected_manifest_sha,
        "v1_fold_manifest_sha_match": v1_fold_sha == expected_v1_fold_sha,
        "raw_official_surface_match": True,
        "all_model_files_present": model_summary["all_model_files_present"],
        "model_training_forbidden": True,
        "test_prediction_forbidden": True,
        "submission_forbidden": True,
    }
    if not all(guards.values()):
        raise AssertionError(f"Frozen-anchor parity guards failed: {guards}")
    fold_report = (
        parity.groupby("fold_v2_exp218_weighted")
        .agg(wells=("well_id", "size"), official_tail_rows=("official_tail_rows", "sum"))
        .reset_index()
        .rename(columns={"fold_v2_exp218_weighted": "fold"})
    )
    out = output_dir()
    parity_path = out / f"{OUTPUT_PREFIX}_fold_manifest.csv"
    fold_report_path = out / f"{OUTPUT_PREFIX}_fold_report.csv"
    parity.to_csv(parity_path, index=False)
    fold_report.to_csv(fold_report_path, index=False)
    changed = int(parity["fold_changed_from_v1"].sum())
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "frozen_anchor_parity_preflight_passed",
        "route": nested(config, "experiment.route"),
        "runtime": "kaggle_cpu" if is_kaggle_runtime() else "local_debug",
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_anchor_oof": oof_summary,
        "frozen_anchor_models": model_summary,
        "fold_parity": {
            "assignment": nested(config, "validation.fold_assignment"),
            "matching_v1_wells": int(len(parity) - changed),
            "changed_from_v1_wells": changed,
            "v1_match_rate": float((len(parity) - changed) / len(parity)),
            "v1_fold_manifest_sha256": v1_fold_sha,
            "fold_rows": {
                str(row.fold): int(row.official_tail_rows)
                for row in fold_report.itertuples(index=False)
            },
        },
        "guards": guards,
        "content_sha256": {
            "parity_fold_manifest": canonical_csv_sha256(parity),
            "fold_report": canonical_csv_sha256(fold_report),
        },
        "execution": {
            "active_audits": 1,
            "lightgbm_configs": 0,
            "folds_trained": 0,
            "boosters": 0,
            "parent_control_retrained": False,
        },
        "model_training_performed": False,
        "inference_prediction_performed": False,
        "submission_created": False,
        "generated_files": {
            "parity_fold_manifest": str(parity_path),
            "fold_report": str(fold_report_path),
        },
    }
    summary_path = out / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(parity.head(10).to_string(index=False))
    print(fold_report.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %% [markdown]
# The first execution is a Kaggle CPU guard. Local execution remains blocked unless an
# approved smoke debug explicitly sets `EXPERIMENT_ALLOW_LOCAL=1`.

# %%
if not is_kaggle_runtime() and os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError(
        "This notebook is Kaggle-first. Set EXPERIMENT_ALLOW_LOCAL=1 only for an approved debug."
    )

GUARD_SUMMARY = run_guard()
