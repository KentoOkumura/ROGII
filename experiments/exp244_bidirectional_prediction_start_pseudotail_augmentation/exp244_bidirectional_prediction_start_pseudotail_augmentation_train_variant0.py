# %% [markdown]
# # exp244 dual-start confidence-shrink meta-validation
#
# This CPU-only notebook evaluates one pre-registered, target-unfitted shrink formula on
# frozen exp218 OOF predictions. It never trains or fine-tunes the parent model.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and input identity
# 3. Known-prefix dual-start backtest features
# 4. Frozen exp218 OOF assembly
# 5. Official-start metrics and adoption guards
# 6. Generated files and reproducibility summary

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
VARIANT_KIND = "dual_start_confidence_shrink"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_{VARIANT_KIND}"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Configuration and input identity


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


def sha256_file(path: Path, *, decompressed_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed_gzip:
        handle = gzip.open(path, "rb")
    else:
        handle = path.open("rb")
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


def resolve_artifact(filename: str, configured: Any = None) -> Path:
    candidates: list[Path] = []
    if configured:
        path = Path(str(configured))
        candidates.extend([path, ROOT / path] if not path.is_absolute() else [path])
    candidates.extend(
        [
            PACKAGE_DIR / "dependencies" / filename,
            ROOT / "dependencies" / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.rglob(filename)))
    for path in candidates:
        if path.exists() and path.is_file() and path.name == filename:
            return path
    raise FileNotFoundError(f"Could not resolve required artifact: {filename}")


def assert_input_sha(path: Path, expected: str, *, decompressed_gzip: bool = False) -> str:
    actual = sha256_file(path, decompressed_gzip=decompressed_gzip)
    if actual != expected:
        raise AssertionError(f"Input SHA mismatch for {path.name}: {actual} != {expected}")
    return actual


# %% [markdown]
# ## 3. Known-prefix dual-start backtest features


# %%
def well_id_from_path(path: Path) -> str:
    suffix = "__horizontal_well.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected horizontal filename: {path.name}")
    return path.name[: -len(suffix)]


def local_linear_backtest(
    md: np.ndarray,
    tvt: np.ndarray,
    pseudo_start: int,
    actual_start: int,
    window_rows: int,
) -> dict[str, float]:
    fit_start = max(0, pseudo_start - window_rows + 1)
    fit_rows = np.arange(fit_start, pseudo_start + 1, dtype=int)
    eval_rows = np.arange(pseudo_start + 1, actual_start + 1, dtype=int)
    if len(fit_rows) < 2 or len(eval_rows) == 0:
        raise ValueError("Insufficient rows for local-linear known-prefix backtest")
    x0 = float(md[pseudo_start])
    x = md[fit_rows] - x0
    y = tvt[fit_rows]
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        raise ValueError("Insufficient finite rows for local-linear fit")
    design = np.column_stack([x[valid], np.ones(int(valid.sum()), dtype=float)])
    slope, intercept = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    prediction = intercept + slope * (md[eval_rows] - x0)
    error = prediction - tvt[eval_rows]
    hold_error = tvt[pseudo_start] - tvt[eval_rows]
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(hold_error)):
        raise AssertionError("Known-prefix backtest produced non-finite errors")
    return {
        "linear_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "linear_bias": float(np.mean(error)),
        "hold_rmse": float(np.sqrt(np.mean(np.square(hold_error)))),
        "slope": float(slope),
        "fit_rows": int(valid.sum()),
        "eval_rows": int(len(eval_rows)),
    }


def alpha_from_dual_start(rmse_values: list[float], config: dict[str, Any]) -> float:
    lower = float(nested(config, "confidence_shrink_meta_validation.risk_no_shrink_rmse", 10.0))
    upper = float(nested(config, "confidence_shrink_meta_validation.risk_max_shrink_rmse", 30.0))
    maximum_shrink = float(
        nested(config, "confidence_shrink_meta_validation.max_shrink_fraction", 0.05)
    )
    risk = float(min(rmse_values))
    severity = float(np.clip((risk - lower) / (upper - lower), 0.0, 1.0))
    alpha = 1.0 - maximum_shrink * severity
    alpha_min = float(nested(config, "confidence_shrink_meta_validation.alpha_min", 0.95))
    alpha_max = float(nested(config, "confidence_shrink_meta_validation.alpha_max", 1.0))
    return float(np.clip(alpha, alpha_min, alpha_max))


def build_calibration_features(train_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    pattern = str(nested(config, "data.horizontal_glob", "*__horizontal_well.csv"))
    md_column = str(nested(config, "data.md_column", "MD"))
    target_column = str(nested(config, "data.target_column", "TVT"))
    input_column = str(nested(config, "data.input_target_column", "TVT_input"))
    offsets = [
        int(value)
        for value in nested(config, "confidence_shrink_meta_validation.early_offsets_rows")
    ]
    if offsets != [-1000, -250]:
        raise AssertionError("This pre-registered variant requires offsets [-1000, -250]")
    window = int(nested(config, "confidence_shrink_meta_validation.local_linear_window_rows", 128))
    rows: list[dict[str, Any]] = []
    for path in sorted(train_dir.glob(pattern), key=well_id_from_path):
        frame = pd.read_csv(path, usecols=[md_column, target_column, input_column])
        md = pd.to_numeric(frame[md_column], errors="coerce").to_numpy(float)
        tvt = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(float)
        tvt_input = pd.to_numeric(frame[input_column], errors="coerce").to_numpy(float)
        known = np.flatnonzero(np.isfinite(tvt_input))
        if known.size == 0 or not np.array_equal(known, np.arange(int(known[-1]) + 1)):
            raise AssertionError(f"Non-contiguous known prefix: {path.name}")
        actual_start = int(known[-1])
        if not np.allclose(tvt_input[: actual_start + 1], tvt[: actual_start + 1]):
            raise AssertionError(f"Known TVT_input differs from TVT: {path.name}")
        result: dict[str, Any] = {
            "well_id": well_id_from_path(path),
            "official_start_index": actual_start,
            "official_tail_rows": int(len(frame) - actual_start - 1),
            "official_anchor_tvt": float(tvt_input[actual_start]),
            "eligible_dual_start": True,
        }
        linear_rmses: list[float] = []
        for offset in offsets:
            label = f"m{abs(offset)}"
            pseudo_start = actual_start + offset
            if pseudo_start < 1:
                result["eligible_dual_start"] = False
                break
            audit = local_linear_backtest(md, tvt, pseudo_start, actual_start, window)
            for name, value in audit.items():
                result[f"{name}_{label}"] = value
            result[f"pseudo_start_index_{label}"] = pseudo_start
            linear_rmses.append(float(audit["linear_rmse"]))
        if bool(result["eligible_dual_start"]) and len(linear_rmses) == len(offsets):
            result["dual_start_risk_rmse"] = float(min(linear_rmses))
            result["dual_start_rmse_gap"] = float(abs(linear_rmses[0] - linear_rmses[1]))
            result["alpha"] = alpha_from_dual_start(linear_rmses, config)
        else:
            result["dual_start_risk_rmse"] = np.nan
            result["dual_start_rmse_gap"] = np.nan
            result["alpha"] = 1.0
        rows.append(result)
    features = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)
    if features.empty or features["well_id"].duplicated().any():
        raise AssertionError("Calibration feature table is empty or duplicated")
    alpha_min = float(nested(config, "confidence_shrink_meta_validation.alpha_min"))
    alpha_max = float(nested(config, "confidence_shrink_meta_validation.alpha_max"))
    if not features["alpha"].between(alpha_min, alpha_max, inclusive="both").all():
        raise AssertionError("Confidence shrink alpha is outside the registered bounds")
    return features


# %% [markdown]
# ## 4. Frozen exp218 OOF assembly


# %%
def load_frozen_oof(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target_tvt",
        "pred_tvt",
    ]
    frame = pd.read_csv(path, usecols=columns)
    expected = {
        "variant": str(nested(config, "frozen_anchor_parity.variant")),
        "mode": str(nested(config, "frozen_anchor_parity.mode")),
        "model": str(nested(config, "frozen_anchor_parity.model")),
    }
    for column, value in expected.items():
        if set(frame[column].astype(str)) != {value}:
            raise AssertionError(f"Frozen OOF {column} does not match {value}")
    if len(frame) != int(nested(config, "frozen_anchor_parity.expected_rows")):
        raise AssertionError("Frozen OOF row count changed")
    frame = frame.rename(columns={"well": "well_id", "pred_tvt": "raw_pred_tvt"})
    suffix = frame["id"].astype(str).str.rpartition("_", expand=True)
    if not np.array_equal(suffix[0].to_numpy(str), frame["well_id"].astype(str).to_numpy()):
        raise AssertionError("Frozen OOF id/well mismatch")
    frame["row_index"] = pd.to_numeric(suffix[2], errors="raise").astype(np.int32)
    return frame.drop(columns=["variant", "mode", "model"])


def assemble_evaluation_frame(
    oof: pd.DataFrame,
    calibration: pd.DataFrame,
    parity: pd.DataFrame,
    hidden: pd.DataFrame,
) -> pd.DataFrame:
    parity_columns = [
        "well_id",
        "official_start_index",
        "official_tail_rows",
        "fold_v2_exp218_weighted",
    ]
    frame = oof.merge(
        calibration,
        on="well_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_calibration"),
    )
    frame = frame.merge(
        parity[parity_columns],
        on="well_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_parity"),
    )
    hidden_columns = [
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]
    frame = frame.merge(hidden[hidden_columns], on="well_id", how="left", validate="many_to_one")
    required = [
        "alpha",
        "official_anchor_tvt",
        "official_start_index_parity",
        "fold_v2_exp218_weighted",
        *hidden_columns[1:],
    ]
    if frame[required].isna().any().any():
        raise AssertionError("Evaluation joins left missing parity/calibration/hidden values")
    if not np.allclose(frame["last_known_tvt"], frame["official_anchor_tvt"], atol=0.002):
        raise AssertionError("Frozen OOF anchor differs from raw official anchor")
    frame["eval_step"] = (
        frame["row_index"] - frame["official_start_index_parity"].astype(np.int32) - 1
    )
    if int(frame["eval_step"].min()) != 0:
        raise AssertionError("Official OOF does not start at eval_step zero")
    frame["calibrated_pred_tvt"] = frame["last_known_tvt"] + frame["alpha"] * (
        frame["raw_pred_tvt"] - frame["last_known_tvt"]
    )
    frame["fold"] = frame["fold_v2_exp218_weighted"].astype(int)
    return frame


# %% [markdown]
# ## 5. Official-start metrics and adoption guards


# %%
def rmse(truth: pd.Series, prediction: pd.Series) -> float:
    error = prediction.to_numpy(float) - truth.to_numpy(float)
    return float(np.sqrt(np.mean(np.square(error))))


def metric_row(frame: pd.DataFrame, surface: str) -> dict[str, Any]:
    raw = rmse(frame["target_tvt"], frame["raw_pred_tvt"])
    calibrated = rmse(frame["target_tvt"], frame["calibrated_pred_tvt"])
    return {
        "surface": surface,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "raw_rmse": raw,
        "calibrated_rmse": calibrated,
        "delta_rmse": calibrated - raw,
    }


def build_metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [metric_row(frame, "overall")]
    buckets = [
        ("000_050", 0, 49),
        ("050_100", 50, 99),
        ("100_250", 100, 249),
        ("250_500", 250, 499),
        ("500_1000", 500, 999),
        ("1000_plus", 1000, None),
    ]
    for name, lower, upper in buckets:
        mask = frame["eval_step"].ge(lower)
        if upper is not None:
            mask &= frame["eval_step"].le(upper)
        rows.append(metric_row(frame.loc[mask], name))
    rows.append(
        metric_row(
            frame.loc[frame["verification_like_spatial_role"].eq("valid")],
            "hidden_like_spatial",
        )
    )
    rows.append(
        metric_row(
            frame.loc[frame["verification_like_typewell_purged_role"].eq("valid")],
            "hidden_like_typewell_purged",
        )
    )
    metrics = pd.DataFrame(rows)
    fold_metrics = pd.DataFrame(
        [metric_row(group, f"fold_{fold}") for fold, group in frame.groupby("fold", sort=True)]
    )
    by_well = pd.DataFrame(
        [metric_row(group, str(well)) for well, group in frame.groupby("well_id", sort=True)]
    ).rename(columns={"surface": "well_id"})
    return metrics, fold_metrics, by_well


def adoption_guards(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    lookup = metrics.set_index("surface")
    max_regression = float(
        nested(
            config,
            "confidence_shrink_meta_validation.adoption_guards.max_worst_well_regression",
            2.0,
        )
    )
    min_folds = int(
        nested(
            config,
            "confidence_shrink_meta_validation.adoption_guards.min_improved_folds",
            3,
        )
    )
    improved_folds = int(fold_metrics["delta_rmse"].lt(0.0).sum())
    values = {
        "overall_improved": bool(lookup.loc["overall", "delta_rmse"] < 0.0),
        "rows_1000_plus_non_worse": bool(lookup.loc["1000_plus", "delta_rmse"] <= 0.0),
        "hidden_like_spatial_non_worse": bool(
            lookup.loc["hidden_like_spatial", "delta_rmse"] <= 0.0
        ),
        "hidden_like_typewell_purged_non_worse": bool(
            lookup.loc["hidden_like_typewell_purged", "delta_rmse"] <= 0.0
        ),
        "worst_well_regression_within_limit": bool(by_well["delta_rmse"].max() <= max_regression),
        "minimum_improved_folds": bool(improved_folds >= min_folds),
        "improved_folds": improved_folds,
        "required_improved_folds": min_folds,
        "worst_well_regression": float(by_well["delta_rmse"].max()),
        "max_worst_well_regression": max_regression,
    }
    values["adoption_supported"] = all(
        value for value in values.values() if isinstance(value, bool)
    )
    return values


# %% [markdown]
# ## 6. Generated files and reproducibility summary


# %%
def run_meta_validation() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    section = "confidence_shrink_meta_validation"
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "variant": nested(config, f"{section}.variant"),
                "active_variants": nested(config, f"{section}.active_variants"),
                "lightgbm_configs": nested(config, f"{section}.lightgbm_configs"),
                "folds_trained": nested(config, f"{section}.folds_trained"),
                "boosters": nested(config, f"{section}.boosters"),
                "parent_control_retrained": nested(config, f"{section}.parent_control_retrained"),
            },
            indent=2,
        )
    )
    oof_filename = "exp218_gr_wavelet_rotation_confidence_features_on_exp148_predictions.csv.gz"
    parity_filename = f"{EXPERIMENT_NAME}_frozen_anchor_parity_fold_manifest.csv"
    hidden_filename = "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
    oof_path = resolve_artifact(oof_filename, nested(config, "data.exp218_oof_predictions_local"))
    parity_path = resolve_artifact(parity_filename)
    hidden_path = resolve_artifact(hidden_filename)
    input_sha = {
        "exp218_oof_decompressed": assert_input_sha(
            oof_path,
            str(nested(config, "frozen_anchor_parity.expected_oof_decompressed_sha256")),
            decompressed_gzip=True,
        ),
        "v2_parity_fold_manifest": assert_input_sha(
            parity_path,
            str(nested(config, f"{section}.expected_parity_fold_manifest_sha256")),
        ),
        "exp115_hidden_assignment": assert_input_sha(
            hidden_path,
            str(nested(config, f"{section}.expected_hidden_assignment_sha256")),
        ),
    }
    calibration = build_calibration_features(resolve_train_dir(config), config)
    parity = pd.read_csv(parity_path, dtype={"well_id": str})
    hidden = pd.read_csv(hidden_path, dtype={"well_id": str})
    oof = load_frozen_oof(oof_path, config)
    frame = assemble_evaluation_frame(oof, calibration, parity, hidden)
    metrics, fold_metrics, by_well = build_metric_tables(frame)
    guards = adoption_guards(metrics, fold_metrics, by_well, config)
    eligible = calibration["eligible_dual_start"].astype(bool)
    used = calibration["alpha"].lt(1.0 - 1e-12)
    calibration_summary = {
        "wells": int(len(calibration)),
        "eligible_dual_start_wells": int(eligible.sum()),
        "used_wells": int(used.sum()),
        "use_rate_all_wells": float(used.mean()),
        "use_rate_eligible_wells": float(used[eligible].mean()),
        "alpha_min": float(calibration["alpha"].min()),
        "alpha_mean": float(calibration["alpha"].mean()),
        "alpha_max": float(calibration["alpha"].max()),
        "start_rmse_gap_median": float(calibration.loc[eligible, "dual_start_rmse_gap"].median()),
        "start_rmse_gap_p95": float(
            calibration.loc[eligible, "dual_start_rmse_gap"].quantile(0.95)
        ),
    }
    out = output_dir()
    files = {
        "calibration_features": out / f"{OUTPUT_PREFIX}_calibration_features.csv",
        "metrics": out / f"{OUTPUT_PREFIX}_metrics.csv",
        "fold_metrics": out / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "by_well": out / f"{OUTPUT_PREFIX}_by_well.csv",
        "oof": out / f"{OUTPUT_PREFIX}_oof.csv.gz",
    }
    calibration.to_csv(files["calibration_features"], index=False)
    metrics.to_csv(files["metrics"], index=False)
    fold_metrics.to_csv(files["fold_metrics"], index=False)
    by_well.to_csv(files["by_well"], index=False)
    oof_output = frame[
        [
            "id",
            "well_id",
            "fold",
            "eval_step",
            "target_tvt",
            "last_known_tvt",
            "raw_pred_tvt",
            "alpha",
            "calibrated_pred_tvt",
        ]
    ].copy()
    oof_output.to_csv(files["oof"], index=False, compression={"method": "gzip", "mtime": 0})
    content_sha = {
        "calibration_features": canonical_csv_sha256(calibration),
        "metrics": canonical_csv_sha256(metrics),
        "fold_metrics": canonical_csv_sha256(fold_metrics),
        "by_well": canonical_csv_sha256(by_well),
        "oof_decompressed": canonical_csv_sha256(oof_output),
    }
    overall = metrics.set_index("surface").loc["overall"]
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "confidence_shrink_meta_validation_complete",
        "route": nested(config, "experiment.route"),
        "variant": nested(config, f"{section}.variant"),
        "runtime": "kaggle_cpu" if is_kaggle_runtime() else "local_debug",
        "elapsed_seconds": time.perf_counter() - started,
        "raw_exp218_oof_rmse": float(overall["raw_rmse"]),
        "calibrated_oof_rmse": float(overall["calibrated_rmse"]),
        "delta_rmse": float(overall["delta_rmse"]),
        "calibration": calibration_summary,
        "guards": guards,
        "input_sha256": input_sha,
        "content_sha256": content_sha,
        "execution": {
            "active_variants": int(nested(config, f"{section}.active_variants")),
            "lightgbm_configs": 0,
            "folds_trained": 0,
            "boosters": 0,
            "parent_control_retrained": False,
            "model_training_performed": False,
            "test_prediction_performed": False,
            "submission_created": False,
        },
        "generated_files": {name: str(path) for name, path in files.items()},
    }
    summary_path = out / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("Calibration summary:")
    print(json.dumps(calibration_summary, indent=2, sort_keys=True))
    print("Official-start metrics:")
    print(metrics.to_string(index=False))
    print("Fold metrics:")
    print(fold_metrics.to_string(index=False))
    print("Worst regressions:")
    print(by_well.nlargest(10, "delta_rmse").to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


# %% [markdown]
# The first execution must be on Kaggle CPU. Local execution remains blocked unless an
# approved smoke debug explicitly sets `EXPERIMENT_ALLOW_LOCAL=1`.

# %%
if not is_kaggle_runtime() and os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
    raise RuntimeError(
        "This notebook is Kaggle-first. Set EXPERIMENT_ALLOW_LOCAL=1 only for an approved debug."
    )

META_VALIDATION_SUMMARY = run_meta_validation()
