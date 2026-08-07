from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

EXP063_LOCAL_ARTIFACTS = (
    Path("experiments")
    / "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit"
    / "artifacts"
)
EXP029_FEATURE_PATH = (
    Path("experiments")
    / "exp029_public_sel15_pf_oof_feature_generation"
    / "features"
    / "public_sel15_pf_oof_features.csv.gz"
)
OOF_PREDICTIONS = "ravaghi_vs_pixiux_public_replay_oof_predictions.csv.gz"
INFERENCE_PREDICTIONS = "ravaghi_vs_pixiux_public_replay_inference_test_predictions.csv.gz"
TRACKER_TRAIN_FEATURES = "ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz"
TRACKER_TEST_FEATURES = "ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz"
REFERENCE_SUBMISSION = "submission.csv"
DEFAULT_VARIANT = "pixiux_likpf_public_replay"
DEFAULT_MODEL = "lgb_mean"
FULL_MODEL_DIRNAME = "exp068_exp039_cv_full_lgb_models"
FULL_MODEL_PREDICTIONS = "exp068_exp039_cv_full_model_inference_predictions.csv.gz"
FULL_MODEL_METRICS = "exp068_exp039_cv_full_model_inference_metrics.csv"
FULL_MODEL_SUMMARY = "exp068_exp039_cv_full_model_inference_summary.json"
FULL_MODEL_FEATURE_SCHEMA = "exp068_exp039_cv_full_model_feature_schema.csv"
FULL_MODEL_IMPORTANCE = "exp068_exp039_cv_full_model_feature_importance.csv"

EXP039_META_COLUMNS = [
    "id",
    "well_id",
    "fold",
    "cutoff_row",
    "row_idx",
    "eval_step",
    "eval_fraction",
    "distance_bucket",
    "target_tvt",
    "last_anchor_tvt",
    "pf_pred",
    "beam_pred",
]
TRACKER_META_COLUMNS = {"id", "well", "target"}
EXP063_LGB_FEATURE_EXCLUDE = {"target"}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def stable_fold(value: str, n_folds: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(n_folds)


def exp063_lgb_configs(*, use_gpu: bool = False, fast: bool = False) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "n_jobs": -1,
        "max_bin": 255,
        "deterministic": True,
        "force_col_wise": True,
    }
    if use_gpu:
        base.update(device_type="gpu", gpu_use_dp=False)
    n_estimators = 600 if fast else 5000
    return [
        {
            **base,
            "num_leaves": 255,
            "min_child_samples": 15,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "learning_rate": 0.03,
            "n_estimators": n_estimators,
            "seed": 123,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 0,
        },
        {
            **base,
            "num_leaves": 64,
            "min_child_samples": 40,
            "subsample": 0.474,
            "subsample_freq": 1,
            "colsample_bytree": 0.393,
            "reg_lambda": 95.75,
            "reg_alpha": 10.79,
            "min_child_weight": 0.24,
            "learning_rate": 0.0093,
            "n_estimators": min(2 * n_estimators, 10000),
            "random_state": 29,
        },
    ]


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            EXP063_LOCAL_ARTIFACTS / filename,
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates[:60])
    raise FileNotFoundError(f"Artifact not found: {filename}. Checked:\n{checked}")


def find_artifact_dir(dirname: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path.cwd() / dirname,
            Path.cwd() / "artifacts" / dirname,
            Path.cwd().parent / "artifacts" / dirname,
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{dirname}"))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_dir() and (candidate / "manifest.json").exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates[:60])
    raise FileNotFoundError(f"Artifact directory not found: {dirname}. Checked:\n{checked}")


def find_path(path: str | Path, *, filename: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and (Path.cwd() / candidate).exists():
        return Path.cwd() / candidate
    if filename is not None:
        return find_artifact(filename, candidate)
    raise FileNotFoundError(f"Path not found: {path}")


def load_exp039_cv_surface(path: str | Path | None = None) -> pd.DataFrame:
    source = find_path(
        path or EXP029_FEATURE_PATH,
        filename=EXP029_FEATURE_PATH.name,
    )
    surface = pd.read_csv(source, usecols=lambda col: col in EXP039_META_COLUMNS)
    missing = sorted(set(EXP039_META_COLUMNS) - set(surface.columns))
    if missing:
        raise ValueError(f"{source} is missing exp039 CV columns: {missing}")
    surface["id"] = surface["id"].astype(str)
    surface["well_id"] = surface["well_id"].astype(str)
    return surface


def load_exp063_tracker_features(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], Path]:
    source = find_artifact(TRACKER_TRAIN_FEATURES, path)
    tracker = pd.read_csv(source)
    missing = sorted({"id", "well", "last_known_tvt"} - set(tracker.columns))
    if missing:
        raise ValueError(f"{source} is missing exp063 tracker columns: {missing}")
    tracker["id"] = tracker["id"].astype(str)
    tracker["well"] = tracker["id"].str.rsplit("_", n=1).str[0]
    feature_columns = [
        col
        for col in tracker.columns
        if col not in TRACKER_META_COLUMNS | EXP063_LGB_FEATURE_EXCLUDE
    ]
    for col in feature_columns:
        tracker[col] = pd.to_numeric(tracker[col], errors="coerce").astype(np.float32)
    return tracker, feature_columns, source


def load_exp063_tracker_test_features(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    source = find_artifact(TRACKER_TEST_FEATURES, path)
    tracker = pd.read_csv(source)
    missing = sorted({"id", "well", "last_known_tvt"} - set(tracker.columns))
    if missing:
        raise ValueError(f"{source} is missing exp063 tracker test columns: {missing}")
    tracker["id"] = tracker["id"].astype(str)
    tracker["well"] = tracker["id"].str.rsplit("_", n=1).str[0]
    for col in tracker.columns:
        if col not in {"id", "well"}:
            tracker[col] = pd.to_numeric(tracker[col], errors="coerce").astype(np.float32)
    return tracker, source


def build_exp039_cv_exp063_frame(
    *,
    exp039_feature_path: str | Path | None = None,
    exp063_tracker_features_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    surface = load_exp039_cv_surface(exp039_feature_path)
    tracker, feature_columns, tracker_source = load_exp063_tracker_features(
        exp063_tracker_features_path
    )
    frame = surface.merge(tracker, on="id", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("No joined rows between exp039 CV surface and exp063 tracker features")
    mismatch = frame["well_id"].astype(str).ne(frame["well"].astype(str))
    if bool(mismatch.any()):
        sample = frame.loc[mismatch, ["id", "well_id", "well"]].head(5).to_dict("records")
        raise ValueError(f"well mismatch after join: {sample}")
    for col in ["target_tvt", "last_anchor_tvt", "last_known_tvt", "fold", "eval_step"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.sort_values(["fold", "well_id", "eval_step"], kind="mergesort").reset_index(
        drop=True
    )
    stats = {
        "exp039_rows": int(len(surface)),
        "exp063_tracker_rows": int(len(tracker)),
        "joined_rows": int(len(frame)),
        "dropped_exp039_rows": int(len(surface) - len(frame)),
        "joined_wells": int(frame["well_id"].nunique()),
        "tracker_source": str(tracker_source),
        "feature_count": int(len(feature_columns)),
    }
    return frame, feature_columns, stats


def exp039_cv_split_codes(
    frame: pd.DataFrame,
    *,
    well_hash_folds: int = 5,
) -> dict[str, tuple[np.ndarray, list[Any]]]:
    original_folds = sorted(int(value) for value in frame["fold"].unique())
    original_fold_map = {fold: idx for idx, fold in enumerate(original_folds)}
    original_codes = frame["fold"].map(original_fold_map).to_numpy(dtype=np.int16)
    well_hash_codes = frame["well_id"].map(
        lambda value: stable_fold(str(value), well_hash_folds)
    ).to_numpy(dtype=np.int16)
    return {
        "leave_one_original_fold_out": (original_codes, original_folds),
        "well_hash_holdout": (well_hash_codes, list(range(well_hash_folds))),
    }


def _fit_exp063_lgb_cv(
    *,
    audit: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_codes: np.ndarray,
    split_labels: list[Any],
    use_gpu: bool = False,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_train_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y_true = frame["target_tvt"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    y_residual = y_true - base
    configs = exp063_lgb_configs(use_gpu=use_gpu, fast=fast)
    unique_splits = sorted(int(value) for value in np.unique(split_codes))
    rng = np.random.default_rng(42)
    oof_by_model: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_index, params in enumerate(configs):
        oof = np.full(len(frame), np.nan, dtype=np.float32)
        for split in unique_splits:
            valid_idx = np.flatnonzero(split_codes == split)
            train_idx = np.flatnonzero(split_codes != split)
            if max_train_rows is not None and len(train_idx) > max_train_rows:
                train_idx = np.sort(
                    rng.choice(train_idx, size=int(max_train_rows), replace=False)
                )
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y_residual[train_idx],
                eval_set=[(x_matrix[valid_idx], y_residual[valid_idx])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
            oof[valid_idx] = pred
            pred_tvt = base[valid_idx] + pred
            metric_rows.append(
                {
                    "audit": audit,
                    "model": f"lgb{model_index}",
                    "split": split_labels[split] if split < len(split_labels) else split,
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(y_true[valid_idx], pred_tvt),
                    "rmse_residual": rmse(y_residual[valid_idx], pred),
                }
            )
            print(
                json.dumps(
                    {
                        "audit": audit,
                        "model": f"lgb{model_index}",
                        "split": int(split),
                        "rows": int(len(valid_idx)),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not np.isfinite(oof).all():
            raise ValueError(f"{audit}/lgb{model_index}: non-finite OOF predictions")
        oof_by_model.append(oof)
        metric_rows.append(
            {
                "audit": audit,
                "model": f"lgb{model_index}",
                "split": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "best_iteration": None,
                "rmse_tvt": rmse(y_true, base + oof),
                "rmse_residual": rmse(y_residual, oof),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well_id"].to_numpy(),
                    "audit": audit,
                    "model": f"lgb{model_index}",
                    "target_tvt": y_true,
                    "last_known_tvt": base,
                    "pred_delta": oof,
                    "pred_tvt": base + oof,
                }
            )
        )

    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    metric_rows.append(
        {
            "audit": audit,
            "model": "lgb_mean",
            "split": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "best_iteration": None,
            "rmse_tvt": rmse(y_true, base + ensemble),
            "rmse_residual": rmse(y_residual, ensemble),
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well_id"].to_numpy(),
                "audit": audit,
                "model": "lgb_mean",
                "target_tvt": y_true,
                "last_known_tvt": base,
                "pred_delta": ensemble,
                "pred_tvt": base + ensemble,
            }
        )
    )
    by_well = (
        prediction_frames[-1]
        .assign(error_tvt=lambda df: df["pred_tvt"] - df["target_tvt"])
        .groupby(["audit", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
    )
    return pd.DataFrame(metric_rows), by_well, pd.concat(prediction_frames, ignore_index=True)


def _best_iterations_from_cv(
    metrics: pd.DataFrame,
    *,
    primary_audit: str,
    default_configs: list[dict[str, Any]],
) -> dict[int, int]:
    best_iterations: dict[int, int] = {}
    cv_rows = metrics[
        metrics["audit"].astype(str).eq(primary_audit)
        & metrics["model"].astype(str).str.match(r"^lgb\d+$")
        & metrics["split"].astype(str).ne("pooled")
    ].copy()
    for model_index, params in enumerate(default_configs):
        model_name = f"lgb{model_index}"
        values = pd.to_numeric(
            cv_rows.loc[cv_rows["model"].eq(model_name), "best_iteration"],
            errors="coerce",
        ).dropna()
        if values.empty:
            best_iter = int(params.get("n_estimators", 0))
        else:
            best_iter = int(max(1, round(float(values.median()))))
        best_iterations[model_index] = best_iter
    return best_iterations


def _fit_full_lgb_models(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    metrics: pd.DataFrame,
    primary_audit: str,
    use_gpu: bool,
    fast: bool,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor

    model_dir = output_dir / FULL_MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    configs = exp063_lgb_configs(use_gpu=use_gpu, fast=fast)
    best_iterations = _best_iterations_from_cv(
        metrics,
        primary_audit=primary_audit,
        default_configs=configs,
    )
    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y_true = frame["target_tvt"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    y_residual = y_true - base
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    train_prediction_frames: list[pd.DataFrame] = []

    for model_index, params in enumerate(configs):
        fit_params = dict(params)
        fit_params["n_estimators"] = int(best_iterations[model_index])
        model = LGBMRegressor(**fit_params)
        model.fit(
            x_matrix,
            y_residual,
        )
        model_name = f"lgb{model_index}"
        model_file = f"exp068_exp039_cv_full__{model_name}.txt"
        model.booster_.save_model(str(model_dir / model_file))
        pred_delta = model.predict(x_matrix).astype(np.float32)
        train_prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well_id"].to_numpy(),
                    "model": model_name,
                    "target_tvt": y_true,
                    "last_known_tvt": base,
                    "pred_delta": pred_delta,
                    "pred_tvt": base + pred_delta,
                }
            )
        )
        model_rows.append(
            {
                "model": model_name,
                "model_index": int(model_index),
                "best_iteration_source": primary_audit,
                "best_iteration": int(fit_params["n_estimators"]),
                "file": model_file,
                "sha256": sha256_file(model_dir / model_file),
                "train_rmse_tvt": rmse(y_true, base + pred_delta),
                "train_rmse_residual": rmse(y_residual, pred_delta),
            }
        )
        for col, imp in zip(feature_columns, model.feature_importances_, strict=False):
            importance_rows.append(
                {
                    "model": model_name,
                    "feature": col,
                    "importance": float(imp),
                }
            )
        print(
            json.dumps(
                {
                    "mode": "full_train_model",
                    "model": model_name,
                    "rows": int(len(frame)),
                    "features": int(len(feature_columns)),
                    "n_estimators": int(fit_params["n_estimators"]),
                    "train_rmse_tvt": model_rows[-1]["train_rmse_tvt"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ensemble_delta = np.mean(
        np.vstack([df["pred_delta"].to_numpy(np.float32) for df in train_prediction_frames]),
        axis=0,
    ).astype(np.float32)
    train_prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well_id"].to_numpy(),
                "model": "lgb_mean",
                "target_tvt": y_true,
                "last_known_tvt": base,
                "pred_delta": ensemble_delta,
                "pred_tvt": base + ensemble_delta,
            }
        )
    )
    pd.concat(train_prediction_frames, ignore_index=True).to_csv(
        output_dir / "exp068_exp039_cv_full_model_train_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame({"feature": feature_columns}).to_csv(
        output_dir / FULL_MODEL_FEATURE_SCHEMA,
        index=False,
    )
    pd.DataFrame(importance_rows).to_csv(output_dir / FULL_MODEL_IMPORTANCE, index=False)
    manifest = {
        "experiment": "exp068_equivalent_pixiux_inference_port",
        "mode": "exp039_cv_full_train_lgb_models",
        "model_family": "exp063_public_lightgbm_configs",
        "variant": DEFAULT_VARIANT,
        "target": "target_tvt_minus_last_known_tvt",
        "base_prediction": "last_known_tvt",
        "train_rows": int(len(frame)),
        "train_wells": int(frame["well_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "features": feature_columns,
        "primary_audit_for_best_iteration": primary_audit,
        "models": model_rows,
        "ensemble_train_rmse_tvt": rmse(y_true, base + ensemble_delta),
        "ensemble_train_rmse_residual": rmse(y_residual, ensemble_delta),
        "artifacts": {
            "feature_schema": FULL_MODEL_FEATURE_SCHEMA,
            "feature_importance": FULL_MODEL_IMPORTANCE,
            "train_predictions": "exp068_exp039_cv_full_model_train_predictions.csv.gz",
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "model_dir": FULL_MODEL_DIRNAME,
        "manifest": f"{FULL_MODEL_DIRNAME}/manifest.json",
        "feature_schema": FULL_MODEL_FEATURE_SCHEMA,
        "feature_importance": FULL_MODEL_IMPORTANCE,
        "train_predictions": "exp068_exp039_cv_full_model_train_predictions.csv.gz",
        "models": model_rows,
        "ensemble_train_rmse_tvt": manifest["ensemble_train_rmse_tvt"],
        "ensemble_train_rmse_residual": manifest["ensemble_train_rmse_residual"],
    }


def run_exp063_model_on_exp039_cv(
    *,
    output_dir: str | Path,
    exp039_feature_path: str | Path | None = None,
    exp063_tracker_features_path: str | Path | None = None,
    audits: tuple[str, ...] = ("leave_one_original_fold_out", "well_hash_holdout"),
    well_hash_folds: int = 5,
    use_gpu: bool = False,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_train_rows: int | None = None,
    save_full_models: bool = True,
    primary_audit_for_full_model: str = "leave_one_original_fold_out",
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, feature_columns, join_stats = build_exp039_cv_exp063_frame(
        exp039_feature_path=exp039_feature_path,
        exp063_tracker_features_path=exp063_tracker_features_path,
    )
    split_map = exp039_cv_split_codes(frame, well_hash_folds=well_hash_folds)
    metric_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for audit in audits:
        if audit not in split_map:
            raise ValueError(f"unknown audit: {audit}")
        split_codes, split_labels = split_map[audit]
        metrics, by_well, predictions = _fit_exp063_lgb_cv(
            audit=audit,
            frame=frame,
            feature_columns=feature_columns,
            split_codes=split_codes,
            split_labels=split_labels,
            use_gpu=use_gpu,
            fast=fast,
            early_stopping_rounds=early_stopping_rounds,
            max_train_rows=max_train_rows,
        )
        metric_frames.append(metrics)
        well_frames.append(by_well)
        prediction_frames.append(predictions)

    metrics = pd.concat(metric_frames, ignore_index=True)
    by_well = pd.concat(well_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / "exp063_model_exp039_cv_metrics.csv", index=False)
    by_well.to_csv(output_dir / "exp063_model_exp039_cv_by_well.csv", index=False)
    predictions.to_csv(
        output_dir / "exp063_model_exp039_cv_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    summary_metrics = metrics[(metrics["model"] == "lgb_mean") & (metrics["split"] == "pooled")]
    full_model_artifacts = None
    if save_full_models:
        if primary_audit_for_full_model not in audits:
            raise ValueError(
                "primary_audit_for_full_model must be included in audits: "
                f"{primary_audit_for_full_model} not in {audits}"
            )
        full_model_artifacts = _fit_full_lgb_models(
            frame=frame,
            feature_columns=feature_columns,
            output_dir=output_dir,
            metrics=metrics,
            primary_audit=primary_audit_for_full_model,
            use_gpu=use_gpu,
            fast=fast,
        )
    summary = {
        "experiment": "exp068_equivalent_pixiux_inference_port",
        "mode": "exp063_model_retrained_on_exp039_cv",
        "cv_source": "exp039/exp038 leave-one-original-fold and well-hash audit surface",
        "model_source": "exp063 LightGBM configs with exp063 tracker/PF/Beam output features",
        "join_stats": join_stats,
        "audits": list(audits),
        "lgb_mean_pooled": summary_metrics.to_dict("records"),
        "full_model_artifacts": full_model_artifacts,
        "artifacts": {
            "metrics": "exp063_model_exp039_cv_metrics.csv",
            "by_well": "exp063_model_exp039_cv_by_well.csv",
            "predictions": "exp063_model_exp039_cv_predictions.csv.gz",
            "full_model_dir": FULL_MODEL_DIRNAME if full_model_artifacts else None,
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / "exp063_model_exp039_cv_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def selected_predictions(
    predictions: pd.DataFrame,
    *,
    variant: str = DEFAULT_VARIANT,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    frame = predictions.copy()
    if "variant" in frame.columns:
        frame = frame[frame["variant"].astype(str).eq(variant)]
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str).eq(model)]
    if frame.empty:
        raise ValueError(f"No predictions for variant={variant} model={model}")
    return frame.reset_index(drop=True)


def submission_diff(
    submission_path: Path,
    reference_submission_path: str | Path | None,
    output_dir: Path,
    *,
    target_column: str,
) -> dict[str, Any]:
    if reference_submission_path is None:
        return {"status": "skipped", "reason": "reference_submission_path is null"}
    reference_path = Path(reference_submission_path)
    if not reference_path.exists() and not reference_path.is_absolute():
        cwd_candidate = Path.cwd() / reference_path
        if cwd_candidate.exists():
            reference_path = cwd_candidate
    if not reference_path.exists():
        return {
            "status": "missing_reference",
            "reference_submission_path": str(reference_submission_path),
        }
    try:
        if reference_path.resolve() == submission_path.resolve():
            return {
                "status": "self_reference",
                "reference_submission_path": str(reference_path),
                "reason": "reference submission resolves to current submission",
            }
    except FileNotFoundError:
        pass
    current = pd.read_csv(submission_path)
    reference = pd.read_csv(reference_path)
    reference_target = (
        target_column if target_column in reference.columns else str(reference.columns[1])
    )
    merged = current[["id", target_column]].rename(columns={target_column: "tvt_exp068"}).merge(
        reference[["id", reference_target]].rename(columns={reference_target: "tvt_reference"}),
        on="id",
        how="outer",
        indicator=True,
    )
    merged["diff"] = merged["tvt_exp068"] - merged["tvt_reference"]
    merged.to_csv(output_dir / "exp063_branch_submission_diff.csv", index=False)
    valid = merged["diff"].dropna().astype(float)
    return {
        "status": "complete",
        "reference_submission_path": str(reference_path),
        "rows_current": int(len(current)),
        "rows_reference": int(len(reference)),
        "id_mismatch_rows": int((merged["_merge"] != "both").sum()),
        "diff_min": float(valid.min()) if len(valid) else None,
        "diff_max": float(valid.max()) if len(valid) else None,
        "diff_mean": float(valid.mean()) if len(valid) else None,
        "diff_abs_mean": float(valid.abs().mean()) if len(valid) else None,
        "diff_rmse": float(np.sqrt(np.mean(np.square(valid)))) if len(valid) else None,
        "diff_file": "exp063_branch_submission_diff.csv",
    }


def _load_exp063_public_replay_module() -> Any:
    candidates = [
        EXP063_LOCAL_ARTIFACTS.parent / "public_notebook_replay_audit.py",
        Path.cwd() / "public_notebook_replay_audit.py",
        Path.cwd().parent / "public_notebook_replay_audit.py",
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob("**/public_notebook_replay_audit.py"))
    for candidate in dict.fromkeys(candidates):
        if not candidate.exists():
            continue
        module_name = "exp068_exp063_public_notebook_replay_audit"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    checked = "\n".join(str(path) for path in candidates[:60])
    raise FileNotFoundError(f"public_notebook_replay_audit.py not found. Checked:\n{checked}")


def build_hidden_safe_exp063_test_features(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int = 8,
    fast: bool = False,
    use_gpu: str = "auto",
    max_wells: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    module = _load_exp063_public_replay_module()
    module.configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        fast=fast,
        use_gpu=use_gpu,
        n_train_wells=max_wells,
    )
    test_df, feature_meta = module.build_replay_test_frame()
    test_df = test_df.reset_index(drop=True)
    if "id" not in test_df.columns or "last_known_tvt" not in test_df.columns:
        raise ValueError("generated exp063 test features must include id and last_known_tvt")
    if "well" not in test_df.columns:
        test_df["well"] = test_df["id"].astype(str).str.rsplit("_", n=1).str[0]
    test_df["id"] = test_df["id"].astype(str)
    test_df["well"] = test_df["id"].str.rsplit("_", n=1).str[0]
    return test_df, {
        "source": "generated_with_exp063_public_notebook_replay_audit",
        "module_file": str(Path(module.__file__).resolve()),
        "feature_meta": feature_meta,
    }


def _predict_exp068_full_model(
    test_df: pd.DataFrame,
    model_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import lightgbm as lgb

    manifest = json.loads((model_dir / "manifest.json").read_text())
    features = [str(col) for col in manifest["features"]]
    missing = sorted(col for col in features if col not in test_df.columns)
    if missing:
        raise ValueError(f"test features are missing full-model columns: {missing[:30]}")
    for col in features:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(np.float32)
    if test_df[features].isna().any().any():
        nan_cols = test_df[features].columns[test_df[features].isna().any()].tolist()
        raise ValueError(f"test features contain NaN values for full model: {nan_cols[:30]}")
    x_test = test_df[features].to_numpy(np.float32)
    base = pd.to_numeric(test_df["last_known_tvt"], errors="raise").to_numpy(np.float32)
    model_deltas: list[np.ndarray] = []
    prediction_frames: list[pd.DataFrame] = []
    models = list(manifest.get("models", []))
    if not models:
        raise ValueError(f"No models listed in {model_dir / 'manifest.json'}")
    for item in models:
        booster = lgb.Booster(model_file=str(model_dir / str(item["file"])))
        delta = booster.predict(x_test).astype(np.float32)
        model_deltas.append(delta)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": test_df["id"].astype(str).to_numpy(),
                    "well": test_df["well"].astype(str).to_numpy(),
                    "model": str(item["model"]),
                    "last_known_tvt": base,
                    "pred_delta": delta,
                    "pred_tvt": base + delta,
                }
            )
        )
    if model_name == "lgb_mean":
        selected_delta = np.mean(np.vstack(model_deltas), axis=0).astype(np.float32)
    else:
        matching = [idx for idx, item in enumerate(models) if str(item["model"]) == model_name]
        if not matching:
            raise ValueError(f"No full model named {model_name}; available={models}")
        selected_delta = np.mean(np.vstack([model_deltas[idx] for idx in matching]), axis=0).astype(
            np.float32
        )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": test_df["id"].astype(str).to_numpy(),
                "well": test_df["well"].astype(str).to_numpy(),
                "model": model_name,
                "last_known_tvt": base,
                "pred_delta": selected_delta,
                "pred_tvt": base + selected_delta,
            }
        )
    )
    return pd.concat(prediction_frames, ignore_index=True), {
        "model_dir": str(model_dir),
        "manifest": manifest,
        "feature_count": int(len(features)),
        "model_count": int(len(models)),
        "selected_model": model_name,
    }


def run_exp068_full_model_inference(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    model_artifact_dir: str | Path | None = None,
    tracker_test_features_path: str | Path | None = None,
    reference_submission_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    feature_mode: str = "generate_exp063_replay",
    model: str = DEFAULT_MODEL,
    submission_target_column: str = "tvt",
    n_jobs: int = 8,
    fast: bool = False,
    use_gpu: str = "auto",
    max_wells: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    model_dir = find_artifact_dir(FULL_MODEL_DIRNAME, model_artifact_dir)
    if feature_mode == "artifact":
        test_df, feature_source = load_exp063_tracker_test_features(tracker_test_features_path)
        feature_meta = {
            "source": "exp063_tracker_features_test_artifact",
            "path": str(feature_source),
        }
    elif feature_mode == "generate_exp063_replay":
        if data_dir is None:
            raise ValueError("data_dir is required when feature_mode=generate_exp063_replay")
        test_df, feature_meta = build_hidden_safe_exp063_test_features(
            data_dir=data_dir,
            output_dir=output_dir,
            n_jobs=n_jobs,
            fast=fast,
            use_gpu=use_gpu,
            max_wells=max_wells,
        )
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    predictions, model_meta = _predict_exp068_full_model(test_df, model_dir, model_name=model)
    selected = predictions[predictions["model"].astype(str).eq(model)].copy()
    if selected.empty:
        raise ValueError(f"No predictions for exp068 full model={model}")
    sample = pd.read_csv(sample_submission_path)
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    sample_ids = sample["id"].astype(str)
    pred_ids = selected["id"].astype(str)
    missing_ids = sorted(set(sample_ids) - set(pred_ids))
    extra_ids = sorted(set(pred_ids) - set(sample_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "full-model inference id mismatch: "
            f"missing={len(missing_ids)} extra={len(extra_ids)} "
            f"missing_sample={missing_ids[:5]} extra_sample={extra_ids[:5]}"
        )
    pred_map = dict(zip(pred_ids, selected["pred_tvt"].astype(float), strict=False))
    sample[target_column] = sample_ids.map(pred_map).astype("float64")
    sample.to_csv(submission_path, index=False)
    predictions.to_csv(output_dir / FULL_MODEL_PREDICTIONS, index=False, compression="gzip")
    diagnostics = {
        "mode": "exp068_full_model_inference",
        "model": model,
        "feature_mode": feature_mode,
        "model_dir": str(model_dir),
        "test_feature_rows": int(len(test_df)),
        "submission_rows": int(len(sample)),
        "prediction_rows": int(len(selected)),
        "fallback_rows": 0,
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "sha256": sha256_file(submission_path),
    }
    pd.DataFrame([diagnostics]).to_csv(output_dir / FULL_MODEL_METRICS, index=False)
    diff = submission_diff(
        submission_path,
        reference_submission_path,
        output_dir,
        target_column=target_column,
    )
    summary = {
        "experiment": "exp068_equivalent_pixiux_inference_port",
        "mode": "exp068_full_model_from_exp039_cv_retrain",
        "submission": diagnostics,
        "feature_meta": feature_meta,
        "model_meta": {
            "model_dir": model_meta["model_dir"],
            "feature_count": model_meta["feature_count"],
            "model_count": model_meta["model_count"],
            "selected_model": model_meta["selected_model"],
        },
        "reference_diff": diff,
        "artifacts": {
            "submission": submission_path.name,
            "predictions": FULL_MODEL_PREDICTIONS,
            "metrics": FULL_MODEL_METRICS,
            "summary": FULL_MODEL_SUMMARY,
            "reference_diff": diff.get("diff_file"),
        },
        "excluded": [
            "static exp063 inference prediction artifact",
            "fallback filling for missing hidden ids",
            "modification of exp063 implementation files",
        ],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / FULL_MODEL_SUMMARY).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def train_overlap_summary(ids: pd.Series, data_dir: str | Path | None) -> pd.DataFrame:
    data_path = Path(data_dir) if data_dir is not None else None
    rows = []
    for raw_id in ids.astype(str):
        try:
            well, row_idx = raw_id.rsplit("_", 1)
            row_idx_int = int(row_idx)
        except ValueError:
            well = raw_id
            row_idx_int = -1
        train_overlap = False
        if data_path is not None:
            train_overlap = (data_path / "train" / f"{well}__horizontal_well.csv").exists()
        rows.append(
            {
                "id": raw_id,
                "well": well,
                "row_idx": row_idx_int,
                "train_overlap_visible_like": bool(train_overlap),
            }
        )
    return pd.DataFrame(rows)


def run_exp063_branch_inference_audit(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    inference_predictions_path: str | Path | None = None,
    reference_submission_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    variant: str = DEFAULT_VARIANT,
    model: str = DEFAULT_MODEL,
    submission_target_column: str = "tvt",
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    source_path = find_artifact(INFERENCE_PREDICTIONS, inference_predictions_path)
    predictions = selected_predictions(pd.read_csv(source_path), variant=variant, model=model)
    required = {"id", "well", "last_known_tvt", "pred_delta", "pred_tvt"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{source_path} is missing required columns: {missing}")

    sample = pd.read_csv(sample_submission_path)
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = dict(
        zip(predictions["id"].astype(str), predictions["pred_tvt"].astype(float), strict=False)
    )
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()
    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    branch = train_overlap_summary(sample["id"], data_dir)
    branch["branch_source"] = "exp063_pixiux_lgb_mean_hidden_equivalent"
    branch["tvt"] = sample[target_column].astype(float)
    branch.to_csv(output_dir / "exp063_branch_public_sample_summary.csv", index=False)
    by_source = (
        branch.groupby(["branch_source", "train_overlap_visible_like"], as_index=False)
        .agg(rows=("id", "size"), wells=("well", "nunique"))
        .sort_values(["branch_source", "train_overlap_visible_like"])
    )
    by_source.to_csv(output_dir / "exp063_branch_public_sample_by_source.csv", index=False)
    predictions.to_csv(
        output_dir / "exp063_branch_inference_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    diagnostics = {
        "variant": variant,
        "model": model,
        "source_inference_predictions": str(source_path),
        "test_prediction_rows": int(len(predictions)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "train_overlap_visible_like_rows": int(branch["train_overlap_visible_like"].sum()),
        "train_overlap_visible_like_wells": int(
            branch.loc[branch["train_overlap_visible_like"], "well"].nunique()
        ),
    }
    pd.DataFrame([diagnostics]).to_csv(
        output_dir / "exp063_branch_inference_metrics.csv",
        index=False,
    )
    diff = submission_diff(
        submission_path,
        reference_submission_path,
        output_dir,
        target_column=target_column,
    )
    summary = {
        "experiment": "exp068_equivalent_pixiux_inference_port",
        "mode": "exp039_style_branch_inference_on_exp063",
        "branch_candidate": {
            "variant": variant,
            "model": model,
            "source": "exp063 inference prediction artifact",
        },
        "submission": {
            "path": str(submission_path),
            "rows": int(len(sample)),
            "predicted_rows": int((~missing_mask).sum()),
            "fallback_rows": int(missing_mask.sum()),
            "target_column": target_column,
            "prediction_mean": float(sample[target_column].mean()),
            "prediction_std": float(sample[target_column].std()),
            "prediction_min": float(sample[target_column].min()),
            "prediction_max": float(sample[target_column].max()),
            "sha256": sha256_file(submission_path),
        },
        "public_sample_branch_summary": diagnostics,
        "reference_diff": diff,
        "artifacts": {
            "submission": submission_path.name,
            "predictions": "exp063_branch_inference_predictions.csv.gz",
            "metrics": "exp063_branch_inference_metrics.csv",
            "public_sample_summary": "exp063_branch_public_sample_summary.csv",
            "public_sample_by_source": "exp063_branch_public_sample_by_source.csv",
            "summary": "exp063_branch_inference_summary.json",
            "reference_diff": diff.get("diff_file"),
        },
        "excluded": [
            "new model retraining",
            "PF/Beam regeneration",
            "static visible override",
            "train-label overwrite for public sample rows",
            "modification of exp063 implementation",
        ],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / "exp063_branch_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
