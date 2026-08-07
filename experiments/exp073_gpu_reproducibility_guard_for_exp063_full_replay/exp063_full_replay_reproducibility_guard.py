from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

EXP072_ARTIFACTS = (
    Path("experiments")
    / "exp072_exp063_full_replay_feature_cache"
    / "artifacts"
)
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
FULL_REPLAY_CACHE_SUMMARY = "exp063_full_replay_feature_cache_summary.json"
FULL_REPLAY_TEST_FEATURES = "exp063_full_replay_repro_guard_test_features.csv.gz"
TRACKER_TEST_FEATURES = FULL_REPLAY_TEST_FEATURES
OUTPUT_PREFIX = "exp063_full_replay_repro_guard"
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196
VARIANT = "pixiux_likpf_public_replay"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(np.asarray(y_true, float), np.asarray(y_pred, float))))


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prediction_sha256(ids: pd.Series, values: np.ndarray, *, label: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(label.encode("utf-8"))
    for raw_id in ids.astype(str).to_numpy():
        hasher.update(raw_id.encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(np.asarray(values, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_artifacts: Path = EXP072_ARTIFACTS,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            local_artifacts / filename,
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
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found: {filename}. Checked:\n{checked}")


def find_model_manifest(explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        path = Path(explicit_path)
        candidates.append(path if path.name == "manifest.json" else path / "manifest.json")
    candidates.extend(
        [
            Path.cwd() / "artifacts" / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
            Path.cwd() / f"{OUTPUT_PREFIX}_lgb_models" / "manifest.json",
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.glob(f"**/{OUTPUT_PREFIX}_lgb_models/manifest.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"model manifest not found. Checked:\n{checked}")


def exp063_lgb_config_family(*, fast: bool = False) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "verbose": -1,
        "max_bin": 255,
    }
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


def apply_mode_overrides(
    configs: list[dict[str, Any]],
    mode_config: dict[str, Any],
) -> list[dict[str, Any]]:
    use_gpu = bool(mode_config.get("use_gpu", False))
    common = dict(mode_config.get("common_overrides") or {})
    updated: list[dict[str, Any]] = []
    for params in configs:
        merged = dict(params)
        if use_gpu:
            merged["device_type"] = "gpu"
        else:
            merged.pop("device_type", None)
            merged.pop("gpu_use_dp", None)
        merged.update(common)
        if use_gpu and "gpu_use_dp" not in merged:
            merged["gpu_use_dp"] = False
        updated.append(merged)
    return updated


def load_exp072_full_replay_cache_frame(
    cache_path: str | Path | None = None,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TRAIN_FEATURES, cache_path)
    frame = pd.read_csv(source, nrows=max_rows, dtype={"id": str, "well": str})
    required = {"id", "well", "target", "last_known_tvt"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    feature_columns = [col for col in frame.columns if col not in META_COLUMNS]
    if len(feature_columns) != EXPECTED_FULL_REPLAY_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_REPLAY_FEATURE_COUNT} full replay features, "
            f"got {len(feature_columns)} from {source}"
        )
    for col in ["target", *feature_columns]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    if not np.isfinite(frame[["target", *feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("exp072 full replay cache contains non-finite numeric values")
    schema_path: Path | None = None
    summary_path: Path | None = None
    try:
        schema_path = find_artifact(FULL_REPLAY_FEATURE_SCHEMA)
    except FileNotFoundError:
        schema_path = None
    try:
        summary_path = find_artifact(FULL_REPLAY_CACHE_SUMMARY)
    except FileNotFoundError:
        summary_path = None
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_experiment": "exp072_exp063_full_replay_feature_cache",
        "source_kind": "exp063_full_public_replay_train_feature_cache",
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_file(schema_path) if schema_path else None,
        "summary": str(summary_path) if summary_path else None,
        "summary_sha256": sha256_file(summary_path) if summary_path else None,
    }
    return frame, feature_columns, metadata


def load_exp063_tracker_test_frame(
    tracker_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(FULL_REPLAY_TEST_FEATURES, tracker_path)
    frame = pd.read_csv(source, dtype={"id": str, "well": str})
    required = {"id", "well", "last_known_tvt"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for col in frame.columns:
        if col not in {"id", "well"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    metadata = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": int(len(frame.columns)),
    }
    return frame, metadata


def generate_exp063_tracker_test_frame(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from public_notebook_replay_audit import (
        build_replay_test_frame,
        configure_public_runtime,
        feature_columns_for_variant,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=n_jobs,
        pf_seeds=pf_seeds,
        pf_particles=pf_particles,
        fast=fast,
        use_gpu=use_gpu,
        n_train_wells=None,
    )
    raw_test_frame, generation_meta = build_replay_test_frame()
    feature_columns = feature_columns_for_variant(raw_test_frame, VARIANT)
    if len(feature_columns) != EXPECTED_FULL_REPLAY_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FULL_REPLAY_FEATURE_COUNT} generated test features, "
            f"got {len(feature_columns)}"
        )
    output_columns = ["id", "well", *feature_columns]
    missing = [col for col in output_columns if col not in raw_test_frame.columns]
    if missing:
        raise ValueError(f"Generated test frame is missing columns: {missing[:20]}")
    full_test_frame = raw_test_frame[output_columns].copy()
    full_test_frame["id"] = full_test_frame["id"].astype(str)
    full_test_frame["well"] = full_test_frame["well"].astype(str)
    for col in feature_columns:
        full_test_frame[col] = pd.to_numeric(
            full_test_frame[col],
            errors="coerce",
        ).astype(np.float32)
    if not np.isfinite(full_test_frame[feature_columns].to_numpy(np.float32)).all():
        raise ValueError("Generated full replay test features contain non-finite values")
    test_path = output_dir / FULL_REPLAY_TEST_FEATURES
    full_test_frame.to_csv(test_path, index=False, compression="gzip")
    test_meta = {
        "path": test_path.name,
        "rows": int(len(full_test_frame)),
        "columns": int(len(full_test_frame.columns)),
        "features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "sha256": sha256_file(test_path),
    }
    tracker_frame, read_meta = load_exp063_tracker_test_frame(test_path)
    read_meta.update(
        {
            "source_kind": "raw_test_regenerated_exp063_public_replay",
            "feature_generation": generation_meta,
            "full_replay_test_frame": test_meta,
        }
    )
    return tracker_frame, read_meta


def _fit_one_mode(
    *,
    mode_name: str,
    mode_config: dict[str, Any],
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
    n_splits: int,
    fast: bool,
    early_stopping_rounds: int,
    max_train_rows: int | None,
    save_models: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    x_matrix = frame[feature_columns].to_numpy(np.float32)
    y_residual = frame["target"].to_numpy(np.float32)
    base = frame["last_known_tvt"].to_numpy(np.float32)
    y_tvt = base + y_residual
    groups = frame["well"].to_numpy()
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    by_well_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models" / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "mode": mode_name,
                "rows": int(len(frame)),
                "features": int(len(feature_columns)),
                "configs": int(len(configs)),
                "use_gpu": bool(mode_config.get("use_gpu", False)),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for model_index, params in enumerate(configs):
        oof = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y_residual, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
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
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__lgb{model_index}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(y_tvt[valid_idx], base[valid_idx] + pred),
                    "rmse_residual": rmse(y_residual[valid_idx], pred),
                    "prediction_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred,
                        label=f"{mode_name}/lgb{model_index}/fold{fold}",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            if save_models:
                model_rows.append(
                    {
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": f"{mode_name}/{model_file}",
                        "sha256": model_sha,
                    }
                )
            print(
                json.dumps(
                    {
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "fold": int(fold),
                        "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                        "best_iteration": best_iter,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        oof_by_model.append(oof)
        pred_tvt = base + oof
        pooled_sha = prediction_sha256(
            frame["id"],
            oof,
            label=f"{mode_name}/lgb{model_index}/pooled",
        )
        metric_rows.append(
            {
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "best_iteration": None,
                "rmse_tvt": rmse(y_tvt, pred_tvt),
                "rmse_residual": rmse(y_residual, oof),
                "prediction_sha256": pooled_sha,
                "model_file": None,
                "model_sha256": None,
            }
        )
        pred_frame = pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well"].to_numpy(),
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "target_tvt": y_tvt,
                "last_known_tvt": base,
                "target_delta": y_residual,
                "pred_delta": oof,
                "pred_tvt": pred_tvt,
            }
        )
        prediction_frames.append(pred_frame)
        by_well_frames.append(_by_well_metrics(pred_frame, mode_name, f"lgb{model_index}"))

    ensemble = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_tvt = base + ensemble
    ensemble_sha = prediction_sha256(frame["id"], ensemble, label=f"{mode_name}/lgb_mean/pooled")
    metric_rows.append(
        {
            "mode": mode_name,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "best_iteration": None,
            "rmse_tvt": rmse(y_tvt, ensemble_tvt),
            "rmse_residual": rmse(y_residual, ensemble),
            "prediction_sha256": ensemble_sha,
            "model_file": None,
            "model_sha256": None,
        }
    )
    pred_frame = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "mode": mode_name,
            "model": "lgb_mean",
            "target_tvt": y_tvt,
            "last_known_tvt": base,
            "target_delta": y_residual,
            "pred_delta": ensemble,
            "pred_tvt": ensemble_tvt,
        }
    )
    prediction_frames.append(pred_frame)
    by_well_frames.append(_by_well_metrics(pred_frame, mode_name, "lgb_mean"))
    mode_summary = {
        "mode": mode_name,
        "description": mode_config.get("description"),
        "use_gpu": bool(mode_config.get("use_gpu", False)),
        "common_overrides": mode_config.get("common_overrides") or {},
        "lgb_configs": configs,
        "lgb_mean_prediction_sha256": ensemble_sha,
        "model_count": int(len(model_rows)),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(by_well_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        model_rows,
        mode_summary,
    )


def _by_well_metrics(predictions: pd.DataFrame, mode: str, model: str) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    by_well = (
        frame.groupby("well", as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
        .sort_values(["rmse_tvt", "rows"], ascending=[False, False])
    )
    by_well.insert(0, "model", model)
    by_well.insert(0, "mode", mode)
    return by_well


def run_reproducibility_guard(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None = None,
    modes: dict[str, dict[str, Any]] | None = None,
    active_modes: list[str] | tuple[str, ...] | None = None,
    n_splits: int = 5,
    fast: bool = False,
    early_stopping_rounds: int = 250,
    max_rows: int | None = None,
    max_train_rows: int | None = None,
    save_models: bool = True,
    save_predictions: bool = True,
) -> dict[str, Any]:
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    mode_map = modes or {}
    selected_modes = list(active_modes or mode_map)
    if not selected_modes:
        raise ValueError("No active LightGBM reproducibility modes configured")

    metric_frames: list[pd.DataFrame] = []
    well_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    for mode_name in selected_modes:
        if mode_name not in mode_map:
            raise ValueError(f"active mode is not defined under model.training.modes: {mode_name}")
        metrics, by_well, predictions, models, mode_summary = _fit_one_mode(
            mode_name=mode_name,
            mode_config=mode_map[mode_name],
            frame=frame,
            feature_columns=feature_columns,
            output_dir=output_dir,
            n_splits=n_splits,
            fast=fast,
            early_stopping_rounds=early_stopping_rounds,
            max_train_rows=max_train_rows,
            save_models=save_models,
        )
        metric_frames.append(metrics)
        well_frames.append(by_well)
        prediction_frames.append(predictions)
        model_rows.extend(models)
        mode_summaries.append(mode_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    by_well = pd.concat(well_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    if save_predictions:
        predictions.to_csv(
            output_dir / f"{OUTPUT_PREFIX}_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    pd.DataFrame({"feature": feature_columns}).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        index=False,
    )
    manifest = {
        "experiment": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "parent": "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "exp063_full_replay_lgbm_reproducibility_guard_from_exp072_cache",
        "feature_source": feature_meta,
        "n_splits": int(n_splits),
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    model_root = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    summary = {
        "experiment": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "status": "implemented_not_run" if metrics.empty else "train_completed",
        "mode": "exp063_full_replay_lgbm_reproducibility_guard_from_exp072_cache",
        "parent": "exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "active_modes": selected_modes,
        "pooled_metrics": pooled.to_dict("records"),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "predictions": f"{OUTPUT_PREFIX}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_saved_model_inference(
    *,
    output_dir: str | Path,
    submission_path: str | Path,
    sample_submission_path: str | Path,
    data_dir: str | Path | None = None,
    tracker_test_path: str | Path | None = None,
    model_manifest_path: str | Path | None = None,
    mode_name: str = "gpu_repro_guard_dp_threads8",
    model_name: str = "lgb_mean",
    submission_target_column: str = "tvt",
    regenerate_test_features: bool = True,
    n_jobs: int | None = None,
    pf_seeds: int | None = None,
    pf_particles: int | None = None,
    fast: bool = False,
    use_gpu: str = "auto",
) -> dict[str, Any]:
    import lightgbm as lgb

    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = Path(submission_path)
    manifest_path = find_model_manifest(model_manifest_path)
    model_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if regenerate_test_features:
        if data_dir is None:
            raise ValueError("data_dir is required when regenerate_test_features=True")
        test_frame, test_meta = generate_exp063_tracker_test_frame(
            data_dir=data_dir,
            output_dir=output_dir,
            n_jobs=n_jobs,
            pf_seeds=pf_seeds,
            pf_particles=pf_particles,
            fast=fast,
            use_gpu=use_gpu,
        )
    else:
        test_frame, test_meta = load_exp063_tracker_test_frame(tracker_test_path)
    feature_columns = manifest.get("feature_source", {}).get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError(f"{manifest_path} does not contain feature_source.feature_columns")
    missing = sorted(set(feature_columns) - set(test_frame.columns))
    if missing:
        raise ValueError(f"test tracker frame is missing model features: {missing[:20]}")

    model_rows = [
        item
        for item in manifest.get("models", [])
        if str(item.get("mode")) == mode_name
        and (model_name == "lgb_mean" or str(item.get("model")) == model_name)
    ]
    if not model_rows:
        raise ValueError(f"No saved models for mode={mode_name} model={model_name}")

    x_matrix = test_frame[feature_columns].to_numpy(np.float32)
    pred_delta = np.zeros(len(test_frame), dtype=np.float32)
    loaded_rows: list[dict[str, Any]] = []
    for item in model_rows:
        model_file = model_root / str(item["file"])
        booster = lgb.Booster(model_file=str(model_file))
        pred = booster.predict(x_matrix).astype(np.float32)
        pred_delta += pred / float(len(model_rows))
        loaded_rows.append(
            {
                "mode": item.get("mode"),
                "model": item.get("model"),
                "fold": item.get("fold"),
                "file": str(item.get("file")),
                "sha256": item.get("sha256"),
                "rows": int(len(pred)),
            }
        )

    base = test_frame["last_known_tvt"].to_numpy(np.float32)
    pred_tvt = (base + pred_delta).astype(np.float32)
    predictions = pd.DataFrame(
        {
            "id": test_frame["id"].to_numpy(),
            "well": test_frame["well"].to_numpy(),
            "mode": mode_name,
            "model": model_name,
            "last_known_tvt": base,
            "pred_delta": pred_delta,
            "pred_tvt": pred_tvt,
        }
    )
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_inference_test_predictions.csv.gz"
    predictions.to_csv(predictions_path, index=False, compression="gzip")

    sample = pd.read_csv(sample_submission_path, dtype={"id": str})
    target_column = (
        submission_target_column
        if submission_target_column in sample.columns
        else str(sample.columns[1])
    )
    pred_map = dict(zip(predictions["id"].astype(str), predictions["pred_tvt"], strict=False))
    mapped = sample["id"].astype(str).map(pred_map)
    fallback = float(predictions["pred_tvt"].mean())
    missing_mask = mapped.isna()
    sample[target_column] = mapped.fillna(fallback).astype("float64")
    sample.to_csv(submission_path, index=False)

    submission_sha = sha256_file(submission_path)
    prediction_sha = prediction_sha256(
        predictions["id"],
        pred_delta,
        label=f"{mode_name}/{model_name}/test",
    )
    metrics = {
        "mode": mode_name,
        "model": model_name,
        "model_count": int(len(model_rows)),
        "test_rows": int(len(test_frame)),
        "submission_rows": int(len(sample)),
        "predicted_rows": int((~missing_mask).sum()),
        "fallback_rows": int(missing_mask.sum()),
        "prediction_min": float(sample[target_column].min()),
        "prediction_max": float(sample[target_column].max()),
        "prediction_mean": float(sample[target_column].mean()),
        "prediction_std": float(sample[target_column].std()),
        "prediction_sha256": prediction_sha,
        "submission_sha256": submission_sha,
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_inference_metrics.csv",
        index=False,
    )
    summary = {
        "experiment": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "status": "inference_completed",
        "mode": "saved_lgb_booster_inference_from_exp073_full_replay_train",
        "train_manifest": str(manifest_path),
        "test_feature_source": test_meta,
        "selected": {
            "mode": mode_name,
            "model": model_name,
            "model_count": int(len(model_rows)),
        },
        "metrics": metrics,
        "loaded_models": loaded_rows,
        "artifacts": {
            "predictions": predictions_path.name,
            "metrics": f"{OUTPUT_PREFIX}_inference_metrics.csv",
            "summary": f"{OUTPUT_PREFIX}_inference_summary.json",
            "submission": str(submission_path),
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_inference_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
