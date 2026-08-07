from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
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
OUTPUT_PREFIX = "exp080_u_space_target_ablation"
META_COLUMNS = {"id", "well", "target"}
EXPECTED_FULL_REPLAY_FEATURE_COUNT = 196


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str

    def make_target(
        self,
        y_tvt: np.ndarray,
        t0: np.ndarray,
        z: np.ndarray,
        z0: np.ndarray,
    ) -> np.ndarray:
        if self.name == "dTVT":
            value = y_tvt - t0
        elif self.name == "dTVT_plus_dZ":
            value = (y_tvt - t0) + (z - z0)
        elif self.name == "TVT_plus_Z_abs":
            value = y_tvt + z
        elif self.name == "TVT_plus_Z_minus_T0":
            value = y_tvt + z - t0
        elif self.name == "TVT_plus_Z_minus_T0Z0":
            value = y_tvt + z - (t0 + z0)
        else:
            raise ValueError(f"Unknown target spec: {self.name}")
        return value.astype(np.float32)

    def inverse(
        self,
        pred_target: np.ndarray,
        t0: np.ndarray,
        z: np.ndarray,
        z0: np.ndarray,
    ) -> np.ndarray:
        if self.name == "dTVT":
            value = t0 + pred_target
        elif self.name == "dTVT_plus_dZ":
            value = t0 + pred_target - (z - z0)
        elif self.name == "TVT_plus_Z_abs":
            value = pred_target - z
        elif self.name == "TVT_plus_Z_minus_T0":
            value = pred_target + t0 - z
        elif self.name == "TVT_plus_Z_minus_T0Z0":
            value = pred_target + t0 + z0 - z
        else:
            raise ValueError(f"Unknown target spec: {self.name}")
        return value.astype(np.float32)


TARGET_SPECS = {
    "dTVT": TargetSpec("dTVT", "TVT - T0, equivalent to exp073 last_known_tvt residual."),
    "dTVT_plus_dZ": TargetSpec("dTVT_plus_dZ", "(TVT - T0) + (Z - Z0)."),
    "TVT_plus_Z_abs": TargetSpec("TVT_plus_Z_abs", "TVT + Z absolute U-space value."),
    "TVT_plus_Z_minus_T0": TargetSpec("TVT_plus_Z_minus_T0", "TVT + Z - T0."),
    "TVT_plus_Z_minus_T0Z0": TargetSpec(
        "TVT_plus_Z_minus_T0Z0",
        "TVT + Z - (T0 + Z0).",
    ),
}


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


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
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


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
    required = {"id", "well", "target", "last_known_tvt", "z"}
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


def load_known_prefix_anchors(train_dir: str | Path, wells: list[str] | pd.Series) -> pd.DataFrame:
    train_dir = Path(train_dir)
    rows: list[dict[str, Any]] = []
    for well in sorted(set(map(str, wells))):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"raw train well file not found for anchor recovery: {path}")
        frame = pd.read_csv(
            path,
            usecols=["MD", "Z", "TVT", "TVT_input"],
        )
        known = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()].copy()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for well {well}")
        anchor = known.iloc[-1]
        rows.append(
            {
                "well": well,
                "anchor_md": float(anchor["MD"]),
                "anchor_z0": float(anchor["Z"]),
                "anchor_t0": float(anchor["TVT_input"]),
                "anchor_tvt_true": float(anchor["TVT"]),
                "known_prefix_rows": int(len(known)),
            }
        )
    return pd.DataFrame(rows)


def add_anchor_columns(
    frame: pd.DataFrame,
    train_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = load_known_prefix_anchors(train_dir, frame["well"])
    merged = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if merged[["anchor_t0", "anchor_z0"]].isna().any().any():
        raise ValueError("Anchor merge produced missing T0/Z0 values")
    t0_delta = (
        merged["last_known_tvt"].to_numpy(np.float32)
        - merged["anchor_t0"].to_numpy(np.float32)
    )
    meta = {
        "anchor_wells": int(len(anchors)),
        "anchor_t0_vs_last_known_abs_max": float(np.max(np.abs(t0_delta))),
        "anchor_t0_vs_last_known_abs_mean": float(np.mean(np.abs(t0_delta))),
        "known_prefix_rows_min": int(anchors["known_prefix_rows"].min()),
        "known_prefix_rows_max": int(anchors["known_prefix_rows"].max()),
    }
    if meta["anchor_t0_vs_last_known_abs_max"] > 0.05:
        raise ValueError(
            "Recovered raw prefix T0 does not match feature cache last_known_tvt; "
            f"max abs diff={meta['anchor_t0_vs_last_known_abs_max']}"
        )
    return merged, meta


def _tail_rank(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(-1).to_numpy(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _tail_rank(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def _by_well_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    by_well = (
        frame.groupby(["target_name", "mode", "model", "well"], as_index=False)
        .agg(
            rows=("id", "size"),
            rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
            error_mean=("error_tvt", "mean"),
            error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
        )
        .sort_values(
            ["target_name", "mode", "model", "rmse_tvt"],
            ascending=[True, True, True, False],
        )
    )
    return by_well


def _bucket_metrics(predictions: pd.DataFrame, source_frame: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[["id", "target_name", "mode", "model", "target_tvt", "pred_tvt"]].copy()
    context = source_frame[["id"]].copy()
    distance_source = source_frame.get(
        "md_since",
        pd.Series(np.nan, index=source_frame.index),
    )
    context["distance_bucket"] = _distance_bucket(distance_source)
    context["tail_rank_bucket"] = _tail_rank_bucket(source_frame["id"])
    frame = frame.merge(context, on="id", how="left", validate="many_to_one")
    frame["error_tvt"] = frame["pred_tvt"] - frame["target_tvt"]
    rows: list[pd.DataFrame] = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        grouped = (
            frame.groupby(["target_name", "mode", "model", bucket_col], observed=True)
            .agg(
                rows=("id", "size"),
                rmse_tvt=("error_tvt", lambda value: float(np.sqrt(np.mean(np.square(value))))),
                error_abs_mean=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            )
            .reset_index()
            .rename(columns={bucket_col: "bucket"})
        )
        grouped.insert(3, "bucket_family", bucket_col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def _target_summary(frame: pd.DataFrame, active_specs: list[TargetSpec]) -> pd.DataFrame:
    y_tvt = frame["last_known_tvt"].to_numpy(np.float32) + frame["target"].to_numpy(np.float32)
    t0 = frame["anchor_t0"].to_numpy(np.float32)
    z = frame["z"].to_numpy(np.float32)
    z0 = frame["anchor_z0"].to_numpy(np.float32)
    rows: list[dict[str, Any]] = []
    for spec in active_specs:
        target = spec.make_target(y_tvt, t0, z, z0)
        rows.append(
            {
                "target_name": spec.name,
                "rows": int(len(target)),
                "mean": float(np.mean(target)),
                "std": float(np.std(target)),
                "min": float(np.min(target)),
                "p01": float(np.quantile(target, 0.01)),
                "p50": float(np.quantile(target, 0.50)),
                "p99": float(np.quantile(target, 0.99)),
                "max": float(np.max(target)),
                "abs_p99": float(np.quantile(np.abs(target), 0.99)),
            }
        )
    return pd.DataFrame(rows)


def _fit_one_target_mode(
    *,
    target_spec: TargetSpec,
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    x_matrix = frame[feature_columns].to_numpy(np.float32)
    t0 = frame["anchor_t0"].to_numpy(np.float32)
    z0 = frame["anchor_z0"].to_numpy(np.float32)
    z = frame["z"].to_numpy(np.float32)
    y_tvt = frame["last_known_tvt"].to_numpy(np.float32) + frame["target"].to_numpy(np.float32)
    y_target = target_spec.make_target(y_tvt, t0, z, z0)
    groups = frame["well"].to_numpy()
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(42)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_model: list[np.ndarray] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models" / target_spec.name / mode_name
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "target": target_spec.name,
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
        oof_target = np.zeros(len(frame), dtype=np.float32)
        splits = cv.split(x_matrix, y_target, groups=groups)
        for fold, (train_idx, valid_idx) in enumerate(splits):
            if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                train_idx = np.sort(rng.choice(train_idx, size=int(max_train_rows), replace=False))
            model = LGBMRegressor(**params)
            model.fit(
                x_matrix[train_idx],
                y_target[train_idx],
                eval_set=[(x_matrix[valid_idx], y_target[valid_idx])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(early_stopping_rounds), verbose=False),
                    log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
            pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
            oof_target[valid_idx] = pred
            pred_tvt = target_spec.inverse(pred, t0[valid_idx], z[valid_idx], z0[valid_idx])
            model_file = None
            model_sha = None
            if save_models:
                model_file = f"{mode_name}__lgb{model_index}__fold{fold}.txt"
                model_path = model_dir / model_file
                model.booster_.save_model(str(model_path), num_iteration=best_iter)
                model_sha = sha256_file(model_path)
            metric_rows.append(
                {
                    "target_name": target_spec.name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "fold": int(fold),
                    "rows": int(len(valid_idx)),
                    "train_rows": int(len(train_idx)),
                    "features": int(len(feature_columns)),
                    "best_iteration": best_iter,
                    "rmse_tvt": rmse(y_tvt[valid_idx], pred_tvt),
                    "rmse_target": rmse(y_target[valid_idx], pred),
                    "prediction_target_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred,
                        label=f"{target_spec.name}/{mode_name}/lgb{model_index}/fold{fold}/target",
                    ),
                    "prediction_tvt_sha256": prediction_sha256(
                        frame.iloc[valid_idx]["id"],
                        pred_tvt,
                        label=f"{target_spec.name}/{mode_name}/lgb{model_index}/fold{fold}/tvt",
                    ),
                    "model_file": model_file,
                    "model_sha256": model_sha,
                }
            )
            if save_models:
                model_rows.append(
                    {
                        "target_name": target_spec.name,
                        "mode": mode_name,
                        "model": f"lgb{model_index}",
                        "model_index": int(model_index),
                        "fold": int(fold),
                        "best_iteration": best_iter,
                        "file": f"{target_spec.name}/{mode_name}/{model_file}",
                        "sha256": model_sha,
                    }
                )
            print(
                json.dumps(
                    {
                        "target": target_spec.name,
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
        oof_by_model.append(oof_target)
        pred_tvt = target_spec.inverse(oof_target, t0, z, z0)
        metric_rows.append(
            {
                "target_name": target_spec.name,
                "mode": mode_name,
                "model": f"lgb{model_index}",
                "fold": "pooled",
                "rows": int(len(frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "best_iteration": None,
                "rmse_tvt": rmse(y_tvt, pred_tvt),
                "rmse_target": rmse(y_target, oof_target),
                "prediction_target_sha256": prediction_sha256(
                    frame["id"],
                    oof_target,
                    label=f"{target_spec.name}/{mode_name}/lgb{model_index}/pooled/target",
                ),
                "prediction_tvt_sha256": prediction_sha256(
                    frame["id"],
                    pred_tvt,
                    label=f"{target_spec.name}/{mode_name}/lgb{model_index}/pooled/tvt",
                ),
                "model_file": None,
                "model_sha256": None,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "id": frame["id"].to_numpy(),
                    "well": frame["well"].to_numpy(),
                    "target_name": target_spec.name,
                    "mode": mode_name,
                    "model": f"lgb{model_index}",
                    "target_tvt": y_tvt,
                    "anchor_t0": t0,
                    "anchor_z0": z0,
                    "z": z,
                    "target_value": y_target,
                    "pred_target": oof_target,
                    "pred_tvt": pred_tvt,
                }
            )
        )

    ensemble_target = np.mean(np.vstack(oof_by_model), axis=0).astype(np.float32)
    ensemble_tvt = target_spec.inverse(ensemble_target, t0, z, z0)
    metric_rows.append(
        {
            "target_name": target_spec.name,
            "mode": mode_name,
            "model": "lgb_mean",
            "fold": "pooled",
            "rows": int(len(frame)),
            "train_rows": None,
            "features": int(len(feature_columns)),
            "best_iteration": None,
            "rmse_tvt": rmse(y_tvt, ensemble_tvt),
            "rmse_target": rmse(y_target, ensemble_target),
            "prediction_target_sha256": prediction_sha256(
                frame["id"],
                ensemble_target,
                label=f"{target_spec.name}/{mode_name}/lgb_mean/pooled/target",
            ),
            "prediction_tvt_sha256": prediction_sha256(
                frame["id"],
                ensemble_tvt,
                label=f"{target_spec.name}/{mode_name}/lgb_mean/pooled/tvt",
            ),
            "model_file": None,
            "model_sha256": None,
        }
    )
    prediction_frames.append(
        pd.DataFrame(
            {
                "id": frame["id"].to_numpy(),
                "well": frame["well"].to_numpy(),
                "target_name": target_spec.name,
                "mode": mode_name,
                "model": "lgb_mean",
                "target_tvt": y_tvt,
                "anchor_t0": t0,
                "anchor_z0": z0,
                "z": z,
                "target_value": y_target,
                "pred_target": ensemble_target,
                "pred_tvt": ensemble_tvt,
            }
        )
    )
    mode_summary = {
        "target_name": target_spec.name,
        "mode": mode_name,
        "description": mode_config.get("description"),
        "use_gpu": bool(mode_config.get("use_gpu", False)),
        "common_overrides": mode_config.get("common_overrides") or {},
        "lgb_configs": configs,
        "lgb_mean_prediction_tvt_sha256": metric_rows[-1]["prediction_tvt_sha256"],
        "model_count": int(len(model_rows)),
    }
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
        model_rows,
        mode_summary,
    )


def run_u_space_target_ablation(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None = None,
    target_names: list[str] | tuple[str, ...] | None = None,
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
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    selected_targets = list(target_names or TARGET_SPECS)
    active_specs = [TARGET_SPECS[name] for name in selected_targets]
    mode_map = modes or {}
    selected_modes = list(active_modes or mode_map)
    if not selected_modes:
        raise ValueError("No active LightGBM modes configured")

    target_stats = _target_summary(frame, active_specs)
    target_stats.to_csv(output_dir / f"{OUTPUT_PREFIX}_target_summary.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_rows: list[dict[str, Any]] = []
    mode_summaries: list[dict[str, Any]] = []
    for spec in active_specs:
        for mode_name in selected_modes:
            if mode_name not in mode_map:
                raise ValueError(
                    f"active mode is not defined under model.training.modes: {mode_name}"
                )
            metrics, predictions, models, mode_summary = _fit_one_target_mode(
                target_spec=spec,
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
            prediction_frames.append(predictions)
            model_rows.extend(models)
            mode_summaries.append(mode_summary)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    by_well = _by_well_metrics(predictions)
    bucket_metrics = _bucket_metrics(predictions, frame)

    metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_metrics.csv", index=False)
    by_well.to_csv(output_dir / f"{OUTPUT_PREFIX}_by_well.csv", index=False)
    bucket_metrics.to_csv(output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv", index=False)
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

    model_root = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "exp080_u_space_target_ablation",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "mode": "u_space_target_ablation_from_exp072_cache",
        "feature_source": feature_meta,
        "anchor_source": {
            "train_dir": str(train_dir),
            **anchor_meta,
        },
        "n_splits": int(n_splits),
        "target_specs": [
            {"name": spec.name, "description": spec.description}
            for spec in active_specs
        ],
        "models": model_rows,
        "model_count": int(len(model_rows)),
        "modes": mode_summaries,
    }
    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    lgb_mean = pooled[pooled["model"].eq("lgb_mean")].sort_values("rmse_tvt")
    best = lgb_mean.iloc[0].to_dict() if not lgb_mean.empty else None
    summary = {
        "experiment": "exp080_u_space_target_ablation",
        "status": "train_completed" if not metrics.empty else "implemented_not_run",
        "mode": "u_space_target_ablation_from_exp072_cache",
        "parent": "exp073_gpu_reproducibility_guard_for_exp063_full_replay",
        "cache_parent": "exp072_exp063_full_replay_feature_cache",
        "feature_source": feature_meta,
        "anchor_source": anchor_meta,
        "active_modes": selected_modes,
        "active_targets": selected_targets,
        "best_lgb_mean_by_rmse_tvt": _jsonable(best),
        "pooled_metrics": _jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": f"{OUTPUT_PREFIX}_metrics.csv",
            "by_well": f"{OUTPUT_PREFIX}_by_well.csv",
            "bucket_metrics": f"{OUTPUT_PREFIX}_bucket_metrics.csv",
            "target_summary": f"{OUTPUT_PREFIX}_target_summary.csv",
            "predictions": f"{OUTPUT_PREFIX}_predictions.csv.gz" if save_predictions else None,
            "feature_schema": f"{OUTPUT_PREFIX}_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_lgb_models/manifest.json",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary
