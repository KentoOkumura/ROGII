from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from learned_likelihood_fulltrain_addonly_on_exp092 import (
    add_anchor_columns,
    apply_mode_overrides,
    build_learned_likelihood_features,
    build_u_projection_features,
    exp063_lgb_config_family,
    feature_columns_for_variant,
    load_exp072_full_replay_cache_frame,
    load_learned_likelihood_ml_features,
    prediction_sha256,
    rmse,
    sha256_file,
)


EXPERIMENT_NAME = "exp229_lgb_quantile_band_emission_hmm_on_exp148"
OUTPUT_PREFIX = "exp229_lgb_quantile_band_emission_hmm_on_exp148"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value


def alpha_token(alpha: float) -> str:
    return f"q{int(round(float(alpha) * 100)):02d}"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.maximum(float(alpha) * residual, (float(alpha) - 1.0) * residual)))


def plot_mean_importance(mean_importance: pd.DataFrame, output_path: Path, top_n: int) -> None:
    import matplotlib.pyplot as plt

    if mean_importance.empty:
        return
    top = mean_importance.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.28 * len(top))))
    ax.barh(top["feature"].astype(str), top["mean_importance"].astype(float))
    ax.set_title("Mean LightGBM feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def assemble_exp148_training_surface(
    *,
    output_dir: Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    learned_feature_path: str | Path | None,
    learned_schema_path: str | Path | None,
    learned_summary_path: str | Path | None,
    projection_config: dict[str, Any],
    learned_feature_config: dict[str, Any],
    variant: dict[str, Any],
    max_rows: int | None,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], list[dict[str, Any]]]:
    frame, base_feature_columns, feature_meta = load_exp072_full_replay_cache_frame(
        cache_path,
        max_rows=max_rows,
    )
    frame, anchor_meta = add_anchor_columns(frame, train_dir)
    learned_features_source, learned_source_meta = load_learned_likelihood_ml_features(
        learned_feature_path,
        schema_path=learned_schema_path,
        summary_path=learned_summary_path,
    )
    if projection_config.get("include_lgb_oof_features", False):
        raise NotImplementedError("Nested LGB OOF U-projection features remain disabled for exp229")

    projection_features, projection_group_columns, projection_summary = build_u_projection_features(
        frame,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )
    projection_feature_columns = [
        col for col in projection_features.columns if col not in {"id", "well"}
    ]
    full_frame = pd.concat(
        [
            frame.reset_index(drop=True),
            projection_features[projection_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    learned_features, learned_group_columns, learned_summary = build_learned_likelihood_features(
        learned_features_source,
        full_frame,
        learned_feature_config,
    )
    learned_feature_columns = [col for col in learned_features.columns if col not in {"id", "well"}]
    before_rows = len(full_frame)
    before_wells = int(full_frame["well"].nunique())
    full_frame = full_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if full_frame.empty:
        raise ValueError("No shared rows between exp148 base surface and learned likelihood features")

    feature_group_columns = {
        **projection_group_columns,
        **learned_group_columns,
    }
    feature_columns = feature_columns_for_variant(
        base_feature_columns,
        feature_group_columns,
        variant,
    )
    missing = sorted(set(feature_columns).difference(full_frame.columns))
    if missing:
        raise ValueError(f"selected feature columns are missing from assembled frame: {missing[:40]}")

    projection_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_projection_feature_summary.csv",
        index=False,
    )
    learned_summary.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_learned_feature_summary.csv",
        index=False,
    )
    feature_schema_rows = [
        {
            "variant": str(variant["name"]),
            "feature_index": int(index),
            "feature": feature,
            "is_projection_feature": bool(feature in projection_feature_columns),
            "is_learned_likelihood_feature": bool(feature in learned_feature_columns),
        }
        for index, feature in enumerate(feature_columns)
    ]
    pd.DataFrame(feature_schema_rows).to_csv(
        output_dir / f"{OUTPUT_PREFIX}_quantile_feature_schema.csv",
        index=False,
    )

    metadata = {
        "feature_source": feature_meta,
        "learned_likelihood_feature_source": learned_source_meta,
        "anchor_source": anchor_meta,
        "projection_config": projection_config,
        "learned_feature_config": learned_feature_config,
        "coverage": {
            "base_rows_before_feature_join": int(before_rows),
            "base_wells_before_feature_join": int(before_wells),
            "learned_feature_rows": int(learned_source_meta["rows"]),
            "learned_feature_wells": int(learned_source_meta["wells"]),
            "joined_rows": int(len(full_frame)),
            "joined_wells": int(full_frame["well"].nunique()),
            "dropped_base_rows": int(before_rows - len(full_frame)),
            "dropped_base_wells": int(before_wells - full_frame["well"].nunique()),
            "full_train_coverage_pass": bool(
                before_rows == len(full_frame) and before_wells == full_frame["well"].nunique()
            ),
        },
        "feature_count": int(len(feature_columns)),
    }
    return full_frame, feature_columns, metadata, feature_schema_rows


def select_lgb_configs(
    *,
    mode_config: dict[str, Any],
    fast: bool,
) -> list[tuple[int, dict[str, Any]]]:
    configs = apply_mode_overrides(exp063_lgb_config_family(fast=fast), mode_config)
    selected_indices = [int(v) for v in (mode_config.get("selected_config_indices") or [1])]
    selected: list[tuple[int, dict[str, Any]]] = []
    for index in selected_indices:
        if index < 0 or index >= len(configs):
            raise IndexError(f"selected LightGBM config index out of range: {index}")
        selected.append((index, dict(configs[index])))
    return selected


def corrected_quantile_band(
    *,
    frame: pd.DataFrame,
    low_column: str,
    mid_column: str,
    high_column: str,
    sigma_divisor: float,
    sigma_floor: float,
    sigma_cap: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    low_raw = pd.to_numeric(frame[low_column], errors="coerce").to_numpy(np.float64)
    mid_raw = pd.to_numeric(frame[mid_column], errors="coerce").to_numpy(np.float64)
    high_raw = pd.to_numeric(frame[high_column], errors="coerce").to_numpy(np.float64)
    target = pd.to_numeric(frame["target_tvt"], errors="coerce").to_numpy(np.float64)
    raw_stack = np.vstack([low_raw, mid_raw, high_raw]).T
    if not np.isfinite(raw_stack).all():
        raise ValueError("raw quantile predictions contain non-finite values")
    sorted_stack = np.sort(raw_stack, axis=1)
    low = sorted_stack[:, 0]
    mid = sorted_stack[:, 1]
    high = sorted_stack[:, 2]
    band_width = np.maximum(high - low, 0.0)
    sigma_raw = band_width / float(sigma_divisor)
    sigma = np.clip(sigma_raw, float(sigma_floor), float(sigma_cap))
    crossing_low_mid = low_raw > mid_raw
    crossing_mid_high = mid_raw > high_raw
    crossing_any = crossing_low_mid | crossing_mid_high

    out = frame.copy()
    out["q_low_raw_tvt"] = low_raw.astype(np.float32)
    out["q_mid_raw_tvt"] = mid_raw.astype(np.float32)
    out["q_high_raw_tvt"] = high_raw.astype(np.float32)
    out["q_low_tvt"] = low.astype(np.float32)
    out["q_mid_tvt"] = mid.astype(np.float32)
    out["q_high_tvt"] = high.astype(np.float32)
    out["band_width_tvt"] = band_width.astype(np.float32)
    out["sigma_raw_tvt"] = sigma_raw.astype(np.float32)
    out["sigma_tvt"] = sigma.astype(np.float32)
    out["crossing_low_mid"] = crossing_low_mid.astype(np.int8)
    out["crossing_mid_high"] = crossing_mid_high.astype(np.int8)
    out["crossing_any"] = crossing_any.astype(np.int8)

    summary = {
        "raw_q_mid_rmse": rmse(target, mid_raw),
        "corrected_q_mid_rmse": rmse(target, mid),
        "raw_band_coverage": float(np.mean((target >= low_raw) & (target <= high_raw))),
        "corrected_band_coverage": float(np.mean((target >= low) & (target <= high))),
        "crossing_low_mid_rate": float(np.mean(crossing_low_mid)),
        "crossing_mid_high_rate": float(np.mean(crossing_mid_high)),
        "crossing_any_rate": float(np.mean(crossing_any)),
        "band_width_mean": float(np.mean(band_width)),
        "band_width_p50": float(np.quantile(band_width, 0.50)),
        "band_width_p90": float(np.quantile(band_width, 0.90)),
        "sigma_raw_mean": float(np.mean(sigma_raw)),
        "sigma_effective_mean": float(np.mean(sigma)),
        "sigma_effective_p10": float(np.quantile(sigma, 0.10)),
        "sigma_effective_p50": float(np.quantile(sigma, 0.50)),
        "sigma_effective_p90": float(np.quantile(sigma, 0.90)),
        "sigma_floor_rate": float(np.mean(sigma_raw < float(sigma_floor))),
        "sigma_cap_rate": float(np.mean(sigma_raw > float(sigma_cap))),
    }
    return out, summary


def run_quantile_lgb_train(
    *,
    output_dir: str | Path,
    train_dir: str | Path,
    cache_path: str | Path | None,
    learned_feature_path: str | Path | None,
    learned_schema_path: str | Path | None,
    learned_summary_path: str | Path | None,
    projection_config: dict[str, Any],
    learned_feature_config: dict[str, Any],
    variant: dict[str, Any],
    mode_name: str,
    mode_config: dict[str, Any],
    quantile_config: dict[str, Any],
    n_splits: int,
    fast: bool = False,
    max_rows: int | None = None,
    max_train_rows: int | None = None,
    save_models: bool = True,
    save_predictions: bool = True,
    top_n_importance: int = 40,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from sklearn.model_selection import GroupKFold

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(v) for v in quantile_config.get("alphas", [0.16, 0.50, 0.84])]
    low_alpha = float(quantile_config.get("low_alpha", alphas[0]))
    mid_alpha = float(quantile_config.get("mid_alpha", 0.50))
    high_alpha = float(quantile_config.get("high_alpha", alphas[-1]))
    for required_alpha in (low_alpha, mid_alpha, high_alpha):
        if min(abs(required_alpha - alpha) for alpha in alphas) > 1e-8:
            raise ValueError(f"required alpha {required_alpha} is not in quantile_lgb.alphas={alphas}")

    full_frame, feature_columns, feature_meta, feature_schema_rows = assemble_exp148_training_surface(
        output_dir=output_dir,
        train_dir=train_dir,
        cache_path=cache_path,
        learned_feature_path=learned_feature_path,
        learned_schema_path=learned_schema_path,
        learned_summary_path=learned_summary_path,
        projection_config=projection_config,
        learned_feature_config=learned_feature_config,
        variant=variant,
        max_rows=max_rows,
    )

    x_matrix = full_frame[feature_columns].to_numpy(np.float32)
    y = full_frame["target"].to_numpy(np.float32)
    base = full_frame["last_known_tvt"].to_numpy(np.float32)
    target_tvt = base + y
    groups = full_frame["well"].astype(str).to_numpy()
    cv = GroupKFold(n_splits=int(n_splits))
    rng = np.random.default_rng(int(quantile_config.get("subsample_seed", 42)))
    selected_configs = select_lgb_configs(mode_config=mode_config, fast=fast)
    early_stopping_rounds = int(mode_config.get("early_stopping_rounds", 250))

    model_root = output_dir / f"{OUTPUT_PREFIX}_quantile_lgb_models"
    if save_models:
        model_root.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    oof_by_alpha_model: dict[tuple[float, str], np.ndarray] = {}

    for config_index, base_params in selected_configs:
        model_label = f"lgb{config_index}"
        for alpha in alphas:
            token = alpha_token(alpha)
            oof = np.zeros(len(full_frame), dtype=np.float32)
            params = dict(base_params)
            params["objective"] = "quantile"
            params["alpha"] = float(alpha)
            params.setdefault("metric", "quantile")
            for fold, (train_idx, valid_idx) in enumerate(cv.split(x_matrix, y, groups=groups)):
                if max_train_rows is not None and len(train_idx) > int(max_train_rows):
                    train_idx = np.sort(
                        rng.choice(train_idx, size=int(max_train_rows), replace=False)
                    )
                model = LGBMRegressor(**params)
                model.fit(
                    x_matrix[train_idx],
                    y[train_idx],
                    eval_set=[(x_matrix[valid_idx], y[valid_idx])],
                    eval_metric="quantile",
                    callbacks=[
                        early_stopping(early_stopping_rounds, verbose=False),
                        log_evaluation(0),
                    ],
                )
                best_iter = int(model.best_iteration_ or params.get("n_estimators", 0))
                pred = model.predict(x_matrix[valid_idx], num_iteration=best_iter).astype(np.float32)
                oof[valid_idx] = pred
                pred_tvt = base[valid_idx] + pred
                model_file = None
                model_sha = None
                if save_models:
                    model_file = f"{mode_name}__{model_label}__{token}__fold{fold}.txt"
                    model_path = model_root / model_file
                    model.booster_.save_model(str(model_path), num_iteration=best_iter)
                    model_sha = sha256_file(model_path)
                    model_rows.append(
                        {
                            "mode": mode_name,
                            "model": model_label,
                            "config_index": int(config_index),
                            "alpha": float(alpha),
                            "alpha_token": token,
                            "fold": int(fold),
                            "best_iteration": best_iter,
                            "file": model_file,
                            "sha256": model_sha,
                        }
                    )
                metric_rows.append(
                    {
                        "mode": mode_name,
                        "model": model_label,
                        "config_index": int(config_index),
                        "alpha": float(alpha),
                        "alpha_token": token,
                        "fold": int(fold),
                        "rows": int(len(valid_idx)),
                        "train_rows": int(len(train_idx)),
                        "features": int(len(feature_columns)),
                        "best_iteration": best_iter,
                        "pinball_target": pinball_loss(y[valid_idx], pred, alpha),
                        "rmse_tvt": rmse(target_tvt[valid_idx], pred_tvt),
                        "prediction_sha256": prediction_sha256(
                            full_frame.iloc[valid_idx]["id"],
                            pred_tvt,
                            label=f"{EXPERIMENT_NAME}/{mode_name}/{model_label}/{token}/fold{fold}/tvt",
                        ),
                        "model_file": model_file,
                        "model_sha256": model_sha,
                    }
                )
                for feature, importance in zip(feature_columns, model.feature_importances_, strict=False):
                    importance_rows.append(
                        {
                            "mode": mode_name,
                            "model": model_label,
                            "config_index": int(config_index),
                            "alpha": float(alpha),
                            "alpha_token": token,
                            "fold": int(fold),
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "model": model_label,
                            "alpha": float(alpha),
                            "fold": int(fold),
                            "rmse_tvt": metric_rows[-1]["rmse_tvt"],
                            "pinball_target": metric_rows[-1]["pinball_target"],
                            "best_iteration": best_iter,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            oof_by_alpha_model[(alpha, model_label)] = oof
            pred_tvt = base + oof
            metric_rows.append(
                {
                    "mode": mode_name,
                    "model": model_label,
                    "config_index": int(config_index),
                    "alpha": float(alpha),
                    "alpha_token": token,
                    "fold": "pooled",
                    "rows": int(len(full_frame)),
                    "train_rows": None,
                    "features": int(len(feature_columns)),
                    "best_iteration": None,
                    "pinball_target": pinball_loss(y, oof, alpha),
                    "rmse_tvt": rmse(target_tvt, pred_tvt),
                    "prediction_sha256": prediction_sha256(
                        full_frame["id"],
                        pred_tvt,
                        label=f"{EXPERIMENT_NAME}/{mode_name}/{model_label}/{token}/pooled/tvt",
                    ),
                    "model_file": None,
                    "model_sha256": None,
                }
            )

    prediction_frame = pd.DataFrame(
        {
            "id": full_frame["id"].astype(str).to_numpy(),
            "well": full_frame["well"].astype(str).to_numpy(),
            "last_known_tvt": base,
            "target": y,
            "target_tvt": target_tvt,
            "md_since": full_frame["md_since"].to_numpy(np.float32),
        }
    )
    for alpha in alphas:
        token = alpha_token(alpha)
        model_preds = [
            values
            for (pred_alpha, _model_label), values in oof_by_alpha_model.items()
            if abs(float(pred_alpha) - float(alpha)) <= 1e-8
        ]
        if not model_preds:
            raise ValueError(f"no OOF predictions collected for alpha={alpha}")
        mean_residual = np.mean(np.vstack(model_preds), axis=0).astype(np.float32)
        prediction_frame[f"{token}_target"] = mean_residual
        prediction_frame[f"{token}_tvt"] = (base + mean_residual).astype(np.float32)
        metric_rows.append(
            {
                "mode": mode_name,
                "model": "lgb_quantile_mean",
                "config_index": ",".join(str(index) for index, _ in selected_configs),
                "alpha": float(alpha),
                "alpha_token": token,
                "fold": "pooled",
                "rows": int(len(full_frame)),
                "train_rows": None,
                "features": int(len(feature_columns)),
                "best_iteration": None,
                "pinball_target": pinball_loss(y, mean_residual, alpha),
                "rmse_tvt": rmse(target_tvt, base + mean_residual),
                "prediction_sha256": prediction_sha256(
                    full_frame["id"],
                    base + mean_residual,
                    label=f"{EXPERIMENT_NAME}/{mode_name}/lgb_quantile_mean/{token}/pooled/tvt",
                ),
                "model_file": None,
                "model_sha256": None,
            }
        )

    low_token = alpha_token(low_alpha)
    mid_token = alpha_token(mid_alpha)
    high_token = alpha_token(high_alpha)
    prediction_frame, band_summary = corrected_quantile_band(
        frame=prediction_frame,
        low_column=f"{low_token}_tvt",
        mid_column=f"{mid_token}_tvt",
        high_column=f"{high_token}_tvt",
        sigma_divisor=float(quantile_config.get("band_to_sigma_divisor", 2.0)),
        sigma_floor=float(quantile_config.get("sigma_floor", 6.0)),
        sigma_cap=float(quantile_config.get("sigma_cap", 30.0)),
    )

    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    mean_importance = (
        importance.groupby(["mode", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            records=("importance", "size"),
        )
        .sort_values("mean_importance", ascending=False)
    )

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_quantile_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_quantile_predictions.csv.gz"
    importance_path = output_dir / f"{OUTPUT_PREFIX}_quantile_feature_importance.csv"
    mean_importance_path = output_dir / f"{OUTPUT_PREFIX}_quantile_feature_importance_mean.csv"
    importance_plot_path = output_dir / f"{OUTPUT_PREFIX}_quantile_feature_importance_mean_top.png"
    manifest_path = model_root / "manifest.json"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_quantile_summary.json"

    metrics.to_csv(metrics_path, index=False)
    importance.to_csv(importance_path, index=False)
    mean_importance.to_csv(mean_importance_path, index=False)
    plot_mean_importance(mean_importance, importance_plot_path, int(top_n_importance))
    if save_predictions:
        prediction_frame.to_csv(predictions_path, index=False, compression="gzip")

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "mode": mode_name,
        "variant": variant,
        "feature_meta": feature_meta,
        "feature_schema": feature_schema_rows,
        "quantile_config": quantile_config,
        "selected_lgb_configs": [
            {"config_index": int(index), "params": params}
            for index, params in selected_configs
        ],
        "models": model_rows,
        "model_count": int(len(model_rows)),
    }
    if save_models:
        write_json(manifest_path, manifest)

    pooled = metrics[metrics["fold"].astype(str).eq("pooled")].copy()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "quantile_lgb_train_completed",
        "mode": mode_name,
        "rows": int(len(full_frame)),
        "wells": int(full_frame["well"].nunique()),
        "features": int(len(feature_columns)),
        "n_splits": int(n_splits),
        "alphas": alphas,
        "selected_config_indices": [int(index) for index, _ in selected_configs],
        "booster_count": int(len(model_rows)),
        "parent_control_retraining": False,
        "band_summary": band_summary,
        "pooled_metrics": to_jsonable(pooled.to_dict("records")),
        "artifacts": {
            "metrics": metrics_path.name,
            "predictions": predictions_path.name if save_predictions else None,
            "feature_importance": importance_path.name,
            "feature_importance_mean": mean_importance_path.name,
            "feature_importance_plot": importance_plot_path.name,
            "feature_schema": f"{OUTPUT_PREFIX}_quantile_feature_schema.csv",
            "model_manifest": f"{OUTPUT_PREFIX}_quantile_lgb_models/manifest.json" if save_models else None,
            "summary": summary_path.name,
        },
        "sha256": {
            "metrics": sha256_file(metrics_path),
            "predictions_gzip": sha256_file(predictions_path) if save_predictions else None,
            "feature_importance": sha256_file(importance_path),
            "feature_importance_mean": sha256_file(mean_importance_path),
            "model_manifest": sha256_file(manifest_path) if save_models else None,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary
