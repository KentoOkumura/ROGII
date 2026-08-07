from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
FORMATION_COLUMN = "ANCC"
REQUIRED_COLUMNS = ("MD", "X", "Y", "Z", "TVT", "TVT_input", FORMATION_COLUMN)
METHOD_GLOBAL = "global_median"
METHOD_ROW_KNN = "row_knn_xy"
METHOD_WELL_PLANE = "well_plane_knn"
METHODS = (METHOD_GLOBAL, METHOD_ROW_KNN, METHOD_WELL_PLANE)


@dataclass(frozen=True)
class WellFrame:
    well: str
    frame: pd.DataFrame
    eval_mask: np.ndarray
    anchor_position: int


@dataclass(frozen=True)
class SurfaceFit:
    global_median: float
    row_knn: KNeighborsRegressor | None
    row_xy_mean: np.ndarray | None
    row_xy_scale: np.ndarray | None
    plane_model: WellPlaneKNN | None


def nested_get(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return default if current is None else current


def well_id_from_path(path: Path) -> str:
    name = path.name
    if not name.endswith(HORIZONTAL_SUFFIX):
        raise ValueError(f"not a horizontal well CSV: {path}")
    return name.removesuffix(HORIZONTAL_SUFFIX)


def finite_rmse(error: np.ndarray) -> float:
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def finite_mae(error: np.ndarray) -> float:
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.abs(finite)))


def finite_bias(error: np.ndarray) -> float:
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def finite_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.quantile(finite, q))


def stable_content_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_well(path: Path) -> WellFrame | None:
    try:
        frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    except ValueError:
        return None
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        return None

    known_positions = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    eval_mask = frame["TVT_input"].isna().to_numpy()
    if known_positions.size == 0 or not eval_mask.any():
        return None
    anchor = frame.iloc[int(known_positions[-1])]
    anchor_values = anchor.loc[["MD", "X", "Y", "TVT_input", FORMATION_COLUMN]].to_numpy(
        dtype=float
    )
    if not np.isfinite(anchor_values).all():
        return None
    if frame.loc[eval_mask, FORMATION_COLUMN].notna().sum() == 0:
        return None

    return WellFrame(
        well=well_id_from_path(path),
        frame=frame,
        eval_mask=eval_mask,
        anchor_position=int(known_positions[-1]),
    )


def load_wells(train_dir: Path, *, max_wells: int | None = None) -> list[WellFrame]:
    paths = sorted(train_dir.glob(f"*{HORIZONTAL_SUFFIX}"))
    wells: list[WellFrame] = []
    for path in paths:
        well = load_well(path)
        if well is not None:
            wells.append(well)
        if max_wells is not None and len(wells) >= max_wells:
            break
    if len(wells) < 2:
        raise ValueError(f"not enough valid train wells found in {train_dir}: {len(wells)}")
    return wells


def sample_rows_for_knn(
    wells: list[WellFrame],
    *,
    rng: np.random.Generator,
    max_rows_per_well: int,
    max_rows_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy_parts: list[np.ndarray] = []
    ancc_parts: list[np.ndarray] = []

    for well in wells:
        frame = well.frame
        xy = frame.loc[:, ["X", "Y"]].to_numpy(dtype=float)
        ancc = frame[FORMATION_COLUMN].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(ancc)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        if indices.size > max_rows_per_well:
            indices = rng.choice(indices, size=max_rows_per_well, replace=False)
        xy_parts.append(xy[indices])
        ancc_parts.append(ancc[indices])

    if not xy_parts:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)

    xy_all = np.vstack(xy_parts)
    ancc_all = np.concatenate(ancc_parts)
    if xy_all.shape[0] > max_rows_total:
        selected = rng.choice(xy_all.shape[0], size=max_rows_total, replace=False)
        xy_all = xy_all[selected]
        ancc_all = ancc_all[selected]
    return xy_all, ancc_all


class WellPlaneKNN:
    def __init__(self, wells: list[WellFrame]) -> None:
        rows: list[dict[str, float | str]] = []
        for well in wells:
            frame = well.frame
            valid = (
                frame[["X", "Y", FORMATION_COLUMN]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if valid.empty:
                continue
            rows.append(
                {
                    "well": well.well,
                    "x": float(valid["X"].median()),
                    "y": float(valid["Y"].median()),
                    "ancc": float(valid[FORMATION_COLUMN].median()),
                }
            )

        self.summary = pd.DataFrame(rows)
        if self.summary.empty:
            self.xy = np.empty((0, 2), dtype=float)
            self.values = np.empty(0, dtype=float)
            self.xy_scale = np.ones(2, dtype=float)
            return

        self.xy = self.summary[["x", "y"]].to_numpy(dtype=float)
        self.values = self.summary["ancc"].to_numpy(dtype=float)
        scale = np.std(self.xy, axis=0)
        self.xy_scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    def predict(
        self,
        xy_query: np.ndarray,
        *,
        k: int,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        xy_query = np.asarray(xy_query, dtype=float)
        n_query = int(xy_query.shape[0])
        if self.xy.shape[0] == 0:
            return np.full(n_query, np.nan), np.full(n_query, np.nan)

        k_eff = min(max(1, int(k)), int(self.xy.shape[0]))
        train_scaled = self.xy / self.xy_scale
        pred = np.full(n_query, np.nan, dtype=float)
        min_dist = np.full(n_query, np.nan, dtype=float)
        chunk_size = max(1, int(chunk_size))

        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            query_scaled = xy_query[start:end] / self.xy_scale
            diff = query_scaled[:, None, :] - train_scaled[None, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            neighbor_idx = np.argpartition(distances, kth=k_eff - 1, axis=1)[:, :k_eff]
            neighbor_dist = np.take_along_axis(distances, neighbor_idx, axis=1)
            min_dist[start:end] = np.min(neighbor_dist, axis=1)

            weight = 1.0 / (neighbor_dist + 1e-3)
            x = self.xy[neighbor_idx, 0]
            y = self.xy[neighbor_idx, 1]
            value = self.values[neighbor_idx]
            ones = np.ones_like(x)
            design = np.stack([x, y, ones], axis=2)
            weighted_design = design * weight[:, :, None]
            lhs = np.einsum("nki,nkj->nij", design, weighted_design)
            rhs = np.einsum("nki,nk->ni", design, weight * value)
            ridge = 1e-9
            lhs[:, 0, 0] += ridge
            lhs[:, 1, 1] += ridge
            lhs[:, 2, 2] += ridge
            try:
                coef = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                coef = np.einsum("nij,nj->ni", np.linalg.pinv(lhs), rhs)

            query_xy = xy_query[start:end]
            pred[start:end] = (
                query_xy[:, 0] * coef[:, 0] + query_xy[:, 1] * coef[:, 1] + coef[:, 2]
            )
        return pred, min_dist


def fit_surface(
    train_wells: list[WellFrame],
    *,
    config: dict[str, Any],
    fold_seed: int,
) -> SurfaceFit:
    ancc_values = np.concatenate(
        [
            well.frame[FORMATION_COLUMN].to_numpy(dtype=float)[
                np.isfinite(well.frame[FORMATION_COLUMN].to_numpy(dtype=float))
            ]
            for well in train_wells
        ]
    )
    global_median = float(np.nanmedian(ancc_values))

    rng = np.random.default_rng(fold_seed)
    xy, ancc = sample_rows_for_knn(
        train_wells,
        rng=rng,
        max_rows_per_well=int(nested_get(config, "surface.row_knn.max_rows_per_well", 250)),
        max_rows_total=int(nested_get(config, "surface.row_knn.max_rows_total", 150000)),
    )

    row_knn: KNeighborsRegressor | None = None
    xy_mean: np.ndarray | None = None
    xy_scale: np.ndarray | None = None
    if xy.shape[0] > 0:
        xy_mean = np.mean(xy, axis=0)
        xy_scale = np.std(xy, axis=0)
        xy_scale = np.where(np.isfinite(xy_scale) & (xy_scale > 1e-6), xy_scale, 1.0)
        xy_scaled = (xy - xy_mean) / xy_scale
        n_neighbors = min(int(nested_get(config, "surface.row_knn.n_neighbors", 32)), xy.shape[0])
        row_knn = KNeighborsRegressor(
            n_neighbors=max(1, n_neighbors),
            weights=str(nested_get(config, "surface.row_knn.weights", "distance")),
            algorithm="kd_tree",
        )
        row_knn.fit(xy_scaled, ancc)

    return SurfaceFit(
        global_median=global_median,
        row_knn=row_knn,
        row_xy_mean=xy_mean,
        row_xy_scale=xy_scale,
        plane_model=WellPlaneKNN(train_wells),
    )


def predict_surface(
    fit: SurfaceFit,
    xy: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n_rows = int(xy.shape[0])
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        METHOD_GLOBAL: (
            np.full(n_rows, fit.global_median, dtype=float),
            np.full(n_rows, np.nan, dtype=float),
        )
    }

    if fit.row_knn is not None and fit.row_xy_mean is not None and fit.row_xy_scale is not None:
        scaled = (xy - fit.row_xy_mean) / fit.row_xy_scale
        predictions[METHOD_ROW_KNN] = (
            fit.row_knn.predict(scaled).astype(float),
            np.full(n_rows, np.nan, dtype=float),
        )
    else:
        predictions[METHOD_ROW_KNN] = (
            np.full(n_rows, np.nan, dtype=float),
            np.full(n_rows, np.nan, dtype=float),
        )

    if fit.plane_model is not None:
        predictions[METHOD_WELL_PLANE] = fit.plane_model.predict(
            xy,
            k=int(nested_get(config, "surface.well_plane_knn.n_neighbors", 16)),
            chunk_size=int(nested_get(config, "surface.well_plane_knn.chunk_size", 50000)),
        )
    else:
        predictions[METHOD_WELL_PLANE] = (
            np.full(n_rows, np.nan, dtype=float),
            np.full(n_rows, np.nan, dtype=float),
        )
    return predictions


def build_fold_predictions(
    fold: int,
    valid_wells: list[WellFrame],
    fit: SurfaceFit,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in valid_wells:
        frame = well.frame
        raw_eval_positions = np.flatnonzero(well.eval_mask)
        eval_values = frame.iloc[raw_eval_positions][
            ["MD", "X", "Y", "Z", "TVT", FORMATION_COLUMN]
        ].to_numpy(dtype=float)
        valid_eval = np.isfinite(eval_values).all(axis=1)
        eval_positions = raw_eval_positions[valid_eval]
        if eval_positions.size == 0:
            continue
        eval_frame = frame.iloc[eval_positions].copy()
        xy_eval = eval_frame[["X", "Y"]].to_numpy(dtype=float)
        xy_anchor = frame.iloc[[well.anchor_position]][["X", "Y"]].to_numpy(dtype=float)
        pred_eval = predict_surface(fit, xy_eval, config=config)
        pred_anchor = predict_surface(fit, xy_anchor, config=config)

        out = pd.DataFrame(
            {
                "fold": fold,
                "well": well.well,
                "row_index": eval_positions.astype(np.int32),
                "md": eval_frame["MD"].to_numpy(dtype=float),
                "x": eval_frame["X"].to_numpy(dtype=float),
                "y": eval_frame["Y"].to_numpy(dtype=float),
                "z": eval_frame["Z"].to_numpy(dtype=float),
                "tvt": eval_frame["TVT"].to_numpy(dtype=float),
                "anchor_row_index": int(well.anchor_position),
                "anchor_md": float(frame.iloc[well.anchor_position]["MD"]),
                "anchor_tvt": float(frame.iloc[well.anchor_position]["TVT_input"]),
                "true_ancc": eval_frame[FORMATION_COLUMN].to_numpy(dtype=float),
                "true_anchor_ancc": float(frame.iloc[well.anchor_position][FORMATION_COLUMN]),
            }
        )
        out["md_since_anchor"] = out["md"] - out["anchor_md"]

        for method in METHODS:
            values, distance = pred_eval[method]
            anchor_value = float(pred_anchor[method][0][0])
            out[f"{method}_hat"] = values
            out[f"{method}_anchor_hat"] = anchor_value
            out[f"{method}_delta_hat"] = values - anchor_value
            out[f"{method}_error"] = values - out["true_ancc"].to_numpy(dtype=float)
            out[f"{method}_delta_error"] = (
                values
                - anchor_value
                - (
                    out["true_ancc"].to_numpy(dtype=float)
                    - out["true_anchor_ancc"].to_numpy(dtype=float)
                )
            )
            out[f"{method}_neighbor_dist"] = distance

        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def distance_bucket(md_since: pd.Series) -> pd.Series:
    bins = [-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf]
    labels = ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]
    return pd.cut(md_since.astype(float), bins=bins, labels=labels, right=False).astype(str)


def summarize_method(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    error = frame[f"{method}_error"].to_numpy(dtype=float)
    delta_error = frame[f"{method}_delta_error"].to_numpy(dtype=float)
    by_well = (
        frame.assign(sq_error=error**2)
        .groupby("well", observed=True)["sq_error"]
        .mean()
        .pipe(np.sqrt)
    )
    return {
        "method": method,
        "rows": int(np.isfinite(error).sum()),
        "wells": int(frame["well"].nunique()),
        "rmse": finite_rmse(error),
        "mae": finite_mae(error),
        "bias": finite_bias(error),
        "abs_error_p95": finite_quantile(np.abs(error), 0.95),
        "abs_error_p99": finite_quantile(np.abs(error), 0.99),
        "delta_rmse": finite_rmse(delta_error),
        "delta_mae": finite_mae(delta_error),
        "delta_bias": finite_bias(delta_error),
        "well_rmse_mean": float(by_well.mean()) if len(by_well) else float("nan"),
        "well_rmse_p95": float(by_well.quantile(0.95)) if len(by_well) else float("nan"),
        "well_rmse_max": float(by_well.max()) if len(by_well) else float("nan"),
    }


def summarize_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["distance_bucket"] = distance_bucket(work["md_since_anchor"])
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for bucket, part in work.groupby("distance_bucket", observed=True):
            error = part[f"{method}_error"].to_numpy(dtype=float)
            delta_error = part[f"{method}_delta_error"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "bucket": str(bucket),
                    "rows": int(len(part)),
                    "wells": int(part["well"].nunique()),
                    "rmse": finite_rmse(error),
                    "mae": finite_mae(error),
                    "bias": finite_bias(error),
                    "delta_rmse": finite_rmse(delta_error),
                    "delta_mae": finite_mae(delta_error),
                    "delta_bias": finite_bias(delta_error),
                }
            )
    return pd.DataFrame(rows)


def summarize_target_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buckets = {"all": frame}
    work = frame.copy()
    work["distance_bucket"] = distance_bucket(work["md_since_anchor"])
    for bucket, part in work.groupby("distance_bucket", observed=True):
        buckets[str(bucket)] = part

    for bucket, part in buckets.items():
        control = part["tvt"].to_numpy(dtype=float) - part["anchor_tvt"].to_numpy(dtype=float)
        rows.append(target_summary_row("control_dTVT", bucket, control, part))
        for method in METHODS:
            absolute = part["tvt"].to_numpy(dtype=float) - part[f"{method}_hat"].to_numpy(
                dtype=float
            )
            anchor_relative = control - part[f"{method}_delta_hat"].to_numpy(dtype=float)
            rows.append(
                target_summary_row(
                    f"{method}_absolute_TVT_minus_ANCC_hat",
                    bucket,
                    absolute,
                    part,
                )
            )
            rows.append(
                target_summary_row(f"{method}_anchor_relative", bucket, anchor_relative, part)
            )
    return pd.DataFrame(rows)


def target_summary_row(
    name: str,
    bucket: str,
    values: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    abs_values = np.abs(finite)
    by_well = (
        pd.DataFrame({"well": frame["well"].to_numpy(), "value": values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("well", observed=True)["value"]
        .mean()
    )
    return {
        "target": name,
        "bucket": bucket,
        "rows": int(finite.size),
        "wells": int(frame["well"].nunique()),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "abs_p95": finite_quantile(abs_values, 0.95),
        "abs_p99": finite_quantile(abs_values, 0.99),
        "by_well_mean_std": float(by_well.std()) if len(by_well) else float("nan"),
        "by_well_abs_p95": (
            float(np.quantile(np.abs(by_well), 0.95)) if len(by_well) else float("nan")
        ),
    }


def write_csv(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return stable_content_sha(path)


def run_audit(
    *,
    train_dir: Path,
    artifacts_dir: Path,
    features_dir: Path,
    metrics_path: Path,
    config: dict[str, Any],
    debug: bool = False,
    max_wells: int | None = None,
) -> dict[str, Any]:
    seed = int(nested_get(config, "reproducibility.seed", 42))
    if debug and max_wells is None:
        max_wells = int(nested_get(config, "runtime.debug_n_wells", 40))

    wells = load_wells(train_dir, max_wells=max_wells)
    n_splits = min(int(nested_get(config, "validation.n_folds", 5)), len(wells))
    if n_splits < 2:
        raise ValueError("need at least two folds for fold-safe ANCC surface audit")

    groups = np.array([well.well for well in wells])
    dummy_x = np.zeros((len(wells), 1), dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    fold_frames: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(dummy_x, groups=groups)):
        train_wells = [wells[i] for i in train_idx]
        valid_wells = [wells[i] for i in valid_idx]
        fit = fit_surface(train_wells, config=config, fold_seed=seed + fold)
        fold_frame = build_fold_predictions(fold, valid_wells, fit, config=config)
        fold_frames.append(fold_frame)

    oof = pd.concat(fold_frames, ignore_index=True)
    method_summary = pd.DataFrame([summarize_method(oof, method) for method in METHODS])
    bucket_summary = summarize_buckets(oof)
    target_summary = summarize_target_distributions(oof)

    oof_path = features_dir / "ancc_surface_oof_predictions.csv"
    method_path = artifacts_dir / "method_metrics.csv"
    bucket_path = artifacts_dir / "bucket_metrics.csv"
    target_path = artifacts_dir / "target_distribution_summary.csv"
    sha = {
        "oof_predictions_csv": write_csv(oof, oof_path),
        "method_metrics_csv": write_csv(method_summary, method_path),
        "bucket_metrics_csv": write_csv(bucket_summary, bucket_path),
        "target_distribution_summary_csv": write_csv(target_summary, target_path),
    }

    best_row = method_summary.sort_values(["delta_rmse", "rmse"], ascending=True).iloc[0]
    experiment_name = nested_get(
        config,
        "experiment.name",
        "exp138_ancc_surface_predictability_audit",
    )
    metrics = {
        "experiment": experiment_name,
        "status": "debug_completed" if debug else "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "debug": bool(debug),
        "max_wells": max_wells,
        "n_wells": int(len(wells)),
        "n_folds": int(n_splits),
        "score_rows": int(len(oof)),
        "methods": METHODS,
        "best_by_delta_rmse": {
            "method": str(best_row["method"]),
            "rmse": float(best_row["rmse"]),
            "delta_rmse": float(best_row["delta_rmse"]),
            "well_rmse_max": float(best_row["well_rmse_max"]),
        },
        "method_metrics": method_summary.to_dict(orient="records"),
        "artifacts": {
            "oof_predictions": str(oof_path),
            "method_metrics": str(method_path),
            "bucket_metrics": str(bucket_path),
            "target_distribution_summary": str(target_path),
        },
        "sha256": sha,
        "notes": [
            "LightGBM is intentionally not used.",
            "Validation-fold ANCC is used only for scoring, not for surface fitting.",
            "Downstream target ablation is out of scope for this audit.",
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics
