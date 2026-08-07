from __future__ import annotations

import argparse
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
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
REQUIRED_COLUMNS = ("MD", "X", "Y", "Z", "TVT", "TVT_input", *FORMATION_COLUMNS)
METHOD_GLOBAL = "global_median"
METHOD_ROW_KNN = "row_knn_xy"
METHOD_WELL_PLANE = "well_plane_knn"
METHODS = (METHOD_GLOBAL, METHOD_ROW_KNN, METHOD_WELL_PLANE)
CANDIDATES = ("contact_median", "contact_prefix_weighted", "contact_best_prefix")


@dataclass(frozen=True)
class WellFrame:
    well: str
    frame: pd.DataFrame
    eval_mask: np.ndarray
    known_positions: np.ndarray
    anchor_position: int


@dataclass(frozen=True)
class SurfaceFit:
    global_median: np.ndarray
    contact_refs: np.ndarray
    row_knn: KNeighborsRegressor | None
    row_xy_mean: np.ndarray | None
    row_xy_scale: np.ndarray | None
    plane_model: MultiWellPlaneKNN


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


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


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
    if known_positions.size < 3 or not eval_mask.any():
        return None

    anchor_position = int(known_positions[-1])
    anchor_values = frame.iloc[anchor_position][["MD", "X", "Y", "Z", "TVT_input"]]
    if not np.isfinite(anchor_values.to_numpy(dtype=float)).all():
        return None
    if frame.loc[eval_mask, ["TVT", *FORMATION_COLUMNS]].notna().sum().min() == 0:
        return None

    return WellFrame(
        well=well_id_from_path(path),
        frame=frame,
        eval_mask=eval_mask,
        known_positions=known_positions.astype(np.int32),
        anchor_position=anchor_position,
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


def stack_valid_surface_rows(
    wells: list[WellFrame],
    *,
    rng: np.random.Generator,
    max_rows_per_well: int,
    max_rows_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy_parts: list[np.ndarray] = []
    surface_parts: list[np.ndarray] = []

    for well in wells:
        frame = well.frame
        xy = frame.loc[:, ["X", "Y"]].to_numpy(dtype=float)
        surfaces = frame.loc[:, list(FORMATION_COLUMNS)].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(surfaces).all(axis=1)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            continue
        if indices.size > max_rows_per_well:
            indices = rng.choice(indices, size=max_rows_per_well, replace=False)
        xy_parts.append(xy[indices])
        surface_parts.append(surfaces[indices])

    if not xy_parts:
        return np.empty((0, 2), dtype=float), np.empty((0, len(FORMATION_COLUMNS)), dtype=float)

    xy_all = np.vstack(xy_parts)
    surface_all = np.vstack(surface_parts)
    if xy_all.shape[0] > max_rows_total:
        selected = rng.choice(xy_all.shape[0], size=max_rows_total, replace=False)
        xy_all = xy_all[selected]
        surface_all = surface_all[selected]
    return xy_all, surface_all


def compute_contact_refs(wells: list[WellFrame]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for well in wells:
        frame = well.frame
        values = frame[["TVT", "Z", *FORMATION_COLUMNS]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            continue
        tvt = values[valid, 0]
        z = values[valid, 1]
        surfaces = values[valid, 2:]
        parts.append(tvt.reshape(-1, 1) + z.reshape(-1, 1) - surfaces)
    if not parts:
        return np.zeros(len(FORMATION_COLUMNS), dtype=float)
    return np.nanmedian(np.vstack(parts), axis=0).astype(float)


class MultiWellPlaneKNN:
    def __init__(self, wells: list[WellFrame]) -> None:
        rows: list[dict[str, float | str]] = []
        for well in wells:
            frame = well.frame
            valid = (
                frame[["X", "Y", *FORMATION_COLUMNS]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if valid.empty:
                continue
            row: dict[str, float | str] = {
                "well": well.well,
                "x": float(valid["X"].median()),
                "y": float(valid["Y"].median()),
            }
            for column in FORMATION_COLUMNS:
                row[column] = float(valid[column].median())
            rows.append(row)

        self.summary = pd.DataFrame(rows)
        if self.summary.empty:
            self.xy = np.empty((0, 2), dtype=float)
            self.values = np.empty((0, len(FORMATION_COLUMNS)), dtype=float)
            self.xy_scale = np.ones(2, dtype=float)
            return

        self.xy = self.summary[["x", "y"]].to_numpy(dtype=float)
        self.values = self.summary.loc[:, list(FORMATION_COLUMNS)].to_numpy(dtype=float)
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
        n_targets = len(FORMATION_COLUMNS)
        if self.xy.shape[0] == 0:
            return (
                np.full((n_query, n_targets), np.nan, dtype=float),
                np.full(n_query, np.nan, dtype=float),
            )

        k_eff = min(max(1, int(k)), int(self.xy.shape[0]))
        train_scaled = self.xy / self.xy_scale
        pred = np.full((n_query, n_targets), np.nan, dtype=float)
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
            rhs = np.einsum("nki,nk,nkm->nim", design, weight, value)
            lhs[:, 0, 0] += 1e-9
            lhs[:, 1, 1] += 1e-9
            lhs[:, 2, 2] += 1e-9
            try:
                coef = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coef = np.empty((end - start, 3, n_targets), dtype=float)
                for i in range(end - start):
                    coef[i] = np.linalg.pinv(lhs[i]) @ rhs[i]

            query_design = np.column_stack([xy_query[start:end], np.ones(end - start)])
            pred[start:end] = np.einsum("ni,nim->nm", query_design, coef)
        return pred, min_dist


def fit_surface(
    train_wells: list[WellFrame],
    *,
    config: dict[str, Any],
    fold_seed: int,
) -> SurfaceFit:
    surface_values = np.vstack(
        [
            well.frame.loc[:, list(FORMATION_COLUMNS)].to_numpy(dtype=float)
            for well in train_wells
        ]
    )
    global_median = np.nanmedian(surface_values, axis=0).astype(float)
    contact_refs = compute_contact_refs(train_wells)

    rng = np.random.default_rng(fold_seed)
    xy, surfaces = stack_valid_surface_rows(
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
        row_knn.fit(xy_scaled, surfaces)

    return SurfaceFit(
        global_median=global_median,
        contact_refs=contact_refs,
        row_knn=row_knn,
        row_xy_mean=xy_mean,
        row_xy_scale=xy_scale,
        plane_model=MultiWellPlaneKNN(train_wells),
    )


def predict_surface(
    fit: SurfaceFit,
    xy: np.ndarray,
    *,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n_rows = int(xy.shape[0])
    n_targets = len(FORMATION_COLUMNS)
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        METHOD_GLOBAL: (
            np.tile(fit.global_median.reshape(1, -1), (n_rows, 1)),
            np.full(n_rows, np.nan, dtype=float),
        )
    }

    if fit.row_knn is not None and fit.row_xy_mean is not None and fit.row_xy_scale is not None:
        scaled = (xy - fit.row_xy_mean) / fit.row_xy_scale
        surface = fit.row_knn.predict(scaled).astype(float)
        try:
            dist, _ = fit.row_knn.kneighbors(scaled, n_neighbors=1)
            min_dist = dist[:, 0].astype(float)
        except Exception:
            min_dist = np.full(n_rows, np.nan, dtype=float)
        predictions[METHOD_ROW_KNN] = (surface, min_dist)
    else:
        predictions[METHOD_ROW_KNN] = (
            np.full((n_rows, n_targets), np.nan, dtype=float),
            np.full(n_rows, np.nan, dtype=float),
        )

    predictions[METHOD_WELL_PLANE] = fit.plane_model.predict(
        xy,
        k=int(nested_get(config, "surface.well_plane_knn.n_neighbors", 16)),
        chunk_size=int(nested_get(config, "surface.well_plane_knn.chunk_size", 50000)),
    )
    return predictions


def nanweighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights.reshape(1, -1))
    weighted = np.where(valid, values * weights.reshape(1, -1), 0.0)
    weight_sum = np.where(valid, weights.reshape(1, -1), 0.0).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    mask = weight_sum > 0
    out[mask] = weighted[mask].sum(axis=1) / weight_sum[mask]
    return out


def prefix_calibrate_physical_candidates(
    *,
    fit: SurfaceFit,
    known_z: np.ndarray,
    known_tvt: np.ndarray,
    eval_z: np.ndarray,
    known_surfaces: np.ndarray,
    eval_surfaces: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_known = fit.contact_refs.reshape(1, -1) - (
        known_z.reshape(-1, 1) - known_surfaces
    )
    raw_eval = fit.contact_refs.reshape(1, -1) - (eval_z.reshape(-1, 1) - eval_surfaces)
    offset = np.nanmedian(known_tvt.reshape(-1, 1) - raw_known, axis=0)
    prefix_pred = raw_known + offset.reshape(1, -1)
    eval_by_formation = raw_eval + offset.reshape(1, -1)
    prefix_error = prefix_pred - known_tvt.reshape(-1, 1)
    prefix_mae = np.nanmean(np.abs(prefix_error), axis=0)
    prefix_rmse = np.sqrt(np.nanmean(prefix_error**2, axis=0))

    if np.isfinite(prefix_mae).any():
        best_index = int(np.nanargmin(prefix_mae))
    else:
        best_index = 0
    weights = 1.0 / np.square(np.nan_to_num(prefix_mae, nan=np.nanmedian(prefix_mae)) + 5.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    candidates = {
        "contact_median": np.nanmedian(eval_by_formation, axis=1),
        "contact_prefix_weighted": nanweighted_average(eval_by_formation, weights),
        "contact_best_prefix": eval_by_formation[:, best_index],
    }
    diagnostics = {
        "best_formation": FORMATION_COLUMNS[best_index],
        "best_formation_index": best_index,
        "prefix_mae_best": float(prefix_mae[best_index])
        if np.isfinite(prefix_mae[best_index])
        else float("nan"),
        "prefix_rmse_best": float(prefix_rmse[best_index])
        if np.isfinite(prefix_rmse[best_index])
        else float("nan"),
        "prefix_mae_median": float(np.nanmedian(prefix_mae)),
        "prefix_rmse_median": float(np.nanmedian(prefix_rmse)),
        "formation_weight_sum": float(np.nansum(weights)),
        "offsets": offset,
        "prefix_mae": prefix_mae,
        "prefix_rmse": prefix_rmse,
        "eval_by_formation": eval_by_formation,
    }
    return candidates, diagnostics


def build_fold_predictions(
    fold: int,
    valid_wells: list[WellFrame],
    fit: SurfaceFit,
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    for well in valid_wells:
        frame = well.frame
        raw_eval_positions = np.flatnonzero(well.eval_mask)
        eval_required = frame.iloc[raw_eval_positions][
            ["MD", "X", "Y", "Z", "TVT", *FORMATION_COLUMNS]
        ].to_numpy(dtype=float)
        valid_eval = np.isfinite(eval_required).all(axis=1)
        eval_positions = raw_eval_positions[valid_eval]
        if eval_positions.size == 0:
            continue

        known_positions = well.known_positions
        known_required = frame.iloc[known_positions][["X", "Y", "Z", "TVT_input"]].to_numpy(
            dtype=float
        )
        valid_known = np.isfinite(known_required).all(axis=1)
        known_positions = known_positions[valid_known]
        if known_positions.size < 3:
            continue

        eval_frame = frame.iloc[eval_positions].copy()
        known_frame = frame.iloc[known_positions].copy()
        xy_known = known_frame[["X", "Y"]].to_numpy(dtype=float)
        xy_eval = eval_frame[["X", "Y"]].to_numpy(dtype=float)
        pred_known = predict_surface(fit, xy_known, config=config)
        pred_eval = predict_surface(fit, xy_eval, config=config)

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
            }
        )
        out["md_since_anchor"] = out["md"] - out["anchor_md"]
        known_z = known_frame["Z"].to_numpy(dtype=float)
        known_tvt = known_frame["TVT_input"].to_numpy(dtype=float)
        eval_z = eval_frame["Z"].to_numpy(dtype=float)
        true_surfaces = eval_frame.loc[:, list(FORMATION_COLUMNS)].to_numpy(dtype=float)

        for method in METHODS:
            known_surfaces, known_distance = pred_known[method]
            eval_surfaces, eval_distance = pred_eval[method]
            candidates, diagnostics = prefix_calibrate_physical_candidates(
                fit=fit,
                known_z=known_z,
                known_tvt=known_tvt,
                eval_z=eval_z,
                known_surfaces=known_surfaces,
                eval_surfaces=eval_surfaces,
            )
            surface_error = eval_surfaces - true_surfaces
            surface_abs_error = np.abs(surface_error)
            surface_span = np.nanmax(eval_surfaces, axis=1) - np.nanmin(eval_surfaces, axis=1)
            formation_pred_spread = np.nanmax(
                diagnostics["eval_by_formation"], axis=1
            ) - np.nanmin(diagnostics["eval_by_formation"], axis=1)

            out[f"{method}_neighbor_dist"] = eval_distance
            out[f"{method}_known_neighbor_dist_median"] = finite_median(known_distance)
            out[f"{method}_surface_abs_error_mean"] = np.nanmean(surface_abs_error, axis=1)
            out[f"{method}_surface_abs_error_min"] = np.nanmin(surface_abs_error, axis=1)
            out[f"{method}_surface_span"] = surface_span
            out[f"{method}_formation_pred_spread"] = formation_pred_spread
            out[f"{method}_best_formation_index"] = int(diagnostics["best_formation_index"])
            out[f"{method}_prefix_mae_best"] = float(diagnostics["prefix_mae_best"])
            out[f"{method}_prefix_rmse_best"] = float(diagnostics["prefix_rmse_best"])
            out[f"{method}_prefix_mae_median"] = float(diagnostics["prefix_mae_median"])
            out[f"{method}_prefix_rmse_median"] = float(diagnostics["prefix_rmse_median"])
            for candidate, values in candidates.items():
                out[f"{method}_{candidate}_tvt"] = values
                out[f"{method}_{candidate}_error"] = values - out["tvt"].to_numpy(dtype=float)

            for column_index, column in enumerate(FORMATION_COLUMNS):
                calibration_rows.append(
                    {
                        "fold": fold,
                        "well": well.well,
                        "method": method,
                        "formation": column,
                        "prefix_rows": int(len(known_positions)),
                        "offset": float(diagnostics["offsets"][column_index]),
                        "prefix_mae": float(diagnostics["prefix_mae"][column_index]),
                        "prefix_rmse": float(diagnostics["prefix_rmse"][column_index]),
                        "known_neighbor_dist_median": finite_median(known_distance),
                    }
                )

        row_frames.append(out)

    if not row_frames:
        return pd.DataFrame(), pd.DataFrame(calibration_rows)
    return pd.concat(row_frames, ignore_index=True), pd.DataFrame(calibration_rows)


def distance_bucket(md_since: pd.Series) -> pd.Series:
    bins = [-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf]
    labels = ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]
    return pd.cut(md_since.astype(float), bins=bins, labels=labels, right=False).astype(str)


def candidate_columns() -> list[tuple[str, str, str]]:
    return [
        (method, candidate, f"{method}_{candidate}_error")
        for method in METHODS
        for candidate in CANDIDATES
    ]


def summarize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, candidate, column in candidate_columns():
        error = frame[column].to_numpy(dtype=float)
        by_well = (
            frame.assign(sq_error=error**2)
            .groupby("well", observed=True)["sq_error"]
            .mean()
            .pipe(np.sqrt)
        )
        rows.append(
            {
                "method": method,
                "candidate": candidate,
                "rows": int(np.isfinite(error).sum()),
                "wells": int(frame["well"].nunique()),
                "rmse": finite_rmse(error),
                "mae": finite_mae(error),
                "bias": finite_bias(error),
                "abs_error_p95": finite_quantile(np.abs(error), 0.95),
                "abs_error_p99": finite_quantile(np.abs(error), 0.99),
                "well_rmse_mean": float(by_well.mean()) if len(by_well) else float("nan"),
                "well_rmse_p95": float(by_well.quantile(0.95)) if len(by_well) else float("nan"),
                "well_rmse_max": float(by_well.max()) if len(by_well) else float("nan"),
                "prefix_mae_best_median": float(
                    np.nanmedian(frame[f"{method}_prefix_mae_best"].to_numpy(dtype=float))
                ),
                "formation_pred_spread_median": float(
                    np.nanmedian(frame[f"{method}_formation_pred_spread"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_distance_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["distance_bucket"] = distance_bucket(work["md_since_anchor"])
    rows: list[dict[str, Any]] = []
    for method, candidate, column in candidate_columns():
        for bucket, part in work.groupby("distance_bucket", observed=True):
            error = part[column].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "candidate": candidate,
                    "bucket": str(bucket),
                    "rows": int(len(part)),
                    "wells": int(part["well"].nunique()),
                    "rmse": finite_rmse(error),
                    "mae": finite_mae(error),
                    "bias": finite_bias(error),
                    "prefix_mae_best_median": float(
                        np.nanmedian(part[f"{method}_prefix_mae_best"].to_numpy(dtype=float))
                    ),
                    "neighbor_dist_median": finite_median(
                        part[f"{method}_neighbor_dist"].to_numpy(dtype=float)
                    ),
                    "formation_pred_spread_median": float(
                        np.nanmedian(
                            part[f"{method}_formation_pred_spread"].to_numpy(dtype=float)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_confidence_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, candidate, column in candidate_columns():
        work = frame.copy()
        prefix = work[f"{method}_prefix_mae_best"].to_numpy(dtype=float)
        spread = work[f"{method}_formation_pred_spread"].to_numpy(dtype=float)
        neighbor = work[f"{method}_neighbor_dist"].to_numpy(dtype=float)
        for name, values in (
            ("prefix_mae_best", prefix),
            ("formation_pred_spread", spread),
            ("neighbor_dist", neighbor),
        ):
            finite = values[np.isfinite(values)]
            if finite.size < 4:
                continue
            edges = np.unique(np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0]))
            if edges.size < 3:
                continue
            work["confidence_bucket"] = pd.cut(
                values,
                bins=edges,
                include_lowest=True,
                duplicates="drop",
            ).astype(str)
            for bucket, part in work.groupby("confidence_bucket", observed=True):
                if bucket == "nan":
                    continue
                error = part[column].to_numpy(dtype=float)
                feature_column = f"{method}_{name}"
                rows.append(
                    {
                        "method": method,
                        "candidate": candidate,
                        "confidence_feature": name,
                        "bucket": str(bucket),
                        "rows": int(len(part)),
                        "wells": int(part["well"].nunique()),
                        "rmse": finite_rmse(error),
                        "mae": finite_mae(error),
                        "bias": finite_bias(error),
                        "feature_median": float(
                            np.nanmedian(part[feature_column].to_numpy(dtype=float))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_surface_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        mean_error = frame[f"{method}_surface_abs_error_mean"].to_numpy(dtype=float)
        min_error = frame[f"{method}_surface_abs_error_min"].to_numpy(dtype=float)
        rows.append(
            {
                "method": method,
                "rows": int(len(frame)),
                "wells": int(frame["well"].nunique()),
                "mean_surface_abs_error_mean": float(np.nanmean(mean_error)),
                "mean_surface_abs_error_p95": finite_quantile(mean_error, 0.95),
                "min_surface_abs_error_mean": float(np.nanmean(min_error)),
                "min_surface_abs_error_p95": finite_quantile(min_error, 0.95),
                "surface_span_median": float(
                    np.nanmedian(frame[f"{method}_surface_span"].to_numpy(dtype=float))
                ),
            }
        )
    return pd.DataFrame(rows)


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
        raise ValueError("need at least two folds for fold-safe formation physical audit")

    groups = np.array([well.well for well in wells])
    dummy_x = np.zeros((len(wells), 1), dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    fold_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(dummy_x, groups=groups)):
        train_wells = [wells[i] for i in train_idx]
        valid_wells = [wells[i] for i in valid_idx]
        fit = fit_surface(train_wells, config=config, fold_seed=seed + fold)
        fold_frame, fold_calibration = build_fold_predictions(
            fold,
            valid_wells,
            fit,
            config=config,
        )
        fold_frames.append(fold_frame)
        calibration_frames.append(fold_calibration)

    oof = pd.concat(fold_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    candidate_summary = summarize_candidates(oof)
    bucket_summary = summarize_distance_buckets(oof)
    confidence_summary = summarize_confidence_buckets(oof)
    surface_summary = summarize_surface_metrics(oof)

    feature_path = features_dir / "formation_physical_oof_features.csv"
    calibration_path = artifacts_dir / "formation_prefix_calibration.csv"
    candidate_path = artifacts_dir / "candidate_metrics.csv"
    bucket_path = artifacts_dir / "distance_bucket_metrics.csv"
    confidence_path = artifacts_dir / "confidence_bucket_metrics.csv"
    surface_path = artifacts_dir / "surface_proxy_metrics.csv"
    sha = {
        "formation_physical_oof_features_csv": write_csv(oof, feature_path),
        "formation_prefix_calibration_csv": write_csv(calibration, calibration_path),
        "candidate_metrics_csv": write_csv(candidate_summary, candidate_path),
        "distance_bucket_metrics_csv": write_csv(bucket_summary, bucket_path),
        "confidence_bucket_metrics_csv": write_csv(confidence_summary, confidence_path),
        "surface_proxy_metrics_csv": write_csv(surface_summary, surface_path),
    }

    best_row = candidate_summary.sort_values(["rmse", "well_rmse_max"], ascending=True).iloc[0]
    experiment_name = nested_get(
        config,
        "experiment.name",
        "exp150_formation_physical_imputer_revisit",
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
        "formation_columns": FORMATION_COLUMNS,
        "methods": METHODS,
        "candidates": CANDIDATES,
        "best_candidate": {
            "method": str(best_row["method"]),
            "candidate": str(best_row["candidate"]),
            "rmse": float(best_row["rmse"]),
            "mae": float(best_row["mae"]),
            "well_rmse_max": float(best_row["well_rmse_max"]),
            "prefix_mae_best_median": float(best_row["prefix_mae_best_median"]),
        },
        "candidate_metrics": candidate_summary.to_dict(orient="records"),
        "artifacts": {
            "formation_physical_oof_features": str(feature_path),
            "formation_prefix_calibration": str(calibration_path),
            "candidate_metrics": str(candidate_path),
            "distance_bucket_metrics": str(bucket_path),
            "confidence_bucket_metrics": str(confidence_path),
            "surface_proxy_metrics": str(surface_path),
        },
        "sha256": sha,
        "notes": [
            "No LightGBM, PF/Beam, or GPU training is used.",
            "Validation-fold formation columns and TVT are used only for scoring.",
            "Prediction-time inputs are X/Y/Z and known-prefix TVT_input calibration.",
            "Physical contact output is an audit candidate and add-only feature source.",
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formation physical imputer audit.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--metrics-path", type=Path, default=Path("metrics.json"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-wells", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import yaml

    args = parse_args()
    config: dict[str, Any] = {}
    if args.config is not None and args.config.exists():
        with args.config.open() as fp:
            config = yaml.safe_load(fp) or {}
    metrics = run_audit(
        train_dir=args.train_dir,
        artifacts_dir=args.artifacts_dir,
        features_dir=args.features_dir,
        metrics_path=args.metrics_path,
        config=config,
        debug=args.debug,
        max_wells=args.max_wells,
    )
    print(json.dumps(metrics["best_candidate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
