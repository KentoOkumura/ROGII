from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


@dataclass(frozen=True)
class CurveRecord:
    label: str
    split: str
    well_id: str
    kind: str
    view: str
    path: str
    n_rows: int
    n_valid_points: int
    grid_valid_points: int
    tvt_min: float
    tvt_max: float
    gr_mean: float
    gr_std: float


@dataclass(frozen=True)
class RawCurve:
    label: str
    split: str
    well_id: str
    kind: str
    view: str
    tvt: np.ndarray
    gr: np.ndarray


def finite_float(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def well_id_from_path(path: Path, suffix: str) -> str:
    return path.name.removesuffix(suffix)


def curve_stats(
    *,
    label: str,
    split: str,
    well_id: str,
    kind: str,
    view: str,
    path: str,
    tvt: np.ndarray,
    gr: np.ndarray,
    grid_values: np.ndarray,
) -> CurveRecord:
    valid = np.isfinite(tvt) & np.isfinite(gr)
    grid_valid = np.isfinite(grid_values)
    if valid.any():
        tvt_min = float(np.nanmin(tvt[valid]))
        tvt_max = float(np.nanmax(tvt[valid]))
        gr_mean = float(np.nanmean(gr[valid]))
        gr_std = float(np.nanstd(gr[valid]))
    else:
        tvt_min = tvt_max = gr_mean = gr_std = float("nan")
    return CurveRecord(
        label=label,
        split=split,
        well_id=well_id,
        kind=kind,
        view=view,
        path=path,
        n_rows=int(len(tvt)),
        n_valid_points=int(valid.sum()),
        grid_valid_points=int(grid_valid.sum()),
        tvt_min=tvt_min,
        tvt_max=tvt_max,
        gr_mean=gr_mean,
        gr_std=gr_std,
    )


def collect_global_tvt_bounds(raw_dir: Path) -> tuple[float, float]:
    mins: list[float] = []
    maxs: list[float] = []

    for split in ("train", "test"):
        split_dir = raw_dir / split
        for path in sorted(split_dir.glob(f"*{TYPEWELL_SUFFIX}")):
            tvt = finite_float(pd.read_csv(path, usecols=["TVT"])["TVT"])
            tvt = tvt[np.isfinite(tvt)]
            if tvt.size:
                mins.append(float(tvt.min()))
                maxs.append(float(tvt.max()))

        for path in sorted(split_dir.glob(f"*{HORIZONTAL_SUFFIX}")):
            usecols = ["TVT_input"]
            if split == "train":
                usecols.append("TVT")
            frame = pd.read_csv(path, usecols=usecols)
            for column in usecols:
                tvt = finite_float(frame[column])
                tvt = tvt[np.isfinite(tvt)]
                if tvt.size:
                    mins.append(float(tvt.min()))
                    maxs.append(float(tvt.max()))

    if not mins:
        raise FileNotFoundError(f"No finite TVT values found under {raw_dir}")
    return min(mins), max(maxs)


def points_to_grid(
    tvt: np.ndarray,
    gr: np.ndarray,
    *,
    grid_min: float,
    grid_step: float,
    grid_size: int,
) -> np.ndarray:
    valid = np.isfinite(tvt) & np.isfinite(gr)
    values = np.full(grid_size, np.nan, dtype=np.float32)
    if valid.sum() < 2:
        return values

    idx = np.rint((tvt[valid] - grid_min) / grid_step).astype(np.int64)
    vals = gr[valid].astype(np.float64, copy=False)
    in_range = (idx >= 0) & (idx < grid_size) & np.isfinite(vals)
    idx = idx[in_range]
    vals = vals[in_range]
    if idx.size == 0:
        return values

    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    vals = vals[order]
    unique, start, counts = np.unique(idx, return_index=True, return_counts=True)
    sums = np.add.reduceat(vals, start)
    values[unique] = (sums / counts).astype(np.float32)
    return values


def load_typewell_curves(
    raw_dir: Path,
    *,
    grid_min: float,
    grid_step: float,
    grid_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, RawCurve]]:
    records: list[CurveRecord] = []
    values: list[np.ndarray] = []
    raw: dict[str, RawCurve] = {}

    for split in ("train", "test"):
        for path in sorted((raw_dir / split).glob(f"*{TYPEWELL_SUFFIX}")):
            well_id = well_id_from_path(path, TYPEWELL_SUFFIX)
            frame = pd.read_csv(path, usecols=["TVT", "GR"])
            tvt = finite_float(frame["TVT"])
            gr = finite_float(frame["GR"])
            label = f"typewell:{split}:{well_id}"
            grid_values = points_to_grid(
                tvt, gr, grid_min=grid_min, grid_step=grid_step, grid_size=grid_size
            )
            records.append(
                curve_stats(
                    label=label,
                    split=split,
                    well_id=well_id,
                    kind="typewell",
                    view="native",
                    path=str(path),
                    tvt=tvt,
                    gr=gr,
                    grid_values=grid_values,
                )
            )
            values.append(grid_values)
            raw[label] = RawCurve(
                label=label,
                split=split,
                well_id=well_id,
                kind="typewell",
                view="native",
                tvt=tvt,
                gr=gr,
            )

    if not values:
        raise FileNotFoundError(f"No typewell files found under {raw_dir}")
    return pd.DataFrame([record.__dict__ for record in records]), np.vstack(values), raw


def horizontal_views(split: str, frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    gr = finite_float(frame["GR"])
    tvt_input = finite_float(frame["TVT_input"])
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    prefix = np.isfinite(tvt_input) & np.isfinite(gr)
    out["prefix"] = (tvt_input[prefix], gr[prefix])

    if split == "train":
        tvt = finite_float(frame["TVT"])
        hidden = ~np.isfinite(tvt_input) & np.isfinite(tvt) & np.isfinite(gr)
        full = np.isfinite(tvt) & np.isfinite(gr)
        out["hidden"] = (tvt[hidden], gr[hidden])
        out["full"] = (tvt[full], gr[full])

    return out


def load_horizontal_query_curves(
    raw_dir: Path,
    *,
    grid_min: float,
    grid_step: float,
    grid_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, RawCurve]]:
    records: list[CurveRecord] = []
    values: list[np.ndarray] = []
    raw: dict[str, RawCurve] = {}

    for split in ("train", "test"):
        for path in sorted((raw_dir / split).glob(f"*{HORIZONTAL_SUFFIX}")):
            well_id = well_id_from_path(path, HORIZONTAL_SUFFIX)
            frame = pd.read_csv(path)
            for view, (tvt, gr) in horizontal_views(split, frame).items():
                label = f"horizontal:{split}:{well_id}:{view}"
                grid_values = points_to_grid(
                    tvt, gr, grid_min=grid_min, grid_step=grid_step, grid_size=grid_size
                )
                records.append(
                    curve_stats(
                        label=label,
                        split=split,
                        well_id=well_id,
                        kind="horizontal",
                        view=view,
                        path=str(path),
                        tvt=tvt,
                        gr=gr,
                        grid_values=grid_values,
                    )
                )
                values.append(grid_values)
                raw[label] = RawCurve(
                    label=label,
                    split=split,
                    well_id=well_id,
                    kind="horizontal",
                    view=view,
                    tvt=tvt,
                    gr=gr,
                )

    if not values:
        raise FileNotFoundError(f"No horizontal files found under {raw_dir}")
    return pd.DataFrame([record.__dict__ for record in records]), np.vstack(values), raw


def pairwise_grid_metrics(
    query_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    block_size: int,
) -> dict[str, np.ndarray]:
    q_valid = np.isfinite(query_values)
    c_valid = np.isfinite(candidate_values)
    q = np.where(q_valid, query_values, 0.0).astype(np.float32, copy=False)
    c = np.where(c_valid, candidate_values, 0.0).astype(np.float32, copy=False)
    q2 = q * q
    c2 = c * c
    mq = q_valid.astype(np.float32)
    mc = c_valid.astype(np.float32)

    n_queries = q.shape[0]
    n_candidates = c.shape[0]
    out_shape = (n_queries, n_candidates)
    n = np.zeros(out_shape, dtype=np.float32)
    coverage_q = np.zeros(out_shape, dtype=np.float32)
    coverage_c = np.zeros(out_shape, dtype=np.float32)
    corr = np.full(out_shape, np.nan, dtype=np.float32)
    rmse = np.full(out_shape, np.nan, dtype=np.float32)
    resid_std = np.full(out_shape, np.nan, dtype=np.float32)
    z_rmse = np.full(out_shape, np.nan, dtype=np.float32)

    mc_t = mc.T
    c_t = c.T
    c2_t = c2.T
    q_counts = mq.sum(axis=1)
    c_counts = mc.sum(axis=1)

    for start in range(0, n_queries, block_size):
        stop = min(start + block_size, n_queries)
        qb = q[start:stop]
        q2b = q2[start:stop]
        mqb = mq[start:stop]

        nb = mqb @ mc_t
        sum_q = qb @ mc_t
        sum_c = mqb @ c_t
        sum_q2 = q2b @ mc_t
        sum_c2 = mqb @ c2_t
        sum_qc = qb @ c_t

        with np.errstate(divide="ignore", invalid="ignore"):
            diff2_sum = np.maximum(sum_q2 + sum_c2 - 2.0 * sum_qc, 0.0)
            mean_diff = (sum_q - sum_c) / nb
            rmse_b = np.sqrt(diff2_sum / nb)
            resid_b = np.sqrt(np.maximum(diff2_sum / nb - mean_diff * mean_diff, 0.0))

            cov_q_b = nb / q_counts[start:stop, None]
            cov_c_b = nb / c_counts[None, :]

            q_centered = sum_q2 - (sum_q * sum_q) / nb
            c_centered = sum_c2 - (sum_c * sum_c) / nb
            numerator = sum_qc - (sum_q * sum_c) / nb
            denom = np.sqrt(q_centered * c_centered)
            corr_b = numerator / denom
            corr_b = np.clip(corr_b, -1.0, 1.0)
            z_b = np.sqrt(np.maximum(2.0 - 2.0 * corr_b, 0.0))

        invalid = (nb < 3) | (q_centered <= 1e-8) | (c_centered <= 1e-8)
        corr_b[invalid] = np.nan
        z_b[invalid] = np.nan
        rmse_b[nb < 1] = np.nan
        resid_b[nb < 2] = np.nan

        n[start:stop] = nb.astype(np.float32)
        coverage_q[start:stop] = cov_q_b.astype(np.float32)
        coverage_c[start:stop] = cov_c_b.astype(np.float32)
        corr[start:stop] = corr_b.astype(np.float32)
        rmse[start:stop] = rmse_b.astype(np.float32)
        resid_std[start:stop] = resid_b.astype(np.float32)
        z_rmse[start:stop] = z_b.astype(np.float32)

    return {
        "overlap_grid_points": n,
        "coverage_query": coverage_q,
        "coverage_candidate": coverage_c,
        "corr": corr,
        "raw_rmse": rmse,
        "resid_std": resid_std,
        "z_rmse": z_rmse,
    }


def sorted_candidate_indices(
    z_rmse: np.ndarray,
    coverage: np.ndarray,
    raw_rmse: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return idx
    order = np.lexsort((raw_rmse[idx], -coverage[idx], z_rmse[idx]))
    return idx[order]


def metric_at(metrics: dict[str, np.ndarray], query_idx: int, candidate_idx: int) -> dict[str, Any]:
    return {
        key: (
            float(value[query_idx, candidate_idx])
            if np.isfinite(value[query_idx, candidate_idx])
            else None
        )
        for key, value in metrics.items()
    }


def build_pair_summaries(
    query_index: pd.DataFrame,
    candidate_index: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    *,
    min_overlap_points: int,
    min_coverage_query: float,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_label_to_idx = {
        label: idx for idx, label in enumerate(candidate_index["label"].astype(str))
    }
    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    z = metrics["z_rmse"]
    raw = metrics["raw_rmse"]
    cov = metrics["coverage_query"]
    n = metrics["overlap_grid_points"]

    for qi, query in query_index.reset_index(drop=True).iterrows():
        query_label = str(query["label"])
        split = str(query["split"])
        well_id = str(query["well_id"])
        view = str(query["view"])
        provided_label = f"typewell:{split}:{well_id}"
        provided_idx = candidate_label_to_idx.get(provided_label)

        valid = (
            (n[qi] >= float(min_overlap_points))
            & (cov[qi] >= float(min_coverage_query))
            & np.isfinite(z[qi])
        )
        order = sorted_candidate_indices(z[qi], cov[qi], raw[qi], valid)
        best_idx = int(order[0]) if order.size else None

        provided_rank = None
        if provided_idx is not None and valid[provided_idx] and order.size:
            matches = np.flatnonzero(order == provided_idx)
            if matches.size:
                provided_rank = int(matches[0] + 1)

        base = {
            "query_label": query_label,
            "split": split,
            "well_id": well_id,
            "view": view,
            "query_grid_valid_points": int(query["grid_valid_points"]),
            "provided_label": provided_label if provided_idx is not None else None,
            "provided_in_primary_pool": bool(
                provided_idx is not None and bool(valid[provided_idx])
            ),
            "provided_rank_by_z_rmse": provided_rank,
            "primary_pool_candidates": int(valid.sum()),
        }
        if provided_idx is not None:
            base.update(
                {f"provided_{k}": v for k, v in metric_at(metrics, qi, provided_idx).items()}
            )
        else:
            base.update({f"provided_{k}": None for k in metrics})

        if best_idx is not None:
            best = candidate_index.iloc[best_idx]
            base.update(
                {
                    "best_label": str(best["label"]),
                    "best_split": str(best["split"]),
                    "best_well_id": str(best["well_id"]),
                    "best_is_provided": bool(best_idx == provided_idx),
                    **{f"best_{k}": v for k, v in metric_at(metrics, qi, best_idx).items()},
                }
            )
        else:
            base.update(
                {
                    "best_label": None,
                    "best_split": None,
                    "best_well_id": None,
                    "best_is_provided": False,
                    **{f"best_{k}": None for k in metrics},
                }
            )
        if base.get("provided_z_rmse") is not None and base.get("best_z_rmse") is not None:
            base["best_minus_provided_z_rmse"] = float(base["best_z_rmse"]) - float(
                base["provided_z_rmse"]
            )
        else:
            base["best_minus_provided_z_rmse"] = None
        if base.get("provided_raw_rmse") is not None and base.get("best_raw_rmse") is not None:
            base["best_minus_provided_raw_rmse"] = float(base["best_raw_rmse"]) - float(
                base["provided_raw_rmse"]
            )
        else:
            base["best_minus_provided_raw_rmse"] = None
        rows.append(base)

        for rank, ci in enumerate(order[:top_k], start=1):
            cand = candidate_index.iloc[int(ci)]
            top_rows.append(
                {
                    "query_label": query_label,
                    "split": split,
                    "well_id": well_id,
                    "view": view,
                    "rank_by_z_rmse": rank,
                    "candidate_label": str(cand["label"]),
                    "candidate_split": str(cand["split"]),
                    "candidate_well_id": str(cand["well_id"]),
                    "candidate_is_provided": bool(int(ci) == provided_idx),
                    **metric_at(metrics, qi, int(ci)),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def aggregate_duplicate_tvt(tvt: np.ndarray, gr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(tvt) & np.isfinite(gr)
    if valid.sum() < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    frame = pd.DataFrame({"tvt": tvt[valid], "gr": gr[valid]})
    grouped = frame.groupby("tvt", sort=True, as_index=False)["gr"].mean()
    return grouped["tvt"].to_numpy(np.float64), grouped["gr"].to_numpy(np.float64)


def exact_curve_metrics(query: RawCurve, candidate: RawCurve) -> dict[str, Any]:
    q_tvt, q_gr = aggregate_duplicate_tvt(query.tvt, query.gr)
    c_tvt, c_gr = aggregate_duplicate_tvt(candidate.tvt, candidate.gr)
    if len(q_tvt) < 2 or len(c_tvt) < 2:
        return {
            "exact_rows": 0,
            "exact_coverage_query": 0.0,
            "exact_corr": None,
            "exact_raw_rmse": None,
            "exact_raw_mae": None,
            "exact_resid_std": None,
            "exact_z_rmse": None,
        }

    in_range = (q_tvt >= c_tvt[0]) & (q_tvt <= c_tvt[-1])
    q_tvt = q_tvt[in_range]
    q_gr = q_gr[in_range]
    if len(q_tvt) < 2:
        return {
            "exact_rows": int(len(q_tvt)),
            "exact_coverage_query": float(len(q_tvt) / max(len(query.tvt), 1)),
            "exact_corr": None,
            "exact_raw_rmse": None,
            "exact_raw_mae": None,
            "exact_resid_std": None,
            "exact_z_rmse": None,
        }

    c_at_q = np.interp(q_tvt, c_tvt, c_gr)
    diff = q_gr - c_at_q
    q_std = float(np.std(q_gr))
    c_std = float(np.std(c_at_q))
    if len(q_gr) >= 3 and q_std > 1e-8 and c_std > 1e-8:
        corr = float(np.corrcoef(q_gr, c_at_q)[0, 1])
        corr = float(np.clip(corr, -1.0, 1.0))
        z_rmse = float(math.sqrt(max(0.0, 2.0 - 2.0 * corr)))
    else:
        corr = None
        z_rmse = None
    return {
        "exact_rows": int(len(q_gr)),
        "exact_coverage_query": float(len(q_gr) / max(np.isfinite(query.tvt).sum(), 1)),
        "exact_corr": corr,
        "exact_raw_rmse": float(np.sqrt(np.mean(diff * diff))),
        "exact_raw_mae": float(np.mean(np.abs(diff))),
        "exact_resid_std": float(np.std(diff)),
        "exact_z_rmse": z_rmse,
    }


def self_candidate_for_query(raw_dir: Path, query: RawCurve, source_view: str) -> RawCurve | None:
    path = raw_dir / query.split / f"{query.well_id}{HORIZONTAL_SUFFIX}"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    views = horizontal_views(query.split, frame)
    if source_view not in views:
        return None
    tvt, gr = views[source_view]
    return RawCurve(
        label=f"self_gr:{query.split}:{query.well_id}:{source_view}",
        split=query.split,
        well_id=query.well_id,
        kind="self_gr",
        view=source_view,
        tvt=tvt,
        gr=gr,
    )


def build_exact_selected_metrics(
    raw_dir: Path,
    query_index: pd.DataFrame,
    candidate_raw: dict[str, RawCurve],
    query_raw: dict[str, RawCurve],
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_query = pair_summary.set_index("query_label", drop=False)
    for query_label, query in query_raw.items():
        if query_label not in by_query.index:
            continue
        summary = by_query.loc[query_label]
        if isinstance(summary, pd.DataFrame):
            summary = summary.iloc[0]
        candidates: list[tuple[str, str]] = []
        for role, column in [
            ("provided_typewell", "provided_label"),
            ("best_typewell", "best_label"),
        ]:
            label = summary.get(column)
            if isinstance(label, str) and label in candidate_raw:
                candidates.append((role, label))
        if query.split == "train":
            for source_view in ("prefix", "full"):
                self_curve = self_candidate_for_query(raw_dir, query, source_view)
                if self_curve is None:
                    continue
                role = f"self_gr_{source_view}"
                row = {
                    "query_label": query_label,
                    "split": query.split,
                    "well_id": query.well_id,
                    "view": query.view,
                    "candidate_role": role,
                    "candidate_label": self_curve.label,
                    "candidate_split": self_curve.split,
                    "candidate_well_id": self_curve.well_id,
                    "candidate_kind": self_curve.kind,
                    "candidate_view": self_curve.view,
                }
                row.update(exact_curve_metrics(query, self_curve))
                rows.append(row)
        elif query.view == "prefix":
            self_curve = self_candidate_for_query(raw_dir, query, "prefix")
            if self_curve is not None:
                row = {
                    "query_label": query_label,
                    "split": query.split,
                    "well_id": query.well_id,
                    "view": query.view,
                    "candidate_role": "self_gr_prefix",
                    "candidate_label": self_curve.label,
                    "candidate_split": self_curve.split,
                    "candidate_well_id": self_curve.well_id,
                    "candidate_kind": self_curve.kind,
                    "candidate_view": self_curve.view,
                }
                row.update(exact_curve_metrics(query, self_curve))
                rows.append(row)

        for role, label in candidates:
            candidate = candidate_raw[label]
            row = {
                "query_label": query_label,
                "split": query.split,
                "well_id": query.well_id,
                "view": query.view,
                "candidate_role": role,
                "candidate_label": label,
                "candidate_split": candidate.split,
                "candidate_well_id": candidate.well_id,
                "candidate_kind": candidate.kind,
                "candidate_view": candidate.view,
            }
            row.update(exact_curve_metrics(query, candidate))
            rows.append(row)
    return pd.DataFrame(rows)


def build_prefix_selected_hidden_metrics(
    query_index: pd.DataFrame,
    candidate_index: pd.DataFrame,
    metrics: dict[str, np.ndarray],
    pair_summary: pd.DataFrame,
) -> pd.DataFrame:
    query_label_to_idx = {label: idx for idx, label in enumerate(query_index["label"].astype(str))}
    candidate_label_to_idx = {
        label: idx for idx, label in enumerate(candidate_index["label"].astype(str))
    }
    prefix_best_by_well = {
        row.well_id: row.best_label
        for row in pair_summary.itertuples(index=False)
        if row.split == "train" and row.view == "prefix" and isinstance(row.best_label, str)
    }
    rows: list[dict[str, Any]] = []
    hidden_rows = pair_summary[
        (pair_summary["split"] == "train") & (pair_summary["view"] == "hidden")
    ]
    for row in hidden_rows.itertuples(index=False):
        candidate_label = prefix_best_by_well.get(row.well_id)
        hidden_label = row.query_label
        if candidate_label not in candidate_label_to_idx or hidden_label not in query_label_to_idx:
            continue
        qi = query_label_to_idx[hidden_label]
        ci = candidate_label_to_idx[candidate_label]
        metric = metric_at(metrics, qi, ci)
        rows.append(
            {
                "well_id": row.well_id,
                "hidden_query_label": hidden_label,
                "prefix_selected_label": candidate_label,
                "prefix_selected_is_provided": candidate_label == row.provided_label,
                **{f"prefix_selected_hidden_{k}": v for k, v in metric.items()},
                "provided_hidden_z_rmse": row.provided_z_rmse,
                "provided_hidden_raw_rmse": row.provided_raw_rmse,
                "oracle_hidden_z_rmse": row.best_z_rmse,
                "oracle_hidden_raw_rmse": row.best_raw_rmse,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["prefix_selected_minus_provided_hidden_z_rmse"] = (
            out["prefix_selected_hidden_z_rmse"] - out["provided_hidden_z_rmse"]
        )
        out["prefix_selected_minus_provided_hidden_raw_rmse"] = (
            out["prefix_selected_hidden_raw_rmse"] - out["provided_hidden_raw_rmse"]
        )
    return out


def numeric_summary(values: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_aggregate_summary(
    pair_summary: pd.DataFrame,
    exact_metrics: pd.DataFrame,
    prefix_hidden: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {"by_split_view": []}
    for (split, view), part in pair_summary.groupby(["split", "view"], sort=True):
        provided_valid = part["provided_in_primary_pool"].astype(bool)
        ranks = pd.to_numeric(part["provided_rank_by_z_rmse"], errors="coerce")
        item = {
            "split": split,
            "view": view,
            "queries": int(len(part)),
            "provided_primary_valid_rate": float(provided_valid.mean()) if len(part) else None,
            "provided_best_rate": float(part["best_is_provided"].fillna(False).mean())
            if len(part)
            else None,
            "provided_top5_rate": float((ranks <= 5).mean()) if len(part) else None,
            "provided_median_rank": float(ranks.median()) if ranks.notna().any() else None,
            "provided_corr": numeric_summary(part["provided_corr"]),
            "provided_z_rmse": numeric_summary(part["provided_z_rmse"]),
            "provided_raw_rmse": numeric_summary(part["provided_raw_rmse"]),
            "best_z_rmse": numeric_summary(part["best_z_rmse"]),
            "best_raw_rmse": numeric_summary(part["best_raw_rmse"]),
            "best_minus_provided_z_rmse": numeric_summary(part["best_minus_provided_z_rmse"]),
            "best_minus_provided_raw_rmse": numeric_summary(part["best_minus_provided_raw_rmse"]),
            "provided_corr_lt_0p3_rate": float(
                (pd.to_numeric(part["provided_corr"], errors="coerce") < 0.3).mean()
            ),
            "provided_z_rmse_gt_1p0_rate": float(
                (pd.to_numeric(part["provided_z_rmse"], errors="coerce") > 1.0).mean()
            ),
            "best_candidate_split_counts": {
                str(key): int(value)
                for key, value in part["best_split"].value_counts(dropna=False).items()
            },
        }
        out["by_split_view"].append(item)

    if not exact_metrics.empty:
        out["exact_by_role_view"] = []
        for (view, role), part in exact_metrics.groupby(["view", "candidate_role"], sort=True):
            out["exact_by_role_view"].append(
                {
                    "view": view,
                    "candidate_role": role,
                    "rows": int(len(part)),
                    "coverage": numeric_summary(part["exact_coverage_query"]),
                    "corr": numeric_summary(part["exact_corr"]),
                    "z_rmse": numeric_summary(part["exact_z_rmse"]),
                    "raw_rmse": numeric_summary(part["exact_raw_rmse"]),
                    "raw_mae": numeric_summary(part["exact_raw_mae"]),
                }
            )

    if not prefix_hidden.empty:
        out["prefix_selected_hidden"] = {
            "wells": int(len(prefix_hidden)),
            "prefix_selected_is_provided_rate": float(
                prefix_hidden["prefix_selected_is_provided"].astype(bool).mean()
            ),
            "delta_z_rmse": numeric_summary(
                prefix_hidden["prefix_selected_minus_provided_hidden_z_rmse"]
            ),
            "delta_raw_rmse": numeric_summary(
                prefix_hidden["prefix_selected_minus_provided_hidden_raw_rmse"]
            ),
            "improves_hidden_z_rmse_rate": float(
                (prefix_hidden["prefix_selected_minus_provided_hidden_z_rmse"] < 0).mean()
            ),
            "improves_hidden_raw_rmse_rate": float(
                (prefix_hidden["prefix_selected_minus_provided_hidden_raw_rmse"] < 0).mean()
            ),
        }
    return out


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        if not np.isfinite(float(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(
    output_dir: Path,
    pair_summary: pd.DataFrame,
    exact_metrics: pd.DataFrame,
    prefix_hidden: pd.DataFrame,
    aggregate: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# typewell GR TVT fit audit")
    lines.append("")
    lines.append(f"- generated_at_utc: `{datetime.now(UTC).isoformat()}`")
    lines.append("- primary rank: lower grid `z_rmse`, with minimum overlap/coverage filters")
    lines.append(
        "- `prefix`: rows with finite `TVT_input`; "
        "`hidden`: train rows with missing `TVT_input`; "
        "`full`: all finite train `TVT` rows"
    )
    lines.append("")

    lines.append("## Grid summary")
    lines.append("")
    lines.append(
        "| split | view | wells | provided best rate | provided top5 rate | median rank | "
        "provided corr median | provided zRMSE median | best zRMSE median | "
        "median best-provided zRMSE |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in aggregate["by_split_view"]:
        lines.append(
            (
                "| {split} | {view} | {queries} | {best_rate} | {top5_rate} | "
                "{rank} | {corr} | {pz} | {bz} | {delta} |"
            ).format(
                split=item["split"],
                view=item["view"],
                queries=item["queries"],
                best_rate=fmt(item["provided_best_rate"]),
                top5_rate=fmt(item["provided_top5_rate"]),
                rank=fmt(item["provided_median_rank"]),
                corr=fmt(item["provided_corr"]["median"]),
                pz=fmt(item["provided_z_rmse"]["median"]),
                bz=fmt(item["best_z_rmse"]["median"]),
                delta=fmt(item["best_minus_provided_z_rmse"]["median"]),
            )
        )
    lines.append("")

    if not exact_metrics.empty:
        exact_summary = (
            exact_metrics.groupby(["view", "candidate_role"], sort=True)
            .agg(
                wells=("query_label", "nunique"),
                coverage_median=("exact_coverage_query", "median"),
                corr_median=("exact_corr", "median"),
                z_rmse_median=("exact_z_rmse", "median"),
                raw_rmse_median=("exact_raw_rmse", "median"),
                raw_mae_median=("exact_raw_mae", "median"),
            )
            .reset_index()
        )
        lines.append("## Exact selected-pair summary")
        lines.append("")
        lines.append(
            "| view | role | wells | coverage median | corr median | zRMSE median | "
            "raw RMSE median | raw MAE median |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in exact_summary.itertuples(index=False):
            lines.append(
                
                    f"| {row.view} | {row.candidate_role} | {row.wells} | "
                    f"{fmt(row.coverage_median)} | {fmt(row.corr_median)} | "
                    f"{fmt(row.z_rmse_median)} | {fmt(row.raw_rmse_median)} | "
                    f"{fmt(row.raw_mae_median)} |"
                
            )
        lines.append("")

    if not prefix_hidden.empty:
        lines.append("## Prefix-selected typewell applied to hidden rows")
        lines.append("")
        lines.append(
            "- prefix だけで選んだ best typewell が hidden の provided typewell より zRMSE 改善: "
            + fmt((prefix_hidden["prefix_selected_minus_provided_hidden_z_rmse"] < 0).mean())
        )
        lines.append(
            "- prefix だけで選んだ best typewell が hidden の provided typewell より "
            "raw RMSE 改善: "
            + fmt((prefix_hidden["prefix_selected_minus_provided_hidden_raw_rmse"] < 0).mean())
        )
        lines.append(
            "- median delta zRMSE: "
            + fmt(prefix_hidden["prefix_selected_minus_provided_hidden_z_rmse"].median())
        )
        lines.append(
            "- median delta raw RMSE: "
            + fmt(prefix_hidden["prefix_selected_minus_provided_hidden_raw_rmse"].median())
        )
        lines.append("")

    bad = pair_summary[
        (pair_summary["split"] == "train")
        & (pair_summary["view"] == "hidden")
        & pd.to_numeric(pair_summary["provided_corr"], errors="coerce").lt(0.0)
    ].copy()
    if not bad.empty:
        bad = bad.sort_values(["provided_corr", "provided_z_rmse"]).head(20)
        lines.append("## Worst provided hidden fits by grid corr")
        lines.append("")
        lines.append(
            "| well | provided corr | provided zRMSE | best well | best corr | best zRMSE | "
            "rank |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for row in bad.itertuples(index=False):
            lines.append(
                
                    f"| {row.well_id} | {fmt(row.provided_corr)} | "
                    f"{fmt(row.provided_z_rmse)} | {row.best_well_id} | "
                    f"{fmt(row.best_corr)} | {fmt(row.best_z_rmse)} | "
                    f"{fmt(row.provided_rank_by_z_rmse, 0)} |"
                
            )
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    for filename in [
        "curve_index.csv",
        "typewell_candidate_index.csv",
        "pair_summary.csv",
        "top_typewell_candidates.csv",
        "selected_pair_exact_metrics.csv",
        "prefix_selected_hidden_metrics.csv",
        "aggregate_summary.json",
    ]:
        lines.append(f"- `{filename}`")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify horizontal-vs-typewell GR agreement on TVT and search "
            "alternate typewells."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("studies/typewell_gr_tvt_fit_audit_outputs"),
    )
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--min-overlap-points", type=int, default=80)
    parser.add_argument("--min-coverage-query", type=float, default=0.80)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tvt_min, tvt_max = collect_global_tvt_bounds(args.raw_dir)
    grid_min = math.floor(tvt_min / args.grid_step) * args.grid_step
    grid_max = math.ceil(tvt_max / args.grid_step) * args.grid_step
    grid_size = int(round((grid_max - grid_min) / args.grid_step)) + 1

    candidate_index, candidate_values, candidate_raw = load_typewell_curves(
        args.raw_dir, grid_min=grid_min, grid_step=args.grid_step, grid_size=grid_size
    )
    query_index, query_values, query_raw = load_horizontal_query_curves(
        args.raw_dir, grid_min=grid_min, grid_step=args.grid_step, grid_size=grid_size
    )

    metrics = pairwise_grid_metrics(
        query_values,
        candidate_values,
        block_size=int(args.block_size),
    )
    pair_summary, top_candidates = build_pair_summaries(
        query_index,
        candidate_index,
        metrics,
        min_overlap_points=int(args.min_overlap_points),
        min_coverage_query=float(args.min_coverage_query),
        top_k=int(args.top_k),
    )
    exact_metrics = build_exact_selected_metrics(
        args.raw_dir, query_index, candidate_raw, query_raw, pair_summary
    )
    prefix_hidden = build_prefix_selected_hidden_metrics(
        query_index, candidate_index, metrics, pair_summary
    )
    aggregate = build_aggregate_summary(pair_summary, exact_metrics, prefix_hidden)
    aggregate["config"] = {
        "raw_dir": str(args.raw_dir),
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_size": grid_size,
        "grid_step": args.grid_step,
        "min_overlap_points": args.min_overlap_points,
        "min_coverage_query": args.min_coverage_query,
        "block_size": args.block_size,
        "top_k": args.top_k,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    query_index.to_csv(output_dir / "curve_index.csv", index=False)
    candidate_index.to_csv(output_dir / "typewell_candidate_index.csv", index=False)
    pair_summary.to_csv(output_dir / "pair_summary.csv", index=False)
    top_candidates.to_csv(output_dir / "top_typewell_candidates.csv", index=False)
    exact_metrics.to_csv(output_dir / "selected_pair_exact_metrics.csv", index=False)
    prefix_hidden.to_csv(output_dir / "prefix_selected_hidden_metrics.csv", index=False)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(to_jsonable(aggregate), indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, pair_summary, exact_metrics, prefix_hidden, aggregate)

    print(json.dumps(aggregate["config"], indent=2, sort_keys=True))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
