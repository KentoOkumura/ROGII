from __future__ import annotations

import gzip
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR if (EXP_DIR / "project.yml").exists() else EXP_DIR.parents[1]
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def load_config() -> dict[str, Any]:
    with (EXP_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_first_existing(candidates: list[str]) -> Path:
    checked: list[str] = []
    for raw in candidates:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        checked.append(str(path))
        if path.exists() and path.stat().st_size > 0:
            return path
        if KAGGLE_INPUT_ROOT.exists():
            matches = sorted(
                candidate
                for candidate in KAGGLE_INPUT_ROOT.rglob(Path(raw).name)
                if candidate.is_file() and candidate.stat().st_size > 0
            )
            if matches:
                return matches[0]
    raise FileNotFoundError("No existing non-empty path among: " + ", ".join(checked))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def load_lgb_mean_predictions(path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    validation = cfg["validation"]
    usecols = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target_tvt",
        "pred_tvt",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=750_000):
        mask = (
            chunk["model"].eq(validation["source_prediction_model"])
            & chunk["mode"].eq(validation["source_prediction_mode"])
            & chunk["variant"].eq(validation["source_prediction_variant"])
        )
        part = chunk.loc[mask].copy()
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError(f"No rows matched lgb_mean filters in {path}")
    pred = pd.concat(chunks, ignore_index=True)
    row_idx = pred["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
    pred["row_idx"] = row_idx
    pred["target_tvt"] = pred["target_tvt"].astype(np.float32)
    pred["pred_tvt"] = pred["pred_tvt"].astype(np.float32)
    pred["last_known_tvt"] = pred["last_known_tvt"].astype(np.float32)
    pred["residual"] = (pred["pred_tvt"] - pred["target_tvt"]).astype(np.float32)
    pred["abs_error"] = np.abs(pred["residual"]).astype(np.float32)
    pred["sq_error"] = (pred["residual"].astype(np.float64) ** 2).astype(np.float64)
    pred = pred.drop(columns=["id", "variant", "mode", "model"])
    pred["well"] = pred["well"].astype("category")
    return pred


def attach_raw_and_typewell(
    pred: pd.DataFrame, typewell_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tw = pd.read_csv(typewell_path)
    tw = tw[
        [
            "well_id",
            "exact_typewell_group",
            "typewell_hash",
            "exact_typewell_group_size",
            "shared_typewell",
            "x_mean",
            "y_mean",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
        ]
    ].rename(columns={"well_id": "well"})
    tw["exact_typewell_group"] = tw["exact_typewell_group"].astype(str)
    tw["shared_typewell"] = tw["shared_typewell"].astype(bool)
    group_map = tw.set_index("well")["exact_typewell_group"]
    df = pred.copy()
    df["well"] = df["well"].astype(str)
    df["exact_typewell_group"] = df["well"].map(group_map).astype("category")
    if int(df["exact_typewell_group"].isna().sum()):
        raise ValueError("Missing exact_typewell_group after join")
    return df, tw


def compute_well_summary(df: pd.DataFrame, tw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    gb = df.groupby("well", sort=True, observed=True)
    well = gb.agg(
        rows=("residual", "size"),
        error_mean=("residual", "mean"),
        error_median=("residual", "median"),
        abs_error_mean=("abs_error", "mean"),
        abs_error_median=("abs_error", "median"),
        sq_error_mean=("sq_error", "mean"),
        target_tvt_min=("target_tvt", "min"),
        target_tvt_max=("target_tvt", "max"),
        target_tvt_mean=("target_tvt", "mean"),
        pred_tvt_min=("pred_tvt", "min"),
        pred_tvt_max=("pred_tvt", "max"),
        pred_tvt_mean=("pred_tvt", "mean"),
    ).reset_index()
    well["rmse_tvt"] = np.sqrt(well.pop("sq_error_mean"))
    sign = gb["residual"].agg(
        positive_rate=lambda s: float(np.mean(s.to_numpy() > 0)),
        negative_rate=lambda s: float(np.mean(s.to_numpy() < 0)),
    ).reset_index()
    well = well.merge(sign, on="well", how="left", validate="one_to_one")
    well["sign_consistency"] = well[["positive_rate", "negative_rate"]].max(axis=1)
    well["abs_bias"] = np.abs(well["error_mean"])
    well["bias_rmse_ratio"] = well["abs_bias"] / well["rmse_tvt"].replace(0, np.nan)
    well["target_tvt_span"] = well["target_tvt_max"] - well["target_tvt_min"]
    well["pred_tvt_span"] = well["pred_tvt_max"] - well["pred_tvt_min"]
    readout = cfg["readout"]
    well["offset_flag"] = (
        (well["abs_bias"] >= float(readout["offset_abs_bias_threshold_ft"]))
        & (well["bias_rmse_ratio"] >= float(readout["offset_bias_rmse_ratio_threshold"]))
    )
    well["offset_direction"] = np.where(
        well["error_mean"] > 0,
        "overpredict",
        np.where(well["error_mean"] < 0, "underpredict", "neutral"),
    )
    well = well.merge(tw, on="well", how="left", validate="one_to_one")
    return well.sort_values("rmse_tvt", ascending=False).reset_index(drop=True)


def compute_profiles(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well, group in df.groupby("well", sort=True, observed=True):
        g = group.sort_values("row_idx")
        n = len(g)
        if n == 1:
            profile_bin = np.array([0], dtype=np.int16)
        else:
            profile_bin = np.minimum(
                (np.arange(n, dtype=np.float64) / max(n - 1, 1) * bins).astype(np.int16),
                bins - 1,
            )
        part = pd.DataFrame(
            {
                "well": well,
                "profile_bin": profile_bin,
                "residual": g["residual"].to_numpy(np.float32),
            }
        )
        parts.append(part)
    binned = pd.concat(parts, ignore_index=True)
    prof = binned.groupby(["well", "profile_bin"], observed=True)["residual"].mean().unstack()
    prof = prof.reindex(columns=range(bins))
    prof = prof.interpolate(axis=1, limit_direction="both").fillna(0.0)
    prof.columns = [f"bin_{i:03d}" for i in range(bins)]
    return prof


def correlation_matrix(profile: pd.DataFrame) -> np.ndarray:
    arr = profile.to_numpy(np.float64)
    arr = arr - arr.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(arr, axis=1)
    norm[norm == 0] = np.nan
    scaled = arr / norm[:, None]
    corr = scaled @ scaled.T
    corr[~np.isfinite(corr)] = np.nan
    return corr


def pair_summary(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n_pairs": 0, "mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "n_pairs": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def compute_shape_similarity(
    well: pd.DataFrame, profile: pd.DataFrame, xy_neighbors: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    well_order = list(profile.index)
    pos = {w: i for i, w in enumerate(well_order)}
    corr = correlation_matrix(profile)
    upper = np.triu_indices(len(well_order), 1)
    all_summary = pair_summary(corr[upper])

    rows: list[dict[str, Any]] = []
    same_values: list[float] = []
    for group_name, group in well.groupby("exact_typewell_group", sort=True):
        idx = [pos[w] for w in group["well"] if w in pos]
        if len(idx) < 2:
            vals = np.array([], dtype=float)
        else:
            local = corr[np.ix_(idx, idx)]
            vals = local[np.triu_indices(len(idx), 1)]
            same_values.extend(vals[np.isfinite(vals)].tolist())
        ps = pair_summary(vals)
        rows.append(
            {
                "exact_typewell_group": group_name,
                "shape_corr_pairs": ps["n_pairs"],
                "shape_corr_mean": ps["mean"],
                "shape_corr_median": ps["median"],
                "shape_corr_p25": ps["p25"],
                "shape_corr_p75": ps["p75"],
            }
        )

    xy_vals: list[float] = []
    xy_rows: list[dict[str, Any]] = []
    for w, neighbors in xy_neighbors.items():
        if w not in pos:
            continue
        vals = np.array([corr[pos[w], pos[n]] for n in neighbors if n in pos], dtype=float)
        xy_vals.extend(vals[np.isfinite(vals)].tolist())
        ps = pair_summary(vals)
        xy_rows.append(
            {
                "well": w,
                "xy_neighbor_shape_corr_mean": ps["mean"],
                "xy_neighbor_shape_corr_median": ps["median"],
                "xy_neighbor_shape_corr_pairs": ps["n_pairs"],
            }
        )

    summary = {
        "all_pair_shape_corr": all_summary,
        "same_typewell_shape_corr": pair_summary(np.asarray(same_values, dtype=float)),
        "xy_neighbor_shape_corr": pair_summary(np.asarray(xy_vals, dtype=float)),
    }
    group_shape = pd.DataFrame(rows)
    xy_shape = pd.DataFrame(xy_rows)
    return group_shape, {"summary": summary, "xy_shape": xy_shape}


def compute_xy_neighbors(well: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    work = well.dropna(subset=["x_mean", "y_mean"]).reset_index(drop=True)
    coords = work[["x_mean", "y_mean"]].to_numpy(np.float64)
    bias = work["error_mean"].to_numpy(np.float64)
    rmse = work["rmse_tvt"].to_numpy(np.float64)
    wells = work["well"].tolist()

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    k_eff = min(k, len(work) - 1)
    nn = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    sorted_nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(dist, nn, axis=1), axis=1), axis=1)

    rows: list[dict[str, Any]] = []
    neighbor_map: dict[str, list[str]] = {}
    nearest_bias_diffs: list[float] = []
    nearest_same_sign: list[bool] = []
    for i, idx in enumerate(sorted_nn):
        nb_bias = bias[idx]
        nb_rmse = rmse[idx]
        nb_dist = dist[i, idx]
        nb_wells = [wells[j] for j in idx]
        neighbor_map[wells[i]] = nb_wells
        abs_diff = np.abs(nb_bias - bias[i])
        same_sign = np.sign(nb_bias) == np.sign(bias[i])
        nearest_bias_diffs.extend(abs_diff.tolist())
        nearest_same_sign.extend(same_sign.tolist())
        rows.append(
            {
                "well": wells[i],
                "xy_neighbor_k": int(k_eff),
                "xy_neighbor_wells": ",".join(nb_wells),
                "xy_neighbor_distance_mean": float(np.mean(nb_dist)),
                "xy_neighbor_distance_min": float(np.min(nb_dist)),
                "xy_neighbor_bias_mean": float(np.mean(nb_bias)),
                "xy_neighbor_abs_bias_mean": float(np.mean(np.abs(nb_bias))),
                "xy_neighbor_rmse_mean": float(np.mean(nb_rmse)),
                "xy_neighbor_bias_abs_diff_mean": float(np.mean(abs_diff)),
                "xy_neighbor_bias_same_sign_rate": float(np.mean(same_sign)),
                "xy_local_bias_mean_abs": float(abs(np.mean(np.r_[bias[i], nb_bias]))),
                "xy_local_bias_same_sign_rate": float(
                    np.mean(np.sign(np.r_[bias[i], nb_bias]) == np.sign(np.mean(np.r_[bias[i], nb_bias])))
                ),
            }
        )

    upper = np.triu_indices(len(work), 1)
    all_bias_diff = np.abs(bias[upper[0]] - bias[upper[1]])
    all_same_sign = np.sign(bias[upper[0]]) == np.sign(bias[upper[1]])
    summary = {
        "wells": int(len(work)),
        "neighbor_k": int(k_eff),
        "nearest_bias_abs_diff_mean": float(np.mean(nearest_bias_diffs)),
        "nearest_bias_abs_diff_median": float(np.median(nearest_bias_diffs)),
        "all_pair_bias_abs_diff_mean": float(np.mean(all_bias_diff)),
        "all_pair_bias_abs_diff_median": float(np.median(all_bias_diff)),
        "nearest_bias_same_sign_rate": float(np.mean(nearest_same_sign)),
        "all_pair_bias_same_sign_rate": float(np.mean(all_same_sign)),
        "nearest_vs_all_abs_diff_ratio": safe_div(float(np.mean(nearest_bias_diffs)), float(np.mean(all_bias_diff))),
    }
    return pd.DataFrame(rows), summary, neighbor_map


def compute_typewell_metrics(well: pd.DataFrame, df: pd.DataFrame, group_shape: pd.DataFrame) -> pd.DataFrame:
    row = df.groupby("exact_typewell_group", observed=True).agg(
        rows=("residual", "size"),
        error_mean=("residual", "mean"),
        error_median=("residual", "median"),
        abs_error_mean=("abs_error", "mean"),
        sq_error_mean=("sq_error", "mean"),
        target_tvt_min=("target_tvt", "min"),
        target_tvt_max=("target_tvt", "max"),
    ).reset_index()
    row["rmse_tvt"] = np.sqrt(row.pop("sq_error_mean"))
    row["target_tvt_span"] = row["target_tvt_max"] - row["target_tvt_min"]

    def same_sign_rate(s: pd.Series) -> float:
        arr = s.to_numpy(np.float64)
        return float(max(np.mean(arr > 0), np.mean(arr < 0)))

    wg = well.groupby("exact_typewell_group", sort=True).agg(
        wells=("well", "nunique"),
        exact_typewell_group_size=("exact_typewell_group_size", "max"),
        shared_typewell_wells=("shared_typewell", "sum"),
        well_rmse_mean=("rmse_tvt", "mean"),
        well_rmse_max=("rmse_tvt", "max"),
        well_bias_mean=("error_mean", "mean"),
        well_abs_bias_mean=("abs_bias", "mean"),
        offset_wells=("offset_flag", "sum"),
        positive_bias_wells=("error_mean", lambda s: int(np.sum(s.to_numpy() > 0))),
        negative_bias_wells=("error_mean", lambda s: int(np.sum(s.to_numpy() < 0))),
        well_bias_same_sign_rate=("error_mean", same_sign_rate),
        x_mean=("x_mean", "mean"),
        y_mean=("y_mean", "mean"),
    ).reset_index()
    out = row.merge(wg, on="exact_typewell_group", how="left", validate="one_to_one")
    out["offset_well_rate"] = out["offset_wells"] / out["wells"].replace(0, np.nan)
    out["abs_error_mean_minus_global"] = out["abs_error_mean"] - float(df["abs_error"].mean())
    out = out.merge(group_shape, on="exact_typewell_group", how="left", validate="one_to_one")
    return out.sort_values(["rmse_tvt", "wells"], ascending=[False, False]).reset_index(drop=True)


def compute_typewell_metrics_from_well_summary(well: pd.DataFrame) -> pd.DataFrame:
    def same_sign_rate(s: pd.Series) -> float:
        arr = s.to_numpy(np.float64)
        return float(max(np.mean(arr > 0), np.mean(arr < 0)))

    out = well.groupby("exact_typewell_group", sort=True).agg(
        wells=("well", "nunique"),
        exact_typewell_group_size=("exact_typewell_group_size", "max"),
        shared_typewell_wells=("shared_typewell", "sum"),
        well_rmse_mean=("rmse_tvt", "mean"),
        well_rmse_max=("rmse_tvt", "max"),
        well_bias_mean=("error_mean", "mean"),
        well_abs_bias_mean=("abs_bias", "mean"),
        offset_wells=("offset_flag", "sum"),
        positive_bias_wells=("error_mean", lambda s: int(np.sum(s.to_numpy() > 0))),
        negative_bias_wells=("error_mean", lambda s: int(np.sum(s.to_numpy() < 0))),
        well_bias_same_sign_rate=("error_mean", same_sign_rate),
        x_mean=("x_mean", "mean"),
        y_mean=("y_mean", "mean"),
    ).reset_index()
    out["offset_well_rate"] = out["offset_wells"] / out["wells"].replace(0, np.nan)
    return out


def compute_sharp_steps(
    df: pd.DataFrame, well: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = df[["well", "row_idx", "target_tvt", "pred_tvt", "exact_typewell_group"]].copy()
    work = work.sort_values(["well", "row_idx"]).reset_index(drop=True)
    gb = work.groupby("well", sort=False, observed=True)
    work["true_step"] = gb["target_tvt"].diff()
    work["pred_step"] = gb["pred_tvt"].diff()
    step = work.dropna(subset=["true_step", "pred_step"]).copy()
    step["abs_true_step"] = np.abs(step["true_step"])
    step["abs_pred_step"] = np.abs(step["pred_step"])
    threshold = float(step["abs_true_step"].quantile(float(cfg["readout"]["sharp_step_quantile"])))
    sharp = step[step["abs_true_step"] >= threshold].copy()
    sharp["step_direction"] = np.where(sharp["true_step"] >= 0, "up", "down")
    sharp["step_error"] = sharp["pred_step"] - sharp["true_step"]
    sharp["abs_step_error"] = np.abs(sharp["step_error"])
    sharp["damping_abs_pred_over_true"] = sharp["abs_pred_step"] / sharp["abs_true_step"].replace(0, np.nan)

    all_step_well = step.groupby("well", sort=True).agg(
        all_step_count=("true_step", "size"),
        true_step_abs_mean=("abs_true_step", "mean"),
        true_step_abs_p95=("abs_true_step", lambda s: float(np.quantile(s, 0.95))),
        true_step_abs_p99=("abs_true_step", lambda s: float(np.quantile(s, 0.99))),
        true_step_abs_max=("abs_true_step", "max"),
    ).reset_index()

    if sharp.empty:
        sharp_well = pd.DataFrame(columns=["well"])
    else:
        sharp_well = sharp.groupby("well", sort=True).agg(
            sharp_step_count=("true_step", "size"),
            sharp_up_count=("step_direction", lambda s: int(np.sum(s.to_numpy() == "up"))),
            sharp_down_count=("step_direction", lambda s: int(np.sum(s.to_numpy() == "down"))),
            sharp_true_step_abs_mean=("abs_true_step", "mean"),
            sharp_pred_step_abs_mean=("abs_pred_step", "mean"),
            sharp_abs_step_error_mean=("abs_step_error", "mean"),
            sharp_damping_mean=("damping_abs_pred_over_true", "mean"),
            sharp_damping_median=("damping_abs_pred_over_true", "median"),
        ).reset_index()

    sharp_well = all_step_well.merge(sharp_well, on="well", how="left")
    fill_cols = ["sharp_step_count", "sharp_up_count", "sharp_down_count"]
    for col in fill_cols:
        sharp_well[col] = sharp_well[col].fillna(0).astype(int)
    sharp_well["sharp_step_rate"] = sharp_well["sharp_step_count"] / sharp_well["all_step_count"].replace(0, np.nan)
    sharp_well = sharp_well.merge(
        well[
            [
                "well",
                "rmse_tvt",
                "error_mean",
                "abs_bias",
                "offset_flag",
                "exact_typewell_group",
                "x_mean",
                "y_mean",
            ]
        ],
        on="well",
        how="left",
        validate="one_to_one",
    )
    sharp_typewell = sharp_well.groupby("exact_typewell_group", sort=True).agg(
        wells=("well", "nunique"),
        wells_with_sharp_steps=("sharp_step_count", lambda s: int(np.sum(s.to_numpy() > 0))),
        sharp_step_count=("sharp_step_count", "sum"),
        sharp_step_rate_mean=("sharp_step_rate", "mean"),
        sharp_damping_mean=("sharp_damping_mean", "mean"),
        sharp_abs_step_error_mean=("sharp_abs_step_error_mean", "mean"),
        rmse_tvt_mean=("rmse_tvt", "mean"),
        abs_bias_mean=("abs_bias", "mean"),
    ).reset_index()
    sharp_typewell["wells_with_sharp_step_rate"] = (
        sharp_typewell["wells_with_sharp_steps"] / sharp_typewell["wells"].replace(0, np.nan)
    )
    summary = {
        "sharp_step_quantile": float(cfg["readout"]["sharp_step_quantile"]),
        "sharp_step_abs_true_threshold": threshold,
        "all_step_rows": int(len(step)),
        "sharp_step_rows": int(len(sharp)),
        "sharp_step_row_rate": safe_div(float(len(sharp)), float(len(step))),
        "sharp_step_wells": int(np.sum(sharp_well["sharp_step_count"].to_numpy() > 0)),
        "sharp_step_damping_mean": float(sharp["damping_abs_pred_over_true"].mean()) if len(sharp) else float("nan"),
        "sharp_step_abs_error_mean": float(sharp["abs_step_error"].mean()) if len(sharp) else float("nan"),
    }
    return sharp, sharp_well, sharp_typewell, summary


def color_scale(value: float, vmax: float) -> str:
    if not np.isfinite(value) or vmax <= 0:
        return "#999999"
    t = max(-1.0, min(1.0, value / vmax))
    if t >= 0:
        r, g, b = 210, int(245 - 120 * t), int(245 - 135 * t)
    else:
        r, g, b = int(245 + 120 * t), int(245 + 105 * t), 220
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg_text(path: Path, body: str, width: int, height: int) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n',
        encoding="utf-8",
    )


def write_typewell_bar_svg(typewell: pd.DataFrame, path: Path, top_n: int = 24) -> None:
    data = typewell.sort_values("error_mean", key=lambda s: np.abs(s), ascending=False).head(top_n).copy()
    width, height = 980, 34 * len(data) + 90
    left, right, top = 250, 40, 45
    vmax = max(float(np.nanmax(np.abs(data["error_mean"]))), 1.0)
    scale = (width - left - right) / (2 * vmax)
    parts = [
        '<text x="20" y="26" font-size="18" font-family="sans-serif">typewell group mean residual (pred - true)</text>',
        f'<line x1="{left + vmax * scale}" y1="{top - 15}" x2="{left + vmax * scale}" y2="{height - 30}" stroke="#555"/>',
    ]
    zero = left + vmax * scale
    for i, row in enumerate(data.itertuples(index=False)):
        y = top + i * 34
        val = float(row.error_mean)
        x = zero if val >= 0 else zero + val * scale
        w = abs(val) * scale
        label = html.escape(str(row.exact_typewell_group))
        color = color_scale(val, vmax)
        parts.append(f'<text x="8" y="{y + 14}" font-size="11" font-family="sans-serif">{label}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="18" fill="{color}" stroke="#777"/>')
        parts.append(
            f'<text x="{min(width - 150, max(8, zero + (w + 8) * (1 if val >= 0 else -1))):.1f}" '
            f'y="{y + 14}" font-size="11" font-family="sans-serif">{val:.2f} ft, RMSE {float(row.rmse_tvt):.2f}</text>'
        )
    write_svg_text(path, "\n".join(parts), width, height)


def write_xy_bias_svg(well: pd.DataFrame, path: Path) -> None:
    data = well.dropna(subset=["x_mean", "y_mean"]).copy()
    width, height = 900, 720
    pad = 55
    x = data["x_mean"].to_numpy(np.float64)
    y = data["y_mean"].to_numpy(np.float64)
    bias = data["error_mean"].to_numpy(np.float64)
    vmax = max(float(np.nanquantile(np.abs(bias), 0.98)), 1.0)
    xs = pad + (x - x.min()) / max(x.max() - x.min(), 1.0) * (width - 2 * pad)
    ys = height - pad - (y - y.min()) / max(y.max() - y.min(), 1.0) * (height - 2 * pad)
    parts = [
        '<text x="20" y="28" font-size="18" font-family="sans-serif">well XY mean position colored by mean residual</text>',
        f'<text x="20" y="50" font-size="12" font-family="sans-serif">blue = underpredict, red = overpredict, clipped at p98 abs bias {vmax:.2f} ft</text>',
    ]
    for xi, yi, bi, wf, rm in zip(xs, ys, bias, data["well"], data["rmse_tvt"]):
        color = color_scale(float(bi), vmax)
        title = html.escape(f"{wf}: bias={bi:.2f}, rmse={rm:.2f}")
        r = 4.5 if abs(bi) >= vmax else 3.2
        parts.append(f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="{r:.1f}" fill="{color}" stroke="#333" stroke-width="0.35"><title>{title}</title></circle>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="none" stroke="#555"/>')
    write_svg_text(path, "\n".join(parts), width, height)


def write_residual_profile_svg(profile: pd.DataFrame, well: pd.DataFrame, path: Path, top_n: int = 8) -> None:
    top = well.sort_values("abs_bias", ascending=False).head(top_n)
    width, height = 900, 520
    pad_l, pad_r, pad_t, pad_b = 70, 180, 50, 45
    vals = profile.loc[top["well"]].to_numpy(np.float64)
    vmax = max(float(np.nanmax(np.abs(vals))), 1.0)
    xs = np.linspace(pad_l, width - pad_r, profile.shape[1])
    parts = [
        '<text x="20" y="28" font-size="18" font-family="sans-serif">top offset well residual profiles</text>',
        '<line x1="70" y1="260" x2="720" y2="260" stroke="#777"/>',
    ]
    colors = ["#b2182b", "#2166ac", "#ef8a62", "#67a9cf", "#d6604d", "#4393c3", "#762a83", "#1b7837"]
    for i, row in enumerate(top.itertuples(index=False)):
        ys = pad_t + (1 - (profile.loc[row.well].to_numpy(np.float64) + vmax) / (2 * vmax)) * (height - pad_t - pad_b)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
        color = colors[i % len(colors)]
        label = html.escape(f"{row.well} bias {row.error_mean:.1f} rmse {row.rmse_tvt:.1f}")
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/>')
        parts.append(f'<text x="{width - pad_r + 18}" y="{pad_t + 20 + i * 20}" font-size="11" font-family="sans-serif" fill="{color}">{label}</text>')
    parts.append(f'<text x="12" y="{pad_t + 5}" font-size="11" font-family="sans-serif">+{vmax:.1f}</text>')
    parts.append(f'<text x="18" y="{height - pad_b}" font-size="11" font-family="sans-serif">-{vmax:.1f}</text>')
    write_svg_text(path, "\n".join(parts), width, height)


def write_sharp_step_svg(sharp: pd.DataFrame, path: Path) -> None:
    sample = sharp.copy()
    if len(sample) > 6000:
        stride = max(len(sample) // 6000, 1)
        sample = sample.iloc[::stride].copy()
    width, height = 720, 720
    pad = 70
    if sample.empty:
        write_svg_text(path, '<text x="20" y="30" font-size="18">No sharp steps</text>', width, height)
        return
    true = sample["true_step"].to_numpy(np.float64)
    pred = sample["pred_step"].to_numpy(np.float64)
    lim = max(float(np.nanquantile(np.abs(np.r_[true, pred]), 0.995)), 1.0)
    xs = pad + (true + lim) / (2 * lim) * (width - 2 * pad)
    ys = height - pad - (pred + lim) / (2 * lim) * (height - 2 * pad)
    parts = [
        '<text x="20" y="30" font-size="18" font-family="sans-serif">sharp true TVT step vs predicted step</text>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{pad}" stroke="#777" stroke-dasharray="4 4"/>',
        f'<line x1="{pad}" y1="{height/2}" x2="{width - pad}" y2="{height/2}" stroke="#aaa"/>',
        f'<line x1="{width/2}" y1="{pad}" x2="{width/2}" y2="{height - pad}" stroke="#aaa"/>',
    ]
    for xi, yi, d in zip(xs, ys, sample["step_direction"]):
        color = "#b2182b" if d == "up" else "#2166ac"
        parts.append(f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="2.0" fill="{color}" opacity="0.35"/>')
    parts.append(f'<text x="{pad}" y="{height - 20}" font-size="11" font-family="sans-serif">true step -{lim:.1f} to +{lim:.1f}</text>')
    parts.append(f'<text x="15" y="{pad}" font-size="11" font-family="sans-serif">pred step</text>')
    write_svg_text(path, "\n".join(parts), width, height)


def records(df: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    return json.loads(df.head(n).to_json(orient="records"))


def iter_lgb_mean_chunks(path: Path, cfg: dict[str, Any], chunksize: int = 250_000):
    validation = cfg["validation"]
    usecols = [
        "id",
        "well",
        "variant",
        "mode",
        "model",
        "last_known_tvt",
        "target_tvt",
        "pred_tvt",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        mask = (
            chunk["model"].eq(validation["source_prediction_model"])
            & chunk["mode"].eq(validation["source_prediction_mode"])
            & chunk["variant"].eq(validation["source_prediction_variant"])
        )
        part = chunk.loc[mask, ["well", "id", "last_known_tvt", "target_tvt", "pred_tvt"]].copy()
        if part.empty:
            continue
        part["well"] = part["well"].astype(str)
        part["row_idx"] = part["id"].str.rsplit("_", n=1).str[-1].astype(np.int32)
        part = part.drop(columns=["id"])
        part["target_tvt"] = part["target_tvt"].astype(np.float32)
        part["pred_tvt"] = part["pred_tvt"].astype(np.float32)
        part["last_known_tvt"] = part["last_known_tvt"].astype(np.float32)
        part["residual"] = (part["pred_tvt"] - part["target_tvt"]).astype(np.float32)
        part["abs_error"] = np.abs(part["residual"]).astype(np.float32)
        part["sq_error"] = (part["residual"].astype(np.float64) ** 2).astype(np.float64)
        yield part


def step_arrays_for_chunk(
    part: pd.DataFrame, last_state: dict[str, tuple[int, float, float]]
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    ordered = part.sort_values(["well", "row_idx"])
    for well, group in ordered.groupby("well", sort=False):
        row_idx = group["row_idx"].to_numpy(np.int32)
        target = group["target_tvt"].to_numpy(np.float32)
        pred = group["pred_tvt"].to_numpy(np.float32)
        if len(target) == 0:
            continue
        if well in last_state:
            _, prev_target, prev_pred = last_state[well]
            true_step = np.empty(len(target), dtype=np.float32)
            pred_step = np.empty(len(pred), dtype=np.float32)
            true_step[0] = target[0] - prev_target
            pred_step[0] = pred[0] - prev_pred
            if len(target) > 1:
                true_step[1:] = np.diff(target)
                pred_step[1:] = np.diff(pred)
            step_row_idx = row_idx
            step_target = target
            step_pred = pred
        else:
            if len(target) < 2:
                last_state[well] = (int(row_idx[-1]), float(target[-1]), float(pred[-1]))
                continue
            true_step = np.diff(target).astype(np.float32)
            pred_step = np.diff(pred).astype(np.float32)
            step_row_idx = row_idx[1:]
            step_target = target[1:]
            step_pred = pred[1:]
        out[well] = {
            "row_idx": step_row_idx,
            "target_tvt": step_target,
            "pred_tvt": step_pred,
            "true_step": true_step,
            "pred_step": pred_step,
        }
        last_state[well] = (int(row_idx[-1]), float(target[-1]), float(pred[-1]))
    return out


def combine_additive_aggs(parts: list[pd.DataFrame], key: str) -> pd.DataFrame:
    raw = pd.concat(parts, ignore_index=True)
    agg_spec: dict[str, tuple[str, str]] = {
        "rows": ("rows", "sum"),
        "error_sum": ("error_sum", "sum"),
        "abs_error_sum": ("abs_error_sum", "sum"),
        "sq_error_sum": ("sq_error_sum", "sum"),
        "positive_count": ("positive_count", "sum"),
        "negative_count": ("negative_count", "sum"),
        "target_tvt_min": ("target_tvt_min", "min"),
        "target_tvt_max": ("target_tvt_max", "max"),
        "target_tvt_sum": ("target_tvt_sum", "sum"),
        "pred_tvt_min": ("pred_tvt_min", "min"),
        "pred_tvt_max": ("pred_tvt_max", "max"),
        "pred_tvt_sum": ("pred_tvt_sum", "sum"),
    }
    return raw.groupby(key, sort=True).agg(**agg_spec).reset_index()


def finalize_well_summary(base: pd.DataFrame, tw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    well = base.copy()
    well["error_mean"] = well["error_sum"] / well["rows"]
    well["abs_error_mean"] = well["abs_error_sum"] / well["rows"]
    well["rmse_tvt"] = np.sqrt(well["sq_error_sum"] / well["rows"])
    well["target_tvt_mean"] = well["target_tvt_sum"] / well["rows"]
    well["pred_tvt_mean"] = well["pred_tvt_sum"] / well["rows"]
    well["positive_rate"] = well["positive_count"] / well["rows"]
    well["negative_rate"] = well["negative_count"] / well["rows"]
    well["sign_consistency"] = well[["positive_rate", "negative_rate"]].max(axis=1)
    well["abs_bias"] = np.abs(well["error_mean"])
    well["bias_rmse_ratio"] = well["abs_bias"] / well["rmse_tvt"].replace(0, np.nan)
    well["target_tvt_span"] = well["target_tvt_max"] - well["target_tvt_min"]
    well["pred_tvt_span"] = well["pred_tvt_max"] - well["pred_tvt_min"]
    readout = cfg["readout"]
    well["offset_flag"] = (
        (well["abs_bias"] >= float(readout["offset_abs_bias_threshold_ft"]))
        & (well["bias_rmse_ratio"] >= float(readout["offset_bias_rmse_ratio_threshold"]))
    )
    well["offset_direction"] = np.where(
        well["error_mean"] > 0,
        "overpredict",
        np.where(well["error_mean"] < 0, "underpredict", "neutral"),
    )
    well = well.merge(tw, on="well", how="left", validate="one_to_one")
    drop_cols = [
        "error_sum",
        "abs_error_sum",
        "sq_error_sum",
        "positive_count",
        "negative_count",
        "target_tvt_sum",
        "pred_tvt_sum",
    ]
    return well.drop(columns=drop_cols).sort_values("rmse_tvt", ascending=False).reset_index(drop=True)


def finalize_typewell_row_metrics(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy().rename(columns={"typewell_group": "exact_typewell_group"})
    out["error_mean"] = out["error_sum"] / out["rows"]
    out["abs_error_mean"] = out["abs_error_sum"] / out["rows"]
    out["rmse_tvt"] = np.sqrt(out["sq_error_sum"] / out["rows"])
    out["target_tvt_span"] = out["target_tvt_max"] - out["target_tvt_min"]
    return out[
        [
            "exact_typewell_group",
            "rows",
            "error_mean",
            "abs_error_mean",
            "rmse_tvt",
            "target_tvt_min",
            "target_tvt_max",
            "target_tvt_span",
        ]
    ]


def first_streaming_pass(
    pred_path: Path, cfg: dict[str, Any], well_to_typewell: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, list[np.ndarray]], np.ndarray]:
    well_parts: list[pd.DataFrame] = []
    typewell_parts: list[pd.DataFrame] = []
    step_abs_by_well: dict[str, list[np.ndarray]] = {}
    all_abs_step_parts: list[np.ndarray] = []
    last_state: dict[str, tuple[int, float, float]] = {}

    for part in iter_lgb_mean_chunks(pred_path, cfg):
        part["positive_count"] = (part["residual"] > 0).astype(np.int32)
        part["negative_count"] = (part["residual"] < 0).astype(np.int32)
        well_agg = part.groupby("well", sort=True).agg(
            rows=("residual", "size"),
            error_sum=("residual", "sum"),
            abs_error_sum=("abs_error", "sum"),
            sq_error_sum=("sq_error", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            target_tvt_min=("target_tvt", "min"),
            target_tvt_max=("target_tvt", "max"),
            target_tvt_sum=("target_tvt", "sum"),
            pred_tvt_min=("pred_tvt", "min"),
            pred_tvt_max=("pred_tvt", "max"),
            pred_tvt_sum=("pred_tvt", "sum"),
        ).reset_index()
        well_parts.append(well_agg)

        part["typewell_group"] = part["well"].map(well_to_typewell)
        typewell_agg = part.groupby("typewell_group", sort=True).agg(
            rows=("residual", "size"),
            error_sum=("residual", "sum"),
            abs_error_sum=("abs_error", "sum"),
            sq_error_sum=("sq_error", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            target_tvt_min=("target_tvt", "min"),
            target_tvt_max=("target_tvt", "max"),
            target_tvt_sum=("target_tvt", "sum"),
            pred_tvt_min=("pred_tvt", "min"),
            pred_tvt_max=("pred_tvt", "max"),
            pred_tvt_sum=("pred_tvt", "sum"),
        ).reset_index()
        typewell_parts.append(typewell_agg)

        step_payload = step_arrays_for_chunk(part, last_state)
        for well, arrays in step_payload.items():
            abs_true = np.abs(arrays["true_step"]).astype(np.float32)
            if len(abs_true):
                step_abs_by_well.setdefault(well, []).append(abs_true)
                all_abs_step_parts.append(abs_true)

    well_base = combine_additive_aggs(well_parts, "well")
    typewell_base = combine_additive_aggs(typewell_parts, "typewell_group")
    well_counts = dict(zip(well_base["well"], well_base["rows"].astype(int)))
    all_abs_steps = np.concatenate(all_abs_step_parts) if all_abs_step_parts else np.array([], dtype=np.float32)
    return well_base, typewell_base, well_counts, step_abs_by_well, all_abs_steps


def second_streaming_pass(
    pred_path: Path,
    cfg: dict[str, Any],
    well_counts: dict[str, int],
    well_to_typewell: dict[str, str],
    sharp_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = int(cfg["readout"]["residual_profile_bins"])
    wells = sorted(well_counts)
    well_pos = {well: i for i, well in enumerate(wells)}
    profile_sum = np.zeros((len(wells), bins), dtype=np.float64)
    profile_count = np.zeros((len(wells), bins), dtype=np.int32)
    seen_count = {well: 0 for well in wells}
    sharp_parts: list[pd.DataFrame] = []
    last_state: dict[str, tuple[int, float, float]] = {}

    for part in iter_lgb_mean_chunks(pred_path, cfg):
        ordered = part.sort_values(["well", "row_idx"])
        for well, group in ordered.groupby("well", sort=False):
            total = well_counts[well]
            start = seen_count[well]
            n = len(group)
            if total <= 1:
                bin_idx = np.zeros(n, dtype=np.int16)
            else:
                pos = np.arange(start, start + n, dtype=np.float64)
                bin_idx = np.minimum((pos / max(total - 1, 1) * bins).astype(np.int16), bins - 1)
            wi = well_pos[well]
            residual = group["residual"].to_numpy(np.float32)
            np.add.at(profile_sum[wi], bin_idx, residual)
            np.add.at(profile_count[wi], bin_idx, 1)
            seen_count[well] = start + n

        step_payload = step_arrays_for_chunk(part, last_state)
        for well, arrays in step_payload.items():
            abs_true = np.abs(arrays["true_step"])
            mask = abs_true >= sharp_threshold
            if not np.any(mask):
                continue
            sharp = pd.DataFrame(
                {
                    "well": well,
                    "row_idx": arrays["row_idx"][mask].astype(np.int32),
                    "target_tvt": arrays["target_tvt"][mask].astype(np.float32),
                    "pred_tvt": arrays["pred_tvt"][mask].astype(np.float32),
                    "true_step": arrays["true_step"][mask].astype(np.float32),
                    "pred_step": arrays["pred_step"][mask].astype(np.float32),
                }
            )
            sharp["exact_typewell_group"] = well_to_typewell[well]
            sharp_parts.append(sharp)

    with np.errstate(invalid="ignore", divide="ignore"):
        profile_arr = profile_sum / profile_count
    profile_arr[~np.isfinite(profile_arr)] = np.nan
    profile = pd.DataFrame(profile_arr, index=wells, columns=[f"bin_{i:03d}" for i in range(bins)])
    profile = profile.interpolate(axis=1, limit_direction="both").fillna(0.0)
    profile.index.name = "well"
    sharp_df = pd.concat(sharp_parts, ignore_index=True) if sharp_parts else pd.DataFrame()
    if not sharp_df.empty:
        sharp_df["step_direction"] = np.where(sharp_df["true_step"] >= 0, "up", "down")
        sharp_df["step_error"] = sharp_df["pred_step"] - sharp_df["true_step"]
        sharp_df["abs_true_step"] = np.abs(sharp_df["true_step"])
        sharp_df["abs_pred_step"] = np.abs(sharp_df["pred_step"])
        sharp_df["abs_step_error"] = np.abs(sharp_df["step_error"])
        sharp_df["damping_abs_pred_over_true"] = sharp_df["abs_pred_step"] / sharp_df["abs_true_step"].replace(0, np.nan)
    return profile, sharp_df


def finalize_sharp_metrics(
    sharp: pd.DataFrame,
    step_abs_by_well: dict[str, list[np.ndarray]],
    well: pd.DataFrame,
    cfg: dict[str, Any],
    sharp_threshold: float,
    all_abs_steps: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for well_name, parts in sorted(step_abs_by_well.items()):
        arr = np.concatenate(parts)
        rows.append(
            {
                "well": well_name,
                "all_step_count": int(len(arr)),
                "true_step_abs_mean": float(np.mean(arr)),
                "true_step_abs_p95": float(np.quantile(arr, 0.95)),
                "true_step_abs_p99": float(np.quantile(arr, 0.99)),
                "true_step_abs_max": float(np.max(arr)),
            }
        )
    sharp_well = pd.DataFrame(rows)
    if sharp.empty:
        sharp_agg = pd.DataFrame(columns=["well"])
    else:
        sharp_agg = sharp.groupby("well", sort=True).agg(
            sharp_step_count=("true_step", "size"),
            sharp_up_count=("step_direction", lambda s: int(np.sum(s.to_numpy() == "up"))),
            sharp_down_count=("step_direction", lambda s: int(np.sum(s.to_numpy() == "down"))),
            sharp_true_step_abs_mean=("abs_true_step", "mean"),
            sharp_pred_step_abs_mean=("abs_pred_step", "mean"),
            sharp_abs_step_error_mean=("abs_step_error", "mean"),
            sharp_damping_mean=("damping_abs_pred_over_true", "mean"),
            sharp_damping_median=("damping_abs_pred_over_true", "median"),
        ).reset_index()
    sharp_well = sharp_well.merge(sharp_agg, on="well", how="left")
    for col in ["sharp_step_count", "sharp_up_count", "sharp_down_count"]:
        sharp_well[col] = sharp_well[col].fillna(0).astype(int)
    sharp_well["sharp_step_rate"] = sharp_well["sharp_step_count"] / sharp_well["all_step_count"].replace(0, np.nan)
    sharp_well = sharp_well.merge(
        well[["well", "rmse_tvt", "error_mean", "abs_bias", "offset_flag", "exact_typewell_group", "x_mean", "y_mean"]],
        on="well",
        how="left",
        validate="one_to_one",
    )
    sharp_typewell = sharp_well.groupby("exact_typewell_group", sort=True).agg(
        wells=("well", "nunique"),
        wells_with_sharp_steps=("sharp_step_count", lambda s: int(np.sum(s.to_numpy() > 0))),
        sharp_step_count=("sharp_step_count", "sum"),
        sharp_step_rate_mean=("sharp_step_rate", "mean"),
        sharp_damping_mean=("sharp_damping_mean", "mean"),
        sharp_abs_step_error_mean=("sharp_abs_step_error_mean", "mean"),
        rmse_tvt_mean=("rmse_tvt", "mean"),
        abs_bias_mean=("abs_bias", "mean"),
    ).reset_index()
    sharp_typewell["wells_with_sharp_step_rate"] = (
        sharp_typewell["wells_with_sharp_steps"] / sharp_typewell["wells"].replace(0, np.nan)
    )
    summary = {
        "sharp_step_quantile": float(cfg["readout"]["sharp_step_quantile"]),
        "sharp_step_abs_true_threshold": float(sharp_threshold),
        "all_step_rows": int(len(all_abs_steps)),
        "sharp_step_rows": int(len(sharp)),
        "sharp_step_row_rate": safe_div(float(len(sharp)), float(len(all_abs_steps))),
        "sharp_step_wells": int(np.sum(sharp_well["sharp_step_count"].to_numpy() > 0)),
        "sharp_step_damping_mean": float(sharp["damping_abs_pred_over_true"].mean()) if len(sharp) else float("nan"),
        "sharp_step_abs_error_mean": float(sharp["abs_step_error"].mean()) if len(sharp) else float("nan"),
    }
    return sharp_well, sharp_typewell, summary


def main() -> None:
    cfg = load_config()
    out_dir = resolve_first_existing([str(EXP_DIR)]) / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = resolve_first_existing(cfg["data"]["exp148_prediction_candidates"])
    typewell_path = resolve_first_existing(cfg["data"]["typewell_summary_candidates"])

    tw = pd.read_csv(typewell_path)
    tw = tw[
        [
            "well_id",
            "exact_typewell_group",
            "typewell_hash",
            "exact_typewell_group_size",
            "shared_typewell",
            "x_mean",
            "y_mean",
            "x_start",
            "y_start",
            "x_end",
            "y_end",
        ]
    ].rename(columns={"well_id": "well"})
    tw["well"] = tw["well"].astype(str)
    tw["exact_typewell_group"] = tw["exact_typewell_group"].astype(str)
    tw["shared_typewell"] = tw["shared_typewell"].astype(bool)
    well_to_typewell = dict(zip(tw["well"], tw["exact_typewell_group"]))

    readout_cfg = cfg["readout"]
    well_base, typewell_base, well_counts, step_abs_by_well, all_abs_steps = first_streaming_pass(
        pred_path, cfg, well_to_typewell
    )
    sharp_threshold = float(np.quantile(all_abs_steps, float(readout_cfg["sharp_step_quantile"])))
    profile, sharp = second_streaming_pass(pred_path, cfg, well_counts, well_to_typewell, sharp_threshold)

    well = finalize_well_summary(well_base, tw, cfg)
    xy, xy_summary, neighbor_map = compute_xy_neighbors(well, int(readout_cfg["xy_neighbor_k"]))
    group_shape, shape_payload = compute_shape_similarity(well, profile, neighbor_map)
    xy = xy.merge(shape_payload["xy_shape"], on="well", how="left", validate="one_to_one")
    row_typewell = finalize_typewell_row_metrics(typewell_base)
    typewell = row_typewell.merge(
        compute_typewell_metrics_from_well_summary(well),
        on="exact_typewell_group",
        how="left",
        validate="one_to_one",
    )
    typewell["abs_error_mean_minus_global"] = typewell["abs_error_mean"] - (
        well_base["abs_error_sum"].sum() / well_base["rows"].sum()
    )
    typewell = typewell.merge(group_shape, on="exact_typewell_group", how="left", validate="one_to_one")
    typewell = typewell.sort_values(["rmse_tvt", "wells"], ascending=[False, False]).reset_index(drop=True)
    sharp_well, sharp_typewell, sharp_summary = finalize_sharp_metrics(
        sharp, step_abs_by_well, well, cfg, sharp_threshold, all_abs_steps
    )

    offset = well[well["offset_flag"]].sort_values("abs_bias", ascending=False).reset_index(drop=True)
    high_error = well.sort_values("rmse_tvt", ascending=False).head(int(readout_cfg["high_error_top_n"]))

    overall_rmse = float(math.sqrt(float(well_base["sq_error_sum"].sum() / well_base["rows"].sum())))
    overall = {
        "rows": int(well_base["rows"].sum()),
        "wells": int(well_base["well"].nunique()),
        "typewell_groups": int(well["exact_typewell_group"].nunique()),
        "rmse_tvt": overall_rmse,
        "mae_tvt": float(well_base["abs_error_sum"].sum() / well_base["rows"].sum()),
        "bias_mean": float(well_base["error_sum"].sum() / well_base["rows"].sum()),
        "offset_wells": int(len(offset)),
        "offset_well_rate": safe_div(float(len(offset)), float(well["well"].nunique())),
    }

    outputs = {
        "well_summary": out_dir / "well_error_profile_summary.csv",
        "typewell_metrics": out_dir / "typewell_group_metrics.csv",
        "xy_neighbor_metrics": out_dir / "xy_neighbor_bias_similarity.csv",
        "residual_profiles": out_dir / "well_residual_profiles_100bin.csv",
        "offset_wells": out_dir / "offset_wells.csv",
        "high_error_wells": out_dir / "high_error_wells_top30.csv",
        "sharp_steps": out_dir / "tvt_sharp_step_rows.csv",
        "sharp_step_wells": out_dir / "tvt_sharp_step_wells.csv",
        "sharp_step_typewell": out_dir / "tvt_sharp_step_typewell_metrics.csv",
        "typewell_bar_svg": out_dir / "typewell_group_bias_rmse_top.svg",
        "xy_bias_svg": out_dir / "xy_bias_map.svg",
        "offset_profile_svg": out_dir / "offset_residual_profiles.svg",
        "sharp_step_svg": out_dir / "sharp_step_true_vs_pred.svg",
    }
    write_csv(well, outputs["well_summary"])
    write_csv(typewell, outputs["typewell_metrics"])
    write_csv(xy, outputs["xy_neighbor_metrics"])
    write_csv(profile.reset_index(), outputs["residual_profiles"])
    write_csv(offset, outputs["offset_wells"])
    write_csv(high_error, outputs["high_error_wells"])
    write_csv(sharp, outputs["sharp_steps"])
    write_csv(sharp_well.sort_values("sharp_step_count", ascending=False), outputs["sharp_step_wells"])
    write_csv(sharp_typewell.sort_values("sharp_step_count", ascending=False), outputs["sharp_step_typewell"])

    write_typewell_bar_svg(typewell, outputs["typewell_bar_svg"])
    write_xy_bias_svg(well, outputs["xy_bias_svg"])
    write_residual_profile_svg(profile, well, outputs["offset_profile_svg"])
    write_sharp_step_svg(sharp, outputs["sharp_step_svg"])

    top_typewell_bias = typewell.sort_values("error_mean", key=lambda s: np.abs(s), ascending=False)
    top_typewell_rmse = typewell.sort_values("rmse_tvt", ascending=False)
    xy_pockets = xy.merge(
        well[["well", "error_mean", "rmse_tvt", "exact_typewell_group", "offset_flag"]],
        on="well",
        how="left",
        validate="one_to_one",
    ).sort_values("xy_local_bias_mean_abs", ascending=False)

    summary = {
        "experiment": cfg["experiment"]["name"],
        "status": "complete",
        "diagnostic_only": True,
        "inputs": {
            "prediction_path": str(pred_path),
            "prediction_decompressed_sha256": sha256_gzip_decompressed(pred_path)
            if pred_path.suffix == ".gz"
            else sha256_file(pred_path),
            "typewell_summary_path": str(typewell_path),
            "typewell_summary_sha256": sha256_file(typewell_path),
        },
        "filters": cfg["validation"],
        "overall": overall,
        "typewell": {
            "groups": int(typewell["exact_typewell_group"].nunique()),
            "top_abs_bias_groups": records(top_typewell_bias, 15),
            "top_rmse_groups": records(top_typewell_rmse, 15),
        },
        "xy_neighbor": {
            **xy_summary,
            "top_local_bias_pockets": records(xy_pockets, 15),
        },
        "shape_similarity": shape_payload["summary"],
        "sharp_tvt_steps": sharp_summary,
        "offset": {
            "threshold_abs_bias_ft": float(readout_cfg["offset_abs_bias_threshold_ft"]),
            "threshold_bias_rmse_ratio": float(readout_cfg["offset_bias_rmse_ratio_threshold"]),
            "wells": int(len(offset)),
            "top_offset_wells": records(offset, 20),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    summary_path = out_dir / "readout_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_hashes = {}
    for name, path in {**outputs, "readout_summary": summary_path}.items():
        artifact_hashes[name] = sha256_file(path)
    summary["artifact_sha256"] = artifact_hashes
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = EXP_DIR / "metrics.json"
    metrics = {
        "experiment": cfg["experiment"]["name"],
        "route": cfg["experiment"]["route"],
        "status": "diagnostic_readout_complete",
        "diagnostic_only": True,
        "parent": cfg["lineage"]["parent"],
        "overall": overall,
        "xy_neighbor": xy_summary,
        "shape_similarity": shape_payload["summary"],
        "sharp_tvt_steps": sharp_summary,
        "offset_wells": int(len(offset)),
        "artifacts": {name: str(path.relative_to(EXP_DIR)) for name, path in {**outputs, "readout_summary": summary_path}.items()},
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
