from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exact_hmm_smoother import (
    load_lgb_prediction_series,
    load_well,
    list_well_ids,
    resolve_existing_file,
    run_hmm2,
    sha256_gzip_decompressed,
    sha256_path,
    to_jsonable,
)
from settings import ExperimentPaths, get_nested, load_config


EXPERIMENT_NAME = "exp236_exact_hmm_posterior_bimodality_audit"
DECODER_NAMES = ("posterior_mean", "marginal_map", "dominant_mode_conditional_mean")
DISTANCE_BUCKETS = ("000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _hmm_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "step",
        "n_rates",
        "rate_span",
        "sig_r",
        "sig_p",
        "df",
        "emission",
        "lam",
        "sigma_mode",
        "start_sig",
        "r0_sig",
        "band_pad",
        "mom",
        "rate_center",
    )
    hmm = get_nested(config, "model.hmm") or {}
    return {key: hmm[key] for key in keys if key in hmm}


def _lgb_source_config(config: dict[str, Any]) -> dict[str, Any]:
    source = get_nested(config, "model.lgb_emission") or {}
    return {
        "id_column": "id",
        "prediction_column": source.get("prediction_column", "pred_tvt"),
        "model_filter": source.get("model_filter", "lgb_mean"),
        "candidates": list(get_nested(config, "data.exp148_lgb_mean_oof_candidates") or []),
    }


def _distance_bucket(md_since: np.ndarray) -> np.ndarray:
    values = np.asarray(md_since, dtype=np.float64)
    labels = np.full(len(values), "1000_plus", dtype=object)
    labels[values < 1000.0] = "500_1000"
    labels[values < 500.0] = "250_500"
    labels[values < 250.0] = "100_250"
    labels[values < 100.0] = "050_100"
    labels[values < 50.0] = "000_050"
    return labels


def _local_peaks(values: np.ndarray, min_height: float) -> list[int]:
    """Return target-free local maxima, including finite boundary maxima."""
    peaks: list[int] = []
    n = len(values)
    if n == 0:
        return peaks
    if n == 1:
        return [0] if values[0] >= min_height else []
    if values[0] >= values[1] and values[0] >= min_height:
        peaks.append(0)
    for idx in range(1, n - 1):
        if values[idx] >= min_height and values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            peaks.append(idx)
    if values[-1] > values[-2] and values[-1] >= min_height:
        peaks.append(n - 1)
    return peaks


def _analyse_posterior(
    posterior: np.ndarray,
    grid: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Summarise a marginal posterior without accepting target or error inputs."""
    posterior = np.asarray(posterior, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    rows = posterior.shape[0]
    min_height = float(config["min_peak_height"])
    min_top2_mass = float(config["min_top2_mass"])
    min_ratio = float(config["min_top2_to_top1_mass_ratio"])
    min_gap = float(config["min_peak_separation_ft"])
    min_valley_depth = float(config["min_valley_depth"])
    mean_valley_ratio = float(config["mean_valley_density_ratio_max"])

    output: dict[str, np.ndarray] = {
        "peak_count": np.zeros(rows, dtype=np.int16),
        "top1_tvt": np.full(rows, np.nan, dtype=np.float32),
        "top2_tvt": np.full(rows, np.nan, dtype=np.float32),
        "top1_density": np.full(rows, np.nan, dtype=np.float32),
        "top2_density": np.full(rows, np.nan, dtype=np.float32),
        "top1_mass": np.full(rows, np.nan, dtype=np.float32),
        "top2_mass": np.full(rows, np.nan, dtype=np.float32),
        "top2_to_top1_mass_ratio": np.full(rows, np.nan, dtype=np.float32),
        "peak_separation_ft": np.full(rows, np.nan, dtype=np.float32),
        "valley_tvt": np.full(rows, np.nan, dtype=np.float32),
        "valley_density": np.full(rows, np.nan, dtype=np.float32),
        "valley_depth": np.full(rows, np.nan, dtype=np.float32),
        "posterior_entropy": np.full(rows, np.nan, dtype=np.float32),
        "posterior_mean": np.full(rows, np.nan, dtype=np.float32),
        "marginal_map": np.full(rows, np.nan, dtype=np.float32),
        "dominant_mode_conditional_mean": np.full(rows, np.nan, dtype=np.float32),
        "dominant_mode_tvt": np.full(rows, np.nan, dtype=np.float32),
        "dominant_mode_side": np.full(rows, -1, dtype=np.int8),
        "bimodal_flag": np.zeros(rows, dtype=np.int8),
        "mean_in_valley_flag": np.zeros(rows, dtype=np.int8),
    }

    for row_idx, raw in enumerate(posterior):
        total = float(np.sum(raw))
        if not np.isfinite(total) or total <= 0.0:
            continue
        probs = raw / total
        mean = float(np.dot(probs, grid))
        map_idx = int(np.argmax(probs))
        output["posterior_mean"][row_idx] = mean
        output["marginal_map"][row_idx] = grid[map_idx]
        positive = probs[probs > 0.0]
        output["posterior_entropy"][row_idx] = float(-np.sum(positive * np.log(positive)) / math.log(len(grid)))

        peaks = _local_peaks(probs, min_height)
        output["peak_count"][row_idx] = len(peaks)
        if not peaks:
            output["dominant_mode_conditional_mean"][row_idx] = mean
            output["dominant_mode_tvt"][row_idx] = grid[map_idx]
            continue

        ranked = sorted(peaks, key=lambda idx: (-float(probs[idx]), idx))
        top1_idx = ranked[0]
        output["top1_tvt"][row_idx] = grid[top1_idx]
        output["top1_density"][row_idx] = probs[top1_idx]
        output["dominant_mode_conditional_mean"][row_idx] = mean
        output["dominant_mode_tvt"][row_idx] = grid[top1_idx]
        if len(ranked) < 2:
            continue

        top2_idx = ranked[1]
        low_idx, high_idx = sorted((top1_idx, top2_idx))
        valley_idx = low_idx + int(np.argmin(probs[low_idx : high_idx + 1]))
        lower_mass = float(np.sum(probs[: valley_idx + 1]))
        upper_mass = float(np.sum(probs[valley_idx + 1 :]))
        if top1_idx <= valley_idx:
            top1_mass, top2_mass = lower_mass, upper_mass
        else:
            top1_mass, top2_mass = upper_mass, lower_mass
        min_peak_density = float(min(probs[top1_idx], probs[top2_idx]))
        valley_density = float(probs[valley_idx])
        valley_depth = 0.0 if min_peak_density <= 0.0 else 1.0 - valley_density / min_peak_density
        gap = float(abs(grid[top1_idx] - grid[top2_idx]))
        mass_ratio = 0.0 if top1_mass <= 0.0 else float(top2_mass / top1_mass)

        output["top2_tvt"][row_idx] = grid[top2_idx]
        output["top2_density"][row_idx] = probs[top2_idx]
        output["top1_mass"][row_idx] = top1_mass
        output["top2_mass"][row_idx] = top2_mass
        output["top2_to_top1_mass_ratio"][row_idx] = mass_ratio
        output["peak_separation_ft"][row_idx] = gap
        output["valley_tvt"][row_idx] = grid[valley_idx]
        output["valley_density"][row_idx] = valley_density
        output["valley_depth"][row_idx] = valley_depth

        bimodal = (
            top2_mass >= min_top2_mass
            and mass_ratio >= min_ratio
            and gap >= min_gap
            and valley_depth >= min_valley_depth
        )
        if not bimodal:
            continue
        output["bimodal_flag"][row_idx] = 1
        if lower_mass >= upper_mass:
            mode_probs = probs[: valley_idx + 1]
            mode_grid = grid[: valley_idx + 1]
            dominant_side = 0
            dominant_peak = low_idx
        else:
            mode_probs = probs[valley_idx + 1 :]
            mode_grid = grid[valley_idx + 1 :]
            dominant_side = 1
            dominant_peak = high_idx
        mode_mass = float(np.sum(mode_probs))
        conditional_mean = float(np.dot(mode_probs, mode_grid) / mode_mass) if mode_mass > 0.0 else mean
        mean_grid_idx = int(np.argmin(np.abs(grid - mean)))
        mean_density = float(probs[mean_grid_idx])
        is_between = float(grid[low_idx]) <= mean <= float(grid[high_idx])
        valley_limit = valley_density * mean_valley_ratio + 1e-12
        output["dominant_mode_conditional_mean"][row_idx] = conditional_mean
        output["dominant_mode_tvt"][row_idx] = grid[dominant_peak]
        output["dominant_mode_side"][row_idx] = dominant_side
        if is_between and mean_density <= valley_limit:
            output["mean_in_valley_flag"][row_idx] = 1
    return output


def _annotate_mode_segments(frame: pd.DataFrame, allowance_ft: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frame = frame.copy()
    bimodal = frame["bimodal_flag"].to_numpy(dtype=bool)
    sides = frame["dominant_mode_side"].to_numpy(dtype=np.int8)
    centers = frame["dominant_mode_tvt"].to_numpy(dtype=np.float64)
    segment_ids = np.full(len(frame), -1, dtype=np.int32)
    mass_switch = np.zeros(len(frame), dtype=np.int8)
    track_break = np.zeros(len(frame), dtype=np.int8)
    segments: list[dict[str, Any]] = []
    active_start: int | None = None
    segment_id = -1
    for idx in range(len(frame)):
        if not bimodal[idx]:
            if active_start is not None:
                active_start = None
            continue
        if idx == 0 or not bimodal[idx - 1]:
            segment_id += 1
            active_start = idx
        else:
            if sides[idx] != sides[idx - 1]:
                mass_switch[idx] = 1
            if abs(centers[idx] - centers[idx - 1]) > allowance_ft:
                track_break[idx] = 1
        segment_ids[idx] = segment_id

    frame["bimodal_segment_id"] = segment_ids
    frame["mode_mass_switch_from_prev"] = mass_switch
    frame["mode_track_break_from_prev"] = track_break
    for current_id, group in frame[frame["bimodal_segment_id"] >= 0].groupby("bimodal_segment_id", sort=True):
        segments.append(
            {
                "well": str(group["well"].iloc[0]),
                "bimodal_segment_id": int(current_id),
                "rows": int(len(group)),
                "start_id": str(group["id"].iloc[0]),
                "end_id": str(group["id"].iloc[-1]),
                "start_md_since": float(group["md_since"].iloc[0]),
                "end_md_since": float(group["md_since"].iloc[-1]),
                "mean_entropy": float(group["posterior_entropy"].mean()),
                "mean_valley_depth": float(group["valley_depth"].mean()),
                "mean_peak_separation_ft": float(group["peak_separation_ft"].mean()),
                "mean_in_valley_rows": int(group["mean_in_valley_flag"].sum()),
                "mode_mass_switches": int(group["mode_mass_switch_from_prev"].sum()),
                "mode_track_breaks": int(group["mode_track_break_from_prev"].sum()),
            }
        )
    return frame, segments


@dataclass
class MetricAccumulator:
    rows: int = 0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    sum_signed_error: float = 0.0
    within10: int = 0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        if len(truth) == 0:
            return
        error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
        self.rows += int(len(error))
        self.sum_squared_error += float(np.sum(error**2))
        self.sum_absolute_error += float(np.sum(np.abs(error)))
        self.sum_signed_error += float(np.sum(error))
        self.within10 += int(np.sum(np.abs(error) <= 10.0))

    def as_row(self, candidate: str, scope: str, value: str) -> dict[str, Any]:
        return {
            "candidate": candidate,
            "scope": scope,
            "scope_value": value,
            "rows": self.rows,
            "rmse": math.sqrt(self.sum_squared_error / self.rows) if self.rows else None,
            "mae": self.sum_absolute_error / self.rows if self.rows else None,
            "bias": self.sum_signed_error / self.rows if self.rows else None,
            "within10": self.within10 / self.rows if self.rows else None,
        }


def _update_decoder_metrics(
    accumulators: dict[tuple[str, str, str], MetricAccumulator],
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    bucket: np.ndarray,
    bimodal: np.ndarray,
    mean_in_valley: np.ndarray,
    hidden_like: dict[str, bool],
) -> None:
    masks: list[tuple[str, str, np.ndarray]] = [
        ("overall", "all", np.ones(len(truth), dtype=bool)),
        ("bimodal", "false", ~bimodal),
        ("bimodal", "true", bimodal),
        ("mean_in_valley", "false", ~mean_in_valley),
        ("mean_in_valley", "true", mean_in_valley),
    ]
    for bucket_name in DISTANCE_BUCKETS:
        masks.append(("distance_bucket", bucket_name, bucket == bucket_name))
    for split_name, enabled in hidden_like.items():
        if enabled:
            masks.append(("hidden_like", split_name, np.ones(len(truth), dtype=bool)))
    for candidate, prediction in predictions.items():
        for scope, value, mask in masks:
            if not np.any(mask):
                continue
            accumulators[(candidate, scope, value)].update(truth[mask], prediction[mask])


def _step_delta(values: np.ndarray, last_known_tvt: float) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64)
    previous = np.empty(len(prediction), dtype=np.float64)
    previous[0] = last_known_tvt
    if len(prediction) > 1:
        previous[1:] = prediction[:-1]
    return np.abs(prediction - previous)


def _hidden_like_lookup(paths: ExperimentPaths, config: dict[str, Any]) -> dict[str, set[str]]:
    enabled = bool(get_nested(config, "comparison.hidden_like.enabled"))
    if not enabled:
        return {}
    candidates = list(get_nested(config, "data.exp115_fold_assignment_candidates") or [])
    path = resolve_existing_file(paths.root, candidates)
    frame = pd.read_csv(path, dtype={"well_id": str})
    lookups: dict[str, set[str]] = {}
    for name, column in (get_nested(config, "comparison.hidden_like.valid_role_columns") or {}).items():
        if column not in frame.columns:
            raise KeyError(f"hidden-like assignment is missing role column {column!r}: {path}")
        lookups[str(name)] = set(frame.loc[frame[column].astype(str) == "valid", "well_id"].astype(str))
    return lookups


def _append_gzip_csv(frame: pd.DataFrame, path: Path, wrote_header: bool) -> bool:
    frame.to_csv(path, index=False, compression="gzip", mode="a", header=not wrote_header)
    return True


def _well_summary(frame: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "well": str(frame["well"].iloc[0]),
        "rows": int(len(frame)),
        "bimodal_rows": int(frame["bimodal_flag"].sum()),
        "bimodal_rate": float(frame["bimodal_flag"].mean()),
        "mean_in_valley_rows": int(frame["mean_in_valley_flag"].sum()),
        "mean_in_valley_rate": float(frame["mean_in_valley_flag"].mean()),
        "mode_mass_switches": int(frame["mode_mass_switch_from_prev"].sum()),
        "mode_track_breaks": int(frame["mode_track_break_from_prev"].sum()),
        "mean_entropy": float(frame["posterior_entropy"].mean()),
        "mean_valley_depth_bimodal": float(frame.loc[frame["bimodal_flag"] == 1, "valley_depth"].mean()) if int(frame["bimodal_flag"].sum()) else None,
    }
    truth = frame["target_tvt_readout_only"].to_numpy(np.float64)
    for candidate in DECODER_NAMES:
        prediction = frame[candidate].to_numpy(np.float64)
        row[f"{candidate}_rmse"] = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    row["decoder_rmse_gap_mean_minus_dominant"] = row["posterior_mean_rmse"] - row["dominant_mode_conditional_mean_rmse"]
    bimodal = frame["bimodal_flag"].to_numpy(dtype=bool)
    if np.any(bimodal):
        top1 = frame.loc[bimodal, "top1_tvt"].to_numpy(np.float64)
        top2 = frame.loc[bimodal, "top2_tvt"].to_numpy(np.float64)
        y = truth[bimodal]
        row["oracle_top2_mean_abs_error"] = float(np.mean(np.minimum(np.abs(y - top1), np.abs(y - top2))))
        row["oracle_top2_within10"] = float(np.mean(np.minimum(np.abs(y - top1), np.abs(y - top2)) <= 10.0))
    else:
        row["oracle_top2_mean_abs_error"] = None
        row["oracle_top2_within10"] = None
    return row


def _plot_well(
    well: str,
    paths: ExperimentPaths,
    config: dict[str, Any],
    lgb_predictions: pd.Series,
    output_path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    audit = get_nested(config, "posterior_audit") or {}
    horizontal, typewell = load_well(well, paths.train_data_dir)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_index = horizontal.index[~known_mask].to_numpy(np.int64)
    ids = [f"{well}_{int(index)}" for index in eval_index]
    center = lgb_predictions.reindex(ids)
    if int(center.isna().sum()):
        raise ValueError(f"{well} has missing exp148 OOF centers while plotting")
    lgb = get_nested(config, "model.lgb_emission") or {}
    result = run_hmm2(
        horizontal,
        typewell,
        **_hmm_config(config),
        return_post=True,
        lgb_tvt=center.to_numpy(np.float64),
        lgb_sigma=float(lgb["sigma"]),
        lgb_lambda=float(lgb["lambda"]),
        lgb_emission_clip=float(lgb["emission_clip"]),
    )
    local = _analyse_posterior(result["post"], result["grid"], audit)
    marker = np.where((local["mean_in_valley_flag"] == 1) | (local["bimodal_flag"] == 1))[0]
    center_row = int(marker[0]) if len(marker) else int(np.argmax(local["valley_depth"]))
    half = max(1, int(audit["plot_rows_per_well"]) // 2)
    start, end = max(0, center_row - half), min(len(eval_index), center_row + half)
    post = result["post"][start:end]
    grid = result["grid"]
    md = result["md_eval"][start:end]
    truth = horizontal.loc[eval_index[start:end], "TVT"].to_numpy(np.float64)
    figure, axis = plt.subplots(figsize=(12, 6))
    extent = [float(md[0]), float(md[-1]), float(grid[0]), float(grid[-1])]
    image = axis.imshow(post.T, origin="lower", aspect="auto", extent=extent, cmap="magma")
    axis.plot(md, local["posterior_mean"][start:end], label="posterior mean", color="cyan", linewidth=1.4)
    axis.plot(md, local["marginal_map"][start:end], label="marginal MAP", color="lime", linewidth=1.0)
    axis.plot(md, local["dominant_mode_conditional_mean"][start:end], label="dominant conditional mean", color="white", linewidth=1.0)
    axis.plot(md, truth, label="train TVT (readout only)", color="deepskyblue", linestyle="--", linewidth=0.8)
    flagged = local["mean_in_valley_flag"][start:end].astype(bool)
    if np.any(flagged):
        axis.scatter(md[flagged], local["posterior_mean"][start:end][flagged], s=9, color="red", label="mean in valley")
    figure.colorbar(image, ax=axis, label="posterior mass")
    axis.set(title=f"{EXPERIMENT_NAME}: {well}", xlabel="MD since last known TVT", ylabel="TVT")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return {"well": well, "path": str(output_path), "center_row": center_row, "rows_plotted": int(end - start)}


def run_posterior_bimodality_audit() -> dict[str, Any]:
    paths = ExperimentPaths()
    paths.ensure_output_dirs()
    config = load_config()
    audit = get_nested(config, "posterior_audit") or {}
    lgb_config = get_nested(config, "model.lgb_emission") or {}
    source, source_meta = load_lgb_prediction_series(paths.root, "exp148_lgb_mean", _lgb_source_config(config))
    source_path = Path(str(source_meta["path"]))
    hidden_lookup = _hidden_like_lookup(paths, config)
    accumulators: dict[tuple[str, str, str], MetricAccumulator] = defaultdict(MetricAccumulator)
    step_values: dict[str, list[np.ndarray]] = {name: [] for name in DECODER_NAMES}
    oracle_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    well_status: list[dict[str, Any]] = []
    row_path = paths.artifacts_dir / f"{audit['output_prefix']}_row_summary.csv.gz"
    wrote_row_header = False
    started = time.time()
    wells = list_well_ids(paths.train_data_dir)

    for ordinal, well in enumerate(wells, start=1):
        well_started = time.time()
        horizontal, typewell = load_well(well, paths.train_data_dir)
        known_mask = horizontal["TVT_input"].notna().to_numpy()
        eval_index = horizontal.index[~known_mask].to_numpy(np.int64)
        if not len(eval_index):
            well_status.append({"well": well, "status": "skipped_no_eval_rows", "rows": 0})
            continue
        ids = [f"{well}_{int(index)}" for index in eval_index]
        lgb_center = source.reindex(ids)
        missing = int(lgb_center.isna().sum())
        if missing:
            examples = lgb_center[lgb_center.isna()].index[:5].tolist()
            raise ValueError(f"{well}: exp148 lgb_mean OOF missing {missing} eval ids, examples={examples}")
        result = run_hmm2(
            horizontal,
            typewell,
            **_hmm_config(config),
            return_post=True,
            lgb_tvt=lgb_center.to_numpy(np.float64),
            lgb_sigma=float(lgb_config["sigma"]),
            lgb_lambda=float(lgb_config["lambda"]),
            lgb_emission_clip=float(lgb_config["emission_clip"]),
        )
        target_free = _analyse_posterior(result["post"], result["grid"], audit)
        known = horizontal.loc[known_mask]
        last_known_tvt = float(known["TVT_input"].iloc[-1])
        last_md = float(known["MD"].iloc[-1])
        md_since = horizontal.loc[eval_index, "MD"].to_numpy(np.float64) - last_md
        row_frame = pd.DataFrame(
            {
                "id": ids,
                "well": well,
                "md_since": md_since.astype(np.float32),
                "last_known_tvt": np.full(len(ids), last_known_tvt, dtype=np.float32),
                **target_free,
            }
        )
        row_frame, segments = _annotate_mode_segments(row_frame, float(audit["mode_track_allowance_ft"]))
        truth = horizontal.loc[eval_index, "TVT"].to_numpy(np.float64)
        row_frame["target_tvt_readout_only"] = truth.astype(np.float32)
        bucket = _distance_bucket(md_since)
        row_frame["distance_bucket"] = bucket
        predictions = {name: row_frame[name].to_numpy(np.float64) for name in DECODER_NAMES}
        bimodal = row_frame["bimodal_flag"].to_numpy(dtype=bool)
        valley = row_frame["mean_in_valley_flag"].to_numpy(dtype=bool)
        active_hidden = {name: well in valid_wells for name, valid_wells in hidden_lookup.items()}
        _update_decoder_metrics(accumulators, truth, predictions, bucket, bimodal, valley, active_hidden)
        for candidate, prediction in predictions.items():
            step_values[candidate].append(_step_delta(prediction, last_known_tvt))
        if np.any(bimodal):
            min_abs = np.minimum(
                np.abs(truth[bimodal] - row_frame.loc[bimodal, "top1_tvt"].to_numpy(np.float64)),
                np.abs(truth[bimodal] - row_frame.loc[bimodal, "top2_tvt"].to_numpy(np.float64)),
            )
            oracle_rows.append(
                {
                    "well": well,
                    "rows": int(len(min_abs)),
                    "mean_abs_error": float(np.mean(min_abs)),
                    "within_fixed_threshold": int(np.sum(min_abs <= float(audit["oracle_top2_within_ft"]))),
                }
            )
        well_rows.append(_well_summary(row_frame))
        segment_rows.extend(segments)
        if bool(audit.get("write_row_summary", True)):
            wrote_row_header = _append_gzip_csv(row_frame, row_path, wrote_row_header)
        well_status.append(
            {
                "well": well,
                "status": "ok",
                "ordinal": ordinal,
                "rows": int(len(row_frame)),
                "grid_size": int(len(result["grid"])),
                "elapsed_seconds": round(time.time() - well_started, 3),
            }
        )
        del result, target_free, row_frame

    decoder_metrics = pd.DataFrame(
        [accumulator.as_row(candidate, scope, value) for (candidate, scope, value), accumulator in accumulators.items()]
    ).sort_values(["scope", "scope_value", "candidate"], ignore_index=True)
    overall_metrics = decoder_metrics[decoder_metrics["scope"] == "overall"].copy()
    distance_metrics = decoder_metrics[decoder_metrics["scope"] == "distance_bucket"].copy()
    hidden_metrics = decoder_metrics[decoder_metrics["scope"] == "hidden_like"].copy()
    well_summary = pd.DataFrame(well_rows).sort_values("well", ignore_index=True)
    segment_summary = pd.DataFrame(segment_rows).sort_values(["well", "bimodal_segment_id"], ignore_index=True) if segment_rows else pd.DataFrame()
    status_frame = pd.DataFrame(well_status).sort_values("well", ignore_index=True)
    step_rows: list[dict[str, Any]] = []
    for candidate, chunks in step_values.items():
        values = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
        row: dict[str, Any] = {
            "candidate": candidate,
            "rows": int(len(values)),
            "abs_step_delta_mean": float(np.mean(values)) if len(values) else None,
            "abs_step_delta_p95": float(np.quantile(values, 0.95)) if len(values) else None,
            "abs_step_delta_p99": float(np.quantile(values, 0.99)) if len(values) else None,
        }
        for threshold in get_nested(config, "comparison.step_delta_thresholds") or []:
            row[f"rate_abs_step_delta_gt_{str(threshold).replace('.', 'p')}"] = float(np.mean(values > float(threshold))) if len(values) else None
        step_rows.append(row)
    step_metrics = pd.DataFrame(step_rows)
    oracle = pd.DataFrame(oracle_rows)
    oracle_summary = {
        "scope": "bimodal_rows_only_diagnostic",
        "wells": int(oracle["well"].nunique()) if len(oracle) else 0,
        "rows": int(oracle["rows"].sum()) if len(oracle) else 0,
        "mean_abs_error": float(np.average(oracle["mean_abs_error"], weights=oracle["rows"])) if len(oracle) else None,
        f"within{int(audit['oracle_top2_within_ft'])}": float(oracle["within_fixed_threshold"].sum() / oracle["rows"].sum()) if len(oracle) else None,
    }
    plot_index = well_summary.sort_values(
        ["mean_in_valley_rows", "bimodal_rows", "well"],
        ascending=[False, False, True],
        ignore_index=True,
    ).head(int(audit["plot_well_limit"])).copy()
    plot_index["plot_rank"] = np.arange(1, len(plot_index) + 1, dtype=np.int16)

    prefix = str(audit["output_prefix"])
    paths_by_name = {
        "row_summary": row_path,
        "segment_summary": paths.artifacts_dir / f"{prefix}_segment_summary.csv",
        "well_summary": paths.artifacts_dir / f"{prefix}_well_summary.csv",
        "decoder_metrics": paths.artifacts_dir / f"{prefix}_decoder_metrics.csv",
        "distance_bucket_metrics": paths.artifacts_dir / f"{prefix}_distance_bucket_metrics.csv",
        "hidden_like_metrics": paths.artifacts_dir / f"{prefix}_hidden_like_metrics.csv",
        "step_delta_rates": paths.artifacts_dir / f"{prefix}_step_delta_rates.csv",
        "oracle_top2_metrics": paths.artifacts_dir / f"{prefix}_oracle_top2_metrics.csv",
        "plot_index": paths.artifacts_dir / f"{prefix}_plot_index.csv",
        "well_status": paths.artifacts_dir / f"{prefix}_well_status.csv",
        "summary": paths.artifacts_dir / f"{prefix}_summary.json",
    }
    segment_summary.to_csv(paths_by_name["segment_summary"], index=False)
    well_summary.to_csv(paths_by_name["well_summary"], index=False)
    decoder_metrics.to_csv(paths_by_name["decoder_metrics"], index=False)
    distance_metrics.to_csv(paths_by_name["distance_bucket_metrics"], index=False)
    hidden_metrics.to_csv(paths_by_name["hidden_like_metrics"], index=False)
    step_metrics.to_csv(paths_by_name["step_delta_rates"], index=False)
    pd.DataFrame([oracle_summary]).to_csv(paths_by_name["oracle_top2_metrics"], index=False)
    plot_index.to_csv(paths_by_name["plot_index"], index=False)
    status_frame.to_csv(paths_by_name["well_status"], index=False)

    plot_rows: list[dict[str, Any]] = []
    for _, row in plot_index.iterrows():
        plot_path = paths.artifacts_dir / f"{prefix}_posterior_{row['well']}.png"
        plot_rows.append(_plot_well(str(row["well"]), paths, config, source, plot_path))
    if plot_rows:
        plot_manifest = paths.artifacts_dir / f"{prefix}_plot_manifest.csv"
        pd.DataFrame(plot_rows).to_csv(plot_manifest, index=False)
        paths_by_name["plot_manifest"] = plot_manifest

    source_sha = sha256_path(source_path)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_posterior_diagnostic",
        "route": "ensemble",
        "rows": int(status_frame.loc[status_frame["status"] == "ok", "rows"].sum()),
        "wells": int((status_frame["status"] == "ok").sum()),
        "elapsed_seconds": round(time.time() - started, 3),
        "fixed_parent_variant": {
            "parent": "exp221_lgb_oof_gaussian_emission_hmm_on_exp148",
            "candidate": "hmm_lgb_exp148_lgb_mean_s2000_l0500",
            "hmm": _hmm_config(config),
            "lgb_emission": lgb_config,
        },
        "posterior_audit": audit,
        "source": {
            **source_meta,
            "raw_sha256": source_sha,
            "decompressed_content_sha256": sha256_gzip_decompressed(source_path) if source_path.suffix == ".gz" else None,
        },
        "bimodality": {
            "rows": int(well_summary["rows"].sum()),
            "bimodal_rows": int(well_summary["bimodal_rows"].sum()),
            "mean_in_valley_rows": int(well_summary["mean_in_valley_rows"].sum()),
            "mode_mass_switches": int(well_summary["mode_mass_switches"].sum()),
            "mode_track_breaks": int(well_summary["mode_track_breaks"].sum()),
        },
        "overall_decoder_metrics": overall_metrics.to_dict(orient="records"),
        "oracle_top2_diagnostic": oracle_summary,
        "hidden_like_metrics_available": bool(len(hidden_metrics)),
        "artifacts": {name: str(path) for name, path in paths_by_name.items()},
        "sha256": {},
    }
    for name, path in paths_by_name.items():
        if path.exists() and path != paths_by_name["summary"]:
            summary["sha256"][name] = sha256_path(path)
            if path.suffix == ".gz":
                summary["sha256"][f"{name}_decompressed"] = sha256_gzip_decompressed(path)
    write_json(paths_by_name["summary"], summary)
    summary["sha256"]["summary"] = sha256_path(paths_by_name["summary"])
    write_json(paths_by_name["summary"], summary)
    primary = overall_metrics.loc[
        overall_metrics["candidate"].astype(str) == str(get_nested(config, "comparison.primary_decoder"))
    ]
    primary_row = primary.iloc[0].to_dict() if len(primary) else {}
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_posterior_diagnostic",
        "route": "ensemble",
        "metric": "train_side_fixed_exp221_hmm_decoder_rmse",
        "cv": primary_row.get("rmse"),
        "public_lb": None,
        "private_lb": None,
        "submitted": False,
        "rows": summary["rows"],
        "wells": summary["wells"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "fixed_parent_variant": summary["fixed_parent_variant"],
        "bimodality": summary["bimodality"],
        "oracle_top2_diagnostic": oracle_summary,
        "summary_sha256": summary["sha256"]["summary"],
        "input_oof_decompressed_sha256": summary["source"]["decompressed_content_sha256"],
        "kernel": None,
        "kernel_version": None,
    }
    paths.metrics_path.write_text(json.dumps(to_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run_posterior_bimodality_audit()
