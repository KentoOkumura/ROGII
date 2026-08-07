from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import hmm_exp226_candidate_selector_on_exp183 as parent
import numpy as np
import pandas as pd
from settings import ExperimentPaths, get_nested, load_config
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp248_candidate_perturbation_augmentation_for_likelihood_ranker"
PROTECTED_LONG_COLUMNS = {
    "id",
    "well",
    "is_oracle",
    "true_tvt",
    "target",
    "abs_error",
    "within_10ft",
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sha256_path(path: str | Path, *, decompressed: bool = False) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64))))


def _sample_sorted_rows(
    row_indices: np.ndarray,
    limit: int | None,
    *,
    seed: int,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if limit is None or len(row_indices) <= int(limit):
        return np.sort(row_indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_indices, size=int(limit), replace=False))


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    return parent._row_indices_from_ids(ids)


def _family_indices(candidate_names: list[str], config: dict[str, Any]) -> dict[str, np.ndarray]:
    configured = get_nested(config, "augmentation.family_groups") or {}
    result: dict[str, np.ndarray] = {}
    for family, names in configured.items():
        indices = [
            candidate_names.index(str(name)) for name in names if str(name) in candidate_names
        ]
        if indices:
            result[str(family)] = np.asarray(indices, dtype=np.int16)
    if not result:
        raise ValueError("augmentation.family_groups produced no candidate indices")
    return result


def _load_horizontal_well(
    well: str,
    *,
    train_dir: Path,
    rolling_window: int,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if well in cache:
        return cache[well]
    path = train_dir / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(f"raw train horizontal well file not found: {path}")
    horizontal = pd.read_csv(path, usecols=["GR", "TVT_input"])
    tvt = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    known = tvt.notna().to_numpy()
    if not known.any():
        raise ValueError(f"well {well} has no known TVT_input prefix")
    prefix_len = int(np.flatnonzero(known)[-1] + 1)
    prefix_tvt = (
        tvt.iloc[:prefix_len]
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .to_numpy(np.float32)
    )
    gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    fallback = float(gr.iloc[:prefix_len].mean())
    if not np.isfinite(fallback):
        fallback = float(gr.mean()) if np.isfinite(float(gr.mean())) else 0.0
    full_gr = (
        gr.interpolate(limit_direction="both")
        .fillna(fallback)
        .rolling(int(rolling_window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )
    if not np.isfinite(prefix_tvt).all() or not np.isfinite(full_gr).all():
        raise ValueError(f"well {well} has non-finite multi-observation inputs")
    cache[well] = (prefix_tvt, full_gr)
    return cache[well]


def _nearest_prefix_indices(prefix_tvt: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, candidate_tvt, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - candidate_tvt) < np.abs(
        sorted_tvt[left] - candidate_tvt
    )
    return order[np.where(choose_right, right, left)].astype(np.int32)


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / (values.std(axis=1, keepdims=True) + 1e-6)


def _multiobs_scores_for_well(
    *,
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    offsets: np.ndarray,
    gr_scale: float,
    out_of_range_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_candidates = candidate_values.shape
    nearest = _nearest_prefix_indices(prefix_tvt, candidate_values.reshape(-1)).reshape(
        n_rows, n_candidates
    )
    eval_vectors: list[np.ndarray] = []
    candidate_vectors: list[np.ndarray] = []
    for offset in offsets:
        eval_idx = np.clip(row_idx + int(offset), 0, len(full_gr) - 1)
        prefix_idx = np.clip(nearest + int(offset), 0, len(full_gr) - 1)
        eval_vectors.append(full_gr[eval_idx])
        candidate_vectors.append(full_gr[prefix_idx])
    eval_matrix = np.stack(eval_vectors, axis=1).astype(np.float32)
    candidate_tensor = np.stack(candidate_vectors, axis=2).astype(np.float32)
    diff_mae = np.mean(np.abs(candidate_tensor - eval_matrix[:, None, :]), axis=2)
    eval_norm = _standardize_rows(eval_matrix)
    flat = candidate_tensor.reshape(n_rows * n_candidates, len(offsets))
    candidate_norm = _standardize_rows(flat).reshape(n_rows, n_candidates, len(offsets))
    ncc = np.mean(candidate_norm * eval_norm[:, None, :], axis=2)
    low = float(np.min(prefix_tvt))
    high = float(np.max(prefix_tvt))
    range_distance = np.maximum(0.0, low - candidate_values) + np.maximum(
        0.0, candidate_values - high
    )
    range_penalty = np.exp(-range_distance / max(out_of_range_scale, 1e-6))
    mae_score = np.exp(-diff_mae / max(gr_scale, 1e-6))
    ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)
    score = np.clip(mae_score * (0.25 + 0.75 * ncc_score) * range_penalty, 0.0, 1.0)
    return score.astype(np.float32), diff_mae.astype(np.float32), ncc.astype(np.float32)


def recompute_multi_observation_features(
    view: pd.DataFrame,
    candidate_values: np.ndarray,
    availability: np.ndarray,
    *,
    candidate_names: list[str],
    config: dict[str, Any],
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    settings = get_nested(config, "augmentation.multi_observation_likelihood") or {}
    paths = ExperimentPaths()
    train_dir = paths.train_data_dir
    offsets = np.asarray(settings.get("observation_offsets", [-24, -12, 0, 12, 24]), dtype=np.int32)
    rolling_window = int(settings.get("gr_rolling_window", 5))
    gr_scale = float(settings.get("gr_scale", 18.0))
    range_scale = float(settings.get("out_of_range_scale", 80.0))
    n_rows, n_candidates = candidate_values.shape
    scores = np.empty((n_rows, n_candidates), dtype=np.float32)
    maes = np.empty_like(scores)
    nccs = np.empty_like(scores)
    row_idx_all = _row_indices_from_ids(view["id"])
    for well, positions in view.groupby("well", sort=False).groups.items():
        pos = np.asarray(list(positions), dtype=np.int64)
        prefix_tvt, full_gr = _load_horizontal_well(
            str(well),
            train_dir=train_dir,
            rolling_window=rolling_window,
            cache=raw_cache,
        )
        if row_idx_all[pos].max(initial=0) >= len(full_gr):
            raise ValueError(f"row index is outside raw horizontal file for well {well}")
        score, obs_mae, obs_ncc = _multiobs_scores_for_well(
            full_gr=full_gr,
            prefix_tvt=prefix_tvt,
            row_idx=row_idx_all[pos],
            candidate_values=candidate_values[pos],
            offsets=offsets,
            gr_scale=gr_scale,
            out_of_range_scale=range_scale,
        )
        scores[pos], maes[pos], nccs[pos] = score, obs_mae, obs_ncc

    masked_score = np.where(availability, scores, -np.inf)
    best = np.argmax(masked_score, axis=1)
    best_score = masked_score[np.arange(n_rows), best]
    if not np.isfinite(best_score).all():
        raise ValueError("an augmented view has no available candidate")
    sorted_score = np.sort(masked_score, axis=1)
    second = sorted_score[:, -2] if n_candidates > 1 else np.zeros(n_rows, dtype=np.float32)
    available_count = availability.sum(axis=1).astype(np.float32)
    safe_score_sum = np.where(availability, scores, 0.0).sum(axis=1)
    generated: dict[str, np.ndarray] = {
        "multiobs_top1": candidate_values[np.arange(n_rows), best].astype(np.float32),
        "multiobs_score_max": best_score.astype(np.float32),
        "multiobs_score_mean": (safe_score_sum / available_count).astype(np.float32),
        "multiobs_score_gap": np.where(available_count > 1, best_score - second, best_score).astype(
            np.float32
        ),
        "multiobs_top1_source_id": best.astype(np.float32),
        "multiobs_top1_mae": maes[np.arange(n_rows), best].astype(np.float32),
        "multiobs_top1_ncc": nccs[np.arange(n_rows), best].astype(np.float32),
    }
    for candidate_idx, name in enumerate(candidate_names):
        generated[f"multiobs_score_{name}"] = scores[:, candidate_idx]
        generated[f"multiobs_mae_{name}"] = maes[:, candidate_idx]
        generated[f"multiobs_ncc_{name}"] = nccs[:, candidate_idx]
    for temperature in settings.get("softmax_temperatures", [0.15, 0.30]):
        temp = float(temperature)
        logits = np.where(availability, scores / max(temp, 1e-6), -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits) * availability
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9
        tag = str(temp).replace(".", "p")
        generated[f"multiobs_softmax_t{tag}"] = np.sum(candidate_values * weights, axis=1).astype(
            np.float32
        )
    if "likpf_mean" in candidate_names:
        likpf = candidate_values[:, candidate_names.index("likpf_mean")]
        for weight in settings.get("likpf_blend_weights", [0.25, 0.50]):
            alpha = float(weight)
            tag = str(alpha).replace(".", "p")
            generated[f"likpf_multiobs_blend_w{tag}"] = (
                (1.0 - alpha) * likpf + alpha * generated["multiobs_top1"]
            ).astype(np.float32)
    view = view.drop(columns=[column for column in generated if column in view.columns])
    view = pd.concat([view, pd.DataFrame(generated, index=view.index)], axis=1)
    return view, scores, maes, nccs


def assemble_parent_candidate_surface(
    *,
    cache_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    max_rows: int | None = None,
) -> tuple[
    pd.DataFrame,
    list[Any],
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, Any],
]:
    config = load_config()
    config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
    candidates = parent.candidate_specs_from_config(config)
    required = parent.build_required_columns(config, candidates)
    frame, source_meta = parent.load_train_feature_cache(
        cache_path=cache_path,
        schema_path=schema_path,
        required_columns=required,
        max_rows=max_rows,
    )
    frame, enrichment_columns, enrichment_meta = parent.add_feature_enrichment(
        frame, config, max_rows=max_rows
    )
    frame, cluster_columns, cluster_meta = parent.add_cluster_prior_confidence_features(
        frame, config, max_rows=max_rows
    )
    frame, external_columns, external_meta = parent.add_hmm_exp226_candidate_sources(frame, config)
    frame, engineered_columns, candidate_values, oracle_labels = (
        parent.add_candidate_labels_and_features(
            frame,
            candidates,
            include_candidate_values=bool(get_nested(config, "ranker.include_candidate_values")),
        )
    )
    all_features = parent.select_numeric_feature_columns(
        frame,
        config,
        [*engineered_columns, *enrichment_columns, *cluster_columns, *external_columns],
    )
    exclude_prefixes = tuple(
        str(value)
        for value in get_nested(
            config, "augmentation.candidate_context.exclude_parent_generated_prefixes"
        )
        or []
    )
    excluded = set(
        str(value)
        for value in get_nested(config, "augmentation.candidate_context.exclude_parent_columns")
        or []
    )
    excluded.update(
        str(value)
        for value in get_nested(config, "ranker.feature_enrichment.base_feature_columns") or []
    )
    feature_columns = [
        column
        for column in all_features
        if column not in excluded and not (exclude_prefixes and column.startswith(exclude_prefixes))
    ]
    meta = {
        "base": source_meta,
        "enrichment": enrichment_meta,
        "cluster": cluster_meta,
        "external": external_meta,
        "parent_feature_count": len(all_features),
        "selected_feature_count": len(feature_columns),
        "excluded_feature_count": len(all_features) - len(feature_columns),
        "excluded_columns": sorted(set(all_features).difference(feature_columns)),
    }
    return frame, candidates, candidate_values, oracle_labels, feature_columns, meta


def build_augmented_candidate_view(
    frame: pd.DataFrame,
    candidate_values: np.ndarray,
    *,
    candidate_names: list[str],
    fold: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n_rows, n_candidates = candidate_values.shape
    values = candidate_values.astype(np.float32, copy=True)
    availability = np.ones((n_rows, n_candidates), dtype=bool)
    transforms = [
        str(value) for value in get_nested(config, "augmentation.enabled_transforms") or []
    ]
    if not transforms:
        raise ValueError("augmentation.enabled_transforms must not be empty")
    rng = np.random.default_rng(
        stable_seed(OUTPUT_PREFIX, "augmentation", fold, get_nested(config, "reproducibility.seed"))
    )
    transform_idx = rng.integers(0, len(transforms), size=n_rows)
    chosen_candidate = rng.integers(0, n_candidates, size=n_rows)
    shift_grid = np.asarray(get_nested(config, "augmentation.shift_grid_ft"), dtype=np.float32)
    drift_grid = np.asarray(
        get_nested(config, "augmentation.drift.amplitude_grid_ft"), dtype=np.float32
    )
    spread_grid = np.asarray(get_nested(config, "augmentation.spread_scale_grid"), dtype=np.float32)
    shift = shift_grid[rng.integers(0, len(shift_grid), size=n_rows)]
    drift_amplitude = drift_grid[rng.integers(0, len(drift_grid), size=n_rows)]
    spread_scale = spread_grid[rng.integers(0, len(spread_grid), size=n_rows)]
    family_map = _family_indices(candidate_names, config)
    family_names = list(family_map)
    chosen_family = rng.integers(0, len(family_names), size=n_rows)
    applied_amplitude = np.zeros(n_rows, dtype=np.float32)
    source_candidate = np.full(n_rows, -1, dtype=np.int16)
    source_family = np.full(n_rows, "", dtype=object)

    for transform_position, transform in enumerate(transforms):
        rows = np.flatnonzero(transform_idx == transform_position)
        if not len(rows):
            continue
        if transform == "fixed_shift":
            cand = chosen_candidate[rows]
            values[rows, cand] += shift[rows]
            source_candidate[rows] = cand.astype(np.int16)
            applied_amplitude[rows] = shift[rows]
        elif transform == "common_datum_shift":
            values[rows] += shift[rows, None]
            applied_amplitude[rows] = shift[rows]
        elif transform == "low_frequency_drift":
            cand = chosen_candidate[rows]
            row_index = _row_indices_from_ids(frame.iloc[rows]["id"]).astype(np.float32)
            eval_len = (
                pd.to_numeric(frame.iloc[rows].get("eval_len", 1.0), errors="coerce")
                .fillna(1.0)
                .to_numpy(np.float32)
            )
            phase = np.clip((row_index + 1.0) / np.maximum(eval_len, 1.0), 0.0, 1.0)
            ramp = 0.5 - 0.5 * np.cos(np.pi * phase)
            delta = drift_amplitude[rows] * ramp
            values[rows, cand] += delta
            source_candidate[rows] = cand.astype(np.int16)
            applied_amplitude[rows] = delta.astype(np.float32)
        elif transform == "candidate_dropout":
            cand = chosen_candidate[rows]
            availability[rows, cand] = False
            source_candidate[rows] = cand.astype(np.int16)
        elif transform == "family_dropout":
            for row in rows:
                family = family_names[int(chosen_family[row])]
                availability[row, family_map[family]] = False
                source_family[row] = family
        elif transform == "target_free_top_dropout":
            score_matrix = np.full((len(rows), n_candidates), -np.inf, dtype=np.float32)
            for candidate_idx, name in enumerate(candidate_names):
                column = f"multiobs_score_{name}"
                if column in frame.columns:
                    score_matrix[:, candidate_idx] = (
                        pd.to_numeric(frame.iloc[rows][column], errors="coerce")
                        .fillna(-np.inf)
                        .to_numpy(np.float32)
                    )
            top = np.argmax(score_matrix, axis=1)
            availability[rows, top] = False
            source_candidate[rows] = top.astype(np.int16)
        elif transform == "spread_scale":
            center = np.median(values[rows], axis=1)
            values[rows] = center[:, None] + spread_scale[rows, None] * (
                values[rows] - center[:, None]
            )
            applied_amplitude[rows] = spread_scale[rows]
        else:
            raise ValueError(f"unsupported augmentation transform: {transform}")

    minimum = int(get_nested(config, "augmentation.min_available_candidates") or 1)
    invalid = np.flatnonzero(availability.sum(axis=1) < minimum)
    if len(invalid):
        availability[invalid, candidate_names.index("likpf_mean")] = True
    if not np.isfinite(values).all():
        raise ValueError("augmented candidate values contain non-finite values")
    inventory = pd.DataFrame(
        {
            "fold": np.int16(fold),
            "id": frame["id"].astype(str).to_numpy(),
            "well": frame["well"].astype(str).to_numpy(),
            "transform": np.asarray(transforms, dtype=object)[transform_idx],
            "source_candidate": [
                candidate_names[idx] if idx >= 0 else "" for idx in source_candidate
            ],
            "source_family": source_family,
            "applied_amplitude": applied_amplitude,
            "available_candidates": availability.sum(axis=1).astype(np.int16),
        }
    )
    return values, availability, inventory


def _view_candidate_context(
    candidate_values: np.ndarray,
    availability: np.ndarray,
    score_matrix: np.ndarray,
    *,
    family_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    masked_values = np.where(availability, candidate_values, np.nan)
    mean = np.nanmean(masked_values, axis=1).astype(np.float32)
    std = np.nanstd(masked_values, axis=1).astype(np.float32)
    minimum = np.nanmin(masked_values, axis=1)
    maximum = np.nanmax(masked_values, axis=1)
    safe_std = np.maximum(std, 1.0)
    masked_score = np.where(availability, score_matrix, -np.inf)
    score_best = np.max(masked_score, axis=1)
    context: dict[str, np.ndarray] = {
        "view_candidate_count": availability.sum(axis=1).astype(np.float32),
        "view_candidate_mean": mean,
        "view_candidate_std": std,
        "view_candidate_range": (maximum - minimum).astype(np.float32),
        "view_score_best": score_best.astype(np.float32),
        "view_candidate_std_safe": safe_std.astype(np.float32),
    }
    for family, indices in family_map.items():
        context[f"view_{family}_available_count"] = (
            availability[:, indices].sum(axis=1).astype(np.float32)
        )
    return context


def build_candidate_long_view(
    source_frame: pd.DataFrame,
    candidate_values: np.ndarray,
    availability: np.ndarray,
    *,
    candidates: list[Any],
    base_feature_columns: list[str],
    config: dict[str, Any],
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    candidate_names = [item.name for item in candidates]
    view = source_frame.copy().reset_index(drop=True)
    last_known = view["last_known_tvt"].to_numpy(np.float32)
    for candidate_idx, spec in enumerate(candidates):
        view[spec.column] = candidate_values[:, candidate_idx]
    delta_map = get_nested(config, "augmentation.candidate_context.recompute_delta_columns") or {}
    for output_column, candidate_name in delta_map.items():
        if str(candidate_name) in candidate_names:
            view[str(output_column)] = (
                candidate_values[:, candidate_names.index(str(candidate_name))] - last_known
            ).astype(np.float32)
    view, engineered, values_check, view_oracle = parent.add_candidate_labels_and_features(
        view, candidates, include_candidate_values=False
    )
    if not np.array_equal(values_check, candidate_values):
        raise ValueError("candidate values changed while rebuilding view context")
    view, score_matrix, _mae_matrix, _ncc_matrix = recompute_multi_observation_features(
        view,
        candidate_values,
        availability,
        candidate_names=candidate_names,
        config=config,
        raw_cache=raw_cache,
    )
    feature_columns = list(dict.fromkeys([*base_feature_columns, *engineered]))
    missing = [column for column in feature_columns if column not in view.columns]
    if missing:
        raise ValueError(f"view feature columns are missing: {missing}")
    long_frame, y_error = parent.build_long_frame(
        view,
        np.arange(len(view), dtype=np.int64),
        candidates,
        row_feature_columns=feature_columns,
        candidate_values=candidate_values,
        oracle_labels=view_oracle,
        sample_rows=None,
        seed=int(get_nested(config, "reproducibility.seed") or 42),
        config=config,
    )
    keep_mask = np.concatenate([availability[:, idx] for idx in range(len(candidates))])
    context = _view_candidate_context(
        candidate_values,
        availability,
        score_matrix,
        family_map=_family_indices(candidate_names, config),
    )
    candidate_context_parts: dict[str, np.ndarray] = {}
    for name, values in context.items():
        candidate_context_parts[name] = np.tile(values, len(candidates))
    row_mean = context["view_candidate_mean"]
    row_std = context["view_candidate_std_safe"]
    candidate_context_parts["candidate_abs_minus_view_mean"] = np.concatenate(
        [np.abs(candidate_values[:, idx] - row_mean) for idx in range(len(candidates))]
    ).astype(np.float32)
    candidate_context_parts["candidate_z_within_view"] = np.concatenate(
        [(candidate_values[:, idx] - row_mean) / row_std for idx in range(len(candidates))]
    ).astype(np.float32)
    candidate_context_parts["candidate_score_gap_from_view_best"] = np.concatenate(
        [context["view_score_best"] - score_matrix[:, idx] for idx in range(len(candidates))]
    ).astype(np.float32)
    for name, values in candidate_context_parts.items():
        long_frame[name] = values
    long_frame = long_frame.loc[keep_mask].reset_index(drop=True)
    y_error = y_error[keep_mask].astype(np.float32)
    y_within10 = (y_error <= 10.0).astype(np.int8)
    long_feature_columns = [
        column
        for column in long_frame.columns
        if column not in PROTECTED_LONG_COLUMNS
        and pd.api.types.is_numeric_dtype(long_frame[column])
    ]
    if "is_oracle" in long_feature_columns:
        raise AssertionError("oracle label leaked into long features")
    forbidden_tokens = ("true_tvt", "abs_error", "oracle", "target")
    leaked = [
        column
        for column in long_feature_columns
        if any(token in column for token in forbidden_tokens)
    ]
    if leaked:
        raise AssertionError(f"protected labels leaked into model features: {leaked}")
    return long_frame, y_error, y_within10, long_feature_columns


def _fit_imputer(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(medians, np.where(bad)[1])
    return values, medians


def _apply_imputer(frame: pd.DataFrame, columns: list[str], medians: np.ndarray) -> np.ndarray:
    values = frame[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(medians, np.where(bad)[1])
    return values


def _predict_clean_validation(
    *,
    frame: pd.DataFrame,
    valid_idx: np.ndarray,
    candidate_values: np.ndarray,
    candidates: list[Any],
    base_feature_columns: list[str],
    config: dict[str, Any],
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    feature_columns: list[str],
    medians: np.ndarray,
    classifier: Any,
    error_model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    n_candidates = len(candidates)
    chunk_size = int(get_nested(config, "augmentation.predict_base_row_chunk_size") or 30000)
    probability = np.empty((len(valid_idx), n_candidates), dtype=np.float32)
    predicted_error = np.empty_like(probability)
    for start in range(0, len(valid_idx), chunk_size):
        stop = min(start + chunk_size, len(valid_idx))
        source_idx = valid_idx[start:stop]
        clean_values = candidate_values[source_idx]
        clean_available = np.ones_like(clean_values, dtype=bool)
        clean_long, _y_error, _y_binary, chunk_features = build_candidate_long_view(
            frame.iloc[source_idx],
            clean_values,
            clean_available,
            candidates=candidates,
            base_feature_columns=base_feature_columns,
            config=config,
            raw_cache=raw_cache,
        )
        if chunk_features != feature_columns:
            raise ValueError("clean validation feature schema differs from train schema")
        x = _apply_imputer(clean_long, feature_columns, medians)
        prob = classifier.predict_proba(x)[:, 1].astype(np.float32)
        err = np.maximum(error_model.predict(x).astype(np.float32), 0.0)
        n_rows = stop - start
        probability[start:stop] = prob.reshape(n_candidates, n_rows).T
        predicted_error[start:stop] = err.reshape(n_candidates, n_rows).T
    return probability, predicted_error


def train_outer_oof_models(
    *,
    frame: pd.DataFrame,
    candidates: list[Any],
    candidate_values: np.ndarray,
    base_feature_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    candidate_names = [item.name for item in candidates]
    n_candidates = len(candidates)
    variants = [str(value) for value in get_nested(config, "model.active_variants") or []]
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    seed = int(get_nested(config, "reproducibility.seed") or 42)
    train_limit = int(get_nested(config, "augmentation.max_train_base_rows_per_fold") or 60000)
    valid_limit = int(
        get_nested(config, "augmentation.max_valid_base_rows_for_early_stopping") or 30000
    )
    log_period = int(get_nested(config, "ranker.log_period") or 100)
    classifier_params = dict(get_nested(config, "ranker.long_models.binary_lgbm.params") or {})
    error_params = dict(get_nested(config, "ranker.long_models.error_lgbm.params") or {})
    folds = list(GroupKFold(n_splits=n_folds).split(frame, groups=frame["well"]))
    oof = {
        variant: {
            "probability": np.empty((len(frame), n_candidates), dtype=np.float32),
            "predicted_error": np.empty((len(frame), n_candidates), dtype=np.float32),
        }
        for variant in variants
    }
    inventory_parts: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    schema: list[str] | None = None
    raw_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    models_dir = output_dir / f"{OUTPUT_PREFIX}_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(folds):
        sampled_train = _sample_sorted_rows(
            train_idx,
            train_limit,
            seed=stable_seed(OUTPUT_PREFIX, "train_rows", fold, seed),
        )
        sampled_valid = _sample_sorted_rows(
            valid_idx,
            valid_limit,
            seed=stable_seed(OUTPUT_PREFIX, "valid_rows", fold, seed),
        )
        clean_values = candidate_values[sampled_train]
        clean_available = np.ones_like(clean_values, dtype=bool)
        clean_long, clean_error, clean_binary, clean_features = build_candidate_long_view(
            frame.iloc[sampled_train],
            clean_values,
            clean_available,
            candidates=candidates,
            base_feature_columns=base_feature_columns,
            config=config,
            raw_cache=raw_cache,
        )
        valid_values = candidate_values[sampled_valid]
        valid_available = np.ones_like(valid_values, dtype=bool)
        valid_long, valid_error, valid_binary, valid_features = build_candidate_long_view(
            frame.iloc[sampled_valid],
            valid_values,
            valid_available,
            candidates=candidates,
            base_feature_columns=base_feature_columns,
            config=config,
            raw_cache=raw_cache,
        )
        if valid_features != clean_features:
            raise ValueError("train and early-stop validation feature schemas differ")
        if schema is None:
            schema = clean_features
        elif schema != clean_features:
            raise ValueError("feature schema changed across folds")

        for variant in variants:
            train_long = clean_long
            train_error = clean_error
            train_binary = clean_binary
            if bool(get_nested(config, f"augmentation.variants.{variant}.add_augmented_view")):
                augmented_values, augmented_available, inventory = build_augmented_candidate_view(
                    frame.iloc[sampled_train].reset_index(drop=True),
                    clean_values,
                    candidate_names=candidate_names,
                    fold=fold,
                    config=config,
                )
                inventory["variant"] = variant
                inventory_parts.append(inventory)
                augmented_long, augmented_error, augmented_binary, augmented_features = (
                    build_candidate_long_view(
                        frame.iloc[sampled_train],
                        augmented_values,
                        augmented_available,
                        candidates=candidates,
                        base_feature_columns=base_feature_columns,
                        config=config,
                        raw_cache=raw_cache,
                    )
                )
                if augmented_features != clean_features:
                    raise ValueError("augmented and clean feature schemas differ")
                train_long = pd.concat([clean_long, augmented_long], ignore_index=True)
                train_error = np.concatenate([clean_error, augmented_error])
                train_binary = np.concatenate([clean_binary, augmented_binary])

            x_train, medians = _fit_imputer(train_long, clean_features)
            x_valid = _apply_imputer(valid_long, clean_features, medians)
            classifier = LGBMClassifier(
                objective="binary",
                random_state=stable_seed(OUTPUT_PREFIX, variant, "classifier", fold, seed),
                **classifier_params,
            )
            classifier.fit(
                x_train,
                train_binary,
                eval_set=[(x_valid, valid_binary)],
                eval_metric="binary_logloss",
                callbacks=[early_stopping(50), log_evaluation(log_period)],
            )
            error_model = LGBMRegressor(
                objective="regression_l1",
                random_state=stable_seed(OUTPUT_PREFIX, variant, "error", fold, seed),
                **error_params,
            )
            error_model.fit(
                x_train,
                train_error,
                eval_set=[(x_valid, valid_error)],
                eval_metric="l1",
                callbacks=[early_stopping(50), log_evaluation(log_period)],
            )
            classifier_path = models_dir / f"{variant}_within10_classifier_fold{fold}.txt"
            error_path = models_dir / f"{variant}_expected_error_fold{fold}.txt"
            classifier.booster_.save_model(str(classifier_path))
            error_model.booster_.save_model(str(error_path))
            np.save(models_dir / f"{variant}_imputer_medians_fold{fold}.npy", medians)
            for objective, model, path in [
                ("within10_classifier", classifier, classifier_path),
                ("expected_error_regressor", error_model, error_path),
            ]:
                manifest.append(
                    {
                        "variant": variant,
                        "objective": objective,
                        "fold": fold,
                        "path": str(path.relative_to(output_dir)),
                        "sha256": sha256_path(path),
                        "best_iteration": int(model.best_iteration_ or model.n_estimators),
                        "train_base_rows": int(len(sampled_train)),
                        "train_long_rows": int(len(train_long)),
                        "valid_base_rows": int(len(sampled_valid)),
                        "valid_long_rows": int(len(valid_long)),
                    }
                )
                for feature, importance in zip(
                    clean_features, model.feature_importances_, strict=True
                ):
                    importance_rows.append(
                        {
                            "variant": variant,
                            "objective": objective,
                            "fold": fold,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
            probability, predicted_error = _predict_clean_validation(
                frame=frame,
                valid_idx=np.asarray(valid_idx, dtype=np.int64),
                candidate_values=candidate_values,
                candidates=candidates,
                base_feature_columns=base_feature_columns,
                config=config,
                raw_cache=raw_cache,
                feature_columns=clean_features,
                medians=medians,
                classifier=classifier,
                error_model=error_model,
            )
            oof[variant]["probability"][valid_idx] = probability
            oof[variant]["predicted_error"][valid_idx] = predicted_error
            del x_train, x_valid, classifier, error_model

    if schema is None:
        raise RuntimeError("no folds were trained")
    inventory_frame = (
        pd.concat(inventory_parts, ignore_index=True)
        if inventory_parts
        else pd.DataFrame(
            columns=[
                "fold",
                "id",
                "well",
                "transform",
                "source_candidate",
                "source_family",
                "applied_amplitude",
                "available_candidates",
                "variant",
            ]
        )
    )
    return oof, inventory_frame, pd.DataFrame(manifest), schema, pd.DataFrame(importance_rows)


def _candidate_metrics(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    probability: np.ndarray,
    predicted_error: np.ndarray,
    *,
    variant: str,
) -> dict[str, Any]:
    actual_error = np.abs(candidate_values - true_tvt[:, None])
    y = (actual_error <= 10.0).astype(np.int8).reshape(-1)
    p = np.clip(probability.reshape(-1).astype(np.float64), 1e-6, 1.0 - 1e-6)
    return {
        "variant": variant,
        "candidate_auc": float(roc_auc_score(y, p)),
        "candidate_logloss": float(log_loss(y, p, labels=[0, 1])),
        "candidate_brier": float(brier_score_loss(y, p)),
        "expected_error_mae": mae(actual_error.reshape(-1), predicted_error.reshape(-1)),
        "candidate_rows": int(len(y)),
        "observed_within10": float(np.mean(y)),
        "predicted_within10": float(np.mean(p)),
    }


def _calibration_tables(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    probability: np.ndarray,
    predicted_error: np.ndarray,
    *,
    variant: str,
) -> pd.DataFrame:
    actual = np.abs(candidate_values - true_tvt[:, None]).reshape(-1)
    p = probability.reshape(-1)
    rows: list[dict[str, Any]] = []
    prob_bins = np.linspace(0.0, 1.0, 11)
    prob_codes = np.clip(np.digitize(p, prob_bins[1:-1], right=False), 0, 9)
    for code in range(10):
        mask = prob_codes == code
        if not mask.any():
            continue
        rows.append(
            {
                "variant": variant,
                "calibration_kind": "within10_probability",
                "bin": code,
                "rows": int(mask.sum()),
                "predicted": float(np.mean(p[mask])),
                "observed": float(np.mean(actual[mask] <= 10.0)),
                "mean_abs_error": float(np.mean(actual[mask])),
            }
        )
    predicted = predicted_error.reshape(-1)
    edges = np.unique(np.quantile(predicted, np.linspace(0.0, 1.0, 11)))
    if len(edges) > 1:
        codes = np.clip(np.digitize(predicted, edges[1:-1], right=False), 0, len(edges) - 2)
        for code in range(len(edges) - 1):
            mask = codes == code
            if not mask.any():
                continue
            rows.append(
                {
                    "variant": variant,
                    "calibration_kind": "expected_error",
                    "bin": code,
                    "rows": int(mask.sum()),
                    "predicted": float(np.mean(predicted[mask])),
                    "observed": float(np.mean(actual[mask])),
                    "mean_abs_error": float(np.mean(actual[mask])),
                }
            )
    return pd.DataFrame(rows)


def _topk_coverage(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    score: np.ndarray,
    *,
    variant: str,
    score_kind: str,
) -> pd.DataFrame:
    actual = np.abs(candidate_values - true_tvt[:, None])
    order = np.argsort(-score if score_kind == "probability" else score, axis=1)
    rows: list[dict[str, Any]] = []
    for k in [1, 3, 5]:
        chosen = np.take_along_axis(actual, order[:, :k], axis=1)
        rows.append(
            {
                "variant": variant,
                "score_kind": score_kind,
                "k": k,
                "within10_coverage": float(np.mean(np.any(chosen <= 10.0, axis=1))),
                "oracle_topk_rmse": float(np.sqrt(np.mean(np.min(chosen, axis=1) ** 2))),
            }
        )
    return pd.DataFrame(rows)


def _margin_calibration(
    true_tvt: np.ndarray,
    candidate_values: np.ndarray,
    predicted_error: np.ndarray,
    *,
    variant: str,
) -> pd.DataFrame:
    order = np.argsort(predicted_error, axis=1)
    best = order[:, 0]
    second = order[:, 1]
    margin = (
        predicted_error[np.arange(len(best)), second] - predicted_error[np.arange(len(best)), best]
    )
    selected = candidate_values[np.arange(len(best)), best]
    actual = np.abs(selected - true_tvt)
    edges = np.unique(np.quantile(margin, np.linspace(0.0, 1.0, 11)))
    rows: list[dict[str, Any]] = []
    if len(edges) <= 1:
        return pd.DataFrame(rows)
    codes = np.clip(np.digitize(margin, edges[1:-1], right=False), 0, len(edges) - 2)
    for code in range(len(edges) - 1):
        mask = codes == code
        if not mask.any():
            continue
        rows.append(
            {
                "variant": variant,
                "bin": code,
                "rows": int(mask.sum()),
                "margin_mean": float(np.mean(margin[mask])),
                "selected_rmse": float(np.sqrt(np.mean(actual[mask] ** 2))),
                "selected_within10": float(np.mean(actual[mask] <= 10.0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_oof(
    *,
    frame: pd.DataFrame,
    candidates: list[Any],
    candidate_values: np.ndarray,
    oof_scores: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    candidate_names = [item.name for item in candidates]
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    oracle_labels = np.argmin(np.abs(candidate_values - true_tvt[:, None]), axis=1).astype(np.int16)
    metrics_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    calibration_parts: list[pd.DataFrame] = []
    topk_parts: list[pd.DataFrame] = []
    margin_parts: list[pd.DataFrame] = []
    wide_predictions = frame[["id", "well"]].copy()
    wide_predictions["true_tvt"] = true_tvt
    by_well_parts: list[pd.DataFrame] = []
    bucket_parts: list[pd.DataFrame] = []
    subgroup_parts: list[pd.DataFrame] = []
    default_idx = candidate_names.index(str(get_nested(config, "selector.default_candidate")))
    allowed_names = [
        str(value) for value in get_nested(config, "selector.allowed_switch_candidates") or []
    ]
    allowed_idx = np.asarray(
        [candidate_names.index(name) for name in allowed_names], dtype=np.int16
    )
    specs = parent.variant_specs_from_config(config)
    if len(specs) != 1:
        raise ValueError(f"expected exactly one fixed Viterbi spec, got {len(specs)}")
    spec = specs[0]

    for variant, scores in oof_scores.items():
        probability = scores["probability"]
        predicted_error = scores["predicted_error"]
        if not np.isfinite(probability).all() or not np.isfinite(predicted_error).all():
            raise ValueError(f"{variant} OOF score surface contains non-finite values")
        candidate_rows.append(
            _candidate_metrics(
                true_tvt,
                candidate_values,
                probability,
                predicted_error,
                variant=variant,
            )
        )
        calibration_parts.append(
            _calibration_tables(
                true_tvt,
                candidate_values,
                probability,
                predicted_error,
                variant=variant,
            )
        )
        topk_parts.append(
            _topk_coverage(
                true_tvt,
                candidate_values,
                probability,
                variant=variant,
                score_kind="probability",
            )
        )
        topk_parts.append(
            _topk_coverage(
                true_tvt,
                candidate_values,
                predicted_error,
                variant=variant,
                score_kind="predicted_error",
            )
        )
        margin_parts.append(
            _margin_calibration(
                true_tvt,
                candidate_values,
                predicted_error,
                variant=variant,
            )
        )
        selections = {
            "probability_rowwise": np.argmax(probability, axis=1).astype(np.int16),
            "expected_error_rowwise": np.argmin(predicted_error, axis=1).astype(np.int16),
        }
        selections["expected_error_fixed_viterbi"] = parent.viterbi_select(
            frame=frame,
            predicted_error=predicted_error,
            candidate_values=candidate_values,
            candidate_names=candidate_names,
            default_idx=default_idx,
            allowed_switch_idx=allowed_idx,
            spec=spec,
        )
        for mode, selected_idx in selections.items():
            selected = candidate_values[np.arange(len(frame)), selected_idx]
            actual = np.abs(selected - true_tvt)
            metrics_rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "rmse_tvt": rmse(true_tvt, selected),
                    "mae_tvt": mae(true_tvt, selected),
                    "within10": float(np.mean(actual <= 10.0)),
                    "oracle_label_accuracy": float(np.mean(selected_idx == oracle_labels)),
                    "path_switch_count": int(
                        np.sum(
                            (selected_idx[1:] != selected_idx[:-1])
                            & (frame["well"].to_numpy()[1:] == frame["well"].to_numpy()[:-1])
                        )
                    ),
                }
            )
            pred = pd.DataFrame(
                {
                    "id": frame["id"].astype(str).to_numpy(),
                    "well": frame["well"].astype(str).to_numpy(),
                    "variant": variant,
                    "mode": mode,
                    "selected_candidate": np.asarray(
                        [candidate_names[idx] for idx in selected_idx]
                    ),
                    "selected_candidate_index": selected_idx,
                    "selected_tvt": selected,
                    "true_tvt": true_tvt,
                    "abs_error": actual,
                    "oracle_label": oracle_labels,
                }
            )
            prefix = f"{variant}_{mode}"
            wide_predictions[f"{prefix}_candidate"] = pred["selected_candidate"].to_numpy()
            wide_predictions[f"{prefix}_tvt"] = selected.astype(np.float32)
            wide_predictions[f"{prefix}_abs_error"] = actual.astype(np.float32)
            by_well_parts.append(parent.summarize_by_well(pred))
            bucket_parts.append(parent.bucket_metrics(pred, frame))
            subgroup_parts.append(parent.subgroup_metrics(pred, frame, config))
    return {
        "metrics": pd.DataFrame(metrics_rows),
        "candidate_metrics": pd.DataFrame(candidate_rows),
        "calibration": pd.concat(calibration_parts, ignore_index=True),
        "topk_coverage": pd.concat(topk_parts, ignore_index=True),
        "margin_calibration": pd.concat(margin_parts, ignore_index=True),
        "predictions": wide_predictions,
        "by_well": pd.concat(by_well_parts, ignore_index=True),
        "bucket_metrics": pd.concat(bucket_parts, ignore_index=True),
        "subgroup_metrics": pd.concat(subgroup_parts, ignore_index=True),
    }


def summarize_decision(results: dict[str, pd.DataFrame], config: dict[str, Any]) -> dict[str, Any]:
    metrics = results["metrics"]
    candidate = results["candidate_metrics"]
    by_well = results["by_well"]
    buckets = results["bucket_metrics"]
    subgroups = results["subgroup_metrics"]
    mode = "expected_error_fixed_viterbi"
    control_rmse = float(
        metrics.loc[
            metrics["variant"].eq("original_only") & metrics["mode"].eq(mode), "rmse_tvt"
        ].iloc[0]
    )
    augmented_rmse = float(
        metrics.loc[
            metrics["variant"].eq("perturbation_augmented") & metrics["mode"].eq(mode),
            "rmse_tvt",
        ].iloc[0]
    )
    control_logloss = float(
        candidate.loc[candidate["variant"].eq("original_only"), "candidate_logloss"].iloc[0]
    )
    augmented_logloss = float(
        candidate.loc[candidate["variant"].eq("perturbation_augmented"), "candidate_logloss"].iloc[
            0
        ]
    )
    bucket_key = ("distance_bucket", "1000_plus")

    def bucket_rmse(variant: str) -> float:
        selected = buckets[
            buckets["variant"].eq(variant)
            & buckets["mode"].eq(mode)
            & buckets["bucket_family"].eq(bucket_key[0])
            & buckets["bucket"].eq(bucket_key[1])
        ]
        return float(selected["rmse_tvt"].iloc[0])

    control_well = by_well[by_well["variant"].eq("original_only") & by_well["mode"].eq(mode)][
        ["well", "rmse_tvt"]
    ].rename(columns={"rmse_tvt": "control_rmse"})
    aug_well = by_well[by_well["variant"].eq("perturbation_augmented") & by_well["mode"].eq(mode)][
        ["well", "rmse_tvt"]
    ].rename(columns={"rmse_tvt": "augmented_rmse"})
    well_delta = control_well.merge(aug_well, on="well", validate="one_to_one")
    well_delta["delta"] = well_delta["augmented_rmse"] - well_delta["control_rmse"]
    hidden_deltas: dict[str, float] = {}
    for subgroup in ["exp115_spatial_valid", "exp115_typewell_purged_valid"]:
        control = subgroups[
            subgroups["variant"].eq("original_only")
            & subgroups["mode"].eq(mode)
            & subgroups["subgroup"].eq(subgroup)
        ]
        augmented = subgroups[
            subgroups["variant"].eq("perturbation_augmented")
            & subgroups["mode"].eq(mode)
            & subgroups["subgroup"].eq(subgroup)
        ]
        if len(control) and len(augmented):
            hidden_deltas[subgroup] = float(
                augmented["rmse_tvt"].iloc[0] - control["rmse_tvt"].iloc[0]
            )
    criteria = get_nested(config, "audit.success_criteria") or {}
    checks = {
        "selected_rmse_nonworse": augmented_rmse <= control_rmse,
        "candidate_logloss_nonworse": augmented_logloss <= control_logloss,
        "1000_plus_nonworse": bucket_rmse("perturbation_augmented") <= bucket_rmse("original_only"),
        "hidden_like_nonworse": all(value <= 0.0 for value in hidden_deltas.values()),
        "worst_well_regression_bounded": float(well_delta["delta"].max())
        <= float(criteria.get("max_worst_well_regression", 0.25)),
    }
    return {
        "adoption_supported": bool(all(checks.values())),
        "checks": checks,
        "delta_rmse": augmented_rmse - control_rmse,
        "delta_candidate_logloss": augmented_logloss - control_logloss,
        "delta_1000_plus_rmse": bucket_rmse("perturbation_augmented")
        - bucket_rmse("original_only"),
        "hidden_like_delta_rmse": hidden_deltas,
        "worst_well_regression": float(well_delta["delta"].max()),
        "worst_well": str(well_delta.loc[well_delta["delta"].idxmax(), "well"]),
    }


def synthetic_augmentation_contract_test() -> dict[str, Any]:
    candidate_names = [
        "pf_ancc",
        "beam_mean",
        "likpf_mean",
        "sc_ens",
        "hyb",
        "tvt_dense",
        "tvt_densew",
        "tvt_dense50",
        "blend_likpf_hmm_w500",
        "hmm_selfgr_boost_only_a070_c100",
        "v6_k16_geometry_gr_u_projection",
    ]
    config = load_config()
    rows = 700
    frame = pd.DataFrame(
        {
            "id": [f"well_{idx % 7}_{idx:04d}" for idx in range(rows)],
            "well": [f"well_{idx % 7}" for idx in range(rows)],
            "eval_len": np.full(rows, rows, dtype=np.float32),
        }
    )
    for idx, name in enumerate(candidate_names):
        frame[f"multiobs_score_{name}"] = np.float32(0.01 * idx)
    values = np.tile(np.arange(len(candidate_names), dtype=np.float32), (rows, 1))
    first = build_augmented_candidate_view(
        frame, values, candidate_names=candidate_names, fold=0, config=config
    )
    second = build_augmented_candidate_view(
        frame, values, candidate_names=candidate_names, fold=0, config=config
    )
    if not np.array_equal(first[0], second[0]) or not np.array_equal(first[1], second[1]):
        raise AssertionError("augmentation is not deterministic")
    transforms = set(first[2]["transform"])
    expected = set(get_nested(config, "augmentation.enabled_transforms") or [])
    if transforms != expected:
        raise AssertionError(f"synthetic coverage mismatch: {transforms} != {expected}")
    if int(first[1].sum(axis=1).min()) < 1:
        raise AssertionError("augmentation removed every candidate")
    return {
        "deterministic": True,
        "rows": rows,
        "transforms": sorted(transforms),
        "minimum_available_candidates": int(first[1].sum(axis=1).min()),
    }


def run_candidate_perturbation_augmentation(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    contract = synthetic_augmentation_contract_test()
    (
        frame,
        candidates,
        candidate_values,
        _oracle_labels,
        base_feature_columns,
        source_meta,
    ) = assemble_parent_candidate_surface(
        cache_path=cache_path,
        schema_path=schema_path,
        max_rows=max_rows,
    )
    oof_scores, inventory, model_manifest, feature_schema, feature_importance = (
        train_outer_oof_models(
            frame=frame,
            candidates=candidates,
            candidate_values=candidate_values,
            base_feature_columns=base_feature_columns,
            config=config,
            output_dir=output_dir,
        )
    )
    results = evaluate_oof(
        frame=frame,
        candidates=candidates,
        candidate_values=candidate_values,
        oof_scores=oof_scores,
        config=config,
    )
    decision = summarize_decision(results, config)
    paths: dict[str, Path] = {
        "metrics": output_dir / f"{OUTPUT_PREFIX}_metrics.csv",
        "candidate_metrics": output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "calibration": output_dir / f"{OUTPUT_PREFIX}_calibration.csv",
        "topk_coverage": output_dir / f"{OUTPUT_PREFIX}_topk_coverage.csv",
        "margin_calibration": output_dir / f"{OUTPUT_PREFIX}_margin_calibration.csv",
        "oof_predictions": output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "bucket_metrics": output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
        "subgroup_metrics": output_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv",
        "augmentation_inventory": output_dir / f"{OUTPUT_PREFIX}_augmentation_inventory.csv.gz",
        "feature_importance_mean": output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
        "feature_schema": output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv",
        "model_manifest": output_dir / f"{OUTPUT_PREFIX}_model_manifest.json",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    for key in [
        "metrics",
        "candidate_metrics",
        "calibration",
        "topk_coverage",
        "margin_calibration",
        "by_well",
        "bucket_metrics",
        "subgroup_metrics",
    ]:
        results[key].to_csv(paths[key], index=False)
    results["predictions"].to_csv(paths["oof_predictions"], index=False, compression="gzip")
    inventory.to_csv(paths["augmentation_inventory"], index=False, compression="gzip")
    pd.DataFrame(
        {"feature_order": np.arange(len(feature_schema), dtype=np.int32), "feature": feature_schema}
    ).to_csv(paths["feature_schema"], index=False)
    paths["model_manifest"].write_text(
        json.dumps(to_jsonable(model_manifest.to_dict("records")), indent=2)
    )
    importance_mean = (
        feature_importance.groupby(["variant", "objective", "feature"], as_index=False)[
            "importance"
        ]
        .mean()
        .sort_values(["variant", "objective", "importance"], ascending=[True, True, False])
    )
    importance_mean.to_csv(paths["feature_importance_mean"], index=False)
    summary = {
        "status": "completed_train_side_adoption_supported"
        if decision["adoption_supported"]
        else "completed_train_side_guard_failed",
        "runtime_seconds": time.time() - started,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "candidate_count": int(len(candidates)),
        "candidate_names": [item.name for item in candidates],
        "base_feature_count": int(len(base_feature_columns)),
        "long_feature_count": int(len(feature_schema)),
        "model_count": int(len(model_manifest)),
        "augmentation_contract": contract,
        "augmentation_inventory_rows": int(len(inventory)),
        "source_meta": to_jsonable(source_meta),
        "decision": decision,
        "metrics": to_jsonable(results["metrics"].to_dict("records")),
        "candidate_metrics": to_jsonable(results["candidate_metrics"].to_dict("records")),
        "artifacts": {key: path.name for key, path in paths.items()},
        "sha256": {
            key: sha256_path(path, decompressed=path.suffix == ".gz")
            for key, path in paths.items()
            if key != "summary"
        },
    }
    paths["summary"].write_text(json.dumps(to_jsonable(summary), indent=2))
    summary["sha256"]["summary"] = sha256_path(paths["summary"])
    return summary


__all__ = [
    "OUTPUT_PREFIX",
    "assemble_parent_candidate_surface",
    "build_augmented_candidate_view",
    "run_candidate_perturbation_augmentation",
    "synthetic_augmentation_contract_test",
]
