from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp136_gr_shape_descriptor_verifier_on_candidate_selector"
EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
EXP101_MANIFEST = "exp101_pf_candidate_ranker_or_nway_classifier_model_manifest.json"
EXP101_FEATURE_SCHEMA = "exp101_pf_candidate_ranker_or_nway_classifier_feature_schema.csv"
DESCRIPTOR_SCORE_VARIANTS = [
    "raw_point_real",
    "banded_shift_real",
    "shape_descriptor_real",
    "combo_descriptor_real",
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    column: str


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(frame: pd.DataFrame, *, value_col: str) -> str:
    digest = hashlib.sha256()
    for row in frame[["id", value_col]].itertuples(index=False):
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row[1]).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(
    filename: str,
    *,
    explicit_path: str | Path | None = None,
    explicit_dir: str | Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    if explicit_dir is not None:
        candidates.append(Path(explicit_dir) / filename)
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("experiments")
            / "exp101_pf_candidate_ranker_or_nway_classifier"
            / "kaggle"
            / "output"
            / "train_v1"
            / "artifacts"
            / filename,
            Path("experiments")
            / "exp099_pf_multi_observation_likelihood_probe"
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / filename,
        ]
    )
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:100])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def resolve_model_path(manifest_path: Path, relative_path: str) -> Path:
    candidates = [
        manifest_path.parent / relative_path,
        manifest_path.parent / Path(relative_path).name,
        Path("experiments")
        / "exp101_pf_candidate_ranker_or_nway_classifier"
        / "kaggle"
        / "output"
        / "train_v1"
        / "artifacts"
        / relative_path,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{Path(relative_path).name}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"model file not found for {relative_path}")


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "gate.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("gate.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("gate.candidates must not be empty")
    return specs


def configured_raw_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "target", "last_known_tvt"}
    required.update(spec.column for spec in candidates)
    for key in [
        "gate.context_columns",
        "gate.multiobs_feature_columns",
        "gate.optional_columns",
    ]:
        required.update(str(value) for value in (get_nested(config, key) or []))
    return sorted(required)


def load_feature_cache(
    *,
    config: dict[str, Any],
    required_columns: list[str],
    max_rows: int | None,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(
        EXP099_FEATURE_CACHE,
        explicit_path=cache_path or get_nested(config, "data.exp099_train_feature_cache_local"),
    )
    schema = find_artifact(
        EXP099_FEATURE_SCHEMA,
        explicit_path=schema_path or get_nested(config, "data.exp099_train_feature_schema_local"),
    )
    header = pd.read_csv(source, nrows=0).columns.tolist()
    load_columns = [column for column in required_columns if column in header]
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required raw columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=load_columns,
        nrows=max_rows,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    meta = {
        "path": str(source),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema_path": str(schema),
        "schema_sha256": sha256_path(schema),
    }
    return frame, meta


def add_candidate_labels_and_features(
    frame: pd.DataFrame, candidates: list[CandidateSpec]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    out = frame.copy()
    out["true_tvt"] = out["last_known_tvt"].astype(np.float32) + out["target"].astype(np.float32)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(out[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
        ]
    )
    if not np.isfinite(candidate_values).all():
        bad = np.argwhere(~np.isfinite(candidate_values))[:5].tolist()
        raise ValueError(f"candidate values contain non-finite values, examples={bad}")
    true_tvt = out["true_tvt"].to_numpy(np.float32)
    errors = np.abs(candidate_values - true_tvt[:, None])
    oracle_labels = np.argmin(errors, axis=1).astype(np.int16)
    out["oracle_label"] = oracle_labels
    out["oracle_candidate"] = np.asarray([candidates[i].name for i in oracle_labels], dtype=object)

    for spec in candidates:
        out[f"{spec.name}_minus_last"] = (
            out[spec.column].astype(np.float32) - out["last_known_tvt"].astype(np.float32)
        )
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            out[f"{left.name}_vs_{right.name}_abs"] = np.abs(
                out[left.column].astype(np.float32) - out[right.column].astype(np.float32)
            )
    value_cols = [spec.column for spec in candidates]
    out["candidate_mean"] = out[value_cols].mean(axis=1).astype(np.float32)
    out["candidate_std"] = out[value_cols].std(axis=1).astype(np.float32)
    out["candidate_range"] = (out[value_cols].max(axis=1) - out[value_cols].min(axis=1)).astype(
        np.float32
    )
    return out, candidate_values, oracle_labels


def load_exp101_feature_columns(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    schema_path = find_artifact(
        str(get_nested(config, "data.exp101_feature_schema") or EXP101_FEATURE_SCHEMA),
        explicit_dir=get_nested(config, "data.exp101_artifact_dir_local"),
    )
    schema = pd.read_csv(schema_path)
    if "feature" not in schema.columns:
        raise ValueError(f"{schema_path} must contain a feature column")
    feature_columns = schema["feature"].astype(str).tolist()
    if not feature_columns:
        raise ValueError(f"{schema_path} contains no features")
    return feature_columns, {"path": str(schema_path), "sha256": sha256_path(schema_path)}


def fit_impute_values(train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]) -> np.ndarray:
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    valid_values = valid[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(valid_values)
    if bad.any():
        valid_values[bad] = np.take(medians, np.where(bad)[1])
    return valid_values


def build_long_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    sample_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if sample_rows is not None and len(row_indices) > sample_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(sample_rows), replace=False))
    chunks: list[pd.DataFrame] = []
    last_known = frame["last_known_tvt"].to_numpy(np.float32)
    for cand_idx, spec in enumerate(candidates):
        part = frame.iloc[row_indices][["id", "well", *row_feature_columns]].copy()
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name_code"] = np.int16(cand_idx)
        part["candidate_tvt"] = candidate_values[row_indices, cand_idx].astype(np.float32)
        part["candidate_minus_last"] = (
            candidate_values[row_indices, cand_idx] - last_known[row_indices]
        ).astype(np.float32)
        score_col = f"multiobs_score_{spec.name}"
        mae_col = f"multiobs_mae_{spec.name}"
        ncc_col = f"multiobs_ncc_{spec.name}"
        part["candidate_multiobs_score"] = (
            frame.iloc[row_indices][score_col].to_numpy(np.float32)
            if score_col in frame.columns
            else 0.0
        )
        part["candidate_multiobs_mae"] = (
            frame.iloc[row_indices][mae_col].to_numpy(np.float32)
            if mae_col in frame.columns
            else 0.0
        )
        part["candidate_multiobs_ncc"] = (
            frame.iloc[row_indices][ncc_col].to_numpy(np.float32)
            if ncc_col in frame.columns
            else 0.0
        )
        part["is_oracle"] = (oracle_labels[row_indices] == cand_idx).astype(np.int8)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def long_feature_columns(long_frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in long_frame.columns
        if col not in {"id", "well", "is_oracle"}
        and pd.api.types.is_numeric_dtype(long_frame[col])
    ]


def fit_impute_long_values(
    train_long: pd.DataFrame, valid_long: pd.DataFrame, columns: list[str]
) -> np.ndarray:
    train_values = train_long[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    valid_values = valid_long[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(valid_values)
    if bad.any():
        valid_values[bad] = np.take(medians, np.where(bad)[1])
    return valid_values


def load_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = find_artifact(
        str(get_nested(config, "data.exp101_model_manifest") or EXP101_MANIFEST),
        explicit_dir=get_nested(config, "data.exp101_artifact_dir_local"),
    )
    with manifest_path.open() as fp:
        manifest = json.load(fp)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list):
        raise ValueError(f"invalid exp101 model manifest: {manifest_path}")
    return manifest_path, manifest


def model_item(manifest: dict[str, Any], variant: str, fold: int) -> dict[str, Any]:
    for item in manifest["models"]:
        if item.get("variant") == variant and int(item.get("fold")) == int(fold):
            return item
    raise KeyError(f"model not found: variant={variant} fold={fold}")


def reconstruct_exp101_scores(
    *,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    oracle_labels: np.ndarray,
    feature_columns: list[str],
    config: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    from lightgbm import Booster

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    max_train_rows = get_nested(config, "gate.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    n_rows = len(frame)
    n_candidates = len(candidates)
    scores = {
        "multiclass_proba": np.zeros((n_rows, n_candidates), dtype=np.float32),
        "binary_proba": np.zeros((n_rows, n_candidates), dtype=np.float32),
        "predicted_error": np.zeros((n_rows, n_candidates), dtype=np.float32),
    }
    manifest_rows: list[dict[str, Any]] = []

    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(frame, oracle_labels, groups=frame["well"]))
    for fold, (train_idx, valid_idx) in enumerate(folds):
        print(f"[fold {fold}] reconstruct scores train={len(train_idx)} valid={len(valid_idx)}")
        train_frame = frame.iloc[train_idx]
        valid_frame = frame.iloc[valid_idx]
        x_valid = fit_impute_values(train_frame, valid_frame, feature_columns)

        multiclass_item = model_item(manifest, "lgb_multiclass", fold)
        multiclass_path = resolve_model_path(manifest_path, str(multiclass_item["path"]))
        multiclass = Booster(model_file=str(multiclass_path))
        multiclass_pred = multiclass.predict(x_valid)
        scores["multiclass_proba"][valid_idx] = np.asarray(multiclass_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **multiclass_item,
                "resolved_path": str(multiclass_path),
                "resolved_sha256": sha256_path(multiclass_path),
            }
        )

        train_long = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=feature_columns,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
        )
        valid_long = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=feature_columns,
            candidate_values=candidate_values,
            oracle_labels=oracle_labels,
            sample_rows=None,
            seed=seed,
        )
        long_columns = long_feature_columns(train_long)
        x_long_valid = fit_impute_long_values(train_long, valid_long, long_columns)

        binary_item = model_item(manifest, "lgb_candidate_binary", fold)
        binary_path = resolve_model_path(manifest_path, str(binary_item["path"]))
        binary = Booster(model_file=str(binary_path))
        binary_pred = binary.predict(x_long_valid).reshape(n_candidates, len(valid_idx)).T
        scores["binary_proba"][valid_idx] = np.asarray(binary_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **binary_item,
                "resolved_path": str(binary_path),
                "resolved_sha256": sha256_path(binary_path),
            }
        )

        error_item = model_item(manifest, "lgb_candidate_error_ranker", fold)
        error_path = resolve_model_path(manifest_path, str(error_item["path"]))
        error_model = Booster(model_file=str(error_path))
        error_pred = error_model.predict(x_long_valid).reshape(n_candidates, len(valid_idx)).T
        scores["predicted_error"][valid_idx] = np.asarray(error_pred, dtype=np.float32)
        manifest_rows.append(
            {
                **error_item,
                "resolved_path": str(error_path),
                "resolved_sha256": sha256_path(error_path),
            }
        )

    return scores, pd.DataFrame(manifest_rows)


def second_margin_high(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    top = order[:, -1]
    top1 = values[np.arange(len(values)), top]
    top2 = values[np.arange(len(values)), order[:, -2]]
    return top.astype(np.int16), (top1 - top2).astype(np.float32)


def second_margin_low(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    top = order[:, 0]
    top1 = values[np.arange(len(values)), top]
    top2 = values[np.arange(len(values)), order[:, 1]]
    return top.astype(np.int16), (top2 - top1).astype(np.float32), top1.astype(np.float32)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    med = float(np.nanmedian(finite))
    q25, q75 = np.nanquantile(finite, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(finite) + 1e-6)
    out = (values - med) / scale
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _nearest_prefix_indices(prefix_tvt: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, candidate_tvt, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - candidate_tvt) < np.abs(
        sorted_tvt[left] - candidate_tvt
    )
    nearest_sorted = np.where(choose_right, right, left)
    return order[nearest_sorted].astype(np.int32)


def _standardize_last_axis(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1e-6
    return centered / scale


def _local_vectors(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    indices = np.clip(centers[..., None] + offsets.astype(np.int32), 0, len(series) - 1)
    return series[indices]


def _local_derivatives(
    series: np.ndarray,
    centers: np.ndarray,
    offsets: np.ndarray,
    *,
    step: int,
) -> np.ndarray:
    left = np.clip(centers[..., None] + offsets.astype(np.int32) - int(step), 0, len(series) - 1)
    right = np.clip(centers[..., None] + offsets.astype(np.int32) + int(step), 0, len(series) - 1)
    return (series[right] - series[left]) / max(2 * int(step), 1)


def _peak_count_proxy(vectors: np.ndarray) -> np.ndarray:
    diffs = np.diff(vectors, axis=-1)
    signs = np.sign(diffs)
    turns = signs[..., 1:] * signs[..., :-1] < 0
    return turns.sum(axis=-1).astype(np.float32)


def _shifted_window_mae(
    series: np.ndarray,
    eval_centers: np.ndarray,
    candidate_centers: np.ndarray,
    window_offsets: np.ndarray,
    shift_offsets: np.ndarray,
) -> np.ndarray:
    eval_vectors = _local_vectors(series, eval_centers, window_offsets)
    shifted_costs = []
    for shift in shift_offsets:
        candidate_vectors = _local_vectors(series, candidate_centers + int(shift), window_offsets)
        shifted_costs.append(np.mean(np.abs(candidate_vectors - eval_vectors[:, None, :]), axis=2))
    return np.min(np.stack(shifted_costs, axis=2), axis=2).astype(np.float32)


def descriptor_scores_for_gr(
    *,
    full_gr: np.ndarray,
    missing_mask: np.ndarray,
    prefix_tvt: np.ndarray,
    row_idx: np.ndarray,
    candidate_values: np.ndarray,
    descriptor_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    window_offsets = np.asarray(
        descriptor_config.get("window_offsets", [-24, -12, 0, 12, 24]),
        dtype=np.int32,
    )
    derivative_offsets = np.asarray(
        descriptor_config.get("derivative_offsets", [-12, 0, 12]),
        dtype=np.int32,
    )
    shift_offsets = np.asarray(
        descriptor_config.get("shift_offsets", [-6, 0, 6]),
        dtype=np.int32,
    )
    derivative_step = int(descriptor_config.get("derivative_step", 3))
    gr_scale = float(descriptor_config.get("gr_scale", 18.0))
    derivative_scale = float(descriptor_config.get("derivative_scale", 9.0))
    energy_scale = float(descriptor_config.get("energy_scale", 18.0))
    shape_scale = float(descriptor_config.get("shape_scale", 1.0))
    band_scale = float(descriptor_config.get("band_scale", 16.0))

    candidate_values = np.nan_to_num(candidate_values, nan=float(prefix_tvt[-1]))
    n_rows, n_candidates = candidate_values.shape
    candidate_centers = _nearest_prefix_indices(prefix_tvt, candidate_values.reshape(-1)).reshape(
        n_rows,
        n_candidates,
    )

    eval_point = full_gr[row_idx]
    candidate_point = full_gr[candidate_centers]
    raw_point = np.abs(candidate_point - eval_point[:, None]).astype(np.float32)

    eval_window = _local_vectors(full_gr, row_idx, window_offsets)
    candidate_window = _local_vectors(full_gr, candidate_centers, window_offsets)
    eval_norm = _standardize_last_axis(eval_window)
    candidate_norm = _standardize_last_axis(candidate_window)
    window_mae = np.mean(np.abs(candidate_window - eval_window[:, None, :]), axis=2)

    eval_derivative = _local_derivatives(
        full_gr,
        row_idx,
        derivative_offsets,
        step=derivative_step,
    )
    candidate_derivative = _local_derivatives(
        full_gr,
        candidate_centers,
        derivative_offsets,
        step=derivative_step,
    )
    derivative_mae = np.mean(
        np.abs(candidate_derivative - eval_derivative[:, None, :]),
        axis=2,
    )

    eval_curvature = np.diff(eval_derivative, n=2, axis=-1)
    candidate_curvature = np.diff(candidate_derivative, n=2, axis=-1)
    if eval_curvature.shape[-1] == 0:
        curvature_mae = np.zeros((n_rows, n_candidates), dtype=np.float32)
    else:
        curvature_mae = np.mean(
            np.abs(candidate_curvature - eval_curvature[:, None, :]),
            axis=2,
        )

    eval_energy = np.sqrt(np.mean(np.square(eval_derivative), axis=1))
    candidate_energy = np.sqrt(np.mean(np.square(candidate_derivative), axis=2))
    energy_abs = np.abs(candidate_energy - eval_energy[:, None])

    eval_peak_count = _peak_count_proxy(eval_window)
    candidate_peak_count = _peak_count_proxy(candidate_window)
    peak_count_abs = np.abs(candidate_peak_count - eval_peak_count[:, None])

    eval_missing = _local_vectors(missing_mask.astype(np.float32), row_idx, window_offsets).mean(
        axis=1,
    )
    candidate_missing = _local_vectors(
        missing_mask.astype(np.float32),
        candidate_centers,
        window_offsets,
    ).mean(axis=2)
    missing_gap_abs = np.abs(candidate_missing - eval_missing[:, None])

    banded_shift = _shifted_window_mae(
        full_gr,
        row_idx,
        candidate_centers,
        window_offsets,
        shift_offsets,
    )
    local_shape = np.mean(np.abs(candidate_norm - eval_norm[:, None, :]), axis=2)

    shape_distance = (
        0.40 * local_shape
        + 0.20 * np.clip(derivative_mae / max(derivative_scale, 1e-6), 0.0, 5.0)
        + 0.15 * np.clip(curvature_mae / max(derivative_scale, 1e-6), 0.0, 5.0)
        + 0.15 * np.clip(energy_abs / max(energy_scale, 1e-6), 0.0, 5.0)
        + 0.05 * np.clip(peak_count_abs / 4.0, 0.0, 2.0)
        + 0.05 * np.clip(missing_gap_abs, 0.0, 1.0)
    ).astype(np.float32)
    combo_cost = (
        0.20 * np.clip(raw_point / max(gr_scale, 1e-6), 0.0, 5.0)
        + 0.25 * np.clip(window_mae / max(gr_scale, 1e-6), 0.0, 5.0)
        + 0.25 * np.clip(banded_shift / max(band_scale, 1e-6), 0.0, 5.0)
        + 0.30 * np.clip(shape_distance / max(shape_scale, 1e-6), 0.0, 5.0)
    ).astype(np.float32)

    return {
        "raw_point_real": np.exp(-raw_point / max(gr_scale, 1e-6)).astype(np.float32),
        "banded_shift_real": np.exp(-banded_shift / max(band_scale, 1e-6)).astype(np.float32),
        "shape_descriptor_real": np.exp(
            -shape_distance / max(shape_scale, 1e-6),
        ).astype(np.float32),
        "combo_descriptor_real": np.exp(-combo_cost).astype(np.float32),
    }


def build_descriptor_score_matrices(
    *,
    frame: pd.DataFrame,
    candidate_values: np.ndarray,
    candidate_names: list[str],
    train_dir: Path,
    descriptor_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    gr_rolling_window = int(descriptor_config.get("gr_rolling_window", 5))
    row_indices = _row_indices_from_ids(frame["id"])
    score_matrices = {
        variant: np.zeros((len(frame), len(candidate_names)), dtype=np.float32)
        for variant in DESCRIPTOR_SCORE_VARIANTS
    }
    well_rows: list[dict[str, Any]] = []

    group_positions = frame.groupby("well", sort=False).groups
    for well_idx, (well, positions) in enumerate(group_positions.items(), start=1):
        if well_idx % int(descriptor_config.get("log_period_wells", 50)) == 0:
            print(
                f"[descriptor] processed {well_idx}/{len(group_positions)} wells",
                flush=True,
            )
        position_idx = np.asarray(list(positions), dtype=np.int64)
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        if not horizontal_path.exists():
            raise FileNotFoundError(f"raw train horizontal well file not found: {horizontal_path}")

        horizontal = pd.read_csv(horizontal_path, usecols=["GR", "TVT_input"])
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        known_mask = tvt_input.notna().to_numpy()
        if not known_mask.any():
            raise ValueError(f"No finite TVT_input prefix rows for well {well}")
        prefix_len = int(np.flatnonzero(known_mask)[-1] + 1)
        prefix_tvt = (
            tvt_input.iloc[:prefix_len]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
        if not np.isfinite(prefix_tvt).all():
            raise ValueError(f"Non-finite prefix TVT after interpolation for well {well}")

        gr_series = pd.to_numeric(horizontal["GR"], errors="coerce")
        missing_mask = gr_series.isna().to_numpy()
        fallback = float(gr_series.iloc[:prefix_len].mean())
        if not np.isfinite(fallback):
            full_mean = float(gr_series.mean())
            fallback = full_mean if np.isfinite(full_mean) else 0.0
        full_gr = (
            gr_series.interpolate(limit_direction="both")
            .fillna(fallback)
            .rolling(gr_rolling_window, center=True, min_periods=1)
            .mean()
            .to_numpy(np.float32)
        )

        well_row_idx = row_indices[position_idx]
        if well_row_idx.min(initial=0) < 0 or well_row_idx.max(initial=0) >= len(horizontal):
            raise ValueError(f"row index out of range for well {well}")
        well_candidate_values = candidate_values[position_idx]
        well_scores = descriptor_scores_for_gr(
            full_gr=full_gr,
            missing_mask=missing_mask,
            prefix_tvt=prefix_tvt,
            row_idx=well_row_idx,
            candidate_values=well_candidate_values,
            descriptor_config=descriptor_config,
        )
        for variant, values in well_scores.items():
            score_matrices[variant][position_idx] = values.astype(np.float32)

        combo = well_scores["combo_descriptor_real"]
        well_rows.append(
            {
                "well": str(well),
                "rows": int(len(position_idx)),
                "known_prefix_rows": int(prefix_len),
                "eval_len": int(max(0, len(horizontal) - prefix_len)),
                "gr_missing_rate": float(missing_mask.mean()),
                "combo_descriptor_real_mean": float(np.mean(combo)),
                "combo_descriptor_real_p90": float(np.quantile(combo, 0.90)),
            }
        )

    for variant, matrix in score_matrices.items():
        if not np.isfinite(matrix).all():
            bad = np.argwhere(~np.isfinite(matrix))[:5].tolist()
            raise ValueError(f"descriptor score matrix has non-finite values: {variant} {bad}")
    return score_matrices, pd.DataFrame(well_rows)


def descriptor_score_summary_rows(
    *,
    descriptor_scores: dict[str, np.ndarray],
    candidate_names: list[str],
    selected_idx: np.ndarray,
    default_idx: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_idx = np.arange(len(selected_idx))
    for variant, matrix in descriptor_scores.items():
        selected_score = matrix[row_idx, selected_idx]
        default_score = matrix[:, default_idx]
        sorted_scores = np.sort(matrix, axis=1)
        score_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
        rows.append(
            {
                "score": f"{variant}_selected",
                "min": float(np.nanmin(selected_score)),
                "p25": float(np.nanquantile(selected_score, 0.25)),
                "median": float(np.nanmedian(selected_score)),
                "p75": float(np.nanquantile(selected_score, 0.75)),
                "max": float(np.nanmax(selected_score)),
                "mean_margin_vs_default": float(np.mean(selected_score - default_score)),
                "mean_top1_top2_gap": float(np.mean(score_gap)),
            }
        )
        for cand_idx, candidate in enumerate(candidate_names):
            values = matrix[:, cand_idx]
            rows.append(
                {
                    "score": f"{variant}_{candidate}",
                    "min": float(np.nanmin(values)),
                    "p25": float(np.nanquantile(values, 0.25)),
                    "median": float(np.nanmedian(values)),
                    "p75": float(np.nanquantile(values, 0.75)),
                    "max": float(np.nanmax(values)),
                    "mean_margin_vs_default": float(np.mean(values - default_score)),
                    "mean_top1_top2_gap": float(np.mean(score_gap)),
                }
            )
    return rows


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def distance_bucket_codes(values: pd.Series | np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = ["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"]
    cats = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=labels,
        include_lowest=True,
    )
    return categorical_codes(cats), labels


def tail_rank_bucket_codes(ids: pd.Series) -> tuple[np.ndarray, list[str]]:
    labels = ["000_099", "100_249", "250_499", "500_999", "1000_plus"]
    cats = pd.cut(
        _row_indices_from_ids(ids),
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=labels,
        include_lowest=True,
    )
    return categorical_codes(cats), labels


def quantile_bucket_codes(
    values: pd.Series | np.ndarray, prefix: str
) -> tuple[np.ndarray, list[str]]:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.nunique(dropna=True) < 4:
        return np.zeros(len(series), dtype=np.int16), [f"{prefix}_unknown"]
    edges = np.unique(np.nanquantile(finite, [0.0, 0.25, 0.50, 0.75, 1.0]))
    if len(edges) < 3:
        return np.zeros(len(series), dtype=np.int16), [f"{prefix}_unknown"]
    labels = [f"{prefix}_q{i + 1}" for i in range(len(edges) - 1)]
    cats = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return categorical_codes(cats), labels


def categorical_codes(values: pd.Series | pd.Categorical) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.cat.codes.to_numpy(np.int16)
    return values.codes.astype(np.int16)


def metrics_for_selection(
    *,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    oracle_labels: np.ndarray,
    default_idx: int,
) -> dict[str, Any]:
    pred = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    err = pred.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    switched = selected_idx != default_idx
    metrics: dict[str, Any] = {
        "variant": variant,
        "mode": mode,
        "rows": int(len(selected_idx)),
        "rmse_tvt": float(np.sqrt(np.mean(np.square(err)))),
        "mae_tvt": float(np.mean(abs_err)),
        "oracle_label_accuracy": float(np.mean(selected_idx == oracle_labels)),
        "switch_rows": int(np.sum(switched)),
        "switch_rate": float(np.mean(switched)),
    }
    for threshold in [1.0, 2.0, 5.0, 10.0]:
        metrics[f"within_{int(threshold)}ft"] = float(np.mean(abs_err <= threshold))
    return metrics


def selection_distribution_rows(
    *, variant: str, mode: str, selected_idx: np.ndarray, candidate_names: list[str]
) -> list[dict[str, Any]]:
    rows = []
    total = len(selected_idx)
    counts = np.bincount(selected_idx.astype(np.int16), minlength=len(candidate_names))
    for idx, count in enumerate(counts):
        if count:
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "selected_candidate": candidate_names[idx],
                    "rows": int(count),
                    "rate": float(count / total) if total else 0.0,
                }
            )
    return rows


def by_well_rows(
    *,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    selected_tvt: np.ndarray,
    true_tvt: np.ndarray,
    well_codes: np.ndarray,
    well_names: np.ndarray,
    order: np.ndarray,
) -> list[dict[str, Any]]:
    n_wells = len(well_names)
    err = selected_tvt.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    counts = np.bincount(well_codes, minlength=n_wells)
    se = np.bincount(well_codes, weights=np.square(err), minlength=n_wells)
    ae = np.bincount(well_codes, weights=abs_err, minlength=n_wells)
    within10 = np.bincount(well_codes, weights=(abs_err <= 10.0).astype(float), minlength=n_wells)
    ordered_sel = selected_idx[order]
    ordered_well = well_codes[order]
    same_well = ordered_well[1:] == ordered_well[:-1]
    switch_mask = same_well & (ordered_sel[1:] != ordered_sel[:-1])
    switch_wells = ordered_well[1:][switch_mask]
    switches = np.bincount(switch_wells, minlength=n_wells)
    rows = []
    for code, well in enumerate(well_names):
        if counts[code] == 0:
            continue
        rows.append(
            {
                "variant": variant,
                "mode": mode,
                "well": str(well),
                "rows": int(counts[code]),
                "rmse_tvt": float(np.sqrt(se[code] / counts[code])),
                "mae_tvt": float(ae[code] / counts[code]),
                "within_10ft": float(within10[code] / counts[code]),
                "path_switch_count": int(switches[code]),
                "path_switch_per_1000_rows": float(switches[code] / counts[code] * 1000.0),
            }
        )
    return rows


def bucket_metric_rows(
    *,
    variant: str,
    mode: str,
    selected_tvt: np.ndarray,
    true_tvt: np.ndarray,
    bucket_defs: list[tuple[str, np.ndarray, list[str]]],
) -> list[dict[str, Any]]:
    err = selected_tvt.astype(np.float64) - true_tvt.astype(np.float64)
    abs_err = np.abs(err)
    rows: list[dict[str, Any]] = []
    for family, codes, labels in bucket_defs:
        valid = codes >= 0
        for code, label in enumerate(labels):
            mask = valid & (codes == code)
            count = int(np.sum(mask))
            if not count:
                continue
            rows.append(
                {
                    "variant": variant,
                    "mode": mode,
                    "bucket_family": family,
                    "bucket": label,
                    "rows": count,
                    "rmse_tvt": float(np.sqrt(np.mean(np.square(err[mask])))),
                    "mae_tvt": float(np.mean(abs_err[mask])),
                    "within_10ft": float(np.mean(abs_err[mask] <= 10.0)),
                }
            )
    return rows


def selected_prediction_frame(
    *,
    frame: pd.DataFrame,
    variant: str,
    mode: str,
    selected_idx: np.ndarray,
    candidate_values: np.ndarray,
    true_tvt: np.ndarray,
    oracle_labels: np.ndarray,
    candidate_names: list[str],
) -> pd.DataFrame:
    selected_tvt = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(np.float32)
    return pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "well": frame["well"].to_numpy(),
            "variant": variant,
            "mode": mode,
            "selected_candidate": np.asarray([candidate_names[i] for i in selected_idx]),
            "selected_candidate_index": selected_idx.astype(np.int16),
            "selected_tvt": selected_tvt,
            "true_tvt": true_tvt.astype(np.float32),
            "abs_error": np.abs(selected_tvt - true_tvt).astype(np.float32),
            "oracle_candidate": np.asarray([candidate_names[i] for i in oracle_labels]),
            "oracle_label": oracle_labels.astype(np.int16),
        }
    )


def prune_short_switch_segments(
    selected_idx: np.ndarray,
    *,
    frame: pd.DataFrame,
    default_idx: int,
    min_segment_len: int,
) -> np.ndarray:
    if min_segment_len <= 1:
        return selected_idx
    out = selected_idx.copy()
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, _well_names = pd.factorize(frame["well"], sort=True)
    order = np.lexsort((row_indices, well_codes.astype(np.int32)))
    ordered_selected = out[order]
    ordered_well = well_codes[order]
    start = 0
    n_rows = len(order)
    while start < n_rows:
        end = start + 1
        while (
            end < n_rows
            and ordered_well[end] == ordered_well[start]
            and ordered_selected[end] == ordered_selected[start]
        ):
            end += 1
        candidate_idx = int(ordered_selected[start])
        if candidate_idx != default_idx and end - start < int(min_segment_len):
            out[order[start:end]] = default_idx
        start = end
    return out


def make_gate_variants(
    *,
    config: dict[str, Any],
    candidate_names: list[str],
    candidate_values: np.ndarray,
    frame: pd.DataFrame,
    error_selected: np.ndarray,
    error_margin: np.ndarray,
    error_top1: np.ndarray,
    binary_margin: np.ndarray,
    multiclass_margin: np.ndarray,
    descriptor_scores: dict[str, np.ndarray] | None = None,
) -> list[tuple[str, str, np.ndarray, dict[str, Any]]]:
    default_name = str(get_nested(config, "gate.default_candidate") or "likpf_mean")
    default_idx = candidate_names.index(default_name)
    allowed_names = [
        str(name) for name in (get_nested(config, "gate.allowed_switch_candidates") or [])
    ]
    allowed = np.asarray([candidate_names.index(name) for name in allowed_names], dtype=np.int16)
    allowed_mask = np.isin(error_selected, allowed)
    delta_vs_likpf = np.abs(
        candidate_values[np.arange(len(error_selected)), error_selected]
        - candidate_values[:, default_idx]
    ).astype(np.float32)
    pf_std = (
        pd.to_numeric(frame["pf_ancc_std"], errors="coerce").fillna(np.inf).to_numpy(np.float32)
        if "pf_ancc_std" in frame.columns
        else np.full(len(frame), np.inf, dtype=np.float32)
    )
    score_map: dict[str, np.ndarray] = {
        "error_margin": error_margin,
        "joint_margin": (
            robust_z(error_margin)
            + 0.50 * robust_z(binary_margin)
            + 0.25 * robust_z(multiclass_margin)
            - 0.30 * robust_z(error_top1)
            - 0.20 * robust_z(delta_vs_likpf)
            - 0.20 * robust_z(pf_std)
        ),
        "conservative_margin": (
            robust_z(error_margin)
            + robust_z(binary_margin)
            - 0.50 * robust_z(error_top1)
            - 0.50 * robust_z(delta_vs_likpf)
            - 0.50 * robust_z(pf_std)
        ),
    }
    descriptor_variant = str(
        get_nested(config, "gate.descriptor.score_variant") or "combo_descriptor_real"
    )
    descriptor_selected_score: np.ndarray | None = None
    descriptor_margin_vs_default: np.ndarray | None = None
    if descriptor_scores:
        if descriptor_variant not in descriptor_scores:
            raise ValueError(f"missing descriptor score variant: {descriptor_variant}")
        descriptor_matrix = descriptor_scores[descriptor_variant]
        row_idx = np.arange(len(error_selected))
        descriptor_selected_score = descriptor_matrix[row_idx, error_selected]
        descriptor_default_score = descriptor_matrix[:, default_idx]
        descriptor_margin_vs_default = descriptor_selected_score - descriptor_default_score
        sorted_descriptor = np.sort(descriptor_matrix, axis=1)
        descriptor_gap = sorted_descriptor[:, -1] - sorted_descriptor[:, -2]
        score_map.update(
            {
                "descriptor_score": descriptor_selected_score,
                "descriptor_margin": descriptor_margin_vs_default,
                "descriptor_joint_margin": (
                    robust_z(error_margin)
                    + 0.65 * robust_z(binary_margin)
                    + 0.40 * robust_z(descriptor_selected_score)
                    + 0.35 * robust_z(descriptor_margin_vs_default)
                    + 0.20 * robust_z(descriptor_gap)
                    - 0.35 * robust_z(error_top1)
                    - 0.25 * robust_z(delta_vs_likpf)
                    - 0.20 * robust_z(pf_std)
                ),
                "descriptor_conservative_margin": (
                    robust_z(error_margin)
                    + robust_z(binary_margin)
                    + 0.75 * robust_z(descriptor_margin_vs_default)
                    + 0.40 * robust_z(descriptor_selected_score)
                    - 0.50 * robust_z(error_top1)
                    - 0.50 * robust_z(delta_vs_likpf)
                    - 0.50 * robust_z(pf_std)
                ),
            }
        )
    variants = []
    caps = [float(value) for value in (get_nested(config, "gate.switch_rate_caps") or [0.01])]
    delta_caps = [
        float(value) for value in (get_nested(config, "gate.max_abs_delta_vs_likpf") or [35.0])
    ]
    std_caps = [
        float(value) for value in (get_nested(config, "gate.max_pf_ancc_std") or [999999.0])
    ]
    score_names = [str(value) for value in (get_nested(config, "gate.confidence_scores") or [])]
    descriptor_score_floors = [
        float(value)
        for value in (get_nested(config, "gate.descriptor.score_floors") or [-np.inf])
    ]
    descriptor_margin_mins = [
        float(value)
        for value in (
            get_nested(config, "gate.descriptor.margin_vs_default_min") or [-np.inf]
        )
    ]
    min_segment_lengths = [
        int(value) for value in (get_nested(config, "gate.min_segment_lengths") or [1])
    ]
    for score_name in score_names:
        if score_name not in score_map:
            raise ValueError(f"Unsupported confidence score: {score_name}")
        raw_score = np.asarray(score_map[score_name], dtype=np.float32)
        uses_descriptor = score_name.startswith("descriptor_")
        floor_values = descriptor_score_floors if uses_descriptor else [-np.inf]
        margin_values = descriptor_margin_mins if uses_descriptor else [-np.inf]
        for switch_cap in caps:
            max_switches = int(math.floor(len(error_selected) * switch_cap))
            if max_switches <= 0:
                continue
            for delta_cap in delta_caps:
                for std_cap in std_caps:
                    for descriptor_floor in floor_values:
                        for descriptor_margin_min in margin_values:
                            descriptor_mask = np.ones(len(error_selected), dtype=bool)
                            if uses_descriptor:
                                if (
                                    descriptor_selected_score is None
                                    or descriptor_margin_vs_default is None
                                ):
                                    raise ValueError(
                                        f"{score_name} requires descriptor score matrices"
                                    )
                                descriptor_mask = (
                                    (descriptor_selected_score >= descriptor_floor)
                                    & (
                                        descriptor_margin_vs_default
                                        >= descriptor_margin_min
                                    )
                                )
                            candidate_mask = (
                                allowed_mask
                                & (error_selected != default_idx)
                                & np.isfinite(raw_score)
                                & (delta_vs_likpf <= delta_cap)
                                & (pf_std <= std_cap)
                                & descriptor_mask
                            )
                            eligible_count = int(np.sum(candidate_mask))
                            selected_base = np.full(
                                len(error_selected),
                                default_idx,
                                dtype=np.int16,
                            )
                            if eligible_count:
                                eligible_idx = np.flatnonzero(candidate_mask)
                                take = min(max_switches, eligible_count)
                                local_scores = raw_score[eligible_idx]
                                if take < eligible_count:
                                    chosen_local = np.argpartition(local_scores, -take)[-take:]
                                    chosen = eligible_idx[chosen_local]
                                    threshold = float(np.min(local_scores[chosen_local]))
                                else:
                                    chosen = eligible_idx
                                    threshold = float(np.min(local_scores))
                                selected_base[chosen] = error_selected[chosen]
                            else:
                                threshold = None
                            for min_segment_len in min_segment_lengths:
                                selected = prune_short_switch_segments(
                                    selected_base,
                                    frame=frame,
                                    default_idx=default_idx,
                                    min_segment_len=min_segment_len,
                                )
                                descriptor_tag = ""
                                if uses_descriptor:
                                    descriptor_tag = (
                                        f"_df{int(round(descriptor_floor * 100)):03d}"
                                        f"_dm{int(round(descriptor_margin_min * 100)):03d}"
                                    )
                                variant = (
                                    f"gate_{score_name}_sr{int(round(switch_cap * 1000)):03d}"
                                    f"_d{int(round(delta_cap)):03d}"
                                    f"_std{int(round(min(std_cap, 999999.0))):06d}"
                                    f"{descriptor_tag}_seg{int(min_segment_len):03d}"
                                )
                                params = {
                                    "score_name": score_name,
                                    "switch_rate_cap": switch_cap,
                                    "max_abs_delta_vs_likpf": delta_cap,
                                    "max_pf_ancc_std": std_cap,
                                    "descriptor_score_variant": (
                                        descriptor_variant if uses_descriptor else None
                                    ),
                                    "descriptor_score_floor": (
                                        descriptor_floor if uses_descriptor else None
                                    ),
                                    "descriptor_margin_vs_default_min": (
                                        descriptor_margin_min if uses_descriptor else None
                                    ),
                                    "min_segment_len": min_segment_len,
                                    "eligible_rows": eligible_count,
                                    "score_threshold": threshold,
                                    "post_segment_switch_rows": int(
                                        np.sum(selected != default_idx)
                                    ),
                                }
                                variants.append((variant, "gated", selected, params))
    return variants


def run_gr_shape_descriptor_verifier(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    paths = ExperimentPaths()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    candidate_names = [spec.name for spec in candidates]
    default_idx = candidate_names.index(
        str(get_nested(config, "gate.default_candidate") or "likpf_mean")
    )

    raw_columns = configured_raw_columns(config, candidates)
    frame, source_meta = load_feature_cache(
        config=config,
        required_columns=raw_columns,
        max_rows=max_rows,
        cache_path=cache_path,
        schema_path=schema_path,
    )
    frame, candidate_values, oracle_labels = add_candidate_labels_and_features(frame, candidates)
    descriptor_scores, descriptor_well_summary = build_descriptor_score_matrices(
        frame=frame,
        candidate_values=candidate_values,
        candidate_names=candidate_names,
        train_dir=paths.train_data_dir,
        descriptor_config=get_nested(config, "gate.descriptor") or {},
    )
    feature_columns, exp101_schema_meta = load_exp101_feature_columns(config)
    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(
            f"exp101 feature schema columns missing after engineering: {missing_features}"
        )

    manifest_path, manifest = load_manifest(config)
    scores, model_manifest = reconstruct_exp101_scores(
        frame=frame,
        candidates=candidates,
        candidate_values=candidate_values,
        oracle_labels=oracle_labels,
        feature_columns=feature_columns,
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    binary_selected, binary_margin = second_margin_high(scores["binary_proba"])
    multiclass_selected, multiclass_margin = second_margin_high(scores["multiclass_proba"])
    error_selected, error_margin, error_top1 = second_margin_low(scores["predicted_error"])
    del binary_selected, multiclass_selected

    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    row_indices = _row_indices_from_ids(frame["id"])
    well_codes, well_names = pd.factorize(frame["well"], sort=True)
    well_codes = well_codes.astype(np.int32)
    order = np.lexsort((row_indices, well_codes))
    bucket_defs: list[tuple[str, np.ndarray, list[str]]] = []
    distance_values = frame["md_since"] if "md_since" in frame.columns else np.nan
    codes, labels = distance_bucket_codes(distance_values)
    bucket_defs.append(("distance_bucket", codes, labels))
    codes, labels = tail_rank_bucket_codes(frame["id"])
    bucket_defs.append(("tail_rank_bucket", codes, labels))
    for source_column, bucket_name in [
        ("eval_len", "eval_len_bucket"),
        ("pf_ancc_std", "pf_seed_std_bucket"),
        ("likpf_mean_d", "likpf_delta_bucket"),
    ]:
        if source_column in frame.columns:
            codes, labels = quantile_bucket_codes(frame[source_column], bucket_name)
            bucket_defs.append((bucket_name, codes, labels))

    variants: list[tuple[str, str, np.ndarray, dict[str, Any]]] = [
        (
            "likpf_mean_single",
            "baseline",
            np.full(len(frame), default_idx, dtype=np.int16),
            {"source": "fixed_default"},
        ),
        (
            "exp101_error_ranker_rowwise",
            "oof",
            error_selected.astype(np.int16),
            {"source": "exp101_lgb_candidate_error_ranker_argmin"},
        ),
        ("oracle", "oracle", oracle_labels.astype(np.int16), {"source": "oracle_best_candidate"}),
    ]
    variants.extend(
        make_gate_variants(
            config=config,
            candidate_names=candidate_names,
            candidate_values=candidate_values,
            frame=frame,
            error_selected=error_selected,
            error_margin=error_margin,
            error_top1=error_top1,
            binary_margin=binary_margin,
            multiclass_margin=multiclass_margin,
            descriptor_scores=descriptor_scores,
        )
    )

    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    by_well_all: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    variant_params: list[dict[str, Any]] = []
    for idx, (variant, mode, selected_idx, params) in enumerate(variants, start=1):
        if idx % int(get_nested(config, "gate.log_period") or 25) == 0:
            print(f"[gate] evaluated {idx}/{len(variants)} variants", flush=True)
        selected_tvt = candidate_values[np.arange(len(selected_idx)), selected_idx].astype(
            np.float32
        )
        metric_rows.append(
            {
                **metrics_for_selection(
                    variant=variant,
                    mode=mode,
                    selected_idx=selected_idx,
                    candidate_values=candidate_values,
                    true_tvt=true_tvt,
                    oracle_labels=oracle_labels,
                    default_idx=default_idx,
                ),
                **{f"param_{key}": value for key, value in params.items()},
            }
        )
        distribution_rows.extend(
            selection_distribution_rows(
                variant=variant,
                mode=mode,
                selected_idx=selected_idx,
                candidate_names=candidate_names,
            )
        )
        by_well_all.extend(
            by_well_rows(
                variant=variant,
                mode=mode,
                selected_idx=selected_idx,
                selected_tvt=selected_tvt,
                true_tvt=true_tvt,
                well_codes=well_codes,
                well_names=well_names,
                order=order,
            )
        )
        bucket_rows.extend(
            bucket_metric_rows(
                variant=variant,
                mode=mode,
                selected_tvt=selected_tvt,
                true_tvt=true_tvt,
                bucket_defs=bucket_defs,
            )
        )
        variant_params.append({"variant": variant, "mode": mode, **params})

    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    distribution = pd.DataFrame(distribution_rows)
    by_well = pd.DataFrame(by_well_all).sort_values(
        ["variant", "mode", "rmse_tvt"], ascending=[True, True, False]
    )
    buckets = pd.DataFrame(bucket_rows).sort_values(["variant", "mode", "bucket_family", "bucket"])
    params_frame = pd.DataFrame(variant_params)

    likpf_rmse = float(metrics.loc[metrics["variant"].eq("likpf_mean_single"), "rmse_tvt"].iloc[0])
    gated = metrics[metrics["mode"].eq("gated")].copy()
    best_gate = gated.sort_values("rmse_tvt").head(1)
    best_gate_variant = str(best_gate["variant"].iloc[0]) if not best_gate.empty else None
    best_gate_delta = (
        float(best_gate["rmse_tvt"].iloc[0] - likpf_rmse) if not best_gate.empty else None
    )
    recommendation = (
        "confidence_gate_supported_for_continuity_audit"
        if best_gate_delta is not None and best_gate_delta < 0.0
        else "confidence_gate_not_supported"
    )

    save_variants = {"likpf_mean_single", "exp101_error_ranker_rowwise", "oracle"}
    if best_gate_variant is not None:
        save_variants.add(best_gate_variant)
    prediction_frames = []
    for variant, mode, selected_idx, _params in variants:
        if variant not in save_variants:
            continue
        prediction_frames.append(
            selected_prediction_frame(
                frame=frame,
                variant=variant,
                mode=mode,
                selected_idx=selected_idx,
                candidate_values=candidate_values,
                true_tvt=true_tvt,
                oracle_labels=oracle_labels,
                candidate_names=candidate_names,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    score_summary = pd.DataFrame(
        [
            {
                "score": name,
                "min": float(np.nanmin(values)),
                "p25": float(np.nanquantile(values, 0.25)),
                "median": float(np.nanmedian(values)),
                "p75": float(np.nanquantile(values, 0.75)),
                "max": float(np.nanmax(values)),
            }
            for name, values in [
                ("error_margin", error_margin),
                ("binary_margin", binary_margin),
                ("multiclass_margin", multiclass_margin),
                ("predicted_error_top1", error_top1),
            ]
        ]
        + descriptor_score_summary_rows(
            descriptor_scores=descriptor_scores,
            candidate_names=candidate_names,
            selected_idx=error_selected,
            default_idx=default_idx,
        )
    )

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    predictions_path = output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz"
    distribution_path = output_dir / f"{OUTPUT_PREFIX}_selection_distribution.csv"
    by_well_path = output_dir / f"{OUTPUT_PREFIX}_by_well.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    params_path = output_dir / f"{OUTPUT_PREFIX}_gate_params.csv"
    score_summary_path = output_dir / f"{OUTPUT_PREFIX}_score_summary.csv"
    model_manifest_path = output_dir / f"{OUTPUT_PREFIX}_exp101_model_manifest_resolved.csv"
    descriptor_well_path = output_dir / f"{OUTPUT_PREFIX}_descriptor_well_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False, compression="gzip")
    distribution.to_csv(distribution_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    params_frame.to_csv(params_path, index=False)
    score_summary.to_csv(score_summary_path, index=False)
    model_manifest.to_csv(model_manifest_path, index=False)
    descriptor_well_summary.to_csv(descriptor_well_path, index=False)

    prediction_hashes = {
        variant: prediction_sha256(group, value_col="selected_tvt")
        for variant, group in predictions.groupby("variant", observed=True)
    }
    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_debug_completed"
        if max_rows is not None
        else "completed_train_side_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "candidates": candidate_names,
        "source": source_meta,
        "exp101": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_path(manifest_path),
            "feature_schema": exp101_schema_meta,
            "resolved_model_count": int(len(model_manifest)),
        },
        "descriptor": {
            "score_variants": list(descriptor_scores),
            "train_dir": str(paths.train_data_dir),
            "well_summary_rows": int(len(descriptor_well_summary)),
            "score_variant_used_for_gate": str(
                get_nested(config, "gate.descriptor.score_variant") or "combo_descriptor_real"
            ),
        },
        "feature_count": int(len(feature_columns)),
        "best_metric": to_jsonable(metrics.iloc[0].to_dict()),
        "decision": {
            "recommendation": recommendation,
            "best_gate_variant": best_gate_variant,
            "best_gate_delta_rmse_vs_likpf": best_gate_delta,
            "likpf_rmse_tvt": likpf_rmse,
        },
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "predictions": sha256_path(predictions_path),
            "predictions_decompressed": sha256_path(predictions_path, decompressed=True),
            "selection_distribution": sha256_path(distribution_path),
            "by_well": sha256_path(by_well_path),
            "bucket_metrics": sha256_path(buckets_path),
            "model_manifest_resolved": sha256_path(model_manifest_path),
            "descriptor_well_summary": sha256_path(descriptor_well_path),
            "prediction_by_variant": prediction_hashes,
        },
        "artifacts": {
            "metrics": metrics_path.name,
            "oof_predictions": predictions_path.name,
            "selection_distribution": distribution_path.name,
            "by_well": by_well_path.name,
            "bucket_metrics": buckets_path.name,
            "gate_params": params_path.name,
            "score_summary": score_summary_path.name,
            "exp101_model_manifest_resolved": model_manifest_path.name,
            "descriptor_well_summary": descriptor_well_path.name,
        },
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--schema-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)
    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not (Path("/kaggle/working").exists())
        else Path("/kaggle/working") / "artifacts"
    )
    max_rows = args.max_rows
    configured_max = get_nested(config, "gate.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_gr_shape_descriptor_verifier(
        output_dir=output_dir,
        cache_path=args.cache_path,
        schema_path=args.schema_path,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
