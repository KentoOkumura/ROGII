from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp145_learned_likelihood_rawtest_feature_generator_parity"
EXP111_PREFIX = "exp111_learned_pf_observation_likelihood_probe"
EXP112_PREFIX = "exp112_learned_pf_likelihood_weight_or_feature_followup"
DEFAULT_EXP099_TRAIN_FEATURES = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
DEFAULT_EXP099_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
DEFAULT_EXP111_SCHEMA = f"{EXP111_PREFIX}_feature_schema.csv"
DEFAULT_EXP111_MANIFEST = f"{EXP111_PREFIX}_model_manifest.json"
DEFAULT_EXP112_SCHEMA = f"{EXP112_PREFIX}_feature_schema.csv"


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


def find_artifact(
    filename: str,
    explicit_path: str | Path | None = None,
    *,
    local_dirs: Iterable[str | Path] = (),
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit = Path(explicit_path)
        candidates.append(explicit if explicit.name == filename else explicit / filename)
    candidates.extend(Path(path) / filename for path in local_dirs)
    candidates.extend([Path.cwd() / filename, Path.cwd() / "artifacts" / filename])
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    checked = "\n".join(str(path) for path in candidates[:120])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "generator.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("generator.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("generator.candidates must not be empty")
    if "likpf_mean" not in {spec.name for spec in specs}:
        raise ValueError("generator.candidates must include likpf_mean")
    return specs


def load_feature_schema(path: Path) -> list[str]:
    schema = pd.read_csv(path)
    if "feature" not in schema.columns:
        raise ValueError(f"{path} must contain a feature column")
    return [str(value) for value in schema["feature"].tolist()]


def source_required_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "last_known_tvt"}
    required.update(spec.column for spec in candidates)
    for key in ["generator.row_context_columns", "generator.multiobs_global_columns"]:
        required.update(str(value) for value in get_nested(config, key) or [])
    for spec in candidates:
        for suffix in ["score", "mae", "ncc"]:
            required.add(f"multiobs_{suffix}_{spec.name}")
    return sorted(required)


def validate_source_header(source: Path, required_columns: list[str]) -> list[str]:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    return header


def numeric_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame


def add_target_free_row_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    include_candidate_values: bool,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    out = numeric_source_frame(frame)
    candidate_values = np.column_stack(
        [
            pd.to_numeric(out[spec.column], errors="coerce").to_numpy(np.float32)
            for spec in candidates
        ]
    )
    if not np.isfinite(candidate_values).all():
        bad = np.argwhere(~np.isfinite(candidate_values))[:5].tolist()
        raise ValueError(f"candidate values contain non-finite values, examples={bad}")

    value_cols = [spec.column for spec in candidates]
    out["candidate_mean"] = out[value_cols].mean(axis=1).astype(np.float32)
    out["candidate_std"] = out[value_cols].std(axis=1).astype(np.float32)
    out["candidate_range"] = (out[value_cols].max(axis=1) - out[value_cols].min(axis=1)).astype(
        np.float32
    )

    engineered = ["candidate_mean", "candidate_std", "candidate_range"]
    for spec in candidates:
        delta_col = f"{spec.name}_minus_last"
        out[delta_col] = out[spec.column].astype(np.float32) - out["last_known_tvt"].astype(
            np.float32
        )
        engineered.append(delta_col)
        if include_candidate_values:
            engineered.append(spec.column)

    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            col = f"{left.name}_vs_{right.name}_abs"
            out[col] = np.abs(
                out[left.column].astype(np.float32) - out[right.column].astype(np.float32)
            )
            engineered.append(col)
    return out, engineered, candidate_values


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.int16)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=np.int16)
    return ranks


def build_candidate_long_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
) -> pd.DataFrame:
    candidate_names = [spec.name for spec in candidates]
    row_mean = candidate_values.mean(axis=1).astype(np.float32)
    row_std = candidate_values.std(axis=1).astype(np.float32)
    row_std_safe = np.where(row_std > 1e-6, row_std, 1.0).astype(np.float32)
    last_known = frame["last_known_tvt"].to_numpy(np.float32)

    score_cols = [f"multiobs_score_{spec.name}" for spec in candidates]
    mae_cols = [f"multiobs_mae_{spec.name}" for spec in candidates]
    ncc_cols = [f"multiobs_ncc_{spec.name}" for spec in candidates]
    score_matrix = (
        frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(np.float32)
    )
    mae_matrix = frame[mae_cols].replace([np.inf, -np.inf], np.nan).fillna(1e9).to_numpy(np.float32)
    ncc_matrix = frame[ncc_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    score_max = score_matrix.max(axis=1).astype(np.float32)
    score_mean = score_matrix.mean(axis=1).astype(np.float32)
    mae_min = mae_matrix.min(axis=1).astype(np.float32)
    ncc_max = ncc_matrix.max(axis=1).astype(np.float32)
    score_rank = rank_desc(score_matrix).astype(np.float32)
    mae_rank = rank_asc(mae_matrix).astype(np.float32)
    ncc_rank = rank_desc(ncc_matrix).astype(np.float32)

    chunks: list[pd.DataFrame] = []
    for cand_idx, _spec in enumerate(candidates):
        part = frame[["id", "well", *row_feature_columns]].copy()
        cand = candidate_values[:, cand_idx].astype(np.float32)
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name"] = candidate_names[cand_idx]
        part["candidate_tvt"] = cand
        part["candidate_minus_last"] = (cand - last_known).astype(np.float32)
        part["candidate_abs_minus_likpf"] = np.abs(
            cand - frame["likpf_mean"].to_numpy(np.float32)
        ).astype(np.float32)
        part["candidate_abs_minus_row_mean"] = np.abs(cand - row_mean).astype(np.float32)
        part["candidate_z_within_row"] = ((cand - row_mean) / row_std_safe).astype(np.float32)
        part["candidate_multiobs_score"] = score_matrix[:, cand_idx]
        part["candidate_multiobs_mae"] = mae_matrix[:, cand_idx]
        part["candidate_multiobs_ncc"] = ncc_matrix[:, cand_idx]
        part["candidate_score_gap_from_best"] = (score_max - score_matrix[:, cand_idx]).astype(
            np.float32
        )
        part["candidate_score_centered"] = (score_matrix[:, cand_idx] - score_mean).astype(
            np.float32
        )
        part["candidate_mae_gap_from_best"] = (mae_matrix[:, cand_idx] - mae_min).astype(np.float32)
        part["candidate_ncc_gap_from_best"] = (ncc_max - ncc_matrix[:, cand_idx]).astype(np.float32)
        part["candidate_score_rank"] = score_rank[:, cand_idx]
        part["candidate_mae_rank"] = mae_rank[:, cand_idx]
        part["candidate_ncc_rank"] = ncc_rank[:, cand_idx]
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def prepare_model_matrix(long_frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in long_frame.columns]
    if missing:
        raise ValueError(f"candidate-long frame missing model features: {missing}")
    values = long_frame[feature_columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.take(medians, np.where(bad)[1])
    return values


def exp111_model_feature_columns(row_feature_columns: list[str]) -> list[str]:
    """Reconstruct exp111 long_feature_columns() order for saved numpy-trained boosters."""
    candidate_columns = [
        "candidate_index",
        "candidate_tvt",
        "candidate_minus_last",
        "candidate_abs_minus_likpf",
        "candidate_abs_minus_row_mean",
        "candidate_z_within_row",
        "candidate_multiobs_score",
        "candidate_multiobs_mae",
        "candidate_multiobs_ncc",
        "candidate_score_gap_from_best",
        "candidate_score_centered",
        "candidate_mae_gap_from_best",
        "candidate_ncc_gap_from_best",
        "candidate_score_rank",
        "candidate_mae_rank",
        "candidate_ncc_rank",
    ]
    return [*row_feature_columns, *candidate_columns]


def load_exp111_models(
    *,
    manifest_path: Path,
    model_root: Path | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    import lightgbm as lgb

    manifest = json.loads(manifest_path.read_text())
    root = model_root or manifest_path.parent
    models = manifest.get("models") or []
    classifier_rows = [item for item in models if item.get("variant") == "within10_classifier"]
    error_rows = [item for item in models if item.get("variant") == "expected_error_regressor"]
    if len(classifier_rows) != 1 or len(error_rows) != 1:
        raise ValueError(f"expected one classifier and one regressor in {manifest_path}")
    classifier_row = classifier_rows[0]
    error_row = error_rows[0]
    classifier_path = root / str(classifier_row["path"])
    error_path = root / str(error_row["path"])
    if not classifier_path.exists() or not error_path.exists():
        raise FileNotFoundError(f"exp111 model files are missing under {root}")
    classifier = lgb.Booster(model_file=str(classifier_path))
    error_model = lgb.Booster(model_file=str(error_path))
    meta = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "model_root": str(root),
        "classifier": {**classifier_row, "resolved_path": str(classifier_path)},
        "expected_error": {**error_row, "resolved_path": str(error_path)},
        "classifier_sha256_actual": sha256_path(classifier_path),
        "expected_error_sha256_actual": sha256_path(error_path),
    }
    return classifier, error_model, meta


def predict_likelihood_matrices(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    candidate_values: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    long_frame = build_candidate_long_features(
        frame,
        candidates,
        row_feature_columns=row_feature_columns,
        candidate_values=candidate_values,
    )
    expected_features = int(classifier.num_feature())
    if len(model_feature_columns) != expected_features:
        raise ValueError(
            f"Reconstructed exp111 model feature count mismatch: "
            f"{len(model_feature_columns)} != booster num_feature {expected_features}"
        )
    x_matrix = prepare_model_matrix(long_frame, model_feature_columns)
    n_rows = len(frame)
    n_candidates = len(candidates)
    probability = classifier.predict(x_matrix).astype(np.float32).reshape(n_candidates, n_rows).T
    pred_error = error_model.predict(x_matrix).astype(np.float32).reshape(n_candidates, n_rows).T
    pred_error = np.maximum(pred_error, 0.0).astype(np.float32)

    def pivot_long_value(column: str) -> np.ndarray:
        return long_frame[column].to_numpy(np.float32).reshape(n_candidates, n_rows).T

    context = {
        "base": frame[["id", "well", "fold", "md_since"]].reset_index(drop=True).copy(),
        "candidate_tvt": candidate_values.astype(np.float32),
        "probability": probability,
        "pred_error": pred_error,
        "multiobs_score": pivot_long_value("candidate_multiobs_score"),
        "multiobs_mae": pivot_long_value("candidate_multiobs_mae"),
        "multiobs_ncc": pivot_long_value("candidate_multiobs_ncc"),
    }
    long_out = long_frame[
        ["id", "well", "candidate_name", "candidate_index", "candidate_tvt"]
    ].copy()
    long_out["pred_within10_prob"] = probability.T.reshape(-1).astype(np.float32)
    long_out["pred_abs_error"] = pred_error.T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_score"] = context["multiobs_score"].T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_mae"] = context["multiobs_mae"].T.reshape(-1).astype(np.float32)
    long_out["baseline_multiobs_ncc"] = context["multiobs_ncc"].T.reshape(-1).astype(np.float32)
    long_out["md_since"] = np.tile(frame["md_since"].to_numpy(np.float32), n_candidates)
    return context, long_out


def build_ml_features(
    context: dict[str, Any], candidates: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    base = context["base"][["id", "well", "fold", "md_since"]].reset_index(drop=True).copy()
    probability = context["probability"]
    pred_error = context["pred_error"]
    candidate_tvt = context["candidate_tvt"]
    multiobs_score = context["multiobs_score"]
    multiobs_mae = context["multiobs_mae"]
    multiobs_ncc = context["multiobs_ncc"]
    likpf_idx = candidates.index("likpf_mean")

    prob_order = np.argsort(-probability, axis=1)
    err_order = np.argsort(pred_error, axis=1)
    prob_sorted = np.take_along_axis(probability, prob_order, axis=1)
    err_sorted = np.take_along_axis(pred_error, err_order, axis=1)
    entropy = -np.sum(
        np.clip(probability, 1e-6, 1.0) * np.log(np.clip(probability, 1e-6, 1.0)), axis=1
    )

    out = base
    out["learned_prob_top1_index"] = prob_order[:, 0].astype(np.int16)
    out["learned_error_top1_index"] = err_order[:, 0].astype(np.int16)
    out["learned_prob_top1_value"] = prob_sorted[:, 0].astype(np.float32)
    out["learned_prob_top2_value"] = prob_sorted[:, 1].astype(np.float32)
    out["learned_prob_margin_top1_top2"] = (prob_sorted[:, 0] - prob_sorted[:, 1]).astype(
        np.float32
    )
    out["learned_prob_entropy"] = entropy.astype(np.float32)
    out["learned_error_top1_value"] = err_sorted[:, 0].astype(np.float32)
    out["learned_error_top2_value"] = err_sorted[:, 1].astype(np.float32)
    out["learned_error_margin_top2_top1"] = (err_sorted[:, 1] - err_sorted[:, 0]).astype(np.float32)
    out["learned_prob_likpf_rank"] = rank_desc(probability)[:, likpf_idx].astype(np.int16)
    out["learned_error_likpf_rank"] = rank_asc(pred_error)[:, likpf_idx].astype(np.int16)
    out["learned_prob_top3_contains_likpf"] = (rank_desc(probability)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["learned_error_top3_contains_likpf"] = (rank_asc(pred_error)[:, likpf_idx] < 3).astype(
        np.int8
    )
    out["candidate_tvt_std"] = candidate_tvt.std(axis=1).astype(np.float32)
    out["candidate_tvt_range"] = (candidate_tvt.max(axis=1) - candidate_tvt.min(axis=1)).astype(
        np.float32
    )
    prob_sum = probability.sum(axis=1)
    prob_sum = np.where(prob_sum > 1e-6, prob_sum, 1.0)
    out["learned_prob_weighted_tvt"] = (
        np.sum(candidate_tvt * probability, axis=1) / prob_sum
    ).astype(np.float32)
    inv_error_weight = 1.0 / np.maximum(pred_error, 1e-3)
    inv_error_sum = inv_error_weight.sum(axis=1)
    out["learned_error_weighted_tvt"] = (
        np.sum(candidate_tvt * inv_error_weight, axis=1) / inv_error_sum
    ).astype(np.float32)

    include_candidate_tvt = bool(
        get_nested(config, "generator.feature_cache.include_candidate_tvt")
    )
    include_multiobs = bool(get_nested(config, "generator.feature_cache.include_multiobs_scores"))
    for idx, candidate in enumerate(candidates):
        out[f"learned_prob_{candidate}"] = probability[:, idx].astype(np.float32)
        out[f"learned_pred_abs_error_{candidate}"] = pred_error[:, idx].astype(np.float32)
        if include_candidate_tvt:
            out[f"candidate_tvt_{candidate}"] = candidate_tvt[:, idx].astype(np.float32)
        if include_multiobs:
            out[f"multiobs_score_{candidate}"] = multiobs_score[:, idx].astype(np.float32)
            out[f"multiobs_mae_{candidate}"] = multiobs_mae[:, idx].astype(np.float32)
            out[f"multiobs_ncc_{candidate}"] = multiobs_ncc[:, idx].astype(np.float32)
    return out


def generate_ml_features_from_frame(
    frame: pd.DataFrame,
    *,
    candidates: list[CandidateSpec],
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    config: dict[str, Any],
    default_fold: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, _engineered, candidate_values = add_target_free_row_features(
        frame,
        candidates,
        include_candidate_values=bool(get_nested(config, "generator.include_candidate_values")),
    )
    if "fold" not in frame.columns:
        frame["fold"] = np.int16(default_fold)
    if "md_since" not in frame.columns:
        extracted = frame["id"].astype(str).str.extract(r"_(\d+)$", expand=False)
        frame["md_since"] = pd.to_numeric(extracted, errors="coerce").fillna(0).astype(np.float32)
    context, long_likelihood = predict_likelihood_matrices(
        frame,
        candidates,
        row_feature_columns=row_feature_columns,
        model_feature_columns=model_feature_columns,
        classifier=classifier,
        error_model=error_model,
        candidate_values=candidate_values,
    )
    ml_features = build_ml_features(
        context,
        [spec.name for spec in candidates],
        config,
    )
    feature_cols = [col for col in ml_features.columns if col not in {"id", "well"}]
    for col in feature_cols:
        ml_features[col] = pd.to_numeric(ml_features[col], errors="coerce").astype(np.float32)
    if not np.isfinite(ml_features[feature_cols].to_numpy(np.float32)).all():
        raise ValueError("generated ML feature frame contains non-finite values")
    return ml_features, long_likelihood


def schema_parity_frame(generated_columns: list[str], reference_columns: list[str]) -> pd.DataFrame:
    rows = []
    max_len = max(len(generated_columns), len(reference_columns))
    for index in range(max_len):
        generated = generated_columns[index] if index < len(generated_columns) else None
        reference = reference_columns[index] if index < len(reference_columns) else None
        rows.append(
            {
                "feature_index": index,
                "generated_feature": generated,
                "reference_feature": reference,
                "matches_position": generated == reference,
                "generated_only": generated is not None and generated not in reference_columns,
                "reference_only": reference is not None and reference not in generated_columns,
            }
        )
    return pd.DataFrame(rows)


def output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"feature_index": range(len(frame.columns)), "feature": frame.columns})


def write_ml_features(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, compression="gzip")


def generate_from_cache(
    *,
    source: Path,
    output_path: Path,
    candidates: list[CandidateSpec],
    row_feature_columns: list[str],
    model_feature_columns: list[str],
    classifier: Any,
    error_model: Any,
    config: dict[str, Any],
    required_columns: list[str],
    chunksize: int,
    max_rows: int | None,
) -> dict[str, Any]:
    validate_source_header(source, required_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    wells: set[str] = set()
    first = True
    long_sample: pd.DataFrame | None = None
    with gzip.open(output_path, "wt", newline="") as fp:
        reader = pd.read_csv(
            source,
            usecols=required_columns,
            dtype={"id": str, "well": str},
            chunksize=chunksize,
            nrows=max_rows,
            low_memory=False,
        )
        for chunk_index, chunk in enumerate(reader):
            features, long_likelihood = generate_ml_features_from_frame(
                chunk,
                candidates=candidates,
                row_feature_columns=row_feature_columns,
                model_feature_columns=model_feature_columns,
                classifier=classifier,
                error_model=error_model,
                config=config,
            )
            if long_sample is None:
                long_sample = long_likelihood.head(50).copy()
            features.to_csv(fp, index=False, header=first)
            first = False
            rows += int(len(features))
            wells.update(features["well"].astype(str).unique().tolist())
            print(f"[cache chunk {chunk_index}] rows={rows:,} wells={len(wells):,}", flush=True)
    return {
        "path": str(output_path),
        "rows": rows,
        "wells": len(wells),
        "sha256": sha256_path(output_path),
        "decompressed_sha256": sha256_path(output_path, decompressed=True),
        "long_likelihood_sample": long_sample,
    }


def generate_rawtest_frame_from_replay(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from public_notebook_replay_audit import (
        build_replay_test_frame,
        configure_public_runtime,
        feature_columns_for_variant,
    )

    output_dir = Path(get_nested(config, "runtime.replay_work_dir") or "/tmp/exp145_replay")
    data_dir = ExperimentPaths().raw_data_dir
    replay = get_nested(config, "generator.rawtest_replay") or {}
    configure_public_runtime(
        data_dir=data_dir,
        output_dir=output_dir,
        n_jobs=int(replay.get("n_jobs") or 8),
        pf_seeds=int(replay.get("pf_seeds") or 128),
        pf_particles=int(replay.get("pf_particles") or 500),
        fast=bool(replay.get("fast", False)),
        use_gpu="cpu",
        n_train_wells=None,
    )
    test_frame, meta = build_replay_test_frame()
    variant = str(
        get_nested(config, "generator.rawtest_replay.variant") or "pixiux_likpf_public_replay"
    )
    feature_columns = feature_columns_for_variant(test_frame, variant)
    meta["feature_columns_for_variant"] = int(len(feature_columns))
    return test_frame, meta


def ensure_candidate_value_columns(
    frame: pd.DataFrame, candidates: list[CandidateSpec]
) -> pd.DataFrame:
    out = frame.copy()
    last_known = pd.to_numeric(out["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    for spec in candidates:
        if spec.column in out.columns:
            continue
        delta_col = f"{spec.name}_d"
        if delta_col in out.columns:
            out[spec.column] = last_known + pd.to_numeric(out[delta_col], errors="coerce").to_numpy(
                np.float32
            )
    return out


def ensure_multiobs_columns(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    required_multiobs = [
        str(value) for value in get_nested(config, "generator.multiobs_global_columns") or []
    ]
    for spec in candidates:
        for suffix in ["score", "mae", "ncc"]:
            required_multiobs.append(f"multiobs_{suffix}_{spec.name}")
    missing = [column for column in required_multiobs if column not in frame.columns]
    if not missing:
        return frame, None

    from pf_multi_observation_likelihood_probe import build_multi_observation_candidate_frame

    out = ensure_candidate_value_columns(frame, candidates)
    candidate_names = [spec.name for spec in candidates]
    missing_candidates = [column for column in candidate_names if column not in out.columns]
    if missing_candidates:
        raise ValueError(f"raw-test frame missing candidate value columns: {missing_candidates}")

    existing_candidates = out[["id", "well", *candidate_names]].copy()
    multiobs_config = get_nested(config, "generator.multi_observation_likelihood") or {}
    multiobs, well_summary = build_multi_observation_candidate_frame(
        out,
        existing_candidates,
        train_dir=ExperimentPaths().test_data_dir,
        candidate_names=candidate_names,
        config=multiobs_config,
    )
    out = out.merge(multiobs, on=["id", "well"], how="left", validate="one_to_one")
    remaining = [column for column in required_multiobs if column not in out.columns]
    if remaining:
        raise ValueError(f"failed to generate raw-test multiobs columns: {remaining}")
    return out, {
        "mode": "generated_exp099_multiobs_from_rawtest_prefix_gr",
        "initial_missing_columns": missing,
        "generated_columns": sorted(set(required_multiobs)),
        "well_summary_rows": int(len(well_summary)),
        "well_summary": to_jsonable(well_summary.to_dict("records")),
    }


def run_generator(
    *,
    output_dir: str | Path,
    mode: str,
    train_cache_path: str | Path | None,
    rawtest_cache_path: str | Path | None,
    exp111_schema_path: str | Path | None,
    exp111_manifest_path: str | Path | None,
    exp112_schema_path: str | Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidate_specs_from_config(config)
    exp111_artifacts = Path(str(get_nested(config, "data.exp111_artifact_dir_local") or ""))
    exp112_artifacts = Path(str(get_nested(config, "data.exp112_artifact_dir_local") or ""))
    exp099_artifacts = Path(str(get_nested(config, "data.exp099_artifact_dir_local") or ""))
    schema_path = find_artifact(
        DEFAULT_EXP111_SCHEMA,
        exp111_schema_path or get_nested(config, "data.exp111_feature_schema"),
        local_dirs=[exp111_artifacts],
    )
    manifest_path = find_artifact(
        DEFAULT_EXP111_MANIFEST,
        exp111_manifest_path or get_nested(config, "data.exp111_model_manifest"),
        local_dirs=[exp111_artifacts],
    )
    reference_schema_path = find_artifact(
        DEFAULT_EXP112_SCHEMA,
        exp112_schema_path or get_nested(config, "data.exp112_feature_schema"),
        local_dirs=[exp112_artifacts],
    )
    row_feature_columns = load_feature_schema(schema_path)
    model_feature_columns = exp111_model_feature_columns(row_feature_columns)
    reference_columns = load_feature_schema(reference_schema_path)
    classifier, error_model, model_meta = load_exp111_models(manifest_path=manifest_path)

    required_columns = source_required_columns(config, candidates)
    chunksize = int(get_nested(config, "generator.chunksize") or 200_000)
    outputs: dict[str, Any] = {}
    generated_columns: list[str] | None = None

    if mode in {"train", "both"}:
        source = find_artifact(
            DEFAULT_EXP099_TRAIN_FEATURES,
            train_cache_path or get_nested(config, "data.exp099_train_feature_cache_local"),
            local_dirs=[exp099_artifacts],
        )
        meta = generate_from_cache(
            source=source,
            output_path=output_dir / f"{OUTPUT_PREFIX}_full_train_ml_features.csv.gz",
            candidates=candidates,
            row_feature_columns=row_feature_columns,
            model_feature_columns=model_feature_columns,
            classifier=classifier,
            error_model=error_model,
            config=config,
            required_columns=required_columns,
            chunksize=chunksize,
            max_rows=max_rows,
        )
        outputs["full_train_ml_features"] = {
            k: v for k, v in meta.items() if k != "long_likelihood_sample"
        }
        if meta["long_likelihood_sample"] is not None:
            sample_path = output_dir / f"{OUTPUT_PREFIX}_full_train_likelihood_long_sample.csv"
            meta["long_likelihood_sample"].to_csv(sample_path, index=False)
            outputs["full_train_likelihood_long_sample"] = str(sample_path)
        generated_columns = pd.read_csv(meta["path"], nrows=0).columns.tolist()

    if mode in {"rawtest", "both"}:
        if rawtest_cache_path is not None:
            test_frame = pd.read_csv(rawtest_cache_path, dtype={"id": str, "well": str})
            rawtest_meta: dict[str, Any] = {
                "source": str(rawtest_cache_path),
                "source_sha256": sha256_path(Path(rawtest_cache_path)),
                "mode": "provided_rawtest_feature_cache",
            }
        else:
            test_frame, rawtest_meta = generate_rawtest_frame_from_replay(config)
        multiobs_meta: dict[str, Any] | None = None
        test_frame, multiobs_meta = ensure_multiobs_columns(
            test_frame,
            candidates,
            config=config,
        )
        test_features, long_likelihood = generate_ml_features_from_frame(
            test_frame[required_columns],
            candidates=candidates,
            row_feature_columns=row_feature_columns,
            model_feature_columns=model_feature_columns,
            classifier=classifier,
            error_model=error_model,
            config=config,
        )
        test_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_ml_features.csv.gz"
        long_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_likelihood_long.csv.gz"
        write_ml_features(test_path, test_features)
        long_likelihood.to_csv(long_path, index=False, compression="gzip")
        outputs["rawtest_ml_features"] = {
            "path": str(test_path),
            "rows": int(len(test_features)),
            "wells": int(test_features["well"].nunique()),
            "sha256": sha256_path(test_path),
            "decompressed_sha256": sha256_path(test_path, decompressed=True),
        }
        outputs["rawtest_likelihood_long"] = {
            "path": str(long_path),
            "rows": int(len(long_likelihood)),
            "sha256": sha256_path(long_path),
            "decompressed_sha256": sha256_path(long_path, decompressed=True),
        }
        outputs["rawtest_source"] = rawtest_meta
        if multiobs_meta is not None:
            outputs["rawtest_multiobs_generation"] = multiobs_meta
        generated_columns = test_features.columns.tolist()

    if generated_columns is None:
        raise ValueError("mode must generate at least one feature file")

    schema_path_out = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    parity_path = output_dir / f"{OUTPUT_PREFIX}_schema_parity.csv"
    output_schema(pd.DataFrame(columns=generated_columns)).to_csv(schema_path_out, index=False)
    parity = schema_parity_frame(generated_columns, reference_columns)
    parity.to_csv(parity_path, index=False)
    parity_pass = bool(
        len(generated_columns) == len(reference_columns)
        and all(
            left == right for left, right in zip(generated_columns, reference_columns, strict=True)
        )
    )

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_smoke_completed"
        if max_rows is not None
        else "implemented_not_run_full",
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "runtime_seconds": float(time.time() - t0),
        "candidates": [spec.name for spec in candidates],
        "row_feature_schema": {
            "path": str(schema_path),
            "sha256": sha256_path(schema_path),
            "feature_count": int(len(row_feature_columns)),
        },
        "exp111_model_feature_order": {
            "feature_count": int(len(model_feature_columns)),
            "features": model_feature_columns,
            "source": "reconstructed_from_exp111_long_feature_columns_order",
        },
        "reference_exp112_schema": {
            "path": str(reference_schema_path),
            "sha256": sha256_path(reference_schema_path),
            "columns": int(len(reference_columns)),
        },
        "generated_schema": {
            "path": str(schema_path_out),
            "sha256": sha256_path(schema_path_out),
            "columns": int(len(generated_columns)),
            "parity_path": str(parity_path),
            "parity_sha256": sha256_path(parity_path),
            "schema_parity_pass": parity_pass,
            "mismatch_rows": int((~parity["matches_position"]).sum()),
        },
        "model": model_meta,
        "outputs": outputs,
        "known_limitations": [
            (
                "exp111 saved fold0 models did not persist the training imputation medians; "
                "this generator imputes per generated batch before LightGBM prediction."
            ),
            "The generator is target-free and does not produce a submission.csv.",
        ],
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    with summary_path.open("w") as fp:
        json.dump(to_jsonable(summary), fp, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["train", "rawtest", "both"], default="train")
    parser.add_argument("--train-cache-path", type=Path, default=None)
    parser.add_argument("--rawtest-cache-path", type=Path, default=None)
    parser.add_argument("--exp111-schema-path", type=Path, default=None)
    parser.add_argument("--exp111-manifest-path", type=Path, default=None)
    parser.add_argument("--exp112-schema-path", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)

    paths = ExperimentPaths()
    config = load_config()
    output_dir = args.output_dir or (
        paths.artifacts_dir
        if not Path("/kaggle/working").exists()
        else Path("/kaggle/working") / "artifacts"
    )
    max_rows = args.max_rows
    configured_max = get_nested(config, "generator.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_generator(
        output_dir=output_dir,
        mode=args.mode,
        train_cache_path=args.train_cache_path,
        rawtest_cache_path=args.rawtest_cache_path,
        exp111_schema_path=args.exp111_schema_path,
        exp111_manifest_path=args.exp111_manifest_path,
        exp112_schema_path=args.exp112_schema_path,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
