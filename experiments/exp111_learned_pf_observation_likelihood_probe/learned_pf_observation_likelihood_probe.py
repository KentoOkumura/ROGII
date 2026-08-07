from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp111_learned_pf_observation_likelihood_probe"
DEFAULT_TRAIN_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_train_features.csv.gz"
)
DEFAULT_TRAIN_FEATURE_SCHEMA = (
    "exp099_pf_multi_observation_likelihood_probe_multiobs_likelihood_probe_feature_schema.csv"
)
PROTECTED_COLUMNS = {"id", "well", "target", "true_tvt", "oracle_label", "oracle_candidate"}


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
    for row in frame[["id", "candidate_name", value_col]].itertuples(index=False):
        digest.update(str(row.id).encode("utf-8"))
        digest.update(b",")
        digest.update(str(row.candidate_name).encode("utf-8"))
        digest.update(b",")
        digest.update(np.float64(row[2]).tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def find_artifact(filename: str, explicit_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
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
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked:\n{checked}")


def candidate_specs_from_config(config: dict[str, Any]) -> list[CandidateSpec]:
    values = get_nested(config, "likelihood.candidates") or []
    specs: list[CandidateSpec] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("likelihood.candidates entries must be mappings")
        specs.append(
            CandidateSpec(name=str(item["name"]), column=str(item.get("column", item["name"])))
        )
    if not specs:
        raise ValueError("likelihood.candidates must not be empty")
    return specs


def build_required_columns(config: dict[str, Any], candidates: list[CandidateSpec]) -> list[str]:
    required = {"id", "well", "target", "last_known_tvt"}
    required.update(spec.column for spec in candidates)
    for key in [
        "likelihood.row_context_columns",
        "likelihood.multiobs_global_columns",
        "likelihood.optional_columns",
    ]:
        values = get_nested(config, key) or []
        required.update(str(value) for value in values)
    for spec in candidates:
        for suffix in ["score", "mae", "ncc"]:
            required.add(f"multiobs_{suffix}_{spec.name}")
    return sorted(required)


def load_train_feature_cache(
    *,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    required_columns: list[str],
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = find_artifact(DEFAULT_TRAIN_FEATURE_CACHE, cache_path)
    schema = find_artifact(DEFAULT_TRAIN_FEATURE_SCHEMA, schema_path)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    frame = pd.read_csv(
        source,
        usecols=required_columns,
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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred.astype(np.float64) - y_true.astype(np.float64)))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred.astype(np.float64) - y_true.astype(np.float64))))


def _row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int32)


def _distance_bucket(values: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def _tail_rank_bucket(ids: pd.Series) -> pd.Categorical:
    ranks = _row_indices_from_ids(ids)
    return pd.cut(
        ranks,
        bins=[-np.inf, 99, 249, 499, 999, np.inf],
        labels=["000_099", "100_249", "250_499", "500_999", "1000_plus"],
        include_lowest=True,
    )


def fit_impute(
    train: pd.DataFrame, valid: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    valid_values = valid[columns].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    train_bad = ~np.isfinite(train_values)
    valid_bad = ~np.isfinite(valid_values)
    if train_bad.any():
        train_values[train_bad] = np.take(medians, np.where(train_bad)[1])
    if valid_bad.any():
        valid_values[valid_bad] = np.take(medians, np.where(valid_bad)[1])
    return train_values, valid_values


def add_row_level_features(
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    *,
    include_candidate_values: bool,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
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
    return out, engineered, candidate_values, oracle_labels


def select_row_feature_columns(
    frame: pd.DataFrame,
    config: dict[str, Any],
    engineered_columns: list[str],
) -> list[str]:
    configured = [
        str(value)
        for value in (
            (get_nested(config, "likelihood.row_context_columns") or [])
            + (get_nested(config, "likelihood.multiobs_global_columns") or [])
        )
    ]
    columns: list[str] = []
    for column in configured + engineered_columns:
        if column in frame.columns and column not in PROTECTED_COLUMNS and column not in columns:
            columns.append(column)
    missing = [column for column in configured if column not in frame.columns]
    if missing:
        raise ValueError(f"configured feature columns are missing: {missing}")
    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()
    ]
    if not numeric_columns:
        raise ValueError("no numeric row feature columns selected")
    return numeric_columns


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row = np.arange(values.shape[0])[:, None]
    ranks[row, order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def _rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row = np.arange(values.shape[0])[:, None]
    ranks[row, order] = np.arange(values.shape[1], dtype=np.float32)
    return ranks


def build_long_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    candidates: list[CandidateSpec],
    *,
    row_feature_columns: list[str],
    candidate_values: np.ndarray,
    sample_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if sample_rows is not None and len(row_indices) > sample_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=int(sample_rows), replace=False))

    candidate_names = [spec.name for spec in candidates]
    true_tvt = frame["true_tvt"].to_numpy(np.float32)
    last_known = frame["last_known_tvt"].to_numpy(np.float32)
    row_mean = candidate_values.mean(axis=1).astype(np.float32)
    row_std = candidate_values.std(axis=1).astype(np.float32)
    row_std_safe = np.where(row_std > 1e-6, row_std, 1.0).astype(np.float32)

    score_cols = [f"multiobs_score_{spec.name}" for spec in candidates]
    mae_cols = [f"multiobs_mae_{spec.name}" for spec in candidates]
    ncc_cols = [f"multiobs_ncc_{spec.name}" for spec in candidates]
    score_matrix = frame[score_cols].replace([np.inf, -np.inf], np.nan).fillna(-1e9).to_numpy(
        np.float32
    )
    mae_matrix = frame[mae_cols].replace([np.inf, -np.inf], np.nan).fillna(1e9).to_numpy(
        np.float32
    )
    ncc_matrix = frame[ncc_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
        np.float32
    )
    score_max = score_matrix.max(axis=1).astype(np.float32)
    score_mean = score_matrix.mean(axis=1).astype(np.float32)
    mae_min = mae_matrix.min(axis=1).astype(np.float32)
    ncc_max = ncc_matrix.max(axis=1).astype(np.float32)
    score_rank = _rank_desc(score_matrix)
    mae_rank = _rank_asc(mae_matrix)
    ncc_rank = _rank_desc(ncc_matrix)

    chunks: list[pd.DataFrame] = []
    for cand_idx, _spec in enumerate(candidates):
        part = frame.iloc[row_indices][["id", "well", *row_feature_columns]].copy()
        cand = candidate_values[row_indices, cand_idx].astype(np.float32)
        error = np.abs(cand - true_tvt[row_indices]).astype(np.float32)
        part["candidate_index"] = np.int16(cand_idx)
        part["candidate_name"] = candidate_names[cand_idx]
        part["candidate_tvt"] = cand
        part["candidate_minus_last"] = (cand - last_known[row_indices]).astype(np.float32)
        part["candidate_abs_minus_likpf"] = np.abs(
            cand - frame["likpf_mean"].to_numpy(np.float32)[row_indices]
        ).astype(np.float32)
        part["candidate_abs_minus_row_mean"] = np.abs(cand - row_mean[row_indices]).astype(
            np.float32
        )
        z_within_row = (cand - row_mean[row_indices]) / row_std_safe[row_indices]
        part["candidate_z_within_row"] = z_within_row.astype(np.float32)
        part["candidate_multiobs_score"] = score_matrix[row_indices, cand_idx]
        part["candidate_multiobs_mae"] = mae_matrix[row_indices, cand_idx]
        part["candidate_multiobs_ncc"] = ncc_matrix[row_indices, cand_idx]
        part["candidate_score_gap_from_best"] = (
            score_max[row_indices] - score_matrix[row_indices, cand_idx]
        ).astype(np.float32)
        part["candidate_score_centered"] = (
            score_matrix[row_indices, cand_idx] - score_mean[row_indices]
        ).astype(np.float32)
        part["candidate_mae_gap_from_best"] = (
            mae_matrix[row_indices, cand_idx] - mae_min[row_indices]
        ).astype(np.float32)
        part["candidate_ncc_gap_from_best"] = (
            ncc_max[row_indices] - ncc_matrix[row_indices, cand_idx]
        ).astype(np.float32)
        part["candidate_score_rank"] = score_rank[row_indices, cand_idx]
        part["candidate_mae_rank"] = mae_rank[row_indices, cand_idx]
        part["candidate_ncc_rank"] = ncc_rank[row_indices, cand_idx]
        part["abs_error"] = error
        part["within_5ft"] = (error <= 5.0).astype(np.int8)
        part["within_10ft"] = (error <= 10.0).astype(np.int8)
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def long_feature_columns(long_frame: pd.DataFrame) -> list[str]:
    blocked = {"id", "well", "candidate_name", "abs_error", "within_5ft", "within_10ft"}
    return [
        column
        for column in long_frame.columns
        if column not in blocked and pd.api.types.is_numeric_dtype(long_frame[column])
    ]


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _safe_logloss(y_true: np.ndarray, prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, np.clip(prob, 1e-6, 1.0 - 1e-6)))


def make_calibration_table(long_oof: pd.DataFrame, score_column: str) -> pd.DataFrame:
    work = long_oof[["within_10ft", "abs_error", score_column]].copy()
    finite = work[np.isfinite(work[score_column])]
    if finite.empty:
        return pd.DataFrame()
    ranks = finite[score_column].rank(method="first")
    finite["bin"] = pd.qcut(ranks, q=min(10, len(finite)), duplicates="drop")
    rows = []
    for bin_value, group in finite.groupby("bin", observed=True):
        rows.append(
            {
                "score": score_column,
                "bin": str(bin_value),
                "rows": int(len(group)),
                "score_min": float(group[score_column].min()),
                "score_max": float(group[score_column].max()),
                "score_mean": float(group[score_column].mean()),
                "observed_within10": float(group["within_10ft"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def topk_metrics(
    *,
    frame: pd.DataFrame,
    eval_indices: np.ndarray,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    probability: np.ndarray,
    predicted_error: np.ndarray,
) -> pd.DataFrame:
    true_tvt = frame["true_tvt"].to_numpy(np.float32)[eval_indices]
    eval_values = candidate_values[eval_indices]
    abs_errors = np.abs(eval_values - true_tvt[:, None])
    candidate_names = [spec.name for spec in candidates]
    rows: list[dict[str, Any]] = []

    def add_selected(name: str, selected: np.ndarray) -> None:
        pred = eval_values[np.arange(len(eval_indices)), selected]
        error = np.abs(pred - true_tvt)
        rows.append(
            {
                "variant": name,
                "mode": "diagnostic_top1",
                "rows": int(len(eval_indices)),
                "rmse_tvt": rmse(true_tvt, pred),
                "mae_tvt": mae(true_tvt, pred),
                "within_5ft": float(np.mean(error <= 5.0)),
                "within_10ft": float(np.mean(error <= 10.0)),
                "pf_ancc_selection_rate": float(
                    np.mean(np.asarray(candidate_names, dtype=object)[selected] == "pf_ancc")
                ),
            }
        )

    add_selected(
        "likpf_mean_single",
        np.full(len(eval_indices), candidate_names.index("likpf_mean"), dtype=np.int16),
    )
    add_selected("learned_prob_top1", np.argmax(probability, axis=1).astype(np.int16))
    add_selected("learned_error_top1", np.argmin(predicted_error, axis=1).astype(np.int16))
    score_cols = [f"multiobs_score_{spec.name}" for spec in candidates]
    score_matrix = frame.iloc[eval_indices][score_cols].to_numpy(np.float32)
    add_selected("multiobs_score_top1", np.argmax(score_matrix, axis=1).astype(np.int16))

    order_prob = np.argsort(-probability, axis=1)
    order_error = np.argsort(predicted_error, axis=1)
    order_score = np.argsort(-score_matrix, axis=1)
    for family, order in [
        ("learned_prob", order_prob),
        ("learned_error", order_error),
        ("multiobs_score", order_score),
    ]:
        for k in [1, 2, 3]:
            chosen = order[:, :k]
            chosen_errors = np.take_along_axis(abs_errors, chosen, axis=1)
            rows.append(
                {
                    "variant": f"{family}_top{k}_coverage",
                    "mode": "topk_coverage",
                    "rows": int(len(eval_indices)),
                    "top_k": k,
                    "oracle_topk_rmse": float(np.sqrt(np.mean(np.min(chosen_errors, axis=1) ** 2))),
                    "oracle_topk_mae": float(np.mean(np.min(chosen_errors, axis=1))),
                    "topk_within_5ft": float(np.mean(np.any(chosen_errors <= 5.0, axis=1))),
                    "topk_within_10ft": float(np.mean(np.any(chosen_errors <= 10.0, axis=1))),
                }
            )
    return pd.DataFrame(rows)


def bucket_likelihood_metrics(long_oof: pd.DataFrame) -> pd.DataFrame:
    context = long_oof[["id", "within_10ft", "pred_within10_prob", "abs_error"]].copy()
    context["distance_bucket"] = _distance_bucket(long_oof.get("md_since", np.nan))
    context["tail_rank_bucket"] = _tail_rank_bucket(long_oof["id"])
    rows = []
    for bucket_col in ["distance_bucket", "tail_rank_bucket"]:
        for bucket, group in context.groupby(bucket_col, observed=True):
            y = group["within_10ft"].to_numpy(np.int8)
            p = group["pred_within10_prob"].to_numpy(np.float32)
            rows.append(
                {
                    "bucket_family": bucket_col,
                    "bucket": str(bucket),
                    "rows": int(len(group)),
                    "observed_within10": float(np.mean(y)),
                    "pred_within10_mean": float(np.mean(p)),
                    "auc": _safe_auc(y, p),
                    "brier": float(brier_score_loss(y, np.clip(p, 1e-6, 1.0 - 1e-6))),
                    "mean_abs_error": float(group["abs_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def train_and_score(
    *,
    frame: pd.DataFrame,
    candidates: list[CandidateSpec],
    candidate_values: np.ndarray,
    row_feature_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
]:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

    seed = int(get_nested(config, "validation.seed") or 42)
    n_folds = int(get_nested(config, "validation.n_folds") or 5)
    run_folds = int(get_nested(config, "likelihood.run_folds") or 1)
    log_period = int(get_nested(config, "likelihood.log_period") or 100)
    max_train_rows = get_nested(config, "likelihood.max_train_rows_per_fold")
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    classifier_params = dict(get_nested(config, "likelihood.classifier_lgbm.params") or {})
    error_params = dict(get_nested(config, "likelihood.error_lgbm.params") or {})
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    cv = GroupKFold(n_splits=n_folds)
    folds = list(cv.split(frame, frame["oracle_label"], groups=frame["well"]))
    model_manifest: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    long_oof_parts: list[pd.DataFrame] = []
    eval_indices_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    predicted_error_parts: list[np.ndarray] = []

    for fold, (train_idx, valid_idx) in enumerate(folds[:run_folds]):
        print(f"[fold {fold}] train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        long_train = build_long_frame(
            frame,
            train_idx,
            candidates,
            row_feature_columns=row_feature_columns,
            candidate_values=candidate_values,
            sample_rows=max_train_rows,
            seed=seed + 101 * fold,
        )
        long_valid = build_long_frame(
            frame,
            valid_idx,
            candidates,
            row_feature_columns=row_feature_columns,
            candidate_values=candidate_values,
            sample_rows=None,
            seed=seed,
        )
        feature_columns = long_feature_columns(long_train)
        x_train, x_valid = fit_impute(long_train, long_valid, feature_columns)
        y_train = long_train["within_10ft"].to_numpy(np.int8)
        y_valid = long_valid["within_10ft"].to_numpy(np.int8)

        classifier = LGBMClassifier(
            objective="binary",
            random_state=seed + fold,
            **classifier_params,
        )
        classifier.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        prob = classifier.predict_proba(x_valid)[:, 1].astype(np.float32)
        model_path = model_dir / f"{OUTPUT_PREFIX}_within10_classifier_fold{fold}.txt"
        classifier.booster_.save_model(str(model_path))
        model_manifest.append(
            {
                "variant": "within10_classifier",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(classifier.best_iteration_ or classifier.n_estimators),
            }
        )
        for feature, importance in zip(
            feature_columns, classifier.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "within10_classifier",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

        error_model = LGBMRegressor(
            objective="regression_l1",
            random_state=seed + 1000 + fold,
            **error_params,
        )
        error_model.fit(
            x_train,
            long_train["abs_error"].to_numpy(np.float32),
            eval_set=[(x_valid, long_valid["abs_error"].to_numpy(np.float32))],
            eval_metric="l1",
            callbacks=[early_stopping(50), log_evaluation(log_period)],
        )
        pred_error = error_model.predict(x_valid).astype(np.float32)
        model_path = model_dir / f"{OUTPUT_PREFIX}_expected_error_fold{fold}.txt"
        error_model.booster_.save_model(str(model_path))
        model_manifest.append(
            {
                "variant": "expected_error_regressor",
                "fold": fold,
                "path": str(model_path.relative_to(output_dir)),
                "sha256": sha256_path(model_path),
                "best_iteration": int(error_model.best_iteration_ or error_model.n_estimators),
            }
        )
        for feature, importance in zip(
            feature_columns, error_model.feature_importances_, strict=False
        ):
            importance_rows.append(
                {
                    "variant": "expected_error_regressor",
                    "fold": fold,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

        output_columns = [
            "id",
            "well",
            "candidate_name",
            "candidate_index",
            "candidate_tvt",
            "abs_error",
            "within_5ft",
            "within_10ft",
        ]
        valid_out = long_valid[output_columns].copy()
        valid_out["fold"] = np.int16(fold)
        valid_out["pred_within10_prob"] = prob
        valid_out["pred_abs_error"] = pred_error
        valid_out["baseline_multiobs_score"] = long_valid["candidate_multiobs_score"].to_numpy(
            np.float32
        )
        valid_out["baseline_multiobs_mae"] = long_valid["candidate_multiobs_mae"].to_numpy(
            np.float32
        )
        valid_out["baseline_multiobs_ncc"] = long_valid["candidate_multiobs_ncc"].to_numpy(
            np.float32
        )
        if "md_since" in long_valid.columns:
            valid_out["md_since"] = long_valid["md_since"].to_numpy(np.float32)
        long_oof_parts.append(valid_out)
        eval_indices_parts.append(valid_idx)
        probability_parts.append(prob.reshape(len(candidates), len(valid_idx)).T)
        predicted_error_parts.append(pred_error.reshape(len(candidates), len(valid_idx)).T)

    long_oof = pd.concat(long_oof_parts, ignore_index=True)
    eval_indices = np.concatenate(eval_indices_parts)
    probability = np.vstack(probability_parts)
    predicted_error = np.vstack(predicted_error_parts)

    y = long_oof["within_10ft"].to_numpy(np.int8)
    learned = long_oof["pred_within10_prob"].to_numpy(np.float32)
    baseline_score = long_oof["baseline_multiobs_score"].to_numpy(np.float32)
    baseline_ncc = long_oof["baseline_multiobs_ncc"].to_numpy(np.float32)
    baseline_neg_mae = -long_oof["baseline_multiobs_mae"].to_numpy(np.float32)
    metric_rows = [
        {
            "variant": "learned_within10_probability",
            "mode": "candidate_likelihood",
            "rows": int(len(long_oof)),
            "candidate_rows": int(len(long_oof)),
            "auc": _safe_auc(y, learned),
            "logloss": _safe_logloss(y, learned),
            "brier": float(brier_score_loss(y, np.clip(learned, 1e-6, 1.0 - 1e-6))),
            "observed_within10": float(np.mean(y)),
            "pred_within10_mean": float(np.mean(learned)),
            "mean_abs_error": float(long_oof["abs_error"].mean()),
        },
        {
            "variant": "baseline_multiobs_score",
            "mode": "candidate_likelihood",
            "rows": int(len(long_oof)),
            "candidate_rows": int(len(long_oof)),
            "auc": _safe_auc(y, baseline_score),
            "logloss": None,
            "brier": None,
            "observed_within10": float(np.mean(y)),
            "pred_within10_mean": None,
            "mean_abs_error": float(long_oof["abs_error"].mean()),
        },
        {
            "variant": "baseline_multiobs_ncc",
            "mode": "candidate_likelihood",
            "rows": int(len(long_oof)),
            "candidate_rows": int(len(long_oof)),
            "auc": _safe_auc(y, baseline_ncc),
            "logloss": None,
            "brier": None,
            "observed_within10": float(np.mean(y)),
            "pred_within10_mean": None,
            "mean_abs_error": float(long_oof["abs_error"].mean()),
        },
        {
            "variant": "baseline_negative_multiobs_mae",
            "mode": "candidate_likelihood",
            "rows": int(len(long_oof)),
            "candidate_rows": int(len(long_oof)),
            "auc": _safe_auc(y, baseline_neg_mae),
            "logloss": None,
            "brier": None,
            "observed_within10": float(np.mean(y)),
            "pred_within10_mean": None,
            "mean_abs_error": float(long_oof["abs_error"].mean()),
        },
    ]
    metrics = pd.DataFrame(metric_rows)
    topk = topk_metrics(
        frame=frame,
        eval_indices=eval_indices,
        candidates=candidates,
        candidate_values=candidate_values,
        probability=probability,
        predicted_error=predicted_error,
    )
    calibration = pd.concat(
        [
            make_calibration_table(long_oof, "pred_within10_prob"),
            make_calibration_table(long_oof, "baseline_multiobs_score"),
        ],
        ignore_index=True,
    )
    buckets = bucket_likelihood_metrics(long_oof)
    importance = pd.DataFrame(importance_rows)
    manifest_path = output_dir / f"{OUTPUT_PREFIX}_model_manifest.json"
    with manifest_path.open("w") as fp:
        json.dump(to_jsonable({"models": model_manifest}), fp, indent=2, sort_keys=True)
    model_manifest_meta = [
        {
            **item,
            "manifest": manifest_path.name,
            "manifest_sha256": sha256_path(manifest_path),
        }
        for item in model_manifest
    ]
    return metrics, topk, calibration, buckets, long_oof, importance, model_manifest_meta


def summarize_decision(metrics: pd.DataFrame, topk: pd.DataFrame) -> dict[str, Any]:
    learned = metrics[metrics["variant"].eq("learned_within10_probability")].head(1)
    baseline = metrics[metrics["variant"].eq("baseline_multiobs_score")].head(1)
    top1 = topk[topk["variant"].eq("learned_prob_top1")].head(1)
    likpf = topk[topk["variant"].eq("likpf_mean_single")].head(1)
    decision = "likelihood_not_run"
    delta_auc = None
    delta_top1_rmse = None
    if not learned.empty and not baseline.empty:
        if pd.notna(learned.iloc[0]["auc"]) and pd.notna(baseline.iloc[0]["auc"]):
            delta_auc = float(learned.iloc[0]["auc"] - baseline.iloc[0]["auc"])
        if not top1.empty and not likpf.empty:
            delta_top1_rmse = float(top1.iloc[0]["rmse_tvt"] - likpf.iloc[0]["rmse_tvt"])
        if delta_auc is not None and delta_auc > 0.02:
            decision = "likelihood_supported_for_pf_weight_or_feature_followup"
        elif delta_auc is not None and delta_auc > 0.0:
            decision = "weak_likelihood_signal_needs_small_pf_weight_ablation"
        else:
            decision = "likelihood_not_supported"
    return {
        "recommendation": decision,
        "delta_auc_vs_multiobs_score": delta_auc,
        "diagnostic_top1_delta_rmse_vs_likpf": delta_top1_rmse,
        "learned_metric": to_jsonable(learned.iloc[0].to_dict()) if not learned.empty else None,
        "baseline_multiobs_metric": to_jsonable(baseline.iloc[0].to_dict())
        if not baseline.empty
        else None,
    }


def run_learned_pf_observation_likelihood_probe(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None,
    schema_path: str | Path | None,
    max_rows: int | None,
) -> dict[str, Any]:
    t0 = time.time()
    config = load_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_specs_from_config(config)
    required_columns = build_required_columns(config, candidates)
    frame, source_meta = load_train_feature_cache(
        cache_path=cache_path,
        schema_path=schema_path,
        required_columns=required_columns,
        max_rows=max_rows,
    )
    frame, engineered_columns, candidate_values, _ = add_row_level_features(
        frame,
        candidates,
        include_candidate_values=bool(get_nested(config, "likelihood.include_candidate_values")),
    )
    row_feature_columns = select_row_feature_columns(frame, config, engineered_columns)
    metrics, topk, calibration, buckets, long_oof, importance, model_manifest = train_and_score(
        frame=frame,
        candidates=candidates,
        candidate_values=candidate_values,
        row_feature_columns=row_feature_columns,
        config=config,
        output_dir=output_dir,
    )
    mean_importance = (
        importance.groupby(["variant", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            folds=("importance", "size"),
        )
        .sort_values(["variant", "mean_importance"], ascending=[True, False])
    )
    decision = summarize_decision(metrics, topk)

    metrics_path = output_dir / f"{OUTPUT_PREFIX}_metrics.csv"
    topk_path = output_dir / f"{OUTPUT_PREFIX}_topk_metrics.csv"
    calibration_path = output_dir / f"{OUTPUT_PREFIX}_calibration.csv"
    buckets_path = output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    long_oof_path = output_dir / f"{OUTPUT_PREFIX}_oof_likelihood_long.csv.gz"
    importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance.csv"
    mean_importance_path = output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv"
    schema_out_path = output_dir / f"{OUTPUT_PREFIX}_feature_schema.csv"
    metrics.to_csv(metrics_path, index=False)
    topk.to_csv(topk_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    buckets.to_csv(buckets_path, index=False)
    long_oof.to_csv(long_oof_path, index=False, compression="gzip")
    importance.to_csv(importance_path, index=False)
    mean_importance.to_csv(mean_importance_path, index=False)
    schema_rows = [
        {"feature_index": idx, "feature": feature}
        for idx, feature in enumerate(row_feature_columns)
    ]
    pd.DataFrame(schema_rows).to_csv(schema_out_path, index=False)

    summary = {
        "experiment": OUTPUT_PREFIX,
        "status": "implemented_debug_completed"
        if max_rows is not None
        else "completed_train_side_smoke_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - t0),
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "candidates": [spec.name for spec in candidates],
        "source": source_meta,
        "row_feature_count": int(len(row_feature_columns)),
        "row_feature_columns": row_feature_columns,
        "decision": to_jsonable(decision),
        "sha256": {
            "metrics": sha256_path(metrics_path),
            "topk_metrics": sha256_path(topk_path),
            "calibration": sha256_path(calibration_path),
            "bucket_metrics": sha256_path(buckets_path),
            "oof_likelihood_long": sha256_path(long_oof_path),
            "oof_likelihood_long_decompressed": sha256_path(long_oof_path, decompressed=True),
            "feature_schema": sha256_path(schema_out_path),
            "oof_probability": prediction_sha256(long_oof, value_col="pred_within10_prob"),
            "oof_predicted_error": prediction_sha256(long_oof, value_col="pred_abs_error"),
        },
        "model_manifest": model_manifest,
        "artifacts": {
            "metrics": metrics_path.name,
            "topk_metrics": topk_path.name,
            "calibration": calibration_path.name,
            "bucket_metrics": buckets_path.name,
            "oof_likelihood_long": long_oof_path.name,
            "feature_importance": importance_path.name,
            "feature_importance_mean": mean_importance_path.name,
            "feature_schema": schema_out_path.name,
            "model_manifest": f"{OUTPUT_PREFIX}_model_manifest.json",
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
    cache_path = args.cache_path or get_nested(config, "data.exp099_train_feature_cache_local")
    schema_path = args.schema_path or get_nested(config, "data.exp099_train_feature_schema_local")
    max_rows = args.max_rows
    configured_max = get_nested(config, "likelihood.max_rows")
    if max_rows is None and configured_max is not None:
        max_rows = int(configured_max)
    return run_learned_pf_observation_likelihood_probe(
        output_dir=output_dir,
        cache_path=cache_path,
        schema_path=schema_path,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
