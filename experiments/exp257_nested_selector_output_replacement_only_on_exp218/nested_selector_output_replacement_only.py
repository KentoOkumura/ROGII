from __future__ import annotations

import gc
import gzip
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # Local schema tests do not require LightGBM.
    lgb = None

OUTPUT_PREFIX = "exp257_nested_selector_output_replacement_only_on_exp218"
SELECTOR_OUTPUT_PREFIX = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
KEYS = ["id", "well"]

LEGACY_CANDIDATES = ["pf_ancc", "beam_mean", "likpf_mean", "sc_ens", "hyb"]

REPLACEMENT_COLUMNS = [
    "ll_learned_prob_top1_index",
    "ll_learned_error_top1_index",
    "ll_learned_prob_top1_value",
    "ll_learned_prob_top2_value",
    "ll_learned_prob_margin_top1_top2",
    "ll_learned_prob_entropy",
    "ll_learned_error_top1_value",
    "ll_learned_error_top2_value",
    "ll_learned_error_margin_top2_top1",
    "ll_learned_prob_likpf_rank",
    "ll_learned_error_likpf_rank",
    "ll_learned_prob_top3_contains_likpf",
    "ll_learned_error_top3_contains_likpf",
    "ll_candidate_tvt_std",
    "ll_candidate_tvt_range",
    "ll_learned_prob_pf_ancc",
    "ll_learned_pred_abs_error_pf_ancc",
    "ll_learned_prob_beam_mean",
    "ll_learned_pred_abs_error_beam_mean",
    "ll_learned_prob_likpf_mean",
    "ll_learned_pred_abs_error_likpf_mean",
    "ll_learned_prob_sc_ens",
    "ll_learned_pred_abs_error_sc_ens",
    "ll_learned_prob_hyb",
    "ll_learned_pred_abs_error_hyb",
    "ll_learned_prob_weighted_tvt_minus_last_known_tvt",
    "ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt",
    "ll_learned_error_weighted_tvt_minus_last_known_tvt",
    "ll_learned_error_weighted_tvt_minus_likpf_mean_tvt",
]

PRESERVED_SELECTOR_INPUT_COLUMNS = [
    "ll_multiobs_score_pf_ancc",
    "ll_multiobs_mae_pf_ancc",
    "ll_multiobs_ncc_pf_ancc",
    "ll_multiobs_score_beam_mean",
    "ll_multiobs_mae_beam_mean",
    "ll_multiobs_ncc_beam_mean",
    "ll_multiobs_score_likpf_mean",
    "ll_multiobs_mae_likpf_mean",
    "ll_multiobs_ncc_likpf_mean",
    "ll_multiobs_score_sc_ens",
    "ll_multiobs_mae_sc_ens",
    "ll_multiobs_ncc_sc_ens",
    "ll_multiobs_score_hyb",
    "ll_multiobs_mae_hyb",
    "ll_multiobs_ncc_hyb",
    "ll_candidate_tvt_pf_ancc_minus_last_known_tvt",
    "ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt",
    "ll_candidate_tvt_beam_mean_minus_last_known_tvt",
    "ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt",
    "ll_candidate_tvt_likpf_mean_minus_last_known_tvt",
    "ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt",
    "ll_candidate_tvt_sc_ens_minus_last_known_tvt",
    "ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt",
    "ll_candidate_tvt_hyb_minus_last_known_tvt",
    "ll_candidate_tvt_hyb_minus_likpf_mean_tvt",
]


def sha256_file(path: str | Path, *, decompressed: bool = False) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_feature_contract(
    feature_columns: list[str],
    *,
    expected_feature_count: int = 380,
) -> pd.DataFrame:
    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"exp218 feature count changed: {len(feature_columns)} != {expected_feature_count}"
        )
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("exp218 feature schema contains duplicate columns")
    forbidden = [column for column in feature_columns if column.startswith("nsel_")]
    if forbidden:
        raise ValueError(f"add-only nsel columns are forbidden: {forbidden[:10]}")

    ll_columns = [column for column in feature_columns if column.startswith("ll_")]
    expected_ll = set(REPLACEMENT_COLUMNS) | set(PRESERVED_SELECTOR_INPUT_COLUMNS)
    if len(REPLACEMENT_COLUMNS) != 29:
        raise AssertionError("replacement selector output contract must contain 29 columns")
    if len(PRESERVED_SELECTOR_INPUT_COLUMNS) != 25:
        raise AssertionError("preserved selector input contract must contain 25 columns")
    if set(ll_columns) != expected_ll or len(ll_columns) != 54:
        raise ValueError(
            {
                "message": "exp218 ll block no longer matches 29 replacement + 25 preserved",
                "ll_count": len(ll_columns),
                "missing": sorted(expected_ll - set(ll_columns)),
                "unexpected": sorted(set(ll_columns) - expected_ll),
            }
        )

    replacement = set(REPLACEMENT_COLUMNS)
    preserved = set(PRESERVED_SELECTOR_INPUT_COLUMNS)
    rows = []
    for index, column in enumerate(feature_columns):
        role = "unchanged_exp218"
        if column in replacement:
            role = "nested_selector_output_replaced"
        elif column in preserved:
            role = "selector_input_diagnostic_preserved"
        rows.append(
            {
                "feature_index": index,
                "feature": column,
                "role": role,
                "is_selector_output_replaced": column in replacement,
                "is_selector_input_diagnostic_preserved": column in preserved,
            }
        )
    return pd.DataFrame(rows)


def validate_selector_summary_sha(
    selector_summary: dict[str, Any],
    expected_by_outer: dict[int, str],
) -> dict[int, str]:
    actual = {
        int(item["outer_fold"]): str(item["sha256_decompressed"])
        for item in selector_summary.get("score_artifacts", [])
    }
    if actual != expected_by_outer:
        raise ValueError(
            {
                "message": "selector nested-score SHA contract mismatch",
                "expected": expected_by_outer,
                "actual": actual,
            }
        )
    if selector_summary.get("selector_v3_nested_score_sha_contract") != "pass":
        raise ValueError("selector summary does not certify the v3 nested-score SHA contract")
    return actual


def load_nested_fold_contracts(
    artifact_dir: str | Path,
    row_count: int,
    fold_count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    artifact_dir = Path(artifact_dir)
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    expected_rows = np.arange(row_count, dtype=np.int64)
    for outer_fold in range(fold_count):
        path = artifact_dir / f"{SELECTOR_OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
        artifact = pd.read_csv(
            path,
            usecols=["row_index", "role"],
            dtype={"row_index": np.int32, "role": "category"},
        )
        if len(artifact) != row_count or artifact["row_index"].duplicated().any():
            raise ValueError(f"outer {outer_fold}: score artifact row contract failed")
        train_rows = np.sort(
            artifact.loc[artifact.role.eq("train"), "row_index"].to_numpy(np.int64)
        )
        valid_rows = np.sort(
            artifact.loc[artifact.role.eq("valid"), "row_index"].to_numpy(np.int64)
        )
        if np.intersect1d(train_rows, valid_rows).size:
            raise ValueError(f"outer {outer_fold}: train/valid rows overlap")
        if not np.array_equal(
            np.sort(np.concatenate([train_rows, valid_rows])), expected_rows
        ):
            raise ValueError(f"outer {outer_fold}: roles do not cover all rows")
        outputs.append((train_rows, valid_rows))
        del artifact
        gc.collect()
    return outputs


def load_nested_score_artifact(
    artifact_dir: str | Path,
    frame: pd.DataFrame,
    outer_fold: int,
    expected_outer: tuple[np.ndarray, np.ndarray],
    candidate_columns: list[str],
) -> dict[str, np.ndarray]:
    artifact_dir = Path(artifact_dir)
    score_columns = [f"pred_error__{name}" for name in candidate_columns]
    path = artifact_dir / f"{SELECTOR_OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
    dtype: dict[str, Any] = {
        "row_index": np.int32,
        "id": str,
        "well": str,
        "role": "category",
        **{column: np.float32 for column in score_columns},
    }
    artifact = pd.read_csv(
        path,
        usecols=["row_index", "role", *KEYS, *score_columns],
        dtype=dtype,
    )
    if len(artifact) != len(frame) or artifact["row_index"].duplicated().any():
        raise ValueError(f"outer {outer_fold}: score artifact row contract failed")
    aligned = artifact.sort_values("row_index")
    if not aligned[KEYS].reset_index(drop=True).equals(
        frame[KEYS].astype(str).reset_index(drop=True)
    ):
        raise ValueError(f"outer {outer_fold}: score artifact id/well mismatch")
    train = artifact.loc[artifact.role.eq("train")].sort_values("row_index")
    valid = artifact.loc[artifact.role.eq("valid")].sort_values("row_index")
    train_rows = train["row_index"].to_numpy(np.int64)
    valid_rows = valid["row_index"].to_numpy(np.int64)
    expected_train, expected_valid = expected_outer
    if not (
        np.array_equal(train_rows, np.sort(expected_train))
        and np.array_equal(valid_rows, np.sort(expected_valid))
    ):
        raise ValueError(f"outer {outer_fold}: role contract changed after initial read")
    train_scores = train[score_columns].to_numpy(np.float32, copy=True)
    valid_scores = valid[score_columns].to_numpy(np.float32, copy=True)
    if not np.isfinite(train_scores).all() or not np.isfinite(valid_scores).all():
        raise ValueError(f"outer {outer_fold}: selector scores contain non-finite values")
    output = {"train_scores": train_scores, "valid_scores": valid_scores}
    del aligned, train, valid, artifact
    gc.collect()
    return output


def _rank_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty_like(order)
    row_index = np.arange(order.shape[0])[:, None]
    positions[row_index, order] = np.arange(order.shape[1])[None, :]
    return positions


def build_selector_output_replacements(
    frame: pd.DataFrame,
    rows: np.ndarray,
    scores: np.ndarray,
    candidate_columns: list[str],
    *,
    error_floor: float = 1e-3,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if scores.shape != (len(rows), len(candidate_columns)):
        raise ValueError(
            f"score shape {scores.shape} != ({len(rows)}, {len(candidate_columns)})"
        )
    if not np.isfinite(scores).all():
        raise ValueError("selector scores contain non-finite values")
    missing_candidates = sorted(set(LEGACY_CANDIDATES) - set(candidate_columns))
    if missing_candidates:
        raise ValueError(f"legacy candidate slots cannot be populated: {missing_candidates}")
    if "likpf_mean" not in candidate_columns:
        raise ValueError("likpf_mean candidate is required for existing rank slots")

    values = frame.iloc[rows][candidate_columns].to_numpy(np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("candidate values contain non-finite values")
    clipped_scores = np.maximum(scores.astype(np.float64), float(error_floor))
    inverse_error = 1.0 / clipped_scores
    probability = inverse_error / inverse_error.sum(axis=1, keepdims=True)
    probability = probability.astype(np.float32)

    prob_order = np.argsort(-probability, axis=1)
    error_order = np.argsort(scores, axis=1)
    prob_sorted = np.take_along_axis(probability, prob_order, axis=1)
    error_sorted = np.take_along_axis(scores, error_order, axis=1)
    prob_rank = _rank_positions(prob_order)
    error_rank = _rank_positions(error_order)
    likpf_index = candidate_columns.index("likpf_mean")

    anchor = frame.iloc[rows]["last_known_tvt"].to_numpy(np.float32)
    likpf_tvt = frame.iloc[rows]["likpf_mean"].to_numpy(np.float32)
    weighted_tvt = np.sum(values * probability, axis=1).astype(np.float32)

    output = pd.DataFrame(index=np.arange(len(rows)))
    output["ll_learned_prob_top1_index"] = prob_order[:, 0]
    output["ll_learned_error_top1_index"] = error_order[:, 0]
    output["ll_learned_prob_top1_value"] = prob_sorted[:, 0]
    output["ll_learned_prob_top2_value"] = prob_sorted[:, 1]
    output["ll_learned_prob_margin_top1_top2"] = prob_sorted[:, 0] - prob_sorted[:, 1]
    output["ll_learned_prob_entropy"] = -np.sum(
        np.clip(probability, 1e-6, 1.0) * np.log(np.clip(probability, 1e-6, 1.0)),
        axis=1,
    )
    output["ll_learned_error_top1_value"] = error_sorted[:, 0]
    output["ll_learned_error_top2_value"] = error_sorted[:, 1]
    output["ll_learned_error_margin_top2_top1"] = error_sorted[:, 1] - error_sorted[:, 0]
    output["ll_learned_prob_likpf_rank"] = prob_rank[:, likpf_index]
    output["ll_learned_error_likpf_rank"] = error_rank[:, likpf_index]
    output["ll_learned_prob_top3_contains_likpf"] = prob_rank[:, likpf_index] < 3
    output["ll_learned_error_top3_contains_likpf"] = error_rank[:, likpf_index] < 3
    output["ll_candidate_tvt_std"] = values.std(axis=1)
    output["ll_candidate_tvt_range"] = np.ptp(values, axis=1)

    for candidate in LEGACY_CANDIDATES:
        index = candidate_columns.index(candidate)
        output[f"ll_learned_prob_{candidate}"] = probability[:, index]
        output[f"ll_learned_pred_abs_error_{candidate}"] = scores[:, index]

    output["ll_learned_prob_weighted_tvt_minus_last_known_tvt"] = weighted_tvt - anchor
    output["ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt"] = weighted_tvt - likpf_tvt
    output["ll_learned_error_weighted_tvt_minus_last_known_tvt"] = weighted_tvt - anchor
    output["ll_learned_error_weighted_tvt_minus_likpf_mean_tvt"] = weighted_tvt - likpf_tvt
    output = output[REPLACEMENT_COLUMNS].astype(np.float32)
    if not np.isfinite(output.to_numpy(np.float32)).all():
        raise ValueError("replacement selector output contains non-finite values")

    meta: dict[str, float | int] = {
        "rows": int(len(rows)),
        "candidate_count": int(len(candidate_columns)),
        "replacement_feature_count": int(output.shape[1]),
        "scores_below_floor": int((scores < float(error_floor)).sum()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "probability_row_sum_abs_max": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
    }
    return output, meta


def replacement_contract_frame() -> pd.DataFrame:
    semantic: dict[str, str] = {
        "ll_learned_prob_top1_index": "all-candidate inverse-error probability top1 code",
        "ll_learned_error_top1_index": "all-candidate predicted-error top1 code",
        "ll_candidate_tvt_std": "all-candidate TVT standard deviation",
        "ll_candidate_tvt_range": "all-candidate TVT range",
    }
    rows = []
    for column in REPLACEMENT_COLUMNS:
        rows.append(
            {
                "feature": column,
                "action": "overwrite_existing_column",
                "source": semantic.get(column, "exp238 nested selector score/candidate adapter"),
            }
        )
    for column in PRESERVED_SELECTOR_INPUT_COLUMNS:
        rows.append(
            {
                "feature": column,
                "action": "preserve_existing_value",
                "source": "exp218 target-free selector input diagnostic",
            }
        )
    return pd.DataFrame(rows)


def fit_final_nested_replacement_only(
    base_frame: pd.DataFrame,
    selector_frame: pd.DataFrame,
    feature_columns: list[str],
    outer: list[tuple[np.ndarray, np.ndarray]],
    nested_score_dir: str | Path,
    candidate_columns: list[str],
    final_param_family: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    error_floor: float = 1e-3,
    early_stopping_rounds: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    if lgb is None:
        raise ModuleNotFoundError("LightGBM is required for final Kaggle training")
    output_dir = Path(output_dir)
    validate_feature_contract(feature_columns)
    if not base_frame[KEYS].astype(str).reset_index(drop=True).equals(
        selector_frame[KEYS].astype(str).reset_index(drop=True)
    ):
        raise ValueError("exp218 and selector frames are not id/well row aligned")

    y = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = anchor + y
    outer_assignment = np.full(len(base_frame), -1, dtype=np.int16)
    oof = [np.full(len(base_frame), np.nan, np.float32) for _ in final_param_family]
    importance_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    replacement_audit: list[dict[str, Any]] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    replacement_set = set(REPLACEMENT_COLUMNS)
    for outer_fold, (train_rows, valid_rows) in enumerate(outer):
        if np.any(outer_assignment[valid_rows] >= 0):
            raise ValueError(f"outer {outer_fold}: validation rows were already assigned")
        outer_assignment[valid_rows] = outer_fold
        score_item = load_nested_score_artifact(
            nested_score_dir,
            selector_frame,
            outer_fold,
            (train_rows, valid_rows),
            candidate_columns,
        )
        train_replacement, train_meta = build_selector_output_replacements(
            selector_frame,
            train_rows,
            score_item["train_scores"],
            candidate_columns,
            error_floor=error_floor,
        )
        valid_replacement, valid_meta = build_selector_output_replacements(
            selector_frame,
            valid_rows,
            score_item["valid_scores"],
            candidate_columns,
            error_floor=error_floor,
        )
        replacement_audit.extend(
            [
                {"outer_fold": outer_fold, "role": "train", **train_meta},
                {"outer_fold": outer_fold, "role": "valid", **valid_meta},
            ]
        )

        x_train_values = np.empty(
            (len(train_rows), len(feature_columns)), dtype=np.float32
        )
        x_valid_values = np.empty(
            (len(valid_rows), len(feature_columns)), dtype=np.float32
        )
        chunk_size = 32
        for start in range(0, len(feature_columns), chunk_size):
            stop = min(start + chunk_size, len(feature_columns))
            columns = feature_columns[start:stop]
            train_chunk = base_frame.iloc[train_rows][columns].to_numpy(
                np.float32, copy=True
            )
            valid_chunk = base_frame.iloc[valid_rows][columns].to_numpy(
                np.float32, copy=True
            )
            for offset, column in enumerate(columns):
                if column in replacement_set:
                    train_chunk[:, offset] = train_replacement[column].to_numpy(np.float32)
                    valid_chunk[:, offset] = valid_replacement[column].to_numpy(np.float32)
            x_train_values[:, start:stop] = train_chunk
            x_valid_values[:, start:stop] = valid_chunk
            del train_chunk, valid_chunk

        x_train = pd.DataFrame(x_train_values, columns=feature_columns, copy=False)
        x_valid = pd.DataFrame(x_valid_values, columns=feature_columns, copy=False)
        del train_replacement, valid_replacement, score_item
        gc.collect()

        for model_index, params in enumerate(final_param_family):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x_train,
                y[train_rows],
                eval_set=[(x_valid, y[valid_rows])],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            prediction = model.predict(
                x_valid, num_iteration=model.best_iteration_
            ).astype(np.float32)
            oof[model_index][valid_rows] = prediction
            model_path = model_dir / f"lgb{model_index}__outer{outer_fold}.txt"
            model.booster_.save_model(
                str(model_path), num_iteration=model.best_iteration_
            )
            manifest.append(
                {
                    "model": f"lgb{model_index}",
                    "outer_fold": outer_fold,
                    "file": str(model_path),
                    "sha256": sha256_file(model_path),
                    "best_iteration": int(model.best_iteration_),
                    "feature_count": len(feature_columns),
                    "selector_replaced_features": len(REPLACEMENT_COLUMNS),
                    "selector_added_features": 0,
                }
            )
            importance_rows.extend(
                {
                    "model": f"lgb{model_index}",
                    "outer_fold": outer_fold,
                    "feature": feature,
                    "importance": float(value),
                }
                for feature, value in zip(feature_columns, model.feature_importances_)
            )
            del model, prediction
            gc.collect()
        del x_train, x_valid, x_train_values, x_valid_values
        gc.collect()

    for model_index, values in enumerate(oof):
        if not np.isfinite(values).all():
            raise AssertionError(f"incomplete final OOF for lgb{model_index}")
    ensemble = np.mean(np.vstack(oof), axis=0).astype(np.float32)
    predictions = base_frame[KEYS + ["last_known_tvt", "target"]].copy()
    if np.any(outer_assignment < 0):
        raise AssertionError("outer fold assignment does not cover all OOF rows")
    predictions["outer_fold"] = outer_assignment
    for model_index, values in enumerate(oof):
        predictions[f"lgb{model_index}_pred_tvt"] = anchor + values
    predictions["lgb_mean_pred_tvt"] = anchor + ensemble

    metric_rows = []
    for name in [*(f"lgb{i}" for i in range(len(oof))), "lgb_mean"]:
        predicted_tvt = predictions[f"{name}_pred_tvt"].to_numpy(np.float32)
        metric_rows.append(
            {
                "model": name,
                "rmse_tvt": float(np.sqrt(np.mean((predicted_tvt - truth) ** 2))),
                "rows": len(predictions),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    importance = pd.DataFrame(importance_rows)
    importance_mean = (
        importance.groupby(["model", "feature"], as_index=False)
        .importance.mean()
        .sort_values(["model", "importance"], ascending=[True, False])
    )
    return metrics, predictions, importance_mean, manifest, replacement_audit
