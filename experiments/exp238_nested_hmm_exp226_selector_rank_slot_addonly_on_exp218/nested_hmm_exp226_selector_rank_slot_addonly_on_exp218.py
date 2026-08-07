from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

OUTPUT_PREFIX = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
KEYS = ["id", "well"]


def _sha(path: Path, *, decompressed: bool = False) -> str:
    h = hashlib.sha256()
    opener = __import__("gzip").open if decompressed and path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def deterministic_outer_inner_splits(
    frame: pd.DataFrame, outer_folds: int, inner_folds: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[list[tuple[np.ndarray, np.ndarray]]]]:
    groups = frame["well"].astype(str).to_numpy()
    outer = list(GroupKFold(outer_folds).split(frame, groups=groups))
    nested: list[list[tuple[np.ndarray, np.ndarray]]] = []
    all_rows = np.arange(len(frame))
    for outer_fold, (outer_train, outer_valid) in enumerate(outer):
        train_wells = set(groups[outer_train])
        valid_wells = set(groups[outer_valid])
        if train_wells & valid_wells:
            raise AssertionError(f"outer fold {outer_fold}: well overlap")
        local = list(
            GroupKFold(inner_folds).split(
                outer_train, groups=groups[outer_train]
            )
        )
        fold_rows: list[tuple[np.ndarray, np.ndarray]] = []
        covered: list[np.ndarray] = []
        for inner_train_local, inner_valid_local in local:
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            if set(groups[inner_train]) & set(groups[inner_valid]):
                raise AssertionError(f"outer fold {outer_fold}: inner well overlap")
            if set(groups[inner_train]) & valid_wells:
                raise AssertionError(f"outer fold {outer_fold}: outer-valid leaked into selector train")
            fold_rows.append((inner_train, inner_valid))
            covered.append(inner_valid)
        if not np.array_equal(np.sort(np.concatenate(covered)), np.sort(outer_train)):
            raise AssertionError(f"outer fold {outer_fold}: inner OOF coverage mismatch")
        if np.intersect1d(outer_valid, all_rows[np.isin(all_rows, outer_train)]).size:
            raise AssertionError(f"outer fold {outer_fold}: row overlap")
        nested.append(fold_rows)
    return outer, nested


def candidate_long(
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_columns: list[str],
    context_columns: list[str],
    *,
    with_target: bool,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    blocks: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    true_tvt = (
        frame["last_known_tvt"].to_numpy(np.float32)
        + frame["target"].to_numpy(np.float32)
    )
    anchor = frame["last_known_tvt"].to_numpy(np.float32)
    for code, column in enumerate(candidate_columns):
        values = frame[column].to_numpy(np.float32)[rows]
        block = frame.iloc[rows][context_columns].reset_index(drop=True).copy()
        block["candidate_code"] = np.float32(code)
        block["candidate_minus_anchor"] = values - anchor[rows]
        block["candidate_abs_minus_anchor"] = np.abs(values - anchor[rows])
        blocks.append(block)
        if with_target:
            labels.append(np.abs(values - true_tvt[rows]).astype(np.float32))
    long = pd.concat(blocks, ignore_index=True)
    target = np.concatenate(labels) if labels else None
    return long, target


def predict_candidate_errors(
    model: lgb.LGBMRegressor,
    frame: pd.DataFrame,
    rows: np.ndarray,
    candidate_columns: list[str],
    context_columns: list[str],
    chunk_rows: int = 50_000,
) -> np.ndarray:
    parts = []
    for start in range(0, len(rows), int(chunk_rows)):
        chunk = rows[start : start + int(chunk_rows)]
        long, _ = candidate_long(
            frame, chunk, candidate_columns, context_columns, with_target=False
        )
        pred = model.predict(long).reshape(len(candidate_columns), len(chunk)).T
        parts.append(np.asarray(pred, dtype=np.float32))
    return np.concatenate(parts, axis=0)


def _bounded_base_rows(
    rows: np.ndarray,
    candidate_count: int,
    max_long_rows: int | None,
    seed: int,
) -> np.ndarray:
    if max_long_rows is None:
        return rows
    max_base_rows = max(1, int(max_long_rows) // int(candidate_count))
    if len(rows) <= max_base_rows:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=max_base_rows, replace=False))


def fit_nested_selector_scores(
    frame: pd.DataFrame,
    outer: list[tuple[np.ndarray, np.ndarray]],
    inner: list[list[tuple[np.ndarray, np.ndarray]]],
    candidate_columns: list[str],
    context_columns: list[str],
    selector_params: dict[str, Any],
    seed: int,
    output_dir: Path,
    max_train_long_rows: int | None = 120_000,
    max_valid_long_rows: int | None = 120_000,
    predict_chunk_rows: int = 50_000,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    outputs: list[dict[str, np.ndarray]] = []
    manifest: list[dict[str, Any]] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold, ((outer_train, outer_valid), inner_splits) in enumerate(zip(outer, inner)):
        train_scores = np.full((len(frame), len(candidate_columns)), np.nan, np.float32)
        valid_models: list[lgb.LGBMRegressor] = []
        for inner_fold, (train_rows, valid_rows) in enumerate(inner_splits):
            fit_train_rows = _bounded_base_rows(
                train_rows, len(candidate_columns), max_train_long_rows,
                seed + 10_000 * outer_fold + 100 * inner_fold,
            )
            fit_valid_rows = _bounded_base_rows(
                valid_rows, len(candidate_columns), max_valid_long_rows,
                seed + 20_000 * outer_fold + 100 * inner_fold,
            )
            x_train, y_train = candidate_long(
                frame, fit_train_rows, candidate_columns, context_columns, with_target=True
            )
            x_valid, y_valid = candidate_long(
                frame, fit_valid_rows, candidate_columns, context_columns, with_target=True
            )
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                random_state=seed + 100 * outer_fold + inner_fold,
                **selector_params,
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            )
            model_path = model_dir / f"selector_outer{outer_fold}_inner{inner_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
            train_scores[valid_rows] = predict_candidate_errors(
                model, frame, valid_rows, candidate_columns, context_columns,
                chunk_rows=predict_chunk_rows,
            )
            valid_models.append(model)
            manifest.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "train_rows": len(train_rows),
                    "valid_rows": len(valid_rows),
                    "fit_train_base_rows": len(fit_train_rows),
                    "fit_valid_base_rows": len(fit_valid_rows),
                    "fit_train_long_rows": len(x_train),
                    "fit_valid_long_rows": len(x_valid),
                    "train_wells": frame.iloc[train_rows]["well"].nunique(),
                    "valid_wells": frame.iloc[valid_rows]["well"].nunique(),
                    "best_iteration": int(model.best_iteration_),
                    "file": str(model_path),
                    "sha256": _sha(model_path),
                    "feature_count": int(model.booster_.num_feature()),
                    "feature_names_json": json.dumps(model.booster_.feature_name()),
                }
            )
            del x_train, x_valid, y_train, y_valid
            gc.collect()
        if not np.isfinite(train_scores[outer_train]).all():
            raise AssertionError(f"outer fold {outer_fold}: incomplete inner OOF scores")
        valid_scores = np.mean(
            [
                predict_candidate_errors(
                    model, frame, outer_valid, candidate_columns, context_columns
                    , chunk_rows=predict_chunk_rows
                )
                for model in valid_models
            ],
            axis=0,
        ).astype(np.float32)
        outputs.append(
            {
                "outer_train": outer_train,
                "outer_valid": outer_valid,
                "train_scores": train_scores[outer_train],
                "valid_scores": valid_scores,
            }
        )
        del valid_models, model
        gc.collect()
    return outputs, manifest


def save_nested_score_artifacts(
    output_dir: Path,
    frame: pd.DataFrame,
    nested_scores: list[dict[str, np.ndarray]],
    candidate_columns: list[str],
) -> list[dict[str, Any]]:
    records = []
    for outer_fold, item in enumerate(nested_scores):
        rows = np.concatenate([item["outer_train"], item["outer_valid"]])
        scores = np.vstack([item["train_scores"], item["valid_scores"]])
        roles = np.concatenate([
            np.repeat("train", len(item["outer_train"])),
            np.repeat("valid", len(item["outer_valid"])),
        ])
        artifact = frame.iloc[rows][KEYS].reset_index(drop=True).copy()
        artifact.insert(0, "row_index", rows)
        artifact.insert(1, "role", roles)
        for index, name in enumerate(candidate_columns):
            artifact[f"pred_error__{name}"] = scores[:, index]
        path = output_dir / f"{OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
        artifact.to_csv(path, index=False, compression="gzip")
        records.append({
            "outer_fold": outer_fold,
            "file": path.name,
            "rows": len(artifact),
            "train_rows": int((roles == "train").sum()),
            "valid_rows": int((roles == "valid").sum()),
            "sha256_decompressed": _sha(path, decompressed=True),
        })
    return records


def load_nested_score_artifacts(
    artifact_dir: Path,
    frame: pd.DataFrame,
    outer: list[tuple[np.ndarray, np.ndarray]] | None,
    candidate_columns: list[str],
) -> list[dict[str, np.ndarray]]:
    outputs = []
    score_columns = [f"pred_error__{name}" for name in candidate_columns]
    fold_count = len(outer) if outer is not None else 5
    for outer_fold in range(fold_count):
        path = artifact_dir / f"{OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
        artifact = pd.read_csv(path, dtype={"id": str, "well": str})
        if len(artifact) != len(frame) or artifact["row_index"].duplicated().any():
            raise ValueError(f"outer {outer_fold}: score artifact row contract failed")
        aligned = artifact.sort_values("row_index")
        if not aligned[KEYS].reset_index(drop=True).equals(frame[KEYS].astype(str).reset_index(drop=True)):
            raise ValueError(f"outer {outer_fold}: score artifact id/well mismatch")
        train = artifact.loc[artifact.role.eq("train")].sort_values("row_index")
        valid = artifact.loc[artifact.role.eq("valid")].sort_values("row_index")
        train_rows = train["row_index"].to_numpy(np.int64)
        valid_rows = valid["row_index"].to_numpy(np.int64)
        if np.intersect1d(train_rows, valid_rows).size:
            raise ValueError(f"outer {outer_fold}: train/valid row overlap in score artifact")
        if not np.array_equal(
            np.sort(np.concatenate([train_rows, valid_rows])), np.arange(len(frame))
        ):
            raise ValueError(f"outer {outer_fold}: score artifact roles do not cover all rows")
        if outer is not None:
            expected_train, expected_valid = outer[outer_fold]
            runtime_fold_match = bool(
                np.array_equal(train_rows, np.sort(expected_train))
                and np.array_equal(valid_rows, np.sort(expected_valid))
            )
        else:
            runtime_fold_match = None
        outputs.append({
            "outer_train": train_rows,
            "outer_valid": valid_rows,
            "train_scores": train[score_columns].to_numpy(np.float32),
            "valid_scores": valid[score_columns].to_numpy(np.float32),
            "runtime_reconstructed_fold_match": runtime_fold_match,
        })
    return outputs


def load_nested_fold_contracts(
    artifact_dir: Path,
    row_count: int,
    fold_count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load only row roles so the five large score matrices are not resident together."""
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    expected_rows = np.arange(row_count, dtype=np.int64)
    for outer_fold in range(fold_count):
        path = artifact_dir / f"{OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
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
        if not np.array_equal(
            np.sort(np.concatenate([train_rows, valid_rows])), expected_rows
        ):
            raise ValueError(f"outer {outer_fold}: score artifact roles do not cover all rows")
        outputs.append((train_rows, valid_rows))
        del artifact
        gc.collect()
    return outputs


def load_nested_score_artifact(
    artifact_dir: Path,
    frame: pd.DataFrame,
    outer_fold: int,
    expected_outer: tuple[np.ndarray, np.ndarray],
    candidate_columns: list[str],
) -> dict[str, np.ndarray]:
    """Load and validate one outer fold's score matrix just before its models run."""
    score_columns = [f"pred_error__{name}" for name in candidate_columns]
    path = artifact_dir / f"{OUTPUT_PREFIX}_nested_scores_outer{outer_fold}.csv.gz"
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
        raise ValueError(f"outer {outer_fold}: score artifact changed after fold-contract read")
    output = {
        "train_scores": train[score_columns].to_numpy(np.float32, copy=True),
        "valid_scores": valid[score_columns].to_numpy(np.float32, copy=True),
    }
    del aligned, train, valid, artifact
    gc.collect()
    return output


def rank_slot_features(
    frame: pd.DataFrame,
    rows: np.ndarray,
    scores: np.ndarray,
    candidate_columns: list[str],
    prefix: str = "nsel_",
) -> pd.DataFrame:
    values = frame.iloc[rows][candidate_columns].to_numpy(np.float32)
    order = np.argsort(scores, axis=1)
    top = np.take_along_axis(values, order[:, :2], axis=1)
    top_scores = np.take_along_axis(scores, order[:, :2], axis=1)
    anchor = frame.iloc[rows]["last_known_tvt"].to_numpy(np.float32)
    out = pd.DataFrame(index=rows)
    out[prefix + "top1_code"] = order[:, 0].astype(np.float32)
    out[prefix + "top2_code"] = order[:, 1].astype(np.float32)
    out[prefix + "top1_minus_anchor"] = top[:, 0] - anchor
    out[prefix + "top2_minus_anchor"] = top[:, 1] - anchor
    out[prefix + "top2_minus_top1"] = top[:, 1] - top[:, 0]
    out[prefix + "error_top1"] = top_scores[:, 0]
    out[prefix + "error_top2"] = top_scores[:, 1]
    out[prefix + "error_margin"] = top_scores[:, 1] - top_scores[:, 0]
    out[prefix + "error_ratio"] = top_scores[:, 0] / np.maximum(top_scores[:, 1], 1e-3)
    out[prefix + "score_mean"] = scores.mean(axis=1)
    out[prefix + "score_std"] = scores.std(axis=1)
    out[prefix + "candidate_std"] = values.std(axis=1)
    out[prefix + "candidate_range"] = np.ptp(values, axis=1)
    for code, name in enumerate(candidate_columns):
        out[prefix + "top1_is_" + name] = (order[:, 0] == code).astype(np.float32)
        out[prefix + "pred_error_" + name] = scores[:, code]
    return out.astype(np.float32)


def selector_safety_readout(
    frame: pd.DataFrame,
    rows: np.ndarray,
    scores: np.ndarray,
    candidate_columns: list[str],
    fallback_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = frame.iloc[rows][candidate_columns].to_numpy(np.float32)
    selected = values[np.arange(len(rows)), np.argmin(scores, axis=1)]
    fallback = frame.iloc[rows][fallback_column].to_numpy(np.float32)
    truth = frame.iloc[rows]["last_known_tvt"].to_numpy(np.float32) + frame.iloc[rows]["target"].to_numpy(np.float32)
    md = frame.iloc[rows]["md_since"].to_numpy(np.float32)
    well = frame.iloc[rows]["well"].astype(str).to_numpy()
    records = []
    masks = {
        "global": np.ones(len(rows), bool),
        "000_050": md <= 50,
        "1000_plus": md >= 1000,
    }
    for bucket, mask in masks.items():
        if not mask.any():
            continue
        base_rmse = float(np.sqrt(np.mean((fallback[mask] - truth[mask]) ** 2)))
        sel_rmse = float(np.sqrt(np.mean((selected[mask] - truth[mask]) ** 2)))
        records.append({"bucket": bucket, "rows": int(mask.sum()), "fallback_rmse": base_rmse, "selector_rmse": sel_rmse, "delta_rmse": sel_rmse - base_rmse})
    detail = pd.DataFrame({"well": well, "truth": truth, "fallback": fallback, "selected": selected})
    by_well = detail.groupby("well").apply(
        lambda g: pd.Series({
            "fallback_rmse": np.sqrt(np.mean((g.fallback-g.truth)**2)),
            "selector_rmse": np.sqrt(np.mean((g.selected-g.truth)**2)),
        }), include_groups=False
    ).reset_index()
    by_well["delta_rmse"] = by_well["selector_rmse"] - by_well["fallback_rmse"]
    return pd.DataFrame(records), by_well


def save_fold_contract(output_dir: Path, frame: pd.DataFrame, outer, inner) -> Path:
    rows = []
    for outer_fold, ((outer_train, outer_valid), inner_splits) in enumerate(zip(outer, inner)):
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
            rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "outer_train_rows": len(outer_train), "outer_valid_rows": len(outer_valid), "inner_train_rows": len(inner_train), "inner_valid_rows": len(inner_valid), "outer_train_wells": frame.iloc[outer_train].well.nunique(), "outer_valid_wells": frame.iloc[outer_valid].well.nunique()})
    path = output_dir / f"{OUTPUT_PREFIX}_fold_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def fit_final_nested_addonly(
    base_frame: pd.DataFrame,
    selector_frame: pd.DataFrame,
    base_feature_columns: list[str],
    outer: list[tuple[np.ndarray, np.ndarray]],
    nested_score_dir: Path,
    candidate_columns: list[str],
    final_param_family: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if not base_frame[KEYS].reset_index(drop=True).equals(selector_frame[KEYS].reset_index(drop=True)):
        raise ValueError("exp218 and selector frames are not id/well row aligned")
    y = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = anchor + y
    oof = [np.full(len(base_frame), np.nan, np.float32) for _ in final_param_family]
    importance_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    model_dir = output_dir / f"{OUTPUT_PREFIX}_lgb_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    all_feature_columns: list[str] | None = None
    for outer_fold, (train_rows, valid_rows) in enumerate(outer):
        score_item = load_nested_score_artifact(
            nested_score_dir,
            selector_frame,
            outer_fold,
            (train_rows, valid_rows),
            candidate_columns,
        )
        train_extra = rank_slot_features(selector_frame, train_rows, score_item["train_scores"], candidate_columns).reset_index(drop=True)
        valid_extra = rank_slot_features(selector_frame, valid_rows, score_item["valid_scores"], candidate_columns).reset_index(drop=True)
        extra_columns = list(train_extra.columns)
        all_feature_columns = [*base_feature_columns, *extra_columns]

        # Fill the final matrices in column chunks.  Building x_base and then
        # concatenating it with selector features temporarily held two full
        # 3M x 380 matrices and caused the Kaggle kernel to be OOM-killed.
        x_train_values = np.empty(
            (len(train_rows), len(all_feature_columns)), dtype=np.float32
        )
        x_valid_values = np.empty(
            (len(valid_rows), len(all_feature_columns)), dtype=np.float32
        )
        chunk_columns = 32
        for start in range(0, len(base_feature_columns), chunk_columns):
            stop = min(start + chunk_columns, len(base_feature_columns))
            columns = base_feature_columns[start:stop]
            x_train_values[:, start:stop] = base_frame.iloc[train_rows][columns].to_numpy(
                dtype=np.float32, copy=True
            )
            x_valid_values[:, start:stop] = base_frame.iloc[valid_rows][columns].to_numpy(
                dtype=np.float32, copy=True
            )
        x_train_values[:, len(base_feature_columns):] = train_extra.to_numpy(
            dtype=np.float32, copy=False
        )
        x_valid_values[:, len(base_feature_columns):] = valid_extra.to_numpy(
            dtype=np.float32, copy=False
        )
        x_train = pd.DataFrame(x_train_values, columns=all_feature_columns, copy=False)
        x_valid = pd.DataFrame(x_valid_values, columns=all_feature_columns, copy=False)
        del train_extra, valid_extra, score_item
        gc.collect()
        for model_index, params in enumerate(final_param_family):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x_train,
                y[train_rows],
                eval_set=[(x_valid, y[valid_rows])],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(100)],
            )
            pred = model.predict(x_valid, num_iteration=model.best_iteration_).astype(np.float32)
            oof[model_index][valid_rows] = pred
            model_path = model_dir / f"lgb{model_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
            manifest.append({"model": f"lgb{model_index}", "outer_fold": outer_fold, "file": str(model_path), "sha256": _sha(model_path), "best_iteration": int(model.best_iteration_), "base_features": len(base_feature_columns), "selector_features": len(extra_columns)})
            importance_rows.extend({"model": f"lgb{model_index}", "outer_fold": outer_fold, "feature": feature, "importance": float(value)} for feature, value in zip(all_feature_columns, model.feature_importances_))
            del model, pred
            gc.collect()
        del x_train, x_valid, x_train_values, x_valid_values
        gc.collect()
    for index, values in enumerate(oof):
        if not np.isfinite(values).all():
            raise AssertionError(f"incomplete final OOF for lgb{index}")
    ensemble = np.mean(np.vstack(oof), axis=0).astype(np.float32)
    prediction = base_frame[KEYS + ["last_known_tvt", "target"]].copy()
    for index, values in enumerate(oof):
        prediction[f"lgb{index}_pred_tvt"] = anchor + values
    prediction["lgb_mean_pred_tvt"] = anchor + ensemble
    metric_rows = []
    for name in [*(f"lgb{i}" for i in range(len(oof))), "lgb_mean"]:
        pred = prediction[f"{name}_pred_tvt"].to_numpy(np.float32)
        metric_rows.append({"model": name, "rmse_tvt": float(np.sqrt(np.mean((pred-truth)**2))), "rows": len(prediction)})
    metrics = pd.DataFrame(metric_rows).sort_values("rmse_tvt")
    importance = pd.DataFrame(importance_rows)
    importance_mean = importance.groupby(["model", "feature"], as_index=False).importance.mean().sort_values(["model", "importance"], ascending=[True, False])
    return metrics, prediction, importance_mean, manifest
