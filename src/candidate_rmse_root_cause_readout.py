from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.candidate_selector_pipeline import sha256_file, write_json


EPS = 1.0e-7
MARGIN_EDGES = np.asarray(
    [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf],
    dtype=np.float64,
)
MARGIN_LABELS = [
    "0_0p05",
    "0p05_0p1",
    "0p1_0p25",
    "0p25_0p5",
    "0p5_1",
    "1_2",
    "2_plus",
]


def _reshape(series: pd.Series, n_candidates: int) -> np.ndarray:
    values = series.to_numpy()
    if len(values) % n_candidates:
        raise ValueError("candidate-long row count does not preserve base-row blocks")
    return values.reshape(-1, n_candidates)


def _init_candidate_stats(n_folds: int, n_candidates: int) -> dict[str, np.ndarray]:
    shape = (n_folds, n_candidates)
    names = [
        "count",
        "actual_sum",
        "actual_sq_sum",
        "parent_pred_sum",
        "exp407_pred_sum",
        "parent_score_abs_error_sum",
        "exp407_score_abs_error_sum",
        "delta_pred_sum",
        "delta_pred_sq_sum",
        "parent_brier_sum",
        "exp407_brier_sum",
        "parent_logloss_sum",
        "exp407_logloss_sum",
    ]
    return {name: np.zeros(shape, dtype=np.float64) for name in names}


def _validate_pair(
    parent: pd.DataFrame,
    exp407: pd.DataFrame,
    *,
    n_candidates: int,
    candidate_order: Sequence[str] | None,
    row_group: int,
) -> list[str]:
    key_columns = ["id", "well", "well_row_idx", "outer_fold", "candidate_id"]
    if not parent[key_columns].equals(exp407[key_columns]):
        raise ValueError(f"parent/exp407 key mismatch in row group {row_group}")
    for column in ("actual_abs_error", "actual_within10", "candidate_tvt"):
        if not np.array_equal(
            parent[column].to_numpy(),
            exp407[column].to_numpy(),
            equal_nan=True,
        ):
            raise ValueError(
                f"parent/exp407 {column} mismatch in row group {row_group}"
            )
    if len(parent) % n_candidates:
        raise ValueError(f"row group {row_group} breaks candidate blocks")
    observed_order = (
        parent["candidate_id"].iloc[:n_candidates].astype(str).tolist()
    )
    expected_order = (
        observed_order if candidate_order is None else list(candidate_order)
    )
    matrix = _reshape(parent["candidate_id"], n_candidates).astype(str)
    if not np.all(matrix == np.asarray(expected_order, dtype=str)[None, :]):
        raise ValueError(f"candidate order mismatch in row group {row_group}")
    return expected_order


def _update_candidate_stats(
    stats: dict[str, np.ndarray],
    folds: np.ndarray,
    actual: np.ndarray,
    actual_binary: np.ndarray,
    parent_pred: np.ndarray,
    exp407_pred: np.ndarray,
    parent_prob: np.ndarray,
    exp407_prob: np.ndarray,
    n_folds: int,
) -> None:
    delta = exp407_pred - parent_pred
    for fold in range(n_folds):
        mask = folds == fold
        if not mask.any():
            continue
        actual_fold = actual[mask]
        binary_fold = actual_binary[mask]
        parent_fold = parent_pred[mask]
        exp407_fold = exp407_pred[mask]
        parent_probability = np.clip(parent_prob[mask], EPS, 1.0 - EPS)
        exp407_probability = np.clip(exp407_prob[mask], EPS, 1.0 - EPS)
        delta_fold = delta[mask]
        stats["count"][fold] += mask.sum(axis=0)
        stats["actual_sum"][fold] += actual_fold.sum(axis=0)
        stats["actual_sq_sum"][fold] += np.square(actual_fold).sum(axis=0)
        stats["parent_pred_sum"][fold] += parent_fold.sum(axis=0)
        stats["exp407_pred_sum"][fold] += exp407_fold.sum(axis=0)
        stats["parent_score_abs_error_sum"][fold] += np.abs(
            parent_fold - actual_fold
        ).sum(axis=0)
        stats["exp407_score_abs_error_sum"][fold] += np.abs(
            exp407_fold - actual_fold
        ).sum(axis=0)
        stats["delta_pred_sum"][fold] += delta_fold.sum(axis=0)
        stats["delta_pred_sq_sum"][fold] += np.square(delta_fold).sum(axis=0)
        stats["parent_brier_sum"][fold] += np.square(
            parent_probability - binary_fold
        ).sum(axis=0)
        stats["exp407_brier_sum"][fold] += np.square(
            exp407_probability - binary_fold
        ).sum(axis=0)
        stats["parent_logloss_sum"][fold] += (
            -(
                binary_fold * np.log(parent_probability)
                + (1.0 - binary_fold) * np.log(1.0 - parent_probability)
            )
        ).sum(axis=0)
        stats["exp407_logloss_sum"][fold] += (
            -(
                binary_fold * np.log(exp407_probability)
                + (1.0 - binary_fold) * np.log(1.0 - exp407_probability)
            )
        ).sum(axis=0)


def _candidate_fold_frame(
    candidate_order: Sequence[str],
    stats: Mapping[str, np.ndarray],
    weight_table: pd.DataFrame,
    *,
    n_folds: int,
) -> pd.DataFrame:
    counts = stats["count"]
    if np.any(counts <= 0):
        raise ValueError("candidate/fold stats contain empty cells")
    delta_mean = stats["delta_pred_sum"] / counts
    delta_var = np.maximum(
        0.0,
        stats["delta_pred_sq_sum"] / counts - np.square(delta_mean),
    )
    rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        for position, candidate_id in enumerate(candidate_order):
            count = counts[fold, position]
            rows.append(
                {
                    "outer_fold": fold,
                    "candidate_position": position,
                    "candidate_id": candidate_id,
                    "rows": int(count),
                    "actual_abs_error_mean": stats["actual_sum"][
                        fold, position
                    ]
                    / count,
                    "actual_candidate_rmse": math.sqrt(
                        stats["actual_sq_sum"][fold, position] / count
                    ),
                    "parent_pred_mean": stats["parent_pred_sum"][fold, position]
                    / count,
                    "exp407_pred_mean": stats["exp407_pred_sum"][fold, position]
                    / count,
                    "delta_pred_mean": delta_mean[fold, position],
                    "delta_pred_std": math.sqrt(delta_var[fold, position]),
                    "parent_score_mae": stats["parent_score_abs_error_sum"][
                        fold, position
                    ]
                    / count,
                    "exp407_score_mae": stats["exp407_score_abs_error_sum"][
                        fold, position
                    ]
                    / count,
                    "delta_score_mae": (
                        stats["exp407_score_abs_error_sum"][fold, position]
                        - stats["parent_score_abs_error_sum"][fold, position]
                    )
                    / count,
                    "parent_logloss": stats["parent_logloss_sum"][fold, position]
                    / count,
                    "exp407_logloss": stats["exp407_logloss_sum"][
                        fold, position
                    ]
                    / count,
                    "delta_logloss": (
                        stats["exp407_logloss_sum"][fold, position]
                        - stats["parent_logloss_sum"][fold, position]
                    )
                    / count,
                    "parent_brier": stats["parent_brier_sum"][fold, position]
                    / count,
                    "exp407_brier": stats["exp407_brier_sum"][fold, position]
                    / count,
                    "delta_brier": (
                        stats["exp407_brier_sum"][fold, position]
                        - stats["parent_brier_sum"][fold, position]
                    )
                    / count,
                }
            )
    frame = pd.DataFrame(rows)
    weights = weight_table.rename(columns={"fit_partition": "outer_fold"})
    merge_columns = [
        "outer_fold",
        "candidate_position",
        "candidate_id",
        "fit_candidate_rmse",
        "final_weight",
    ]
    frame = frame.merge(
        weights[merge_columns],
        on=["outer_fold", "candidate_position", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    if frame[["fit_candidate_rmse", "final_weight"]].isna().any().any():
        raise ValueError("exp407 weight merge produced missing values")
    return frame


def _init_selector_stats(
    names: Sequence[str],
    *,
    n_folds: int,
    n_legal: int,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            "rows": np.zeros(n_folds, dtype=np.int64),
            "sse": np.zeros(n_folds, dtype=np.float64),
            "absolute_error_sum": np.zeros(n_folds, dtype=np.float64),
            "switch_vs_parent": np.zeros(n_folds, dtype=np.int64),
            "selection_count": np.zeros((n_folds, n_legal), dtype=np.int64),
        }
        for name in names
    }


def _first_pass(
    parent_path: Path,
    exp407_path: Path,
    *,
    n_candidates: int,
    n_folds: int,
) -> tuple[list[str], dict[str, np.ndarray]]:
    parent_file = pq.ParquetFile(parent_path)
    exp407_file = pq.ParquetFile(exp407_path)
    if parent_file.metadata.num_rows != exp407_file.metadata.num_rows:
        raise ValueError("parent/exp407 OOF row counts differ")
    if parent_file.metadata.num_row_groups != exp407_file.metadata.num_row_groups:
        raise ValueError("parent/exp407 OOF row-group counts differ")
    columns = [
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "candidate_id",
        "candidate_tvt",
        "actual_abs_error",
        "actual_within10",
        "pred_abs_error",
        "p_within10",
    ]
    stats = _init_candidate_stats(n_folds, n_candidates)
    candidate_order: list[str] | None = None
    for row_group in range(parent_file.metadata.num_row_groups):
        parent = parent_file.read_row_group(
            row_group, columns=columns
        ).to_pandas()
        exp407 = exp407_file.read_row_group(
            row_group, columns=columns
        ).to_pandas()
        candidate_order = _validate_pair(
            parent,
            exp407,
            n_candidates=n_candidates,
            candidate_order=candidate_order,
            row_group=row_group,
        )
        folds = _reshape(parent["outer_fold"], n_candidates)[:, 0].astype(
            np.int64
        )
        _update_candidate_stats(
            stats,
            folds,
            _reshape(parent["actual_abs_error"], n_candidates).astype(np.float64),
            _reshape(parent["actual_within10"], n_candidates).astype(np.float64),
            _reshape(parent["pred_abs_error"], n_candidates).astype(np.float64),
            _reshape(exp407["pred_abs_error"], n_candidates).astype(np.float64),
            _reshape(parent["p_within10"], n_candidates).astype(np.float64),
            _reshape(exp407["p_within10"], n_candidates).astype(np.float64),
            n_folds,
        )
        if (row_group + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "root_cause_first_pass",
                        "row_groups": row_group + 1,
                        "total": parent_file.metadata.num_row_groups,
                    }
                ),
                flush=True,
            )
    if candidate_order is None:
        raise ValueError("candidate-score OOF files contain no rows")
    return candidate_order, stats


def _second_pass(
    parent_path: Path,
    exp407_path: Path,
    treatment_path: Path,
    *,
    candidate_order: Sequence[str],
    candidate_fold: pd.DataFrame,
    primary_domain: Sequence[str],
    n_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n_candidates = len(candidate_order)
    primary_positions = [
        list(candidate_order).index(str(candidate)) for candidate in primary_domain
    ]
    n_legal = len(primary_positions)
    parent_file = pq.ParquetFile(parent_path)
    exp407_file = pq.ParquetFile(exp407_path)
    treatment_file = pq.ParquetFile(treatment_path)
    expected_rows = parent_file.metadata.num_rows
    if (
        exp407_file.metadata.num_rows != expected_rows
        or treatment_file.metadata.num_rows != expected_rows
    ):
        raise ValueError("parent/exp407/treatment OOF row counts differ")
    if (
        exp407_file.metadata.num_row_groups != parent_file.metadata.num_row_groups
        or treatment_file.metadata.num_row_groups
        != parent_file.metadata.num_row_groups
    ):
        raise ValueError("parent/exp407/treatment row-group counts differ")

    pair_columns = [
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "md_since",
        "candidate_id",
        "candidate_tvt",
        "actual_abs_error",
        "actual_within10",
        "pred_abs_error",
        "p_within10",
    ]
    treatment_columns = [
        "id",
        "well",
        "well_row_idx",
        "outer_fold",
        "md_since",
        "candidate_id",
        "candidate_tvt",
        "actual_abs_error",
        "pred_abs_error",
    ]
    delta_lookup = (
        candidate_fold.pivot(
            index="outer_fold",
            columns="candidate_position",
            values="delta_pred_mean",
        )
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
    )
    selector_names = [
        "parent",
        "exp407",
        "global_shift_only",
        "local_change_only",
        "rmse_offset_treatment",
    ]
    selector_stats = _init_selector_stats(
        selector_names, n_folds=n_folds, n_legal=n_legal
    )
    margin_rows = np.zeros((n_folds, len(MARGIN_LABELS)), dtype=np.int64)
    margin_switches = np.zeros((n_folds, len(MARGIN_LABELS)), dtype=np.int64)
    margin_harmful = np.zeros((n_folds, len(MARGIN_LABELS)), dtype=np.int64)
    margin_delta_sse = np.zeros((n_folds, len(MARGIN_LABELS)), dtype=np.float64)
    harmful_switch_count = np.zeros(n_folds, dtype=np.int64)
    helpful_switch_count = np.zeros(n_folds, dtype=np.int64)
    switched_count = np.zeros(n_folds, dtype=np.int64)
    switched_delta_sse = np.zeros(n_folds, dtype=np.float64)
    instability_count = np.zeros((n_folds, n_candidates), dtype=np.float64)
    exp407_delta_sum = np.zeros((n_folds, n_candidates), dtype=np.float64)
    exp407_delta_sq_sum = np.zeros((n_folds, n_candidates), dtype=np.float64)
    treatment_delta_sum = np.zeros((n_folds, n_candidates), dtype=np.float64)
    treatment_delta_sq_sum = np.zeros((n_folds, n_candidates), dtype=np.float64)

    for row_group in range(parent_file.metadata.num_row_groups):
        parent = parent_file.read_row_group(
            row_group, columns=pair_columns
        ).to_pandas()
        exp407 = exp407_file.read_row_group(
            row_group, columns=pair_columns
        ).to_pandas()
        treatment = treatment_file.read_row_group(
            row_group, columns=treatment_columns
        ).to_pandas()
        _validate_pair(
            parent,
            exp407,
            n_candidates=n_candidates,
            candidate_order=candidate_order,
            row_group=row_group,
        )
        key_columns = [
            "id",
            "well",
            "well_row_idx",
            "outer_fold",
            "candidate_id",
        ]
        if not parent[key_columns].equals(treatment[key_columns]):
            raise ValueError(f"treatment key mismatch in row group {row_group}")
        for column in ("candidate_tvt", "actual_abs_error"):
            if not np.array_equal(
                parent[column].to_numpy(),
                treatment[column].to_numpy(),
                equal_nan=True,
            ):
                raise ValueError(
                    f"treatment {column} mismatch in row group {row_group}"
                )

        folds = _reshape(parent["outer_fold"], n_candidates)[:, 0].astype(
            np.int64
        )
        actual_all = _reshape(
            parent["actual_abs_error"], n_candidates
        ).astype(np.float64)
        parent_all = _reshape(parent["pred_abs_error"], n_candidates).astype(
            np.float64
        )
        exp407_all = _reshape(exp407["pred_abs_error"], n_candidates).astype(
            np.float64
        )
        treatment_all = _reshape(
            treatment["pred_abs_error"], n_candidates
        ).astype(np.float64)
        for fold in range(n_folds):
            fold_mask = folds == fold
            if not fold_mask.any():
                continue
            exp_delta = exp407_all[fold_mask] - parent_all[fold_mask]
            treatment_delta = treatment_all[fold_mask] - parent_all[fold_mask]
            instability_count[fold] += fold_mask.sum(axis=0)
            exp407_delta_sum[fold] += exp_delta.sum(axis=0)
            exp407_delta_sq_sum[fold] += np.square(exp_delta).sum(axis=0)
            treatment_delta_sum[fold] += treatment_delta.sum(axis=0)
            treatment_delta_sq_sum[fold] += np.square(treatment_delta).sum(axis=0)

        actual = actual_all[:, primary_positions]
        parent_score = parent_all[:, primary_positions]
        exp407_score = exp407_all[:, primary_positions]
        treatment_score = treatment_all[:, primary_positions]
        shifts = delta_lookup[folds][:, primary_positions]
        selector_indices = {
            "parent": np.argmin(parent_score, axis=1),
            "exp407": np.argmin(exp407_score, axis=1),
            "global_shift_only": np.argmin(parent_score + shifts, axis=1),
            "local_change_only": np.argmin(exp407_score - shifts, axis=1),
            "rmse_offset_treatment": np.argmin(treatment_score, axis=1),
        }
        parent_index = selector_indices["parent"]
        exp407_index = selector_indices["exp407"]
        row_index = np.arange(len(folds))
        parent_actual = actual[row_index, parent_index]
        exp407_actual = actual[row_index, exp407_index]
        switch = exp407_index != parent_index
        delta_sse = np.square(exp407_actual) - np.square(parent_actual)
        sorted_parent = np.partition(parent_score, 1, axis=1)
        margin = sorted_parent[:, 1] - sorted_parent[:, 0]
        margin_bin = np.clip(
            np.searchsorted(MARGIN_EDGES, margin, side="right") - 1,
            0,
            len(MARGIN_LABELS) - 1,
        )

        for fold in range(n_folds):
            fold_mask = folds == fold
            fold_switch = fold_mask & switch
            switched_count[fold] += int(fold_switch.sum())
            harmful_switch_count[fold] += int(
                np.sum(fold_switch & (delta_sse > 0.0))
            )
            helpful_switch_count[fold] += int(
                np.sum(fold_switch & (delta_sse < 0.0))
            )
            switched_delta_sse[fold] += float(delta_sse[fold_switch].sum())
            for bin_index in range(len(MARGIN_LABELS)):
                mask = fold_mask & (margin_bin == bin_index)
                margin_rows[fold, bin_index] += int(mask.sum())
                margin_switches[fold, bin_index] += int(np.sum(mask & switch))
                margin_harmful[fold, bin_index] += int(
                    np.sum(mask & switch & (delta_sse > 0.0))
                )
                margin_delta_sse[fold, bin_index] += float(
                    delta_sse[mask & switch].sum()
                )

        for selector, indices in selector_indices.items():
            chosen = actual[row_index, indices]
            for fold in range(n_folds):
                mask = folds == fold
                selected = indices[mask]
                selected_error = chosen[mask]
                selector_stats[selector]["rows"][fold] += int(mask.sum())
                selector_stats[selector]["sse"][fold] += float(
                    np.square(selected_error).sum()
                )
                selector_stats[selector]["absolute_error_sum"][fold] += float(
                    selected_error.sum()
                )
                selector_stats[selector]["switch_vs_parent"][fold] += int(
                    np.sum(selected != parent_index[mask])
                )
                np.add.at(
                    selector_stats[selector]["selection_count"][fold],
                    selected,
                    1,
                )
        if (row_group + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "root_cause_second_pass",
                        "row_groups": row_group + 1,
                        "total": parent_file.metadata.num_row_groups,
                    }
                ),
                flush=True,
            )

    selector_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for selector in selector_names:
        for fold in range(n_folds):
            rows = int(selector_stats[selector]["rows"][fold])
            selector_rows.append(
                {
                    "selector": selector,
                    "outer_fold": fold,
                    "rows": rows,
                    "rmse": math.sqrt(
                        selector_stats[selector]["sse"][fold] / rows
                    ),
                    "mae": (
                        selector_stats[selector]["absolute_error_sum"][fold] / rows
                    ),
                    "switch_rate_vs_parent": (
                        selector_stats[selector]["switch_vs_parent"][fold] / rows
                    ),
                }
            )
            for local_position, candidate_position in enumerate(primary_positions):
                selection_rows.append(
                    {
                        "selector": selector,
                        "outer_fold": fold,
                        "candidate_position": candidate_position,
                        "candidate_id": candidate_order[candidate_position],
                        "selected_rows": int(
                            selector_stats[selector]["selection_count"][
                                fold, local_position
                            ]
                        ),
                        "selection_rate": (
                            selector_stats[selector]["selection_count"][
                                fold, local_position
                            ]
                            / rows
                        ),
                    }
                )

    margin_output: list[dict[str, Any]] = []
    for fold in range(n_folds):
        for bin_index, label in enumerate(MARGIN_LABELS):
            rows = int(margin_rows[fold, bin_index])
            switches = int(margin_switches[fold, bin_index])
            margin_output.append(
                {
                    "outer_fold": fold,
                    "parent_margin_bucket": label,
                    "rows": rows,
                    "switches": switches,
                    "switch_rate": switches / rows if rows else math.nan,
                    "harmful_switches": int(margin_harmful[fold, bin_index]),
                    "harmful_share_of_switches": (
                        margin_harmful[fold, bin_index] / switches
                        if switches
                        else math.nan
                    ),
                    "switched_delta_sse": margin_delta_sse[fold, bin_index],
                }
            )

    exp_mean = exp407_delta_sum / instability_count
    treatment_mean = treatment_delta_sum / instability_count
    exp_var = np.maximum(
        0.0,
        exp407_delta_sq_sum / instability_count - np.square(exp_mean),
    )
    treatment_var = np.maximum(
        0.0,
        treatment_delta_sq_sum / instability_count - np.square(treatment_mean),
    )
    instability_rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        for position, candidate_id in enumerate(candidate_order):
            instability_rows.append(
                {
                    "outer_fold": fold,
                    "candidate_position": position,
                    "candidate_id": candidate_id,
                    "rows": int(instability_count[fold, position]),
                    "exp407_delta_mean": exp_mean[fold, position],
                    "exp407_centered_delta_std": math.sqrt(
                        exp_var[fold, position]
                    ),
                    "treatment_delta_mean": treatment_mean[fold, position],
                    "treatment_centered_delta_std": math.sqrt(
                        treatment_var[fold, position]
                    ),
                }
            )
    summary = {
        "base_rows": int(sum(selector_stats["parent"]["rows"])),
        "switched_rows": int(switched_count.sum()),
        "switch_rate": float(
            switched_count.sum() / sum(selector_stats["parent"]["rows"])
        ),
        "harmful_switch_rows": int(harmful_switch_count.sum()),
        "helpful_switch_rows": int(helpful_switch_count.sum()),
        "harmful_share_of_non_tie_switches": float(
            harmful_switch_count.sum()
            / (harmful_switch_count.sum() + helpful_switch_count.sum())
        ),
        "switched_delta_sse": float(switched_delta_sse.sum()),
    }
    return (
        pd.DataFrame(selector_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(margin_output),
        pd.DataFrame(instability_rows),
        summary,
    )


def _pooled_selector_summary(selector_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    parent_by_fold = selector_fold[
        selector_fold["selector"].eq("parent")
    ].set_index("outer_fold")["rmse"]
    for selector, frame in selector_fold.groupby("selector", sort=False):
        total_rows = int(frame["rows"].sum())
        pooled_rmse = math.sqrt(
            float(np.sum(np.square(frame["rmse"]) * frame["rows"])) / total_rows
        )
        values = frame.set_index("outer_fold")["rmse"]
        rows.append(
            {
                "selector": selector,
                "rows": total_rows,
                "rmse": pooled_rmse,
                "fold_nonworse_vs_parent": int(
                    np.sum(values <= parent_by_fold + 1.0e-12)
                ),
            }
        )
    return pd.DataFrame(rows)


def run_root_cause_readout(
    *,
    parent_path: Path,
    exp407_path: Path,
    treatment_path: Path,
    exp407_weight_table_path: Path,
    candidate_order: Sequence[str],
    primary_domain: Sequence[str],
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_candidates = [str(item) for item in candidate_order]
    expected_primary = [str(item) for item in primary_domain]
    n_candidates = len(expected_candidates)
    n_folds = int(config["validation"]["outer_folds"])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    observed_order, stats = _first_pass(
        Path(parent_path),
        Path(exp407_path),
        n_candidates=n_candidates,
        n_folds=n_folds,
    )
    if observed_order != expected_candidates:
        raise ValueError("OOF candidate order differs from the frozen contract")
    weight_table = pd.read_csv(exp407_weight_table_path)
    candidate_fold = _candidate_fold_frame(
        observed_order,
        stats,
        weight_table,
        n_folds=n_folds,
    )
    (
        selector_fold,
        selection,
        margin,
        instability,
        switch_summary,
    ) = _second_pass(
        Path(parent_path),
        Path(exp407_path),
        Path(treatment_path),
        candidate_order=observed_order,
        candidate_fold=candidate_fold,
        primary_domain=expected_primary,
        n_folds=n_folds,
    )
    pooled = _pooled_selector_summary(selector_fold)

    selection_pivot = selection.pivot_table(
        index=["outer_fold", "candidate_position", "candidate_id"],
        columns="selector",
        values="selection_rate",
    ).reset_index()
    weight_effect = candidate_fold.merge(
        selection_pivot,
        on=["outer_fold", "candidate_position", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    weight_effect["delta_selection_rate"] = (
        weight_effect["exp407"] - weight_effect["parent"]
    )
    correlation_columns = [
        "delta_pred_mean",
        "delta_pred_std",
        "delta_score_mae",
        "delta_logloss",
        "delta_brier",
        "delta_selection_rate",
    ]
    correlations = {
        "rows": len(weight_effect),
        "final_weight_pearson": {
            column: float(
                weight_effect["final_weight"].corr(
                    weight_effect[column], method="pearson"
                )
            )
            for column in correlation_columns
        },
        "final_weight_spearman": {
            column: float(
                weight_effect["final_weight"].corr(
                    weight_effect[column], method="spearman"
                )
            )
            for column in correlation_columns
        },
    }

    candidate_fold_path = root / "root_cause_candidate_fold_score_shift.csv"
    selector_fold_path = root / "root_cause_selector_counterfactual_by_fold.csv"
    pooled_path = root / "root_cause_selector_counterfactual_pooled.csv"
    selection_path = root / "root_cause_selection_rate_by_candidate_fold.csv"
    margin_path = root / "root_cause_parent_margin_switch_readout.csv"
    weight_effect_path = root / "root_cause_candidate_weight_effect_join.csv"
    instability_path = root / "root_cause_treatment_instability.csv"
    candidate_fold.to_csv(candidate_fold_path, index=False)
    selector_fold.to_csv(selector_fold_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    selection.to_csv(selection_path, index=False)
    margin.to_csv(margin_path, index=False)
    weight_effect.to_csv(weight_effect_path, index=False)
    instability.to_csv(instability_path, index=False)

    pooled_lookup = pooled.set_index("selector").to_dict(orient="index")
    margin_pooled = margin.groupby(
        "parent_margin_bucket", as_index=False
    ).agg(
        rows=("rows", "sum"),
        switches=("switches", "sum"),
        harmful_switches=("harmful_switches", "sum"),
        switched_delta_sse=("switched_delta_sse", "sum"),
    )
    confident_damage = float(
        margin_pooled.loc[
            margin_pooled["parent_margin_bucket"].isin(["0p5_1", "1_2"]),
            "switched_delta_sse",
        ].sum()
    )
    total_damage = float(switch_summary["switched_delta_sse"])
    confident_damage_share = (
        confident_damage / total_damage if total_damage > 0 else math.nan
    )
    root_cfg = config["root_cause_gate"]
    spearman = correlations["final_weight_spearman"]
    root_checks = {
        "global_shift_only_non_regression": float(
            pooled_lookup["global_shift_only"]["rmse"]
        )
        <= float(root_cfg["global_shift_only_hard_rmse_max"]),
        "global_shift_only_fold_consistency": int(
            pooled_lookup["global_shift_only"]["fold_nonworse_vs_parent"]
        )
        >= int(root_cfg["global_shift_only_nonworse_folds_min"]),
        "local_change_only_reproduces_regression": float(
            pooled_lookup["local_change_only"]["rmse"]
        )
        >= float(root_cfg["local_change_only_hard_rmse_min"]),
        "local_change_only_fold_consistency": int(
            pooled_lookup["local_change_only"]["fold_nonworse_vs_parent"]
        )
        <= int(root_cfg["local_change_only_nonworse_folds_max"]),
        "low_weight_candidates_have_larger_local_drift": float(
            spearman["delta_pred_std"]
        )
        <= float(root_cfg["final_weight_delta_pred_std_spearman_max"]),
        "low_weight_candidates_have_worse_score_mae": float(
            spearman["delta_score_mae"]
        )
        <= float(root_cfg["final_weight_delta_score_mae_spearman_max"]),
        "binary_objective_is_misaligned": float(spearman["delta_logloss"])
        <= float(root_cfg["final_weight_delta_logloss_spearman_max"]),
        "candidate_constant_shift_is_not_dose_response": abs(
            float(spearman["delta_pred_mean"])
        )
        <= float(root_cfg["final_weight_delta_pred_mean_spearman_abs_max"]),
        "confident_parent_margin_damage": confident_damage_share
        >= float(root_cfg["confident_margin_net_damage_share_min"]),
    }
    treatment_instability = {
        "exp407_mean_centered_delta_std": float(
            instability["exp407_centered_delta_std"].mean()
        ),
        "treatment_mean_centered_delta_std": float(
            instability["treatment_centered_delta_std"].mean()
        ),
        "treatment_relative_to_exp407": float(
            instability["treatment_centered_delta_std"].mean()
            / instability["exp407_centered_delta_std"].mean()
        ),
    }
    summary = {
        "status": "exp407_root_cause_and_exp414_treatment_surface_readout_complete",
        "candidate_order": observed_order,
        "input_sha256": {
            "parent_candidate_score_oof": sha256_file(parent_path),
            "exp407_candidate_score_oof": sha256_file(exp407_path),
            "treatment_candidate_score_oof": sha256_file(treatment_path),
            "exp407_weight_table": sha256_file(exp407_weight_table_path),
        },
        "switch_summary": switch_summary,
        "weight_effect_correlation": correlations,
        "selector_counterfactual_pooled": pooled.to_dict(orient="records"),
        "confident_parent_margin_0p5_to_2": {
            "net_delta_sse": confident_damage,
            "share_of_total_net_switched_delta_sse": confident_damage_share,
        },
        "treatment_instability": treatment_instability,
        "root_cause_gate": {
            "checks": root_checks,
            "passed": bool(all(root_checks.values())),
            "interpretation": (
                "inverse_rmse_task_importance_caused_distributed_row_local_"
                "score_surface_drift_not_candidate_constant_bias"
            ),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for path in (
                candidate_fold_path,
                selector_fold_path,
                pooled_path,
                selection_path,
                margin_path,
                weight_effect_path,
                instability_path,
            )
        },
    }
    summary_path = root / "root_cause_summary.json"
    write_json(summary_path, summary)
    summary["summary_file_sha256"] = sha256_file(summary_path)
    return summary


__all__ = ["run_root_cause_readout"]
