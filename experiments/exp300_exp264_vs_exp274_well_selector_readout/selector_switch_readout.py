from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/exp300_exp264_vs_exp274_well_selector_readout"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
SOURCE_DIR = ARTIFACT_DIR / "source_inputs"
SCORE_PATH = (
    SOURCE_DIR
    / "exp264_stage_c_v6/artifacts/nested_outer_valid_candidate_score.parquet"
)
EXP274_OOF_PATH = (
    SOURCE_DIR / "exp274_catboost_final_regressor_swap_on_exp238_oof_predictions.csv.gz"
)
EXP264_VIEWER_PATH = ROOT / (
    "experiments/exp264_exp263_candidate_confidence_dual_selector/artifacts/"
    "exp264_exp263_candidate_confidence_dual_selector_stage_d_v3_oof_viewer.csv"
)
WELL_PATH = ARTIFACT_DIR / "well_comparison_and_features.csv"

EXPECTED_ROWS = 3_783_989
EXPECTED_LONG_ROWS = EXPECTED_ROWS * 12
EXPECTED_WELLS = 773
EXPECTED_SCORE_SHA256 = (
    "a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc"
)
EXPECTED_HARD_PRIMARY_RMSE = 8.652531955610227
CHUNK_SIZE = 200_000
WINDOWS = [0, 1, 5, 25, 100]

CANDIDATES = [
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
]
PRIMARY_COUNT = 11


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64) - target.astype(np.float64)
                )
            )
        )
    )


def load_base_arrays(
    well: pd.DataFrame,
) -> dict[str, Any]:
    well_names = well["well"].astype(str).tolist()
    well_to_code = {name: index for index, name in enumerate(well_names)}
    n_wells = len(well_names)
    target = np.empty(EXPECTED_ROWS, np.float32)
    pred274 = np.empty(EXPECTED_ROWS, np.float32)
    pred264 = np.empty(EXPECTED_ROWS, np.float32)
    well_code = np.empty(EXPECTED_ROWS, np.int16)
    raw_row_idx = np.empty(EXPECTED_ROWS, np.int32)
    outer_fold = np.empty(EXPECTED_ROWS, np.int8)
    well_start = np.full(n_wells, -1, np.int64)
    first_raw_row = np.full(n_wells, -1, np.int32)

    iter274 = pd.read_csv(
        EXP274_OOF_PATH,
        usecols=[
            "id",
            "well",
            "outer_fold",
            "target_tvt",
            "catboost_public_cb0_tvt",
        ],
        dtype={
            "id": "string",
            "well": "string",
            "outer_fold": "int8",
            "target_tvt": "float32",
            "catboost_public_cb0_tvt": "float32",
        },
        chunksize=CHUNK_SIZE,
    )
    iter264 = pd.read_csv(
        EXP264_VIEWER_PATH,
        dtype={"id": "string", "tvt": "float32"},
        chunksize=CHUNK_SIZE,
    )
    total = 0
    for chunk_index, (frame274, frame264) in enumerate(
        zip(iter274, iter264, strict=True), start=1
    ):
        if len(frame274) != len(frame264) or not frame274["id"].reset_index(
            drop=True
        ).equals(frame264["id"].reset_index(drop=True)):
            raise ValueError(f"exp264/exp274 ID mismatch in chunk {chunk_index}")
        size = len(frame274)
        row_slice = slice(total, total + size)
        names = frame274["well"].astype(str)
        codes = names.map(well_to_code).to_numpy(np.int16)
        if bool(np.any(codes < 0)):
            raise ValueError("OOF contains a well absent from the well comparison")
        suffix = frame274["id"].str.slice(9).astype("int32").to_numpy()
        positions = np.arange(total, total + size, dtype=np.int64)
        for code in np.unique(codes):
            local = np.flatnonzero(codes == code)
            if well_start[code] < 0:
                well_start[code] = positions[local[0]]
                first_raw_row[code] = suffix[local[0]]
        target[row_slice] = frame274["target_tvt"].to_numpy(np.float32)
        pred274[row_slice] = frame274["catboost_public_cb0_tvt"].to_numpy(
            np.float32
        )
        pred264[row_slice] = frame264["tvt"].to_numpy(np.float32)
        well_code[row_slice] = codes
        raw_row_idx[row_slice] = suffix
        outer_fold[row_slice] = frame274["outer_fold"].to_numpy(np.int8)
        total += size
        print(f"loaded OOF base {total:,}/{EXPECTED_ROWS:,}", flush=True)

    if total != EXPECTED_ROWS or bool(np.any(well_start < 0)):
        raise ValueError("OOF base coverage failed")
    positions = np.arange(EXPECTED_ROWS, dtype=np.int64)
    expected_raw = first_raw_row[well_code] + positions - well_start[well_code]
    if not np.array_equal(expected_raw.astype(np.int32), raw_row_idx):
        raise ValueError("OOF well rows are not contiguous in global order")
    counts = np.bincount(well_code, minlength=n_wells)
    if not np.array_equal(counts, well["rows"].to_numpy(np.int64)):
        raise ValueError("OOF well row counts differ from the well readout")
    return {
        "well_names": well_names,
        "well_to_code": well_to_code,
        "target": target,
        "pred274": pred274,
        "pred264": pred264,
        "well_code": well_code,
        "raw_row_idx": raw_row_idx,
        "outer_fold": outer_fold,
        "well_start": well_start,
        "first_raw_row": first_raw_row,
    }


def row_positions_from_score_chunk(
    chunk: pd.DataFrame,
    block_rows: int,
    base: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    candidate_blocks = chunk["candidate_id"].astype(str).to_numpy().reshape(
        block_rows, len(CANDIDATES)
    )
    if not np.all(candidate_blocks == np.asarray(CANDIDATES)[None, :]):
        raise ValueError("candidate order changed")
    wells = chunk["well"].astype(str).to_numpy().reshape(
        block_rows, len(CANDIDATES)
    )
    if not np.all(wells == wells[:, :1]):
        raise ValueError("candidate block contains multiple wells")
    codes = pd.Series(wells[:, 0]).map(base["well_to_code"]).to_numpy()
    if bool(pd.isna(codes).any()):
        raise ValueError("score contains a well absent from the OOF base")
    codes = codes.astype(np.int16)
    raw_rows = chunk["well_row_idx"].to_numpy(np.int32).reshape(
        block_rows, len(CANDIDATES)
    )
    if not np.all(raw_rows == raw_rows[:, :1]):
        raise ValueError("candidate block row index changed")
    positions = (
        base["well_start"][codes]
        + raw_rows[:, 0]
        - base["first_raw_row"][codes]
    )
    if bool(np.any((positions < 0) | (positions >= EXPECTED_ROWS))):
        raise ValueError("score row mapped outside the OOF base")
    if not np.array_equal(base["well_code"][positions], codes):
        raise ValueError("score well mapping differs from the OOF base")
    return positions.astype(np.int64), codes


def reconstruct_primary_surface(
    base: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selector_code = np.full(EXPECTED_ROWS, -1, np.int8)
    selector_tvt = np.full(EXPECTED_ROWS, np.nan, np.float32)
    selector_margin = np.full(EXPECTED_ROWS, np.nan, np.float32)
    covered = np.zeros(EXPECTED_ROWS, bool)
    parquet = pq.ParquetFile(SCORE_PATH)
    if parquet.metadata.num_rows != EXPECTED_LONG_ROWS:
        raise ValueError("Stage C candidate-long row count changed")
    columns = [
        "well",
        "well_row_idx",
        "candidate_id",
        "candidate_tvt",
        "pred_abs_error",
        "outer_fold",
        "downstream_outer_fold",
        "nested_model_count",
    ]
    processed = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group, columns=columns).to_pandas()
        if len(chunk) % len(CANDIDATES) != 0:
            raise ValueError("row group breaks candidate blocks")
        block_rows = len(chunk) // len(CANDIDATES)
        positions, _ = row_positions_from_score_chunk(chunk, block_rows, base)
        if bool(covered[positions].any()) or len(np.unique(positions)) != block_rows:
            raise ValueError("score row coverage overlaps")
        folds = chunk["outer_fold"].to_numpy(np.int8).reshape(
            block_rows, len(CANDIDATES)
        )
        downstream = chunk["downstream_outer_fold"].to_numpy(np.int8).reshape(
            block_rows, len(CANDIDATES)
        )
        model_count = chunk["nested_model_count"].to_numpy(np.int8).reshape(
            block_rows, len(CANDIDATES)
        )
        contract_checks = {
            "outer_fold_constant": bool(np.all(folds == folds[:, :1])),
            "downstream_fold_constant": bool(
                np.all(downstream == downstream[:, :1])
            ),
            "outer_equals_downstream": bool(
                np.all(folds[:, 0] == downstream[:, 0])
            ),
            "outer_fold_in_range": bool(
                np.all((folds[:, 0] >= 0) & (folds[:, 0] < 5))
            ),
            "nested_model_count_is_4": bool(np.all(model_count == 4)),
        }
        if not all(contract_checks.values()):
            raise ValueError(
                "strict nested outer-valid contract failed: "
                f"row_group={row_group}, checks={contract_checks}"
            )
        values = chunk["candidate_tvt"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )
        scores = chunk["pred_abs_error"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )[:, :PRIMARY_COUNT]
        if not np.isfinite(values).all() or not np.isfinite(scores).all():
            raise ValueError("candidate score contains non-finite values")
        codes = np.argmin(scores, axis=1).astype(np.int8)
        first = scores[np.arange(block_rows), codes]
        second = np.partition(scores, 1, axis=1)[:, 1]
        selector_code[positions] = codes
        selector_tvt[positions] = values[np.arange(block_rows), codes]
        selector_margin[positions] = np.maximum(second - first, 0.0)
        covered[positions] = True
        processed += block_rows
        if row_group % 25 == 0 or row_group + 1 == parquet.num_row_groups:
            print(
                f"reconstructed selector {processed:,}/{EXPECTED_ROWS:,}",
                flush=True,
            )
    if processed != EXPECTED_ROWS or not covered.all():
        raise ValueError("selector surface coverage failed")
    observed = rmse(base["target"], selector_tvt)
    if not np.isclose(observed, EXPECTED_HARD_PRIMARY_RMSE, atol=2e-6):
        raise ValueError(
            f"hard selector RMSE {observed} != {EXPECTED_HARD_PRIMARY_RMSE}"
        )
    return selector_code, selector_tvt, selector_margin


def build_switch_and_runs(
    selector_code: np.ndarray,
    well_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    well_start_mask = np.empty(EXPECTED_ROWS, bool)
    well_start_mask[0] = True
    well_start_mask[1:] = well_code[1:] != well_code[:-1]
    switch = np.zeros(EXPECTED_ROWS, bool)
    switch[1:] = (~well_start_mask[1:]) & (
        selector_code[1:] != selector_code[:-1]
    )
    run_start_mask = well_start_mask | switch
    run_starts = np.flatnonzero(run_start_mask)
    run_ends = np.r_[run_starts[1:], EXPECTED_ROWS]
    run_lengths = (run_ends - run_starts).astype(np.int32)
    run_id = (np.cumsum(run_start_mask, dtype=np.int64) - 1).astype(np.int32)
    run_candidate = selector_code[run_starts]
    run_previous_candidate = np.full(len(run_starts), -1, np.int8)
    noninitial = ~well_start_mask[run_starts]
    run_previous_candidate[noninitial] = selector_code[run_starts[noninitial] - 1]
    previous_run_code_by_row = run_previous_candidate[run_id]

    nearest_switch_distance = np.full(EXPECTED_ROWS, np.iinfo(np.int32).max, np.int32)
    for code in np.unique(well_code):
        positions = np.flatnonzero(well_code == code)
        start, end = positions[0], positions[-1] + 1
        switch_positions = np.flatnonzero(switch[start:end])
        if not len(switch_positions):
            continue
        local = np.arange(end - start, dtype=np.int32)
        insertion = np.searchsorted(switch_positions, local)
        left = np.full(len(local), np.iinfo(np.int32).max, np.int32)
        right = np.full(len(local), np.iinfo(np.int32).max, np.int32)
        has_left = insertion > 0
        left[has_left] = local[has_left] - switch_positions[insertion[has_left] - 1]
        has_right = insertion < len(switch_positions)
        right[has_right] = switch_positions[insertion[has_right]] - local[has_right]
        nearest_switch_distance[start:end] = np.minimum(left, right)
    return (
        switch,
        nearest_switch_distance,
        run_id,
        run_starts,
        run_lengths,
        previous_run_code_by_row,
    )


def reconstruct_previous_run_candidate_sse(
    base: dict[str, Any],
    selector_tvt: np.ndarray,
    run_id: np.ndarray,
    previous_run_code_by_row: np.ndarray,
    n_runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_error = selector_tvt.astype(np.float64) - base["target"].astype(
        np.float64
    )
    selected_sse = np.bincount(
        run_id,
        weights=np.square(selected_error),
        minlength=n_runs,
    ).astype(np.float64)
    previous_sse = np.zeros(n_runs, np.float64)
    parquet = pq.ParquetFile(SCORE_PATH)
    columns = ["well", "well_row_idx", "candidate_id", "candidate_tvt"]
    processed = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group, columns=columns).to_pandas()
        block_rows = len(chunk) // len(CANDIDATES)
        positions, _ = row_positions_from_score_chunk(chunk, block_rows, base)
        values = chunk["candidate_tvt"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )
        previous_codes = previous_run_code_by_row[positions]
        valid = previous_codes >= 0
        if valid.any():
            valid_positions = positions[valid]
            valid_runs = run_id[valid_positions]
            previous_tvt = values[
                np.flatnonzero(valid), previous_codes[valid].astype(np.int16)
            ]
            target = base["target"][valid_positions].astype(np.float64)
            previous_error = previous_tvt.astype(np.float64) - target
            np.add.at(previous_sse, valid_runs, np.square(previous_error))
        processed += block_rows
        if row_group % 50 == 0 or row_group + 1 == parquet.num_row_groups:
            print(
                f"loaded previous-candidate counterfactual {processed:,}/{EXPECTED_ROWS:,}",
                flush=True,
            )
    if processed != EXPECTED_ROWS:
        raise ValueError("counterfactual score coverage failed")
    return selected_sse, previous_sse


def metrics_for_mask(
    mask: np.ndarray,
    target: np.ndarray,
    pred274: np.ndarray,
    pred264: np.ndarray,
    selector_tvt: np.ndarray,
    delta_sse_final: np.ndarray,
    positive_sse_scope: float,
    net_sse_scope: float,
    scope_rows: int,
    well_code: np.ndarray,
) -> dict[str, Any]:
    rows = int(mask.sum())
    if not rows:
        raise ValueError("empty switch window bucket")
    final_delta = float(delta_sse_final[mask].sum())
    positive_delta = float(np.maximum(delta_sse_final[mask], 0.0).sum())
    return {
        "rows": rows,
        "row_share_within_scope": rows / scope_rows,
        "wells": int(np.unique(well_code[mask]).size),
        "exp274_rmse": rmse(target[mask], pred274[mask]),
        "exp264_final_rmse": rmse(target[mask], pred264[mask]),
        "hard_primary_rmse": rmse(target[mask], selector_tvt[mask]),
        "final_mse_delta": final_delta / rows,
        "hard_minus_exp274_mse_delta": float(
            (
                np.square(selector_tvt[mask].astype(np.float64) - target[mask])
                - np.square(pred274[mask].astype(np.float64) - target[mask])
            ).sum()
            / rows
        ),
        "net_final_sse_delta": final_delta,
        "net_sse_share_within_scope": (
            final_delta / net_sse_scope if net_sse_scope != 0 else np.nan
        ),
        "positive_final_sse_delta": positive_delta,
        "positive_sse_share_within_scope": (
            positive_delta / positive_sse_scope if positive_sse_scope > 0 else np.nan
        ),
    }


def build_window_summary(
    base: dict[str, Any],
    well: pd.DataFrame,
    selector_tvt: np.ndarray,
    nearest_switch_distance: np.ndarray,
) -> pd.DataFrame:
    target = base["target"].astype(np.float64)
    pred274 = base["pred274"].astype(np.float64)
    pred264 = base["pred264"].astype(np.float64)
    delta_sse_final = np.square(pred264 - target) - np.square(pred274 - target)
    severe_by_well = well["delta_exp264_vs_exp274"].to_numpy(float) > 3.0
    severe_rows = severe_by_well[base["well_code"]]
    scopes = {
        "all": np.ones(EXPECTED_ROWS, bool),
        "worse_gt3": severe_rows,
        "other": ~severe_rows,
    }
    rows = []
    for scope_name, scope in scopes.items():
        positive_scope = float(np.maximum(delta_sse_final[scope], 0.0).sum())
        net_scope = float(delta_sse_final[scope].sum())
        scope_rows = int(scope.sum())
        for window in WINDOWS:
            within = nearest_switch_distance <= window
            for zone_name, zone in [("within", within), ("outside", ~within)]:
                mask = scope & zone
                row = {
                    "scope": scope_name,
                    "window_rows": window,
                    "zone": zone_name,
                }
                row.update(
                    metrics_for_mask(
                        mask,
                        target,
                        pred274,
                        pred264,
                        selector_tvt,
                        delta_sse_final,
                        positive_scope,
                        net_scope,
                        scope_rows,
                        base["well_code"],
                    )
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_run_frame(
    base: dict[str, Any],
    well: pd.DataFrame,
    selector_code: np.ndarray,
    selector_tvt: np.ndarray,
    run_id: np.ndarray,
    run_starts: np.ndarray,
    run_lengths: np.ndarray,
    previous_run_code_by_row: np.ndarray,
    selected_sse: np.ndarray,
    previous_sse: np.ndarray,
) -> pd.DataFrame:
    n_runs = len(run_starts)
    run_well_code = base["well_code"][run_starts]
    run_candidate_code = selector_code[run_starts]
    run_previous_code = previous_run_code_by_row[run_starts]
    target = base["target"].astype(np.float64)
    final_error = base["pred264"].astype(np.float64) - target
    exp274_error = base["pred274"].astype(np.float64) - target
    hard_error = selector_tvt.astype(np.float64) - target
    final_delta_sse = np.square(final_error) - np.square(exp274_error)
    hard_delta_sse = np.square(hard_error) - np.square(exp274_error)
    final_delta_by_run = np.bincount(
        run_id, weights=final_delta_sse, minlength=n_runs
    )
    hard_delta_by_run = np.bincount(
        run_id, weights=hard_delta_sse, minlength=n_runs
    )
    positive_final_by_run = np.bincount(
        run_id, weights=np.maximum(final_delta_sse, 0.0), minlength=n_runs
    )
    frame = pd.DataFrame(
        {
            "run_id": np.arange(n_runs, dtype=np.int32),
            "well": np.asarray(base["well_names"])[run_well_code],
            "run_start_global_row": run_starts,
            "run_start_well_row_idx": base["raw_row_idx"][run_starts],
            "run_rows": run_lengths,
            "previous_candidate_code": run_previous_code,
            "selected_candidate_code": run_candidate_code,
            "selected_candidate": np.asarray(CANDIDATES)[run_candidate_code],
            "final_vs_exp274_sse_delta": final_delta_by_run,
            "hard_vs_exp274_sse_delta": hard_delta_by_run,
            "positive_final_vs_exp274_sse_delta": positive_final_by_run,
            "selected_hard_sse": selected_sse,
            "previous_candidate_hold_sse": previous_sse,
        }
    )
    frame["previous_candidate"] = np.where(
        frame["previous_candidate_code"] >= 0,
        np.asarray(CANDIDATES)[np.maximum(frame["previous_candidate_code"], 0)],
        "INITIAL_RUN",
    )
    frame["is_initial_run"] = frame["previous_candidate_code"] < 0
    frame["selected_minus_previous_sse"] = (
        frame["selected_hard_sse"] - frame["previous_candidate_hold_sse"]
    )
    frame["selected_minus_previous_mse"] = (
        frame["selected_minus_previous_sse"] / frame["run_rows"]
    )
    frame["selected_hard_rmse"] = np.sqrt(
        frame["selected_hard_sse"] / frame["run_rows"]
    )
    frame["previous_candidate_hold_rmse"] = np.where(
        frame["is_initial_run"],
        np.nan,
        np.sqrt(frame["previous_candidate_hold_sse"] / frame["run_rows"]),
    )
    frame["hard_switch_harmful"] = (
        ~frame["is_initial_run"] & (frame["selected_minus_previous_sse"] > 0)
    )
    frame["final_vs_exp274_mse_delta"] = (
        frame["final_vs_exp274_sse_delta"] / frame["run_rows"]
    )
    frame["hard_vs_exp274_mse_delta"] = (
        frame["hard_vs_exp274_sse_delta"] / frame["run_rows"]
    )
    well_delta = well.set_index("well")["delta_exp264_vs_exp274"]
    frame["well_delta_exp264_vs_exp274"] = frame["well"].map(well_delta)
    frame["well_worse_gt3"] = frame["well_delta_exp264_vs_exp274"] > 3.0
    return frame


def build_run_category_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    frame = run_frame.copy()
    frame["run_category"] = np.select(
        [
            frame["is_initial_run"],
            frame["hard_switch_harmful"],
        ],
        ["initial_run", "switch_harmful_hard_path"],
        default="switch_helpful_hard_path",
    )
    scope_masks = {
        "all": np.ones(len(frame), bool),
        "worse_gt3": frame["well_worse_gt3"].to_numpy(bool),
        "other": ~frame["well_worse_gt3"].to_numpy(bool),
    }
    rows = []
    for scope_name, scope in scope_masks.items():
        positive_total = frame.loc[scope, "positive_final_vs_exp274_sse_delta"].sum()
        scope_rows = frame.loc[scope, "run_rows"].sum()
        for category, group in frame.loc[scope].groupby("run_category"):
            n_rows = int(group["run_rows"].sum())
            selected_sse = float(group["selected_hard_sse"].sum())
            previous_sse = float(group["previous_candidate_hold_sse"].sum())
            rows.append(
                {
                    "scope": scope_name,
                    "run_category": category,
                    "runs": len(group),
                    "rows": n_rows,
                    "row_share": n_rows / scope_rows,
                    "wells": group["well"].nunique(),
                    "final_vs_exp274_mse_delta": float(
                        group["final_vs_exp274_sse_delta"].sum() / n_rows
                    ),
                    "hard_vs_exp274_mse_delta": float(
                        group["hard_vs_exp274_sse_delta"].sum() / n_rows
                    ),
                    "positive_final_sse_share": float(
                        group["positive_final_vs_exp274_sse_delta"].sum()
                        / positive_total
                    ),
                    "selected_hard_rmse": float(np.sqrt(selected_sse / n_rows)),
                    "previous_candidate_hold_rmse": (
                        float(np.sqrt(previous_sse / n_rows))
                        if category != "initial_run"
                        else np.nan
                    ),
                    "selected_minus_previous_mse": (
                        float((selected_sse - previous_sse) / n_rows)
                        if category != "initial_run"
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_transition_summary(run_frame: pd.DataFrame) -> pd.DataFrame:
    switched = run_frame.loc[~run_frame["is_initial_run"]].copy()
    rows = []
    for (previous, selected), group in switched.groupby(
        ["previous_candidate", "selected_candidate"]
    ):
        n_rows = int(group["run_rows"].sum())
        selected_sse = float(group["selected_hard_sse"].sum())
        previous_sse = float(group["previous_candidate_hold_sse"].sum())
        rows.append(
            {
                "previous_candidate": previous,
                "selected_candidate": selected,
                "runs": len(group),
                "rows": n_rows,
                "wells": group["well"].nunique(),
                "worse_gt3_wells": group.loc[
                    group["well_worse_gt3"], "well"
                ].nunique(),
                "harmful_run_rate": float(group["hard_switch_harmful"].mean()),
                "selected_hard_rmse": float(np.sqrt(selected_sse / n_rows)),
                "previous_candidate_hold_rmse": float(
                    np.sqrt(previous_sse / n_rows)
                ),
                "selected_minus_previous_mse": float(
                    (selected_sse - previous_sse) / n_rows
                ),
                "final_vs_exp274_mse_delta": float(
                    group["final_vs_exp274_sse_delta"].sum() / n_rows
                ),
                "positive_final_sse_delta": float(
                    group["positive_final_vs_exp274_sse_delta"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["positive_final_sse_delta", "rows"], ascending=False
    )


def build_by_well_summary(
    base: dict[str, Any],
    well: pd.DataFrame,
    switch: np.ndarray,
    nearest_switch_distance: np.ndarray,
    run_frame: pd.DataFrame,
) -> pd.DataFrame:
    target = base["target"].astype(np.float64)
    delta_sse_final = (
        np.square(base["pred264"].astype(np.float64) - target)
        - np.square(base["pred274"].astype(np.float64) - target)
    )
    records = []
    for code, well_name in enumerate(base["well_names"]):
        mask = base["well_code"] == code
        within5 = mask & (nearest_switch_distance <= 5)
        outside5 = mask & (nearest_switch_distance > 5)
        positive_total = float(np.maximum(delta_sse_final[mask], 0.0).sum())
        runs = run_frame[
            run_frame["well"].eq(well_name) & ~run_frame["is_initial_run"]
        ]
        run_rows = int(runs["run_rows"].sum())
        selected_sse = float(runs["selected_hard_sse"].sum())
        previous_sse = float(runs["previous_candidate_hold_sse"].sum())
        records.append(
            {
                "well": well_name,
                "switches": int(switch[mask].sum()),
                "switches_per_1000": 1000.0 * switch[mask].mean(),
                "within5_row_share": float(within5.sum() / mask.sum()),
                "within5_final_mse_delta": float(
                    delta_sse_final[within5].mean() if within5.any() else np.nan
                ),
                "outside5_final_mse_delta": float(
                    delta_sse_final[outside5].mean() if outside5.any() else np.nan
                ),
                "within5_positive_sse_share": float(
                    np.maximum(delta_sse_final[within5], 0.0).sum()
                    / positive_total
                    if positive_total > 0
                    else np.nan
                ),
                "switch_runs": len(runs),
                "harmful_switch_runs": int(runs["hard_switch_harmful"].sum()),
                "harmful_switch_run_rate": float(
                    runs["hard_switch_harmful"].mean() if len(runs) else np.nan
                ),
                "switch_run_rows": run_rows,
                "selected_minus_previous_mse": float(
                    (selected_sse - previous_sse) / run_rows
                    if run_rows
                    else np.nan
                ),
                "selected_hard_switch_run_rmse": float(
                    np.sqrt(selected_sse / run_rows) if run_rows else np.nan
                ),
                "previous_candidate_hold_switch_run_rmse": float(
                    np.sqrt(previous_sse / run_rows) if run_rows else np.nan
                ),
            }
        )
    return well.merge(pd.DataFrame(records), on="well", validate="one_to_one")


def main() -> None:
    actual_sha = sha256_file(SCORE_PATH)
    if actual_sha != EXPECTED_SCORE_SHA256:
        raise ValueError(f"Stage C score SHA {actual_sha} != {EXPECTED_SCORE_SHA256}")
    well = pd.read_csv(WELL_PATH)
    if len(well) != EXPECTED_WELLS or well["well"].nunique() != EXPECTED_WELLS:
        raise ValueError("well comparison coverage changed")
    base = load_base_arrays(well)
    selector_code, selector_tvt, selector_margin = reconstruct_primary_surface(base)
    (
        switch,
        nearest_switch_distance,
        run_id,
        run_starts,
        run_lengths,
        previous_run_code_by_row,
    ) = build_switch_and_runs(selector_code, base["well_code"])
    selected_sse, previous_sse = reconstruct_previous_run_candidate_sse(
        base,
        selector_tvt,
        run_id,
        previous_run_code_by_row,
        len(run_starts),
    )
    run_frame = build_run_frame(
        base,
        well,
        selector_code,
        selector_tvt,
        run_id,
        run_starts,
        run_lengths,
        previous_run_code_by_row,
        selected_sse,
        previous_sse,
    )
    window_summary = build_window_summary(
        base, well, selector_tvt, nearest_switch_distance
    )
    run_category = build_run_category_summary(run_frame)
    transition = build_transition_summary(run_frame)
    by_well = build_by_well_summary(
        base, well, switch, nearest_switch_distance, run_frame
    )
    top_worse = by_well.sort_values(
        "delta_exp264_vs_exp274", ascending=False
    ).head(100)

    output_frames = {
        "selector_switch_window_summary.csv": window_summary,
        "selector_switch_run_counterfactual.csv": run_frame,
        "selector_switch_run_category_summary.csv": run_category,
        "selector_switch_transition_summary.csv": transition,
        "selector_switch_by_well.csv": by_well,
        "selector_switch_top100_worse_wells.csv": top_worse,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(ARTIFACT_DIR / filename, index=False)

    severe = by_well["delta_exp264_vs_exp274"] > 3.0
    rho, rho_p = spearmanr(
        by_well["switches_per_1000"], by_well["delta_exp264_vs_exp274"]
    )
    switched = run_frame[~run_frame["is_initial_run"]]
    selected_sse_total = float(switched["selected_hard_sse"].sum())
    previous_sse_total = float(switched["previous_candidate_hold_sse"].sum())
    switch_consistent = by_well[
        severe
        & (by_well["selected_minus_previous_mse"] > 0)
        & (
            by_well["within5_positive_sse_share"]
            > by_well["within5_row_share"]
        )
    ].sort_values("delta_exp264_vs_exp274", ascending=False)

    def category_record(scope: str, category: str) -> dict[str, Any]:
        row = run_category[
            run_category["scope"].eq(scope)
            & run_category["run_category"].eq(category)
        ].iloc[0]
        return {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.to_dict().items()
        }

    def window_record(scope: str, window: int, zone: str) -> dict[str, Any]:
        row = window_summary[
            window_summary["scope"].eq(scope)
            & window_summary["window_rows"].eq(window)
            & window_summary["zone"].eq(zone)
        ].iloc[0]
        return {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.to_dict().items()
        }

    summary = {
        "status": "complete_diagnostic_only",
        "rows": EXPECTED_ROWS,
        "wells": EXPECTED_WELLS,
        "question": (
            "Did within-well changes in the corrected Stage C v6 primary hard-top1 "
            "candidate account for Stage D v3 regression versus exp274?"
        ),
        "definition": {
            "switch": "primary pred_abs_error top1 differs from previous row within the same well",
            "run_counterfactual": (
                "evaluate the previous run's hard candidate over the full new run using actual TVT"
            ),
            "exp274_has_selector": False,
            "hard_top1_is_stage_d_final": False,
        },
        "global": {
            "switches": int(switch.sum()),
            "switches_per_1000_rows": float(1000.0 * switch.mean()),
            "selector_margin_mean": float(selector_margin.mean()),
            "hard_primary_rmse": rmse(base["target"], selector_tvt),
            "exp274_rmse": rmse(base["target"], base["pred274"]),
            "exp264_final_rmse": rmse(base["target"], base["pred264"]),
            "switch_rate_spearman_with_well_delta": float(rho),
            "switch_rate_spearman_p": float(rho_p),
        },
        "worse_gt3_vs_other": {
            "wells": int(severe.sum()),
            "switches_per_1000_median": float(
                by_well.loc[severe, "switches_per_1000"].median()
            ),
            "other_switches_per_1000_median": float(
                by_well.loc[~severe, "switches_per_1000"].median()
            ),
            "selected_minus_previous_mse_median": float(
                by_well.loc[severe, "selected_minus_previous_mse"].median()
            ),
            "other_selected_minus_previous_mse_median": float(
                by_well.loc[~severe, "selected_minus_previous_mse"].median()
            ),
            "wells_where_switch_runs_hurt_hard_path": int(
                (by_well.loc[severe, "selected_minus_previous_mse"] > 0).sum()
            ),
            "switch_consistent_well_criteria": (
                "selected_minus_previous_mse > 0 and "
                "within5_positive_sse_share > within5_row_share"
            ),
            "switch_consistent_wells": switch_consistent["well"].tolist(),
        },
        "switch_windows": {
            "all_exact_switch": window_record("all", 0, "within"),
            "all_within5": window_record("all", 5, "within"),
            "all_outside5": window_record("all", 5, "outside"),
            "worse_gt3_within5": window_record("worse_gt3", 5, "within"),
            "worse_gt3_outside5": window_record("worse_gt3", 5, "outside"),
        },
        "run_counterfactual": {
            "switch_runs": int(len(switched)),
            "harmful_switch_run_rate": float(switched["hard_switch_harmful"].mean()),
            "selected_hard_rmse": float(
                np.sqrt(selected_sse_total / switched["run_rows"].sum())
            ),
            "previous_candidate_hold_rmse": float(
                np.sqrt(previous_sse_total / switched["run_rows"].sum())
            ),
            "selected_minus_previous_mse": float(
                (selected_sse_total - previous_sse_total)
                / switched["run_rows"].sum()
            ),
            "all_harmful_switch_runs": category_record(
                "all", "switch_harmful_hard_path"
            ),
            "all_helpful_switch_runs": category_record(
                "all", "switch_helpful_hard_path"
            ),
            "worse_gt3_harmful_switch_runs": category_record(
                "worse_gt3", "switch_harmful_hard_path"
            ),
            "worse_gt3_helpful_switch_runs": category_record(
                "worse_gt3", "switch_helpful_hard_path"
            ),
        },
        "input_sha256": {
            "stage_c_outer_valid_candidate_score": actual_sha,
            "well_comparison": sha256_file(WELL_PATH),
            "exp274_oof_raw_gzip": sha256_file(EXP274_OOF_PATH),
            "exp264_stage_d_viewer": sha256_file(EXP264_VIEWER_PATH),
        },
        "output_sha256": {
            filename: sha256_file(ARTIFACT_DIR / filename)
            for filename in output_frames
        },
        "non_use_contract": [
            "actual-TVT run counterfactual is oracle attribution, not a deployable hold policy",
            "exp274 has no selector path, so no exp274-to-exp264 candidate-change claim is made",
            "Stage D final uses compact selector features and is not the hard top1 candidate path",
            "do not approve candidate removal, hard fallback, or switch suppression from this readout",
        ],
    }
    summary_path = ARTIFACT_DIR / "selector_switch_readout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nSwitch window summary")
    print(window_summary.to_string(index=False))
    print("\nRun category summary")
    print(run_category.to_string(index=False))
    print("\nTop transitions")
    print(transition.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
