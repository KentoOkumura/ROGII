# %% [markdown]
# # exp409 saved selector candidate-switch tail attribution on exp407
#
# 保存済みのcorrected exp264 Stage B v5とexp407 Stage B v1のcandidate-score
# OOFだけを使い、hard-selected candidateの遷移へtail悪化を帰属する。
# model、booster、prediction、inference、submissionは生成しない。

# %% [markdown]
# ## Contents
# 1. Imports and constants
# 2. Runtime and configuration helpers
# 3. Input resolution and SHA checks
# 4. Truth-free selection and transition freeze
# 5. Frozen-truth join and additive SSE attribution
# 6. Tail consistency gate and plots
# 7. Execution orchestration
# 8. Metrics and generated artifacts

# %%
from __future__ import annotations

import glob
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

EXPERIMENT_NAME = "exp409_saved_selector_candidate_switch_tail_attribution_on_exp407"
FREEZE_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "parent_selected_candidate",
    "exp407_selected_candidate",
    "parent_selected_tvt",
    "exp407_selected_tvt",
    "parent_selected_pred_abs_error",
    "exp407_selected_pred_abs_error",
    "switched",
    "transition_id",
    "distance_bucket",
    "verification_like_spatial_role",
    "verification_like_typewell_purged_role",
]
TRUTH_FREE_OOF_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "candidate_id",
    "candidate_tvt",
    "candidate_available",
    "pred_abs_error",
    "feature_schema_sha",
    "candidate_contract_sha",
    "model_fold",
]
TRUTH_OOF_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "candidate_id",
    "candidate_tvt",
    "actual_abs_error",
    "feature_schema_sha",
    "candidate_contract_sha",
    "model_fold",
]
FORBIDDEN_TRUTH_COLUMNS = {
    "actual_abs_error",
    "actual_within10",
    "TVT",
    "error",
    "oracle",
}
SUM_COLUMNS = [
    "rows",
    "switched_rows",
    "parent_sse",
    "exp407_sse",
    "delta_sse",
    "positive_excess_sse",
    "negative_excess_sse",
    "worse_rows",
    "parent_abs_error_sum",
    "exp407_abs_error_sum",
]


# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def locate_config() -> Path:
    cwd = Path.cwd()
    candidates = [
        cwd / "config.yaml",
        cwd / EXPERIMENT_NAME / "config.yaml",
        cwd / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"config.yaml for {EXPERIMENT_NAME} was not found from {cwd}"
    )


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = locate_config() if path is None else Path(path)
    payload = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(payload, dict):
        raise TypeError("config.yaml must contain a mapping")
    return payload


def output_root(config: Mapping[str, Any]) -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/artifacts")
    return Path("experiments") / str(config["experiment"]["name"]) / "artifacts"


def distance_bucket(values: Sequence[float]) -> np.ndarray:
    md = np.asarray(values, dtype=np.float64)
    if not np.isfinite(md).all() or (md < 0).any():
        raise ValueError("md_since must be finite and nonnegative")
    return np.select(
        [md <= 250.0, md <= 500.0, md <= 1000.0],
        ["near_0_250", "250_500", "500_1000"],
        default="1000_plus",
    ).astype(object)


def transition_id(parent: Sequence[str], exp407: Sequence[str]) -> np.ndarray:
    left = np.asarray(parent, dtype=str)
    right = np.asarray(exp407, dtype=str)
    return np.char.add(np.char.add(left, " -> "), right)


# %% [markdown]
# ## 3. Input resolution and SHA checks
#
# 各入力はconfigのpattern順に探索し、最初に見つかった単一fileのSHAが固定値と
# 一致する場合だけ使用する。candidate-score OOFは実行前に全file SHAを計算する。

# %%
def _pattern_matches(pattern: str) -> list[Path]:
    expanded = sorted(Path(item) for item in glob.glob(pattern, recursive=True))
    if expanded:
        return [path for path in expanded if path.is_file()]
    direct = Path(pattern)
    return [direct] if direct.is_file() else []


def resolve_required_file(
    patterns: Sequence[str],
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    found_wrong_sha: list[tuple[str, str]] = []
    for pattern in patterns:
        matches = _pattern_matches(str(pattern))
        if len(matches) > 1:
            raise RuntimeError(f"{label}: pattern is ambiguous: {pattern}: {matches}")
        if not matches:
            continue
        path = matches[0]
        actual = sha256_file(path)
        if actual != str(expected_sha256):
            found_wrong_sha.append((str(path), actual))
            continue
        return path
    if found_wrong_sha:
        raise RuntimeError(
            f"{label}: files were found but SHA mismatched: {found_wrong_sha}"
        )
    raise FileNotFoundError(f"{label}: no input matched patterns: {list(patterns)}")


def load_hidden_assignment(path: Path, expected_wells: int) -> pd.DataFrame:
    assignment = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    missing = required - set(assignment.columns)
    if missing:
        raise ValueError(f"hidden-like assignment is missing columns: {sorted(missing)}")
    if assignment["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate well_id")
    if len(assignment) != int(expected_wells):
        raise ValueError(
            f"hidden-like assignment wells {len(assignment)} != {expected_wells}"
        )
    allowed_spatial = {"train", "valid"}
    allowed_purged = {"train", "valid", "purged_train_excluded"}
    if not set(assignment["verification_like_spatial_role"]) <= allowed_spatial:
        raise ValueError("unexpected spatial hidden-like role")
    if not set(assignment["verification_like_typewell_purged_role"]) <= allowed_purged:
        raise ValueError("unexpected typewell-purged hidden-like role")
    return assignment[
        [
            "well_id",
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        ]
    ].copy()


def iter_parquet_batches(
    path: Path,
    columns: Sequence[str],
    *,
    batch_rows: int,
) -> Iterable[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for batch in parquet.iter_batches(
        batch_size=int(batch_rows),
        columns=list(columns),
        use_threads=False,
    ):
        yield batch.to_pandas()


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


# %% [markdown]
# ## 4. Truth-free selection and transition freeze
#
# このphaseでは`actual_abs_error`、`actual_within10`、TVT truth、oracleを
# 読まない。両surfaceのcandidate orderとtarget-free candidate valueを全件照合し、
# 同じ11候補domainで`pred_abs_error` argminを再現する。

# %%
def validate_candidate_long_layout(
    frame: pd.DataFrame,
    candidate_order: Sequence[str],
    *,
    source: str,
    allow_truth: bool,
) -> dict[str, np.ndarray]:
    if not allow_truth:
        forbidden = FORBIDDEN_TRUTH_COLUMNS & set(frame.columns)
        if forbidden:
            raise ValueError(
                f"{source}: truth columns are forbidden before freeze: {sorted(forbidden)}"
            )
    n_candidates = len(candidate_order)
    if len(frame) == 0 or len(frame) % n_candidates:
        raise ValueError(
            f"{source}: batch rows {len(frame)} are not complete {n_candidates}-candidate blocks"
        )
    n_base = len(frame) // n_candidates
    candidate_matrix = frame["candidate_id"].astype(str).to_numpy().reshape(
        n_base, n_candidates
    )
    expected = np.asarray(candidate_order, dtype=object)
    if not np.array_equal(candidate_matrix, np.broadcast_to(expected, candidate_matrix.shape)):
        raise ValueError(f"{source}: candidate order differs from frozen contract")
    matrices: dict[str, np.ndarray] = {"candidate_id": candidate_matrix}
    for column in ["id", "well", "well_row_idx", "outer_fold", "md_since"]:
        matrix = frame[column].to_numpy().reshape(n_base, n_candidates)
        if not np.all(matrix == matrix[:, :1]):
            raise ValueError(f"{source}: {column} differs inside a candidate block")
        matrices[column] = matrix
    if "model_fold" in frame.columns:
        model_fold = frame["model_fold"].to_numpy()
        if not np.array_equal(model_fold, frame["outer_fold"].to_numpy()):
            raise ValueError(f"{source}: model_fold does not match outer_fold")
    for column in ["feature_schema_sha", "candidate_contract_sha"]:
        if column in frame.columns:
            unique = frame[column].astype(str).unique()
            if len(unique) != 1:
                raise ValueError(f"{source}: {column} is not constant inside batch")
            matrices[column] = unique
    return matrices


def select_truth_free_batch(
    frame: pd.DataFrame,
    candidate_order: Sequence[str],
    selectable_candidates: Sequence[str],
    *,
    source: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    layout = validate_candidate_long_layout(
        frame, candidate_order, source=source, allow_truth=False
    )
    n_candidates = len(candidate_order)
    n_base = len(frame) // n_candidates
    index = {name: position for position, name in enumerate(candidate_order)}
    if len(index) != n_candidates:
        raise ValueError("candidate_order contains duplicates")
    if not set(selectable_candidates) <= set(candidate_order):
        raise ValueError("selectable domain contains an unknown candidate")
    selectable_positions = np.asarray(
        [index[str(name)] for name in selectable_candidates], dtype=np.int64
    )
    available = frame["candidate_available"].astype(bool).to_numpy().reshape(
        n_base, n_candidates
    )
    if not available[:, selectable_positions].all():
        raise ValueError(f"{source}: a selectable candidate is unavailable")
    predicted = frame["pred_abs_error"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError(f"{source}: pred_abs_error is nonfinite or negative")
    candidate_tvt = frame["candidate_tvt"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    if not np.isfinite(candidate_tvt).all():
        raise ValueError(f"{source}: candidate_tvt is nonfinite")
    selected_local = np.argmin(predicted[:, selectable_positions], axis=1)
    selected_position = selectable_positions[selected_local]
    rows = np.arange(n_base, dtype=np.int64)
    result = pd.DataFrame(
        {
            "id": layout["id"][:, 0].astype(str),
            "well": layout["well"][:, 0].astype(str),
            "well_row_idx": layout["well_row_idx"][:, 0].astype(np.int64),
            "outer_fold": layout["outer_fold"][:, 0].astype(np.int8),
            "md_since": layout["md_since"][:, 0].astype(np.float64),
            "selected_candidate": np.asarray(candidate_order, dtype=object)[
                selected_position
            ],
            "selected_tvt": candidate_tvt[rows, selected_position],
            "selected_pred_abs_error": predicted[rows, selected_position],
        }
    )
    return result, candidate_tvt


def build_selection_freeze_batch(
    parent_long: pd.DataFrame,
    exp407_long: pd.DataFrame,
    hidden_assignment: pd.DataFrame,
    candidate_order: Sequence[str],
    selectable_candidates: Sequence[str],
    *,
    expected_feature_schema_sha256: str | None = None,
    expected_candidate_contract_sha256: str | None = None,
) -> pd.DataFrame:
    parent, parent_values = select_truth_free_batch(
        parent_long,
        candidate_order,
        selectable_candidates,
        source="parent",
    )
    exp407, exp407_values = select_truth_free_batch(
        exp407_long,
        candidate_order,
        selectable_candidates,
        source="exp407",
    )
    key_columns = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
    if not parent[key_columns].equals(exp407[key_columns]):
        raise ValueError("parent and exp407 base key/order differ")
    if not np.allclose(parent_values, exp407_values, rtol=0.0, atol=1.0e-6):
        raise ValueError("parent and exp407 candidate values differ")
    for label, frame in [("parent", parent_long), ("exp407", exp407_long)]:
        if expected_feature_schema_sha256 is not None:
            values = set(frame["feature_schema_sha"].astype(str).unique())
            if values != {str(expected_feature_schema_sha256)}:
                raise ValueError(f"{label}: feature schema SHA mismatch: {values}")
        if expected_candidate_contract_sha256 is not None:
            values = set(frame["candidate_contract_sha"].astype(str).unique())
            if values != {str(expected_candidate_contract_sha256)}:
                raise ValueError(f"{label}: candidate contract SHA mismatch: {values}")
    freeze = parent[key_columns].copy()
    freeze["parent_selected_candidate"] = parent["selected_candidate"].astype(str)
    freeze["exp407_selected_candidate"] = exp407["selected_candidate"].astype(str)
    freeze["parent_selected_tvt"] = parent["selected_tvt"].to_numpy(np.float64)
    freeze["exp407_selected_tvt"] = exp407["selected_tvt"].to_numpy(np.float64)
    freeze["parent_selected_pred_abs_error"] = parent[
        "selected_pred_abs_error"
    ].to_numpy(np.float64)
    freeze["exp407_selected_pred_abs_error"] = exp407[
        "selected_pred_abs_error"
    ].to_numpy(np.float64)
    freeze["switched"] = (
        freeze["parent_selected_candidate"] != freeze["exp407_selected_candidate"]
    )
    freeze["transition_id"] = transition_id(
        freeze["parent_selected_candidate"], freeze["exp407_selected_candidate"]
    )
    freeze["distance_bucket"] = distance_bucket(freeze["md_since"])
    freeze = freeze.merge(
        hidden_assignment,
        left_on="well",
        right_on="well_id",
        how="left",
        validate="many_to_one",
    ).drop(columns=["well_id"])
    hidden_columns = [
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ]
    if freeze[hidden_columns].isna().any().any():
        raise ValueError("a freeze row lacks hidden-like assignment")
    return freeze[FREEZE_COLUMNS]


def write_selection_freeze(
    *,
    parent_path: Path,
    exp407_path: Path,
    hidden_assignment: pd.DataFrame,
    output_path: Path,
    candidate_order: Sequence[str],
    selectable_candidates: Sequence[str],
    batch_base_rows: int,
    expected_feature_schema_sha256: str,
    expected_candidate_contract_sha256: str,
    expected_base_rows: int,
    expected_long_rows: int,
) -> dict[str, Any]:
    parent_rows = parquet_row_count(parent_path)
    exp407_rows = parquet_row_count(exp407_path)
    if parent_rows != expected_long_rows or exp407_rows != expected_long_rows:
        raise ValueError(
            f"candidate-long row mismatch: parent={parent_rows}, exp407={exp407_rows}, "
            f"expected={expected_long_rows}"
        )
    n_candidates = len(candidate_order)
    batch_long_rows = int(batch_base_rows) * n_candidates
    parent_batches = iter_parquet_batches(
        parent_path, TRUTH_FREE_OOF_COLUMNS, batch_rows=batch_long_rows
    )
    exp407_batches = iter_parquet_batches(
        exp407_path, TRUTH_FREE_OOF_COLUMNS, batch_rows=batch_long_rows
    )
    writer: pq.ParquetWriter | None = None
    base_rows = 0
    transition_counts: list[pd.DataFrame] = []
    try:
        for batch_index, pair in enumerate(
            zip_longest(parent_batches, exp407_batches), start=1
        ):
            parent_long, exp407_long = pair
            if parent_long is None or exp407_long is None:
                raise ValueError("parent and exp407 parquet batch counts differ")
            if len(parent_long) != len(exp407_long):
                raise ValueError(
                    f"batch {batch_index}: parent and exp407 long row counts differ"
                )
            freeze = build_selection_freeze_batch(
                parent_long,
                exp407_long,
                hidden_assignment,
                candidate_order,
                selectable_candidates,
                expected_feature_schema_sha256=expected_feature_schema_sha256,
                expected_candidate_contract_sha256=expected_candidate_contract_sha256,
            )
            table = pa.Table.from_pandas(freeze, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            base_rows += len(freeze)
            transition_counts.append(
                freeze.groupby(
                    [
                        "parent_selected_candidate",
                        "exp407_selected_candidate",
                        "transition_id",
                    ],
                    as_index=False,
                    observed=True,
                ).size()
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("selection freeze received no rows")
    if base_rows != int(expected_base_rows):
        raise ValueError(f"freeze rows {base_rows} != expected {expected_base_rows}")
    inventory = (
        pd.concat(transition_counts, ignore_index=True)
        .groupby(
            [
                "parent_selected_candidate",
                "exp407_selected_candidate",
                "transition_id",
            ],
            as_index=False,
            observed=True,
        )["size"]
        .sum()
        .rename(columns={"size": "rows"})
        .sort_values(["rows", "transition_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {
        "status": "truth_free_selection_freeze_complete",
        "rows": int(base_rows),
        "transition_count": int(len(inventory)),
        "switched_rows": int(
            inventory.loc[
                inventory["parent_selected_candidate"]
                != inventory["exp407_selected_candidate"],
                "rows",
            ].sum()
        ),
        "selection_freeze_sha256": sha256_file(output_path),
        "truth_columns_read": [],
        "forbidden_truth_read_count": 0,
        "transition_inventory": inventory.to_dict(orient="records"),
    }


# %% [markdown]
# ## 5. Frozen-truth join and additive SSE attribution
#
# freeze fileがcloseされSHA固定された後だけ、両OOFの`actual_abs_error`を読む。
# 帰属の主量は加法的な`exp407 squared error - parent squared error`である。

# %%
def build_truth_attribution_batch(
    freeze: pd.DataFrame,
    parent_truth_long: pd.DataFrame,
    exp407_truth_long: pd.DataFrame,
    candidate_order: Sequence[str],
) -> pd.DataFrame:
    parent_layout = validate_candidate_long_layout(
        parent_truth_long,
        candidate_order,
        source="parent_truth",
        allow_truth=True,
    )
    exp407_layout = validate_candidate_long_layout(
        exp407_truth_long,
        candidate_order,
        source="exp407_truth",
        allow_truth=True,
    )
    n_candidates = len(candidate_order)
    n_base = len(parent_truth_long) // n_candidates
    if len(exp407_truth_long) != len(parent_truth_long) or len(freeze) != n_base:
        raise ValueError("freeze and truth batch row counts differ")
    key_columns = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
    parent_keys = pd.DataFrame(
        {
            column: parent_layout[column][:, 0]
            for column in key_columns
        }
    )
    parent_keys["id"] = parent_keys["id"].astype(str)
    parent_keys["well"] = parent_keys["well"].astype(str)
    exp407_keys = pd.DataFrame(
        {
            column: exp407_layout[column][:, 0]
            for column in key_columns
        }
    )
    exp407_keys["id"] = exp407_keys["id"].astype(str)
    exp407_keys["well"] = exp407_keys["well"].astype(str)
    frozen_keys = freeze[key_columns].reset_index(drop=True)
    comparisons = [
        (
            "id",
            np.array_equal(
                parent_keys["id"].astype(str).to_numpy(),
                exp407_keys["id"].astype(str).to_numpy(),
            )
            and np.array_equal(
                parent_keys["id"].astype(str).to_numpy(),
                frozen_keys["id"].astype(str).to_numpy(),
            ),
        ),
        (
            "well",
            np.array_equal(
                parent_keys["well"].astype(str).to_numpy(),
                exp407_keys["well"].astype(str).to_numpy(),
            )
            and np.array_equal(
                parent_keys["well"].astype(str).to_numpy(),
                frozen_keys["well"].astype(str).to_numpy(),
            ),
        ),
        *[
            (
                column,
                np.array_equal(
                    parent_keys[column].to_numpy(np.int64),
                    exp407_keys[column].to_numpy(np.int64),
                )
                and np.array_equal(
                    parent_keys[column].to_numpy(np.int64),
                    frozen_keys[column].to_numpy(np.int64),
                ),
            )
            for column in ["well_row_idx", "outer_fold"]
        ],
        (
            "md_since",
            np.allclose(
                parent_keys["md_since"].to_numpy(np.float64),
                exp407_keys["md_since"].to_numpy(np.float64),
                rtol=0.0,
                atol=0.0,
            )
            and np.allclose(
                parent_keys["md_since"].to_numpy(np.float64),
                frozen_keys["md_since"].to_numpy(np.float64),
                rtol=0.0,
                atol=0.0,
            ),
        ),
    ]
    failed_keys = [column for column, passed in comparisons if not passed]
    if failed_keys:
        raise ValueError(f"truth batch keys differ from frozen keys: {failed_keys}")
    parent_values = parent_truth_long["candidate_tvt"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    exp407_values = exp407_truth_long["candidate_tvt"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    if not np.allclose(parent_values, exp407_values, rtol=0.0, atol=1.0e-6):
        raise ValueError("candidate values differ during truth join")
    parent_actual = parent_truth_long["actual_abs_error"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    exp407_actual = exp407_truth_long["actual_abs_error"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    if not np.isfinite(parent_actual).all() or not np.isfinite(exp407_actual).all():
        raise ValueError("actual_abs_error is nonfinite")
    if not np.allclose(parent_actual, exp407_actual, rtol=0.0, atol=1.0e-5):
        raise ValueError("parent and exp407 actual_abs_error surfaces differ")
    positions = {name: index for index, name in enumerate(candidate_order)}
    try:
        parent_position = freeze["parent_selected_candidate"].map(positions).to_numpy(
            np.int64
        )
        exp407_position = freeze["exp407_selected_candidate"].map(positions).to_numpy(
            np.int64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("freeze contains an unknown selected candidate") from exc
    if (parent_position < 0).any() or (exp407_position < 0).any():
        raise ValueError("freeze contains an unknown selected candidate")
    rows = np.arange(n_base, dtype=np.int64)
    if not np.allclose(
        parent_values[rows, parent_position],
        freeze["parent_selected_tvt"].to_numpy(np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("parent selected candidate value differs from freeze")
    if not np.allclose(
        exp407_values[rows, exp407_position],
        freeze["exp407_selected_tvt"].to_numpy(np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("exp407 selected candidate value differs from freeze")
    parent_abs_error = parent_actual[rows, parent_position]
    exp407_abs_error = exp407_actual[rows, exp407_position]
    output = freeze.copy()
    output["parent_abs_error"] = parent_abs_error
    output["exp407_abs_error"] = exp407_abs_error
    output["parent_squared_error"] = np.square(parent_abs_error)
    output["exp407_squared_error"] = np.square(exp407_abs_error)
    output["delta_abs_error"] = exp407_abs_error - parent_abs_error
    output["delta_sse"] = (
        output["exp407_squared_error"] - output["parent_squared_error"]
    )
    output["exp407_worse"] = output["delta_sse"] > 0.0
    return output


def aggregate_attribution(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["rows"] = 1
    work["switched_rows"] = work["switched"].astype(np.int64)
    work["parent_sse"] = work["parent_squared_error"]
    work["exp407_sse"] = work["exp407_squared_error"]
    work["positive_excess_sse"] = work["delta_sse"].clip(lower=0.0)
    work["negative_excess_sse"] = work["delta_sse"].clip(upper=0.0)
    work["worse_rows"] = work["exp407_worse"].astype(np.int64)
    work["parent_abs_error_sum"] = work["parent_abs_error"]
    work["exp407_abs_error_sum"] = work["exp407_abs_error"]
    return (
        work.groupby(
            list(group_columns),
            as_index=False,
            observed=True,
            dropna=False,
        )[SUM_COLUMNS]
        .sum()
        .reset_index(drop=True)
    )


def combine_partial_aggregates(
    partials: Sequence[pd.DataFrame],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame(columns=[*group_columns, *SUM_COLUMNS])
    combined = (
        pd.concat(partials, ignore_index=True)
        .groupby(
            list(group_columns),
            as_index=False,
            observed=True,
            dropna=False,
        )[SUM_COLUMNS]
        .sum()
    )
    rows = combined["rows"].to_numpy(np.float64)
    combined["parent_rmse"] = np.sqrt(combined["parent_sse"] / rows)
    combined["exp407_rmse"] = np.sqrt(combined["exp407_sse"] / rows)
    combined["delta_rmse"] = combined["exp407_rmse"] - combined["parent_rmse"]
    combined["parent_mae"] = combined["parent_abs_error_sum"] / rows
    combined["exp407_mae"] = combined["exp407_abs_error_sum"] / rows
    combined["delta_mae"] = combined["exp407_mae"] - combined["parent_mae"]
    combined["worse_row_rate"] = combined["worse_rows"] / rows
    positive_total_columns = [
        column
        for column in group_columns
        if column
        not in {
            "parent_selected_candidate",
            "exp407_selected_candidate",
            "transition_id",
        }
    ]
    if positive_total_columns:
        denominator = combined.groupby(
            positive_total_columns,
            observed=True,
            dropna=False,
        )["positive_excess_sse"].transform("sum")
    else:
        denominator = pd.Series(
            float(combined["positive_excess_sse"].sum()), index=combined.index
        )
    combined["positive_excess_sse_share"] = np.divide(
        combined["positive_excess_sse"],
        denominator,
        out=np.zeros(len(combined), dtype=np.float64),
        where=denominator.to_numpy(np.float64) > 0.0,
    )
    return combined.sort_values(
        [*positive_total_columns, "delta_sse", "transition_id"],
        ascending=[*[True] * len(positive_total_columns), False, True],
    ).reset_index(drop=True)


def add_hidden_scope_rows(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    definitions = [
        ("hidden_like_spatial", "verification_like_spatial_role"),
        (
            "hidden_like_typewell_purged",
            "verification_like_typewell_purged_role",
        ),
    ]
    for scope, column in definitions:
        selected = frame.loc[frame[column].eq("valid")].copy()
        selected["scope"] = scope
        parts.append(selected)
    if not parts:
        return pd.DataFrame(columns=[*frame.columns, "scope"])
    return pd.concat(parts, ignore_index=True)


# %% [markdown]
# ## 6. Tail consistency gate and plots
#
# magnitude shareの閾値は設けない。各scope/foldでpositive excess SSEが最大の
# directed switchをcandidate-orderから生成したtransition IDで安定順位付けする。

# %%
def rank1_positive_transition_by_scope_fold(
    table: pd.DataFrame,
    scopes: Sequence[str],
    expected_folds: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for fold in expected_folds:
            selected = table.loc[
                table["scope"].eq(scope)
                & table["outer_fold"].eq(int(fold))
                & (
                    table["parent_selected_candidate"]
                    != table["exp407_selected_candidate"]
                )
                & (table["delta_sse"] > 0.0)
            ].sort_values(["delta_sse", "transition_id"], ascending=[False, True])
            if selected.empty:
                rows.append(
                    {
                        "scope": scope,
                        "outer_fold": int(fold),
                        "transition_id": None,
                        "delta_sse": 0.0,
                        "positive_excess_sse_share": 0.0,
                        "rows": 0,
                    }
                )
                continue
            top = selected.iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "outer_fold": int(fold),
                    "transition_id": str(top["transition_id"]),
                    "delta_sse": float(top["delta_sse"]),
                    "positive_excess_sse_share": float(
                        top["positive_excess_sse_share"]
                    ),
                    "rows": int(top["rows"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_tail_consistency_gate(
    distance_table: pd.DataFrame,
    hidden_table: pd.DataFrame,
    by_well_table: pd.DataFrame,
    *,
    expected_folds: Sequence[int],
    minimum_rank1_folds: int,
    preregistered_worst_well: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    distance_scope = distance_table.copy()
    distance_scope["scope"] = "distance_" + distance_scope[
        "distance_bucket"
    ].astype(str)
    scope_table = pd.concat(
        [
            distance_scope.drop(columns=["distance_bucket"]),
            hidden_table,
        ],
        ignore_index=True,
        sort=False,
    )
    required_scopes = [
        "distance_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    top = rank1_positive_transition_by_scope_fold(
        scope_table, required_scopes, expected_folds
    )
    counts = (
        top.dropna(subset=["transition_id"])
        .groupby(["transition_id", "scope"], as_index=False)
        .size()
        .pivot(index="transition_id", columns="scope", values="size")
        .fillna(0)
        .astype(int)
    )
    for scope in required_scopes:
        if scope not in counts.columns:
            counts[scope] = 0
    consistent = counts.loc[
        (counts[required_scopes] >= int(minimum_rank1_folds)).all(axis=1)
    ]
    consistent_transitions = sorted(consistent.index.astype(str).tolist())
    cause_transition = consistent_transitions[0] if consistent_transitions else None
    worst = (
        by_well_table.loc[
            by_well_table["well"].astype(str).eq(str(preregistered_worst_well))
            & (
                by_well_table["parent_selected_candidate"]
                != by_well_table["exp407_selected_candidate"]
            )
            & (by_well_table["delta_sse"] > 0.0)
        ]
        .groupby(
            [
                "parent_selected_candidate",
                "exp407_selected_candidate",
                "transition_id",
            ],
            as_index=False,
            observed=True,
        )[["rows", "delta_sse", "positive_excess_sse"]]
        .sum()
        .sort_values(["delta_sse", "transition_id"], ascending=[False, True])
    )
    worst_top = None if worst.empty else str(worst.iloc[0]["transition_id"])
    scope_fold_inventory_complete = bool(
        len(top) == len(required_scopes) * len(expected_folds)
        and top["transition_id"].notna().all()
    )
    same_worst = cause_transition is not None and cause_transition == worst_top
    supported = bool(
        scope_fold_inventory_complete
        and len(consistent_transitions) == 1
        and same_worst
    )
    count_records = (
        counts.reset_index()[["transition_id", *required_scopes]]
        .sort_values(required_scopes, ascending=False)
        .to_dict(orient="records")
    )
    gate = {
        "status": "tail_consistency_gate_complete",
        "required_scopes": required_scopes,
        "expected_folds": [int(value) for value in expected_folds],
        "minimum_rank1_folds_per_scope": int(minimum_rank1_folds),
        "magnitude_threshold": None,
        "preregistered_worst_well": str(preregistered_worst_well),
        "scope_fold_inventory_complete": scope_fold_inventory_complete,
        "consistent_transitions": consistent_transitions,
        "cause_transition": cause_transition,
        "worst_well_rank1_positive_transition": worst_top,
        "cause_matches_worst_well": bool(same_worst),
        "rank1_fold_counts": count_records,
        "passed": supported,
        "decision": (
            "candidate_switch_tail_cause_supported"
            if supported
            else "diffuse_or_nonreproducible_candidate_switch_cause"
        ),
        "exp407_status_change": "none_exp407_remains_scientific_fail_closed",
    }
    return gate, top


def save_top_transition_plot(overall: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = (
        overall.loc[
            overall["parent_selected_candidate"]
            != overall["exp407_selected_candidate"]
        ]
        .sort_values(["delta_sse", "transition_id"], ascending=[False, True])
        .head(20)
        .sort_values("delta_sse", ascending=True)
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    if selected.empty:
        axis.text(0.5, 0.5, "No switched transition", ha="center", va="center")
        axis.set_axis_off()
    else:
        colors = np.where(selected["delta_sse"] >= 0.0, "#c44e52", "#4c72b0")
        axis.barh(selected["transition_id"], selected["delta_sse"], color=colors)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("exp407 SSE - parent SSE (ft²)")
        axis.set_ylabel("parent -> exp407 selected candidate")
        axis.set_title("Top candidate-switch excess-SSE attribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# %% [markdown]
# ## 7. Execution orchestration
#
# `execution.run_approved=false`ではinputを読み始めず、implementation-onlyで停止する。
# Kaggle実行には親OOF private input作成と明示承認が必要。

# %%
def run_diagnostic(config: Mapping[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config["data"]
    validation_cfg = config["validation"]
    candidate_cfg = config["candidate_bank"]
    runtime_cfg = config["runtime"]
    parent_path = resolve_required_file(
        data_cfg["parent_candidate_score_oof"]["patterns"],
        data_cfg["parent_candidate_score_oof"]["sha256"],
        label="parent candidate-score OOF",
    )
    exp407_path = resolve_required_file(
        data_cfg["exp407_candidate_score_oof"]["patterns"],
        data_cfg["exp407_candidate_score_oof"]["sha256"],
        label="exp407 candidate-score OOF",
    )
    hidden_path = resolve_required_file(
        data_cfg["hidden_like_assignment"]["patterns"],
        data_cfg["hidden_like_assignment"]["sha256"],
        label="hidden-like assignment",
    )
    hidden_assignment = load_hidden_assignment(
        hidden_path, int(validation_cfg["expected_wells"])
    )
    candidate_order = list(candidate_cfg["candidate_order"])
    selectable = list(candidate_cfg["selectable_domain"])
    batch_base_rows = int(runtime_cfg["parquet_batch_base_rows"])
    expected_base_rows = int(validation_cfg["expected_base_rows"])
    expected_long_rows = int(validation_cfg["expected_candidate_long_rows"])

    freeze_path = artifacts_dir / "selection_freeze.parquet"
    freeze_manifest = write_selection_freeze(
        parent_path=parent_path,
        exp407_path=exp407_path,
        hidden_assignment=hidden_assignment,
        output_path=freeze_path,
        candidate_order=candidate_order,
        selectable_candidates=selectable,
        batch_base_rows=batch_base_rows,
        expected_feature_schema_sha256=data_cfg["expected_feature_schema_sha256"],
        expected_candidate_contract_sha256=data_cfg[
            "expected_candidate_contract_logical_sha256"
        ],
        expected_base_rows=expected_base_rows,
        expected_long_rows=expected_long_rows,
    )
    freeze_manifest.update(
        {
            "parent_candidate_score_oof": {
                "path": str(parent_path),
                "sha256": sha256_file(parent_path),
            },
            "exp407_candidate_score_oof": {
                "path": str(exp407_path),
                "sha256": sha256_file(exp407_path),
            },
            "hidden_like_assignment": {
                "path": str(hidden_path),
                "sha256": sha256_file(hidden_path),
            },
            "candidate_order": candidate_order,
            "selectable_domain": selectable,
        }
    )
    freeze_manifest_path = artifacts_dir / "selection_freeze_manifest.json"
    write_json(freeze_manifest_path, freeze_manifest)
    freeze_manifest_sha = sha256_file(freeze_manifest_path)

    row_output_path = artifacts_dir / "selector_transition_row_attribution.parquet"
    freeze_batches = iter_parquet_batches(
        freeze_path, FREEZE_COLUMNS, batch_rows=batch_base_rows
    )
    batch_long_rows = batch_base_rows * len(candidate_order)
    parent_truth_batches = iter_parquet_batches(
        parent_path, TRUTH_OOF_COLUMNS, batch_rows=batch_long_rows
    )
    exp407_truth_batches = iter_parquet_batches(
        exp407_path, TRUTH_OOF_COLUMNS, batch_rows=batch_long_rows
    )
    writer: pq.ParquetWriter | None = None
    attributed_rows = 0
    partial_overall: list[pd.DataFrame] = []
    partial_fold: list[pd.DataFrame] = []
    partial_distance: list[pd.DataFrame] = []
    partial_hidden: list[pd.DataFrame] = []
    partial_well: list[pd.DataFrame] = []
    transition_columns = [
        "parent_selected_candidate",
        "exp407_selected_candidate",
        "transition_id",
    ]
    try:
        for batch_index, triple in enumerate(
            zip_longest(
                freeze_batches,
                parent_truth_batches,
                exp407_truth_batches,
            ),
            start=1,
        ):
            freeze, parent_truth, exp407_truth = triple
            if freeze is None or parent_truth is None or exp407_truth is None:
                raise ValueError("freeze and truth parquet batch counts differ")
            attribution = build_truth_attribution_batch(
                freeze, parent_truth, exp407_truth, candidate_order
            )
            table = pa.Table.from_pandas(attribution, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    row_output_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
            attributed_rows += len(attribution)
            partial_overall.append(
                aggregate_attribution(attribution, transition_columns)
            )
            partial_fold.append(
                aggregate_attribution(
                    attribution, ["outer_fold", *transition_columns]
                )
            )
            partial_distance.append(
                aggregate_attribution(
                    attribution,
                    ["distance_bucket", "outer_fold", *transition_columns],
                )
            )
            hidden = add_hidden_scope_rows(attribution)
            partial_hidden.append(
                aggregate_attribution(
                    hidden,
                    ["scope", "outer_fold", *transition_columns],
                )
            )
            partial_well.append(
                aggregate_attribution(
                    attribution,
                    ["well", "outer_fold", *transition_columns],
                )
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("truth attribution received no rows")
    if attributed_rows != expected_base_rows:
        raise ValueError(
            f"attribution rows {attributed_rows} != expected {expected_base_rows}"
        )

    overall = combine_partial_aggregates(partial_overall, transition_columns)
    by_fold = combine_partial_aggregates(
        partial_fold, ["outer_fold", *transition_columns]
    )
    by_distance = combine_partial_aggregates(
        partial_distance,
        ["distance_bucket", "outer_fold", *transition_columns],
    )
    hidden_like = combine_partial_aggregates(
        partial_hidden, ["scope", "outer_fold", *transition_columns]
    )
    by_well = combine_partial_aggregates(
        partial_well, ["well", "outer_fold", *transition_columns]
    )
    tables = {
        "transition_attribution_overall.csv": overall,
        "transition_attribution_by_fold.csv": by_fold,
        "transition_attribution_by_distance.csv": by_distance,
        "transition_attribution_hidden_like.csv": hidden_like,
        "transition_attribution_by_well.csv": by_well,
    }
    for filename, table in tables.items():
        table.to_csv(artifacts_dir / filename, index=False)

    concentration = validation_cfg["concentration_gate"]
    gate, top = evaluate_tail_consistency_gate(
        by_distance,
        hidden_like,
        by_well,
        expected_folds=validation_cfg["expected_folds"],
        minimum_rank1_folds=int(
            concentration["minimum_rank1_folds_per_scope"]
        ),
        preregistered_worst_well=str(
            validation_cfg["preregistered_worst_well"]
        ),
    )
    top_path = artifacts_dir / "tail_top_transition_by_scope_fold.csv"
    top.to_csv(top_path, index=False)
    gate.update(
        {
            "selection_freeze_sha256": sha256_file(freeze_path),
            "selection_freeze_manifest_sha256": freeze_manifest_sha,
            "row_attribution_sha256": sha256_file(row_output_path),
            "rows": int(attributed_rows),
        }
    )
    gate_path = artifacts_dir / "tail_consistency_gate.json"
    write_json(gate_path, gate)
    plot_path = artifacts_dir / "transition_delta_sse_top20.png"
    save_top_transition_plot(overall, plot_path)

    truth_ledger = pd.DataFrame(
        [
            {
                "phase": "selection_freeze",
                "truth_columns_read": "",
                "actual_abs_error_read": False,
                "selection_or_scope_mutation_allowed": True,
                "freeze_sha_required_before_phase": False,
            },
            {
                "phase": "frozen_truth_attribution",
                "truth_columns_read": "actual_abs_error",
                "actual_abs_error_read": True,
                "selection_or_scope_mutation_allowed": False,
                "freeze_sha_required_before_phase": True,
            },
        ]
    )
    truth_ledger_path = artifacts_dir / "truth_read_ledger.csv"
    truth_ledger.to_csv(truth_ledger_path, index=False)

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(artifacts_dir.iterdir())
        if path.is_file() and path.name != "reproducibility_manifest.json"
    }
    reproducibility = {
        "status": "saved_oof_candidate_switch_tail_attribution_complete",
        "experiment": config["experiment"]["name"],
        "rng": "none",
        "models": 0,
        "boosters": 0,
        "predictions_generated": 0,
        "parent_candidate_score_oof_sha256": sha256_file(parent_path),
        "exp407_candidate_score_oof_sha256": sha256_file(exp407_path),
        "hidden_like_assignment_sha256": sha256_file(hidden_path),
        "selection_freeze_sha256": sha256_file(freeze_path),
        "selection_freeze_manifest_sha256": freeze_manifest_sha,
        "truth_read_ledger_sha256": sha256_file(truth_ledger_path),
        "row_attribution_sha256": sha256_file(row_output_path),
        "gate_sha256": sha256_file(gate_path),
        "artifact_sha256": artifact_hashes,
        "deterministic_anchor": False,
        "deterministic_anchor_reason": (
            "Kaggle input source and kernel version are not recorded until an approved run."
        ),
    }
    write_json(
        artifacts_dir / "reproducibility_manifest.json", reproducibility
    )
    return gate


# %% [markdown]
# ## 8. Metrics and generated artifacts
#
# 実行時はfreeze、truth ledger、row attribution、5種の集計表、4-of-5 gate、
# 上位transition plot、reproducibility manifestを保存する。

# %%
def main() -> None:
    config = load_config()
    execution = config["execution"]
    print(
        json.dumps(
            {
                "experiment": config["experiment"]["name"],
                "route": config["experiment"]["route"],
                "scope": config["implementation"]["scope"],
                "run_approved": bool(execution["run_approved"]),
                "models": int(execution["models"]),
                "boosters": int(execution["boosters"]),
                "predictions_generated": int(
                    execution["prediction_rows_generated"]
                ),
                "parent_oof_private_input_created": bool(
                    execution["private_parent_oof_input_created"]
                ),
            },
            indent=2,
        )
    )
    if not bool(execution["run_approved"]):
        print(
            "Implementation-only guard: no input is read and no diagnostic is run. "
            "Create the approved private parent-OOF input and obtain explicit Kaggle "
            "execution approval before opening execution.run_approved."
        )
        return
    if not bool(execution["private_parent_oof_input_created"]):
        raise RuntimeError(
            "execution is fail-closed until the approved private parent OOF input exists"
        )
    if any(
        int(execution[key]) != 0
        for key in [
            "variants",
            "models",
            "folds_for_fitting",
            "boosters",
            "prediction_rows_generated",
            "control_retraining",
            "pf_runs",
            "hmm_runs",
            "beam_runs",
            "inference_runs",
            "submissions",
        ]
    ):
        raise RuntimeError("exp409 execution contract requires every generation count to be zero")
    gate = run_diagnostic(config, output_root(config))
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
