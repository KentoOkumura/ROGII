# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp415 fold-safe RMSE-prior bounded nudge on exp264 — train/readout
#
# corrected exp264の保存済みcandidate-score OOFと、各outer modelのexact fit rowsから
# 計算済みの候補RMSEだけを使うzero-booster confirmation readout。
#
# 候補RMSEをtask weightへ変換せず、親scoreへadditive priorとして加える。
# prior候補方向へのTVT補正は各行±0.25 ftに制限し、任意scopeのRMSE悪化を
# 数学的に最大0.25 ftへ制限する。

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment boundary
# 2. Runtime, paths, serialization, and Parquet helpers
# 3. Candidate RMSE and bounded-nudge policy helpers
# 4. Two-phase leakage and metric helpers
# 5. Setup, config, inputs, and compute contract
# 6. Phase 1 — truth-free policy freeze
# 7. Phase 2 — evaluation and risk certificate
# 8. Gate, metrics, diagnostics, and generated artifacts

# %% [markdown]
# ## 1. Imports and immutable experiment boundary

# %%
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

EXPERIMENT_NAME = "exp415_fold_safe_rmse_prior_bounded_nudge_on_exp264"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TRUTH_FREE_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "candidate_id",
    "candidate_tvt",
    "pred_abs_error",
    "feature_schema_sha",
    "candidate_contract_sha",
    "model_fold",
]
EVALUATION_COLUMNS = [
    "id",
    "well",
    "well_row_idx",
    "outer_fold",
    "md_since",
    "candidate_id",
    "candidate_tvt",
    "actual_abs_error",
]
KEY_COLUMNS = ["id", "well", "well_row_idx", "outer_fold"]
DISTANCE_BUCKETS = (
    ("near_0_250", -np.inf, 250.0),
    ("250_500", 250.0, 500.0),
    ("500_1000", 500.0, 1000.0),
    ("1000_plus", 1000.0, np.inf),
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP415_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)
if EXECUTE_NOTEBOOK:
    import matplotlib.pyplot as plt
    from IPython.display import display

# %% [markdown]
# ## 2. Runtime, paths, serialization, and Parquet helpers

# %%
def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(dict(value)), indent=2, ensure_ascii=False) + "\n"
    )


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.select_dtypes(include=["string"]).columns:
        normalized[column] = normalized[column].astype(object)
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode())
    digest.update("|".join(str(dtype) for dtype in normalized.dtypes).encode())
    hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=True
    ).to_numpy(np.uint64)
    digest.update(hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


class IncrementalParquetWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0
        self.row_groups = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError(f"cannot write an empty Parquet batch: {self.path}")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.path, table.schema, compression="zstd"
            )
        self.writer.write_table(table)
        self.rows += len(frame)
        self.row_groups += 1

    def close(self) -> None:
        if self.writer is None:
            raise ValueError(f"no Parquet rows were written: {self.path}")
        self.writer.close()
        self.writer = None


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def runtime_output_dir() -> Path:
    if EXECUTE_NOTEBOOK and KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "artifacts"
    return experiment_dir() / "artifacts"


def find_support_file(filename: str) -> Path:
    for candidate in (
        experiment_dir() / filename,
        Path.cwd() / filename,
        KAGGLE_WORKING_ROOT / filename,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"support file not found: {filename}")


def search_roots() -> list[Path]:
    return [
        Path.cwd(),
        project_root(),
        KAGGLE_INPUT_ROOT,
        Path("/tmp/kaggle-output"),
    ]


def resolve_pinned_file(
    patterns: Sequence[str],
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    matches: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        direct = Path(pattern)
        if direct.is_file() and direct.stat().st_size > 0:
            matches.add(direct)
        if direct.is_absolute():
            matches.update(
                Path(item)
                for item in glob.glob(pattern, recursive=True)
                if Path(item).is_file() and Path(item).stat().st_size > 0
            )
            continue
        for root in search_roots():
            if not root.exists():
                continue
            matches.update(
                path
                for path in root.glob(pattern)
                if path.is_file() and path.stat().st_size > 0
            )
    valid = sorted(
        path for path in matches if sha256_file(path) == str(expected_sha256)
    )
    if not valid:
        observed = {str(path): sha256_file(path) for path in sorted(matches)}
        raise FileNotFoundError(
            f"{label} did not resolve with pinned SHA {expected_sha256}; "
            f"observed={observed}"
        )
    return valid[0]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value

# %% [markdown]
# ## 3. Candidate RMSE and bounded-nudge policy helpers
#
# policy入力はtarget-free列とfit-partition RMSEだけ。
# `actual_abs_error`を渡せない関数境界にして、selection leakageを防ぐ。

# %%
def validate_static_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("experiment name changed")
    if config["experiment"]["route"] != "ensemble":
        raise ValueError("exp415 must remain on the ensemble route")
    candidate_order = [str(item) for item in config["candidate_bank"]["order"]]
    primary_domain = [
        str(item) for item in config["candidate_bank"]["primary_domain"]
    ]
    if len(candidate_order) != 12 or len(set(candidate_order)) != 12:
        raise ValueError("candidate order must contain 12 unique candidates")
    if primary_domain != candidate_order[:11]:
        raise ValueError("primary domain must be the first 11 frozen candidates")
    if config["candidate_bank"]["excluded_fixed_fallback"] != candidate_order[-1]:
        raise ValueError("fixed fallback exclusion changed")

    policy = config["policy"]
    expected_policy = {
        "candidate_rmse_coefficient": 1.0,
        "blend_parent_weight": 0.5,
        "blend_prior_weight": 0.5,
        "max_abs_correction_ft": 0.25,
    }
    for key, expected in expected_policy.items():
        if float(policy[key]) != expected:
            raise ValueError(f"fixed policy value changed: {key}")
    if [float(item) for item in policy["correction_clip_ft"]] != [-0.25, 0.25]:
        raise ValueError("correction clip changed")

    execution = config["execution"]
    cost = {
        key: int(execution[key])
        for key in (
            "active_variants",
            "models",
            "model_configs",
            "folds_for_fit",
            "boosters",
            "control_retraining",
            "pf_runs",
            "hmm_runs",
            "beam_runs",
            "gpu_runs",
            "inference_runs",
            "submissions",
        )
    }
    expected_cost = {
        "active_variants": 1,
        "models": 0,
        "model_configs": 0,
        "folds_for_fit": 0,
        "boosters": 0,
        "control_retraining": 0,
        "pf_runs": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
        "inference_runs": 0,
        "submissions": 0,
    }
    if cost != expected_cost:
        raise ValueError(f"zero-booster execution contract changed: {cost}")
    return {
        "candidate_order": candidate_order,
        "primary_domain": primary_domain,
        "cost": cost,
        "run_approved": bool(execution["run_approved"]),
    }


def load_candidate_rmse_matrix(
    path: Path,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    table = pd.read_csv(path)
    candidates = [str(item) for item in config["candidate_bank"]["order"]]
    expected_rows = int(config["data"]["candidate_rmse_table"]["expected_rows"])
    if len(table) != expected_rows:
        raise ValueError("candidate RMSE table row count changed")
    required = {
        "fit_partition",
        "candidate_position",
        "candidate_id",
        "fit_candidate_rmse",
        "fit_row_count",
        "fit_long_row_count",
        "fit_row_id_content_sha256",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"candidate RMSE table missing columns: {missing}")
    ordered = table.sort_values(
        ["fit_partition", "candidate_position"], kind="stable"
    ).reset_index(drop=True)
    expected_ids = candidates * 5
    if ordered["candidate_id"].astype(str).tolist() != expected_ids:
        raise ValueError("candidate RMSE table order changed")
    if ordered["fit_partition"].astype(int).tolist() != np.repeat(
        np.arange(5), 12
    ).tolist():
        raise ValueError("candidate RMSE fold inventory changed")
    if ordered["candidate_position"].astype(int).tolist() != np.tile(
        np.arange(12), 5
    ).tolist():
        raise ValueError("candidate RMSE position inventory changed")
    rmse = pd.to_numeric(
        ordered["fit_candidate_rmse"], errors="raise"
    ).to_numpy(np.float64)
    if not np.isfinite(rmse).all() or np.any(rmse <= 0):
        raise ValueError("candidate RMSE values are invalid")
    expected_fit_rows = int(
        config["data"]["candidate_rmse_table"][
            "expected_fit_base_rows_per_fold"
        ]
    )
    if not ordered["fit_row_count"].astype(int).eq(expected_fit_rows).all():
        raise ValueError("candidate RMSE fit row count changed")
    if not ordered["fit_long_row_count"].astype(int).eq(
        expected_fit_rows * 12
    ).all():
        raise ValueError("candidate RMSE fit long-row count changed")
    if (
        ordered.groupby("fit_partition")["fit_row_id_content_sha256"].nunique()
        != 1
    ).any():
        raise ValueError("fit row-ID SHA differs within an outer partition")
    matrix = rmse.reshape(5, 12)
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(ordered),
        "logical_sha256": logical_frame_sha256(ordered),
        "folds": sorted(ordered["fit_partition"].astype(int).unique().tolist()),
        "candidate_count": ordered["candidate_id"].nunique(),
        "fit_row_id_sha_by_fold": {
            str(int(fold)): str(group["fit_row_id_content_sha256"].iloc[0])
            for fold, group in ordered.groupby("fit_partition", sort=True)
        },
        "policy_columns_used": ["fit_candidate_rmse"],
        "weight_columns_used": [],
    }
    return matrix, ordered, audit


def validate_candidate_long_layout(
    frame: pd.DataFrame,
    candidate_order: Sequence[str],
    *,
    source: str,
) -> dict[str, np.ndarray]:
    candidates = [str(item) for item in candidate_order]
    n_candidates = len(candidates)
    if frame.empty or len(frame) % n_candidates:
        raise ValueError(f"{source}: incomplete candidate-long blocks")
    n_base = len(frame) // n_candidates
    candidate_matrix = frame["candidate_id"].astype(str).to_numpy().reshape(
        n_base, n_candidates
    )
    if not np.all(candidate_matrix == np.asarray(candidates)[None, :]):
        raise ValueError(f"{source}: candidate order changed")
    output: dict[str, np.ndarray] = {}
    for column in KEY_COLUMNS + ["md_since"]:
        matrix = frame[column].to_numpy().reshape(n_base, n_candidates)
        if not np.all(matrix == matrix[:, :1]):
            raise ValueError(f"{source}: {column} changes within a base row")
        output[column] = matrix[:, 0]
    if "model_fold" in frame:
        model_fold = frame["model_fold"].to_numpy().reshape(n_base, n_candidates)
        if not np.all(model_fold == model_fold[:, :1]):
            raise ValueError(f"{source}: model_fold changes within a base row")
        if not np.array_equal(
            model_fold[:, 0].astype(np.int64),
            output["outer_fold"].astype(np.int64),
        ):
            raise ValueError(f"{source}: model_fold differs from outer_fold")
        output["model_fold"] = model_fold[:, 0]
    output["candidate_id"] = candidate_matrix
    return output


def bounded_rmse_prior_policy(
    parent_score: np.ndarray,
    candidate_tvt: np.ndarray,
    outer_fold: np.ndarray,
    rmse_matrix: np.ndarray,
    candidate_order: Sequence[str],
    primary_domain: Sequence[str],
    *,
    rmse_coefficient: float = 1.0,
    blend_prior_weight: float = 0.5,
    max_abs_correction_ft: float = 0.25,
) -> dict[str, np.ndarray]:
    score = np.asarray(parent_score, dtype=np.float64)
    tvt = np.asarray(candidate_tvt, dtype=np.float64)
    folds = np.asarray(outer_fold, dtype=np.int64)
    rmse = np.asarray(rmse_matrix, dtype=np.float64)
    candidates = [str(item) for item in candidate_order]
    primary = [str(item) for item in primary_domain]
    if score.shape != tvt.shape or score.ndim != 2:
        raise ValueError("score and candidate TVT matrices must align")
    if score.shape[1] != len(candidates):
        raise ValueError("candidate matrix width changed")
    if folds.shape != (len(score),):
        raise ValueError("outer fold vector does not match the base rows")
    if rmse.ndim != 2 or rmse.shape[1] != len(candidates):
        raise ValueError("candidate RMSE matrix shape changed")
    if not primary or len(primary) != len(set(primary)):
        raise ValueError("primary candidate domain must be non-empty and unique")
    if any(name not in candidates for name in primary):
        raise ValueError("primary candidate domain contains an unknown candidate")
    if (
        not np.isfinite(score).all()
        or not np.isfinite(tvt).all()
        or not np.isfinite(rmse).all()
        or np.any(rmse <= 0)
    ):
        raise ValueError("bounded-nudge inputs contain invalid values")
    if np.any((folds < 0) | (folds >= rmse.shape[0])):
        raise ValueError("outer fold is outside the RMSE table")
    if not math.isfinite(rmse_coefficient) or rmse_coefficient < 0:
        raise ValueError("candidate RMSE coefficient must be finite and non-negative")
    if not math.isfinite(blend_prior_weight) or not 0 <= blend_prior_weight <= 1:
        raise ValueError("prior blend weight must be between zero and one")
    if not math.isfinite(max_abs_correction_ft) or max_abs_correction_ft <= 0:
        raise ValueError("correction cap must be finite and positive")
    positions = [candidates.index(name) for name in primary]
    parent_local = np.argmin(score[:, positions], axis=1)
    prior_score = (
        score[:, positions]
        + float(rmse_coefficient) * rmse[folds][:, positions]
    )
    prior_local = np.argmin(prior_score, axis=1)
    primary_positions = np.asarray(positions, dtype=np.int64)
    parent_position = primary_positions[parent_local]
    prior_position = primary_positions[prior_local]
    rows = np.arange(len(score))
    parent_tvt = tvt[rows, parent_position]
    prior_tvt = tvt[rows, prior_position]
    raw_nudge = float(blend_prior_weight) * (prior_tvt - parent_tvt)
    correction = np.clip(
        raw_nudge,
        -float(max_abs_correction_ft),
        float(max_abs_correction_ft),
    )
    prediction = parent_tvt + correction
    if not np.isfinite(prediction).all():
        raise ValueError("bounded RMSE-prior prediction is non-finite")
    if float(np.max(np.abs(correction))) > float(max_abs_correction_ft) + 1e-12:
        raise AssertionError("bounded correction exceeds its risk budget")
    return {
        "parent_position": parent_position,
        "prior_position": prior_position,
        "parent_tvt": parent_tvt,
        "prior_tvt": prior_tvt,
        "parent_score": score[rows, parent_position],
        "prior_parent_score": score[rows, prior_position],
        "prior_adjusted_score": prior_score[rows, prior_local],
        "raw_nudge": raw_nudge,
        "correction": correction,
        "prediction": prediction,
    }


def truth_free_policy_batch(
    frame: pd.DataFrame,
    rmse_matrix: np.ndarray,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    forbidden = set(config["validation"]["truth_free_phase"]["forbidden_columns"])
    present_forbidden = sorted(forbidden.intersection(frame.columns))
    if present_forbidden:
        raise ValueError(
            f"truth-free policy received forbidden columns: {present_forbidden}"
        )
    candidate_order = [str(item) for item in config["candidate_bank"]["order"]]
    layout = validate_candidate_long_layout(
        frame, candidate_order, source="truth_free_parent_oof"
    )
    n_candidates = len(candidate_order)
    n_base = len(frame) // n_candidates
    candidate_tvt = frame["candidate_tvt"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    parent_score = frame["pred_abs_error"].to_numpy(np.float64).reshape(
        n_base, n_candidates
    )
    folds = layout["outer_fold"].astype(np.int64)
    policy_cfg = config["policy"]
    result = bounded_rmse_prior_policy(
        parent_score,
        candidate_tvt,
        folds,
        rmse_matrix,
        candidate_order,
        config["candidate_bank"]["primary_domain"],
        rmse_coefficient=float(policy_cfg["candidate_rmse_coefficient"]),
        blend_prior_weight=float(policy_cfg["blend_prior_weight"]),
        max_abs_correction_ft=float(policy_cfg["max_abs_correction_ft"]),
    )
    parent_position = result["parent_position"]
    prior_position = result["prior_position"]
    candidates = np.asarray(candidate_order, dtype=object)
    freeze = pd.DataFrame(
        {
            "id": layout["id"].astype(str),
            "well": layout["well"].astype(str),
            "well_row_idx": layout["well_row_idx"].astype(np.int32),
            "outer_fold": folds.astype(np.int8),
            "md_since": layout["md_since"].astype(np.float32),
            "parent_candidate_position": parent_position.astype(np.int16),
            "parent_candidate_id": candidates[parent_position],
            "prior_candidate_position": prior_position.astype(np.int16),
            "prior_candidate_id": candidates[prior_position],
            "parent_candidate_tvt": result["parent_tvt"].astype(np.float64),
            "prior_candidate_tvt": result["prior_tvt"].astype(np.float64),
            "parent_pred_abs_error": result["parent_score"].astype(np.float32),
            "prior_parent_pred_abs_error": result["prior_parent_score"].astype(
                np.float32
            ),
            "prior_adjusted_score": result["prior_adjusted_score"].astype(
                np.float32
            ),
            "raw_nudge_ft": result["raw_nudge"].astype(np.float64),
            "bounded_correction_ft": result["correction"].astype(np.float64),
            "bounded_prediction_tvt": result["prediction"].astype(np.float64),
        }
    )
    return freeze

# %% [markdown]
# ## 4. Two-phase leakage and metric helpers

# %%
def reconstruct_true_tvt(
    candidate_tvt: np.ndarray,
    actual_abs_error: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(candidate_tvt, dtype=np.float64)
    errors = np.asarray(actual_abs_error, dtype=np.float64)
    if values.shape != errors.shape or values.ndim != 2:
        raise ValueError("candidate values and actual errors must align")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
    ):
        raise ValueError("truth reconstruction inputs are invalid")
    plus = values[:, 0] + errors[:, 0]
    minus = values[:, 0] - errors[:, 0]
    plus_residual = np.mean(
        np.abs(np.abs(values - plus[:, None]) - errors), axis=1
    )
    minus_residual = np.mean(
        np.abs(np.abs(values - minus[:, None]) - errors), axis=1
    )
    truth = np.where(plus_residual <= minus_residual, plus, minus)
    residual = np.abs(np.abs(values - truth[:, None]) - errors)
    max_residual = float(np.max(residual))
    if max_residual > float(tolerance):
        raise ValueError(
            f"true TVT reconstruction residual {max_residual} exceeds {tolerance}"
        )
    return truth, max_residual


def validate_freeze_alignment(
    freeze: pd.DataFrame,
    evaluation_long: pd.DataFrame,
    candidate_order: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    layout = validate_candidate_long_layout(
        evaluation_long, candidate_order, source="evaluation_parent_oof"
    )
    if len(freeze) != len(layout["id"]):
        raise ValueError("freeze/evaluation base-row counts differ")
    for column in KEY_COLUMNS + ["md_since"]:
        left = freeze[column].to_numpy()
        right = layout[column]
        if column in {"id", "well"}:
            matches = np.array_equal(left.astype(str), right.astype(str))
        else:
            matches = np.array_equal(left, right)
        if not matches:
            raise ValueError(f"freeze/evaluation key mismatch: {column}")
    n_candidates = len(candidate_order)
    candidate_tvt = evaluation_long["candidate_tvt"].to_numpy(
        np.float64
    ).reshape(-1, n_candidates)
    actual_abs_error = evaluation_long["actual_abs_error"].to_numpy(
        np.float64
    ).reshape(-1, n_candidates)
    return layout, candidate_tvt, actual_abs_error


def _update_stat(
    stats: dict[tuple[str, str], dict[str, float]],
    key: tuple[str, str],
    parent_error: np.ndarray,
    new_error: np.ndarray,
    correction: np.ndarray,
) -> None:
    if not len(parent_error):
        return
    item = stats[key]
    item["rows"] += len(parent_error)
    item["parent_sse"] += float(np.square(parent_error).sum())
    item["new_sse"] += float(np.square(new_error).sum())
    item["correction_sq_sum"] += float(np.square(correction).sum())
    item["correction_abs_max"] = max(
        item["correction_abs_max"], float(np.max(np.abs(correction)))
    )


def update_metric_stats(
    stats: dict[tuple[str, str], dict[str, float]],
    base: pd.DataFrame,
    parent_error: np.ndarray,
    new_error: np.ndarray,
    correction: np.ndarray,
    hidden_wells: Mapping[str, set[str]],
) -> None:
    _update_stat(
        stats,
        ("overall", "overall"),
        parent_error,
        new_error,
        correction,
    )
    folds = base["outer_fold"].to_numpy(np.int64)
    for fold in range(5):
        mask = folds == fold
        _update_stat(
            stats,
            ("outer_fold", str(fold)),
            parent_error[mask],
            new_error[mask],
            correction[mask],
        )
    md = base["md_since"].to_numpy(np.float64)
    for bucket, lower, upper in DISTANCE_BUCKETS:
        mask = (md >= lower) & (md < upper)
        _update_stat(
            stats,
            ("distance_bucket", bucket),
            parent_error[mask],
            new_error[mask],
            correction[mask],
        )
    wells = base["well"].astype(str).to_numpy()
    for scope, scope_wells in hidden_wells.items():
        mask = np.isin(wells, np.asarray(sorted(scope_wells), dtype=object))
        _update_stat(
            stats,
            ("hidden_like", scope),
            parent_error[mask],
            new_error[mask],
            correction[mask],
        )

    work = pd.DataFrame(
        {
            "well": wells,
            "parent_sse": np.square(parent_error),
            "new_sse": np.square(new_error),
            "correction_sq": np.square(correction),
            "correction_abs": np.abs(correction),
        }
    )
    grouped = work.groupby("well", sort=False).agg(
        rows=("well", "size"),
        parent_sse=("parent_sse", "sum"),
        new_sse=("new_sse", "sum"),
        correction_sq_sum=("correction_sq", "sum"),
        correction_abs_max=("correction_abs", "max"),
    )
    for well, row in grouped.iterrows():
        item = stats[("well", str(well))]
        item["rows"] += int(row["rows"])
        item["parent_sse"] += float(row["parent_sse"])
        item["new_sse"] += float(row["new_sse"])
        item["correction_sq_sum"] += float(row["correction_sq_sum"])
        item["correction_abs_max"] = max(
            item["correction_abs_max"], float(row["correction_abs_max"])
        )


def metric_frame(
    stats: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    tolerance: float,
    correction_cap: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scope_type, scope), item in sorted(stats.items()):
        count = int(item["rows"])
        if count <= 0:
            continue
        parent_rmse = math.sqrt(float(item["parent_sse"]) / count)
        new_rmse = math.sqrt(float(item["new_sse"]) / count)
        correction_rms = math.sqrt(float(item["correction_sq_sum"]) / count)
        correction_abs_max = float(item["correction_abs_max"])
        delta = new_rmse - parent_rmse
        rows.append(
            {
                "scope_type": scope_type,
                "scope": scope,
                "rows": count,
                "parent_rmse": parent_rmse,
                "bounded_nudge_rmse": new_rmse,
                "delta_rmse_new_minus_parent": delta,
                "correction_rms": correction_rms,
                "correction_abs_max": correction_abs_max,
                "delta_lte_correction_rms": bool(
                    delta <= correction_rms + tolerance
                ),
                "correction_rms_lte_abs_max": bool(
                    correction_rms <= correction_abs_max + tolerance
                ),
                "abs_max_lte_cap": bool(
                    correction_abs_max <= correction_cap + tolerance
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_gate(
    metrics: pd.DataFrame,
    technical_checks: Mapping[str, bool],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate_cfg = config["scientific_gate"]
    overall = metrics[
        metrics["scope_type"].eq("overall")
        & metrics["scope"].eq("overall")
    ].iloc[0]
    fold = metrics[metrics["scope_type"].eq("outer_fold")]
    distance = metrics[metrics["scope_type"].eq("distance_bucket")]
    hidden = metrics[metrics["scope_type"].eq("hidden_like")]
    wells = metrics[metrics["scope_type"].eq("well")]
    improvement = float(overall["parent_rmse"] - overall["bounded_nudge_rmse"])
    scientific_checks = {
        "minimum_overall_improvement": improvement
        >= float(gate_cfg["minimum_overall_rmse_improvement_ft"]),
        "all_folds_nonworse": int(
            (fold["delta_rmse_new_minus_parent"] <= 0.0).sum()
        )
        == int(gate_cfg["required_nonworse_folds"]),
        "all_distance_buckets_nonworse": int(
            (distance["delta_rmse_new_minus_parent"] <= 0.0).sum()
        )
        == int(gate_cfg["required_nonworse_distance_buckets"]),
        "all_hidden_like_scopes_nonworse": int(
            (hidden["delta_rmse_new_minus_parent"] <= 0.0).sum()
        )
        == int(gate_cfg["required_nonworse_hidden_like_scopes"]),
        "worst_well_within_risk_budget": float(
            wells["delta_rmse_new_minus_parent"].max()
        )
        <= float(gate_cfg["maximum_worst_well_regression_ft"]),
        "risk_certificate_all_scopes": bool(
            metrics[
                [
                    "delta_lte_correction_rms",
                    "correction_rms_lte_abs_max",
                    "abs_max_lte_cap",
                ]
            ]
            .astype(bool)
            .all()
            .all()
        ),
    }
    technical_passed = bool(all(bool(value) for value in technical_checks.values()))
    scientific_passed = bool(all(scientific_checks.values()))
    if not technical_passed:
        decision = "technical_fail_fix_same_frozen_policy_only"
    elif scientific_passed:
        decision = "rmse_prior_bounded_nudge_method_confirmed_on_saved_oof"
    else:
        decision = "scientific_fail_close_without_policy_rescue"
    return {
        "technical": {
            "checks": dict(technical_checks),
            "passed": technical_passed,
        },
        "scientific": {
            "checks": scientific_checks,
            "metrics": {
                "parent_oof_rmse": float(overall["parent_rmse"]),
                "bounded_nudge_oof_rmse": float(overall["bounded_nudge_rmse"]),
                "overall_improvement_ft": improvement,
                "nonworse_folds": int(
                    (fold["delta_rmse_new_minus_parent"] <= 0.0).sum()
                ),
                "nonworse_distance_buckets": int(
                    (distance["delta_rmse_new_minus_parent"] <= 0.0).sum()
                ),
                "nonworse_hidden_like_scopes": int(
                    (hidden["delta_rmse_new_minus_parent"] <= 0.0).sum()
                ),
                "observed_worst_well_regression_ft": float(
                    wells["delta_rmse_new_minus_parent"].max()
                ),
                "maximum_abs_correction_ft": float(
                    metrics["correction_abs_max"].max()
                ),
            },
            "passed": scientific_passed,
        },
        "decision": decision,
    }

# %% [markdown]
# ## 5. Setup, config, inputs, and compute contract

# %%
CONFIG_PATH = find_support_file("config.yaml")
JUPYTEXT_SOURCE_PATH = find_support_file(
    f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"
)
CONFIG = read_yaml(CONFIG_PATH)
STATIC = validate_static_contract(CONFIG)
OUTPUT_DIR = runtime_output_dir()
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "candidate_order": STATIC["candidate_order"],
            "primary_domain": STATIC["primary_domain"],
            "policy": CONFIG["policy"],
            "risk_certificate": CONFIG["risk_certificate"],
            "cost": STATIC["cost"],
            "run_approved": STATIC["run_approved"],
        },
        indent=2,
    )
)

if EXECUTE_NOTEBOOK:
    if not bool(CONFIG["execution"]["run_approved"]):
        raise RuntimeError(
            "exp415 Kaggle execution is not approved. Keep run_approved=false "
            "until the user approves canonical Notebook adoption and the "
            "zero-booster private CPU readout."
        )
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parent_cfg = CONFIG["data"]["parent_candidate_score_oof"]
    parent_oof_path = resolve_pinned_file(
        parent_cfg["patterns"],
        parent_cfg["sha256"],
        label="corrected exp264 Stage B v5 candidate-score OOF",
    )
    rmse_cfg = CONFIG["data"]["candidate_rmse_table"]
    candidate_rmse_path = resolve_pinned_file(
        rmse_cfg["patterns"],
        rmse_cfg["sha256"],
        label="exp407 exact-fit candidate RMSE table",
    )
    hidden_cfg = CONFIG["data"]["hidden_like_assignment"]
    hidden_path = resolve_pinned_file(
        hidden_cfg["patterns"],
        hidden_cfg["sha256"],
        label="hidden-like assignment",
    )
    rmse_matrix, rmse_table, rmse_audit = load_candidate_rmse_matrix(
        candidate_rmse_path, CONFIG
    )
    parent_file = pq.ParquetFile(parent_oof_path)
    if int(parent_file.metadata.num_rows) != int(
        CONFIG["validation"]["expected_candidate_long_rows"]
    ):
        raise ValueError("parent candidate-score OOF row count changed")
    input_manifest = {
        "config": {
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "jupytext_source": {
            "path": str(JUPYTEXT_SOURCE_PATH),
            "sha256": sha256_file(JUPYTEXT_SOURCE_PATH),
        },
        "parent_candidate_score_oof": {
            "path": str(parent_oof_path),
            "sha256": sha256_file(parent_oof_path),
            "rows": int(parent_file.metadata.num_rows),
            "row_groups": int(parent_file.metadata.num_row_groups),
        },
        "candidate_rmse_table": rmse_audit,
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "sha256": sha256_file(hidden_path),
        },
        "cost": STATIC["cost"],
    }
    write_json(OUTPUT_DIR / "input_manifest.json", input_manifest)
    display(rmse_table)
    print(json.dumps(input_manifest, indent=2))

# %% [markdown]
# ## 6. Phase 1 — truth-free policy freeze
#
# `actual_abs_error`をschemaから除外してparent/prior候補、candidate TVT、補正量を
# freezeする。freeze fileをcloseしてSHAを確定するまでtruth phaseへ進まない。

# %%
if EXECUTE_NOTEBOOK:
    freeze_path = OUTPUT_DIR / "truth_free_policy_freeze.parquet"
    freeze_writer = IncrementalParquetWriter(freeze_path)
    feature_schema_values: set[str] = set()
    candidate_contract_values: set[str] = set()
    wells_seen: set[str] = set()
    folds_seen: set[int] = set()
    selection_counts: dict[tuple[int, str, str], int] = defaultdict(int)
    correction_rows = 0
    correction_nonzero_rows = 0
    correction_clipped_rows = 0
    correction_abs_max = 0.0
    correction_abs_sum = 0.0
    correction_sq_sum = 0.0
    cap = float(CONFIG["policy"]["max_abs_correction_ft"])

    for row_group in range(parent_file.metadata.num_row_groups):
        truth_free_long = parent_file.read_row_group(
            row_group, columns=TRUTH_FREE_COLUMNS
        ).to_pandas()
        freeze = truth_free_policy_batch(truth_free_long, rmse_matrix, CONFIG)
        freeze_writer.write(freeze)
        feature_schema_values.update(
            truth_free_long["feature_schema_sha"].astype(str).unique()
        )
        candidate_contract_values.update(
            truth_free_long["candidate_contract_sha"].astype(str).unique()
        )
        wells_seen.update(freeze["well"].astype(str))
        folds_seen.update(freeze["outer_fold"].astype(int).unique().tolist())
        for fold, group in freeze.groupby("outer_fold", sort=False):
            for role, column in (
                ("parent", "parent_candidate_id"),
                ("rmse_prior", "prior_candidate_id"),
            ):
                counts = group[column].astype(str).value_counts()
                for candidate_id, count in counts.items():
                    selection_counts[
                        (int(fold), role, str(candidate_id))
                    ] += int(count)
        correction = freeze["bounded_correction_ft"].to_numpy(np.float64)
        raw_nudge = freeze["raw_nudge_ft"].to_numpy(np.float64)
        correction_rows += len(correction)
        correction_nonzero_rows += int(np.sum(correction != 0.0))
        correction_clipped_rows += int(
            np.sum(np.abs(raw_nudge) > cap + 1.0e-12)
        )
        correction_abs_max = max(
            correction_abs_max, float(np.max(np.abs(correction)))
        )
        correction_abs_sum += float(np.abs(correction).sum())
        correction_sq_sum += float(np.square(correction).sum())
        if (row_group + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "truth_free_freeze",
                        "row_groups": row_group + 1,
                        "total": parent_file.metadata.num_row_groups,
                    }
                ),
                flush=True,
            )
    freeze_writer.close()
    freeze_sha = sha256_file(freeze_path)
    if feature_schema_values != {
        str(CONFIG["data"]["expected_feature_schema_sha256"])
    }:
        raise ValueError("parent feature schema SHA changed")
    if candidate_contract_values != {
        str(CONFIG["data"]["expected_candidate_contract_logical_sha256"])
    }:
        raise ValueError("parent candidate contract logical SHA changed")
    if folds_seen != set(CONFIG["validation"]["expected_folds"]):
        raise ValueError("parent fold inventory changed")
    if len(wells_seen) != int(CONFIG["validation"]["expected_wells"]):
        raise ValueError("parent well inventory changed")
    if freeze_writer.rows != int(CONFIG["validation"]["expected_base_rows"]):
        raise ValueError("truth-free freeze base-row count changed")
    if correction_abs_max > cap + 1.0e-12:
        raise ValueError("truth-free correction exceeds risk budget")

    selection_frame = pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "selector_role": role,
                "candidate_id": candidate_id,
                "selected_rows": count,
            }
            for (fold, role, candidate_id), count in sorted(
                selection_counts.items()
            )
        ]
    )
    selection_frame.to_csv(
        OUTPUT_DIR / "candidate_selection_by_fold.csv", index=False
    )
    freeze_manifest = {
        "status": "truth_free_policy_frozen",
        "path": freeze_path.name,
        "sha256": freeze_sha,
        "rows": freeze_writer.rows,
        "row_groups": freeze_writer.row_groups,
        "wells": len(wells_seen),
        "folds": sorted(folds_seen),
        "candidate_order": STATIC["candidate_order"],
        "primary_domain": STATIC["primary_domain"],
        "feature_schema_sha256": next(iter(feature_schema_values)),
        "candidate_contract_logical_sha256": next(
            iter(candidate_contract_values)
        ),
        "truth_columns_read": [],
        "forbidden_truth_reads": 0,
        "policy": CONFIG["policy"],
        "correction": {
            "rows": correction_rows,
            "nonzero_rows": correction_nonzero_rows,
            "clipped_rows": correction_clipped_rows,
            "mean_abs": correction_abs_sum / correction_rows,
            "rms": math.sqrt(correction_sq_sum / correction_rows),
            "max_abs": correction_abs_max,
        },
    }
    write_json(OUTPUT_DIR / "truth_free_policy_freeze_manifest.json", freeze_manifest)
    print(json.dumps(freeze_manifest, indent=2))

# %% [markdown]
# ## 7. Phase 2 — evaluation and risk certificate
#
# freeze SHA確定後だけ`actual_abs_error`を読む。12候補のvalue / absolute errorから
# true TVTをexact reconstructionし、overall / fold / distance / hidden-like /
# wellのRMSEとrisk inequalityを集計する。

# %%
if EXECUTE_NOTEBOOK:
    freeze_file = pq.ParquetFile(freeze_path)
    if freeze_file.metadata.num_row_groups != parent_file.metadata.num_row_groups:
        raise ValueError("freeze/parent row-group counts differ")
    assignment = pd.read_csv(hidden_path, dtype={"well_id": str})
    if assignment["well_id"].astype(str).duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    hidden_wells = {
        scope: set(
            assignment.loc[
                assignment[details["column"]].eq(details["included_value"]),
                "well_id",
            ].astype(str)
        )
        for scope, details in CONFIG["validation"]["hidden_like_scopes"].items()
    }
    metric_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "rows": 0,
            "parent_sse": 0.0,
            "new_sse": 0.0,
            "correction_sq_sum": 0.0,
            "correction_abs_max": 0.0,
        }
    )
    prediction_path = OUTPUT_DIR / "bounded_nudge_oof_predictions.parquet"
    prediction_writer = IncrementalParquetWriter(prediction_path)
    truth_reconstruction_max_abs_error = 0.0
    correction_application_max_abs_error = 0.0
    evaluation_rows = 0
    truth_read_started_after_freeze_sha = bool(freeze_sha)

    for row_group in range(parent_file.metadata.num_row_groups):
        freeze = freeze_file.read_row_group(row_group).to_pandas()
        evaluation_long = parent_file.read_row_group(
            row_group, columns=EVALUATION_COLUMNS
        ).to_pandas()
        layout, candidate_tvt, actual_abs_error = validate_freeze_alignment(
            freeze, evaluation_long, STATIC["candidate_order"]
        )
        truth, reconstruction_error = reconstruct_true_tvt(
            candidate_tvt,
            actual_abs_error,
            tolerance=float(
                CONFIG["validation"]["evaluation_phase"][
                    "truth_reconstruction_max_abs_tolerance"
                ]
            ),
        )
        truth_reconstruction_max_abs_error = max(
            truth_reconstruction_max_abs_error, reconstruction_error
        )
        parent_prediction = freeze["parent_candidate_tvt"].to_numpy(np.float64)
        bounded_prediction = freeze["bounded_prediction_tvt"].to_numpy(
            np.float64
        )
        declared_correction = freeze["bounded_correction_ft"].to_numpy(np.float64)
        correction = bounded_prediction - parent_prediction
        correction_application_max_abs_error = max(
            correction_application_max_abs_error,
            float(np.max(np.abs(correction - declared_correction))),
        )
        if correction_application_max_abs_error > 1.0e-12:
            raise ValueError(
                "stored prediction does not apply the declared bounded correction"
            )
        parent_error = parent_prediction - truth
        new_error = bounded_prediction - truth
        base = freeze[KEY_COLUMNS + ["md_since"]].copy()
        update_metric_stats(
            metric_stats,
            base,
            parent_error,
            new_error,
            correction,
            hidden_wells,
        )
        prediction = freeze.copy()
        prediction["true_tvt"] = truth.astype(np.float32)
        prediction["parent_error"] = parent_error.astype(np.float32)
        prediction["bounded_nudge_error"] = new_error.astype(np.float32)
        prediction["parent_squared_error"] = np.square(parent_error).astype(
            np.float32
        )
        prediction["bounded_nudge_squared_error"] = np.square(
            new_error
        ).astype(np.float32)
        prediction_writer.write(prediction)
        evaluation_rows += len(prediction)
        if (row_group + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "phase": "evaluation",
                        "row_groups": row_group + 1,
                        "total": parent_file.metadata.num_row_groups,
                    }
                ),
                flush=True,
            )
    prediction_writer.close()
    if evaluation_rows != int(CONFIG["validation"]["expected_base_rows"]):
        raise ValueError("evaluation base-row count changed")

    tolerance = float(
        CONFIG["scientific_gate"]["risk_bound_absolute_tolerance"]
    )
    metrics = metric_frame(
        metric_stats,
        tolerance=tolerance,
        correction_cap=cap,
    )
    metrics.to_csv(OUTPUT_DIR / "bounded_nudge_metrics_all_scopes.csv", index=False)
    for scope_type, filename in (
        ("outer_fold", "bounded_nudge_metrics_by_fold.csv"),
        ("distance_bucket", "bounded_nudge_metrics_by_distance.csv"),
        ("hidden_like", "bounded_nudge_metrics_hidden_like.csv"),
        ("well", "bounded_nudge_metrics_by_well.csv"),
    ):
        metrics[metrics["scope_type"].eq(scope_type)].to_csv(
            OUTPUT_DIR / filename, index=False
        )
    risk_certificate = {
        "status": "minkowski_rmse_bound_verified",
        "inequality": CONFIG["risk_certificate"]["inequality"],
        "configured_maximum_scope_rmse_regression_ft": cap,
        "evaluated_scope_rows": len(metrics),
        "delta_lte_correction_rms_all": bool(
            metrics["delta_lte_correction_rms"].all()
        ),
        "correction_rms_lte_abs_max_all": bool(
            metrics["correction_rms_lte_abs_max"].all()
        ),
        "abs_max_lte_cap_all": bool(metrics["abs_max_lte_cap"].all()),
        "observed_max_abs_correction_ft": float(
            metrics["correction_abs_max"].max()
        ),
        "observed_max_scope_rmse_regression_ft": float(
            metrics["delta_rmse_new_minus_parent"].max()
        ),
        "observed_worst_well_rmse_regression_ft": float(
            metrics.loc[
                metrics["scope_type"].eq("well"),
                "delta_rmse_new_minus_parent",
            ].max()
        ),
    }
    write_json(OUTPUT_DIR / "risk_certificate.json", risk_certificate)
    truth_read_ledger = {
        "truth_free_phase": {
            "columns_read": TRUTH_FREE_COLUMNS,
            "truth_columns_read": [],
            "forbidden_truth_reads": 0,
            "freeze_sha256": freeze_sha,
        },
        "evaluation_phase": {
            "started_after_freeze_sha": truth_read_started_after_freeze_sha,
            "columns_read": EVALUATION_COLUMNS,
            "truth_columns_read": ["actual_abs_error"],
            "rows": evaluation_rows,
            "truth_reconstruction_max_abs_error": (
                truth_reconstruction_max_abs_error
            ),
            "correction_application_max_abs_error": (
                correction_application_max_abs_error
            ),
        },
    }
    write_json(OUTPUT_DIR / "truth_read_ledger.json", truth_read_ledger)

# %% [markdown]
# ## 8. Gate, metrics, diagnostics, and generated artifacts

# %%
if EXECUTE_NOTEBOOK:
    technical_checks = {
        "input_parent_sha": sha256_file(parent_oof_path)
        == str(CONFIG["data"]["parent_candidate_score_oof"]["sha256"]),
        "input_candidate_rmse_sha": sha256_file(candidate_rmse_path)
        == str(CONFIG["data"]["candidate_rmse_table"]["sha256"]),
        "input_hidden_assignment_sha": sha256_file(hidden_path)
        == str(CONFIG["data"]["hidden_like_assignment"]["sha256"]),
        "base_rows": freeze_writer.rows
        == int(CONFIG["validation"]["expected_base_rows"]),
        "evaluation_rows": evaluation_rows
        == int(CONFIG["validation"]["expected_base_rows"]),
        "well_count": len(wells_seen)
        == int(CONFIG["validation"]["expected_wells"]),
        "fold_inventory": folds_seen
        == set(CONFIG["validation"]["expected_folds"]),
        "candidate_rmse_rows": len(rmse_table)
        == int(CONFIG["data"]["candidate_rmse_table"]["expected_rows"]),
        "truth_free_forbidden_reads": int(
            freeze_manifest["forbidden_truth_reads"]
        )
        == 0,
        "evaluation_after_freeze_sha": truth_read_started_after_freeze_sha,
        "truth_reconstruction": truth_reconstruction_max_abs_error
        <= float(
            CONFIG["validation"]["evaluation_phase"][
                "truth_reconstruction_max_abs_tolerance"
            ]
        ),
        "correction_bound": correction_abs_max <= cap + tolerance,
        "correction_application_exact": (
            correction_application_max_abs_error <= 1.0e-12
        ),
        "risk_certificate": bool(
            risk_certificate["delta_lte_correction_rms_all"]
            and risk_certificate["correction_rms_lte_abs_max_all"]
            and risk_certificate["abs_max_lte_cap_all"]
        ),
        "zero_booster_contract": STATIC["cost"]
        == {
            "active_variants": 1,
            "models": 0,
            "model_configs": 0,
            "folds_for_fit": 0,
            "boosters": 0,
            "control_retraining": 0,
            "pf_runs": 0,
            "hmm_runs": 0,
            "beam_runs": 0,
            "gpu_runs": 0,
            "inference_runs": 0,
            "submissions": 0,
        },
    }
    gate = evaluate_gate(metrics, technical_checks, CONFIG)
    output_artifacts = {}
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file() and path.name != "reproducibility_manifest.json":
            output_artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    summary = {
        "status": "exp415_saved_oof_confirmation_complete",
        "experiment": EXPERIMENT_NAME,
        "route": CONFIG["experiment"]["route"],
        "diagnostic_only": True,
        "route_anchor_update_allowed": False,
        "source_config_sha256": sha256_file(CONFIG_PATH),
        "jupytext_source_sha256": sha256_file(JUPYTEXT_SOURCE_PATH),
        "inputs": input_manifest,
        "cost": STATIC["cost"],
        "freeze": freeze_manifest,
        "truth_read_ledger": truth_read_ledger,
        "risk_certificate": risk_certificate,
        "gate": gate,
        "artifacts": output_artifacts,
    }
    gate_path = OUTPUT_DIR / "exp415_confirmation_gate.json"
    write_json(gate_path, summary)
    summary["gate_file_sha256"] = sha256_file(gate_path)
    reproducibility = {
        "status": "exp415_saved_oof_confirmation_complete",
        "source_config_sha256": summary["source_config_sha256"],
        "jupytext_source_sha256": summary["jupytext_source_sha256"],
        "parent_candidate_score_oof_sha256": sha256_file(parent_oof_path),
        "candidate_rmse_table_sha256": sha256_file(candidate_rmse_path),
        "hidden_like_assignment_sha256": sha256_file(hidden_path),
        "truth_free_freeze_sha256": freeze_sha,
        "bounded_nudge_oof_predictions_sha256": sha256_file(prediction_path),
        "bounded_nudge_metrics_all_scopes_sha256": sha256_file(
            OUTPUT_DIR / "bounded_nudge_metrics_all_scopes.csv"
        ),
        "bounded_nudge_metrics_by_fold_sha256": sha256_file(
            OUTPUT_DIR / "bounded_nudge_metrics_by_fold.csv"
        ),
        "bounded_nudge_metrics_by_distance_sha256": sha256_file(
            OUTPUT_DIR / "bounded_nudge_metrics_by_distance.csv"
        ),
        "bounded_nudge_metrics_hidden_like_sha256": sha256_file(
            OUTPUT_DIR / "bounded_nudge_metrics_hidden_like.csv"
        ),
        "bounded_nudge_metrics_by_well_sha256": sha256_file(
            OUTPUT_DIR / "bounded_nudge_metrics_by_well.csv"
        ),
        "candidate_selection_by_fold_sha256": sha256_file(
            OUTPUT_DIR / "candidate_selection_by_fold.csv"
        ),
        "risk_certificate_sha256": sha256_file(
            OUTPUT_DIR / "risk_certificate.json"
        ),
        "confirmation_gate_sha256": summary["gate_file_sha256"],
        "decision": gate["decision"],
        "model_count": 0,
        "submission_generated": False,
    }
    write_json(
        OUTPUT_DIR / "reproducibility_manifest.json", reproducibility
    )

    display(metrics[metrics["scope_type"].ne("well")])
    display(
        metrics[metrics["scope_type"].eq("well")]
        .sort_values("delta_rmse_new_minus_parent", ascending=False)
        .head(30)
    )
    display(selection_frame)
    print(json.dumps(gate, indent=2))
    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")

    non_well = metrics[metrics["scope_type"].ne("well")].copy()
    axis = non_well.plot.bar(
        x="scope",
        y="delta_rmse_new_minus_parent",
        figsize=(12, 5),
        legend=False,
        title="exp415 bounded nudge RMSE delta vs corrected exp264",
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_ylabel("RMSE delta (ft)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bounded_nudge_scope_delta_rmse.png", dpi=140)
    plt.show()

    print("Generated files")
    for generated in sorted(OUTPUT_DIR.iterdir()):
        if generated.is_file():
            print(generated.name, generated.stat().st_size)
