# %% [markdown]
# # exp423 same-typewell GR-DTW truth-warp transfer readout — train
#
# This compact self-contained notebook implements the frozen Stage 0 contract.
# It creates no fitted model, reruns no PF/Beam process, and produces no
# inference or submission output. Query/outer-valid truth is attached only
# after all deployable candidates, controls, donor ranks, and content hashes
# have been frozen.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Input contract and late-truth ledger
# 5. GR suffix profile preprocessing
# 6. Constrained DTW and donor truth-warp transfer
# 7. Outer-fold candidate generation and target-free freeze
# 8. Late truth attachment and diagnostic readouts
# 9. Technical/scientific gates and generated artifacts
# 10. Setup and configuration preview
# 11. Run the approved Kaggle CPU readout

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit
except ImportError:  # pragma: no cover - Kaggle/runtime contract requires numba.

    def njit(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs

        def decorate(function: Any) -> Any:
            return function

        return decorate


EXPERIMENT_NAME = "exp423_same_typewell_gr_dtw_truth_warp_transfer_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

EXP099_FEATURE_CACHE = (
    "exp099_pf_multi_observation_likelihood_probe_"
    "multiobs_likelihood_probe_train_features.csv.gz"
)
EXP065_ASSIGNMENTS = "common_typewell_cluster_assignments.csv"
EXP109_OOF = "exp109_typewell_neighbor_prior_features_oof_predictions.csv.gz"
EXP115_ASSIGNMENTS = (
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
)

PRIMARY_CANDIDATE = "analog_top5_median"
TOP1_CANDIDATE = "analog_top1"
RANDOM_CONTROL = "stable_random_same_group"
ORACLE_CANDIDATE = "analog_top5_oracle_well"
PARENT_REFERENCE = "exp109_best_fixed"
LIKPF_REFERENCE = "exp099_likpf_mean"
PARENT_SOURCE_COLUMN = "native_overlap_0p999_likpf_mean_corr_a0p2_c40"

TARGET_FREE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "fold",
    "md_since",
    "eval_len",
    "last_known_tvt",
    "typewell_group_id",
    "supported",
    "eligible_donor_count",
    "used_donor_count",
    "top1_donor_well",
    "random_donor_well",
    TOP1_CANDIDATE,
    PRIMARY_CANDIDATE,
    RANDOM_CONTROL,
    PARENT_REFERENCE,
    LIKPF_REFERENCE,
)
DEPLOYABLE_CANDIDATES = (
    TOP1_CANDIDATE,
    PRIMARY_CANDIDATE,
    RANDOM_CONTROL,
    PARENT_REFERENCE,
    LIKPF_REFERENCE,
)
EVALUATION_CANDIDATES = (
    TOP1_CANDIDATE,
    PRIMARY_CANDIDATE,
    RANDOM_CONTROL,
    ORACLE_CANDIDATE,
    PARENT_REFERENCE,
    LIKPF_REFERENCE,
)


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers


# %%
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
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    if decompressed:
        stream: Any = gzip.open(path, "rb")
    else:
        stream = path.open("rb")
    with stream as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    schema = [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "nullable": bool(frame[column].isna().any()),
        }
        for column in frame.columns
    ]
    return canonical_json_sha256(schema)


def logical_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"logical SHA columns are missing: {missing}")
    ordered = frame.loc[:, list(columns)].copy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [(str(column), str(ordered[column].dtype)) for column in ordered.columns],
            separators=(",", ":"),
        ).encode()
    )
    row_hashes = pd.util.hash_pandas_object(ordered, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def find_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("experiments") / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml",
        Path("config.yaml"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        value = yaml.safe_load(path.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return path
    raise FileNotFoundError("exp423 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = find_config_path() if path is None else path
    value = yaml.safe_load(source.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def resolve_existing_file(filename: str, candidates: Iterable[str | Path]) -> Path:
    checked: list[str] = []
    for raw in candidates:
        path = Path(raw)
        candidate = path / filename if path.is_dir() else path
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    local_defaults = [
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "artifacts" / filename,
        Path("artifacts") / filename,
    ]
    for candidate in local_defaults:
        checked.append(str(candidate))
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"required input {filename} was not found; checked:\n"
        + "\n".join(checked[:100])
    )


def resolve_raw_train_dir(config: dict[str, Any]) -> Path:
    candidates = [
        Path(str(value))
        for value in (get_nested(config, "data.raw_train_candidates") or [])
    ]
    candidates.extend(
        [
            PACKAGE_DIR / "data" / "raw" / "train",
            Path("data/raw/train"),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    raise FileNotFoundError("raw train directory containing horizontal well CSVs was not found")


def verify_expected_sha(
    path: Path,
    *,
    expected_raw: str | None = None,
    expected_decompressed: str | None = None,
) -> dict[str, Any]:
    raw_sha = sha256_path(path)
    if expected_raw and raw_sha != expected_raw:
        raise ValueError(f"raw SHA mismatch for {path}: {raw_sha}")
    decompressed_sha: str | None = None
    if path.suffix == ".gz":
        decompressed_sha = sha256_path(path, decompressed=True)
        if expected_decompressed and decompressed_sha != expected_decompressed:
            raise ValueError(f"decompressed SHA mismatch for {path}: {decompressed_sha}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "sha_match": True,
    }


def build_raw_horizontal_manifest(
    train_dir: Path,
    wells: Sequence[str],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for well in sorted({str(value) for value in wells}):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        files.append(
            {
                "well": well,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "raw_sha256": sha256_path(path),
            }
        )
    return {
        "path": str(train_dir),
        "horizontal_files": len(files),
        "combined_manifest_sha256": canonical_json_sha256(files),
        "files": files,
        "safe_reader_columns": ["MD", "GR", "TVT_input"],
        "query_truth_column_exposed": False,
        "sha_match": True,
    }


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def execution_counts(config: dict[str, Any]) -> dict[str, int]:
    return {
        "audit_variants": int(get_nested(config, "execution.audit_variants")),
        "lightgbm_configs": int(get_nested(config, "execution.lightgbm_configs")),
        "trained_folds": int(get_nested(config, "execution.trained_folds")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "pf_well_runs": int(get_nested(config, "execution.pf_well_runs")),
        "hmm_well_runs": int(get_nested(config, "execution.hmm_well_runs")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
        "reporting_folds": int(get_nested(config, "validation.n_folds")),
    }


def validate_scientific_contract(
    config: dict[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp423 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != "exp109_typewell_neighbor_prior_features":
        raise ValueError("exp423 parent contract changed")
    if not bool(get_nested(config, "implementation.enabled")):
        raise RuntimeError("exp423 implementation must be enabled")
    if get_nested(config, "evaluation.primary_candidate") != PRIMARY_CANDIDATE:
        raise ValueError("primary candidate changed")
    if int(get_nested(config, "data.donor_pool.top_k")) != 5:
        raise ValueError("top_k must remain 5")
    if int(get_nested(config, "gr_similarity.resampling.n_points")) != 256:
        raise ValueError("DTW profile length must remain 256")
    if int(get_nested(config, "gr_similarity.dtw.sakoe_chiba_band_points")) != 32:
        raise ValueError("DTW band must remain 32")
    if int(get_nested(config, "gr_similarity.dtw.max_consecutive_horizontal_or_vertical")) != 4:
        raise ValueError("DTW run-length limit must remain 4")
    if bool(get_nested(config, "decision.rescue_grid_allowed")):
        raise ValueError("post-hoc rescue grid is forbidden")
    expected_counts = {
        "audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "reporting_folds": 5,
    }
    observed_counts = execution_counts(config)
    if observed_counts != expected_counts:
        raise ValueError(
            f"execution count contract changed: {observed_counts} != {expected_counts}"
        )
    if require_run_approval and not bool(get_nested(config, "execution.audit_run_approved")):
        raise RuntimeError("Kaggle CPU audit run is not approved")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": "exp109_typewell_neighbor_prior_features",
        "folds": 5,
        "seed": 42,
        "query_truth_policy": "late_join_after_target_free_freeze",
        "donor_policy": "same_native_overlap_1_group_outer_train_only",
        "primary_candidate": PRIMARY_CANDIDATE,
        "top_k": 5,
        "gr_points": 256,
        "dtw_band_points": 32,
        "dtw_max_axis_run": 4,
        "execution_counts": observed_counts,
        "success_gates": get_nested(config, "success_gates"),
        "rescue_grid_allowed": False,
    }
    contract["scientific_contract_sha256"] = canonical_json_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Input contract and late-truth ledger


# %%
@dataclass
class TruthAccessLedger:
    query_truth_rows_before_freeze: int = 0
    query_truth_rows_after_freeze: int = 0
    donor_truth_rows_by_fold: dict[int, int] = field(default_factory=dict)
    target_free_frozen: bool = False
    frozen_content_sha256: str | None = None

    def record_donor_truth(self, fold: int, rows: int) -> None:
        if self.target_free_frozen:
            raise RuntimeError("donor truth must be materialized before target-free freeze")
        self.donor_truth_rows_by_fold[int(fold)] = (
            self.donor_truth_rows_by_fold.get(int(fold), 0) + int(rows)
        )

    def require_frozen(self) -> None:
        if not self.target_free_frozen or not self.frozen_content_sha256:
            raise RuntimeError("query truth access requires a frozen target-free content SHA")

    def mark_frozen(self, content_sha256: str) -> None:
        if self.query_truth_rows_before_freeze != 0:
            raise RuntimeError("query truth rows were opened before target-free freeze")
        if len(content_sha256) != 64:
            raise ValueError("frozen content SHA must be a SHA256 hex digest")
        self.target_free_frozen = True
        self.frozen_content_sha256 = content_sha256

    def record_query_truth_after_freeze(self, rows: int) -> None:
        self.require_frozen()
        self.query_truth_rows_after_freeze += int(rows)

    def snapshot(self) -> dict[str, Any]:
        return {
            "query_truth_rows_before_freeze": self.query_truth_rows_before_freeze,
            "query_truth_rows_after_freeze": self.query_truth_rows_after_freeze,
            "donor_truth_rows_by_fold": dict(sorted(self.donor_truth_rows_by_fold.items())),
            "target_free_frozen": self.target_free_frozen,
            "frozen_content_sha256": self.frozen_content_sha256,
        }


def _parse_row_idx(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    numeric = pd.to_numeric(extracted, errors="coerce")
    if numeric.isna().any():
        examples = ids[numeric.isna()].head(5).tolist()
        raise ValueError(f"could not parse row_idx from ids: {examples}")
    return numeric.to_numpy(np.int64)


def deterministic_well_folds(
    wells: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    ordered = np.asarray(sorted({str(well) for well in wells}), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    parts = np.array_split(shuffled, n_folds)
    rows = [
        {"well": str(well), "fold": int(fold)}
        for fold, part in enumerate(parts)
        for well in part.tolist()
    ]
    result = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    if len(result) != len(ordered) or not result["well"].is_unique:
        raise AssertionError("fold assignment must contain each well exactly once")
    return result


def _input_spec(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.inputs.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.inputs.{key} must be a mapping")
    return value


def resolve_input(config: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    spec = _input_spec(config, key)
    path = resolve_existing_file(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    manifest = verify_expected_sha(
        path,
        expected_raw=spec.get("expected_raw_sha256"),
        expected_decompressed=spec.get("expected_decompressed_sha256"),
    )
    manifest["name"] = key
    return path, manifest


def load_target_free_inventory(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    path, manifest = resolve_input(config, "exp099_feature_cache")
    safe_columns = [
        "id",
        "well",
        "last_known_tvt",
        "likpf_mean",
        "md_since",
        "eval_len",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "target" not in header:
        raise ValueError("exp099 cache must contain a post-freeze target column")
    missing = sorted(set(safe_columns) - set(header))
    if missing:
        raise ValueError(f"exp099 cache safe columns missing: {missing}")
    frame = pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = _parse_row_idx(frame["id"])
    for column in ("last_known_tvt", "likpf_mean", "md_since", "eval_len"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.duplicated(["well", "row_idx"]).any() or frame["id"].duplicated().any():
        raise ValueError("exp099 safe inventory has duplicate row identity")
    if not np.isfinite(
        frame[["last_known_tvt", "likpf_mean", "md_since", "eval_len"]].to_numpy(
            np.float64
        )
    ).all():
        raise ValueError("exp099 safe inventory contains non-finite values")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError("exp099 row/well inventory does not match exp423 contract")
    manifest.update({"rows": len(frame), "wells": int(frame["well"].nunique())})
    return frame, manifest, path


def load_typewell_groups(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp065_cluster_assignments")
    frame = pd.read_csv(path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    if not required.issubset(frame.columns):
        raise ValueError(f"cluster assignments missing {sorted(required - set(frame.columns))}")
    selected = frame.loc[
        frame["method"].astype(str).eq("native_overlap")
        & frame["threshold"].astype(str).eq("1"),
        ["well_id", "cluster_id", "cluster_size"],
    ].copy()
    selected["well_id"] = selected["well_id"].astype(str)
    selected["cluster_id"] = selected["cluster_id"].astype(str)
    selected["cluster_size"] = pd.to_numeric(
        selected["cluster_size"], errors="raise"
    ).astype(np.int64)
    if selected["well_id"].duplicated().any():
        raise ValueError("native_overlap=1 assignment has duplicate wells")
    manifest.update(
        {
            "selected_rows": len(selected),
            "selected_wells": int(selected["well_id"].nunique()),
            "selected_groups": int(selected["cluster_id"].nunique()),
        }
    )
    return selected, manifest


def load_parent_reference(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp109_oof")
    columns = ["id", "well", PARENT_SOURCE_COLUMN]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f"exp109 OOF reference missing {missing}")
    frame = pd.read_csv(path, usecols=columns, dtype={"id": str, "well": str})
    frame = frame.rename(columns={PARENT_SOURCE_COLUMN: PARENT_REFERENCE})
    frame[PARENT_REFERENCE] = pd.to_numeric(frame[PARENT_REFERENCE], errors="raise")
    if frame.duplicated(["id", "well"]).any():
        raise ValueError("exp109 reference has duplicate identity")
    if not np.isfinite(frame[PARENT_REFERENCE]).all():
        raise ValueError("exp109 reference must be finite")
    manifest.update({"rows": len(frame), "wells": int(frame["well"].nunique())})
    return frame, manifest


def build_target_free_row_inventory(
    safe: pd.DataFrame,
    groups: pd.DataFrame,
    parent: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, tuple[set[str], set[str]]]]:
    folds = deterministic_well_folds(
        safe["well"].unique().tolist(),
        n_folds=int(get_nested(config, "validation.n_folds")),
        seed=int(get_nested(config, "validation.seed")),
    )
    work = safe.merge(folds, on="well", how="left", validate="many_to_one")
    work = work.merge(
        groups.rename(
            columns={
                "well_id": "well",
                "cluster_id": "typewell_group_id",
                "cluster_size": "typewell_group_size",
            }
        ),
        on="well",
        how="left",
        validate="many_to_one",
    )
    work = work.merge(parent, on=["id", "well"], how="left", validate="one_to_one")
    work = work.rename(columns={"likpf_mean": LIKPF_REFERENCE})
    if work["fold"].isna().any():
        raise ValueError("fold assignment is incomplete")
    if work[PARENT_REFERENCE].isna().any():
        raise ValueError("exp109 reference row coverage is incomplete")
    work["fold"] = work["fold"].astype(np.int64)
    work = work.sort_values(["fold", "well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    all_wells = set(work["well"].astype(str))
    split_map: dict[int, tuple[set[str], set[str]]] = {}
    for fold in sorted(work["fold"].unique()):
        valid = set(work.loc[work["fold"].eq(fold), "well"].astype(str))
        train = all_wells - valid
        if train & valid:
            raise AssertionError("outer-train and outer-valid wells overlap")
        split_map[int(fold)] = (train, valid)
    return work, split_map


def load_hidden_like_assignments(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp115_hidden_like_assignments")
    frame = pd.read_csv(path, dtype={"well_id": str})
    role_columns = get_nested(config, "evaluation.hidden_like_role_columns") or {}
    required = {"well_id", *[str(value) for value in role_columns.values()]}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"hidden-like assignment columns missing: {sorted(required - set(frame.columns))}"
        )
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments must be one row per well")
    manifest.update({"rows": len(frame), "wells": int(frame["well_id"].nunique())})
    return frame, manifest


# %% [markdown]
# ## 5. GR suffix profile preprocessing


# %%
@dataclass(frozen=True)
class SuffixProfile:
    well: str
    row_idx: np.ndarray
    md: np.ndarray
    progress: np.ndarray
    anchor_tvt: float
    gr_resampled: np.ndarray
    gr_normalized: np.ndarray
    support_mask: np.ndarray
    finite_fraction: float
    robust_center: float
    robust_scale: float


@dataclass(frozen=True)
class DonorTruth:
    well: str
    progress: np.ndarray
    tvt: np.ndarray


def centered_rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("rolling median window must be a positive odd integer")
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(np.float64)
    )


def normalized_progress(md: np.ndarray) -> np.ndarray:
    values = np.asarray(md, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("suffix MD must be non-empty and finite")
    if np.any(np.diff(values) <= 0):
        raise ValueError("suffix MD must be strictly increasing")
    span = float(values[-1] - values[0])
    if span <= 0:
        if len(values) == 1:
            return np.zeros(1, dtype=np.float64)
        raise ValueError("suffix MD span must be positive")
    return (values - values[0]) / span


def nearest_support_mask(
    source_progress: np.ndarray,
    source_finite: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    position = np.searchsorted(source_progress, grid, side="left")
    right = np.clip(position, 0, len(source_progress) - 1)
    left = np.clip(position - 1, 0, len(source_progress) - 1)
    choose_right = np.abs(source_progress[right] - grid) < np.abs(
        grid - source_progress[left]
    )
    nearest = np.where(choose_right, right, left)
    return np.asarray(source_finite, dtype=bool)[nearest]


def preprocess_suffix_gr(
    md: np.ndarray,
    gr: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    progress = normalized_progress(md)
    raw_gr = np.asarray(gr, dtype=np.float64)
    if len(raw_gr) != len(progress):
        raise ValueError("GR and MD lengths differ")
    finite_raw = np.isfinite(raw_gr)
    finite_fraction = float(finite_raw.mean())
    minimum_fraction = float(get_nested(config, "gr_similarity.support.min_finite_fraction"))
    window = int(get_nested(config, "gr_similarity.smoothing.window_rows"))
    n_points = int(get_nested(config, "gr_similarity.resampling.n_points"))
    grid = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    smoothed = centered_rolling_median(raw_gr, window).copy()
    # The smoother may summarize finite neighbours, but it must not turn an
    # originally missing observation into supported GR evidence.
    smoothed[~finite_raw] = np.nan
    finite_smooth = np.isfinite(smoothed)
    resampled = np.full(n_points, np.nan, dtype=np.float64)
    if finite_smooth.sum() >= 2:
        resampled = np.interp(
            grid,
            progress[finite_smooth],
            smoothed[finite_smooth],
            left=np.nan,
            right=np.nan,
        )
    support_mask = nearest_support_mask(progress, finite_raw, grid)
    support_mask &= np.isfinite(resampled)
    supported_values = resampled[support_mask]
    center = float(np.median(supported_values)) if supported_values.size else np.nan
    mad = (
        float(np.median(np.abs(supported_values - center)))
        if supported_values.size
        else np.nan
    )
    scale = 1.4826 * mad if np.isfinite(mad) else np.nan
    min_scale = float(get_nested(config, "gr_similarity.normalization.min_scale"))
    normalized = np.full(n_points, np.nan, dtype=np.float64)
    profile_valid = (
        finite_fraction >= minimum_fraction
        and np.isfinite(scale)
        and scale > min_scale
        and support_mask.mean() >= minimum_fraction
    )
    if profile_valid:
        normalized[np.isfinite(resampled)] = (
            resampled[np.isfinite(resampled)] - center
        ) / scale
    return {
        "progress": progress,
        "grid": grid,
        "resampled": resampled,
        "normalized": normalized,
        "support_mask": support_mask,
        "finite_fraction": finite_fraction,
        "center": center,
        "scale": scale,
        "valid": bool(profile_valid),
    }


def load_safe_suffix_profile(
    well: str,
    rows: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> SuffixProfile:
    path = train_dir / f"{well}__horizontal_well.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "TVT" not in header:
        raise ValueError(f"{path} must contain TVT for the later truth reader")
    horizontal = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
    if "TVT" in horizontal.columns:
        raise AssertionError("target-free profile reader exposed TVT")
    query = rows.sort_values("row_idx", kind="mergesort")
    row_idx = query["row_idx"].to_numpy(np.int64)
    if row_idx.min(initial=0) < 0 or row_idx.max(initial=-1) >= len(horizontal):
        raise ValueError(f"row_idx outside horizontal frame for {well}")
    if len(np.unique(row_idx)) != len(row_idx):
        raise ValueError(f"duplicate row_idx for {well}")
    suffix = horizontal.iloc[row_idx]
    if suffix["TVT_input"].notna().any():
        raise ValueError(f"pseudo suffix contains observed TVT_input for {well}")
    known_before = pd.to_numeric(
        horizontal.loc[
            horizontal.index < int(row_idx[0]), "TVT_input"
        ],
        errors="coerce",
    ).dropna()
    if known_before.empty:
        raise ValueError(f"no query anchor before pseudo suffix for {well}")
    anchor = float(known_before.iloc[-1])
    inventory_anchor = pd.to_numeric(query["last_known_tvt"], errors="raise").to_numpy(
        np.float64
    )
    if not np.allclose(inventory_anchor, anchor, rtol=0.0, atol=2e-3):
        raise ValueError(f"exp099 anchor parity failed for {well}")
    md = pd.to_numeric(suffix["MD"], errors="raise").to_numpy(np.float64)
    gr = pd.to_numeric(suffix["GR"], errors="coerce").to_numpy(np.float64)
    prepared = preprocess_suffix_gr(md, gr, config)
    return SuffixProfile(
        well=str(well),
        row_idx=row_idx,
        md=md,
        progress=prepared["progress"],
        anchor_tvt=anchor,
        gr_resampled=prepared["resampled"],
        gr_normalized=prepared["normalized"],
        support_mask=prepared["support_mask"],
        finite_fraction=float(prepared["finite_fraction"]),
        robust_center=float(prepared["center"]),
        robust_scale=float(prepared["scale"]),
    )


def profile_is_valid(profile: SuffixProfile, config: dict[str, Any]) -> bool:
    minimum_fraction = float(get_nested(config, "gr_similarity.support.min_finite_fraction"))
    min_scale = float(get_nested(config, "gr_similarity.normalization.min_scale"))
    return bool(
        profile.finite_fraction >= minimum_fraction
        and profile.support_mask.mean() >= minimum_fraction
        and np.isfinite(profile.robust_scale)
        and profile.robust_scale > min_scale
        and np.isfinite(profile.gr_normalized).sum() >= 2
    )


def load_outer_train_donor_truth(
    well: str,
    profile: SuffixProfile,
    train_dir: Path,
    *,
    fold: int,
    outer_train_wells: set[str],
    outer_valid_wells: set[str],
    ledger: TruthAccessLedger,
) -> DonorTruth:
    if well not in outer_train_wells or well in outer_valid_wells:
        raise RuntimeError(f"donor {well} is not strictly outer-train for fold {fold}")
    path = train_dir / f"{well}__horizontal_well.csv"
    truth = pd.read_csv(path, usecols=["TVT"])
    selected = pd.to_numeric(
        truth.iloc[profile.row_idx]["TVT"], errors="raise"
    ).to_numpy(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError(f"donor suffix truth is non-finite for {well}")
    ledger.record_donor_truth(fold, len(selected))
    return DonorTruth(well=well, progress=profile.progress.copy(), tvt=selected)


# %% [markdown]
# ## 6. Constrained DTW and donor truth-warp transfer


# %%
@njit(cache=False)
def _constrained_dtw_numba(
    query: np.ndarray,
    donor: np.ndarray,
    band: int,
    max_run: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    n = len(query)
    m = len(donor)
    n_states = 1 + 2 * max_run
    inf = np.inf
    costs = np.full((n, m, n_states), inf, dtype=np.float64)
    previous_state = np.full((n, m, n_states), -1, dtype=np.int8)
    if not np.isfinite(query[0]) or not np.isfinite(donor[0]):
        return inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0
    costs[0, 0, 0] = (query[0] - donor[0]) ** 2

    for i in range(n):
        j_min = max(0, i - band)
        j_max = min(m - 1, i + band)
        for j in range(j_min, j_max + 1):
            if i == 0 and j == 0:
                continue
            if not np.isfinite(query[i]) or not np.isfinite(donor[j]):
                continue
            point_cost = (query[i] - donor[j]) ** 2

            if i > 0 and j > 0:
                best_state = 0
                best_cost = costs[i - 1, j - 1, 0]
                for state in range(1, n_states):
                    value = costs[i - 1, j - 1, state]
                    if value < best_cost:
                        best_cost = value
                        best_state = state
                if np.isfinite(best_cost):
                    costs[i, j, 0] = best_cost + point_cost
                    previous_state[i, j, 0] = best_state

            if i > 0:
                best_state = 0
                best_cost = costs[i - 1, j, 0]
                for state in range(max_run + 1, n_states):
                    value = costs[i - 1, j, state]
                    if value < best_cost:
                        best_cost = value
                        best_state = state
                if np.isfinite(best_cost):
                    costs[i, j, 1] = best_cost + point_cost
                    previous_state[i, j, 1] = best_state
                for run in range(2, max_run + 1):
                    source_state = run - 1
                    value = costs[i - 1, j, source_state]
                    if np.isfinite(value):
                        costs[i, j, run] = value + point_cost
                        previous_state[i, j, run] = source_state

            if j > 0:
                horizontal_start = max_run + 1
                best_state = 0
                best_cost = costs[i, j - 1, 0]
                for state in range(1, max_run + 1):
                    value = costs[i, j - 1, state]
                    if value < best_cost:
                        best_cost = value
                        best_state = state
                if np.isfinite(best_cost):
                    costs[i, j, horizontal_start] = best_cost + point_cost
                    previous_state[i, j, horizontal_start] = best_state
                for run in range(2, max_run + 1):
                    target_state = max_run + run
                    source_state = target_state - 1
                    value = costs[i, j - 1, source_state]
                    if np.isfinite(value):
                        costs[i, j, target_state] = value + point_cost
                        previous_state[i, j, target_state] = source_state

    final_state = 0
    final_cost = costs[n - 1, m - 1, 0]
    for state in range(1, n_states):
        value = costs[n - 1, m - 1, state]
        if value < final_cost:
            final_cost = value
            final_state = state
    if not np.isfinite(final_cost):
        return inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0

    max_path = n + m
    query_path = np.empty(max_path, dtype=np.int32)
    donor_path = np.empty(max_path, dtype=np.int32)
    length = 0
    i = n - 1
    j = m - 1
    state = final_state
    while True:
        query_path[length] = i
        donor_path[length] = j
        length += 1
        if i == 0 and j == 0:
            break
        previous = int(previous_state[i, j, state])
        if previous < 0:
            return inf, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), 0
        if state == 0:
            i -= 1
            j -= 1
        elif state <= max_run:
            i -= 1
        else:
            j -= 1
        state = previous

    out_query = np.empty(length, dtype=np.int32)
    out_donor = np.empty(length, dtype=np.int32)
    for index in range(length):
        out_query[index] = query_path[length - 1 - index]
        out_donor[index] = donor_path[length - 1 - index]
    return final_cost / length, out_query, out_donor, length


def constrained_dtw(
    query: np.ndarray,
    donor: np.ndarray,
    *,
    band: int,
    max_run: int,
) -> dict[str, Any]:
    left = np.asarray(query, dtype=np.float64)
    right = np.asarray(donor, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError("DTW inputs must be equal-length one-dimensional arrays")
    cost, query_path, donor_path, length = _constrained_dtw_numba(
        left, right, int(band), int(max_run)
    )
    if not np.isfinite(cost) or length == 0:
        raise ValueError("no finite constrained DTW path")
    query_steps = np.diff(query_path)
    donor_steps = np.diff(donor_path)
    allowed = (
        ((query_steps == 1) & (donor_steps == 1))
        | ((query_steps == 1) & (donor_steps == 0))
        | ((query_steps == 0) & (donor_steps == 1))
    )
    if not bool(allowed.all()):
        raise AssertionError("DTW path contains a forbidden step")
    if np.max(np.abs(query_path - donor_path), initial=0) > band:
        raise AssertionError("DTW path escaped the Sakoe-Chiba band")
    vertical_run = 0
    horizontal_run = 0
    max_vertical = 0
    max_horizontal = 0
    for q_step, d_step in zip(query_steps, donor_steps, strict=True):
        if q_step == 1 and d_step == 0:
            vertical_run += 1
            horizontal_run = 0
        elif q_step == 0 and d_step == 1:
            horizontal_run += 1
            vertical_run = 0
        else:
            vertical_run = 0
            horizontal_run = 0
        max_vertical = max(max_vertical, vertical_run)
        max_horizontal = max(max_horizontal, horizontal_run)
    if max_vertical > max_run or max_horizontal > max_run:
        raise AssertionError("DTW path exceeded the fixed axis-run limit")
    return {
        "normalized_cost": float(cost),
        "query_path": query_path,
        "donor_path": donor_path,
        "path_length": int(length),
        "max_vertical_run": int(max_vertical),
        "max_horizontal_run": int(max_horizontal),
    }


def query_to_donor_mapping(
    query_path: np.ndarray,
    donor_path: np.ndarray,
    *,
    n_points: int,
) -> np.ndarray:
    query_path = np.asarray(query_path, dtype=np.int64)
    donor_path = np.asarray(donor_path, dtype=np.int64)
    if len(query_path) != len(donor_path) or len(query_path) == 0:
        raise ValueError("DTW path is empty or misaligned")
    mapped = np.full(n_points, np.nan, dtype=np.float64)
    for query_index in range(n_points):
        values = donor_path[query_path == query_index]
        if values.size:
            mapped[query_index] = float(np.median(values))
    finite = np.isfinite(mapped)
    if finite.sum() < 2:
        raise ValueError("DTW path cannot define a query-to-donor mapping")
    grid = np.arange(n_points, dtype=np.float64)
    mapped = np.interp(grid, grid[finite], mapped[finite])
    mapped = np.maximum.accumulate(mapped)
    mapped = np.clip(mapped / max(1, n_points - 1), 0.0, 1.0)
    return mapped


def transfer_donor_truth_warp(
    query: SuffixProfile,
    donor_truth: DonorTruth,
    dtw: dict[str, Any],
) -> np.ndarray:
    n_points = len(query.gr_normalized)
    mapping = query_to_donor_mapping(
        dtw["query_path"],
        dtw["donor_path"],
        n_points=n_points,
    )
    grid = np.linspace(0.0, 1.0, n_points)
    donor_progress_at_query = np.interp(query.progress, grid, mapping)
    donor_tvt = np.interp(
        donor_progress_at_query,
        donor_truth.progress,
        donor_truth.tvt,
        left=np.nan,
        right=np.nan,
    )
    donor_origin_progress = float(np.interp(0.0, grid, mapping))
    donor_origin_tvt = float(
        np.interp(
            donor_origin_progress,
            donor_truth.progress,
            donor_truth.tvt,
            left=np.nan,
            right=np.nan,
        )
    )
    prediction = query.anchor_tvt + donor_tvt - donor_origin_tvt
    if not np.isfinite(prediction).all():
        raise ValueError(f"transferred path is not finite for donor {donor_truth.well}")
    if not np.isclose(prediction[0], query.anchor_tvt, rtol=0.0, atol=1e-8):
        raise AssertionError("transferred path is not continuous at query anchor")
    return prediction.astype(np.float32)


def stable_random_donor(query_well: str, eligible_donors: Sequence[str]) -> str:
    ordered = sorted({str(well) for well in eligible_donors})
    if not ordered:
        raise ValueError("stable random control requires an eligible donor")
    digest = hashlib.sha256(str(query_well).encode()).digest()
    index = int.from_bytes(digest[:8], byteorder="big", signed=False) % len(ordered)
    return ordered[index]


# %% [markdown]
# ## 7. Outer-fold candidate generation and target-free freeze


# %%
@dataclass
class CandidateBundle:
    target_free_rows: pd.DataFrame
    donor_predictions: np.ndarray
    donor_rank_by_well: dict[str, list[dict[str, Any]]]
    dtw_diagnostics: pd.DataFrame
    fold_audit: pd.DataFrame


def pair_common_support(
    query: SuffixProfile,
    donor: SuffixProfile,
) -> float:
    return float(np.mean(query.support_mask & donor.support_mask))


def _pair_dtw(
    query: SuffixProfile,
    donor: SuffixProfile,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    minimum_common = float(
        get_nested(config, "gr_similarity.support.min_pair_common_fraction")
    )
    common_fraction = pair_common_support(query, donor)
    if common_fraction < minimum_common:
        return None
    band = int(get_nested(config, "gr_similarity.dtw.sakoe_chiba_band_points"))
    max_run = int(
        get_nested(config, "gr_similarity.dtw.max_consecutive_horizontal_or_vertical")
    )
    try:
        result = constrained_dtw(
            query.gr_normalized,
            donor.gr_normalized,
            band=band,
            max_run=max_run,
        )
    except ValueError:
        return None
    result["common_support_fraction"] = common_fraction
    return result


def _group_lookup(
    inventory: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    well_meta = inventory[
        ["well", "typewell_group_id"]
    ].drop_duplicates("well", keep="first")
    well_to_group = {
        str(row.well): str(row.typewell_group_id)
        for row in well_meta.itertuples()
        if pd.notna(row.typewell_group_id)
    }
    group_to_wells = {
        str(group): sorted(part["well"].astype(str).tolist())
        for group, part in well_meta.dropna(subset=["typewell_group_id"]).groupby(
            "typewell_group_id", sort=True
        )
    }
    return well_to_group, group_to_wells


def generate_target_free_candidates(
    inventory: pd.DataFrame,
    split_map: dict[int, tuple[set[str], set[str]]],
    safe_profiles: dict[str, SuffixProfile],
    train_dir: Path,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> CandidateBundle:
    if "true_tvt" in inventory.columns or "target" in inventory.columns:
        ledger.query_truth_rows_before_freeze += len(inventory)
        raise RuntimeError("target-free candidate generation received query truth")
    n_rows = len(inventory)
    top_k = int(get_nested(config, "data.donor_pool.top_k"))
    min_donors = int(get_nested(config, "data.donor_pool.min_group_donors"))
    parent_values = inventory[PARENT_REFERENCE].to_numpy(np.float64)
    top1 = parent_values.copy()
    primary = parent_values.copy()
    random_control = parent_values.copy()
    donor_predictions = np.full((n_rows, top_k), np.nan, dtype=np.float32)
    supported = np.zeros(n_rows, dtype=bool)
    eligible_count = np.zeros(n_rows, dtype=np.int16)
    used_count = np.zeros(n_rows, dtype=np.int8)
    top1_donor = np.full(n_rows, "", dtype=object)
    random_donor = np.full(n_rows, "", dtype=object)
    donor_rank_by_well: dict[str, list[dict[str, Any]]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    well_to_group, group_to_wells = _group_lookup(inventory)

    for fold in sorted(split_map):
        outer_train, outer_valid = split_map[fold]
        if outer_train & outer_valid:
            raise AssertionError("donor/query well intersection is non-zero")
        donor_truth_cache: dict[str, DonorTruth] = {}
        fold_query_wells = sorted(outer_valid)
        fold_supported = 0
        fold_pairs = 0
        fold_selected_paths = 0
        for query_well in fold_query_wells:
            query_positions = inventory.index[
                inventory["well"].astype(str).eq(query_well)
            ].to_numpy(np.int64)
            if query_positions.size == 0:
                continue
            query_profile = safe_profiles[query_well]
            group = well_to_group.get(query_well)
            if group is None or not profile_is_valid(query_profile, config):
                donor_rank_by_well[query_well] = []
                continue
            donor_pool = [
                donor
                for donor in group_to_wells.get(group, [])
                if donor in outer_train
                and donor != query_well
                and profile_is_valid(safe_profiles[donor], config)
            ]
            ranked: list[dict[str, Any]] = []
            for donor_well in sorted(donor_pool):
                dtw = _pair_dtw(query_profile, safe_profiles[donor_well], config)
                if dtw is None:
                    continue
                ranked.append({"donor_well": donor_well, **dtw})
            ranked.sort(key=lambda item: (float(item["normalized_cost"]), item["donor_well"]))
            fold_pairs += len(ranked)
            if len(ranked) < min_donors:
                donor_rank_by_well[query_well] = []
                continue
            selected = ranked[:top_k]
            random_well = stable_random_donor(
                query_well, [str(item["donor_well"]) for item in ranked]
            )
            selected_by_well = {str(item["donor_well"]): item for item in selected}
            random_item = next(
                item for item in ranked if str(item["donor_well"]) == random_well
            )
            transfer_items = dict(selected_by_well)
            transfer_items[random_well] = random_item
            transferred: dict[str, np.ndarray] = {}
            for donor_well, item in transfer_items.items():
                if donor_well not in donor_truth_cache:
                    donor_truth_cache[donor_well] = load_outer_train_donor_truth(
                        donor_well,
                        safe_profiles[donor_well],
                        train_dir,
                        fold=fold,
                        outer_train_wells=outer_train,
                        outer_valid_wells=outer_valid,
                        ledger=ledger,
                    )
                transferred[donor_well] = transfer_donor_truth_warp(
                    query_profile,
                    donor_truth_cache[donor_well],
                    item,
                )
            selected_predictions = np.column_stack(
                [transferred[str(item["donor_well"])] for item in selected]
            )
            if selected_predictions.shape[0] != len(query_positions):
                raise AssertionError("transferred path row count mismatch")
            if not np.isfinite(selected_predictions).all():
                raise ValueError("supported donor prediction is not finite")
            donor_predictions[query_positions, : len(selected)] = selected_predictions
            top1[query_positions] = selected_predictions[:, 0]
            primary[query_positions] = np.median(selected_predictions, axis=1)
            random_control[query_positions] = transferred[random_well]
            supported[query_positions] = True
            eligible_count[query_positions] = len(ranked)
            used_count[query_positions] = len(selected)
            top1_donor[query_positions] = str(selected[0]["donor_well"])
            random_donor[query_positions] = random_well
            fold_supported += 1
            fold_selected_paths += len(selected)
            donor_rank_by_well[query_well] = []
            for rank, item in enumerate(selected, start=1):
                record = {
                    "fold": int(fold),
                    "query_well": query_well,
                    "typewell_group_id": group,
                    "path_role": "top_k",
                    "donor_rank": int(rank),
                    "donor_well": str(item["donor_well"]),
                    "stable_random_selected": str(item["donor_well"]) == random_well,
                    "normalized_cost": float(item["normalized_cost"]),
                    "common_support_fraction": float(item["common_support_fraction"]),
                    "path_length": int(item["path_length"]),
                    "max_vertical_run": int(item["max_vertical_run"]),
                    "max_horizontal_run": int(item["max_horizontal_run"]),
                    "query_path_json": json.dumps(
                        item["query_path"].astype(int).tolist(), separators=(",", ":")
                    ),
                    "donor_path_json": json.dumps(
                        item["donor_path"].astype(int).tolist(), separators=(",", ":")
                    ),
                }
                donor_rank_by_well[query_well].append(record)
                diagnostic_rows.append(record)
            if random_well not in selected_by_well:
                random_rank = next(
                    index
                    for index, item in enumerate(ranked, start=1)
                    if str(item["donor_well"]) == random_well
                )
                diagnostic_rows.append(
                    {
                        "fold": int(fold),
                        "query_well": query_well,
                        "typewell_group_id": group,
                        "path_role": "stable_random_control",
                        "donor_rank": int(random_rank),
                        "donor_well": random_well,
                        "stable_random_selected": True,
                        "normalized_cost": float(random_item["normalized_cost"]),
                        "common_support_fraction": float(
                            random_item["common_support_fraction"]
                        ),
                        "path_length": int(random_item["path_length"]),
                        "max_vertical_run": int(random_item["max_vertical_run"]),
                        "max_horizontal_run": int(
                            random_item["max_horizontal_run"]
                        ),
                        "query_path_json": json.dumps(
                            random_item["query_path"].astype(int).tolist(),
                            separators=(",", ":"),
                        ),
                        "donor_path_json": json.dumps(
                            random_item["donor_path"].astype(int).tolist(),
                            separators=(",", ":"),
                        ),
                    }
                )
        fold_rows.append(
            {
                "fold": int(fold),
                "outer_train_wells": len(outer_train),
                "outer_valid_wells": len(outer_valid),
                "donor_query_intersection": len(outer_train & outer_valid),
                "supported_wells": int(fold_supported),
                "eligible_dtw_pairs": int(fold_pairs),
                "selected_donor_paths": int(fold_selected_paths),
            }
        )

    target_free = inventory.copy()
    target_free["supported"] = supported
    target_free["eligible_donor_count"] = eligible_count
    target_free["used_donor_count"] = used_count
    target_free["top1_donor_well"] = top1_donor
    target_free["random_donor_well"] = random_donor
    target_free[TOP1_CANDIDATE] = top1
    target_free[PRIMARY_CANDIDATE] = primary
    target_free[RANDOM_CONTROL] = random_control
    target_free = target_free.loc[:, list(TARGET_FREE_COLUMNS)].copy()
    if target_free.duplicated(["well", "row_idx"]).any() or target_free["id"].duplicated().any():
        raise ValueError("generated target-free rows have duplicate identity")
    if not np.isfinite(target_free[list(DEPLOYABLE_CANDIDATES)].to_numpy(np.float64)).all():
        raise ValueError("deployable candidate/fallback coverage is not finite")
    return CandidateBundle(
        target_free_rows=target_free,
        donor_predictions=donor_predictions,
        donor_rank_by_well=donor_rank_by_well,
        dtw_diagnostics=pd.DataFrame(diagnostic_rows),
        fold_audit=pd.DataFrame(fold_rows),
    )


def write_donor_prediction_long(
    bundle: CandidateBundle,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = True
    rows_written = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                base = bundle.target_free_rows
                for rank in range(bundle.donor_predictions.shape[1]):
                    values = bundle.donor_predictions[:, rank]
                    mask = np.isfinite(values)
                    if not mask.any():
                        continue
                    donor_by_well = {
                        well: records[rank]["donor_well"]
                        for well, records in bundle.donor_rank_by_well.items()
                        if len(records) > rank
                    }
                    part = base.loc[
                        mask, ["id", "well", "row_idx", "fold"]
                    ].copy()
                    part["donor_rank"] = rank + 1
                    part["donor_well"] = part["well"].map(donor_by_well)
                    part["analog_prediction"] = values[mask]
                    if part["donor_well"].isna().any():
                        raise AssertionError("donor long artifact lost donor identity")
                    part.to_csv(text, index=False, header=header)
                    header = False
                    rows_written += len(part)
    return {
        "path": str(path),
        "rows": int(rows_written),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_path(path, decompressed=True),
    }


def freeze_target_free_candidates(
    bundle: CandidateBundle,
    input_manifest: dict[str, Any],
    scientific_contract: dict[str, Any],
    artifacts_dir: Path,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    target_free = bundle.target_free_rows
    logical_sha = logical_frame_sha256(target_free, TARGET_FREE_COLUMNS)
    schema_sha = schema_sha256(target_free)
    ledger.mark_frozen(logical_sha)
    target_free_path = artifacts_dir / f"{OUTPUT_PREFIX}_target_free_candidate_freeze.csv.gz"
    diagnostics_path = artifacts_dir / f"{OUTPUT_PREFIX}_dtw_path_diagnostics.csv.gz"
    fold_audit_path = artifacts_dir / f"{OUTPUT_PREFIX}_fold_separation_audit.csv"
    donor_long_path = artifacts_dir / f"{OUTPUT_PREFIX}_analog_donor_paths.csv.gz"
    input_manifest_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.json"
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_target_free_freeze_contract.json"
    write_deterministic_gzip_csv(target_free, target_free_path)
    write_deterministic_gzip_csv(bundle.dtw_diagnostics, diagnostics_path)
    bundle.fold_audit.to_csv(fold_audit_path, index=False)
    donor_long_manifest = write_donor_prediction_long(bundle, donor_long_path)
    write_json(input_manifest_path, input_manifest)
    expected_content = get_nested(
        config, "reproducibility.expected_target_free_content_sha256"
    )
    deterministic_match = bool(expected_content and expected_content == logical_sha)
    freeze = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "rows": len(target_free),
        "wells": int(target_free["well"].nunique()),
        "supported_rows": int(target_free["supported"].sum()),
        "supported_wells": int(
            target_free.groupby("well", sort=False)["supported"].any().sum()
        ),
        "schema_sha256": schema_sha,
        "logical_content_sha256": logical_sha,
        "expected_rerun_content_sha256": expected_content,
        "deterministic_content_sha_match": deterministic_match,
        "determinism_status": (
            "matched_independent_reference"
            if deterministic_match
            else "pending_independent_rerun_reference"
        ),
        "truth_access_ledger_at_freeze": ledger.snapshot(),
        "artifacts": {
            "target_free_candidate_freeze": {
                "path": str(target_free_path),
                "raw_sha256": sha256_path(target_free_path),
                "decompressed_sha256": sha256_path(
                    target_free_path, decompressed=True
                ),
            },
            "dtw_path_diagnostics": {
                "path": str(diagnostics_path),
                "raw_sha256": sha256_path(diagnostics_path),
                "decompressed_sha256": sha256_path(
                    diagnostics_path, decompressed=True
                ),
            },
            "fold_separation_audit": {
                "path": str(fold_audit_path),
                "raw_sha256": sha256_path(fold_audit_path),
            },
            "analog_donor_paths": donor_long_manifest,
            "input_manifest": {
                "path": str(input_manifest_path),
                "raw_sha256": sha256_path(input_manifest_path),
            },
        },
    }
    write_json(freeze_path, freeze)
    freeze["artifacts"]["freeze_contract"] = {
        "path": str(freeze_path),
        "raw_sha256": sha256_path(freeze_path),
    }
    return freeze


# %% [markdown]
# ## 8. Late truth attachment and diagnostic readouts


# %%
def load_query_truth_after_freeze(
    exp099_path: Path,
    frozen: pd.DataFrame,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    ledger.require_frozen()
    columns = ["id", "well", "target", "last_known_tvt"]
    truth = pd.read_csv(
        exp099_path,
        usecols=columns,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    truth["target"] = pd.to_numeric(truth["target"], errors="raise")
    truth["last_known_tvt"] = pd.to_numeric(truth["last_known_tvt"], errors="raise")
    truth["true_tvt"] = truth["last_known_tvt"] + truth["target"]
    truth = truth[["id", "well", "true_tvt"]]
    if truth.duplicated(["id", "well"]).any() or not np.isfinite(truth["true_tvt"]).all():
        raise ValueError("post-freeze query truth is invalid")
    ledger.record_query_truth_after_freeze(len(truth))
    merged = frozen.merge(truth, on=["id", "well"], how="left", validate="one_to_one")
    if merged["true_tvt"].isna().any() or len(merged) != len(frozen):
        raise ValueError("query truth row identity coverage is incomplete")
    return merged


def attach_well_oracle(
    readout: pd.DataFrame,
    donor_predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(readout) != donor_predictions.shape[0]:
        raise ValueError("donor prediction matrix row count mismatch")
    result = readout.copy()
    oracle = result[PARENT_REFERENCE].to_numpy(np.float64).copy()
    selected_rank = np.zeros(len(result), dtype=np.int8)
    donor_metric_rows: list[dict[str, Any]] = []
    truth = result["true_tvt"].to_numpy(np.float64)
    for well, positions_index in result.groupby("well", sort=False).groups.items():
        positions = np.asarray(list(positions_index), dtype=np.int64)
        fold = int(result.loc[positions[0], "fold"])
        candidates = donor_predictions[positions]
        valid_ranks = [
            rank
            for rank in range(candidates.shape[1])
            if np.isfinite(candidates[:, rank]).all()
        ]
        if not valid_ranks:
            continue
        rmses: list[tuple[float, int]] = []
        for rank in valid_ranks:
            error = candidates[:, rank].astype(np.float64) - truth[positions]
            rmse = float(np.sqrt(np.mean(error**2)))
            rmses.append((rmse, rank))
            donor_metric_rows.append(
                {
                    "well": str(well),
                    "fold": fold,
                    "donor_rank": rank + 1,
                    "rows": len(positions),
                    "sse": float(np.sum(error**2)),
                    "path_rmse": rmse,
                }
            )
        best_rmse, best_rank = min(rmses, key=lambda item: (item[0], item[1]))
        del best_rmse
        oracle[positions] = candidates[:, best_rank]
        selected_rank[positions] = best_rank + 1
    result[ORACLE_CANDIDATE] = oracle
    result["oracle_selected_rank"] = selected_rank
    return result, pd.DataFrame(donor_metric_rows)


def metric_record(
    frame: pd.DataFrame,
    candidate: str,
    *,
    scope: str,
) -> dict[str, Any]:
    truth = pd.to_numeric(frame["true_tvt"], errors="raise").to_numpy(np.float64)
    prediction = pd.to_numeric(frame[candidate], errors="coerce").to_numpy(np.float64)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if not mask.any():
        return {
            "scope": scope,
            "candidate": candidate,
            "rows": 0,
            "wells": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "within10": np.nan,
        }
    error = prediction[mask] - truth[mask]
    return {
        "scope": scope,
        "candidate": candidate,
        "rows": int(mask.sum()),
        "wells": int(frame.loc[mask, "well"].nunique()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def build_scope_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    scopes: dict[str, pd.Series] = {
        "overall": pd.Series(True, index=readout.index),
        "1000_plus": readout["md_since"].ge(1000.0),
    }
    role_columns = get_nested(config, "evaluation.hidden_like_role_columns") or {}
    if not hidden_assignments.empty:
        role_table = hidden_assignments.set_index("well_id")
        for scope, role_column in role_columns.items():
            valid_wells = set(
                role_table.index[
                    role_table[str(role_column)].astype(str).eq("valid")
                ].astype(str)
            )
            scopes[str(scope)] = readout["well"].astype(str).isin(valid_wells)
    rows = [
        metric_record(readout.loc[mask], candidate, scope=scope)
        for scope, mask in scopes.items()
        for candidate in EVALUATION_CANDIDATES
    ]
    return pd.DataFrame(rows)


def build_fold_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            **metric_record(part, candidate, scope=f"fold_{int(fold)}"),
            "fold": int(fold),
        }
        for fold, part in readout.groupby("fold", sort=True)
        for candidate in EVALUATION_CANDIDATES
    ]
    return pd.DataFrame(rows)


def build_by_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in readout.groupby("well", sort=True):
        fold = int(part["fold"].iloc[0])
        values = {
            candidate: metric_record(part, candidate, scope=str(well))["rmse"]
            for candidate in EVALUATION_CANDIDATES
        }
        row: dict[str, Any] = {
            "well": str(well),
            "fold": fold,
            "rows": len(part),
            "supported": bool(part["supported"].all()),
        }
        for candidate, rmse in values.items():
            row[f"rmse__{candidate}"] = rmse
        row["primary_minus_exp109_rmse"] = (
            values[PRIMARY_CANDIDATE] - values[PARENT_REFERENCE]
        )
        row["top1_minus_random_rmse"] = (
            values[TOP1_CANDIDATE] - values[RANDOM_CONTROL]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    left = pd.Series(x, dtype=float)
    right = pd.Series(y, dtype=float)
    mask = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return np.nan
    return float(left[mask].rank(method="average").corr(right[mask].rank(method="average")))


def build_dtw_error_readout(
    diagnostics: pd.DataFrame,
    donor_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "query_well", "donor_rank"]
    metrics = donor_metrics.rename(columns={"well": "query_well"})
    selected = diagnostics.loc[diagnostics["path_role"].eq("top_k")].copy()
    merged = selected.merge(metrics, on=keys, how="left", validate="one_to_one")
    if merged["path_rmse"].isna().any():
        raise ValueError("donor error readout did not cover every selected path")
    rows = [
        {
            "scope": "pooled",
            "fold": None,
            "pairs": len(merged),
            "spearman_dtw_cost_vs_path_rmse": spearman_rank_correlation(
                merged["normalized_cost"], merged["path_rmse"]
            ),
        }
    ]
    for fold, part in merged.groupby("fold", sort=True):
        rows.append(
            {
                "scope": f"fold_{int(fold)}",
                "fold": int(fold),
                "pairs": len(part),
                "spearman_dtw_cost_vs_path_rmse": spearman_rank_correlation(
                    part["normalized_cost"], part["path_rmse"]
                ),
            }
        )
    return merged, pd.DataFrame(rows)


def build_donor_rank_metrics(donor_readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, part in donor_readout.groupby("donor_rank", sort=True):
        total_rows = int(part["rows"].sum())
        rows.append(
            {
                "donor_rank": int(rank),
                "wells": int(part["query_well"].nunique()),
                "rows": total_rows,
                "pooled_rmse": float(
                    np.sqrt(part["sse"].sum() / max(total_rows, 1))
                ),
                "mean_well_rmse": float(part["path_rmse"].mean()),
                "median_well_rmse": float(part["path_rmse"].median()),
                "mean_dtw_cost": float(part["normalized_cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 9. Technical/scientific gates and generated artifacts


# %%
def _metric_lookup(
    metrics: pd.DataFrame,
    *,
    scope: str,
    candidate: str,
) -> float:
    selected = metrics.loc[
        metrics["scope"].astype(str).eq(scope)
        & metrics["candidate"].astype(str).eq(candidate),
        "rmse",
    ]
    if len(selected) != 1:
        raise ValueError(f"metric lookup is not unique for {scope}/{candidate}")
    return float(selected.iloc[0])


def evaluate_technical_gate(
    bundle: CandidateBundle,
    freeze: dict[str, Any],
    ledger: TruthAccessLedger,
    input_manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = bundle.target_free_rows
    supported_well_fraction = float(
        rows.groupby("well", sort=False)["supported"].any().mean()
    )
    supported_row_fraction = float(rows["supported"].mean())
    supported_prediction_finite = float(
        np.isfinite(
            rows.loc[rows["supported"], [TOP1_CANDIDATE, PRIMARY_CANDIDATE]].to_numpy(
                np.float64
            )
        ).mean()
    )
    thresholds = get_nested(config, "success_gates.technical")
    checks = {
        "all_input_sha_matches": all(
            bool(item.get("sha_match", True))
            for item in input_manifest["inputs"].values()
            if isinstance(item, dict)
        ),
        "donor_query_intersection_zero": bool(
            bundle.fold_audit["donor_query_intersection"].eq(0).all()
        ),
        "query_truth_reads_before_freeze_zero": (
            ledger.query_truth_rows_before_freeze
            <= int(thresholds["query_truth_reads_before_freeze_max"])
        ),
        "row_identity_unique": not rows.duplicated(["well", "row_idx"]).any()
        and not rows["id"].duplicated().any(),
        "supported_well_fraction": supported_well_fraction
        >= float(thresholds["supported_well_fraction_min"]),
        "supported_score_row_fraction": supported_row_fraction
        >= float(thresholds["supported_score_row_fraction_min"]),
        "supported_path_finite_fraction": supported_prediction_finite
        >= float(thresholds["supported_path_finite_fraction_min"]),
        "deterministic_content_sha_match": bool(
            freeze["deterministic_content_sha_match"]
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail_or_pending_rerun",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "supported_well_fraction": supported_well_fraction,
        "supported_score_row_fraction": supported_row_fraction,
        "supported_path_finite_fraction": supported_prediction_finite,
        "determinism_status": freeze["determinism_status"],
    }


def evaluate_scientific_gate(
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    dtw_spearman: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = get_nested(config, "success_gates.scientific")
    oracle_gain = _metric_lookup(
        scope_metrics, scope="overall", candidate=PARENT_REFERENCE
    ) - _metric_lookup(scope_metrics, scope="overall", candidate=ORACLE_CANDIDATE)
    top1_random_gain = _metric_lookup(
        scope_metrics, scope="overall", candidate=RANDOM_CONTROL
    ) - _metric_lookup(scope_metrics, scope="overall", candidate=TOP1_CANDIDATE)
    primary_gain = _metric_lookup(
        scope_metrics, scope="overall", candidate=PARENT_REFERENCE
    ) - _metric_lookup(scope_metrics, scope="overall", candidate=PRIMARY_CANDIDATE)

    def nonworse_folds(candidate: str, reference: str) -> int:
        candidate_rows = fold_metrics.loc[
            fold_metrics["candidate"].eq(candidate), ["fold", "rmse"]
        ].rename(columns={"rmse": "candidate_rmse"})
        reference_rows = fold_metrics.loc[
            fold_metrics["candidate"].eq(reference), ["fold", "rmse"]
        ].rename(columns={"rmse": "reference_rmse"})
        joined = candidate_rows.merge(reference_rows, on="fold", validate="one_to_one")
        return int((joined["candidate_rmse"] <= joined["reference_rmse"]).sum())

    oracle_nonworse = nonworse_folds(ORACLE_CANDIDATE, PARENT_REFERENCE)
    top1_nonworse = nonworse_folds(TOP1_CANDIDATE, RANDOM_CONTROL)
    primary_nonworse = nonworse_folds(PRIMARY_CANDIDATE, PARENT_REFERENCE)
    pooled_spearman = float(
        dtw_spearman.loc[dtw_spearman["scope"].eq("pooled"), "spearman_dtw_cost_vs_path_rmse"].iloc[
            0
        ]
    )
    positive_spearman_folds = int(
        (
            dtw_spearman.loc[dtw_spearman["fold"].notna(), "spearman_dtw_cost_vs_path_rmse"]
            > 0.0
        ).sum()
    )
    scope_deltas = {
        scope: _metric_lookup(
            scope_metrics, scope=scope, candidate=PRIMARY_CANDIDATE
        )
        - _metric_lookup(scope_metrics, scope=scope, candidate=PARENT_REFERENCE)
        for scope in (
            "1000_plus",
            "hidden_like_spatial",
            "hidden_like_typewell_purged",
        )
    }
    primary_delta = by_well["primary_minus_exp109_rmse"].to_numpy(np.float64)
    p95_delta = float(np.quantile(primary_delta, 0.95))
    worst_delta = float(np.max(primary_delta))
    checks = {
        "oracle_overall_gain": oracle_gain
        >= float(thresholds["oracle_vs_exp109_rmse_gain_min_ft"]),
        "oracle_fold_consistency": oracle_nonworse
        >= int(thresholds["oracle_nonworse_folds_min"]),
        "dtw_cost_error_spearman": pooled_spearman
        >= float(thresholds["dtw_cost_error_spearman_min"]),
        "dtw_spearman_fold_consistency": positive_spearman_folds
        >= int(thresholds["positive_spearman_folds_min"]),
        "top1_vs_random_gain": top1_random_gain
        >= float(thresholds["top1_vs_random_rmse_gain_min_ft"]),
        "top1_vs_random_fold_consistency": top1_nonworse
        >= int(thresholds["top1_vs_random_nonworse_folds_min"]),
        "primary_overall_gain": primary_gain
        >= float(thresholds["primary_vs_exp109_rmse_gain_min_ft"]),
        "primary_fold_consistency": primary_nonworse
        >= int(thresholds["primary_nonworse_folds_min"]),
        "bucket_1000_plus_nonworse": scope_deltas["1000_plus"]
        <= float(thresholds["bucket_1000_plus_delta_max_ft"]),
        "hidden_like_spatial_nonworse": scope_deltas["hidden_like_spatial"]
        <= float(thresholds["hidden_like_delta_max_ft"]),
        "hidden_like_typewell_purged_nonworse": scope_deltas[
            "hidden_like_typewell_purged"
        ]
        <= float(thresholds["hidden_like_delta_max_ft"]),
        "by_well_p95_safety": p95_delta
        <= float(thresholds["by_well_delta_p95_max_ft"]),
        "by_well_worst_safety": worst_delta
        <= float(thresholds["by_well_delta_worst_max_ft"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "oracle_gain_vs_exp109_ft": oracle_gain,
        "oracle_nonworse_folds": oracle_nonworse,
        "dtw_cost_error_spearman": pooled_spearman,
        "positive_spearman_folds": positive_spearman_folds,
        "top1_gain_vs_random_ft": top1_random_gain,
        "top1_nonworse_folds": top1_nonworse,
        "primary_gain_vs_exp109_ft": primary_gain,
        "primary_nonworse_folds": primary_nonworse,
        "scope_primary_minus_exp109_rmse_ft": scope_deltas,
        "by_well_primary_minus_exp109_p95_ft": p95_delta,
        "by_well_primary_minus_exp109_worst_ft": worst_delta,
    }


def decide_result(
    technical: dict[str, Any],
    scientific: dict[str, Any],
) -> str:
    if not technical["passed"]:
        if (
            technical["determinism_status"] == "pending_independent_rerun_reference"
            and all(
                value
                for key, value in technical["checks"].items()
                if key != "deterministic_content_sha_match"
            )
        ):
            return "technical_pass_pending_independent_determinism_rerun"
        return "invalid_technical_gate_failed"
    if not scientific["checks"]["oracle_overall_gain"] or not scientific["checks"][
        "oracle_fold_consistency"
    ]:
        return "close_truth_warp_transfer"
    if not scientific["passed"]:
        return "headroom_only_selection_failed"
    return "all_gates_pass_design_separate_test_parity_experiment"


def save_readout_artifacts(
    readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    donor_readout: pd.DataFrame,
    donor_rank_metrics: pd.DataFrame,
    dtw_spearman: pd.DataFrame,
    technical: dict[str, Any],
    scientific: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    paths = {
        "oof_predictions": artifacts_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
        "scope_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_scope_metrics.csv",
        "fold_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "by_well": artifacts_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "donor_error_readout": artifacts_dir
        / f"{OUTPUT_PREFIX}_donor_error_readout.csv.gz",
        "donor_rank_metrics": artifacts_dir
        / f"{OUTPUT_PREFIX}_donor_rank_metrics.csv",
        "dtw_spearman": artifacts_dir / f"{OUTPUT_PREFIX}_dtw_spearman.csv",
        "technical_gate": artifacts_dir / f"{OUTPUT_PREFIX}_technical_gate.json",
        "scientific_gate": artifacts_dir / f"{OUTPUT_PREFIX}_scientific_gate.json",
    }
    write_deterministic_gzip_csv(readout, paths["oof_predictions"])
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    write_deterministic_gzip_csv(donor_readout, paths["donor_error_readout"])
    donor_rank_metrics.to_csv(paths["donor_rank_metrics"], index=False)
    dtw_spearman.to_csv(paths["dtw_spearman"], index=False)
    write_json(paths["technical_gate"], technical)
    write_json(paths["scientific_gate"], scientific)
    manifest: dict[str, Any] = {}
    for key, path in paths.items():
        manifest[key] = {
            "path": str(path),
            "raw_sha256": sha256_path(path),
            "decompressed_sha256": (
                sha256_path(path, decompressed=True) if path.suffix == ".gz" else None
            ),
        }
    return manifest


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    scientific_contract = validate_scientific_contract(
        config, require_run_approval=True
    )
    if not is_kaggle_runtime():
        raise RuntimeError("exp423 audit must run in the approved Kaggle CPU notebook")
    artifacts_dir = KAGGLE_WORKING_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ledger = TruthAccessLedger()

    safe, exp099_manifest, exp099_path = load_target_free_inventory(config)
    groups, exp065_manifest = load_typewell_groups(config)
    parent, exp109_manifest = load_parent_reference(config)
    inventory, split_map = build_target_free_row_inventory(
        safe, groups, parent, config
    )
    train_dir = resolve_raw_train_dir(config)
    safe_profiles = {
        str(well): load_safe_suffix_profile(
            str(well), part, train_dir, config
        )
        for well, part in inventory.groupby("well", sort=True)
    }
    raw_manifest = build_raw_horizontal_manifest(
        train_dir, inventory["well"].unique().tolist()
    )
    input_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "exp099_feature_cache": exp099_manifest,
            "exp065_cluster_assignments": exp065_manifest,
            "exp109_oof": exp109_manifest,
            "raw_train": raw_manifest,
        },
        "row_inventory_logical_sha256": logical_frame_sha256(
            inventory,
            ["id", "well", "row_idx", "fold", "md_since", "last_known_tvt"],
        ),
        "config_sha256": canonical_json_sha256(config),
    }
    bundle = generate_target_free_candidates(
        inventory,
        split_map,
        safe_profiles,
        train_dir,
        config,
        ledger,
    )
    freeze = freeze_target_free_candidates(
        bundle,
        input_manifest,
        scientific_contract,
        artifacts_dir,
        config,
        ledger,
    )
    readout = load_query_truth_after_freeze(
        exp099_path, bundle.target_free_rows, ledger
    )
    readout, donor_metrics = attach_well_oracle(
        readout, bundle.donor_predictions
    )
    hidden, hidden_manifest = load_hidden_like_assignments(config)
    input_manifest["inputs"]["exp115_hidden_like_assignments"] = hidden_manifest
    scope_metrics = build_scope_metrics(readout, hidden, config)
    fold_metrics = build_fold_metrics(readout)
    by_well = build_by_well_metrics(readout)
    donor_readout, dtw_spearman = build_dtw_error_readout(
        bundle.dtw_diagnostics, donor_metrics
    )
    donor_rank_metrics = build_donor_rank_metrics(donor_readout)
    technical = evaluate_technical_gate(
        bundle, freeze, ledger, input_manifest, config
    )
    scientific = evaluate_scientific_gate(
        scope_metrics, fold_metrics, by_well, dtw_spearman, config
    )
    decision = decide_result(technical, scientific)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - started,
        "route": "pf_beam",
        "stage": "zero_model_train_side_oof_readout",
        "rows": len(readout),
        "wells": int(readout["well"].nunique()),
        "scientific_contract": scientific_contract,
        "freeze": freeze,
        "postfreeze_inputs": {
            "exp115_hidden_like_assignments": hidden_manifest,
        },
        "truth_access_ledger_final": ledger.snapshot(),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "decision": decision,
        "execution_counts": execution_counts(config),
        "submission_in_scope": False,
    }
    artifact_manifest = save_readout_artifacts(
        readout,
        scope_metrics,
        fold_metrics,
        by_well,
        donor_readout,
        donor_rank_metrics,
        dtw_spearman,
        technical,
        scientific,
        artifacts_dir,
    )
    summary["readout_artifacts"] = artifact_manifest
    write_json(
        artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
        summary,
    )
    write_json(
        KAGGLE_WORKING_ROOT / "metrics.json",
        {
            "experiment": EXPERIMENT_NAME,
            "status": "completed_train_side_readout",
            "route": "pf_beam",
            "decision": decision,
            "technical_gate_passed": technical["passed"],
            "scientific_gate_passed": scientific["passed"],
            "primary_gain_vs_exp109_ft": scientific[
                "primary_gain_vs_exp109_ft"
            ],
            "oracle_gain_vs_exp109_ft": scientific["oracle_gain_vs_exp109_ft"],
            "dtw_cost_error_spearman": scientific["dtw_cost_error_spearman"],
            "rows": len(readout),
            "wells": int(readout["well"].nunique()),
            "public_lb": None,
            "private_lb": None,
            "submission_in_scope": False,
        },
    )
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
CONFIG = load_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "primary_candidate": PRIMARY_CANDIDATE,
            "execution_counts": execution_counts(CONFIG),
            "audit_run_approved": bool(
                get_nested(CONFIG, "execution.audit_run_approved")
            ),
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
        },
        indent=2,
        sort_keys=True,
    )
)


# %% [markdown]
# ## 11. Run the approved Kaggle CPU readout
#
# The user explicitly approved the Kaggle CPU audit on 2026-07-28.
# `execution.audit_run_approved` is the fail-closed runtime guard. Inference
# and submission remain outside this experiment.


# %%
SUMMARY: dict[str, Any] | None = None
if os.environ.get("EXP423_IMPORT_ONLY") != "1":
    if is_kaggle_runtime():
        SUMMARY = run_audit(CONFIG)
        print(json.dumps(to_jsonable(SUMMARY), indent=2, sort_keys=True))
    else:
        print(
            "Implementation loaded. Full local execution is disabled; "
            "use the separately approved Kaggle CPU run."
        )
