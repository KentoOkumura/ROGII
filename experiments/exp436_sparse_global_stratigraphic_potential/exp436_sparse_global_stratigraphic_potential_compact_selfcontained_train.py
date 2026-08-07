# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp436 sparse global stratigraphic potential
#
# This notebook fits six formation-specific global potentials from outer-train
# first contacts. It never reads outer-valid formation columns or suffix truth
# before the candidate prediction bundle is frozen. Stage 0, Stage 1, and
# Stage 2 remain independently authorization-gated and fail closed.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment contract
# 2. Notebook-safe paths, SHA, generated-artifact, and role-read helpers
# 3. Fold identity and guarded raw-data loaders
# 4. First-contact nodes and deterministic same-formation graphs
# 5. Sparse Huber potential solver
# 6. Fixed-support potential query and anchored path generation
# 7. Stage 0 resource/integrity preflight
# 8. Stage 1 rolling-origin and target-free OOF freeze
# 9. Stage 2 truth-late direct OOF readout
# 10. Guarded execution and configuration preview

# %% [markdown]
# ## 1. Imports and immutable experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import yaml  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.sparse.linalg import lsqr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

EXPERIMENT_NAME = "exp436_sparse_global_stratigraphic_potential"
OUTPUT_PREFIX = EXPERIMENT_NAME
FORMATION_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
FORMATION_INDEX = {name: index for index, name in enumerate(FORMATION_NAMES)}
SOURCE_COLUMNS = ("MD", "X", "Y", "Z", "TVT", *FORMATION_NAMES)
TARGET_SAFE_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TARGET_FORBIDDEN_COLUMNS = {
    "TVT",
    "TVT_suffix",
    "GR",
    *FORMATION_NAMES,
}
FOLD_IDENTITY_COLUMNS = ("well_id", "row_idx", "suffix_offset", "fold")
EXP226_LATE_COLUMNS = (
    "well_id",
    "row_idx",
    "suffix_offset",
    "tvt_true",
    "tvt_pred",
    "fold",
)
EXECUTE_NOTEBOOK = os.environ.get("EXP436_IMPORT_ONLY", "0") != "1"
EXPECTED_COUNTS = {
    "scientific_candidates": 1,
    "report_only_single_formation_paths": 6,
    "reporting_folds": 5,
    "global_surface_fits": 30,
    "maximum_sparse_solves_including_irls": 180,
    "fitted_ml_models": 0,
    "lightgbm_configs": 0,
    "trained_ml_folds": 0,
    "boosters": 0,
    "hmm_runs": 0,
    "pf_runs": 0,
    "beam_runs": 0,
    "parent_control_regeneration": 0,
}


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = {
        "field.graph_edges.nearest_unique_wells": 8,
        "field.graph_edges.maximum_distance_ft": 4000.0,
        "field.graph_edges.kernel_bandwidth_ft": 1000.0,
        "field.objective.huber_delta": 1.345,
        "field.objective.irls_updates": 5,
        "field.objective.graph_smoothness_weight": 1.0,
        "field.objective.graph_laplacian_bending_weight": 0.05,
        "field.objective.ridge_weight": 1.0e-8,
        "field.solver.atol": 1.0e-6,
        "field.solver.btol": 1.0e-6,
        "field.solver.iteration_limit": 2000,
        "query.control_md_stride_ft": 64.0,
        "query.maximum_unique_source_wells_per_formation": 16,
        "query.minimum_unique_source_wells_per_formation": 8,
        "query.maximum_distance_ft": 4000.0,
        "query.formation_support.minimum_fixed_formations_per_well": 4,
    }
    for path, expected in fixed.items():
        actual = get_nested(config, path)
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"frozen scientific contract changed at {path}")
    if tuple(get_nested(config, "contact_observations.formation_order", ())) != (
        FORMATION_NAMES
    ):
        raise ValueError("formation order changed")
    if get_nested(config, "contact_observations.selection") != (
        "first_crossing_in_increasing_md"
    ):
        raise ValueError("first-contact selection changed")
    if get_nested(config, "query.prediction.primary_delta_formula") != (
        "equal_weight_mean_over_fixed_supported_formations"
    ):
        raise ValueError("primary fixed equal-weight aggregation changed")
    if bool(get_nested(config, "model.uses_target_formation")):
        raise ValueError("target formation use is forbidden")
    if bool(get_nested(config, "model.uses_gr")):
        raise ValueError("GR use is forbidden")
    if bool(get_nested(config, "model.uses_parent_prediction_in_candidate")):
        raise ValueError("parent fallback/blend is forbidden")
    if bool(get_nested(config, "design.same_oof_rescue_allowed")):
        raise ValueError("same-OOF rescue is forbidden")
    return {
        "formations": list(FORMATION_NAMES),
        "contact": "outer_train_first_Z_minus_F_zero",
        "fields_per_fold": 6,
        "folds": 5,
        "total_fields": 30,
        "primary": "fixed_supported_equal_weight_anchor_difference",
        "target_raw_allowlist": list(TARGET_SAFE_COLUMNS),
        "target_formation_reads": 0,
        "gr_reads": 0,
        "parent_fallback": False,
    }


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_stage0_authorization: bool,
) -> dict[str, int]:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp436 route must remain pf_beam")
    if not bool(get_nested(config, "design.implementation_authorized")):
        raise RuntimeError("exp436 implementation is not authorized")
    if not bool(get_nested(config, "implementation.enabled")):
        raise RuntimeError("exp436 implementation is disabled")
    validate_scientific_contract(config)
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1))
        for key in EXPECTED_COUNTS
    }
    if observed != EXPECTED_COUNTS:
        raise ValueError(
            f"execution count contract changed: {observed} != {EXPECTED_COUNTS}"
        )
    if bool(get_nested(config, "runtime.inference_enabled")):
        raise ValueError("inference must remain disabled before Stage 2 promotion")
    if bool(get_nested(config, "runtime.submission_enabled")):
        raise ValueError("submission must remain disabled before promotion")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("exp436 must remain CPU-only")
    if bool(get_nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("exp436 must remain offline")
    if require_stage0_authorization:
        required = {
            "runtime.run_approved": bool(get_nested(config, "runtime.run_approved")),
            "authorization.canonical_train_notebook_adopted": bool(
                get_nested(config, "authorization.canonical_train_notebook_adopted")
            ),
            "authorization.kaggle_package_authorized": bool(
                get_nested(config, "authorization.kaggle_package_authorized")
            ),
            "authorization.kaggle_push_authorized": bool(
                get_nested(config, "authorization.kaggle_push_authorized")
            ),
            "authorization.kaggle_execution_authorized": bool(
                get_nested(config, "authorization.kaggle_execution_authorized")
            ),
            "authorization.stage0_run_authorized": bool(
                get_nested(config, "authorization.stage0_run_authorized")
            ),
            "runtime.kaggle.train_run_on_push": bool(
                get_nested(config, "runtime.kaggle.train_run_on_push")
            ),
        }
        missing = [path for path, enabled in required.items() if not enabled]
        if missing:
            raise RuntimeError(
                "Stage 0 execution remains locked; missing authorization: "
                + ", ".join(missing)
            )
    return observed


# %% [markdown]
# ## 2. Notebook-safe paths, SHA, generated-artifact, and role-read helpers

# %%
def project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "project.yml").exists():
            return candidate
    return start


def config_path() -> Path:
    root = project_root()
    candidates = [
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        root / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"exp436 config not found: {candidates}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = config_path() if path is None else path
    with selected.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def runtime_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return project_root() / "experiments" / EXPERIMENT_NAME


def artifacts_dir() -> Path:
    path = runtime_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_path() -> Path:
    return runtime_dir() / "metrics.json"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if isinstance(normalized[column].dtype, pd.CategoricalDtype):
            normalized[column] = normalized[column].astype(str)
        elif normalized[column].dtype == object:
            normalized[column] = normalized[column].map(
                lambda item: "<NA>" if pd.isna(item) else str(item)
            )
    digest = hashlib.sha256()
    digest.update("|".join(map(str, normalized.columns)).encode())
    digest.update("|".join(map(str, normalized.dtypes)).encode())
    digest.update(
        pd.util.hash_pandas_object(normalized, index=False)
        .to_numpy(dtype="<u8", copy=False)
        .tobytes()
    )
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    return stable_json_sha256(
        [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    )


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(to_jsonable(payload), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return {"path": str(path), "file_sha256": sha256_file(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")
    content_sha = sha256_file(path)
    return {
        "path": str(path),
        "rows": len(frame),
        "file_sha256": content_sha,
        "logical_sha256": content_sha,
        "readback_logical_sha256": content_sha,
        "schema_sha256": frame_schema_sha256(frame),
        "stream_readback_verified": True,
    }


def write_deterministic_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
    )
    decompressed_sha = sha256_decompressed_gzip(path)
    return {
        "path": str(path),
        "rows": len(frame),
        "file_sha256": sha256_file(path),
        "decompressed_sha256": decompressed_sha,
        "logical_sha256": decompressed_sha,
        "readback_logical_sha256": decompressed_sha,
        "schema_sha256": frame_schema_sha256(frame),
        "stream_readback_verified": True,
    }


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return value / (1024.0**3)
    return value / (1024.0**2)


def runtime_versions() -> dict[str, Any]:
    return {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "blas_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


@dataclass
class RoleReadLedger:
    records: list[dict[str, Any]] = field(default_factory=list)
    target_free_frozen: bool = False

    def record_source(
        self,
        *,
        fold: int,
        well_id: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        if self.target_free_frozen:
            raise RuntimeError("source contact reads cannot occur after freeze")
        self.records.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "role": "outer_train",
                "phase": "source_contact_extraction",
                "columns": "|".join(map(str, columns)),
                "rows": int(rows),
                "forbidden_hits": 0,
                "after_freeze": False,
            }
        )

    def record_target_safe(
        self,
        *,
        fold: int,
        well_id: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        requested = tuple(map(str, columns))
        forbidden = TARGET_FORBIDDEN_COLUMNS.intersection(requested)
        if forbidden:
            raise ValueError(
                f"target-safe loader rejected forbidden columns: {sorted(forbidden)}"
            )
        if requested != TARGET_SAFE_COLUMNS:
            raise ValueError(
                f"target-safe loader requires exact allowlist {TARGET_SAFE_COLUMNS}"
            )
        if self.target_free_frozen:
            raise RuntimeError("target-safe generation cannot occur after freeze")
        self.records.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "role": "outer_valid",
                "phase": "target_safe_generation",
                "columns": "|".join(requested),
                "rows": int(rows),
                "forbidden_hits": 0,
                "after_freeze": False,
            }
        )

    def freeze(self) -> None:
        if self.target_free_frozen:
            raise RuntimeError("target-free bundle is already frozen")
        self.target_free_frozen = True

    def record_truth_late(
        self,
        *,
        source: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        if not self.target_free_frozen:
            raise RuntimeError("suffix truth cannot be read before prediction freeze")
        self.records.append(
            {
                "fold": -1,
                "well_id": str(source),
                "role": "truth_late",
                "phase": "truth_late_join",
                "columns": "|".join(map(str, columns)),
                "rows": int(rows),
                "forbidden_hits": 0,
                "after_freeze": True,
            }
        )

    def frame(self) -> pd.DataFrame:
        columns = [
            "fold",
            "well_id",
            "role",
            "phase",
            "columns",
            "rows",
            "forbidden_hits",
            "after_freeze",
        ]
        if not self.records:
            return pd.DataFrame(columns=columns)
        return (
            pd.DataFrame(self.records, columns=columns)
            .sort_values(
                ["after_freeze", "fold", "role", "well_id", "phase"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )


# %% [markdown]
# ## 3. Fold identity and guarded raw-data loaders

# %%
@dataclass(frozen=True)
class FoldIdentity:
    rows: pd.DataFrame
    by_well: dict[str, int]
    path: Path
    evidence: dict[str, Any]


@dataclass(frozen=True)
class TargetWell:
    well_id: str
    fold: int
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tvt_input: np.ndarray
    suffix_row_idx: np.ndarray
    suffix_offset: np.ndarray

    @property
    def anchor_row(self) -> int:
        finite = np.flatnonzero(np.isfinite(self.tvt_input))
        if len(finite) == 0:
            raise ValueError(f"{self.well_id} has no TVT_input anchor")
        return int(finite[-1])


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    output: list[Path] = []
    for raw in patterns:
        value = str(raw)
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if any(character in str(path) for character in "*?[]"):
            output.extend(sorted(path.parent.glob(path.name)))
        elif path.exists():
            output.append(path)
    unique: dict[str, Path] = {}
    for path in output:
        unique[str(path.resolve())] = path.resolve()
    return [unique[key] for key in sorted(unique)]


def resolve_file(
    patterns: Sequence[str],
    *,
    label: str,
    expected_file_sha256: str | None = None,
) -> Path:
    paths = [path for path in expand_paths(patterns) if path.is_file()]
    if expected_file_sha256 is not None:
        paths = [
            path
            for path in paths
            if sha256_file(path) == str(expected_file_sha256)
        ]
    if not paths:
        raise FileNotFoundError(f"{label} was not found from {list(patterns)}")
    if len(paths) > 1:
        hashes = {sha256_file(path) for path in paths}
        if len(hashes) != 1:
            raise ValueError(f"{label} resolves to non-identical files: {paths}")
    return paths[0]


def load_fold_identity(config: Mapping[str, Any]) -> FoldIdentity:
    specification = get_nested(config, "data.exp226_oof")
    path = resolve_file(specification["patterns"], label="exp226 saved OOF")
    expected_sha = str(specification["expected_decompressed_sha256"])
    actual_sha = sha256_decompressed_gzip(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"exp226 OOF decompressed SHA mismatch: {actual_sha} != {expected_sha}"
        )
    requested = tuple(map(str, specification["pre_freeze_columns"]))
    if requested != FOLD_IDENTITY_COLUMNS:
        raise ValueError("exp226 pre-freeze allowlist changed")
    rows = pd.read_csv(
        path,
        usecols=list(requested),
        dtype={
            "well_id": str,
            "row_idx": "int32",
            "suffix_offset": "int32",
            "fold": "int8",
        },
    )
    rows["well_id"] = rows["well_id"].astype(str)
    rows = rows.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if rows.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 fold identity has duplicate row keys")
    well_fold = rows[["well_id", "fold"]].drop_duplicates()
    if well_fold["well_id"].duplicated().any():
        raise ValueError("one well maps to multiple outer folds")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = set(map(int, get_nested(config, "validation.expected_folds")))
    if len(rows) != expected_rows:
        raise ValueError(f"fold identity row count {len(rows)} != {expected_rows}")
    if len(well_fold) != expected_wells:
        raise ValueError(f"fold identity well count {len(well_fold)} != {expected_wells}")
    if set(well_fold["fold"].astype(int)) != expected_folds:
        raise ValueError("fold identity inventory differs from the frozen contract")
    by_well = {
        str(row.well_id): int(row.fold)
        for row in well_fold.sort_values("well_id").itertuples(index=False)
    }
    evidence = {
        "phase": "pre_freeze_fold_identity",
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decompressed_sha256": actual_sha,
        "logical_sha256": logical_frame_sha256(rows),
        "schema_sha256": frame_schema_sha256(rows),
        "rows": len(rows),
        "wells": len(by_well),
        "columns_read": list(requested),
        "truth_columns_read": 0,
    }
    return FoldIdentity(rows=rows, by_well=by_well, path=path, evidence=evidence)


def resolve_raw_train(
    config: Mapping[str, Any],
    expected_wells: set[str],
) -> tuple[Path, dict[str, Path]]:
    patterns = get_nested(config, "data.raw_train_dir_patterns", [])
    glob_pattern = str(get_nested(config, "data.raw_horizontal_glob"))
    reports: list[dict[str, Any]] = []
    for directory in [path for path in expand_paths(patterns) if path.is_dir()]:
        files = sorted(directory.glob(glob_pattern))
        mapping = {
            path.name.split("__horizontal_well.csv", 1)[0]: path
            for path in files
        }
        reports.append(
            {
                "directory": str(directory),
                "files": len(files),
                "wells": len(mapping),
            }
        )
        if set(mapping) == expected_wells:
            return directory, mapping
    raise FileNotFoundError(
        f"no raw train directory matches exp226 wells: {reports}"
    )


def load_target_safe(
    path: Path,
    *,
    fold: int,
    identity_rows: pd.DataFrame,
    ledger: RoleReadLedger,
) -> TargetWell:
    frame = pd.read_csv(path, usecols=list(TARGET_SAFE_COLUMNS))
    well_id = path.name.split("__horizontal_well.csv", 1)[0]
    ledger.record_target_safe(
        fold=fold,
        well_id=well_id,
        columns=TARGET_SAFE_COLUMNS,
        rows=len(frame),
    )
    arrays = {
        column: pd.to_numeric(frame[column], errors="raise").to_numpy(np.float64)
        for column in TARGET_SAFE_COLUMNS
    }
    for column in ("MD", "X", "Y", "Z"):
        if not np.isfinite(arrays[column]).all():
            raise ValueError(f"{well_id} target-safe {column} is non-finite")
    if np.any(np.diff(arrays["MD"]) <= 0.0):
        raise ValueError(f"{well_id} MD must be strictly increasing")
    finite = np.isfinite(arrays["TVT_input"])
    if not finite.any():
        raise ValueError(f"{well_id} has no finite TVT_input")
    anchor = int(np.flatnonzero(finite)[-1])
    if not finite[: anchor + 1].all() or finite[anchor + 1 :].any():
        raise ValueError(f"{well_id} TVT_input must be one contiguous prefix")
    unknown = np.flatnonzero(~finite).astype(np.int32)
    expected = identity_rows.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(unknown, expected["row_idx"].to_numpy(np.int32)):
        raise ValueError(f"{well_id} suffix row identity differs from exp226")
    suffix_offset = expected["suffix_offset"].to_numpy(np.int32)
    if not np.array_equal(
        suffix_offset,
        np.arange(len(suffix_offset), dtype=np.int32),
    ):
        raise ValueError(f"{well_id} suffix offsets are not contiguous")
    return TargetWell(
        well_id=well_id,
        fold=int(fold),
        md=arrays["MD"],
        x=arrays["X"],
        y=arrays["Y"],
        z=arrays["Z"],
        tvt_input=arrays["TVT_input"],
        suffix_row_idx=unknown,
        suffix_offset=suffix_offset,
    )


# %% [markdown]
# ## 4. First-contact nodes and deterministic same-formation graphs

# %%
def interpolate_at_fraction(
    values: np.ndarray | None,
    left: int,
    fraction: float,
) -> float:
    if values is None:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if fraction == 0.0 or left == len(array) - 1:
        return float(array[left])
    return float(array[left] + fraction * (array[left + 1] - array[left]))


def first_crossing(
    md: np.ndarray,
    residual: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
    tvt: np.ndarray | None = None,
) -> dict[str, float] | None:
    md_array = np.asarray(md, dtype=np.float64)
    residual_array = np.asarray(residual, dtype=np.float64)
    if len(md_array) != len(residual_array):
        raise ValueError("MD and crossing residual lengths differ")
    if len(md_array) == 0:
        return None
    if not np.isfinite(md_array).all() or np.any(np.diff(md_array) <= 0.0):
        raise ValueError("contact MD must be finite and strictly increasing")
    for left in range(len(md_array)):
        left_value = residual_array[left]
        if not np.isfinite(left_value):
            continue
        if left_value == 0.0:
            fraction = 0.0
        elif left + 1 < len(md_array):
            right_value = residual_array[left + 1]
            if not np.isfinite(right_value) or left_value * right_value >= 0.0:
                continue
            fraction = float(left_value / (left_value - right_value))
        else:
            continue
        return {
            "contact_md": interpolate_at_fraction(md_array, left, fraction),
            "contact_x": interpolate_at_fraction(x, left, fraction),
            "contact_y": interpolate_at_fraction(y, left, fraction),
            "contact_z": interpolate_at_fraction(z, left, fraction),
            "contact_tvt": interpolate_at_fraction(tvt, left, fraction),
            "left_row_idx": float(left),
            "fraction": float(fraction),
        }
    return None


def extract_first_contact_nodes(
    frame: pd.DataFrame,
    *,
    fold: int,
    well_id: str,
) -> pd.DataFrame:
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"source frame misses columns: {sorted(missing)}")
    md = frame["MD"].to_numpy(np.float64)
    x = frame["X"].to_numpy(np.float64)
    y = frame["Y"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    tvt = frame["TVT"].to_numpy(np.float64)
    if not np.isfinite(np.column_stack([md, x, y, z, tvt])).all():
        raise ValueError(f"{well_id} source geometry/TVT is non-finite")
    records: list[dict[str, Any]] = []
    for formation_index, formation in enumerate(FORMATION_NAMES):
        values = frame[formation].to_numpy(np.float64)
        contact = first_crossing(
            md,
            z - values,
            x=x,
            y=y,
            z=z,
            tvt=tvt,
        )
        if contact is None:
            continue
        u_contact = float(contact["contact_tvt"] + contact["contact_z"])
        record = {
            "fold": int(fold),
            "formation": formation,
            "formation_index": int(formation_index),
            "well_id": str(well_id),
            **contact,
            "u_contact": u_contact,
        }
        records.append(record)
    columns = [
        "fold",
        "formation",
        "formation_index",
        "well_id",
        "contact_md",
        "contact_x",
        "contact_y",
        "contact_z",
        "contact_tvt",
        "left_row_idx",
        "fraction",
        "u_contact",
    ]
    return pd.DataFrame(records, columns=columns)


def build_fold_contact_nodes(
    *,
    fold: int,
    source_wells: Sequence[str],
    path_by_well: Mapping[str, Path],
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well_id in sorted(map(str, source_wells)):
        path = path_by_well[well_id]
        frame = pd.read_csv(path, usecols=list(SOURCE_COLUMNS))
        ledger.record_source(
            fold=fold,
            well_id=well_id,
            columns=SOURCE_COLUMNS,
            rows=len(frame),
        )
        parts.append(
            extract_first_contact_nodes(
                frame,
                fold=fold,
                well_id=well_id,
            )
        )
    if not parts:
        raise ValueError(f"fold {fold} has no source contacts")
    nodes = pd.concat(parts, ignore_index=True)
    nodes = nodes.sort_values(
        ["fold", "formation_index", "well_id", "contact_md"],
        kind="mergesort",
    ).reset_index(drop=True)
    if nodes.duplicated(
        ["fold", "formation_index", "well_id", "contact_md"]
    ).any():
        raise ValueError("duplicate contact/node stable key")
    finite_columns = [
        "contact_md",
        "contact_x",
        "contact_y",
        "contact_z",
        "contact_tvt",
        "u_contact",
    ]
    if not np.isfinite(nodes[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("contact catalog contains non-finite values")
    return nodes


def build_graph_edges(
    nodes: pd.DataFrame,
    *,
    nearest_wells: int,
    maximum_distance_ft: float,
    bandwidth_ft: float,
) -> pd.DataFrame:
    required = {
        "fold",
        "formation",
        "formation_index",
        "well_id",
        "contact_md",
        "contact_x",
        "contact_y",
    }
    if required - set(nodes.columns):
        raise ValueError("node catalog lacks graph columns")
    if nodes["fold"].nunique() != 1 or nodes["formation_index"].nunique() != 1:
        raise ValueError("graph builder accepts one fold and one formation")
    ordered = nodes.sort_values(
        ["well_id", "contact_md"],
        kind="mergesort",
    ).reset_index(drop=True)
    if ordered["well_id"].duplicated().any():
        raise ValueError("graph requires at most one node per source well")
    xy = ordered[["contact_x", "contact_y"]].to_numpy(np.float64)
    wells = ordered["well_id"].astype(str).to_numpy()
    contact_md = ordered["contact_md"].to_numpy(np.float64)
    pairs: set[tuple[int, int]] = set()
    for left in range(len(ordered)):
        distance = np.hypot(
            xy[:, 0] - xy[left, 0],
            xy[:, 1] - xy[left, 1],
        )
        candidates = [
            right
            for right in range(len(ordered))
            if right != left
            and wells[right] != wells[left]
            and np.isfinite(distance[right])
            and distance[right] <= float(maximum_distance_ft)
        ]
        candidates.sort(
            key=lambda right: (
                float(distance[right]),
                str(wells[right]),
                float(contact_md[right]),
                int(right),
            )
        )
        for right in candidates[: int(nearest_wells)]:
            pairs.add((min(left, right), max(left, right)))
    records = []
    fold = int(ordered["fold"].iloc[0])
    formation = str(ordered["formation"].iloc[0])
    formation_index = int(ordered["formation_index"].iloc[0])
    for left, right in sorted(pairs):
        distance = float(np.hypot(*(xy[right] - xy[left])))
        weight = float(
            np.exp(-distance**2 / (2.0 * float(bandwidth_ft) ** 2))
        )
        records.append(
            {
                "fold": fold,
                "formation": formation,
                "formation_index": formation_index,
                "left_node": int(left),
                "right_node": int(right),
                "left_well": str(wells[left]),
                "right_well": str(wells[right]),
                "distance_ft": distance,
                "weight": weight,
            }
        )
    columns = [
        "fold",
        "formation",
        "formation_index",
        "left_node",
        "right_node",
        "left_well",
        "right_well",
        "distance_ft",
        "weight",
    ]
    edges = pd.DataFrame(records, columns=columns)
    if edges.duplicated(
        ["fold", "formation_index", "left_node", "right_node"]
    ).any():
        raise ValueError("duplicate graph edge stable key")
    return edges


def graph_component_count(node_count: int, edges: pd.DataFrame) -> int:
    if node_count <= 0:
        return 0
    if edges.empty:
        return int(node_count)
    rows = np.r_[
        edges["left_node"].to_numpy(np.int64),
        edges["right_node"].to_numpy(np.int64),
    ]
    columns = np.r_[
        edges["right_node"].to_numpy(np.int64),
        edges["left_node"].to_numpy(np.int64),
    ]
    adjacency = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(node_count, node_count),
    )
    components, _ = connected_components(adjacency, directed=False)
    return int(components)


# %% [markdown]
# ## 5. Sparse Huber potential solver

# %%
@dataclass(frozen=True)
class SparsePotentialSurface:
    fold: int
    formation: str
    formation_index: int
    nodes: pd.DataFrame
    edges: pd.DataFrame
    solved_u: np.ndarray
    scale: float
    diagnostics: dict[str, Any]


def weighted_incidence_matrix(
    node_count: int,
    edges: pd.DataFrame,
) -> sp.csr_matrix:
    if edges.empty:
        return sp.csr_matrix((0, node_count), dtype=np.float64)
    edge_index = np.arange(len(edges), dtype=np.int64)
    rows = np.repeat(edge_index, 2)
    columns = np.column_stack(
        [
            edges["left_node"].to_numpy(np.int64),
            edges["right_node"].to_numpy(np.int64),
        ]
    ).ravel()
    root_weight = np.sqrt(edges["weight"].to_numpy(np.float64))
    data = np.column_stack([-root_weight, root_weight]).ravel()
    return sp.csr_matrix(
        (data, (rows, columns)),
        shape=(len(edges), node_count),
    )


def solve_sparse_potential(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: Mapping[str, Any],
) -> SparsePotentialSurface:
    minimum_nodes = int(get_nested(config, "field.node.minimum_source_wells_per_formation"))
    if len(nodes) < minimum_nodes:
        raise ValueError(
            f"surface has {len(nodes)} source wells, below {minimum_nodes}"
        )
    ordered = nodes.sort_values(
        ["well_id", "contact_md"],
        kind="mergesort",
    ).reset_index(drop=True)
    if ordered["well_id"].duplicated().any():
        raise ValueError("surface node set has duplicate source wells")
    if edges.empty:
        raise ValueError("surface graph has no edges")
    response = ordered["u_contact"].to_numpy(np.float64)
    center = float(np.median(response))
    scale = max(
        float(np.median(np.abs(response - center))),
        1.0e-6,
    )
    incidence = weighted_incidence_matrix(len(ordered), edges)
    laplacian = (incidence.T @ incidence).tocsr()
    smooth_weight = float(
        get_nested(config, "field.objective.graph_smoothness_weight")
    )
    bending_weight = float(
        get_nested(config, "field.objective.graph_laplacian_bending_weight")
    )
    ridge_weight = float(get_nested(config, "field.objective.ridge_weight"))
    huber_delta = float(get_nested(config, "field.objective.huber_delta"))
    updates = int(get_nested(config, "field.objective.irls_updates"))
    atol = float(get_nested(config, "field.solver.atol"))
    btol = float(get_nested(config, "field.solver.btol"))
    iteration_limit = int(get_nested(config, "field.solver.iteration_limit"))
    accepted = set(map(int, get_nested(config, "field.solver.accepted_istop")))
    observation_weight = np.ones(len(ordered), dtype=np.float64)
    solution = response.copy()
    standardized_response = (response - center) / scale
    iterations: list[dict[str, Any]] = []
    for update_index in range(updates + 1):
        observation = sp.diags(np.sqrt(observation_weight), format="csr")
        blocks = [
            observation,
            incidence * math.sqrt(smooth_weight),
            laplacian * math.sqrt(bending_weight),
            sp.eye(len(ordered), format="csr")
            * math.sqrt(ridge_weight),
        ]
        matrix = sp.vstack(blocks, format="csr")
        rhs = np.r_[
            standardized_response * np.sqrt(observation_weight),
            np.zeros(incidence.shape[0], dtype=np.float64),
            np.zeros(len(ordered), dtype=np.float64),
            np.full(
                len(ordered),
                -math.sqrt(ridge_weight) * center / scale,
                dtype=np.float64,
            ),
        ]
        result = lsqr(
            matrix,
            rhs,
            atol=atol,
            btol=btol,
            iter_lim=iteration_limit,
            show=False,
        )
        standardized_solution = np.asarray(result[0], dtype=np.float64)
        solution = center + scale * standardized_solution
        istop = int(result[1])
        if istop not in accepted:
            raise RuntimeError(
                f"LSQR failed for fold={ordered['fold'].iloc[0]} "
                f"formation={ordered['formation'].iloc[0]}: istop={istop}"
            )
        standardized = (solution - response) / scale
        absolute = np.abs(standardized)
        iterations.append(
            {
                "update": int(update_index),
                "istop": istop,
                "iterations": int(result[2]),
                "r1norm": float(result[3]),
                "r2norm": float(result[4]),
                "anorm": float(result[5]),
                "acond": float(result[6]),
                "arnorm": float(result[7]),
                "xnorm": float(result[8]),
                "max_abs_standardized_observation_residual": float(
                    absolute.max(initial=0.0)
                ),
            }
        )
        if update_index < updates:
            observation_weight = np.ones_like(absolute)
            outside = absolute > huber_delta
            observation_weight[outside] = huber_delta / absolute[outside]
    if not np.isfinite(solution).all():
        raise RuntimeError("sparse potential solution is non-finite")
    diagnostics = {
        "fold": int(ordered["fold"].iloc[0]),
        "formation": str(ordered["formation"].iloc[0]),
        "formation_index": int(ordered["formation_index"].iloc[0]),
        "nodes": len(ordered),
        "edges": len(edges),
        "components": graph_component_count(len(ordered), edges),
        "scale_mad": scale,
        "sparse_solves": updates + 1,
        "accepted": True,
        "solution_min": float(solution.min()),
        "solution_max": float(solution.max()),
        "solution_sha256": hashlib.sha256(
            solution.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "iterations": iterations,
    }
    return SparsePotentialSurface(
        fold=int(ordered["fold"].iloc[0]),
        formation=str(ordered["formation"].iloc[0]),
        formation_index=int(ordered["formation_index"].iloc[0]),
        nodes=ordered,
        edges=edges,
        solved_u=solution,
        scale=scale,
        diagnostics=diagnostics,
    )


# %% [markdown]
# ## 6. Fixed-support potential query and anchored path generation

# %%
def query_sparse_potential(
    surface: SparsePotentialSurface,
    query_xy: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    query = np.atleast_2d(np.asarray(query_xy, dtype=np.float64))
    if query.shape[1] != 2 or not np.isfinite(query).all():
        raise ValueError("surface query must contain finite XY pairs")
    nodes = surface.nodes
    node_xy = nodes[["contact_x", "contact_y"]].to_numpy(np.float64)
    node_wells = nodes["well_id"].astype(str).to_numpy()
    node_md = nodes["contact_md"].to_numpy(np.float64)
    if len(node_xy) != len(surface.solved_u):
        raise ValueError("surface node/value length mismatch")
    maximum_sources = int(
        get_nested(config, "query.maximum_unique_source_wells_per_formation")
    )
    minimum_sources = int(
        get_nested(config, "query.minimum_unique_source_wells_per_formation")
    )
    maximum_distance = float(get_nested(config, "query.maximum_distance_ft"))
    bandwidth_low, bandwidth_high = map(
        float,
        get_nested(config, "query.bandwidth_clip_ft"),
    )
    chunk_rows = int(get_nested(config, "query.query_chunk_rows"))
    tree = cKDTree(node_xy)
    prediction = np.full(len(query), np.nan, dtype=np.float64)
    support_records: list[dict[str, Any]] = []
    for start in range(0, len(query), max(1, chunk_rows)):
        end = min(start + max(1, chunk_rows), len(query))
        candidate_indices = tree.query_ball_point(
            query[start:end],
            r=maximum_distance,
            return_sorted=False,
            workers=1,
        )
        for local_row in range(end - start):
            local_indices = np.asarray(
                candidate_indices[local_row],
                dtype=np.int64,
            )
            local_distance = np.hypot(
                node_xy[local_indices, 0] - query[start + local_row, 0],
                node_xy[local_indices, 1] - query[start + local_row, 1],
            )
            candidates = [
                (float(value), int(node_index))
                for value, node_index in zip(
                    local_distance,
                    local_indices,
                    strict=True,
                )
                if np.isfinite(value)
                and int(node_index) < len(node_xy)
                and float(value) <= maximum_distance
            ]
            candidates.sort(
                key=lambda item: (
                    item[0],
                    str(node_wells[item[1]]),
                    float(node_md[item[1]]),
                    item[1],
                )
            )
            selected: list[tuple[float, int]] = []
            selected_wells: set[str] = set()
            for item in candidates:
                well = str(node_wells[item[1]])
                if well in selected_wells:
                    continue
                selected.append(item)
                selected_wells.add(well)
                if len(selected) == maximum_sources:
                    break
            supported = len(selected) >= minimum_sources
            bandwidth = float("nan")
            if supported:
                selected_distance = np.asarray(
                    [item[0] for item in selected],
                    dtype=np.float64,
                )
                selected_index = np.asarray(
                    [item[1] for item in selected],
                    dtype=np.int64,
                )
                bandwidth = float(
                    np.clip(
                        selected_distance[-1],
                        bandwidth_low,
                        bandwidth_high,
                    )
                )
                weight = np.exp(
                    -np.square(selected_distance)
                    / (2.0 * bandwidth**2)
                )
                denominator = float(weight.sum())
                if denominator > 0.0 and np.isfinite(denominator):
                    prediction[start + local_row] = float(
                        np.dot(weight, surface.solved_u[selected_index])
                        / denominator
                    )
                else:
                    supported = False
            support_records.append(
                {
                    "query_row": int(start + local_row),
                    "fold": int(surface.fold),
                    "formation": surface.formation,
                    "formation_index": int(surface.formation_index),
                    "supported": bool(
                        supported and np.isfinite(prediction[start + local_row])
                    ),
                    "unique_source_wells": int(len(selected)),
                    "nearest_distance_ft": (
                        float(selected[0][0]) if selected else float("nan")
                    ),
                    "farthest_distance_ft": (
                        float(selected[-1][0]) if selected else float("nan")
                    ),
                    "bandwidth_ft": bandwidth,
                }
            )
    return prediction, pd.DataFrame(support_records)


def build_control_points(
    well: TargetWell,
    *,
    start_row: int,
    end_row: int,
    stride_ft: float,
) -> pd.DataFrame:
    if not 0 <= start_row < end_row < len(well.md):
        raise ValueError("invalid control-point interval")
    start_md = float(well.md[start_row])
    end_md = float(well.md[end_row])
    count = int(math.floor((end_md - start_md) / float(stride_ft)))
    control_md = start_md + np.arange(count + 1, dtype=np.float64) * float(
        stride_ft
    )
    control_md = control_md[control_md <= end_md]
    if len(control_md) == 0 or not math.isclose(
        float(control_md[0]),
        start_md,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        control_md = np.r_[start_md, control_md]
    if not math.isclose(
        float(control_md[-1]),
        end_md,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        control_md = np.r_[control_md, end_md]
    return pd.DataFrame(
        {
            "control_index": np.arange(len(control_md), dtype=np.int32),
            "MD": control_md,
            "X": np.interp(control_md, well.md, well.x),
            "Y": np.interp(control_md, well.md, well.y),
        }
    )


def predict_interval(
    well: TargetWell,
    fields: Mapping[str, SparsePotentialSurface],
    *,
    start_row: int,
    row_indices: np.ndarray,
    config: Mapping[str, Any],
    purpose: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.asarray(row_indices, dtype=np.int64)
    if len(rows) == 0:
        raise ValueError("prediction interval is empty")
    if rows[0] <= start_row or np.any(np.diff(rows) <= 0):
        raise ValueError("prediction rows must increase after the anchor")
    controls = build_control_points(
        well,
        start_row=start_row,
        end_row=int(rows[-1]),
        stride_ft=float(get_nested(config, "query.control_md_stride_ft")),
    )
    query_xy = controls[["X", "Y"]].to_numpy(np.float64)
    control_potential: dict[str, np.ndarray] = {}
    support_parts: list[pd.DataFrame] = []
    supported_formations: list[str] = []
    for formation in FORMATION_NAMES:
        if formation not in fields:
            raise ValueError(f"fold field missing formation {formation}")
        values, support = query_sparse_potential(
            fields[formation],
            query_xy,
            config,
        )
        support = support.copy()
        support.insert(0, "well_id", well.well_id)
        support.insert(1, "purpose", purpose)
        support["control_md"] = controls["MD"].to_numpy(np.float64)
        support_parts.append(support)
        control_potential[formation] = values
        if bool(support["supported"].all()) and np.isfinite(values).all():
            supported_formations.append(formation)
    minimum_formations = int(
        get_nested(config, "query.formation_support.minimum_fixed_formations_per_well")
    )
    output = pd.DataFrame(
        {
            "well_id": well.well_id,
            "fold": int(well.fold),
            "row_idx": rows.astype(np.int32),
            "MD": well.md[rows],
            "distance_from_anchor": well.md[rows] - well.md[start_row],
        }
    )
    formation_predictions: dict[str, np.ndarray] = {}
    for formation in FORMATION_NAMES:
        values = control_potential[formation]
        if formation in supported_formations:
            interpolated = np.interp(
                well.md[rows],
                controls["MD"].to_numpy(np.float64),
                values,
            )
            delta_u = interpolated - float(values[0])
            formation_predictions[formation] = (
                float(well.tvt_input[start_row])
                + delta_u
                - (well.z[rows] - well.z[start_row])
            )
        else:
            formation_predictions[formation] = np.full(
                len(rows),
                np.nan,
                dtype=np.float64,
            )
        output[f"tvt_pred_{formation}"] = formation_predictions[formation]
    if len(supported_formations) >= minimum_formations:
        stack = np.column_stack(
            [formation_predictions[name] for name in supported_formations]
        )
        primary = np.mean(stack, axis=1)
    else:
        primary = np.full(len(rows), np.nan, dtype=np.float64)
    output["tvt_pred_exp436"] = primary
    output["supported_formations"] = "|".join(supported_formations)
    output["supported_formation_count"] = int(len(supported_formations))
    output["anchor_row_idx"] = int(start_row)
    output["anchor_tvt_input"] = float(well.tvt_input[start_row])
    support_frame = pd.concat(support_parts, ignore_index=True)
    support_frame["fixed_supported_formation"] = support_frame[
        "formation"
    ].isin(supported_formations)
    support_frame["fixed_supported_formation_count"] = int(
        len(supported_formations)
    )
    return output, support_frame


def stable_sample_wells(
    wells: Sequence[str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    ranked = sorted(
        (
            hashlib.sha256(f"{seed}:{well}".encode()).hexdigest(),
            str(well),
        )
        for well in wells
    )
    return [well for _, well in ranked[: min(int(count), len(ranked))]]


# %% [markdown]
# ## 7. Stage 0 resource/integrity preflight

# %%
@dataclass
class Stage0Bundle:
    identity: FoldIdentity
    path_by_well: dict[str, Path]
    nodes_by_fold: dict[int, pd.DataFrame]
    edges_by_key: dict[tuple[int, str], pd.DataFrame]
    preflight_fields: dict[str, SparsePotentialSurface]
    contact_census: pd.DataFrame
    graph_census: pd.DataFrame
    solver_manifest: pd.DataFrame
    query_support: pd.DataFrame
    preflight_predictions: pd.DataFrame
    role_ledger: RoleReadLedger
    decision: dict[str, Any]
    elapsed_seconds: float


def build_contact_and_graph_census(
    identity: FoldIdentity,
    path_by_well: Mapping[str, Path],
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> tuple[
    dict[int, pd.DataFrame],
    dict[tuple[int, str], pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
]:
    nodes_by_fold: dict[int, pd.DataFrame] = {}
    edges_by_key: dict[tuple[int, str], pd.DataFrame] = {}
    contact_records: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    folds = sorted(set(identity.by_well.values()))
    for fold in folds:
        source_wells = sorted(
            well
            for well, assigned in identity.by_well.items()
            if int(assigned) != int(fold)
        )
        target_wells = {
            well
            for well, assigned in identity.by_well.items()
            if int(assigned) == int(fold)
        }
        overlap = set(source_wells).intersection(target_wells)
        if overlap:
            raise ValueError(f"fold {fold} source/valid overlap: {sorted(overlap)[:5]}")
        nodes = build_fold_contact_nodes(
            fold=int(fold),
            source_wells=source_wells,
            path_by_well=path_by_well,
            ledger=ledger,
        )
        nodes_by_fold[int(fold)] = nodes
        for formation_index, formation in enumerate(FORMATION_NAMES):
            selected = (
                nodes.loc[nodes["formation"].eq(formation)]
                .sort_values(["well_id", "contact_md"], kind="mergesort")
                .reset_index(drop=True)
            )
            edges = build_graph_edges(
                selected,
                nearest_wells=int(
                    get_nested(config, "field.graph_edges.nearest_unique_wells")
                ),
                maximum_distance_ft=float(
                    get_nested(config, "field.graph_edges.maximum_distance_ft")
                ),
                bandwidth_ft=float(
                    get_nested(config, "field.graph_edges.kernel_bandwidth_ft")
                ),
            )
            edges_by_key[(int(fold), formation)] = edges
            contact_records.append(
                {
                    "fold": int(fold),
                    "formation": formation,
                    "formation_index": int(formation_index),
                    "source_wells": int(selected["well_id"].nunique()),
                    "nodes": len(selected),
                    "finite_fraction": float(
                        np.isfinite(
                            selected[
                                [
                                    "contact_md",
                                    "contact_x",
                                    "contact_y",
                                    "contact_z",
                                    "contact_tvt",
                                    "u_contact",
                                ]
                            ].to_numpy(np.float64)
                        ).mean()
                    ),
                    "duplicate_stable_keys": int(
                        selected.duplicated(
                            ["fold", "formation_index", "well_id", "contact_md"]
                        ).sum()
                    ),
                    "outer_valid_overlap": int(
                        selected["well_id"].astype(str).isin(target_wells).sum()
                    ),
                }
            )
            graph_records.append(
                {
                    "fold": int(fold),
                    "formation": formation,
                    "formation_index": int(formation_index),
                    "nodes": len(selected),
                    "edges": len(edges),
                    "components": graph_component_count(len(selected), edges),
                    "isolated_nodes": int(
                        len(selected)
                        - len(
                            set(edges["left_node"].astype(int))
                            | set(edges["right_node"].astype(int))
                        )
                    ),
                    "duplicate_edge_keys": int(
                        edges.duplicated(
                            [
                                "fold",
                                "formation_index",
                                "left_node",
                                "right_node",
                            ]
                        ).sum()
                    ),
                    "cross_formation_edges": int(
                        edges["formation"].ne(formation).sum()
                    ),
                }
            )
    return (
        nodes_by_fold,
        edges_by_key,
        pd.DataFrame(contact_records),
        pd.DataFrame(graph_records),
    )


def fit_fold_fields(
    *,
    fold: int,
    nodes: pd.DataFrame,
    edges_by_key: Mapping[tuple[int, str], pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, SparsePotentialSurface]:
    fields: dict[str, SparsePotentialSurface] = {}
    for formation in FORMATION_NAMES:
        selected = (
            nodes.loc[nodes["formation"].eq(formation)]
            .sort_values(["well_id", "contact_md"], kind="mergesort")
            .reset_index(drop=True)
        )
        fields[formation] = solve_sparse_potential(
            selected,
            edges_by_key[(int(fold), formation)],
            config,
        )
    return fields


def fit_fold_fields_fail_closed(
    *,
    fold: int,
    nodes: pd.DataFrame,
    edges_by_key: Mapping[tuple[int, str], pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[dict[str, SparsePotentialSurface], pd.DataFrame]:
    fields: dict[str, SparsePotentialSurface] = {}
    diagnostics: list[dict[str, Any]] = []
    for formation_index, formation in enumerate(FORMATION_NAMES):
        selected = (
            nodes.loc[nodes["formation"].eq(formation)]
            .sort_values(["well_id", "contact_md"], kind="mergesort")
            .reset_index(drop=True)
        )
        edges = edges_by_key[(int(fold), formation)]
        try:
            surface = solve_sparse_potential(selected, edges, config)
        except (ValueError, RuntimeError) as error:
            diagnostics.append(
                {
                    "fold": int(fold),
                    "formation": formation,
                    "formation_index": int(formation_index),
                    "nodes": len(selected),
                    "edges": len(edges),
                    "components": graph_component_count(len(selected), edges),
                    "scale_mad": float("nan"),
                    "sparse_solves": 0,
                    "accepted": False,
                    "solution_min": float("nan"),
                    "solution_max": float("nan"),
                    "solution_sha256": None,
                    "iterations": [],
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                }
            )
            continue
        fields[formation] = surface
        diagnostics.append(
            {
                **surface.diagnostics,
                "failure_type": None,
                "failure_reason": None,
            }
        )
    manifest = (
        pd.DataFrame(diagnostics)
        .sort_values("formation_index", kind="mergesort")
        .reset_index(drop=True)
    )
    return fields, manifest


def stage0_preflight(
    config: Mapping[str, Any],
) -> Stage0Bundle:
    start = time.perf_counter()
    identity = load_fold_identity(config)
    _, path_by_well = resolve_raw_train(config, set(identity.by_well))
    ledger = RoleReadLedger()
    census_start = time.perf_counter()
    (
        nodes_by_fold,
        edges_by_key,
        contact_census,
        graph_census,
    ) = build_contact_and_graph_census(
        identity,
        path_by_well,
        ledger,
        config,
    )
    census_seconds = time.perf_counter() - census_start
    preflight_fold = int(
        get_nested(config, "gates.stage0_target_free_resource_integrity.preflight_surface_fold")
    )
    solve_start = time.perf_counter()
    preflight_fields, solver_manifest = fit_fold_fields_fail_closed(
        fold=preflight_fold,
        nodes=nodes_by_fold[preflight_fold],
        edges_by_key=edges_by_key,
        config=config,
    )
    solve_seconds = time.perf_counter() - solve_start
    target_wells = sorted(
        well
        for well, fold in identity.by_well.items()
        if int(fold) == preflight_fold
    )
    sample_wells = stable_sample_wells(
        target_wells,
        count=int(
            get_nested(
                config,
                "gates.stage0_target_free_resource_integrity.preflight_target_wells",
            )
        ),
        seed=int(get_nested(config, "validation.seed")),
    )
    prediction_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    query_start = time.perf_counter()
    if len(preflight_fields) == len(FORMATION_NAMES):
        for well_id in sample_wells:
            identity_rows = identity.rows.loc[identity.rows["well_id"].eq(well_id)]
            well = load_target_safe(
                path_by_well[well_id],
                fold=preflight_fold,
                identity_rows=identity_rows,
                ledger=ledger,
            )
            prediction, support = predict_interval(
                well,
                preflight_fields,
                start_row=well.anchor_row,
                row_indices=well.suffix_row_idx,
                config=config,
                purpose="stage0_preflight_suffix",
            )
            prediction_parts.append(prediction)
            support_parts.append(support)
    query_seconds = time.perf_counter() - query_start
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame(
            columns=[
                "well_id",
                "supported_formation_count",
                "tvt_pred_exp436",
            ]
        )
    )
    query_support = (
        pd.concat(support_parts, ignore_index=True)
        if support_parts
        else pd.DataFrame(
            columns=[
                "fixed_supported_formation",
                "unique_source_wells",
            ]
        )
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    sample_rows = max(1, len(predictions))
    projected_seconds = (
        census_seconds
        + solve_seconds * int(get_nested(config, "validation.n_folds"))
        + query_seconds * expected_rows / sample_rows
    )
    gate = get_nested(config, "gates.stage0_target_free_resource_integrity")
    supported_count = (
        predictions[["well_id", "supported_formation_count"]]
        .drop_duplicates()["supported_formation_count"]
        .to_numpy(np.float64)
    )
    query_unique = query_support.loc[
        query_support["fixed_supported_formation"],
        "unique_source_wells",
    ].to_numpy(np.float64)
    finite_source = float(contact_census["finite_fraction"].min())
    source_well_min = int(contact_census["source_wells"].min())
    solver_fraction = float(solver_manifest["accepted"].mean())
    finite_primary = (
        float(
            np.isfinite(
                predictions["tvt_pred_exp436"].to_numpy(np.float64)
            ).mean()
        )
        if len(predictions)
        else 0.0
    )
    supported_p05 = (
        float(np.quantile(supported_count, 0.05))
        if len(supported_count)
        else 0.0
    )
    query_unique_p05 = (
        float(np.quantile(query_unique, 0.05))
        if len(query_unique)
        else 0.0
    )
    observed = {
        "rows": len(identity.rows),
        "wells": len(identity.by_well),
        "folds": len(set(identity.by_well.values())),
        "source_valid_overlap": int(contact_census["outer_valid_overlap"].sum()),
        "target_gr_reads": 0,
        "target_formation_reads": 0,
        "valid_suffix_truth_reads": 0,
        "duplicate_contact_node_edge_keys": int(
            contact_census["duplicate_stable_keys"].sum()
            + graph_census["duplicate_edge_keys"].sum()
        ),
        "finite_source_coverage": finite_source,
        "source_wells_per_formation_min": source_well_min,
        "successful_surface_solve_fraction": solver_fraction,
        "finite_primary_query_row_coverage": finite_primary,
        "supported_formations_per_well_p05": supported_p05,
        "query_unique_source_wells_p05": query_unique_p05,
        "projected_runtime_seconds": float(projected_seconds),
        "projected_peak_rss_gb": peak_rss_gb(),
        "preflight_sample_wells_selected": len(sample_wells),
        "preflight_sample_wells_queried": (
            predictions["well_id"].nunique() if len(predictions) else 0
        ),
        "preflight_sample_rows": len(predictions),
        "preflight_sparse_solves": int(solver_manifest["sparse_solves"].sum()),
    }
    checks = {
        "expected_rows": observed["rows"] == int(gate["expected_rows"]),
        "expected_wells": observed["wells"] == int(gate["expected_wells"]),
        "expected_folds": observed["folds"] == int(gate["expected_folds"]),
        "source_valid_overlap": observed["source_valid_overlap"]
        <= int(gate["source_valid_overlap_max"]),
        "target_gr_reads": observed["target_gr_reads"]
        <= int(gate["target_gr_reads_max"]),
        "target_formation_reads": observed["target_formation_reads"]
        <= int(gate["target_formation_reads_max"]),
        "valid_suffix_truth_reads": observed["valid_suffix_truth_reads"]
        <= int(gate["valid_suffix_truth_reads_max"]),
        "duplicate_keys": observed["duplicate_contact_node_edge_keys"]
        <= int(gate["duplicate_contact_node_edge_keys_max"]),
        "finite_source_coverage": observed["finite_source_coverage"]
        >= float(gate["finite_source_coverage_min"]),
        "source_wells_per_formation": observed["source_wells_per_formation_min"]
        >= int(gate["source_wells_per_formation_min"]),
        "surface_solves": observed["successful_surface_solve_fraction"]
        >= float(gate["successful_surface_solve_fraction_min"]),
        "finite_primary_query": observed["finite_primary_query_row_coverage"]
        >= float(gate["finite_primary_query_row_coverage_min"]),
        "supported_formations": observed["supported_formations_per_well_p05"]
        >= float(gate["supported_formations_per_well_p05_min"]),
        "query_unique_source_wells": observed["query_unique_source_wells_p05"]
        >= float(gate["query_unique_source_wells_p05_min"]),
        "runtime": observed["projected_runtime_seconds"]
        <= float(gate["projected_runtime_seconds_max"]),
        "rss": observed["projected_peak_rss_gb"]
        <= float(gate["projected_peak_rss_gb_max"]),
    }
    decision = {
        "stage": "stage0_target_free_resource_integrity",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
        "failure_policy": str(gate["fail_action"]),
    }
    return Stage0Bundle(
        identity=identity,
        path_by_well=dict(path_by_well),
        nodes_by_fold=nodes_by_fold,
        edges_by_key=edges_by_key,
        preflight_fields=preflight_fields,
        contact_census=contact_census,
        graph_census=graph_census,
        solver_manifest=solver_manifest,
        query_support=query_support,
        preflight_predictions=predictions,
        role_ledger=ledger,
        decision=decision,
        elapsed_seconds=time.perf_counter() - start,
    )


def persist_stage0(
    bundle: Stage0Bundle,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    contact_nodes = pd.concat(
        [bundle.nodes_by_fold[fold] for fold in sorted(bundle.nodes_by_fold)],
        ignore_index=True,
    )
    graph_edges = pd.concat(
        [bundle.edges_by_key[key] for key in sorted(bundle.edges_by_key)],
        ignore_index=True,
    )
    paths = {
        "fold_identity": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_fold_identity.csv.gz",
            bundle.identity.rows,
        ),
        "contact_nodes": write_csv(
            output / f"{OUTPUT_PREFIX}_contact_nodes.csv",
            contact_nodes,
        ),
        "graph_edges": write_csv(
            output / f"{OUTPUT_PREFIX}_graph_edges.csv",
            graph_edges,
        ),
        "contact_census": write_csv(
            output / f"{OUTPUT_PREFIX}_contact_census.csv",
            bundle.contact_census,
        ),
        "graph_census": write_csv(
            output / f"{OUTPUT_PREFIX}_graph_census.csv",
            bundle.graph_census,
        ),
        "solver_manifest": write_csv(
            output / f"{OUTPUT_PREFIX}_stage0_solver_manifest.csv",
            bundle.solver_manifest.drop(columns=["iterations"]),
        ),
        "query_support": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_stage0_query_support.csv.gz",
            bundle.query_support,
        ),
        "preflight_predictions": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_stage0_preflight_predictions.csv.gz",
            bundle.preflight_predictions,
        ),
        "role_ledger": write_csv(
            output / f"{OUTPUT_PREFIX}_stage0_role_read_ledger.csv",
            bundle.role_ledger.frame(),
        ),
    }
    decision_path = output / f"{OUTPUT_PREFIX}_stage0_decision.json"
    paths["decision"] = write_json(decision_path, bundle.decision)
    return {
        "artifacts": paths,
        "artifact_bundle_sha256": stable_json_sha256(
            {
                name: evidence.get(
                    "logical_sha256",
                    evidence.get("file_sha256"),
                )
                for name, evidence in paths.items()
            }
        ),
    }


# %% [markdown]
# ## 8. Stage 1 rolling-origin and target-free OOF freeze

# %%
@dataclass
class TargetFreeBundle:
    fields_by_fold: dict[int, dict[str, SparsePotentialSurface]]
    predictions: pd.DataFrame
    rolling_origin: pd.DataFrame
    query_support: pd.DataFrame
    solver_manifest: pd.DataFrame
    freeze_manifest: dict[str, Any]
    artifact_evidence: dict[str, Any]
    stage1_decision: dict[str, Any]


def rolling_origin_rows(
    well: TargetWell,
    fields: Mapping[str, SparsePotentialSurface],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    gate = get_nested(config, "gates.stage1_prefix_rolling_origin")
    anchor = well.anchor_row
    prefix_span = float(well.md[anchor] - well.md[0])
    if prefix_span < float(gate["minimum_prefix_md_ft"]):
        return None, None
    pseudo_md = float(well.md[anchor] - float(gate["heldout_prefix_md_ft"]))
    candidates = np.flatnonzero(well.md <= pseudo_md)
    if len(candidates) == 0:
        return None, None
    pseudo_anchor = int(candidates[-1])
    rows = np.arange(pseudo_anchor + 1, anchor + 1, dtype=np.int64)
    if len(rows) == 0:
        return None, None
    prediction, support = predict_interval(
        well,
        fields,
        start_row=pseudo_anchor,
        row_indices=rows,
        config=config,
        purpose="stage1_prefix_rolling_origin",
    )
    prediction["tvt_visible"] = well.tvt_input[rows]
    prediction["tvt_null_constant_u"] = (
        float(well.tvt_input[pseudo_anchor])
        - (well.z[rows] - well.z[pseudo_anchor])
    )
    prediction["endpoint"] = False
    prediction.loc[prediction.index[-1], "endpoint"] = True
    prediction["prefix_span_ft"] = prefix_span
    prediction["heldout_span_ft"] = (
        well.md[rows] - well.md[pseudo_anchor]
    )
    return prediction, support


def build_target_free_predictions(
    stage0: Stage0Bundle,
    config: Mapping[str, Any],
    output: Path,
) -> TargetFreeBundle:
    fields_by_fold: dict[int, dict[str, SparsePotentialSurface]] = {
        int(
            get_nested(
                config,
                "gates.stage0_target_free_resource_integrity.preflight_surface_fold",
            )
        ): stage0.preflight_fields
    }
    solver_records = [
        surface.diagnostics
        for fields in fields_by_fold.values()
        for surface in fields.values()
    ]
    for fold in sorted(stage0.nodes_by_fold):
        if fold in fields_by_fold:
            continue
        fields = fit_fold_fields(
            fold=fold,
            nodes=stage0.nodes_by_fold[fold],
            edges_by_key=stage0.edges_by_key,
            config=config,
        )
        fields_by_fold[fold] = fields
        solver_records.extend(surface.diagnostics for surface in fields.values())
    if len(fields_by_fold) != int(get_nested(config, "validation.n_folds")):
        raise ValueError("full target-free field inventory is incomplete")
    if sum(len(fields) for fields in fields_by_fold.values()) != int(
        get_nested(config, "execution.global_surface_fits")
    ):
        raise ValueError("global surface fit count differs from the contract")
    prediction_parts: list[pd.DataFrame] = []
    rolling_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    for fold in sorted(fields_by_fold):
        wells = sorted(
            well
            for well, assigned in stage0.identity.by_well.items()
            if int(assigned) == int(fold)
        )
        for well_id in wells:
            identity_rows = stage0.identity.rows.loc[
                stage0.identity.rows["well_id"].eq(well_id)
            ]
            well = load_target_safe(
                stage0.path_by_well[well_id],
                fold=fold,
                identity_rows=identity_rows,
                ledger=stage0.role_ledger,
            )
            prediction, support = predict_interval(
                well,
                fields_by_fold[fold],
                start_row=well.anchor_row,
                row_indices=well.suffix_row_idx,
                config=config,
                purpose="target_free_oof_suffix",
            )
            prediction["suffix_offset"] = well.suffix_offset
            prediction_parts.append(prediction)
            support_parts.append(support)
            rolling, rolling_support = rolling_origin_rows(
                well,
                fields_by_fold[fold],
                config,
            )
            if rolling is not None and rolling_support is not None:
                rolling_parts.append(rolling)
                support_parts.append(rolling_support)
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    rolling_origin = (
        pd.concat(rolling_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
        if rolling_parts
        else pd.DataFrame()
    )
    query_support = (
        pd.concat(support_parts, ignore_index=True)
        .sort_values(
            ["purpose", "fold", "well_id", "formation_index", "query_row"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if len(predictions) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("target-free prediction row count differs from exp226")
    if predictions.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("target-free prediction row keys are duplicated")
    if predictions["well_id"].nunique() != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("target-free prediction well count differs from exp226")
    stage0.role_ledger.freeze()
    solver_manifest = (
        pd.DataFrame(solver_records)
        .sort_values(["fold", "formation_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    if int(solver_manifest["sparse_solves"].sum()) > int(
        get_nested(config, "execution.maximum_sparse_solves_including_irls")
    ):
        raise ValueError("sparse solve count exceeds the frozen maximum")
    output.mkdir(parents=True, exist_ok=True)
    artifact_evidence = {
        "predictions": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_target_free_oof_predictions.csv.gz",
            predictions,
        ),
        "rolling_origin": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_prefix_rolling_origin.csv.gz",
            rolling_origin,
        ),
        "query_support": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_target_free_query_support.csv.gz",
            query_support,
        ),
        "solver_manifest": write_csv(
            output / f"{OUTPUT_PREFIX}_full_solver_manifest.csv",
            solver_manifest.drop(columns=["iterations"]),
        ),
        "role_ledger": write_csv(
            output / f"{OUTPUT_PREFIX}_target_free_role_read_ledger.csv",
            stage0.role_ledger.frame(),
        ),
    }
    for evidence in artifact_evidence.values():
        if "logical_sha256" in evidence and (
            evidence["logical_sha256"] != evidence["readback_logical_sha256"]
        ):
            raise RuntimeError("target-free generated artifact failed readback SHA")
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "target_free_frozen": True,
        "outer_valid_suffix_truth_reads_before_freeze": 0,
        "outer_valid_formation_reads_before_freeze": 0,
        "target_gr_reads_before_freeze": 0,
        "predictions": artifact_evidence["predictions"],
        "rolling_origin": artifact_evidence["rolling_origin"],
        "query_support": artifact_evidence["query_support"],
        "solver_manifest": artifact_evidence["solver_manifest"],
        "role_ledger": artifact_evidence["role_ledger"],
        "bundle_logical_sha256": stable_json_sha256(
            {
                name: evidence["logical_sha256"]
                for name, evidence in artifact_evidence.items()
            }
        ),
    }
    artifact_evidence["freeze_manifest"] = write_json(
        output / f"{OUTPUT_PREFIX}_target_free_freeze_manifest.json",
        freeze_manifest,
    )
    stage1_decision = evaluate_stage1_rolling_origin(
        rolling_origin,
        config,
    )
    return TargetFreeBundle(
        fields_by_fold=fields_by_fold,
        predictions=predictions,
        rolling_origin=rolling_origin,
        query_support=query_support,
        solver_manifest=solver_manifest,
        freeze_manifest=freeze_manifest,
        artifact_evidence=artifact_evidence,
        stage1_decision=stage1_decision,
    )


def root_mean_square(error: np.ndarray) -> float:
    values = np.asarray(error, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def evaluate_stage1_rolling_origin(
    rolling: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "gates.stage1_prefix_rolling_origin")
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if rolling.empty:
        observed = {
            "eligible_well_coverage": 0.0,
            "candidate_rmse": float("inf"),
            "null_rmse": float("inf"),
            "pooled_relative_rmse_gain": float("-inf"),
            "positive_fold_count": 0,
            "endpoint_absolute_error_delta_ft": float("inf"),
        }
    else:
        finite = (
            np.isfinite(rolling["tvt_pred_exp436"])
            & np.isfinite(rolling["tvt_null_constant_u"])
            & np.isfinite(rolling["tvt_visible"])
        )
        scored = rolling.loc[finite].copy()
        eligible_wells = int(scored["well_id"].nunique())
        candidate_error = (
            scored["tvt_pred_exp436"] - scored["tvt_visible"]
        ).to_numpy(np.float64)
        null_error = (
            scored["tvt_null_constant_u"] - scored["tvt_visible"]
        ).to_numpy(np.float64)
        candidate_rmse = root_mean_square(candidate_error)
        null_rmse = root_mean_square(null_error)
        relative_gain = (
            (null_rmse - candidate_rmse) / null_rmse
            if np.isfinite(null_rmse) and null_rmse > 0.0
            else float("-inf")
        )
        fold_records = []
        for fold in range(int(get_nested(config, "validation.n_folds"))):
            selected = scored.loc[scored["fold"].astype(int).eq(fold)]
            fold_records.append(
                {
                    "fold": fold,
                    "candidate_rmse": root_mean_square(
                        (
                            selected["tvt_pred_exp436"]
                            - selected["tvt_visible"]
                        ).to_numpy(np.float64)
                    ),
                    "null_rmse": root_mean_square(
                        (
                            selected["tvt_null_constant_u"]
                            - selected["tvt_visible"]
                        ).to_numpy(np.float64)
                    ),
                }
            )
        fold_metrics = pd.DataFrame(fold_records)
        positive_folds = int(
            (
                fold_metrics["candidate_rmse"] < fold_metrics["null_rmse"]
            ).sum()
        )
        endpoints = scored.loc[scored["endpoint"].astype(bool)]
        endpoint_delta = float(
            np.mean(
                np.abs(endpoints["tvt_pred_exp436"] - endpoints["tvt_visible"])
                - np.abs(
                    endpoints["tvt_null_constant_u"] - endpoints["tvt_visible"]
                )
            )
        )
        observed = {
            "eligible_well_coverage": float(eligible_wells / expected_wells),
            "eligible_wells": eligible_wells,
            "candidate_rmse": candidate_rmse,
            "null_rmse": null_rmse,
            "pooled_relative_rmse_gain": float(relative_gain),
            "positive_fold_count": positive_folds,
            "endpoint_absolute_error_delta_ft": endpoint_delta,
            "fold_metrics": fold_records,
        }
    checks = {
        "eligible_well_coverage": observed["eligible_well_coverage"]
        >= float(gate["eligible_well_coverage_min"]),
        "pooled_relative_rmse_gain": observed["pooled_relative_rmse_gain"]
        >= float(gate["pooled_relative_rmse_gain_min"]),
        "positive_fold_count": observed["positive_fold_count"]
        >= int(gate["positive_fold_count_min"]),
        "endpoint_absolute_error_delta": observed[
            "endpoint_absolute_error_delta_ft"
        ]
        <= float(gate["endpoint_absolute_error_delta_max_ft"]),
    }
    return {
        "stage": "stage1_prefix_rolling_origin",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
        "failure_policy": str(gate["fail_action"]),
    }


# %% [markdown]
# ## 9. Stage 2 truth-late direct OOF readout

# %%
def verify_target_free_freeze(bundle: TargetFreeBundle) -> None:
    if not bool(bundle.freeze_manifest.get("target_free_frozen")):
        raise RuntimeError("target-free prediction bundle is not frozen")
    for name in (
        "predictions",
        "rolling_origin",
        "query_support",
        "solver_manifest",
        "role_ledger",
    ):
        evidence = bundle.artifact_evidence[name]
        path = Path(evidence["path"])
        if sha256_file(path) != evidence["file_sha256"]:
            raise RuntimeError(f"frozen generated artifact changed: {name}")
        if evidence["logical_sha256"] != evidence["readback_logical_sha256"]:
            raise RuntimeError(f"frozen generated artifact SHA mismatch: {name}")


def load_exp226_truth_control_after_freeze(
    stage0: Stage0Bundle,
    target_free: TargetFreeBundle,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    verify_target_free_freeze(target_free)
    specification = get_nested(config, "data.exp226_oof")
    requested = tuple(map(str, specification["post_freeze_columns"]))
    if requested != EXP226_LATE_COLUMNS:
        raise ValueError("exp226 late-read allowlist changed")
    if sha256_decompressed_gzip(stage0.identity.path) != str(
        specification["expected_decompressed_sha256"]
    ):
        raise RuntimeError("exp226 saved OOF changed after prediction freeze")
    frame = pd.read_csv(
        stage0.identity.path,
        usecols=list(requested),
        dtype={"well_id": str},
    )
    stage0.role_ledger.record_truth_late(
        source="exp226_saved_oof",
        columns=requested,
        rows=len(frame),
    )
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("late exp226 rows are duplicated")
    prediction = target_free.predictions.copy()
    scored = prediction.merge(
        frame,
        on=["well_id", "row_idx", "suffix_offset", "fold"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(prediction):
        raise ValueError("late truth join changed target-free prediction rows")
    actual_control_rmse = root_mean_square(
        (scored["tvt_pred"] - scored["tvt_true"]).to_numpy(np.float64)
    )
    expected_control_rmse = float(
        get_nested(config, "gates.stage2_truth_late_oof.control_rmse_ft")
    )
    if not math.isclose(
        actual_control_rmse,
        expected_control_rmse,
        rel_tol=0.0,
        abs_tol=1.0e-3,
    ):
        raise ValueError(
            f"exp226 control parity failed: {actual_control_rmse} "
            f"!= {expected_control_rmse}"
        )
    return scored


def load_hidden_like_roles_after_freeze(
    stage0: Stage0Bundle,
    target_free: TargetFreeBundle,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    verify_target_free_freeze(target_free)
    specification = get_nested(config, "data.hidden_like_assignment")
    path = resolve_file(
        specification["patterns"],
        label="hidden-like assignments",
        expected_file_sha256=str(specification["expected_file_sha256"]),
    )
    frame = pd.read_csv(path, dtype={str(specification["well_column"]): str})
    columns = [
        str(specification["well_column"]),
        *map(str, specification["role_columns"].values()),
    ]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"hidden-like assignments miss {sorted(missing)}")
    selected = frame[columns].copy()
    if selected[columns[0]].duplicated().any():
        raise ValueError("hidden-like assignments duplicate well ids")
    stage0.role_ledger.record_truth_late(
        source="hidden_like_assignment",
        columns=columns,
        rows=len(selected),
    )
    return selected


def metric_record(
    frame: pd.DataFrame,
    *,
    scope: str,
    scope_value: str,
) -> dict[str, Any]:
    candidate = frame["tvt_pred_exp436"].to_numpy(np.float64)
    control = frame["tvt_pred"].to_numpy(np.float64)
    truth = frame["tvt_true"].to_numpy(np.float64)
    candidate_rmse = root_mean_square(candidate - truth)
    control_rmse = root_mean_square(control - truth)
    return {
        "scope": scope,
        "scope_value": scope_value,
        "rows": len(frame),
        "exp436_rmse": candidate_rmse,
        "exp226_rmse": control_rmse,
        "gain_exp226_minus_exp436": control_rmse - candidate_rmse,
        "delta_exp436_minus_exp226": candidate_rmse - control_rmse,
    }


def build_stage2_readout(
    scored: pd.DataFrame,
    hidden_roles: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    well_column = str(
        get_nested(config, "data.hidden_like_assignment.well_column")
    )
    roles = scored.merge(
        hidden_roles,
        left_on="well_id",
        right_on=well_column,
        how="left",
        validate="many_to_one",
    )
    hidden_role_columns = list(
        get_nested(config, "data.hidden_like_assignment.role_columns").values()
    )
    if roles[hidden_role_columns].isna().any().any():
        raise ValueError("hidden-like roles do not cover every OOF well")
    records = [metric_record(roles, scope="pooled", scope_value="all")]
    for fold, block in roles.groupby("fold", sort=True):
        records.append(
            metric_record(block, scope="fold", scope_value=str(int(fold)))
        )
    distance = roles["distance_from_anchor"].to_numpy(np.float64)
    role_columns = get_nested(config, "data.hidden_like_assignment.role_columns")
    scopes = {
        "near_0_250": (distance >= 0.0) & (distance < 250.0),
        "mid_250_1000": (distance >= 250.0) & (distance < 1000.0),
        "1000_plus": distance >= 1000.0,
        "hidden_like_spatial": roles[
            str(role_columns["hidden_like_spatial"])
        ].astype(str).eq("valid").to_numpy(),
        "hidden_like_typewell_purged": roles[
            str(role_columns["hidden_like_typewell_purged"])
        ].astype(str).eq("valid").to_numpy(),
    }
    for name, mask in scopes.items():
        if not np.any(mask):
            raise ValueError(f"Stage 2 scope is empty: {name}")
        records.append(
            metric_record(
                roles.loc[mask],
                scope="scope",
                scope_value=name,
            )
        )
    by_well = pd.DataFrame(
        [
            {
                "well_id": str(well),
                "fold": int(block["fold"].iloc[0]),
                **metric_record(block, scope="well", scope_value=str(well)),
            }
            for well, block in roles.groupby("well_id", sort=True)
        ]
    )
    metrics = pd.DataFrame(records)
    gate = get_nested(config, "gates.stage2_truth_late_oof")
    pooled = metrics.loc[
        metrics["scope"].eq("pooled")
        & metrics["scope_value"].eq("all")
    ].iloc[0]
    fold_metrics = metrics.loc[metrics["scope"].eq("fold")]
    scope_metrics = metrics.loc[metrics["scope"].eq("scope")].set_index(
        "scope_value"
    )
    positive_folds = int(
        (fold_metrics["delta_exp436_minus_exp226"] < 0.0).sum()
    )
    by_well_delta = by_well["delta_exp436_minus_exp226"].to_numpy(np.float64)
    correlation = float(
        np.corrcoef(
            roles["tvt_pred_exp436"].to_numpy(np.float64),
            roles["tvt_pred"].to_numpy(np.float64),
        )[0, 1]
    )
    observed = {
        "candidate_rmse_ft": float(pooled["exp436_rmse"]),
        "control_rmse_ft": float(pooled["exp226_rmse"]),
        "gain_vs_control_ft": float(pooled["gain_exp226_minus_exp436"]),
        "positive_fold_count": positive_folds,
        "suffix_1000_plus_gain_ft": float(
            scope_metrics.loc["1000_plus", "gain_exp226_minus_exp436"]
        ),
        "hidden_like_spatial_delta_ft": float(
            scope_metrics.loc[
                "hidden_like_spatial",
                "delta_exp436_minus_exp226",
            ]
        ),
        "hidden_like_typewell_purged_delta_ft": float(
            scope_metrics.loc[
                "hidden_like_typewell_purged",
                "delta_exp436_minus_exp226",
            ]
        ),
        "near_0_250_delta_ft": float(
            scope_metrics.loc[
                "near_0_250",
                "delta_exp436_minus_exp226",
            ]
        ),
        "by_well_delta_p95_ft": float(np.quantile(by_well_delta, 0.95)),
        "worst_well_delta_ft": float(np.max(by_well_delta)),
        "prediction_correlation_vs_control": correlation,
    }
    checks = {
        "candidate_rmse": observed["candidate_rmse_ft"]
        <= float(gate["candidate_rmse_max_ft"]),
        "gain_vs_control": observed["gain_vs_control_ft"]
        >= float(gate["gain_vs_control_min_ft"]),
        "positive_fold_count": observed["positive_fold_count"]
        >= int(gate["positive_fold_count_min"]),
        "suffix_1000_plus": observed["suffix_1000_plus_gain_ft"]
        >= float(gate["suffix_1000_plus_gain_min_ft"]),
        "hidden_like_spatial": observed["hidden_like_spatial_delta_ft"]
        <= float(gate["hidden_like_spatial_delta_max_ft"]),
        "hidden_like_typewell_purged": observed[
            "hidden_like_typewell_purged_delta_ft"
        ]
        <= float(gate["hidden_like_typewell_purged_delta_max_ft"]),
        "near_0_250": observed["near_0_250_delta_ft"]
        <= float(gate["near_0_250_delta_max_ft"]),
        "by_well_delta_p95": observed["by_well_delta_p95_ft"]
        <= float(gate["by_well_delta_p95_max_ft"]),
        "worst_well_delta": observed["worst_well_delta_ft"]
        <= float(gate["worst_well_delta_max_ft"]),
        "prediction_correlation": observed["prediction_correlation_vs_control"]
        <= float(gate["prediction_correlation_vs_control_max"]),
    }
    decision = {
        "stage": "stage2_truth_late_direct_oof",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": observed,
        "failure_policy": str(gate["fail_action"]),
    }
    return metrics, by_well, decision


def persist_stage2(
    *,
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    scored: pd.DataFrame,
    stage0: Stage0Bundle,
    target_free: TargetFreeBundle,
    decision: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    evidence = {
        "metrics": write_csv(
            output / f"{OUTPUT_PREFIX}_stage2_metrics.csv",
            metrics,
        ),
        "by_well": write_csv(
            output / f"{OUTPUT_PREFIX}_stage2_by_well.csv",
            by_well,
        ),
        "scored_oof": write_deterministic_gzip_csv(
            output / f"{OUTPUT_PREFIX}_stage2_scored_oof.csv.gz",
            scored,
        ),
        "truth_late_role_ledger": write_csv(
            output / f"{OUTPUT_PREFIX}_truth_late_role_read_ledger.csv",
            stage0.role_ledger.frame(),
        ),
        "decision": write_json(
            output / f"{OUTPUT_PREFIX}_stage2_decision.json",
            decision,
        ),
    }
    return {
        "artifacts": evidence,
        "prediction_logical_sha256": target_free.artifact_evidence[
            "predictions"
        ]["logical_sha256"],
        "truth_late_bundle_sha256": stable_json_sha256(
            {
                name: item.get("logical_sha256", item.get("file_sha256"))
                for name, item in evidence.items()
            }
        ),
    }


# %% [markdown]
# ## 10. Guarded execution and configuration preview

# %%
def write_runtime_metrics(payload: Mapping[str, Any]) -> None:
    write_json(metrics_path(), payload)


def run_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_stage0_authorization=True)
    output = artifacts_dir()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent/control:", get_nested(config, "lineage.parent"))
    print(
        "Execution contract: 1 candidate / 6 report-only paths / "
        "5 folds / 30 global fields / <=180 sparse solves / "
        "0 ML / 0 HMM / 0 PF / 0 Beam / 0 GPU / 0 parent rerun"
    )
    stage0 = stage0_preflight(config)
    stage0_artifacts = persist_stage0(stage0, output)
    print("Stage 0:", json.dumps(to_jsonable(stage0.decision), indent=2))
    base_summary: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "runtime_versions": runtime_versions(),
        "stage0": stage0.decision,
        "stage0_artifacts": stage0_artifacts,
        "stage1": None,
        "stage2": None,
        "inference": False,
        "submission": False,
    }
    if not stage0.decision["passed"]:
        base_summary["status"] = "stage0_fail_closed"
        write_runtime_metrics(base_summary)
        return base_summary
    if not bool(get_nested(config, "authorization.stage1_run_authorized")):
        base_summary["status"] = "stage0_pass_stage1_execution_locked"
        write_runtime_metrics(base_summary)
        return base_summary
    target_free = build_target_free_predictions(stage0, config, output)
    base_summary["stage1"] = target_free.stage1_decision
    base_summary["target_free_freeze"] = target_free.freeze_manifest
    print("Stage 1:", json.dumps(to_jsonable(target_free.stage1_decision), indent=2))
    if not target_free.stage1_decision["passed"]:
        base_summary["status"] = "stage1_fail_closed_before_truth"
        write_runtime_metrics(base_summary)
        return base_summary
    if not bool(get_nested(config, "authorization.stage2_run_authorized")):
        base_summary["status"] = "stage1_pass_stage2_truth_execution_locked"
        write_runtime_metrics(base_summary)
        return base_summary
    scored = load_exp226_truth_control_after_freeze(
        stage0,
        target_free,
        config,
    )
    hidden_roles = load_hidden_like_roles_after_freeze(
        stage0,
        target_free,
        config,
    )
    stage2_metrics, by_well, stage2_decision = build_stage2_readout(
        scored,
        hidden_roles,
        config,
    )
    stage2_artifacts = persist_stage2(
        metrics=stage2_metrics,
        by_well=by_well,
        scored=scored,
        stage0=stage0,
        target_free=target_free,
        decision=stage2_decision,
        output=output,
    )
    base_summary["stage2"] = stage2_decision
    base_summary["stage2_artifacts"] = stage2_artifacts
    base_summary["status"] = (
        "stage2_pass_candidate_for_separate_inference_decision"
        if stage2_decision["passed"]
        else "stage2_fail_closed"
    )
    print("Stage 2:", json.dumps(to_jsonable(stage2_decision), indent=2))
    write_runtime_metrics(base_summary)
    return base_summary


# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = config_path()
    CONFIG = load_config(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    print(
        json.dumps(
            to_jsonable(
                {
                    "experiment": get_nested(CONFIG, "experiment"),
                    "lineage": get_nested(CONFIG, "lineage"),
                    "design": get_nested(CONFIG, "design"),
                    "field": get_nested(CONFIG, "field"),
                    "query": get_nested(CONFIG, "query"),
                    "gates": get_nested(CONFIG, "gates"),
                    "execution": get_nested(CONFIG, "execution"),
                    "authorization": get_nested(CONFIG, "authorization"),
                }
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    SUMMARY = run_experiment(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY), indent=2, ensure_ascii=False))
