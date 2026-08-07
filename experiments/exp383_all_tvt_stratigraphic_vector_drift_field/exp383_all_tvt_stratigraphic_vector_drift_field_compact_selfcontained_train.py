# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # exp383 all-TVT stratigraphic vector drift field
#
# This implementation reconstructs a deterministic, fold-safe physical field from
# all outer-train TVT rows.  It consumes the saved exp226 OOF only as a fixed
# fallback/control and does not regenerate the parent.  Stage 1 truth is unread
# until all target-free outputs have been frozen by logical-content SHA.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment contract
# 2. Runtime, configuration, SHA, and artifact helpers
# 3. Fold roles, guarded raw reads, and exp226 control
# 4. Fold-safe formation surfaces and 29-dimensional signatures
# 5. All-TVT multiscale donor catalog
# 6. Stratigraphic absolute/vector field
# 7. Prefix calibration, exp226 shrink, and physical path solve
# 8. Stage 0 target-free freeze and integrity/resource gate
# 9. Late truth join and Stage 1 readout
# 10. Setup, configuration preview, and execution

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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

EXPERIMENT_NAME = "exp383_all_tvt_stratigraphic_vector_drift_field"
PARENT_EXPERIMENT = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
)
IMPORT_ONLY_ENV = "EXP383_IMPORT_ONLY"

FORMATION_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SURFACE_COLUMNS = tuple(f"surface_{name}" for name in FORMATION_NAMES)
SURFACE_GRAD_X_COLUMNS = tuple(f"surface_grad_x_{name}" for name in FORMATION_NAMES)
SURFACE_GRAD_Y_COLUMNS = tuple(f"surface_grad_y_{name}" for name in FORMATION_NAMES)
SURFACE_VARIANCE_COLUMNS = tuple(
    f"surface_variance_{name}" for name in FORMATION_NAMES
)
THICKNESS_COLUMNS = tuple(
    f"thickness_{FORMATION_NAMES[index]}_{FORMATION_NAMES[index + 1]}"
    for index in range(len(FORMATION_NAMES) - 1)
)
SIGNATURE_COLUMNS = tuple(f"signature_{index:02d}" for index in range(29))
TARGET_ALLOWED_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TARGET_FORBIDDEN_COLUMNS = frozenset({"TVT", "GR", *FORMATION_NAMES})
SOURCE_COLUMNS = (
    "MD",
    "X",
    "Y",
    "Z",
    "TVT",
    "TVT_input",
    *FORMATION_NAMES,
)
EXP226_SAFE_COLUMNS = ("well_id", "row_idx", "suffix_offset", "tvt_pred", "fold")
EXP226_TRUTH_COLUMNS = EXP226_SAFE_COLUMNS + ("tvt_true",)


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values in {list(columns)}")


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_kaggle_authorization: bool,
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp383 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp383 parent must remain exp226")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise ValueError("implementation authorization is not recorded")
    if int(get_nested(config, "runtime.scientific_candidates", -1)) != 1:
        raise ValueError("exactly one scientific candidate is allowed")
    if int(get_nested(config, "runtime.reporting_folds", -1)) != 5:
        raise ValueError("exactly five reporting folds are required")
    for key in (
        "runtime.fitted_models",
        "runtime.hmm_runs",
        "runtime.pf_runs",
        "runtime.beam_runs",
        "runtime.lightgbm_boosters",
    ):
        if int(get_nested(config, key, -1)) != 0:
            raise ValueError(f"{key} must remain zero")
    if bool(get_nested(config, "runtime.replay_parent_control", True)):
        raise ValueError("saved exp226 control must not be regenerated")
    mode = str(get_nested(config, "execution.current_mode", ""))
    if mode not in {"stage0_resource_preflight", "full_run"}:
        raise ValueError(f"unsupported exp383 execution mode: {mode}")
    if mode == "stage0_resource_preflight":
        if int(get_nested(config, "execution.preflight_max_wells", -1)) != int(
            get_nested(config, "runtime.resource_preflight_wells", -2)
        ):
            raise ValueError("preflight well count must match the fixed resource contract")
    if mode == "full_run" and not bool(
        get_nested(config, "execution.full_run_authorized", False)
    ):
        raise RuntimeError("exp383 full run is not authorized")
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized", False)
    ):
        raise RuntimeError(
            "Kaggle execution is not authorized; approve the 16-well CPU resource "
            "preflight and full run separately"
        )


# %% [markdown]
# ## 2. Runtime, configuration, SHA, and artifact helpers

# %%
PACKAGE_DIR = Path.cwd()


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def config_path() -> Path:
    local = PACKAGE_DIR / "config.yaml"
    if local.exists():
        return local
    root = find_project_root()
    candidate = root / "experiments" / EXPERIMENT_NAME / "config.yaml"
    if candidate.exists():
        return candidate
    matches = sorted(Path("/kaggle/working").rglob("config.yaml"))
    for match in matches:
        try:
            value = yaml.safe_load(match.read_text()) or {}
        except Exception:
            continue
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return match
    raise FileNotFoundError("exp383 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    value = yaml.safe_load((path or config_path()).read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def output_root() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return find_project_root()


def artifacts_dir() -> Path:
    if Path("/kaggle/working").exists():
        path = Path("/kaggle/working") / "artifacts"
    else:
        path = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_json_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(val) for key, val in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, Path):
            return str(item)
        return item

    return json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_cell(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return str(int(value))
    return str(value)


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return sha256_bytes(stable_json_bytes(schema))


def frame_content_sha256(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
) -> str:
    work = frame.copy()
    if sort_columns:
        work = work.sort_values(list(sort_columns), kind="mergesort")
    work = work.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(stable_json_bytes([str(column) for column in work.columns]))
    for row in work.itertuples(index=False, name=None):
        digest.update("\x1f".join(_canonical_cell(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json.loads(stable_json_bytes(value)), indent=2) + "\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix == ".gz":
        frame.to_csv(path, index=False, compression="gzip")
    else:
        frame.to_csv(path, index=False)


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if value > 10_000_000 else 1024.0 * 1024.0
    return value / divisor


def resolve_candidate_file(
    directories: Sequence[str | Path],
    filename: str,
) -> Path:
    for directory in directories:
        candidate = Path(directory) / filename
        if candidate.exists():
            return candidate
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if root.exists():
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"{filename} was not found in configured candidates")


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    candidates = [
        configured,
        find_project_root() / configured,
        Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*__horizontal_well.csv")):
            return candidate
    if Path("/kaggle/input").exists():
        for match in sorted(
            Path("/kaggle/input").rglob("*__horizontal_well.csv")
        ):
            if match.parent.name == "train":
                return match.parent
    raise FileNotFoundError("train horizontal-well directory was not found")


def robust_center_scale(
    values: np.ndarray,
    *,
    scale_floor: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    center = np.nanmedian(array, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(array - center[None, :]), axis=0)
    fallback = np.nanstd(array, axis=0)
    scale = np.where(np.isfinite(scale) & (scale >= scale_floor), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale >= scale_floor), scale, scale_floor)
    return center, scale


def huber_location(
    values: np.ndarray,
    *,
    delta: float = 1.345,
    iterations: int = 5,
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("Huber location requires finite values")
    location = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - location)))
    if not np.isfinite(scale) or scale < 1.0e-9:
        return location
    for _ in range(iterations):
        residual = (finite - location) / scale
        weights = np.ones_like(residual)
        outside = np.abs(residual) > delta
        weights[outside] = delta / np.abs(residual[outside])
        location = float(np.sum(weights * finite) / np.sum(weights))
    return location


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean(np.square(error))))


# %% [markdown]
# ## 3. Fold roles, guarded raw reads, and exp226 control

# %%
@dataclass
class RoleReadLedger:
    source_full_rows: int = 0
    target_safe_rows: int = 0
    valid_formation_reads: int = 0
    valid_suffix_truth_reads: int = 0
    valid_reference_overlap: int = 0
    truth_joined_after_freeze: bool = False

    def record_source(self, frame: pd.DataFrame) -> None:
        require_columns(frame, SOURCE_COLUMNS, "outer-train source")
        self.source_full_rows += len(frame)

    def record_target(self, columns: Iterable[str], rows: int) -> None:
        found = TARGET_FORBIDDEN_COLUMNS.intersection(map(str, columns))
        formations = set(FORMATION_NAMES).intersection(found)
        self.valid_formation_reads += len(formations)
        if "TVT" in found:
            self.valid_suffix_truth_reads += 1
        if found:
            raise ValueError(
                f"outer-valid target read contains forbidden columns: {sorted(found)}"
            )
        self.target_safe_rows += int(rows)

    def record_role_overlap(
        self,
        source_wells: Iterable[str],
        target_wells: Iterable[str],
    ) -> None:
        overlap = set(map(str, source_wells)).intersection(map(str, target_wells))
        self.valid_reference_overlap += len(overlap)
        if overlap:
            raise ValueError(f"outer-valid wells leaked into references: {sorted(overlap)[:5]}")

    def mark_truth_join(self, frozen_hashes: Mapping[str, str]) -> None:
        if not frozen_hashes:
            raise ValueError("truth cannot be joined before target-free SHA freeze")
        self.truth_joined_after_freeze = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_full_rows": self.source_full_rows,
            "target_safe_rows": self.target_safe_rows,
            "valid_formation_reads": self.valid_formation_reads,
            "valid_suffix_truth_reads": self.valid_suffix_truth_reads,
            "valid_reference_overlap": self.valid_reference_overlap,
            "truth_joined_after_freeze": self.truth_joined_after_freeze,
        }


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


def assign_group_folds(
    well_ids: Sequence[str],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    ordered = sorted(
        map(str, well_ids),
        key=lambda well: (hashlib.sha256(f"{seed}:{well}".encode()).hexdigest(), well),
    )
    return {well: index % int(n_folds) for index, well in enumerate(ordered)}


def read_source_well(path: Path, fold: int, ledger: RoleReadLedger) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(SOURCE_COLUMNS))
    ledger.record_source(frame)
    frame = frame.copy()
    frame["well_id"] = well_id_from_path(path)
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["fold"] = int(fold)
    frame["role"] = "outer_train"
    return frame


def read_target_safe_well(
    path: Path,
    fold: int,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(TARGET_ALLOWED_COLUMNS))
    ledger.record_target(frame.columns, len(frame))
    frame = frame.copy()
    frame["well_id"] = well_id_from_path(path)
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["fold"] = int(fold)
    frame["role"] = "outer_valid"
    return frame


def validate_parent_oof_contract(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    full_run: bool,
) -> None:
    require_columns(frame, EXP226_SAFE_COLUMNS, "exp226 OOF")
    folds_per_well = frame.groupby("well_id", sort=True)["fold"].nunique()
    if not folds_per_well.eq(1).all():
        raise ValueError("exp226 OOF assigns more than one fold to a well")
    if full_run:
        if len(frame) != int(get_nested(config, "validation.expected_rows")):
            raise ValueError("exp226 OOF row count differs from the frozen contract")
        if frame["well_id"].nunique() != int(
            get_nested(config, "validation.expected_wells")
        ):
            raise ValueError("exp226 OOF well count differs from the frozen contract")
        expected_folds = list(get_nested(config, "validation.expected_folds"))
        if sorted(frame["fold"].unique().tolist()) != expected_folds:
            raise ValueError("exp226 OOF folds differ from the frozen contract")


def load_parent_oof(
    config: Mapping[str, Any],
    *,
    include_truth: bool,
    frozen_hashes: Mapping[str, str] | None = None,
    ledger: RoleReadLedger | None = None,
) -> tuple[pd.DataFrame, Path]:
    filename = str(get_nested(config, "data.parent_exp226.filename"))
    candidates = list(get_nested(config, "data.parent_exp226.candidates", []))
    path = resolve_candidate_file(candidates, filename)
    expected = str(
        get_nested(config, "data.parent_exp226.expected_oof_decompressed_sha256")
    )
    actual = sha256_decompressed_csv(path)
    if actual != expected:
        raise ValueError(f"exp226 OOF SHA mismatch: expected {expected}, got {actual}")
    if include_truth:
        if ledger is None:
            raise ValueError("truth read requires a role ledger")
        ledger.mark_truth_join(frozen_hashes or {})
        columns = list(EXP226_TRUTH_COLUMNS)
    else:
        columns = list(EXP226_SAFE_COLUMNS)
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["fold"] = frame["fold"].astype(int)
    validate_parent_oof_contract(frame, config, full_run=True)
    return frame, path


def validate_fold_identity(
    fold_by_well: Mapping[str, int],
    parent_oof: pd.DataFrame,
) -> None:
    observed = (
        parent_oof[["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    if set(observed) != set(fold_by_well):
        raise ValueError("raw train wells and exp226 OOF well ids differ")
    mismatches = [
        well
        for well, fold in fold_by_well.items()
        if int(observed[well]) != int(fold)
    ]
    if mismatches:
        raise ValueError(f"exp226 fold identity mismatch: {mismatches[:5]}")


# %% [markdown]
# ## 4. Fold-safe formation surfaces and 29-dimensional signatures

# %%
def deterministic_md_decimation(md: np.ndarray, step_ft: float) -> np.ndarray:
    values = np.asarray(md, dtype=float)
    if not len(values):
        return np.asarray([], dtype=int)
    order = np.argsort(values, kind="mergesort")
    selected = [int(order[0])]
    last = float(values[order[0]])
    for index in order[1:]:
        current = float(values[index])
        if current - last >= step_ft:
            selected.append(int(index))
            last = current
    if int(order[-1]) != selected[-1]:
        selected.append(int(order[-1]))
    return np.asarray(selected, dtype=int)


def build_surface_points(
    source_wells: Sequence[pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    step = float(get_nested(config, "method.formation_surface.row_decimation_md_ft"))
    records: list[pd.DataFrame] = []
    for frame in source_wells:
        ordered = frame.sort_values(["MD", "row_idx"], kind="mergesort")
        selected = deterministic_md_decimation(ordered["MD"].to_numpy(dtype=float), step)
        part = ordered.iloc[selected][
            ["fold", "role", "well_id", "row_idx", "MD", "X", "Y", *FORMATION_NAMES]
        ].copy()
        finite = np.isfinite(
            part[["MD", "X", "Y", *FORMATION_NAMES]].to_numpy(dtype=float)
        ).all(axis=1)
        records.append(part.loc[finite])
    if not records:
        raise ValueError("surface point catalog is empty")
    return pd.concat(records, ignore_index=True).sort_values(
        ["fold", "well_id", "MD", "row_idx"], kind="mergesort"
    )


def _weighted_plane(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    ridge_trace_ratio: float,
) -> tuple[np.ndarray, float, float]:
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    weights = np.asarray(weights, dtype=float)
    normal = design.T @ (design * weights[:, None])
    trace = float(np.trace(normal))
    ridge = ridge_trace_ratio * max(trace, 1.0)
    penalty = np.diag([ridge * 1.0e-6, ridge, ridge])
    regularized = normal + penalty
    rhs = design.T @ (weights * response)
    coefficients = np.linalg.solve(regularized, rhs)
    residual = response - design @ coefficients
    variance = float(
        np.sum(weights * np.square(residual)) / max(float(np.sum(weights)), 1.0e-12)
    )
    return coefficients, max(variance, 1.0e-9), float(np.linalg.cond(regularized))


def _stable_capped_selection(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    *,
    exclude_well: str | None,
    unique_wells: int,
    max_per_well: int,
    max_points: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    work = candidates.copy()
    work["_distance_ft"] = np.asarray(distances, dtype=float)
    if exclude_well is not None:
        work = work.loc[work["well_id"].astype(str).ne(str(exclude_well))]
    work = work.sort_values(
        ["_distance_ft", "well_id", "MD", "row_idx"], kind="mergesort"
    )
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index, row in work.iterrows():
        well = str(row["well_id"])
        if well not in counts and len(counts) >= unique_wells:
            continue
        if counts.get(well, 0) >= max_per_well:
            continue
        selected.append(index)
        counts[well] = counts.get(well, 0) + 1
        if len(selected) >= max_points:
            break
    result = work.loc[selected].copy()
    return result.drop(columns=["_distance_ft"]), result["_distance_ft"].to_numpy(float)


def fit_surface_record(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    query: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    unique_wells = int(get_nested(config, "method.formation_surface.unique_wells"))
    maximum_per_well = int(
        get_nested(config, "method.formation_surface.max_points_per_well")
    )
    maximum_points = int(get_nested(config, "method.formation_surface.max_points"))
    exclude = str(query["well_id"]) if str(query.get("role")) == "outer_train" else None
    selected, selected_distances = _stable_capped_selection(
        candidates,
        distances,
        exclude_well=exclude,
        unique_wells=unique_wells,
        max_per_well=maximum_per_well,
        max_points=maximum_points,
    )
    if selected["well_id"].nunique() < unique_wells:
        return None
    unique_distance = (
        selected.assign(_distance_ft=selected_distances)
        .groupby("well_id", sort=True)["_distance_ft"]
        .min()
        .sort_values()
        .to_numpy(dtype=float)
    )
    bandwidth = float(unique_distance[unique_wells - 1])
    clip = list(get_nested(config, "method.formation_surface.adaptive_bandwidth_clip_ft"))
    bandwidth = float(np.clip(bandwidth, float(clip[0]), float(clip[1])))
    weights = np.exp(
        -0.5 * np.square(selected_distances / max(bandwidth, 1.0e-9))
    )
    dx = selected["X"].to_numpy(dtype=float) - float(query["X"])
    dy = selected["Y"].to_numpy(dtype=float) - float(query["Y"])
    design = np.column_stack([np.ones(len(selected)), dx, dy])
    ridge = float(get_nested(config, "method.formation_surface.ridge_trace_ratio"))
    record: dict[str, Any] = {
        "surface_available": True,
        "surface_bandwidth_ft": bandwidth,
        "surface_nearest_distance_ft": float(np.min(selected_distances)),
        "surface_unique_wells": int(selected["well_id"].nunique()),
        "surface_points": int(len(selected)),
    }
    conditions: list[float] = []
    for name in FORMATION_NAMES:
        response = selected[name].to_numpy(dtype=float)
        coefficients, variance, condition = _weighted_plane(
            design, response, weights, ridge
        )
        record[f"surface_{name}"] = float(coefficients[0])
        record[f"surface_grad_x_{name}"] = float(coefficients[1])
        record[f"surface_grad_y_{name}"] = float(coefficients[2])
        record[f"surface_variance_{name}"] = variance
        conditions.append(condition)
    record["surface_condition_number"] = float(max(conditions))
    record["surface_variance_mean"] = float(
        np.mean([record[column] for column in SURFACE_VARIANCE_COLUMNS])
    )
    return record


def query_surface_fields(
    surface_points: pd.DataFrame,
    queries: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if surface_points.empty:
        raise ValueError("surface points are empty")
    points_xy = surface_points[["X", "Y"]].to_numpy(dtype=float)
    initial_query = min(
        int(get_nested(config, "method.formation_surface.initial_neighbor_query", 256)),
        len(surface_points),
    )
    maximum_query = min(
        int(get_nested(config, "method.formation_surface.maximum_neighbor_query", 2048)),
        len(surface_points),
    )
    batch_rows = int(
        get_nested(config, "method.formation_surface.query_batch_rows", 128)
    )
    model = NearestNeighbors(n_neighbors=initial_query, algorithm="auto", n_jobs=1)
    model.fit(points_xy)
    output: list[dict[str, Any]] = []
    ordered = queries.reset_index(drop=True)
    for start in range(0, len(ordered), batch_rows):
        block = ordered.iloc[start : start + batch_rows]
        distances, indices = model.kneighbors(
            block[["X", "Y"]].to_numpy(dtype=float),
            n_neighbors=initial_query,
            return_distance=True,
        )
        for local_index, (_, query) in enumerate(block.iterrows()):
            candidate = surface_points.iloc[indices[local_index]]
            record = fit_surface_record(
                candidate,
                distances[local_index],
                query,
                config,
            )
            if record is None and maximum_query > initial_query:
                retry_distance, retry_index = model.kneighbors(
                    np.asarray([[float(query["X"]), float(query["Y"])]]),
                    n_neighbors=maximum_query,
                    return_distance=True,
                )
                record = fit_surface_record(
                    surface_points.iloc[retry_index[0]],
                    retry_distance[0],
                    query,
                    config,
                )
            base = {
                "fold": int(query["fold"]),
                "role": str(query["role"]),
                "well_id": str(query["well_id"]),
                "row_idx": int(query.get("row_idx", -1)),
                "query_id": str(query.get("query_id", "")),
                "MD": float(query["MD"]),
            }
            if record is None:
                record = {"surface_available": False}
            output.append({**base, **record})
    return pd.DataFrame(output)


def raw_signature_matrix(
    frame: pd.DataFrame,
    reference_s: np.ndarray,
) -> np.ndarray:
    require_columns(
        frame,
        (
            *SURFACE_COLUMNS,
            *SURFACE_GRAD_X_COLUMNS,
            *SURFACE_GRAD_Y_COLUMNS,
            *SURFACE_VARIANCE_COLUMNS,
        ),
        "surface frame",
    )
    surfaces = frame[list(SURFACE_COLUMNS)].to_numpy(dtype=float)
    relative = np.asarray(reference_s, dtype=float)[:, None] - surfaces
    thickness = np.diff(surfaces, axis=1)
    grad_x = frame[list(SURFACE_GRAD_X_COLUMNS)].to_numpy(dtype=float)
    grad_y = frame[list(SURFACE_GRAD_Y_COLUMNS)].to_numpy(dtype=float)
    variance = frame[list(SURFACE_VARIANCE_COLUMNS)].to_numpy(dtype=float)
    signature = np.column_stack([relative, thickness, grad_x, grad_y, variance])
    if signature.shape[1] != len(SIGNATURE_COLUMNS):
        raise AssertionError("stratigraphic signature must have 29 dimensions")
    return signature


def attach_standardized_signatures(
    donors: pd.DataFrame,
    queries: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor_raw = raw_signature_matrix(
        donors, donors["S_true"].to_numpy(dtype=float)
    )
    query_raw = raw_signature_matrix(
        queries, queries["base_path_s"].to_numpy(dtype=float)
    )
    floor = float(get_nested(config, "method.signature.robust_scale_floor", 1.0e-6))
    center, scale = robust_center_scale(donor_raw, scale_floor=floor)
    donor_z = np.clip((donor_raw - center[None, :]) / scale[None, :], -3.0, 3.0)
    query_z = np.clip((query_raw - center[None, :]) / scale[None, :], -3.0, 3.0)
    donors = donors.copy()
    queries = queries.copy()
    for index, column in enumerate(SIGNATURE_COLUMNS):
        donors[column] = donor_z[:, index]
        queries[column] = query_z[:, index]
    statistics = pd.DataFrame(
        {
            "fold": int(donors["fold"].iloc[0]),
            "dimension": np.arange(len(SIGNATURE_COLUMNS), dtype=int),
            "column": SIGNATURE_COLUMNS,
            "center": center,
            "scale": scale,
        }
    )
    return donors, queries, statistics


# %% [markdown]
# ## 5. All-TVT multiscale donor catalog

# %%
def fixed_huber_linear_fit(
    md: np.ndarray,
    values: np.ndarray,
    center_md: float,
    *,
    delta: float,
    iterations: int,
) -> tuple[float, float, float]:
    x = np.asarray(md, dtype=float) - float(center_md)
    y = np.asarray(values, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - design @ coefficients
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        if not np.isfinite(scale) or scale < 1.0e-9:
            break
        standardized = residual / scale
        weights = np.ones_like(standardized)
        outside = np.abs(standardized) > delta
        weights[outside] = delta / np.abs(standardized[outside])
        normal = design.T @ (design * weights[:, None])
        rhs = design.T @ (weights * y)
        coefficients = np.linalg.solve(normal + np.eye(2) * 1.0e-12, rhs)
    residual = y - design @ coefficients
    variance = float(np.mean(np.square(residual)))
    return float(coefficients[0]), float(coefficients[1]), max(variance, 1.0e-9)


def build_well_donor_windows(
    source: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    ordered = source.sort_values(["MD", "row_idx"], kind="mergesort")
    md = ordered["MD"].to_numpy(dtype=float)
    s_true = (
        ordered["TVT"].to_numpy(dtype=float)
        + ordered["Z"].to_numpy(dtype=float)
    )
    lengths = list(get_nested(config, "method.donor_catalog.window_lengths_ft"))
    strides = list(get_nested(config, "method.donor_catalog.window_strides_ft"))
    if len(lengths) != len(strides):
        raise ValueError("window lengths and strides must align")
    minimum_rows = int(
        get_nested(config, "method.donor_catalog.minimum_rows_per_window", 8)
    )
    delta = float(get_nested(config, "method.donor_catalog.huber_delta", 1.345))
    iterations = int(get_nested(config, "method.donor_catalog.huber_iterations", 5))
    records: list[dict[str, Any]] = []
    for length, stride in zip(lengths, strides, strict=True):
        length = float(length)
        stride = float(stride)
        centers = np.arange(md.min(), md.max() + 0.5 * stride, stride)
        centers = np.clip(centers, md.min(), md.max())
        centers = np.unique(centers)
        for center_md in centers:
            mask = np.abs(md - center_md) <= 0.5 * length
            if int(mask.sum()) < minimum_rows:
                continue
            s_center, rate, variance = fixed_huber_linear_fit(
                md[mask],
                s_true[mask],
                center_md,
                delta=delta,
                iterations=iterations,
            )
            x_center, tangent_x, _ = fixed_huber_linear_fit(
                md[mask],
                ordered.loc[mask, "X"].to_numpy(dtype=float),
                center_md,
                delta=delta,
                iterations=iterations,
            )
            y_center, tangent_y, _ = fixed_huber_linear_fit(
                md[mask],
                ordered.loc[mask, "Y"].to_numpy(dtype=float),
                center_md,
                delta=delta,
                iterations=iterations,
            )
            nearest = int(np.argmin(np.abs(md - center_md)))
            records.append(
                {
                    "fold": int(ordered["fold"].iloc[0]),
                    "role": "outer_train",
                    "well_id": str(ordered["well_id"].iloc[0]),
                    "row_idx": int(ordered["row_idx"].iloc[nearest]),
                    "MD": float(center_md),
                    "X": x_center,
                    "Y": y_center,
                    "Z": float(np.interp(center_md, md, ordered["Z"].to_numpy(float))),
                    "window_scale_ft": length,
                    "S_true": s_center,
                    "tangent_x": tangent_x,
                    "tangent_y": tangent_y,
                    "rate_true": rate,
                    "window_residual_variance": variance,
                    "window_rows": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


def attach_donor_query_ids(donors: pd.DataFrame) -> pd.DataFrame:
    output = donors.copy()
    output["query_id"] = [
        (
            f"f{int(row.fold):02d}_donor_{row.well_id}_"
            f"{int(row.window_scale_ft):04d}_{int(round(float(row.MD) * 1000)):012d}"
        )
        for row in output.itertuples()
    ]
    if output["query_id"].duplicated().any():
        duplicates = (
            output.loc[output["query_id"].duplicated(keep=False), "query_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise ValueError(f"donor query_id is not unique: {duplicates}")
    return output


def attach_surface_results_to_donors(
    donors: pd.DataFrame,
    surface: pd.DataFrame,
) -> pd.DataFrame:
    surface_columns = [
        column
        for column in surface.columns
        if column
        not in {"fold", "role", "well_id", "row_idx", "query_id", "MD"}
    ]
    keys = ["fold", "well_id", "row_idx", "query_id", "MD"]
    joined = donors.merge(
        surface[[*keys, *surface_columns]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(donors):
        raise ValueError("surface join changed donor node count")
    return joined


def build_multiscale_donor_catalog(
    source_wells: Sequence[pd.DataFrame],
    surface_points: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, float]:
    raw_parts = [build_well_donor_windows(frame, config) for frame in source_wells]
    raw = pd.concat(raw_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "window_scale_ft", "MD"], kind="mergesort"
    )
    raw = attach_donor_query_ids(raw)
    surface = query_surface_fields(surface_points, raw, config)
    joined = attach_surface_results_to_donors(raw, surface)
    coverage = float(joined["surface_available"].fillna(False).mean())
    available = joined.loc[joined["surface_available"].fillna(False)].copy()
    require_finite(
        available,
        (
            "S_true",
            "rate_true",
            *SURFACE_COLUMNS,
            *SURFACE_GRAD_X_COLUMNS,
            *SURFACE_GRAD_Y_COLUMNS,
            *SURFACE_VARIANCE_COLUMNS,
        ),
        "surface-conditioned donor catalog",
    )
    return available, coverage


# %% [markdown]
# ## 6. Stratigraphic absolute/vector field

# %%
def build_query_grid(
    target: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    ordered = target.sort_values(["MD", "row_idx"], kind="mergesort").reset_index(drop=True)
    md = ordered["MD"].to_numpy(dtype=float)
    step = float(get_nested(config, "method.path_solver.query_grid_md_ft", 64.0))
    grid = np.arange(md.min(), md.max() + 0.5 * step, step)
    grid = np.clip(grid, md.min(), md.max())
    indices = np.searchsorted(md, grid, side="left")
    indices = np.clip(indices, 0, len(md) - 1)
    left = np.clip(indices - 1, 0, len(md) - 1)
    choose_left = np.abs(md[left] - grid) <= np.abs(md[indices] - grid)
    indices = np.where(choose_left, left, indices)
    known = np.flatnonzero(np.isfinite(ordered["TVT_input"].to_numpy(dtype=float)))
    extra = [0, len(ordered) - 1]
    if len(known):
        extra.append(int(known[-1]))
    selected = np.unique(np.concatenate([indices, np.asarray(extra, dtype=int)]))
    full_tangent_x = np.gradient(
        ordered["X"].to_numpy(dtype=float), md, edge_order=1
    )
    full_tangent_y = np.gradient(
        ordered["Y"].to_numpy(dtype=float), md, edge_order=1
    )
    query = ordered.iloc[selected].copy()
    query["tangent_x"] = full_tangent_x[selected]
    query["tangent_y"] = full_tangent_y[selected]
    query["query_id"] = [
        f"f{int(row.fold):02d}_{row.well_id}_{int(row.row_idx):07d}"
        for row in query.itertuples()
    ]
    return query.sort_values(["MD", "row_idx"], kind="mergesort").reset_index(drop=True)


def attach_exp226_fallback(
    query: pd.DataFrame,
    target_safe: pd.DataFrame,
    parent_safe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    well = str(target_safe["well_id"].iloc[0])
    parent = parent_safe.loc[parent_safe["well_id"].eq(well)].copy()
    if parent.empty:
        raise ValueError(f"exp226 OOF has no rows for {well}")
    full = target_safe.sort_values("row_idx", kind="mergesort").copy()
    suffix = parent[["row_idx", "tvt_pred", "suffix_offset"]]
    full = full.merge(suffix, on="row_idx", how="left", validate="one_to_one")
    prefix_s = full["TVT_input"].to_numpy(dtype=float) + full["Z"].to_numpy(dtype=float)
    fallback_s = full["tvt_pred"].to_numpy(dtype=float) + full["Z"].to_numpy(dtype=float)
    base_s = np.where(np.isfinite(prefix_s), prefix_s, fallback_s)
    if not np.isfinite(base_s).all():
        raise ValueError(f"exp226 base path is incomplete for {well}")
    md = full["MD"].to_numpy(dtype=float)
    fallback_rate = np.gradient(base_s, md, edge_order=1)
    by_row = pd.DataFrame(
        {
            "row_idx": full["row_idx"].to_numpy(dtype=int),
            "base_path_s": base_s,
            "fallback_rate": fallback_rate,
        }
    )
    attached = query.merge(by_row, on="row_idx", how="left", validate="one_to_one")
    require_finite(
        attached,
        ("base_path_s", "fallback_rate"),
        f"exp226 query fallback {well}",
    )
    anchor_rows = np.flatnonzero(np.isfinite(full["TVT_input"].to_numpy(dtype=float)))
    if not len(anchor_rows):
        raise ValueError(f"{well} has no finite TVT_input prefix")
    anchor_md = float(full["MD"].iloc[int(anchor_rows[-1])])
    safe_keys = parent.merge(
        full[["row_idx", "MD", "Z"]],
        on="row_idx",
        how="left",
        validate="one_to_one",
    )
    safe_keys["distance_from_anchor"] = (
        safe_keys["MD"].to_numpy(dtype=float) - anchor_md
    )
    safe_keys = safe_keys.rename(columns={"tvt_pred": "exp226_prediction"})
    return attached, safe_keys


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    square = float(np.sum(np.square(weights)))
    return total * total / square if square > 0 else 0.0


def _select_vector_donors(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    query: pd.Series,
    config: Mapping[str, Any],
    *,
    exclude_same_well: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    return _stable_capped_selection(
        candidates,
        distances,
        exclude_well=str(query["well_id"]) if exclude_same_well else None,
        unique_wells=int(get_nested(config, "method.vector_field.unique_wells")),
        max_per_well=int(
            get_nested(config, "method.vector_field.max_nodes_per_well")
        ),
        max_points=int(get_nested(config, "method.vector_field.max_nodes")),
    )


def fit_vector_field_record(
    candidates: pd.DataFrame,
    distances: np.ndarray,
    query: pd.Series,
    config: Mapping[str, Any],
    *,
    exclude_same_well: bool,
) -> dict[str, Any] | None:
    selected, selected_distances = _select_vector_donors(
        candidates,
        distances,
        query,
        config,
        exclude_same_well=exclude_same_well,
    )
    minimum_wells = int(
        get_nested(config, "method.vector_field.minimum_unique_wells", 8)
    )
    if selected["well_id"].nunique() < minimum_wells:
        return None
    unique_distance = (
        selected.assign(_distance_ft=selected_distances)
        .groupby("well_id", sort=True)["_distance_ft"]
        .min()
        .sort_values()
        .to_numpy(dtype=float)
    )
    bandwidth_rank = int(
        get_nested(
            config,
            "method.vector_field.adaptive_bandwidth_unique_well_rank",
            24,
        )
    )
    bandwidth_index = min(bandwidth_rank - 1, len(unique_distance) - 1)
    clip = list(
        get_nested(config, "method.vector_field.adaptive_bandwidth_clip_ft")
    )
    bandwidth = float(
        np.clip(unique_distance[bandwidth_index], float(clip[0]), float(clip[1]))
    )
    donor_signature = selected[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
    query_signature = query[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
    signature_delta = np.clip(
        donor_signature - query_signature[None, :], -3.0, 3.0
    )
    formation_weight = np.exp(
        -0.5 * np.mean(np.square(signature_delta), axis=1)
    )
    xy_weight = np.exp(
        -0.5 * np.square(selected_distances / max(bandwidth, 1.0e-9))
    )
    residual_variance = np.maximum(
        selected["window_residual_variance"].to_numpy(dtype=float), 1.0e-4
    )
    weights = xy_weight * formation_weight / residual_variance
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return None
    dx = selected["X"].to_numpy(dtype=float) - float(query["X"])
    dy = selected["Y"].to_numpy(dtype=float) - float(query["Y"])
    design = np.column_stack([np.ones(len(selected)), dx, dy])
    ridge = float(get_nested(config, "method.vector_field.ridge_trace_ratio"))
    absolute_candidates: list[float] = []
    rate_candidates: list[float] = []
    absolute_variances: list[float] = []
    rate_variances: list[float] = []
    conditions: list[float] = []
    for surface_column, grad_x_column, grad_y_column in zip(
        SURFACE_COLUMNS,
        SURFACE_GRAD_X_COLUMNS,
        SURFACE_GRAD_Y_COLUMNS,
        strict=True,
    ):
        response = (
            selected["S_true"].to_numpy(dtype=float)
            - selected[surface_column].to_numpy(dtype=float)
        )
        coefficients, absolute_variance, condition = _weighted_plane(
            design, response, weights, ridge
        )
        absolute = float(query[surface_column]) + float(coefficients[0])
        gradient_x = float(query[grad_x_column]) + float(coefficients[1])
        gradient_y = float(query[grad_y_column]) + float(coefficients[2])
        rate = (
            float(query["tangent_x"]) * gradient_x
            + float(query["tangent_y"]) * gradient_y
        )
        predicted_donor_rate = (
            selected["tangent_x"].to_numpy(dtype=float)
            * (
                selected[grad_x_column].to_numpy(dtype=float)
                + float(coefficients[1])
            )
            + selected["tangent_y"].to_numpy(dtype=float)
            * (
                selected[grad_y_column].to_numpy(dtype=float)
                + float(coefficients[2])
            )
        )
        rate_residual = (
            selected["rate_true"].to_numpy(dtype=float) - predicted_donor_rate
        )
        rate_variance = float(
            np.sum(weights * np.square(rate_residual))
            / max(float(np.sum(weights)), 1.0e-12)
        )
        absolute_candidates.append(absolute)
        rate_candidates.append(rate)
        absolute_variances.append(absolute_variance)
        rate_variances.append(max(rate_variance, 1.0e-9))
        conditions.append(condition)
    if not np.isfinite(absolute_candidates + rate_candidates).all():
        return None
    surface_variance = float(
        np.mean(query[list(SURFACE_VARIANCE_COLUMNS)].to_numpy(dtype=float))
    )
    return {
        "field_available": True,
        "field_absolute_s": float(np.median(absolute_candidates)),
        "field_rate": float(np.median(rate_candidates)),
        "field_absolute_variance": float(
            np.median(absolute_variances) + surface_variance
        ),
        "field_rate_variance": float(np.median(rate_variances)),
        "field_support_ess": _effective_sample_size(weights),
        "field_unique_wells": int(selected["well_id"].nunique()),
        "field_condition_number": float(max(conditions)),
        "field_surface_variance": surface_variance,
        "field_selected_nodes": int(len(selected)),
        "field_bandwidth_ft": bandwidth,
    }


def generate_vector_fields(
    donor_catalog: pd.DataFrame,
    queries: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    exclude_same_well: bool,
) -> pd.DataFrame:
    points_xy = donor_catalog[["X", "Y"]].to_numpy(dtype=float)
    initial_query = min(
        int(get_nested(config, "method.vector_field.initial_neighbor_query", 256)),
        len(donor_catalog),
    )
    maximum_query = min(
        int(get_nested(config, "method.vector_field.maximum_neighbor_query", 2048)),
        len(donor_catalog),
    )
    batch_rows = int(get_nested(config, "method.vector_field.query_batch_rows", 128))
    model = NearestNeighbors(n_neighbors=initial_query, algorithm="auto", n_jobs=1)
    model.fit(points_xy)
    ordered = queries.reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for start in range(0, len(ordered), batch_rows):
        block = ordered.iloc[start : start + batch_rows]
        distances, indices = model.kneighbors(
            block[["X", "Y"]].to_numpy(dtype=float),
            n_neighbors=initial_query,
            return_distance=True,
        )
        for local_index, (_, query) in enumerate(block.iterrows()):
            record = fit_vector_field_record(
                donor_catalog.iloc[indices[local_index]],
                distances[local_index],
                query,
                config,
                exclude_same_well=exclude_same_well,
            )
            if record is None and maximum_query > initial_query:
                retry_distance, retry_index = model.kneighbors(
                    np.asarray([[float(query["X"]), float(query["Y"])]]),
                    n_neighbors=maximum_query,
                    return_distance=True,
                )
                record = fit_vector_field_record(
                    donor_catalog.iloc[retry_index[0]],
                    retry_distance[0],
                    query,
                    config,
                    exclude_same_well=exclude_same_well,
                )
            base = {
                "fold": int(query["fold"]),
                "well_id": str(query["well_id"]),
                "row_idx": int(query.get("row_idx", -1)),
                "query_id": str(query.get("query_id", "")),
                "MD": float(query["MD"]),
            }
            records.append(
                {**base, **(record if record is not None else {"field_available": False})}
            )
    return pd.DataFrame(records)


def attach_field_results(
    query: pd.DataFrame,
    fields: pd.DataFrame,
) -> pd.DataFrame:
    field_columns = [
        column
        for column in fields.columns
        if column not in {"fold", "well_id", "row_idx", "query_id", "MD"}
    ]
    joined = query.merge(
        fields[
            ["fold", "well_id", "row_idx", "query_id", "MD", *field_columns]
        ],
        on=["fold", "well_id", "row_idx", "query_id", "MD"],
        how="left",
        validate="one_to_one",
    )
    joined["field_available"] = joined["field_available"].fillna(False)
    return joined


def build_exp384_donor_nodes(
    donor_catalog: pd.DataFrame,
    donor_fields: pd.DataFrame,
) -> pd.DataFrame:
    primary = donor_catalog.loc[
        donor_catalog["window_scale_ft"].eq(256.0)
    ].copy()
    joined = attach_field_results(primary, donor_fields)
    joined = joined.loc[joined["field_available"]].copy()
    for name, surface_column in zip(
        FORMATION_NAMES, SURFACE_COLUMNS, strict=True
    ):
        joined[f"fault_surface_residual_{name}"] = (
            joined["S_true"].to_numpy(dtype=float)
            - joined[surface_column].to_numpy(dtype=float)
        )
    for index, column in enumerate(THICKNESS_COLUMNS):
        joined[column] = (
            joined[SURFACE_COLUMNS[index + 1]].to_numpy(dtype=float)
            - joined[SURFACE_COLUMNS[index]].to_numpy(dtype=float)
        )
    joined["smooth_absolute_residual"] = (
        joined["S_true"].to_numpy(dtype=float)
        - joined["field_absolute_s"].to_numpy(dtype=float)
    )
    joined["smooth_rate_residual"] = (
        joined["rate_true"].to_numpy(dtype=float)
        - joined["field_rate"].to_numpy(dtype=float)
    )
    return joined.sort_values(["fold", "well_id", "MD"], kind="mergesort")


# %% [markdown]
# ## 7. Prefix calibration, exp226 shrink, and physical path solve

# %%
def field_confidence(
    support_ess: np.ndarray,
    unique_wells: np.ndarray,
    condition_number: np.ndarray,
    surface_variance: np.ndarray,
    surface_variance_reference: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    support_full = float(get_nested(config, "method.fallback.support_ess_full"))
    wells_full = float(get_nested(config, "method.fallback.support_unique_wells_full"))
    condition_reference = float(
        get_nested(config, "method.fallback.condition_reference")
    )
    support = np.clip(np.asarray(support_ess, float) / support_full, 0.0, 1.0)
    wells = np.clip(np.asarray(unique_wells, float) / wells_full, 0.0, 1.0)
    condition = np.maximum(np.asarray(condition_number, float), condition_reference)
    condition_part = np.clip(
        math.log10(condition_reference) / np.log10(condition), 0.0, 1.0
    )
    reference = np.maximum(np.asarray(surface_variance_reference, float), 1.0e-9)
    surface = np.exp(-np.asarray(surface_variance, float) / reference)
    return support * wells * condition_part * surface


def calibrate_prefix_for_well(
    query: pd.DataFrame,
    target_safe: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = query.sort_values(["MD", "row_idx"], kind="mergesort").copy()
    if not bool(ordered["field_available"].all()):
        ordered["prefix_bias_ft"] = 0.0
        ordered["calibrated_absolute_s"] = ordered["base_path_s"].to_numpy(float)
        ordered["field_confidence"] = 0.0
        ordered["final_rate"] = ordered["fallback_rate"].to_numpy(float)
        return ordered, {
            "well_id": str(ordered["well_id"].iloc[0]),
            "available": False,
            "known_prefix_rows": int(np.isfinite(target_safe["TVT_input"]).sum()),
            "prefix_bias_ft": 0.0,
            "prefix_huber_rmse_ft": float("nan"),
            "status": "field_coverage_exp226_fallback",
        }
    known = target_safe.loc[np.isfinite(target_safe["TVT_input"])].sort_values("MD")
    if known.empty:
        raise ValueError("prefix calibration requires finite TVT_input")
    query_md = ordered["MD"].to_numpy(dtype=float)
    raw_absolute = ordered["field_absolute_s"].to_numpy(dtype=float)
    known_md = known["MD"].to_numpy(dtype=float)
    if known_md.min() < query_md.min() or known_md.max() > query_md.max():
        raise ValueError("prefix MD lies outside the field query grid")
    interpolated = np.interp(known_md, query_md, raw_absolute)
    input_s = (
        known["TVT_input"].to_numpy(dtype=float)
        + known["Z"].to_numpy(dtype=float)
    )
    residual = input_s - interpolated
    bias = huber_location(
        residual,
        delta=float(get_nested(config, "method.prefix_calibration.huber_delta")),
        iterations=int(
            get_nested(config, "method.prefix_calibration.huber_iterations")
        ),
    )
    centered = residual - bias
    reference = ordered["surface_variance_reference"].to_numpy(dtype=float)
    confidence = field_confidence(
        ordered["field_support_ess"].to_numpy(dtype=float),
        ordered["field_unique_wells"].to_numpy(dtype=float),
        ordered["field_condition_number"].to_numpy(dtype=float),
        ordered["field_surface_variance"].to_numpy(dtype=float),
        reference,
        config,
    )
    ordered["surface_variance_reference"] = reference
    ordered["prefix_bias_ft"] = bias
    ordered["calibrated_absolute_s"] = raw_absolute + bias
    ordered["field_confidence"] = confidence
    ordered["final_rate"] = (
        confidence * ordered["field_rate"].to_numpy(dtype=float)
        + (1.0 - confidence) * ordered["fallback_rate"].to_numpy(dtype=float)
    )
    return ordered, {
        "well_id": str(ordered["well_id"].iloc[0]),
        "available": True,
        "known_prefix_rows": int(len(known)),
        "prefix_bias_ft": bias,
        "prefix_huber_rmse_ft": float(np.sqrt(np.mean(np.square(centered)))),
        "status": "calibrated",
    }


def solve_path_for_well(
    query: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    ordered = query.sort_values(["MD", "row_idx"], kind="mergesort")
    if not bool(ordered["field_available"].all()):
        return ordered["base_path_s"].to_numpy(dtype=float), "coverage_exp226_fallback"
    md = ordered["MD"].to_numpy(dtype=float)
    if len(md) < 2 or np.any(np.diff(md) <= 0):
        return ordered["base_path_s"].to_numpy(dtype=float), "invalid_md_exp226_fallback"
    n_rows = len(ordered)
    equations: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    floor = float(get_nested(config, "method.path_solver.variance_floor", 1.0e-9))
    absolute = ordered["calibrated_absolute_s"].to_numpy(dtype=float)
    absolute_variance = np.maximum(
        ordered["field_absolute_variance"].to_numpy(dtype=float), floor
    )
    for index in range(n_rows):
        row = np.zeros(n_rows, dtype=float)
        row[index] = 1.0
        equations.append(row)
        targets.append(float(absolute[index]))
        weights.append(float(1.0 / absolute_variance[index]))
    rate = ordered["final_rate"].to_numpy(dtype=float)
    rate_variance = np.maximum(
        ordered["field_rate_variance"].to_numpy(dtype=float), floor
    )
    for index in range(n_rows - 1):
        delta_md = float(md[index + 1] - md[index])
        row = np.zeros(n_rows, dtype=float)
        row[index] = -1.0
        row[index + 1] = 1.0
        equations.append(row)
        targets.append(float(0.5 * (rate[index] + rate[index + 1]) * delta_md))
        interval_variance = 0.5 * (
            rate_variance[index] + rate_variance[index + 1]
        )
        weights.append(float(1.0 / interval_variance))
    curvature = float(get_nested(config, "method.path_solver.curvature_weight"))
    for index in range(1, n_rows - 1):
        left = float(md[index] - md[index - 1])
        right = float(md[index + 1] - md[index])
        row = np.zeros(n_rows, dtype=float)
        row[index - 1] = 1.0 / left
        row[index] = -(1.0 / left + 1.0 / right)
        row[index + 1] = 1.0 / right
        equations.append(row)
        targets.append(0.0)
        weights.append(curvature)
    design = np.vstack(equations)
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    normal = design.T @ (design * weight[:, None])
    rhs = design.T @ (weight * target)
    known = np.isfinite(ordered["TVT_input"].to_numpy(dtype=float))
    known_value = (
        ordered["TVT_input"].to_numpy(dtype=float)
        + ordered["Z"].to_numpy(dtype=float)
    )
    unknown = ~known
    solution = np.empty(n_rows, dtype=float)
    solution[known] = known_value[known]
    if unknown.any():
        normal_uu = normal[np.ix_(unknown, unknown)]
        rhs_u = rhs[unknown] - normal[np.ix_(unknown, known)] @ solution[known]
        try:
            solution[unknown] = np.linalg.solve(normal_uu, rhs_u)
        except np.linalg.LinAlgError:
            return (
                ordered["base_path_s"].to_numpy(dtype=float),
                "rank_deficient_exp226_fallback",
            )
    if not np.isfinite(solution).all():
        return ordered["base_path_s"].to_numpy(dtype=float), "nonfinite_exp226_fallback"
    return solution, "vector_field"


def solve_and_interpolate_prediction(
    calibrated_query: pd.DataFrame,
    safe_keys: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = calibrated_query.sort_values(["MD", "row_idx"], kind="mergesort").copy()
    solution, status = solve_path_for_well(ordered, config)
    ordered["path_s"] = solution
    ordered["path_status"] = status
    target = safe_keys.sort_values("row_idx", kind="mergesort").copy()
    if "fallback" in status:
        target["exp383_prediction"] = target["exp226_prediction"].to_numpy(dtype=float)
    else:
        query_md = ordered["MD"].to_numpy(dtype=float)
        target_md = target["MD"].to_numpy(dtype=float)
        if target_md.min() < query_md.min() or target_md.max() > query_md.max():
            target["exp383_prediction"] = target["exp226_prediction"].to_numpy(float)
            status = "coverage_exp226_fallback"
        else:
            interpolated_s = np.interp(
                target_md, query_md, ordered["path_s"].to_numpy(dtype=float)
            )
            target["exp383_prediction"] = (
                interpolated_s - target["Z"].to_numpy(dtype=float)
            )
    target["exp383_path_status"] = status
    return ordered, target


def to_exp384_query_contract(query: pd.DataFrame) -> pd.DataFrame:
    output = query.copy()
    available = output["field_available"].astype(bool).to_numpy()
    output["base_absolute_s"] = np.where(
        available,
        output["field_absolute_s"].to_numpy(dtype=float),
        output["base_path_s"].to_numpy(dtype=float),
    )
    output["base_rate"] = np.where(
        available,
        output["field_rate"].to_numpy(dtype=float),
        output["fallback_rate"].to_numpy(dtype=float),
    )
    output["base_absolute_variance"] = np.where(
        available,
        output["field_absolute_variance"].to_numpy(dtype=float),
        1.0e9,
    )
    output["base_rate_variance"] = np.where(
        available,
        output["field_rate_variance"].to_numpy(dtype=float),
        1.0e9,
    )
    output["base_support_ess"] = np.where(
        available, output["field_support_ess"].to_numpy(dtype=float), 0.0
    )
    output["base_unique_wells"] = np.where(
        available, output["field_unique_wells"].to_numpy(dtype=float), 0.0
    )
    output["base_condition_number"] = np.where(
        available,
        output["field_condition_number"].to_numpy(dtype=float),
        1.0e12,
    )
    output["base_surface_variance"] = np.where(
        available,
        output["field_surface_variance"].to_numpy(dtype=float),
        output["surface_variance_reference"].to_numpy(dtype=float),
    )
    output["base_path_s"] = output["path_s"]
    return output


def _attach_surface_results(
    query: pd.DataFrame,
    surface: pd.DataFrame,
) -> pd.DataFrame:
    payload = [
        column
        for column in surface.columns
        if column
        not in {"fold", "role", "well_id", "row_idx", "query_id", "MD"}
    ]
    joined = query.merge(
        surface[
            ["fold", "well_id", "row_idx", "query_id", "MD", *payload]
        ],
        on=["fold", "well_id", "row_idx", "query_id", "MD"],
        how="left",
        validate="one_to_one",
    )
    joined["surface_available"] = joined["surface_available"].fillna(False)
    return joined


def run_fold_target_free(
    *,
    fold: int,
    file_by_well: Mapping[str, Path],
    fold_by_well: Mapping[str, int],
    parent_safe: pd.DataFrame,
    selected_target_wells: set[str] | None,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fold_started = time.perf_counter()
    source_ids = sorted(
        well for well, assigned in fold_by_well.items() if int(assigned) != int(fold)
    )
    target_ids = sorted(
        well for well, assigned in fold_by_well.items() if int(assigned) == int(fold)
    )
    if selected_target_wells is not None:
        target_ids = [well for well in target_ids if well in selected_target_wells]
    if not target_ids:
        return {
            "fold": fold,
            "empty": True,
            "fixed_seconds": 0.0,
            "target_seconds": 0.0,
        }
    ledger.record_role_overlap(source_ids, target_ids)
    sources = [read_source_well(file_by_well[well], fold, ledger) for well in source_ids]
    targets = {
        well: read_target_safe_well(file_by_well[well], fold, ledger)
        for well in target_ids
    }
    surface_points = build_surface_points(sources, config)
    donor_catalog, donor_surface_coverage = build_multiscale_donor_catalog(
        sources, surface_points, config
    )
    fixed_seconds_before_donor_field = time.perf_counter()

    query_parts: list[pd.DataFrame] = []
    safe_key_parts: list[pd.DataFrame] = []
    for well in target_ids:
        query = build_query_grid(targets[well], config)
        query, safe_keys = attach_exp226_fallback(
            query,
            targets[well],
            parent_safe.loc[parent_safe["fold"].eq(fold)],
        )
        query_parts.append(query)
        safe_key_parts.append(safe_keys)
    query_raw = pd.concat(query_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "MD", "row_idx"], kind="mergesort"
    )
    safe_keys = pd.concat(safe_key_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "row_idx"], kind="mergesort"
    )
    query_surface = query_surface_fields(surface_points, query_raw, config)
    query = _attach_surface_results(query_raw, query_surface)
    query_surface_coverage = float(query["surface_available"].mean())

    available_query = query.loc[query["surface_available"]].copy()
    donor_catalog, signed_query, signature_stats = attach_standardized_signatures(
        donor_catalog,
        available_query,
        config,
    )
    query = query.merge(
        signed_query[
            ["fold", "well_id", "row_idx", "query_id", "MD", *SIGNATURE_COLUMNS]
        ],
        on=["fold", "well_id", "row_idx", "query_id", "MD"],
        how="left",
        validate="one_to_one",
    )

    primary = donor_catalog.loc[
        donor_catalog["window_scale_ft"].eq(256.0)
    ].copy()
    donor_fields = generate_vector_fields(
        donor_catalog,
        primary,
        config,
        exclude_same_well=True,
    )
    donor_vector_field_coverage = float(
        donor_fields["field_available"].fillna(False).mean()
    )
    donor_nodes_256 = build_exp384_donor_nodes(donor_catalog, donor_fields)
    fixed_seconds = time.perf_counter() - fold_started

    query_fields = generate_vector_fields(
        donor_catalog,
        available_query.merge(
            signed_query[
                ["fold", "well_id", "row_idx", "query_id", "MD", *SIGNATURE_COLUMNS]
            ],
            on=["fold", "well_id", "row_idx", "query_id", "MD"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_signed"),
        ),
        config,
        exclude_same_well=False,
    )
    query = attach_field_results(query, query_fields)
    surface_variance_reference = float(
        np.median(donor_catalog["surface_variance_mean"].to_numpy(dtype=float))
    )
    query["surface_variance_reference"] = surface_variance_reference
    prefix_records: list[dict[str, Any]] = []
    solved_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    for well in target_ids:
        block = query.loc[query["well_id"].eq(well)].copy()
        calibrated, prefix_record = calibrate_prefix_for_well(
            block, targets[well], config
        )
        safe = safe_keys.loc[safe_keys["well_id"].eq(well)].copy()
        solved, prediction = solve_and_interpolate_prediction(
            calibrated, safe, config
        )
        prefix_records.append(prefix_record)
        solved_parts.append(solved)
        prediction_parts.append(prediction)
    solved = pd.concat(solved_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "MD", "row_idx"], kind="mergesort"
    )
    prediction = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["fold", "well_id", "row_idx"], kind="mergesort"
    )
    exp384_query = to_exp384_query_contract(solved)
    target_seconds = time.perf_counter() - fold_started - fixed_seconds
    return {
        "fold": fold,
        "empty": False,
        "surface_points": surface_points,
        "donor_catalog": donor_catalog,
        "donor_nodes_256": donor_nodes_256,
        "signature_stats": signature_stats,
        "query_fields": exp384_query,
        "prefix_calibration": pd.DataFrame(prefix_records),
        "path": solved,
        "prediction": prediction,
        "donor_surface_coverage": donor_surface_coverage,
        "query_surface_coverage": query_surface_coverage,
        "donor_vector_field_coverage": donor_vector_field_coverage,
        "fixed_seconds": fixed_seconds,
        "target_seconds": max(target_seconds, 0.0),
        "processed_wells": len(target_ids),
        "source_preparation_checkpoint_seconds": (
            fixed_seconds_before_donor_field - fold_started
        ),
    }


# %% [markdown]
# ## 8. Stage 0 target-free freeze and integrity/resource gate

# %%
TARGET_FREE_SORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "surface_points": ("fold", "well_id", "MD", "row_idx"),
    "donor_catalog": ("fold", "well_id", "window_scale_ft", "MD"),
    "donor_nodes_256": ("fold", "well_id", "MD"),
    "signature_stats": ("fold", "dimension"),
    "query_fields": ("fold", "well_id", "MD", "row_idx"),
    "prefix_calibration": ("well_id",),
    "path": ("fold", "well_id", "MD", "row_idx"),
    "prediction": ("fold", "well_id", "row_idx"),
}


def freeze_target_free_outputs(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, str]:
    missing = sorted(set(TARGET_FREE_SORT_COLUMNS).difference(frames))
    if missing:
        raise ValueError(f"target-free freeze is missing frames: {missing}")
    return {
        name: frame_content_sha256(
            frames[name],
            sort_columns=TARGET_FREE_SORT_COLUMNS[name],
        )
        for name in sorted(TARGET_FREE_SORT_COLUMNS)
    }


def _finite_fraction(frame: pd.DataFrame, columns: Sequence[str]) -> float:
    if frame.empty or any(column not in frame for column in columns):
        return 0.0
    return float(
        np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all(axis=1).mean()
    )


def evaluate_stage0(
    *,
    frames: Mapping[str, pd.DataFrame],
    fold_results: Sequence[Mapping[str, Any]],
    ledger: RoleReadLedger,
    elapsed_seconds: float,
    full_run: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage0_target_free")
    query = frames["query_fields"]
    prediction = frames["prediction"]
    prefix = frames["prefix_calibration"]
    available = query["field_available"].astype(bool)
    support = query.loc[available, "field_support_ess"].to_numpy(dtype=float)
    wells = query.loc[available, "field_unique_wells"].to_numpy(dtype=float)
    processed_wells = int(prediction["well_id"].nunique())
    fixed_seconds = float(
        sum(float(result.get("fixed_seconds", 0.0)) for result in fold_results)
    )
    target_seconds = float(
        sum(float(result.get("target_seconds", 0.0)) for result in fold_results)
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    projected_runtime = (
        elapsed_seconds
        if full_run
        else fixed_seconds
        + target_seconds * expected_wells / max(processed_wells, 1)
    )
    donor_surface_coverage = min(
        float(result["donor_surface_coverage"])
        for result in fold_results
        if not bool(result.get("empty"))
    )
    query_surface_coverage = min(
        float(result["query_surface_coverage"])
        for result in fold_results
        if not bool(result.get("empty"))
    )
    donor_vector_field_coverage = min(
        float(result["donor_vector_field_coverage"])
        for result in fold_results
        if not bool(result.get("empty"))
    )
    finite_columns = (
        *SURFACE_COLUMNS,
        *SURFACE_GRAD_X_COLUMNS,
        *SURFACE_GRAD_Y_COLUMNS,
        *SURFACE_VARIANCE_COLUMNS,
        *SIGNATURE_COLUMNS,
        "base_absolute_s",
        "base_rate",
        "base_absolute_variance",
        "base_rate_variance",
        "fallback_rate",
        "base_path_s",
    )
    observed = {
        "score_rows": int(len(prediction)),
        "wells": processed_wells,
        "folds": sorted(prediction["fold"].astype(int).unique().tolist()),
        "surface_primary_coverage": min(
            donor_surface_coverage, query_surface_coverage
        ),
        "donor_surface_coverage": donor_surface_coverage,
        "query_surface_coverage": query_surface_coverage,
        "vector_field_coverage": min(
            float(available.mean()), donor_vector_field_coverage
        ),
        "query_vector_field_coverage": float(available.mean()),
        "donor_vector_field_coverage": donor_vector_field_coverage,
        "effective_donors_p05": (
            float(np.percentile(support, 5)) if len(support) else 0.0
        ),
        "unique_donor_wells_p05": (
            float(np.percentile(wells, 5)) if len(wells) else 0.0
        ),
        "prefix_calibration_coverage": float(prefix["available"].astype(bool).mean()),
        "finite_coverage": min(
            _finite_fraction(query, finite_columns),
            _finite_fraction(
                prediction, ("exp226_prediction", "exp383_prediction")
            ),
        ),
        "valid_reference_overlap": ledger.valid_reference_overlap,
        "valid_formation_reads": ledger.valid_formation_reads,
        "valid_suffix_truth_reads": ledger.valid_suffix_truth_reads,
        "projected_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": peak_rss_gb(),
    }
    checks = {
        "surface_primary_coverage": observed["surface_primary_coverage"]
        >= float(gates["surface_primary_coverage_min"]),
        "vector_field_coverage": observed["vector_field_coverage"]
        >= float(gates["vector_field_coverage_min"]),
        "effective_donors_p05": observed["effective_donors_p05"]
        >= float(gates["effective_donors_p05_min"]),
        "unique_donor_wells_p05": observed["unique_donor_wells_p05"]
        >= float(gates["unique_donor_wells_p05_min"]),
        "prefix_calibration_coverage": observed["prefix_calibration_coverage"]
        >= float(gates["prefix_calibration_coverage_min"]),
        "finite_coverage": observed["finite_coverage"]
        >= float(gates["finite_coverage_min"]),
        "valid_reference_overlap": observed["valid_reference_overlap"]
        <= int(gates["valid_reference_overlap_max"]),
        "valid_formation_reads": observed["valid_formation_reads"]
        <= int(gates["valid_formation_reads_max"]),
        "valid_suffix_truth_reads": observed["valid_suffix_truth_reads"]
        <= int(gates["valid_suffix_truth_reads_max"]),
        "projected_runtime": observed["projected_runtime_seconds"]
        <= float(gates["projected_runtime_seconds_max"]),
        "projected_peak_rss": observed["projected_peak_rss_gb"]
        <= float(gates["projected_peak_rss_gb_max"]),
    }
    if full_run:
        checks.update(
            {
                "score_rows": observed["score_rows"]
                == int(get_nested(config, "validation.expected_rows")),
                "wells": observed["wells"]
                == int(get_nested(config, "validation.expected_wells")),
                "folds": observed["folds"]
                == list(get_nested(config, "validation.expected_folds")),
            }
        )
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}


ARTIFACT_FILENAMES = {
    "surface_points": f"{EXPERIMENT_NAME}_surface_points.parquet",
    "donor_catalog": f"{EXPERIMENT_NAME}_donor_catalog.parquet",
    "donor_nodes_256": f"{EXPERIMENT_NAME}_donor_nodes_256.parquet",
    "signature_stats": f"{EXPERIMENT_NAME}_signature_stats.parquet",
    "query_fields": f"{EXPERIMENT_NAME}_query_fields.parquet",
    "prefix_calibration": f"{EXPERIMENT_NAME}_prefix_calibration.parquet",
    "path": f"{EXPERIMENT_NAME}_query_path.parquet",
    "prediction": f"{EXPERIMENT_NAME}_oof_keys_without_truth.parquet",
    "oof_with_truth": f"{EXPERIMENT_NAME}_oof_with_truth.parquet",
}


def artifact_records(
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, frame in frames.items():
        if name not in ARTIFACT_FILENAMES:
            continue
        filename = ARTIFACT_FILENAMES[name]
        path = output / filename
        write_table(frame, path)
        sort_columns = (
            TARGET_FREE_SORT_COLUMNS.get(name)
            or ("fold", "well_id", "row_idx")
        )
        sort_columns = tuple(column for column in sort_columns if column in frame.columns)
        records[name] = {
            "filename": filename,
            "rows": int(len(frame)),
            "schema_sha256": frame_schema_sha256(frame),
            "logical_content_sha256": frame_content_sha256(
                frame, sort_columns=sort_columns
            ),
            "logical_sort_columns": list(sort_columns),
            "file_sha256": sha256_file(path),
        }
    if "prediction" in records:
        records["oof_keys_without_truth"] = records.pop("prediction")
    return records


# %% [markdown]
# ## 9. Late truth join and Stage 1 readout

# %%
def load_hidden_like_assignments(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path]:
    filename = str(get_nested(config, "data.hidden_like_assignments.filename"))
    candidates = list(get_nested(config, "data.hidden_like_assignments.candidates", []))
    path = resolve_candidate_file(candidates, filename)
    expected = str(get_nested(config, "data.hidden_like_assignments.expected_sha256"))
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"hidden-like assignment SHA mismatch: expected {expected}, got {actual}"
        )
    roles = pd.read_csv(path, dtype={"well_id": str})
    require_columns(
        roles,
        (
            "well_id",
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        ),
        "hidden-like assignments",
    )
    if roles["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments contain duplicate well ids")
    return roles, path


def late_join_truth(
    prediction: pd.DataFrame,
    parent_truth: pd.DataFrame,
    hidden_roles: pd.DataFrame,
    frozen_hashes: Mapping[str, str],
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    if not ledger.truth_joined_after_freeze:
        ledger.mark_truth_join(frozen_hashes)
    truth = parent_truth[
        ["fold", "well_id", "row_idx", "tvt_true"]
    ].copy()
    scored = prediction.merge(
        truth,
        on=["fold", "well_id", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(prediction):
        raise ValueError("late truth join changed prediction row count")
    roles = hidden_roles[
        [
            "well_id",
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        ]
    ]
    scored = scored.merge(roles, on="well_id", how="left", validate="many_to_one")
    if scored["verification_like_spatial_role"].isna().any():
        raise ValueError("hidden-like assignments do not cover all OOF wells")
    scored["hidden_like_spatial"] = (
        scored["verification_like_spatial_role"].astype(str).eq("valid")
    )
    scored["hidden_like_typewell_purged"] = (
        scored["verification_like_typewell_purged_role"].astype(str).eq("valid")
    )
    return scored


def _metric_record(frame: pd.DataFrame, scope: str, value: str) -> dict[str, Any]:
    truth = frame["tvt_true"].to_numpy(dtype=float)
    control = frame["exp226_prediction"].to_numpy(dtype=float)
    candidate = frame["exp383_prediction"].to_numpy(dtype=float)
    control_rmse = rmse(truth, control)
    candidate_rmse = rmse(truth, candidate)
    return {
        "scope": scope,
        "scope_value": value,
        "rows": int(len(frame)),
        "exp226_rmse": control_rmse,
        "exp383_rmse": candidate_rmse,
        "gain_exp226_minus_exp383": control_rmse - candidate_rmse,
        "delta_exp383_minus_exp226": candidate_rmse - control_rmse,
    }


def build_stage1_readout(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = [_metric_record(scored, "pooled", "all")]
    for fold, block in scored.groupby("fold", sort=True):
        rows.append(_metric_record(block, "fold", str(int(fold))))
    distance = scored["distance_from_anchor"].to_numpy(dtype=float)
    scopes = {
        "near_0_250": (distance >= 0.0) & (distance < 250.0),
        "mid_250_1000": (distance >= 250.0) & (distance < 1000.0),
        "long_1000_plus": distance >= 1000.0,
        "hidden_like_spatial": scored["hidden_like_spatial"].to_numpy(dtype=bool),
        "hidden_like_typewell_purged": scored[
            "hidden_like_typewell_purged"
        ].to_numpy(dtype=bool),
    }
    for name, mask in scopes.items():
        if mask.any():
            rows.append(_metric_record(scored.loc[mask], "scope", name))
    by_well = pd.DataFrame(
        [
            {
                "well_id": str(well),
                **_metric_record(block, "well", str(well)),
            }
            for well, block in scored.groupby("well_id", sort=True)
        ]
    )
    delta = by_well["delta_exp383_minus_exp226"].to_numpy(dtype=float)
    tail = {
        "improved_wells": int((delta < 0).sum()),
        "worse_wells": int((delta > 0).sum()),
        "same_wells": int((delta == 0).sum()),
        "delta_p95_ft": float(np.percentile(delta, 95)),
        "delta_worst_ft": float(np.max(delta)),
        "worse_by_1ft_wells": int((delta >= 1.0).sum()),
        "worse_by_3ft_wells": int((delta >= 3.0).sum()),
        "worse_by_5ft_wells": int((delta >= 5.0).sum()),
    }
    return pd.DataFrame(rows), by_well, tail


def evaluate_stage1(
    metrics: pd.DataFrame,
    scored: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage1_direct")
    pooled = metrics.loc[
        metrics["scope"].eq("pooled") & metrics["scope_value"].eq("all")
    ].iloc[0]
    folds = metrics.loc[metrics["scope"].eq("fold")]
    scopes = metrics.loc[metrics["scope"].eq("scope")].set_index("scope_value")

    def gain(name: str) -> float:
        if name not in scopes.index:
            return float("-inf")
        return float(scopes.loc[name, "gain_exp226_minus_exp383"])

    positive_folds = int((folds["gain_exp226_minus_exp383"] > 0).sum())
    correlation = float(
        np.corrcoef(
            scored["exp226_prediction"].to_numpy(dtype=float),
            scored["exp383_prediction"].to_numpy(dtype=float),
        )[0, 1]
    )
    observed = {
        "pooled_rmse_gain_vs_exp226_ft": float(
            pooled["gain_exp226_minus_exp383"]
        ),
        "pooled_rmse_ft": float(pooled["exp383_rmse"]),
        "positive_folds": positive_folds,
        "long_1000_plus_gain_ft": gain("long_1000_plus"),
        "hidden_like_spatial_gain_ft": gain("hidden_like_spatial"),
        "hidden_like_typewell_purged_gain_ft": gain(
            "hidden_like_typewell_purged"
        ),
        "near_0_250_delta_ft": -gain("near_0_250"),
        "correlation_vs_exp226": correlation,
    }
    checks = {
        "pooled_gain": observed["pooled_rmse_gain_vs_exp226_ft"]
        >= float(gates["pooled_rmse_gain_vs_exp226_ft_min"]),
        "pooled_rmse": observed["pooled_rmse_ft"]
        <= float(gates["pooled_rmse_max_ft"]),
        "positive_folds": observed["positive_folds"]
        >= int(gates["positive_folds_min"]),
        "long_1000_plus": observed["long_1000_plus_gain_ft"]
        >= float(gates["long_1000_plus_gain_ft_min"]),
        "hidden_like_spatial": observed["hidden_like_spatial_gain_ft"]
        >= float(gates["hidden_like_spatial_gain_ft_min"]),
        "hidden_like_typewell_purged": observed[
            "hidden_like_typewell_purged_gain_ft"
        ]
        >= float(gates["hidden_like_typewell_purged_gain_ft_min"]),
        "near_0_250": observed["near_0_250_delta_ft"]
        <= float(gates["near_0_250_regression_max_ft"]),
        "correlation": observed["correlation_vs_exp226"]
        <= float(gates["max_correlation_vs_exp226"]),
    }
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}


def build_manifest(
    *,
    stage0: Mapping[str, Any],
    stage1: Mapping[str, Any] | None,
    records: Mapping[str, Any],
    prefix: pd.DataFrame,
    parent_path: Path,
    hidden_path: Path | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    finite_prefix_rmse = prefix["prefix_huber_rmse_ft"].to_numpy(dtype=float)
    finite_prefix_rmse = finite_prefix_rmse[np.isfinite(finite_prefix_rmse)]
    return {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "stage0": stage0,
        "stage1": stage1,
        "validation": {
            "score_rows": int(get_nested(config, "validation.expected_rows")),
            "wells": int(get_nested(config, "validation.expected_wells")),
            "folds": list(get_nested(config, "validation.expected_folds")),
        },
        "parent": {
            "experiment": PARENT_EXPERIMENT,
            "oof_path": str(parent_path),
            "oof_decompressed_sha256": sha256_decompressed_csv(parent_path),
            "regenerated": False,
        },
        "hidden_like_assignment": (
            {
                "path": str(hidden_path),
                "sha256": sha256_file(hidden_path),
            }
            if hidden_path is not None
            else None
        ),
        "calibration": {
            "prefix_loo_huber_rmse_p50_ft": (
                float(np.median(finite_prefix_rmse))
                if len(finite_prefix_rmse)
                else None
            )
        },
        "solver_contract": {
            "logical_sha256": sha256_bytes(
                stable_json_bytes(
                    {
                        "method": get_nested(config, "method"),
                        "validation": get_nested(config, "validation"),
                        "gates": get_nested(config, "gates"),
                    }
                )
            )
        },
        "artifacts": dict(records),
        "deterministic_anchor": False,
    }


# %% [markdown]
# ## 10. Setup, configuration preview, and execution

# %%
def run_train() -> dict[str, Any]:
    config = load_config()
    validate_execution_contract(config, require_kaggle_authorization=True)
    train_dir = resolve_train_dir(config)
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    file_by_well = {well_id_from_path(path): path for path in files}
    if len(file_by_well) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError(
            f"expected {get_nested(config, 'validation.expected_wells')} train wells, "
            f"found {len(file_by_well)}"
        )
    parent_safe, parent_path = load_parent_oof(config, include_truth=False)
    fold_by_well = assign_group_folds(
        sorted(file_by_well),
        int(get_nested(config, "validation.n_folds")),
        int(get_nested(config, "validation.seed")),
    )
    validate_fold_identity(fold_by_well, parent_safe)

    maximum_wells_raw = os.environ.get("EXP383_MAX_WELLS")
    if (
        maximum_wells_raw is None
        and get_nested(config, "execution.current_mode")
        == "stage0_resource_preflight"
    ):
        maximum_wells_raw = str(
            int(get_nested(config, "execution.preflight_max_wells"))
        )
    selected_target_wells: set[str] | None = None
    if maximum_wells_raw:
        maximum_wells = int(maximum_wells_raw)
        selected_target_wells = set(sorted(file_by_well)[:maximum_wells])
    full_run = selected_target_wells is None
    ledger = RoleReadLedger()
    started = time.perf_counter()
    fold_results = [
        run_fold_target_free(
            fold=fold,
            file_by_well=file_by_well,
            fold_by_well=fold_by_well,
            parent_safe=parent_safe,
            selected_target_wells=selected_target_wells,
            ledger=ledger,
            config=config,
        )
        for fold in list(get_nested(config, "validation.expected_folds"))
    ]
    nonempty = [result for result in fold_results if not bool(result.get("empty"))]
    if not nonempty:
        raise ValueError("no target wells were selected")
    frames = {
        name: pd.concat(
            [result[name] for result in nonempty], ignore_index=True
        ).sort_values(list(TARGET_FREE_SORT_COLUMNS[name]), kind="mergesort")
        for name in TARGET_FREE_SORT_COLUMNS
    }
    prediction_contract = frames["prediction"][
        [
            "fold",
            "well_id",
            "row_idx",
            "MD",
            "Z",
            "exp226_prediction",
            "exp383_prediction",
            "exp383_path_status",
            "distance_from_anchor",
        ]
    ].copy()
    frames["prediction"] = prediction_contract
    elapsed = time.perf_counter() - started
    frozen_hashes = freeze_target_free_outputs(frames)
    stage0 = evaluate_stage0(
        frames=frames,
        fold_results=fold_results,
        ledger=ledger,
        elapsed_seconds=elapsed,
        full_run=full_run,
        config=config,
    )
    output = artifacts_dir()
    records = artifact_records(frames, output)
    write_json(output / f"{EXPERIMENT_NAME}_target_free_sha.json", frozen_hashes)
    metrics: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_pass" if stage0["passed"] else "stage0_fail_closed",
        "stage0": stage0,
        "stage1": None,
        "target_free_sha256": frozen_hashes,
        "ledger": ledger.as_dict(),
        "runtime_seconds": elapsed,
        "full_run": full_run,
    }
    hidden_path: Path | None = None
    if not stage0["passed"]:
        manifest = build_manifest(
            stage0=stage0,
            stage1=None,
            records=records,
            prefix=frames["prefix_calibration"],
            parent_path=parent_path,
            hidden_path=None,
            config=config,
        )
        write_json(output / f"{EXPERIMENT_NAME}_manifest.json", manifest)
        write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
        return metrics
    if not full_run:
        metrics["status"] = "stage0_resource_preflight_pass"
        metrics["stage1"] = {
            "passed": None,
            "status": "not_opened_during_resource_preflight",
        }
        manifest = build_manifest(
            stage0=stage0,
            stage1=metrics["stage1"],
            records=records,
            prefix=frames["prefix_calibration"],
            parent_path=parent_path,
            hidden_path=None,
            config=config,
        )
        write_json(output / f"{EXPERIMENT_NAME}_manifest.json", manifest)
        write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
        return metrics

    parent_truth, _ = load_parent_oof(
        config,
        include_truth=True,
        frozen_hashes=frozen_hashes,
        ledger=ledger,
    )
    hidden_roles, hidden_path = load_hidden_like_assignments(config)
    scored = late_join_truth(
        frames["prediction"],
        parent_truth,
        hidden_roles,
        frozen_hashes,
        ledger,
    )
    stage1_metrics, by_well, by_well_tail = build_stage1_readout(scored)
    stage1 = evaluate_stage1(stage1_metrics, scored, config)
    stage1["by_well_tail"] = by_well_tail
    metrics["stage1"] = stage1
    metrics["status"] = "stage1_pass" if stage1["passed"] else "stage1_fail_closed"
    metrics["ledger"] = ledger.as_dict()
    records.update(
        artifact_records({"oof_with_truth": scored}, output)
    )
    write_table(
        stage1_metrics, output / f"{EXPERIMENT_NAME}_stage1_metrics.csv"
    )
    write_table(by_well, output / f"{EXPERIMENT_NAME}_by_well_metrics.csv")
    manifest = build_manifest(
        stage0=stage0,
        stage1=stage1,
        records=records,
        prefix=frames["prefix_calibration"],
        parent_path=parent_path,
        hidden_path=hidden_path,
        config=config,
    )
    write_json(output / f"{EXPERIMENT_NAME}_manifest.json", manifest)
    write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
    return metrics


# %%
CONFIG_PREVIEW = load_config()
validate_execution_contract(CONFIG_PREVIEW, require_kaggle_authorization=False)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG_PREVIEW, "experiment.route"))
print("Parent:", get_nested(CONFIG_PREVIEW, "lineage.parent"))
print("Status:", get_nested(CONFIG_PREVIEW, "experiment.status"))
print(
    "Execution authorized:",
    get_nested(CONFIG_PREVIEW, "execution.kaggle_execution_authorized"),
)
print(
    "Counts:",
    {
        "scientific_candidates": get_nested(
            CONFIG_PREVIEW, "runtime.scientific_candidates"
        ),
        "reporting_folds": get_nested(CONFIG_PREVIEW, "runtime.reporting_folds"),
        "models": get_nested(CONFIG_PREVIEW, "runtime.fitted_models"),
        "hmm": get_nested(CONFIG_PREVIEW, "runtime.hmm_runs"),
        "pf": get_nested(CONFIG_PREVIEW, "runtime.pf_runs"),
        "beam": get_nested(CONFIG_PREVIEW, "runtime.beam_runs"),
        "boosters": get_nested(CONFIG_PREVIEW, "runtime.lightgbm_boosters"),
        "parent_control_replay": get_nested(
            CONFIG_PREVIEW, "runtime.replay_parent_control"
        ),
    },
)
if os.environ.get(IMPORT_ONLY_ENV) != "1":
    RESULT = run_train()
    print(json.dumps(RESULT, indent=2, default=str))
