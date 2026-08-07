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
# # exp377 formation-relative K16 slope identifiability readout
#
# This CPU-only diagnostic keeps the exp226 outer folds, K16 boundaries,
# directional projection, and XY local-linear donor kernel fixed. For each of
# six train-only formation surfaces it interpolates `d(S-F)/dMD`, adds an
# outer-train-only estimate of `dF/dMD`, and evaluates the pre-registered
# median-of-six reconstruction only after the complete target-free bundle has
# been frozen. No HMM, PF, fitted model, current-test prediction, inference, or
# submission is created.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. K16 geometry, formation-surface, and XY-kernel helpers
# 4. Exp226 fold identity and role-safe raw-data loaders
# 5. Fold-local relative-rate construction and target-free freeze
# 6. Stage 0 integrity gate
# 7. Truth late join and Stage 1 identifiability readout
# 8. Metrics, generated artifacts, and fail-closed decision
# 9. Setup, configuration preview, and execution

# %% [markdown]
# ## 1. Imports and immutable experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp377_formation_relative_k16_slope_identifiability_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
FORMATION_COLUMNS = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
FORMATION_CANDIDATES = tuple(f"frk16_{name}" for name in FORMATION_COLUMNS)
PRIMARY_CANDIDATE = "frk16_robust_median6"
SAFE_OOF_COLUMNS = ("well_id", "row_idx", "suffix_offset", "fold")
FORBIDDEN_PRE_FREEZE_COLUMNS = (
    "tvt_true",
    "TVT",
    "error",
    "abs_error",
    "oracle_rank",
)
EXPECTED_EXECUTION = {
    "scientific_variants": 1,
    "reporting_surfaces": 6,
    "reporting_folds": 5,
    "model_configs": 0,
    "trained_folds": 0,
    "lightgbm_boosters": 0,
    "hmm_runs": 0,
    "pf_runs": 0,
}
EXECUTE_NOTEBOOK = os.environ.get("EXP377_IMPORT_ONLY", "0") != "1"


# %% [markdown]
# ## 2. Runtime, path, SHA, and serialization helpers

# %%
def in_notebook_runtime() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "project.yml").exists():
            return candidate
    return Path.cwd()


def experiment_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return project_root() / "experiments" / EXPERIMENT_NAME


def runtime_artifacts_dir() -> Path:
    path = experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    return experiment_dir() / "metrics.json"


def find_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
        project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp377 config.yaml was not found")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(
    path: Path,
    chunk_bytes: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_object_cell(value: Any) -> Any:
    if isinstance(value, Mapping):
        return json.dumps(
            to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(
            to_jsonable(value),
            separators=(",", ":"),
            ensure_ascii=False,
        )
    if value is None:
        return "<NA>"
    return value


def normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(_normalize_object_cell)
        elif isinstance(result[column].dtype, pd.StringDtype):
            result[column] = result[column].fillna("<NA>").astype(str)
    return result


def frame_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    selected = normalized_frame(selected)
    hashed = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype=np.uint64
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(selected.columns), separators=(",", ":")).encode()
    )
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    return json_sha256(
        [
            {"column": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ]
    )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, ensure_ascii=False) + "\n"
    )


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def write_gzip_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": sha256_decompressed_gzip(path),
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }


def expand_existing_paths(patterns: Sequence[str]) -> list[Path]:
    root = project_root()
    found: dict[str, Path] = {}
    for raw in patterns:
        candidate = Path(raw)
        direct = candidate if candidate.is_absolute() else root / candidate
        if direct.exists():
            found[str(direct.resolve())] = direct
        if any(character in raw for character in "*?[]"):
            base = Path("/") if candidate.is_absolute() else root
            pattern = raw.lstrip("/") if candidate.is_absolute() else raw
            for path in base.glob(pattern):
                found[str(path.resolve())] = path
    return [found[key] for key in sorted(found)]


def resolve_file(
    patterns: Sequence[str],
    *,
    expected_sha256: str | None = None,
    decompressed: bool = False,
) -> Path:
    candidates = [path for path in expand_existing_paths(patterns) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no file matched: {patterns}")
    if expected_sha256 is None:
        return candidates[0]
    for path in candidates:
        actual = (
            sha256_decompressed_gzip(path)
            if decompressed
            else sha256_file(path)
        )
        if actual == expected_sha256:
            return path
    evidence = {
        str(path): (
            sha256_decompressed_gzip(path)
            if decompressed
            else sha256_file(path)
        )
        for path in candidates
    }
    raise ValueError(f"no candidate matched expected SHA: {evidence}")


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_kaggle_authorization: bool,
) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp377 route must be pf_beam")
    for key, expected in EXPECTED_EXECUTION.items():
        actual = int(get_nested(config, f"runtime.{key}", -1))
        if actual != expected:
            raise ValueError(
                f"runtime.{key} changed: expected {expected}, got {actual}"
            )
    if bool(get_nested(config, "runtime.parent_control_regeneration")):
        raise ValueError("parent/control regeneration is forbidden")
    if bool(get_nested(config, "execution.inference_enabled")):
        raise ValueError("exp377 inference must remain disabled")
    if bool(get_nested(config, "execution.submission_enabled")):
        raise ValueError("exp377 submission must remain disabled")
    fixed_method_values = {
        "method.segment_count": 16,
        "method.directional_projection.theta0_deg": 118.4,
        "method.directional_projection.minimum_abs_projection": 0.3,
        "method.donor_kernel.nearest_segments": 50,
        "method.donor_kernel.bandwidth_ft": 500.0,
        "method.donor_kernel.ridge": 1.0,
        "method.formation_surface.nearest_wells": 10,
        "method.primary_aggregation.minimum_finite_formations": 6,
    }
    for path, expected in fixed_method_values.items():
        actual = float(get_nested(config, path, float("nan")))
        if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{path} changed: expected {expected}, got {actual}"
            )
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized")
    ):
        raise RuntimeError(
            "Kaggle execution is not authorized. Implementation and static "
            "validation are complete, but package/push/run requires separate "
            "user direction."
        )
    target_version = int(get_nested(config, "execution.kaggle_target_version", 1))
    if require_kaggle_authorization and target_version == 2:
        if not bool(get_nested(config, "execution.kaggle_v2_execution_authorized")):
            raise RuntimeError("Kaggle v2 execution is not authorized")
        report_only = list(
            get_nested(config, "gates.stage0_integrity.report_only_checks", [])
        )
        if report_only != ["effective_donors_p05"]:
            raise ValueError(
                "v2 must retain effective_donors_p05 as the sole "
                f"report-only check, got {report_only}"
            )


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not finite.any():
        return float("nan")
    return float(
        np.sqrt(np.mean(np.square(actual_array[finite] - predicted_array[finite])))
    )


# %% [markdown]
# ## 3. K16 geometry, formation-surface, and XY-kernel helpers

# %%
@dataclass(frozen=True)
class K16Contract:
    segments: int = 16
    theta0_deg: float = 118.4
    minimum_abs_projection: float = 0.3
    local_linear_k: int = 50
    bandwidth_ft: float = 500.0
    ridge: float = 1.0
    formation_k: int = 10
    minimum_finite_formations: int = 6


@dataclass(frozen=True)
class WellGeometry:
    well_id: str
    fold: int
    row_count: int
    anchor_row_idx: int
    suffix_row_idx: np.ndarray
    suffix_offset: np.ndarray
    segment_id: np.ndarray
    segment_start_idx: np.ndarray
    segment_end_idx: np.ndarray
    segment_mid_xy: np.ndarray
    segment_projection: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    md: np.ndarray
    tvt_input: np.ndarray

    @property
    def suffix_rows(self) -> int:
        return len(self.suffix_row_idx)

    @property
    def anchor_s(self) -> float:
        return float(
            self.tvt_input[self.anchor_row_idx] + self.z[self.anchor_row_idx]
        )


@dataclass(frozen=True)
class KernelPrediction:
    values: np.ndarray
    effective_donors: np.ndarray
    selected_donors: np.ndarray
    nearest_distance: np.ndarray
    fallback: np.ndarray


@dataclass
class FormationPlaneKNN:
    wells: np.ndarray
    xy: np.ndarray
    formation_medians: np.ndarray
    k: int = 10

    def __post_init__(self) -> None:
        self.wells = np.asarray(self.wells, dtype=object)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.formation_medians = np.asarray(
            self.formation_medians, dtype=np.float64
        )
        if len(self.wells) < self.k:
            raise ValueError("formation plane requires at least k references")
        if self.xy.shape != (len(self.wells), 2):
            raise ValueError("formation reference XY shape mismatch")
        if self.formation_medians.shape != (
            len(self.wells),
            len(FORMATION_COLUMNS),
        ):
            raise ValueError("formation reference value shape mismatch")
        if not (
            np.isfinite(self.xy).all()
            and np.isfinite(self.formation_medians).all()
        ):
            raise ValueError("formation reference contains nonfinite values")
        scale = np.std(self.xy, axis=0)
        self.scale = np.where(scale < 1.0e-3, 1.0, scale)

    def predict(
        self,
        query_xy: np.ndarray,
        *,
        target_well: str | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        query = np.atleast_2d(np.asarray(query_xy, dtype=np.float64))
        output = np.empty((len(query), len(FORMATION_COLUMNS)), dtype=np.float64)
        fallback = np.zeros(len(query), dtype=bool)
        effective = np.empty(len(query), dtype=np.float64)
        nearest = np.empty(len(query), dtype=np.float64)
        own_mask = (
            self.wells.astype(str) == str(target_well)
            if target_well is not None
            else np.zeros(len(self.wells), dtype=bool)
        )
        scaled_reference = self.xy / self.scale
        well_keys = self.wells.astype(str)
        stable_index = np.arange(len(self.wells), dtype=np.int64)
        for row_index, point in enumerate(query):
            distance = np.sqrt(
                np.square(scaled_reference - point / self.scale).sum(axis=1)
            )
            distance[own_mask] = np.inf
            order = np.lexsort((stable_index, well_keys, distance))
            selected = order[np.isfinite(distance[order])][: self.k]
            if len(selected) != self.k:
                raise ValueError("formation query lacks k role-safe references")
            selected_distance = distance[selected]
            weight = 1.0 / (selected_distance + 1.0e-3)
            weight_sum = float(weight.sum())
            effective[row_index] = weight_sum**2 / float(np.square(weight).sum())
            nearest[row_index] = float(selected_distance.min())
            design = np.column_stack(
                [
                    self.xy[selected, 0] - point[0],
                    self.xy[selected, 1] - point[1],
                    np.ones(len(selected)),
                ]
            )
            normal = (design * weight[:, None]).T @ design
            normal += np.diag([1.0e-9, 1.0e-9, 1.0e-12])
            rhs = (design * weight[:, None]).T @ self.formation_medians[selected]
            try:
                coefficient = np.linalg.solve(normal, rhs)
                prediction = coefficient[2]
            except np.linalg.LinAlgError:
                prediction = np.average(
                    self.formation_medians[selected],
                    axis=0,
                    weights=weight,
                )
                fallback[row_index] = True
            if not np.isfinite(prediction).all():
                prediction = np.average(
                    self.formation_medians[selected],
                    axis=0,
                    weights=weight,
                )
                fallback[row_index] = True
            output[row_index] = prediction
        return output, {
            "fallback": fallback,
            "effective_donors": effective,
            "nearest_distance": nearest,
        }


def k16_segment_ids(n_rows: int, segments: int = 16) -> np.ndarray:
    if n_rows <= 0:
        raise ValueError("K16 requires a non-empty suffix")
    edges = np.linspace(0.0, float(n_rows), segments + 1)
    step_index = np.arange(1.0, n_rows + 1.0)
    return np.clip(
        np.searchsorted(edges[1:], step_index, side="left"),
        0,
        segments - 1,
    ).astype(np.int16)


def segment_bounds(
    suffix_row_idx: np.ndarray,
    segment_id: np.ndarray,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    start = np.empty(segments, dtype=np.int64)
    end = np.empty(segments, dtype=np.int64)
    for segment in range(segments):
        selected = suffix_row_idx[segment_id == segment]
        if len(selected) == 0:
            raise ValueError(f"K16 segment {segment} is empty")
        start[segment] = int(selected[0])
        end[segment] = int(selected[-1])
    return start, end


def segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    suffix_row_idx: np.ndarray,
    segment_id: np.ndarray,
    contract: K16Contract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start, end = segment_bounds(
        suffix_row_idx,
        segment_id,
        contract.segments,
    )
    midpoint = np.column_stack(
        [
            (x[start] + x[end]) / 2.0,
            (y[start] + y[end]) / 2.0,
        ]
    )
    azimuth = np.arctan2(y[end] - y[start], x[end] - x[start])
    projection = np.cos(azimuth - np.radians(contract.theta0_deg))
    return start, end, midpoint, projection


def median_segment_step_rates(
    md: np.ndarray,
    values: np.ndarray,
    suffix_row_idx: np.ndarray,
    segment_id: np.ndarray,
    segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    md = np.asarray(md, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    rates = np.full(segments, np.nan, dtype=np.float64)
    counts = np.zeros(segments, dtype=np.int32)
    for segment in range(segments):
        rows = suffix_row_idx[segment_id == segment]
        if len(rows) < 2:
            continue
        delta_md = np.diff(md[rows])
        delta_value = np.diff(values[rows])
        finite = (
            np.isfinite(delta_md)
            & np.isfinite(delta_value)
            & (delta_md > 0.0)
        )
        step_rate = delta_value[finite] / delta_md[finite]
        counts[segment] = len(step_rate)
        if len(step_rate):
            rates[segment] = float(np.median(step_rate))
    return rates, counts


def stable_nearest_indices(
    squared_distance: np.ndarray,
    donor_well: np.ndarray,
    donor_segment: np.ndarray,
    k: int,
) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(squared_distance))
    if len(finite) == 0:
        return finite
    order = np.lexsort(
        (
            donor_segment[finite],
            donor_well[finite].astype(str),
            squared_distance[finite],
        )
    )
    return finite[order[: min(k, len(order))]]


def local_linear_predict(
    field: pd.DataFrame,
    query_xy: np.ndarray,
    contract: K16Contract,
) -> KernelPrediction:
    required = {"x", "y", "value", "donor_well", "donor_segment"}
    if not required.issubset(field.columns):
        raise ValueError(f"donor field misses {sorted(required - set(field.columns))}")
    donor_xy = field[["x", "y"]].to_numpy(np.float64)
    donor_value = field["value"].to_numpy(np.float64)
    donor_well = field["donor_well"].astype(str).to_numpy()
    donor_segment = field["donor_segment"].to_numpy(np.int16)
    query = np.atleast_2d(np.asarray(query_xy, dtype=np.float64))
    values = np.full(len(query), np.nan, dtype=np.float64)
    effective = np.zeros(len(query), dtype=np.float64)
    selected_count = np.zeros(len(query), dtype=np.int32)
    nearest = np.full(len(query), np.inf, dtype=np.float64)
    fallback = np.zeros(len(query), dtype=bool)
    for row_index, point in enumerate(query):
        squared_distance = np.square(donor_xy - point).sum(axis=1)
        selected = stable_nearest_indices(
            squared_distance,
            donor_well,
            donor_segment,
            contract.local_linear_k,
        )
        selected_count[row_index] = len(selected)
        if len(selected) == 0:
            fallback[row_index] = True
            continue
        selected_d2 = squared_distance[selected]
        nearest[row_index] = float(np.sqrt(selected_d2.min()))
        weight = np.exp(
            np.maximum(
                -selected_d2 / (2.0 * contract.bandwidth_ft**2),
                -700.0,
            )
        )
        weight_sum = float(weight.sum())
        effective[row_index] = weight_sum**2 / float(np.square(weight).sum())
        delta = (donor_xy[selected] - point) / 1000.0
        design = np.column_stack([np.ones(len(selected)), delta])
        ridge = contract.ridge * weight_sum * np.diag([0.0, 1.0, 1.0])
        normal = (design * weight[:, None]).T @ design + ridge
        rhs = (design * weight[:, None]).T @ donor_value[selected]
        try:
            values[row_index] = float(np.linalg.solve(normal, rhs)[0])
        except np.linalg.LinAlgError:
            values[row_index] = float(
                np.linalg.lstsq(
                    normal + np.eye(3) * 1.0e-9,
                    rhs,
                    rcond=None,
                )[0][0]
            )
            fallback[row_index] = True
    return KernelPrediction(
        values=values,
        effective_donors=effective,
        selected_donors=selected_count,
        nearest_distance=nearest,
        fallback=fallback,
    )


# %% [markdown]
# ## 4. Exp226 fold identity and role-safe raw-data loaders
#
# The pre-freeze exp226 reader has an explicit four-column allowlist. Raw
# outer-valid files are opened once with `X/Y/Z/MD/TVT_input`; they never carry
# `TVT` or a formation column. Each fold then opens those columns only for its
# outer-train source wells.

# %%
def resolve_raw_train_dir(
    config: Mapping[str, Any],
    expected_wells: set[str] | None = None,
) -> tuple[Path, list[Path]]:
    patterns = [
        str(value)
        for value in get_nested(config, "data.raw_train_dir_patterns", [])
    ]
    directories = [path for path in expand_existing_paths(patterns) if path.is_dir()]
    glob_pattern = str(get_nested(config, "data.raw_horizontal_glob"))
    evidence: list[dict[str, Any]] = []
    for directory in directories:
        files = sorted(directory.glob(glob_pattern))
        wells = {
            path.name.split("__horizontal_well.csv", 1)[0] for path in files
        }
        evidence.append(
            {"directory": str(directory), "files": len(files), "wells": len(wells)}
        )
        if expected_wells is None or wells == expected_wells:
            return directory, files
    raise FileNotFoundError(
        f"no raw train directory matched the well inventory: {evidence}"
    )


def load_exp226_fold_identity(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    specification = get_nested(config, "data.exp226_oof")
    path = resolve_file(
        [str(value) for value in specification["patterns"]],
        expected_sha256=str(specification["expected_decompressed_sha256"]),
        decompressed=True,
    )
    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(SAFE_OOF_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"exp226 OOF misses safe identity columns: {missing}")
    frame = pd.read_csv(
        path,
        usecols=list(SAFE_OOF_COLUMNS),
        dtype={
            "well_id": "string",
            "row_idx": "int32",
            "suffix_offset": "int32",
            "fold": "int8",
        },
    )
    frame["well_id"] = frame["well_id"].astype(str)
    frame = frame.sort_values(
        ["well_id", "row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 fold identity contains duplicate well/row keys")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("each well must have exactly one exp226 outer fold")
    for forbidden in FORBIDDEN_PRE_FREEZE_COLUMNS:
        if forbidden in frame.columns:
            raise RuntimeError(f"forbidden pre-freeze column loaded: {forbidden}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = {
        int(value) for value in get_nested(config, "validation.expected_folds")
    }
    if len(frame) != expected_rows:
        raise ValueError(
            f"exp226 row count mismatch: {len(frame)} != {expected_rows}"
        )
    if frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp226 well count mismatch")
    if set(frame["fold"].astype(int).unique()) != expected_folds:
        raise ValueError("exp226 fold inventory mismatch")
    evidence = {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": sha256_decompressed_gzip(path),
        "rows": len(frame),
        "wells": frame["well_id"].nunique(),
        "safe_columns_loaded": list(frame.columns),
        "source_columns": list(header.columns),
        "forbidden_columns_loaded": [],
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }
    return frame, evidence


def build_raw_input_manifest(horizontal_files: Sequence[Path]) -> pd.DataFrame:
    records = []
    for path in horizontal_files:
        records.append(
            {
                "well_id": path.name.split("__horizontal_well.csv", 1)[0],
                "path": str(path),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(records).sort_values(
        "well_id",
        kind="mergesort",
    ).reset_index(drop=True)


def last_known_index(tvt_input: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(tvt_input))
    if len(finite) == 0:
        raise ValueError("well has no visible TVT_input anchor")
    index = int(finite.max())
    if not np.isfinite(tvt_input[: index + 1]).all():
        raise ValueError("TVT_input prefix is not contiguous")
    if np.isfinite(tvt_input[index + 1 :]).any():
        raise ValueError("TVT_input suffix contains a finite value after the cut")
    return index


def build_target_free_geometry(
    path: Path,
    identity: pd.DataFrame,
    contract: K16Contract,
) -> WellGeometry:
    safe_columns = ["X", "Y", "Z", "MD", "TVT_input"]
    frame = pd.read_csv(path, usecols=safe_columns)
    if any(column in frame.columns for column in ("TVT", *FORMATION_COLUMNS)):
        raise RuntimeError("truth or formation crossed the target-free loader")
    well_id = path.name.split("__horizontal_well.csv", 1)[0]
    identity = identity.sort_values("row_idx", kind="mergesort").reset_index(
        drop=True
    )
    row_idx = identity["row_idx"].to_numpy(np.int64)
    suffix_offset = identity["suffix_offset"].to_numpy(np.int32)
    if not np.array_equal(suffix_offset, np.arange(len(identity), dtype=np.int32)):
        raise ValueError(f"{well_id} suffix_offset is not contiguous")
    tvt_input = frame["TVT_input"].to_numpy(np.float64)
    anchor = last_known_index(tvt_input)
    unknown = np.flatnonzero(~np.isfinite(tvt_input))
    if not np.array_equal(row_idx, unknown):
        raise ValueError(f"{well_id} exp226 rows differ from raw unknown suffix")
    md = frame["MD"].to_numpy(np.float64)
    x = frame["X"].to_numpy(np.float64)
    y = frame["Y"].to_numpy(np.float64)
    z = frame["Z"].to_numpy(np.float64)
    if not (
        np.isfinite(md).all()
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.isfinite(z).all()
    ):
        raise ValueError(f"{well_id} target-free geometry contains nonfinite values")
    delta_md = np.diff(md[np.r_[anchor, row_idx]])
    if not np.isfinite(delta_md).all() or np.any(delta_md <= 0.0):
        raise ValueError(f"{well_id} MD is not strictly increasing across suffix")
    segment_id = k16_segment_ids(len(row_idx), contract.segments)
    start, end, midpoint, projection = segment_geometry(
        x,
        y,
        row_idx,
        segment_id,
        contract,
    )
    fold_values = identity["fold"].astype(int).unique()
    if len(fold_values) != 1:
        raise ValueError(f"{well_id} has multiple folds")
    return WellGeometry(
        well_id=well_id,
        fold=int(fold_values[0]),
        row_count=len(frame),
        anchor_row_idx=anchor,
        suffix_row_idx=row_idx,
        suffix_offset=suffix_offset,
        segment_id=segment_id,
        segment_start_idx=start,
        segment_end_idx=end,
        segment_mid_xy=midpoint,
        segment_projection=projection,
        x=x,
        y=y,
        z=z,
        md=md,
        tvt_input=tvt_input,
    )


def load_target_free_geometries(
    horizontal_files: Sequence[Path],
    identity: pd.DataFrame,
    contract: K16Contract,
) -> dict[str, WellGeometry]:
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in horizontal_files
    }
    geometry: dict[str, WellGeometry] = {}
    for well_id, well_identity in identity.groupby(
        "well_id",
        sort=True,
        observed=True,
    ):
        well = str(well_id)
        geometry[well] = build_target_free_geometry(
            path_by_well[well],
            well_identity,
            contract,
        )
    return geometry


def execution_contract_from_config(config: Mapping[str, Any]) -> K16Contract:
    return K16Contract(
        segments=int(get_nested(config, "method.segment_count")),
        theta0_deg=float(
            get_nested(config, "method.directional_projection.theta0_deg")
        ),
        minimum_abs_projection=float(
            get_nested(
                config,
                "method.directional_projection.minimum_abs_projection",
            )
        ),
        local_linear_k=int(
            get_nested(config, "method.donor_kernel.nearest_segments")
        ),
        bandwidth_ft=float(
            get_nested(config, "method.donor_kernel.bandwidth_ft")
        ),
        ridge=float(get_nested(config, "method.donor_kernel.ridge")),
        formation_k=int(
            get_nested(config, "method.formation_surface.nearest_wells")
        ),
        minimum_finite_formations=int(
            get_nested(
                config,
                "method.primary_aggregation.minimum_finite_formations",
            )
        ),
    )


# %% [markdown]
# ## 5. Fold-local relative-rate construction and target-free freeze

# %%
@dataclass(frozen=True)
class FoldDonorBundle:
    fold: int
    donor_rows: pd.DataFrame
    fields: dict[str, pd.DataFrame]
    formation_plane: FormationPlaneKNN
    reference_manifest: dict[str, Any]


@dataclass(frozen=True)
class TargetFreeBundle:
    input_manifest: pd.DataFrame
    fold_manifest: pd.DataFrame
    role_read_ledger: pd.DataFrame
    donor_fields: pd.DataFrame
    segment_schedule: pd.DataFrame
    primary_paths: pd.DataFrame
    artifact_evidence: dict[str, dict[str, Any]]
    freeze_manifest: dict[str, Any]


def build_fold_donor_bundle(
    fold: int,
    source_wells: Sequence[str],
    path_by_well: Mapping[str, Path],
    geometry_by_well: Mapping[str, WellGeometry],
    contract: K16Contract,
) -> FoldDonorBundle:
    donor_records: list[dict[str, Any]] = []
    reference_wells: list[str] = []
    reference_xy: list[np.ndarray] = []
    reference_formation: list[np.ndarray] = []
    unavailable_reference_wells: list[str] = []
    columns = ["X", "Y", "Z", "MD", "TVT", *FORMATION_COLUMNS]
    for well_id in sorted(map(str, source_wells)):
        path = path_by_well[well_id]
        frame = pd.read_csv(path, usecols=columns)
        geometry = geometry_by_well[well_id]
        if len(frame) != geometry.row_count:
            raise ValueError(f"{well_id} source row count changed")
        source_xy = frame[["X", "Y"]].to_numpy(np.float64)
        if not np.allclose(
            source_xy,
            np.column_stack([geometry.x, geometry.y]),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(f"{well_id} source/target-free XY parity failed")
        formation = frame[list(FORMATION_COLUMNS)].to_numpy(np.float64)
        finite_reference = np.isfinite(source_xy).all(axis=1) & np.isfinite(
            formation
        ).all(axis=1)
        if not finite_reference.any():
            unavailable_reference_wells.append(well_id)
        else:
            reference_wells.append(well_id)
            reference_xy.append(np.median(source_xy[finite_reference], axis=0))
            reference_formation.append(
                np.median(formation[finite_reference], axis=0)
            )
        tvt = frame["TVT"].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        md = frame["MD"].to_numpy(np.float64)
        if not (
            np.isfinite(tvt).all()
            and np.isfinite(z).all()
            and np.isfinite(md).all()
        ):
            raise ValueError(f"{well_id} donor truth geometry is nonfinite")
        structural_position = tvt + z
        direct_rate, direct_steps = median_segment_step_rates(
            md,
            structural_position,
            geometry.suffix_row_idx,
            geometry.segment_id,
            contract.segments,
        )
        relative_rate = np.empty(
            (contract.segments, len(FORMATION_COLUMNS)),
            dtype=np.float64,
        )
        relative_steps = np.empty_like(relative_rate, dtype=np.int32)
        for formation_index in range(len(FORMATION_COLUMNS)):
            rate, count = median_segment_step_rates(
                md,
                structural_position - formation[:, formation_index],
                geometry.suffix_row_idx,
                geometry.segment_id,
                contract.segments,
            )
            relative_rate[:, formation_index] = rate
            relative_steps[:, formation_index] = count
        for segment in range(contract.segments):
            projection = float(geometry.segment_projection[segment])
            eligible = (
                math.isfinite(projection)
                and abs(projection) > contract.minimum_abs_projection
                and math.isfinite(float(direct_rate[segment]))
                and np.isfinite(relative_rate[segment]).all()
            )
            record: dict[str, Any] = {
                "outer_fold": fold,
                "donor_well": well_id,
                "donor_segment": segment,
                "x": float(geometry.segment_mid_xy[segment, 0]),
                "y": float(geometry.segment_mid_xy[segment, 1]),
                "projection": projection,
                "eligible": eligible,
                "direct_dsdmd": float(direct_rate[segment]),
                "direct_valid_steps": int(direct_steps[segment]),
            }
            for formation_index, formation_name in enumerate(FORMATION_COLUMNS):
                record[f"relative_{formation_name}_ds_minus_f_dmd"] = float(
                    relative_rate[segment, formation_index]
                )
                record[f"relative_{formation_name}_valid_steps"] = int(
                    relative_steps[segment, formation_index]
                )
            donor_records.append(record)
    if len(reference_wells) < contract.formation_k:
        raise ValueError("fold formation reference has fewer than k wells")
    plane = FormationPlaneKNN(
        wells=np.asarray(reference_wells, dtype=object),
        xy=np.asarray(reference_xy, dtype=np.float64),
        formation_medians=np.asarray(reference_formation, dtype=np.float64),
        k=contract.formation_k,
    )
    donor_frame = pd.DataFrame(donor_records).sort_values(
        ["donor_well", "donor_segment"],
        kind="mergesort",
    ).reset_index(drop=True)
    reference_frame = pd.DataFrame(
        {
            "well_id": reference_wells,
            "x": np.asarray(reference_xy)[:, 0],
            "y": np.asarray(reference_xy)[:, 1],
            **{
                formation_name: np.asarray(reference_formation)[:, formation_index]
                for formation_index, formation_name in enumerate(FORMATION_COLUMNS)
            },
        }
    )
    reference_manifest = {
        "outer_fold": fold,
        "requested_source_wells": len(set(map(str, source_wells))),
        "reference_wells": len(reference_wells),
        "unavailable_reference_wells": unavailable_reference_wells,
        "reference_well_sha256": json_sha256(reference_wells),
        "reference_content_sha256": frame_content_sha256(reference_frame),
        "donor_field_content_sha256": frame_content_sha256(donor_frame),
    }
    field_columns = [
        "direct_dsdmd",
        *[
            f"relative_{formation_name}_ds_minus_f_dmd"
            for formation_name in FORMATION_COLUMNS
        ],
    ]
    fields = {
        column: donor_field_for(donor_frame, column)
        for column in field_columns
    }
    return FoldDonorBundle(
        fold=fold,
        donor_rows=donor_frame,
        fields=fields,
        formation_plane=plane,
        reference_manifest=reference_manifest,
    )


def donor_field_for(
    donor_rows: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    selected = donor_rows.loc[
        donor_rows["eligible"].astype(bool)
        & np.isfinite(donor_rows[value_column].to_numpy(np.float64)),
        ["x", "y", "donor_well", "donor_segment", "projection", value_column],
    ].copy()
    selected["value"] = (
        selected[value_column].to_numpy(np.float64)
        / selected["projection"].to_numpy(np.float64)
    )
    return selected[
        ["x", "y", "value", "donor_well", "donor_segment"]
    ].reset_index(drop=True)


def reconstruct_relative_rate(
    projected_relative_field: np.ndarray,
    target_projection: np.ndarray,
    target_surface_rate: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(projected_relative_field, dtype=np.float64)
        * np.asarray(target_projection, dtype=np.float64)
        + np.asarray(target_surface_rate, dtype=np.float64)
    )


def robust_median_primary(
    reconstructed_rates: np.ndarray,
    minimum_finite_formations: int,
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(reconstructed_rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != len(FORMATION_COLUMNS):
        raise ValueError("reconstructed rate matrix must have six columns")
    finite_count = np.isfinite(rates).sum(axis=1)
    primary = np.full(len(rates), np.nan, dtype=np.float64)
    eligible = finite_count >= int(minimum_finite_formations)
    primary[eligible] = np.nanmedian(rates[eligible], axis=1)
    return primary, finite_count


def integrate_segment_rates(
    geometry: WellGeometry,
    rate_by_segment: np.ndarray,
) -> np.ndarray:
    rate_by_segment = np.asarray(rate_by_segment, dtype=np.float64)
    if rate_by_segment.shape != (16,):
        raise ValueError("integration requires exactly 16 segment rates")
    row_rate = rate_by_segment[geometry.segment_id]
    md_suffix = geometry.md[geometry.suffix_row_idx]
    delta_md = np.diff(
        np.r_[geometry.md[geometry.anchor_row_idx], md_suffix]
    )
    if not np.isfinite(row_rate).all():
        return np.full(len(row_rate), np.nan, dtype=np.float64)
    structural_position = geometry.anchor_s + np.cumsum(row_rate * delta_md)
    return structural_position - geometry.z[geometry.suffix_row_idx]


def predict_valid_well(
    fold: int,
    geometry: WellGeometry,
    donor_bundle: FoldDonorBundle,
    contract: K16Contract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_prediction = local_linear_predict(
        donor_bundle.fields["direct_dsdmd"],
        geometry.segment_mid_xy,
        contract,
    )
    baseline_rate = (
        direct_prediction.values * geometry.segment_projection
    )

    endpoint_rows = np.concatenate(
        [geometry.segment_start_idx, geometry.segment_end_idx]
    )
    endpoint_xy = np.column_stack(
        [geometry.x[endpoint_rows], geometry.y[endpoint_rows]]
    )
    surface_values, surface_support = donor_bundle.formation_plane.predict(
        endpoint_xy,
        target_well=geometry.well_id,
    )
    surface_start = surface_values[: contract.segments]
    surface_end = surface_values[contract.segments :]
    surface_delta_md = (
        geometry.md[geometry.segment_end_idx]
        - geometry.md[geometry.segment_start_idx]
    )
    if np.any(surface_delta_md <= 0.0):
        raise ValueError(f"{geometry.well_id} has nonpositive segment MD span")
    surface_rate = (surface_end - surface_start) / surface_delta_md[:, None]
    reconstructed = np.full_like(surface_rate, np.nan)
    kernel_support: list[KernelPrediction] = []
    for formation_index, formation_name in enumerate(FORMATION_COLUMNS):
        relative_column = f"relative_{formation_name}_ds_minus_f_dmd"
        prediction = local_linear_predict(
            donor_bundle.fields[relative_column],
            geometry.segment_mid_xy,
            contract,
        )
        kernel_support.append(prediction)
        reconstructed[:, formation_index] = reconstruct_relative_rate(
            prediction.values,
            geometry.segment_projection,
            surface_rate[:, formation_index],
        )

    primary_rate, finite_count = robust_median_primary(
        reconstructed,
        contract.minimum_finite_formations,
    )
    kernel_effective = np.column_stack(
        [
            direct_prediction.effective_donors,
            *[item.effective_donors for item in kernel_support],
        ]
    )
    kernel_fallback = np.column_stack(
        [
            direct_prediction.fallback,
            *[item.fallback for item in kernel_support],
        ]
    )
    endpoint_fallback = surface_support["fallback"].reshape(
        2,
        contract.segments,
    ).T
    endpoint_effective = surface_support["effective_donors"].reshape(
        2,
        contract.segments,
    ).T

    segment_records: list[dict[str, Any]] = []
    for segment in range(contract.segments):
        record: dict[str, Any] = {
            "well_id": geometry.well_id,
            "fold": fold,
            "segment_id": segment,
            "segment_start_row_idx": int(geometry.segment_start_idx[segment]),
            "segment_end_row_idx": int(geometry.segment_end_idx[segment]),
            "segment_mid_x": float(geometry.segment_mid_xy[segment, 0]),
            "segment_mid_y": float(geometry.segment_mid_xy[segment, 1]),
            "segment_projection": float(geometry.segment_projection[segment]),
            "baseline_direct_rate": float(baseline_rate[segment]),
            PRIMARY_CANDIDATE: float(primary_rate[segment]),
            "finite_formation_count": int(finite_count[segment]),
            "effective_donors_min": float(
                np.nanmin(kernel_effective[segment])
            ),
            "surface_effective_donors_min": float(
                np.nanmin(endpoint_effective[segment])
            ),
            "kernel_fallback": bool(kernel_fallback[segment].any()),
            "surface_fallback": bool(endpoint_fallback[segment].any()),
        }
        for formation_index, formation_name in enumerate(FORMATION_COLUMNS):
            record[f"surface_{formation_name}_start"] = float(
                surface_start[segment, formation_index]
            )
            record[f"surface_{formation_name}_end"] = float(
                surface_end[segment, formation_index]
            )
            record[f"surface_{formation_name}_dmd"] = float(
                surface_rate[segment, formation_index]
            )
            record[f"frk16_{formation_name}"] = float(
                reconstructed[segment, formation_index]
            )
        segment_records.append(record)
    segment_frame = pd.DataFrame(segment_records)

    baseline_path = integrate_segment_rates(geometry, baseline_rate)
    primary_path = integrate_segment_rates(geometry, primary_rate)
    row_frame = pd.DataFrame(
        {
            "well_id": geometry.well_id,
            "row_idx": geometry.suffix_row_idx.astype(np.int32),
            "suffix_offset": geometry.suffix_offset.astype(np.int32),
            "suffix_rows": geometry.suffix_rows,
            "fold": fold,
            "segment_id": geometry.segment_id.astype(np.int16),
            "md_since": (
                geometry.md[geometry.suffix_row_idx]
                - geometry.md[geometry.anchor_row_idx]
            ),
            "baseline_direct_path": baseline_path,
            PRIMARY_CANDIDATE: primary_path,
        }
    )
    return segment_frame, row_frame


def build_target_free_bundle(
    config: Mapping[str, Any],
    identity: pd.DataFrame,
    identity_evidence: Mapping[str, Any],
    raw_dir: Path,
    horizontal_files: Sequence[Path],
    geometry_by_well: Mapping[str, WellGeometry],
    contract: K16Contract,
    artifacts_dir: Path,
) -> TargetFreeBundle:
    del raw_dir
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in horizontal_files
    }
    fold_by_well = (
        identity[["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    donor_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.DataFrame] = []
    path_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []
    for fold in expected_folds:
        source_wells = sorted(
            well for well, assigned in fold_by_well.items() if assigned != fold
        )
        valid_wells = sorted(
            well for well, assigned in fold_by_well.items() if assigned == fold
        )
        overlap = sorted(set(source_wells) & set(valid_wells))
        if overlap:
            raise RuntimeError(f"fold {fold} source/valid overlap: {overlap[:5]}")
        donor_bundle = build_fold_donor_bundle(
            fold,
            source_wells,
            path_by_well,
            geometry_by_well,
            contract,
        )
        donor_parts.append(donor_bundle.donor_rows)
        fold_records.append(
            {
                "outer_fold": fold,
                "source_wells": len(source_wells),
                "valid_wells": len(valid_wells),
                "source_valid_overlap": len(overlap),
                "source_well_sha256": json_sha256(source_wells),
                "valid_well_sha256": json_sha256(valid_wells),
                **donor_bundle.reference_manifest,
            }
        )
        role_records.extend(
            [
                {
                    "outer_fold": fold,
                    "role": "outer_train_source",
                    "wells": len(source_wells),
                    "truth_file_reads": len(source_wells),
                    "formation_file_reads": len(source_wells),
                },
                {
                    "outer_fold": fold,
                    "role": "outer_valid_target",
                    "wells": len(valid_wells),
                    "truth_file_reads": 0,
                    "formation_file_reads": 0,
                },
            ]
        )
        for well_id in valid_wells:
            segment_frame, row_frame = predict_valid_well(
                fold,
                geometry_by_well[well_id],
                donor_bundle,
                contract,
            )
            segment_parts.append(segment_frame)
            path_parts.append(row_frame)

    input_manifest = build_raw_input_manifest(horizontal_files)
    fold_manifest = pd.DataFrame(fold_records).sort_values(
        "outer_fold",
        kind="mergesort",
    ).reset_index(drop=True)
    role_ledger = pd.DataFrame(role_records).sort_values(
        ["outer_fold", "role"],
        kind="mergesort",
    ).reset_index(drop=True)
    donor_fields = pd.concat(donor_parts, ignore_index=True).sort_values(
        ["outer_fold", "donor_well", "donor_segment"],
        kind="mergesort",
    ).reset_index(drop=True)
    segment_schedule = pd.concat(segment_parts, ignore_index=True).sort_values(
        ["well_id", "segment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    primary_paths = pd.concat(path_parts, ignore_index=True).sort_values(
        ["well_id", "row_idx"],
        kind="mergesort",
    ).reset_index(drop=True)

    paths = {
        "input_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "fold_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_fold_manifest.csv",
        "role_read_ledger": (
            artifacts_dir / f"{OUTPUT_PREFIX}_role_read_ledger.csv"
        ),
        "donor_fields": (
            artifacts_dir / f"{OUTPUT_PREFIX}_donor_relative_rate_fields.csv.gz"
        ),
        "segment_schedule": (
            artifacts_dir / f"{OUTPUT_PREFIX}_segment_rate_schedule.csv.gz"
        ),
        "primary_paths": (
            artifacts_dir / f"{OUTPUT_PREFIX}_primary_path_schedule.csv.gz"
        ),
    }
    for key, frame in (
        ("input_manifest", input_manifest),
        ("fold_manifest", fold_manifest),
        ("role_read_ledger", role_ledger),
    ):
        write_frame(paths[key], frame)
    artifact_evidence = {
        "input_manifest": {
            "path": str(paths["input_manifest"]),
            "rows": len(input_manifest),
            "file_sha256": sha256_file(paths["input_manifest"]),
            "logical_content_sha256": frame_content_sha256(input_manifest),
            "schema_sha256": frame_schema_sha256(input_manifest),
        },
        "fold_manifest": {
            "path": str(paths["fold_manifest"]),
            "rows": len(fold_manifest),
            "file_sha256": sha256_file(paths["fold_manifest"]),
            "logical_content_sha256": frame_content_sha256(fold_manifest),
            "schema_sha256": frame_schema_sha256(fold_manifest),
        },
        "role_read_ledger": {
            "path": str(paths["role_read_ledger"]),
            "rows": len(role_ledger),
            "file_sha256": sha256_file(paths["role_read_ledger"]),
            "logical_content_sha256": frame_content_sha256(role_ledger),
            "schema_sha256": frame_schema_sha256(role_ledger),
        },
        "donor_fields": write_gzip_frame(paths["donor_fields"], donor_fields),
        "segment_schedule": write_gzip_frame(
            paths["segment_schedule"],
            segment_schedule,
        ),
        "primary_paths": write_gzip_frame(
            paths["primary_paths"],
            primary_paths,
        ),
    }
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "truth_access_before_freeze": 0,
        "validation_formation_reads": int(
            role_ledger.loc[
                role_ledger["role"].eq("outer_valid_target"),
                "formation_file_reads",
            ].sum()
        ),
        "validation_truth_reads": int(
            role_ledger.loc[
                role_ledger["role"].eq("outer_valid_target"),
                "truth_file_reads",
            ].sum()
        ),
        "exp226_identity": dict(identity_evidence),
        "artifacts": artifact_evidence,
        "bundle_logical_sha256": json_sha256(
            {
                key: evidence["logical_content_sha256"]
                for key, evidence in artifact_evidence.items()
            }
        ),
    }
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_json(freeze_path, freeze_manifest)
    freeze_manifest["freeze_manifest_path"] = str(freeze_path)
    freeze_manifest["freeze_manifest_file_sha256"] = sha256_file(freeze_path)
    return TargetFreeBundle(
        input_manifest=input_manifest,
        fold_manifest=fold_manifest,
        role_read_ledger=role_ledger,
        donor_fields=donor_fields,
        segment_schedule=segment_schedule,
        primary_paths=primary_paths,
        artifact_evidence=artifact_evidence,
        freeze_manifest=freeze_manifest,
    )


# %% [markdown]
# ## 6. Stage 0 integrity gate

# %%
def evaluate_stage0_integrity(
    bundle: TargetFreeBundle,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage0_integrity")
    expected_rows = int(gates["expected_rows"])
    expected_wells = int(gates["expected_wells"])
    expected_segments = int(gates["expected_segments"])
    segment = bundle.segment_schedule
    paths = bundle.primary_paths
    role = bundle.role_read_ledger
    fold_manifest = bundle.fold_manifest
    primary_coverage = float(
        np.isfinite(segment[PRIMARY_CANDIDATE].to_numpy(np.float64)).mean()
    )
    surface_fallback_fraction = float(
        segment["surface_fallback"].astype(bool).mean()
    )
    effective_donors_p05 = float(
        segment["effective_donors_min"].quantile(0.05)
    )
    valid_role = role["role"].eq("outer_valid_target")
    validation_truth_reads = int(
        role.loc[valid_role, "truth_file_reads"].sum()
    )
    validation_formation_reads = int(
        role.loc[valid_role, "formation_file_reads"].sum()
    )
    source_valid_overlap = int(fold_manifest["source_valid_overlap"].sum())
    checks = {
        "row_count_exact": len(paths) == expected_rows,
        "well_count_exact": paths["well_id"].nunique() == expected_wells,
        "segment_count_exact": len(segment) == expected_segments,
        "sixteen_segments_per_well": bool(
            segment.groupby("well_id", sort=False)["segment_id"]
            .nunique()
            .eq(16)
            .all()
        ),
        "outer_fold_runs_exact": len(fold_manifest)
        == int(gates["expected_outer_fold_runs"]),
        "fold_inventory_exact": set(paths["fold"].astype(int).unique())
        == set(int(value) for value in get_nested(config, "validation.expected_folds")),
        "source_valid_overlap_zero": source_valid_overlap
        == int(gates["source_valid_overlap"]),
        "validation_truth_reads_zero": validation_truth_reads
        == int(gates["validation_truth_reads"]),
        "validation_formation_reads_zero": validation_formation_reads
        == int(gates["validation_formation_reads"]),
        "truth_access_before_freeze_zero": int(
            bundle.freeze_manifest["truth_access_before_freeze"]
        )
        == 0,
        "surface_fallback_fraction": surface_fallback_fraction
        <= float(gates["target_surface_fallback_fraction_max"]),
        "primary_coverage": primary_coverage
        >= float(gates["primary_coverage_min"]),
        "effective_donors_p05": effective_donors_p05
        >= float(gates["effective_donors_p05_min"]),
        "primary_paths_finite": bool(
            np.isfinite(paths[PRIMARY_CANDIDATE].to_numpy(np.float64)).all()
        ),
        "baseline_paths_finite": bool(
            np.isfinite(paths["baseline_direct_path"].to_numpy(np.float64)).all()
        ),
        "frozen_artifact_inventory": set(bundle.artifact_evidence)
        == {
            "input_manifest",
            "fold_manifest",
            "role_read_ledger",
            "donor_fields",
            "segment_schedule",
            "primary_paths",
        },
    }
    report_only_check_names = {
        str(value)
        for value in gates.get("report_only_checks", [])
    }
    unknown_report_only = report_only_check_names - set(checks)
    if unknown_report_only:
        raise ValueError(
            "unknown Stage 0 report-only checks: "
            f"{sorted(unknown_report_only)}"
        )
    blocking_checks = {
        key: value
        for key, value in checks.items()
        if key not in report_only_check_names
    }
    passed = bool(all(blocking_checks.values()))
    return {
        "stage": "stage0_integrity",
        "passed": passed,
        "checks": checks,
        "blocking_checks": blocking_checks,
        "report_only_checks": {
            key: checks[key] for key in sorted(report_only_check_names)
        },
        "report_only_warning": bool(
            any(not checks[key] for key in report_only_check_names)
        ),
        "rows": len(paths),
        "wells": int(paths["well_id"].nunique()),
        "segments": len(segment),
        "outer_fold_runs": len(fold_manifest),
        "primary_coverage": primary_coverage,
        "surface_fallback_fraction": surface_fallback_fraction,
        "effective_donors_p05": effective_donors_p05,
        "validation_truth_reads": validation_truth_reads,
        "validation_formation_reads": validation_formation_reads,
        "source_valid_overlap": source_valid_overlap,
        "bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "fail_action": (
            None
            if passed
            else "stop_before_truth_join_without_surface_or_kernel_rescue"
        ),
    }


# %% [markdown]
# ## 7. Truth late join and Stage 1 identifiability readout

# %%
@dataclass(frozen=True)
class TruthReadout:
    paths: pd.DataFrame
    segment_actual: pd.DataFrame
    truth_manifest: pd.DataFrame


def verify_frozen_bundle(bundle: TargetFreeBundle) -> None:
    frames = {
        "input_manifest": bundle.input_manifest,
        "fold_manifest": bundle.fold_manifest,
        "role_read_ledger": bundle.role_read_ledger,
        "donor_fields": bundle.donor_fields,
        "segment_schedule": bundle.segment_schedule,
        "primary_paths": bundle.primary_paths,
    }
    logical = {
        key: frame_content_sha256(frame) for key, frame in frames.items()
    }
    expected = {
        key: evidence["logical_content_sha256"]
        for key, evidence in bundle.artifact_evidence.items()
    }
    if logical != expected:
        raise RuntimeError("target-free bundle changed before truth late join")
    if json_sha256(logical) != bundle.freeze_manifest["bundle_logical_sha256"]:
        raise RuntimeError("target-free bundle aggregate SHA mismatch")


def load_truth_after_freeze(
    bundle: TargetFreeBundle,
    geometry_by_well: Mapping[str, WellGeometry],
    path_by_well: Mapping[str, Path],
    contract: K16Contract,
) -> TruthReadout:
    verify_frozen_bundle(bundle)
    paths = bundle.primary_paths.copy()
    truth = np.full(len(paths), np.nan, dtype=np.float64)
    segment_actual_records: list[dict[str, Any]] = []
    truth_manifest_records: list[dict[str, Any]] = []
    segment_fold = (
        bundle.segment_schedule[["well_id", "fold"]]
        .drop_duplicates()
        .set_index("well_id")["fold"]
        .astype(int)
        .to_dict()
    )
    for well_id, positions in paths.groupby(
        "well_id",
        sort=True,
        observed=True,
    ).indices.items():
        well = str(well_id)
        integer_positions = np.asarray(positions, dtype=np.int64)
        geometry = geometry_by_well[well]
        path = path_by_well[well]
        frame = pd.read_csv(path, usecols=["TVT"])
        tvt = frame["TVT"].to_numpy(np.float64)
        if len(tvt) != geometry.row_count or not np.isfinite(tvt).all():
            raise ValueError(f"{well} truth late join is invalid")
        row_idx = paths.loc[integer_positions, "row_idx"].to_numpy(np.int64)
        if not np.array_equal(row_idx, geometry.suffix_row_idx):
            raise ValueError(f"{well} truth late-join row order changed")
        truth[integer_positions] = tvt[row_idx]
        actual_rate, valid_steps = median_segment_step_rates(
            geometry.md,
            tvt + geometry.z,
            geometry.suffix_row_idx,
            geometry.segment_id,
            contract.segments,
        )
        for segment in range(contract.segments):
            segment_actual_records.append(
                {
                    "well_id": well,
                    "fold": int(segment_fold[well]),
                    "segment_id": segment,
                    "actual_dsdmd": float(actual_rate[segment]),
                    "actual_valid_steps": int(valid_steps[segment]),
                }
            )
        digest = hashlib.sha256()
        digest.update(np.asarray(tvt[row_idx], dtype="<f8").tobytes())
        truth_manifest_records.append(
            {
                "well_id": well,
                "path": str(path),
                "file_sha256": sha256_file(path),
                "truth_rows": len(row_idx),
                "suffix_truth_content_sha256": digest.hexdigest(),
            }
        )
    if not np.isfinite(truth).all():
        raise ValueError("truth late join left nonfinite rows")
    paths["tvt_true_readout_only"] = truth
    segment_actual = pd.DataFrame(segment_actual_records).sort_values(
        ["well_id", "segment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    truth_manifest = pd.DataFrame(truth_manifest_records).sort_values(
        "well_id",
        kind="mergesort",
    ).reset_index(drop=True)
    return TruthReadout(
        paths=paths,
        segment_actual=segment_actual,
        truth_manifest=truth_manifest,
    )


def metric_record(
    frame: pd.DataFrame,
    *,
    candidate: str,
    scope: str,
    scope_value: str,
    actual_column: str,
    baseline_column: str,
    candidate_column: str,
) -> dict[str, Any]:
    baseline_rmse = rmse(frame[actual_column], frame[baseline_column])
    candidate_rmse = rmse(frame[actual_column], frame[candidate_column])
    return {
        "candidate": candidate,
        "scope": scope,
        "scope_value": scope_value,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_candidate_minus_baseline": candidate_rmse - baseline_rmse,
        "gain_baseline_minus_candidate": baseline_rmse - candidate_rmse,
        "gain_fraction": (
            (baseline_rmse - candidate_rmse) / baseline_rmse
            if baseline_rmse > 0.0
            else float("nan")
        ),
    }


def build_segment_rate_metrics(
    segment_schedule: pd.DataFrame,
    segment_actual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = segment_schedule.merge(
        segment_actual,
        on=["well_id", "fold", "segment_id"],
        how="left",
        validate="one_to_one",
    )
    if not np.isfinite(merged["actual_dsdmd"].to_numpy(np.float64)).all():
        raise ValueError("actual segment rate is missing after late join")
    records: list[dict[str, Any]] = []
    candidates = [*FORMATION_CANDIDATES, PRIMARY_CANDIDATE]
    for candidate in candidates:
        records.append(
            metric_record(
                merged,
                candidate=candidate,
                scope="pooled",
                scope_value="all",
                actual_column="actual_dsdmd",
                baseline_column="baseline_direct_rate",
                candidate_column=candidate,
            )
        )
        for fold, fold_frame in merged.groupby("fold", sort=True):
            records.append(
                metric_record(
                    fold_frame,
                    candidate=candidate,
                    scope="fold",
                    scope_value=str(int(fold)),
                    actual_column="actual_dsdmd",
                    baseline_column="baseline_direct_rate",
                    candidate_column=candidate,
                )
            )
    return (
        pd.DataFrame(records),
        merged,
    )


def path_scope_mask(
    frame: pd.DataFrame,
    definition: Mapping[str, Any],
) -> np.ndarray:
    kind = str(definition["kind"])
    if kind == "suffix_offset_less_than":
        return frame["suffix_offset"].to_numpy(np.int64) < int(
            definition["value"]
        )
    if kind == "well_suffix_rows_at_least":
        return frame["suffix_rows"].to_numpy(np.int64) >= int(
            definition["value"]
        )
    if kind == "pooled_contract_alias":
        return np.ones(len(frame), dtype=bool)
    raise ValueError(f"unknown path scope kind: {kind}")


def _sse_record(
    actual: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    finite = (
        np.isfinite(actual) & np.isfinite(baseline) & np.isfinite(candidate)
    )
    return {
        "rows": int(finite.sum()),
        "baseline_sse": float(np.square(actual[finite] - baseline[finite]).sum()),
        "candidate_sse": float(
            np.square(actual[finite] - candidate[finite]).sum()
        ),
    }


def _merge_sse(
    accumulator: dict[tuple[str, str, str], dict[str, float | int]],
    key: tuple[str, str, str],
    record: Mapping[str, float | int],
) -> None:
    current = accumulator.setdefault(
        key,
        {"rows": 0, "baseline_sse": 0.0, "candidate_sse": 0.0},
    )
    current["rows"] = int(current["rows"]) + int(record["rows"])
    current["baseline_sse"] = float(current["baseline_sse"]) + float(
        record["baseline_sse"]
    )
    current["candidate_sse"] = float(current["candidate_sse"]) + float(
        record["candidate_sse"]
    )


def build_path_metrics(
    truth_readout: TruthReadout,
    segment_schedule: pd.DataFrame,
    geometry_by_well: Mapping[str, WellGeometry],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    paths = truth_readout.paths
    scopes = get_nested(config, "validation.scopes")
    accumulator: dict[
        tuple[str, str, str],
        dict[str, float | int],
    ] = {}
    segment_lookup = segment_schedule.set_index(["well_id", "segment_id"])
    for well_id, positions in paths.groupby(
        "well_id",
        sort=True,
        observed=True,
    ).indices.items():
        well = str(well_id)
        integer_positions = np.asarray(positions, dtype=np.int64)
        well_paths = paths.loc[integer_positions].sort_values(
            "row_idx",
            kind="mergesort",
        )
        geometry = geometry_by_well[well]
        actual = well_paths["tvt_true_readout_only"].to_numpy(np.float64)
        baseline = well_paths["baseline_direct_path"].to_numpy(np.float64)
        fold = int(well_paths["fold"].iloc[0])
        for candidate in (*FORMATION_CANDIDATES, PRIMARY_CANDIDATE):
            if candidate == PRIMARY_CANDIDATE:
                prediction = well_paths[PRIMARY_CANDIDATE].to_numpy(np.float64)
            else:
                rates = np.asarray(
                    [
                        float(segment_lookup.loc[(well, segment), candidate])
                        for segment in range(16)
                    ],
                    dtype=np.float64,
                )
                prediction = integrate_segment_rates(geometry, rates)
            _merge_sse(
                accumulator,
                (candidate, "pooled", "all"),
                _sse_record(actual, baseline, prediction),
            )
            _merge_sse(
                accumulator,
                (candidate, "fold", str(fold)),
                _sse_record(actual, baseline, prediction),
            )
            if candidate == PRIMARY_CANDIDATE:
                for scope_name, definition in scopes.items():
                    mask = path_scope_mask(well_paths, definition)
                    _merge_sse(
                        accumulator,
                        (candidate, "scope", str(scope_name)),
                        _sse_record(
                            actual[mask],
                            baseline[mask],
                            prediction[mask],
                        ),
                    )
    records: list[dict[str, Any]] = []
    well_counts = paths.groupby("fold")["well_id"].nunique().astype(int).to_dict()
    for (candidate, scope, scope_value), value in sorted(accumulator.items()):
        rows = int(value["rows"])
        baseline_rmse = math.sqrt(float(value["baseline_sse"]) / rows)
        candidate_rmse = math.sqrt(float(value["candidate_sse"]) / rows)
        records.append(
            {
                "candidate": candidate,
                "scope": scope,
                "scope_value": scope_value,
                "rows": rows,
                "wells": (
                    int(well_counts[int(scope_value)])
                    if scope == "fold"
                    else int(paths["well_id"].nunique())
                ),
                "baseline_rmse": baseline_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_candidate_minus_baseline": (
                    candidate_rmse - baseline_rmse
                ),
                "gain_baseline_minus_candidate": (
                    baseline_rmse - candidate_rmse
                ),
                "gain_fraction": (
                    (baseline_rmse - candidate_rmse) / baseline_rmse
                    if baseline_rmse > 0.0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(records)


def build_by_well_metrics(paths: pd.DataFrame) -> pd.DataFrame:
    records = []
    for well_id, frame in paths.groupby("well_id", sort=True, observed=True):
        baseline_rmse = rmse(
            frame["tvt_true_readout_only"],
            frame["baseline_direct_path"],
        )
        candidate_rmse = rmse(
            frame["tvt_true_readout_only"],
            frame[PRIMARY_CANDIDATE],
        )
        records.append(
            {
                "well_id": str(well_id),
                "fold": int(frame["fold"].iloc[0]),
                "rows": len(frame),
                "baseline_rmse": baseline_rmse,
                "candidate_rmse": candidate_rmse,
                "delta_candidate_minus_baseline": (
                    candidate_rmse - baseline_rmse
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        "well_id",
        kind="mergesort",
    ).reset_index(drop=True)


def one_metric(
    metrics: pd.DataFrame,
    candidate: str,
    scope: str,
    scope_value: str,
) -> pd.Series:
    selected = metrics.loc[
        metrics["candidate"].eq(candidate)
        & metrics["scope"].eq(scope)
        & metrics["scope_value"].astype(str).eq(str(scope_value))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric for {candidate}/{scope}/{scope_value}"
        )
    return selected.iloc[0]


def evaluate_stage1_identifiability(
    segment_metrics: pd.DataFrame,
    path_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage1_identifiability")
    rate_pooled = one_metric(
        segment_metrics,
        PRIMARY_CANDIDATE,
        "pooled",
        "all",
    )
    path_pooled = one_metric(
        path_metrics,
        PRIMARY_CANDIDATE,
        "pooled",
        "all",
    )
    rate_fold_gain = [
        float(
            one_metric(
                segment_metrics,
                PRIMARY_CANDIDATE,
                "fold",
                str(fold),
            )["gain_baseline_minus_candidate"]
        )
        for fold in get_nested(config, "validation.expected_folds")
    ]
    path_fold_gain = [
        float(
            one_metric(
                path_metrics,
                PRIMARY_CANDIDATE,
                "fold",
                str(fold),
            )["gain_baseline_minus_candidate"]
        )
        for fold in get_nested(config, "validation.expected_folds")
    ]
    scope_deltas = {
        str(scope_name): float(
            one_metric(
                path_metrics,
                PRIMARY_CANDIDATE,
                "scope",
                str(scope_name),
            )["delta_candidate_minus_baseline"]
        )
        for scope_name in get_nested(config, "validation.scopes")
    }
    p95_well_delta = float(
        by_well["delta_candidate_minus_baseline"].quantile(0.95)
    )
    worst_row = by_well.loc[
        by_well["delta_candidate_minus_baseline"].idxmax()
    ]
    worst_delta = float(worst_row["delta_candidate_minus_baseline"])
    checks = {
        "rate_relative_gain": float(rate_pooled["gain_fraction"])
        >= float(gates["donor_rate_rmse_relative_gain_min"]),
        "cumulative_path_gain": float(
            path_pooled["gain_baseline_minus_candidate"]
        )
        >= float(gates["cumulative_path_rmse_gain_ft_min"]),
        "positive_rate_folds": int(np.sum(np.asarray(rate_fold_gain) > 0.0))
        >= int(gates["positive_rate_fold_count_min"]),
        "positive_path_folds": int(np.sum(np.asarray(path_fold_gain) > 0.0))
        >= int(gates["positive_path_fold_count_min"]),
        "scope_tail_guard": all(
            delta <= float(gates["scope_regression_tolerance_ft"])
            for delta in scope_deltas.values()
        ),
        "p95_well_guard": p95_well_delta
        <= float(gates["p95_well_regression_max_ft"]),
        "worst_well_guard": worst_delta
        <= float(gates["worst_well_regression_max_ft"]),
    }
    return {
        "stage": "stage1_identifiability",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rate": {
            "baseline_rmse": float(rate_pooled["baseline_rmse"]),
            "candidate_rmse": float(rate_pooled["candidate_rmse"]),
            "relative_gain": float(rate_pooled["gain_fraction"]),
            "fold_gains": rate_fold_gain,
            "positive_folds": int(np.sum(np.asarray(rate_fold_gain) > 0.0)),
        },
        "path": {
            "baseline_rmse": float(path_pooled["baseline_rmse"]),
            "candidate_rmse": float(path_pooled["candidate_rmse"]),
            "gain_ft": float(path_pooled["gain_baseline_minus_candidate"]),
            "fold_gains": path_fold_gain,
            "positive_folds": int(np.sum(np.asarray(path_fold_gain) > 0.0)),
            "scope_deltas": scope_deltas,
        },
        "by_well": {
            "improved": int(
                (by_well["delta_candidate_minus_baseline"] < 0.0).sum()
            ),
            "worsened": int(
                (by_well["delta_candidate_minus_baseline"] > 0.0).sum()
            ),
            "median_delta": float(
                by_well["delta_candidate_minus_baseline"].median()
            ),
            "p95_delta": p95_well_delta,
            "worst_well_id": str(worst_row["well_id"]),
            "worst_delta": worst_delta,
        },
        "fail_action": (
            None
            if all(checks.values())
            else "close_branch_and_block_exp378_exp379_exp380_without_rescue_grid"
        ),
    }


# %% [markdown]
# ## 8. Metrics, generated artifacts, and fail-closed decision

# %%
def artifact_sha_record(path: Path) -> dict[str, Any]:
    return {
        "artifact": path.name,
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "decompressed_content_sha256": (
            sha256_decompressed_gzip(path) if path.suffix == ".gz" else None
        ),
    }


def write_sha_manifest(
    artifacts_dir: Path,
    artifact_paths: Iterable[Path],
) -> Path:
    unique = {
        str(path.resolve()): path
        for path in artifact_paths
        if path.exists()
    }
    records = [
        artifact_sha_record(unique[key]) for key in sorted(unique)
    ]
    manifest = pd.DataFrame(records)
    path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    write_frame(path, manifest)
    return path


def write_stage0_failure(
    bundle: TargetFreeBundle,
    stage0: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    stage0_path = (
        artifacts_dir / f"{OUTPUT_PREFIX}_stage0_integrity_guard.json"
    )
    write_json(stage0_path, stage0)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_integrity_failed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "stage0": dict(stage0),
        "stage1": None,
        "truth_loaded": False,
        "execution": {
            **EXPECTED_EXECUTION,
            "parent_control_regeneration": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        "decision": (
            "close_before_truth_join_and_block_exp378_exp379_exp380_"
            "without_rescue_grid"
        ),
    }
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    frozen_paths = [
        Path(evidence["path"])
        for evidence in bundle.artifact_evidence.values()
    ]
    freeze_path = Path(bundle.freeze_manifest["freeze_manifest_path"])
    sha_manifest_path = write_sha_manifest(
        artifacts_dir,
        [*frozen_paths, freeze_path, stage0_path, summary_path],
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_integrity_failed",
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "stage0_passed": False,
        "stage1_passed": False,
        "truth_loaded": False,
        "decision": summary["decision"],
        "bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "sha_manifest_file_sha256": sha256_file(sha_manifest_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def persist_stage1_outputs(
    bundle: TargetFreeBundle,
    truth_readout: TruthReadout,
    segment_readout: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    path_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    stage0: Mapping[str, Any],
    stage1: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    paths = {
        "stage0": artifacts_dir / f"{OUTPUT_PREFIX}_stage0_integrity_guard.json",
        "truth_manifest": (
            artifacts_dir / f"{OUTPUT_PREFIX}_truth_late_join_manifest.csv"
        ),
        "segment_actual": (
            artifacts_dir / f"{OUTPUT_PREFIX}_segment_actual_rate_readout.csv.gz"
        ),
        "segment_metrics": (
            artifacts_dir / f"{OUTPUT_PREFIX}_segment_rate_metrics.csv"
        ),
        "path_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_path_metrics.csv",
        "by_well": artifacts_dir / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "stage1": (
            artifacts_dir / f"{OUTPUT_PREFIX}_stage1_identifiability_guard.json"
        ),
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    write_json(paths["stage0"], stage0)
    write_frame(paths["truth_manifest"], truth_readout.truth_manifest)
    segment_actual_evidence = write_gzip_frame(
        paths["segment_actual"],
        segment_readout,
    )
    write_frame(paths["segment_metrics"], segment_metrics)
    write_frame(paths["path_metrics"], path_metrics)
    write_frame(paths["by_well"], by_well)
    write_json(paths["stage1"], stage1)

    if bool(stage1["passed"]):
        decision = (
            "exp377_passed_request_separate_exp378_implementation_or_"
            "execution_decision"
        )
    else:
        decision = (
            "close_and_block_exp378_exp379_exp380_without_surface_kernel_"
            "or_scope_rescue"
        )
    primary_rate = one_metric(
        segment_metrics,
        PRIMARY_CANDIDATE,
        "pooled",
        "all",
    )
    primary_path = one_metric(
        path_metrics,
        PRIMARY_CANDIDATE,
        "pooled",
        "all",
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_and_stage1_completed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "rows": len(bundle.primary_paths),
        "wells": int(bundle.primary_paths["well_id"].nunique()),
        "segments": len(bundle.segment_schedule),
        "stage0": dict(stage0),
        "stage1": dict(stage1),
        "truth_loaded": True,
        "primary": PRIMARY_CANDIDATE,
        "primary_rate_rmse": float(primary_rate["candidate_rmse"]),
        "primary_path_rmse": float(primary_path["candidate_rmse"]),
        "bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "truth_manifest_logical_sha256": frame_content_sha256(
            truth_readout.truth_manifest
        ),
        "segment_actual_logical_sha256": segment_actual_evidence[
            "logical_content_sha256"
        ],
        "execution": {
            **EXPECTED_EXECUTION,
            "parent_control_regeneration": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        "decision": decision,
    }
    write_json(paths["summary"], summary)
    frozen_paths = [
        Path(evidence["path"])
        for evidence in bundle.artifact_evidence.values()
    ]
    freeze_path = Path(bundle.freeze_manifest["freeze_manifest_path"])
    sha_manifest_path = write_sha_manifest(
        artifacts_dir,
        [
            *frozen_paths,
            freeze_path,
            *paths.values(),
        ],
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_and_stage1_completed",
        "route": "pf_beam",
        "cv": float(primary_path["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "stage0_passed": bool(stage0["passed"]),
        "stage1_passed": bool(stage1["passed"]),
        "truth_loaded": True,
        "decision": decision,
        "primary": {
            "candidate": PRIMARY_CANDIDATE,
            "rate_baseline_rmse": float(primary_rate["baseline_rmse"]),
            "rate_candidate_rmse": float(primary_rate["candidate_rmse"]),
            "rate_relative_gain": float(primary_rate["gain_fraction"]),
            "path_baseline_rmse": float(primary_path["baseline_rmse"]),
            "path_candidate_rmse": float(primary_path["candidate_rmse"]),
            "path_gain_ft": float(
                primary_path["gain_baseline_minus_candidate"]
            ),
        },
        "bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "truth_manifest_logical_sha256": frame_content_sha256(
            truth_readout.truth_manifest
        ),
        "segment_actual_logical_sha256": segment_actual_evidence[
            "logical_content_sha256"
        ],
        "sha_manifest_file_sha256": sha256_file(sha_manifest_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_kaggle_authorization=True)
    artifacts_dir = runtime_artifacts_dir()
    contract = execution_contract_from_config(config)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Primary:", PRIMARY_CANDIDATE)
    print(
        "Execution: 1 diagnostic / 6 reporting surfaces / 5 outer folds / "
        "0 model / 0 HMM / 0 PF / 0 booster"
    )
    print("Parent/control regeneration: 0")

    identity, identity_evidence = load_exp226_fold_identity(config)
    expected_wells = set(identity["well_id"].astype(str))
    raw_dir, horizontal_files = resolve_raw_train_dir(config, expected_wells)
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in horizontal_files
    }
    geometry_by_well = load_target_free_geometries(
        horizontal_files,
        identity,
        contract,
    )
    print("Target-free geometry loaded:", len(geometry_by_well), "wells")
    print("Target truth/formation columns loaded: 0 / 0")

    bundle = build_target_free_bundle(
        config,
        identity,
        identity_evidence,
        raw_dir,
        horizontal_files,
        geometry_by_well,
        contract,
        artifacts_dir,
    )
    print(
        "Target-free bundle frozen:",
        bundle.freeze_manifest["bundle_logical_sha256"],
    )
    stage0 = evaluate_stage0_integrity(bundle, config)
    stage0_path = (
        artifacts_dir / f"{OUTPUT_PREFIX}_stage0_integrity_guard.json"
    )
    write_json(stage0_path, stage0)
    print("Stage 0 integrity PASS:", stage0["passed"])
    print(json.dumps(to_jsonable(stage0), indent=2, ensure_ascii=False))
    if not stage0["passed"]:
        summary = write_stage0_failure(bundle, stage0, artifacts_dir)
        print("Truth join skipped. Decision:", summary["decision"])
        return summary

    truth_readout = load_truth_after_freeze(
        bundle,
        geometry_by_well,
        path_by_well,
        contract,
    )
    segment_metrics, segment_readout = build_segment_rate_metrics(
        bundle.segment_schedule,
        truth_readout.segment_actual,
    )
    path_metrics = build_path_metrics(
        truth_readout,
        bundle.segment_schedule,
        geometry_by_well,
        config,
    )
    by_well = build_by_well_metrics(truth_readout.paths)
    stage1 = evaluate_stage1_identifiability(
        segment_metrics,
        path_metrics,
        by_well,
        config,
    )
    print("Stage 1 identifiability PASS:", stage1["passed"])
    print(json.dumps(to_jsonable(stage1), indent=2, ensure_ascii=False))
    summary = persist_stage1_outputs(
        bundle,
        truth_readout,
        segment_readout,
        segment_metrics,
        path_metrics,
        by_well,
        stage0,
        stage1,
        artifacts_dir,
    )
    print("Decision:", summary["decision"])
    print("Generated artifacts:", artifacts_dir)
    return summary


# %% [markdown]
# ## 9. Setup, configuration preview, and execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    validate_execution_contract(CONFIG, require_kaggle_authorization=True)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment"),
                "lineage": get_nested(CONFIG, "lineage"),
                "validation": get_nested(CONFIG, "validation"),
                "method": get_nested(CONFIG, "method"),
                "gates": get_nested(CONFIG, "gates"),
                "runtime": get_nested(CONFIG, "runtime"),
                "execution": get_nested(CONFIG, "execution"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    SUMMARY = run_audit(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY), indent=2, ensure_ascii=False))
