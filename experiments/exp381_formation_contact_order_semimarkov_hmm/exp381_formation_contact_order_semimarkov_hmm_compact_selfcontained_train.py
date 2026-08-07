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
# # exp381 formation contact order semi-Markov HMM: Stage 0
#
# This notebook implements only the preregistered zero-HMM contact
# predictability audit. It freezes fold-safe formation surfaces, geometric
# crossings, prefix-calibrated contact TVT predictions, and a 16-well resource
# readout before validation truth or validation formation columns are parsed.
# The seven-state semi-Markov HMM remains unimplemented pending a Stage 0 PASS
# and separate user approval.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment contract
# 2. Runtime, configuration, SHA, and artifact helpers
# 3. Execution and role-read guards
# 4. Formation surface, crossing, and prefix-calibration helpers
# 5. Fold identity and guarded raw-data loaders
# 6. Fold-local source fitting and target-free contact generation
# 7. Target-free freeze and 16-well resource readout
# 8. Validation-truth late join and contact metrics
# 9. Fixed Stage 0 AND gate and generated artifacts
# 10. Setup, configuration preview, and execution

# %% [markdown]
# ## 1. Imports and immutable experiment contract

# %%
from __future__ import annotations

import gzip
import hashlib
import io
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

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

EXPERIMENT_NAME = "exp381_formation_contact_order_semimarkov_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
FORMATION_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
FORMATION_INDEX = {name: index for index, name in enumerate(FORMATION_NAMES)}
TARGET_SAFE_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TARGET_TRUTH_COLUMNS = ("MD", "Z", "TVT", *FORMATION_NAMES)
SOURCE_COLUMNS = ("MD", "X", "Y", "Z", "TVT", *FORMATION_NAMES)
SAFE_OOF_COLUMNS = ("well_id", "row_idx", "suffix_offset", "fold")
FORBIDDEN_PRE_FREEZE_COLUMNS = (
    "tvt_true",
    "TVT",
    "error",
    "abs_error",
    "oracle_rank",
)
PRIMARY_METHOD = "formation_plane_knn"
CONTROL_METHOD = "constant_surface"
EXECUTE_NOTEBOOK = os.environ.get("EXP381_IMPORT_ONLY", "0") != "1"
EXPECTED_RUNTIME_COUNTS = {
    "scientific_diagnostics": 1,
    "reporting_surfaces": 6,
    "reporting_folds": 5,
    "fitted_models": 0,
    "model_configs": 0,
    "trained_folds": 0,
    "lightgbm_boosters": 0,
    "hmm_runs": 0,
    "pf_runs": 0,
    "beam_runs": 0,
}

# %% [markdown]
# ## 2. Runtime, configuration, SHA, and artifact helpers

# %%
def project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def find_config_path() -> Path:
    root = project_root()
    candidates = [
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        Path("/kaggle/working/config.yaml"),
        root / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found in {candidates}")


def runtime_experiment_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return project_root() / "experiments" / EXPERIMENT_NAME


def runtime_artifacts_dir() -> Path:
    path = runtime_experiment_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_metrics_path() -> Path:
    return runtime_experiment_dir() / "metrics.json"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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


def array_content_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def frame_content_sha256(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(
        buffer,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_jsonable(value),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    path.write_text(payload + "\n")


def write_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "file_sha256": sha256_file(path),
        "logical_content_sha256": frame_content_sha256(frame),
        "schema_sha256": frame_schema_sha256(frame),
    }


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
    evidence: dict[str, str] = {}
    for path in candidates:
        actual = (
            sha256_decompressed_gzip(path) if decompressed else sha256_file(path)
        )
        evidence[str(path)] = actual
        if actual == expected_sha256:
            return path
    raise ValueError(f"no candidate matched expected SHA: {evidence}")


def max_rss_gb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. Kaggle and the repository runtime
    # are Linux, while the branch keeps the helper portable for unit tests.
    bytes_value = raw if raw > 10_000_000 else raw * 1024.0
    return bytes_value / (1024.0**3)


# %% [markdown]
# ## 3. Execution and role-read guards

# %%
def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_authorization: bool,
) -> None:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp381 route must remain pf_beam")
    if not bool(get_nested(config, "execution.implementation_authorized")):
        raise RuntimeError("exp381 implementation is not authorized")
    for key, expected in EXPECTED_RUNTIME_COUNTS.items():
        actual = int(get_nested(config, f"runtime.{key}", -1))
        if actual != expected:
            raise ValueError(
                f"runtime.{key} changed: expected {expected}, got {actual}"
            )
    fixed_values = {
        "contact_model.surface.primary.nearest_wells": 10,
        "contact_model.surface.primary.minimum_finite_donors_per_formation": 10,
        "contact_model.eligibility.minimum_triple_matched_formations": 2,
        "gates.stage0_contact_predictability.resource_audit_wells": 16,
        "semimarkov_hmm.duration_potential_scale": 0.10,
        "semimarkov_hmm.duration_squared_error_clip": 25,
    }
    for path, expected in fixed_values.items():
        actual = float(get_nested(config, path, float("nan")))
        if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path} changed: expected {expected}, got {actual}")
    if tuple(get_nested(config, "contact_model.formation_order", [])) != FORMATION_NAMES:
        raise ValueError("formation order changed")
    if get_nested(config, "contact_model.crossing.selection") != (
        "first_crossing_in_increasing_md"
    ):
        raise ValueError("crossing selection changed")
    if get_nested(
        config,
        "contact_model.surface.primary.missing_formation_policy",
    ) != "formation_specific_finite_outer_train_donors":
        raise ValueError("missing formation donor policy changed")
    if bool(get_nested(config, "runtime.parent_control_regeneration")):
        raise ValueError("parent/control regeneration is forbidden")
    if int(get_nested(config, "runtime.hmm_runs")) != 0:
        raise ValueError("Stage 0 must execute zero HMM runs")
    if bool(get_nested(config, "execution.stage1_implementation_authorized")):
        raise ValueError("Stage 1 implementation is outside the current scope")
    if bool(get_nested(config, "execution.inference_enabled")):
        raise ValueError("exp381 inference must remain disabled")
    if bool(get_nested(config, "execution.submission_enabled")):
        raise ValueError("exp381 submission must remain disabled")
    if require_run_authorization:
        authorization = (
            bool(get_nested(config, "execution.kaggle_execution_authorized"))
            and bool(get_nested(config, "execution.stage0_run_authorized"))
        )
        if not authorization:
            raise RuntimeError(
                "Stage 0 Kaggle execution is not authorized. Static implementation "
                "is complete; canonical notebook adoption/package/push/run require "
                "separate user direction."
            )
        required_true = (
            "execution.canonical_train_notebook_adopted",
            "execution.kaggle_package_authorized",
            "execution.kaggle_push_authorized",
            "runtime.kaggle.train_run_on_push",
        )
        for path in required_true:
            if not bool(get_nested(config, path)):
                raise RuntimeError(f"{path} must be true for the authorized Stage 0 run")
        if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
            raise ValueError("Stage 0 must remain CPU-only")
        if bool(get_nested(config, "runtime.kaggle.enable_internet")):
            raise ValueError("Stage 0 must remain offline")


@dataclass
class RoleReadLedger:
    records: list[dict[str, Any]] = field(default_factory=list)
    target_free_frozen: bool = False

    def _record(
        self,
        *,
        fold: int,
        well_id: str,
        role: str,
        phase: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        self.records.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "role": role,
                "phase": phase,
                "columns": "|".join(str(column) for column in columns),
                "rows": int(rows),
                "target_free_frozen": bool(self.target_free_frozen),
            }
        )

    def record_source(
        self,
        fold: int,
        well_id: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        if self.target_free_frozen:
            raise RuntimeError("source fitting cannot occur after target-free freeze")
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_train",
            phase="source_fit",
            columns=columns,
            rows=rows,
        )

    def record_target_safe(
        self,
        fold: int,
        well_id: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        forbidden = {"TVT", *FORMATION_NAMES}.intersection(columns)
        if forbidden:
            raise ValueError(f"target-safe read contains forbidden columns: {forbidden}")
        if self.target_free_frozen:
            raise RuntimeError("target-safe reads must finish before freeze")
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_valid",
            phase="target_safe_generation",
            columns=columns,
            rows=rows,
        )

    def freeze(self) -> None:
        if self.target_free_frozen:
            raise RuntimeError("target-free bundle already frozen")
        self.target_free_frozen = True

    def record_target_truth(
        self,
        fold: int,
        well_id: str,
        columns: Sequence[str],
        rows: int,
    ) -> None:
        if not self.target_free_frozen:
            raise RuntimeError("validation truth cannot be read before freeze")
        expected = {"TVT", *FORMATION_NAMES}
        if not expected.issubset(columns):
            raise ValueError("late truth read must declare TVT and six formations")
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_valid",
            phase="truth_late_join",
            columns=columns,
            rows=rows,
        )

    def pre_freeze_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.records)
        if frame.empty:
            return frame
        return frame.loc[
            frame["phase"].ne("truth_late_join")
        ].sort_values(
            ["fold", "role", "well_id", "phase"],
            kind="mergesort",
        ).reset_index(drop=True)

    def late_truth_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.records)
        if frame.empty:
            return frame
        return frame.loc[
            frame["phase"].eq("truth_late_join")
        ].sort_values(
            ["fold", "well_id"],
            kind="mergesort",
        ).reset_index(drop=True)


# %% [markdown]
# ## 4. Formation surface, crossing, and prefix-calibration helpers

# %%
@dataclass
class FormationPlaneKNN:
    wells: np.ndarray
    xy: np.ndarray
    formation_medians: np.ndarray
    k: int = 10
    chunk_rows: int = 50_000

    def __post_init__(self) -> None:
        self.wells = np.asarray(self.wells, dtype=object)
        self.xy = np.asarray(self.xy, dtype=np.float64)
        self.formation_medians = np.asarray(
            self.formation_medians,
            dtype=np.float64,
        )
        if len(self.wells) < self.k:
            raise ValueError("formation plane requires at least k reference wells")
        if self.xy.shape != (len(self.wells), 2):
            raise ValueError("formation reference XY shape mismatch")
        if self.formation_medians.shape != (
            len(self.wells),
            len(FORMATION_NAMES),
        ):
            raise ValueError("formation reference value shape mismatch")
        if not np.isfinite(self.xy).all():
            raise ValueError("formation reference XY contains nonfinite values")
        finite_reference = np.isfinite(self.formation_medians)
        reference_counts = finite_reference.sum(axis=0)
        if np.any(reference_counts < self.k):
            raise ValueError(
                "each formation plane requires at least k finite reference wells: "
                f"{reference_counts.tolist()}"
            )
        scale = np.std(self.xy, axis=0)
        self.scale = np.where(np.isfinite(scale) & (scale >= 1e-3), scale, 1.0)
        self.formation_reference_indices = [
            np.flatnonzero(finite_reference[:, formation_index])
            for formation_index in range(len(FORMATION_NAMES))
        ]
        self.formation_trees = [
            cKDTree(self.xy[indices] / self.scale)
            for indices in self.formation_reference_indices
        ]

    def predict(
        self,
        query_xy: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        query = np.atleast_2d(np.asarray(query_xy, dtype=np.float64))
        if query.shape[1] != 2 or not np.isfinite(query).all():
            raise ValueError("formation query must be finite XY")
        output = np.empty((len(query), len(FORMATION_NAMES)), dtype=np.float64)
        fallback_by_formation = np.zeros(
            (len(query), len(FORMATION_NAMES)),
            dtype=bool,
        )
        effective_by_formation = np.empty(
            (len(query), len(FORMATION_NAMES)),
            dtype=np.float64,
        )
        nearest_by_formation = np.empty(
            (len(query), len(FORMATION_NAMES)),
            dtype=np.float64,
        )
        chunk_rows = max(1, int(self.chunk_rows))
        for start in range(0, len(query), chunk_rows):
            end = min(start + chunk_rows, len(query))
            point = query[start:end]
            for formation_index in range(len(FORMATION_NAMES)):
                reference_indices = self.formation_reference_indices[
                    formation_index
                ]
                distance, local_neighbor = self.formation_trees[
                    formation_index
                ].query(
                    point / self.scale,
                    k=self.k,
                    workers=1,
                )
                distance = np.asarray(distance, dtype=np.float64)
                local_neighbor = np.asarray(local_neighbor, dtype=np.int64)
                if self.k == 1:
                    distance = distance[:, None]
                    local_neighbor = local_neighbor[:, None]
                neighbor = reference_indices[local_neighbor]
                response = self.formation_medians[neighbor, formation_index]
                weight = 1.0 / (distance + 1e-3)
                delta = (
                    self.xy[neighbor] - point[:, None, :]
                ) / self.scale[None, None, :]
                design = np.concatenate(
                    [
                        delta,
                        np.ones((len(point), self.k, 1), dtype=np.float64),
                    ],
                    axis=2,
                )
                normal = np.einsum("nki,nk,nkj->nij", design, weight, design)
                normal[:, 0, 0] += 1e-9
                normal[:, 1, 1] += 1e-9
                normal[:, 2, 2] += 1e-12
                rhs = np.einsum("nki,nk,nk->ni", design, weight, response)
                try:
                    coefficient = np.linalg.solve(normal, rhs[:, :, None])
                    prediction = coefficient[:, 2, 0]
                except np.linalg.LinAlgError:
                    prediction = np.full(
                        len(point),
                        np.nan,
                        dtype=np.float64,
                    )
                bad = ~np.isfinite(prediction)
                if bad.any():
                    prediction[bad] = np.average(
                        response[bad],
                        axis=1,
                        weights=weight[bad],
                    )
                    fallback_by_formation[start:end, formation_index] = bad
                output[start:end, formation_index] = prediction
                sum_weight = weight.sum(axis=1)
                effective_by_formation[start:end, formation_index] = (
                    np.square(sum_weight) / np.square(weight).sum(axis=1)
                )
                nearest_by_formation[start:end, formation_index] = distance[:, 0]
        return output, {
            "fallback": fallback_by_formation.any(axis=1),
            "effective_donors": effective_by_formation.min(axis=1),
            "nearest_distance_scaled": nearest_by_formation.max(axis=1),
            "formation_reference_counts": np.asarray(
                [len(indices) for indices in self.formation_reference_indices],
                dtype=np.int64,
            ),
        }


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
        raise ValueError("MD and crossing residual length mismatch")
    if len(md_array) == 0:
        return None
    finite_md = np.isfinite(md_array)
    if not finite_md.all() or np.any(np.diff(md_array) <= 0.0):
        raise ValueError("crossing MD must be finite and strictly increasing")
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
            "md": interpolate_at_fraction(md_array, left, fraction),
            "x": interpolate_at_fraction(x, left, fraction),
            "y": interpolate_at_fraction(y, left, fraction),
            "z": interpolate_at_fraction(z, left, fraction),
            "tvt": interpolate_at_fraction(tvt, left, fraction),
            "left_row_idx": float(left),
            "fraction": float(fraction),
        }
    return None


def extract_first_contacts(
    *,
    md: np.ndarray,
    z: np.ndarray,
    surfaces: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    tvt: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    surface_array = np.asarray(surfaces, dtype=np.float64)
    if surface_array.shape != (len(md), len(FORMATION_NAMES)):
        raise ValueError("surface matrix shape mismatch")
    contacts: list[dict[str, Any]] = []
    for formation_index, formation in enumerate(FORMATION_NAMES):
        contact = first_crossing(
            md,
            np.asarray(z, dtype=np.float64) - surface_array[:, formation_index],
            x=x,
            y=y,
            z=z,
            tvt=tvt,
        )
        if contact is None:
            continue
        contacts.append(
            {
                "formation": formation,
                "formation_index": formation_index,
                **contact,
            }
        )
    return contacts


def prefix_additive_offset(
    tvt_input: np.ndarray,
    z: np.ndarray,
    predicted_surfaces: np.ndarray,
    contact_centers: np.ndarray,
) -> tuple[float, int]:
    tvt_array = np.asarray(tvt_input, dtype=np.float64)
    z_array = np.asarray(z, dtype=np.float64)
    surfaces = np.asarray(predicted_surfaces, dtype=np.float64)
    centers = np.asarray(contact_centers, dtype=np.float64)
    if surfaces.shape != (len(tvt_array), len(FORMATION_NAMES)):
        raise ValueError("prefix surface shape mismatch")
    if centers.shape != (len(FORMATION_NAMES),):
        raise ValueError("contact center shape mismatch")
    prefix = np.isfinite(tvt_array) & np.isfinite(z_array)
    if not prefix.any():
        return float("nan"), 0
    implied_contact = (
        tvt_array[prefix, None]
        + z_array[prefix, None]
        - surfaces[prefix]
    )
    residual = implied_contact - centers[None, :]
    finite = residual[np.isfinite(residual)]
    if len(finite) == 0:
        return float("nan"), 0
    return float(np.median(finite)), int(len(finite))


# %% [markdown]
# ## 5. Fold identity and guarded raw-data loaders

# %%
@dataclass(frozen=True)
class TargetGeometry:
    well_id: str
    fold: int
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    tvt_input: np.ndarray
    suffix_row_idx: np.ndarray

    @property
    def rows(self) -> int:
        return len(self.md)

    @property
    def prefix_rows(self) -> int:
        return int(np.isfinite(self.tvt_input).sum())


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
        raise ValueError("fold identity contains duplicate well/row keys")
    if not frame.groupby("well_id", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("each well must have exactly one outer fold")
    for forbidden in FORBIDDEN_PRE_FREEZE_COLUMNS:
        if forbidden in frame.columns:
            raise RuntimeError(f"forbidden pre-freeze column loaded: {forbidden}")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count mismatch")
    if frame["well_id"].nunique() != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("exp226 OOF well count mismatch")
    if set(frame["fold"].astype(int).unique()) != set(
        int(value) for value in get_nested(config, "validation.expected_folds")
    ):
        raise ValueError("exp226 OOF fold inventory mismatch")
    return frame, {
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


def resolve_raw_train_dir(
    config: Mapping[str, Any],
    expected_wells: set[str],
) -> tuple[Path, list[Path]]:
    directories = [
        path
        for path in expand_existing_paths(
            [
                str(value)
                for value in get_nested(config, "data.raw_train_dir_patterns", [])
            ]
        )
        if path.is_dir()
    ]
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
        if wells == expected_wells:
            return directory, files
    raise FileNotFoundError(
        f"no raw train directory matched fold identity inventory: {evidence}"
    )


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


def load_target_geometry(
    path: Path,
    *,
    fold: int,
    identity: pd.DataFrame,
    ledger: RoleReadLedger,
) -> TargetGeometry:
    frame = pd.read_csv(path, usecols=list(TARGET_SAFE_COLUMNS))
    well_id = path.name.split("__horizontal_well.csv", 1)[0]
    ledger.record_target_safe(fold, well_id, TARGET_SAFE_COLUMNS, len(frame))
    arrays = {
        column: frame[column].to_numpy(np.float64)
        for column in TARGET_SAFE_COLUMNS
    }
    for column in ("MD", "X", "Y", "Z"):
        if not np.isfinite(arrays[column]).all():
            raise ValueError(f"{well_id} has nonfinite target-safe {column}")
    if np.any(np.diff(arrays["MD"]) <= 0.0):
        raise ValueError(f"{well_id} MD must be strictly increasing")
    finite_prefix = np.flatnonzero(np.isfinite(arrays["TVT_input"]))
    if len(finite_prefix) == 0:
        raise ValueError(f"{well_id} has no known TVT_input prefix")
    last_known = int(finite_prefix[-1])
    if not np.isfinite(arrays["TVT_input"][: last_known + 1]).all():
        raise ValueError(f"{well_id} TVT_input prefix is not contiguous")
    if np.isfinite(arrays["TVT_input"][last_known + 1 :]).any():
        raise ValueError(f"{well_id} TVT_input has finite suffix values")
    unknown = np.flatnonzero(~np.isfinite(arrays["TVT_input"]))
    ordered_identity = identity.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(
        unknown,
        ordered_identity["row_idx"].to_numpy(np.int64),
    ):
        raise ValueError(f"{well_id} suffix rows differ from fold identity")
    if not np.array_equal(
        ordered_identity["suffix_offset"].to_numpy(np.int32),
        np.arange(len(unknown), dtype=np.int32),
    ):
        raise ValueError(f"{well_id} suffix offsets are not contiguous")
    return TargetGeometry(
        well_id=well_id,
        fold=int(fold),
        md=arrays["MD"],
        x=arrays["X"],
        y=arrays["Y"],
        z=arrays["Z"],
        tvt_input=arrays["TVT_input"],
        suffix_row_idx=unknown,
    )


# %% [markdown]
# ## 6. Fold-local source fitting and target-free contact generation

# %%
@dataclass(frozen=True)
class FoldContactModel:
    fold: int
    surface: FormationPlaneKNN
    constant_surface: np.ndarray
    contact_centers: np.ndarray
    surface_references: pd.DataFrame
    contact_center_frame: pd.DataFrame


def fit_fold_contact_model(
    *,
    fold: int,
    source_wells: Sequence[str],
    path_by_well: Mapping[str, Path],
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> FoldContactModel:
    reference_records: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    for well_id in sorted(source_wells):
        path = path_by_well[well_id]
        frame = pd.read_csv(path, usecols=list(SOURCE_COLUMNS))
        ledger.record_source(fold, well_id, SOURCE_COLUMNS, len(frame))
        required_values = frame.loc[
            :,
            ["MD", "X", "Y", "Z", "TVT"],
        ].to_numpy(np.float64)
        if not np.isfinite(required_values).all():
            raise ValueError(
                f"{well_id} source geometry/TVT columns contain nonfinite values"
            )
        md = frame["MD"].to_numpy(np.float64)
        if np.any(np.diff(md) <= 0.0):
            raise ValueError(f"{well_id} source MD must be strictly increasing")
        reference: dict[str, Any] = {
            "fold": int(fold),
            "role": "outer_train",
            "well_id": well_id,
            "x": float(frame["X"].median()),
            "y": float(frame["Y"].median()),
        }
        for formation in FORMATION_NAMES:
            formation_values = frame[formation].to_numpy(np.float64)
            formation_finite = np.isfinite(formation_values)
            if formation_finite.any() and not formation_finite.all():
                raise ValueError(
                    f"{well_id} {formation} must be wholly finite or wholly missing"
                )
            reference[f"surface_{formation}"] = (
                float(np.median(formation_values))
                if formation_finite.all()
                else np.nan
            )
        reference_records.append(reference)
        true_contacts = extract_first_contacts(
            md=md,
            z=frame["Z"].to_numpy(np.float64),
            surfaces=frame.loc[:, list(FORMATION_NAMES)].to_numpy(np.float64),
            x=frame["X"].to_numpy(np.float64),
            y=frame["Y"].to_numpy(np.float64),
            tvt=frame["TVT"].to_numpy(np.float64),
        )
        for contact in true_contacts:
            contact_records.append(
                {
                    "fold": int(fold),
                    "well_id": well_id,
                    "formation": contact["formation"],
                    "formation_index": contact["formation_index"],
                    "contact_md": contact["md"],
                    "contact_tvt": contact["tvt"],
                    "contact_x": contact["x"],
                    "contact_y": contact["y"],
                    "contact_z": contact["z"],
                }
            )
    references = pd.DataFrame(reference_records).sort_values(
        "well_id",
        kind="mergesort",
    ).reset_index(drop=True)
    if len(references) < int(
        get_nested(config, "contact_model.surface.primary.nearest_wells")
    ):
        raise ValueError(f"fold {fold} lacks surface reference wells")
    formation_matrix = references[
        [f"surface_{name}" for name in FORMATION_NAMES]
    ].to_numpy(np.float64)
    contact_frame = pd.DataFrame(contact_records)
    center_records = []
    centers = np.full(len(FORMATION_NAMES), np.nan, dtype=np.float64)
    for formation_index, formation in enumerate(FORMATION_NAMES):
        selected = contact_frame.loc[
            contact_frame["formation"].eq(formation),
            "contact_tvt",
        ].to_numpy(np.float64)
        selected = selected[np.isfinite(selected)]
        if len(selected) == 0:
            raise ValueError(f"fold {fold} has no source contact for {formation}")
        centers[formation_index] = float(np.median(selected))
        center_records.append(
            {
                "fold": int(fold),
                "formation": formation,
                "formation_index": formation_index,
                "source_contact_count": int(len(selected)),
                "contact_tvt_center": centers[formation_index],
                "constant_surface": float(
                    np.nanmedian(formation_matrix[:, formation_index])
                ),
            }
        )
    constant_surface = np.nanmedian(formation_matrix, axis=0)
    surface = FormationPlaneKNN(
        wells=references["well_id"].astype(str).to_numpy(),
        xy=references[["x", "y"]].to_numpy(np.float64),
        formation_medians=formation_matrix,
        k=int(get_nested(config, "contact_model.surface.primary.nearest_wells")),
        chunk_rows=int(
            get_nested(config, "contact_model.surface.primary.query_chunk_rows")
        ),
    )
    return FoldContactModel(
        fold=int(fold),
        surface=surface,
        constant_surface=constant_surface,
        contact_centers=centers,
        surface_references=references,
        contact_center_frame=pd.DataFrame(center_records),
    )


def stable_resource_wells(
    well_ids: Sequence[str],
    count: int,
) -> set[str]:
    ranked = sorted(
        (hashlib.sha256(str(well).encode()).hexdigest(), str(well))
        for well in well_ids
    )
    return {well for _, well in ranked[: min(count, len(ranked))]}


def predict_target_contacts(
    *,
    geometry: TargetGeometry,
    model: FoldContactModel,
    resource_selected: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    start = time.perf_counter()
    plane_surface, support = model.surface.predict(
        np.column_stack([geometry.x, geometry.y])
    )
    constant_surface = np.broadcast_to(
        model.constant_surface[None, :],
        plane_surface.shape,
    ).copy()
    plane_offset, plane_prefix_values = prefix_additive_offset(
        geometry.tvt_input,
        geometry.z,
        plane_surface,
        model.contact_centers,
    )
    constant_offset, constant_prefix_values = prefix_additive_offset(
        geometry.tvt_input,
        geometry.z,
        constant_surface,
        model.contact_centers,
    )
    records: list[dict[str, Any]] = []
    method_inputs = [
        (PRIMARY_METHOD, plane_surface, plane_offset, plane_prefix_values),
        (
            CONTROL_METHOD,
            constant_surface,
            constant_offset,
            constant_prefix_values,
        ),
    ]
    for method, surfaces, offset, offset_count in method_inputs:
        contacts = extract_first_contacts(
            md=geometry.md,
            z=geometry.z,
            surfaces=surfaces,
            x=geometry.x,
            y=geometry.y,
        )
        for contact in contacts:
            formation_index = int(contact["formation_index"])
            records.append(
                {
                    "fold": int(geometry.fold),
                    "well_id": geometry.well_id,
                    "method": method,
                    "formation": contact["formation"],
                    "formation_index": formation_index,
                    "predicted_md": contact["md"],
                    "predicted_tvt": (
                        model.contact_centers[formation_index] + offset
                        if np.isfinite(offset)
                        else np.nan
                    ),
                    "contact_x": contact["x"],
                    "contact_y": contact["y"],
                    "contact_z": contact["z"],
                    "prefix_offset": offset,
                    "prefix_offset_value_count": int(offset_count),
                    "surface_fallback_fraction": (
                        float(np.mean(support["fallback"]))
                        if method == PRIMARY_METHOD
                        else 0.0
                    ),
                    "surface_effective_donors_p05": (
                        float(np.quantile(support["effective_donors"], 0.05))
                        if method == PRIMARY_METHOD
                        else float(len(model.surface.wells))
                    ),
                }
            )
    elapsed = time.perf_counter() - start
    target_manifest = {
        "fold": int(geometry.fold),
        "well_id": geometry.well_id,
        "target_safe_rows": int(geometry.rows),
        "prefix_rows": int(geometry.prefix_rows),
        "suffix_rows": int(len(geometry.suffix_row_idx)),
        "plane_surface_content_sha256": array_content_sha256(plane_surface),
        "constant_surface_content_sha256": array_content_sha256(constant_surface),
        "plane_prefix_offset": plane_offset,
        "constant_prefix_offset": constant_offset,
        "plane_crossing_count": int(
            sum(record["method"] == PRIMARY_METHOD for record in records)
        ),
        "constant_crossing_count": int(
            sum(record["method"] == CONTROL_METHOD for record in records)
        ),
        "surface_fallback_fraction": float(np.mean(support["fallback"])),
        "surface_effective_donors_p05": float(
            np.quantile(support["effective_donors"], 0.05)
        ),
    }
    resource_record = None
    if resource_selected:
        resource_record = {
            "fold": int(geometry.fold),
            "well_id": geometry.well_id,
            "rows": int(geometry.rows),
            "prefix_rows": int(geometry.prefix_rows),
            "elapsed_seconds": float(elapsed),
            "max_rss_gb_after_well": max_rss_gb(),
            "plane_crossings": target_manifest["plane_crossing_count"],
            "constant_crossings": target_manifest["constant_crossing_count"],
        }
    return records, target_manifest, resource_record


# %% [markdown]
# ## 7. Target-free freeze and 16-well resource readout

# %%
@dataclass(frozen=True)
class TargetFreeBundle:
    identity: pd.DataFrame
    target_manifest: pd.DataFrame
    crossings: pd.DataFrame
    surface_references: pd.DataFrame
    contact_centers: pd.DataFrame
    resource_audit: pd.DataFrame
    freeze_manifest: dict[str, Any]
    frozen_paths: tuple[Path, ...]
    ledger: RoleReadLedger


def build_target_free_bundle(
    *,
    config: Mapping[str, Any],
    identity: pd.DataFrame,
    identity_evidence: Mapping[str, Any],
    horizontal_files: Sequence[Path],
    path_by_well: Mapping[str, Path],
    artifacts_dir: Path,
) -> TargetFreeBundle:
    well_fold = (
        identity[["well_id", "fold"]]
        .drop_duplicates()
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_wells = set(well_fold["well_id"].astype(str))
    input_manifest = build_raw_input_manifest(horizontal_files)
    resource_well_set = stable_resource_wells(
        sorted(expected_wells),
        int(get_nested(config, "runtime.resource_audit_wells")),
    )
    ledger = RoleReadLedger()
    crossing_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    reference_frames: list[pd.DataFrame] = []
    center_frames: list[pd.DataFrame] = []
    resource_records: list[dict[str, Any]] = []
    for fold in sorted(well_fold["fold"].astype(int).unique()):
        source_wells = well_fold.loc[
            well_fold["fold"].astype(int).ne(fold),
            "well_id",
        ].astype(str).tolist()
        target_wells = well_fold.loc[
            well_fold["fold"].astype(int).eq(fold),
            "well_id",
        ].astype(str).tolist()
        model = fit_fold_contact_model(
            fold=int(fold),
            source_wells=source_wells,
            path_by_well=path_by_well,
            ledger=ledger,
            config=config,
        )
        reference_frames.append(model.surface_references)
        center_frames.append(model.contact_center_frame)
        for well_id in sorted(target_wells):
            well_identity = identity.loc[identity["well_id"].eq(well_id)]
            geometry = load_target_geometry(
                path_by_well[well_id],
                fold=int(fold),
                identity=well_identity,
                ledger=ledger,
            )
            contacts, target_manifest, resource_record = predict_target_contacts(
                geometry=geometry,
                model=model,
                resource_selected=well_id in resource_well_set,
            )
            crossing_records.extend(contacts)
            target_records.append(target_manifest)
            if resource_record is not None:
                resource_records.append(resource_record)
    ledger.freeze()
    surface_references = pd.concat(reference_frames, ignore_index=True).sort_values(
        ["fold", "well_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    contact_centers = pd.concat(center_frames, ignore_index=True).sort_values(
        ["fold", "formation_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    crossings = pd.DataFrame(crossing_records).sort_values(
        ["fold", "well_id", "method", "formation_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    target_manifest = pd.DataFrame(target_records).sort_values(
        ["fold", "well_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    resource_audit = pd.DataFrame(resource_records).sort_values(
        ["fold", "well_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(target_manifest) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("target manifest well count mismatch")
    if len(resource_audit) != int(get_nested(config, "runtime.resource_audit_wells")):
        raise ValueError("resource audit well count mismatch")
    role_ledger = ledger.pre_freeze_frame()
    paths = {
        "input_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "fold_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_fold_manifest.csv",
        "role_ledger": artifacts_dir / f"{OUTPUT_PREFIX}_role_read_ledger.csv",
        "surface_references": (
            artifacts_dir / f"{OUTPUT_PREFIX}_surface_references.csv.gz"
        ),
        "contact_centers": artifacts_dir / f"{OUTPUT_PREFIX}_contact_centers.csv",
        "crossings": (
            artifacts_dir / f"{OUTPUT_PREFIX}_target_free_crossings.csv.gz"
        ),
        "resource": artifacts_dir / f"{OUTPUT_PREFIX}_resource_audit.csv",
    }
    evidence = {
        "input_manifest": write_frame(paths["input_manifest"], input_manifest),
        "fold_manifest": write_frame(paths["fold_manifest"], target_manifest),
        "role_ledger": write_frame(paths["role_ledger"], role_ledger),
        "surface_references": write_gzip_frame(
            paths["surface_references"],
            surface_references,
        ),
        "contact_centers": write_frame(
            paths["contact_centers"],
            contact_centers,
        ),
        "crossings": write_gzip_frame(paths["crossings"], crossings),
        "resource": write_frame(paths["resource"], resource_audit),
    }
    bundle_digest = hashlib.sha256()
    for key in sorted(evidence):
        bundle_digest.update(key.encode())
        bundle_digest.update(
            str(evidence[key]["logical_content_sha256"]).encode()
        )
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "target_free_frozen": True,
        "validation_truth_reads_before_freeze": 0,
        "validation_formation_reads_before_freeze": 0,
        "fold_identity": dict(identity_evidence),
        "target_wells": len(target_manifest),
        "target_safe_rows": int(target_manifest["target_safe_rows"].sum()),
        "surface_reference_rows": len(surface_references),
        "target_free_crossing_rows": len(crossings),
        "resource_audit_wells": len(resource_audit),
        "artifact_evidence": evidence,
        "bundle_logical_sha256": bundle_digest.hexdigest(),
    }
    write_json(freeze_path, freeze_manifest)
    freeze_manifest["freeze_manifest_path"] = str(freeze_path)
    return TargetFreeBundle(
        identity=identity,
        target_manifest=target_manifest,
        crossings=crossings,
        surface_references=surface_references,
        contact_centers=contact_centers,
        resource_audit=resource_audit,
        freeze_manifest=freeze_manifest,
        frozen_paths=tuple([*paths.values(), freeze_path]),
        ledger=ledger,
    )


# %% [markdown]
# ## 8. Validation-truth late join and contact metrics

# %%
@dataclass(frozen=True)
class TruthReadout:
    truth_manifest: pd.DataFrame
    contact_events: pd.DataFrame
    fold_metrics: pd.DataFrame
    formation_metrics: pd.DataFrame
    pooled: dict[str, Any]


def load_truth_contacts_after_freeze(
    *,
    bundle: TargetFreeBundle,
    path_by_well: Mapping[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not bundle.ledger.target_free_frozen:
        raise RuntimeError("truth late join requires a frozen target-free bundle")
    well_fold = (
        bundle.identity[["well_id", "fold"]]
        .drop_duplicates()
        .sort_values("well_id", kind="mergesort")
    )
    truth_records: list[dict[str, Any]] = []
    for row in well_fold.itertuples(index=False):
        well_id = str(row.well_id)
        fold = int(row.fold)
        path = path_by_well[well_id]
        frame = pd.read_csv(path, usecols=list(TARGET_TRUTH_COLUMNS))
        bundle.ledger.record_target_truth(
            fold,
            well_id,
            TARGET_TRUTH_COLUMNS,
            len(frame),
        )
        contacts = extract_first_contacts(
            md=frame["MD"].to_numpy(np.float64),
            z=frame["Z"].to_numpy(np.float64),
            surfaces=frame.loc[:, list(FORMATION_NAMES)].to_numpy(np.float64),
            tvt=frame["TVT"].to_numpy(np.float64),
        )
        for contact in contacts:
            truth_records.append(
                {
                    "fold": fold,
                    "well_id": well_id,
                    "formation": contact["formation"],
                    "formation_index": contact["formation_index"],
                    "true_md": contact["md"],
                    "true_tvt": contact["tvt"],
                    "true_z": contact["z"],
                }
            )
    truth = pd.DataFrame(truth_records).sort_values(
        ["fold", "well_id", "formation_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return truth, bundle.ledger.late_truth_frame()


def paired_contact_events(
    truth: pd.DataFrame,
    crossings: pd.DataFrame,
    *,
    minimum_formations: int,
) -> pd.DataFrame:
    keys = ["fold", "well_id", "formation", "formation_index"]
    plane = crossings.loc[crossings["method"].eq(PRIMARY_METHOD)].copy()
    control = crossings.loc[crossings["method"].eq(CONTROL_METHOD)].copy()
    plane = plane.rename(
        columns={
            "predicted_md": "plane_md",
            "predicted_tvt": "plane_tvt",
            "prefix_offset": "plane_prefix_offset",
        }
    )
    control = control.rename(
        columns={
            "predicted_md": "constant_md",
            "predicted_tvt": "constant_tvt",
            "prefix_offset": "constant_prefix_offset",
        }
    )
    plane_columns = [
        *keys,
        "plane_md",
        "plane_tvt",
        "plane_prefix_offset",
        "surface_fallback_fraction",
        "surface_effective_donors_p05",
    ]
    control_columns = [
        *keys,
        "constant_md",
        "constant_tvt",
        "constant_prefix_offset",
    ]
    events = (
        truth.merge(plane[plane_columns], on=keys, how="inner", validate="one_to_one")
        .merge(
            control[control_columns],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )
    finite = (
        np.isfinite(events["true_md"])
        & np.isfinite(events["true_tvt"])
        & np.isfinite(events["plane_md"])
        & np.isfinite(events["plane_tvt"])
        & np.isfinite(events["constant_md"])
        & np.isfinite(events["plane_prefix_offset"])
        & np.isfinite(events["constant_prefix_offset"])
    )
    events = events.loc[finite].copy()
    count = events.groupby(["fold", "well_id"], sort=False)["formation"].transform(
        "size"
    )
    events["eligible"] = count.ge(int(minimum_formations))
    events["plane_md_abs_error"] = np.abs(events["plane_md"] - events["true_md"])
    events["constant_md_abs_error"] = np.abs(
        events["constant_md"] - events["true_md"]
    )
    events["plane_tvt_error"] = events["plane_tvt"] - events["true_tvt"]
    return events.reset_index(drop=True)


def order_readout(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fold",
        "well_id",
        "matched_formations",
        "truth_fixed_order",
        "plane_fixed_order",
        "correct_order",
    ]
    records: list[dict[str, Any]] = []
    eligible = events.loc[events["eligible"]].copy()
    for (fold, well_id), frame in eligible.groupby(
        ["fold", "well_id"],
        sort=True,
        observed=True,
    ):
        ordered = frame.sort_values("formation_index", kind="mergesort")
        truth_order = bool(np.all(np.diff(ordered["true_md"].to_numpy(float)) > 0))
        plane_order = bool(np.all(np.diff(ordered["plane_md"].to_numpy(float)) > 0))
        records.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "matched_formations": len(ordered),
                "truth_fixed_order": truth_order,
                "plane_fixed_order": plane_order,
                "correct_order": bool(truth_order and plane_order),
            }
        )
    return pd.DataFrame(records, columns=columns)


def rmse(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def contact_metric_record(
    *,
    fold: int | str,
    events: pd.DataFrame,
    order: pd.DataFrame,
    total_wells: int,
) -> dict[str, Any]:
    eligible = events.loc[events["eligible"]]
    eligible_wells = int(eligible["well_id"].nunique())
    plane_error = eligible["plane_md_abs_error"].to_numpy(np.float64)
    constant_error = eligible["constant_md_abs_error"].to_numpy(np.float64)
    plane_mae = float(np.mean(plane_error)) if len(plane_error) else float("nan")
    constant_mae = (
        float(np.mean(constant_error)) if len(constant_error) else float("nan")
    )
    return {
        "fold": fold,
        "total_wells": int(total_wells),
        "eligible_wells": eligible_wells,
        "eligible_well_fraction": (
            float(eligible_wells / total_wells) if total_wells else float("nan")
        ),
        "contact_event_count": int(len(eligible)),
        "crossing_md_mae_ft": plane_mae,
        "crossing_md_p90_ft": (
            float(np.quantile(plane_error, 0.90))
            if len(plane_error)
            else float("nan")
        ),
        "constant_crossing_md_mae_ft": constant_mae,
        "gain_vs_constant_surface_ft": constant_mae - plane_mae,
        "contact_tvt_rmse_ft": rmse(eligible["plane_tvt_error"]),
        "correct_order_fraction": (
            float(order["correct_order"].mean()) if len(order) else float("nan")
        ),
        "surface_fallback_fraction": (
            float(eligible["surface_fallback_fraction"].mean())
            if len(eligible)
            else float("nan")
        ),
        "surface_effective_donors_p05": (
            float(np.quantile(eligible["surface_effective_donors_p05"], 0.05))
            if len(eligible)
            else float("nan")
        ),
    }


def build_contact_metrics(
    *,
    events: pd.DataFrame,
    target_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    order = order_readout(events)
    fold_records = []
    for fold in sorted(target_manifest["fold"].astype(int).unique()):
        fold_events = events.loc[events["fold"].astype(int).eq(fold)]
        fold_order = order.loc[order["fold"].astype(int).eq(fold)]
        total_wells = int(
            target_manifest.loc[
                target_manifest["fold"].astype(int).eq(fold),
                "well_id",
            ].nunique()
        )
        fold_records.append(
            contact_metric_record(
                fold=int(fold),
                events=fold_events,
                order=fold_order,
                total_wells=total_wells,
            )
        )
    fold_metrics = pd.DataFrame(fold_records)
    pooled = contact_metric_record(
        fold="pooled",
        events=events,
        order=order,
        total_wells=int(target_manifest["well_id"].nunique()),
    )
    formation_records = []
    eligible = events.loc[events["eligible"]]
    for formation_index, formation in enumerate(FORMATION_NAMES):
        selected = eligible.loc[eligible["formation"].eq(formation)]
        formation_records.append(
            {
                "formation": formation,
                "formation_index": formation_index,
                "contact_event_count": len(selected),
                "eligible_wells": int(selected["well_id"].nunique()),
                "crossing_md_mae_ft": (
                    float(selected["plane_md_abs_error"].mean())
                    if len(selected)
                    else float("nan")
                ),
                "crossing_md_p90_ft": (
                    float(np.quantile(selected["plane_md_abs_error"], 0.90))
                    if len(selected)
                    else float("nan")
                ),
                "constant_crossing_md_mae_ft": (
                    float(selected["constant_md_abs_error"].mean())
                    if len(selected)
                    else float("nan")
                ),
                "contact_tvt_rmse_ft": rmse(selected["plane_tvt_error"]),
            }
        )
    return fold_metrics, pd.DataFrame(formation_records), pooled


def build_truth_readout(
    *,
    bundle: TargetFreeBundle,
    path_by_well: Mapping[str, Path],
    config: Mapping[str, Any],
) -> TruthReadout:
    truth, truth_manifest = load_truth_contacts_after_freeze(
        bundle=bundle,
        path_by_well=path_by_well,
    )
    events = paired_contact_events(
        truth,
        bundle.crossings,
        minimum_formations=int(
            get_nested(
                config,
                "contact_model.eligibility.minimum_triple_matched_formations",
            )
        ),
    )
    fold_metrics, formation_metrics, pooled = build_contact_metrics(
        events=events,
        target_manifest=bundle.target_manifest,
    )
    return TruthReadout(
        truth_manifest=truth_manifest,
        contact_events=events,
        fold_metrics=fold_metrics,
        formation_metrics=formation_metrics,
        pooled=pooled,
    )


# %% [markdown]
# ## 9. Fixed Stage 0 AND gate and generated artifacts

# %%
def evaluate_stage0_gate(
    pooled: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "gates.stage0_contact_predictability")
    positive_fold_count = int(
        (
            np.isfinite(fold_metrics["gain_vs_constant_surface_ft"])
            & fold_metrics["gain_vs_constant_surface_ft"].gt(0.0)
        ).sum()
    )
    checks = {
        "eligible_well_fraction": {
            "value": float(pooled["eligible_well_fraction"]),
            "threshold": float(gate["eligible_well_fraction_min"]),
            "operator": ">=",
            "passed": float(pooled["eligible_well_fraction"])
            >= float(gate["eligible_well_fraction_min"]),
        },
        "contact_event_count": {
            "value": int(pooled["contact_event_count"]),
            "threshold": int(gate["contact_event_count_min"]),
            "operator": ">=",
            "passed": int(pooled["contact_event_count"])
            >= int(gate["contact_event_count_min"]),
        },
        "crossing_md_mae_ft": {
            "value": float(pooled["crossing_md_mae_ft"]),
            "threshold": float(gate["crossing_md_mae_ft_max"]),
            "operator": "<=",
            "passed": float(pooled["crossing_md_mae_ft"])
            <= float(gate["crossing_md_mae_ft_max"]),
        },
        "crossing_md_p90_ft": {
            "value": float(pooled["crossing_md_p90_ft"]),
            "threshold": float(gate["crossing_md_p90_ft_max"]),
            "operator": "<=",
            "passed": float(pooled["crossing_md_p90_ft"])
            <= float(gate["crossing_md_p90_ft_max"]),
        },
        "contact_tvt_rmse_ft": {
            "value": float(pooled["contact_tvt_rmse_ft"]),
            "threshold": float(gate["contact_tvt_rmse_ft_max"]),
            "operator": "<=",
            "passed": float(pooled["contact_tvt_rmse_ft"])
            <= float(gate["contact_tvt_rmse_ft_max"]),
        },
        "correct_order_fraction": {
            "value": float(pooled["correct_order_fraction"]),
            "threshold": float(gate["correct_order_fraction_min"]),
            "operator": ">=",
            "passed": float(pooled["correct_order_fraction"])
            >= float(gate["correct_order_fraction_min"]),
        },
        "positive_fold_count": {
            "value": positive_fold_count,
            "threshold": int(gate["positive_fold_count_min"]),
            "operator": ">=",
            "passed": positive_fold_count >= int(gate["positive_fold_count_min"]),
        },
        "gain_vs_constant_surface_ft": {
            "value": float(pooled["gain_vs_constant_surface_ft"]),
            "threshold": float(gate["gain_vs_constant_surface_ft_min"]),
            "operator": ">=",
            "passed": float(pooled["gain_vs_constant_surface_ft"])
            >= float(gate["gain_vs_constant_surface_ft_min"]),
        },
    }
    return {
        "stage": "stage0_contact_predictability",
        "passed": bool(all(check["passed"] for check in checks.values())),
        "checks": checks,
        "positive_fold_count": positive_fold_count,
        "all_checks_required": True,
    }


def write_sha_manifest(
    artifacts_dir: Path,
    paths: Sequence[Path],
) -> Path:
    records = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        record = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "decompressed_content_sha256": "",
        }
        if path.suffix == ".gz":
            record["decompressed_content_sha256"] = sha256_decompressed_gzip(path)
        records.append(record)
    manifest = pd.DataFrame(records).sort_values("path", kind="mergesort")
    path = artifacts_dir / f"{OUTPUT_PREFIX}_sha_manifest.csv"
    write_frame(path, manifest)
    return path


def persist_stage0_outputs(
    *,
    bundle: TargetFreeBundle,
    truth: TruthReadout,
    stage0: Mapping[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    paths = {
        "truth_manifest": (
            artifacts_dir / f"{OUTPUT_PREFIX}_truth_late_join_manifest.csv"
        ),
        "events": (
            artifacts_dir / f"{OUTPUT_PREFIX}_contact_event_readout.csv.gz"
        ),
        "fold_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "formation_metrics": (
            artifacts_dir / f"{OUTPUT_PREFIX}_formation_metrics.csv"
        ),
        "stage0": artifacts_dir / f"{OUTPUT_PREFIX}_stage0_gate.json",
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    evidence = {
        "truth_manifest": write_frame(paths["truth_manifest"], truth.truth_manifest),
        "events": write_gzip_frame(paths["events"], truth.contact_events),
        "fold_metrics": write_frame(paths["fold_metrics"], truth.fold_metrics),
        "formation_metrics": write_frame(
            paths["formation_metrics"],
            truth.formation_metrics,
        ),
    }
    write_json(paths["stage0"], stage0)
    decision = (
        "stage0_passed_request_separate_stage1_implementation_approval"
        if bool(stage0["passed"])
        else "stage0_failed_close_without_semimarkov_hmm"
    )
    resource = bundle.resource_audit
    projected_runtime = (
        float(resource["elapsed_seconds"].mean())
        * int(bundle.target_manifest["well_id"].nunique())
        if len(resource)
        else float("nan")
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed",
        "route": "pf_beam",
        "completed_at": datetime.now(UTC).isoformat(),
        "stage0": dict(stage0),
        "pooled": truth.pooled,
        "target_free_bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "truth_manifest_logical_sha256": evidence["truth_manifest"][
            "logical_content_sha256"
        ],
        "contact_event_logical_sha256": evidence["events"][
            "logical_content_sha256"
        ],
        "resource": {
            "audited_wells": len(resource),
            "elapsed_seconds_sum": (
                float(resource["elapsed_seconds"].sum()) if len(resource) else None
            ),
            "projected_773_well_seconds": projected_runtime,
            "max_rss_gb": (
                float(resource["max_rss_gb_after_well"].max())
                if len(resource)
                else None
            ),
            "gate_mode": "report_only",
        },
        "execution": {
            **EXPECTED_RUNTIME_COUNTS,
            "parent_control_regeneration": 0,
            "gpu": False,
            "inference": False,
            "submission": False,
        },
        "stage1_implemented": False,
        "stage1_run": False,
        "decision": decision,
    }
    write_json(paths["summary"], summary)
    sha_manifest_path = write_sha_manifest(
        artifacts_dir,
        [*bundle.frozen_paths, *paths.values()],
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_completed",
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "stage0_passed": bool(stage0["passed"]),
        "stage1_implemented": False,
        "stage1_run": False,
        "pooled": truth.pooled,
        "gate": dict(stage0),
        "decision": decision,
        "target_free_bundle_logical_sha256": bundle.freeze_manifest[
            "bundle_logical_sha256"
        ],
        "contact_event_logical_sha256": evidence["events"][
            "logical_content_sha256"
        ],
        "sha_manifest_file_sha256": sha256_file(sha_manifest_path),
    }
    write_json(runtime_metrics_path(), metrics)
    return summary


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_execution_contract(config, require_run_authorization=True)
    artifacts_dir = runtime_artifacts_dir()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print(
        "Execution: 1 diagnostic / 6 reporting surfaces / 5 outer folds / "
        "0 model / 0 HMM / 0 PF / 0 Beam / 0 LightGBM booster"
    )
    print("Parent/control regeneration: 0")
    identity, identity_evidence = load_exp226_fold_identity(config)
    expected_wells = set(identity["well_id"].astype(str))
    _, horizontal_files = resolve_raw_train_dir(config, expected_wells)
    path_by_well = {
        path.name.split("__horizontal_well.csv", 1)[0]: path
        for path in horizontal_files
    }
    bundle = build_target_free_bundle(
        config=config,
        identity=identity,
        identity_evidence=identity_evidence,
        horizontal_files=horizontal_files,
        path_by_well=path_by_well,
        artifacts_dir=artifacts_dir,
    )
    print(
        "Target-free bundle frozen:",
        bundle.freeze_manifest["bundle_logical_sha256"],
    )
    print("Validation truth/formation reads before freeze: 0 / 0")
    truth = build_truth_readout(
        bundle=bundle,
        path_by_well=path_by_well,
        config=config,
    )
    stage0 = evaluate_stage0_gate(truth.pooled, truth.fold_metrics, config)
    print("Stage 0 PASS:", stage0["passed"])
    print(json.dumps(to_jsonable(stage0), indent=2, ensure_ascii=False))
    summary = persist_stage0_outputs(
        bundle=bundle,
        truth=truth,
        stage0=stage0,
        artifacts_dir=artifacts_dir,
    )
    print("Stage 1 implemented/run: 0 / 0")
    print("Decision:", summary["decision"])
    print("Generated artifacts:", artifacts_dir)
    return summary


# %% [markdown]
# ## 10. Setup, configuration preview, and execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG_PATH = find_config_path()
    CONFIG = read_yaml(CONFIG_PATH)
    print("Config:", CONFIG_PATH)
    print("Config SHA256:", sha256_file(CONFIG_PATH))
    validate_execution_contract(CONFIG, require_run_authorization=True)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment"),
                "lineage": get_nested(CONFIG, "lineage"),
                "validation": get_nested(CONFIG, "validation"),
                "contact_model": get_nested(CONFIG, "contact_model"),
                "gates": get_nested(CONFIG, "gates"),
                "runtime": get_nested(CONFIG, "runtime"),
                "execution": get_nested(CONFIG, "execution"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    SUMMARY = run_stage0(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY), indent=2, ensure_ascii=False))
