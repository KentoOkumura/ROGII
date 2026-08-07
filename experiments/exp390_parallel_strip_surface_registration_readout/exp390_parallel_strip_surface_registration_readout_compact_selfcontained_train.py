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
# # exp390 parallel strip surface registration readout — train
#
# This deterministic, fold-safe readout registers nearly parallel horizontal
# wells in a query-centric along-track/cross-track coordinate. It evaluates one
# two-sided strip-surface candidate against the saved exp226 OOF prediction.

# %% [markdown]
# ## Contents
#
# 1. Imports and frozen execution contract
# 2. Runtime, path, table, and SHA helpers
# 3. Fold-safe input and role-read guards
# 4. Query-centric strip geometry and pair eligibility
# 5. Same-s donor interpolation and robust cross-track fitting
# 6. Prefix calibration and exp226 fallback candidate
# 7. Stage 0 target-free support/resource gate
# 8. Stage 1 known-prefix rolling-origin gate
# 9. Stage 2 truth-late score and promotion-safety readout
# 10. Artifact manifest and execution orchestration

# %% [markdown]
# ## 1. Imports and frozen execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp390_parallel_strip_surface_registration_readout"
PARENT_EXPERIMENT = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
)
PACKAGE_DIR = Path.cwd()
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

SOURCE_COLUMNS = ("MD", "X", "Y", "Z", "TVT")
TARGET_SAFE_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
FORBIDDEN_TARGET_COLUMNS = (
    "TVT",
    "GR",
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)
PARENT_SAFE_COLUMNS = ("well_id", "row_idx", "suffix_offset", "tvt_pred", "fold")
PARENT_TRUTH_COLUMNS = (*PARENT_SAFE_COLUMNS, "tvt_true")
CANDIDATE_NAME = "parallel_strip_two_sided_fallback_exp226"


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_kaggle_authorization: bool,
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name differs from the exp390 contract")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp390 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp390 parent must remain exp226")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise ValueError("exp390 implementation is not authorized")
    if int(get_nested(config, "runtime.scientific_variants")) != 1:
        raise ValueError("exp390 permits exactly one scientific variant")
    if int(get_nested(config, "runtime.active_candidates")) != 1:
        raise ValueError("exp390 permits exactly one active candidate")
    if int(get_nested(config, "runtime.reporting_folds")) != 5:
        raise ValueError("exp390 requires five reporting folds")
    zero_counts = (
        "runtime.fitted_models",
        "runtime.model_configs",
        "runtime.trained_folds",
        "runtime.lightgbm_boosters",
        "runtime.hmm_runs",
        "runtime.pf_runs",
        "runtime.beam_runs",
    )
    nonzero = {key: get_nested(config, key) for key in zero_counts if get_nested(config, key) != 0}
    if nonzero:
        raise ValueError(f"fitted-model/PF/Beam counts must remain zero: {nonzero}")
    if bool(get_nested(config, "runtime.parent_control_regeneration", True)):
        raise ValueError("the exp226 control must not be regenerated")
    if bool(get_nested(config, "execution.inference_enabled", False)):
        raise ValueError("exp390 inference must remain disabled")
    if bool(get_nested(config, "execution.submission_enabled", False)):
        raise ValueError("exp390 submission must remain disabled")
    if require_kaggle_authorization:
        if not bool(get_nested(config, "execution.kaggle_execution_authorized", False)):
            raise RuntimeError("Kaggle execution is not authorized for exp390")
        if get_nested(config, "execution.current_mode") not in {
            "stage0_resource_preflight",
            "full_run",
        }:
            raise RuntimeError("exp390 execution mode is not preflight or full_run")


# %% [markdown]
# ## 2. Runtime, path, table, and SHA helpers

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found in: {candidates}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    value = yaml.safe_load((path or config_path()).read_text())
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain a mapping")
    return value


def is_kaggle_runtime() -> bool:
    return KAGGLE_WORKING_ROOT.exists() and KAGGLE_INPUT_ROOT.exists()


def output_root() -> Path:
    if is_kaggle_runtime():
        return KAGGLE_WORKING_ROOT
    return find_project_root() / "experiments" / EXPERIMENT_NAME


def artifacts_dir() -> Path:
    path = output_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: (
            item.item()
            if isinstance(item, np.generic)
            else str(item)
        ),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return sha256_bytes(stable_json_bytes(schema))


def frame_content_sha256(
    frame: pd.DataFrame,
    sort_columns: Sequence[str],
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(columns or frame.columns)
    ordered = frame.loc[:, selected].sort_values(
        list(sort_columns),
        kind="mergesort",
        na_position="last",
    )
    digest = hashlib.sha256()
    digest.update((",".join(selected) + "\n").encode())
    for start in range(0, len(ordered), 50_000):
        text = ordered.iloc[start : start + 50_000].to_csv(
            index=False,
            header=False,
            float_format="%.17g",
            na_rep="<NA>",
            lineterminator="\n",
        )
        digest.update(text.encode())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.name.endswith(".csv.gz"):
        frame.to_csv(
            path,
            index=False,
            float_format="%.17g",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False, float_format="%.17g")


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if os.uname().sysname != "Darwin" else 1024.0**3
    return value / divisor


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    y = np.asarray(list(actual) if not isinstance(actual, np.ndarray) else actual, dtype=float)
    p = np.asarray(
        list(predicted) if not isinstance(predicted, np.ndarray) else predicted,
        dtype=float,
    )
    finite = np.isfinite(y) & np.isfinite(p)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y[finite] - p[finite]))))


def resolve_candidate_file(candidates: Sequence[str], filename: str) -> Path:
    root = find_project_root()
    checked: list[Path] = []
    for raw in candidates:
        base = Path(raw)
        if not base.is_absolute():
            base = root / base
        candidate = base if base.name == filename else base / filename
        checked.append(candidate)
        if candidate.exists():
            return candidate
    kaggle_matches = (
        sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if KAGGLE_INPUT_ROOT.exists()
        else []
    )
    for candidate in kaggle_matches:
        return candidate
    raise FileNotFoundError(f"{filename} was not found; checked {checked}")


def select_train_dir(
    matches: Sequence[Path],
    expected_wells: int,
) -> Path:
    counts: dict[Path, int] = {}
    for path in matches:
        counts[path.parent] = counts.get(path.parent, 0) + 1
    eligible = sorted(
        directory
        for directory, count in counts.items()
        if count == expected_wells
    )
    if len(eligible) == 1:
        return eligible[0]
    summary = {
        str(directory): count
        for directory, count in sorted(counts.items())
    }
    if not eligible:
        raise FileNotFoundError(
            f"no horizontal-well directory contains exactly {expected_wells} files; "
            f"candidate counts={summary}"
        )
    raise RuntimeError(
        f"multiple horizontal-well directories contain exactly {expected_wells} files; "
        f"eligible={list(map(str, eligible))}; candidate counts={summary}"
    )


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    local = find_project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))
    if local.exists():
        return select_train_dir(
            sorted(local.glob("*__horizontal_well.csv")),
            expected_wells,
        )
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob("*__horizontal_well.csv"))
        if matches:
            return select_train_dir(matches, expected_wells)
    raise FileNotFoundError("horizontal-well train directory was not found")


# %% [markdown]
# ## 3. Fold-safe input and role-read guards

# %%
@dataclass
class RoleReadLedger:
    source_rows: int = 0
    target_safe_rows: int = 0
    target_suffix_truth_reads: int = 0
    target_raw_formation_reads: int = 0
    target_gr_reads: int = 0
    source_valid_overlap: int = 0
    frozen: bool = False
    truth_joined_after_freeze: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_source(self, columns: Sequence[str], rows: int, fold: int) -> None:
        unexpected = set(columns).difference(SOURCE_COLUMNS)
        if unexpected:
            raise ValueError(f"source reader requested unexpected columns: {sorted(unexpected)}")
        self.source_rows += int(rows)
        self.events.append({"fold": int(fold), "role": "outer_train", "rows": int(rows)})

    def record_target(self, columns: Sequence[str], rows: int, fold: int) -> None:
        requested = set(columns)
        forbidden = requested.intersection(FORBIDDEN_TARGET_COLUMNS)
        if forbidden:
            self.target_suffix_truth_reads += int("TVT" in forbidden)
            self.target_gr_reads += int("GR" in forbidden)
            self.target_raw_formation_reads += len(
                forbidden.intersection(set(FORBIDDEN_TARGET_COLUMNS[2:]))
            )
            raise ValueError(f"target-safe reader requested forbidden columns: {sorted(forbidden)}")
        self.target_safe_rows += int(rows)
        self.events.append({"fold": int(fold), "role": "outer_valid_safe", "rows": int(rows)})

    def record_role_overlap(
        self,
        source_wells: Iterable[str],
        target_wells: Iterable[str],
        fold: int = -1,
    ) -> None:
        overlap = sorted(set(source_wells).intersection(target_wells))
        self.source_valid_overlap += len(overlap)
        if overlap:
            raise ValueError(f"outer-train/outer-valid role overlap leaked: {overlap[:5]}")
        self.events.append({"fold": int(fold), "role": "role_overlap_check", "rows": 0})

    def freeze(self) -> None:
        self.frozen = True

    def record_truth_late(self, rows: int) -> None:
        if not self.frozen:
            raise RuntimeError("target suffix truth cannot be read before prediction freeze")
        self.truth_joined_after_freeze = True
        self.target_suffix_truth_reads += int(rows)
        self.events.append({"fold": -1, "role": "truth_late", "rows": int(rows)})

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("events")
        return value

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


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


def read_source_well(
    path: Path,
    fold: int,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(SOURCE_COLUMNS))
    ledger.record_source(frame.columns, len(frame), fold)
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
    frame = pd.read_csv(path, usecols=list(TARGET_SAFE_COLUMNS))
    ledger.record_target(frame.columns, len(frame), fold)
    frame["well_id"] = well_id_from_path(path)
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["fold"] = int(fold)
    frame["role"] = "outer_valid"
    return frame


def validate_parent_oof_contract(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    require_columns(frame, PARENT_SAFE_COLUMNS, "exp226 OOF")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count differs from the frozen contract")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 OOF well count differs from the frozen contract")
    if sorted(frame["fold"].astype(int).unique().tolist()) != list(
        get_nested(config, "validation.expected_folds")
    ):
        raise ValueError("exp226 OOF folds differ from the frozen contract")
    folds_per_well = frame.groupby("well_id", sort=True)["fold"].nunique()
    if not folds_per_well.eq(1).all():
        raise ValueError("exp226 OOF assigns more than one fold to a well")


def load_parent_oof(
    config: Mapping[str, Any],
    *,
    include_truth: bool,
    ledger: RoleReadLedger | None = None,
) -> tuple[pd.DataFrame, Path]:
    filename = str(get_nested(config, "data.parent_exp226.filename"))
    candidates = list(get_nested(config, "data.parent_exp226.candidates", []))
    path = resolve_candidate_file(candidates, filename)
    expected = str(get_nested(config, "data.parent_exp226.expected_decompressed_sha256"))
    actual = sha256_decompressed_csv(path)
    if actual != expected:
        raise ValueError(f"exp226 OOF SHA mismatch: expected {expected}, got {actual}")
    if include_truth:
        if ledger is None:
            raise ValueError("truth-late parent read requires a role ledger")
        ledger.record_truth_late(int(get_nested(config, "validation.expected_rows")))
        columns = list(PARENT_TRUTH_COLUMNS)
    else:
        columns = list(PARENT_SAFE_COLUMNS)
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["fold"] = frame["fold"].astype(int)
    validate_parent_oof_contract(frame, config)
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
    mismatch = [
        well
        for well, fold in fold_by_well.items()
        if int(observed[well]) != int(fold)
    ]
    if mismatch:
        raise ValueError(f"exp226 fold identity mismatch: {mismatch[:5]}")


# %% [markdown]
# ## 4. Query-centric strip geometry and pair eligibility

# %%
@dataclass(frozen=True)
class Axis2D:
    es_x: float
    es_y: float
    en_x: float
    en_y: float
    centroid_x: float
    centroid_y: float


@dataclass
class DonorTrack:
    well_id: str
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    surface: np.ndarray
    axis: Axis2D


def canonical_pca_axis(x: Sequence[float], y: Sequence[float]) -> Axis2D:
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    finite = np.isfinite(xy).all(axis=1)
    if finite.sum() < 2:
        raise ValueError("at least two finite XY rows are required for PCA")
    clean = xy[finite]
    center = clean.mean(axis=0)
    centered = clean - center
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))].astype(float)
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0.0:
        axis *= -1.0
    axis /= np.linalg.norm(axis)
    return Axis2D(
        es_x=float(axis[0]),
        es_y=float(axis[1]),
        en_x=float(-axis[1]),
        en_y=float(axis[0]),
        centroid_x=float(center[0]),
        centroid_y=float(center[1]),
    )


def axial_angle_mismatch_deg(first: Axis2D, second: Axis2D) -> float:
    dot = abs(first.es_x * second.es_x + first.es_y * second.es_y)
    return float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))


def project_xy(
    x: Sequence[float],
    y: Sequence[float],
    axis: Axis2D,
) -> tuple[np.ndarray, np.ndarray]:
    x_value = np.asarray(x, dtype=float)
    y_value = np.asarray(y, dtype=float)
    along = x_value * axis.es_x + y_value * axis.es_y
    normal = x_value * axis.en_x + y_value * axis.en_y
    return along, normal


def projected_overlap_fraction(
    query_min: float,
    query_max: float,
    donor_min: float,
    donor_max: float,
) -> float:
    query_span = max(float(query_max) - float(query_min), 0.0)
    if query_span <= 0.0:
        return 0.0
    overlap = max(
        0.0,
        min(float(query_max), float(donor_max))
        - max(float(query_min), float(donor_min)),
    )
    return float(overlap / query_span)


def monotone_step_fraction(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) < 2:
        return 0.0
    delta = np.diff(finite)
    return float(np.mean(delta > 0.0))


def _bbox_interval(
    x: np.ndarray,
    y: np.ndarray,
    direction_x: float,
    direction_y: float,
) -> tuple[float, float]:
    corners_x = np.asarray([np.nanmin(x), np.nanmin(x), np.nanmax(x), np.nanmax(x)])
    corners_y = np.asarray([np.nanmin(y), np.nanmax(y), np.nanmin(y), np.nanmax(y)])
    projection = corners_x * direction_x + corners_y * direction_y
    return float(np.min(projection)), float(np.max(projection))


def _interval_distance(
    first_min: float,
    first_max: float,
    second_min: float,
    second_max: float,
) -> float:
    if first_max < second_min:
        return float(second_min - first_max)
    if second_max < first_min:
        return float(first_min - second_max)
    return 0.0


def _sorted_unique_projection(
    along: np.ndarray,
    *values: np.ndarray,
) -> tuple[np.ndarray, ...]:
    order = np.argsort(along, kind="mergesort")
    ordered_s = along[order]
    finite = np.isfinite(ordered_s)
    for value in values:
        finite &= np.isfinite(value[order])
    ordered_s = ordered_s[finite]
    ordered_values = [value[order][finite] for value in values]
    if len(ordered_s) == 0:
        return (ordered_s, *ordered_values)
    unique_s, first, counts = np.unique(ordered_s, return_index=True, return_counts=True)
    if np.all(counts == 1):
        return (unique_s, *ordered_values)
    collapsed = [
        np.add.reduceat(value, first) / counts
        for value in ordered_values
    ]
    return (unique_s, *collapsed)


def build_donor_track(frame: pd.DataFrame) -> DonorTrack:
    require_columns(frame, ("well_id", "MD", "X", "Y", "Z", "TVT"), "source well")
    ordered = frame.sort_values(["MD", "row_idx"], kind="mergesort")
    return DonorTrack(
        well_id=str(ordered["well_id"].iloc[0]),
        md=ordered["MD"].to_numpy(dtype=float),
        x=ordered["X"].to_numpy(dtype=float),
        y=ordered["Y"].to_numpy(dtype=float),
        surface=(
            ordered["TVT"].to_numpy(dtype=float)
            + ordered["Z"].to_numpy(dtype=float)
        ),
        axis=canonical_pca_axis(ordered["X"], ordered["Y"]),
    )


def build_query_geometry(
    target: pd.DataFrame,
    parent_rows: pd.DataFrame,
) -> dict[str, Any]:
    axis = canonical_pca_axis(target["X"], target["Y"])
    score = target.merge(
        parent_rows[["row_idx"]],
        on="row_idx",
        how="inner",
        validate="one_to_one",
    ).sort_values("row_idx", kind="mergesort")
    if score.empty:
        raise ValueError("query well has no exp226 score rows")
    score_s, score_n = project_xy(score["X"], score["Y"], axis)
    full_s, full_n = project_xy(target["X"], target["Y"], axis)
    query_s, query_n = _sorted_unique_projection(full_s, full_n)
    return {
        "axis": axis,
        "score_s": score_s,
        "score_n": score_n,
        "score_s_min": float(np.min(score_s)),
        "score_s_max": float(np.max(score_s)),
        "query_s": query_s,
        "query_n": query_n,
        "score_rows": int(len(score)),
    }


def evaluate_pair(
    query_well: str,
    query_geometry: Mapping[str, Any],
    donor: DonorTrack,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    axis = query_geometry["axis"]
    angle = axial_angle_mismatch_deg(axis, donor.axis)
    angle_max = float(
        get_nested(config, "strip_coordinate.pair_eligibility.axial_angle_mismatch_deg_max")
    )
    result: dict[str, Any] = {
        "query_well_id": str(query_well),
        "donor_well_id": str(donor.well_id),
        "angle_mismatch_deg": angle,
        "projected_overlap": 0.0,
        "median_abs_cross_track_distance_ft": float("nan"),
        "projected_s_monotone_step_fraction": 0.0,
        "eligible": False,
        "reason": "angle",
    }
    if donor.well_id == query_well:
        result["reason"] = "self"
        return result
    if angle > angle_max:
        return result

    donor_s_bbox = _bbox_interval(donor.x, donor.y, axis.es_x, axis.es_y)
    overlap_upper = projected_overlap_fraction(
        query_geometry["score_s_min"],
        query_geometry["score_s_max"],
        donor_s_bbox[0],
        donor_s_bbox[1],
    )
    overlap_min = float(
        get_nested(config, "strip_coordinate.pair_eligibility.projected_along_track_overlap_min")
    )
    if overlap_upper < overlap_min:
        result["reason"] = "overlap"
        return result

    donor_n_bbox = _bbox_interval(donor.x, donor.y, axis.en_x, axis.en_y)
    query_n_min = float(np.min(query_geometry["score_n"]))
    query_n_max = float(np.max(query_geometry["score_n"]))
    cross_max = float(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.median_abs_cross_track_distance_ft_max",
        )
    )
    if _interval_distance(*donor_n_bbox, query_n_min, query_n_max) > cross_max:
        result["reason"] = "cross_track"
        return result

    donor_s, donor_n = project_xy(donor.x, donor.y, axis)
    monotone = monotone_step_fraction(donor_s)
    result["projected_s_monotone_step_fraction"] = monotone
    if monotone < float(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.projected_s_monotone_step_fraction_min",
        )
    ):
        result["reason"] = "non_monotone"
        return result
    donor_s_sorted, donor_n_sorted = _sorted_unique_projection(donor_s, donor_n)
    if len(donor_s_sorted) < 2:
        result["reason"] = "non_monotone"
        return result
    overlap = projected_overlap_fraction(
        query_geometry["score_s_min"],
        query_geometry["score_s_max"],
        donor_s_sorted[0],
        donor_s_sorted[-1],
    )
    result["projected_overlap"] = overlap
    if overlap < overlap_min:
        result["reason"] = "overlap"
        return result
    sample_s = np.asarray(query_geometry["score_s"], dtype=float)
    inside = (sample_s >= donor_s_sorted[0]) & (sample_s <= donor_s_sorted[-1])
    if not inside.any():
        result["reason"] = "no_overlap"
        return result
    donor_n_at_query = np.interp(sample_s[inside], donor_s_sorted, donor_n_sorted)
    query_n_at_s = np.interp(
        sample_s[inside],
        np.asarray(query_geometry["query_s"], dtype=float),
        np.asarray(query_geometry["query_n"], dtype=float),
    )
    cross = float(np.median(np.abs(donor_n_at_query - query_n_at_s)))
    result["median_abs_cross_track_distance_ft"] = cross
    if cross > cross_max:
        result["reason"] = "cross_track"
        return result
    result["eligible"] = True
    result["reason"] = "eligible"
    return result


def select_eligible_pairs(
    query_well: str,
    query_geometry: Mapping[str, Any],
    donors: Mapping[str, DonorTrack],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    eligible = [
        evaluate_pair(query_well, query_geometry, donors[well], config)
        for well in sorted(donors)
    ]
    frame = pd.DataFrame(eligible)
    frame = frame.loc[frame["eligible"]].copy()
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["median_abs_cross_track_distance_ft", "donor_well_id"],
        kind="mergesort",
    )
    limit = int(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.maximum_unique_donor_wells",
        )
    )
    frame["selected"] = False
    frame.loc[frame.index[:limit], "selected"] = True
    return frame.reset_index(drop=True)


# %% [markdown]
# ## 5. Same-s donor interpolation and robust cross-track fitting

# %%
def build_query_nodes(target: pd.DataFrame, step_ft: float) -> pd.DataFrame:
    ordered = target.sort_values(["MD", "row_idx"], kind="mergesort")
    md = ordered["MD"].to_numpy(dtype=float)
    finite = np.isfinite(md)
    if finite.sum() < 2:
        return pd.DataFrame()
    md = md[finite]
    start = float(md[0])
    stop = float(md[-1])
    grid = np.arange(start, stop + 0.5 * step_ft, step_ft, dtype=float)
    grid = grid[grid <= stop]
    if len(grid) == 0 or not np.isclose(grid[-1], stop):
        grid = np.append(grid, stop)
    return pd.DataFrame(
        {
            "fold": int(ordered["fold"].iloc[0]),
            "well_id": str(ordered["well_id"].iloc[0]),
            "node_idx": np.arange(len(grid), dtype=np.int32),
            "node_md": grid,
            "query_x": np.interp(grid, md, ordered.loc[finite, "X"].to_numpy(float)),
            "query_y": np.interp(grid, md, ordered.loc[finite, "Y"].to_numpy(float)),
        }
    )


def interpolate_donor_at_nodes(
    nodes: pd.DataFrame,
    donor: DonorTrack,
    axis: Axis2D,
    circular_shift_rows: int,
) -> pd.DataFrame:
    donor_s, _ = project_xy(donor.x, donor.y, axis)
    circular_surface = np.roll(donor.surface, int(circular_shift_rows))
    donor_s, donor_x, donor_y, surface, circular = _sorted_unique_projection(
        donor_s,
        donor.x,
        donor.y,
        donor.surface,
        circular_surface,
    )
    if len(donor_s) < 2:
        return pd.DataFrame()
    node_s, _ = project_xy(nodes["query_x"], nodes["query_y"], axis)
    inside = (node_s >= donor_s[0]) & (node_s <= donor_s[-1])
    if not inside.any():
        return pd.DataFrame()
    selected = nodes.loc[inside].copy()
    selected_s = node_s[inside]
    donor_x = np.interp(selected_s, donor_s, donor_x)
    donor_y = np.interp(selected_s, donor_s, donor_y)
    selected["node_s"] = selected_s
    selected["donor_well_id"] = donor.well_id
    selected["donor_x"] = donor_x
    selected["donor_y"] = donor_y
    selected["donor_n"] = (
        (donor_x - selected["query_x"].to_numpy(float)) * axis.en_x
        + (donor_y - selected["query_y"].to_numpy(float)) * axis.en_y
    )
    selected["donor_surface"] = np.interp(selected_s, donor_s, surface)
    selected["donor_surface_circular"] = np.interp(selected_s, donor_s, circular)
    return selected[
        [
            "fold",
            "well_id",
            "node_idx",
            "node_md",
            "node_s",
            "donor_well_id",
            "donor_n",
            "donor_surface",
            "donor_surface_circular",
        ]
    ]


def weighted_huber_local_linear(
    n: Sequence[float],
    response: Sequence[float],
    *,
    bandwidth: float,
    huber_delta: float,
    iterations: int,
    slope_ridge_trace_ratio: float,
) -> tuple[float, float] | None:
    x = np.asarray(n, dtype=float)
    y = np.asarray(response, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2 or np.ptp(x) <= 1.0e-12:
        return None
    base = np.exp(-np.square(x) / (2.0 * bandwidth * bandwidth))
    design = np.column_stack([np.ones(len(x)), x])
    robust = np.ones(len(x), dtype=float)
    beta = np.asarray([np.average(y, weights=base), 0.0], dtype=float)
    for _ in range(int(iterations)):
        weights = base * robust
        normal = design.T @ (weights[:, None] * design)
        ridge = float(np.trace(normal) * slope_ridge_trace_ratio)
        normal[1, 1] += ridge
        rhs = design.T @ (weights * y)
        try:
            beta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return None
        residual = y - design @ beta
        center = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - center)))
        scale = max(scale, 1.0e-9)
        cutoff = huber_delta * scale
        magnitude = np.abs(residual - center)
        robust = np.ones(len(x), dtype=float)
        mask = magnitude > cutoff
        robust[mask] = cutoff / magnitude[mask]
    if not np.isfinite(beta).all():
        return None
    return float(beta[0]), float(beta[1])


def fit_strip_nodes(
    nodes: pd.DataFrame,
    donor_samples: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    minimum = int(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.minimum_unique_donor_wells",
        )
    )
    min_positive = int(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.minimum_positive_side_wells",
        )
    )
    min_negative = int(
        get_nested(
            config,
            "strip_coordinate.pair_eligibility.minimum_negative_side_wells",
        )
    )
    fit_config = get_nested(config, "strip_coordinate.surface_fit")
    by_node = {
        int(node): block.sort_values(
            ["donor_n", "donor_well_id"],
            key=lambda series: series.abs() if series.name == "donor_n" else series,
            kind="mergesort",
        )
        for node, block in donor_samples.groupby("node_idx", sort=True)
    }
    records: list[dict[str, Any]] = []
    for row in nodes.itertuples(index=False):
        block = by_node.get(int(row.node_idx), pd.DataFrame())
        record: dict[str, Any] = {
            "fold": int(row.fold),
            "well_id": str(row.well_id),
            "node_idx": int(row.node_idx),
            "node_md": float(row.node_md),
            "node_s": float(
                block["node_s"].iloc[0] if not block.empty else float("nan")
            ),
            "unique_donors": int(block["donor_well_id"].nunique()) if not block.empty else 0,
            "positive_side_wells": int((block["donor_n"] > 0.0).sum()) if not block.empty else 0,
            "negative_side_wells": int((block["donor_n"] < 0.0).sum()) if not block.empty else 0,
            "status": "no_overlap",
            "fit_valid": False,
            "intercept_raw": float("nan"),
            "slope_raw": float("nan"),
            "intercept_circular_raw": float("nan"),
            "slope_circular_raw": float("nan"),
        }
        if block.empty:
            records.append(record)
            continue
        if record["unique_donors"] < minimum:
            record["status"] = "low_support"
            records.append(record)
            continue
        if (
            record["positive_side_wells"] < min_positive
            or record["negative_side_wells"] < min_negative
        ):
            record["status"] = "one_sided"
            records.append(record)
            continue
        common = {
            "bandwidth": float(fit_config["cross_track_bandwidth_ft"]),
            "huber_delta": float(fit_config["huber_delta"]),
            "iterations": int(fit_config["irls_iterations"]),
            "slope_ridge_trace_ratio": float(fit_config["slope_ridge_trace_ratio"]),
        }
        real = weighted_huber_local_linear(
            block["donor_n"],
            block["donor_surface"],
            **common,
        )
        circular = weighted_huber_local_linear(
            block["donor_n"],
            block["donor_surface_circular"],
            **common,
        )
        if real is None or circular is None:
            record["status"] = "fit_failed"
            records.append(record)
            continue
        record.update(
            {
                "status": "eligible",
                "fit_valid": True,
                "intercept_raw": real[0],
                "slope_raw": real[1],
                "intercept_circular_raw": circular[0],
                "slope_circular_raw": circular[1],
            }
        )
        records.append(record)
    fitted = pd.DataFrame(records)
    window = int(fit_config["along_track_smoothing_nodes"])
    valid = fitted["fit_valid"].to_numpy(dtype=bool)
    segment = np.cumsum(np.r_[True, np.diff(valid.astype(int)) != 0])
    fitted["valid_segment"] = np.where(valid, segment, -1)
    for source, destination in (
        ("intercept_raw", "intercept_smooth"),
        ("slope_raw", "slope_smooth"),
        ("intercept_circular_raw", "intercept_circular_smooth"),
        ("slope_circular_raw", "slope_circular_smooth"),
    ):
        fitted[destination] = np.nan
        for _, index in fitted.loc[valid].groupby("valid_segment", sort=True).groups.items():
            values = fitted.loc[index, source]
            fitted.loc[index, destination] = values.rolling(
                window=window,
                center=True,
                min_periods=1,
            ).median()
    return fitted


def interpolate_strip_to_rows(
    target: pd.DataFrame,
    fitted_nodes: pd.DataFrame,
) -> pd.DataFrame:
    output = target[
        ["fold", "well_id", "row_idx", "MD", "X", "Y", "Z", "TVT_input"]
    ].sort_values("row_idx", kind="mergesort").copy()
    output["strip_surface_raw"] = np.nan
    output["strip_surface_circular_raw"] = np.nan
    output["strip_valid"] = False
    output["strip_edge_of_family"] = False
    output["strip_support_reason"] = "outside_valid_segment"
    valid_nodes = fitted_nodes.loc[fitted_nodes["fit_valid"]].copy()
    for _, block in valid_nodes.groupby("valid_segment", sort=True):
        block = block.sort_values("node_md", kind="mergesort")
        if len(block) < 2:
            continue
        inside = output["MD"].between(
            float(block["node_md"].iloc[0]),
            float(block["node_md"].iloc[-1]),
            inclusive="both",
        )
        output.loc[inside, "strip_surface_raw"] = np.interp(
            output.loc[inside, "MD"],
            block["node_md"],
            block["intercept_smooth"],
        )
        output.loc[inside, "strip_surface_circular_raw"] = np.interp(
            output.loc[inside, "MD"],
            block["node_md"],
            block["intercept_circular_smooth"],
        )
        output.loc[inside, "strip_valid"] = True
        output.loc[inside, "strip_support_reason"] = "eligible"
        row_md = output.loc[inside, "MD"].to_numpy(dtype=float)
        block_md = block["node_md"].to_numpy(dtype=float)
        nearest = np.abs(row_md[:, None] - block_md[None, :]).argmin(axis=1)
        side_minimum = np.minimum(
            block["positive_side_wells"].to_numpy(dtype=int),
            block["negative_side_wells"].to_numpy(dtype=int),
        )
        output.loc[inside, "strip_edge_of_family"] = side_minimum[nearest] <= 1
    invalid = ~output["strip_valid"]
    if invalid.any() and not fitted_nodes.empty:
        node_md = fitted_nodes["node_md"].to_numpy(dtype=float)
        row_md = output.loc[invalid, "MD"].to_numpy(dtype=float)
        nearest = np.abs(row_md[:, None] - node_md[None, :]).argmin(axis=1)
        output.loc[invalid, "strip_support_reason"] = (
            fitted_nodes.iloc[nearest]["status"].to_numpy()
        )
    return output


# %% [markdown]
# ## 6. Prefix calibration and exp226 fallback candidate

# %%
def huber_location(
    values: Sequence[float],
    *,
    delta: float,
    iterations: int,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan")
    location = float(np.median(array))
    for _ in range(int(iterations)):
        residual = array - location
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        scale = max(scale, 1.0e-9)
        cutoff = delta * scale
        magnitude = np.abs(residual)
        weights = np.ones(len(array), dtype=float)
        mask = magnitude > cutoff
        weights[mask] = cutoff / magnitude[mask]
        location = float(np.average(array, weights=weights))
    return location


def calibrate_prefix(
    rows: pd.DataFrame,
    surface_column: str,
    config: Mapping[str, Any],
    calibration_mask: np.ndarray | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    prefix_config = get_nested(config, "strip_coordinate.prefix_calibration")
    base_mask = (
        rows["TVT_input"].notna()
        & rows["Z"].notna()
        & rows[surface_column].notna()
        & rows["strip_valid"].astype(bool)
    ).to_numpy(copy=True)
    if calibration_mask is not None:
        base_mask &= np.asarray(calibration_mask, dtype=bool)
    residual = (
        rows.loc[base_mask, "TVT_input"].to_numpy(float)
        + rows.loc[base_mask, "Z"].to_numpy(float)
        - rows.loc[base_mask, surface_column].to_numpy(float)
    )
    minimum = int(prefix_config["minimum_finite_prefix_rows"])
    offset = (
        huber_location(
            residual,
            delta=float(prefix_config["huber_delta"]),
            iterations=int(prefix_config["iterations"]),
        )
        if len(residual) >= minimum
        else float("nan")
    )
    calibrated = rows[surface_column].astype(float) + offset
    reconstruction = calibrated.to_numpy(float) - rows["Z"].to_numpy(float)
    prefix_rmse = rmse(
        rows.loc[base_mask, "TVT_input"].to_numpy(float),
        reconstruction[base_mask],
    )
    return calibrated, {
        "well_id": str(rows["well_id"].iloc[0]),
        "fold": int(rows["fold"].iloc[0]),
        "surface_column": surface_column,
        "finite_prefix_rows": int(len(residual)),
        "calibration_valid": bool(np.isfinite(offset)),
        "vertical_gauge_offset": float(offset),
        "prefix_reconstruction_rmse": float(prefix_rmse),
    }


def attach_final_candidate(
    rows: pd.DataFrame,
    parent_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real, real_record = calibrate_prefix(rows, "strip_surface_raw", config)
    circular, circular_record = calibrate_prefix(
        rows,
        "strip_surface_circular_raw",
        config,
    )
    enriched = rows.copy()
    if "strip_edge_of_family" not in enriched:
        enriched["strip_edge_of_family"] = False
    enriched["strip_surface_calibrated"] = real
    enriched["strip_surface_circular_calibrated"] = circular
    enriched["strip_tvt_prediction"] = (
        enriched["strip_surface_calibrated"] - enriched["Z"]
    )
    suffix = parent_rows.merge(
        enriched[
            [
                "row_idx",
                "MD",
                "Z",
                "strip_valid",
                "strip_edge_of_family",
                "strip_support_reason",
                "strip_surface_raw",
                "strip_surface_calibrated",
                "strip_tvt_prediction",
            ]
        ],
        on="row_idx",
        how="left",
        validate="one_to_one",
    )
    use_strip = (
        suffix["strip_valid"].fillna(False).astype(bool)
        & suffix["strip_tvt_prediction"].notna()
    )
    suffix["exp390_prediction"] = suffix["tvt_pred"].to_numpy(dtype=float)
    suffix.loc[use_strip, "exp390_prediction"] = suffix.loc[
        use_strip, "strip_tvt_prediction"
    ]
    suffix["candidate_status"] = np.where(use_strip, "parallel_strip", "exp226_fallback")
    suffix["strip_eligible"] = use_strip
    suffix = suffix.rename(columns={"tvt_pred": "exp226_prediction"})
    calibration = pd.DataFrame([real_record, circular_record])
    return enriched, suffix, calibration


def build_well_strip_candidate(
    target: pd.DataFrame,
    parent_rows: pd.DataFrame,
    donors: Mapping[str, DonorTrack],
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    query_well = str(target["well_id"].iloc[0])
    geometry = build_query_geometry(target, parent_rows)
    pair = select_eligible_pairs(query_well, geometry, donors, config)
    selected_ids = (
        pair.loc[pair["selected"], "donor_well_id"].astype(str).tolist()
        if not pair.empty
        else []
    )
    circular_contract = get_nested(
        config,
        "stages.stage1_prefix_rolling_origin.circular_control",
    )
    circular_shift_rows = (
        int(str(circular_contract).split("_")[-2])
        if isinstance(circular_contract, str)
        else 512
    )
    step = float(get_nested(config, "strip_coordinate.query_nodes.grid_md_ft"))
    nodes = build_query_nodes(target, step)
    samples = [
        interpolate_donor_at_nodes(
            nodes,
            donors[well],
            geometry["axis"],
            circular_shift_rows,
        )
        for well in selected_ids
    ]
    donor_samples = (
        pd.concat([item for item in samples if not item.empty], ignore_index=True)
        if any(not item.empty for item in samples)
        else pd.DataFrame(
            columns=[
                "fold",
                "well_id",
                "node_idx",
                "node_md",
                "node_s",
                "donor_well_id",
                "donor_n",
                "donor_surface",
                "donor_surface_circular",
            ]
        )
    )
    fitted = fit_strip_nodes(nodes, donor_samples, config)
    rows = interpolate_strip_to_rows(target, fitted)
    rows, prediction, calibration = attach_final_candidate(rows, parent_rows, config)
    selected_pair = (
        pair.loc[pair["selected"]].copy()
        if not pair.empty
        else pd.DataFrame()
    )
    pair_angle_p95 = (
        float(np.percentile(selected_pair["angle_mismatch_deg"], 95))
        if len(selected_pair)
        else float("nan")
    )
    pair_cross_track_median = (
        float(np.median(selected_pair["median_abs_cross_track_distance_ft"]))
        if len(selected_pair)
        else float("nan")
    )
    pair_overlap_p05 = (
        float(np.percentile(selected_pair["projected_overlap"], 5))
        if len(selected_pair)
        else float("nan")
    )
    real_calibration = calibration.loc[
        calibration["surface_column"].eq("strip_surface_raw")
    ].iloc[0]
    prediction["pair_angle_p95_deg"] = pair_angle_p95
    prediction["pair_cross_track_median_ft"] = pair_cross_track_median
    prediction["pair_overlap_p05"] = pair_overlap_p05
    prediction["prefix_reconstruction_rmse"] = float(
        real_calibration["prefix_reconstruction_rmse"]
    )
    axis = geometry["axis"]
    geometry_summary = pd.DataFrame(
        [
            {
                "fold": int(target["fold"].iloc[0]),
                "well_id": query_well,
                "rows": int(len(target)),
                "score_rows": int(geometry["score_rows"]),
                "axis_es_x": axis.es_x,
                "axis_es_y": axis.es_y,
                "axis_en_x": axis.en_x,
                "axis_en_y": axis.en_y,
                "eligible_pairs": int(len(pair)),
                "selected_pairs": int(len(selected_ids)),
                "strip_eligible_rows": int(prediction["strip_eligible"].sum()),
                "pair_angle_p95_deg": pair_angle_p95,
                "pair_cross_track_median_ft": pair_cross_track_median,
                "pair_overlap_p05": pair_overlap_p05,
                "prefix_reconstruction_rmse": float(
                    real_calibration["prefix_reconstruction_rmse"]
                ),
            }
        ]
    )
    if not pair.empty:
        pair.insert(0, "fold", int(target["fold"].iloc[0]))
    return {
        "geometry_summary": geometry_summary,
        "eligible_pairs": pair,
        "query_node_donors": donor_samples,
        "strip_fit_diagnostics": fitted,
        "query_rows": rows,
        "prediction": prediction,
        "prefix_calibration": calibration,
    }


# %% [markdown]
# ## 7. Stage 0 target-free support/resource gate

# %%
FRAME_SORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "fold_manifest": ("fold", "well_id"),
    "geometry_summary": ("fold", "well_id"),
    "eligible_pairs": ("fold", "query_well_id", "donor_well_id"),
    "query_node_donors": ("fold", "well_id", "node_idx", "donor_well_id"),
    "strip_fit_diagnostics": ("fold", "well_id", "node_idx"),
    "query_rows": ("fold", "well_id", "row_idx"),
    "prediction": ("fold", "well_id", "row_idx"),
    "prefix_calibration": ("fold", "well_id", "surface_column"),
}


def _concat_results(
    results: Sequence[Mapping[str, pd.DataFrame]],
    name: str,
) -> pd.DataFrame:
    values = [result[name] for result in results if not result[name].empty]
    if not values:
        return pd.DataFrame()
    return pd.concat(values, ignore_index=True).sort_values(
        list(FRAME_SORT_COLUMNS[name]),
        kind="mergesort",
    )


def run_fold_target_free(
    fold: int,
    file_by_well: Mapping[str, Path],
    fold_by_well: Mapping[str, int],
    parent_safe: pd.DataFrame,
    selected_targets: set[str] | None,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> list[dict[str, pd.DataFrame]]:
    source_ids = sorted(
        well for well, assigned in fold_by_well.items() if int(assigned) != int(fold)
    )
    target_ids = sorted(
        well
        for well, assigned in fold_by_well.items()
        if int(assigned) == int(fold)
        and (selected_targets is None or well in selected_targets)
    )
    if not target_ids:
        return []
    ledger.record_role_overlap(source_ids, target_ids, fold)
    donors = {
        well: build_donor_track(read_source_well(file_by_well[well], fold, ledger))
        for well in source_ids
    }
    results: list[dict[str, pd.DataFrame]] = []
    for well in target_ids:
        target = read_target_safe_well(file_by_well[well], fold, ledger)
        parent_rows = parent_safe.loc[
            parent_safe["well_id"].eq(well) & parent_safe["fold"].eq(fold)
        ].sort_values("row_idx", kind="mergesort")
        results.append(build_well_strip_candidate(target, parent_rows, donors, config))
    return results


def select_preflight_wells(
    fold_by_well: Mapping[str, int],
    count: int,
    folds: Sequence[int],
) -> set[str]:
    by_fold = {
        int(fold): sorted(
            well for well, assigned in fold_by_well.items() if int(assigned) == int(fold)
        )
        for fold in folds
    }
    selected: list[str] = []
    cursor = {int(fold): 0 for fold in folds}
    while len(selected) < count:
        progressed = False
        for fold in folds:
            index = cursor[int(fold)]
            values = by_fold[int(fold)]
            if index < len(values) and len(selected) < count:
                selected.append(values[index])
                cursor[int(fold)] += 1
                progressed = True
        if not progressed:
            break
    return set(selected)


def evaluate_stage0(
    frames: Mapping[str, pd.DataFrame],
    ledger: RoleReadLedger,
    parent_safe: pd.DataFrame,
    *,
    elapsed_seconds: float,
    full_run: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = get_nested(config, "stages.stage0_target_free")
    prediction = frames["prediction"]
    fitted = frames["strip_fit_diagnostics"]
    pair = frames["eligible_pairs"]
    processed_wells = int(prediction["well_id"].nunique())
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    projected_runtime = (
        float(elapsed_seconds)
        if full_run
        else float(elapsed_seconds) * expected_wells / max(processed_wells, 1)
    )
    eligible_nodes = fitted.loc[fitted["fit_valid"]]
    observed = {
        "input_rows": int(len(parent_safe)),
        "input_wells": int(parent_safe["well_id"].nunique()),
        "input_folds": sorted(parent_safe["fold"].astype(int).unique().tolist()),
        "processed_rows": int(len(prediction)),
        "processed_wells": processed_wells,
        "finite_fallback_prediction_coverage": float(
            prediction["exp226_prediction"].notna().mean()
        ),
        "two_sided_strip_row_coverage": float(prediction["strip_eligible"].mean()),
        "two_sided_strip_well_coverage": float(
            prediction.groupby("well_id", sort=True)["strip_eligible"].any().mean()
        ),
        "eligible_node_unique_donor_p05": (
            float(np.percentile(eligible_nodes["unique_donors"], 5))
            if len(eligible_nodes)
            else 0.0
        ),
        "eligible_pair_angle_p95_deg": (
            float(np.percentile(pair["angle_mismatch_deg"], 95))
            if len(pair)
            else float("inf")
        ),
        "eligible_pair_overlap_p05": (
            float(np.percentile(pair["projected_overlap"], 5))
            if len(pair)
            else 0.0
        ),
        "target_suffix_truth_reads": int(ledger.target_suffix_truth_reads),
        "target_raw_formation_reads": int(ledger.target_raw_formation_reads),
        "target_gr_reads": int(ledger.target_gr_reads),
        "source_valid_overlap": int(ledger.source_valid_overlap),
        "elapsed_seconds": float(elapsed_seconds),
        "projected_full_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": float(peak_rss_gb()),
        "full_run": bool(full_run),
    }
    checks = {
        "rows": observed["input_rows"] == int(gate["expected_rows"]),
        "wells": observed["input_wells"] == int(gate["expected_wells"]),
        "folds": observed["input_folds"] == list(get_nested(config, "validation.expected_folds")),
        "finite_fallback": observed["finite_fallback_prediction_coverage"]
        >= float(gate["finite_fallback_prediction_coverage_min"]),
        "two_sided_rows": observed["two_sided_strip_row_coverage"]
        >= float(gate["two_sided_strip_row_coverage_min"]),
        "two_sided_wells": observed["two_sided_strip_well_coverage"]
        >= float(gate["two_sided_strip_well_coverage_min"]),
        "donor_p05": observed["eligible_node_unique_donor_p05"]
        >= float(gate["eligible_node_unique_donor_p05_min"]),
        "angle_p95": observed["eligible_pair_angle_p95_deg"]
        <= float(gate["eligible_pair_angle_p95_deg_max"]),
        "overlap_p05": observed["eligible_pair_overlap_p05"]
        >= float(gate["eligible_pair_overlap_p05_min"]),
        "truth_reads": observed["target_suffix_truth_reads"]
        <= int(gate["target_suffix_truth_reads_max"]),
        "formation_reads": observed["target_raw_formation_reads"]
        <= int(gate["target_raw_formation_reads_max"]),
        "gr_reads": observed["target_gr_reads"] <= int(gate["target_gr_reads_max"]),
        "role_overlap": observed["source_valid_overlap"]
        <= int(gate["source_valid_overlap_max"]),
        "runtime": observed["projected_full_runtime_seconds"]
        <= float(gate["projected_runtime_seconds_max"]),
        "rss": observed["projected_peak_rss_gb"]
        <= float(gate["projected_peak_rss_gb_max"]),
    }
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}


# %% [markdown]
# ## 8. Stage 1 known-prefix rolling-origin gate

# %%
def build_prefix_rolling_origin(
    query_rows: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage = get_nested(config, "stages.stage1_prefix_rolling_origin")
    heldout_rows = int(stage["heldout_prefix_rows"])
    minimum_rows = int(stage["minimum_known_prefix_rows"])
    records: list[pd.DataFrame] = []
    well_summary: list[dict[str, Any]] = []
    for (fold, well), block in query_rows.groupby(["fold", "well_id"], sort=True):
        block = block.sort_values("row_idx", kind="mergesort").copy()
        prefix = block.loc[block["TVT_input"].notna()].copy()
        if len(prefix) < minimum_rows:
            well_summary.append(
                {
                    "fold": int(fold),
                    "well_id": str(well),
                    "known_prefix_rows": int(len(prefix)),
                    "heldout_rows": 0,
                    "eligible_rows": 0,
                    "eligible_coverage": 0.0,
                    "status": "insufficient_prefix",
                }
            )
            continue
        heldout_index = prefix.index[-heldout_rows:]
        calibration_index = prefix.index[:-heldout_rows]
        mask = block.index.isin(calibration_index)
        real_surface, real_record = calibrate_prefix(
            block,
            "strip_surface_raw",
            config,
            mask,
        )
        circular_surface, circular_record = calibrate_prefix(
            block,
            "strip_surface_circular_raw",
            config,
            mask,
        )
        heldout = block.loc[heldout_index].copy()
        heldout["real_prediction"] = (
            real_surface.loc[heldout_index].to_numpy(float)
            - heldout["Z"].to_numpy(float)
        )
        heldout["circular_prediction"] = (
            circular_surface.loc[heldout_index].to_numpy(float)
            - heldout["Z"].to_numpy(float)
        )
        cut = block.loc[calibration_index[-1]]
        anchor_surface = float(cut["TVT_input"] + cut["Z"])
        heldout["vertical_only_prediction"] = (
            anchor_surface - heldout["Z"].to_numpy(float)
        )
        common = (
            heldout["strip_valid"].astype(bool)
            & heldout[
                [
                    "TVT_input",
                    "real_prediction",
                    "circular_prediction",
                    "vertical_only_prediction",
                ]
            ]
            .notna()
            .all(axis=1)
        )
        scored = heldout.loc[common].copy()
        scored["fold"] = int(fold)
        scored["well_id"] = str(well)
        records.append(scored)
        well_summary.append(
            {
                "fold": int(fold),
                "well_id": str(well),
                "known_prefix_rows": int(len(prefix)),
                "heldout_rows": int(len(heldout)),
                "eligible_rows": int(common.sum()),
                "eligible_coverage": float(common.mean()),
                "real_calibration_rows": int(real_record["finite_prefix_rows"]),
                "circular_calibration_rows": int(circular_record["finite_prefix_rows"]),
                "status": "scored" if common.any() else "no_common_scope",
            }
        )
    scored = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return scored, pd.DataFrame(well_summary)


def evaluate_stage1(
    scored: pd.DataFrame,
    well_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    stage = get_nested(config, "stages.stage1_prefix_rolling_origin")
    if scored.empty:
        observed = {
            "vertical_only_rmse": float("nan"),
            "real_rmse": float("nan"),
            "circular_rmse": float("nan"),
            "heldout_prefix_rmse_gain_ft": float("-inf"),
            "real_minus_circular_control_gain_ft": float("-inf"),
            "positive_folds": 0,
            "eligible_heldout_coverage": 0.0,
        }
        checks = {name: False for name in ("gain", "folds", "coverage", "circular")}
        return {"passed": False, "observed": observed, "checks": checks}, pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scope, block in [("pooled", scored), *[
        (f"fold_{int(fold)}", value)
        for fold, value in scored.groupby("fold", sort=True)
    ]]:
        actual = block["TVT_input"].to_numpy(float)
        vertical = rmse(actual, block["vertical_only_prediction"].to_numpy(float))
        real = rmse(actual, block["real_prediction"].to_numpy(float))
        circular = rmse(actual, block["circular_prediction"].to_numpy(float))
        rows.append(
            {
                "scope": scope,
                "rows": int(len(block)),
                "vertical_only_rmse": vertical,
                "real_strip_rmse": real,
                "circular_strip_rmse": circular,
                "gain_vertical_minus_real": vertical - real,
                "gain_circular_minus_real": circular - real,
            }
        )
    metrics = pd.DataFrame(rows)
    pooled = metrics.loc[metrics["scope"].eq("pooled")].iloc[0]
    folds = metrics.loc[metrics["scope"].str.startswith("fold_")]
    eligible_rows = int(well_summary["eligible_rows"].sum())
    heldout_rows = int(well_summary["heldout_rows"].sum())
    observed = {
        "vertical_only_rmse": float(pooled["vertical_only_rmse"]),
        "real_rmse": float(pooled["real_strip_rmse"]),
        "circular_rmse": float(pooled["circular_strip_rmse"]),
        "heldout_prefix_rmse_gain_ft": float(pooled["gain_vertical_minus_real"]),
        "real_minus_circular_control_gain_ft": float(pooled["gain_circular_minus_real"]),
        "positive_folds": int((folds["gain_vertical_minus_real"] > 0.0).sum()),
        "eligible_heldout_coverage": float(eligible_rows / max(heldout_rows, 1)),
    }
    checks = {
        "gain": observed["heldout_prefix_rmse_gain_ft"]
        >= float(stage["rmse_gain_ft_min"]),
        "folds": observed["positive_folds"] >= int(stage["positive_fold_count_min"]),
        "coverage": observed["eligible_heldout_coverage"]
        >= float(stage["eligible_heldout_coverage_min"]),
        "circular": observed["real_minus_circular_control_gain_ft"]
        >= float(stage["real_minus_circular_control_gain_ft_min"]),
    }
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}, metrics


def freeze_target_free_outputs(
    frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> dict[str, str]:
    hashes = {
        name: frame_content_sha256(frame, FRAME_SORT_COLUMNS[name])
        for name, frame in frames.items()
        if name in FRAME_SORT_COLUMNS and not frame.empty
    }
    hashes["solver_contract"] = sha256_bytes(
        stable_json_bytes(
            {
                "strip_coordinate": get_nested(config, "strip_coordinate"),
                "stages": get_nested(config, "stages"),
                "validation": get_nested(config, "validation"),
            }
        )
    )
    ledger.freeze()
    return hashes


# %% [markdown]
# ## 9. Stage 2 truth-late score and promotion-safety readout

# %%
def load_hidden_like_assignments(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path]:
    filename = str(get_nested(config, "data.hidden_like.filename"))
    candidates = list(get_nested(config, "data.hidden_like.candidates", []))
    path = resolve_candidate_file(candidates, filename)
    frame = pd.read_csv(path, dtype={"well_id": str})
    require_columns(
        frame,
        (
            "well_id",
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        ),
        "hidden-like assignments",
    )
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments contain duplicate well ids")
    return frame, path


def late_join_truth(
    prediction: pd.DataFrame,
    parent_truth: pd.DataFrame,
    hidden_roles: pd.DataFrame,
) -> pd.DataFrame:
    scored = prediction.merge(
        parent_truth[["fold", "well_id", "row_idx", "tvt_true"]],
        on=["fold", "well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    if scored["tvt_true"].isna().any():
        raise ValueError("truth-late join left missing target values")
    scored = scored.merge(
        hidden_roles[
            [
                "well_id",
                "verification_like_spatial_role",
                "verification_like_typewell_purged_role",
            ]
        ],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    scored["hidden_like_spatial"] = scored[
        "verification_like_spatial_role"
    ].eq("valid")
    scored["hidden_like_typewell_purged"] = scored[
        "verification_like_typewell_purged_role"
    ].eq("valid")
    return scored.sort_values(["fold", "well_id", "row_idx"], kind="mergesort")


def metric_record(
    frame: pd.DataFrame,
    scope: str,
    value: str,
) -> dict[str, Any]:
    control = rmse(frame["tvt_true"].to_numpy(float), frame["exp226_prediction"].to_numpy(float))
    candidate = rmse(frame["tvt_true"].to_numpy(float), frame["exp390_prediction"].to_numpy(float))
    return {
        "scope": scope,
        "value": value,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "exp226_rmse": control,
        "exp390_rmse": candidate,
        "gain_exp226_minus_exp390": control - candidate,
        "delta_exp390_minus_exp226": candidate - control,
    }


def build_stage2_metrics(
    scored: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [metric_record(scored, "pooled", "all")]
    records.extend(
        metric_record(block, "fold", str(int(fold)))
        for fold, block in scored.groupby("fold", sort=True)
    )
    scopes = {
        "strip_eligible": scored["strip_eligible"].astype(bool),
        "edge_of_family": scored["strip_edge_of_family"].astype(bool),
        "near_0_250": scored["suffix_offset"].between(0, 249, inclusive="both"),
        "mid_250_1000": scored["suffix_offset"].between(250, 999, inclusive="both"),
        "1000_plus": scored["suffix_offset"].ge(1000),
        "hidden_like_spatial": scored["hidden_like_spatial"].astype(bool),
        "hidden_like_typewell_purged": scored["hidden_like_typewell_purged"].astype(bool),
    }
    for name, mask in scopes.items():
        block = scored.loc[mask]
        records.append(metric_record(block, "report_scope", name))
    for reason, block in scored.groupby("strip_support_reason", dropna=False, sort=True):
        records.append(metric_record(block, "support_reason", str(reason)))
    by_well = pd.DataFrame(
        [
            {
                "well_id": str(well),
                "pair_angle_p95_deg": float(block["pair_angle_p95_deg"].iloc[0]),
                "pair_cross_track_median_ft": float(
                    block["pair_cross_track_median_ft"].iloc[0]
                ),
                "pair_overlap_p05": float(block["pair_overlap_p05"].iloc[0]),
                "prefix_reconstruction_rmse": float(
                    block["prefix_reconstruction_rmse"].iloc[0]
                ),
                "strip_eligible_fraction": float(block["strip_eligible"].mean()),
                "edge_of_family_fraction": float(
                    block["strip_edge_of_family"].mean()
                ),
                **metric_record(block, "well", str(well)),
            }
            for well, block in scored.groupby("well_id", sort=True)
        ]
    )
    return pd.DataFrame(records), by_well


def build_oracle_scope_metrics(
    scored: pd.DataFrame,
    block_sizes: Sequence[int],
) -> pd.DataFrame:
    control_error = np.square(
        scored["tvt_true"].to_numpy(float) - scored["exp226_prediction"].to_numpy(float)
    )
    control_rmse = float(np.sqrt(np.mean(control_error)))
    records: list[dict[str, Any]] = []
    ordered = scored.sort_values(["well_id", "row_idx"], kind="mergesort").copy()
    ordered["control_sse"] = np.square(
        ordered["tvt_true"] - ordered["exp226_prediction"]
    )
    ordered["candidate_sse"] = np.square(
        ordered["tvt_true"] - ordered["exp390_prediction"]
    )
    for size in block_sizes:
        if int(size) == -1:
            ordered["oracle_block"] = ordered["well_id"].astype(str)
            label = "whole_well"
        else:
            position = ordered.groupby("well_id", sort=False).cumcount()
            ordered["oracle_block"] = (
                ordered["well_id"].astype(str)
                + ":"
                + (position // int(size)).astype(str)
            )
            label = f"H{int(size)}" if int(size) > 1 else "row"
        grouped = ordered.groupby("oracle_block", sort=True)[
            ["control_sse", "candidate_sse"]
        ].sum()
        oracle_sse = float(grouped.min(axis=1).sum())
        oracle_rmse = float(np.sqrt(oracle_sse / len(ordered)))
        records.append(
            {
                "scope": label,
                "block_rows": int(size),
                "rows": int(len(ordered)),
                "blocks": int(len(grouped)),
                "exp226_rmse": control_rmse,
                "oracle_rmse": oracle_rmse,
                "oracle_gain_ft": control_rmse - oracle_rmse,
            }
        )
    return pd.DataFrame(records)


def evaluate_stage2(
    metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    support = get_nested(config, "stages.stage2_truth_late.scientific_support")
    safety = get_nested(config, "stages.stage2_truth_late.promotion_safety")
    pooled = metrics.loc[metrics["scope"].eq("pooled")].iloc[0]
    folds = metrics.loc[metrics["scope"].eq("fold")]

    def scope_gain(name: str) -> float:
        row = metrics.loc[
            metrics["scope"].eq("report_scope") & metrics["value"].eq(name)
        ]
        return float(row["gain_exp226_minus_exp390"].iloc[0]) if len(row) else float("-inf")

    delta = by_well["delta_exp390_minus_exp226"].to_numpy(dtype=float)
    observed = {
        "pooled_rmse_gain_vs_exp226_ft": float(pooled["gain_exp226_minus_exp390"]),
        "positive_folds": int((folds["gain_exp226_minus_exp390"] > 0.0).sum()),
        "eligible_rows_rmse_gain_vs_exp226_ft": scope_gain("strip_eligible"),
        "distance_1000_plus_rmse_gain_ft": scope_gain("1000_plus"),
        "near_0_250_regression_ft": -scope_gain("near_0_250"),
        "hidden_like_spatial_regression_ft": -scope_gain("hidden_like_spatial"),
        "hidden_like_typewell_purged_regression_ft": -scope_gain(
            "hidden_like_typewell_purged"
        ),
        "improved_or_equal_wells": int(np.sum(delta <= 0.0)),
        "worse_wells": int(np.sum(delta > 0.0)),
        "by_well_delta_p95_ft": float(np.percentile(delta, 95)),
        "worst_well_delta_ft": float(np.max(delta)),
    }
    scientific_checks = {
        "pooled_gain": observed["pooled_rmse_gain_vs_exp226_ft"]
        >= float(support["pooled_rmse_gain_vs_exp226_ft_min"]),
        "positive_folds": observed["positive_folds"]
        >= int(support["positive_fold_count_min"]),
        "eligible_gain": observed["eligible_rows_rmse_gain_vs_exp226_ft"]
        >= float(support["eligible_rows_rmse_gain_vs_exp226_ft_min"]),
        "distance_1000_plus": observed["distance_1000_plus_rmse_gain_ft"]
        >= float(support["distance_1000_plus_rmse_gain_ft_min"]),
        "near": observed["near_0_250_regression_ft"]
        <= float(support["near_0_250_regression_tolerance_ft"]),
        "hidden_spatial": observed["hidden_like_spatial_regression_ft"]
        <= float(support["hidden_like_spatial_regression_tolerance_ft"]),
        "hidden_typewell": observed["hidden_like_typewell_purged_regression_ft"]
        <= float(support["hidden_like_typewell_purged_regression_tolerance_ft"]),
    }
    safety_checks = {
        "well_majority": observed["improved_or_equal_wells"] >= observed["worse_wells"],
        "well_delta_p95": observed["by_well_delta_p95_ft"]
        <= float(safety["by_well_delta_p95_max_ft"]),
        "worst_well": observed["worst_well_delta_ft"]
        <= float(safety["worst_well_delta_max_ft"]),
    }
    return {
        "scientific_support_passed": bool(all(scientific_checks.values())),
        "promotion_safety_passed": bool(all(safety_checks.values())),
        "passed": bool(all(scientific_checks.values()) and all(safety_checks.values())),
        "observed": observed,
        "scientific_checks": scientific_checks,
        "promotion_safety_checks": safety_checks,
    }


# %% [markdown]
# ## 10. Artifact manifest and execution orchestration

# %%
ARTIFACT_FILENAMES = {
    "fold_manifest": f"{EXPERIMENT_NAME}_fold_manifest.csv",
    "role_read_ledger": f"{EXPERIMENT_NAME}_role_read_ledger.csv",
    "geometry_summary": f"{EXPERIMENT_NAME}_geometry_summary.csv",
    "eligible_pairs": f"{EXPERIMENT_NAME}_eligible_pairs.csv",
    "query_node_donors": f"{EXPERIMENT_NAME}_query_node_donors.parquet",
    "strip_fit_diagnostics": f"{EXPERIMENT_NAME}_strip_fit_diagnostics.csv",
    "prefix_calibration": f"{EXPERIMENT_NAME}_prefix_calibration.csv",
    "prefix_rolling_metrics": f"{EXPERIMENT_NAME}_prefix_rolling_metrics.csv",
    "oof_predictions": f"{EXPERIMENT_NAME}_oof_predictions.csv.gz",
    "candidate_metrics": f"{EXPERIMENT_NAME}_candidate_metrics.csv",
    "scope_metrics": f"{EXPERIMENT_NAME}_scope_metrics.csv",
    "by_well": f"{EXPERIMENT_NAME}_by_well.csv",
    "oracle_scope_metrics": f"{EXPERIMENT_NAME}_oracle_scope_metrics.csv",
}


def persist_frames(
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, frame in frames.items():
        if name not in ARTIFACT_FILENAMES or frame.empty:
            continue
        path = output / ARTIFACT_FILENAMES[name]
        write_table(frame, path)
        record: dict[str, Any] = {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": list(map(str, frame.columns)),
            "schema_sha256": frame_schema_sha256(frame),
            "file_sha256": sha256_file(path),
        }
        if path.name.endswith(".csv.gz"):
            record["decompressed_content_sha256"] = sha256_decompressed_csv(path)
        if name in FRAME_SORT_COLUMNS:
            record["logical_content_sha256"] = frame_content_sha256(
                frame,
                FRAME_SORT_COLUMNS[name],
            )
        records[name] = record
    return records


def build_sha_manifest(records: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": value.get("path"),
                "rows": value.get("rows"),
                "schema_sha256": value.get("schema_sha256"),
                "logical_content_sha256": value.get("logical_content_sha256"),
                "file_sha256": value.get("file_sha256"),
                "decompressed_content_sha256": value.get("decompressed_content_sha256"),
            }
            for name, value in sorted(records.items())
        ]
    )


def run_train() -> dict[str, Any]:
    config = load_config()
    validate_execution_contract(config, require_kaggle_authorization=True)
    train_dir = resolve_train_dir(config)
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    file_by_well = {well_id_from_path(path): path for path in files}
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(file_by_well) != expected_wells:
        raise ValueError(f"expected {expected_wells} horizontal wells, found {len(file_by_well)}")

    parent_safe, parent_path = load_parent_oof(config, include_truth=False)
    folds = list(get_nested(config, "validation.expected_folds"))
    fold_by_well = assign_group_folds(
        sorted(file_by_well),
        int(get_nested(config, "validation.n_folds")),
        int(get_nested(config, "validation.seed")),
    )
    validate_fold_identity(fold_by_well, parent_safe)
    fold_manifest = pd.DataFrame(
        [
            {"well_id": well, "fold": int(fold)}
            for well, fold in sorted(fold_by_well.items())
        ]
    )[["fold", "well_id"]].sort_values(["fold", "well_id"], kind="mergesort")

    mode = str(get_nested(config, "execution.current_mode"))
    selected_targets: set[str] | None = None
    if mode == "stage0_resource_preflight":
        selected_targets = select_preflight_wells(
            fold_by_well,
            int(get_nested(config, "stages.stage0_target_free.resource_audit_wells")),
            folds,
        )
    explicit_max = os.environ.get("EXP390_MAX_WELLS")
    if explicit_max:
        selected_targets = select_preflight_wells(fold_by_well, int(explicit_max), folds)
    full_run = selected_targets is None

    ledger = RoleReadLedger()
    started = time.perf_counter()
    well_results: list[dict[str, pd.DataFrame]] = []
    for fold in folds:
        well_results.extend(
            run_fold_target_free(
                int(fold),
                file_by_well,
                fold_by_well,
                parent_safe,
                selected_targets,
                ledger,
                config,
            )
        )
    if not well_results:
        raise ValueError("no query wells were processed")
    frames = {
        name: _concat_results(well_results, name)
        for name in (
            "geometry_summary",
            "eligible_pairs",
            "query_node_donors",
            "strip_fit_diagnostics",
            "query_rows",
            "prediction",
            "prefix_calibration",
        )
    }
    frames["fold_manifest"] = fold_manifest
    elapsed = time.perf_counter() - started
    stage0 = evaluate_stage0(
        frames,
        ledger,
        parent_safe,
        elapsed_seconds=elapsed,
        full_run=full_run,
        config=config,
    )
    output = artifacts_dir()
    write_json(output / f"{EXPERIMENT_NAME}_stage0_guard.json", stage0)
    persistable = {
        "fold_manifest": frames["fold_manifest"],
        "role_read_ledger": ledger.as_frame(),
        "geometry_summary": frames["geometry_summary"],
        "eligible_pairs": frames["eligible_pairs"],
        "query_node_donors": frames["query_node_donors"],
        "strip_fit_diagnostics": frames["strip_fit_diagnostics"],
        "prefix_calibration": frames["prefix_calibration"],
    }
    records = persist_frames(persistable, output)
    metrics: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "route": "pf_beam",
        "candidate": CANDIDATE_NAME,
        "parent": PARENT_EXPERIMENT,
        "parent_oof_path": str(parent_path),
        "parent_oof_decompressed_sha256": sha256_decompressed_csv(parent_path),
        "full_run": full_run,
        "runtime_seconds": elapsed,
        "stage0": stage0,
        "stage1": None,
        "stage2": None,
        "ledger": ledger.as_dict(),
        "deterministic_anchor": False,
    }
    if not stage0["passed"]:
        metrics["status"] = "stage0_fail_closed"
    elif not full_run:
        metrics["status"] = "stage0_resource_preflight_pass"
        metrics["stage1"] = {"passed": None, "status": "not_opened_in_preflight"}
    else:
        rolling_rows, rolling_wells = build_prefix_rolling_origin(
            frames["query_rows"],
            config,
        )
        stage1, rolling_metrics = evaluate_stage1(rolling_rows, rolling_wells, config)
        metrics["stage1"] = stage1
        persistable.update({"prefix_rolling_metrics": rolling_metrics})
        records.update(
            persist_frames({"prefix_rolling_metrics": rolling_metrics}, output)
        )
        if not stage1["passed"]:
            metrics["status"] = "stage1_fail_closed"
        else:
            frozen_hashes = freeze_target_free_outputs(frames, config, ledger)
            write_json(
                output / f"{EXPERIMENT_NAME}_target_free_sha.json",
                frozen_hashes,
            )
            parent_truth, _ = load_parent_oof(
                config,
                include_truth=True,
                ledger=ledger,
            )
            hidden_roles, hidden_path = load_hidden_like_assignments(config)
            scored = late_join_truth(frames["prediction"], parent_truth, hidden_roles)
            scope_metrics, by_well = build_stage2_metrics(scored)
            oracle = build_oracle_scope_metrics(
                scored,
                list(get_nested(config, "stages.stage2_truth_late.report_only_oracle_blocks_rows")),
            )
            stage2 = evaluate_stage2(scope_metrics, by_well, config)
            metrics["stage2"] = stage2
            metrics["status"] = (
                "stage2_scientific_and_promotion_pass"
                if stage2["passed"]
                else (
                    "stage2_scientific_pass_promotion_fail_closed"
                    if stage2["scientific_support_passed"]
                    else "stage2_scientific_fail_closed"
                )
            )
            metrics["target_free_sha256"] = frozen_hashes
            metrics["hidden_like_path"] = str(hidden_path)
            oof = scored[
                [
                    "fold",
                    "well_id",
                    "row_idx",
                    "suffix_offset",
                    "tvt_true",
                    "exp226_prediction",
                    "exp390_prediction",
                    "strip_eligible",
                    "strip_edge_of_family",
                    "candidate_status",
                    "strip_support_reason",
                ]
            ]
            records.update(
                persist_frames(
                    {
                        "oof_predictions": oof,
                        "candidate_metrics": scope_metrics.loc[
                            scope_metrics["scope"].isin(["pooled", "fold"])
                        ],
                        "scope_metrics": scope_metrics,
                        "by_well": by_well,
                        "oracle_scope_metrics": oracle,
                    },
                    output,
                )
            )
            metrics["ledger"] = ledger.as_dict()

    sha_manifest = build_sha_manifest(records)
    sha_path = output / f"{EXPERIMENT_NAME}_sha_manifest.csv"
    write_table(sha_manifest, sha_path)
    summary = {
        **metrics,
        "artifacts": records,
        "sha_manifest_path": str(sha_path),
        "sha_manifest_sha256": sha256_file(sha_path),
    }
    write_json(output / f"{EXPERIMENT_NAME}_summary.json", summary)
    write_json(output_root() / "metrics.json", metrics)
    return metrics


# %%
CONFIG_PREVIEW = load_config()
validate_execution_contract(CONFIG_PREVIEW, require_kaggle_authorization=False)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG_PREVIEW, "experiment.route"))
print("Parent:", get_nested(CONFIG_PREVIEW, "lineage.parent"))
print("Status:", get_nested(CONFIG_PREVIEW, "experiment.status"))
print("Candidate:", get_nested(CONFIG_PREVIEW, "strip_coordinate.candidate.name"))
print(
    "Execution counts:",
    {
        "scientific_variants": get_nested(CONFIG_PREVIEW, "runtime.scientific_variants"),
        "active_candidates": get_nested(CONFIG_PREVIEW, "runtime.active_candidates"),
        "reporting_folds": get_nested(CONFIG_PREVIEW, "runtime.reporting_folds"),
        "fitted_models": get_nested(CONFIG_PREVIEW, "runtime.fitted_models"),
        "boosters": get_nested(CONFIG_PREVIEW, "runtime.lightgbm_boosters"),
        "parent_control_regeneration": get_nested(
            CONFIG_PREVIEW,
            "runtime.parent_control_regeneration",
        ),
    },
)
print(
    "Authorization:",
    {
        "implementation": get_nested(
            CONFIG_PREVIEW,
            "execution.implementation_authorized",
        ),
        "canonical_notebook": get_nested(
            CONFIG_PREVIEW,
            "execution.canonical_notebook_adoption_authorized",
        ),
        "package": get_nested(CONFIG_PREVIEW, "execution.kaggle_package_authorized"),
        "push": get_nested(CONFIG_PREVIEW, "execution.kaggle_push_authorized"),
        "run": get_nested(CONFIG_PREVIEW, "execution.kaggle_execution_authorized"),
        "inference": get_nested(CONFIG_PREVIEW, "execution.inference_enabled"),
    },
)

if os.environ.get("EXP390_IMPORT_ONLY", "0") != "1":
    RUN_METRICS = run_train()
    print(json.dumps(RUN_METRICS, indent=2, default=str))
