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
# # exp384 fault-aware piecewise stratigraphic vector field
#
# This notebook implements the implementation-only candidate approved on
# 2026-07-24.  It does not regenerate exp383.  A Kaggle run is fail-closed until
# the saved exp383 Stage 0/1 PASS manifest and every pinned logical-content SHA
# are available.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable experiment contract
# 2. Runtime, configuration, SHA, and artifact helpers
# 3. Parent exp383 contract and role-read guard
# 4. Deterministic outer-train fault graph
# 5. Piecewise component field
# 6. Target domain posterior and exp383-compatible path solve
# 7. Stage 0 target-free freeze and integrity gate
# 8. Late truth join and Stage 1 direct readout
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

EXPERIMENT_NAME = "exp384_fault_aware_piecewise_stratigraphic_vector_field"
PARENT_EXPERIMENT = "exp383_all_tvt_stratigraphic_vector_drift_field"
IMPORT_ONLY_ENV = "EXP384_IMPORT_ONLY"

FORMATION_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
SIGNATURE_COLUMNS = tuple(f"signature_{index:02d}" for index in range(29))
SURFACE_COLUMNS = tuple(f"surface_{name}" for name in FORMATION_NAMES)
SURFACE_GRAD_X_COLUMNS = tuple(f"surface_grad_x_{name}" for name in FORMATION_NAMES)
SURFACE_GRAD_Y_COLUMNS = tuple(f"surface_grad_y_{name}" for name in FORMATION_NAMES)
SURFACE_VARIANCE_COLUMNS = tuple(f"surface_variance_{name}" for name in FORMATION_NAMES)
FAULT_SURFACE_RESIDUAL_COLUMNS = tuple(
    f"fault_surface_residual_{name}" for name in FORMATION_NAMES
)
THICKNESS_COLUMNS = tuple(
    f"thickness_{FORMATION_NAMES[index]}_{FORMATION_NAMES[index + 1]}"
    for index in range(len(FORMATION_NAMES) - 1)
)
FAULT_FORMATION_COLUMNS = (
    FAULT_SURFACE_RESIDUAL_COLUMNS
    + THICKNESS_COLUMNS
    + SURFACE_GRAD_X_COLUMNS
    + SURFACE_GRAD_Y_COLUMNS
)
FAULT_STRUCTURAL_COLUMNS = ("smooth_absolute_residual", "smooth_rate_residual")

DONOR_REQUIRED_COLUMNS = (
    "fold",
    "role",
    "well_id",
    "MD",
    "X",
    "Y",
    "Z",
    "window_scale_ft",
    "S_true",
    "tangent_x",
    "tangent_y",
    "rate_true",
    "window_residual_variance",
) + (
    SIGNATURE_COLUMNS
    + SURFACE_COLUMNS
    + SURFACE_GRAD_X_COLUMNS
    + SURFACE_GRAD_Y_COLUMNS
    + SURFACE_VARIANCE_COLUMNS
)
DONOR_GRAPH_REQUIRED_COLUMNS = DONOR_REQUIRED_COLUMNS + FAULT_FORMATION_COLUMNS + (
    "smooth_absolute_residual",
    "smooth_rate_residual",
)
QUERY_REQUIRED_COLUMNS = (
    "fold",
    "role",
    "query_id",
    "well_id",
    "MD",
    "X",
    "Y",
    "Z",
    "TVT_input",
    "tangent_x",
    "tangent_y",
    "base_absolute_s",
    "base_rate",
    "base_absolute_variance",
    "base_rate_variance",
    "base_support_ess",
    "base_unique_wells",
    "base_condition_number",
    "base_surface_variance",
    "surface_variance_reference",
    "fallback_rate",
    "base_path_s",
) + (
    SIGNATURE_COLUMNS
    + SURFACE_COLUMNS
    + SURFACE_GRAD_X_COLUMNS
    + SURFACE_GRAD_Y_COLUMNS
    + SURFACE_VARIANCE_COLUMNS
)

TARGET_ALLOWED_COLUMNS = frozenset({"MD", "X", "Y", "Z", "TVT_input"})
TARGET_FORBIDDEN_COLUMNS = frozenset(
    {"TVT", "GR", *FORMATION_NAMES, "error", "abs_error", "oracle_domain"}
)
PRIMARY_CANDIDATE = "fault_aware_piecewise"


def require_columns(frame: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values in {list(columns)}")


def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_kaggle_authorization: bool,
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp384 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != PARENT_EXPERIMENT:
        raise ValueError("exp384 parent must remain exp383")
    expected_zero = (
        "runtime.fitted_models",
        "runtime.hmm_runs",
        "runtime.pf_runs",
        "runtime.beam_runs",
        "runtime.lightgbm_boosters",
    )
    for key in expected_zero:
        if int(get_nested(config, key, -1)) != 0:
            raise ValueError(f"{key} must remain zero")
    if int(get_nested(config, "runtime.scientific_candidates", -1)) != 1:
        raise ValueError("exactly one scientific candidate is allowed")
    if int(get_nested(config, "runtime.reporting_folds", -1)) != 5:
        raise ValueError("exactly five reporting folds are required")
    if bool(get_nested(config, "runtime.replay_parent_control", True)):
        raise ValueError("saved exp383 control must not be regenerated")
    if not bool(
        get_nested(
            config,
            "method.graph.component_eligibility_requires_fault_boundary_edge",
            False,
        )
    ):
        raise ValueError("piecewise components must require a cut fault-boundary edge")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise ValueError("implementation authorization is not recorded")
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized", False)
    ):
        raise RuntimeError(
            "Kaggle execution is not authorized; exp383 Stage 0/1 PASS and a separate "
            "run approval are required"
        )


# %% [markdown]
# ## 2. Runtime, configuration, SHA, and artifact helpers

# %%
PACKAGE_DIR = Path.cwd()


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    candidates = (start, *start.parents)
    for candidate in candidates:
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
LOCAL_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME


def config_path() -> Path:
    candidates = (
        PACKAGE_DIR / "config.yaml",
        LOCAL_EXPERIMENT_DIR / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("exp384 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    selected = path or config_path()
    value = yaml.safe_load(selected.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{selected} must contain a YAML mapping")
    return value


def output_root() -> Path:
    explicit = os.environ.get("EXP384_OUTPUT_DIR")
    if explicit:
        return Path(explicit)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return LOCAL_EXPERIMENT_DIR


def artifacts_dir() -> Path:
    result = output_root() / "artifacts"
    result.mkdir(parents=True, exist_ok=True)
    return result


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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
    if value is None or pd.isna(value):
        return "<NA>"
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("logical content SHA does not accept non-finite numbers")
        return format(number, ".17g")
    return str(value)


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return sha256_bytes(stable_json_bytes(schema))


def frame_content_sha256(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
) -> str:
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame[columns].copy()
    keys = list(sort_columns or columns)
    missing_keys = sorted(set(keys).difference(canonical.columns))
    if missing_keys:
        raise ValueError(f"SHA sort columns are missing: {missing_keys}")
    if keys:
        canonical = canonical.sort_values(keys, kind="mergesort", na_position="first")
    digest = hashlib.sha256()
    digest.update(stable_json_bytes(columns))
    for row in canonical.itertuples(index=False, name=None):
        digest.update("\x1f".join(_canonical_cell(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_table(path: Path) -> pd.DataFrame:
    suffixes = path.suffixes
    if ".parquet" in suffixes:
        return pd.read_parquet(path)
    if ".csv" in suffixes:
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif ".csv" in path.suffixes:
        compression = "gzip" if path.suffix == ".gz" else None
        frame.to_csv(path, index=False, compression=compression)
    else:
        raise ValueError(f"unsupported table format: {path}")


def resolve_parent_artifact_dir(config: Mapping[str, Any]) -> Path:
    explicit = os.environ.get("EXP384_PARENT_ARTIFACT_DIR")
    configured = get_nested(config, "data.parent_artifacts.candidates", [])
    candidates = ([explicit] if explicit else []) + list(configured)
    for raw in candidates:
        candidate = Path(str(raw))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "saved exp383 artifact directory was not found; set EXP384_PARENT_ARTIFACT_DIR"
    )


def peak_rss_gb() -> float:
    # Linux ru_maxrss is KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)


def robust_center_scale(
    values: np.ndarray,
    *,
    scale_floor: float = 1.0e-9,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    center = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - center), axis=0)
    scale = np.maximum(1.4826 * mad, scale_floor)
    return center, scale


def huber_location(
    values: np.ndarray,
    *,
    delta: float = 1.345,
    iterations: int = 5,
) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    location = float(np.median(finite))
    scale = max(1.4826 * float(np.median(np.abs(finite - location))), 1.0e-9)
    for _ in range(iterations):
        residual = (finite - location) / scale
        weights = np.ones_like(residual)
        outside = np.abs(residual) > delta
        weights[outside] = delta / np.abs(residual[outside])
        location = float(np.sum(weights * finite) / np.sum(weights))
    return location


def huber_centered_rmse(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    location = huber_location(finite)
    residual = finite - location
    return location, float(np.sqrt(np.mean(np.square(residual))))


# %% [markdown]
# ## 3. Parent exp383 contract and role-read guard

# %%
@dataclass
class RoleReadLedger:
    graph_outer_valid_rows: int = 0
    valid_formation_reads: int = 0
    valid_suffix_truth_reads: int = 0
    truth_joined_after_freeze: bool = False

    def record_graph_roles(self, frame: pd.DataFrame) -> None:
        require_columns(frame, ("role",), "graph nodes")
        invalid = ~frame["role"].astype(str).eq("outer_train")
        self.graph_outer_valid_rows += int(invalid.sum())
        if invalid.any():
            raise ValueError("fault graph contains non-outer-train rows")

    def record_target_columns(self, columns: Iterable[str]) -> None:
        found = TARGET_FORBIDDEN_COLUMNS.intersection(map(str, columns))
        formation = set(FORMATION_NAMES).intersection(found)
        self.valid_formation_reads += len(formation)
        if "TVT" in found:
            self.valid_suffix_truth_reads += 1
        if found:
            raise ValueError(f"target-safe input contains forbidden columns: {sorted(found)}")

    def mark_truth_join(self, frozen_hashes: Mapping[str, str]) -> None:
        if not frozen_hashes:
            raise ValueError("truth cannot be joined before target-free content SHA freeze")
        self.truth_joined_after_freeze = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph_outer_valid_rows": self.graph_outer_valid_rows,
            "valid_formation_reads": self.valid_formation_reads,
            "valid_suffix_truth_reads": self.valid_suffix_truth_reads,
            "truth_joined_after_freeze": self.truth_joined_after_freeze,
        }


def parent_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(stable_json_bytes(manifest))


def validate_parent_manifest(
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if manifest.get("experiment") != PARENT_EXPERIMENT:
        raise ValueError("parent manifest is not exp383")
    if manifest.get("stage0", {}).get("passed") is not True:
        raise ValueError("exp383 Stage 0 PASS is required")
    if manifest.get("stage1", {}).get("passed") is not True:
        raise ValueError("exp383 Stage 1 PASS is required")
    parent_validation = manifest.get("validation", {})
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = list(get_nested(config, "validation.expected_folds"))
    if int(parent_validation.get("score_rows", -1)) != expected_rows:
        raise ValueError("exp383 manifest score-row count does not match exp384")
    if int(parent_validation.get("wells", -1)) != expected_wells:
        raise ValueError("exp383 manifest well count does not match exp384")
    if list(parent_validation.get("folds", [])) != expected_folds:
        raise ValueError("exp383 manifest folds do not match exp384")
    pinned = get_nested(
        config,
        "data.parent_artifacts.expected_manifest_logical_sha256",
    )
    if not isinstance(pinned, str) or len(pinned) != 64:
        raise ValueError("exp383 manifest logical SHA is not pinned in config.yaml")
    actual = parent_manifest_sha256(manifest)
    if actual != pinned:
        raise ValueError(f"exp383 manifest SHA mismatch: expected {pinned}, got {actual}")
    required = get_nested(config, "data.parent_artifacts.files", {})
    manifest_artifacts = manifest.get("artifacts", {})
    for logical_name in set(required).difference({"manifest"}):
        record = manifest_artifacts.get(logical_name)
        if not isinstance(record, Mapping):
            raise ValueError(f"parent manifest is missing artifact record {logical_name}")
        sha = record.get("logical_content_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(f"parent artifact {logical_name} has no logical SHA")


def load_pinned_parent_table(
    parent_dir: Path,
    logical_name: str,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    filename = get_nested(config, f"data.parent_artifacts.files.{logical_name}")
    if not filename:
        raise ValueError(f"no filename configured for {logical_name}")
    path = parent_dir / str(filename)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = read_table(path)
    record = manifest["artifacts"][logical_name]
    expected_schema = record.get("schema_sha256")
    expected_content = record.get("logical_content_sha256")
    actual_schema = frame_schema_sha256(frame)
    sort_columns = record.get("logical_sort_columns")
    actual_content = frame_content_sha256(frame, sort_columns=sort_columns)
    if expected_schema and actual_schema != expected_schema:
        raise ValueError(f"{logical_name} schema SHA mismatch")
    if actual_content != expected_content:
        raise ValueError(f"{logical_name} logical content SHA mismatch")
    return frame


def validate_target_safe_query(
    query: pd.DataFrame,
    ledger: RoleReadLedger,
    *,
    allowed_roles: frozenset[str] = frozenset({"outer_valid"}),
) -> None:
    require_columns(query, QUERY_REQUIRED_COLUMNS, "exp383 query field")
    roles = set(query["role"].astype(str).unique())
    if not roles.issubset(allowed_roles):
        raise ValueError(f"query rows have unsupported roles: {sorted(roles)}")
    if query["query_id"].astype(str).duplicated().any():
        raise ValueError("query_id must be globally unique")
    fold_counts = query.groupby("well_id", sort=False)["fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise ValueError("each target well must belong to exactly one fold")
    # The artifact contains derived surface/signature values, never raw target formation columns.
    ledger.record_target_columns(query.columns)


def validate_truth_free_oof_keys(frame: pd.DataFrame, ledger: RoleReadLedger) -> None:
    forbidden = {
        "tvt_true",
        "TVT",
        "error",
        "abs_error",
        "oracle_domain",
        *FORMATION_NAMES,
    }
    found = forbidden.intersection(map(str, frame.columns))
    if found:
        if "tvt_true" in found or "TVT" in found:
            ledger.valid_suffix_truth_reads += 1
        ledger.valid_formation_reads += len(set(FORMATION_NAMES).intersection(found))
        raise ValueError(
            f"pre-freeze OOF key artifact contains forbidden columns: {sorted(found)}"
        )


def validate_oof_key_contract(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    full_run: bool,
) -> None:
    require_columns(
        frame,
        ("fold", "well_id", "row_idx", "MD", "Z", "exp383_prediction"),
        "pre-freeze OOF keys",
    )
    if frame.duplicated(["fold", "well_id", "row_idx"]).any():
        raise ValueError("pre-freeze OOF keys are not unique")
    folds = sorted(int(value) for value in frame["fold"].unique())
    if full_run and folds != list(get_nested(config, "validation.expected_folds")):
        raise ValueError("pre-freeze OOF key folds do not match the fixed five folds")
    if full_run and len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("pre-freeze OOF key row count does not match 3,783,989")
    if full_run and frame["well_id"].nunique() != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("pre-freeze OOF key well count does not match 773")


# %% [markdown]
# ## 4. Deterministic outer-train fault graph

# %%
def canonicalize_graph_nodes(
    donor_nodes: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    require_columns(donor_nodes, DONOR_GRAPH_REQUIRED_COLUMNS, "exp383 donor catalog")
    scale = float(get_nested(config, "method.graph.node_scale_ft"))
    selected = donor_nodes.loc[
        donor_nodes["window_scale_ft"].astype(float).eq(scale)
    ].copy()
    if selected.empty:
        raise ValueError(f"exp383 donor catalog has no {scale:g} ft nodes")
    ledger.record_graph_roles(selected)
    finite_columns = (
        "MD",
        "X",
        "Y",
        "S_true",
        "rate_true",
        *FAULT_FORMATION_COLUMNS,
        *FAULT_STRUCTURAL_COLUMNS,
    )
    require_finite(selected, finite_columns, "fault graph nodes")
    selected["well_id"] = selected["well_id"].astype(str)
    selected = selected.sort_values(
        ["fold", "well_id", "MD", "X", "Y"], kind="mergesort"
    ).reset_index(drop=True)
    selected["node_id"] = np.arange(len(selected), dtype=np.int64)
    return selected


def _nearest_unique_well_edges_for_fold(
    nodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    desired = int(get_nested(config, "method.graph.nearest_unique_wells"))
    max_distance = float(get_nested(config, "method.graph.max_edge_distance_ft"))
    initial_k = int(get_nested(config, "method.graph.initial_neighbor_query", 128))
    maximum_k = int(get_nested(config, "method.graph.maximum_neighbor_query", 2048))
    batch_rows = int(get_nested(config, "method.graph.query_batch_rows", 2048))
    xy = nodes[["X", "Y"]].to_numpy(dtype=float)
    wells = nodes["well_id"].astype(str).to_numpy()
    node_ids = nodes["node_id"].to_numpy(dtype=np.int64)
    md = nodes["MD"].to_numpy(dtype=float)
    if len(nodes) < 2:
        return pd.DataFrame(columns=["fold", "node_u", "node_v", "distance_ft"])
    query_k = min(len(nodes), max(initial_k, desired + 1), maximum_k)
    neighbor_model = NearestNeighbors(
        n_neighbors=query_k,
        algorithm="auto",
        metric="euclidean",
        n_jobs=1,
    ).fit(xy)
    edge_distance: dict[tuple[int, int], float] = {}
    for start in range(0, len(nodes), batch_rows):
        stop = min(start + batch_rows, len(nodes))
        distances, indices = neighbor_model.kneighbors(xy[start:stop])
        for local_row, (distance_row, index_row) in enumerate(
            zip(distances, indices, strict=True)
        ):
            source_pos = start + local_row
            source_well = wells[source_pos]
            def canonical_candidates(
                candidate_distances: np.ndarray,
                candidate_indices: np.ndarray,
                *,
                source_position: int = source_pos,
                source_well_id: str = source_well,
            ) -> list[tuple[float, str, float, int]]:
                candidates: list[tuple[float, str, float, int]] = []
                for distance, target_pos in zip(
                    candidate_distances, candidate_indices, strict=True
                ):
                    target_pos = int(target_pos)
                    if (
                        target_pos == source_position
                        or wells[target_pos] == source_well_id
                    ):
                        continue
                    if float(distance) > max_distance:
                        continue
                    candidates.append(
                        (
                            float(distance),
                            wells[target_pos],
                            float(md[target_pos]),
                            int(node_ids[target_pos]),
                        )
                    )
                return sorted(candidates)

            candidates = canonical_candidates(distance_row, index_row)
            if (
                len({candidate[1] for candidate in candidates}) < desired
                and query_k < len(nodes)
                and float(distance_row[-1]) <= max_distance
            ):
                # Rare exactness fallback: the first query can be dominated by dense
                # nodes from a few wells. Query this row against all nodes so the
                # "nearest 12 unique wells" contract is exact, not approximate.
                exact_distance, exact_index = neighbor_model.kneighbors(
                    xy[source_pos : source_pos + 1],
                    n_neighbors=len(nodes),
                )
                candidates = canonical_candidates(exact_distance[0], exact_index[0])
            used_wells: set[str] = set()
            for distance, target_well, _, target_id in candidates:
                if target_well in used_wells:
                    continue
                used_wells.add(target_well)
                source_id = int(node_ids[source_pos])
                key = (min(source_id, target_id), max(source_id, target_id))
                previous = edge_distance.get(key)
                if previous is None or distance < previous:
                    edge_distance[key] = distance
                if len(used_wells) == desired:
                    break
    fold = int(nodes["fold"].iloc[0])
    records = [
        {
            "fold": fold,
            "node_u": node_u,
            "node_v": node_v,
            "distance_ft": distance,
        }
        for (node_u, node_v), distance in edge_distance.items()
    ]
    return pd.DataFrame(records).sort_values(
        ["fold", "node_u", "node_v"], kind="mergesort"
    ).reset_index(drop=True)


def build_fault_graph(
    graph_nodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    require_columns(
        graph_nodes,
        ("node_id", *FAULT_FORMATION_COLUMNS, *FAULT_STRUCTURAL_COLUMNS),
        "graph nodes",
    )
    all_edges: list[pd.DataFrame] = []
    standardized: dict[int, pd.DataFrame] = {}
    scale_floor = float(get_nested(config, "method.graph.robust_scale_floor", 1.0e-9))
    for fold, fold_nodes in graph_nodes.groupby("fold", sort=True):
        fold_nodes = fold_nodes.copy()
        matrix = fold_nodes[list(FAULT_FORMATION_COLUMNS + FAULT_STRUCTURAL_COLUMNS)].to_numpy(
            dtype=float
        )
        center, scale = robust_center_scale(matrix, scale_floor=scale_floor)
        z = (matrix - center) / scale
        z_columns = [f"z_{column}" for column in FAULT_FORMATION_COLUMNS + FAULT_STRUCTURAL_COLUMNS]
        z_frame = pd.DataFrame(z, columns=z_columns, index=fold_nodes["node_id"].to_numpy())
        standardized[int(fold)] = z_frame
        all_edges.append(_nearest_unique_well_edges_for_fold(fold_nodes, config))
    if not all_edges:
        raise ValueError("fault graph has no folds")
    edges = pd.concat(all_edges, ignore_index=True)
    if edges.empty:
        raise ValueError("fault graph has no cross-well edges")
    formation_threshold = float(
        get_nested(config, "method.graph.formation_jump_squared_mean_min")
    )
    structural_threshold = float(
        get_nested(config, "method.graph.structural_jump_abs_z_min")
    )
    formation_scores: list[float] = []
    structural_scores: list[float] = []
    for row in edges.itertuples(index=False):
        z_frame = standardized[int(row.fold)]
        left = z_frame.loc[int(row.node_u)].to_numpy(dtype=float)
        right = z_frame.loc[int(row.node_v)].to_numpy(dtype=float)
        difference = np.abs(left - right)
        formation = difference[: len(FAULT_FORMATION_COLUMNS)]
        structural = difference[len(FAULT_FORMATION_COLUMNS) :]
        formation_scores.append(float(np.mean(np.square(np.clip(formation, 0.0, 6.0)))))
        structural_scores.append(float(np.max(structural)))
    edges["formation_jump_score"] = formation_scores
    edges["structural_jump_score"] = structural_scores
    edges["formation_jump"] = edges["formation_jump_score"].ge(formation_threshold)
    edges["structural_jump"] = edges["structural_jump_score"].ge(structural_threshold)
    edges["cut"] = edges["formation_jump"] & edges["structural_jump"]
    return edges.sort_values(["fold", "node_u", "node_v"], kind="mergesort").reset_index(
        drop=True
    )


class StableUnionFind:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        low, high = sorted((root_left, root_right))
        self.parent[high] = low


def assign_fault_components(
    graph_nodes: pd.DataFrame,
    edges: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_wells = int(get_nested(config, "method.graph.min_component_unique_wells"))
    assignments: list[pd.DataFrame] = []
    component_records: list[dict[str, Any]] = []
    for fold, fold_nodes in graph_nodes.groupby("fold", sort=True):
        fold_nodes = fold_nodes.sort_values(["well_id", "MD", "node_id"], kind="mergesort")
        node_ids = fold_nodes["node_id"].astype(int).tolist()
        union_find = StableUnionFind(node_ids)
        fold_edges = edges.loc[edges["fold"].eq(fold) & ~edges["cut"]]
        for edge in fold_edges.itertuples(index=False):
            union_find.union(int(edge.node_u), int(edge.node_v))
        members: dict[int, list[int]] = {}
        for node_id in node_ids:
            members.setdefault(union_find.find(node_id), []).append(node_id)
        node_lookup = fold_nodes.set_index("node_id")
        ordered_roots = sorted(
            members,
            key=lambda root: min(
                (str(node_lookup.loc[node_id, "well_id"]), float(node_lookup.loc[node_id, "MD"]))
                for node_id in members[root]
            ),
        )
        root_to_component = {
            root: f"f{int(fold):02d}_c{rank:06d}"
            for rank, root in enumerate(ordered_roots)
        }
        fold_assignment = fold_nodes[["fold", "node_id", "well_id", "MD"]].copy()
        fold_assignment["component_id"] = [
            root_to_component[union_find.find(int(node_id))]
            for node_id in fold_assignment["node_id"]
        ]
        well_counts = fold_assignment.groupby("component_id", sort=True)["well_id"].nunique()
        fold_assignment["component_unique_wells"] = fold_assignment["component_id"].map(
            well_counts
        )
        node_to_component = fold_assignment.set_index("node_id")["component_id"].to_dict()
        boundary_edges: dict[str, set[tuple[int, int]]] = {
            component_id: set() for component_id in fold_assignment["component_id"].unique()
        }
        cut_edges = edges.loc[edges["fold"].eq(fold) & edges["cut"]]
        for edge in cut_edges.itertuples(index=False):
            key = (int(edge.node_u), int(edge.node_v))
            left_component = node_to_component[int(edge.node_u)]
            right_component = node_to_component[int(edge.node_v)]
            if left_component != right_component:
                boundary_edges[left_component].add(key)
                boundary_edges[right_component].add(key)
        boundary_counts = {
            component_id: len(values) for component_id, values in boundary_edges.items()
        }
        fold_assignment["component_fault_boundary_edges"] = fold_assignment[
            "component_id"
        ].map(boundary_counts)
        fold_assignment["component_eligible"] = fold_assignment[
            "component_unique_wells"
        ].ge(minimum_wells) & fold_assignment["component_fault_boundary_edges"].gt(0)
        assignments.append(fold_assignment)
        for component_id, group in fold_assignment.groupby("component_id", sort=True):
            component_records.append(
                {
                    "fold": int(fold),
                    "component_id": component_id,
                    "node_count": int(len(group)),
                    "unique_wells": int(group["well_id"].nunique()),
                    "fault_boundary_edges": int(
                        group["component_fault_boundary_edges"].iloc[0]
                    ),
                    "eligible": bool(group["component_eligible"].iloc[0]),
                    "minimum_well_id": str(group["well_id"].min()),
                    "minimum_md": float(group["MD"].min()),
                }
            )
    assignment_frame = pd.concat(assignments, ignore_index=True).sort_values(
        ["fold", "node_id"], kind="mergesort"
    )
    component_frame = pd.DataFrame(component_records).sort_values(
        ["fold", "component_id"], kind="mergesort"
    )
    return assignment_frame.reset_index(drop=True), component_frame.reset_index(drop=True)


# %% [markdown]
# ## 5. Piecewise component field

# %%
@dataclass(frozen=True)
class ComponentStats:
    fold: int
    component_id: str
    node_count: int
    unique_wells: int
    x_center: float
    y_center: float
    xy_bandwidth_ft: float
    signature_center: np.ndarray
    signature_precision: np.ndarray
    surface_uncertainty_reference: float


def build_component_catalog(
    graph_nodes: pd.DataFrame,
    assignments: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[str, ComponentStats], pd.DataFrame]:
    joined = graph_nodes.merge(
        assignments[["node_id", "component_id", "component_eligible"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    minimum_query_wells = int(
        get_nested(config, "method.component_field.min_query_component_unique_wells")
    )
    bandwidth_min, bandwidth_max = map(
        float,
        get_nested(config, "method.posterior.component_bandwidth_clip_ft", [500.0, 4000.0]),
    )
    ridge_ratio = float(
        get_nested(
            config,
            "method.posterior.signature_covariance_ridge_ratio",
            1.0e-6,
        )
    )
    catalog: dict[str, ComponentStats] = {}
    records: list[dict[str, Any]] = []
    for component_id, group in joined.loc[joined["component_eligible"]].groupby(
        "component_id", sort=True
    ):
        unique_wells = int(group["well_id"].nunique())
        if unique_wells < minimum_query_wells:
            continue
        signature = group[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(signature).all():
            continue
        center = np.median(signature, axis=0)
        centered = signature - center
        covariance = centered.T @ centered / max(len(signature) - 1, 1)
        trace = float(np.trace(covariance))
        ridge = ridge_ratio * max(trace / len(SIGNATURE_COLUMNS), 1.0)
        covariance = covariance + ridge * np.eye(len(SIGNATURE_COLUMNS))
        precision = np.linalg.pinv(covariance, hermitian=True)
        x_center = float(np.median(group["X"]))
        y_center = float(np.median(group["Y"]))
        radius = np.sqrt(
            np.square(group["X"].to_numpy(dtype=float) - x_center)
            + np.square(group["Y"].to_numpy(dtype=float) - y_center)
        )
        bandwidth = float(np.clip(np.percentile(radius, 75), bandwidth_min, bandwidth_max))
        uncertainty = group[list(SURFACE_VARIANCE_COLUMNS)].to_numpy(dtype=float)
        uncertainty_ref = float(np.nanmedian(uncertainty))
        if not math.isfinite(uncertainty_ref) or uncertainty_ref <= 0:
            uncertainty_ref = 1.0
        stats = ComponentStats(
            fold=int(group["fold"].iloc[0]),
            component_id=str(component_id),
            node_count=int(len(group)),
            unique_wells=unique_wells,
            x_center=x_center,
            y_center=y_center,
            xy_bandwidth_ft=bandwidth,
            signature_center=center,
            signature_precision=precision,
            surface_uncertainty_reference=uncertainty_ref,
        )
        catalog[str(component_id)] = stats
        records.append(
            {
                "fold": stats.fold,
                "component_id": stats.component_id,
                "node_count": stats.node_count,
                "unique_wells": stats.unique_wells,
                "x_center": stats.x_center,
                "y_center": stats.y_center,
                "xy_bandwidth_ft": stats.xy_bandwidth_ft,
                "surface_uncertainty_reference": stats.surface_uncertainty_reference,
                "signature_center_json": json.dumps(stats.signature_center.tolist()),
                "signature_precision_json": json.dumps(stats.signature_precision.tolist()),
            }
        )
    return catalog, pd.DataFrame(records)


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    square = float(np.sum(np.square(weights)))
    return total * total / square if square > 0 else 0.0


def _select_component_donors(
    donors: pd.DataFrame,
    query: pd.Series,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    maximum_wells = int(get_nested(config, "method.component_field.unique_wells", 32))
    maximum_per_well = int(get_nested(config, "method.component_field.max_nodes_per_well", 4))
    maximum_nodes = int(get_nested(config, "method.component_field.max_nodes", 128))
    work = donors.copy()
    work["distance_ft"] = np.sqrt(
        np.square(work["X"].to_numpy(dtype=float) - float(query["X"]))
        + np.square(work["Y"].to_numpy(dtype=float) - float(query["Y"]))
    )
    work = work.sort_values(["distance_ft", "well_id", "MD", "node_id"], kind="mergesort")
    selected: list[int] = []
    well_counts: dict[str, int] = {}
    for index, row in work.iterrows():
        well = str(row["well_id"])
        if well not in well_counts and len(well_counts) >= maximum_wells:
            continue
        if well_counts.get(well, 0) >= maximum_per_well:
            continue
        selected.append(index)
        well_counts[well] = well_counts.get(well, 0) + 1
        if len(selected) == maximum_nodes:
            break
    return work.loc[selected].copy()


def _weighted_plane(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    ridge_trace_ratio: float,
) -> tuple[np.ndarray, float, float]:
    weighted_design = design * weights[:, None]
    normal = design.T @ weighted_design
    trace = float(np.trace(normal))
    ridge = ridge_trace_ratio * max(trace, 1.0)
    penalty = np.diag([ridge * 1.0e-6, ridge, ridge])
    regularized = normal + penalty
    rhs = design.T @ (weights * response)
    coefficients = np.linalg.solve(regularized, rhs)
    residual = response - design @ coefficients
    variance = float(np.sum(weights * np.square(residual)) / max(np.sum(weights), 1.0e-12))
    condition = float(np.linalg.cond(regularized))
    return coefficients, max(variance, 1.0e-9), condition


def fit_component_field(
    donors: pd.DataFrame,
    query: pd.Series,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected = _select_component_donors(donors, query, config)
    minimum_wells = int(
        get_nested(config, "method.component_field.min_query_component_unique_wells")
    )
    if selected["well_id"].nunique() < minimum_wells:
        return None
    distances = selected["distance_ft"].to_numpy(dtype=float)
    unique_distance = (
        selected.groupby("well_id", sort=True)["distance_ft"].min().sort_values().to_numpy()
    )
    bandwidth_index = min(23, len(unique_distance) - 1)
    bandwidth = float(np.clip(unique_distance[bandwidth_index], 500.0, 4000.0))
    query_signature = query[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
    donor_signature = selected[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
    signature_delta = np.clip(donor_signature - query_signature[None, :], -3.0, 3.0)
    formation_weight = np.exp(-0.5 * np.mean(np.square(signature_delta), axis=1))
    xy_weight = np.exp(-0.5 * np.square(distances / max(bandwidth, 1.0e-9)))
    residual_variance = np.maximum(
        selected["window_residual_variance"].to_numpy(dtype=float), 1.0e-4
    )
    weights = xy_weight * formation_weight / residual_variance
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return None
    dx = selected["X"].to_numpy(dtype=float) - float(query["X"])
    dy = selected["Y"].to_numpy(dtype=float) - float(query["Y"])
    design = np.column_stack([np.ones(len(selected)), dx, dy])
    ridge_ratio = float(get_nested(config, "method.component_field.ridge_trace_ratio", 1.0e-4))
    absolute_candidates: list[float] = []
    rate_candidates: list[float] = []
    absolute_variances: list[float] = []
    rate_variances: list[float] = []
    conditions: list[float] = []
    for _name, surface_column, grad_x_column, grad_y_column in zip(
        FORMATION_NAMES,
        SURFACE_COLUMNS,
        SURFACE_GRAD_X_COLUMNS,
        SURFACE_GRAD_Y_COLUMNS,
        strict=True,
    ):
        relative_s = (
            selected["S_true"].to_numpy(dtype=float)
            - selected[surface_column].to_numpy(dtype=float)
        )
        coefficients, abs_variance, condition = _weighted_plane(
            design, relative_s, weights, ridge_ratio
        )
        absolute_s = float(query[surface_column]) + float(coefficients[0])
        gradient_x = float(query[grad_x_column]) + float(coefficients[1])
        gradient_y = float(query[grad_y_column]) + float(coefficients[2])
        rate = float(query["tangent_x"]) * gradient_x + float(query["tangent_y"]) * gradient_y
        donor_rate_prediction = (
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
        rate_residual = selected["rate_true"].to_numpy(dtype=float) - donor_rate_prediction
        rate_variance = float(
            np.sum(weights * np.square(rate_residual)) / max(np.sum(weights), 1.0e-12)
        )
        absolute_candidates.append(absolute_s)
        rate_candidates.append(rate)
        absolute_variances.append(max(abs_variance, 1.0e-9))
        rate_variances.append(max(rate_variance, 1.0e-9))
        conditions.append(condition)
    if not np.isfinite(absolute_candidates + rate_candidates).all():
        return None
    surface_variance = float(
        np.mean(query[list(SURFACE_VARIANCE_COLUMNS)].to_numpy(dtype=float))
    )
    return {
        "field_absolute_s": float(np.median(absolute_candidates)),
        "field_rate": float(np.median(rate_candidates)),
        "absolute_variance": float(np.median(absolute_variances) + surface_variance),
        "rate_variance": float(np.median(rate_variances)),
        "support_ess": _effective_sample_size(weights),
        "unique_wells": int(selected["well_id"].nunique()),
        "condition_number": float(np.max(conditions)),
        "surface_variance": surface_variance,
        "selected_nodes": int(len(selected)),
        "bandwidth_ft": bandwidth,
    }


def generate_component_fields(
    query: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    assignments: pd.DataFrame,
    catalog: Mapping[str, ComponentStats],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    limit = int(get_nested(config, "method.component_field.query_component_limit"))
    node_components = graph_nodes.merge(
        assignments[["node_id", "component_id", "component_eligible"]],
        on="node_id",
        how="left",
        validate="one_to_one",
    )
    donor_groups = {
        component_id: group.copy()
        for component_id, group in node_components.loc[
            node_components["component_eligible"]
        ].groupby("component_id", sort=True)
        if component_id in catalog
    }
    by_fold: dict[int, list[ComponentStats]] = {}
    for stats in catalog.values():
        by_fold.setdefault(stats.fold, []).append(stats)
    for values in by_fold.values():
        values.sort(key=lambda item: item.component_id)
    records: list[dict[str, Any]] = []
    ordered_query = query.sort_values(
        ["fold", "well_id", "MD", "query_id"], kind="mergesort"
    )
    for row in ordered_query.itertuples(index=False):
        series = pd.Series(row._asdict())
        candidates = by_fold.get(int(series["fold"]), [])
        candidates = sorted(
            candidates,
            key=lambda stats: (
                math.hypot(
                    float(series["X"]) - stats.x_center,
                    float(series["Y"]) - stats.y_center,
                ),
                stats.component_id,
            ),
        )[:limit]
        for stats in candidates:
            fitted = fit_component_field(donor_groups[stats.component_id], series, config)
            if fitted is None:
                continue
            signature_delta = (
                series[list(SIGNATURE_COLUMNS)].to_numpy(dtype=float)
                - stats.signature_center
            )
            mahalanobis = float(signature_delta @ stats.signature_precision @ signature_delta)
            xy_distance = math.hypot(
                float(series["X"]) - stats.x_center,
                float(series["Y"]) - stats.y_center,
            )
            surface_penalty = math.log1p(
                max(float(fitted["surface_variance"]), 0.0)
                / max(stats.surface_uncertainty_reference, 1.0e-9)
            )
            target_free_log_weight = (
                -0.5 * mahalanobis
                -0.5 * (xy_distance / stats.xy_bandwidth_ft) ** 2
                -0.5 * surface_penalty
            )
            records.append(
                {
                    "fold": int(series["fold"]),
                    "query_id": str(series["query_id"]),
                    "well_id": str(series["well_id"]),
                    "MD": float(series["MD"]),
                    "component_id": stats.component_id,
                    "signature_distance_squared": mahalanobis,
                    "component_xy_distance_ft": xy_distance,
                    "target_free_log_weight": target_free_log_weight,
                    **fitted,
                }
            )
    columns = [
        "fold",
        "query_id",
        "well_id",
        "MD",
        "component_id",
        "signature_distance_squared",
        "component_xy_distance_ft",
        "target_free_log_weight",
        "field_absolute_s",
        "field_rate",
        "absolute_variance",
        "rate_variance",
        "support_ess",
        "unique_wells",
        "condition_number",
        "surface_variance",
        "selected_nodes",
        "bandwidth_ft",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        ["fold", "well_id", "MD", "component_id"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 6. Target domain posterior and exp383-compatible path solve

# %%
def build_prefix_likelihood(
    query: pd.DataFrame,
    component_fields: pd.DataFrame,
    *,
    prefix_scale_ft: float,
) -> pd.DataFrame:
    require_columns(query, ("query_id", "well_id", "Z", "TVT_input"), "query")
    if not math.isfinite(prefix_scale_ft) or prefix_scale_ft <= 0:
        raise ValueError("outer-train LOO prefix RMSE p50 must be finite and positive")
    known = query.loc[
        np.isfinite(query["TVT_input"].to_numpy(dtype=float)),
        ["query_id", "well_id", "Z", "TVT_input"],
    ].copy()
    known["S_input"] = known["TVT_input"].astype(float) + known["Z"].astype(float)
    joined = component_fields.merge(
        known[["query_id", "S_input"]],
        on="query_id",
        how="inner",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    for (well_id, component_id), group in joined.groupby(
        ["well_id", "component_id"], sort=True
    ):
        residual = (
            group["S_input"].to_numpy(dtype=float)
            - group["field_absolute_s"].to_numpy(dtype=float)
        )
        bias, rmse = huber_centered_rmse(residual)
        records.append(
            {
                "well_id": str(well_id),
                "component_id": str(component_id),
                "prefix_rows": int(len(group)),
                "prefix_bias_ft": bias,
                "prefix_huber_rmse_ft": rmse,
                "prefix_log_likelihood": -0.5 * (rmse / prefix_scale_ft) ** 2,
            }
        )
    columns = [
        "well_id",
        "component_id",
        "prefix_rows",
        "prefix_bias_ft",
        "prefix_huber_rmse_ft",
        "prefix_log_likelihood",
    ]
    return pd.DataFrame(records, columns=columns)


def build_domain_posterior(
    query: pd.DataFrame,
    component_fields: pd.DataFrame,
    prefix_likelihood: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    base_floor = float(get_nested(config, "method.posterior.smooth_base_mass_floor"))
    temperature = float(get_nested(config, "method.posterior.temperature"))
    if not (0.0 < base_floor < 1.0):
        raise ValueError("smooth base mass floor must be in (0, 1)")
    if temperature != 1.0:
        raise ValueError("posterior temperature is frozen at 1.0")
    prefix_columns = ["well_id", "component_id", "prefix_log_likelihood"]
    prefix_frame = (
        prefix_likelihood[prefix_columns]
        if not prefix_likelihood.empty
        else pd.DataFrame(columns=prefix_columns)
    )
    weighted = component_fields.merge(
        prefix_frame,
        on=["well_id", "component_id"],
        how="left",
        validate="many_to_one",
    )
    weighted["prefix_log_likelihood"] = weighted["prefix_log_likelihood"].fillna(0.0)
    weighted["log_weight"] = (
        weighted["target_free_log_weight"] + weighted["prefix_log_likelihood"]
    )
    field_groups = {
        str(query_id): group.sort_values("component_id", kind="mergesort")
        for query_id, group in weighted.groupby("query_id", sort=False)
    }
    records: list[dict[str, Any]] = []
    ordered_query = query.sort_values(
        ["fold", "well_id", "MD", "query_id"], kind="mergesort"
    )
    for row in ordered_query.itertuples(index=False):
        query_id = str(row.query_id)
        group = field_groups.get(query_id)
        if group is None or group.empty:
            records.append(
                {
                    "fold": int(row.fold),
                    "query_id": query_id,
                    "well_id": str(row.well_id),
                    "MD": float(row.MD),
                    "domain": "base",
                    "component_id": "",
                    "posterior_mass": 1.0,
                    "eligible": False,
                }
            )
            continue
        log_weight = group["log_weight"].to_numpy(dtype=float)
        finite = np.isfinite(log_weight)
        if not finite.any():
            records.append(
                {
                    "fold": int(row.fold),
                    "query_id": query_id,
                    "well_id": str(row.well_id),
                    "MD": float(row.MD),
                    "domain": "base",
                    "component_id": "",
                    "posterior_mass": 1.0,
                    "eligible": False,
                }
            )
            continue
        group = group.loc[finite].copy()
        log_weight = log_weight[finite]
        shifted = log_weight - float(np.max(log_weight))
        component_mass = np.exp(shifted)
        component_mass = (1.0 - base_floor) * component_mass / component_mass.sum()
        records.append(
            {
                "fold": int(row.fold),
                "query_id": query_id,
                "well_id": str(row.well_id),
                "MD": float(row.MD),
                "domain": "base",
                "component_id": "",
                "posterior_mass": base_floor,
                "eligible": True,
            }
        )
        for component_id, mass in zip(
            group["component_id"], component_mass, strict=True
        ):
            records.append(
                {
                    "fold": int(row.fold),
                    "query_id": query_id,
                    "well_id": str(row.well_id),
                    "MD": float(row.MD),
                    "domain": "component",
                    "component_id": str(component_id),
                    "posterior_mass": float(mass),
                    "eligible": True,
                }
            )
    posterior = pd.DataFrame(records)
    sums = posterior.groupby("query_id", sort=False)["posterior_mass"].sum()
    if float(np.max(np.abs(sums.to_numpy(dtype=float) - 1.0))) > 1.0e-12:
        raise RuntimeError("domain posterior does not sum to one")
    return posterior.sort_values(
        ["fold", "well_id", "MD", "domain", "component_id"], kind="mergesort"
    ).reset_index(drop=True)


def marginalize_fields(
    query: pd.DataFrame,
    component_fields: pd.DataFrame,
    posterior: pd.DataFrame,
) -> pd.DataFrame:
    component_lookup = component_fields.set_index(["query_id", "component_id"])
    records: list[dict[str, Any]] = []
    for query_row in query.sort_values(["fold", "well_id", "MD"], kind="mergesort").itertuples(
        index=False
    ):
        query_id = str(query_row.query_id)
        weights = posterior.loc[posterior["query_id"].eq(query_id)]
        eligible = bool(weights["eligible"].max())
        if not eligible:
            records.append(
                {
                    "fold": int(query_row.fold),
                    "query_id": query_id,
                    "well_id": str(query_row.well_id),
                    "MD": float(query_row.MD),
                    "eligible": False,
                    "mixed_absolute_s": float(query_row.base_absolute_s),
                    "mixed_rate": float(query_row.base_rate),
                    "mixed_absolute_variance": float(query_row.base_absolute_variance),
                    "mixed_rate_variance": float(query_row.base_rate_variance),
                    "mixed_support_ess": float(query_row.base_support_ess),
                    "mixed_unique_wells": float(query_row.base_unique_wells),
                    "mixed_condition_number": float(query_row.base_condition_number),
                    "mixed_surface_variance": float(query_row.base_surface_variance),
                    "component_posterior_mass": 0.0,
                    "dominant_component": "base",
                    "posterior_entropy": 0.0,
                }
            )
            continue
        means_abs: list[float] = []
        means_rate: list[float] = []
        vars_abs: list[float] = []
        vars_rate: list[float] = []
        support: list[float] = []
        unique_wells: list[float] = []
        condition: list[float] = []
        surface_variance: list[float] = []
        masses: list[float] = []
        for weight in weights.itertuples(index=False):
            mass = float(weight.posterior_mass)
            if weight.domain == "base":
                means_abs.append(float(query_row.base_absolute_s))
                means_rate.append(float(query_row.base_rate))
                vars_abs.append(float(query_row.base_absolute_variance))
                vars_rate.append(float(query_row.base_rate_variance))
                support.append(float(query_row.base_support_ess))
                unique_wells.append(float(query_row.base_unique_wells))
                condition.append(float(query_row.base_condition_number))
                surface_variance.append(float(query_row.base_surface_variance))
            else:
                field = component_lookup.loc[(query_id, str(weight.component_id))]
                means_abs.append(float(field["field_absolute_s"]))
                means_rate.append(float(field["field_rate"]))
                vars_abs.append(float(field["absolute_variance"]))
                vars_rate.append(float(field["rate_variance"]))
                support.append(float(field["support_ess"]))
                unique_wells.append(float(field["unique_wells"]))
                condition.append(float(field["condition_number"]))
                surface_variance.append(float(field["surface_variance"]))
            masses.append(mass)
        mass_array = np.asarray(masses, dtype=float)
        abs_array = np.asarray(means_abs, dtype=float)
        rate_array = np.asarray(means_rate, dtype=float)
        mean_abs = float(np.sum(mass_array * abs_array))
        mean_rate = float(np.sum(mass_array * rate_array))
        total_abs_variance = float(
            np.sum(mass_array * (np.asarray(vars_abs) + np.square(abs_array)))
            - mean_abs**2
        )
        total_rate_variance = float(
            np.sum(mass_array * (np.asarray(vars_rate) + np.square(rate_array)))
            - mean_rate**2
        )
        component_weights = weights.loc[weights["domain"].eq("component")].sort_values(
            ["posterior_mass", "component_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        component_mass = float(component_weights["posterior_mass"].sum())
        dominant_component = (
            str(component_weights.iloc[0]["component_id"])
            if not component_weights.empty
            else "base"
        )
        positive_mass = mass_array[mass_array > 0]
        entropy = float(-np.sum(positive_mass * np.log(positive_mass)))
        records.append(
            {
                "fold": int(query_row.fold),
                "query_id": query_id,
                "well_id": str(query_row.well_id),
                "MD": float(query_row.MD),
                "eligible": True,
                "mixed_absolute_s": mean_abs,
                "mixed_rate": mean_rate,
                "mixed_absolute_variance": max(total_abs_variance, 1.0e-9),
                "mixed_rate_variance": max(total_rate_variance, 1.0e-9),
                "mixed_support_ess": float(np.sum(mass_array * np.asarray(support))),
                "mixed_unique_wells": float(
                    np.sum(mass_array * np.asarray(unique_wells))
                ),
                "mixed_condition_number": float(
                    np.sum(mass_array * np.asarray(condition))
                ),
                "mixed_surface_variance": float(
                    np.sum(mass_array * np.asarray(surface_variance))
                ),
                "component_posterior_mass": component_mass,
                "dominant_component": dominant_component,
                "posterior_entropy": entropy,
            }
        )
    return pd.DataFrame(records)


def exp383_field_confidence(
    support_ess: np.ndarray,
    unique_wells: np.ndarray,
    condition_number: np.ndarray,
    surface_variance: np.ndarray,
    surface_variance_reference: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    support_full = float(get_nested(config, "method.path.support_ess_full", 32.0))
    wells_full = float(get_nested(config, "method.path.support_unique_wells_full", 24.0))
    condition_reference = float(get_nested(config, "method.path.condition_reference", 1.0e4))
    c_support = np.clip(np.asarray(support_ess, dtype=float) / support_full, 0.0, 1.0)
    c_wells = np.clip(np.asarray(unique_wells, dtype=float) / wells_full, 0.0, 1.0)
    condition = np.maximum(np.asarray(condition_number, dtype=float), condition_reference)
    c_condition = np.clip(
        math.log10(condition_reference) / np.log10(condition), 0.0, 1.0
    )
    reference = np.maximum(np.asarray(surface_variance_reference, dtype=float), 1.0e-9)
    c_surface = np.exp(-np.asarray(surface_variance, dtype=float) / reference)
    return c_support * c_wells * c_condition * c_surface


def prepare_path_inputs(
    query: pd.DataFrame,
    mixed: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    joined = query.merge(
        mixed,
        on=["fold", "query_id", "well_id", "MD"],
        how="left",
        validate="one_to_one",
    )
    require_finite(
        joined,
        (
            "mixed_absolute_s",
            "mixed_rate",
            "mixed_absolute_variance",
            "mixed_rate_variance",
        ),
        "marginalized field",
    )
    confidence = exp383_field_confidence(
        joined["mixed_support_ess"].to_numpy(dtype=float),
        joined["mixed_unique_wells"].to_numpy(dtype=float),
        joined["mixed_condition_number"].to_numpy(dtype=float),
        joined["mixed_surface_variance"].to_numpy(dtype=float),
        joined["surface_variance_reference"].to_numpy(dtype=float),
        config,
    )
    joined["field_confidence"] = confidence
    joined["final_rate"] = (
        confidence * joined["mixed_rate"].to_numpy(dtype=float)
        + (1.0 - confidence) * joined["fallback_rate"].to_numpy(dtype=float)
    )
    joined["calibrated_absolute_s"] = joined["mixed_absolute_s"].to_numpy(dtype=float)
    for _, indices in joined.groupby("well_id", sort=True).groups.items():
        block = joined.loc[indices]
        known = np.isfinite(block["TVT_input"].to_numpy(dtype=float))
        if not known.any():
            raise ValueError("every target well requires at least one finite TVT_input prefix")
        input_s = (
            block.loc[known, "TVT_input"].to_numpy(dtype=float)
            + block.loc[known, "Z"].to_numpy(dtype=float)
        )
        residual = input_s - block.loc[known, "mixed_absolute_s"].to_numpy(dtype=float)
        bias = huber_location(residual)
        joined.loc[indices, "prefix_bias_ft"] = bias
        joined.loc[indices, "calibrated_absolute_s"] = (
            block["mixed_absolute_s"].to_numpy(dtype=float) + bias
        )
    return joined


def solve_path_for_well(
    block: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    ordered = block.sort_values(["MD", "query_id"], kind="mergesort")
    md = ordered["MD"].to_numpy(dtype=float)
    if len(md) == 0 or np.any(np.diff(md) <= 0):
        return ordered["base_path_s"].to_numpy(dtype=float), "invalid_md_exp383_fallback"
    n_rows = len(ordered)
    equations: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    absolute = ordered["calibrated_absolute_s"].to_numpy(dtype=float)
    absolute_variance = np.maximum(
        ordered["mixed_absolute_variance"].to_numpy(dtype=float), 1.0e-9
    )
    for index in range(n_rows):
        row = np.zeros(n_rows, dtype=float)
        row[index] = 1.0
        equations.append(row)
        targets.append(float(absolute[index]))
        weights.append(float(1.0 / absolute_variance[index]))
    rate = ordered["final_rate"].to_numpy(dtype=float)
    rate_variance = np.maximum(ordered["mixed_rate_variance"].to_numpy(dtype=float), 1.0e-9)
    for index in range(n_rows - 1):
        delta_md = float(md[index + 1] - md[index])
        row = np.zeros(n_rows, dtype=float)
        row[index] = -1.0
        row[index + 1] = 1.0
        equations.append(row)
        targets.append(float(0.5 * (rate[index] + rate[index + 1]) * delta_md))
        weights.append(float(1.0 / (0.5 * (rate_variance[index] + rate_variance[index + 1]))))
    curvature_weight = float(get_nested(config, "method.path.curvature_weight", 1.0e-3))
    for index in range(1, n_rows - 1):
        left = float(md[index] - md[index - 1])
        right = float(md[index + 1] - md[index])
        row = np.zeros(n_rows, dtype=float)
        row[index - 1] = 1.0 / left
        row[index] = -(1.0 / left + 1.0 / right)
        row[index + 1] = 1.0 / right
        equations.append(row)
        targets.append(0.0)
        weights.append(curvature_weight)
    design = np.vstack(equations)
    target = np.asarray(targets, dtype=float)
    weight = np.asarray(weights, dtype=float)
    normal = design.T @ (design * weight[:, None])
    rhs = design.T @ (weight * target)
    prefix_known = np.isfinite(ordered["TVT_input"].to_numpy(dtype=float))
    ineligible = ~ordered["eligible"].astype(bool).to_numpy()
    known = prefix_known | ineligible
    known_value = np.full(n_rows, np.nan, dtype=float)
    known_value[prefix_known] = (
        ordered.loc[prefix_known, "TVT_input"].to_numpy(dtype=float)
        + ordered.loc[prefix_known, "Z"].to_numpy(dtype=float)
    )
    known_value[ineligible & ~prefix_known] = ordered.loc[
        ineligible & ~prefix_known, "base_path_s"
    ].to_numpy(dtype=float)
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
                "rank_deficient_exp383_fallback",
            )
    if not np.isfinite(solution).all():
        return ordered["base_path_s"].to_numpy(dtype=float), "nonfinite_exp383_fallback"
    return solution, "piecewise"


def solve_all_paths(path_inputs: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for _well_id, block in path_inputs.groupby("well_id", sort=True):
        ordered = block.sort_values(["MD", "query_id"], kind="mergesort").copy()
        if not bool(ordered["eligible"].any()):
            solution = ordered["base_path_s"].to_numpy(dtype=float)
            status = "exact_exp383_fallback"
        else:
            solution, status = solve_path_for_well(ordered, config)
        ordered["path_s"] = solution
        ordered["path_status"] = status
        records.append(ordered)
    return pd.concat(records, ignore_index=True).sort_values(
        ["fold", "well_id", "MD"], kind="mergesort"
    )


def interpolate_oof_prediction(
    parent_oof_keys: pd.DataFrame,
    solved_query: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        parent_oof_keys,
        ("fold", "well_id", "row_idx", "MD", "Z", "exp383_prediction"),
        "parent OOF keys",
    )
    records: list[pd.DataFrame] = []
    for well_id, target in parent_oof_keys.groupby("well_id", sort=True):
        query = solved_query.loc[solved_query["well_id"].eq(str(well_id))].sort_values("MD")
        if query.empty:
            raise ValueError(f"no solved query path for {well_id}")
        target = target.sort_values("row_idx", kind="mergesort").copy()
        if (
            not bool(query["eligible"].any())
            or query["path_status"].str.contains("fallback").any()
        ):
            target["exp384_prediction"] = target["exp383_prediction"].to_numpy(dtype=float)
            target["exp384_path_status"] = "exact_exp383_fallback"
            target["fault_domain"] = "base"
            target["component_posterior_mass"] = 0.0
        else:
            query_md = query["MD"].to_numpy(dtype=float)
            query_s = query["path_s"].to_numpy(dtype=float)
            target_md = target["MD"].to_numpy(dtype=float)
            if target_md.min() < query_md.min() or target_md.max() > query_md.max():
                target["exp384_prediction"] = target["exp383_prediction"].to_numpy(dtype=float)
                target["exp384_path_status"] = "coverage_exp383_fallback"
                target["fault_domain"] = "base"
                target["component_posterior_mass"] = 0.0
            else:
                interpolated_s = np.interp(target_md, query_md, query_s)
                target["exp384_prediction"] = interpolated_s - target["Z"].to_numpy(dtype=float)
                target["exp384_path_status"] = "piecewise"
                right = np.searchsorted(query_md, target_md, side="left")
                right = np.clip(right, 0, len(query_md) - 1)
                left = np.clip(right - 1, 0, len(query_md) - 1)
                choose_right = np.abs(query_md[right] - target_md) < np.abs(
                    query_md[left] - target_md
                )
                nearest = np.where(choose_right, right, left)
                target["fault_domain"] = query["dominant_component"].to_numpy()[nearest]
                target["component_posterior_mass"] = query[
                    "component_posterior_mass"
                ].to_numpy(dtype=float)[nearest]
        records.append(target)
    return pd.concat(records, ignore_index=True).sort_values(
        ["fold", "well_id", "row_idx"], kind="mergesort"
    )


# %% [markdown]
# ## 7. Stage 0 target-free freeze and integrity gate

# %%
def freeze_target_free_outputs(frames: Mapping[str, pd.DataFrame]) -> dict[str, str]:
    required = {
        "graph_nodes",
        "graph_edges",
        "components",
        "component_fields",
        "posterior",
        "prefix_likelihood",
        "path",
        "prediction",
    }
    missing = sorted(required.difference(frames))
    if missing:
        raise ValueError(f"target-free freeze is missing frames: {missing}")
    return {
        name: frame_content_sha256(frame)
        for name, frame in sorted(frames.items())
    }


def evaluate_stage0(
    *,
    query: pd.DataFrame,
    component_fields: pd.DataFrame,
    posterior: pd.DataFrame,
    solved_query: pd.DataFrame,
    ledger: RoleReadLedger,
    runtime_seconds: float,
    processed_wells: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage0_target_free")
    query_count = max(len(query), 1)
    eligible_query_ids = set(component_fields["query_id"].astype(str))
    eligible_coverage = len(eligible_query_ids) / query_count
    component_wells_p05 = (
        float(np.percentile(component_fields["unique_wells"], 5))
        if not component_fields.empty
        else 0.0
    )
    posterior_sum_error = float(
        np.max(
            np.abs(
                posterior.groupby("query_id")["posterior_mass"].sum().to_numpy(dtype=float)
                - 1.0
            )
        )
    )
    base_mass = posterior.loc[posterior["domain"].eq("base"), "posterior_mass"]
    finite_columns = [
        "mixed_absolute_s",
        "mixed_rate",
        "mixed_absolute_variance",
        "mixed_rate_variance",
        "path_s",
    ]
    finite_coverage = float(
        np.isfinite(solved_query[finite_columns].to_numpy(dtype=float)).all(axis=1).mean()
    )
    ineligible = ~solved_query["eligible"].astype(bool)
    if ineligible.any():
        absolute_parity = float(
            np.max(
                np.abs(
                    solved_query.loc[ineligible, "mixed_absolute_s"].to_numpy(dtype=float)
                    - solved_query.loc[ineligible, "base_absolute_s"].to_numpy(dtype=float)
                )
            )
        )
        rate_parity = float(
            np.max(
                np.abs(
                    solved_query.loc[ineligible, "mixed_rate"].to_numpy(dtype=float)
                    - solved_query.loc[ineligible, "base_rate"].to_numpy(dtype=float)
                )
            )
        )
        path_parity = float(
            np.max(
                np.abs(
                    solved_query.loc[ineligible, "path_s"].to_numpy(dtype=float)
                    - solved_query.loc[ineligible, "base_path_s"].to_numpy(dtype=float)
                )
            )
        )
        parity = max(absolute_parity, rate_parity, path_parity)
    else:
        parity = 0.0
    projected_runtime = runtime_seconds / max(processed_wells, 1) * int(
        get_nested(config, "validation.expected_wells")
    )
    observed = {
        "eligible_query_coverage": eligible_coverage,
        "component_unique_wells_p05": component_wells_p05,
        "finite_coverage": finite_coverage,
        "posterior_sum_max_abs_error": posterior_sum_error,
        "smooth_base_mass_min": float(base_mass.min()) if not base_mass.empty else 0.0,
        "exp383_fallback_parity_max_abs_ft": parity,
        "valid_graph_rows": ledger.graph_outer_valid_rows,
        "valid_formation_reads": ledger.valid_formation_reads,
        "valid_suffix_truth_reads": ledger.valid_suffix_truth_reads,
        "projected_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": peak_rss_gb(),
    }
    checks = {
        "eligible_query_coverage": observed["eligible_query_coverage"]
        >= float(gates["eligible_query_coverage_min"]),
        "component_unique_wells_p05": observed["component_unique_wells_p05"]
        >= float(gates["component_unique_wells_p05_min"]),
        "finite_coverage": observed["finite_coverage"] >= float(gates["finite_coverage_min"]),
        "posterior_sum": observed["posterior_sum_max_abs_error"]
        <= float(gates["posterior_sum_max_abs_error"]),
        "smooth_base_mass": observed["smooth_base_mass_min"]
        >= float(gates["smooth_base_mass_min"]),
        "exp383_fallback_parity": observed["exp383_fallback_parity_max_abs_ft"]
        <= float(gates["exp383_fallback_parity_max_abs_ft"]),
        "valid_graph_rows": observed["valid_graph_rows"] <= int(gates["valid_graph_rows_max"]),
        "valid_formation_reads": observed["valid_formation_reads"]
        <= int(gates["valid_formation_reads_max"]),
        "valid_suffix_truth_reads": observed["valid_suffix_truth_reads"]
        <= int(gates["valid_suffix_truth_reads_max"]),
        "projected_runtime": observed["projected_runtime_seconds"]
        <= float(gates["projected_runtime_seconds_max"]),
        "projected_peak_rss": observed["projected_peak_rss_gb"]
        <= float(gates["projected_peak_rss_gb_max"]),
    }
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}


# %% [markdown]
# ## 8. Late truth join and Stage 1 direct readout

# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(np.asarray(truth, dtype=float) - np.asarray(prediction, dtype=float))
            )
        )
    )


def late_join_truth(
    prediction: pd.DataFrame,
    parent_oof: pd.DataFrame,
    frozen_hashes: Mapping[str, str],
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    ledger.mark_truth_join(frozen_hashes)
    required = (
        "fold",
        "well_id",
        "row_idx",
        "tvt_true",
        "exp383_prediction",
        "distance_from_anchor",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    )
    require_columns(parent_oof, required, "exp383 OOF")
    truth_columns = [
        "fold",
        "well_id",
        "row_idx",
        "tvt_true",
        "distance_from_anchor",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]
    merged = prediction.merge(
        parent_oof[truth_columns],
        on=["fold", "well_id", "row_idx"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(prediction):
        raise ValueError("late truth join changed prediction row count")
    return merged


def _metric_record(frame: pd.DataFrame, scope: str, value: str) -> dict[str, Any]:
    truth = frame["tvt_true"].to_numpy(dtype=float)
    control = frame["exp383_prediction"].to_numpy(dtype=float)
    candidate = frame["exp384_prediction"].to_numpy(dtype=float)
    control_rmse = rmse(truth, control)
    candidate_rmse = rmse(truth, candidate)
    return {
        "scope": scope,
        "scope_value": value,
        "rows": int(len(frame)),
        "exp383_rmse": control_rmse,
        "exp384_rmse": candidate_rmse,
        "gain_exp383_minus_exp384": control_rmse - candidate_rmse,
        "delta_exp384_minus_exp383": candidate_rmse - control_rmse,
    }


def build_stage1_readout(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [_metric_record(scored, "pooled", "all")]
    for fold, group in scored.groupby("fold", sort=True):
        records.append(_metric_record(group, "fold", str(int(fold))))
    distance = scored["distance_from_anchor"].to_numpy(dtype=float)
    scopes = {
        "near_0_250": distance <= 250.0,
        "long_1000_plus": distance >= 1000.0,
        "hidden_like_spatial": scored["hidden_like_spatial"].astype(bool).to_numpy(),
        "hidden_like_typewell_purged": scored[
            "hidden_like_typewell_purged"
        ].astype(bool).to_numpy(),
        "eligible_rows": scored["fault_domain"].ne("base").to_numpy(),
    }
    for name, mask in scopes.items():
        if mask.any():
            records.append(_metric_record(scored.loc[mask], "scope", name))
    for domain, group in scored.groupby("fault_domain", sort=True):
        records.append(_metric_record(group, "fault_domain", str(domain)))
    by_well = pd.DataFrame(
        [
            {
                "well_id": str(well_id),
                **_metric_record(group, "well", str(well_id)),
            }
            for well_id, group in scored.groupby("well_id", sort=True)
        ]
    )
    return pd.DataFrame(records), by_well


def evaluate_stage1(
    metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage1_direct")
    pooled = metrics.loc[
        metrics["scope"].eq("pooled") & metrics["scope_value"].eq("all")
    ].iloc[0]
    folds = metrics.loc[metrics["scope"].eq("fold")]
    scopes = metrics.loc[metrics["scope"].eq("scope")].set_index("scope_value")

    def scope_gain(name: str) -> float:
        if name not in scopes.index:
            return float("-inf")
        return float(scopes.loc[name, "gain_exp383_minus_exp384"])

    positive_folds = int((folds["gain_exp383_minus_exp384"] > 0).sum())
    observed = {
        "pooled_rmse_gain_vs_exp383_ft": float(pooled["gain_exp383_minus_exp384"]),
        "positive_folds": positive_folds,
        "long_1000_plus_gain_ft": scope_gain("long_1000_plus"),
        "hidden_like_spatial_gain_ft": scope_gain("hidden_like_spatial"),
        "hidden_like_typewell_purged_gain_ft": scope_gain(
            "hidden_like_typewell_purged"
        ),
        "near_0_250_delta_ft": -scope_gain("near_0_250"),
        "eligible_row_gain_ft": scope_gain("eligible_rows"),
    }
    checks = {
        "pooled_gain": observed["pooled_rmse_gain_vs_exp383_ft"]
        >= float(gates["pooled_rmse_gain_vs_exp383_ft_min"]),
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
        "eligible_rows": observed["eligible_row_gain_ft"]
        >= float(gates["eligible_row_gain_ft_min"]),
    }
    return {"passed": bool(all(checks.values())), "observed": observed, "checks": checks}


# %% [markdown]
# ## 9. Setup, configuration preview, and execution

# %%
def run_train() -> dict[str, Any]:
    config = load_config()
    validate_execution_contract(config, require_kaggle_authorization=True)
    parent_dir = resolve_parent_artifact_dir(config)
    manifest_filename = str(
        get_nested(config, "data.parent_artifacts.files.manifest", "exp383_manifest.json")
    )
    manifest = json.loads((parent_dir / manifest_filename).read_text())
    validate_parent_manifest(manifest, config)
    ledger = RoleReadLedger()
    donor = load_pinned_parent_table(parent_dir, "donor_nodes_256", manifest, config)
    query = load_pinned_parent_table(parent_dir, "query_fields", manifest, config)
    validate_target_safe_query(query, ledger)
    query["well_id"] = query["well_id"].astype(str)
    query["query_id"] = query["query_id"].astype(str)
    graph_nodes = canonicalize_graph_nodes(donor, config, ledger)

    maximum_wells_raw = os.environ.get("EXP384_MAX_WELLS")
    selected_wells: list[str] | None = None
    if maximum_wells_raw:
        maximum_wells = int(maximum_wells_raw)
        selected_wells = sorted(query["well_id"].astype(str).unique())[:maximum_wells]
        query = query.loc[query["well_id"].astype(str).isin(selected_wells)].copy()

    started = time.perf_counter()
    edges = build_fault_graph(graph_nodes, config)
    assignments, components = assign_fault_components(graph_nodes, edges, config)
    catalog, component_catalog_frame = build_component_catalog(
        graph_nodes, assignments, config
    )
    component_fields = generate_component_fields(
        query, graph_nodes, assignments, catalog, config
    )
    prefix_scale = float(manifest["calibration"]["prefix_loo_huber_rmse_p50_ft"])
    prefix_likelihood = build_prefix_likelihood(
        query, component_fields, prefix_scale_ft=prefix_scale
    )
    posterior = build_domain_posterior(
        query, component_fields, prefix_likelihood, config
    )
    mixed = marginalize_fields(query, component_fields, posterior)
    path_inputs = prepare_path_inputs(query, mixed, config)
    solved_query = solve_all_paths(path_inputs, config)

    parent_oof_keys = load_pinned_parent_table(
        parent_dir, "oof_keys_without_truth", manifest, config
    )
    parent_oof_keys["well_id"] = parent_oof_keys["well_id"].astype(str)
    validate_truth_free_oof_keys(parent_oof_keys, ledger)
    if selected_wells is not None:
        parent_oof_keys = parent_oof_keys.loc[
            parent_oof_keys["well_id"].isin(selected_wells)
        ].copy()
    validate_oof_key_contract(
        parent_oof_keys,
        config,
        full_run=selected_wells is None,
    )
    prediction = interpolate_oof_prediction(parent_oof_keys, solved_query)
    elapsed = time.perf_counter() - started
    frames = {
        "graph_nodes": graph_nodes,
        "graph_edges": edges,
        "components": components,
        "component_fields": component_fields,
        "posterior": posterior,
        "prefix_likelihood": prefix_likelihood,
        "path": solved_query[
            ["fold", "query_id", "well_id", "MD", "path_s", "path_status"]
        ],
        "prediction": prediction[
            [
                "fold",
                "well_id",
                "row_idx",
                "exp383_prediction",
                "exp384_prediction",
                "exp384_path_status",
                "fault_domain",
                "component_posterior_mass",
            ]
        ],
    }
    frozen_hashes = freeze_target_free_outputs(frames)
    stage0 = evaluate_stage0(
        query=query,
        component_fields=component_fields,
        posterior=posterior,
        solved_query=solved_query,
        ledger=ledger,
        runtime_seconds=elapsed,
        processed_wells=int(query["well_id"].nunique()),
        config=config,
    )

    output = artifacts_dir()
    generated = {
        "graph_edges": edges,
        "component_assignments": assignments,
        "component_catalog": component_catalog_frame,
        "component_fields": component_fields,
        "prefix_likelihood": prefix_likelihood,
        "posterior": posterior,
        "solved_query": solved_query,
        "prediction": prediction,
    }
    for name, frame in generated.items():
        write_table(frame, output / f"{EXPERIMENT_NAME}_{name}.parquet")
    write_json(output / f"{EXPERIMENT_NAME}_target_free_sha.json", frozen_hashes)
    metrics: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_pass" if stage0["passed"] else "stage0_fail_closed",
        "parent_manifest_logical_sha256": parent_manifest_sha256(manifest),
        "target_free_sha256": frozen_hashes,
        "stage0": stage0,
        "stage1": None,
        "ledger": ledger.as_dict(),
        "runtime_seconds": elapsed,
    }
    if not stage0["passed"]:
        write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
        return metrics
    if selected_wells is not None:
        metrics["status"] = "stage0_resource_preflight_pass"
        metrics["stage1"] = {
            "passed": None,
            "status": "not_opened_during_resource_preflight",
        }
        write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
        return metrics

    parent_oof = load_pinned_parent_table(parent_dir, "oof_with_truth", manifest, config)
    scored = late_join_truth(prediction, parent_oof, frozen_hashes, ledger)
    stage1_metrics, by_well = build_stage1_readout(scored)
    stage1 = evaluate_stage1(stage1_metrics, config)
    metrics["stage1"] = stage1
    metrics["status"] = "stage1_pass" if stage1["passed"] else "stage1_fail_closed"
    metrics["ledger"] = ledger.as_dict()
    write_table(
        stage1_metrics,
        output / f"{EXPERIMENT_NAME}_stage1_metrics.csv",
    )
    write_table(by_well, output / f"{EXPERIMENT_NAME}_by_well_metrics.csv")
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
        "scientific_candidates": get_nested(CONFIG_PREVIEW, "runtime.scientific_candidates"),
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

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    RUN_METRICS = run_train()
    print(json.dumps(RUN_METRICS, indent=2, ensure_ascii=False))
