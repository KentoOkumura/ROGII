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
# # exp406 loop-closed multi-well RGT fixed16 Stage 0
#
# This CPU-only diagnostic builds horizontal-GR pairwise correspondences for the
# frozen exp386 fixed16 wells and projects their relative TVT offsets onto one
# loop-closed graph. Outer-valid target columns are limited to the current-test
# observable contract before the graph artifacts are frozen. The final 512 rows
# of the visible prefix are scored only after that freeze.
#
# The exp226 comparison is deliberately narrow: for each fixed16 pseudo-cut, the
# original K16 geometry donor field and adaptive Kappa are rebuilt from that
# fold's outer-train wells. It produces only the `tvt_geop`-equivalent control on
# the held-out prefix. It does not rerun the official OOF, GR correction,
# U-projection, current-test inference, or any fitted ML/PF/HMM/Beam model.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Runtime, path, SHA, and serialization helpers
# 3. Dependency, fold, Type-Well, and guarded-read helpers
# 4. Block construction, donor selection, and GR morphology
# 5. Pairwise edges and deterministic circular control
# 6. Fundamental cycles and Huber IRLS loop closure
# 7. Fixed16 target-free graph, freeze, and technical gates
# 8. Fixed16-only exp226 K16 geometry control and prefix readout
# 9. Resource projection, artifact manifest, and orchestration
# 10. Configuration preview and guarded execution

# %% [markdown]
# ## 1. Imports and immutable execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.linalg import lsqr

EXPERIMENT_NAME = "exp406_loop_closed_multiwell_rgt_fixed16_stage0"
EXP226_NAME = "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
EXP405_NAME = "exp405_geometry_reinjected_interval_semimarkov_fusion"
EXP065_NAME = "exp065_typewell_supertype_cluster_cv_audit"
IMPORT_ONLY_ENV = "EXP406_IMPORT_ONLY"

TARGET_ALLOWED_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT_input")
TARGET_FORBIDDEN_COLUMNS = frozenset(
    {"TVT", "ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"}
)
DONOR_COLUMNS = ("MD", "X", "Y", "Z", "GR", "TVT")
EXP226_SAFE_COLUMNS = ("well_id", "row_idx", "fold", "tvt_geop")

BLOCK_SORT_COLUMNS = ("fold", "well_id", "block_id")
EDGE_SORT_COLUMNS = (
    "fold",
    "source_well_id",
    "target_well_id",
    "source_block_id",
    "target_block_id",
    "edge_rank",
)
CYCLE_SORT_COLUMNS = ("fold", "component_id", "cycle_id")
GAUGE_SORT_COLUMNS = ("fold", "well_id", "block_id")


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


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_kaggle_authorization: bool,
) -> dict[str, int]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp406 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != EXP405_NAME:
        raise ValueError("exp406 parent must remain exp405")
    if not bool(get_nested(config, "implementation.enabled", False)):
        raise RuntimeError("exp406 implementation is not enabled")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise RuntimeError("exp406 implementation authorization is not recorded")
    if not bool(get_nested(config, "implementation.unlock_condition_satisfied", False)):
        raise RuntimeError("exp405 scientific-fail dependency is not satisfied")
    if bool(get_nested(config, "execution.full_oof_stage1_authorized", True)):
        raise ValueError("full OOF Stage 1 must remain disabled")
    if bool(get_nested(config, "execution.current_test_authorized", True)):
        raise ValueError("current-test execution must remain disabled")
    if bool(get_nested(config, "execution.inference_authorized", True)):
        raise ValueError("inference must remain disabled")
    if bool(get_nested(config, "execution.submission_authorized", True)):
        raise ValueError("submission must remain disabled")
    if get_nested(config, "prefix_rolling_origin.comparison") != (
        "fixed16_recomputed_exp226_k16_geometry_tvt_geop"
    ):
        raise ValueError("prefix control must remain the approved fixed16 exp226 geometry replay")
    if not bool(
        get_nested(
            config,
            "prefix_rolling_origin.exp226_geometry_control.user_approved",
            False,
        )
    ):
        raise RuntimeError("fixed16 exp226 geometry replay approval is not recorded")

    expected = {
        "runtime.scientific_diagnostics": 1,
        "runtime.target_wells": 16,
        "runtime.reporting_folds": 5,
        "runtime.graph_contexts": 5,
        "runtime.model_configs": 0,
        "runtime.trained_folds": 0,
        "runtime.lightgbm_boosters": 0,
        "runtime.hmm_runs": 0,
        "runtime.pf_runs": 0,
        "runtime.beam_runs": 0,
    }
    for key, expected_value in expected.items():
        observed = int(get_nested(config, key, -1))
        if observed != expected_value:
            raise ValueError(f"{key} must remain {expected_value}, got {observed}")
    if int(get_nested(config, "pairwise_gr.block_rows", -1)) != 256:
        raise ValueError("pairwise block_rows must remain 256")
    if int(get_nested(config, "pairwise_gr.stride_rows", -1)) != 128:
        raise ValueError("pairwise stride_rows must remain 128")
    if int(
        get_nested(config, "pairwise_gr.donor_pool.maximum_unique_wells", -1)
    ) != 12:
        raise ValueError("pairwise donor count must remain 12")
    if int(
        get_nested(
            config,
            "pairwise_gr.edge_selection.maximum_edges_per_target_block",
            -1,
        )
    ) != 4:
        raise ValueError("edge cap must remain 4")
    shift = get_nested(config, "pairwise_gr.shift_grid_ft", {})
    if (
        float(shift.get("minimum", math.nan)),
        float(shift.get("maximum", math.nan)),
        float(shift.get("step", math.nan)),
    ) != (-55.0, 55.0, 5.0):
        raise ValueError("shift grid must remain [-55, 55] in 5-ft steps")
    if int(get_nested(config, "loop_closure.iterations", -1)) != 10:
        raise ValueError("loop closure must run exactly 10 IRLS iterations")
    if float(get_nested(config, "loop_closure.huber_delta_ft", math.nan)) != 5.0:
        raise ValueError("Huber delta must remain 5 ft")
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized", False)
    ):
        raise RuntimeError(
            "exp406 Kaggle execution is not authorized; implementation approval "
            "does not authorize package push or execution"
        )
    return {key.rsplit(".", 1)[-1]: value for key, value in expected.items()}


# %% [markdown]
# ## 2. Runtime, path, SHA, and serialization helpers

# %%
PACKAGE_DIR = Path.cwd()


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = [
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in sorted(Path("/kaggle/working").rglob("config.yaml")):
        try:
            value = yaml.safe_load(candidate.read_text()) or {}
        except Exception:
            continue
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError("exp406 config.yaml was not found")


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
    def default(item: Any) -> Any:
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        if pd.isna(item) and not isinstance(item, str):
            return None
        raise TypeError(f"cannot JSON serialize {type(item)!r}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]
    return sha256_bytes(stable_json_bytes(schema))


def frame_content_sha256(
    frame: pd.DataFrame,
    sort_columns: Sequence[str] = (),
) -> str:
    ordered = frame.copy()
    keys = [column for column in sort_columns if column in ordered.columns]
    if keys:
        ordered = ordered.sort_values(keys, kind="mergesort")
    ordered = ordered.reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode()
    return sha256_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(
        json.dumps(
            convert(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_table(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.reset_index(drop=True)
    if path.suffix == ".parquet":
        ordered.to_parquet(path, index=False)
    else:
        ordered.to_csv(path, index=False)


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if value > 1024**3:
        return value / 1024**3
    return value / 1024**2


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not finite.any():
        return math.nan
    return float(
        np.sqrt(np.mean(np.square(actual_array[finite] - predicted_array[finite])))
    )


def resolve_candidate_file(candidates: Sequence[str], filename: str) -> Path:
    root = find_project_root()
    roots = [Path(value) for value in candidates]
    roots.extend(
        [
            root / "artifacts",
            root / "experiments" / EXP226_NAME / "artifacts",
            root / "experiments" / EXP405_NAME / "artifacts",
            root / "experiments" / EXP065_NAME / "artifacts",
        ]
    )
    for candidate in roots:
        path = candidate if candidate.name == filename else candidate / filename
        if path.exists() and path.stat().st_size:
            return path
    matches: list[Path] = []
    for search_root in (Path("/kaggle/input"), Path("/tmp")):
        if search_root.exists():
            matches.extend(search_root.rglob(filename))
    matches = [path for path in matches if path.stat().st_size]
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"could not resolve non-empty artifact {filename}")


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    candidates = [
        find_project_root() / str(get_nested(config, "data.train_dir", "data/raw/train")),
        Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*__horizontal_well.csv")):
            return candidate
    for candidate in sorted(Path("/kaggle/input").rglob("train")):
        if candidate.is_dir() and list(candidate.glob("*__horizontal_well.csv")):
            return candidate
    raise FileNotFoundError("raw train horizontal-well directory was not found")


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", maxsplit=1)[0]


# %% [markdown]
# ## 3. Dependency, fold, Type-Well, and guarded-read helpers

# %%
@dataclass
class RoleReadLedger:
    events: list[dict[str, Any]] = field(default_factory=list)
    frozen: bool = False
    source_target_overlap: int = 0

    def _record(
        self,
        *,
        fold: int,
        well_id: str,
        role: str,
        columns: Iterable[str],
        rows: int,
    ) -> None:
        names = tuple(map(str, columns))
        before = not self.frozen
        forbidden = TARGET_FORBIDDEN_COLUMNS.intersection(names)
        if role == "outer_valid_target_free" and forbidden:
            raise ValueError(
                "target-free outer-valid read contains forbidden columns: "
                f"{sorted(forbidden)}"
            )
        if role == "outer_valid_prefix_truth_late" and not self.frozen:
            raise ValueError("prefix truth cannot be attached before graph SHA freeze")
        self.events.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "role": role,
                "rows": int(rows),
                "column_signature": ",".join(names),
                "suffix_truth_reads": int(
                    role.startswith("outer_valid") and "TVT" in names
                ),
                "target_formation_reads": int(
                    role.startswith("outer_valid")
                    and bool(TARGET_FORBIDDEN_COLUMNS.difference({"TVT"}).intersection(names))
                ),
                "hidden_role_reads": int("hidden" in role),
                "before_target_freeze": bool(before),
            }
        )

    def record_source(
        self,
        fold: int,
        well_id: str,
        columns: Iterable[str],
        rows: int,
    ) -> None:
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_train_donor",
            columns=columns,
            rows=rows,
        )

    def record_target_safe(
        self,
        fold: int,
        well_id: str,
        columns: Iterable[str],
        rows: int,
    ) -> None:
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_valid_target_free",
            columns=columns,
            rows=rows,
        )

    def record_prefix_truth_late(self, fold: int, well_id: str, rows: int) -> None:
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_valid_prefix_truth_late",
            columns=("TVT_input",),
            rows=rows,
        )

    def validate_disjoint(
        self,
        source_wells: Iterable[str],
        target_wells: Iterable[str],
    ) -> None:
        overlap = set(map(str, source_wells)).intersection(map(str, target_wells))
        self.source_target_overlap += len(overlap)
        if overlap:
            raise ValueError(
                f"outer-valid wells leaked into donor pool: {sorted(overlap)[:5]}"
            )

    def mark_frozen(self, hashes: Mapping[str, str]) -> None:
        required = {"pairwise_edges", "cycle_basis", "loop_closed_gauge"}
        if not required.issubset(hashes):
            raise ValueError(f"target-free freeze is missing {sorted(required - set(hashes))}")
        self.frozen = True

    def as_frame(self) -> pd.DataFrame:
        columns = [
            "fold",
            "well_id",
            "role",
            "rows",
            "column_signature",
            "suffix_truth_reads",
            "target_formation_reads",
            "hidden_role_reads",
            "before_target_freeze",
        ]
        frame = pd.DataFrame(self.events, columns=columns)
        if frame.empty:
            return frame
        return frame.sort_values(
            ["fold", "role", "well_id"], kind="mergesort"
        ).reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        frame = self.as_frame()
        before = frame.loc[frame["before_target_freeze"]] if len(frame) else frame
        target = before["role"].eq("outer_valid_target_free") if len(before) else []
        return {
            "source_target_overlap": int(self.source_target_overlap),
            "suffix_truth_reads_before_freeze": (
                int(before.loc[target, "suffix_truth_reads"].sum()) if len(before) else 0
            ),
            "target_formation_reads_before_freeze": (
                int(before.loc[target, "target_formation_reads"].sum())
                if len(before)
                else 0
            ),
            "hidden_role_reads_before_freeze": (
                int(before["hidden_role_reads"].sum()) if len(before) else 0
            ),
            "prefix_truth_joined_after_freeze": bool(
                len(frame)
                and frame["role"].eq("outer_valid_prefix_truth_late").any()
                and not frame.loc[
                    frame["role"].eq("outer_valid_prefix_truth_late"),
                    "before_target_freeze",
                ].any()
            ),
        }


def load_exp405_decision(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    section = get_nested(config, "data.exp405_decision", {})
    filename = str(
        section.get(
            "filename",
            f"{EXP405_NAME}_summary.json",
        )
    )
    path = resolve_candidate_file(list(section.get("candidates", [])), filename)
    value = json.loads(path.read_text())
    required_decision = str(section["required_decision"])
    required_sha = str(section["decision_sha256"])
    if value.get("status") != required_decision:
        raise ValueError("exp405 decision status does not unlock exp406")
    if value.get("decision_sha256") != required_sha:
        raise ValueError("exp405 decision SHA does not match the frozen contract")
    gate = value.get("gate") or {}
    if not bool(gate.get("technical_pass")) or bool(gate.get("scientific_pass")):
        raise ValueError("exp405 must be technically valid and scientifically failed")
    return value, path


def load_exp226_oof(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path]:
    section = get_nested(config, "data.exp226", {})
    filename = str(section["filename"])
    path = resolve_candidate_file(list(section.get("candidates", [])), filename)
    actual = sha256_decompressed_csv(path)
    expected = str(section["expected_oof_decompressed_sha256"])
    if actual != expected:
        raise ValueError(f"exp226 OOF SHA mismatch: expected {expected}, got {actual}")
    frame = pd.read_csv(
        path,
        usecols=list(EXP226_SAFE_COLUMNS),
        dtype={"well_id": str},
    )
    require_columns(frame, EXP226_SAFE_COLUMNS, "saved exp226 OOF")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = frame["row_idx"].astype(np.int32)
    frame["fold"] = frame["fold"].astype(np.int8)
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("saved exp226 OOF row count differs from the contract")
    if frame["well_id"].nunique() != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("saved exp226 OOF well count differs from the contract")
    return frame, path


def load_typewell_groups(
    config: Mapping[str, Any],
) -> tuple[dict[str, str], pd.DataFrame, Path]:
    section = get_nested(config, "data.typewell_group_assignments", {})
    path = resolve_candidate_file(
        list(section.get("candidates", [])),
        str(section["filename"]),
    )
    expected = str(section["expected_file_sha256"])
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Type-Well assignment SHA mismatch: {actual}")
    frame = pd.read_csv(path, dtype=str)
    require_columns(
        frame,
        ("method", "threshold", "cluster_id", "well_id"),
        "Type-Well group assignments",
    )
    method = str(section["method"])
    threshold = str(section["threshold"])
    subset = frame.loc[
        frame["method"].eq(method) & frame["threshold"].eq(threshold)
    ].copy()
    if subset["well_id"].duplicated().any():
        raise ValueError("Type-Well group assignment must be unique per well")
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if subset["well_id"].nunique() != expected_wells:
        raise ValueError("Type-Well assignments do not cover every train well")
    lookup = dict(zip(subset["well_id"], subset["cluster_id"], strict=True))
    return lookup, subset, path


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


def validate_fold_identity(
    fold_by_well: Mapping[str, int],
    exp226_oof: pd.DataFrame,
) -> None:
    observed = (
        exp226_oof[["well_id", "fold"]]
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


def select_fixed16_wells(
    fold_by_well: Mapping[str, int],
    maximum_wells: int,
    folds: Sequence[int],
) -> list[str]:
    by_fold = {
        int(fold): sorted(
            well for well, assigned in fold_by_well.items() if int(assigned) == int(fold)
        )
        for fold in folds
    }
    selected: list[str] = []
    offset = 0
    while len(selected) < maximum_wells:
        progressed = False
        for fold in folds:
            wells = by_fold[int(fold)]
            if offset < len(wells):
                selected.append(wells[offset])
                progressed = True
                if len(selected) == maximum_wells:
                    break
        if not progressed:
            break
        offset += 1
    if len(selected) != maximum_wells:
        raise ValueError(f"fixed16 selector returned {len(selected)} wells")
    return selected


def read_target_safe_well(
    path: Path,
    fold: int,
    exp226_oof: pd.DataFrame,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(TARGET_ALLOWED_COLUMNS))
    well_id = well_id_from_path(path)
    ledger.record_target_safe(fold, well_id, frame.columns, len(frame))
    frame = frame.copy()
    frame["well_id"] = well_id
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    parent = exp226_oof.loc[
        exp226_oof["well_id"].eq(well_id),
        ["row_idx", "tvt_geop"],
    ]
    frame = frame.merge(parent, on="row_idx", how="left", validate="one_to_one")
    frame["base_tvt"] = frame["TVT_input"].where(
        frame["TVT_input"].notna(),
        frame["tvt_geop"],
    )
    frame["fold"] = int(fold)
    frame["role"] = "outer_valid"
    return frame


def read_donor_well(
    path: Path,
    fold: int,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=list(DONOR_COLUMNS))
    well_id = well_id_from_path(path)
    ledger.record_source(fold, well_id, frame.columns, len(frame))
    frame = frame.copy()
    frame["well_id"] = well_id
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["base_tvt"] = frame["TVT"].to_numpy(dtype=float)
    frame["fold"] = int(fold)
    frame["role"] = "outer_train"
    return frame


# %% [markdown]
# ## 4. Block construction, donor selection, and GR morphology

# %%
@dataclass(frozen=True)
class BlockSlice:
    block_id: int
    start: int
    stop: int
    center_row_idx: int
    center_tvt: float


def block_slices(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[BlockSlice]:
    rows = int(get_nested(config, "pairwise_gr.block_rows"))
    stride = int(get_nested(config, "pairwise_gr.stride_rows"))
    minimum = int(get_nested(config, "pairwise_gr.minimum_finite_pairs"))
    include_final = bool(get_nested(config, "pairwise_gr.include_final_short_block"))
    starts = list(range(0, max(len(frame) - rows + 1, 1), stride))
    if not starts:
        starts = [0]
    final_start = max(len(frame) - rows, 0)
    if include_final and final_start not in starts:
        starts.append(final_start)
    starts = sorted(set(starts))
    output: list[BlockSlice] = []
    base = frame["base_tvt"].to_numpy(dtype=float)
    row_idx = frame["row_idx"].to_numpy(dtype=int)
    for block_id, start in enumerate(starts):
        stop = min(start + rows, len(frame))
        if stop - start < minimum:
            continue
        center = start + (stop - start - 1) // 2
        output.append(
            BlockSlice(
                block_id=block_id,
                start=start,
                stop=stop,
                center_row_idx=int(row_idx[center]),
                center_tvt=float(base[center]),
            )
        )
    return output


def well_xy(frame: pd.DataFrame) -> tuple[float, float]:
    return (
        float(np.nanmedian(frame["X"].to_numpy(dtype=float))),
        float(np.nanmedian(frame["Y"].to_numpy(dtype=float))),
    )


def select_donor_wells(
    query_well: str,
    donor_candidates: Sequence[str],
    contexts: Mapping[str, pd.DataFrame],
    typewell_by_well: Mapping[str, str],
    maximum: int,
) -> list[str]:
    query_xy = well_xy(contexts[query_well])
    query_group = typewell_by_well.get(query_well)
    ranked: list[tuple[int, float, str]] = []
    for donor in donor_candidates:
        if donor == query_well:
            continue
        donor_xy = well_xy(contexts[donor])
        distance = math.hypot(donor_xy[0] - query_xy[0], donor_xy[1] - query_xy[1])
        same_rank = int(
            query_group is None or typewell_by_well.get(donor) != query_group
        )
        ranked.append((same_rank, distance, donor))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    return [well for _, _, well in ranked[:maximum]]


def centered_full_window_mean(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if window <= 1:
        return array.copy()
    return (
        pd.Series(array)
        .rolling(window, center=True, min_periods=window)
        .mean()
        .to_numpy(dtype=float)
    )


def robust_standardize(
    values: np.ndarray,
    scale_clip: Sequence[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.full(len(array), np.nan, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return output
    median = float(np.median(array[finite]))
    scale = float(np.median(np.abs(array[finite] - median)) * 1.4826)
    scale = float(np.clip(scale, float(scale_clip[0]), float(scale_clip[1])))
    output[finite] = (array[finite] - median) / max(scale, 1.0e-12)
    return output


def pearson_ncc(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    count = int(finite.sum())
    if count < 3:
        return math.nan, count
    x = x[finite] - float(np.mean(x[finite]))
    y = y[finite] - float(np.mean(y[finite]))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1.0e-12:
        return math.nan, count
    return float(np.dot(x, y) / denominator), count


def morphology_score(
    query_gr: np.ndarray,
    donor_gr: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    components = get_nested(config, "pairwise_gr.morphology.components")
    scale_clip = list(get_nested(config, "pairwise_gr.morphology.scale_clip"))
    minimum = int(get_nested(config, "pairwise_gr.minimum_finite_pairs"))
    minimum_fraction = float(get_nested(config, "pairwise_gr.minimum_pair_fraction"))
    weighted = 0.0
    weight_sum = 0.0
    details: dict[str, float] = {}
    raw_pairs = 0
    for name, section in components.items():
        window = int(section["window_rows"])
        weight = float(section["weight"])
        left = centered_full_window_mean(query_gr, window)
        right = centered_full_window_mean(donor_gr, window)
        left = robust_standardize(left, scale_clip)
        right = robust_standardize(right, scale_clip)
        score, pairs = pearson_ncc(left, right)
        if name == "raw":
            raw_pairs = pairs
        details[f"{name}_ncc"] = score
        details[f"{name}_pairs"] = float(pairs)
        if np.isfinite(score) and pairs >= minimum:
            weighted += weight * score
            weight_sum += weight
    pair_fraction = raw_pairs / max(len(query_gr), 1)
    eligible = (
        raw_pairs >= minimum
        and pair_fraction >= minimum_fraction
        and weight_sum > 0.0
    )
    score = weighted / weight_sum if eligible else math.nan
    return {
        "score": float(score),
        "finite_pairs": int(raw_pairs),
        "pair_fraction": float(pair_fraction),
        "eligible": bool(eligible and np.isfinite(score)),
        **details,
    }


def stable_circular_control(
    values: np.ndarray,
    well_id: str,
    block_id: int,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    output = array.copy()
    finite_index = np.flatnonzero(np.isfinite(array))
    if len(finite_index) < 2:
        return output, 0
    minimum_fraction = float(
        get_nested(config, "pairwise_gr.circular_control.minimum_rotation_fraction")
    )
    minimum = max(int(math.ceil(minimum_fraction * len(finite_index))), 1)
    maximum = max(len(finite_index) - minimum, minimum)
    if maximum >= len(finite_index):
        maximum = len(finite_index) - 1
    key = f"exp406::circular::{well_id}::{block_id}".encode()
    raw = int(hashlib.sha256(key).hexdigest()[:16], 16)
    span = max(maximum - minimum + 1, 1)
    offset = minimum + raw % span
    output[finite_index] = np.roll(array[finite_index], offset)
    return output, int(offset)


def interpolation_profile(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    part = frame.loc[
        np.isfinite(frame["base_tvt"]) & np.isfinite(frame["GR"]),
        ["base_tvt", "GR"],
    ].copy()
    if part.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    part = (
        part.groupby("base_tvt", sort=True, as_index=False)["GR"]
        .median()
        .sort_values("base_tvt", kind="mergesort")
    )
    return (
        part["base_tvt"].to_numpy(dtype=float),
        part["GR"].to_numpy(dtype=float),
    )


def nearest_block_id(blocks: Sequence[BlockSlice], tvt: float) -> int:
    if not blocks:
        return -1
    return min(
        blocks,
        key=lambda block: (abs(block.center_tvt - tvt), block.block_id),
    ).block_id


# %% [markdown]
# ## 5. Pairwise edges and deterministic circular control

# %%
def shift_grid(config: Mapping[str, Any]) -> np.ndarray:
    section = get_nested(config, "pairwise_gr.shift_grid_ft")
    values = np.arange(
        float(section["minimum"]),
        float(section["maximum"]) + 0.5 * float(section["step"]),
        float(section["step"]),
        dtype=float,
    )
    expected = int(section["expected_states"])
    if len(values) != expected:
        raise ValueError(f"shift grid has {len(values)} states, expected {expected}")
    return values


def build_pairwise_edges(
    *,
    fold: int,
    contexts: Mapping[str, pd.DataFrame],
    target_wells: Sequence[str],
    donor_wells: Sequence[str],
    typewell_by_well: Mapping[str, str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_set = set(map(str, target_wells))
    donor_set = set(map(str, donor_wells))
    if target_set.intersection(donor_set):
        raise ValueError("target wells cannot appear in the donor graph")
    maximum_donors = int(
        get_nested(config, "pairwise_gr.donor_pool.maximum_unique_wells")
    )
    maximum_edges = int(
        get_nested(
            config,
            "pairwise_gr.edge_selection.maximum_edges_per_target_block",
        )
    )
    minimum_ncc = float(
        get_nested(config, "pairwise_gr.edge_selection.minimum_ncc")
    )
    shifts = shift_grid(config)
    blocks_by_well = {
        well: block_slices(frame, config)
        for well, frame in sorted(contexts.items())
    }
    profiles = {
        well: interpolation_profile(contexts[well])
        for well in sorted(donor_set)
    }
    block_rows: list[dict[str, Any]] = []
    for well, blocks in sorted(blocks_by_well.items()):
        frame = contexts[well]
        for block in blocks:
            block_rows.append(
                {
                    "fold": int(fold),
                    "well_id": well,
                    "role": "outer_valid" if well in target_set else "outer_train",
                    "block_id": int(block.block_id),
                    "start": int(block.start),
                    "stop": int(block.stop),
                    "rows": int(block.stop - block.start),
                    "center_row_idx": int(block.center_row_idx),
                    "center_tvt": float(block.center_tvt),
                    "last_finite_prefix_row": int(
                        np.flatnonzero(
                            np.isfinite(frame["TVT_input"].to_numpy(dtype=float))
                        )[-1]
                    )
                    if "TVT_input" in frame and frame["TVT_input"].notna().any()
                    else -1,
                }
            )

    edge_rows: list[dict[str, Any]] = []
    rejection: defaultdict[tuple[str, str], int] = defaultdict(int)
    query_wells = sorted(contexts)
    for query_well in query_wells:
        query = contexts[query_well]
        donors = select_donor_wells(
            query_well,
            sorted(donor_set),
            contexts,
            typewell_by_well,
            maximum_donors,
        )
        if not donors:
            rejection[(query_well, "no_donor_wells")] += (
                len(blocks_by_well[query_well]) * len(shifts)
            )
            continue
        query_base = query["base_tvt"].to_numpy(dtype=float)
        query_gr_all = query["GR"].to_numpy(dtype=float)
        for block in blocks_by_well[query_well]:
            query_base_block = query_base[block.start : block.stop]
            query_gr = query_gr_all[block.start : block.stop]
            candidates: list[dict[str, Any]] = []
            for donor_well in donors:
                donor_tvt, donor_gr = profiles[donor_well]
                if len(donor_tvt) < 2:
                    rejection[(query_well, "insufficient_finite_pairs")] += len(shifts)
                    continue
                for shift_ft in shifts:
                    candidate_tvt = query_base_block + shift_ft
                    finite_progress = candidate_tvt[np.isfinite(candidate_tvt)]
                    if (
                        len(finite_progress) < 2
                        or float(np.median(np.diff(finite_progress))) <= 0.0
                    ):
                        rejection[(query_well, "nonpositive_local_tvt_progress")] += 1
                        continue
                    sampled = np.interp(
                        candidate_tvt,
                        donor_tvt,
                        donor_gr,
                        left=np.nan,
                        right=np.nan,
                    )
                    real = morphology_score(query_gr, sampled, config)
                    if not real["eligible"]:
                        reason = (
                            "insufficient_finite_pairs"
                            if int(real["finite_pairs"])
                            < int(get_nested(config, "pairwise_gr.minimum_finite_pairs"))
                            else "insufficient_pair_fraction"
                        )
                        rejection[(query_well, reason)] += 1
                        continue
                    if not np.isfinite(real["score"]):
                        rejection[(query_well, "nonfinite_ncc")] += 1
                        continue
                    if float(real["score"]) < minimum_ncc:
                        rejection[(query_well, "ncc_below_minimum")] += 1
                        continue
                    circular_gr, rotation = stable_circular_control(
                        sampled,
                        query_well,
                        block.block_id,
                        config,
                    )
                    circular = morphology_score(query_gr, circular_gr, config)
                    matched_tvt = float(block.center_tvt + shift_ft)
                    donor_block_id = nearest_block_id(
                        blocks_by_well[donor_well],
                        matched_tvt,
                    )
                    if donor_block_id < 0:
                        rejection[(query_well, "nonfinite_relative_offset")] += 1
                        continue
                    candidates.append(
                        {
                            "fold": int(fold),
                            "source_well_id": query_well,
                            "target_well_id": donor_well,
                            "source_role": (
                                "outer_valid" if query_well in target_set else "outer_train"
                            ),
                            "target_role": "outer_train",
                            "source_block_id": int(block.block_id),
                            "target_block_id": int(donor_block_id),
                            "source_center_row_idx": int(block.center_row_idx),
                            "source_center_tvt": float(block.center_tvt),
                            "matched_donor_tvt": matched_tvt,
                            "shift_ft": float(shift_ft),
                            "relative_offset_ft": float(-shift_ft),
                            "real_ncc": float(real["score"]),
                            "circular_ncc": (
                                float(circular["score"])
                                if np.isfinite(circular["score"])
                                else math.nan
                            ),
                            "real_minus_circular_ncc": (
                                float(real["score"] - circular["score"])
                                if np.isfinite(circular["score"])
                                else math.nan
                            ),
                            "finite_pairs": int(real["finite_pairs"]),
                            "pair_fraction": float(real["pair_fraction"]),
                            "circular_rotation": int(rotation),
                            "same_typewell": bool(
                                typewell_by_well.get(query_well)
                                == typewell_by_well.get(donor_well)
                            ),
                        }
                    )
            candidates.sort(
                key=lambda row: (
                    -row["real_ncc"],
                    -row["finite_pairs"],
                    row["target_well_id"],
                    row["shift_ft"],
                )
            )
            for rank, row in enumerate(candidates[:maximum_edges], start=1):
                row = dict(row)
                row["edge_rank"] = rank
                row["edge_id"] = (
                    f"f{fold}:{row['source_well_id']}:{row['source_block_id']}"
                    f"->{row['target_well_id']}:{row['target_block_id']}:r{rank}"
                )
                edge_rows.append(row)
            rejection[(query_well, "retained")] += min(len(candidates), maximum_edges)
            rejection[(query_well, "topk_pruned")] += max(
                len(candidates) - maximum_edges,
                0,
            )

    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges = edges.sort_values(list(EDGE_SORT_COLUMNS), kind="mergesort")
        if edges["edge_id"].duplicated().any():
            raise ValueError("pairwise edge ids must be unique")
        if edges["target_well_id"].isin(target_set).any():
            raise ValueError("outer-valid target appeared on the donor side")
    funnel = pd.DataFrame(
        [
            {
                "fold": int(fold),
                "well_id": well,
                "reason": reason,
                "count": int(count),
            }
            for (well, reason), count in sorted(rejection.items())
        ]
    )
    block_frame = pd.DataFrame(block_rows).sort_values(
        list(BLOCK_SORT_COLUMNS),
        kind="mergesort",
    )
    return edges.reset_index(drop=True), funnel.reset_index(drop=True), block_frame


# %% [markdown]
# ## 6. Fundamental cycles and Huber IRLS loop closure

# %%
def node_key(well_id: str, block_id: int) -> str:
    return f"{well_id}::{int(block_id):06d}"


class UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right
        return True


def edge_nodes(edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    source = np.array(
        [
            node_key(well, block)
            for well, block in zip(
                edges["source_well_id"],
                edges["source_block_id"],
                strict=True,
            )
        ],
        dtype=object,
    )
    target = np.array(
        [
            node_key(well, block)
            for well, block in zip(
                edges["target_well_id"],
                edges["target_block_id"],
                strict=True,
            )
        ],
        dtype=object,
    )
    return source, target


def connected_components(
    source_nodes: Sequence[str],
    target_nodes: Sequence[str],
) -> tuple[dict[str, int], list[list[str]]]:
    adjacency: defaultdict[str, list[str]] = defaultdict(list)
    for left, right in zip(source_nodes, target_nodes, strict=True):
        adjacency[str(left)].append(str(right))
        adjacency[str(right)].append(str(left))
    unseen = set(adjacency)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        members: list[str] = []
        while queue:
            node = queue.popleft()
            members.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(members))
    components.sort(key=lambda values: values[0])
    lookup = {
        node: component_id
        for component_id, members in enumerate(components)
        for node in members
    }
    return lookup, components


def tree_path(
    start: str,
    stop: str,
    adjacency: Mapping[str, Sequence[tuple[str, int, int]]],
) -> list[tuple[int, int]]:
    queue = deque([start])
    previous: dict[str, tuple[str, int, int] | None] = {start: None}
    while queue and stop not in previous:
        node = queue.popleft()
        for neighbor, edge_index, direction in sorted(
            adjacency.get(node, []),
            key=lambda item: (item[0], item[1], item[2]),
        ):
            if neighbor in previous:
                continue
            previous[neighbor] = (node, edge_index, direction)
            queue.append(neighbor)
    if stop not in previous:
        raise ValueError(f"tree path does not connect {start} to {stop}")
    output: list[tuple[int, int]] = []
    node = stop
    while node != start:
        step = previous[node]
        if step is None:
            raise AssertionError("unexpected empty predecessor")
        parent, edge_index, direction_parent_to_node = step
        output.append((edge_index, direction_parent_to_node))
        node = parent
    return list(reversed(output))


def fundamental_cycle_spec(
    edges: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, int], list[list[str]]]:
    if edges.empty:
        return [], {}, []
    source_nodes, target_nodes = edge_nodes(edges)
    all_nodes = sorted(set(source_nodes).union(target_nodes))
    component_by_node, components = connected_components(source_nodes, target_nodes)
    union = UnionFind(all_nodes)
    tree_adjacency: defaultdict[str, list[tuple[str, int, int]]] = defaultdict(list)
    non_tree: list[int] = []
    for edge_index, (source, target) in enumerate(
        zip(source_nodes, target_nodes, strict=True)
    ):
        source = str(source)
        target = str(target)
        if union.union(source, target):
            tree_adjacency[source].append((target, edge_index, 1))
            tree_adjacency[target].append((source, edge_index, -1))
        else:
            non_tree.append(edge_index)
    cycles: list[dict[str, Any]] = []
    for cycle_id, edge_index in enumerate(non_tree):
        source = str(source_nodes[edge_index])
        target = str(target_nodes[edge_index])
        path = tree_path(source, target, tree_adjacency)
        cycles.append(
            {
                "cycle_id": int(cycle_id),
                "component_id": int(component_by_node[source]),
                "tree_path": path,
                "non_tree_edge_index": int(edge_index),
            }
        )
    return cycles, component_by_node, components


def solve_loop_closed_graph(
    edges: pd.DataFrame,
    blocks: pd.DataFrame,
    target_wells: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if edges.empty:
        empty_gauge = pd.DataFrame(
            columns=[
                "fold",
                "well_id",
                "block_id",
                "node_id",
                "component_id",
                "role",
                "base_tvt",
                "solved_offset_ft",
                "finite_gauge",
                "connected_target",
                "anchor_kind",
            ]
        )
        return edges.copy(), empty_gauge, {
            "cycles": pd.DataFrame(),
            "component_count": 0,
            "fundamental_cycles": 0,
        }
    ordered = edges.sort_values(list(EDGE_SORT_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )
    source_nodes, target_nodes = edge_nodes(ordered)
    nodes = sorted(set(source_nodes).union(target_nodes))
    node_index = {node: index for index, node in enumerate(nodes)}
    row_index = np.repeat(np.arange(len(ordered)), 2)
    column_index = np.array(
        [
            value
            for source, target in zip(source_nodes, target_nodes, strict=True)
            for value in (node_index[str(source)], node_index[str(target)])
        ],
        dtype=int,
    )
    incidence = sparse.csr_matrix(
        (
            np.tile(np.array([-1.0, 1.0]), len(ordered)),
            (row_index, column_index),
        ),
        shape=(len(ordered), len(nodes)),
    )
    measurement = ordered["relative_offset_ft"].to_numpy(dtype=float)
    base_weight = (
        np.clip(ordered["real_ncc"].to_numpy(dtype=float), 0.01, 1.0)
        * np.clip(ordered["pair_fraction"].to_numpy(dtype=float), 0.01, 1.0)
    )
    cycles, component_by_node, components = fundamental_cycle_spec(ordered)

    block_index = blocks.copy()
    block_index["node_id"] = [
        node_key(well, block)
        for well, block in zip(
            block_index["well_id"],
            block_index["block_id"],
            strict=True,
        )
    ]
    block_index = block_index.loc[block_index["node_id"].isin(nodes)].copy()
    role_by_node = block_index.set_index("node_id")["role"].to_dict()
    last_prefix_by_well = (
        block_index.groupby("well_id", sort=True)["last_finite_prefix_row"].max().to_dict()
    )
    anchors: list[tuple[str, str]] = []
    for component_id, members in enumerate(components):
        donors = sorted(
            node
            for node in members
            if role_by_node.get(node) == "outer_train"
        )
        chosen = donors[0] if donors else members[0]
        anchors.append((chosen, f"component_{component_id}_minimum_donor"))
    target_set = set(map(str, target_wells))
    for well in sorted(target_set):
        rows = block_index.loc[block_index["well_id"].eq(well)]
        if rows.empty:
            continue
        last_prefix = int(last_prefix_by_well.get(well, -1))
        if last_prefix < 0:
            continue
        chosen_row = rows.iloc[
            np.argmin(
                np.abs(
                    rows["center_row_idx"].to_numpy(dtype=int) - last_prefix
                )
            )
        ]
        anchors.append(
            (
                str(chosen_row["node_id"]),
                "target_visible_prefix_last_finite",
            )
        )
    deduplicated_anchors: dict[str, list[str]] = defaultdict(list)
    for node, kind in anchors:
        deduplicated_anchors[node].append(kind)
    anchor_nodes = sorted(deduplicated_anchors)
    anchor_matrix = sparse.csr_matrix(
        (
            np.ones(len(anchor_nodes)),
            (
                np.arange(len(anchor_nodes)),
                [node_index[node] for node in anchor_nodes],
            ),
        ),
        shape=(len(anchor_nodes), len(nodes)),
    )
    anchor_weight = float(
        get_nested(config, "loop_closure.component_anchor.weight", 1.0e6)
    )
    ridge = float(get_nested(config, "loop_closure.ridge"))
    huber_delta = float(get_nested(config, "loop_closure.huber_delta_ft"))
    iterations = int(get_nested(config, "loop_closure.iterations"))
    edge_weight = base_weight.copy()
    solution = np.zeros(len(nodes), dtype=float)
    for _ in range(iterations):
        weighted_incidence = incidence.multiply(np.sqrt(edge_weight)[:, None])
        weighted_measurement = measurement * np.sqrt(edge_weight)
        design = sparse.vstack(
            [
                weighted_incidence,
                anchor_matrix * math.sqrt(anchor_weight),
                sparse.eye(len(nodes), format="csr") * math.sqrt(ridge),
            ],
            format="csr",
        )
        response = np.concatenate(
            [
                weighted_measurement,
                np.zeros(len(anchor_nodes), dtype=float),
                np.zeros(len(nodes), dtype=float),
            ]
        )
        solution = lsqr(
            design,
            response,
            atol=1.0e-12,
            btol=1.0e-12,
            iter_lim=max(1000, len(nodes) * 5),
        )[0]
        residual = incidence @ solution - measurement
        huber_weight = np.ones(len(residual), dtype=float)
        outside = np.abs(residual) > huber_delta
        huber_weight[outside] = huber_delta / np.maximum(
            np.abs(residual[outside]),
            1.0e-12,
        )
        edge_weight = base_weight * huber_weight

    solved_measurement = incidence @ solution
    solved_edges = ordered.copy()
    solved_edges["solved_relative_offset_ft"] = solved_measurement
    solved_edges["edge_residual_ft"] = solved_measurement - measurement
    solved_edges["irls_weight"] = edge_weight
    cycle_rows: list[dict[str, Any]] = []
    for cycle in cycles:
        path = cycle["tree_path"]
        non_tree = int(cycle["non_tree_edge_index"])
        raw_path = sum(
            direction * measurement[edge_index]
            for edge_index, direction in path
        )
        solved_path = sum(
            direction * solved_measurement[edge_index]
            for edge_index, direction in path
        )
        raw_residual = float(raw_path - measurement[non_tree])
        solved_residual = float(solved_path - solved_measurement[non_tree])
        cycle_rows.append(
            {
                "fold": int(ordered.iloc[non_tree]["fold"]),
                "component_id": int(cycle["component_id"]),
                "cycle_id": int(cycle["cycle_id"]),
                "non_tree_edge_id": str(ordered.iloc[non_tree]["edge_id"]),
                "tree_edge_count": int(len(path)),
                "tree_edge_ids": json.dumps(
                    [
                        {
                            "edge_id": str(ordered.iloc[edge_index]["edge_id"]),
                            "direction": int(direction),
                        }
                        for edge_index, direction in path
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "raw_cycle_residual_ft": raw_residual,
                "solved_cycle_residual_ft": solved_residual,
                "raw_cycle_residual_abs_ft": abs(raw_residual),
                "solved_cycle_residual_abs_ft": abs(solved_residual),
            }
        )
    cycle_frame = pd.DataFrame(cycle_rows)
    if not cycle_frame.empty:
        cycle_frame = cycle_frame.sort_values(
            list(CYCLE_SORT_COLUMNS),
            kind="mergesort",
        ).reset_index(drop=True)

    block_lookup = block_index.set_index("node_id")
    gauge_rows: list[dict[str, Any]] = []
    for node in nodes:
        row = block_lookup.loc[node]
        well = str(row["well_id"])
        offset = float(solution[node_index[node]])
        gauge_rows.append(
            {
                "fold": int(row["fold"]),
                "well_id": well,
                "block_id": int(row["block_id"]),
                "node_id": node,
                "component_id": int(component_by_node[node]),
                "role": str(row["role"]),
                "center_row_idx": int(row["center_row_idx"]),
                "base_tvt": float(row["center_tvt"]),
                "solved_offset_ft": offset,
                "finite_gauge": bool(np.isfinite(offset)),
                "connected_target": bool(well in target_set),
                "anchor_kind": "|".join(deduplicated_anchors.get(node, [])),
            }
        )
    gauge = pd.DataFrame(gauge_rows).sort_values(
        list(GAUGE_SORT_COLUMNS),
        kind="mergesort",
    )
    manifest = {
        "cycles": cycle_frame,
        "component_count": int(len(components)),
        "fundamental_cycles": int(len(cycle_frame)),
        "edge_residual_abs_p95_ft": float(
            np.quantile(np.abs(solved_edges["edge_residual_ft"]), 0.95)
        ),
        "anchor_nodes": int(len(anchor_nodes)),
    }
    return solved_edges, gauge.reset_index(drop=True), manifest


# %% [markdown]
# ## 7. Fixed16 target-free graph, freeze, and technical gates

# %%
def read_observable_xy(path: Path) -> tuple[float, float]:
    frame = pd.read_csv(path, usecols=["X", "Y"])
    return (
        float(np.nanmedian(frame["X"].to_numpy(dtype=float))),
        float(np.nanmedian(frame["Y"].to_numpy(dtype=float))),
    )


def xy_context(value: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame({"X": [value[0]], "Y": [value[1]]})


def target_row_coverage(
    target: pd.DataFrame,
    gauge: pd.DataFrame,
) -> tuple[int, int]:
    well = str(target["well_id"].iloc[0])
    nodes = gauge.loc[
        gauge["well_id"].eq(well) & gauge["finite_gauge"].astype(bool)
    ].sort_values("center_row_idx", kind="mergesort")
    eligible = np.isfinite(target["base_tvt"].to_numpy(dtype=float))
    denominator = int(eligible.sum())
    if len(nodes) < 2:
        return 0, denominator
    row_idx = target["row_idx"].to_numpy(dtype=float)
    correction = np.interp(
        row_idx,
        nodes["center_row_idx"].to_numpy(dtype=float),
        nodes["solved_offset_ft"].to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )
    finite = eligible & np.isfinite(correction)
    return int(finite.sum()), denominator


def run_fold_target_free(
    *,
    fold: int,
    target_wells: Sequence[str],
    file_by_well: Mapping[str, Path],
    xy_by_well: Mapping[str, tuple[float, float]],
    fold_by_well: Mapping[str, int],
    exp226_oof: pd.DataFrame,
    typewell_by_well: Mapping[str, str],
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_wells = sorted(
        well for well, assigned in fold_by_well.items() if int(assigned) != int(fold)
    )
    ledger.validate_disjoint(source_wells, target_wells)
    metadata_contexts = {
        well: xy_context(xy_by_well[well])
        for well in sorted(set(source_wells).union(target_wells))
    }
    maximum = int(get_nested(config, "pairwise_gr.donor_pool.maximum_unique_wells"))
    selected_by_target = {
        well: select_donor_wells(
            well,
            source_wells,
            metadata_contexts,
            typewell_by_well,
            maximum,
        )
        for well in sorted(target_wells)
    }
    if any(len(values) != maximum for values in selected_by_target.values()):
        raise ValueError("each fixed16 target must have exactly 12 donor wells")
    graph_donors = sorted(
        {
            donor
            for values in selected_by_target.values()
            for donor in values
        }
    )
    contexts: dict[str, pd.DataFrame] = {}
    for well in sorted(target_wells):
        contexts[well] = read_target_safe_well(
            file_by_well[well],
            fold,
            exp226_oof,
            ledger,
        )
    for well in graph_donors:
        contexts[well] = read_donor_well(file_by_well[well], fold, ledger)
    edges, funnel, blocks = build_pairwise_edges(
        fold=fold,
        contexts=contexts,
        target_wells=target_wells,
        donor_wells=graph_donors,
        typewell_by_well=typewell_by_well,
        config=config,
    )
    solved_edges, gauge, solver = solve_loop_closed_graph(
        edges,
        blocks,
        target_wells,
        config,
    )
    coverage_rows: list[dict[str, Any]] = []
    for well in sorted(target_wells):
        finite_rows, eligible_rows = target_row_coverage(contexts[well], gauge)
        target_blocks = blocks.loc[
            blocks["well_id"].eq(well) & blocks["role"].eq("outer_valid")
        ]
        queried = (
            solved_edges.loc[solved_edges["source_well_id"].eq(well), "source_block_id"]
            .nunique()
            if not solved_edges.empty
            else 0
        )
        connected = bool(
            not gauge.loc[
                gauge["well_id"].eq(well) & gauge["finite_gauge"].astype(bool)
            ].empty
        )
        coverage_rows.append(
            {
                "fold": int(fold),
                "well_id": well,
                "target_blocks": int(len(target_blocks)),
                "queried_blocks": int(queried),
                "query_coverage": float(queried / max(len(target_blocks), 1)),
                "connected_target": connected,
                "finite_rows": int(finite_rows),
                "eligible_rows": int(eligible_rows),
                "finite_rgt_coverage": float(finite_rows / max(eligible_rows, 1)),
                "donor_count": int(len(selected_by_target[well])),
                "donor_wells": "|".join(selected_by_target[well]),
            }
        )
    return {
        "fold": int(fold),
        "target_wells": list(sorted(target_wells)),
        "graph_donors": graph_donors,
        "contexts": contexts,
        "edges": solved_edges,
        "funnel": funnel,
        "blocks": blocks,
        "cycles": solver["cycles"],
        "gauge": gauge,
        "coverage": pd.DataFrame(coverage_rows),
        "solver_manifest": {
            key: value for key, value in solver.items() if key != "cycles"
        },
    }


def concat_frames(
    frames: Sequence[pd.DataFrame],
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    output = pd.concat(nonempty, ignore_index=True)
    keys = [column for column in sort_columns if column in output.columns]
    if keys:
        output = output.sort_values(keys, kind="mergesort")
    return output.reset_index(drop=True)


def freeze_target_free_frames(
    frames: Mapping[str, pd.DataFrame],
    ledger: RoleReadLedger,
) -> dict[str, str]:
    sort_contract = {
        "fixed16_manifest": ("fold", "well_id"),
        "pairwise_edges": EDGE_SORT_COLUMNS,
        "edge_rejection_funnel": ("fold", "well_id", "reason"),
        "cycle_basis": CYCLE_SORT_COLUMNS,
        "loop_closed_gauge": GAUGE_SORT_COLUMNS,
        "negative_control_metrics": ("scope",),
    }
    hashes = {
        name: frame_content_sha256(frame, sort_contract[name])
        for name, frame in frames.items()
        if name in sort_contract
    }
    ledger.mark_frozen(hashes)
    return hashes


def negative_control_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", edges)]
    if not edges.empty:
        scopes.extend(
            (f"fold_{int(fold)}", part)
            for fold, part in edges.groupby("fold", sort=True)
        )
    for scope, part in scopes:
        if part.empty or not {"real_ncc", "circular_ncc"}.issubset(part.columns):
            finite = pd.DataFrame(columns=["real_ncc", "circular_ncc"])
        else:
            finite = part.loc[
                np.isfinite(part["real_ncc"])
                & np.isfinite(part["circular_ncc"])
            ]
        rows.append(
            {
                "scope": scope,
                "fold": (
                    int(scope.split("_", 1)[1]) if scope.startswith("fold_") else -1
                ),
                "edges": int(len(finite)),
                "real_ncc_mean": (
                    float(finite["real_ncc"].mean()) if len(finite) else math.nan
                ),
                "circular_ncc_mean": (
                    float(finite["circular_ncc"].mean()) if len(finite) else math.nan
                ),
                "real_minus_circular_ncc": (
                    float(
                        (
                            finite["real_ncc"] - finite["circular_ncc"]
                        ).mean()
                    )
                    if len(finite)
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_target_free_gates(
    *,
    fixed16_manifest: pd.DataFrame,
    edges: pd.DataFrame,
    cycles: pd.DataFrame,
    ledger: RoleReadLedger,
    elapsed_seconds: float,
    observed_peak_rss_gb: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical = get_nested(config, "gates.technical")
    negative = negative_control_metrics(edges)
    target_wells = int(fixed16_manifest["well_id"].nunique())
    reporting_folds = int(fixed16_manifest["fold"].nunique())
    target_blocks = int(fixed16_manifest["target_blocks"].sum())
    queried_blocks = int(fixed16_manifest["queried_blocks"].sum())
    eligible_rows = int(fixed16_manifest["eligible_rows"].sum())
    finite_rows = int(fixed16_manifest["finite_rows"].sum())
    graph_query_coverage = queried_blocks / max(target_blocks, 1)
    connected_coverage = float(fixed16_manifest["connected_target"].mean())
    finite_coverage = finite_rows / max(eligible_rows, 1)
    raw_cycle_p95 = (
        float(cycles["raw_cycle_residual_abs_ft"].quantile(0.95))
        if not cycles.empty
        else math.inf
    )
    solved_cycle_p95 = (
        float(cycles["solved_cycle_residual_abs_ft"].quantile(0.95))
        if not cycles.empty
        else math.inf
    )
    reduction = (
        float((raw_cycle_p95 - solved_cycle_p95) / raw_cycle_p95)
        if np.isfinite(raw_cycle_p95) and raw_cycle_p95 > 0
        else float(solved_cycle_p95 == 0.0)
    )
    projected_seconds = elapsed_seconds * int(
        get_nested(config, "validation.expected_wells")
    ) / max(target_wells, 1)
    ledger_summary = ledger.summary()
    observed = {
        "target_wells": target_wells,
        "reporting_folds": reporting_folds,
        "graph_query_coverage": float(graph_query_coverage),
        "connected_target_coverage": connected_coverage,
        "finite_loop_closed_rgt_coverage": float(finite_coverage),
        "fundamental_cycles": int(len(cycles)),
        "raw_cycle_residual_p95_ft": raw_cycle_p95,
        "solved_cycle_residual_p95_ft": solved_cycle_p95,
        "cycle_residual_p95_reduction_fraction": reduction,
        "projected_full_runtime_seconds": float(projected_seconds),
        "projected_peak_rss_gb": float(observed_peak_rss_gb),
        **ledger_summary,
    }
    checks = {
        "target_wells": target_wells == int(technical["target_wells"]),
        "fixed16_match_exp386": bool(technical["fixed16_match_exp386"]),
        "reporting_folds_present": reporting_folds
        == int(technical["reporting_folds_present"]),
        "source_target_overlap": ledger_summary["source_target_overlap"]
        <= int(technical["source_target_overlap_max"]),
        "suffix_truth_reads_before_freeze": ledger_summary[
            "suffix_truth_reads_before_freeze"
        ]
        <= int(technical["suffix_truth_reads_before_freeze_max"]),
        "target_formation_reads_before_freeze": ledger_summary[
            "target_formation_reads_before_freeze"
        ]
        <= int(technical["target_formation_reads_before_freeze_max"]),
        "hidden_role_reads_before_freeze": ledger_summary[
            "hidden_role_reads_before_freeze"
        ]
        <= int(technical["hidden_role_reads_before_freeze_max"]),
        "graph_query_coverage": graph_query_coverage
        >= float(technical["graph_query_coverage_min"]),
        "connected_target_coverage": connected_coverage
        >= float(technical["connected_target_coverage_min"]),
        "finite_loop_closed_rgt_coverage": finite_coverage
        >= float(technical["finite_loop_closed_rgt_coverage_min"]),
        "fundamental_cycles": len(cycles)
        >= int(technical["fundamental_cycles_min"]),
        "solved_cycle_residual_p95": solved_cycle_p95
        <= float(technical["solved_cycle_residual_p95_ft_max"]),
        "cycle_residual_reduction": reduction
        >= float(technical["cycle_residual_p95_reduction_fraction_min"]),
        "projected_full_runtime": projected_seconds
        <= float(technical["projected_full_runtime_seconds_max"]),
        "projected_peak_rss": observed_peak_rss_gb
        <= float(technical["projected_peak_rss_gb_max"]),
    }
    scientific = get_nested(config, "gates.scientific")
    pooled_negative = negative.loc[negative["scope"].eq("pooled")]
    pooled_gain = (
        float(pooled_negative.iloc[0]["real_minus_circular_ncc"])
        if len(pooled_negative)
        else -math.inf
    )
    fold_negative = negative.loc[negative["scope"].str.startswith("fold_")]
    better_folds = int((fold_negative["real_minus_circular_ncc"] > 0.0).sum())
    negative_checks = {
        "real_ncc_gain_vs_circular": pooled_gain
        >= float(scientific["real_ncc_gain_vs_circular_min"]),
        "real_better_than_circular_folds": better_folds
        >= int(scientific["real_better_than_circular_fold_count_min"]),
    }
    return {
        "technical_pass": bool(all(checks.values())),
        "technical_checks": checks,
        "target_free_observed": observed,
        "negative_control_pass": bool(all(negative_checks.values())),
        "negative_control_checks": negative_checks,
        "negative_control_observed": {
            "pooled_real_minus_circular_ncc": pooled_gain,
            "better_fold_count": better_folds,
        },
        "negative_control_metrics": negative,
    }


# %% [markdown]
# ## 8. Fixed16-only exp226 K16 geometry control and prefix readout

# %%
@dataclass(frozen=True)
class K16Params:
    theta0: float
    k_segments: int
    local_linear_k: int
    local_linear_bandwidth: float
    local_linear_ridge: float
    smooth_rho: float
    gate: float
    field_min_proj: float
    kbins: tuple[float, ...]
    kappa_regimes: tuple[float, ...]
    rot_max_deg: float
    ancc_theta_bandwidth: float

    @property
    def n_bins(self) -> int:
        return len(self.kbins) - 1

    @property
    def kappa_dim(self) -> int:
        return 2 * self.n_bins + 2


@dataclass
class K16Well:
    wid: str
    wi: int
    s: int
    n: int
    ndz: np.ndarray
    anchor: float
    segid: np.ndarray
    mid: np.ndarray
    proj: np.ndarray
    az: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r0: np.ndarray | None = None
    anc: np.ndarray | None = None
    c_raw: np.ndarray | None = None
    c_sm: np.ndarray | None = None


@dataclass(frozen=True)
class K16Fields:
    f_raw: np.ndarray
    f_sm: np.ndarray
    surface_points: np.ndarray
    global_theta: float


def k16_params(config: Mapping[str, Any]) -> K16Params:
    raw = get_nested(
        config,
        "prefix_rolling_origin.exp226_geometry_control.k16_params",
    )
    return K16Params(
        theta0=float(raw["theta0"]),
        k_segments=int(raw["k_segments"]),
        local_linear_k=int(raw["local_linear_k"]),
        local_linear_bandwidth=float(raw["local_linear_bandwidth"]),
        local_linear_ridge=float(raw["local_linear_ridge"]),
        smooth_rho=float(raw["smooth_rho"]),
        gate=float(raw["gate"]),
        field_min_proj=float(raw["field_min_proj"]),
        kbins=tuple(map(float, raw["kbins"])),
        kappa_regimes=tuple(map(float, raw["kappa_regimes"])),
        rot_max_deg=float(raw["rot_max_deg"]),
        ancc_theta_bandwidth=float(raw["ancc_theta_bandwidth"]),
    )


def last_known_index(values: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(np.asarray(values, dtype=float)))
    if not len(finite):
        raise ValueError("well has no finite TVT_input")
    return int(finite[-1])


def k16_segment_geometry(
    x: np.ndarray,
    y: np.ndarray,
    s: int,
    n: int,
    params: K16Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_idx = np.arange(1, n + 1.0)
    segid = np.clip(
        np.searchsorted(edges[1:], step_idx, side="left"),
        0,
        params.k_segments - 1,
    )
    mid = np.empty((params.k_segments, 2), dtype=float)
    proj = np.empty(params.k_segments, dtype=float)
    az = np.empty(params.k_segments, dtype=float)
    theta = np.radians(params.theta0)
    last_idx = len(x) - 1
    for segment in range(params.k_segments):
        first = min(s + 1 + int(edges[segment]), last_idx)
        last_raw = s + 1 + max(
            int(edges[segment + 1]) - 1,
            int(edges[segment]),
        )
        last = min(max(last_raw, first), last_idx)
        az[segment] = np.arctan2(y[last] - y[first], x[last] - x[first])
        mid[segment] = ((x[first] + x[last]) / 2.0, (y[first] + y[last]) / 2.0)
        proj[segment] = np.cos(az[segment] - theta)
    return segid.astype(int), mid, proj, az


def k16_fit_coeffs(
    r0: np.ndarray,
    u: np.ndarray,
    n: int,
    params: K16Params,
    rho: float,
) -> np.ndarray:
    steps = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, params.k_segments + 1)
    phi = np.column_stack(
        [
            np.clip(
                steps - edges[index],
                0,
                edges[index + 1] - edges[index],
            )
            for index in range(params.k_segments)
        ]
    )
    matrix = phi.T @ phi
    if rho > 0:
        difference = np.diff(np.eye(params.k_segments), axis=0)
        scale = float(np.mean(np.diag(matrix))) if matrix.size else 1.0
        matrix = matrix + rho * max(scale, 1.0e-9) * difference.T @ difference
    response = phi.T @ (r0 - u)
    try:
        return np.linalg.solve(matrix, response)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(
            matrix + np.eye(params.k_segments) * 1.0e-9,
            response,
            rcond=None,
        )[0]


def load_k16_source_well(
    path: Path,
    wi: int,
    params: K16Params,
) -> K16Well:
    frame = pd.read_csv(
        path,
        usecols=["X", "Y", "Z", "TVT", "TVT_input", "ANCC"],
    )
    x = frame["X"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    z = frame["Z"].to_numpy(dtype=float)
    tvt = frame["TVT"].to_numpy(dtype=float)
    tvt_input = frame["TVT_input"].to_numpy(dtype=float)
    s = last_known_index(tvt_input)
    ndz = -np.diff(z)[s:]
    n = len(ndz)
    if n <= 0:
        raise ValueError(f"{well_id_from_path(path)} has no K16 suffix rows")
    u = np.cumsum(ndz)
    r0 = tvt[s + 1 :] - tvt[s]
    segid, mid, proj, az = k16_segment_geometry(x, y, s, n, params)
    return K16Well(
        wid=well_id_from_path(path),
        wi=int(wi),
        s=int(s),
        n=int(n),
        ndz=ndz,
        anchor=float(tvt[s]),
        segid=segid,
        mid=mid,
        proj=proj,
        az=az,
        x=x,
        y=y,
        z=z,
        r0=r0,
        anc=frame["ANCC"].to_numpy(dtype=float),
        c_raw=k16_fit_coeffs(r0, u, n, params, rho=0.0),
        c_sm=k16_fit_coeffs(r0, u, n, params, rho=params.smooth_rho),
    )


def make_k16_pseudo_target(
    frame: pd.DataFrame,
    pseudo_cut_row: int,
    heldout_rows: int,
    params: K16Params,
) -> K16Well:
    x = frame["X"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    z = frame["Z"].to_numpy(dtype=float)
    tvt_input = frame["TVT_input"].to_numpy(dtype=float)
    if pseudo_cut_row < 0 or not np.isfinite(tvt_input[pseudo_cut_row]):
        raise ValueError("pseudo-cut anchor must be finite")
    stop = pseudo_cut_row + 1 + heldout_rows
    if stop > len(frame):
        raise ValueError("pseudo-cut heldout window exceeds the raw well")
    ndz = -np.diff(z)[pseudo_cut_row : stop - 1]
    if len(ndz) != heldout_rows:
        raise ValueError("pseudo-cut geometry row count mismatch")
    segid, mid, proj, az = k16_segment_geometry(
        x,
        y,
        pseudo_cut_row,
        heldout_rows,
        params,
    )
    return K16Well(
        wid=str(frame["well_id"].iloc[0]),
        wi=-1,
        s=int(pseudo_cut_row),
        n=int(heldout_rows),
        ndz=ndz,
        anchor=float(tvt_input[pseudo_cut_row]),
        segid=segid,
        mid=mid,
        proj=proj,
        az=az,
        x=x,
        y=y,
        z=z,
    )


def build_k16_fields(
    wells: Sequence[K16Well],
    params: K16Params,
) -> K16Fields:
    def pack(attribute: str) -> np.ndarray:
        rows: list[tuple[float, float, float, float]] = []
        for well in wells:
            coefficients = getattr(well, attribute)
            if coefficients is None:
                continue
            for segment in range(params.k_segments):
                if abs(well.proj[segment]) > params.field_min_proj:
                    rows.append(
                        (
                            float(well.mid[segment, 0]),
                            float(well.mid[segment, 1]),
                            float(coefficients[segment] / well.proj[segment]),
                            float(well.wi),
                        )
                    )
        if not rows:
            raise ValueError("empty K16 donor field")
        return np.asarray(rows, dtype=float)

    surface_parts: list[np.ndarray] = []
    for well in wells:
        if well.anc is None:
            continue
        step = max(len(well.x) // 120, 1)
        part = np.column_stack(
            [
                well.x[::step],
                well.y[::step],
                well.anc[::step],
                np.full(len(well.anc[::step]), well.wi, dtype=float),
            ]
        )
        surface_parts.append(part)
    if not surface_parts:
        raise ValueError("empty K16 ANCC surface sample")
    surface = np.vstack(surface_parts)
    surface = surface[np.isfinite(surface[:, 2])]
    design = np.column_stack(
        [
            np.ones(len(surface)),
            surface[:, 0] - surface[:, 0].mean(),
            surface[:, 1] - surface[:, 1].mean(),
        ]
    )
    beta = np.linalg.lstsq(design, surface[:, 2], rcond=None)[0]
    return K16Fields(
        f_raw=pack("c_raw"),
        f_sm=pack("c_sm"),
        surface_points=surface,
        global_theta=float(np.arctan2(beta[2], beta[1])),
    )


def safe_nearest_indices(
    squared_distance: np.ndarray,
    candidates: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(candidates) == 0:
        return candidates
    selected = min(max(int(count), 1), len(candidates))
    return candidates[
        np.argpartition(squared_distance[candidates], selected - 1)[:selected]
    ]


def k16_local_linear(
    field_values: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    params: K16Params,
    minimum_distance: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    keep = field_values[:, 3] != own_wi
    x = field_values[keep, 0]
    y = field_values[keep, 1]
    value = field_values[keep, 2]
    prediction = np.empty(len(mid), dtype=float)
    donor_distance = np.empty(len(mid), dtype=float)
    for index, point in enumerate(mid):
        squared = (x - point[0]) ** 2 + (y - point[1]) ** 2
        candidates = (
            np.flatnonzero(squared >= minimum_distance**2)
            if minimum_distance
            else np.arange(len(squared))
        )
        selected = safe_nearest_indices(
            squared,
            candidates,
            params.local_linear_k,
        )
        if not len(selected):
            prediction[index] = float(np.median(value))
            donor_distance[index] = math.inf
            continue
        weight = np.exp(
            np.maximum(
                -squared[selected] / (2.0 * params.local_linear_bandwidth**2),
                -700.0,
            )
        )
        dx = (x[selected] - point[0]) / 1000.0
        dy = (y[selected] - point[1]) / 1000.0
        local_design = np.column_stack([np.ones(len(selected)), dx, dy])
        ridge = (
            params.local_linear_ridge
            * np.sum(weight)
            * np.diag([0.0, 1.0, 1.0])
        )
        matrix = (local_design * weight[:, None]).T @ local_design + ridge
        response = (local_design * weight[:, None]).T @ value[selected]
        try:
            prediction[index] = np.linalg.solve(matrix, response)[0]
        except np.linalg.LinAlgError:
            prediction[index] = np.linalg.lstsq(
                matrix + np.eye(3) * 1.0e-9,
                response,
                rcond=None,
            )[0][0]
        nearest = np.sort(squared[selected])[: min(15, len(selected))]
        donor_distance[index] = float(np.sqrt(np.median(nearest)))
    return prediction, donor_distance


def k16_kernel_mean(
    field_values: np.ndarray,
    own_wi: int,
    mid: np.ndarray,
    minimum_distance: float,
) -> np.ndarray:
    keep = field_values[:, 3] != own_wi
    x = field_values[keep, 0]
    y = field_values[keep, 1]
    value = field_values[keep, 2]
    output = np.empty(len(mid), dtype=float)
    for index, point in enumerate(mid):
        squared = (x - point[0]) ** 2 + (y - point[1]) ** 2
        candidates = (
            np.flatnonzero(squared >= minimum_distance**2)
            if minimum_distance
            else np.arange(len(squared))
        )
        selected = safe_nearest_indices(squared, candidates, 15)
        if not len(selected):
            output[index] = float(np.median(value))
            continue
        weight = np.exp(np.maximum(-squared[selected] / (2.0 * 500.0**2), -700))
        output[index] = float(np.sum(weight * value[selected]) / np.sum(weight))
    return output


def k16_theta_loc_at(
    surface: np.ndarray,
    mids: np.ndarray,
    own_wi: int,
    global_theta: float,
    params: K16Params,
) -> np.ndarray:
    output = np.empty(len(mids), dtype=float)
    bandwidth = params.ancc_theta_bandwidth
    for index, mid in enumerate(mids):
        squared = (surface[:, 0] - mid[0]) ** 2 + (surface[:, 1] - mid[1]) ** 2
        mask = (squared < (4 * bandwidth) ** 2) & (surface[:, 3] != own_wi)
        if int(mask.sum()) < 30:
            output[index] = global_theta
            continue
        weight = np.exp(-squared[mask] / (2 * bandwidth**2))
        x = surface[mask, 0] - mid[0]
        y = surface[mask, 1] - mid[1]
        z = surface[mask, 2]
        matrix = np.array(
            [
                [np.sum(weight), np.sum(weight * x), np.sum(weight * y)],
                [
                    np.sum(weight * x),
                    np.sum(weight * x * x),
                    np.sum(weight * x * y),
                ],
                [
                    np.sum(weight * y),
                    np.sum(weight * x * y),
                    np.sum(weight * y * y),
                ],
            ]
        )
        response = np.array(
            [np.sum(weight * z), np.sum(weight * x * z), np.sum(weight * y * z)]
        )
        try:
            beta = np.linalg.solve(matrix, response)
            output[index] = np.arctan2(beta[2], beta[1])
        except np.linalg.LinAlgError:
            output[index] = global_theta
    return output


def k16_committee_inputs(
    well: K16Well,
    fields: K16Fields,
    params: K16Params,
    minimum_distance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not (np.abs(well.proj) < params.gate).any():
        return None
    theta = k16_theta_loc_at(
        fields.surface_points,
        well.mid,
        well.wi,
        fields.global_theta,
        params,
    )
    rotation = np.degrees(
        np.abs(
            np.arctan2(
                np.sin(theta - np.radians(params.theta0)),
                np.cos(theta - np.radians(params.theta0)),
            )
        )
    )
    kernel = k16_kernel_mean(
        fields.f_raw,
        well.wi,
        well.mid,
        minimum_distance,
    )
    local = kernel * np.cos(well.az - theta)
    mask = (
        (np.abs(well.proj[well.segid]) < params.gate)
        & (rotation < params.rot_max_deg)[well.segid]
    )
    return local, mask


def k16_build_columns(
    well: K16Well,
    raw: np.ndarray,
    smooth: np.ndarray,
    donor_distance: np.ndarray,
    params: K16Params,
    substitute: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    gated = np.abs(well.proj[well.segid]) < params.gate
    raw_step = np.where(gated, 0.0, well.ndz + raw[well.segid])
    smooth_step = np.where(gated, 0.0, well.ndz + smooth[well.segid])
    bucket = np.digitize(donor_distance, params.kbins[1:-1])[well.segid]
    position = (well.segid + 0.5) / params.k_segments
    columns = [
        np.cumsum(np.where(bucket == index, raw_step, 0.0))
        for index in range(params.n_bins)
    ]
    columns.extend(
        np.cumsum(np.where(bucket == index, smooth_step, 0.0))
        for index in range(params.n_bins)
    )
    columns.append(
        np.cumsum(0.5 * (raw_step + smooth_step) * np.sqrt(position))
    )
    if substitute is None:
        columns.append(np.zeros(len(well.ndz), dtype=float))
    else:
        columns.append(
            np.cumsum(
                np.where(
                    substitute[1],
                    well.ndz + substitute[0][well.segid],
                    0.0,
                )
            )
        )
    return np.column_stack(columns)


def k16_well_design(
    well: K16Well,
    fields: K16Fields,
    params: K16Params,
    minimum_distance: float = 0.0,
) -> np.ndarray:
    raw_field, donor_distance = k16_local_linear(
        fields.f_raw,
        well.wi,
        well.mid,
        params,
        minimum_distance,
    )
    smooth_field, _ = k16_local_linear(
        fields.f_sm,
        well.wi,
        well.mid,
        params,
        minimum_distance,
    )
    substitute = k16_committee_inputs(
        well,
        fields,
        params,
        minimum_distance,
    )
    return k16_build_columns(
        well,
        raw_field * well.proj,
        smooth_field * well.proj,
        donor_distance,
        params,
        substitute,
    )


def fit_k16_kappa(
    wells: Sequence[K16Well],
    fields: K16Fields,
    params: K16Params,
) -> np.ndarray:
    matrix = np.zeros((params.kappa_dim, params.kappa_dim), dtype=float)
    response = np.zeros(params.kappa_dim, dtype=float)
    for regime in params.kappa_regimes:
        for well in wells:
            if well.r0 is None:
                continue
            design = k16_well_design(
                well,
                fields,
                params,
                minimum_distance=regime,
            )
            matrix += design.T @ design
            response += design.T @ well.r0
    return np.linalg.lstsq(matrix, response, rcond=None)[0]


def build_exp226_geometry_control(
    *,
    source_wells: Sequence[str],
    target_frames: Mapping[str, pd.DataFrame],
    file_by_well: Mapping[str, Path],
    config: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    params = k16_params(config)
    sources = [
        load_k16_source_well(file_by_well[well], index, params)
        for index, well in enumerate(sorted(source_wells))
    ]
    fields = build_k16_fields(sources, params)
    kappa = fit_k16_kappa(sources, fields, params)
    heldout_rows = int(get_nested(config, "prefix_rolling_origin.heldout_prefix_rows"))
    minimum_prefix = int(
        get_nested(config, "prefix_rolling_origin.minimum_original_prefix_rows")
    )
    controls: dict[str, pd.DataFrame] = {}
    for well, frame in sorted(target_frames.items()):
        finite = np.flatnonzero(
            np.isfinite(frame["TVT_input"].to_numpy(dtype=float))
        )
        if len(finite) < minimum_prefix:
            continue
        heldout = finite[-heldout_rows:]
        if not np.array_equal(
            heldout,
            np.arange(heldout[0], heldout[0] + heldout_rows),
        ):
            raise ValueError(f"{well} visible-prefix heldout rows are not contiguous")
        pseudo_cut = int(heldout[0] - 1)
        target = make_k16_pseudo_target(
            frame,
            pseudo_cut,
            heldout_rows,
            params,
        )
        design = k16_well_design(target, fields, params)
        prediction = target.anchor + design @ kappa
        controls[well] = pd.DataFrame(
            {
                "well_id": well,
                "row_idx": heldout.astype(np.int32),
                "exp226_geometry_control": prediction,
                "pseudo_cut_row": pseudo_cut,
            }
        )
    manifest = {
        "source_wells": int(len(sources)),
        "target_wells": int(len(controls)),
        "heldout_rows_per_well": heldout_rows,
        "kappa": kappa.tolist(),
        "kappa_sha256": sha256_bytes(np.asarray(kappa, dtype="<f8").tobytes()),
        "control_scope": "fixed16_pseudo_cut_only",
        "gr_correction": False,
        "u_projection": False,
        "official_oof_regenerated": False,
    }
    return controls, manifest


def interpolate_gauge_correction(
    gauge: pd.DataFrame,
    well: str,
    row_idx: np.ndarray,
) -> np.ndarray:
    nodes = gauge.loc[
        gauge["well_id"].eq(well) & gauge["finite_gauge"].astype(bool)
    ].sort_values("center_row_idx", kind="mergesort")
    if len(nodes) < 2:
        return np.full(len(row_idx), np.nan, dtype=float)
    return np.interp(
        np.asarray(row_idx, dtype=float),
        nodes["center_row_idx"].to_numpy(dtype=float),
        nodes["solved_offset_ft"].to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )


def build_prefix_rolling_origin(
    *,
    fold_results: Sequence[Mapping[str, Any]],
    fold_by_well: Mapping[str, int],
    file_by_well: Mapping[str, Path],
    typewell_by_well: Mapping[str, str],
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    readout_parts: list[pd.DataFrame] = []
    replay_manifests: list[dict[str, Any]] = []
    for result in fold_results:
        fold = int(result["fold"])
        target_wells = list(result["target_wells"])
        original_targets = {
            well: result["contexts"][well].copy()
            for well in target_wells
        }
        source_wells = sorted(
            well
            for well, assigned in fold_by_well.items()
            if int(assigned) != fold
        )
        controls, control_manifest = build_exp226_geometry_control(
            source_wells=source_wells,
            target_frames=original_targets,
            file_by_well=file_by_well,
            config=config,
        )
        eligible_targets = sorted(controls)
        masked_contexts = {
            well: result["contexts"][well].copy()
            for well in result["graph_donors"]
        }
        for well in eligible_targets:
            control = controls[well]
            indices = control["row_idx"].to_numpy(dtype=int)
            masked = original_targets[well].copy()
            masked.loc[indices, "TVT_input"] = np.nan
            masked.loc[indices, "base_tvt"] = control[
                "exp226_geometry_control"
            ].to_numpy(dtype=float)
            masked_contexts[well] = masked
        edges, _, blocks = build_pairwise_edges(
            fold=fold,
            contexts=masked_contexts,
            target_wells=eligible_targets,
            donor_wells=result["graph_donors"],
            typewell_by_well=typewell_by_well,
            config=config,
        )
        solved_edges, gauge, solver = solve_loop_closed_graph(
            edges,
            blocks,
            eligible_targets,
            config,
        )
        replay_manifest = {
            "fold": fold,
            **control_manifest,
            "rolling_edge_logical_sha256": frame_content_sha256(
                solved_edges,
                EDGE_SORT_COLUMNS,
            ),
            "rolling_gauge_logical_sha256": frame_content_sha256(
                gauge,
                GAUGE_SORT_COLUMNS,
            ),
            "rolling_fundamental_cycles": int(
                solver["fundamental_cycles"]
            ),
        }
        replay_manifests.append(replay_manifest)
        for well in eligible_targets:
            control = controls[well].copy()
            row_idx = control["row_idx"].to_numpy(dtype=int)
            correction = interpolate_gauge_correction(gauge, well, row_idx)
            control["loop_closed_prediction"] = (
                control["exp226_geometry_control"].to_numpy(dtype=float)
                + correction
            )
            original = original_targets[well]
            truth = original.loc[row_idx, "TVT_input"].to_numpy(dtype=float)
            ledger.record_prefix_truth_late(fold, well, len(row_idx))
            control["tvt_true"] = truth
            control["fold"] = fold
            control["graph_correction_ft"] = correction
            control["control_squared_error"] = np.square(
                truth - control["exp226_geometry_control"].to_numpy(dtype=float)
            )
            control["loop_closed_squared_error"] = np.square(
                truth - control["loop_closed_prediction"].to_numpy(dtype=float)
            )
            control["rolling_edge_logical_sha256"] = replay_manifest[
                "rolling_edge_logical_sha256"
            ]
            control["rolling_gauge_logical_sha256"] = replay_manifest[
                "rolling_gauge_logical_sha256"
            ]
            readout_parts.append(control)
    readout = concat_frames(
        readout_parts,
        ("fold", "well_id", "row_idx"),
    )
    metric_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", readout)]
    if not readout.empty:
        scopes.extend(
            (f"fold_{int(fold)}", part)
            for fold, part in readout.groupby("fold", sort=True)
        )
    for scope, part in scopes:
        required = {
            "tvt_true",
            "exp226_geometry_control",
            "loop_closed_prediction",
        }
        if part.empty or not required.issubset(part.columns):
            finite = pd.DataFrame(columns=sorted(required))
        else:
            finite = part.loc[
                np.isfinite(part["tvt_true"])
                & np.isfinite(part["exp226_geometry_control"])
                & np.isfinite(part["loop_closed_prediction"])
            ]
        control_rmse = (
            rmse(
                finite["tvt_true"].to_numpy(dtype=float),
                finite["exp226_geometry_control"].to_numpy(dtype=float),
            )
            if len(finite)
            else math.nan
        )
        graph_rmse = (
            rmse(
                finite["tvt_true"].to_numpy(dtype=float),
                finite["loop_closed_prediction"].to_numpy(dtype=float),
            )
            if len(finite)
            else math.nan
        )
        metric_rows.append(
            {
                "scope": scope,
                "fold": (
                    int(scope.split("_", 1)[1]) if scope.startswith("fold_") else -1
                ),
                "rows": int(len(finite)),
                "exp226_geometry_control_rmse_ft": control_rmse,
                "loop_closed_rmse_ft": graph_rmse,
                "gain_vs_exp226_ft": (
                    control_rmse - graph_rmse
                    if np.isfinite(control_rmse) and np.isfinite(graph_rmse)
                    else math.nan
                ),
            }
        )
    return readout, pd.DataFrame(metric_rows), replay_manifests


def evaluate_prefix_gate(
    prefix_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    scientific = get_nested(config, "gates.scientific")
    pooled = prefix_metrics.loc[prefix_metrics["scope"].eq("pooled")]
    pooled_gain = (
        float(pooled.iloc[0]["gain_vs_exp226_ft"]) if len(pooled) else -math.inf
    )
    folds = prefix_metrics.loc[prefix_metrics["scope"].str.startswith("fold_")]
    positive_folds = int((folds["gain_vs_exp226_ft"] > 0.0).sum())
    checks = {
        "prefix_rolling_origin_gain": pooled_gain
        >= float(scientific["prefix_rolling_origin_gain_vs_exp226_ft_min"]),
        "prefix_positive_folds": positive_folds
        >= int(scientific["prefix_gain_positive_fold_count_min"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "prefix_rolling_origin_gain_vs_exp226_ft": pooled_gain,
            "prefix_gain_positive_fold_count": positive_folds,
        },
    }


# %% [markdown]
# ## 9. Resource projection, artifact manifest, and orchestration

# %%
ARTIFACT_FILENAMES = {
    "fixed16_manifest": f"{EXPERIMENT_NAME}_fixed16_manifest.csv",
    "role_read_ledger": f"{EXPERIMENT_NAME}_role_read_ledger.csv",
    "pairwise_edges": f"{EXPERIMENT_NAME}_pairwise_edges.parquet",
    "edge_rejection_funnel": f"{EXPERIMENT_NAME}_edge_rejection_funnel.csv",
    "cycle_basis": f"{EXPERIMENT_NAME}_cycle_basis.parquet",
    "loop_closed_gauge": f"{EXPERIMENT_NAME}_loop_closed_gauge.parquet",
    "negative_control_metrics": f"{EXPERIMENT_NAME}_negative_control_metrics.csv",
    "prefix_readout": f"{EXPERIMENT_NAME}_prefix_readout.csv",
}

ARTIFACT_SORT_COLUMNS = {
    "fixed16_manifest": ("fold", "well_id"),
    "role_read_ledger": ("fold", "role", "well_id"),
    "pairwise_edges": EDGE_SORT_COLUMNS,
    "edge_rejection_funnel": ("fold", "well_id", "reason"),
    "cycle_basis": CYCLE_SORT_COLUMNS,
    "loop_closed_gauge": GAUGE_SORT_COLUMNS,
    "negative_control_metrics": ("scope",),
    "prefix_readout": ("fold", "well_id", "row_idx"),
}


def persist_frames(
    frames: Mapping[str, pd.DataFrame],
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records: dict[str, Any] = {}
    sha_rows: list[dict[str, Any]] = []
    for name, filename in ARTIFACT_FILENAMES.items():
        frame = frames.get(name, pd.DataFrame())
        path = output / filename
        write_table(frame, path)
        record = {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "schema_sha256": frame_schema_sha256(frame),
            "logical_content_sha256": frame_content_sha256(
                frame,
                ARTIFACT_SORT_COLUMNS[name],
            ),
            "file_sha256": sha256_file(path),
        }
        records[name] = record
        sha_rows.append({"artifact": name, **record})
    return records, pd.DataFrame(sha_rows)


def resource_projection(
    *,
    target_free_elapsed_seconds: float,
    total_elapsed_seconds: float,
    fixed16_wells: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    projected_seconds = (
        target_free_elapsed_seconds * expected_wells / max(fixed16_wells, 1)
    )
    peak = peak_rss_gb()
    return {
        "fixed16_target_free_elapsed_seconds": float(target_free_elapsed_seconds),
        "fixed16_total_diagnostic_elapsed_seconds": float(total_elapsed_seconds),
        "fixed16_prefix_diagnostic_elapsed_seconds": float(
            max(total_elapsed_seconds - target_free_elapsed_seconds, 0.0)
        ),
        "fixed16_target_wells": int(fixed16_wells),
        "projected_full_wells": expected_wells,
        "projected_full_runtime_seconds": float(projected_seconds),
        "peak_rss_gb": float(peak),
        "runtime_pass": projected_seconds
        <= float(get_nested(config, "gates.technical.projected_full_runtime_seconds_max")),
        "rss_pass": peak
        <= float(get_nested(config, "gates.technical.projected_peak_rss_gb_max")),
        "projection_method": (
            "fixed16_target_free_graph_elapsed_linear_well_scaling;"
            "prefix_pseudocut_control_is_stage0_diagnostic_only"
        ),
    }


def final_decision(
    target_free_gate: Mapping[str, Any],
    prefix_gate: Mapping[str, Any] | None,
    resources: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_checks = dict(target_free_gate["technical_checks"])
    technical_checks["projected_full_runtime"] = bool(resources["runtime_pass"])
    technical_checks["projected_peak_rss"] = bool(resources["rss_pass"])
    technical_pass = bool(all(technical_checks.values()))
    negative_pass = bool(target_free_gate["negative_control_pass"])
    prefix_pass = bool(prefix_gate and prefix_gate["passed"])
    passed = bool(technical_pass and negative_pass and prefix_pass)
    status = (
        str(get_nested(config, "gates.decision.pass"))
        if passed
        else str(get_nested(config, "gates.decision.fail"))
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "passed": passed,
        "technical_pass": technical_pass,
        "technical_checks": technical_checks,
        "negative_control_pass": negative_pass,
        "negative_control_checks": target_free_gate["negative_control_checks"],
        "prefix_pass": prefix_pass,
        "prefix_checks": prefix_gate["checks"] if prefix_gate else {},
        "target_free_observed": target_free_gate["target_free_observed"],
        "negative_control_observed": target_free_gate[
            "negative_control_observed"
        ],
        "prefix_observed": prefix_gate["observed"] if prefix_gate else {},
        "resource_observed": dict(resources),
        "full_oof_stage1_eligible": passed,
        "full_oof_stage1_authorized": False,
        "current_test_implemented": False,
        "inference_implemented": False,
        "submission_created": False,
    }


def run_train() -> dict[str, Any]:
    config = load_config()
    validate_execution_contract(config, require_kaggle_authorization=True)
    started = time.perf_counter()
    train_dir = resolve_train_dir(config)
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    file_by_well = {well_id_from_path(path): path for path in files}
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(file_by_well) != expected_wells:
        raise ValueError(f"expected {expected_wells} raw train wells, found {len(file_by_well)}")

    exp405_decision, exp405_path = load_exp405_decision(config)
    exp226_oof, exp226_path = load_exp226_oof(config)
    typewell_by_well, typewell_assignments, typewell_path = load_typewell_groups(
        config
    )
    folds = list(map(int, get_nested(config, "validation.expected_folds")))
    fold_by_well = assign_group_folds(
        sorted(file_by_well),
        int(get_nested(config, "validation.n_folds")),
        int(get_nested(config, "validation.seed")),
    )
    validate_fold_identity(fold_by_well, exp226_oof)
    fixed16 = select_fixed16_wells(
        fold_by_well,
        int(get_nested(config, "validation.fixed16_target_wells")),
        folds,
    )
    selected_by_fold = {
        fold: sorted(well for well in fixed16 if fold_by_well[well] == fold)
        for fold in folds
    }
    if any(not selected_by_fold[fold] for fold in folds):
        raise ValueError("fixed16 must include all five reporting folds")

    xy_by_well = {
        well: read_observable_xy(path)
        for well, path in sorted(file_by_well.items())
    }
    ledger = RoleReadLedger()
    fold_results = [
        run_fold_target_free(
            fold=fold,
            target_wells=selected_by_fold[fold],
            file_by_well=file_by_well,
            xy_by_well=xy_by_well,
            fold_by_well=fold_by_well,
            exp226_oof=exp226_oof,
            typewell_by_well=typewell_by_well,
            ledger=ledger,
            config=config,
        )
        for fold in folds
    ]
    frames: dict[str, pd.DataFrame] = {
        "fixed16_manifest": concat_frames(
            [result["coverage"] for result in fold_results],
            ("fold", "well_id"),
        ),
        "pairwise_edges": concat_frames(
            [result["edges"] for result in fold_results],
            EDGE_SORT_COLUMNS,
        ),
        "edge_rejection_funnel": concat_frames(
            [result["funnel"] for result in fold_results],
            ("fold", "well_id", "reason"),
        ),
        "cycle_basis": concat_frames(
            [result["cycles"] for result in fold_results],
            CYCLE_SORT_COLUMNS,
        ),
        "loop_closed_gauge": concat_frames(
            [result["gauge"] for result in fold_results],
            GAUGE_SORT_COLUMNS,
        ),
    }
    frames["negative_control_metrics"] = negative_control_metrics(
        frames["pairwise_edges"]
    )
    target_free_hashes = freeze_target_free_frames(frames, ledger)
    target_free_elapsed = time.perf_counter() - started
    target_free_gate = evaluate_target_free_gates(
        fixed16_manifest=frames["fixed16_manifest"],
        edges=frames["pairwise_edges"],
        cycles=frames["cycle_basis"],
        ledger=ledger,
        elapsed_seconds=target_free_elapsed,
        observed_peak_rss_gb=peak_rss_gb(),
        config=config,
    )
    prefix_gate: dict[str, Any] | None = None
    prefix_metrics = pd.DataFrame()
    prefix_manifests: list[dict[str, Any]] = []
    if target_free_gate["technical_pass"]:
        prefix_readout, prefix_metrics, prefix_manifests = build_prefix_rolling_origin(
            fold_results=fold_results,
            fold_by_well=fold_by_well,
            file_by_well=file_by_well,
            typewell_by_well=typewell_by_well,
            ledger=ledger,
            config=config,
        )
        frames["prefix_readout"] = prefix_readout
        prefix_gate = evaluate_prefix_gate(prefix_metrics, config)
    else:
        frames["prefix_readout"] = pd.DataFrame()
    frames["role_read_ledger"] = ledger.as_frame()

    elapsed = time.perf_counter() - started
    resources = resource_projection(
        target_free_elapsed_seconds=target_free_elapsed,
        total_elapsed_seconds=elapsed,
        fixed16_wells=len(fixed16),
        config=config,
    )
    gate = final_decision(target_free_gate, prefix_gate, resources, config)
    output = artifacts_dir()
    records, sha_manifest = persist_frames(frames, output)
    sha_path = output / f"{EXPERIMENT_NAME}_sha_manifest.csv"
    sha_manifest.to_csv(sha_path, index=False)
    resource_path = output / f"{EXPERIMENT_NAME}_resource_projection.json"
    write_json(resource_path, resources)
    gate_path = output / f"{EXPERIMENT_NAME}_gate.json"
    write_json(gate_path, gate)
    decision_manifest = {
        "source_experiment": EXP405_NAME,
        "source_path": str(exp405_path),
        "source_file_sha256": sha256_file(exp405_path),
        "decision": exp405_decision["status"],
        "decision_sha256": exp405_decision["decision_sha256"],
        "technical_pass": bool(exp405_decision["gate"]["technical_pass"]),
        "scientific_pass": bool(exp405_decision["gate"]["scientific_pass"]),
    }
    decision_path = output / f"{EXPERIMENT_NAME}_exp405_decision_manifest.json"
    write_json(decision_path, decision_manifest)
    contract = {
        "experiment": get_nested(config, "experiment"),
        "lineage": get_nested(config, "lineage"),
        "validation": get_nested(config, "validation"),
        "pairwise_gr": get_nested(config, "pairwise_gr"),
        "loop_closure": get_nested(config, "loop_closure"),
        "prefix_rolling_origin": get_nested(config, "prefix_rolling_origin"),
        "gates": get_nested(config, "gates"),
        "runtime": get_nested(config, "runtime"),
        "execution": get_nested(config, "execution"),
    }
    contract_path = output / f"{EXPERIMENT_NAME}_contract.json"
    write_json(contract_path, contract)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": gate["status"],
        "passed": gate["passed"],
        "fixed16_wells": fixed16,
        "folds": folds,
        "elapsed_seconds": elapsed,
        "target_free_elapsed_seconds": target_free_elapsed,
        "target_free_logical_sha256": target_free_hashes,
        "prefix_replay_manifests": prefix_manifests,
        "prefix_metrics": prefix_metrics.to_dict(orient="records"),
        "gate": gate,
        "resources": resources,
        "inputs": {
            "exp405_decision_path": str(exp405_path),
            "exp405_decision_file_sha256": sha256_file(exp405_path),
            "exp226_oof_path": str(exp226_path),
            "exp226_oof_decompressed_sha256": sha256_decompressed_csv(exp226_path),
            "typewell_assignment_path": str(typewell_path),
            "typewell_assignment_file_sha256": sha256_file(typewell_path),
            "typewell_assignment_rows": int(len(typewell_assignments)),
            "raw_wells": int(len(file_by_well)),
        },
        "artifacts": records,
        "sha_manifest_file_sha256": sha256_file(sha_path),
        "contract_file_sha256": sha256_file(contract_path),
        "gate_file_sha256": sha256_file(gate_path),
        "resource_file_sha256": sha256_file(resource_path),
        "exp405_decision_manifest_file_sha256": sha256_file(decision_path),
        "execution_count": {
            "scientific_diagnostics": 1,
            "target_wells": 16,
            "reporting_folds": 5,
            "models": 0,
            "boosters": 0,
            "pf_runs": 0,
            "hmm_runs": 0,
            "beam_runs": 0,
            "exp226_fixed16_prefix_geometry_replays": len(prefix_manifests),
        },
        "unknown_suffix_prediction_persisted": False,
        "deterministic_anchor": False,
    }
    summary_path = output / f"{EXPERIMENT_NAME}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": gate["status"],
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "summary": summary,
    }
    metrics_path = (
        output_root() / "metrics.json"
        if Path("/kaggle/working").exists()
        else find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"
    )
    write_json(metrics_path, metrics)
    print(json.dumps(summary, indent=2, default=str))
    return summary


# %% [markdown]
# ## 10. Configuration preview and guarded execution

# %%
CONFIG = load_config()
CONTRACT_COUNTS = validate_execution_contract(
    CONFIG,
    require_kaggle_authorization=False,
)

print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Parent:", get_nested(CONFIG, "lineage.parent"))
print("Stage:", get_nested(CONFIG, "execution.current_stage"))
print("Fixed targets / folds:", CONTRACT_COUNTS["target_wells"], CONTRACT_COUNTS["reporting_folds"])
print(
    "Pairwise:",
    get_nested(CONFIG, "pairwise_gr.block_rows"),
    get_nested(CONFIG, "pairwise_gr.stride_rows"),
    get_nested(CONFIG, "pairwise_gr.shift_grid_ft"),
)
print(
    "Exp226 control:",
    get_nested(CONFIG, "prefix_rolling_origin.comparison"),
    "fixed16 pseudo-cut only",
)
print("Kaggle execution authorized:", get_nested(CONFIG, "execution.kaggle_execution_authorized"))
print("Inference / submission:", False, False)

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    RUN_SUMMARY = run_train()
