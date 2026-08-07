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
# # exp386 cycle-consistent RGT scenario bank
#
# This train-side audit builds a deterministic relative-geologic-time graph from
# outer-train wells only. Outer-valid wells expose only trajectory coordinates and
# `TVT_input`; their GR, raw formation columns, and suffix TVT remain unread until
# the target-free bank has been frozen by logical-content SHA.

# %% [markdown]
# ## Contents
#
# 1. Imports and immutable execution contract
# 2. Runtime, path, SHA, and table helpers
# 3. Fold roles, guarded readers, and saved exp226 control
# 4. Ordered-formation RGT and 64/32-ft graph nodes
# 5. Cross-well correspondence graph and cycle-consistent solve
# 6. Deterministic k-shortest scenario paths and reference-GR templates
# 7. Target-free freeze, Stage 0 gate, and resource projection
# 8. Rolling-origin prefix and truth-late H512 oracle readouts
# 9. Artifact manifest and execution orchestration
# 10. Configuration preview

# %% [markdown]
# ## 1. Imports and immutable execution contract

# %%
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.linalg import lsqr

EXPERIMENT_NAME = "exp386_cycle_consistent_rgt_scenario_bank"
PARENT_EXPERIMENT = (
    "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
)
INDEPENDENT_PARENT = "independent_topology_first_rgt_physics_family"
IMPORT_ONLY_ENV = "EXP386_IMPORT_ONLY"

FORMATION_NAMES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
THICKNESS_COLUMNS = tuple(
    f"thickness_{FORMATION_NAMES[index]}_{FORMATION_NAMES[index + 1]}"
    for index in range(len(FORMATION_NAMES) - 1)
)
TARGET_ALLOWED_COLUMNS = ("MD", "X", "Y", "Z", "TVT_input")
TARGET_FORBIDDEN_COLUMNS = frozenset({"TVT", "GR", *FORMATION_NAMES})
SOURCE_COLUMNS = ("MD", "X", "Y", "Z", "TVT", "GR", *FORMATION_NAMES)
PARENT_SAFE_COLUMNS = ("well_id", "row_idx", "suffix_offset", "tvt_pred", "fold")

NODE_SORT_COLUMNS = ("fold", "well_id", "MD", "node_id")
EDGE_SORT_COLUMNS = ("fold", "source_well_id", "target_well_id", "edge_id")
PATH_SORT_COLUMNS = ("fold", "well_id", "scenario_rank", "control_index")
REFERENCE_SORT_COLUMNS = (
    "fold",
    "well_id",
    "scenario_rank",
    "control_index",
)


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
) -> None:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("wrong experiment config")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp386 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != INDEPENDENT_PARENT:
        raise ValueError("exp386 must remain an independent topology-first family")
    if not bool(get_nested(config, "execution.implementation_authorized", False)):
        raise ValueError("implementation authorization is not recorded")
    if bool(get_nested(config, "execution.inference_enabled", True)):
        raise ValueError("exp386 inference must remain disabled")
    if bool(get_nested(config, "execution.submission_enabled", True)):
        raise ValueError("exp386 submission must remain disabled")
    expected = {
        "runtime.scientific_variants": 1,
        "runtime.reporting_folds": 5,
        "runtime.fitted_models": 0,
        "runtime.model_configs": 0,
        "runtime.trained_folds": 0,
        "runtime.lightgbm_boosters": 0,
        "runtime.hmm_runs": 0,
        "runtime.pf_runs": 0,
        "runtime.beam_runs": 0,
        "runtime.graph_fold_solves": 5,
        "runtime.target_well_path_solves": 773,
    }
    for key, required in expected.items():
        if int(get_nested(config, key, -1)) != required:
            raise ValueError(f"{key} must remain {required}")
    if bool(get_nested(config, "runtime.parent_control_regeneration", True)):
        raise ValueError("saved exp226 control must not be regenerated")
    if int(
        get_nested(config, "rgt.scenario_bank.minimum_scenarios_per_well", -1)
    ) != 8:
        raise ValueError("minimum scenario count must remain 8")
    if int(
        get_nested(config, "rgt.scenario_bank.maximum_scenarios_per_well", -1)
    ) != 32:
        raise ValueError("maximum scenario count must remain 32")
    mode = str(get_nested(config, "execution.current_mode", ""))
    if mode not in {"stage0_resource_preflight", "full_run"}:
        raise ValueError(f"unsupported exp386 execution mode: {mode}")
    if mode == "full_run" and not bool(
        get_nested(config, "execution.full_run_authorized", False)
    ):
        raise RuntimeError("exp386 full run is not authorized")
    if require_kaggle_authorization and not bool(
        get_nested(config, "execution.kaggle_execution_authorized", False)
    ):
        raise RuntimeError(
            "exp386 Kaggle execution is not authorized; canonical notebook "
            "adoption, package, push, and run require a separate approval"
        )


# %% [markdown]
# ## 2. Runtime, path, SHA, and table helpers

# %%
PACKAGE_DIR = Path.cwd()


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def config_path() -> Path:
    candidates = [PACKAGE_DIR / "config.yaml"]
    root = find_project_root()
    candidates.append(root / "experiments" / EXPERIMENT_NAME / "config.yaml")
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
    raise FileNotFoundError("exp386 config.yaml was not found")


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
            value_float = float(item)
            return value_float if math.isfinite(value_float) else str(value_float)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"cannot serialize {type(item)!r}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_decompressed_csv(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "<NA>"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    return str(value)


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return sha256_bytes(stable_json_bytes(schema))


def frame_content_sha256(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
) -> str:
    work = frame
    if sort_columns:
        existing = [column for column in sort_columns if column in work.columns]
        if existing:
            work = work.sort_values(existing, kind="mergesort")
    digest = hashlib.sha256()
    digest.update(stable_json_bytes([str(column) for column in work.columns]))
    for row in work.itertuples(index=False, name=None):
        digest.update(
            ("\x1f".join(_canonical_cell(value) for value in row) + "\n").encode()
        )
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
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
    if len(actual_array) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def resolve_candidate_file(candidates: Sequence[str], filename: str) -> Path:
    roots = [Path(value) for value in candidates]
    root = find_project_root()
    roots.extend(
        [
            root / "artifacts",
            root / "experiments" / PARENT_EXPERIMENT / "artifacts",
        ]
    )
    for candidate in roots:
        path = candidate if candidate.name == filename else candidate / filename
        if path.exists():
            return path
    matches: list[Path] = []
    for search_root in (Path("/kaggle/input"), Path("/tmp")):
        if search_root.exists():
            matches.extend(search_root.rglob(filename))
    if matches:
        return sorted(matches)[0]
    raise FileNotFoundError(f"could not resolve {filename}")


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
# ## 3. Fold roles, guarded readers, and saved exp226 control

# %%
@dataclass
class RoleReadLedger:
    events: list[dict[str, Any]] = field(default_factory=list)
    frozen: bool = False
    source_valid_overlap: int = 0

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
        forbidden = TARGET_FORBIDDEN_COLUMNS.intersection(names)
        self.events.append(
            {
                "fold": int(fold),
                "well_id": str(well_id),
                "role": role,
                "rows": int(rows),
                "column_signature": ",".join(names),
                "target_gr_reads": int(role.startswith("outer_valid") and "GR" in names),
                "valid_formation_reads": int(
                    role.startswith("outer_valid")
                    and bool(set(FORMATION_NAMES).intersection(names))
                ),
                "valid_suffix_truth_reads": int(
                    role.startswith("outer_valid") and "TVT" in names
                ),
                "after_target_freeze": bool(self.frozen),
            }
        )
        if role == "outer_valid_target_free" and forbidden:
            raise ValueError(
                f"target-free outer-valid read contains forbidden columns: "
                f"{sorted(forbidden)}"
            )
        if role == "outer_valid_truth_late" and not self.frozen:
            raise ValueError("suffix truth cannot be read before target-free SHA freeze")

    def record_source(
        self,
        *,
        fold: int,
        well_id: str,
        columns: Iterable[str],
        rows: int,
    ) -> None:
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_train_source",
            columns=columns,
            rows=rows,
        )

    def record_target_safe(
        self,
        *,
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

    def record_truth_late(self, *, fold: int, well_id: str, rows: int) -> None:
        self._record(
            fold=fold,
            well_id=well_id,
            role="outer_valid_truth_late",
            columns=("TVT",),
            rows=rows,
        )

    def validate_disjoint(
        self,
        source_wells: Iterable[str],
        target_wells: Iterable[str],
    ) -> None:
        overlap = set(map(str, source_wells)).intersection(map(str, target_wells))
        self.source_valid_overlap += len(overlap)
        if overlap:
            raise ValueError(f"outer-valid wells leaked into source graph: {sorted(overlap)[:5]}")

    def mark_frozen(self, hashes: Mapping[str, str]) -> None:
        if not hashes:
            raise ValueError("target-free freeze requires logical content hashes")
        self.frozen = True

    def as_frame(self) -> pd.DataFrame:
        columns = [
            "fold",
            "well_id",
            "role",
            "rows",
            "column_signature",
            "target_gr_reads",
            "valid_formation_reads",
            "valid_suffix_truth_reads",
            "after_target_freeze",
        ]
        return pd.DataFrame(self.events, columns=columns).sort_values(
            ["fold", "role", "well_id"], kind="mergesort"
        )

    def summary(self) -> dict[str, Any]:
        frame = self.as_frame()
        before = frame.loc[~frame["after_target_freeze"]] if len(frame) else frame
        target = before["role"].eq("outer_valid_target_free") if len(before) else []
        return {
            "target_safe_rows": int(before.loc[target, "rows"].sum()) if len(before) else 0,
            "target_gr_reads_before_freeze": (
                int(before.loc[target, "target_gr_reads"].sum()) if len(before) else 0
            ),
            "valid_formation_reads_before_freeze": (
                int(before.loc[target, "valid_formation_reads"].sum()) if len(before) else 0
            ),
            "valid_suffix_truth_reads_before_freeze": (
                int(before.loc[target, "valid_suffix_truth_reads"].sum())
                if len(before)
                else 0
            ),
            "source_valid_overlap": int(self.source_valid_overlap),
            "truth_joined_after_freeze": bool(
                len(frame)
                and frame["role"].eq("outer_valid_truth_late").any()
                and frame.loc[
                    frame["role"].eq("outer_valid_truth_late"), "after_target_freeze"
                ].all()
            ),
        }


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
    well_id = well_id_from_path(path)
    ledger.record_source(
        fold=fold,
        well_id=well_id,
        columns=frame.columns,
        rows=len(frame),
    )
    frame = frame.copy()
    frame["well_id"] = well_id
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
    well_id = well_id_from_path(path)
    ledger.record_target_safe(
        fold=fold,
        well_id=well_id,
        columns=frame.columns,
        rows=len(frame),
    )
    frame = frame.copy()
    frame["well_id"] = well_id
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    frame["fold"] = int(fold)
    frame["role"] = "outer_valid"
    return frame


def read_target_truth_late(
    path: Path,
    fold: int,
    ledger: RoleReadLedger,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["TVT"])
    well_id = well_id_from_path(path)
    ledger.record_truth_late(fold=fold, well_id=well_id, rows=len(frame))
    frame = frame.copy()
    frame["well_id"] = well_id
    frame["row_idx"] = np.arange(len(frame), dtype=np.int32)
    return frame


def load_parent_oof(config: Mapping[str, Any]) -> tuple[pd.DataFrame, Path]:
    filename = str(get_nested(config, "data.parent_exp226.filename"))
    candidates = list(get_nested(config, "data.parent_exp226.candidates", []))
    path = resolve_candidate_file(candidates, filename)
    expected = str(
        get_nested(config, "data.parent_exp226.expected_oof_decompressed_sha256")
    )
    actual = sha256_decompressed_csv(path)
    if actual != expected:
        raise ValueError(f"exp226 OOF SHA mismatch: expected {expected}, got {actual}")
    frame = pd.read_csv(path, usecols=list(PARENT_SAFE_COLUMNS), dtype={"well_id": str})
    require_columns(frame, PARENT_SAFE_COLUMNS, "saved exp226 OOF")
    frame["well_id"] = frame["well_id"].astype(str)
    frame["fold"] = frame["fold"].astype(int)
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("saved exp226 row count differs from the frozen contract")
    if frame["well_id"].nunique() != int(
        get_nested(config, "validation.expected_wells")
    ):
        raise ValueError("saved exp226 well count differs from the frozen contract")
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
# ## 4. Ordered-formation RGT and 64/32-ft graph nodes

# %%
def ordered_formation_rgt(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ("TVT", "Z", *FORMATION_NAMES),
        "outer-train RGT source",
    )
    formations_z = frame[list(FORMATION_NAMES)].to_numpy(dtype=float)
    tvt = frame["TVT"].to_numpy(dtype=float)
    z = frame["Z"].to_numpy(dtype=float)
    structural_s = tvt + z
    formation_s = tvt[:, None] + formations_z
    thickness = formation_s[:, :-1] - formation_s[:, 1:]
    valid = (
        np.isfinite(structural_s)
        & np.isfinite(formation_s).all(axis=1)
        & (thickness > 0.0).all(axis=1)
    )
    interval = np.sum(
        structural_s[:, None] < formation_s[:, :-1],
        axis=1,
    )
    interval = np.clip(interval, 0, len(FORMATION_NAMES) - 2).astype(np.int16)
    row = np.arange(len(frame))
    top = formation_s[row, interval]
    bottom = formation_s[row, interval + 1]
    denominator = top - bottom
    rgt = interval.astype(float) + (top - structural_s) / denominator
    rgt[~valid] = np.nan
    output = frame.copy()
    output["structural_s"] = structural_s
    output["formation_interval_id"] = interval
    output["rgt"] = rgt
    output["rgt_available"] = valid
    for index, column in enumerate(THICKNESS_COLUMNS):
        output[column] = thickness[:, index]
    return output


def fixed_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    center: float,
) -> tuple[float, float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(x_values)), x_values - float(center)])
    coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
    residual = y_values - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(np.mean(np.square(residual))),
    )


def build_rgt_nodes_for_well(
    source: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    converted = ordered_formation_rgt(source)
    coverage = float(converted["rgt_available"].mean()) if len(converted) else 0.0
    available = converted.loc[converted["rgt_available"]].sort_values(
        ["MD", "row_idx"], kind="mergesort"
    )
    width = float(get_nested(config, "rgt.graph_nodes.md_window_ft"))
    stride = float(get_nested(config, "rgt.graph_nodes.md_stride_ft"))
    minimum_rows = int(get_nested(config, "rgt.graph_nodes.minimum_rows"))
    records: list[dict[str, Any]] = []
    if len(available):
        md = available["MD"].to_numpy(dtype=float)
        centers = np.arange(md.min(), md.max() + 0.5 * stride, stride)
        centers = np.unique(np.clip(centers, md.min(), md.max()))
        for node_index, center in enumerate(centers):
            mask = np.abs(md - float(center)) <= 0.5 * width
            window = available.loc[mask]
            if len(window) < minimum_rows:
                continue
            rgt_center, rgt_slope, rgt_variance = fixed_linear_fit(
                window["MD"].to_numpy(float),
                window["rgt"].to_numpy(float),
                float(center),
            )
            s_center, s_slope, s_variance = fixed_linear_fit(
                window["MD"].to_numpy(float),
                window["structural_s"].to_numpy(float),
                float(center),
            )
            x_center, dx_dmd, _ = fixed_linear_fit(
                window["MD"].to_numpy(float),
                window["X"].to_numpy(float),
                float(center),
            )
            y_center, dy_dmd, _ = fixed_linear_fit(
                window["MD"].to_numpy(float),
                window["Y"].to_numpy(float),
                float(center),
            )
            z_center, dz_dmd, _ = fixed_linear_fit(
                window["MD"].to_numpy(float),
                window["Z"].to_numpy(float),
                float(center),
            )
            direction = np.asarray([dx_dmd, dy_dmd, dz_dmd], dtype=float)
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                direction /= norm
            interval = int(np.clip(math.floor(rgt_center), 0, 4))
            stretch = s_slope / rgt_slope if rgt_slope > 0.0 else math.nan
            gr = window["GR"].to_numpy(dtype=float)
            finite_gr = gr[np.isfinite(gr)]
            gr_difference = np.diff(finite_gr)
            well_id = str(window["well_id"].iloc[0])
            fold = int(window["fold"].iloc[0])
            record: dict[str, Any] = {
                "fold": fold,
                "well_id": well_id,
                "node_id": f"f{fold:02d}_{well_id}_n{node_index:06d}",
                "node_index": int(node_index),
                "md_start": float(window["MD"].min()),
                "md_end": float(window["MD"].max()),
                "MD": float(center),
                "X": x_center,
                "Y": y_center,
                "Z": z_center,
                "rgt_median": rgt_center,
                "rgt_slope_per_md": rgt_slope,
                "rgt_residual_variance": rgt_variance,
                "structural_position_median": s_center,
                "structural_slope_per_md": s_slope,
                "structural_residual_variance": s_variance,
                "formation_interval_id": interval,
                "rgt_stretch_ft_per_interval": stretch,
                "trajectory_dx": float(direction[0]),
                "trajectory_dy": float(direction[1]),
                "trajectory_dz": float(direction[2]),
                "rows": int(len(window)),
                "reference_gr_median": (
                    float(np.median(finite_gr)) if len(finite_gr) else math.nan
                ),
                "reference_gr_diff_median": (
                    float(np.median(gr_difference))
                    if len(gr_difference)
                    else math.nan
                ),
            }
            for column in THICKNESS_COLUMNS:
                record[column] = float(np.median(window[column].to_numpy(float)))
            records.append(record)
    columns = [
        "fold",
        "well_id",
        "node_id",
        "node_index",
        "md_start",
        "md_end",
        "MD",
        "X",
        "Y",
        "Z",
        "rgt_median",
        "rgt_slope_per_md",
        "rgt_residual_variance",
        "structural_position_median",
        "structural_slope_per_md",
        "structural_residual_variance",
        "formation_interval_id",
        "rgt_stretch_ft_per_interval",
        "trajectory_dx",
        "trajectory_dy",
        "trajectory_dz",
        "rows",
        "reference_gr_median",
        "reference_gr_diff_median",
        *THICKNESS_COLUMNS,
    ]
    nodes = pd.DataFrame(records, columns=columns)
    if len(nodes):
        nodes = nodes.sort_values(list(NODE_SORT_COLUMNS), kind="mergesort")
    return nodes, {
        "well_id": str(source["well_id"].iloc[0]),
        "rows": int(len(source)),
        "rgt_available_rows": int(converted["rgt_available"].sum()),
        "rgt_source_coverage": coverage,
        "nodes": int(len(nodes)),
    }


# %% [markdown]
# ## 5. Cross-well correspondence graph and cycle-consistent solve

# %%
def well_profiles(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for well_id, group in nodes.groupby("well_id", sort=True):
        direction = group[
            ["trajectory_dx", "trajectory_dy", "trajectory_dz"]
        ].median().to_numpy(dtype=float, copy=True)
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            direction /= norm
        records.append(
            {
                "fold": int(group["fold"].iloc[0]),
                "well_id": str(well_id),
                "centroid_x": float(group["X"].median()),
                "centroid_y": float(group["Y"].median()),
                "direction_x": float(direction[0]),
                "direction_y": float(direction[1]),
                "direction_z": float(direction[2]),
                "node_count": int(len(group)),
            }
        )
    return pd.DataFrame(records).sort_values("well_id", kind="mergesort")


def nearest_well_pairs(
    profiles: pd.DataFrame,
    nearest_unique_wells: int,
) -> tuple[list[tuple[str, str]], dict[str, float]]:
    if profiles.empty:
        return [], {}
    ordered = profiles.sort_values("well_id", kind="mergesort").reset_index(drop=True)
    xy = ordered[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    well_ids = ordered["well_id"].astype(str).tolist()
    pairs: set[tuple[str, str]] = set()
    bandwidth: dict[str, float] = {}
    for index, well_id in enumerate(well_ids):
        distances = np.sqrt(np.sum(np.square(xy - xy[index]), axis=1))
        candidates = [
            (float(distances[other]), well_ids[other])
            for other in range(len(well_ids))
            if other != index
        ]
        selected = sorted(candidates, key=lambda item: (item[0], item[1]))[
            :nearest_unique_wells
        ]
        if selected:
            bandwidth[well_id] = max(float(selected[-1][0]), 1.0)
        for _, other in selected:
            pairs.add(tuple(sorted((well_id, other))))
    return sorted(pairs), bandwidth


def leave_one_well_stretch_bounds(
    nodes: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[tuple[str, int], tuple[float, float]]:
    values = nodes.loc[
        np.isfinite(nodes["rgt_stretch_ft_per_interval"])
        & nodes["rgt_stretch_ft_per_interval"].gt(0.0)
    ].copy()
    bounds: dict[tuple[str, int], tuple[float, float]] = {}
    wells = sorted(nodes["well_id"].astype(str).unique())
    for well_id in wells:
        other = values.loc[values["well_id"].astype(str).ne(well_id)]
        for interval in range(len(FORMATION_NAMES) - 1):
            selected = other.loc[
                other["formation_interval_id"].eq(interval),
                "rgt_stretch_ft_per_interval",
            ].to_numpy(dtype=float)
            if len(selected) < 2:
                bounds[(well_id, interval)] = (math.nan, math.nan)
            else:
                bounds[(well_id, interval)] = (
                    float(np.quantile(selected, lower_quantile)),
                    float(np.quantile(selected, upper_quantile)),
                )
    return bounds


def _node_at_common_rgt(node: pd.Series, common_rgt: float) -> float:
    return float(node["structural_position_median"]) + float(
        node["rgt_stretch_ft_per_interval"]
    ) * (float(common_rgt) - float(node["rgt_median"]))


def match_well_pair(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_bandwidth: float,
    target_bandwidth: float,
    stretch_bounds: Mapping[tuple[str, int], tuple[float, float]],
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    source_well = str(source["well_id"].iloc[0])
    target_well = str(target["well_id"].iloc[0])
    maximum_matches = int(
        get_nested(config, "rgt.graph_edges.maximum_nodes_per_source_well")
    )
    possible: list[tuple[float, str, str, pd.Series, pd.Series]] = []
    for interval in range(len(FORMATION_NAMES) - 1):
        left = source.loc[
            source["formation_interval_id"].eq(interval)
            & source["rgt_slope_per_md"].gt(0.0)
            & source["rgt_stretch_ft_per_interval"].gt(0.0)
        ].sort_values(["rgt_median", "node_id"], kind="mergesort")
        right = target.loc[
            target["formation_interval_id"].eq(interval)
            & target["rgt_slope_per_md"].gt(0.0)
            & target["rgt_stretch_ft_per_interval"].gt(0.0)
        ].sort_values(["rgt_median", "node_id"], kind="mergesort")
        if left.empty or right.empty:
            continue
        source_bounds = stretch_bounds.get((source_well, interval), (math.nan, math.nan))
        target_bounds = stretch_bounds.get((target_well, interval), (math.nan, math.nan))
        for _, left_node in left.iterrows():
            left_stretch = float(left_node["rgt_stretch_ft_per_interval"])
            if np.isfinite(source_bounds).all() and not (
                float(source_bounds[0]) <= left_stretch <= float(source_bounds[1])
            ):
                continue
            distances = np.abs(
                right["rgt_median"].to_numpy(dtype=float)
                - float(left_node["rgt_median"])
            )
            order = np.argsort(distances, kind="mergesort")
            for right_index in order[:1]:
                right_node = right.iloc[int(right_index)]
                right_stretch = float(right_node["rgt_stretch_ft_per_interval"])
                if np.isfinite(target_bounds).all() and not (
                    float(target_bounds[0]) <= right_stretch <= float(target_bounds[1])
                ):
                    continue
                possible.append(
                    (
                        float(distances[int(right_index)]),
                        str(left_node["node_id"]),
                        str(right_node["node_id"]),
                        left_node,
                        right_node,
                    )
                )
    maximum_edge_candidates = int(
        get_nested(config, "rgt.graph_edges.maximum_edge_candidates")
    )
    capped_candidates = sorted(
        possible,
        key=lambda item: (item[0], item[1], item[2]),
    )[:maximum_edge_candidates]
    selected = capped_candidates[:maximum_matches]
    if not selected:
        return None
    offsets_ft: list[float] = []
    offset_scales: list[float] = []
    rgt_mismatch: list[float] = []
    stretch_ratio: list[float] = []
    direction_cost: list[float] = []
    node_pairs: list[str] = []
    for difference, _, _, left_node, right_node in selected:
        common_rgt = 0.5 * (
            float(left_node["rgt_median"]) + float(right_node["rgt_median"])
        )
        offsets_ft.append(
            _node_at_common_rgt(right_node, common_rgt)
            - _node_at_common_rgt(left_node, common_rgt)
        )
        interval = int(left_node["formation_interval_id"])
        thickness_column = THICKNESS_COLUMNS[interval]
        offset_scales.append(
            0.5
            * (
                float(left_node[thickness_column])
                + float(right_node[thickness_column])
            )
        )
        rgt_mismatch.append(float(difference))
        stretch_ratio.append(
            abs(
                math.log(
                    float(right_node["rgt_stretch_ft_per_interval"])
                    / float(left_node["rgt_stretch_ft_per_interval"])
                )
            )
        )
        left_direction = left_node[
            ["trajectory_dx", "trajectory_dy", "trajectory_dz"]
        ].to_numpy(dtype=float)
        right_direction = right_node[
            ["trajectory_dx", "trajectory_dy", "trajectory_dz"]
        ].to_numpy(dtype=float)
        direction_cost.append(
            float(np.clip(1.0 - np.dot(left_direction, right_direction), 0.0, 2.0))
        )
        node_pairs.append(f"{left_node['node_id']}->{right_node['node_id']}")
    scale_ft = max(float(np.median(offset_scales)), 1.0e-9)
    raw_offset_ft = float(np.median(offsets_ft))
    raw_offset_interval = raw_offset_ft / scale_ft
    source_xy = source[["X", "Y"]].median().to_numpy(dtype=float)
    target_xy = target[["X", "Y"]].median().to_numpy(dtype=float)
    spatial_distance = float(np.linalg.norm(source_xy - target_xy))
    spatial_scale = max(
        0.5 * (float(source_bandwidth) + float(target_bandwidth)),
        1.0,
    )
    weights = get_nested(config, "rgt.graph_edges.graph_cost", {})
    graph_cost = (
        float(weights["spatial_distance_weight"]) * spatial_distance / spatial_scale
        + float(weights["rgt_correspondence_weight"]) * float(np.mean(rgt_mismatch))
        + float(weights["log_stretch_ratio_weight"]) * float(np.mean(stretch_ratio))
        + float(weights["trajectory_direction_weight"]) * float(np.mean(direction_cost))
    )
    fold = int(source["fold"].iloc[0])
    return {
        "fold": fold,
        "source_well_id": source_well,
        "target_well_id": target_well,
        "edge_id": f"f{fold:02d}_{source_well}__{target_well}",
        "match_count": int(len(selected)),
        "source_node_ids": "|".join(pair.split("->", maxsplit=1)[0] for pair in node_pairs),
        "target_node_ids": "|".join(pair.split("->", maxsplit=1)[1] for pair in node_pairs),
        "formation_interval_scale_ft": scale_ft,
        "raw_offset_ft": raw_offset_ft,
        "raw_offset_interval": raw_offset_interval,
        "spatial_distance_ft": spatial_distance,
        "mean_rgt_mismatch": float(np.mean(rgt_mismatch)),
        "mean_abs_log_stretch_ratio": float(np.mean(stretch_ratio)),
        "mean_trajectory_cost": float(np.mean(direction_cost)),
        "graph_edge_cost": float(graph_cost),
    }


def build_correspondence_edges(
    nodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = well_profiles(nodes)
    neighbor_count = int(get_nested(config, "rgt.graph_edges.nearest_unique_wells"))
    pairs, bandwidth = nearest_well_pairs(profiles, neighbor_count)
    stretch = get_nested(config, "rgt.graph_edges.stretch_bounds", {})
    bounds = leave_one_well_stretch_bounds(
        nodes,
        float(stretch["lower_quantile"]),
        float(stretch["upper_quantile"]),
    )
    grouped = {
        str(well): group.sort_values(["rgt_median", "node_id"], kind="mergesort")
        for well, group in nodes.groupby("well_id", sort=True)
    }
    records: list[dict[str, Any]] = []
    for source_well, target_well in pairs:
        record = match_well_pair(
            grouped[source_well],
            grouped[target_well],
            source_bandwidth=bandwidth[source_well],
            target_bandwidth=bandwidth[target_well],
            stretch_bounds=bounds,
            config=config,
        )
        if record is not None:
            records.append(record)
    edges = pd.DataFrame(records)
    if len(edges):
        edges = edges.sort_values(list(EDGE_SORT_COLUMNS), kind="mergesort")
    return edges, profiles


class StableUnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}
        self.rank = {str(value): 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right] or (
            self.rank[root_left] == self.rank[root_right]
            and root_left > root_right
        ):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def solve_cycle_consistent_graph(
    edges: pd.DataFrame,
    well_ids: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ordered_wells = sorted(map(str, well_ids))
    index = {well: offset for offset, well in enumerate(ordered_wells)}
    output = edges.copy()
    if output.empty:
        potentials = pd.DataFrame(
            {
                "well_id": ordered_wells,
                "cycle_potential_interval": np.zeros(len(ordered_wells)),
                "component_id": np.arange(len(ordered_wells), dtype=int),
            }
        )
        return output, potentials, {
            "components": len(ordered_wells),
            "edges": 0,
            "fundamental_cycles": 0,
            "cycle_residual_p95": math.inf,
        }
    row_count = len(output)
    rows = np.repeat(np.arange(row_count), 2)
    columns = np.empty(row_count * 2, dtype=int)
    values = np.empty(row_count * 2, dtype=float)
    for edge_index, edge in enumerate(output.itertuples(index=False)):
        columns[2 * edge_index] = index[str(edge.source_well_id)]
        columns[2 * edge_index + 1] = index[str(edge.target_well_id)]
        values[2 * edge_index] = -1.0
        values[2 * edge_index + 1] = 1.0
    incidence = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(row_count, len(ordered_wells)),
    ).tocsr()
    response = output["raw_offset_interval"].to_numpy(dtype=float)
    base_weight = np.exp(
        -np.clip(output["graph_edge_cost"].to_numpy(dtype=float), 0.0, 50.0)
    )
    union = StableUnionFind(ordered_wells)
    for edge in output.itertuples(index=False):
        union.union(str(edge.source_well_id), str(edge.target_well_id))
    roots = {well: union.find(well) for well in ordered_wells}
    root_order = {root: position for position, root in enumerate(sorted(set(roots.values())))}
    gauge_rows = []
    gauge_columns = []
    for root in sorted(set(roots.values())):
        gauge_rows.append(root_order[root])
        gauge_columns.append(index[min(well for well, value in roots.items() if value == root)])
    gauge = sparse.coo_matrix(
        (
            np.ones(len(gauge_rows)),
            (np.asarray(gauge_rows), np.asarray(gauge_columns)),
        ),
        shape=(len(gauge_rows), len(ordered_wells)),
    ).tocsr()
    huber_delta = float(get_nested(config, "rgt.graph_edges.huber_delta"))
    iterations = int(get_nested(config, "rgt.graph_edges.huber_iterations"))
    ridge = float(get_nested(config, "rgt.graph_edges.solve_ridge"))
    weights = base_weight.copy()
    potential = np.zeros(len(ordered_wells), dtype=float)
    for _ in range(iterations):
        weighted = sparse.diags(np.sqrt(np.maximum(weights, 1.0e-12))) @ incidence
        weighted_response = np.sqrt(np.maximum(weights, 1.0e-12)) * response
        ridge_matrix = sparse.eye(len(ordered_wells), format="csr") * math.sqrt(ridge)
        design = sparse.vstack([weighted, gauge, ridge_matrix], format="csr")
        target = np.concatenate(
            [
                weighted_response,
                np.zeros(gauge.shape[0]),
                np.zeros(len(ordered_wells)),
            ]
        )
        potential = lsqr(design, target, atol=1.0e-12, btol=1.0e-12)[0]
        residual = incidence @ potential - response
        robust = np.ones_like(residual)
        outside = np.abs(residual) > huber_delta
        robust[outside] = huber_delta / np.abs(residual[outside])
        weights = base_weight * robust
    residual = incidence @ potential - response
    output["solved_offset_interval"] = [
        potential[index[str(row.target_well_id)]]
        - potential[index[str(row.source_well_id)]]
        for row in output.itertuples(index=False)
    ]
    output["cycle_residual_interval"] = residual
    output["cycle_residual_abs_interval"] = np.abs(residual)
    tree = StableUnionFind(ordered_wells)
    basis: list[bool] = []
    cycle_ids: list[str] = []
    cycle_index = 0
    stable_order = output.sort_values(
        ["graph_edge_cost", "source_well_id", "target_well_id", "edge_id"],
        kind="mergesort",
    ).index
    is_cycle_by_index: dict[int, bool] = {}
    cycle_id_by_index: dict[int, str] = {}
    for edge_index in stable_order:
        edge = output.loc[edge_index]
        closes = not tree.union(
            str(edge["source_well_id"]),
            str(edge["target_well_id"]),
        )
        is_cycle_by_index[int(edge_index)] = closes
        if closes:
            cycle_id_by_index[int(edge_index)] = (
                f"f{int(edge['fold']):02d}_cycle_{cycle_index:06d}"
            )
            cycle_index += 1
        else:
            cycle_id_by_index[int(edge_index)] = ""
    for edge_index in output.index:
        basis.append(is_cycle_by_index[int(edge_index)])
        cycle_ids.append(cycle_id_by_index[int(edge_index)])
    output["fundamental_cycle_edge"] = basis
    output["cycle_id"] = cycle_ids
    potentials = pd.DataFrame(
        {
            "well_id": ordered_wells,
            "cycle_potential_interval": potential,
            "component_id": [root_order[roots[well]] for well in ordered_wells],
        }
    )
    cycle_residual = output.loc[
        output["fundamental_cycle_edge"], "cycle_residual_abs_interval"
    ].to_numpy(dtype=float)
    manifest = {
        "components": int(len(set(roots.values()))),
        "nodes": int(len(ordered_wells)),
        "edges": int(len(output)),
        "tree_edges": int((~output["fundamental_cycle_edge"]).sum()),
        "fundamental_cycles": int(output["fundamental_cycle_edge"].sum()),
        "cycle_residual_p50": (
            float(np.quantile(cycle_residual, 0.50))
            if len(cycle_residual)
            else math.inf
        ),
        "cycle_residual_p95": (
            float(np.quantile(cycle_residual, 0.95))
            if len(cycle_residual)
            else math.inf
        ),
    }
    output = output.sort_values(list(EDGE_SORT_COLUMNS), kind="mergesort")
    return output, potentials, manifest


# %% [markdown]
# ## 6. Deterministic k-shortest scenario paths and reference-GR templates

# %%
def deterministic_k_shortest_paths(
    adjacency: Mapping[str, Sequence[tuple[str, float]]],
    start: str,
    goal: str,
    maximum_paths: int,
) -> list[tuple[float, tuple[str, ...]]]:
    if maximum_paths <= 0:
        return []
    heap: list[tuple[float, tuple[str, ...]]] = [(0.0, (str(start),))]
    results: list[tuple[float, tuple[str, ...]]] = []
    while heap and len(results) < maximum_paths:
        cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            results.append((float(cost), path))
            continue
        neighbors = sorted(
            adjacency.get(node, ()),
            key=lambda item: (float(item[1]), str(item[0])),
        )
        for neighbor, edge_cost in neighbors:
            neighbor_text = str(neighbor)
            if neighbor_text in path:
                continue
            value = float(edge_cost)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("k-shortest graph costs must be finite and nonnegative")
            heapq.heappush(
                heap,
                (float(cost) + value, (*path, neighbor_text)),
            )
    return results


def build_target_control_grid(
    target: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    require_columns(target, TARGET_ALLOWED_COLUMNS, "target-safe well")
    forbidden = TARGET_FORBIDDEN_COLUMNS.intersection(target.columns)
    if forbidden:
        raise ValueError(f"target scenario input exposes forbidden columns: {sorted(forbidden)}")
    ordered = target.sort_values(["MD", "row_idx"], kind="mergesort").reset_index(drop=True)
    md = ordered["MD"].to_numpy(dtype=float)
    stride = float(get_nested(config, "rgt.target_path.control_stride_ft"))
    grid = np.arange(md.min(), md.max() + 0.5 * stride, stride)
    grid = np.unique(np.clip(grid, md.min(), md.max()))
    positions = np.searchsorted(md, grid, side="left")
    positions = np.clip(positions, 0, len(md) - 1)
    left = np.clip(positions - 1, 0, len(md) - 1)
    choose_left = np.abs(md[left] - grid) <= np.abs(md[positions] - grid)
    positions = np.where(choose_left, left, positions)
    known = np.flatnonzero(np.isfinite(ordered["TVT_input"].to_numpy(dtype=float)))
    extra = [0, len(ordered) - 1]
    if len(known):
        extra.append(int(known[-1]))
    selected = np.unique(np.concatenate([positions, np.asarray(extra, dtype=int)]))
    control = ordered.iloc[selected].copy()
    control["control_index"] = np.arange(len(control), dtype=np.int32)
    return control


def select_target_candidate_wells(
    control: pd.DataFrame,
    profiles: pd.DataFrame,
    maximum_wells: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    if profiles.empty:
        return [], {}, {}
    known = control.loc[np.isfinite(control["TVT_input"].to_numpy(dtype=float))]
    prefix_reference = known if len(known) else control.iloc[:1]
    suffix_reference = control.iloc[-max(1, min(8, len(control))) :]
    prefix_xy = prefix_reference[["X", "Y"]].median().to_numpy(dtype=float)
    suffix_xy = suffix_reference[["X", "Y"]].median().to_numpy(dtype=float)
    records: list[tuple[float, str, float, float]] = []
    for row in profiles.itertuples(index=False):
        point = np.asarray([float(row.centroid_x), float(row.centroid_y)])
        prefix_distance = float(np.linalg.norm(point - prefix_xy))
        suffix_distance = float(np.linalg.norm(point - suffix_xy))
        records.append(
            (
                0.5 * (prefix_distance + suffix_distance),
                str(row.well_id),
                prefix_distance,
                suffix_distance,
            )
        )
    selected = sorted(records, key=lambda item: (item[0], item[1]))[:maximum_wells]
    wells = [item[1] for item in selected]
    prefix = {item[1]: item[2] for item in selected}
    suffix = {item[1]: item[3] for item in selected}
    return wells, prefix, suffix


def build_target_route_graph(
    candidate_wells: Sequence[str],
    prefix_distance: Mapping[str, float],
    suffix_distance: Mapping[str, float],
    edges: pd.DataFrame,
) -> dict[str, list[tuple[str, float]]]:
    start = "__target_start__"
    goal = "__target_goal__"
    candidates = sorted(map(str, candidate_wells))
    all_distances = [
        float(prefix_distance[well]) for well in candidates
    ] + [float(suffix_distance[well]) for well in candidates]
    scale = max(float(np.quantile(all_distances, 0.95)), 1.0) if all_distances else 1.0
    adjacency: dict[str, list[tuple[str, float]]] = {start: [], goal: []}
    for well in candidates:
        adjacency.setdefault(well, [])
        adjacency[start].append((well, float(prefix_distance[well]) / scale))
        adjacency[well].append((goal, float(suffix_distance[well]) / scale))
    allowed = set(candidates)
    for edge in edges.itertuples(index=False):
        source = str(edge.source_well_id)
        target = str(edge.target_well_id)
        if source not in allowed or target not in allowed:
            continue
        cost = float(edge.graph_edge_cost) + abs(
            float(edge.cycle_residual_interval)
        )
        adjacency[source].append((target, cost))
        adjacency[target].append((source, cost))
    for node in adjacency:
        adjacency[node] = sorted(adjacency[node], key=lambda item: (item[1], item[0]))
    return adjacency


def _normalized_positions(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    span = float(array.max() - array.min())
    if span <= 0.0:
        return np.zeros(len(array), dtype=float)
    return (array - array.min()) / span


def _nearest_source_node_ids(
    normalized_query: np.ndarray,
    donor_nodes: pd.DataFrame,
) -> np.ndarray:
    donor_position = _normalized_positions(donor_nodes["MD"].to_numpy(dtype=float))
    order = np.searchsorted(donor_position, normalized_query, side="left")
    order = np.clip(order, 0, len(donor_nodes) - 1)
    left = np.clip(order - 1, 0, len(donor_nodes) - 1)
    choose_left = (
        np.abs(donor_position[left] - normalized_query)
        <= np.abs(donor_position[order] - normalized_query)
    )
    selected = np.where(choose_left, left, order)
    return donor_nodes.iloc[selected]["node_id"].astype(str).to_numpy()


def integrate_route_path(
    route: Sequence[str],
    control: pd.DataFrame,
    nodes_by_well: Mapping[str, pd.DataFrame],
) -> dict[str, Any] | None:
    if not route:
        return None
    ordered_control = control.sort_values("MD", kind="mergesort").reset_index(drop=True)
    md = ordered_control["MD"].to_numpy(dtype=float)
    normalized = _normalized_positions(md)
    route_count = len(route)
    route_position = np.minimum(
        np.floor(normalized * route_count).astype(int),
        route_count - 1,
    )
    local_position = np.clip(normalized * route_count - route_position, 0.0, 1.0)
    s_rate = np.full(len(ordered_control), np.nan, dtype=float)
    rgt_rate = np.full(len(ordered_control), np.nan, dtype=float)
    source_rgt = np.full(len(ordered_control), np.nan, dtype=float)
    source_node_ids = np.full(len(ordered_control), "", dtype=object)
    source_gr = np.full(len(ordered_control), np.nan, dtype=float)
    source_gr_difference = np.full(len(ordered_control), np.nan, dtype=float)
    stretches: list[float] = []
    for route_index, well_id in enumerate(route):
        mask = route_position == route_index
        if not mask.any():
            continue
        donor = nodes_by_well[str(well_id)].sort_values(
            ["MD", "node_id"], kind="mergesort"
        )
        donor_u = _normalized_positions(donor["MD"].to_numpy(dtype=float))
        query_u = local_position[mask]
        for column, destination in (
            ("structural_slope_per_md", s_rate),
            ("rgt_slope_per_md", rgt_rate),
            ("rgt_median", source_rgt),
            ("reference_gr_median", source_gr),
            ("reference_gr_diff_median", source_gr_difference),
        ):
            destination[mask] = np.interp(
                query_u,
                donor_u,
                donor[column].to_numpy(dtype=float),
            )
        source_node_ids[mask] = _nearest_source_node_ids(query_u, donor)
        stretches.extend(
            donor["rgt_stretch_ft_per_interval"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
            .tolist()
        )
    if (
        not np.isfinite(s_rate).all()
        or not np.isfinite(rgt_rate).all()
        or np.any(rgt_rate <= 0.0)
    ):
        return None
    cumulative_s = np.zeros(len(md), dtype=float)
    cumulative_rgt = np.zeros(len(md), dtype=float)
    for index in range(1, len(md)):
        delta_md = float(md[index] - md[index - 1])
        if delta_md <= 0.0:
            return None
        cumulative_s[index] = cumulative_s[index - 1] + 0.5 * (
            s_rate[index - 1] + s_rate[index]
        ) * delta_md
        cumulative_rgt[index] = cumulative_rgt[index - 1] + 0.5 * (
            rgt_rate[index - 1] + rgt_rate[index]
        ) * delta_md
    known = np.flatnonzero(
        np.isfinite(ordered_control["TVT_input"].to_numpy(dtype=float))
    )
    if len(known) == 0:
        return None
    anchor = int(known[-1])
    anchor_s = float(ordered_control.loc[anchor, "TVT_input"]) + float(
        ordered_control.loc[anchor, "Z"]
    )
    structural_s = cumulative_s + anchor_s - cumulative_s[anchor]
    rgt = cumulative_rgt + source_rgt[anchor] - cumulative_rgt[anchor]
    tvt = structural_s - ordered_control["Z"].to_numpy(dtype=float)
    if np.any(np.diff(rgt) <= 0.0) or np.any(np.diff(tvt) < -1.0e-9):
        return None
    prefix = np.isfinite(ordered_control["TVT_input"].to_numpy(dtype=float))
    prefix_rmse = rmse(
        ordered_control.loc[prefix, "TVT_input"].to_numpy(dtype=float),
        tvt[prefix],
    )
    stretch_array = np.asarray(stretches, dtype=float)
    stretch_cost = (
        float(np.mean(np.abs(np.log(stretch_array / np.median(stretch_array)))))
        if len(stretch_array) and np.median(stretch_array) > 0.0
        else math.inf
    )
    return {
        "control": ordered_control,
        "tvt": tvt,
        "rgt": rgt,
        "structural_s": structural_s,
        "source_node_ids": source_node_ids,
        "source_gr": source_gr,
        "source_gr_difference": source_gr_difference,
        "prefix_rmse": prefix_rmse,
        "stretch_cost": stretch_cost,
    }


def _route_edge_diagnostics(
    route: Sequence[str],
    edges: pd.DataFrame,
) -> tuple[float, float]:
    if len(route) <= 1:
        incident = edges.loc[
            edges["source_well_id"].eq(route[0])
            | edges["target_well_id"].eq(route[0])
        ]
        if incident.empty:
            return 0.0, math.inf
        return (
            float(incident["graph_edge_cost"].median()),
            float(incident["cycle_residual_abs_interval"].median()),
        )
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, edge in edges.iterrows():
        lookup[tuple(sorted((str(edge["source_well_id"]), str(edge["target_well_id"]))))] = edge
    graph_cost: list[float] = []
    cycle_cost: list[float] = []
    for source, target in zip(route[:-1], route[1:], strict=True):
        edge = lookup.get(tuple(sorted((str(source), str(target)))))
        if edge is None:
            return math.inf, math.inf
        graph_cost.append(float(edge["graph_edge_cost"]))
        cycle_cost.append(float(edge["cycle_residual_abs_interval"]))
    return float(np.mean(graph_cost)), float(np.mean(cycle_cost))


def _path_sha256(
    *,
    route: Sequence[str],
    md: np.ndarray,
    tvt: np.ndarray,
    rgt: np.ndarray,
) -> str:
    return sha256_bytes(
        stable_json_bytes(
            {
                "route": list(map(str, route)),
                "md": np.round(np.asarray(md, dtype=float), 8).tolist(),
                "tvt": np.round(np.asarray(tvt, dtype=float), 8).tolist(),
                "rgt": np.round(np.asarray(rgt, dtype=float), 8).tolist(),
            }
        )
    )


def generate_scenario_bank_for_well(
    target: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    profiles: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    control = build_target_control_grid(target, config)
    maximum_neighbor_wells = int(
        get_nested(config, "rgt.graph_edges.nearest_unique_wells")
    )
    candidates, prefix_distance, suffix_distance = select_target_candidate_wells(
        control,
        profiles,
        maximum_neighbor_wells,
    )
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), {
            "well_id": str(target["well_id"].iloc[0]),
            "query_nodes": int(len(control)),
            "covered_query_nodes": 0,
            "raw_routes": 0,
            "scenario_count": 0,
        }
    route_graph = build_target_route_graph(
        candidates,
        prefix_distance,
        suffix_distance,
        edges,
    )
    raw_limit = int(get_nested(config, "rgt.target_path.route_enumeration_limit"))
    shortest = deterministic_k_shortest_paths(
        route_graph,
        "__target_start__",
        "__target_goal__",
        raw_limit,
    )
    nodes_by_well = {
        str(well): group
        for well, group in nodes.loc[nodes["well_id"].isin(candidates)].groupby(
            "well_id", sort=True
        )
    }
    weights = get_nested(config, "rgt.scenario_bank.prior_cost_weights", {})
    prefix_scale = float(
        get_nested(config, "rgt.scenario_bank.prefix_gauge_misfit_scale_ft")
    )
    raw_paths: list[dict[str, Any]] = []
    for k_cost, path in shortest:
        route = tuple(node for node in path[1:-1] if node in nodes_by_well)
        if not route:
            continue
        integrated = integrate_route_path(route, control, nodes_by_well)
        if integrated is None:
            continue
        graph_cost, cycle_cost = _route_edge_diagnostics(route, edges)
        if not np.isfinite([graph_cost, cycle_cost, integrated["stretch_cost"]]).all():
            continue
        prefix_cost = float(integrated["prefix_rmse"]) / max(prefix_scale, 1.0e-9)
        total = (
            float(weights["graph_edge_cost"]) * (float(k_cost) + graph_cost)
            + float(weights["cycle_residual_cost"]) * cycle_cost
            + float(weights["prefix_gauge_misfit"]) * prefix_cost
            + float(weights["rgt_stretch_cost"]) * float(integrated["stretch_cost"])
        )
        path_sha = _path_sha256(
            route=route,
            md=integrated["control"]["MD"].to_numpy(dtype=float),
            tvt=integrated["tvt"],
            rgt=integrated["rgt"],
        )
        raw_paths.append(
            {
                "route": route,
                "node_sequence": tuple(
                    map(str, integrated["source_node_ids"].tolist())
                ),
                "integrated": integrated,
                "k_shortest_cost": float(k_cost),
                "graph_edge_cost": graph_cost,
                "cycle_residual_cost": cycle_cost,
                "prefix_gauge_misfit": prefix_cost,
                "rgt_stretch_cost": float(integrated["stretch_cost"]),
                "total_graph_cost": float(total),
                "path_sha256": path_sha,
            }
        )
    raw_paths = sorted(
        raw_paths,
        key=lambda item: (
            item["total_graph_cost"],
            item["route"],
            item["node_sequence"],
            item["path_sha256"],
        ),
    )
    diversity = float(
        get_nested(config, "rgt.scenario_bank.path_rms_separation_ft_min")
    )
    maximum_scenarios = int(
        get_nested(config, "rgt.scenario_bank.maximum_scenarios_per_well")
    )
    accepted: list[dict[str, Any]] = []
    for candidate in raw_paths:
        vector = np.asarray(candidate["integrated"]["tvt"], dtype=float)
        if all(
            rmse(vector, np.asarray(previous["integrated"]["tvt"], dtype=float))
            >= diversity
            for previous in accepted
        ):
            accepted.append(candidate)
        if len(accepted) >= maximum_scenarios:
            break
    path_records: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    fold = int(target["fold"].iloc[0])
    well_id = str(target["well_id"].iloc[0])
    for scenario_rank, candidate in enumerate(accepted):
        integrated = candidate["integrated"]
        scenario_id = f"f{fold:02d}_{well_id}_scenario_{scenario_rank:02d}"
        route_text = "|".join(candidate["route"])
        for row_index, row in enumerate(integrated["control"].itertuples(index=False)):
            base = {
                "fold": fold,
                "well_id": well_id,
                "scenario_rank": int(scenario_rank),
                "scenario_id": scenario_id,
                "control_index": int(row.control_index),
                "row_idx": int(row.row_idx),
                "MD": float(row.MD),
                "X": float(row.X),
                "Y": float(row.Y),
                "Z": float(row.Z),
                "tvt_path": float(integrated["tvt"][row_index]),
                "structural_s_path": float(integrated["structural_s"][row_index]),
                "rgt_path": float(integrated["rgt"][row_index]),
                "source_node_id": str(integrated["source_node_ids"][row_index]),
                "source_well_sequence": route_text,
                "k_shortest_cost": float(candidate["k_shortest_cost"]),
                "graph_edge_cost": float(candidate["graph_edge_cost"]),
                "cycle_residual_cost": float(candidate["cycle_residual_cost"]),
                "prefix_gauge_misfit": float(candidate["prefix_gauge_misfit"]),
                "rgt_stretch_cost": float(candidate["rgt_stretch_cost"]),
                "total_graph_cost": float(candidate["total_graph_cost"]),
                "path_sha256": str(candidate["path_sha256"]),
            }
            path_records.append(base)
            reference_records.append(
                {
                    "fold": fold,
                    "well_id": well_id,
                    "scenario_rank": int(scenario_rank),
                    "scenario_id": scenario_id,
                    "control_index": int(row.control_index),
                    "MD": float(row.MD),
                    "rgt_path": float(integrated["rgt"][row_index]),
                    "source_node_id": str(integrated["source_node_ids"][row_index]),
                    "reference_gr": float(integrated["source_gr"][row_index]),
                    "reference_gr_diff": float(
                        integrated["source_gr_difference"][row_index]
                    ),
                    "path_sha256": str(candidate["path_sha256"]),
                }
            )
    paths = pd.DataFrame(path_records)
    references = pd.DataFrame(reference_records)
    if len(paths):
        paths = paths.sort_values(list(PATH_SORT_COLUMNS), kind="mergesort")
        references = references.sort_values(
            list(REFERENCE_SORT_COLUMNS), kind="mergesort"
        )
    return paths, references, {
        "fold": fold,
        "well_id": well_id,
        "query_nodes": int(len(control)),
        "covered_query_nodes": int(len(control) if raw_paths else 0),
        "candidate_source_wells": int(len(candidates)),
        "raw_routes": int(len(raw_paths)),
        "scenario_count": int(len(accepted)),
        "finite_paths": bool(
            len(paths)
            and np.isfinite(
                paths[["tvt_path", "rgt_path", "total_graph_cost"]].to_numpy(float)
            ).all()
        ),
    }


# %% [markdown]
# ## 7. Target-free freeze, Stage 0 gate, and resource projection

# %%
def _concat_nonempty(
    frames: Sequence[pd.DataFrame],
    *,
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    selected = [frame for frame in frames if frame is not None and len(frame)]
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True).sort_values(
        [column for column in sort_columns if column in selected[0].columns],
        kind="mergesort",
    )


def run_fold_target_free(
    *,
    fold: int,
    file_by_well: Mapping[str, Path],
    input_sha_by_well: Mapping[str, str],
    fold_by_well: Mapping[str, int],
    selected_target_wells: set[str] | None,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source_wells = sorted(
        well for well, assigned in fold_by_well.items() if int(assigned) != int(fold)
    )
    target_wells = sorted(
        well
        for well, assigned in fold_by_well.items()
        if int(assigned) == int(fold)
        and (selected_target_wells is None or well in selected_target_wells)
    )
    ledger.validate_disjoint(source_wells, target_wells)
    fold_manifest = pd.DataFrame(
        [
            {
                "fold": int(fold),
                "well_id": well,
                "role": "outer_train" if well in set(source_wells) else "outer_valid",
                "selected_target": bool(well in set(target_wells)),
                "horizontal_file_sha256": str(input_sha_by_well[well]),
            }
            for well in sorted(set(source_wells).union(
                well for well, assigned in fold_by_well.items() if int(assigned) == int(fold)
            ))
        ]
    )
    graph_started = time.perf_counter()
    node_parts: list[pd.DataFrame] = []
    rgt_diagnostics: list[dict[str, Any]] = []
    for well_id in source_wells:
        source = read_source_well(file_by_well[well_id], fold, ledger)
        nodes, diagnostics = build_rgt_nodes_for_well(source, config)
        node_parts.append(nodes)
        rgt_diagnostics.append(diagnostics)
    nodes = _concat_nonempty(node_parts, sort_columns=NODE_SORT_COLUMNS)
    if nodes.empty:
        raise ValueError(f"fold {fold} RGT graph has no nodes")
    raw_edges, profiles = build_correspondence_edges(nodes, config)
    edges, potentials, cycle = solve_cycle_consistent_graph(
        raw_edges,
        sorted(nodes["well_id"].astype(str).unique()),
        config,
    )
    graph_seconds = time.perf_counter() - graph_started
    path_started = time.perf_counter()
    path_parts: list[pd.DataFrame] = []
    reference_parts: list[pd.DataFrame] = []
    bank_diagnostics: list[dict[str, Any]] = []
    for well_id in target_wells:
        target = read_target_safe_well(file_by_well[well_id], fold, ledger)
        paths, references, diagnostics = generate_scenario_bank_for_well(
            target,
            nodes,
            edges,
            profiles,
            config,
        )
        path_parts.append(paths)
        reference_parts.append(references)
        bank_diagnostics.append(diagnostics)
    path_seconds = time.perf_counter() - path_started
    paths = _concat_nonempty(path_parts, sort_columns=PATH_SORT_COLUMNS)
    references = _concat_nonempty(
        reference_parts,
        sort_columns=REFERENCE_SORT_COLUMNS,
    )
    return {
        "fold": int(fold),
        "source_wells": source_wells,
        "target_wells": target_wells,
        "fold_manifest": fold_manifest,
        "nodes": nodes,
        "edges": edges,
        "potentials": potentials.assign(fold=int(fold)),
        "profiles": profiles,
        "paths": paths,
        "references": references,
        "rgt_diagnostics": pd.DataFrame(rgt_diagnostics),
        "bank_diagnostics": pd.DataFrame(bank_diagnostics),
        "cycle_manifest": {"fold": int(fold), **cycle},
        "graph_seconds": float(graph_seconds),
        "path_seconds": float(path_seconds),
    }


def freeze_target_free_outputs(
    frames: Mapping[str, pd.DataFrame],
    cycle_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    sort_contract = {
        "fold_manifest": ("fold", "role", "well_id"),
        "role_read_ledger": ("fold", "role", "well_id"),
        "rgt_nodes": NODE_SORT_COLUMNS,
        "graph_edges": EDGE_SORT_COLUMNS,
        "target_paths": PATH_SORT_COLUMNS,
        "reference_gr_templates": REFERENCE_SORT_COLUMNS,
    }
    hashes = {
        name: frame_content_sha256(frame, sort_columns=sort_contract[name])
        for name, frame in frames.items()
        if name in sort_contract
    }
    hashes["cycle_manifest"] = sha256_bytes(
        stable_json_bytes(list(cycle_manifests))
    )
    return hashes


def evaluate_stage0(
    *,
    frames: Mapping[str, pd.DataFrame],
    fold_results: Sequence[Mapping[str, Any]],
    cycle_manifests: Sequence[Mapping[str, Any]],
    ledger: RoleReadLedger,
    full_run: bool,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "gates.stage0_target_free", {})
    rgt = frames["rgt_diagnostics"]
    bank = frames["bank_diagnostics"]
    total_source_rows = int(rgt["rows"].sum()) if len(rgt) else 0
    rgt_available_rows = int(rgt["rgt_available_rows"].sum()) if len(rgt) else 0
    rgt_coverage = (
        rgt_available_rows / total_source_rows if total_source_rows else 0.0
    )
    total_queries = int(bank["query_nodes"].sum()) if len(bank) else 0
    covered_queries = int(bank["covered_query_nodes"].sum()) if len(bank) else 0
    graph_query_coverage = (
        covered_queries / total_queries if total_queries else 0.0
    )
    scenario_counts = (
        bank["scenario_count"].to_numpy(dtype=float)
        if len(bank)
        else np.asarray([], dtype=float)
    )
    minimum_scenarios = int(
        get_nested(config, "rgt.scenario_bank.minimum_scenarios_per_well")
    )
    bank_coverage = (
        float(np.mean(scenario_counts >= minimum_scenarios))
        if len(scenario_counts)
        else 0.0
    )
    scenario_p05 = (
        float(np.quantile(scenario_counts, 0.05))
        if len(scenario_counts)
        else 0.0
    )
    paths = frames["target_paths"]
    finite_path_coverage = (
        float(
            np.isfinite(
                paths[["tvt_path", "rgt_path", "total_graph_cost"]].to_numpy(float)
            ).all(axis=1).mean()
        )
        if len(paths)
        else 0.0
    )
    graph_edges = frames["graph_edges"]
    cycle_residuals = (
        graph_edges.loc[
            graph_edges["fundamental_cycle_edge"].astype(bool),
            "cycle_residual_abs_interval",
        ].to_numpy(dtype=float)
        if len(graph_edges)
        else np.asarray([], dtype=float)
    )
    cycle_p95 = (
        float(np.quantile(cycle_residuals, 0.95))
        if len(cycle_residuals)
        else math.inf
    )
    selected_wells = int(bank["well_id"].nunique()) if len(bank) else 0
    graph_seconds = float(sum(float(result["graph_seconds"]) for result in fold_results))
    target_seconds = float(sum(float(result["path_seconds"]) for result in fold_results))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    projected_runtime = graph_seconds + (
        target_seconds * expected_wells / selected_wells
        if selected_wells
        else math.inf
    )
    ledger_summary = ledger.summary()
    target_safe_rows = int(ledger_summary["target_safe_rows"])
    target_well_count = int(bank["well_id"].nunique()) if len(bank) else 0
    expected_folds = list(get_nested(config, "validation.expected_folds"))
    observed_folds = sorted(
        int(value)
        for value in bank["fold"].unique().tolist()
    ) if len(bank) else []
    checks = {
        "rgt_source_coverage": rgt_coverage
        >= float(gates["rgt_source_coverage_min"]),
        "graph_query_coverage": graph_query_coverage
        >= float(gates["graph_query_coverage_min"]),
        "scenario_bank_well_coverage": bank_coverage
        >= float(gates["scenario_bank_well_coverage_min"]),
        "scenario_count_p05": scenario_p05
        >= float(gates["scenario_count_p05_min"]),
        "finite_path_coverage": finite_path_coverage
        >= float(gates["finite_path_coverage_min"]),
        "cycle_residual_p95": cycle_p95
        <= float(gates["cycle_residual_p95_max"]),
        "target_gr_reads_zero": int(
            ledger_summary["target_gr_reads_before_freeze"]
        )
        <= int(gates["target_gr_reads_max"]),
        "valid_formation_reads_zero": int(
            ledger_summary["valid_formation_reads_before_freeze"]
        )
        <= int(gates["valid_formation_reads_max"]),
        "valid_suffix_truth_reads_zero": int(
            ledger_summary["valid_suffix_truth_reads_before_freeze"]
        )
        <= int(gates["valid_suffix_truth_reads_max"]),
        "source_valid_overlap_zero": int(ledger_summary["source_valid_overlap"])
        <= int(gates["source_valid_overlap_max"]),
        "projected_runtime": projected_runtime
        <= float(gates["projected_runtime_seconds_max"]),
        "projected_peak_rss": peak_rss_gb()
        <= float(gates["projected_peak_rss_gb_max"]),
    }
    full_contract = {
        "rows": target_safe_rows
        == int(get_nested(config, "validation.expected_rows")),
        "wells": target_well_count
        == int(get_nested(config, "validation.expected_wells")),
        "folds": observed_folds == expected_folds,
    }
    if full_run:
        checks.update({f"full_{key}": value for key, value in full_contract.items()})
    return {
        "passed": bool(all(checks.values())),
        "mode": "full_run" if full_run else "stage0_resource_preflight",
        "checks": checks,
        "full_contract": full_contract,
        "full_contract_evaluated": bool(full_run),
        "rows": target_safe_rows,
        "wells": target_well_count,
        "folds": observed_folds,
        "rgt_source_coverage": float(rgt_coverage),
        "graph_query_coverage": float(graph_query_coverage),
        "scenario_bank_well_coverage": float(bank_coverage),
        "scenario_count_p05": float(scenario_p05),
        "finite_path_coverage": float(finite_path_coverage),
        "cycle_residual_p95": float(cycle_p95),
        "graph_runtime_seconds": graph_seconds,
        "sample_target_runtime_seconds": target_seconds,
        "projected_runtime_seconds": float(projected_runtime),
        "projected_peak_rss_gb": float(peak_rss_gb()),
        "ledger": ledger_summary,
    }


# %% [markdown]
# ## 8. Rolling-origin prefix and truth-late H512 oracle readouts

# %%
def interpolate_scenario(
    scenario: pd.DataFrame,
    query_md: np.ndarray,
) -> np.ndarray:
    ordered = scenario.sort_values(["MD", "control_index"], kind="mergesort")
    return np.interp(
        np.asarray(query_md, dtype=float),
        ordered["MD"].to_numpy(dtype=float),
        ordered["tvt_path"].to_numpy(dtype=float),
    )


def _scenario_sse(
    paths: pd.DataFrame,
    query_md: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, int | None]:
    values: list[tuple[float, int]] = []
    for scenario_rank, scenario in paths.groupby("scenario_rank", sort=True):
        prediction = interpolate_scenario(scenario, query_md)
        if np.isfinite(prediction).all():
            values.append(
                (
                    float(np.sum(np.square(np.asarray(truth, dtype=float) - prediction))),
                    int(scenario_rank),
                )
            )
    if not values:
        return math.inf, None
    return min(values, key=lambda item: (item[0], item[1]))


def build_prefix_rolling_origin_readout(
    *,
    fold_results: Sequence[Mapping[str, Any]],
    file_by_well: Mapping[str, Path],
    parent_oof: pd.DataFrame,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gates = get_nested(config, "gates.stage1_prefix_rolling_origin", {})
    heldout_rows = int(gates["heldout_prefix_rows"])
    minimum_known = int(gates["minimum_known_prefix_rows"])
    records: list[dict[str, Any]] = []
    for result in fold_results:
        fold = int(result["fold"])
        for well_id in result["target_wells"]:
            target = read_target_safe_well(file_by_well[well_id], fold, ledger)
            finite = np.flatnonzero(np.isfinite(target["TVT_input"].to_numpy(float)))
            if len(finite) < minimum_known:
                continue
            heldout = finite[-heldout_rows:]
            truth = target.loc[heldout, "TVT_input"].to_numpy(dtype=float)
            masked = target.copy()
            masked.loc[heldout, "TVT_input"] = np.nan
            paths, _, diagnostics = generate_scenario_bank_for_well(
                masked,
                result["nodes"],
                result["edges"],
                result["profiles"],
                config,
            )
            rolling_bank_sha256 = frame_content_sha256(
                paths,
                sort_columns=PATH_SORT_COLUMNS,
            )
            oracle_sse, best_rank = _scenario_sse(
                paths,
                target.loc[heldout, "MD"].to_numpy(dtype=float),
                truth,
            )
            control = parent_oof.loc[
                parent_oof["well_id"].eq(well_id)
                & parent_oof["row_idx"].isin(target.loc[heldout, "row_idx"])
            ].sort_values("row_idx", kind="mergesort")
            if len(control) != len(heldout):
                raise ValueError(f"exp226 prefix control rows are missing for {well_id}")
            control_sse = float(
                np.sum(
                    np.square(
                        truth
                        - control["tvt_pred"].to_numpy(dtype=float)
                    )
                )
            )
            rows = int(len(heldout))
            records.append(
                {
                    "fold": fold,
                    "well_id": well_id,
                    "rows": rows,
                    "scenario_count": int(diagnostics["scenario_count"]),
                    "rolling_bank_sha256": rolling_bank_sha256,
                    "best_scenario_rank": best_rank,
                    "oracle_sse": oracle_sse,
                    "control_sse": control_sse,
                    "oracle_rmse": math.sqrt(oracle_sse / rows),
                    "control_rmse": math.sqrt(control_sse / rows),
                    "gain_ft": math.sqrt(control_sse / rows)
                    - math.sqrt(oracle_sse / rows),
                }
            )
    readout = pd.DataFrame(records)
    fold_records: list[dict[str, Any]] = []
    for fold in list(get_nested(config, "validation.expected_folds")):
        group = readout.loc[readout["fold"].eq(fold)]
        rows = int(group["rows"].sum()) if len(group) else 0
        oracle_sse = float(group["oracle_sse"].sum()) if len(group) else math.inf
        control_sse = float(group["control_sse"].sum()) if len(group) else math.inf
        fold_records.append(
            {
                "fold": int(fold),
                "rows": rows,
                "oracle_rmse": math.sqrt(oracle_sse / rows) if rows else math.inf,
                "control_rmse": math.sqrt(control_sse / rows) if rows else math.inf,
            }
        )
    folds = pd.DataFrame(fold_records)
    total_rows = int(readout["rows"].sum()) if len(readout) else 0
    pooled_oracle = (
        math.sqrt(float(readout["oracle_sse"].sum()) / total_rows)
        if total_rows
        else math.inf
    )
    pooled_control = (
        math.sqrt(float(readout["control_sse"].sum()) / total_rows)
        if total_rows
        else math.inf
    )
    coverage = (
        float(np.isfinite(readout["oracle_rmse"]).mean()) if len(readout) else 0.0
    )
    positive_folds = int((folds["oracle_rmse"] < folds["control_rmse"]).sum())
    checks = {
        "oracle_gain": pooled_control - pooled_oracle
        >= float(gates["oracle_gain_vs_exp226_ft_min"]),
        "positive_folds": positive_folds
        >= int(gates["positive_fold_count_min"]),
        "candidate_coverage": coverage
        >= float(gates["candidate_coverage_min"]),
    }
    return readout, {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "rows": total_rows,
        "wells": int(readout["well_id"].nunique()) if len(readout) else 0,
        "oracle_rmse": float(pooled_oracle),
        "control_rmse": float(pooled_control),
        "oracle_gain_ft": float(pooled_control - pooled_oracle),
        "positive_folds": positive_folds,
        "candidate_coverage": coverage,
        "fold_metrics": fold_records,
    }


def build_truth_late_h512_readout(
    *,
    paths: pd.DataFrame,
    fold_results: Sequence[Mapping[str, Any]],
    file_by_well: Mapping[str, Path],
    parent_oof: pd.DataFrame,
    bank_diagnostics: pd.DataFrame,
    ledger: RoleReadLedger,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizon = int(get_nested(config, "rgt.scenario_bank.oracle_granularity_rows"))
    block_records: list[dict[str, Any]] = []
    for result in fold_results:
        fold = int(result["fold"])
        for well_id in result["target_wells"]:
            safe = read_target_safe_well(file_by_well[well_id], fold, ledger)
            truth_frame = read_target_truth_late(file_by_well[well_id], fold, ledger)
            merged = safe.merge(
                truth_frame,
                on=["well_id", "row_idx"],
                how="left",
                validate="one_to_one",
            )
            suffix = merged.loc[merged["TVT_input"].isna()].sort_values(
                "row_idx", kind="mergesort"
            )
            well_paths = paths.loc[paths["well_id"].eq(well_id)]
            control = parent_oof.loc[
                parent_oof["well_id"].eq(well_id)
                & parent_oof["row_idx"].isin(suffix["row_idx"])
            ].sort_values("row_idx", kind="mergesort")
            if len(control) != len(suffix):
                raise ValueError(f"exp226 suffix control rows are missing for {well_id}")
            control_by_row = control.set_index("row_idx")["tvt_pred"]
            for block_index, start in enumerate(range(0, len(suffix), horizon)):
                block = suffix.iloc[start : start + horizon]
                query_md = block["MD"].to_numpy(dtype=float)
                truth = block["TVT"].to_numpy(dtype=float)
                oracle_sse, best_rank = _scenario_sse(well_paths, query_md, truth)
                control_prediction = (
                    block["row_idx"].map(control_by_row).to_numpy(dtype=float)
                )
                control_sse = float(
                    np.sum(np.square(truth - control_prediction))
                )
                suffix_offset = np.arange(start, start + len(block), dtype=int)
                median_offset = float(np.median(suffix_offset))
                if median_offset < 250:
                    scope = "near_0_250"
                elif median_offset < 1000:
                    scope = "mid_250_1000"
                else:
                    scope = "1000_plus"
                block_records.append(
                    {
                        "fold": fold,
                        "well_id": well_id,
                        "block_index": int(block_index),
                        "scope": scope,
                        "rows": int(len(block)),
                        "best_scenario_rank": best_rank,
                        "oracle_sse": oracle_sse,
                        "control_sse": control_sse,
                        "oracle_rmse": math.sqrt(oracle_sse / len(block)),
                        "control_rmse": math.sqrt(control_sse / len(block)),
                    }
                )
    blocks = pd.DataFrame(block_records)

    def aggregate(group: pd.DataFrame, scope: str, fold: int | str) -> dict[str, Any]:
        rows = int(group["rows"].sum()) if len(group) else 0
        oracle_sse = float(group["oracle_sse"].sum()) if len(group) else math.inf
        control_sse = float(group["control_sse"].sum()) if len(group) else math.inf
        oracle_value = math.sqrt(oracle_sse / rows) if rows else math.inf
        control_value = math.sqrt(control_sse / rows) if rows else math.inf
        return {
            "record_type": "summary",
            "scope": scope,
            "fold": fold,
            "rows": rows,
            "oracle_rmse": oracle_value,
            "control_rmse": control_value,
            "gain_ft": control_value - oracle_value,
        }

    summaries = [aggregate(blocks, "overall", "all")]
    for fold in list(get_nested(config, "validation.expected_folds")):
        summaries.append(aggregate(blocks.loc[blocks["fold"].eq(fold)], "overall", fold))
    for scope in ("near_0_250", "mid_250_1000", "1000_plus"):
        summaries.append(
            aggregate(blocks.loc[blocks["scope"].eq(scope)], scope, "all")
        )
    summary = pd.DataFrame(summaries)
    gates = get_nested(config, "gates.stage2_truth_late_scenario_bank", {})
    pooled = summary.loc[
        summary["scope"].eq("overall") & summary["fold"].eq("all")
    ].iloc[0]
    folds = summary.loc[
        summary["scope"].eq("overall") & summary["fold"].ne("all")
    ]
    scopes = summary.loc[summary["scope"].ne("overall")]
    minimum_scenarios = int(
        get_nested(config, "rgt.scenario_bank.minimum_scenarios_per_well")
    )
    bank_coverage = (
        float(
            np.mean(
                bank_diagnostics["scenario_count"].to_numpy(dtype=float)
                >= minimum_scenarios
            )
        )
        if len(bank_diagnostics)
        else 0.0
    )
    unique_fraction = (
        float(
            np.mean(
                bank_diagnostics["scenario_count"].to_numpy(dtype=float) >= 2
            )
        )
        if len(bank_diagnostics)
        else 0.0
    )
    positive_folds = int((folds["oracle_rmse"] < folds["control_rmse"]).sum())
    scope_tolerance = float(gates["scope_oracle_regression_tolerance_ft"])
    checks = {
        "scenario_oracle_rmse": float(pooled["oracle_rmse"])
        <= float(gates["scenario_oracle_rmse_max_ft"]),
        "positive_folds": positive_folds
        >= int(gates["positive_fold_count_min"]),
        "scenario_bank_well_coverage": bank_coverage
        >= float(gates["scenario_bank_well_coverage_min"]),
        "unique_scenario_well_fraction": unique_fraction
        >= float(gates["unique_scenario_well_fraction_min"]),
        "scope_nonregression": bool(
            len(scopes)
            and (
                scopes["oracle_rmse"]
                <= scopes["control_rmse"] + scope_tolerance
            ).all()
        ),
    }
    readout = pd.concat(
        [
            blocks.assign(record_type="h512_block"),
            summary,
        ],
        ignore_index=True,
        sort=False,
    )
    readout["fold"] = readout["fold"].astype(str)
    return readout, {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "oracle_granularity_rows": horizon,
        "oracle_rmse": float(pooled["oracle_rmse"]),
        "control_rmse": float(pooled["control_rmse"]),
        "oracle_gain_ft": float(pooled["gain_ft"]),
        "positive_folds": positive_folds,
        "scenario_bank_well_coverage": bank_coverage,
        "unique_scenario_well_fraction": unique_fraction,
        "summary": summaries,
    }


# %% [markdown]
# ## 9. Artifact manifest and execution orchestration

# %%
ARTIFACT_FILENAMES = {
    "fold_manifest": f"{EXPERIMENT_NAME}_fold_manifest.csv",
    "role_read_ledger": f"{EXPERIMENT_NAME}_role_read_ledger.csv",
    "rgt_nodes": f"{EXPERIMENT_NAME}_rgt_nodes.parquet",
    "graph_edges": f"{EXPERIMENT_NAME}_graph_edges.parquet",
    "target_paths": f"{EXPERIMENT_NAME}_target_paths.parquet",
    "reference_gr_templates": f"{EXPERIMENT_NAME}_reference_gr_templates.parquet",
    "prefix_readout": f"{EXPERIMENT_NAME}_prefix_readout.csv",
    "truth_late_oracle_metrics": (
        f"{EXPERIMENT_NAME}_truth_late_oracle_metrics.csv"
    ),
}

ARTIFACT_SORT_COLUMNS = {
    "fold_manifest": ("fold", "role", "well_id"),
    "role_read_ledger": ("fold", "role", "well_id"),
    "rgt_nodes": NODE_SORT_COLUMNS,
    "graph_edges": EDGE_SORT_COLUMNS,
    "target_paths": PATH_SORT_COLUMNS,
    "reference_gr_templates": REFERENCE_SORT_COLUMNS,
    "prefix_readout": ("fold", "well_id"),
    "truth_late_oracle_metrics": (
        "record_type",
        "scope",
        "fold",
        "well_id",
        "block_index",
    ),
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
        logical = frame_content_sha256(
            frame,
            sort_columns=ARTIFACT_SORT_COLUMNS[name],
        )
        schema = frame_schema_sha256(frame)
        file_sha = sha256_file(path)
        record = {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "schema_sha256": schema,
            "logical_content_sha256": logical,
            "file_sha256": file_sha,
        }
        records[name] = record
        sha_rows.append({"artifact": name, **record})
    return records, pd.DataFrame(sha_rows)


def select_preflight_wells(
    fold_by_well: Mapping[str, int],
    maximum_wells: int,
    folds: Sequence[int],
) -> set[str]:
    by_fold = {
        int(fold): sorted(
            well for well, assigned in fold_by_well.items() if int(assigned) == int(fold)
        )
        for fold in folds
    }
    selected: list[str] = []
    offset = 0
    while len(selected) < maximum_wells:
        progress = False
        for fold in folds:
            wells = by_fold[int(fold)]
            if offset < len(wells):
                selected.append(wells[offset])
                progress = True
                if len(selected) >= maximum_wells:
                    break
        if not progress:
            break
        offset += 1
    return set(selected)


def build_manifest(
    *,
    config: Mapping[str, Any],
    parent_path: Path,
    target_free_hashes: Mapping[str, str],
    stage0: Mapping[str, Any],
    stage1: Mapping[str, Any] | None,
    stage2: Mapping[str, Any] | None,
    cycle_manifests: Sequence[Mapping[str, Any]],
    records: Mapping[str, Any],
    ledger: RoleReadLedger,
    full_run: bool,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "status": (
            "stage2_pass"
            if stage2 and stage2.get("passed")
            else "stage2_fail_closed"
            if stage2 is not None
            else "stage1_fail_closed"
            if stage1 is not None and not stage1.get("passed")
            else "stage0_resource_preflight_pass"
            if not full_run and stage0.get("passed")
            else "stage0_fail_closed"
        ),
        "full_run": bool(full_run),
        "parent": {
            "experiment": PARENT_EXPERIMENT,
            "regenerated": False,
            "path": str(parent_path),
            "decompressed_content_sha256": sha256_decompressed_csv(parent_path),
        },
        "stage0": dict(stage0),
        "stage1": dict(stage1) if stage1 is not None else None,
        "stage2": dict(stage2) if stage2 is not None else None,
        "target_free_logical_sha256": dict(target_free_hashes),
        "cycle_manifests": list(cycle_manifests),
        "role_read_ledger": ledger.summary(),
        "solver_contract_sha256": sha256_bytes(
            stable_json_bytes(
                {
                    "rgt": get_nested(config, "rgt"),
                    "validation": get_nested(config, "validation"),
                    "gates": get_nested(config, "gates"),
                }
            )
        ),
        "artifacts": dict(records),
        "deterministic_anchor": False,
        "deterministic_anchor_reason": (
            "first_successful_run_requires_identical_graph_scenario_and_prediction_"
            "content_sha_on_rerun"
        ),
    }


def run_train() -> dict[str, Any]:
    config = load_config()
    validate_execution_contract(config, require_kaggle_authorization=True)
    train_dir = resolve_train_dir(config)
    files = sorted(train_dir.glob("*__horizontal_well.csv"))
    file_by_well = {well_id_from_path(path): path for path in files}
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(file_by_well) != expected_wells:
        raise ValueError(f"expected {expected_wells} raw train wells, found {len(file_by_well)}")
    parent_oof, parent_path = load_parent_oof(config)
    input_sha_by_well = {
        well_id: sha256_file(path)
        for well_id, path in sorted(file_by_well.items())
    }
    folds = list(get_nested(config, "validation.expected_folds"))
    fold_by_well = assign_group_folds(
        sorted(file_by_well),
        int(get_nested(config, "validation.n_folds")),
        int(get_nested(config, "validation.seed")),
    )
    validate_fold_identity(fold_by_well, parent_oof)
    full_run = str(get_nested(config, "execution.current_mode")) == "full_run"
    selected_target_wells: set[str] | None = None
    if not full_run:
        selected_target_wells = select_preflight_wells(
            fold_by_well,
            int(get_nested(config, "execution.preflight_max_wells")),
            folds,
        )
    maximum_wells_raw = os.environ.get("EXP386_MAX_WELLS")
    if maximum_wells_raw:
        selected_target_wells = select_preflight_wells(
            fold_by_well,
            int(maximum_wells_raw),
            folds,
        )
        full_run = False
    ledger = RoleReadLedger()
    started = time.perf_counter()
    fold_results = [
        run_fold_target_free(
            fold=int(fold),
            file_by_well=file_by_well,
            input_sha_by_well=input_sha_by_well,
            fold_by_well=fold_by_well,
            selected_target_wells=selected_target_wells,
            ledger=ledger,
            config=config,
        )
        for fold in folds
    ]
    frames: dict[str, pd.DataFrame] = {
        "fold_manifest": _concat_nonempty(
            [result["fold_manifest"] for result in fold_results],
            sort_columns=("fold", "role", "well_id"),
        ),
        "rgt_nodes": _concat_nonempty(
            [result["nodes"] for result in fold_results],
            sort_columns=NODE_SORT_COLUMNS,
        ),
        "graph_edges": _concat_nonempty(
            [result["edges"] for result in fold_results],
            sort_columns=EDGE_SORT_COLUMNS,
        ),
        "target_paths": _concat_nonempty(
            [result["paths"] for result in fold_results],
            sort_columns=PATH_SORT_COLUMNS,
        ),
        "reference_gr_templates": _concat_nonempty(
            [result["references"] for result in fold_results],
            sort_columns=REFERENCE_SORT_COLUMNS,
        ),
        "rgt_diagnostics": _concat_nonempty(
            [result["rgt_diagnostics"] for result in fold_results],
            sort_columns=("well_id",),
        ),
        "bank_diagnostics": _concat_nonempty(
            [result["bank_diagnostics"] for result in fold_results],
            sort_columns=("fold", "well_id"),
        ),
    }
    frames["role_read_ledger"] = ledger.as_frame()
    cycle_manifests = [result["cycle_manifest"] for result in fold_results]
    target_free_hashes = freeze_target_free_outputs(frames, cycle_manifests)
    ledger.mark_frozen(target_free_hashes)
    stage0 = evaluate_stage0(
        frames=frames,
        fold_results=fold_results,
        cycle_manifests=cycle_manifests,
        ledger=ledger,
        full_run=full_run,
        config=config,
    )
    prefix_readout = pd.DataFrame()
    truth_readout = pd.DataFrame()
    stage1: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None
    if full_run and stage0["passed"]:
        prefix_readout, stage1 = build_prefix_rolling_origin_readout(
            fold_results=fold_results,
            file_by_well=file_by_well,
            parent_oof=parent_oof,
            ledger=ledger,
            config=config,
        )
        if stage1["passed"]:
            truth_readout, stage2 = build_truth_late_h512_readout(
                paths=frames["target_paths"],
                fold_results=fold_results,
                file_by_well=file_by_well,
                parent_oof=parent_oof,
                bank_diagnostics=frames["bank_diagnostics"],
                ledger=ledger,
                config=config,
            )
    frames["role_read_ledger"] = ledger.as_frame()
    frames["prefix_readout"] = prefix_readout
    frames["truth_late_oracle_metrics"] = truth_readout
    output = artifacts_dir()
    write_json(output / f"{EXPERIMENT_NAME}_cycle_manifest.json", cycle_manifests)
    write_json(output / f"{EXPERIMENT_NAME}_stage0_guard.json", stage0)
    records, sha_manifest = persist_frames(frames, output)
    write_table(sha_manifest, output / f"{EXPERIMENT_NAME}_sha_manifest.csv")
    manifest = build_manifest(
        config=config,
        parent_path=parent_path,
        target_free_hashes=target_free_hashes,
        stage0=stage0,
        stage1=stage1,
        stage2=stage2,
        cycle_manifests=cycle_manifests,
        records=records,
        ledger=ledger,
        full_run=full_run,
    )
    write_json(output / f"{EXPERIMENT_NAME}_manifest.json", manifest)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": manifest["status"],
        "stage0": stage0,
        "stage1": stage1,
        "stage2": stage2,
        "runtime_seconds": float(time.perf_counter() - started),
        "full_run": bool(full_run),
        "runtime_counts": {
            "scientific_variants": 1,
            "graph_fold_solves_completed": 5,
            "target_well_path_solves_completed": int(
                frames["bank_diagnostics"]["well_id"].nunique()
            ),
            "lightgbm_boosters": 0,
            "hmm_runs": 0,
            "pf_runs": 0,
            "beam_runs": 0,
        },
        "target_free_logical_sha256": target_free_hashes,
        "deterministic_anchor": False,
    }
    write_json(output / f"{EXPERIMENT_NAME}_metrics.json", metrics)
    return metrics


# %% [markdown]
# ## 10. Configuration preview

# %%
CONFIG_PREVIEW = load_config()
validate_execution_contract(CONFIG_PREVIEW, require_kaggle_authorization=False)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG_PREVIEW, "experiment.route"))
print("Lineage:", get_nested(CONFIG_PREVIEW, "lineage.parent"))
print("Status:", get_nested(CONFIG_PREVIEW, "experiment.status"))
print(
    "Execution authorized:",
    get_nested(CONFIG_PREVIEW, "execution.kaggle_execution_authorized"),
)
print(
    "Planned counts:",
    {
        "scientific_variants": get_nested(
            CONFIG_PREVIEW, "runtime.scientific_variants"
        ),
        "graph_fold_solves": get_nested(
            CONFIG_PREVIEW, "runtime.graph_fold_solves"
        ),
        "target_well_path_solves": get_nested(
            CONFIG_PREVIEW, "runtime.target_well_path_solves"
        ),
        "models": get_nested(CONFIG_PREVIEW, "runtime.fitted_models"),
        "hmm": get_nested(CONFIG_PREVIEW, "runtime.hmm_runs"),
        "pf": get_nested(CONFIG_PREVIEW, "runtime.pf_runs"),
        "beam": get_nested(CONFIG_PREVIEW, "runtime.beam_runs"),
        "boosters": get_nested(CONFIG_PREVIEW, "runtime.lightgbm_boosters"),
        "parent_control_regeneration": get_nested(
            CONFIG_PREVIEW, "runtime.parent_control_regeneration"
        ),
    },
)

if os.environ.get(IMPORT_ONLY_ENV, "0") != "1":
    RESULT = run_train()
    print(json.dumps(RESULT, indent=2, default=str))
