# %% [markdown]
# # exp428 similar-well GR registration-map transfer readout — train
#
# This compact self-contained notebook implements the frozen zero-model Stage 0
# readout. It transfers only a donor's Type Well–Horizontal GR registration
# offset. It never transfers donor TVT paths, geometry, rates, or truth warps,
# and it never creates a TVT prediction, fitted model, inference output, or
# submission.

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Notebook-safe configuration, paths, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Input inventory, fold separation, and late-truth ledger
# 5. Type Well axis graph
# 6. Horizontal suffix GR preprocessing and constrained DTW
# 7. Registration-map estimation
# 8. Outer-fold donor selection and target-free freeze
# 9. Late query-reference construction and readouts
# 10. Technical/scientific gates and generated artifacts
# 11. Setup and configuration preview
# 12. Run the separately approved Kaggle CPU audit

# %%
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit
except ImportError:  # pragma: no cover - Kaggle/runtime contract includes numba.

    def njit(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs

        def decorate(function: Any) -> Any:
            return function

        return decorate


EXPERIMENT_NAME = "exp428_similar_well_gr_registration_map_transfer_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

PRIMARY = "selected_top1_global_shift"
ZERO = "zero_shift"
RANDOM = "stable_random_same_group"
GROUP_MEDIAN = "same_group_median_global_shift"
ORACLE = "top5_oracle_global_shift"
GLOBAL_CANDIDATES = (PRIMARY, ZERO, RANDOM, GROUP_MEDIAN, ORACLE)

TARGET_FREE_WELL_COLUMNS = (
    "well",
    "fold",
    "typewell_group_id",
    "axis_offset_query_ft",
    "supported",
    "eligible_donor_count",
    "selected_donor_count",
    "top1_donor_well",
    "random_donor_well",
    "top1_dtw_cost",
    "top1_donor_stretch_ft_per_suffix",
    "top1_donor_local_warp_mad_ft",
    PRIMARY,
    ZERO,
    RANDOM,
    GROUP_MEDIAN,
    "top5_donor_wells_json",
    "top5_global_shifts_ft_json",
)
TARGET_FREE_BLOCK_COLUMNS = (
    "well",
    "fold",
    "block_id",
    "block_start_row",
    "block_end_row_exclusive",
    "block_center_progress",
    "top1_local_shift_ft",
)


# %% [markdown]
# ## 2. Notebook-safe configuration, paths, and SHA helpers


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
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
    stream: Any = gzip.open(path, "rb") if decompressed else path.open("rb")
    with stream as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    return canonical_json_sha256(
        [
            {
                "name": str(column),
                "dtype": str(frame[column].dtype),
                "nullable": bool(frame[column].isna().any()),
            }
            for column in frame.columns
        ]
    )


def logical_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    buffer = io.StringIO()
    selected.to_csv(buffer, index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(buffer.getvalue().encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.12g",
                )


def config_candidates() -> list[Path]:
    return [
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / "config.yaml",
        PACKAGE_DIR / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]


def load_config() -> dict[str, Any]:
    for path in config_candidates():
        if path.exists():
            value = yaml.safe_load(path.read_text()) or {}
            if not isinstance(value, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                return value
    raise FileNotFoundError(f"config.yaml not found in {config_candidates()}")


def resolve_existing_file(
    filename: str,
    candidates: Iterable[str | Path],
) -> Path:
    checked: list[str] = []
    for raw in candidates:
        path = Path(raw)
        candidate = path / filename if path.is_dir() else path
        checked.append(str(candidate))
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    local_defaults = [
        PACKAGE_DIR / filename,
        PACKAGE_DIR / "artifacts" / filename,
        PACKAGE_DIR / "experiments" / EXPERIMENT_NAME / filename,
    ]
    for candidate in local_defaults:
        checked.append(str(candidate))
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    search_roots = [PACKAGE_DIR / "experiments", KAGGLE_INPUT_ROOT]
    for root in search_roots:
        if root.exists():
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"required input {filename} was not found; checked:\n"
        + "\n".join(checked[:80])
    )


def resolve_raw_train_dir(config: dict[str, Any]) -> Path:
    candidates = [
        Path(str(value))
        for value in get_nested(config, "data.raw_train_candidates", [])
    ]
    candidates.extend([PACKAGE_DIR / "data/raw/train", Path("data/raw/train")])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    raise FileNotFoundError("raw train directory containing well CSV files was not found")


def input_spec(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.inputs.{key}", {})
    if not isinstance(value, dict):
        raise ValueError(f"data.inputs.{key} must be a mapping")
    return value


def resolve_input(config: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    spec = input_spec(config, key)
    candidates = list(spec.get("candidates", []))
    if spec.get("local_path"):
        candidates.insert(0, spec["local_path"])
    path = resolve_existing_file(str(spec["filename"]), candidates)
    raw_sha = sha256_path(path)
    expected_raw = spec.get("expected_raw_sha256")
    if expected_raw and raw_sha != expected_raw:
        raise ValueError(f"raw SHA mismatch for {key}: {raw_sha}")
    decompressed_sha: str | None = None
    if path.suffix == ".gz":
        decompressed_sha = sha256_path(path, decompressed=True)
        expected_decompressed = spec.get("expected_decompressed_sha256")
        if expected_decompressed and decompressed_sha != expected_decompressed:
            raise ValueError(f"decompressed SHA mismatch for {key}: {decompressed_sha}")
    return path, {
        "name": key,
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
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
        "reporting_folds": int(get_nested(config, "validation.n_folds")),
        "lightgbm_configs": int(get_nested(config, "execution.lightgbm_configs")),
        "trained_folds": int(get_nested(config, "execution.trained_folds")),
        "boosters": int(get_nested(config, "execution.boosters")),
        "pf_well_runs": int(get_nested(config, "execution.pf_well_runs")),
        "hmm_well_runs": int(get_nested(config, "execution.hmm_well_runs")),
        "beam_well_runs": int(get_nested(config, "execution.beam_well_runs")),
        "gpu_runs": int(get_nested(config, "execution.gpu_runs")),
        "parent_control_reruns": int(
            get_nested(config, "execution.parent_control_reruns")
        ),
    }


def validate_scientific_contract(
    config: dict[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("unexpected experiment name")
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp428 route must remain pf_beam")
    if (
        get_nested(config, "lineage.parent")
        != "exp423_same_typewell_gr_dtw_truth_warp_transfer_readout"
    ):
        raise ValueError("exp428 parent contract changed")
    if not bool(get_nested(config, "implementation.design_frozen")):
        raise RuntimeError("the frozen design must remain enabled")
    if not bool(get_nested(config, "implementation.implementation_approved")):
        raise RuntimeError("exp428 implementation is not approved")
    if get_nested(config, "candidates.primary.name") != PRIMARY:
        raise ValueError("primary candidate changed")
    if list(get_nested(config, "registration_map.shift_grid_ft")) != [
        -80.0,
        -40.0,
        -20.0,
        -10.0,
        -5.0,
        -2.0,
        0.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ]:
        raise ValueError("registration shift grid changed")
    if int(get_nested(config, "registration_map.block_rows")) != 512:
        raise ValueError("registration block size changed")
    if (
        int(
            get_nested(
                config,
                "well_similarity.horizontal_gr_preprocessing.progress_grid_points",
            )
        )
        != 256
    ):
        raise ValueError("GR profile grid changed")
    if int(get_nested(config, "well_similarity.dtw.sakoe_chiba_band_points")) != 32:
        raise ValueError("DTW band changed")
    if bool(get_nested(config, "decision.rescue_grid_allowed")):
        raise ValueError("post-hoc rescue grid is forbidden")
    expected_counts = {
        "audit_variants": 1,
        "reporting_folds": 5,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "parent_control_reruns": 0,
    }
    observed_counts = execution_counts(config)
    if observed_counts != expected_counts:
        raise ValueError(
            f"execution count contract changed: {observed_counts} != {expected_counts}"
        )
    if require_run_approval and not bool(
        get_nested(config, "execution.kaggle_run_approved")
    ):
        raise PermissionError("Kaggle CPU audit run is not approved")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "primary": PRIMARY,
        "fold_identity": get_nested(config, "validation.fold_identity"),
        "query_truth_policy": get_nested(
            config, "validation.outer_valid_truth_policy"
        ),
        "axis_contract": get_nested(config, "typewell_axis_contract"),
        "registration_map": get_nested(config, "registration_map"),
        "well_similarity": get_nested(config, "well_similarity"),
        "success_gates": get_nested(config, "success_gates"),
        "execution_counts": observed_counts,
        "rescue_grid_allowed": False,
    }
    contract["scientific_contract_sha256"] = canonical_json_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Input inventory, fold separation, and late-truth ledger


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
            raise RuntimeError("donor truth must be read before target-free freeze")
        self.donor_truth_rows_by_fold[int(fold)] = (
            self.donor_truth_rows_by_fold.get(int(fold), 0) + int(rows)
        )

    def reject_query_truth_before_freeze(self, rows: int) -> None:
        self.query_truth_rows_before_freeze += int(rows)
        raise RuntimeError("query truth cannot be read before target-free freeze")

    def mark_frozen(self, content_sha256: str) -> None:
        if self.query_truth_rows_before_freeze != 0:
            raise RuntimeError("query truth was read before target-free freeze")
        if len(content_sha256) != 64:
            raise ValueError("frozen content SHA must be a SHA256 digest")
        self.target_free_frozen = True
        self.frozen_content_sha256 = content_sha256

    def require_frozen(self) -> None:
        if not self.target_free_frozen or not self.frozen_content_sha256:
            raise RuntimeError("query truth access requires a frozen target-free SHA")

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


def parse_row_idx(ids: pd.Series) -> np.ndarray:
    values = pd.to_numeric(
        ids.astype(str).str.extract(r"_(\d+)$", expand=False),
        errors="coerce",
    )
    if values.isna().any():
        raise ValueError("could not parse row index from inventory IDs")
    return values.to_numpy(np.int64)


def deterministic_well_folds(
    wells: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    ordered = np.asarray(sorted({str(well) for well in wells}), dtype=object)
    shuffled = ordered.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    rows = [
        {"well": str(well), "fold": int(fold)}
        for fold, part in enumerate(np.array_split(shuffled, n_folds))
        for well in part.tolist()
    ]
    result = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(
        drop=True
    )
    if len(result) != len(ordered) or not result["well"].is_unique:
        raise AssertionError("fold assignment must contain every well exactly once")
    return result


def load_target_free_inventory(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp099_inventory")
    safe_columns = list(
        get_nested(config, "data.inputs.exp099_inventory.prefreeze_safe_columns")
    )
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "target" not in header:
        raise ValueError("exp099 inventory must retain target for post-freeze evaluation")
    missing = sorted(set(safe_columns) - set(header))
    if missing:
        raise ValueError(f"safe inventory columns missing: {missing}")
    frame = pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = parse_row_idx(frame["id"])
    if frame["id"].duplicated().any() or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("safe inventory contains duplicate row identities")
    if frame["well"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("safe inventory well count differs from the frozen contract")
    manifest.update({"rows": len(frame), "wells": int(frame["well"].nunique())})
    return frame, path, manifest


def load_typewell_groups(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp065_cluster_assignments")
    frame = pd.read_csv(path, dtype=str)
    required = {
        "method",
        "threshold",
        "cluster_id",
        "well_id",
        "representative_well_id",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"group assignment columns missing: {sorted(required-set(frame.columns))}")
    selected = frame.loc[
        frame["method"].eq("native_overlap") & frame["threshold"].eq("1"),
        ["well_id", "cluster_id", "representative_well_id"],
    ].copy()
    selected = selected.rename(
        columns={
            "well_id": "well",
            "cluster_id": "typewell_group_id",
            "representative_well_id": "representative_well",
        }
    )
    if selected["well"].duplicated().any():
        raise ValueError("native_overlap=1 assignment has duplicate wells")
    manifest.update(
        {
            "selected_wells": int(selected["well"].nunique()),
            "selected_groups": int(selected["typewell_group_id"].nunique()),
        }
    )
    return selected, manifest


def build_inventory(
    safe: pd.DataFrame,
    groups: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, tuple[set[str], set[str]]]]:
    folds = deterministic_well_folds(
        safe["well"].unique().tolist(),
        n_folds=int(get_nested(config, "validation.n_folds")),
        seed=int(get_nested(config, "validation.seed")),
    )
    inventory = safe.merge(folds, on="well", how="left", validate="many_to_one")
    inventory = inventory.merge(groups, on="well", how="left", validate="many_to_one")
    inventory["fold"] = inventory["fold"].astype(np.int64)
    inventory = inventory.sort_values(
        ["fold", "well", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    all_wells = set(inventory["well"].astype(str))
    split_map: dict[int, tuple[set[str], set[str]]] = {}
    for fold in range(int(get_nested(config, "validation.n_folds"))):
        valid = set(inventory.loc[inventory["fold"].eq(fold), "well"].astype(str))
        train = all_wells - valid
        if train & valid:
            raise AssertionError("outer-train and outer-valid wells overlap")
        split_map[fold] = (train, valid)
    return inventory, split_map


def load_hidden_like_assignments(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path, manifest = resolve_input(config, "exp115_hidden_like_assignments")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {
        "well_id",
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"hidden-like assignment columns missing: {missing}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments must be one row per well")
    manifest.update({"rows": len(frame), "wells": int(frame["well_id"].nunique())})
    return frame, manifest


# %% [markdown]
# ## 5. Type Well axis graph


# %%
def build_typewell_axis_graph(
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tolerance = float(
        get_nested(config, "typewell_axis_contract.graph_cycle_tolerance_ft")
    )
    edge_tolerance = float(
        get_nested(config, "typewell_axis_contract.edge_consistency_tolerance_ft")
    )
    exact_min = float(
        get_nested(config, "typewell_axis_contract.pair_filter.exact_match_rate_min")
    )
    overlap_min = float(
        get_nested(
            config,
            "typewell_axis_contract.pair_filter.overlap_fraction_shorter_min",
        )
    )
    numeric_columns = [
        "exact_match_rate",
        "overlap_fraction_shorter",
        "tvt_delta_b_minus_a_median",
        "tvt_delta_b_minus_a_min",
        "tvt_delta_b_minus_a_max",
    ]
    work = pairs.copy()
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
    work = work.loc[
        work["exact_match_rate"].ge(exact_min)
        & work["overlap_fraction_shorter"].ge(overlap_min)
    ].copy()
    well_group = assignments.set_index("well")["typewell_group_id"].astype(str).to_dict()
    work["group_a"] = work["well_id_a"].astype(str).map(well_group)
    work["group_b"] = work["well_id_b"].astype(str).map(well_group)
    work = work.loc[work["group_a"].notna() & work["group_a"].eq(work["group_b"])]
    edge_span = (
        work["tvt_delta_b_minus_a_max"] - work["tvt_delta_b_minus_a_min"]
    ).abs()
    edge_conflicts = int(edge_span.gt(edge_tolerance).sum())
    edge_invalid_groups = set(
        work.loc[edge_span.gt(edge_tolerance), "group_a"].astype(str)
    )

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in work.itertuples():
        a = str(row.well_id_a)
        b = str(row.well_id_b)
        delta = float(row.tvt_delta_b_minus_a_median)
        adjacency[a].append((b, delta))
        adjacency[b].append((a, -delta))

    rows: list[dict[str, Any]] = []
    cycle_conflicts = 0
    disconnected_wells = 0
    invalid_groups: set[str] = set(edge_invalid_groups)
    for group, part in assignments.groupby("typewell_group_id", sort=True):
        wells = sorted(part["well"].astype(str))
        representative = str(part["representative_well"].iloc[0])
        if representative not in wells:
            raise ValueError(f"representative {representative} is outside group {group}")
        offsets: dict[str, float] = {representative: 0.0}
        queue: deque[str] = deque([representative])
        while queue:
            current = queue.popleft()
            for neighbour, delta in sorted(adjacency.get(current, [])):
                if neighbour not in wells:
                    continue
                proposed = offsets[current] + delta
                if neighbour not in offsets:
                    offsets[neighbour] = proposed
                    queue.append(neighbour)
                elif abs(offsets[neighbour] - proposed) > tolerance:
                    cycle_conflicts += 1
                    invalid_groups.add(str(group))
        disconnected = sorted(set(wells) - set(offsets))
        if len(wells) == 1:
            disconnected = []
        if disconnected:
            disconnected_wells += len(disconnected)
            invalid_groups.add(str(group))
        group_valid = str(group) not in invalid_groups
        for well in wells:
            rows.append(
                {
                    "well": well,
                    "typewell_group_id": str(group),
                    "representative_well": representative,
                    "axis_offset_ft": offsets.get(well, np.nan),
                    "axis_connected": well in offsets,
                    "axis_group_valid": group_valid,
                }
            )
    result = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(
        drop=True
    )
    if result["well"].duplicated().any():
        raise ValueError("axis graph contains duplicate wells")
    audit = {
        "selected_edges": len(work),
        "edge_consistency_conflicts": edge_conflicts,
        "cycle_conflicts": cycle_conflicts,
        "disconnected_wells": disconnected_wells,
        "invalid_groups": len(invalid_groups),
        "typewell_axis_graph_conflicts": (
            edge_conflicts + cycle_conflicts + disconnected_wells
        ),
        "axis_offsets_sha256": logical_frame_sha256(result),
    }
    return result, audit


def load_axis_graph_inputs(
    assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    path, manifest = resolve_input(config, "exp065_native_overlap_pairs")
    pairs = pd.read_csv(path, dtype={"well_id_a": str, "well_id_b": str})
    required = {
        "well_id_a",
        "well_id_b",
        "exact_match_rate",
        "overlap_fraction_shorter",
        "tvt_delta_b_minus_a_median",
        "tvt_delta_b_minus_a_min",
        "tvt_delta_b_minus_a_max",
    }
    if not required.issubset(pairs.columns):
        missing = sorted(required - set(pairs.columns))
        raise ValueError(f"native-overlap pair columns missing: {missing}")
    graph, audit = build_typewell_axis_graph(assignments, pairs, config)
    manifest.update(audit)
    return graph, audit, manifest


# %% [markdown]
# ## 6. Horizontal suffix GR preprocessing and constrained DTW


# %%
@dataclass(frozen=True)
class SuffixProfile:
    well: str
    row_idx: np.ndarray
    md: np.ndarray
    progress: np.ndarray
    gr_raw: np.ndarray
    gr_normalized: np.ndarray
    support_mask: np.ndarray
    finite_fraction: float
    robust_scale: float


def normalized_progress(md: np.ndarray) -> np.ndarray:
    values = np.asarray(md, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("suffix MD must be non-empty and finite")
    if np.any(np.diff(values) <= 0):
        raise ValueError("suffix MD must be strictly increasing")
    span = float(values[-1] - values[0])
    if span <= 0:
        raise ValueError("suffix MD span must be positive")
    return (values - values[0]) / span


def preprocess_suffix_gr(
    md: np.ndarray,
    gr: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    progress = normalized_progress(md)
    raw = np.asarray(gr, dtype=np.float64)
    finite = np.isfinite(raw)
    window = 5
    smoothed = (
        pd.Series(raw)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(np.float64)
    ).copy()
    smoothed[~finite] = np.nan
    points = int(
        get_nested(
            config,
            "well_similarity.horizontal_gr_preprocessing.progress_grid_points",
        )
    )
    grid = np.linspace(0.0, 1.0, points)
    resampled = np.full(points, np.nan)
    if np.isfinite(smoothed).sum() >= 2:
        valid = np.isfinite(smoothed)
        resampled = np.interp(
            grid, progress[valid], smoothed[valid], left=np.nan, right=np.nan
        )
    position = np.searchsorted(progress, grid, side="left")
    right = np.clip(position, 0, len(progress) - 1)
    left = np.clip(position - 1, 0, len(progress) - 1)
    nearest = np.where(
        np.abs(progress[right] - grid) < np.abs(grid - progress[left]), right, left
    )
    support = finite[nearest] & np.isfinite(resampled)
    values = resampled[support]
    center = float(np.median(values)) if len(values) else np.nan
    mad = float(np.median(np.abs(values - center))) if len(values) else np.nan
    scale = 1.4826 * mad if np.isfinite(mad) else np.nan
    normalized = np.full(points, np.nan)
    if np.isfinite(scale) and scale > 1.0e-9:
        finite_resampled = np.isfinite(resampled)
        # Keep the raw-observation support mask for the fixed support gates, but
        # let constrained DTW traverse deterministic interpolation across
        # internal GR gaps.  This matches the exp423 parent preprocessing.
        normalized[finite_resampled] = (
            resampled[finite_resampled] - center
        ) / scale
    return {
        "progress": progress,
        "normalized": normalized,
        "support": support,
        "finite_fraction": float(finite.mean()),
        "scale": scale,
    }


def load_safe_suffix_profile(
    well: str,
    rows: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> SuffixProfile:
    path = train_dir / f"{well}__horizontal_well.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if "TVT" not in header:
        raise ValueError(f"{path} must contain TVT for the post-freeze reader")
    horizontal = pd.read_csv(path, usecols=["MD", "GR", "TVT_input"])
    if "TVT" in horizontal.columns:
        raise AssertionError("safe suffix reader exposed query TVT")
    ordered = rows.sort_values("row_idx", kind="mergesort")
    row_idx = ordered["row_idx"].to_numpy(np.int64)
    if len(row_idx) == 0 or row_idx.min() < 0 or row_idx.max() >= len(horizontal):
        raise ValueError(f"invalid suffix row inventory for {well}")
    suffix = horizontal.iloc[row_idx]
    if suffix["TVT_input"].notna().any():
        raise ValueError(f"pseudo suffix contains known TVT_input for {well}")
    md = pd.to_numeric(suffix["MD"], errors="raise").to_numpy(np.float64)
    gr = pd.to_numeric(suffix["GR"], errors="coerce").to_numpy(np.float64)
    prepared = preprocess_suffix_gr(md, gr, config)
    return SuffixProfile(
        well=str(well),
        row_idx=row_idx,
        md=md,
        progress=prepared["progress"],
        gr_raw=gr,
        gr_normalized=prepared["normalized"],
        support_mask=prepared["support"],
        finite_fraction=float(prepared["finite_fraction"]),
        robust_scale=float(prepared["scale"]),
    )


def profile_is_valid(profile: SuffixProfile, config: dict[str, Any]) -> bool:
    minimum = float(
        get_nested(
            config,
            "well_similarity.horizontal_gr_preprocessing.minimum_finite_fraction",
        )
    )
    return bool(
        profile.finite_fraction >= minimum
        and profile.support_mask.mean() >= minimum
        and np.isfinite(profile.robust_scale)
        and profile.robust_scale > 1.0e-9
    )


@njit(cache=False)
def _constrained_dtw_numba(
    query: np.ndarray,
    donor: np.ndarray,
    band: int,
    max_run: int,
) -> tuple[float, np.ndarray, np.ndarray, int]:
    n = len(query)
    m = len(donor)
    states = 1 + 2 * max_run
    costs = np.full((n, m, states), np.inf)
    previous = np.full((n, m, states), -1, dtype=np.int8)
    if not np.isfinite(query[0]) or not np.isfinite(donor[0]):
        return np.inf, np.empty(0, np.int32), np.empty(0, np.int32), 0
    costs[0, 0, 0] = (query[0] - donor[0]) ** 2
    for i in range(n):
        for j in range(max(0, i - band), min(m - 1, i + band) + 1):
            if i == 0 and j == 0:
                continue
            if not np.isfinite(query[i]) or not np.isfinite(donor[j]):
                continue
            point = (query[i] - donor[j]) ** 2
            if i > 0 and j > 0:
                best_state = 0
                best = costs[i - 1, j - 1, 0]
                for state in range(1, states):
                    if costs[i - 1, j - 1, state] < best:
                        best = costs[i - 1, j - 1, state]
                        best_state = state
                if np.isfinite(best):
                    costs[i, j, 0] = best + point
                    previous[i, j, 0] = best_state
            if i > 0:
                best_state = 0
                best = costs[i - 1, j, 0]
                for state in range(max_run + 1, states):
                    if costs[i - 1, j, state] < best:
                        best = costs[i - 1, j, state]
                        best_state = state
                if np.isfinite(best):
                    costs[i, j, 1] = best + point
                    previous[i, j, 1] = best_state
                for run in range(2, max_run + 1):
                    if np.isfinite(costs[i - 1, j, run - 1]):
                        costs[i, j, run] = costs[i - 1, j, run - 1] + point
                        previous[i, j, run] = run - 1
            if j > 0:
                start = max_run + 1
                best_state = 0
                best = costs[i, j - 1, 0]
                for state in range(1, max_run + 1):
                    if costs[i, j - 1, state] < best:
                        best = costs[i, j - 1, state]
                        best_state = state
                if np.isfinite(best):
                    costs[i, j, start] = best + point
                    previous[i, j, start] = best_state
                for run in range(2, max_run + 1):
                    target = max_run + run
                    if np.isfinite(costs[i, j - 1, target - 1]):
                        costs[i, j, target] = costs[i, j - 1, target - 1] + point
                        previous[i, j, target] = target - 1
    state = 0
    final = costs[n - 1, m - 1, 0]
    for candidate in range(1, states):
        if costs[n - 1, m - 1, candidate] < final:
            final = costs[n - 1, m - 1, candidate]
            state = candidate
    if not np.isfinite(final):
        return np.inf, np.empty(0, np.int32), np.empty(0, np.int32), 0
    qi = np.empty(n + m, np.int32)
    di = np.empty(n + m, np.int32)
    length = 0
    i = n - 1
    j = m - 1
    while True:
        qi[length] = i
        di[length] = j
        length += 1
        if i == 0 and j == 0:
            break
        prior = int(previous[i, j, state])
        if prior < 0:
            return np.inf, np.empty(0, np.int32), np.empty(0, np.int32), 0
        if state == 0:
            i -= 1
            j -= 1
        elif state <= max_run:
            i -= 1
        else:
            j -= 1
        state = prior
    out_q = np.empty(length, np.int32)
    out_d = np.empty(length, np.int32)
    for index in range(length):
        out_q[index] = qi[length - 1 - index]
        out_d[index] = di[length - 1 - index]
    return final / length, out_q, out_d, length


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
        raise ValueError("DTW inputs must be equal-length vectors")
    cost, query_path, donor_path, length = _constrained_dtw_numba(
        left, right, band, max_run
    )
    if not np.isfinite(cost) or length == 0:
        raise ValueError("no finite constrained DTW path")
    return {
        "normalized_cost": float(cost),
        "query_path": query_path,
        "donor_path": donor_path,
        "path_length": int(length),
    }


def pair_dtw(
    query: SuffixProfile,
    donor: SuffixProfile,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    common = float(np.mean(query.support_mask & donor.support_mask))
    minimum = float(
        get_nested(
            config,
            "well_similarity.horizontal_gr_preprocessing.minimum_pair_common_fraction",
        )
    )
    if common < minimum:
        return None
    try:
        result = constrained_dtw(
            query.gr_normalized,
            donor.gr_normalized,
            band=int(get_nested(config, "well_similarity.dtw.sakoe_chiba_band_points")),
            max_run=int(
                get_nested(
                    config,
                    "well_similarity.dtw.max_consecutive_horizontal_or_vertical",
                )
            ),
        )
    except ValueError:
        return None
    result["common_support_fraction"] = common
    return result


def query_to_donor_progress(
    query_path: np.ndarray,
    donor_path: np.ndarray,
    query_progress: np.ndarray,
    *,
    n_points: int,
) -> np.ndarray:
    mapped = np.full(n_points, np.nan)
    for query_index in range(n_points):
        values = donor_path[query_path == query_index]
        if len(values):
            mapped[query_index] = float(np.median(values))
    finite = np.isfinite(mapped)
    if finite.sum() < 2:
        raise ValueError("DTW path does not define a progress mapping")
    grid_index = np.arange(n_points, dtype=float)
    mapped = np.interp(grid_index, grid_index[finite], mapped[finite])
    mapped = np.maximum.accumulate(mapped) / max(1, n_points - 1)
    return np.interp(
        np.asarray(query_progress, dtype=float),
        np.linspace(0.0, 1.0, n_points),
        np.clip(mapped, 0.0, 1.0),
    )


def stable_random_donor(query_well: str, eligible_donors: Sequence[str]) -> str:
    ordered = sorted({str(well) for well in eligible_donors})
    if not ordered:
        raise ValueError("stable random donor requires a non-empty donor set")
    digest = hashlib.sha256(str(query_well).encode()).digest()
    index = int.from_bytes(digest[:8], "big") % len(ordered)
    return ordered[index]


# %% [markdown]
# ## 7. Registration-map estimation


# %%
@dataclass(frozen=True)
class RegistrationMap:
    well: str
    block_scores: pd.DataFrame
    block_summary: pd.DataFrame
    global_shift_ft: float
    identifiable_blocks: int
    total_blocks: int
    progress: np.ndarray
    local_shift_ft: np.ndarray
    stretch_ft_per_suffix: float
    local_warp_mad_ft: float

    @property
    def supported(self) -> bool:
        return bool(np.isfinite(self.global_shift_ft) and self.identifiable_blocks >= 3)

    def interpolate(self, progress: np.ndarray) -> np.ndarray:
        values = np.asarray(progress, dtype=np.float64)
        if len(self.progress) == 0:
            return np.full(len(values), np.nan)
        return np.interp(
            values,
            self.progress,
            self.local_shift_ft,
            left=self.local_shift_ft[0],
            right=self.local_shift_ft[-1],
        )


def raw_finite_zncc(
    left: np.ndarray,
    right: np.ndarray,
    *,
    minimum_pairs: int,
) -> tuple[float, int]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    count = int(mask.sum())
    if count < minimum_pairs:
        return np.nan, count
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator <= 0.0:
        return np.nan, count
    return float(np.sum(x * y) / denominator), count


def normalized_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"TVT", "GR"}
    if not required.issubset(typewell.columns):
        raise ValueError("Type Well table must contain TVT and GR")
    work = typewell[["TVT", "GR"]].copy()
    work["TVT"] = pd.to_numeric(work["TVT"], errors="coerce")
    work["GR"] = pd.to_numeric(work["GR"], errors="coerce")
    work = work.dropna().groupby("TVT", sort=True, as_index=False)["GR"].mean()
    tvt = work["TVT"].to_numpy(np.float64)
    gr = work["GR"].to_numpy(np.float64)
    if len(tvt) < 2 or np.any(np.diff(tvt) <= 0):
        raise ValueError("Type Well TVT axis must contain two increasing finite points")
    return tvt, gr


def sample_typewell_gr(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tvt: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(tvt, dtype=np.float64),
        typewell_tvt,
        typewell_gr,
        left=np.nan,
        right=np.nan,
    )


def registration_blocks(profile: SuffixProfile, block_rows: int) -> pd.DataFrame:
    rows = []
    for block_id, start in enumerate(range(0, len(profile.row_idx), block_rows)):
        end = min(start + block_rows, len(profile.row_idx))
        rows.append(
            {
                "block_id": block_id,
                "block_start": start,
                "block_end": end,
                "block_start_row": int(profile.row_idx[start]),
                "block_end_row_exclusive": int(profile.row_idx[end - 1]) + 1,
                "block_center_progress": float(
                    np.median(profile.progress[start:end])
                ),
            }
        )
    return pd.DataFrame(rows)


def estimate_registration_map(
    well: str,
    profile: SuffixProfile,
    true_tvt: np.ndarray,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> RegistrationMap:
    truth = np.asarray(true_tvt, dtype=np.float64)
    if len(truth) != len(profile.row_idx) or not np.isfinite(truth).all():
        raise ValueError(f"registration truth is invalid for {well}")
    type_tvt, type_gr = normalized_typewell(typewell)
    shifts = [
        float(value) for value in get_nested(config, "registration_map.shift_grid_ft")
    ]
    minimum_pairs = int(
        get_nested(config, "registration_map.minimum_finite_pairs")
    )
    best_min = float(
        get_nested(config, "registration_map.identifiable_block.best_zncc_min")
    )
    margin_min = float(
        get_nested(
            config,
            "registration_map.identifiable_block.best_minus_second_zncc_min",
        )
    )
    blocks = registration_blocks(
        profile, int(get_nested(config, "registration_map.block_rows"))
    )
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for block in blocks.itertuples():
        start = int(block.block_start)
        end = int(block.block_end)
        candidates: list[tuple[float, float, int]] = []
        for shift in shifts:
            sampled = sample_typewell_gr(
                type_tvt, type_gr, truth[start:end] + shift
            )
            zncc, pairs = raw_finite_zncc(
                profile.gr_raw[start:end], sampled, minimum_pairs=minimum_pairs
            )
            score_rows.append(
                {
                    "well": str(well),
                    "block_id": int(block.block_id),
                    "block_start_row": int(block.block_start_row),
                    "block_end_row_exclusive": int(block.block_end_row_exclusive),
                    "block_center_progress": float(block.block_center_progress),
                    "shift_ft": shift,
                    "zncc": zncc,
                    "finite_pairs": pairs,
                }
            )
            if np.isfinite(zncc):
                candidates.append((float(zncc), shift, pairs))
        ranked = sorted(candidates, key=lambda item: (-item[0], abs(item[1]), item[1]))
        best_zncc = ranked[0][0] if ranked else np.nan
        best_shift = ranked[0][1] if ranked else np.nan
        second_zncc = ranked[1][0] if len(ranked) > 1 else np.nan
        margin = best_zncc - second_zncc if len(ranked) > 1 else np.nan
        identifiable = bool(
            np.isfinite(best_zncc)
            and np.isfinite(margin)
            and best_zncc >= best_min
            and margin >= margin_min
        )
        summary_rows.append(
            {
                "well": str(well),
                "block_id": int(block.block_id),
                "block_start_row": int(block.block_start_row),
                "block_end_row_exclusive": int(block.block_end_row_exclusive),
                "block_center_progress": float(block.block_center_progress),
                "best_shift_ft": best_shift,
                "best_zncc": best_zncc,
                "second_zncc": second_zncc,
                "zncc_margin": margin,
                "identifiable": identifiable,
            }
        )
    scores = pd.DataFrame(score_rows)
    summary = pd.DataFrame(summary_rows)
    identified = summary.loc[summary["identifiable"]].copy()
    minimum_blocks = int(
        get_nested(
            config, "registration_map.donor_map.minimum_identifiable_blocks"
        )
    )
    if len(identified) >= minimum_blocks:
        progress = identified["block_center_progress"].to_numpy(np.float64)
        local = identified["best_shift_ft"].to_numpy(np.float64)
        global_shift = float(np.median(local))
        centered_progress = progress - progress.mean()
        denominator = float(np.sum(centered_progress**2))
        slope = (
            float(np.sum(centered_progress * (local - local.mean())) / denominator)
            if denominator > 0
            else 0.0
        )
        intercept = float(local.mean() - slope * progress.mean())
        residual = local - (intercept + slope * progress)
        residual_center = float(np.median(residual))
        warp_mad = float(np.median(np.abs(residual - residual_center)))
    else:
        progress = np.empty(0, dtype=np.float64)
        local = np.empty(0, dtype=np.float64)
        global_shift = np.nan
        slope = np.nan
        warp_mad = np.nan
    return RegistrationMap(
        well=str(well),
        block_scores=scores,
        block_summary=summary,
        global_shift_ft=global_shift,
        identifiable_blocks=len(identified),
        total_blocks=len(summary),
        progress=progress,
        local_shift_ft=local,
        stretch_ft_per_suffix=slope,
        local_warp_mad_ft=warp_mad,
    )


def read_registration_inputs(
    well: str,
    profile: SuffixProfile,
    train_dir: Path,
    *,
    fold: int,
    outer_train: set[str],
    outer_valid: set[str],
    ledger: TruthAccessLedger,
    query_after_freeze: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    if query_after_freeze:
        ledger.require_frozen()
        if well not in outer_valid or well in outer_train:
            raise RuntimeError(f"query {well} is not strictly outer-valid for fold {fold}")
    else:
        if well not in outer_train or well in outer_valid:
            raise RuntimeError(f"donor {well} is not strictly outer-train for fold {fold}")
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=["TVT"])
    true_tvt = pd.to_numeric(
        horizontal.iloc[profile.row_idx]["TVT"], errors="raise"
    ).to_numpy(np.float64)
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    if query_after_freeze:
        ledger.record_query_truth_after_freeze(len(true_tvt))
    else:
        ledger.record_donor_truth(fold, len(true_tvt))
    return true_tvt, typewell


def score_global_registration_shift(
    profile: SuffixProfile,
    true_tvt: np.ndarray,
    typewell: pd.DataFrame,
    shift_ft: float,
    config: dict[str, Any],
) -> float:
    type_tvt, type_gr = normalized_typewell(typewell)
    sampled = sample_typewell_gr(type_tvt, type_gr, true_tvt + float(shift_ft))
    score, _ = raw_finite_zncc(
        profile.gr_raw,
        sampled,
        minimum_pairs=int(get_nested(config, "registration_map.minimum_finite_pairs")),
    )
    return score


# %% [markdown]
# ## 8. Outer-fold donor selection and target-free freeze


# %%
@dataclass
class CandidateBundle:
    wells: pd.DataFrame
    blocks: pd.DataFrame
    donor_maps: pd.DataFrame
    donor_block_scores: pd.DataFrame
    donor_rankings: pd.DataFrame
    fold_audit: pd.DataFrame
    top5_shifts: dict[str, list[float]]


def axis_lookup(axis_graph: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row.well): {
            "group": str(row.typewell_group_id),
            "offset": float(row.axis_offset_ft),
            "valid": bool(row.axis_group_valid) and bool(row.axis_connected),
        }
        for row in axis_graph.itertuples()
    }


def transfer_shift(
    donor_shift: float,
    *,
    query_axis_offset: float,
    donor_axis_offset: float,
) -> float:
    return float(donor_shift + query_axis_offset - donor_axis_offset)


def generate_target_free_candidates(
    inventory: pd.DataFrame,
    split_map: dict[int, tuple[set[str], set[str]]],
    profiles: dict[str, SuffixProfile],
    axis_graph: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> CandidateBundle:
    if "target" in inventory.columns or "TVT" in inventory.columns:
        ledger.reject_query_truth_before_freeze(len(inventory))
    axes = axis_lookup(axis_graph)
    group_wells = {
        str(group): sorted(part["well"].astype(str).unique().tolist())
        for group, part in inventory.dropna(subset=["typewell_group_id"]).groupby(
            "typewell_group_id", sort=True
        )
    }
    well_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    donor_map_rows: list[dict[str, Any]] = []
    donor_score_rows: list[pd.DataFrame] = []
    ranking_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    top5_shifts: dict[str, list[float]] = {}
    top_k = int(get_nested(config, "well_similarity.selected_donors"))

    for fold, (outer_train, outer_valid) in sorted(split_map.items()):
        if outer_train & outer_valid:
            raise AssertionError("donor/query intersection is non-zero")
        query_groups = {
            axes[well]["group"]
            for well in outer_valid
            if well in axes and axes[well]["valid"]
        }
        needed_donors = sorted(
            {
                well
                for group in query_groups
                for well in group_wells.get(group, [])
                if well in outer_train
                and well in axes
                and axes[well]["valid"]
                and profile_is_valid(profiles[well], config)
            }
        )
        maps: dict[str, RegistrationMap] = {}
        for donor in needed_donors:
            true_tvt, typewell = read_registration_inputs(
                donor,
                profiles[donor],
                train_dir,
                fold=fold,
                outer_train=outer_train,
                outer_valid=outer_valid,
                ledger=ledger,
                query_after_freeze=False,
            )
            registration = estimate_registration_map(
                donor, profiles[donor], true_tvt, typewell, config
            )
            maps[donor] = registration
            donor_map_rows.append(
                {
                    "fold": fold,
                    "donor_well": donor,
                    "typewell_group_id": axes[donor]["group"],
                    "axis_offset_ft": axes[donor]["offset"],
                    "global_shift_ft": registration.global_shift_ft,
                    "identifiable_blocks": registration.identifiable_blocks,
                    "total_blocks": registration.total_blocks,
                    "supported": registration.supported,
                    "stretch_ft_per_suffix": registration.stretch_ft_per_suffix,
                    "local_warp_mad_ft": registration.local_warp_mad_ft,
                }
            )
            scores = registration.block_scores.copy()
            scores.insert(0, "fold", fold)
            donor_score_rows.append(scores)

        fold_supported = 0
        fold_pairs = 0
        for query in sorted(outer_valid):
            profile = profiles[query]
            query_blocks = registration_blocks(
                profile, int(get_nested(config, "registration_map.block_rows"))
            )
            base = {
                "well": query,
                "fold": fold,
                "typewell_group_id": axes.get(query, {}).get("group", ""),
                "axis_offset_query_ft": axes.get(query, {}).get("offset", np.nan),
                "supported": False,
                "eligible_donor_count": 0,
                "selected_donor_count": 0,
                "top1_donor_well": "",
                "random_donor_well": "",
                "top1_dtw_cost": np.nan,
                "top1_donor_stretch_ft_per_suffix": np.nan,
                "top1_donor_local_warp_mad_ft": np.nan,
                PRIMARY: 0.0,
                ZERO: 0.0,
                RANDOM: 0.0,
                GROUP_MEDIAN: 0.0,
                "top5_donor_wells_json": "[]",
                "top5_global_shifts_ft_json": "[]",
            }
            local_prediction = np.zeros(len(query_blocks), dtype=np.float64)
            if (
                query in axes
                and axes[query]["valid"]
                and profile_is_valid(profile, config)
            ):
                group = axes[query]["group"]
                donor_pool = [
                    donor
                    for donor in group_wells.get(group, [])
                    if donor in outer_train
                    and donor in maps
                    and maps[donor].supported
                    and donor != query
                ]
                ranked: list[dict[str, Any]] = []
                for donor in sorted(donor_pool):
                    dtw = pair_dtw(profile, profiles[donor], config)
                    if dtw is not None:
                        ranked.append({"donor_well": donor, **dtw})
                ranked.sort(
                    key=lambda item: (
                        float(item["normalized_cost"]),
                        str(item["donor_well"]),
                    )
                )
                fold_pairs += len(ranked)
                if ranked:
                    selected = ranked[:top_k]
                    random_well = stable_random_donor(
                        query, [str(item["donor_well"]) for item in ranked]
                    )
                    top1 = str(selected[0]["donor_well"])
                    query_axis = float(axes[query]["offset"])

                    def converted(donor: str) -> float:
                        return transfer_shift(
                            maps[donor].global_shift_ft,
                            query_axis_offset=query_axis,
                            donor_axis_offset=float(axes[donor]["offset"]),
                        )

                    selected_shifts = [
                        converted(str(item["donor_well"])) for item in selected
                    ]
                    group_shift_values = [
                        converted(donor) for donor in donor_pool
                    ]
                    base.update(
                        {
                            "supported": True,
                            "eligible_donor_count": len(ranked),
                            "selected_donor_count": len(selected),
                            "top1_donor_well": top1,
                            "random_donor_well": random_well,
                            "top1_dtw_cost": float(selected[0]["normalized_cost"]),
                            "top1_donor_stretch_ft_per_suffix": maps[
                                top1
                            ].stretch_ft_per_suffix,
                            "top1_donor_local_warp_mad_ft": maps[
                                top1
                            ].local_warp_mad_ft,
                            PRIMARY: selected_shifts[0],
                            RANDOM: converted(random_well),
                            GROUP_MEDIAN: float(np.median(group_shift_values)),
                            "top5_donor_wells_json": json.dumps(
                                [str(item["donor_well"]) for item in selected],
                                separators=(",", ":"),
                            ),
                            "top5_global_shifts_ft_json": json.dumps(
                                selected_shifts, separators=(",", ":")
                            ),
                        }
                    )
                    top5_shifts[query] = selected_shifts
                    query_progress = query_blocks[
                        "block_center_progress"
                    ].to_numpy(np.float64)
                    donor_progress = query_to_donor_progress(
                        selected[0]["query_path"],
                        selected[0]["donor_path"],
                        query_progress,
                        n_points=len(profile.gr_normalized),
                    )
                    local_prediction = maps[top1].interpolate(donor_progress)
                    local_prediction += query_axis - float(axes[top1]["offset"])
                    fold_supported += 1
                    for rank, item in enumerate(selected, start=1):
                        ranking_rows.append(
                            {
                                "fold": fold,
                                "query_well": query,
                                "donor_rank": rank,
                                "donor_well": str(item["donor_well"]),
                                "normalized_cost": float(item["normalized_cost"]),
                                "common_support_fraction": float(
                                    item["common_support_fraction"]
                                ),
                                "path_length": int(item["path_length"]),
                                "query_path_json": json.dumps(
                                    item["query_path"].astype(int).tolist(),
                                    separators=(",", ":"),
                                ),
                                "donor_path_json": json.dumps(
                                    item["donor_path"].astype(int).tolist(),
                                    separators=(",", ":"),
                                ),
                            }
                        )
            top5_shifts.setdefault(query, [])
            well_rows.append(base)
            for block, local_shift in zip(
                query_blocks.itertuples(), local_prediction, strict=True
            ):
                block_rows.append(
                    {
                        "well": query,
                        "fold": fold,
                        "block_id": int(block.block_id),
                        "block_start_row": int(block.block_start_row),
                        "block_end_row_exclusive": int(block.block_end_row_exclusive),
                        "block_center_progress": float(block.block_center_progress),
                        "top1_local_shift_ft": float(local_shift),
                    }
                )
        fold_rows.append(
            {
                "fold": fold,
                "outer_train_wells": len(outer_train),
                "outer_valid_wells": len(outer_valid),
                "donor_query_intersection": len(outer_train & outer_valid),
                "donor_maps_built": len(maps),
                "supported_donor_maps": sum(value.supported for value in maps.values()),
                "eligible_dtw_pairs": fold_pairs,
                "supported_query_wells": fold_supported,
            }
        )
    wells = pd.DataFrame(well_rows).loc[:, list(TARGET_FREE_WELL_COLUMNS)]
    blocks = pd.DataFrame(block_rows).loc[:, list(TARGET_FREE_BLOCK_COLUMNS)]
    if wells["well"].duplicated().any() or blocks.duplicated(["well", "block_id"]).any():
        raise ValueError("target-free output has duplicate identities")
    if not np.isfinite(
        wells.loc[wells["supported"], [PRIMARY, ZERO, RANDOM, GROUP_MEDIAN]].to_numpy(
            np.float64
        )
    ).all():
        raise ValueError("supported global-shift candidates must be finite")
    return CandidateBundle(
        wells=wells,
        blocks=blocks,
        donor_maps=pd.DataFrame(donor_map_rows),
        donor_block_scores=(
            pd.concat(donor_score_rows, ignore_index=True)
            if donor_score_rows
            else pd.DataFrame()
        ),
        donor_rankings=pd.DataFrame(ranking_rows),
        fold_audit=pd.DataFrame(fold_rows),
        top5_shifts=top5_shifts,
    )


def freeze_target_free_candidates(
    bundle: CandidateBundle,
    axis_graph: pd.DataFrame,
    axis_audit: dict[str, Any],
    input_manifest: dict[str, Any],
    scientific_contract: dict[str, Any],
    config: dict[str, Any],
    artifacts_dir: Path,
    ledger: TruthAccessLedger,
) -> dict[str, Any]:
    component_sha = {
        "wells": logical_frame_sha256(bundle.wells, TARGET_FREE_WELL_COLUMNS),
        "blocks": logical_frame_sha256(bundle.blocks, TARGET_FREE_BLOCK_COLUMNS),
        "donor_maps": logical_frame_sha256(bundle.donor_maps),
        "donor_block_scores": logical_frame_sha256(bundle.donor_block_scores),
        "donor_rankings": logical_frame_sha256(bundle.donor_rankings),
        "axis_graph": logical_frame_sha256(axis_graph),
    }
    logical_sha = canonical_json_sha256(component_sha)
    ledger.mark_frozen(logical_sha)
    paths = {
        "target_free_wells": artifacts_dir / f"{OUTPUT_PREFIX}_target_free_wells.csv.gz",
        "target_free_blocks": artifacts_dir / f"{OUTPUT_PREFIX}_target_free_blocks.csv.gz",
        "donor_maps": artifacts_dir / f"{OUTPUT_PREFIX}_donor_registration_maps.csv.gz",
        "donor_block_scores": artifacts_dir
        / f"{OUTPUT_PREFIX}_donor_registration_block_scores.csv.gz",
        "donor_rankings": artifacts_dir / f"{OUTPUT_PREFIX}_donor_rankings.csv.gz",
        "fold_audit": artifacts_dir / f"{OUTPUT_PREFIX}_fold_separation_audit.csv",
        "axis_graph": artifacts_dir / f"{OUTPUT_PREFIX}_typewell_axis_graph.csv",
        "input_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.json",
    }
    for key, frame in (
        ("target_free_wells", bundle.wells),
        ("target_free_blocks", bundle.blocks),
        ("donor_maps", bundle.donor_maps),
        ("donor_block_scores", bundle.donor_block_scores),
        ("donor_rankings", bundle.donor_rankings),
    ):
        write_deterministic_gzip_csv(frame, paths[key])
    bundle.fold_audit.to_csv(paths["fold_audit"], index=False)
    axis_graph.to_csv(paths["axis_graph"], index=False)
    write_json(paths["input_manifest"], input_manifest)
    expected = get_nested(
        config, "reproducibility.expected_target_free_content_sha256"
    )
    matched = bool(expected and expected == logical_sha)
    artifact_manifest = {}
    for key, path in paths.items():
        artifact_manifest[key] = {
            "path": str(path),
            "raw_sha256": sha256_path(path),
            "decompressed_sha256": (
                sha256_path(path, decompressed=True) if path.suffix == ".gz" else None
            ),
        }
    freeze = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "logical_component_sha256": component_sha,
        "logical_content_sha256": logical_sha,
        "expected_rerun_content_sha256": expected,
        "deterministic_content_sha_match": matched,
        "determinism_status": (
            "matched_independent_reference"
            if matched
            else "pending_independent_rerun_reference"
        ),
        "well_schema_sha256": schema_sha256(bundle.wells),
        "block_schema_sha256": schema_sha256(bundle.blocks),
        "typewell_axis_audit": axis_audit,
        "truth_access_ledger_at_freeze": ledger.snapshot(),
        "artifacts": artifact_manifest,
    }
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_target_free_freeze_contract.json"
    write_json(freeze_path, freeze)
    freeze["artifacts"]["freeze_contract"] = {
        "path": str(freeze_path),
        "raw_sha256": sha256_path(freeze_path),
    }
    return freeze


# %% [markdown]
# ## 9. Late query-reference construction and readouts


# %%
def attach_query_references_after_freeze(
    bundle: CandidateBundle,
    profiles: dict[str, SuffixProfile],
    split_map: dict[int, tuple[set[str], set[str]]],
    train_dir: Path,
    hidden: pd.DataFrame,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger.require_frozen()
    result_rows: list[dict[str, Any]] = []
    reference_blocks: list[pd.DataFrame] = []
    hidden_lookup = hidden.set_index("well_id")
    for candidate in bundle.wells.itertuples(index=False):
        well = str(candidate.well)
        fold = int(candidate.fold)
        outer_train, outer_valid = split_map[fold]
        true_tvt, typewell = read_registration_inputs(
            well,
            profiles[well],
            train_dir,
            fold=fold,
            outer_train=outer_train,
            outer_valid=outer_valid,
            ledger=ledger,
            query_after_freeze=True,
        )
        reference = estimate_registration_map(
            well, profiles[well], true_tvt, typewell, config
        )
        selected = bundle.wells.loc[bundle.wells["well"].eq(well)].iloc[0]
        top5 = bundle.top5_shifts[well]
        oracle = (
            min(top5, key=lambda value: (abs(value - reference.global_shift_ft), value))
            if top5 and reference.supported
            else 0.0
        )
        supported = bool(selected["supported"] and reference.supported)
        primary_shift = float(selected[PRIMARY])
        zero_zncc = (
            score_global_registration_shift(
                profiles[well], true_tvt, typewell, 0.0, config
            )
            if reference.supported
            else np.nan
        )
        primary_zncc = (
            score_global_registration_shift(
                profiles[well], true_tvt, typewell, primary_shift, config
            )
            if supported
            else np.nan
        )
        hidden_row = hidden_lookup.loc[well] if well in hidden_lookup.index else None
        result_rows.append(
            {
                **selected.to_dict(),
                ORACLE: oracle,
                "query_reference_global_shift_ft": reference.global_shift_ft,
                "query_identifiable_blocks": reference.identifiable_blocks,
                "query_total_blocks": reference.total_blocks,
                "query_reference_supported": reference.supported,
                "evaluation_supported": supported,
                "query_stretch_ft_per_suffix": reference.stretch_ft_per_suffix,
                "query_local_warp_mad_ft": reference.local_warp_mad_ft,
                "stretch_transfer_abs_error_ft": abs(
                    float(selected["top1_donor_stretch_ft_per_suffix"])
                    - reference.stretch_ft_per_suffix
                ),
                "local_warp_mad_transfer_abs_error_ft": abs(
                    float(selected["top1_donor_local_warp_mad_ft"])
                    - reference.local_warp_mad_ft
                ),
                "zero_shift_zncc": zero_zncc,
                "primary_shift_zncc": primary_zncc,
                "zncc_gain_vs_zero": primary_zncc - zero_zncc,
                "verification_like_spatial_role": (
                    str(hidden_row["verification_like_spatial_role"])
                    if hidden_row is not None
                    else ""
                ),
                "verification_like_typewell_purged_role": (
                    str(hidden_row["verification_like_typewell_purged_role"])
                    if hidden_row is not None
                    else ""
                ),
            }
        )
        blocks = reference.block_summary.drop(columns=["well"]).copy()
        blocks.insert(0, "fold", fold)
        blocks.insert(0, "well", well)
        predicted = bundle.blocks.loc[
            bundle.blocks["well"].eq(well),
            ["block_id", "top1_local_shift_ft"],
        ]
        blocks = blocks.merge(predicted, on="block_id", how="left", validate="one_to_one")
        blocks["top1_global_shift_ft"] = primary_shift
        blocks["evaluation_supported"] = supported & blocks["identifiable"]
        blocks["verification_like_spatial_role"] = (
            str(hidden_row["verification_like_spatial_role"])
            if hidden_row is not None
            else ""
        )
        blocks["verification_like_typewell_purged_role"] = (
            str(hidden_row["verification_like_typewell_purged_role"])
            if hidden_row is not None
            else ""
        )
        reference_blocks.append(blocks)
    wells = pd.DataFrame(result_rows).sort_values(
        ["fold", "well"], kind="mergesort"
    ).reset_index(drop=True)
    blocks = pd.concat(reference_blocks, ignore_index=True)
    return wells, blocks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    left = pd.Series(x, dtype=float)
    right = pd.Series(y, dtype=float)
    mask = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return np.nan
    return float(left[mask].rank().corr(right[mask].rank()))


def global_metric(
    frame: pd.DataFrame,
    candidate: str,
    *,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[frame["evaluation_supported"]].copy()
    truth = selected["query_reference_global_shift_ft"].to_numpy(np.float64)
    prediction = selected[candidate].to_numpy(np.float64)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    error = prediction[mask] - truth[mask]
    return {
        "scope": scope,
        "candidate": candidate,
        "wells": int(mask.sum()),
        "global_shift_mae_ft": (
            float(np.mean(np.abs(error))) if mask.any() else np.nan
        ),
        "within_2ft": float(np.mean(np.abs(error) <= 2.0)) if mask.any() else np.nan,
        "within_5ft": float(np.mean(np.abs(error) <= 5.0)) if mask.any() else np.nan,
        "shift_sign_accuracy": (
            float(np.mean(np.sign(prediction[mask]) == np.sign(truth[mask])))
            if mask.any()
            else np.nan
        ),
    }


def build_global_metrics(wells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: dict[str, pd.Series] = {
        "overall": pd.Series(True, index=wells.index),
        "hidden_like_spatial": wells["verification_like_spatial_role"].eq("valid"),
        "hidden_like_typewell_purged": wells[
            "verification_like_typewell_purged_role"
        ].eq("valid"),
    }
    scope_rows = [
        global_metric(wells.loc[mask], candidate, scope=scope)
        for scope, mask in scopes.items()
        for candidate in GLOBAL_CANDIDATES
    ]
    fold_rows = [
        {
            **global_metric(part, candidate, scope=f"fold_{int(fold)}"),
            "fold": int(fold),
        }
        for fold, part in wells.groupby("fold", sort=True)
        for candidate in GLOBAL_CANDIDATES
    ]
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def build_by_well_metrics(wells: pd.DataFrame) -> pd.DataFrame:
    selected = wells.loc[wells["evaluation_supported"]].copy()
    truth = selected["query_reference_global_shift_ft"]
    for candidate in GLOBAL_CANDIDATES:
        selected[f"abs_error__{candidate}"] = (selected[candidate] - truth).abs()
    selected["primary_minus_zero_abs_error"] = (
        selected[f"abs_error__{PRIMARY}"] - selected[f"abs_error__{ZERO}"]
    )
    return selected


def build_spearman_readout(wells: pd.DataFrame) -> pd.DataFrame:
    selected = wells.loc[wells["evaluation_supported"]].copy()
    selected["primary_abs_error"] = (
        selected[PRIMARY] - selected["query_reference_global_shift_ft"]
    ).abs()
    rows = [
        {
            "scope": "pooled",
            "fold": np.nan,
            "wells": len(selected),
            "spearman_dtw_cost_vs_transfer_error": spearman_rank_correlation(
                selected["top1_dtw_cost"], selected["primary_abs_error"]
            ),
        }
    ]
    for fold, part in selected.groupby("fold", sort=True):
        rows.append(
            {
                "scope": f"fold_{int(fold)}",
                "fold": int(fold),
                "wells": len(part),
                "spearman_dtw_cost_vs_transfer_error": spearman_rank_correlation(
                    part["top1_dtw_cost"], part["primary_abs_error"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_local_metrics(blocks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = blocks.loc[blocks["evaluation_supported"]].copy()
    selected["local_abs_error_ft"] = (
        selected["top1_local_shift_ft"] - selected["best_shift_ft"]
    ).abs()
    selected["global_abs_error_ft"] = (
        selected["top1_global_shift_ft"] - selected["best_shift_ft"]
    ).abs()
    well_rows = []
    for well, part in selected.groupby("well", sort=True):
        well_rows.append(
            {
                "well": str(well),
                "fold": int(part["fold"].iloc[0]),
                "blocks": len(part),
                "local_block_mae_ft": float(part["local_abs_error_ft"].mean()),
                "global_block_mae_ft": float(part["global_abs_error_ft"].mean()),
                "verification_like_spatial_role": str(
                    part["verification_like_spatial_role"].iloc[0]
                ),
                "verification_like_typewell_purged_role": str(
                    part["verification_like_typewell_purged_role"].iloc[0]
                ),
            }
        )
    by_well = pd.DataFrame(well_rows)
    if by_well.empty:
        by_well = pd.DataFrame(
            columns=[
                "well",
                "fold",
                "blocks",
                "local_block_mae_ft",
                "global_block_mae_ft",
                "verification_like_spatial_role",
                "verification_like_typewell_purged_role",
            ]
        )
    scopes = {
        "overall": pd.Series(True, index=by_well.index),
        "hidden_like_spatial": by_well["verification_like_spatial_role"].eq(
            "valid"
        ),
        "hidden_like_typewell_purged": by_well[
            "verification_like_typewell_purged_role"
        ].eq("valid"),
    }
    rows = []
    for scope, mask in scopes.items():
        part = by_well.loc[mask]
        rows.append(
            {
                "scope": scope,
                "fold": np.nan,
                "wells": len(part),
                "local_block_mae_ft": part["local_block_mae_ft"].mean(),
                "global_block_mae_ft": part["global_block_mae_ft"].mean(),
            }
        )
    for fold in range(5):
        part = by_well.loc[by_well["fold"].eq(fold)]
        rows.append(
            {
                "scope": f"fold_{fold}",
                "fold": fold,
                "wells": len(part),
                "local_block_mae_ft": part["local_block_mae_ft"].mean(),
                "global_block_mae_ft": part["global_block_mae_ft"].mean(),
            }
        )
    return by_well, pd.DataFrame(rows)


def build_mapping_shape_metrics(wells: pd.DataFrame) -> pd.DataFrame:
    selected = wells.loc[wells["evaluation_supported"]].copy()
    fields = (
        (
            "global_shift",
            PRIMARY,
            "query_reference_global_shift_ft",
        ),
        (
            "stretch",
            "top1_donor_stretch_ft_per_suffix",
            "query_stretch_ft_per_suffix",
        ),
        (
            "local_warp_mad",
            "top1_donor_local_warp_mad_ft",
            "query_local_warp_mad_ft",
        ),
    )
    rows = []
    for scope, part in [
        ("pooled", selected),
        *[
            (f"fold_{fold}", fold_part)
            for fold, fold_part in selected.groupby("fold", sort=True)
        ],
    ]:
        fold_value = (
            int(scope.split("_")[-1]) if scope.startswith("fold_") else np.nan
        )
        for field, donor_column, query_column in fields:
            rows.append(
                {
                    "scope": scope,
                    "fold": fold_value,
                    "mapping_field": field,
                    "wells": len(part),
                    "spearman_donor_vs_query": spearman_rank_correlation(
                        part[donor_column], part[query_column]
                    ),
                    "mean_absolute_transfer_error_ft": float(
                        (part[donor_column] - part[query_column]).abs().mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 10. Technical/scientific gates and generated artifacts


# %%
def metric_lookup(
    metrics: pd.DataFrame,
    *,
    scope: str,
    candidate: str,
) -> float:
    values = metrics.loc[
        metrics["scope"].eq(scope) & metrics["candidate"].eq(candidate),
        "global_shift_mae_ft",
    ]
    if len(values) != 1:
        raise ValueError(f"metric lookup is not unique for {scope}/{candidate}")
    return float(values.iloc[0])


def nonworse_folds(
    fold_metrics: pd.DataFrame,
    candidate: str,
    reference: str,
) -> int:
    left = fold_metrics.loc[
        fold_metrics["candidate"].eq(candidate), ["fold", "global_shift_mae_ft"]
    ].rename(columns={"global_shift_mae_ft": "candidate_mae"})
    right = fold_metrics.loc[
        fold_metrics["candidate"].eq(reference), ["fold", "global_shift_mae_ft"]
    ].rename(columns={"global_shift_mae_ft": "reference_mae"})
    merged = left.merge(right, on="fold", validate="one_to_one")
    return int((merged["candidate_mae"] <= merged["reference_mae"]).sum())


def evaluate_technical_gate(
    bundle: CandidateBundle,
    wells: pd.DataFrame,
    blocks: pd.DataFrame,
    axis_audit: dict[str, Any],
    freeze: dict[str, Any],
    ledger: TruthAccessLedger,
    config: dict[str, Any],
) -> dict[str, Any]:
    thresholds = get_nested(config, "success_gates.technical")
    supported_fraction = float(bundle.wells["supported"].mean())
    identifiable_fraction = float(blocks["identifiable"].mean())
    supported_finite = float(
        np.isfinite(
            wells.loc[
                wells["evaluation_supported"],
                [PRIMARY, ZERO, RANDOM, GROUP_MEDIAN],
            ].to_numpy(np.float64)
        ).mean()
    )
    checks = {
        "folds_complete": bundle.fold_audit["fold"].nunique()
        == int(thresholds["folds_complete"]),
        "donor_query_intersection_zero": bundle.fold_audit[
            "donor_query_intersection"
        ].max()
        <= int(thresholds["donor_query_intersection_max"]),
        "query_truth_reads_before_freeze_zero": ledger.query_truth_rows_before_freeze
        <= int(thresholds["query_truth_reads_before_freeze_max"]),
        "typewell_axis_graph_conflicts_zero": axis_audit[
            "typewell_axis_graph_conflicts"
        ]
        <= int(thresholds["typewell_axis_graph_conflicts_max"]),
        "supported_query_well_fraction": supported_fraction
        >= float(thresholds["supported_query_well_fraction_min"]),
        "identifiable_query_block_fraction": identifiable_fraction
        >= float(thresholds["identifiable_query_block_fraction_min"]),
        "supported_prediction_finite_fraction": supported_finite
        >= float(thresholds["supported_prediction_finite_fraction_min"]),
        "deterministic_content_sha_match": bool(
            freeze["deterministic_content_sha_match"]
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "status": "pass" if all(checks.values()) else "fail_or_pending_rerun",
        "checks": checks,
        "supported_query_well_fraction": supported_fraction,
        "identifiable_query_block_fraction": identifiable_fraction,
        "supported_prediction_finite_fraction": supported_finite,
        "determinism_status": freeze["determinism_status"],
    }


def evaluate_scientific_gate(
    wells: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    spearman: pd.DataFrame,
    local_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = get_nested(config, "success_gates.scientific")
    zero_mae = metric_lookup(scope_metrics, scope="overall", candidate=ZERO)
    gains = {
        "oracle_vs_zero": zero_mae
        - metric_lookup(scope_metrics, scope="overall", candidate=ORACLE),
        "primary_vs_zero": zero_mae
        - metric_lookup(scope_metrics, scope="overall", candidate=PRIMARY),
        "primary_vs_random": metric_lookup(
            scope_metrics, scope="overall", candidate=RANDOM
        )
        - metric_lookup(scope_metrics, scope="overall", candidate=PRIMARY),
        "primary_vs_group_median": metric_lookup(
            scope_metrics, scope="overall", candidate=GROUP_MEDIAN
        )
        - metric_lookup(scope_metrics, scope="overall", candidate=PRIMARY),
    }
    folds = {
        "oracle_vs_zero": nonworse_folds(fold_metrics, ORACLE, ZERO),
        "primary_vs_zero": nonworse_folds(fold_metrics, PRIMARY, ZERO),
        "primary_vs_random": nonworse_folds(fold_metrics, PRIMARY, RANDOM),
        "primary_vs_group_median": nonworse_folds(
            fold_metrics, PRIMARY, GROUP_MEDIAN
        ),
    }
    pooled_spearman = float(
        spearman.loc[spearman["scope"].eq("pooled"), "spearman_dtw_cost_vs_transfer_error"].iloc[0]
    )
    positive_spearman_folds = int(
        (
            spearman.loc[
                spearman["fold"].notna(), "spearman_dtw_cost_vs_transfer_error"
            ]
            > 0.0
        ).sum()
    )
    mean_zncc_gain = float(
        wells.loc[wells["evaluation_supported"], "zncc_gain_vs_zero"].mean()
    )
    positive_zncc_folds = int(
        (
            wells.loc[wells["evaluation_supported"]]
            .groupby("fold")["zncc_gain_vs_zero"]
            .mean()
            > 0.0
        ).sum()
    )
    hidden_deltas = {
        scope: metric_lookup(scope_metrics, scope=scope, candidate=PRIMARY)
        - metric_lookup(scope_metrics, scope=scope, candidate=ZERO)
        for scope in ("hidden_like_spatial", "hidden_like_typewell_purged")
    }
    p90 = (
        float(np.quantile(by_well["primary_minus_zero_abs_error"], 0.90))
        if len(by_well)
        else np.nan
    )
    checks = {
        "oracle_gain": gains["oracle_vs_zero"]
        >= float(thresholds["oracle_vs_zero_global_shift_mae_gain_min_ft"]),
        "oracle_fold_consistency": folds["oracle_vs_zero"]
        >= int(thresholds["oracle_nonworse_folds_min"]),
        "primary_zero_gain": gains["primary_vs_zero"]
        >= float(thresholds["primary_vs_zero_global_shift_mae_gain_min_ft"]),
        "primary_zero_fold_consistency": folds["primary_vs_zero"]
        >= int(thresholds["primary_vs_zero_nonworse_folds_min"]),
        "primary_random_gain": gains["primary_vs_random"]
        >= float(thresholds["primary_vs_random_global_shift_mae_gain_min_ft"]),
        "primary_random_fold_consistency": folds["primary_vs_random"]
        >= int(thresholds["primary_vs_random_nonworse_folds_min"]),
        "primary_group_median_gain": gains["primary_vs_group_median"]
        >= float(
            thresholds["primary_vs_group_median_global_shift_mae_gain_min_ft"]
        ),
        "primary_group_median_fold_consistency": folds["primary_vs_group_median"]
        >= int(thresholds["primary_vs_group_median_nonworse_folds_min"]),
        "dtw_cost_error_spearman": pooled_spearman
        >= float(thresholds["gr_dtw_cost_vs_global_shift_error_spearman_min"]),
        "dtw_spearman_fold_consistency": positive_spearman_folds
        >= int(thresholds["positive_spearman_folds_min"]),
        "mean_zncc_gain": mean_zncc_gain
        >= float(thresholds["mean_query_zncc_gain_vs_zero_shift_min"]),
        "zncc_gain_fold_consistency": positive_zncc_folds
        >= int(thresholds["positive_zncc_gain_folds_min"]),
        "hidden_like_spatial_nonworse": hidden_deltas["hidden_like_spatial"]
        <= float(thresholds["hidden_like_shift_mae_delta_max_ft"]),
        "hidden_like_typewell_purged_nonworse": hidden_deltas[
            "hidden_like_typewell_purged"
        ]
        <= float(thresholds["hidden_like_shift_mae_delta_max_ft"]),
        "by_well_p90_safety": p90
        <= float(thresholds["by_well_shift_abs_error_delta_p90_max_ft"]),
    }
    global_gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "mae_gains_ft": gains,
        "nonworse_folds": folds,
        "pooled_dtw_cost_error_spearman": pooled_spearman,
        "positive_spearman_folds": positive_spearman_folds,
        "mean_zncc_gain_vs_zero": mean_zncc_gain,
        "positive_zncc_gain_folds": positive_zncc_folds,
        "hidden_like_primary_minus_zero_mae_ft": hidden_deltas,
        "by_well_primary_minus_zero_abs_error_p90_ft": p90,
    }
    local_thresholds = get_nested(config, "success_gates.local_shape_incremental")
    overall_local = local_metrics.loc[local_metrics["scope"].eq("overall")].iloc[0]
    local_gain = float(
        overall_local["global_block_mae_ft"] - overall_local["local_block_mae_ft"]
    )
    fold_local = local_metrics.loc[local_metrics["fold"].notna()]
    local_nonworse = int(
        (fold_local["local_block_mae_ft"] <= fold_local["global_block_mae_ft"]).sum()
    )
    local_hidden_deltas = {
        scope: float(
            local_metrics.loc[
                local_metrics["scope"].eq(scope), "local_block_mae_ft"
            ].iloc[0]
            - local_metrics.loc[
                local_metrics["scope"].eq(scope), "global_block_mae_ft"
            ].iloc[0]
        )
        for scope in ("hidden_like_spatial", "hidden_like_typewell_purged")
    }
    local_checks = {
        "global_gate_passed": bool(global_gate["passed"]),
        "local_gain": local_gain
        >= float(local_thresholds["local_vs_global_block_shift_mae_gain_min_ft"]),
        "fold_consistency": local_nonworse
        >= int(local_thresholds["local_vs_global_nonworse_folds_min"]),
        "hidden_like_spatial_nonworse": local_hidden_deltas[
            "hidden_like_spatial"
        ]
        <= float(local_thresholds["hidden_like_block_shift_mae_delta_max_ft"]),
        "hidden_like_typewell_purged_nonworse": local_hidden_deltas[
            "hidden_like_typewell_purged"
        ]
        <= float(local_thresholds["hidden_like_block_shift_mae_delta_max_ft"]),
    }
    local_gate = {
        "passed": bool(all(local_checks.values())),
        "evaluated_only_after_global_pass": True,
        "checks": local_checks,
        "local_vs_global_block_mae_gain_ft": local_gain,
        "local_nonworse_folds": local_nonworse,
        "hidden_like_local_minus_global_block_mae_ft": local_hidden_deltas,
    }
    return global_gate, local_gate


def decide_result(
    technical: dict[str, Any],
    scientific: dict[str, Any],
    local: dict[str, Any],
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
        return "invalid_or_insufficient_registration_support"
    if not scientific["checks"]["oracle_gain"] or not scientific["checks"][
        "oracle_fold_consistency"
    ]:
        return "close_cross_well_registration_map_transfer"
    if not scientific["passed"]:
        return "registration_map_headroom_but_gr_similarity_selection_failed"
    if not local["passed"]:
        return "support_global_shift_transfer_only"
    return "support_global_and_local_registration_shape_transfer"


def save_readout_artifacts(
    wells: pd.DataFrame,
    blocks: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    spearman: pd.DataFrame,
    local_by_well: pd.DataFrame,
    local_metrics: pd.DataFrame,
    mapping_shape_metrics: pd.DataFrame,
    technical: dict[str, Any],
    scientific: dict[str, Any],
    local_gate: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, Any]:
    frames = {
        "well_readout": wells,
        "query_reference_blocks": blocks,
        "scope_metrics": scope_metrics,
        "fold_metrics": fold_metrics,
        "by_well": by_well,
        "dtw_spearman": spearman,
        "local_by_well": local_by_well,
        "local_metrics": local_metrics,
        "mapping_shape_metrics": mapping_shape_metrics,
    }
    manifest: dict[str, Any] = {}
    for name, frame in frames.items():
        path = artifacts_dir / f"{OUTPUT_PREFIX}_{name}.csv.gz"
        write_deterministic_gzip_csv(frame, path)
        manifest[name] = {
            "path": str(path),
            "raw_sha256": sha256_path(path),
            "decompressed_sha256": sha256_path(path, decompressed=True),
        }
    for name, value in (
        ("technical_gate", technical),
        ("scientific_gate", scientific),
        ("local_shape_gate", local_gate),
    ):
        path = artifacts_dir / f"{OUTPUT_PREFIX}_{name}.json"
        write_json(path, value)
        manifest[name] = {"path": str(path), "raw_sha256": sha256_path(path)}
    return manifest


def build_raw_manifest(train_dir: Path, wells: Sequence[str]) -> dict[str, Any]:
    rows = []
    for well in sorted({str(value) for value in wells}):
        for kind in ("horizontal_well", "typewell"):
            path = train_dir / f"{well}__{kind}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "well": well,
                    "kind": kind,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "raw_sha256": sha256_path(path),
                }
            )
    return {
        "path": str(train_dir),
        "files": len(rows),
        "combined_manifest_sha256": canonical_json_sha256(rows),
        "safe_horizontal_reader_columns": ["MD", "GR", "TVT_input"],
        "query_truth_exposed_before_freeze": False,
        "sha_match": True,
    }


def run_audit(config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    contract = validate_scientific_contract(config, require_run_approval=True)
    if not is_kaggle_runtime():
        raise RuntimeError("exp428 audit must run in the approved Kaggle CPU notebook")
    artifacts_dir = KAGGLE_WORKING_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ledger = TruthAccessLedger()

    safe, _exp099_path, exp099_manifest = load_target_free_inventory(config)
    groups, groups_manifest = load_typewell_groups(config)
    inventory, split_map = build_inventory(safe, groups, config)
    axis_graph, axis_audit, pairs_manifest = load_axis_graph_inputs(groups, config)
    hidden, hidden_manifest = load_hidden_like_assignments(config)
    train_dir = resolve_raw_train_dir(config)
    profiles = {
        str(well): load_safe_suffix_profile(str(well), part, train_dir, config)
        for well, part in inventory.groupby("well", sort=True)
    }
    raw_manifest = build_raw_manifest(train_dir, profiles)
    input_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "exp099_inventory": exp099_manifest,
            "exp065_cluster_assignments": groups_manifest,
            "exp065_native_overlap_pairs": pairs_manifest,
            "exp115_hidden_like_assignments": hidden_manifest,
            "raw_train": raw_manifest,
        },
        "config_sha256": canonical_json_sha256(config),
        "row_fold_inventory_sha256": logical_frame_sha256(
            inventory,
            ["id", "well", "row_idx", "fold", "typewell_group_id"],
        ),
    }
    bundle = generate_target_free_candidates(
        inventory,
        split_map,
        profiles,
        axis_graph,
        train_dir,
        config,
        ledger,
    )
    freeze = freeze_target_free_candidates(
        bundle,
        axis_graph,
        axis_audit,
        input_manifest,
        contract,
        config,
        artifacts_dir,
        ledger,
    )
    wells, blocks = attach_query_references_after_freeze(
        bundle,
        profiles,
        split_map,
        train_dir,
        hidden,
        config,
        ledger,
    )
    scope_metrics, fold_metrics = build_global_metrics(wells)
    by_well = build_by_well_metrics(wells)
    spearman = build_spearman_readout(wells)
    local_by_well, local_metrics = build_local_metrics(blocks)
    mapping_shape_metrics = build_mapping_shape_metrics(wells)
    technical = evaluate_technical_gate(
        bundle, wells, blocks, axis_audit, freeze, ledger, config
    )
    scientific, local_gate = evaluate_scientific_gate(
        wells,
        scope_metrics,
        fold_metrics,
        by_well,
        spearman,
        local_metrics,
        config,
    )
    decision = decide_result(technical, scientific, local_gate)
    artifact_manifest = save_readout_artifacts(
        wells,
        blocks,
        scope_metrics,
        fold_metrics,
        by_well,
        spearman,
        local_by_well,
        local_metrics,
        mapping_shape_metrics,
        technical,
        scientific,
        local_gate,
        artifacts_dir,
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - started,
        "route": "pf_beam",
        "stage": "zero_model_train_side_registration_transfer_readout",
        "scientific_contract": contract,
        "execution_counts": execution_counts(config),
        "freeze": freeze,
        "truth_access_ledger_final": ledger.snapshot(),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "local_shape_gate": local_gate,
        "decision": decision,
        "artifacts": artifact_manifest,
        "tv_t_prediction_created": False,
        "submission_in_scope": False,
    }
    write_json(artifacts_dir / f"{OUTPUT_PREFIX}_summary.json", summary)
    write_json(
        KAGGLE_WORKING_ROOT / "metrics.json",
        {
            "experiment": EXPERIMENT_NAME,
            "status": "completed_train_side_readout",
            "route": "pf_beam",
            "decision": decision,
            "technical_gate_passed": technical["passed"],
            "scientific_gate_passed": scientific["passed"],
            "local_shape_gate_passed": local_gate["passed"],
            "primary_vs_zero_mae_gain_ft": scientific["mae_gains_ft"][
                "primary_vs_zero"
            ],
            "oracle_vs_zero_mae_gain_ft": scientific["mae_gains_ft"][
                "oracle_vs_zero"
            ],
            "public_lb": None,
            "private_lb": None,
            "submission_in_scope": False,
        },
    )
    return summary


# %% [markdown]
# ## 11. Setup and configuration preview


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
            "primary": PRIMARY,
            "execution_counts": execution_counts(CONFIG),
            "kaggle_run_approved": bool(
                get_nested(CONFIG, "execution.kaggle_run_approved")
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
# ## 12. Run the separately approved Kaggle CPU audit
#
# Implementation and canonical train-notebook adoption do not authorize a
# package, push, or run. The fail-closed `execution.kaggle_run_approved` flag
# remains false until a separate user request.


# %%
SUMMARY: dict[str, Any] | None = None
if os.environ.get("EXP428_IMPORT_ONLY") != "1" and is_kaggle_runtime():
    if bool(get_nested(CONFIG, "execution.kaggle_run_approved")):
        SUMMARY = run_audit(CONFIG)
        print(json.dumps(to_jsonable(SUMMARY), indent=2, sort_keys=True))
    else:
        print("exp428 implementation loaded; Kaggle CPU run remains unapproved.")
