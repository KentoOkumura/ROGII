# %% [markdown]
# # exp511 exp413 transductive K16 neighbor-rate postprocess — train candidate
#
# This compact self-contained candidate audits one frozen, predicted-only
# postprocess. Each exp413 outer-valid fold is treated as a pseudo-test batch.
# Other wells in that batch contribute only their exp413 prediction and raw
# X/Y/Z geometry. Truth, hidden-like roles, and error outcomes are attached
# only after the candidate parquet and its content hashes have been frozen.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment boundary
# 2. Notebook-safe paths, serialization, hashes, and resolvers
# 3. Frozen implementation and execution contract
# 4. Saved exp413 OOF and raw-geometry allowlist
# 5. K16 coefficient and segment-geometry helpers
# 6. Fold-local donor field and local-linear consensus
# 7. Truth-free prediction generation and freeze
# 8. Truth-late metrics, diagnostics, and all-AND gate
# 9. Stage A orchestration and generated artifacts
# 10. Setup and implementation-only stop

# %% [markdown]
# ## 1. Imports and immutable experiment boundary

# %%
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import yaml

EXPERIMENT_NAME = "exp511_exp413_transductive_k16_neighbor_rate_postprocess"
PARENT_EXPERIMENT = "exp413_scale5_likpf_full_replacement_on_exp335"
PREDICTION_COLUMN = "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
CONTROL_COLUMN = "raw_exp413_stage_d_oof"
PRIMARY_COLUMN = "transductive_k16_neighbor_rate_a005_cap025"
CORRECTION_COLUMN = "transductive_k16_neighbor_rate_correction"
TRUTH_FREE_OOF_COLUMNS = ("id", "well", "outer_fold", PREDICTION_COLUMN)
TRUTH_LATE_OOF_COLUMNS = (
    "id",
    "md_since",
    "last_known_tvt",
    "target",
    "actual_tvt",
)
PREDICTION_ALLOWLIST = ("well", "row_idx", "fold", "pred_tvt", "X", "Y", "Z")
PREDICTION_FREEZE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "fold",
    CONTROL_COLUMN,
    PRIMARY_COLUMN,
    CORRECTION_COLUMN,
)
FIELD_COLUMNS = (
    "fold",
    "donor_well",
    "donor_segment",
    "source_row",
    "x",
    "y",
    "projection",
    "predicted_k16_coefficient",
    "normalized_field_rate",
    "field_eligible",
)
SUPPORT_COLUMNS = (
    "fold",
    "well",
    "segment",
    "source_row",
    "query_x",
    "query_y",
    "query_projection",
    "own_coefficient",
    "neighbor_field_rate",
    "neighbor_coefficient",
    "delta_coefficient",
    "selected_segments",
    "unique_donor_wells",
    "nearest_donor_distance_ft",
    "median_selected_distance_ft",
    "effective_segments",
    "query_projection_pass",
    "finite_prediction_pass",
    "minimum_unique_wells_pass",
    "supported",
    "self_donor_segments",
)
FIXED_SCOPE_ORDER = (
    "md_0_250",
    "md_250_1000",
    "md_1000_plus",
    "hidden_like_spatial",
    "hidden_like_typewell_purged",
)
HIDDEN_ROLE_COLUMNS = {
    "hidden_like_spatial": "verification_like_spatial_role",
    "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
}
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


# %% [markdown]
# ## 2. Notebook-safe paths, serialization, hashes, and resolvers

# %%
def locate_experiment_dir(start: Path = PACKAGE_DIR) -> Path:
    candidates = (
        start,
        start / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    )
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate
    for candidate in (start, *start.parents):
        path = candidate / "experiments" / EXPERIMENT_NAME
        if (path / "config.yaml").is_file():
            return path
    raise FileNotFoundError(f"Could not locate {EXPERIMENT_NAME}")


EXPERIMENT_DIR = locate_experiment_dir()


def find_project_root(start: Path = EXPERIMENT_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


ROOT = find_project_root()


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def output_dir() -> Path:
    return (
        KAGGLE_WORKING_ROOT / "artifacts"
        if is_kaggle_runtime()
        else EXPERIMENT_DIR / "artifacts"
    )


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"YAML must contain a mapping: {path}")
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def logical_frame_sha256(frame: pd.DataFrame, chunk_rows: int = 100_000) -> str:
    normalized = frame.copy()
    logical_dtypes: list[str] = []
    for column in normalized.columns:
        if pd.api.types.is_object_dtype(normalized[column]) or isinstance(
            normalized[column].dtype, pd.StringDtype
        ):
            normalized[column] = normalized[column].astype(str)
            logical_dtypes.append("string")
        else:
            logical_dtypes.append(str(normalized[column].dtype))
    digest = hashlib.sha256()
    digest.update("\x1f".join(normalized.columns.astype(str)).encode("utf-8"))
    digest.update("\x1f".join(logical_dtypes).encode("utf-8"))
    for start in range(0, len(normalized), int(chunk_rows)):
        chunk = normalized.iloc[start : start + int(chunk_rows)]
        hashes = pd.util.hash_pandas_object(chunk, index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def array_sha256(values: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest = hashlib.sha256()
    digest.update(sha256_json({"shape": list(array.shape), "dtype": dtype}).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def search_roots() -> list[Path]:
    return [KAGGLE_INPUT_ROOT, KAGGLE_WORKING_ROOT, Path("/tmp"), ROOT, Path.cwd()]


def expand_pattern(pattern: str) -> list[Path]:
    raw = Path(pattern)
    if raw.is_absolute():
        return [Path(value) for value in glob.glob(pattern, recursive=True)]
    paths: list[Path] = []
    for root in search_roots():
        paths.extend(Path(value) for value in glob.glob(str(root / pattern), recursive=True))
    return paths


def resolve_sha_qualified_file(
    patterns: Sequence[str],
    expected_sha256: str,
    *,
    explicit_env: str | None = None,
) -> Path:
    ordered = (
        ([str(os.environ[explicit_env])] if explicit_env and os.environ.get(explicit_env) else [])
        + [str(value) for value in patterns]
    )
    mismatches: dict[str, str] = {}
    seen: set[str] = set()
    for pattern in ordered:
        for candidate in expand_pattern(pattern):
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            observed = sha256_file(candidate)
            if observed == str(expected_sha256):
                return candidate
            mismatches[key] = observed
    raise FileNotFoundError(
        f"No SHA-qualified file found for {ordered}; mismatches={mismatches}"
    )


def resolve_parent_evidence_root(config: Mapping[str, Any]) -> Path:
    spec = config["data"]["exp413_oof"]
    files = dict(spec["evidence_files"])
    expected = {
        "oof": str(spec["expected_oof_sha256"]),
        "fold_metrics": str(spec["expected_fold_metrics_sha256"]),
        "scope_metrics": str(spec["expected_scope_metrics_sha256"]),
        "hidden_like_metrics": str(spec["expected_hidden_like_metrics_sha256"]),
        "by_well": str(spec["expected_by_well_sha256"]),
    }
    mismatches: list[dict[str, Any]] = []
    for pattern in spec["root_patterns"]:
        for candidate in expand_pattern(str(pattern)):
            if not candidate.is_dir():
                continue
            paths = {label: candidate / str(filename) for label, filename in files.items()}
            if not all(path.is_file() for path in paths.values()):
                continue
            observed = {label: sha256_file(path) for label, path in paths.items()}
            if observed == expected:
                return candidate
            mismatches.append({"root": str(candidate), "sha256": observed})
    raise FileNotFoundError(f"No complete SHA-qualified exp413 Stage D root: {mismatches}")


def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, dict[str, Path]]:
    spec = config["data"]["raw_geometry"]
    patterns: list[str] = []
    if os.environ.get("EXP511_RAW_TRAIN_DIR"):
        patterns.append(str(os.environ["EXP511_RAW_TRAIN_DIR"]))
    patterns.extend(str(value) for value in spec["root_patterns"])
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    def inspect_candidate(candidate: Path) -> dict[str, Path] | None:
        key = str(candidate.resolve())
        if key in seen or not candidate.is_dir():
            return None
        seen.add(key)
        files = sorted(candidate.glob(str(spec["horizontal_glob"])))
        by_well = {
            path.name.split("__horizontal_well.csv", 1)[0]: path for path in files
        }
        evidence.append(
            {
                "directory": str(candidate),
                "files": len(files),
                "wells": len(by_well),
            }
        )
        return by_well if set(by_well) == expected_wells else None

    for pattern in patterns:
        for candidate in expand_pattern(pattern):
            by_well = inspect_candidate(candidate)
            if by_well is not None:
                return candidate, by_well

    # Kaggle competition sources may be mounted below /kaggle/input/competitions.
    # The fallback remains fail-closed: only a directory whose filename-derived
    # well set exactly equals the frozen OOF inventory is accepted.
    if KAGGLE_INPUT_ROOT.is_dir():
        for candidate in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            by_well = inspect_candidate(candidate)
            if by_well is not None:
                return candidate, by_well
    raise FileNotFoundError(f"No raw geometry root matched all OOF wells: {evidence}")


# %% [markdown]
# ## 3. Frozen implementation and execution contract

# %%
@dataclass(frozen=True)
class K16Contract:
    segments: int = 16
    coefficient_smoothing_rho: float = 10.0
    theta0_deg: float = 118.4
    minimum_abs_projection: float = 0.3
    local_linear_k: int = 50
    bandwidth_ft: float = 500.0
    ridge: float = 1.0
    minimum_unique_donor_wells: int = 8
    alpha: float = 0.05
    correction_cap_ft: float = 0.25


@dataclass(frozen=True)
class WellK16:
    fold: int
    well: str
    positions: np.ndarray
    row_idx: np.ndarray
    prediction: np.ndarray
    phi: np.ndarray
    coefficients: np.ndarray
    segment_id: np.ndarray
    segment_mid_xy: np.ndarray
    segment_projection: np.ndarray
    segment_source_row: np.ndarray


@dataclass(frozen=True)
class LocalLinearResult:
    values: np.ndarray
    selected_segments: np.ndarray
    unique_donor_wells: np.ndarray
    nearest_distance: np.ndarray
    median_distance: np.ndarray
    effective_segments: np.ndarray
    finite: np.ndarray
    self_donor_segments: np.ndarray


def contract_from_config(config: Mapping[str, Any]) -> K16Contract:
    primary = config["postprocess"]["primary"]
    return K16Contract(
        segments=int(primary["k_segments"]),
        coefficient_smoothing_rho=float(primary["coefficient_smoothing_rho"]),
        theta0_deg=float(primary["theta0_deg"]),
        minimum_abs_projection=float(primary["field_min_abs_projection"]),
        local_linear_k=int(primary["local_linear_k_segments"]),
        bandwidth_ft=float(primary["local_linear_bandwidth_ft"]),
        ridge=float(primary["local_linear_ridge"]),
        minimum_unique_donor_wells=int(primary["min_unique_donor_wells"]),
        alpha=float(primary["alpha"]),
        correction_cap_ft=float(primary["final_correction_cap_ft"]),
    )


def validate_static_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    if config["experiment"]["name"] != EXPERIMENT_NAME:
        raise ValueError("experiment name changed")
    if config["experiment"]["route"] != "ensemble":
        raise ValueError("exp511 route must remain ensemble")
    if config["lineage"]["parent"] != PARENT_EXPERIMENT:
        raise ValueError("exp511 parent changed")
    if not bool(config["authorization"]["implementation_approved"]):
        raise ValueError("implementation approval is not recorded")
    if not all(
        bool(config["authorization"][key])
        for key in (
            "canonical_notebook_adoption_approved",
            "kaggle_package_approved",
            "kaggle_run_approved",
        )
    ):
        raise ValueError("canonical Notebook, package, and Stage A run approvals are required")
    if any(
        bool(config["authorization"][key])
        for key in (
            "inference_implementation_approved",
            "inference_run_approved",
            "submission_approved",
        )
    ):
        raise ValueError("train-only authorization boundary was exceeded")
    implementation = config["implementation"]
    required_implementation = {
        "code_created": True,
        "jupytext_source_created": True,
        "compact_selfcontained_candidate_created": True,
        "dedicated_contract_tests_created": True,
        "canonical_train_notebook_is_template_placeholder": False,
        "canonical_inference_notebook_is_template_placeholder": True,
        "kaggle_package_created": True,
        "inference_implemented": False,
    }
    observed_implementation = {
        key: bool(implementation[key]) for key in required_implementation
    }
    if observed_implementation != required_implementation:
        raise ValueError(f"implementation boundary changed: {observed_implementation}")

    primary = config["postprocess"]["primary"]
    expected_primary = {
        "id": PRIMARY_COLUMN,
        "selectable": True,
        "input_surface": "exp413_final_tvt_oof",
        "donor_surface": "same_outer_valid_fold_other_wells_predicted_tvt_plus_z",
        "target_surface": "target_well_predicted_tvt_plus_z",
        "score_anchor": "first_score_row_prediction",
        "k_segments": 16,
        "coefficient_smoothing_rho": 10.0,
        "theta0_deg": 118.4,
        "field_min_abs_projection": 0.3,
        "local_linear_k_segments": 50,
        "local_linear_bandwidth_ft": 500.0,
        "local_linear_ridge": 1.0,
        "min_unique_donor_wells": 8,
        "exclude_self_well": True,
        "alpha": 0.05,
        "final_correction_cap_ft": 0.25,
        "first_score_row_correction_ft": 0.0,
        "explicit_fade": False,
        "reanchor_after_correction": False,
        "clip_base_prediction": False,
        "u_projection_after_correction": False,
        "dtype": "float64",
        "stable_tie_break": ["distance", "donor_well", "donor_segment", "source_row"],
    }
    observed_primary = {key: primary[key] for key in expected_primary}
    if observed_primary != expected_primary:
        raise ValueError(f"fixed primary changed: {observed_primary}")
    if config["postprocess"]["report_only_candidates"] != []:
        raise ValueError("report-only candidates are prohibited")

    cost = config["execution_contract"]
    expected_cost = {
        "scientific_primary_variants": 1,
        "report_only_variants": 0,
        "trained_models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "total_boosters": 0,
        "hmm_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "parent_or_control_retraining": 0,
        "gpu_runs": 0,
    }
    observed_cost = {key: int(cost[key]) for key in expected_cost}
    if observed_cost != expected_cost:
        raise ValueError(f"execution inventory changed: {observed_cost}")

    gate = config["promotion_gate"]
    expected_gate = {
        "all_and": True,
        "technical_all_pass": True,
        "pooled_gain_min_ft": 0.01,
        "nonworse_folds_required": 4,
        "fixed_scope_max_delta_ft": 0.02,
        "by_well_p95_max_delta_ft": 0.25,
        "by_well_worst_max_delta_ft": 0.25,
        "first_score_row_abs_correction_max_ft": 1.0e-12,
        "all_row_abs_correction_max_ft": 0.250000001,
    }
    if {key: gate[key] for key in expected_gate} != expected_gate:
        raise ValueError("promotion gate changed")
    if tuple(gate["fixed_scopes"]) != FIXED_SCOPE_ORDER:
        raise ValueError("fixed scope order changed")
    if (
        gate["fail_decision"]
        != "FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_FADE_SCOPE_OR_GATE_RESCUE"
    ):
        raise ValueError("fail-close decision changed")
    if config["execution"]["stage_a"] != {
        "implementation_approved": True,
        "run_approved": True,
        "enabled": True,
        "completed": True,
        "kernel_version": 4,
        "decision": (
            "FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_"
            "SUPPORT_FADE_SCOPE_OR_GATE_RESCUE"
        ),
    }:
        raise ValueError("Stage A completion state changed")
    return {
        "status": "pass",
        "primary": observed_primary,
        "cost": observed_cost,
        "gate": expected_gate,
        "implementation": observed_implementation,
    }


def require_stage_a_authorization(config: Mapping[str, Any]) -> None:
    if not is_kaggle_runtime():
        raise RuntimeError("exp511 Stage A must run on Kaggle private CPU")
    if not bool(config["authorization"]["kaggle_run_approved"]):
        raise RuntimeError("exp511 Kaggle Stage A run is not approved")
    execution = config["execution"]["stage_a"]
    if not bool(execution["run_approved"]) or not bool(execution["enabled"]):
        raise RuntimeError("exp511 Stage A remains disabled")
    if bool(execution["completed"]):
        raise RuntimeError("exp511 Stage A is already complete and closed")


# %% [markdown]
# ## 4. Saved exp413 OOF and raw-geometry allowlist

# %%
def parse_row_idx(ids: pd.Series, wells: pd.Series) -> np.ndarray:
    split = ids.astype(str).str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2 or not split[0].astype(str).eq(wells.astype(str)).all():
        raise ValueError("exp413 id/well logical-key contract changed")
    return pd.to_numeric(split[1], errors="raise").to_numpy(np.int64)


def verify_parent_evidence(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    spec = config["data"]["exp413_oof"]
    expected = {
        "oof": str(spec["expected_oof_sha256"]),
        "fold_metrics": str(spec["expected_fold_metrics_sha256"]),
        "scope_metrics": str(spec["expected_scope_metrics_sha256"]),
        "hidden_like_metrics": str(spec["expected_hidden_like_metrics_sha256"]),
        "by_well": str(spec["expected_by_well_sha256"]),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for label, filename in spec["evidence_files"].items():
        path = root / str(filename)
        observed = sha256_file(path)
        if observed != expected[label]:
            raise ValueError(f"exp413 {label} SHA mismatch")
        evidence[label] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return evidence


def load_truth_free_parent_oof(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_wells: int,
    expected_folds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    observed_sha = sha256_file(path)
    if observed_sha != str(expected_sha256):
        raise ValueError("exp413 Stage D OOF SHA mismatch")
    available_columns = tuple(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(TRUTH_FREE_OOF_COLUMNS) - set(available_columns))
    if missing:
        raise ValueError(f"exp413 truth-free OOF columns missing: {missing}")
    source = pd.read_parquet(path, columns=list(TRUTH_FREE_OOF_COLUMNS))
    if tuple(source.columns) != TRUTH_FREE_OOF_COLUMNS:
        raise ValueError("truth-free OOF loader column order changed")
    frame = pd.DataFrame(
        {
            "id": source["id"].astype(str),
            "well": source["well"].astype(str),
            "row_idx": parse_row_idx(source["id"], source["well"]),
            "fold": pd.to_numeric(source["outer_fold"], errors="raise").to_numpy(np.int8),
            "pred_tvt": pd.to_numeric(
                source[PREDICTION_COLUMN], errors="raise"
            ).to_numpy(np.float64),
        }
    )
    if len(frame) != int(expected_rows):
        raise ValueError(f"exp413 row count mismatch: {len(frame)} != {expected_rows}")
    if int(frame["well"].nunique()) != int(expected_wells):
        raise ValueError("exp413 well count mismatch")
    if frame["id"].duplicated().any() or frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp413 logical keys are duplicated")
    if sorted(int(value) for value in frame["fold"].unique()) != list(expected_folds):
        raise ValueError("exp413 fold inventory changed")
    if not frame.groupby("well", sort=False)["fold"].nunique().eq(1).all():
        raise ValueError("an exp413 well spans multiple folds")
    if not np.isfinite(frame["pred_tvt"].to_numpy(np.float64)).all():
        raise ValueError("exp413 prediction contains nonfinite values")
    for well, positions in frame.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        rows = frame.iloc[pos]["row_idx"].to_numpy(np.int64)
        if len(rows) < 17 or not np.all(np.diff(rows) == 1):
            raise ValueError(f"{well} score rows are not one contiguous K16 path")
    fold_frame = (
        frame[["well", "fold"]]
        .drop_duplicates()
        .sort_values("well", kind="stable")
        .reset_index(drop=True)
    )
    manifest = {
        "source_path": str(path),
        "source_file_sha256": observed_sha,
        "source_schema": list(available_columns),
        "loaded_columns": list(TRUTH_FREE_OOF_COLUMNS),
        "truth_or_error_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "logical_key_sha256": logical_frame_sha256(frame[["id", "well", "row_idx"]]),
        "global_row_order_sha256": logical_frame_sha256(frame[["id"]]),
        "per_well_row_order_sha256": logical_frame_sha256(frame[["well", "row_idx"]]),
        "fold_assignment_logical_sha256": logical_frame_sha256(fold_frame),
        "prediction_sha256": array_sha256(frame["pred_tvt"].to_numpy(np.float64)),
    }
    return frame, manifest


def attach_raw_geometry_allowlist(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_wells = set(frame["well"].astype(str))
    raw_root, files = resolve_raw_train_dir(config, expected_wells)
    allowed_columns = tuple(config["data"]["raw_geometry"]["allowed_columns"])
    if allowed_columns != ("X", "Y", "Z"):
        raise ValueError("raw geometry allowlist changed")
    geometry = np.empty((len(frame), 3), dtype=np.float64)
    inventory: list[dict[str, Any]] = []
    for well, positions in frame.groupby("well", sort=True).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        path = files[str(well)]
        raw = pd.read_csv(path, usecols=list(allowed_columns))
        if tuple(raw.columns) != allowed_columns:
            raw = raw.loc[:, list(allowed_columns)]
        row_idx = frame.iloc[pos]["row_idx"].to_numpy(np.int64)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=-1) >= len(raw):
            raise ValueError(f"raw geometry row index is out of range for {well}")
        values = raw.iloc[row_idx].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"raw X/Y/Z contains nonfinite values for {well}")
        geometry[pos] = values
        inventory.append(
            {
                "well": str(well),
                "filename": path.name,
                "rows": len(raw),
                "bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
            }
        )
    allowlist = pd.DataFrame(
        {
            "well": frame["well"].astype(str),
            "row_idx": frame["row_idx"].to_numpy(np.int64),
            "fold": frame["fold"].to_numpy(np.int8),
            "pred_tvt": frame["pred_tvt"].to_numpy(np.float64),
            "X": geometry[:, 0],
            "Y": geometry[:, 1],
            "Z": geometry[:, 2],
        }
    )
    if tuple(allowlist.columns) != PREDICTION_ALLOWLIST:
        raise ValueError("prediction-phase allowlist schema changed")
    manifest = {
        "root": str(raw_root),
        "loaded_columns": list(allowed_columns),
        "forbidden_columns_loaded": 0,
        "files": len(inventory),
        "wells": len(inventory),
        "file_inventory_sha256": sha256_json(inventory),
        "geometry_logical_sha256": logical_frame_sha256(allowlist),
        "inventory": inventory,
    }
    return allowlist, manifest


def resolve_fold_manifest(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = config["data"]["fold_manifest"]
    path = resolve_sha_qualified_file(
        spec["patterns"],
        str(spec["expected_sha256"]),
        explicit_env="EXP511_FOLD_MANIFEST",
    )
    frame = pd.read_csv(path)
    required = {
        "downstream_outer_fold",
        "inner_fold",
        "outer_train_wells",
        "outer_valid_wells",
    }
    if not required.issubset(frame.columns):
        raise ValueError("nested fold manifest schema changed")
    if sorted(frame["downstream_outer_fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("nested fold manifest outer folds changed")
    return path, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "outer_folds": sorted(int(value) for value in frame["downstream_outer_fold"].unique()),
    }


def resolve_hidden_assignment(config: Mapping[str, Any]) -> Path:
    spec = config["data"]["hidden_like_assignment"]
    return resolve_sha_qualified_file(
        spec["patterns"],
        str(spec["expected_sha256"]),
        explicit_env="EXP511_HIDDEN_ASSIGNMENT",
    )


# %% [markdown]
# ## 5. K16 coefficient and segment-geometry helpers

# %%
def exact_k16_segment_ids(n_transitions: int, segments: int = 16) -> np.ndarray:
    if int(segments) != 16:
        raise ValueError("exp511 is fixed to K16")
    if int(n_transitions) < int(segments):
        raise ValueError("K16 requires at least one transition per segment")
    edges = np.linspace(0.0, float(n_transitions), int(segments) + 1)
    step_index = np.arange(1.0, float(n_transitions) + 1.0)
    return np.clip(
        np.searchsorted(edges[1:], step_index, side="left"),
        0,
        int(segments) - 1,
    ).astype(np.int16)


def cumulative_rate_basis(n_transitions: int, segments: int = 16) -> np.ndarray:
    exact_k16_segment_ids(n_transitions, segments)
    edges = np.linspace(0.0, float(n_transitions), int(segments) + 1)
    step_index = np.arange(1.0, float(n_transitions) + 1.0)
    return np.column_stack(
        [
            np.clip(
                step_index - edges[segment],
                0.0,
                edges[segment + 1] - edges[segment],
            )
            for segment in range(int(segments))
        ]
    ).astype(np.float64)


def solve_smoothed_coefficients(
    prediction: Sequence[float],
    z: Sequence[float],
    *,
    rho: float = 10.0,
    segments: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    pred = np.asarray(prediction, dtype=np.float64)
    z_values = np.asarray(z, dtype=np.float64)
    if len(pred) != len(z_values) or len(pred) <= int(segments):
        raise ValueError("invalid K16 predicted TVT/Z path")
    if not np.isfinite(np.column_stack([pred, z_values])).all():
        raise ValueError("K16 predicted TVT/Z path contains nonfinite values")
    transitions = len(pred) - 1
    phi = cumulative_rate_basis(transitions, segments)
    u = np.cumsum(-np.diff(z_values))
    residual = pred[1:] - pred[0] - u
    normal = phi.T @ phi
    difference = np.diff(np.eye(int(segments)), axis=0)
    scale = float(np.mean(np.diag(normal))) if normal.size else 1.0
    penalized = normal + float(rho) * max(scale, 1.0e-9) * (
        difference.T @ difference
    )
    rhs = phi.T @ residual
    fallback = False
    try:
        coefficients = np.linalg.solve(penalized, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(
            penalized + np.eye(int(segments)) * 1.0e-9,
            rhs,
            rcond=None,
        )[0]
        fallback = True
    fitted = phi @ coefficients
    diagnostics = {
        "rank": int(np.linalg.matrix_rank(phi)),
        "condition": float(np.linalg.cond(penalized)),
        "solver_fallback": fallback,
        "residual_rms_ft": float(np.sqrt(np.mean(np.square(residual - fitted)))),
    }
    return coefficients, phi, residual, diagnostics


def segment_geometry(
    x: Sequence[float],
    y: Sequence[float],
    row_idx: Sequence[int],
    segment_id: np.ndarray,
    contract: K16Contract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    rows = np.asarray(row_idx, dtype=np.int64)
    destination = np.arange(1, len(rows), dtype=np.int64)
    if len(destination) != len(segment_id):
        raise ValueError("K16 segment geometry length mismatch")
    midpoint = np.empty((contract.segments, 2), dtype=np.float64)
    projection = np.empty(contract.segments, dtype=np.float64)
    source_row = np.empty(contract.segments, dtype=np.int64)
    theta = np.radians(contract.theta0_deg)
    for segment in range(contract.segments):
        selected = destination[segment_id == segment]
        if len(selected) == 0:
            raise ValueError(f"K16 segment {segment} is empty")
        start = int(selected[0])
        end = int(selected[-1])
        midpoint[segment] = (
            (x_values[start] + x_values[end]) / 2.0,
            (y_values[start] + y_values[end]) / 2.0,
        )
        azimuth = np.arctan2(
            y_values[end] - y_values[start],
            x_values[end] - x_values[start],
        )
        projection[segment] = np.cos(azimuth - theta)
        source_row[segment] = rows[start]
    return midpoint, projection, source_row


def build_coefficient_field(
    allowlist: pd.DataFrame,
    contract: K16Contract,
) -> tuple[pd.DataFrame, dict[tuple[int, str], WellK16], dict[str, Any]]:
    if tuple(allowlist.columns) != PREDICTION_ALLOWLIST:
        raise ValueError("coefficient builder received a non-allowlisted frame")
    states: dict[tuple[int, str], WellK16] = {}
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (fold, well), positions in allowlist.groupby(
        ["fold", "well"], sort=True
    ).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        part = allowlist.iloc[pos]
        prediction = part["pred_tvt"].to_numpy(np.float64)
        row_idx = part["row_idx"].to_numpy(np.int64)
        coefficients, phi, _residual, diagnostic = solve_smoothed_coefficients(
            prediction,
            part["Z"].to_numpy(np.float64),
            rho=contract.coefficient_smoothing_rho,
            segments=contract.segments,
        )
        segment_id = exact_k16_segment_ids(len(prediction) - 1, contract.segments)
        midpoint, projection, source_row = segment_geometry(
            part["X"],
            part["Y"],
            row_idx,
            segment_id,
            contract,
        )
        state = WellK16(
            fold=int(fold),
            well=str(well),
            positions=pos,
            row_idx=row_idx,
            prediction=prediction,
            phi=phi,
            coefficients=coefficients,
            segment_id=segment_id,
            segment_mid_xy=midpoint,
            segment_projection=projection,
            segment_source_row=source_row,
        )
        states[(int(fold), str(well))] = state
        diagnostics.append({"fold": int(fold), "well": str(well), **diagnostic})
        for segment in range(contract.segments):
            eligible = bool(
                np.isfinite(coefficients[segment])
                and np.isfinite(projection[segment])
                and abs(projection[segment]) >= contract.minimum_abs_projection
            )
            rows.append(
                {
                    "fold": int(fold),
                    "donor_well": str(well),
                    "donor_segment": segment,
                    "source_row": int(source_row[segment]),
                    "x": float(midpoint[segment, 0]),
                    "y": float(midpoint[segment, 1]),
                    "projection": float(projection[segment]),
                    "predicted_k16_coefficient": float(coefficients[segment]),
                    "normalized_field_rate": (
                        float(coefficients[segment] / projection[segment])
                        if eligible
                        else np.nan
                    ),
                    "field_eligible": eligible,
                }
            )
    field = pd.DataFrame(rows, columns=list(FIELD_COLUMNS))
    if len(field) != contract.segments * len(states):
        raise ValueError("K16 coefficient field row count changed")
    if field.duplicated(["fold", "donor_well", "donor_segment"]).any():
        raise ValueError("K16 coefficient field keys are duplicated")
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = {
        "wells": len(states),
        "rows": len(field),
        "eligible_rows": int(field["field_eligible"].sum()),
        "eligible_rate": float(field["field_eligible"].mean()),
        "basis_rank_min": int(diagnostic_frame["rank"].min()),
        "basis_rank_max": int(diagnostic_frame["rank"].max()),
        "solver_fallback_wells": int(diagnostic_frame["solver_fallback"].sum()),
        "condition_max": float(diagnostic_frame["condition"].max()),
        "fit_residual_rms_p50_ft": float(
            diagnostic_frame["residual_rms_ft"].quantile(0.50)
        ),
        "fit_residual_rms_p95_ft": float(
            diagnostic_frame["residual_rms_ft"].quantile(0.95)
        ),
    }
    return field, states, summary


# %% [markdown]
# ## 6. Fold-local donor field and local-linear consensus

# %%
def local_linear_consensus(
    fold_field: pd.DataFrame,
    state: WellK16,
    contract: K16Contract,
) -> LocalLinearResult:
    eligible = fold_field[
        fold_field["field_eligible"]
        & fold_field["donor_well"].astype(str).ne(state.well)
    ].copy()
    donor_xy = eligible[["x", "y"]].to_numpy(np.float64)
    donor_value = eligible["normalized_field_rate"].to_numpy(np.float64)
    donor_well = eligible["donor_well"].astype(str).to_numpy()
    donor_segment = eligible["donor_segment"].to_numpy(np.int16)
    source_row = eligible["source_row"].to_numpy(np.int64)
    query = state.segment_mid_xy
    values = np.full(contract.segments, np.nan, dtype=np.float64)
    selected_count = np.zeros(contract.segments, dtype=np.int32)
    unique_wells = np.zeros(contract.segments, dtype=np.int32)
    nearest = np.full(contract.segments, np.inf, dtype=np.float64)
    median_distance = np.full(contract.segments, np.inf, dtype=np.float64)
    effective = np.zeros(contract.segments, dtype=np.float64)
    finite = np.zeros(contract.segments, dtype=bool)
    self_count = np.zeros(contract.segments, dtype=np.int32)
    for segment, point in enumerate(query):
        if len(eligible) == 0:
            continue
        squared_distance = np.square(donor_xy - point).sum(axis=1)
        finite_index = np.flatnonzero(
            np.isfinite(squared_distance) & np.isfinite(donor_value)
        )
        order = np.lexsort(
            (
                source_row[finite_index],
                donor_segment[finite_index],
                donor_well[finite_index],
                squared_distance[finite_index],
            )
        )
        selected = finite_index[order[: min(contract.local_linear_k, len(order))]]
        selected_count[segment] = len(selected)
        if len(selected) == 0:
            continue
        self_count[segment] = int((donor_well[selected] == state.well).sum())
        unique_wells[segment] = len(np.unique(donor_well[selected]))
        selected_d2 = squared_distance[selected]
        distance = np.sqrt(selected_d2)
        nearest[segment] = float(distance.min())
        median_distance[segment] = float(np.median(distance))
        log_weight = -selected_d2 / (2.0 * contract.bandwidth_ft**2)
        log_weight -= float(log_weight.max())
        weight = np.exp(np.maximum(log_weight, -700.0))
        weight_sum = float(weight.sum())
        effective[segment] = weight_sum**2 / float(np.square(weight).sum())
        delta = (donor_xy[selected] - point) / 1000.0
        design = np.column_stack([np.ones(len(selected)), delta])
        ridge = contract.ridge * weight_sum * np.diag([0.0, 1.0, 1.0])
        normal = (design * weight[:, None]).T @ design + ridge
        rhs = (design * weight[:, None]).T @ donor_value[selected]
        try:
            value = float(np.linalg.solve(normal, rhs)[0])
        except np.linalg.LinAlgError:
            value = float(
                np.linalg.lstsq(
                    normal + np.eye(3) * 1.0e-9,
                    rhs,
                    rcond=None,
                )[0][0]
            )
        values[segment] = value
        finite[segment] = bool(np.isfinite(value))
    return LocalLinearResult(
        values=values,
        selected_segments=selected_count,
        unique_donor_wells=unique_wells,
        nearest_distance=nearest,
        median_distance=median_distance,
        effective_segments=effective,
        finite=finite,
        self_donor_segments=self_count,
    )


def generate_truth_free_predictions(
    base: pd.DataFrame,
    allowlist: pd.DataFrame,
    contract: K16Contract,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not base[["well", "row_idx", "fold"]].reset_index(drop=True).equals(
        allowlist[["well", "row_idx", "fold"]].reset_index(drop=True)
    ):
        raise ValueError("raw geometry allowlist changed OOF keys or row order")
    field, states, field_summary = build_coefficient_field(allowlist, contract)
    correction = np.zeros(len(base), dtype=np.float64)
    candidate = base["pred_tvt"].to_numpy(np.float64).copy()
    support_rows: list[dict[str, Any]] = []
    for (fold, well), state in sorted(states.items()):
        fold_field = field[field["fold"].eq(fold)]
        consensus = local_linear_consensus(fold_field, state, contract)
        query_projection_pass = (
            np.isfinite(state.segment_projection)
            & (np.abs(state.segment_projection) >= contract.minimum_abs_projection)
        )
        unique_pass = (
            consensus.unique_donor_wells >= contract.minimum_unique_donor_wells
        )
        supported = (
            query_projection_pass
            & consensus.finite
            & unique_pass
            & (consensus.self_donor_segments == 0)
        )
        neighbor_coefficient = consensus.values * state.segment_projection
        delta_coefficient = np.where(
            supported,
            neighbor_coefficient - state.coefficients,
            0.0,
        )
        raw_correction = np.concatenate(
            [np.zeros(1, dtype=np.float64), state.phi @ delta_coefficient]
        )
        final_correction = np.clip(
            contract.alpha * raw_correction,
            -contract.correction_cap_ft,
            contract.correction_cap_ft,
        )
        if final_correction[0] != 0.0:
            raise ValueError(f"{well} first score-row correction is not exactly zero")
        correction[state.positions] = final_correction
        candidate[state.positions] = state.prediction + final_correction
        for segment in range(contract.segments):
            support_rows.append(
                {
                    "fold": fold,
                    "well": well,
                    "segment": segment,
                    "source_row": int(state.segment_source_row[segment]),
                    "query_x": float(state.segment_mid_xy[segment, 0]),
                    "query_y": float(state.segment_mid_xy[segment, 1]),
                    "query_projection": float(state.segment_projection[segment]),
                    "own_coefficient": float(state.coefficients[segment]),
                    "neighbor_field_rate": float(consensus.values[segment]),
                    "neighbor_coefficient": float(neighbor_coefficient[segment]),
                    "delta_coefficient": float(delta_coefficient[segment]),
                    "selected_segments": int(consensus.selected_segments[segment]),
                    "unique_donor_wells": int(consensus.unique_donor_wells[segment]),
                    "nearest_donor_distance_ft": float(consensus.nearest_distance[segment]),
                    "median_selected_distance_ft": float(
                        consensus.median_distance[segment]
                    ),
                    "effective_segments": float(consensus.effective_segments[segment]),
                    "query_projection_pass": bool(query_projection_pass[segment]),
                    "finite_prediction_pass": bool(consensus.finite[segment]),
                    "minimum_unique_wells_pass": bool(unique_pass[segment]),
                    "supported": bool(supported[segment]),
                    "self_donor_segments": int(
                        consensus.self_donor_segments[segment]
                    ),
                }
            )
    support = pd.DataFrame(support_rows, columns=list(SUPPORT_COLUMNS))
    predictions = pd.DataFrame(
        {
            "id": base["id"].astype(str),
            "well": base["well"].astype(str),
            "row_idx": base["row_idx"].to_numpy(np.int64),
            "fold": base["fold"].to_numpy(np.int8),
            CONTROL_COLUMN: base["pred_tvt"].to_numpy(np.float64),
            PRIMARY_COLUMN: candidate,
            CORRECTION_COLUMN: correction,
        }
    )
    if tuple(predictions.columns) != PREDICTION_FREEZE_COLUMNS:
        raise ValueError("prediction freeze schema changed")
    if not np.isfinite(
        predictions[[CONTROL_COLUMN, PRIMARY_COLUMN, CORRECTION_COLUMN]].to_numpy()
    ).all():
        raise ValueError("truth-free prediction contains nonfinite values")
    if int(support["self_donor_segments"].sum()) != 0:
        raise ValueError("self donor reached the local-linear fit")
    first_positions = np.asarray(
        [
            positions[0]
            for positions in predictions.groupby("well", sort=False).indices.values()
        ],
        dtype=np.int64,
    )
    if np.max(np.abs(correction[first_positions]), initial=0.0) > 0.0:
        raise ValueError("first score-row continuity contract failed")
    if np.max(np.abs(correction), initial=0.0) > contract.correction_cap_ft + 1.0e-12:
        raise ValueError("correction cap contract failed")
    manifest = {
        "status": "truth_free_predictions_ready_for_freeze",
        "prediction_allowlist": list(PREDICTION_ALLOWLIST),
        "truth_or_error_columns": 0,
        "rows": len(predictions),
        "wells": int(predictions["well"].nunique()),
        "folds": sorted(int(value) for value in predictions["fold"].unique()),
        "prediction_content_sha256": {
            CONTROL_COLUMN: array_sha256(
                predictions[CONTROL_COLUMN].to_numpy(np.float64)
            ),
            PRIMARY_COLUMN: array_sha256(
                predictions[PRIMARY_COLUMN].to_numpy(np.float64)
            ),
            CORRECTION_COLUMN: array_sha256(
                predictions[CORRECTION_COLUMN].to_numpy(np.float64)
            ),
        },
        "prediction_logical_sha256": logical_frame_sha256(predictions),
        "coefficient_field_logical_sha256": logical_frame_sha256(field),
        "support_ledger_logical_sha256": logical_frame_sha256(support),
        "field": field_summary,
        "support": {
            "rows": len(support),
            "supported_segments": int(support["supported"].sum()),
            "supported_rate": float(support["supported"].mean()),
            "self_donor_segments": int(support["self_donor_segments"].sum()),
            "unique_donor_wells_p05": float(
                support["unique_donor_wells"].quantile(0.05)
            ),
            "unique_donor_wells_p50": float(
                support["unique_donor_wells"].quantile(0.50)
            ),
            "nearest_distance_p50_ft": float(
                support["nearest_donor_distance_ft"]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .quantile(0.50)
            ),
        },
        "correction_abs_max_ft": float(np.abs(correction).max(initial=0.0)),
        "first_score_row_abs_correction_max_ft": float(
            np.abs(correction[first_positions]).max(initial=0.0)
        ),
    }
    return predictions, field, support, manifest


# %% [markdown]
# ## 7. Truth-free prediction generation and freeze

# %%
def write_parquet_and_verify(
    frame: pd.DataFrame,
    path: Path,
    columns: Sequence[str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(columns)].to_parquet(path, index=False, compression="zstd")
    reloaded = pd.read_parquet(path, columns=list(columns))
    expected_logical = logical_frame_sha256(frame.loc[:, list(columns)])
    observed_logical = logical_frame_sha256(reloaded)
    if observed_logical != expected_logical:
        raise ValueError(f"parquet round-trip changed logical content: {path.name}")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "logical_sha256": observed_logical,
        "rows": len(reloaded),
        "columns": list(reloaded.columns),
    }


def freeze_truth_free_outputs(
    predictions: pd.DataFrame,
    field: pd.DataFrame,
    support: pd.DataFrame,
    artifacts_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    files = {
        "prediction": write_parquet_and_verify(
            predictions,
            artifacts_dir / "exp511_frozen_predictions.parquet",
            PREDICTION_FREEZE_COLUMNS,
        ),
        "coefficient_field": write_parquet_and_verify(
            field,
            artifacts_dir / "exp511_k16_coefficient_field.parquet",
            FIELD_COLUMNS,
        ),
        "support_ledger": write_parquet_and_verify(
            support,
            artifacts_dir / "exp511_neighbor_support_ledger.parquet",
            SUPPORT_COLUMNS,
        ),
    }
    freeze = {
        "status": "prediction_frozen_before_truth_or_role_access",
        "created_at": datetime.now(UTC).isoformat(),
        "truth_or_error_columns": 0,
        "hidden_role_columns": 0,
        "prediction_manifest": dict(manifest),
        "files": files,
        "prediction_readback_passed": True,
        "field_readback_passed": True,
        "support_readback_passed": True,
    }
    path = artifacts_dir / "exp511_prediction_freeze.json"
    freeze_sha = write_json(path, freeze)
    return freeze, freeze_sha


# %% [markdown]
# ## 8. Truth-late metrics, diagnostics, and all-AND gate

# %%
def load_truth_late(
    path: Path,
    predictions: pd.DataFrame,
    *,
    prediction_freeze_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not prediction_freeze_sha256:
        raise ValueError("truth requires a frozen prediction SHA")
    source = pd.read_parquet(path, columns=list(TRUTH_LATE_OOF_COLUMNS))
    if tuple(source.columns) != TRUTH_LATE_OOF_COLUMNS:
        raise ValueError("truth-late loader column order changed")
    if not source["id"].astype(str).reset_index(drop=True).equals(
        predictions["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("truth-late OOF row order differs from prediction freeze")
    late = pd.DataFrame(
        {
            "id": source["id"].astype(str),
            "md_since": pd.to_numeric(source["md_since"], errors="raise").to_numpy(
                np.float64
            ),
            "last_known_tvt": pd.to_numeric(
                source["last_known_tvt"], errors="raise"
            ).to_numpy(np.float64),
            "target": pd.to_numeric(source["target"], errors="raise").to_numpy(
                np.float64
            ),
            "actual_tvt": pd.to_numeric(
                source["actual_tvt"], errors="raise"
            ).to_numpy(np.float64),
        }
    )
    if not np.isfinite(late.iloc[:, 1:].to_numpy(np.float64)).all():
        raise ValueError("truth-late values contain nonfinite values")
    reconstructed = late["last_known_tvt"] + late["target"]
    parity = float(
        np.max(
            np.abs(
                reconstructed.to_numpy(np.float64)
                - late["actual_tvt"].to_numpy(np.float64)
            ),
            initial=0.0,
        )
    )
    if parity > 1.0e-4:
        raise ValueError("actual_tvt differs from last_known_tvt + target")
    return late, {
        "loaded_after_prediction_freeze": True,
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "rows": len(late),
        "actual_target_parity_max_abs_ft": parity,
        "actual_tvt_sha256": array_sha256(late["actual_tvt"].to_numpy(np.float64)),
    }


def load_hidden_assignment_late(
    path: Path,
    predictions: pd.DataFrame,
    expected_sha256: str,
    *,
    prediction_freeze_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not prediction_freeze_sha256:
        raise ValueError("hidden-like roles require a frozen prediction SHA")
    observed = sha256_file(path)
    if observed != str(expected_sha256):
        raise ValueError("hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *HIDDEN_ROLE_COLUMNS.values()}
    if not required.issubset(assignment.columns) or assignment["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contract failed")
    if not set(predictions["well"].astype(str)).issubset(
        set(assignment["well_id"].astype(str))
    ):
        raise ValueError("hidden-like assignment does not cover exp511 wells")
    return assignment, {
        "path": str(path),
        "sha256": observed,
        "rows": len(assignment),
        "loaded_after_prediction_freeze": True,
        "prediction_freeze_sha256": prediction_freeze_sha256,
    }


def rmse(actual: Sequence[float], prediction: Sequence[float]) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(actual_array - prediction_array))))


def score_pair(
    truth: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    if not bool(mask.any()):
        raise ValueError("exp511 metric scope is empty")
    parent = rmse(truth[mask], control[mask])
    primary = rmse(truth[mask], candidate[mask])
    return parent, primary, primary - parent


def evaluate_predictions(
    predictions: pd.DataFrame,
    truth_late: pd.DataFrame,
    assignment: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    truth = truth_late["actual_tvt"].to_numpy(np.float64)
    control = predictions[CONTROL_COLUMN].to_numpy(np.float64)
    candidate = predictions[PRIMARY_COLUMN].to_numpy(np.float64)
    correction = predictions[CORRECTION_COLUMN].to_numpy(np.float64)
    fold = predictions["fold"].to_numpy(np.int8)
    md_since = truth_late["md_since"].to_numpy(np.float64)
    metric_rows: list[dict[str, Any]] = []

    def append_metric(kind: str, scope: str, mask: np.ndarray) -> None:
        parent, primary, delta = score_pair(truth, control, candidate, mask)
        metric_rows.append(
            {
                "kind": kind,
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(predictions.loc[mask, "well"].nunique()),
                "exp413_rmse": parent,
                "exp511_rmse": primary,
                "delta_exp511_minus_exp413": delta,
                "gain_ft": -delta,
            }
        )

    append_metric("pooled", "all", np.ones(len(predictions), dtype=bool))
    for outer_fold in range(5):
        append_metric("fold", str(outer_fold), fold == outer_fold)
    scope_masks: dict[str, np.ndarray] = {
        "md_0_250": md_since <= 250.0,
        "md_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "md_1000_plus": md_since >= 1000.0,
    }
    assignment_by_well = assignment.set_index("well_id")
    for scope, role_column in HIDDEN_ROLE_COLUMNS.items():
        scope_masks[scope] = (
            predictions["well"]
            .astype(str)
            .map(assignment_by_well[role_column])
            .eq("valid")
            .to_numpy()
        )
    for scope in FIXED_SCOPE_ORDER:
        append_metric("scope", scope, scope_masks[scope])
    metrics = pd.DataFrame(metric_rows)

    joined = predictions[["well", "fold"]].copy()
    joined["truth"] = truth
    joined["control"] = control
    joined["candidate"] = candidate
    by_well_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    support_by_well = support.groupby("well", sort=True)
    for well, positions in predictions.groupby("well", sort=True).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        parent = rmse(truth[pos], control[pos])
        primary = rmse(truth[pos], candidate[pos])
        well_support = support_by_well.get_group(str(well))
        by_well_rows.append(
            {
                "well": str(well),
                "fold": int(fold[pos[0]]),
                "rows": len(pos),
                "exp413_rmse": parent,
                "exp511_rmse": primary,
                "delta_exp511_minus_exp413": primary - parent,
            }
        )
        well_correction = correction[pos]
        continuity_rows.append(
            {
                "well": str(well),
                "fold": int(fold[pos[0]]),
                "rows": len(pos),
                "first_score_row_correction_ft": float(well_correction[0]),
                "correction_abs_mean_ft": float(np.abs(well_correction).mean()),
                "correction_abs_max_ft": float(
                    np.abs(well_correction).max(initial=0.0)
                ),
                "supported_segments": int(well_support["supported"].sum()),
                "supported_segment_rate": float(well_support["supported"].mean()),
                "unique_donor_wells_min": int(
                    well_support["unique_donor_wells"].min()
                ),
                "unique_donor_wells_median": float(
                    well_support["unique_donor_wells"].median()
                ),
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    continuity = pd.DataFrame(continuity_rows)
    pooled = metrics[(metrics["kind"] == "pooled")].iloc[0]
    tail_delta = by_well["delta_exp511_minus_exp413"]
    finite_distance = support["median_selected_distance_ft"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    summary = {
        "pooled": {
            "rows": len(predictions),
            "wells": int(predictions["well"].nunique()),
            "exp413_rmse": float(pooled["exp413_rmse"]),
            "exp511_rmse": float(pooled["exp511_rmse"]),
            "gain_ft": float(pooled["gain_ft"]),
        },
        "tail": {
            "delta_p50_ft": float(tail_delta.quantile(0.50)),
            "delta_p90_ft": float(tail_delta.quantile(0.90)),
            "delta_p95_ft": float(tail_delta.quantile(0.95)),
            "delta_p99_ft": float(tail_delta.quantile(0.99)),
            "worst_well": str(by_well.loc[tail_delta.idxmax(), "well"]),
            "worst_delta_ft": float(tail_delta.max()),
        },
        "continuity": {
            "first_score_row_abs_correction_max_ft": float(
                continuity["first_score_row_correction_ft"].abs().max()
            ),
            "all_row_abs_correction_max_ft": float(
                np.abs(correction).max(initial=0.0)
            ),
            "row_order_change_count": 0,
            "nonfinite_prediction_count": int((~np.isfinite(candidate)).sum()),
        },
        "mechanism": {
            "segments": len(support),
            "supported_segments": int(support["supported"].sum()),
            "supported_segment_rate": float(support["supported"].mean()),
            "unique_donor_wells_p05": float(
                support["unique_donor_wells"].quantile(0.05)
            ),
            "unique_donor_wells_p50": float(
                support["unique_donor_wells"].quantile(0.50)
            ),
            "median_selected_distance_p50_ft": float(finite_distance.quantile(0.50)),
            "median_selected_distance_p95_ft": float(finite_distance.quantile(0.95)),
            "raw_delta_coefficient_abs_p50": float(
                support["delta_coefficient"].abs().quantile(0.50)
            ),
            "raw_delta_coefficient_abs_p95": float(
                support["delta_coefficient"].abs().quantile(0.95)
            ),
            "correction_abs_p50_ft": float(np.quantile(np.abs(correction), 0.50)),
            "correction_abs_p95_ft": float(np.quantile(np.abs(correction), 0.95)),
            "correction_cap_hit_rows": int(
                np.isclose(np.abs(correction), 0.25, rtol=0.0, atol=1.0e-12).sum()
            ),
        },
    }
    return summary, {
        "pooled_fold_scope": metrics,
        "by_well": by_well,
        "continuity": continuity,
    }


def build_promotion_gate(
    summary: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    technical_checks: Mapping[str, bool],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["promotion_gate"]
    metrics = tables["pooled_fold_scope"]
    folds = metrics[metrics["kind"].eq("fold")]
    scopes = metrics[metrics["kind"].eq("scope")].set_index("scope")
    nonworse_folds = int((folds["delta_exp511_minus_exp413"] <= 0.0).sum())
    maximum_scope_delta = float(
        scopes.loc[list(FIXED_SCOPE_ORDER), "delta_exp511_minus_exp413"].max()
    )
    checks = {
        "technical_all_pass": bool(technical_checks) and bool(
            all(technical_checks.values())
        ),
        "pooled_gain_at_least_0p01_ft": float(summary["pooled"]["gain_ft"])
        >= float(gate["pooled_gain_min_ft"]),
        "nonworse_folds_at_least_4_of_5": nonworse_folds
        >= int(gate["nonworse_folds_required"]),
        "all_fixed_scopes_within_plus_0p02_ft": maximum_scope_delta
        <= float(gate["fixed_scope_max_delta_ft"]),
        "by_well_p95_within_plus_0p25_ft": float(
            summary["tail"]["delta_p95_ft"]
        )
        <= float(gate["by_well_p95_max_delta_ft"]),
        "worst_well_within_plus_0p25_ft": float(
            summary["tail"]["worst_delta_ft"]
        )
        <= float(gate["by_well_worst_max_delta_ft"]),
        "first_score_row_exact_zero": float(
            summary["continuity"]["first_score_row_abs_correction_max_ft"]
        )
        <= float(gate["first_score_row_abs_correction_max_ft"]),
        "all_rows_within_correction_cap": float(
            summary["continuity"]["all_row_abs_correction_max_ft"]
        )
        <= float(gate["all_row_abs_correction_max_ft"]),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "decision": (
            "PASS_QUALIFY_SAME_EXP_INFERENCE_IMPLEMENTATION_FOR_SEPARATE_APPROVAL"
            if passed
            else str(gate["fail_decision"])
        ),
        "checks": checks,
        "technical_checks": dict(technical_checks),
        "gain_ft": float(summary["pooled"]["gain_ft"]),
        "nonworse_folds": nonworse_folds,
        "maximum_fixed_scope_delta_ft": maximum_scope_delta,
        "by_well_delta_p95_ft": float(summary["tail"]["delta_p95_ft"]),
        "worst_well_delta_ft": float(summary["tail"]["worst_delta_ft"]),
        "first_score_row_abs_correction_max_ft": float(
            summary["continuity"]["first_score_row_abs_correction_max_ft"]
        ),
        "all_row_abs_correction_max_ft": float(
            summary["continuity"]["all_row_abs_correction_max_ft"]
        ),
        "same_oof_rescue_performed": False,
        "inference_automatically_approved": False,
    }


# %% [markdown]
# ## 9. Stage A orchestration and generated artifacts

# %%
def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def run_stage_a(config: Mapping[str, Any]) -> dict[str, Any]:
    require_stage_a_authorization(config)
    artifacts_dir = output_dir()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    parent_root = resolve_parent_evidence_root(config)
    parent_evidence = verify_parent_evidence(parent_root, config)
    fold_manifest_path, fold_manifest_evidence = resolve_fold_manifest(config)
    del fold_manifest_path
    oof_path = parent_root / str(config["data"]["exp413_oof"]["file"])
    base, oof_manifest = load_truth_free_parent_oof(
        oof_path,
        expected_sha256=str(config["data"]["exp413_oof"]["expected_oof_sha256"]),
        expected_rows=int(config["validation"]["expected_rows"]),
        expected_wells=int(config["validation"]["expected_wells"]),
        expected_folds=[int(value) for value in config["validation"]["expected_folds"]],
    )
    allowlist, geometry_manifest = attach_raw_geometry_allowlist(base, config)
    input_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": runtime_versions(),
        "parent_evidence": parent_evidence,
        "fold_manifest": fold_manifest_evidence,
        "oof": oof_manifest,
        "raw_geometry": geometry_manifest,
        "prediction_allowlist": list(PREDICTION_ALLOWLIST),
        "truth_or_error_reads_before_freeze": 0,
        "hidden_role_reads_before_freeze": 0,
    }
    write_json(artifacts_dir / "exp511_input_manifest.json", input_manifest)

    predictions, field, support, prediction_manifest = generate_truth_free_predictions(
        base, allowlist, contract_from_config(config)
    )
    freeze, freeze_sha = freeze_truth_free_outputs(
        predictions,
        field,
        support,
        artifacts_dir,
        prediction_manifest,
    )

    truth_late, truth_manifest = load_truth_late(
        oof_path,
        predictions,
        prediction_freeze_sha256=freeze_sha,
    )
    hidden_path = resolve_hidden_assignment(config)
    assignment, hidden_manifest = load_hidden_assignment_late(
        hidden_path,
        predictions,
        str(config["data"]["hidden_like_assignment"]["expected_sha256"]),
        prediction_freeze_sha256=freeze_sha,
    )
    observed_control_rmse = rmse(
        truth_late["actual_tvt"], predictions[CONTROL_COLUMN]
    )
    expected_control_rmse = float(config["data"]["exp413_oof"]["expected_cv"])
    first_positions = np.asarray(
        [
            positions[0]
            for positions in predictions.groupby("well", sort=False).indices.values()
        ],
        dtype=np.int64,
    )
    technical_checks = {
        "static_contract": STATIC_CONTRACT["status"] == "pass",
        "parent_evidence_sha": True,
        "fold_manifest_sha": fold_manifest_evidence["sha256"]
        == str(config["validation"]["fold_manifest_sha256"]),
        "expected_rows": len(predictions) == int(config["validation"]["expected_rows"]),
        "expected_wells": int(predictions["well"].nunique())
        == int(config["validation"]["expected_wells"]),
        "expected_folds": sorted(int(value) for value in predictions["fold"].unique())
        == [int(value) for value in config["validation"]["expected_folds"]],
        "truth_free_oof_allowlist": oof_manifest["truth_or_error_columns_loaded"] == 0,
        "raw_geometry_allowlist": geometry_manifest["loaded_columns"] == ["X", "Y", "Z"]
        and geometry_manifest["forbidden_columns_loaded"] == 0,
        "input_content_sha_recorded": bool(
            oof_manifest["prediction_sha256"]
            and geometry_manifest["geometry_logical_sha256"]
        ),
        "key_and_order_frozen": bool(
            oof_manifest["logical_key_sha256"]
            and oof_manifest["global_row_order_sha256"]
            and oof_manifest["per_well_row_order_sha256"]
        ),
        "prediction_freeze_before_truth": bool(
            truth_manifest["loaded_after_prediction_freeze"]
            and hidden_manifest["loaded_after_prediction_freeze"]
        ),
        "prediction_readback": bool(freeze["prediction_readback_passed"]),
        "field_readback": bool(freeze["field_readback_passed"]),
        "support_readback": bool(freeze["support_readback_passed"]),
        "self_donor_zero": int(support["self_donor_segments"].sum()) == 0,
        "prediction_finite": bool(
            np.isfinite(predictions[[CONTROL_COLUMN, PRIMARY_COLUMN]].to_numpy()).all()
        ),
        "control_rmse_parity": abs(observed_control_rmse - expected_control_rmse)
        <= 1.0e-9,
        "actual_target_parity": truth_manifest["actual_target_parity_max_abs_ft"]
        <= 1.0e-4,
        "first_score_row_exact_zero": float(
            predictions.iloc[first_positions][CORRECTION_COLUMN].abs().max()
        )
        <= 1.0e-12,
        "correction_cap": float(predictions[CORRECTION_COLUMN].abs().max())
        <= 0.250000001,
        "no_model_or_physics_rerun": all(
            int(STATIC_CONTRACT["cost"][key]) == 0
            for key in (
                "trained_models",
                "total_boosters",
                "hmm_runs",
                "pf_runs",
                "beam_runs",
                "parent_or_control_retraining",
                "gpu_runs",
            )
        ),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp511 technical checks failed: {technical_checks}")

    summary, tables = evaluate_predictions(
        predictions, truth_late, assignment, support
    )
    gate = build_promotion_gate(summary, tables, technical_checks, config)
    tables["pooled_fold_scope"].to_csv(
        artifacts_dir / "exp511_pooled_fold_scope_metrics.csv", index=False
    )
    tables["by_well"].to_csv(
        artifacts_dir / "exp511_by_well_metrics.csv", index=False
    )
    tables["continuity"].to_csv(
        artifacts_dir / "exp511_continuity_metrics.csv", index=False
    )
    summary_payload = {
        "schema_version": "1.0.0",
        "experiment": EXPERIMENT_NAME,
        "status": "stage_a_complete_pass" if gate["passed"] else "stage_a_complete_fail",
        "route": "ensemble",
        "cost": STATIC_CONTRACT["cost"],
        "summary": summary,
        "promotion_gate": gate,
        "truth_late": truth_manifest,
        "hidden_assignment": hidden_manifest,
        "prediction_freeze_sha256": freeze_sha,
        "inference_implemented": False,
        "submission_created": False,
    }
    summary_sha = write_json(
        artifacts_dir / "exp511_summary.json", summary_payload
    )
    artifact_names = list(config["artifacts"]["expected_train"])
    artifact_sha = {
        name: sha256_file(artifacts_dir / name)
        for name in artifact_names
        if name != "exp511_sha_manifest.json"
    }
    sha_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed_policy": config["reproducibility"]["seed_policy"],
        "environment": runtime_versions(),
        "input_manifest_sha256": sha256_file(
            artifacts_dir / "exp511_input_manifest.json"
        ),
        "prediction_freeze_sha256": freeze_sha,
        "summary_sha256": summary_sha,
        "artifact_sha256": artifact_sha,
        "prediction_content_sha256": prediction_manifest[
            "prediction_content_sha256"
        ],
        "coefficient_field_logical_sha256": prediction_manifest[
            "coefficient_field_logical_sha256"
        ],
        "support_ledger_logical_sha256": prediction_manifest[
            "support_ledger_logical_sha256"
        ],
        "deterministic_anchor": False,
        "independent_rerun_required": True,
    }
    write_json(artifacts_dir / "exp511_sha_manifest.json", sha_manifest)
    metrics_path = KAGGLE_WORKING_ROOT / "metrics.json"
    write_json(metrics_path, summary_payload)
    print(json.dumps(to_jsonable(summary_payload), indent=2, ensure_ascii=False))
    return summary_payload


# %% [markdown]
# ## 10. Setup and implementation-only stop

# %%
CONFIG = read_yaml(EXPERIMENT_DIR / "config.yaml")
STATIC_CONTRACT = validate_static_contract(CONFIG)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "parent": CONFIG["lineage"]["parent"],
            "primary": PRIMARY_COLUMN,
            "prediction_allowlist": list(PREDICTION_ALLOWLIST),
            "cost": STATIC_CONTRACT["cost"],
            "canonical_notebook_adopted": CONFIG["authorization"][
                "canonical_notebook_adoption_approved"
            ],
            "kaggle_run_approved": CONFIG["authorization"]["kaggle_run_approved"],
            "stage_a_enabled": CONFIG["execution"]["stage_a"]["enabled"],
        },
        indent=2,
    )
)

if os.environ.get("EXP511_IMPORT_ONLY", "0") == "1":
    print("EXP511_IMPORT_ONLY=1: contract import completed; Stage A not run.")
else:
    STAGE_A_RESULT = run_stage_a(CONFIG)
