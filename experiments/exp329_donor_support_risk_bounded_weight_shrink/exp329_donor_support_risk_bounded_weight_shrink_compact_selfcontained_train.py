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
# # exp329 donor-support risk bounded-weight-shrink Stage 0
#
# This zero-model CPU readout reconstructs the exact fold-safe exp226 K16 donor
# support without regenerating any parent prediction.  Six unsigned support
# primitives are frozen, converted to exp263-readout-fold-safe empirical
# percentiles, and compared with a deterministic within-well circular control.
# Suffix truth is attached only after every support, risk, control, and saved
# prediction input has been persisted and content-hashed.  Stage 1 prediction
# changes remain deliberately unavailable pending a separate approval.

# %% [markdown]
# ## Contents
# 1. Imports and immutable Stage 0 boundary
# 2. Runtime, configuration, path, SHA, and serialization helpers
# 3. Frozen scientific and execution contract
# 4. Saved exp263 / exp226 / hidden-like input checks
# 5. Exp226 K16 geometry and source-fold-safe donor support reconstruction
# 6. Outer-train ECDF risk and deterministic circular control
# 7. Target-free row contract and late-truth attachment
# 8. Segment AUC, benefit, scope, coverage, and fixed decision
# 9. Full Kaggle CPU orchestration and generated artifacts
# 10. Setup, contract preview, and guarded execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp329_donor_support_risk_bounded_weight_shrink"
OUTPUT_PREFIX = "exp329"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TARGET_FREE_FORBIDDEN = {
    "TVT",
    "tvt_true",
    "true_tvt",
    "target",
    "error",
    "abs_error",
    "absolute_error",
    "oracle_label",
    "oracle_candidate",
}
RISK_SPECS = (
    ("donor_distance", 1.0, "donor_distance_high"),
    ("effective_sample_size", -1.0, "effective_sample_size_low"),
    ("local_linear_log1p_condition", 1.0, "local_linear_condition_high"),
    ("raw_donor_weighted_mad", 1.0, "raw_donor_weighted_mad_high"),
    ("smoothed_donor_weighted_mad", 1.0, "smoothed_donor_weighted_mad_high"),
    (
        "raw_smoothed_local_linear_abs_disagreement",
        1.0,
        "raw_smoothed_local_linear_disagreement_high",
    ),
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP329_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, SHA, and serialization helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    candidates = (Path.cwd() / "config.yaml", experiment_dir() / "config.yaml")
    for path in candidates:
        if path.exists():
            value = read_yaml(path)
            if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
                return value
    raise FileNotFoundError(f"exp329 config was not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    output = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    return (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "metrics.json"
    )


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(str(column).encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "schema_sha256": dataframe_schema_sha(frame),
        "content_sha256": dataframe_content_sha(frame),
    }


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "schema_sha256": dataframe_schema_sha(frame),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def assert_no_target_columns(frame: pd.DataFrame, *, stage: str) -> None:
    leaked = sorted(TARGET_FREE_FORBIDDEN.intersection(frame.columns))
    if leaked:
        raise ValueError(f"{stage} contains forbidden pre-freeze columns: {leaked}")


# %% [markdown]
# ## 3. Frozen scientific and execution contract


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_kaggle_approval: bool
) -> None:
    checks = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.scope": "stage0_train_side_readout",
        "validation.n_folds": 5,
        "validation.readout_fold_policy": "use_saved_exp263_outer_fold_for_metrics",
        "support_risk.segment_scale": "exp226_k16",
        "support_risk.signed_direction_allowed": False,
        "stage_0_readout.enabled_after_implementation": True,
        "stage_0_readout.negative_control.kind": ("within_well_nonzero_circular_shift_of_k16_risk"),
        "stage_1_bounded_shrink.enabled_after_implementation": False,
        "execution.implementation_approved": True,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "execution_contract.stage_0.scientific_risk_scores": 1,
        "execution_contract.stage_0.diagnostic_controls": 1,
        "execution_contract.stage_0.support_well_runs": 773,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in checks.items():
        if get_nested(config, key) != expected:
            raise ValueError(f"exp329 frozen contract changed: {key} must be {expected!r}")
    numeric = {
        "exp226_geometry.theta0": 118.4,
        "exp226_geometry.k_segments": 16.0,
        "exp226_geometry.local_linear_k": 50.0,
        "exp226_geometry.local_linear_bandwidth": 500.0,
        "exp226_geometry.local_linear_ridge": 1.0,
        "exp226_geometry.smooth_rho": 10.0,
        "exp226_geometry.field_min_proj": 0.3,
        "stage_0_readout.pass_requires_all.minimum_pooled_auc": 0.60,
        "stage_0_readout.pass_requires_all.minimum_real_minus_control_auc": 0.05,
        "stage_0_readout.top_risk_minimum": 0.90,
        "stage_0_readout.bottom_risk_maximum": 0.10,
        "stage_0_readout.activation_risk_strictly_greater_than": 0.80,
        "stage_1_bounded_shrink.maximum_shrink_fraction": 0.25,
        "stage_1_bounded_shrink.maximum_absolute_move_ft": 5.0,
        "stage_1_bounded_shrink.minimum_md_since_last_known_ft": 250.0,
    }
    for key, expected in numeric.items():
        if float(get_nested(config, key)) != expected:
            raise ValueError(f"exp329 frozen contract changed: {key} must be {expected}")
    expected_features = [item[2] for item in RISK_SPECS]
    if list(get_nested(config, "support_risk.primitive_features") or []) != expected_features:
        raise ValueError("exp329 fixes exactly six unsigned donor-support risk features")
    expected_formula = {"exp226_k16": 0.50, "likpf_mean": 0.25, "exact_hmm": 0.25}
    actual_formula = {
        str(key): float(value)
        for key, value in (get_nested(config, "data.exp263_cache.expected_formula") or {}).items()
    }
    if actual_formula != expected_formula:
        raise ValueError("exp329 fixes the saved exp263 0.50/0.25/0.25 formula")
    forbidden = set(get_nested(config, "forbidden") or [])
    required_forbidden = {
        "signed_neighbor_error_or_bias_transfer",
        "true_error_or_oracle_gate",
        "k12_k24_stability_gate",
        "gr_likelihood_gate",
        "exp226_likpf_hmm_prediction_regeneration",
        "threshold_alpha_clip_or_destination_grid",
        "hmm_pf_beam_or_model_training",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp329 forbidden-operation inventory is incomplete")
    if require_kaggle_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise RuntimeError("exp329 Kaggle Stage 0 package/push/run is not approved")


# %% [markdown]
# ## 4. Saved exp263 / exp226 / hidden-like input checks


# %%
def resolve_exp263_manifest(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp263_cache") or {}
    expected_sha = str(spec["expected_manifest_sha256"])
    candidates = [str(value) for value in spec.get("manifest_candidates", [])]
    for source in spec.get("kaggle_sources", []):
        candidates.extend(
            [
                str(Path(str(source)) / "artifacts" / "cache_manifest.json"),
                str(Path(str(source)) / "cache_manifest.json"),
            ]
        )
    matches: list[Path] = []
    for raw in candidates:
        candidate = Path(raw)
        for path in (candidate, project_root() / candidate, Path.cwd() / candidate):
            if path.is_file() and sha256_path(path) == expected_sha:
                matches.append(path.resolve())
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob("**/cache_manifest.json")):
            if path.is_file() and sha256_path(path) == expected_sha:
                matches.append(path.resolve())
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise FileNotFoundError("exp263 cache manifest with the frozen SHA was not found")
    return unique[0]


def _cache_partition_path(
    cache_root: Path,
    candidate_id: str,
    fold: int,
    expected_name: str,
    expected_sha256: str,
) -> Path:
    direct = cache_root / "candidate_values" / candidate_id / f"fold={fold}" / expected_name
    if direct.is_file() and sha256_path(direct) == expected_sha256:
        return direct
    pattern = f"**/candidate_values/{candidate_id}/fold={fold}/{expected_name}"
    search_roots = [cache_root]
    if len(cache_root.parents) >= 2:
        search_roots.append(cache_root.parents[1])
    if KAGGLE_INPUT_ROOT.exists():
        search_roots.append(KAGGLE_INPUT_ROOT)
    matches = []
    for root in dict.fromkeys(search_roots):
        matches.extend(
            path.resolve()
            for path in sorted(root.glob(pattern))
            if path.is_file() and sha256_path(path) == expected_sha256
        )
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise FileNotFoundError(
            f"exp263 partition with frozen SHA was not found for "
            f"{candidate_id}/fold={fold}/{expected_name}"
        )
    return unique[0]


def _load_exp263_component(
    cache_root: Path, manifest: Mapping[str, Any], candidate_id: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = list((manifest.get("candidate_value_partitions") or {}).get(candidate_id, []))
    if len(records) != 5:
        raise ValueError(f"exp263 manifest requires five partitions for {candidate_id}")
    parts: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for fold, record in enumerate(records):
        expected_name = Path(str(record["path"])).name
        path = _cache_partition_path(
            cache_root,
            candidate_id,
            fold,
            expected_name,
            str(record["file_sha256"]),
        )
        actual_sha = sha256_path(path)
        if actual_sha != str(record["file_sha256"]):
            raise ValueError(f"exp263 partition SHA mismatch for {candidate_id} fold {fold}")
        frame = pd.read_parquet(
            path,
            columns=["well", "well_row_idx", "outer_fold", "md_since", "candidate_tvt"],
        )
        if len(frame) != int(record["rows"]):
            raise ValueError(f"exp263 partition row mismatch for {candidate_id} fold {fold}")
        if not bool((pd.to_numeric(frame["outer_fold"], errors="raise") == fold).all()):
            raise ValueError(f"exp263 partition fold identity mismatch for {candidate_id}")
        parts.append(frame)
        evidence.append(
            {
                "candidate_id": candidate_id,
                "fold": fold,
                "path": str(path),
                "rows": len(frame),
                "raw_sha256": actual_sha,
                "manifest_content_sha256": str(record["content_sha256"]),
            }
        )
    output = pd.concat(parts, ignore_index=True)
    output["well"] = output["well"].astype(str)
    output["well_row_idx"] = pd.to_numeric(output["well_row_idx"], errors="raise").astype(np.int64)
    output["outer_fold"] = pd.to_numeric(output["outer_fold"], errors="raise").astype(np.int8)
    output["md_since"] = pd.to_numeric(output["md_since"], errors="raise").astype(np.float64)
    output["candidate_tvt"] = pd.to_numeric(output["candidate_tvt"], errors="raise").astype(
        np.float32
    )
    output = output.sort_values(["well", "well_row_idx"], kind="mergesort").reset_index(drop=True)
    if output.duplicated(["well", "well_row_idx"]).any():
        raise ValueError(f"exp263 {candidate_id} contains duplicate identities")
    if not np.isfinite(output[["md_since", "candidate_tvt"]].to_numpy()).all():
        raise ValueError(f"exp263 {candidate_id} contains non-finite values")
    return output, evidence


def materialize_exp263_fixed_blend(
    exp226_k16: np.ndarray, likpf_mean: np.ndarray, exact_hmm: np.ndarray
) -> np.ndarray:
    components = [
        np.asarray(exp226_k16, dtype=np.float32),
        np.asarray(likpf_mean, dtype=np.float32),
        np.asarray(exact_hmm, dtype=np.float32),
    ]
    if len({item.shape for item in components}) != 1:
        raise ValueError("exp263 fixed-blend components must have identical shape")
    output = np.zeros(components[0].shape, dtype=np.float64)
    for weight, component in zip((0.50, 0.25, 0.25), components, strict=True):
        output += weight * component.astype(np.float64)
    return output.astype(np.float32)


def materialize_other_blend(likpf_mean: np.ndarray, exact_hmm: np.ndarray) -> np.ndarray:
    left = np.asarray(likpf_mean, dtype=np.float32)
    right = np.asarray(exact_hmm, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("non-exp226 components must have identical shape")
    output = 0.50 * left.astype(np.float64) + 0.50 * right.astype(np.float64)
    return output.astype(np.float32)


def arrays_equal_with_missing(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare identity arrays across numeric and object dtypes, treating aligned NaNs as equal."""
    left_series = pd.Series(np.asarray(left)).reset_index(drop=True)
    right_series = pd.Series(np.asarray(right)).reset_index(drop=True)
    return bool(left_series.equals(right_series))


def load_exp263_base(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = resolve_exp263_manifest(config)
    manifest = json.loads(manifest_path.read_text())
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if (
        int(manifest.get("rows", -1)) != int(get_nested(config, "validation.expected_rows"))
        or int(manifest.get("wells", -1)) != int(get_nested(config, "validation.expected_wells"))
        or int(manifest.get("folds", -1)) != len(expected_folds)
    ):
        raise ValueError("exp263 cache coverage differs from the exp329 contract")
    base: pd.DataFrame | None = None
    evidence: list[dict[str, Any]] = []
    values: dict[str, np.ndarray] = {}
    keys = ["well", "well_row_idx", "outer_fold", "md_since"]
    for candidate_id in ("exp226_k16", "likpf_mean", "exact_hmm"):
        frame, part_evidence = _load_exp263_component(manifest_path.parent, manifest, candidate_id)
        evidence.extend(part_evidence)
        if base is None:
            base = frame[keys].copy()
        else:
            if len(base) != len(frame):
                raise ValueError("exp263 component row counts differ")
            for column in keys:
                if not arrays_equal_with_missing(
                    base[column].to_numpy(), frame[column].to_numpy()
                ):
                    raise ValueError(f"exp263 component identity mismatch in {column}")
        values[candidate_id] = frame["candidate_tvt"].to_numpy(np.float32)
    assert base is not None
    p_base = materialize_exp263_fixed_blend(
        values["exp226_k16"], values["likpf_mean"], values["exact_hmm"]
    )
    p_other = materialize_other_blend(values["likpf_mean"], values["exact_hmm"])
    direct = (
        0.50 * values["exp226_k16"].astype(np.float64)
        + 0.25 * values["likpf_mean"].astype(np.float64)
        + 0.25 * values["exact_hmm"].astype(np.float64)
    ).astype(np.float32)
    parity = float(
        np.max(np.abs(p_base.astype(np.float64) - direct.astype(np.float64)), initial=0.0)
    )
    if parity > float(
        get_nested(config, "validation.technical_guards.maximum_exp263_formula_parity_abs_ft")
    ):
        raise ValueError("exp263 formula parity guard failed")
    base = base.rename(
        columns={
            "well": "well_id",
            "well_row_idx": "row_idx",
            "outer_fold": "readout_fold",
            "md_since": "md_since_ft",
        }
    )
    base["p226"] = values["exp226_k16"].astype(np.float64)
    base["p_likpf"] = values["likpf_mean"].astype(np.float64)
    base["p_exact_hmm"] = values["exact_hmm"].astype(np.float64)
    base["p_base"] = p_base.astype(np.float64)
    base["p_other"] = p_other.astype(np.float64)
    return base, {
        "name": "exp263_saved_candidate_cache",
        "path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "rows": len(base),
        "wells": int(base["well_id"].nunique()),
        "folds": sorted(int(value) for value in base["readout_fold"].unique()),
        "formula_parity_max_abs_ft": parity,
        "partition_evidence": evidence,
    }


def audit_saved_fold_relationship(
    well_ids: Iterable[str],
    exp226_source_folds: Iterable[int],
    exp263_readout_folds: Iterable[int],
    expected_folds: Iterable[int],
) -> dict[str, Any]:
    relationship = pd.DataFrame(
        {
            "well_id": np.asarray(list(well_ids), dtype=object).astype(str),
            "exp226_source_fold": np.asarray(list(exp226_source_folds), dtype=np.int64),
            "exp263_readout_fold": np.asarray(list(exp263_readout_folds), dtype=np.int64),
        }
    )
    required = sorted(int(value) for value in expected_folds)
    if relationship.empty:
        raise ValueError("saved fold relationship requires rows")
    for column in ("exp226_source_fold", "exp263_readout_fold"):
        if sorted(int(value) for value in relationship[column].unique()) != required:
            raise ValueError(f"{column} does not contain the fixed five-fold set")
        if int(relationship.groupby("well_id", sort=False)[column].nunique().max()) != 1:
            raise ValueError(f"{column} must be constant within each well")
    contingency = (
        relationship.groupby(
            ["exp226_source_fold", "exp263_readout_fold"], sort=True, observed=True
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    return {
        "policy": "exp226_source_fold_for_donors_exp263_outer_fold_for_readout",
        "same_fold_row_fraction": float(
            (relationship["exp226_source_fold"] == relationship["exp263_readout_fold"]).mean()
        ),
        "row_contingency": contingency.to_dict(orient="records"),
    }


def load_exp226_safe(
    config: Mapping[str, Any], base: pd.DataFrame
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["target_free_columns"]]
    if set(spec["forbidden_pre_freeze_columns"]).intersection(safe_columns):
        raise ValueError("exp226 safe-column list contains forbidden truth/error columns")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    assert_no_target_columns(frame, stage="exp226 saved OOF")
    for column in ("fold", "row_idx", "suffix_offset"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_pred"] = pd.to_numeric(frame["tvt_pred"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(frame["tvt_pred"]).all():
        raise ValueError("exp226 target-free OOF identity/value guard failed")
    if len(frame) != len(base):
        raise ValueError("exp226 and exp263 row counts differ")
    for column in ("well_id", "row_idx"):
        if not np.array_equal(frame[column].to_numpy(), base[column].to_numpy()):
            raise ValueError(f"exp226/exp263 identity mismatch in {column}")
    fold_audit = audit_saved_fold_relationship(
        frame["well_id"],
        frame["fold"],
        base["readout_fold"],
        get_nested(config, "validation.expected_folds"),
    )
    anchor_delta = np.abs(
        frame["tvt_pred"].to_numpy(np.float32).astype(np.float64)
        - base["p226"].to_numpy(np.float64)
    )
    anchor_parity = float(anchor_delta.max(initial=0.0))
    if anchor_parity > 1e-5:
        raise ValueError("saved exp226 OOF and exp263 cached anchor differ")
    output = base.copy()
    output["exp226_source_fold"] = frame["fold"].to_numpy(np.int8)
    output["suffix_offset"] = frame["suffix_offset"].to_numpy(np.int64)
    return (
        output,
        path,
        {
            "name": "exp226_saved_oof_target_free",
            "path": str(path),
            "bytes": path.stat().st_size,
            "raw_sha256": sha256_path(path),
            "decompressed_sha256": decompressed_sha,
            "rows": len(frame),
            "wells": int(frame["well_id"].nunique()),
            "source_folds": sorted(int(value) for value in frame["fold"].unique()),
            "cached_anchor_parity_max_abs_ft": anchor_parity,
            "fold_relationship": fold_audit,
        },
    )


def load_exp226_truth(path: Path, *, target_free_contract_sha256: str) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("late truth requires a frozen target-free contract SHA")
    frame = pd.read_csv(path, usecols=["well_id", "row_idx", "tvt_true"], dtype={"well_id": str})
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("late exp226 truth rows must be unique and finite")
    return frame


def load_hidden_like_assignments(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns) or frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment schema/identity guard failed")
    return frame, {
        "name": "exp115_hidden_like_assignments_readout_only",
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


# %% [markdown]
# ## 5. Exp226 K16 geometry and source-fold-safe donor support reconstruction


# %%
@dataclass(frozen=True)
class K16Params:
    theta0: float = 118.4
    k_segments: int = 16
    local_linear_k: int = 50
    local_linear_bandwidth: float = 500.0
    local_linear_ridge: float = 1.0
    smooth_rho: float = 10.0
    field_min_proj: float = 0.3


@dataclass
class WellGeometry:
    wid: str
    wi: int
    s: int
    n: int
    ndz: np.ndarray
    anchor: float
    segid: np.ndarray
    mid: np.ndarray
    proj: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    c_raw: np.ndarray | None = None
    c_smoothed: np.ndarray | None = None

    @property
    def suffix_row_idx(self) -> np.ndarray:
        return np.arange(self.s + 1, self.s + 1 + self.n, dtype=np.int64)


def params_from_config(config: Mapping[str, Any]) -> K16Params:
    spec = get_nested(config, "exp226_geometry") or {}
    params = K16Params(
        theta0=float(spec["theta0"]),
        k_segments=int(spec["k_segments"]),
        local_linear_k=int(spec["local_linear_k"]),
        local_linear_bandwidth=float(spec["local_linear_bandwidth"]),
        local_linear_ridge=float(spec["local_linear_ridge"]),
        smooth_rho=float(spec["smooth_rho"]),
        field_min_proj=float(spec["field_min_proj"]),
    )
    if params != K16Params():
        raise ValueError("exp329 must preserve every exp226 support parameter")
    return params


def last_known_index(tvt_input: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(tvt_input))
    if not len(finite):
        raise ValueError("well has no finite TVT_input anchor")
    return int(finite[-1])


def segment_geometry(
    x: np.ndarray, y: np.ndarray, s: int, n: int, params: K16Params
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, n, params.k_segments + 1)
    step_idx = np.arange(1, n + 1.0)
    segid = np.clip(np.searchsorted(edges[1:], step_idx, side="left"), 0, params.k_segments - 1)
    mid = np.empty((params.k_segments, 2), dtype=np.float64)
    proj = np.empty(params.k_segments, dtype=np.float64)
    az = np.empty(params.k_segments, dtype=np.float64)
    theta = np.radians(params.theta0)
    last_idx = len(x) - 1
    for segment in range(params.k_segments):
        first = min(s + 1 + int(edges[segment]), last_idx)
        last_raw = s + 1 + max(int(edges[segment + 1]) - 1, int(edges[segment]))
        last = min(max(last_raw, first), last_idx)
        az[segment] = np.arctan2(y[last] - y[first], x[last] - x[first])
        mid[segment] = ((x[first] + x[last]) / 2.0, (y[first] + y[last]) / 2.0)
        proj[segment] = np.cos(az[segment] - theta)
    return segid.astype(np.int16), mid, proj


def fit_coeffs(r0: np.ndarray, u: np.ndarray, n: int, params: K16Params, rho: float) -> np.ndarray:
    t = np.arange(1, n + 1.0)
    edges = np.linspace(0, n, params.k_segments + 1)
    phi = np.column_stack(
        [np.clip(t - edges[j], 0, edges[j + 1] - edges[j]) for j in range(params.k_segments)]
    )
    matrix = phi.T @ phi
    if rho > 0:
        difference = np.diff(np.eye(params.k_segments), axis=0)
        scale = float(np.mean(np.diag(matrix))) if matrix.size else 1.0
        matrix = matrix + rho * max(scale, 1e-9) * difference.T @ difference
    rhs = phi.T @ (r0 - u)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix + np.eye(params.k_segments) * 1e-9, rhs, rcond=None)[0]


def resolve_raw_train_dir(
    config: Mapping[str, Any], expected_wells: set[str]
) -> tuple[Path, list[Path]]:
    candidates: list[Path] = []
    for raw in get_nested(config, "data.raw_train_dir_patterns") or []:
        path = Path(str(raw))
        for candidate in (path, project_root() / path, Path.cwd() / path):
            if candidate.is_dir():
                candidates.append(candidate.resolve())
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.resolve() for path in KAGGLE_INPUT_ROOT.glob("**/train") if path.is_dir()
        )
    for directory in dict.fromkeys(candidates):
        files = sorted(directory.glob(str(get_nested(config, "data.raw_horizontal_glob"))))
        wells = {path.name.replace("__horizontal_well.csv", "") for path in files}
        if wells == expected_wells and len(files) == len(expected_wells):
            return directory, files
    raise FileNotFoundError("raw train directory with the exact 773-well inventory was not found")


def validate_raw_well_identity(
    config: Mapping[str, Any], raw_dir: Path, horizontal_files: Sequence[Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    for horizontal_path in horizontal_files:
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        records.append(
            {
                "well_id": well,
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(records).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    identity_sha = dataframe_content_sha(
        frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    if identity_sha != str(get_nested(config, "data.expected_raw_well_identity_sha256")):
        raise ValueError("raw well-file identity SHA differs from the frozen contract")
    return frame, {
        "name": "raw_train_well_file_identity",
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": identity_sha,
    }


def load_target_free_wells(
    horizontal_files: Sequence[Path], params: K16Params
) -> list[WellGeometry]:
    wells: list[WellGeometry] = []
    for wi, path in enumerate(horizontal_files):
        frame = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT_input"])
        x = frame["X"].to_numpy(np.float64)
        y = frame["Y"].to_numpy(np.float64)
        z = frame["Z"].to_numpy(np.float64)
        visible = frame["TVT_input"].to_numpy(np.float64)
        s = last_known_index(visible)
        ndz = -np.diff(z)[s:]
        if not len(ndz):
            raise ValueError(f"{path.name} has no prediction suffix")
        segid, mid, proj = segment_geometry(x, y, s, len(ndz), params)
        wells.append(
            WellGeometry(
                wid=path.name.replace("__horizontal_well.csv", ""),
                wi=wi,
                s=s,
                n=len(ndz),
                ndz=ndz,
                anchor=float(visible[s]),
                segid=segid,
                mid=mid,
                proj=proj,
                x=x,
                y=y,
                z=z,
            )
        )
    return wells


def attach_donor_fit_truth(
    well: WellGeometry, horizontal_path: Path, params: K16Params
) -> WellGeometry:
    tvt = pd.read_csv(horizontal_path, usecols=["TVT"])["TVT"].to_numpy(np.float64)
    if len(tvt) != len(well.z) or not np.isfinite(tvt).all():
        raise ValueError(f"invalid donor-fit TVT: {horizontal_path}")
    if not math.isclose(float(tvt[well.s]), well.anchor, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"donor known-anchor parity failed: {well.wid}")
    r0 = tvt[well.s + 1 :] - tvt[well.s]
    u = np.cumsum(well.ndz)
    return replace(
        well,
        c_raw=fit_coeffs(r0, u, well.n, params, rho=0.0),
        c_smoothed=fit_coeffs(r0, u, well.n, params, rho=params.smooth_rho),
    )


def build_support_fields(
    source_wells: Sequence[WellGeometry], params: K16Params
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows: list[tuple[float, ...]] = []
    smoothed_rows: list[tuple[float, ...]] = []
    for well in source_wells:
        if well.c_raw is None or well.c_smoothed is None:
            raise ValueError(f"donor-fit coefficients are missing for {well.wid}")
        for segment in range(params.k_segments):
            if abs(well.proj[segment]) <= params.field_min_proj:
                continue
            common = (
                float(well.mid[segment, 0]),
                float(well.mid[segment, 1]),
                float(well.wi),
                float(segment),
            )
            raw_rows.append(
                (
                    common[0],
                    common[1],
                    float(well.c_raw[segment] / well.proj[segment]),
                    common[2],
                    common[3],
                )
            )
            smoothed_rows.append(
                (
                    common[0],
                    common[1],
                    float(well.c_smoothed[segment] / well.proj[segment]),
                    common[2],
                    common[3],
                )
            )
    raw = np.asarray(raw_rows, dtype=np.float64)
    smoothed = np.asarray(smoothed_rows, dtype=np.float64)
    if raw.shape != smoothed.shape or raw.ndim != 2 or raw.shape[1] != 5:
        raise ValueError("raw/smoothed exp226 donor field shape mismatch")
    if not np.array_equal(raw[:, [0, 1, 3, 4]], smoothed[:, [0, 1, 3, 4]]):
        raise ValueError("raw/smoothed donor identities differ")
    return raw, smoothed


def _safe_nearest_indices(d2: np.ndarray, k: int) -> np.ndarray:
    if not len(d2):
        return np.empty(0, dtype=np.int64)
    count = min(max(int(k), 1), len(d2))
    return np.argpartition(d2, count - 1)[:count]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if value.shape != weight.shape or not len(value) or not np.isfinite(value).all():
        raise ValueError("weighted median requires aligned finite values")
    order = np.argsort(value, kind="stable")
    ordered_weight = weight[order]
    threshold = 0.5 * float(ordered_weight.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weight), threshold, side="left"))
    return float(value[order[min(index, len(order) - 1)]])


def solve_local_linear(
    x_design: np.ndarray, weights: np.ndarray, values: np.ndarray, ridge_scale: float
) -> tuple[float, float]:
    ridge = ridge_scale * float(weights.sum()) * np.diag([0.0, 1.0, 1.0])
    normal = (x_design * weights[:, None]).T @ x_design + ridge
    rhs = (x_design * weights[:, None]).T @ values
    try:
        intercept = float(np.linalg.solve(normal, rhs)[0])
    except np.linalg.LinAlgError:
        intercept = float(np.linalg.lstsq(normal + np.eye(3) * 1e-9, rhs, rcond=None)[0][0])
    condition = float(np.linalg.cond(normal))
    if not math.isfinite(intercept) or not math.isfinite(condition):
        raise ValueError("non-finite local-linear support diagnostic")
    return intercept, condition


def reconstruct_well_support(
    target: WellGeometry,
    raw_field: np.ndarray,
    smoothed_field: np.ndarray,
    params: K16Params,
    wi_to_well: Mapping[int, str],
    source_fold_by_well: Mapping[str, int],
    readout_fold: int,
    target_source_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if np.any(raw_field[:, 3].astype(np.int64) == target.wi):
        raise ValueError(f"validation well leaked into its donor field: {target.wid}")
    feature_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for segment in range(params.k_segments):
        d2 = np.square(raw_field[:, 0] - target.mid[segment, 0]) + np.square(
            raw_field[:, 1] - target.mid[segment, 1]
        )
        selected = _safe_nearest_indices(d2, params.local_linear_k)
        if len(selected) != params.local_linear_k:
            raise ValueError("exp329 requires the complete exp226 k50 donor set")
        weights = np.exp(
            np.maximum(-d2[selected] / (2.0 * params.local_linear_bandwidth**2), -700.0)
        )
        dx = (raw_field[selected, 0] - target.mid[segment, 0]) / 1000.0
        dy = (raw_field[selected, 1] - target.mid[segment, 1]) / 1000.0
        design = np.column_stack([np.ones(len(selected)), dx, dy])
        raw_intercept, condition = solve_local_linear(
            design, weights, raw_field[selected, 2], params.local_linear_ridge
        )
        smoothed_intercept, smoothed_condition = solve_local_linear(
            design, weights, smoothed_field[selected, 2], params.local_linear_ridge
        )
        if not math.isclose(condition, smoothed_condition, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("raw/smoothed local-linear normal matrices differ")
        weight_sum = float(weights.sum())
        effective_n = weight_sum**2 / float(np.square(weights).sum())
        raw_center = weighted_median(raw_field[selected, 2], weights)
        smoothed_center = weighted_median(smoothed_field[selected, 2], weights)
        raw_mad = weighted_median(np.abs(raw_field[selected, 2] - raw_center), weights)
        smoothed_mad = weighted_median(
            np.abs(smoothed_field[selected, 2] - smoothed_center), weights
        )
        donor_distance = float(np.sqrt(np.median(np.sort(d2[selected])[: min(15, len(selected))])))
        feature_rows.append(
            {
                "well_id": target.wid,
                "readout_fold": readout_fold,
                "exp226_source_fold": target_source_fold,
                "segment_id": segment,
                "donor_count": len(selected),
                "donor_distance": donor_distance,
                "effective_sample_size": effective_n,
                "local_linear_log1p_condition": math.log1p(condition),
                "raw_donor_weighted_mad": raw_mad,
                "smoothed_donor_weighted_mad": smoothed_mad,
                "raw_smoothed_local_linear_abs_disagreement": abs(
                    raw_intercept - smoothed_intercept
                ),
            }
        )
        canonical = sorted(
            range(len(selected)),
            key=lambda index: (
                float(d2[selected[index]]),
                wi_to_well[int(raw_field[selected[index], 3])],
                int(raw_field[selected[index], 4]),
            ),
        )
        for donor_rank, selected_position in enumerate(canonical, start=1):
            field_index = int(selected[selected_position])
            donor_well = wi_to_well[int(raw_field[field_index, 3])]
            donor_fold = int(source_fold_by_well[donor_well])
            if donor_fold == target_source_fold:
                raise ValueError("exp226 validation source fold entered the donor ledger")
            ledger_rows.append(
                {
                    "target_well_id": target.wid,
                    "readout_fold": readout_fold,
                    "exp226_source_fold": target_source_fold,
                    "target_segment_id": segment,
                    "donor_rank": donor_rank,
                    "donor_well_id": donor_well,
                    "donor_source_fold": donor_fold,
                    "donor_segment_id": int(raw_field[field_index, 4]),
                    "distance_ft": float(np.sqrt(d2[field_index])),
                    "kernel_weight": float(weights[selected_position]),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(ledger_rows)


def build_support_primitives(
    config: Mapping[str, Any],
    base: pd.DataFrame,
    horizontal_files: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = params_from_config(config)
    target_wells = load_target_free_wells(horizontal_files, params)
    by_path = {path.name.replace("__horizontal_well.csv", ""): path for path in horizontal_files}
    donor_wells = {
        well.wid: attach_donor_fit_truth(well, by_path[well.wid], params) for well in target_wells
    }
    source_fold_by_well = (
        base[["well_id", "exp226_source_fold"]]
        .drop_duplicates()
        .set_index("well_id")["exp226_source_fold"]
        .astype(int)
        .to_dict()
    )
    readout_fold_by_well = (
        base[["well_id", "readout_fold"]]
        .drop_duplicates()
        .set_index("well_id")["readout_fold"]
        .astype(int)
        .to_dict()
    )
    wi_to_well = {well.wi: well.wid for well in target_wells}
    feature_parts: list[pd.DataFrame] = []
    ledger_parts: list[pd.DataFrame] = []
    fold_audit_rows: list[dict[str, Any]] = []
    for source_fold in sorted(
        int(value) for value in get_nested(config, "validation.expected_folds")
    ):
        source = [
            donor_wells[well.wid]
            for well in target_wells
            if source_fold_by_well[well.wid] != source_fold
        ]
        valid = [well for well in target_wells if source_fold_by_well[well.wid] == source_fold]
        source_ids = {well.wid for well in source}
        valid_ids = {well.wid for well in valid}
        if source_ids & valid_ids:
            raise ValueError("exp226 source/validation donor fold overlap")
        raw_field, smoothed_field = build_support_fields(source, params)
        fold_audit_rows.append(
            {
                "exp226_source_fold": source_fold,
                "source_wells": len(source),
                "validation_wells": len(valid),
                "donor_field_rows": len(raw_field),
                "source_validation_overlap": 0,
                "raw_field_sha256": array_sha256(raw_field),
                "smoothed_field_sha256": array_sha256(smoothed_field),
            }
        )
        for index, target in enumerate(valid, start=1):
            features, ledger = reconstruct_well_support(
                target,
                raw_field,
                smoothed_field,
                params,
                wi_to_well,
                source_fold_by_well,
                readout_fold_by_well[target.wid],
                source_fold,
            )
            feature_parts.append(features)
            ledger_parts.append(ledger)
            if index % 25 == 0 or index == len(valid):
                print(
                    f"support source_fold={source_fold} wells={index}/{len(valid)} "
                    f"donors={len(source)}"
                )
    support = (
        pd.concat(feature_parts, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    ledger = (
        pd.concat(ledger_parts, ignore_index=True)
        .sort_values(["target_well_id", "target_segment_id", "donor_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    expected_segments = len(target_wells) * params.k_segments
    if len(support) != expected_segments or support.duplicated(["well_id", "segment_id"]).any():
        raise ValueError("support primitive segment coverage failed")
    numeric = support[[item[0] for item in RISK_SPECS]].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("support primitives contain non-finite values")
    row_layout_parts = []
    for well in target_wells:
        row_layout_parts.append(
            pd.DataFrame(
                {
                    "well_id": well.wid,
                    "row_idx": well.suffix_row_idx,
                    "segment_id": well.segid.astype(np.int16),
                }
            )
        )
    row_layout = (
        pd.concat(row_layout_parts, ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    return support, ledger, row_layout, pd.DataFrame(fold_audit_rows)


# %% [markdown]
# ## 6. Outer-train ECDF risk and deterministic circular control


# %%
def stable_empirical_percentile(
    train_values: np.ndarray,
    train_wells: Sequence[str],
    train_segments: np.ndarray,
    query_values: np.ndarray,
    query_wells: Sequence[str],
    query_segments: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(train_values, dtype=np.float64)
    query = np.asarray(query_values, dtype=np.float64)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("ECDF requires non-empty finite reference and finite query")
    order = sorted(
        range(len(reference)),
        key=lambda index: (
            float(reference[index]),
            str(train_wells[index]),
            int(train_segments[index]),
        ),
    )
    sorted_values = reference[order]
    sorted_keys = [(str(train_wells[index]), int(train_segments[index])) for index in order]
    result = np.empty(len(query), dtype=np.float64)
    for index, (value, well, segment) in enumerate(
        zip(query, query_wells, query_segments, strict=True)
    ):
        left = int(np.searchsorted(sorted_values, value, side="left"))
        right = int(np.searchsorted(sorted_values, value, side="right"))
        if right > left:
            count = left + bisect_right(sorted_keys[left:right], (str(well), int(segment)))
        else:
            count = left
        result[index] = count / len(reference)
    return result


def _cdf_rows(
    evaluation_fold: int,
    feature: str,
    values: np.ndarray,
    wells: Sequence[str],
    segments: np.ndarray,
) -> pd.DataFrame:
    order = sorted(
        range(len(values)),
        key=lambda index: (float(values[index]), str(wells[index]), int(segments[index])),
    )
    return pd.DataFrame(
        {
            "evaluation_fold": evaluation_fold,
            "feature": feature,
            "reference_rank": np.arange(1, len(order) + 1, dtype=np.int64),
            "directed_value": np.asarray(values, dtype=np.float64)[order],
            "reference_well_id": np.asarray(wells, dtype=object)[order],
            "reference_segment_id": np.asarray(segments, dtype=np.int16)[order],
            "empirical_percentile": np.arange(1, len(order) + 1, dtype=np.float64) / len(order),
        }
    )


def fit_outer_train_risk(
    support: pd.DataFrame, expected_folds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_parts: list[pd.DataFrame] = []
    cdf_parts: list[pd.DataFrame] = []
    for fold in expected_folds:
        train = support.loc[support["readout_fold"] != fold].copy()
        valid = support.loc[support["readout_fold"] == fold].copy()
        if train.empty or valid.empty:
            raise ValueError(f"readout fold {fold} has empty ECDF train/validation")
        train_percentiles: list[np.ndarray] = []
        valid_percentiles: list[np.ndarray] = []
        for column, direction, risk_name in RISK_SPECS:
            train_directed = direction * train[column].to_numpy(np.float64)
            valid_directed = direction * valid[column].to_numpy(np.float64)
            train_pct = stable_empirical_percentile(
                train_directed,
                train["well_id"].astype(str).tolist(),
                train["segment_id"].to_numpy(np.int16),
                train_directed,
                train["well_id"].astype(str).tolist(),
                train["segment_id"].to_numpy(np.int16),
            )
            valid_pct = stable_empirical_percentile(
                train_directed,
                train["well_id"].astype(str).tolist(),
                train["segment_id"].to_numpy(np.int16),
                valid_directed,
                valid["well_id"].astype(str).tolist(),
                valid["segment_id"].to_numpy(np.int16),
            )
            train_percentiles.append(train_pct)
            valid_percentiles.append(valid_pct)
            valid[risk_name] = valid_pct
            cdf_parts.append(
                _cdf_rows(
                    fold,
                    risk_name,
                    train_directed,
                    train["well_id"].astype(str).tolist(),
                    train["segment_id"].to_numpy(np.int16),
                )
            )
        train_composite = np.mean(np.column_stack(train_percentiles), axis=1)
        valid_composite = np.mean(np.column_stack(valid_percentiles), axis=1)
        valid["raw_composite"] = valid_composite
        valid["risk_score"] = stable_empirical_percentile(
            train_composite,
            train["well_id"].astype(str).tolist(),
            train["segment_id"].to_numpy(np.int16),
            valid_composite,
            valid["well_id"].astype(str).tolist(),
            valid["segment_id"].to_numpy(np.int16),
        )
        cdf_parts.append(
            _cdf_rows(
                fold,
                "raw_composite",
                train_composite,
                train["well_id"].astype(str).tolist(),
                train["segment_id"].to_numpy(np.int16),
            )
        )
        scored_parts.append(valid)
    scored = (
        pd.concat(scored_parts, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(scored) != len(support) or not np.isfinite(scored["risk_score"]).all():
        raise ValueError("outer-train risk coverage/finite guard failed")
    return scored, pd.concat(cdf_parts, ignore_index=True)


def stable_circular_offset(well_id: str, segment_count: int, key_prefix: str) -> int:
    if segment_count <= 1:
        raise ValueError("exp329 circular control requires at least two segments")
    digest = hashlib.sha256(f"{key_prefix}|{well_id}".encode()).hexdigest()
    return 1 + int(digest[:16], 16) % (segment_count - 1)


def add_circular_control(risk: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    output_parts: list[pd.DataFrame] = []
    prefix = str(get_nested(config, "stage_0_readout.negative_control.key_prefix"))
    expected_segments = int(get_nested(config, "exp226_geometry.k_segments"))
    for well, part in risk.groupby("well_id", sort=True):
        ordered = part.sort_values("segment_id", kind="mergesort").copy()
        if len(ordered) != expected_segments or not np.array_equal(
            ordered["segment_id"].to_numpy(np.int64), np.arange(expected_segments)
        ):
            raise ValueError(f"{well} does not have the fixed K16 segment inventory")
        offset = stable_circular_offset(str(well), len(ordered), prefix)
        ordered["control_risk_score"] = np.roll(ordered["risk_score"].to_numpy(np.float64), offset)
        ordered["control_offset_segments"] = offset
        if not np.array_equal(
            np.sort(ordered["risk_score"].to_numpy(np.float64)),
            np.sort(ordered["control_risk_score"].to_numpy(np.float64)),
        ):
            raise RuntimeError("circular control failed to preserve within-well risk values")
        output_parts.append(ordered)
    return (
        pd.concat(output_parts, ignore_index=True)
        .sort_values(["well_id", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )


# %% [markdown]
# ## 7. Target-free row contract and late-truth attachment


# %%
def build_target_free_rows(
    base: pd.DataFrame, row_layout: pd.DataFrame, segment_risk: pd.DataFrame
) -> pd.DataFrame:
    keys = ["well_id", "row_idx"]
    output = base.merge(row_layout, on=keys, how="left", validate="one_to_one")
    risk_columns = [
        "well_id",
        "readout_fold",
        "exp226_source_fold",
        "segment_id",
        "risk_score",
        "control_risk_score",
        "control_offset_segments",
    ]
    output = output.merge(
        segment_risk[risk_columns],
        on=[
            "well_id",
            "readout_fold",
            "exp226_source_fold",
            "segment_id",
        ],
        how="left",
        validate="many_to_one",
    )
    if output[["segment_id", "risk_score", "control_risk_score"]].isna().any().any():
        raise ValueError("target-free row/segment risk join coverage failed")
    output = output.sort_values(keys, kind="mergesort").reset_index(drop=True)
    assert_no_target_columns(output, stage="target-free row contract")
    return output


def build_target_free_contract(
    config: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    evidence = {}
    for name, artifact in artifacts.items():
        evidence[name] = {
            key: artifact.get(key)
            for key in (
                "rows",
                "raw_sha256",
                "decompressed_sha256",
                "schema_sha256",
                "content_sha256",
            )
            if artifact.get(key) is not None
        }
    contract = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage0_target_free_freeze",
        "route": "pf_beam",
        "truth_attached": False,
        "exp226_geometry": get_nested(config, "exp226_geometry"),
        "support_risk": get_nested(config, "support_risk"),
        "negative_control": get_nested(config, "stage_0_readout.negative_control"),
        "saved_formula": get_nested(config, "data.exp263_cache.expected_formula"),
        "artifact_evidence": evidence,
        "forbidden": get_nested(config, "forbidden"),
    }
    contract["target_free_contract_sha256"] = mapping_sha256(contract)
    return contract


def attach_truth_after_freeze(
    target_free_rows: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("truth attachment requires a frozen target-free contract SHA")
    joined = target_free_rows.merge(
        truth, on=["well_id", "row_idx"], how="left", validate="one_to_one"
    )
    if joined["tvt_true"].isna().any() or len(joined) != len(target_free_rows):
        raise ValueError("late truth join failed full row identity coverage")
    return joined.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 8. Segment AUC, benefit, scope, coverage, and fixed decision


# %%
def rmse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape or not len(truth):
        raise ValueError("RMSE inputs must be aligned and non-empty")
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    label = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    if label.shape != score.shape or not len(label) or not np.isfinite(score).all():
        raise ValueError("AUC requires aligned finite scores")
    positives = int(label.sum())
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[label].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def build_segment_benefit(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["well_id", "readout_fold", "exp226_source_fold", "segment_id"]
    for key, part in joined.groupby(keys, sort=True, observed=True):
        truth = part["tvt_true"].to_numpy(np.float64)
        base_rmse = rmse(truth, part["p_base"].to_numpy(np.float64))
        other_rmse = rmse(truth, part["p_other"].to_numpy(np.float64))
        risk_values = part["risk_score"].unique()
        control_values = part["control_risk_score"].unique()
        if len(risk_values) != 1 or len(control_values) != 1:
            raise ValueError("one segment must have one real/control risk value")
        benefit = base_rmse - other_rmse
        rows.append(
            {
                "well_id": str(key[0]),
                "readout_fold": int(key[1]),
                "exp226_source_fold": int(key[2]),
                "segment_id": int(key[3]),
                "rows": len(part),
                "md_since_min_ft": float(part["md_since_ft"].min()),
                "md_since_max_ft": float(part["md_since_ft"].max()),
                "risk_score": float(risk_values[0]),
                "control_risk_score": float(control_values[0]),
                "base_rmse": base_rmse,
                "destination_rmse": other_rmse,
                "segment_benefit_ft": benefit,
                "positive_benefit": benefit > 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_auc_metrics(segment_benefit: pd.DataFrame) -> pd.DataFrame:
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [("pooled", None, segment_benefit)]
    scopes.extend(
        (f"readout_fold_{int(fold)}", int(fold), part)
        for fold, part in segment_benefit.groupby("readout_fold", sort=True)
    )
    rows = []
    for scope, fold, part in scopes:
        labels = part["positive_benefit"].to_numpy(bool)
        for risk_kind, column in (
            ("real", "risk_score"),
            ("circular_control", "control_risk_score"),
        ):
            rows.append(
                {
                    "scope": scope,
                    "readout_fold": fold,
                    "risk_kind": risk_kind,
                    "segments": len(part),
                    "positive_segments": int(labels.sum()),
                    "positive_share": float(labels.mean()),
                    "auc": binary_auc(labels, part[column].to_numpy(np.float64)),
                }
            )
    return pd.DataFrame(rows)


def destination_metric(frame: pd.DataFrame, scope: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "scope": scope,
            "rows": 0,
            "wells": 0,
            "base_rmse": np.nan,
            "destination_rmse": np.nan,
            "destination_gain_ft": np.nan,
        }
    truth = frame["tvt_true"].to_numpy(np.float64)
    base_rmse = rmse(truth, frame["p_base"].to_numpy(np.float64))
    destination_rmse = rmse(truth, frame["p_other"].to_numpy(np.float64))
    return {
        "scope": scope,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "base_rmse": base_rmse,
        "destination_rmse": destination_rmse,
        "destination_gain_ft": base_rmse - destination_rmse,
    }


def build_scope_metrics(
    joined: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    top = float(get_nested(config, "stage_0_readout.top_risk_minimum"))
    top_rows = joined.loc[joined["risk_score"] >= top]
    long_min = float(get_nested(config, "validation.scopes.long_tail.minimum_md_since_ft"))
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("overall_top_risk", top_rows),
        ("long_tail_1000_plus_top_risk", top_rows.loc[top_rows["md_since_ft"] >= long_min]),
    ]
    roles = hidden_assignments.set_index("well_id")
    for scope_name, role_column in (
        get_nested(config, "data.hidden_like.role_columns") or {}
    ).items():
        valid_wells = set(roles.index[roles[str(role_column)].astype(str) == "valid"].astype(str))
        scopes.append(
            (
                f"{scope_name}_top_risk",
                top_rows.loc[top_rows["well_id"].astype(str).isin(valid_wells)],
            )
        )
    return pd.DataFrame([destination_metric(frame, name) for name, frame in scopes])


def evaluate_stage0_decision(
    joined: pd.DataFrame,
    support: pd.DataFrame,
    segment_benefit: pd.DataFrame,
    auc_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    input_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical = get_nested(config, "validation.technical_guards") or {}
    scientific = get_nested(config, "stage_0_readout.pass_requires_all") or {}
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_segments = expected_wells * int(get_nested(config, "exp226_geometry.k_segments"))
    activation_threshold = float(
        get_nested(config, "stage_0_readout.activation_risk_strictly_greater_than")
    )
    activated = joined["risk_score"].to_numpy(np.float64) > activation_threshold
    activated_fraction = float(activated.mean())
    activated_wells = int(joined.loc[activated, "well_id"].nunique())
    activated_folds = int(joined.loc[activated, "readout_fold"].nunique())
    hard_checks = {
        "expected_rows": len(joined) == expected_rows,
        "expected_wells": joined["well_id"].nunique() == expected_wells,
        "expected_readout_folds": sorted(int(v) for v in joined["readout_fold"].unique())
        == expected_folds,
        "expected_source_folds": sorted(int(v) for v in joined["exp226_source_fold"].unique())
        == expected_folds,
        "segment_feature_coverage": len(support) == expected_segments,
        "row_identity_coverage": float(input_audit["row_identity_coverage"])
        >= float(technical["required_row_identity_coverage"]),
        "well_identity_coverage": float(input_audit["well_identity_coverage"])
        >= float(technical["required_well_identity_coverage"]),
        "finite_support_coverage": float(
            np.isfinite(support[[item[0] for item in RISK_SPECS]].to_numpy()).mean()
        )
        >= float(technical["required_segment_feature_coverage"]),
        "finite_saved_prediction_coverage": float(
            np.isfinite(joined[["p_base", "p_other", "p226"]].to_numpy()).mean()
        )
        >= float(technical["required_finite_prediction_coverage"]),
        "exp263_formula_parity": float(input_audit["exp263_formula_parity_max_abs_ft"])
        <= float(technical["maximum_exp263_formula_parity_abs_ft"]),
    }
    coverage_checks = {
        "minimum_activated_row_fraction": activated_fraction
        >= float(technical["minimum_activated_row_fraction"]),
        "maximum_activated_row_fraction": activated_fraction
        <= float(technical["maximum_activated_row_fraction"]),
        "minimum_activated_wells": activated_wells >= int(technical["minimum_activated_wells"]),
        "minimum_activated_folds": activated_folds
        >= int(technical["minimum_folds_with_activation"]),
    }
    auc = auc_metrics.set_index(["scope", "risk_kind"])["auc"]
    pooled_real_auc = float(auc.loc[("pooled", "real")])
    pooled_control_auc = float(auc.loc[("pooled", "circular_control")])
    fold_real = auc_metrics.loc[
        (auc_metrics["risk_kind"] == "real") & (auc_metrics["scope"] != "pooled")
    ]
    folds_above_half = int((fold_real["auc"] > 0.5).sum())
    top_threshold = float(get_nested(config, "stage_0_readout.top_risk_minimum"))
    bottom_threshold = float(get_nested(config, "stage_0_readout.bottom_risk_maximum"))
    top_benefit = segment_benefit.loc[
        segment_benefit["risk_score"] >= top_threshold, "segment_benefit_ft"
    ]
    bottom_benefit = segment_benefit.loc[
        segment_benefit["risk_score"] <= bottom_threshold, "segment_benefit_ft"
    ]
    top_mean = float(top_benefit.mean())
    bottom_mean = float(bottom_benefit.mean())
    top_minus_bottom = top_mean - bottom_mean
    scope = scope_metrics.set_index("scope")
    science_checks = {
        "pooled_auc": pooled_real_auc >= float(scientific["minimum_pooled_auc"]),
        "folds_auc_above_half": folds_above_half
        >= int(scientific["minimum_folds_with_auc_above_half"]),
        "real_minus_control_auc": pooled_real_auc - pooled_control_auc
        >= float(scientific["minimum_real_minus_control_auc"]),
        "top_risk_mean_benefit": top_mean
        >= float(scientific["minimum_top_risk_decile_mean_benefit_ft"]),
        "top_minus_bottom_benefit": top_minus_bottom
        >= float(scientific["minimum_top_minus_bottom_risk_decile_benefit_ft"]),
        "long_tail_direction_non_regression": float(
            scope.loc["long_tail_1000_plus_top_risk", "destination_gain_ft"]
        )
        >= 0.0,
        "hidden_like_spatial_direction_non_regression": float(
            scope.loc["hidden_like_spatial_top_risk", "destination_gain_ft"]
        )
        >= 0.0,
        "hidden_like_typewell_purged_direction_non_regression": float(
            scope.loc["hidden_like_typewell_purged_top_risk", "destination_gain_ft"]
        )
        >= 0.0,
    }
    if not all(hard_checks.values()):
        decision = "FAIL_TECHNICAL"
    elif not all(coverage_checks.values()):
        decision = "INCONCLUSIVE_COVERAGE"
    elif all(science_checks.values()):
        decision = "PASS"
    else:
        decision = "FAIL"
    return {
        "decision": decision,
        "technical_hard_checks": hard_checks,
        "coverage_checks": coverage_checks,
        "scientific_checks": science_checks,
        "activated_rows": int(activated.sum()),
        "activated_row_fraction": activated_fraction,
        "activated_wells": activated_wells,
        "activated_folds": activated_folds,
        "pooled_real_auc": pooled_real_auc,
        "pooled_control_auc": pooled_control_auc,
        "real_minus_control_auc": pooled_real_auc - pooled_control_auc,
        "folds_auc_above_half": folds_above_half,
        "top_risk_segment_mean_benefit_ft": top_mean,
        "bottom_risk_segment_mean_benefit_ft": bottom_mean,
        "top_minus_bottom_segment_benefit_ft": top_minus_bottom,
        "stage1_available": False,
        "next_action": (
            "request separate Stage 1 implementation approval"
            if decision == "PASS"
            else "close branch without rescue grid"
        ),
    }


# %% [markdown]
# ## 9. Full Kaggle CPU orchestration and generated artifacts


# %%
def run_stage0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp329 Stage 0 must run on Kaggle; local execution requires explicit "
            "EXPERIMENT_ALLOW_LOCAL=1 approval."
        )
    validate_scientific_contract(config, require_kaggle_approval=True)
    started = time.time()
    base, exp263_manifest = load_exp263_base(config)
    base, exp226_path, exp226_manifest = load_exp226_safe(config, base)
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    expected_well_set = set(base["well_id"].astype(str).unique())
    raw_dir, horizontal_files = resolve_raw_train_dir(config, expected_well_set)
    raw_well_manifest, raw_manifest = validate_raw_well_identity(config, raw_dir, horizontal_files)

    support, donor_ledger, row_layout, donor_fold_audit = build_support_primitives(
        config, base, horizontal_files
    )
    segment_risk, cdf = fit_outer_train_risk(
        support, [int(value) for value in get_nested(config, "validation.expected_folds")]
    )
    segment_risk = add_circular_control(segment_risk, config)
    target_free_rows = build_target_free_rows(base, row_layout, segment_risk)

    artifacts = artifact_dir()
    raw_manifest_artifact = write_csv(
        raw_well_manifest, artifacts / f"{OUTPUT_PREFIX}_raw_well_manifest.csv"
    )
    input_manifest = pd.DataFrame(
        [
            {
                **exp263_manifest,
                "partition_evidence": json.dumps(exp263_manifest["partition_evidence"]),
            },
            exp226_manifest,
            hidden_manifest,
            raw_manifest,
            {
                "name": "exp226_source_fold_donor_audit",
                "path": "generated_in_memory_before_truth_freeze",
                "rows": len(donor_fold_audit),
                "wells": int(donor_fold_audit["validation_wells"].sum()),
                "content_sha256": dataframe_content_sha(donor_fold_audit),
            },
        ]
    )
    input_artifact = write_csv(input_manifest, artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv")
    donor_artifact = write_csv_gzip(
        donor_ledger, artifacts / f"{OUTPUT_PREFIX}_donor_ledger.csv.gz"
    )
    support_artifact = write_csv_gzip(
        support, artifacts / f"{OUTPUT_PREFIX}_support_primitives.csv.gz"
    )
    cdf_artifact = write_csv_gzip(cdf, artifacts / f"{OUTPUT_PREFIX}_outer_train_cdf.csv.gz")
    risk_artifact = write_csv_gzip(segment_risk, artifacts / f"{OUTPUT_PREFIX}_segment_risk.csv.gz")
    row_artifact = write_csv_gzip(
        target_free_rows, artifacts / f"{OUTPUT_PREFIX}_target_free_rows.csv.gz"
    )
    freeze_artifacts = {
        "input_manifest": input_artifact,
        "raw_well_manifest": raw_manifest_artifact,
        "donor_ledger": donor_artifact,
        "support_primitives": support_artifact,
        "outer_train_cdf": cdf_artifact,
        "segment_risk": risk_artifact,
        "target_free_rows": row_artifact,
    }
    contract = build_target_free_contract(config, freeze_artifacts)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_target_free_contract.json"
    write_json(contract_path, contract)

    # First evaluation-truth read: every target-free value and its content SHA
    # has already been persisted above.
    truth = load_exp226_truth(
        exp226_path,
        target_free_contract_sha256=str(contract["target_free_contract_sha256"]),
    )
    joined = attach_truth_after_freeze(
        target_free_rows,
        truth,
        target_free_contract_sha256=str(contract["target_free_contract_sha256"]),
    )
    segment_benefit = build_segment_benefit(joined)
    auc_metrics = build_auc_metrics(segment_benefit)
    scope_metrics = build_scope_metrics(joined, hidden_assignments, config)
    decision = evaluate_stage0_decision(
        joined,
        support,
        segment_benefit,
        auc_metrics,
        scope_metrics,
        {
            "row_identity_coverage": 1.0,
            "well_identity_coverage": 1.0,
            "exp263_formula_parity_max_abs_ft": exp263_manifest["formula_parity_max_abs_ft"],
        },
        config,
    )
    segment_artifact = write_csv(
        segment_benefit, artifacts / f"{OUTPUT_PREFIX}_segment_benefit.csv"
    )
    auc_artifact = write_csv(auc_metrics, artifacts / f"{OUTPUT_PREFIX}_auc_metrics.csv")
    scope_artifact = write_csv(scope_metrics, artifacts / f"{OUTPUT_PREFIX}_scope_metrics.csv")
    decision_path = artifacts / f"{OUTPUT_PREFIX}_stage0_decision.json"
    write_json(decision_path, decision)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_readout_completed",
        "route": "pf_beam",
        "runtime_seconds": time.time() - started,
        "rows": len(joined),
        "wells": int(joined["well_id"].nunique()),
        "segments": len(segment_benefit),
        "execution": {
            "scientific_risk_scores": 1,
            "diagnostic_controls": 1,
            "support_well_runs": int(joined["well_id"].nunique()),
            "prediction_candidates": 0,
            "models": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_pf_beam_runs": 0,
            "parent_prediction_regeneration": False,
        },
        "decision": decision,
        "truth_attachment": {
            "stage": "after_support_ecdf_risk_control_saved_prediction_freeze",
            "target_free_contract_sha256": contract["target_free_contract_sha256"],
        },
        "artifacts": {
            **freeze_artifacts,
            "target_free_contract": {
                "path": str(contract_path),
                "raw_sha256": sha256_path(contract_path),
            },
            "segment_benefit": segment_artifact,
            "auc_metrics": auc_artifact,
            "scope_metrics": scope_artifact,
            "stage0_decision": {
                "path": str(decision_path),
                "raw_sha256": sha256_path(decision_path),
            },
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": decision["next_action"],
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage0_readout_completed",
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "segment_auc_and_rmse_benefit",
        "decision": decision["decision"],
        "diagnostic": decision,
        "target_free_contract_sha256": contract["target_free_contract_sha256"],
        "notes": "Stage 0 only; no prediction change, model, inference, or submission.",
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup, contract preview, and guarded execution

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG, require_kaggle_approval=False)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "anchor": get_nested(CONFIG, "lineage.anchor"),
                "implementation_scope": get_nested(CONFIG, "implementation.scope"),
                "stage0_enabled": get_nested(
                    CONFIG, "stage_0_readout.enabled_after_implementation"
                ),
                "stage1_enabled": get_nested(
                    CONFIG, "stage_1_bounded_shrink.enabled_after_implementation"
                ),
                "support_well_runs": get_nested(
                    CONFIG, "execution_contract.stage_0.support_well_runs"
                ),
                "scientific_risk_scores": get_nested(
                    CONFIG, "execution_contract.stage_0.scientific_risk_scores"
                ),
                "diagnostic_controls": get_nested(
                    CONFIG, "execution_contract.stage_0.diagnostic_controls"
                ),
                "models": get_nested(CONFIG, "execution_contract.stage_0.model_configs"),
                "trained_folds": get_nested(CONFIG, "execution_contract.stage_0.trained_folds"),
                "boosters": get_nested(CONFIG, "execution_contract.stage_0.boosters"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "run_stage0": get_nested(CONFIG, "execution.run_stage_0"),
                "run_stage1": get_nested(CONFIG, "execution.run_stage_1"),
                "inference": get_nested(CONFIG, "execution.run_inference"),
                "submission": get_nested(CONFIG, "execution.create_submission"),
            },
            indent=2,
        )
    )


# %%
if EXECUTE_NOTEBOOK and bool(get_nested(CONFIG or {}, "execution.run_stage_0")):
    assert CONFIG is not None
    EXP329_STAGE0_SUMMARY = run_stage0_experiment(CONFIG)
elif EXECUTE_NOTEBOOK:
    print("exp329 Stage 0 execution is disabled; implementation preview only.")
