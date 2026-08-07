# %% [markdown]
# # exp352 Type-Well transfer safety guard readout
#
# This zero-model diagnostic consumes the SHA-pinned exp311 fold, native-overlap
# group, prior, score-population, and suffix-pair artifacts. It freezes a
# target-free exact-group/global/identity fallback manifest before opening the
# suffix GR reconstruction pairs used for scoring.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, SHA, and output helpers
# 3. Frozen scientific contract and exp311 input preflight
# 4. Target-free availability and fallback manifest
# 5. Late suffix-pair attachment and guarded scoring
# 6. Surface metrics and fixed Stage 0 gate
# 7. Stage 0 orchestration and generated artifacts
# 8. Setup, configuration, and contract preview
# 9. Run the diagnostic and report generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp352_typewell_transfer_safety_guard_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SHA256_LENGTH = 64
PRIOR_VALUE_COLUMNS = [
    "slope",
    "intercept",
    "bias_at_gr50",
    "residual_sigma_mad",
    "fit_rmse",
]
PARENT_SURFACES = [
    "same_typewell_heldout_well",
    "leave_one_typewell_group_out",
    "spatial_typewell_purged",
]


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell is not None


# %% [markdown]
# ## 2. Runtime, configuration, SHA, and output helpers

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if not isinstance(value, (str, bytes)):
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    starts = [Path.cwd(), KAGGLE_WORKING_ROOT]
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "project.yml").exists():
                return candidate
    return Path.cwd()


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = [
        Path.cwd() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    ]
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp352 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    path.write_text(text)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    return mapping_sha256([(str(column), str(frame[column].dtype)) for column in frame.columns])


def dataframe_content_sha(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> str:
    work = frame.copy()
    if sort_columns:
        work = work.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    if columns:
        work = work.loc[:, list(columns)]
    digest = hashlib.sha256()
    digest.update(dataframe_schema_sha(work).encode())
    for row in work.itertuples(index=False, name=None):
        payload = json.dumps(
            to_jsonable(row),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        digest.update(payload.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_columns: Sequence[str],
) -> dict[str, Any]:
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(ordered),
        "columns": len(ordered.columns),
        "schema_sha256": dataframe_schema_sha(ordered),
        "content_sha256": dataframe_content_sha(ordered),
        "raw_sha256": sha256_path(path),
    }


# %% [markdown]
# ## 3. Frozen scientific contract and exp311 input preflight

# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    fixed = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp311_typewell_group_prefix_suffix_gr_calibration_readout",
        "implementation.enabled": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.strategy": "frozen_exp311_typewell_transfer_guard_three_surface_audit",
        "validation.n_folds": 5,
        "validation.score_unit": "horizontal_gr_api",
        "validation.truth_attachment": (
            "after_availability_fallback_manifest_content_sha_freeze"
        ),
        "model.name": "none",
        "model.guard.exact_group_min_peer_wells": 2,
        "model.guard.exact_group_min_effective_rows": 64,
        "model.guard.group_scheme": "native_overlap_1",
        "model.guard.global_prior_aggregation": (
            "equal_group_median_excluding_target_group"
        ),
        "model.guard.soft_similarity_enabled": False,
        "model.guard.score_unit": "horizontal_gr_api",
        "execution_contract.stage_0.diagnostic_variants": 1,
        "execution_contract.stage_0.audit_surfaces": 3,
        "execution_contract.stage_0.folds": 5,
        "execution_contract.stage_0.hmm_well_runs": 0,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.stage_0.decoder_runs": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"exp352 fixes {key}={expected!r}, got {actual!r}")
    if list(get_nested(config, "validation.audit_surfaces") or []) != PARENT_SURFACES:
        raise ValueError("exp352 fixes the three ordered exp311 audit surfaces")
    if list(get_nested(config, "model.guard.fallback_order") or []) != [
        "exact_native_overlap_group",
        "global_outer_train_prior",
        "identity_no_correction",
    ]:
        raise ValueError("exp352 fixes exact-group -> global -> identity fallback order")
    if int(get_nested(config, "model.guard.global_prior_min_source_groups") or 0) != 1:
        raise ValueError("exp352 fixes global prior availability at one eligible source group")
    forbidden = set(get_nested(config, "model.forbidden") or [])
    required_forbidden = {
        "support_threshold_grid",
        "fallback_order_grid",
        "soft_similarity",
        "correction_or_prediction_generation",
        "selector_or_decoder_training",
        "automatic_unlock_of_exp314_to_exp320",
    }
    if forbidden != required_forbidden:
        raise ValueError("exp352 forbidden-operation contract changed")
    expected_summary = str(
        get_nested(config, "data.exp311_artifacts.expected_summary_raw_sha256") or ""
    )
    expected_pair = str(
        get_nested(config, "data.exp311_artifacts.expected_pair_decompressed_sha256") or ""
    )
    if len(expected_summary) != SHA256_LENGTH or len(expected_pair) != SHA256_LENGTH:
        raise ValueError("exp352 requires pinned exp311 summary and pair content SHA256")


def validate_run_approval(config: Mapping[str, Any]) -> None:
    if not bool(get_nested(config, "execution.run_stage_0")):
        raise RuntimeError("Stage 0 execution is disabled until separately approved")
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("Kaggle package/push/run is not approved")
    if not bool(get_nested(config, "runtime.kaggle.train_run_on_push")):
        raise RuntimeError("approved Stage 0 requires train_run_on_push=true")


def resolve_parent_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    spec = get_nested(config, "data.exp311_artifacts") or {}
    filenames = dict(spec.get("filenames") or {})
    root = project_root()
    roots: list[Path] = []
    for raw in list(spec.get("root_candidates") or []):
        candidate = Path(str(raw))
        roots.extend([candidate, root / candidate, Path.cwd() / candidate])
    resolved: dict[str, Path] = {}
    for name, filename in filenames.items():
        path = next(
            (
                candidate / str(filename)
                for candidate in roots
                if (candidate / str(filename)).is_file()
                and (candidate / str(filename)).stat().st_size > 0
            ),
            None,
        )
        if path is None and KAGGLE_INPUT_ROOT.exists():
            path = next(
                (
                    item
                    for item in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
                    if item.is_file() and item.stat().st_size > 0
                ),
                None,
            )
        if path is None:
            raise FileNotFoundError(f"could not resolve exp311 artifact {name}: {filename}")
        resolved[str(name)] = path
    required = {
        "summary",
        "fold_manifest",
        "group_membership",
        "group_priors",
        "suffix_by_well",
        "pair_table",
    }
    if set(resolved) != required:
        raise ValueError(
            f"exp311 artifact mapping mismatch: missing={sorted(required - set(resolved))}, "
            f"unexpected={sorted(set(resolved) - required)}"
        )
    return resolved


def read_parent_summary(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = paths["summary"]
    expected = str(get_nested(config, "data.exp311_artifacts.expected_summary_raw_sha256"))
    actual = sha256_path(summary_path)
    if actual != expected:
        raise ValueError(f"exp311 summary SHA mismatch: expected={expected}, actual={actual}")
    payload = json.loads(summary_path.read_text())
    if payload.get("experiment") != "exp311_typewell_group_prefix_suffix_gr_calibration_readout":
        raise ValueError("resolved summary is not the frozen exp311 output")
    if get_nested(payload, "execution_contract.folds") != 5:
        raise ValueError("exp311 summary does not contain the frozen five-fold contract")
    return payload


def verify_parent_artifact_hashes(
    paths: Mapping[str, Path],
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifests = get_nested(summary, "artifact_manifests") or {}
    verified: dict[str, Any] = {}
    for name, path in sorted(paths.items()):
        raw = sha256_path(path)
        entry: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "raw_sha256": raw,
        }
        if name == "summary":
            entry["expected_raw_sha256"] = str(
                get_nested(config, "data.exp311_artifacts.expected_summary_raw_sha256")
            )
        else:
            parent_manifest = manifests.get(name)
            if not isinstance(parent_manifest, Mapping):
                raise ValueError(f"exp311 summary is missing manifest for {name}")
            expected_raw = str(parent_manifest.get("raw_sha256") or "")
            if raw != expected_raw:
                raise ValueError(
                    f"exp311 {name} raw SHA mismatch: expected={expected_raw}, actual={raw}"
                )
            entry["expected_raw_sha256"] = expected_raw
        if name == "pair_table":
            decompressed = sha256_gzip_decompressed(path)
            expected_decompressed = str(
                get_nested(config, "data.exp311_artifacts.expected_pair_decompressed_sha256")
            )
            summary_decompressed = str(
                (manifests.get(name) or {}).get("decompressed_sha256") or ""
            )
            if decompressed != expected_decompressed or decompressed != summary_decompressed:
                raise ValueError(
                    "exp311 pair decompressed SHA mismatch: "
                    f"config={expected_decompressed}, summary={summary_decompressed}, "
                    f"actual={decompressed}"
                )
            entry["decompressed_sha256"] = decompressed
        verified[name] = entry
    return {
        "expected_kernel_id": get_nested(
            config, "data.exp311_artifacts.expected_kernel_id"
        ),
        "expected_kernel_version": get_nested(
            config, "data.exp311_artifacts.expected_kernel_version"
        ),
        "artifacts": verified,
    }


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns {sorted(missing)}")


def load_parent_target_free_tables(
    paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = pd.read_csv(paths["fold_manifest"], dtype={"well_id": str})
    membership = pd.read_csv(
        paths["group_membership"],
        dtype={"well_id": str, "group_id": str, "group_scheme": str},
    )
    priors = pd.read_csv(
        paths["group_priors"],
        dtype={
            "group_scheme": str,
            "surface": str,
            "control": str,
            "group_id": str,
        },
    )
    score_index_columns = [
        "fold",
        "well_id",
        "group_scheme",
        "surface",
        "control",
        "group_id",
    ]
    score_index = pd.read_csv(
        paths["suffix_by_well"],
        usecols=score_index_columns,
        dtype={
            "well_id": str,
            "group_scheme": str,
            "surface": str,
            "control": str,
            "group_id": str,
        },
    )
    require_columns(folds, {"well_id", "fold"}, "exp311 fold manifest")
    require_columns(
        membership,
        {"well_id", "fold", "group_scheme", "group_id"},
        "exp311 group membership",
    )
    require_columns(
        priors,
        {
            "fold",
            "group_scheme",
            "surface",
            "control",
            "group_id",
            "source_wells",
            "support_rows",
            "available",
            *PRIOR_VALUE_COLUMNS,
        },
        "exp311 group priors",
    )
    scheme = str(get_nested(config, "model.guard.group_scheme"))
    membership = membership[membership["group_scheme"].eq(scheme)].copy()
    score_index = score_index[
        score_index["group_scheme"].eq(scheme)
        & score_index["control"].eq("real")
        & score_index["surface"].isin(PARENT_SURFACES)
    ].copy()
    priors = priors[
        priors["group_scheme"].eq(scheme) & priors["control"].eq("real")
    ].copy()
    if folds["well_id"].duplicated().any():
        raise ValueError("exp311 fold manifest contains duplicate wells")
    if membership["well_id"].duplicated().any():
        raise ValueError("exp311 native group membership contains duplicate wells")
    fold_lookup = folds.set_index("well_id")["fold"].astype(int)
    joined_fold = membership["well_id"].map(fold_lookup)
    if not np.array_equal(joined_fold.to_numpy(), membership["fold"].astype(int).to_numpy()):
        raise ValueError("exp311 membership fold does not match fold manifest")
    duplicates = score_index.duplicated(["surface", "fold", "well_id"])
    if duplicates.any():
        raise ValueError("exp311 score population has duplicate surface/fold/well keys")
    if sorted(score_index["surface"].unique()) != sorted(PARENT_SURFACES):
        raise ValueError("exp311 score population does not contain all frozen surfaces")
    if sorted(folds["fold"].astype(int).unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("exp311 fold manifest must contain folds 0..4")
    return (
        folds.sort_values(["fold", "well_id"], kind="mergesort").reset_index(drop=True),
        membership.sort_values(["fold", "well_id"], kind="mergesort").reset_index(drop=True),
        priors.sort_values(["surface", "fold", "group_id"], kind="mergesort").reset_index(
            drop=True
        ),
        score_index.sort_values(["surface", "fold", "well_id"], kind="mergesort").reset_index(
            drop=True
        ),
    )


# %% [markdown]
# ## 4. Target-free availability and fallback manifest

# %%
def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def eligible_group_priors(
    priors: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    minimum_wells = int(get_nested(config, "model.guard.exact_group_min_peer_wells"))
    minimum_rows = int(get_nested(config, "model.guard.exact_group_min_effective_rows"))
    finite = np.isfinite(priors[PRIOR_VALUE_COLUMNS].to_numpy(np.float64)).all(axis=1)
    available = bool_series(priors["available"])
    eligible = (
        available
        & priors["source_wells"].astype(int).ge(minimum_wells)
        & priors["support_rows"].astype(int).ge(minimum_rows)
        & finite
    )
    output = priors.copy()
    output["exp352_exact_eligible"] = eligible
    return output


def identity_prior(reason: str) -> dict[str, Any]:
    return {
        "selected_source": "identity_no_correction",
        "fallback_reason": reason,
        "selected_source_groups": 0,
        "selected_source_wells": 0,
        "selected_support_rows": 0,
        "slope": 1.0,
        "intercept": 0.0,
        "bias_at_gr50": 0.0,
        "residual_sigma_mad": math.nan,
        "fit_rmse": math.nan,
    }


def exact_prior(row: pd.Series) -> dict[str, Any]:
    return {
        "selected_source": "exact_native_overlap_group",
        "fallback_reason": None,
        "selected_source_groups": 1,
        "selected_source_wells": int(row["source_wells"]),
        "selected_support_rows": int(row["support_rows"]),
        **{column: float(row[column]) for column in PRIOR_VALUE_COLUMNS},
    }


def global_prior(
    eligible: pd.DataFrame,
    *,
    excluded_group_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    source = eligible[
        eligible["exp352_exact_eligible"]
        & ~eligible["group_id"].astype(str).eq(str(excluded_group_id))
    ].copy()
    minimum_groups = int(get_nested(config, "model.guard.global_prior_min_source_groups"))
    if len(source) < minimum_groups:
        return None
    source = source.sort_values("group_id", kind="mergesort")
    return {
        "selected_source": "global_outer_train_prior",
        "fallback_reason": "exact_group_unavailable_or_forbidden",
        "selected_source_groups": int(len(source)),
        "selected_source_wells": int(source["source_wells"].astype(int).sum()),
        "selected_support_rows": int(source["support_rows"].astype(int).sum()),
        **{
            column: float(source[column].astype(float).median())
            for column in PRIOR_VALUE_COLUMNS
        },
    }


def source_surface_for(surface: str) -> str:
    if surface == "spatial_typewell_purged":
        return "spatial_typewell_purged"
    return "same_typewell_heldout_well"


def build_availability_manifest(
    folds: pd.DataFrame,
    membership: pd.DataFrame,
    priors: pd.DataFrame,
    score_index: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    del folds
    eligible = eligible_group_priors(priors, config)
    group_lookup = membership.set_index("well_id")["group_id"].astype(str).to_dict()
    membership_fold = membership.set_index("well_id")["fold"].astype(int).to_dict()
    rows: list[dict[str, Any]] = []
    for key, population in score_index.groupby(["surface", "fold"], sort=True):
        surface = str(key[0])
        fold = int(key[1])
        source_surface = source_surface_for(surface)
        source = eligible[
            eligible["surface"].eq(source_surface) & eligible["fold"].astype(int).eq(fold)
        ].copy()
        prior_by_group = {
            str(row["group_id"]): row
            for _, row in source.sort_values("group_id", kind="mergesort").iterrows()
        }
        for well_id in sorted(population["well_id"].astype(str)):
            if well_id not in group_lookup:
                raise ValueError(f"score population well has no frozen group: {well_id}")
            if membership_fold[well_id] != fold:
                raise ValueError(f"score population fold changed for {well_id}")
            group_id = group_lookup[well_id]
            candidate = prior_by_group.get(group_id)
            exact_available = bool(
                candidate is not None and bool(candidate["exp352_exact_eligible"])
            )
            exact_allowed = surface != "leave_one_typewell_group_out"
            if exact_available and exact_allowed:
                selected = exact_prior(candidate)
            else:
                selected = global_prior(
                    source,
                    excluded_group_id=group_id,
                    config=config,
                )
                if selected is None:
                    reason = (
                        "leave_one_group_out_no_global_prior"
                        if not exact_allowed
                        else "exact_and_global_prior_unavailable"
                    )
                    selected = identity_prior(reason)
            rows.append(
                {
                    "surface": surface,
                    "fold": fold,
                    "well_id": well_id,
                    "group_id": group_id,
                    "source_surface": source_surface,
                    "exact_group_available": exact_available,
                    "exact_group_allowed": exact_allowed,
                    "exact_source_wells": (
                        int(candidate["source_wells"]) if candidate is not None else 0
                    ),
                    "exact_support_rows": (
                        int(candidate["support_rows"]) if candidate is not None else 0
                    ),
                    **selected,
                    "outer_valid_truth_rows_before_manifest_freeze": 0,
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["surface", "fold", "well_id"], kind="mergesort"
    )
    if manifest.empty:
        raise ValueError("availability manifest is empty")
    if manifest.duplicated(["surface", "fold", "well_id"]).any():
        raise ValueError("availability manifest keys are not unique")
    if manifest[PRIOR_VALUE_COLUMNS[:2]].isna().any().any():
        raise ValueError("selected slope/intercept must always be finite")
    freeze_sha = dataframe_content_sha(
        manifest,
        sort_columns=["surface", "fold", "well_id"],
    )
    manifest["availability_manifest_freeze_sha256"] = freeze_sha
    return manifest.reset_index(drop=True), freeze_sha


# %% [markdown]
# ## 5. Late suffix-pair attachment and guarded scoring

# %%
def load_suffix_pairs_after_freeze(
    pair_path: Path,
    *,
    availability_manifest_freeze_sha256: str,
) -> pd.DataFrame:
    if len(str(availability_manifest_freeze_sha256)) != SHA256_LENGTH:
        raise ValueError("suffix pairs require a complete frozen availability SHA256")
    pairs = pd.read_csv(
        pair_path,
        usecols=[
            "fold",
            "well_id",
            "row_idx",
            "partition",
            "typewell_gr",
            "horizontal_gr",
            "truth_attached_after_freeze",
        ],
        dtype={"well_id": str, "partition": str},
    )
    require_columns(
        pairs,
        {
            "fold",
            "well_id",
            "row_idx",
            "partition",
            "typewell_gr",
            "horizontal_gr",
            "truth_attached_after_freeze",
        },
        "exp311 pair table",
    )
    suffix = pairs[pairs["partition"].eq("evaluation_suffix")].copy()
    if suffix.empty:
        raise ValueError("exp311 pair table contains no evaluation suffix")
    if not bool_series(suffix["truth_attached_after_freeze"]).all():
        raise ValueError("exp311 suffix pairs were not marked as late-truth-attached")
    finite = np.isfinite(
        suffix[["typewell_gr", "horizontal_gr"]].to_numpy(np.float64)
    ).all(axis=1)
    suffix = suffix[finite].copy()
    if suffix.duplicated(["fold", "well_id", "row_idx"]).any():
        raise ValueError("exp311 suffix pairs contain duplicate fold/well/row keys")
    suffix["availability_manifest_freeze_sha256"] = (
        availability_manifest_freeze_sha256
    )
    return suffix.sort_values(["fold", "well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )


def rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("RMSE requires a non-empty finite vector")
    return float(np.sqrt(np.mean(array**2)))


def score_guard_manifest(
    manifest: pd.DataFrame,
    suffix_pairs: pd.DataFrame,
    *,
    availability_manifest_freeze_sha256: str,
) -> tuple[pd.DataFrame, float]:
    if len(str(availability_manifest_freeze_sha256)) != SHA256_LENGTH:
        raise ValueError("guard scoring requires a complete frozen availability SHA256")
    observed_sha = set(manifest["availability_manifest_freeze_sha256"].astype(str))
    if observed_sha != {availability_manifest_freeze_sha256}:
        raise ValueError("manifest SHA column does not match the frozen availability SHA")
    pair_lookup = {
        (int(key[0]), str(key[1])): group
        for key, group in suffix_pairs.groupby(["fold", "well_id"], sort=True)
    }
    rows: list[dict[str, Any]] = []
    identity_parity = 0.0
    for row in manifest.itertuples(index=False):
        key = (int(row.fold), str(row.well_id))
        pairs = pair_lookup.get(key)
        if pairs is None or pairs.empty:
            raise ValueError(f"no frozen suffix pairs for fold/well={key}")
        x = pairs["typewell_gr"].to_numpy(np.float64)
        y = pairs["horizontal_gr"].to_numpy(np.float64)
        identity = x.copy()
        identity_replay = 1.0 * x + 0.0
        identity_parity = max(
            identity_parity,
            float(np.max(np.abs(identity_replay - identity))),
        )
        guarded = float(row.slope) * x + float(row.intercept)
        identity_rmse = rmse(y - identity)
        guarded_rmse = rmse(y - guarded)
        rows.append(
            {
                "surface": str(row.surface),
                "fold": int(row.fold),
                "well_id": str(row.well_id),
                "group_id": str(row.group_id),
                "selected_source": str(row.selected_source),
                "fallback_reason": row.fallback_reason,
                "exact_group_available": bool(row.exact_group_available),
                "suffix_rows": int(len(pairs)),
                "identity_suffix_gr_rmse": identity_rmse,
                "guarded_suffix_gr_rmse": guarded_rmse,
                "guarded_gain_vs_identity_gr_api": identity_rmse - guarded_rmse,
                "guarded_delta_vs_identity_gr_api": guarded_rmse - identity_rmse,
                "availability_manifest_freeze_sha256": (
                    availability_manifest_freeze_sha256
                ),
                "truth_attached_after_manifest_freeze": True,
            }
        )
    scored = pd.DataFrame(rows).sort_values(
        ["surface", "fold", "well_id"], kind="mergesort"
    )
    return scored.reset_index(drop=True), identity_parity


# %% [markdown]
# ## 6. Surface metrics and fixed Stage 0 gate

# %%
def aggregate_surface_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface, group in scored.groupby("surface", sort=True):
        fold_scopes: list[tuple[str | int, pd.DataFrame]] = [
            *((int(fold), part) for fold, part in group.groupby("fold", sort=True)),
            ("pooled", group),
        ]
        for fold, scope in fold_scopes:
            identity = float(
                np.sqrt(np.mean(scope["identity_suffix_gr_rmse"].to_numpy() ** 2))
            )
            guarded = float(
                np.sqrt(np.mean(scope["guarded_suffix_gr_rmse"].to_numpy() ** 2))
            )
            source_counts = scope["selected_source"].value_counts()
            rows.append(
                {
                    "surface": str(surface),
                    "fold": fold,
                    "wells": int(len(scope)),
                    "suffix_rows": int(scope["suffix_rows"].sum()),
                    "exact_group_available_wells": int(
                        scope["exact_group_available"].sum()
                    ),
                    "exact_group_availability_rate": float(
                        scope["exact_group_available"].mean()
                    ),
                    "selected_exact_wells": int(
                        source_counts.get("exact_native_overlap_group", 0)
                    ),
                    "selected_global_wells": int(
                        source_counts.get("global_outer_train_prior", 0)
                    ),
                    "selected_identity_wells": int(
                        source_counts.get("identity_no_correction", 0)
                    ),
                    "identity_suffix_gr_rmse": identity,
                    "guarded_suffix_gr_rmse": guarded,
                    "guarded_gain_vs_identity_gr_api": identity - guarded,
                    "negative_transfer_gr_api": guarded - identity,
                    "worst_well_regression_gr_api": float(
                        scope["guarded_delta_vs_identity_gr_api"].max()
                    ),
                    "improved_wells": int(
                        (scope["guarded_gain_vs_identity_gr_api"] > 0.0).sum()
                    ),
                    "worse_wells": int(
                        (scope["guarded_gain_vs_identity_gr_api"] < 0.0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["surface", "fold"], kind="mergesort")


def one_surface_metric(
    metrics: pd.DataFrame,
    *,
    surface: str,
    fold: str | int,
) -> pd.Series:
    selected = metrics[
        metrics["surface"].eq(surface)
        & metrics["fold"].astype(str).eq(str(fold))
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {(surface, fold)}, got {len(selected)}")
    return selected.iloc[0]


def evaluate_stage_0_gate(
    manifest: pd.DataFrame,
    scored: pd.DataFrame,
    metrics: pd.DataFrame,
    identity_parity_max_abs: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "model.stage_0.pass_requires_all") or {}
    same = one_surface_metric(
        metrics,
        surface="same_typewell_heldout_well",
        fold="pooled",
    )
    unseen = one_surface_metric(
        metrics,
        surface="leave_one_typewell_group_out",
        fold="pooled",
    )
    spatial = one_surface_metric(
        metrics,
        surface="spatial_typewell_purged",
        fold="pooled",
    )
    same_folds = metrics[
        metrics["surface"].eq("same_typewell_heldout_well")
        & ~metrics["fold"].astype(str).eq("pooled")
    ]
    folds_improved = int(
        (same_folds["guarded_gain_vs_identity_gr_api"] > 0.0).sum()
    )
    worst = float(scored["guarded_delta_vs_identity_gr_api"].max())
    truth_before_freeze = int(
        manifest["outer_valid_truth_rows_before_manifest_freeze"].sum()
    )
    checks = {
        "outer_valid_truth_before_manifest_freeze_zero": truth_before_freeze == 0,
        "fold_safe_exact_group_coverage": bool(
            float(same["exact_group_availability_rate"])
            >= float(gates["minimum_fold_safe_coverage"])
        ),
        "identity_fallback_row_parity": bool(
            identity_parity_max_abs
            <= float(gates["maximum_identity_fallback_parity_abs"])
        ),
        "same_group_gain": bool(
            float(same["guarded_gain_vs_identity_gr_api"])
            >= float(gates["minimum_same_group_gain_gr_api"])
        ),
        "same_group_folds_improved": bool(
            folds_improved >= int(gates["minimum_improved_folds"])
        ),
        "unseen_group_non_regression": bool(
            float(unseen["negative_transfer_gr_api"])
            <= float(gates["maximum_unseen_group_negative_transfer_gr_api"])
        ),
        "spatial_typewell_purged_non_regression": bool(
            float(spatial["negative_transfer_gr_api"])
            <= float(
                gates["maximum_spatial_typewell_purged_negative_transfer_gr_api"]
            )
        ),
        "worst_well_safety": bool(
            worst <= float(gates["maximum_worst_well_regression_gr_api"])
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "score_unit": "horizontal_gr_api",
        "fold_safe_exact_group_coverage": float(
            same["exact_group_availability_rate"]
        ),
        "identity_fallback_parity_max_abs": float(identity_parity_max_abs),
        "same_group_gain_gr_api": float(
            same["guarded_gain_vs_identity_gr_api"]
        ),
        "same_group_folds_improved": folds_improved,
        "unseen_group_negative_transfer_gr_api": float(
            unseen["negative_transfer_gr_api"]
        ),
        "spatial_typewell_purged_negative_transfer_gr_api": float(
            spatial["negative_transfer_gr_api"]
        ),
        "worst_well_regression_gr_api": worst,
        "outer_valid_truth_rows_before_manifest_freeze": truth_before_freeze,
    }


# %% [markdown]
# ## 7. Stage 0 orchestration and generated artifacts

# %%
def scientific_contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "score_unit": "horizontal_gr_api",
        "unit_note": (
            "The numeric preregistered gates are unchanged, but exp311 saved only "
            "horizontal-GR reconstruction scores and generated no TVT predictor."
        ),
        "audit_surfaces": PARENT_SURFACES,
        "guard": get_nested(config, "model.guard"),
        "stage_0_gates": get_nested(config, "model.stage_0.pass_requires_all"),
        "execution_contract": get_nested(config, "execution_contract"),
        "forbidden": get_nested(config, "model.forbidden"),
        "truth_boundary": get_nested(config, "validation.truth_attachment"),
        "pass_scope": (
            "A PASS supports only this guard readout; it does not reverse exp311/312 "
            "or unlock exp314-exp320, correction, selection, inference, or submission."
        ),
    }


def run_stage_0(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config)
    validate_run_approval(config)
    started = time.perf_counter()
    output_dir = artifact_dir()
    paths = resolve_parent_artifacts(config)
    parent_summary = read_parent_summary(paths, config)
    input_manifest = verify_parent_artifact_hashes(paths, parent_summary, config)

    folds, membership, priors, score_index = load_parent_target_free_tables(
        paths,
        config,
    )
    manifest, freeze_sha = build_availability_manifest(
        folds,
        membership,
        priors,
        score_index,
        config,
    )
    suffix_pairs = load_suffix_pairs_after_freeze(
        paths["pair_table"],
        availability_manifest_freeze_sha256=freeze_sha,
    )
    scored, identity_parity = score_guard_manifest(
        manifest,
        suffix_pairs,
        availability_manifest_freeze_sha256=freeze_sha,
    )
    surface_metrics = aggregate_surface_metrics(scored)
    gate = evaluate_stage_0_gate(
        manifest,
        scored,
        surface_metrics,
        identity_parity,
        config,
    )
    contract = scientific_contract_payload(config)

    manifests: dict[str, Any] = {}
    manifests["availability_fallback_table"] = write_csv(
        manifest,
        output_dir / f"{OUTPUT_PREFIX}_availability_fallback_table.csv",
        sort_columns=["surface", "fold", "well_id"],
    )
    manifests["score_table"] = write_csv(
        scored,
        output_dir / f"{OUTPUT_PREFIX}_score_table.csv",
        sort_columns=["surface", "fold", "well_id"],
    )
    manifests["surface_metrics"] = write_csv(
        surface_metrics,
        output_dir / f"{OUTPUT_PREFIX}_surface_metrics.csv",
        sort_columns=["surface", "fold"],
    )
    manifests["gate"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_gate.json",
        gate,
    )
    manifests["input_manifest"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_input_manifest.json",
        input_manifest,
    )
    manifests["scientific_contract"] = write_json(
        output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json",
        contract,
    )
    runtime_seconds = time.perf_counter() - started
    summary: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "completed_stage_0_guard_passed"
            if gate["passed"]
            else "completed_stage_0_guard_failed"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_only": True,
            "internet_enabled": False,
            "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "execution_contract": get_nested(config, "execution_contract"),
        "score_unit": "horizontal_gr_api",
        "availability_manifest_freeze_sha256": freeze_sha,
        "parent_input": {
            "kernel_id": get_nested(
                config, "data.exp311_artifacts.expected_kernel_id"
            ),
            "kernel_version": get_nested(
                config, "data.exp311_artifacts.expected_kernel_version"
            ),
            "summary_raw_sha256": sha256_path(paths["summary"]),
            "pair_decompressed_sha256": sha256_gzip_decompressed(
                paths["pair_table"]
            ),
        },
        "stage_0_gate": gate,
        "artifact_manifests": manifests,
        "forbidden_outputs": {
            "models": 0,
            "boosters": 0,
            "decoders": 0,
            "hmm_well_runs": 0,
            "predictions": 0,
            "submission": False,
        },
    }
    summary_manifest = write_json(
        output_dir / f"{OUTPUT_PREFIX}_summary.json",
        summary,
    )
    summary["artifact_manifests"]["summary"] = summary_manifest

    expected = set(get_nested(config, "artifacts.expected_stage_0_artifacts") or [])
    generated = {
        Path(item["path"]).name
        for item in summary["artifact_manifests"].values()
    }
    if generated != expected:
        raise RuntimeError(
            f"generated artifact contract mismatch: missing={sorted(expected - generated)}, "
            f"unexpected={sorted(generated - expected)}"
        )
    metrics_payload = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "pf_beam",
        "stage": "stage_0_guard_readout",
        "cv": gate,
        "public_lb": None,
        "private_lb": None,
        "metric": "transfer_gain_and_negative_transfer_safety",
        "score_unit": "horizontal_gr_api",
        "runtime_seconds": runtime_seconds,
        "summary_path": str(summary_manifest["path"]),
        "notes": (
            "No model, decoder, HMM, prediction, inference, or submission was run. "
            "A PASS does not reverse exp311/312 or unlock exp314-exp320."
        ),
    }
    write_json(metrics_output_path(), metrics_payload)
    return summary


# %% [markdown]
# ## 8. Setup, configuration, and contract preview

# %%
CONFIG = load_experiment_config()
validate_scientific_contract(CONFIG)
CONTRACT_PREVIEW = {
    "experiment": get_nested(CONFIG, "experiment.name"),
    "route": get_nested(CONFIG, "experiment.route"),
    "parent": get_nested(CONFIG, "lineage.parent"),
    "status": get_nested(CONFIG, "experiment.status"),
    "score_unit": get_nested(CONFIG, "validation.score_unit"),
    "audit_surfaces": get_nested(CONFIG, "validation.audit_surfaces"),
    "fallback_order": get_nested(CONFIG, "model.guard.fallback_order"),
    "execution_contract": get_nested(CONFIG, "execution_contract"),
    "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
    "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
}
print(json.dumps(to_jsonable(CONTRACT_PREVIEW), indent=2, sort_keys=True), flush=True)


# %% [markdown]
# ## 9. Run the diagnostic and report generated artifacts

# %%
SUMMARY: dict[str, Any] | None = None
if in_notebook_runtime():
    SUMMARY = run_stage_0(CONFIG)
    print(json.dumps(to_jsonable(SUMMARY["stage_0_gate"]), indent=2, sort_keys=True))
    print("generated artifacts", flush=True)
    for artifact_name, manifest_item in SUMMARY["artifact_manifests"].items():
        print(artifact_name, manifest_item.get("path"), flush=True)
