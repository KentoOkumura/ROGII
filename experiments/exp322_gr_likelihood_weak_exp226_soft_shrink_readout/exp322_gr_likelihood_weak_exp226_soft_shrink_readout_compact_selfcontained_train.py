# %% [markdown]
# # exp322 GR-likelihood-weak exp226 soft-shrink readout
#
# This zero-model train-side readout keeps the submitted exp263 fixed blend as
# the base prediction.  It scores a fixed 13-shift raw-GR likelihood bank around
# saved exp226 K16, freezes every target-free score/gate/prediction artifact, and
# only then attaches suffix truth for the fixed scientific decision.

# %% [markdown]
# ## Contents
# 1. Imports and fixed execution boundary
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract
# 4. Saved exp263 / exp226 / raw-well input checks
# 5. Exp280-parity Gaussian shift-likelihood scoring
# 6. Outer-train gate, circular control, and bounded shrink
# 7. Target-free freeze and late-truth readout
# 8. Fold, scope, well, and fixed decision metrics
# 9. Full Kaggle CPU orchestration and generated artifacts
# 10. Setup, contract preview, and execution

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp322_gr_likelihood_weak_exp226_soft_shrink_readout"
OUTPUT_PREFIX = "exp322"
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
    "truth_nearest_shift",
}
TARGET_FREE_PREDICTION_COLUMNS = [
    "well_id",
    "fold",
    "row_idx",
    "suffix_offset",
    "block_id",
    "md_since_ft",
    "p_base",
    "p226",
    "real_block_gate",
    "control_block_gate",
    "real_eligible",
    "control_eligible",
    "real_prediction",
    "control_prediction",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP322_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


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
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "AGENTS.md").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        value = read_yaml(path)
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return value
    raise FileNotFoundError(f"exp322 config not found in {[str(path) for path in candidates]}")


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


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
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
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking requires one finite score per shift")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int16)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int16)
    return ranks


def assert_no_target_columns(frame: pd.DataFrame, *, stage: str) -> None:
    leaked = sorted(TARGET_FREE_FORBIDDEN.intersection(frame.columns))
    if leaked:
        raise ValueError(f"{stage} target-free input contains forbidden columns: {leaked}")


# %% [markdown]
# ## 3. Frozen scientific contract


# %%
def validate_scientific_contract(
    config: Mapping[str, Any], *, require_kaggle_approval: bool = False
) -> None:
    expected_shifts = [
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
    ]
    checks = {
        "experiment.route": "pf_beam",
        "implementation.enabled": True,
        "implementation.scope": "train_side_readout",
        "validation.fold_policy": (
            "use_saved_exp263_outer_fold_for_readout_audit_saved_exp226_oof_source_fold_separately"
        ),
        "likelihood.block_rows": 512,
        "likelihood.block_policy": "non_overlapping_from_suffix_start_keep_short_tail",
        "likelihood.score_aggregation": "mean_row_log_likelihood",
        "likelihood.tie_policy": "config_shift_bank_order",
        "likelihood.emission.kind": "exp209_gaussian_raw_gr",
        "likelihood.emission.sigma_mode": "known_prefix_residual_std",
        "likelihood.emission.missing_gr_policy": "interpolate_both_directions_then_typewell_mean",
        "likelihood.emission.typewell_gr_policy": (
            "sort_tvt_ffill_bfill_then_linear_interp_endpoint_hold"
        ),
        "gate.threshold_fit": "outer_train_four_folds_target_free_blocks",
        "shrink.base_candidate": "exp226_w500_50_50",
        "shrink.destination_candidate": "exp226_k16",
        "negative_control.kind": "within_well_nonzero_circular_shift_of_block_gate",
        "execution_contract.active_candidates": 1,
        "execution_contract.diagnostic_controls": 1,
        "execution_contract.fold_strata": 5,
        "execution_contract.lightgbm_configs": 0,
        "execution_contract.trained_folds": 0,
        "execution_contract.total_boosters": 0,
        "execution_contract.hmm_well_runs": 0,
        "execution_contract.pf_well_runs": 0,
        "execution_contract.beam_well_runs": 0,
        "execution_contract.k16_well_runs": 0,
        "execution_contract.parent_control_retraining": False,
        "execution_contract.parent_prediction_regeneration": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected in checks.items():
        if get_nested(config, key) != expected:
            raise ValueError(f"exp322 frozen contract changed: {key} must be {expected!r}")
    shifts = [float(value) for value in get_nested(config, "likelihood.shift_bank_ft") or []]
    if shifts != expected_shifts:
        raise ValueError("exp322 fixes the approved 13-shift bank")
    sigma_clip = [
        float(value) for value in get_nested(config, "likelihood.emission.sigma_clip") or []
    ]
    if sigma_clip != [10.0, 60.0]:
        raise ValueError("exp322 fixes GR sigma clip [10, 60]")
    likelihood_clip = [
        float(value)
        for value in get_nested(config, "likelihood.emission.row_log_likelihood_clip") or []
    ]
    if likelihood_clip != [-600.0, 0.0]:
        raise ValueError("exp322 fixes the exp280-compatible legacy clip contract")
    numeric = {
        "gate.weak_gr_requires_all.maximum_margin_quantile": 0.20,
        "gate.weak_gr_requires_all.minimum_normalized_entropy_quantile": 0.80,
        "gate.exp226_admissible_requires_any.maximum_zero_rank": 3.0,
        "gate.exp226_admissible_requires_any.maximum_zero_gap_quantile": 0.20,
        "gate.minimum_raw_observed_gr_share": 0.80,
        "gate.minimum_md_since_last_known_ft": 250.0,
        "shrink.alpha": 0.25,
        "shrink.maximum_absolute_move_ft": 10.0,
    }
    for key, expected in numeric.items():
        if float(get_nested(config, key)) != expected:
            raise ValueError(f"exp322 frozen contract changed: {key} must be {expected}")
    for key in (
        "shrink.hard_replacement",
        "shrink.top1_shift_correction",
        "shrink.boundary_taper",
        "gate.absolute_log_likelihood_gate_enabled",
        "execution_contract.inference",
        "execution_contract.submission",
    ):
        if bool(get_nested(config, key)):
            raise ValueError(f"exp322 forbids {key}")
    if require_kaggle_approval and not bool(
        get_nested(config, "execution_contract.kaggle_push_approved")
    ):
        raise RuntimeError("exp322 Kaggle CPU package/push/run is not approved")


# %% [markdown]
# ## 4. Saved exp263 / exp226 / raw-well input checks


# %%
def resolve_exp263_manifest(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp263_cache") or {}
    expected_sha = str(spec["expected_manifest_sha256"])
    candidates = [str(value) for value in spec.get("manifest_candidates", [])]
    for source in spec.get("kaggle_sources", []):
        candidates.append(str(Path(str(source)) / "artifacts" / "cache_manifest.json"))
        candidates.append(str(Path(str(source)) / "cache_manifest.json"))
    matches: list[Path] = []
    root = project_root()
    for raw in candidates:
        candidate = Path(raw)
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            if path.is_file() and sha256_path(path) == expected_sha:
                matches.append(path)
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob("**/cache_manifest.json")):
            if path.is_file() and sha256_path(path) == expected_sha:
                matches.append(path)
    unique = list(dict.fromkeys(path.resolve() for path in matches))
    if not unique:
        raise FileNotFoundError("exp263 cache_manifest.json with the frozen SHA was not found")
    return unique[0]


def _cache_partition_path(
    cache_root: Path, candidate_id: str, fold: int, expected_name: str
) -> Path:
    direct = cache_root / "candidate_values" / candidate_id / f"fold={fold}" / expected_name
    if direct.is_file():
        return direct
    matches = sorted(
        cache_root.glob(f"**/candidate_values/{candidate_id}/fold={fold}/{expected_name}")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one exp263 partition for {candidate_id}/fold={fold}/{expected_name}"
        )
    return matches[0]


def _load_exp263_component(
    cache_root: Path,
    manifest: Mapping[str, Any],
    candidate_id: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records = list((manifest.get("candidate_value_partitions") or {}).get(candidate_id, []))
    if len(records) != 5:
        raise ValueError(f"exp263 manifest requires five partitions for {candidate_id}")
    parts: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for fold, record in enumerate(records):
        expected_name = Path(str(record["path"])).name
        path = _cache_partition_path(cache_root, candidate_id, fold, expected_name)
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
    exp226_k16: np.ndarray,
    likpf_mean: np.ndarray,
    exact_hmm: np.ndarray,
) -> np.ndarray:
    components = [
        np.asarray(exp226_k16, dtype=np.float32),
        np.asarray(likpf_mean, dtype=np.float32),
        np.asarray(exact_hmm, dtype=np.float32),
    ]
    if len({item.shape for item in components}) != 1:
        raise ValueError("exp263 fixed-blend components must have the same shape")
    # Match CandidateCache.materialize: promote saved float32 primitive values to
    # float64, accumulate the registered weights in component order, then store
    # the virtual candidate as float32.
    output = np.zeros(components[0].shape, dtype=np.float64)
    for weight, component in zip((0.50, 0.25, 0.25), components, strict=True):
        output += weight * component.astype(np.float64)
    return output.astype(np.float32)


def load_exp263_base(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = resolve_exp263_manifest(config)
    manifest = json.loads(manifest_path.read_text())
    expected = get_nested(config, "validation") or {}
    if (
        int(manifest.get("rows", -1)) != int(expected["expected_rows"])
        or int(manifest.get("wells", -1)) != int(expected["expected_wells"])
        or int(manifest.get("folds", -1)) != len(expected["expected_folds"])
    ):
        raise ValueError("exp263 cache manifest coverage differs from exp322 contract")
    component_ids = ["exp226_k16", "likpf_mean", "exact_hmm"]
    base: pd.DataFrame | None = None
    partition_evidence: list[dict[str, Any]] = []
    values: dict[str, np.ndarray] = {}
    key_columns = ["well", "well_row_idx", "outer_fold", "md_since"]
    for candidate_id in component_ids:
        frame, evidence = _load_exp263_component(manifest_path.parent, manifest, candidate_id)
        partition_evidence.extend(evidence)
        if base is None:
            base = frame[key_columns].copy()
        else:
            if len(base) != len(frame):
                raise ValueError("exp263 component row counts differ")
            for column in key_columns:
                left = base[column].to_numpy()
                right = frame[column].to_numpy()
                equal = (
                    np.array_equal(left, right, equal_nan=True)
                    if column == "md_since"
                    else np.array_equal(left, right)
                )
                if not equal:
                    raise ValueError(f"exp263 component identity mismatch in {column}")
        values[candidate_id] = frame["candidate_tvt"].to_numpy(np.float32)
    assert base is not None
    formula = get_nested(config, "data.exp263_cache.expected_formula") or {}
    expected_formula = {"exp226_k16": 0.50, "likpf_mean": 0.25, "exact_hmm": 0.25}
    if {str(key): float(value) for key, value in formula.items()} != expected_formula:
        raise ValueError("exp263 fixed formula differs from the frozen 0.50/0.25/0.25 contract")
    p_base = materialize_exp263_fixed_blend(
        values["exp226_k16"], values["likpf_mean"], values["exact_hmm"]
    )
    direct = (
        0.50 * values["exp226_k16"].astype(np.float64)
        + 0.25 * values["likpf_mean"].astype(np.float64)
        + 0.25 * values["exact_hmm"].astype(np.float64)
    ).astype(np.float32)
    parity_max_abs = float(
        np.max(np.abs(p_base.astype(np.float64) - direct.astype(np.float64)), initial=0.0)
    )
    maximum = float(
        get_nested(config, "validation.technical_guards.maximum_exp263_formula_parity_abs_ft")
    )
    if parity_max_abs > maximum:
        raise ValueError("exp263 fixed formula parity guard failed")
    base = base.rename(
        columns={
            "well": "well_id",
            "well_row_idx": "row_idx",
            "outer_fold": "fold",
            "md_since": "md_since_ft",
        }
    )
    base["p226"] = values["exp226_k16"].astype(np.float64)
    base["p_base"] = p_base.astype(np.float64)
    input_manifest = {
        "name": "exp263_saved_stage0_candidate_cache",
        "path": str(manifest_path.parent),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_path(manifest_path),
        "rows": len(base),
        "wells": int(base["well_id"].nunique()),
        "folds": sorted(int(value) for value in base["fold"].unique()),
        "formula_parity_max_abs_ft": parity_max_abs,
        "partition_evidence": partition_evidence,
    }
    return base, input_manifest


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
    if relationship.empty:
        raise ValueError("saved fold relationship requires rows")
    required = sorted(int(value) for value in expected_folds)
    for column in ("exp226_source_fold", "exp263_readout_fold"):
        actual = sorted(int(value) for value in relationship[column].unique())
        if actual != required:
            raise ValueError(f"{column} must contain the fixed five-fold set")
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
        "policy": ("exp263_outer_fold_is_readout_stratum_exp226_original_oof_fold_is_audited_only"),
        "exp226_source_fold_counts": {
            str(int(key)): int(value)
            for key, value in relationship["exp226_source_fold"]
            .value_counts(sort=False)
            .sort_index()
            .items()
        },
        "exp263_readout_fold_counts": {
            str(int(key)): int(value)
            for key, value in relationship["exp263_readout_fold"]
            .value_counts(sort=False)
            .sort_index()
            .items()
        },
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
    actual_decompressed_sha = sha256_gzip_decompressed(path)
    if actual_decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["target_free_columns"]]
    forbidden = set(str(value) for value in spec["forbidden_pre_freeze_columns"])
    if forbidden.intersection(safe_columns):
        raise ValueError("exp226 target-free column list contains truth/error columns")
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    assert_no_target_columns(frame, stage="exp226 OOF")
    for column in ("row_idx", "suffix_offset", "fold"):
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
    fold_relationship = audit_saved_fold_relationship(
        frame["well_id"].to_numpy(),
        frame["fold"].to_numpy(),
        base["fold"].to_numpy(),
        get_nested(config, "validation.expected_folds") or [],
    )
    anchor_parity = np.abs(
        frame["tvt_pred"].to_numpy(np.float32).astype(np.float64)
        - base["p226"].to_numpy(np.float64)
    )
    anchor_parity_max = float(anchor_parity.max(initial=0.0))
    if anchor_parity_max > 1e-5:
        raise ValueError("exp226 OOF and exp263 cached exp226 anchor differ")
    output = base.copy()
    output["suffix_offset"] = frame["suffix_offset"].to_numpy(np.int64)
    output = output[
        [
            "well_id",
            "fold",
            "row_idx",
            "suffix_offset",
            "md_since_ft",
            "p_base",
            "p226",
        ]
    ]
    manifest = {
        "name": "exp226_saved_oof_target_free_columns",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": actual_decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "target_free_columns": safe_columns,
        "cached_anchor_parity_max_abs_ft": anchor_parity_max,
        "fold_relationship": fold_relationship,
    }
    return output, path, manifest


def load_exp226_truth(
    path: Path,
    *,
    frozen_target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not frozen_target_free_contract_sha256:
        raise ValueError("truth attachment requires a frozen target-free contract SHA")
    frame = pd.read_csv(
        path,
        usecols=["well_id", "row_idx", "tvt_true"],
        dtype={"well_id": str},
    )
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("exp226 late truth rows must be unique and finite")
    return frame


def load_hidden_like_assignments(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"name": "hidden_like_assignments", "enabled": False}
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
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column != "TVT")
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    return frame


# %% [markdown]
# ## 5. Exp280-parity Gaussian shift-likelihood scoring


# %%
def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free GR preparation forbids horizontal TVT")
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal_without_truth.columns):
        missing = sorted(required_horizontal - set(horizontal_without_truth.columns))
        raise ValueError(f"horizontal missing {missing}")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy()).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    known = horizontal_without_truth.loc[horizontal_without_truth["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("well requires at least four known-prefix rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "likelihood.emission.sigma_clip")
    ]
    gr_sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix GR residual sigma is not finite")
    gr_fill = float(np.nanmean(typewell_gr))
    all_gr = (
        pd.to_numeric(horizontal_without_truth["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "gr_sigma": gr_sigma,
        "all_gr_interpolated": all_gr,
        "known_rows": len(known),
        "known_residual_mean": float(np.mean(residual)),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def score_well_target_free(
    anchor_rows: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    assert_no_target_columns(anchor_rows, stage="exp322 anchor rows")
    required = {
        "well_id",
        "fold",
        "row_idx",
        "suffix_offset",
        "md_since_ft",
        "p226",
    }
    if not required.issubset(anchor_rows.columns):
        raise ValueError(f"anchor rows missing {sorted(required - set(anchor_rows.columns))}")
    rows = anchor_rows.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if rows.empty or rows["well_id"].nunique() != 1 or rows["fold"].nunique() != 1:
        raise ValueError("score_well_target_free requires one non-empty well/fold")
    row_idx = rows["row_idx"].to_numpy(np.int64)
    suffix_offset = rows["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(rows), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero within each well")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("row_idx lies outside the raw horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("anchor rows must align only to unknown suffix rows")
    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    shifts = np.asarray(get_nested(config, "likelihood.shift_bank_ft"), dtype=np.float64)
    p226 = rows["p226"].to_numpy(np.float64)
    candidate_tvt = p226[:, None] + shifts[None, :]
    expected_gr = np.column_stack(
        [
            np.interp(candidate_tvt[:, slot], prepared["typewell_tvt"], prepared["typewell_gr"])
            for slot in range(len(shifts))
        ]
    )
    raw_gr = prepared["all_gr_interpolated"][row_idx]
    zscore = (raw_gr[:, None] - expected_gr) / float(prepared["gr_sigma"])
    # exp280 clips squared z at 600 before multiplying by -0.5.  The exp322
    # config keeps the approved [-600, 0] legacy label, while this computation
    # intentionally preserves exp280 numerical parity.
    legacy_clip = abs(float(get_nested(config, "likelihood.emission.row_log_likelihood_clip")[0]))
    log_likelihood = -0.5 * np.minimum(np.square(zscore), legacy_clip)
    if not np.isfinite(log_likelihood).all():
        raise ValueError("target-free likelihood must be finite")
    observed_gr = pd.to_numeric(horizontal_without_truth.iloc[row_idx]["GR"], errors="coerce")
    md = pd.to_numeric(horizontal_without_truth["MD"], errors="raise").to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_without_truth["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    raw_md_since = md[row_idx] - md[last_known]
    if not np.allclose(
        raw_md_since,
        rows["md_since_ft"].to_numpy(np.float64),
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("exp263 md_since does not match raw MD identity")
    block_rows = int(get_nested(config, "likelihood.block_rows"))
    block_id = suffix_offset // block_rows
    native = (candidate_tvt >= prepared["typewell_tvt"].min()) & (
        candidate_tvt <= prepared["typewell_tvt"].max()
    )
    extension = float(get_nested(config, "likelihood.typewell_extension_ft"))
    extended = (candidate_tvt >= prepared["typewell_tvt"].min() - extension) & (
        candidate_tvt <= prepared["typewell_tvt"].max() + extension
    )
    well = str(rows["well_id"].iloc[0])
    fold = int(rows["fold"].iloc[0])
    output_rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        scores = log_likelihood[mask].mean(axis=0)
        score_sums = log_likelihood[mask].sum(axis=0)
        ranks = rank_descending(scores)
        positions = np.flatnonzero(mask)
        for slot, shift in enumerate(shifts):
            output_rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix_offset[positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[positions[-1]]),
                    "block_start_row_idx": int(row_idx[positions[0]]),
                    "block_end_row_idx": int(row_idx[positions[-1]]),
                    "block_row_count": int(mask.sum()),
                    "md_since_min_ft": float(raw_md_since[mask].min()),
                    "md_since_max_ft": float(raw_md_since[mask].max()),
                    "observed_gr_share": float(observed_gr.iloc[positions].notna().mean()),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "likelihood_mean": float(scores[slot]),
                    "likelihood_sum": float(score_sums[slot]),
                    "likelihood_rank": int(ranks[slot]),
                    "native_typewell_coverage": float(native[mask, slot].mean()),
                    "extended_typewell_coverage": float(extended[mask, slot].mean()),
                }
            )
    output = pd.DataFrame(output_rows).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    manifest = {
        "well_id": well,
        "fold": fold,
        "horizontal_rows": len(horizontal_without_truth),
        "evaluation_rows": len(rows),
        "blocks": int(block_id.max() + 1),
        "known_rows": int(prepared["known_rows"]),
        "last_known_row_idx": last_known,
        "gr_sigma": float(prepared["gr_sigma"]),
        "known_residual_mean": float(prepared["known_residual_mean"]),
        "known_residual_std_unclipped": float(prepared["known_residual_std_unclipped"]),
        "observed_eval_gr_share": float(observed_gr.notna().mean()),
        "score_finite_coverage": float(np.isfinite(log_likelihood).mean()),
    }
    return output.reset_index(drop=True), manifest


def build_block_features(
    target_free_scores: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    shifts = np.asarray(get_nested(config, "likelihood.shift_bank_ft"), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for (well, fold, block), part in target_free_scores.groupby(
        ["well_id", "fold", "block_id"], sort=True, observed=True
    ):
        part = part.sort_values("shift_slot", kind="mergesort")
        if len(part) != len(shifts) or not np.array_equal(
            part["shift_ft"].to_numpy(np.float64), shifts
        ):
            raise ValueError(f"shift-bank alignment failed for {well} block {block}")
        scores = part["likelihood_mean"].to_numpy(np.float64)
        ranks = rank_descending(scores)
        ordered = np.sort(scores)[::-1]
        probability = np.exp(scores - np.max(scores))
        probability /= probability.sum()
        entropy = float(-np.sum(probability * np.log(probability)) / np.log(len(shifts)))
        zero_slot = int(np.flatnonzero(np.isclose(shifts, 0.0))[0])
        first = part.iloc[0]
        rows.append(
            {
                "well_id": str(well),
                "fold": int(fold),
                "block_id": int(block),
                "block_start_suffix_offset": int(first["block_start_suffix_offset"]),
                "block_end_suffix_offset": int(first["block_end_suffix_offset"]),
                "block_start_row_idx": int(first["block_start_row_idx"]),
                "block_end_row_idx": int(first["block_end_row_idx"]),
                "block_row_count": int(first["block_row_count"]),
                "md_since_min_ft": float(first["md_since_min_ft"]),
                "md_since_max_ft": float(first["md_since_max_ft"]),
                "observed_gr_share": float(first["observed_gr_share"]),
                "top1_top2_margin": float(ordered[0] - ordered[1]),
                "normalized_entropy": entropy,
                "zero_rank": int(ranks[zero_slot]),
                "best_minus_zero_gap": float(ordered[0] - scores[zero_slot]),
                "top1_shift_ft": float(shifts[int(np.argmax(scores))]),
            }
        )
    output = (
        pd.DataFrame(rows)
        .sort_values(["well_id", "block_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not np.isfinite(
        output[
            [
                "top1_top2_margin",
                "normalized_entropy",
                "best_minus_zero_gap",
                "observed_gr_share",
            ]
        ].to_numpy(np.float64)
    ).all():
        raise ValueError("block features must be finite")
    return output


# %% [markdown]
# ## 6. Outer-train gate, circular control, and bounded shrink


# %%
def fit_outer_train_thresholds(
    block_features: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    margin_q = float(get_nested(config, "gate.weak_gr_requires_all.maximum_margin_quantile"))
    entropy_q = float(
        get_nested(config, "gate.weak_gr_requires_all.minimum_normalized_entropy_quantile")
    )
    zero_gap_q = float(
        get_nested(config, "gate.exp226_admissible_requires_any.maximum_zero_gap_quantile")
    )
    rows = []
    for fold in expected_folds:
        outer_train = block_features.loc[block_features["fold"] != fold]
        if outer_train.empty or fold not in set(block_features["fold"]):
            raise ValueError(f"outer-train threshold fold {fold} has empty train/validation")
        rows.append(
            {
                "fold": fold,
                "outer_train_blocks": len(outer_train),
                "margin_q20": float(outer_train["top1_top2_margin"].quantile(margin_q)),
                "entropy_q80": float(outer_train["normalized_entropy"].quantile(entropy_q)),
                "zero_gap_q20": float(outer_train["best_minus_zero_gap"].quantile(zero_gap_q)),
            }
        )
    return pd.DataFrame(rows)


def stable_circular_offset(well_id: str, block_count: int, key_prefix: str) -> int:
    if block_count <= 1:
        return 0
    prefix = hashlib.sha256(f"{key_prefix}|{well_id}".encode()).hexdigest()[:8]
    return 1 + int(prefix, 16) % (block_count - 1)


def apply_real_and_control_gates(
    block_features: pd.DataFrame,
    thresholds: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    output = block_features.merge(thresholds, on="fold", how="left", validate="many_to_one")
    if output[["margin_q20", "entropy_q80", "zero_gap_q20"]].isna().any().any():
        raise ValueError("outer-train thresholds failed to cover every validation block")
    maximum_zero_rank = int(
        get_nested(config, "gate.exp226_admissible_requires_any.maximum_zero_rank")
    )
    minimum_observed = float(get_nested(config, "gate.minimum_raw_observed_gr_share"))
    output["weak_margin"] = output["top1_top2_margin"] <= output["margin_q20"]
    output["weak_entropy"] = output["normalized_entropy"] >= output["entropy_q80"]
    output["weak_gr"] = output["weak_margin"] & output["weak_entropy"]
    output["zero_rank_admissible"] = output["zero_rank"] <= maximum_zero_rank
    output["zero_gap_admissible"] = output["best_minus_zero_gap"] <= output["zero_gap_q20"]
    output["exp226_admissible"] = output["zero_rank_admissible"] | output["zero_gap_admissible"]
    output["gr_observed_enough"] = output["observed_gr_share"] >= minimum_observed
    output["real_block_gate"] = (
        output["weak_gr"] & output["exp226_admissible"] & output["gr_observed_enough"]
    )
    output["control_block_gate"] = False
    output["control_offset_blocks"] = 0
    key_prefix = str(get_nested(config, "negative_control.key_prefix"))
    for well, indices in output.groupby("well_id", sort=True).groups.items():
        ordered_indices = output.loc[indices].sort_values("block_id", kind="mergesort").index
        gates = output.loc[ordered_indices, "real_block_gate"].to_numpy(bool)
        offset = stable_circular_offset(str(well), len(gates), key_prefix)
        control = np.roll(gates, offset) if len(gates) > 1 else gates.copy()
        output.loc[ordered_indices, "control_block_gate"] = control
        output.loc[ordered_indices, "control_offset_blocks"] = offset
        if int(control.sum()) != int(gates.sum()):
            raise RuntimeError("circular control did not preserve within-well activation count")
        if len(gates) > 1 and offset == 0:
            raise RuntimeError("multi-block circular control offset must be nonzero")
    return output.sort_values(["well_id", "block_id"], kind="mergesort").reset_index(drop=True)


def build_target_free_predictions(
    base_rows: pd.DataFrame,
    block_gates: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    assert_no_target_columns(base_rows, stage="exp322 base prediction")
    output = base_rows.copy()
    block_rows = int(get_nested(config, "likelihood.block_rows"))
    output["block_id"] = output["suffix_offset"].to_numpy(np.int64) // block_rows
    output = output.merge(
        block_gates[["well_id", "fold", "block_id", "real_block_gate", "control_block_gate"]],
        on=["well_id", "fold", "block_id"],
        how="left",
        validate="many_to_one",
    )
    if output[["real_block_gate", "control_block_gate"]].isna().any().any():
        raise ValueError("block gate did not cover every prediction row")
    minimum_md = float(get_nested(config, "gate.minimum_md_since_last_known_ft"))
    output["real_eligible"] = output["real_block_gate"].astype(bool) & (
        output["md_since_ft"] >= minimum_md
    )
    output["control_eligible"] = output["control_block_gate"].astype(bool) & (
        output["md_since_ft"] >= minimum_md
    )
    alpha = float(get_nested(config, "shrink.alpha"))
    maximum_move = float(get_nested(config, "shrink.maximum_absolute_move_ft"))
    move = np.clip(
        alpha * (output["p226"].to_numpy(np.float64) - output["p_base"].to_numpy(np.float64)),
        -maximum_move,
        maximum_move,
    )
    base = output["p_base"].to_numpy(np.float64)
    output["real_prediction"] = base + output["real_eligible"].to_numpy(bool) * move
    output["control_prediction"] = base + output["control_eligible"].to_numpy(bool) * move
    output = (
        output[TARGET_FREE_PREDICTION_COLUMNS]
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert_no_target_columns(output, stage="exp322 target-free predictions")
    if not np.isfinite(
        output[["p_base", "p226", "real_prediction", "control_prediction"]].to_numpy(np.float64)
    ).all():
        raise ValueError("target-free candidate prediction must be finite")
    near = output["md_since_ft"].to_numpy(np.float64) < minimum_md
    if not np.array_equal(
        output.loc[near, "p_base"].to_numpy(np.float64),
        output.loc[near, "real_prediction"].to_numpy(np.float64),
    ):
        raise ValueError("near-range target-free parity guard failed")
    return output


# %% [markdown]
# ## 7. Target-free freeze and late-truth readout


# %%
def build_target_free_contract(
    config: Mapping[str, Any],
    score_artifact: Mapping[str, Any],
    gate_artifact: Mapping[str, Any],
    prediction_artifact: Mapping[str, Any],
    input_manifest: pd.DataFrame,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "route": get_nested(config, "experiment.route"),
        "lineage": get_nested(config, "lineage"),
        "likelihood": get_nested(config, "likelihood"),
        "gate": get_nested(config, "gate"),
        "shrink": get_nested(config, "shrink"),
        "negative_control": get_nested(config, "negative_control"),
        # Decompressed CSV bytes are the primary reproducibility evidence for
        # gzip outputs.  Frame-content and schema hashes remain secondary guards.
        "target_free_score_content_sha256": score_artifact["decompressed_sha256"],
        "target_free_score_frame_content_sha256": score_artifact["content_sha256"],
        "target_free_score_schema_sha256": score_artifact["schema_sha256"],
        "target_free_gate_content_sha256": gate_artifact["decompressed_sha256"],
        "target_free_gate_frame_content_sha256": gate_artifact["content_sha256"],
        "target_free_gate_schema_sha256": gate_artifact["schema_sha256"],
        "target_free_prediction_content_sha256": prediction_artifact["decompressed_sha256"],
        "target_free_prediction_frame_content_sha256": prediction_artifact["content_sha256"],
        "target_free_prediction_schema_sha256": prediction_artifact["schema_sha256"],
        "input_manifest_content_sha256": dataframe_content_sha(input_manifest),
    }
    contract["target_free_contract_sha256"] = mapping_sha256(contract)
    return contract


def attach_truth_after_freeze(
    target_free_predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    target_free_contract_sha256: str,
) -> pd.DataFrame:
    if not target_free_contract_sha256:
        raise ValueError("late truth join requires a frozen target-free contract")
    assert_no_target_columns(target_free_predictions, stage="late truth join")
    joined = target_free_predictions.merge(
        truth, on=["well_id", "row_idx"], how="left", validate="one_to_one"
    )
    if len(joined) != len(target_free_predictions) or joined["tvt_true"].isna().any():
        raise ValueError("late truth join failed full row identity coverage")
    return joined.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError("RMSE inputs must be aligned and non-empty")
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def prediction_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "scope": scope,
            "rows": 0,
            "wells": 0,
            "base_rmse": np.nan,
            "real_rmse": np.nan,
            "control_rmse": np.nan,
            "real_rmse_delta_ft": np.nan,
            "control_rmse_delta_ft": np.nan,
            "real_rmse_gain_ft": np.nan,
            "control_rmse_gain_ft": np.nan,
            "real_gain_minus_control_gain_ft": np.nan,
            "real_eligible_rows": 0,
            "control_eligible_rows": 0,
            "changed_rows": 0,
            "changed_row_fraction": 0.0,
        }
    truth = frame["tvt_true"].to_numpy(np.float64)
    base_rmse = rmse(truth, frame["p_base"].to_numpy(np.float64))
    real_rmse = rmse(truth, frame["real_prediction"].to_numpy(np.float64))
    control_rmse = rmse(truth, frame["control_prediction"].to_numpy(np.float64))
    changed = frame["real_prediction"].to_numpy(np.float64) != frame["p_base"].to_numpy(np.float64)
    return {
        "scope": scope,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "base_rmse": base_rmse,
        "real_rmse": real_rmse,
        "control_rmse": control_rmse,
        "real_rmse_delta_ft": real_rmse - base_rmse,
        "control_rmse_delta_ft": control_rmse - base_rmse,
        "real_rmse_gain_ft": base_rmse - real_rmse,
        "control_rmse_gain_ft": base_rmse - control_rmse,
        "real_gain_minus_control_gain_ft": control_rmse - real_rmse,
        "real_eligible_rows": int(frame["real_eligible"].sum()),
        "control_eligible_rows": int(frame["control_eligible"].sum()),
        "changed_rows": int(changed.sum()),
        "changed_row_fraction": float(changed.mean()),
    }


def build_fold_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in joined.groupby("fold", sort=True):
        row = prediction_metric_row(part, scope=f"fold_{int(fold)}")
        row["fold"] = int(fold)
        rows.append(row)
    return pd.DataFrame(rows)


def build_scope_metrics(
    joined: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    near_max = float(get_nested(config, "validation.scopes.near.maximum_md_since_ft"))
    long_min = float(get_nested(config, "validation.scopes.long_tail.minimum_md_since_ft"))
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("overall", joined),
        ("activated_subset", joined.loc[joined["real_eligible"].astype(bool)]),
        ("near_0_250", joined.loc[joined["md_since_ft"] < near_max]),
        ("long_tail_1000_plus", joined.loc[joined["md_since_ft"] >= long_min]),
    ]
    if hidden_assignments.empty:
        raise ValueError("exp322 scientific contract requires hidden-like assignments")
    role_columns = get_nested(config, "data.hidden_like.role_columns") or {}
    roles = hidden_assignments.set_index("well_id")
    for scope_name, role_column in role_columns.items():
        valid_wells = set(roles.index[roles[str(role_column)].astype(str) == "valid"].astype(str))
        scopes.append(
            (str(scope_name), joined.loc[joined["well_id"].astype(str).isin(valid_wells)])
        )
    return pd.DataFrame([prediction_metric_row(frame, scope=name) for name, frame in scopes])


def build_by_well_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, part in joined.groupby("well_id", sort=True):
        row = prediction_metric_row(part, scope=str(well))
        row["well_id"] = str(well)
        row["fold"] = int(part["fold"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# %% [markdown]
# ## 8. Fold, scope, well, and fixed decision metrics


# %%
def evaluate_fixed_decision(
    joined: pd.DataFrame,
    target_free_scores: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    input_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical = get_nested(config, "validation.technical_guards") or {}
    scientific = get_nested(config, "validation.scientific_pass_requires_all") or {}
    changed = joined["real_prediction"].to_numpy(np.float64) != joined["p_base"].to_numpy(
        np.float64
    )
    changed_wells = int(joined.loc[changed, "well_id"].nunique())
    changed_folds = int(joined.loc[changed, "fold"].nunique())
    changed_fraction = float(changed.mean())
    expected_folds = [int(value) for value in technical["required_fold_set"]]
    actual_folds = sorted(int(value) for value in joined["fold"].unique())
    score_finite = float(
        np.isfinite(target_free_scores["likelihood_mean"].to_numpy(np.float64)).mean()
    )
    prediction_finite = float(
        np.isfinite(
            joined[["p_base", "p226", "real_prediction", "control_prediction"]].to_numpy(np.float64)
        ).mean()
    )
    hard_checks = {
        "expected_rows": len(joined) == int(get_nested(config, "validation.expected_rows")),
        "expected_wells": joined["well_id"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "expected_folds": actual_folds == expected_folds,
        "row_identity_coverage": float(input_audit["row_identity_coverage"])
        >= float(technical["required_row_identity_coverage"]),
        "well_identity_coverage": float(input_audit["well_identity_coverage"])
        >= float(technical["required_well_identity_coverage"]),
        "finite_score_coverage": score_finite >= float(technical["required_finite_score_coverage"]),
        "finite_prediction_coverage": prediction_finite
        >= float(technical["required_finite_prediction_coverage"]),
        "exp263_formula_parity": float(input_audit["exp263_formula_parity_max_abs_ft"])
        <= float(technical["maximum_exp263_formula_parity_abs_ft"]),
    }
    coverage_checks = {
        "minimum_changed_row_fraction": changed_fraction
        >= float(technical["minimum_changed_row_fraction"]),
        "maximum_changed_row_fraction": changed_fraction
        <= float(technical["maximum_changed_row_fraction"]),
        "minimum_changed_wells": changed_wells >= int(technical["minimum_changed_wells"]),
        "minimum_changed_folds": changed_folds >= int(technical["minimum_folds_with_changed_rows"]),
    }
    scope = scope_metrics.set_index("scope")
    overall = scope.loc["overall"]
    activated = scope.loc["activated_subset"]
    near = joined["md_since_ft"].to_numpy(np.float64) < float(
        get_nested(config, "gate.minimum_md_since_last_known_ft")
    )
    near_parity = np.array_equal(
        joined.loc[near, "p_base"].to_numpy(np.float64),
        joined.loc[near, "real_prediction"].to_numpy(np.float64),
    )
    by_well_delta = by_well["real_rmse_delta_ft"].to_numpy(np.float64)
    by_well_p95 = float(np.quantile(by_well_delta, 0.95))
    worst_well_delta = float(by_well_delta.max(initial=-np.inf))
    improved_folds = int((fold_metrics["real_rmse_delta_ft"] < 0.0).sum())
    science_checks = {
        "overall_gain": float(overall["real_rmse_gain_ft"])
        >= float(scientific["minimum_overall_rmse_gain_ft"]),
        "improved_folds": improved_folds >= int(scientific["minimum_improved_folds"]),
        "activated_subset_gain": float(activated["real_rmse_gain_ft"])
        >= float(scientific["minimum_activated_subset_rmse_gain_ft"]),
        "near_prediction_bitwise_parity": near_parity,
        "long_tail_nonworse": float(scope.loc["long_tail_1000_plus", "real_rmse_delta_ft"])
        <= float(scientific["maximum_long_tail_rmse_delta_ft"]),
        "hidden_like_spatial_nonworse": float(
            scope.loc["hidden_like_spatial", "real_rmse_delta_ft"]
        )
        <= float(scientific["maximum_hidden_like_spatial_rmse_delta_ft"]),
        "hidden_like_typewell_purged_nonworse": float(
            scope.loc["hidden_like_typewell_purged", "real_rmse_delta_ft"]
        )
        <= float(scientific["maximum_hidden_like_typewell_purged_rmse_delta_ft"]),
        "by_well_delta_p95_nonworse": by_well_p95
        <= float(scientific["maximum_by_well_delta_p95_ft"]),
        "worst_well_nonworse": worst_well_delta <= float(scientific["maximum_worst_well_delta_ft"]),
        "real_beats_circular_control": float(overall["real_gain_minus_control_gain_ft"])
        >= float(scientific["minimum_real_gain_minus_circular_control_gain_ft"]),
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
        "changed_rows": int(changed.sum()),
        "changed_row_fraction": changed_fraction,
        "changed_wells": changed_wells,
        "changed_folds": changed_folds,
        "actual_folds": actual_folds,
        "score_finite_coverage": score_finite,
        "prediction_finite_coverage": prediction_finite,
        "improved_folds": improved_folds,
        "by_well_rmse_delta_p95_ft": by_well_p95,
        "worst_well_rmse_delta_ft": worst_well_delta,
        "overall": overall.to_dict(),
        "activated_subset": activated.to_dict(),
    }


# %% [markdown]
# ## 9. Full Kaggle CPU orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp322 readout must run on Kaggle; local execution requires explicit "
            "EXPERIMENT_ALLOW_LOCAL=1 approval."
        )
    validate_scientific_contract(config, require_kaggle_approval=True)
    started = time.time()
    base, exp263_manifest = load_exp263_base(config)
    base, exp226_path, exp226_manifest = load_exp226_safe(config, base)
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    raw_dir = train_data_dir(config)
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(raw_wells) != expected_wells or set(raw_wells) != set(base["well_id"].unique()):
        raise ValueError("raw train, exp263, and exp226 well sets do not match")

    score_parts: list[pd.DataFrame] = []
    well_manifest_rows: list[dict[str, Any]] = []
    for index, (well, anchor_rows) in enumerate(base.groupby("well_id", sort=True), start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"missing raw well pair for {well}")
        horizontal = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        well_scores, well_manifest = score_well_target_free(
            anchor_rows, horizontal, typewell, config
        )
        well_manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        score_parts.append(well_scores)
        well_manifest_rows.append(well_manifest)
        if index % 25 == 0 or index == expected_wells:
            print(f"target-free scoring wells={index}/{expected_wells}")

    target_free_scores = (
        pd.concat(score_parts, ignore_index=True)
        .sort_values(["well_id", "block_id", "shift_slot"], kind="mergesort")
        .reset_index(drop=True)
    )
    block_features = build_block_features(target_free_scores, config)
    thresholds = fit_outer_train_thresholds(block_features, config)
    block_gates = apply_real_and_control_gates(block_features, thresholds, config)
    target_free_predictions = build_target_free_predictions(base, block_gates, config)

    well_manifest = pd.DataFrame(well_manifest_rows).sort_values("well_id", kind="mergesort")
    input_manifest = pd.DataFrame(
        [
            exp263_manifest,
            exp226_manifest,
            hidden_manifest,
            {
                "name": "raw_train_horizontal_and_typewell",
                "path": str(raw_dir),
                "rows": int(well_manifest["horizontal_rows"].sum()),
                "wells": len(well_manifest),
                "raw_sha256": dataframe_content_sha(
                    well_manifest,
                    ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
            },
        ]
    )
    artifacts = artifact_dir()
    input_artifact = write_csv(input_manifest, artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv")
    score_artifact = write_csv_gzip(
        target_free_scores,
        artifacts / f"{OUTPUT_PREFIX}_target_free_shift_scores.csv.gz",
    )
    gate_artifact = write_csv_gzip(
        block_gates,
        artifacts / f"{OUTPUT_PREFIX}_target_free_block_gate.csv.gz",
    )
    prediction_artifact = write_csv_gzip(
        target_free_predictions,
        artifacts / f"{OUTPUT_PREFIX}_target_free_predictions.csv.gz",
    )
    contract = build_target_free_contract(
        config, score_artifact, gate_artifact, prediction_artifact, input_manifest
    )
    contract_path = artifacts / f"{OUTPUT_PREFIX}_score_contract.json"
    write_json(contract_path, contract)

    # Suffix truth is first read here, after all target-free artifacts and their
    # content hashes have been persisted.
    truth = load_exp226_truth(
        exp226_path,
        frozen_target_free_contract_sha256=str(contract["target_free_contract_sha256"]),
    )
    joined = attach_truth_after_freeze(
        target_free_predictions,
        truth,
        target_free_contract_sha256=str(contract["target_free_contract_sha256"]),
    )
    fold_metrics = build_fold_metrics(joined)
    scope_metrics = build_scope_metrics(joined, hidden_assignments, config)
    by_well = build_by_well_metrics(joined)
    input_audit = {
        "row_identity_coverage": 1.0,
        "well_identity_coverage": 1.0,
        "exp263_formula_parity_max_abs_ft": exp263_manifest["formula_parity_max_abs_ft"],
    }
    decision = evaluate_fixed_decision(
        joined,
        target_free_scores,
        fold_metrics,
        scope_metrics,
        by_well,
        input_audit,
        config,
    )
    fold_artifact = write_csv(fold_metrics, artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv")
    scope_artifact = write_csv(scope_metrics, artifacts / f"{OUTPUT_PREFIX}_scope_metrics.csv")
    well_artifact = write_csv(by_well, artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_readout_completed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(joined),
        "wells": int(joined["well_id"].nunique()),
        "blocks": len(block_gates),
        "shift_candidates": len(get_nested(config, "likelihood.shift_bank_ft")),
        "execution": {
            "active_candidates": 1,
            "diagnostic_controls": 1,
            "fold_strata": 5,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "decoder_regeneration": 0,
            "parent_retraining": False,
        },
        "decision": decision,
        "truth_attachment": {
            "stage": "after_target_free_score_gate_prediction_freeze",
            "target_free_contract_sha256": contract["target_free_contract_sha256"],
        },
        "artifacts": {
            "input_manifest": input_artifact,
            "target_free_scores": score_artifact,
            "target_free_block_gate": gate_artifact,
            "target_free_predictions": prediction_artifact,
            "score_contract": {
                "path": str(contract_path),
                "raw_sha256": sha256_path(contract_path),
            },
            "fold_metrics": fold_artifact,
            "scope_metrics": scope_artifact,
            "by_well_metrics": well_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "next_action": "request separate inference design approval only if decision is PASS",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_readout_completed",
        "route": "pf_beam",
        "cv": float(decision["overall"]["real_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "parent_reference_cv": float(decision["overall"]["base_rmse"]),
        "decision": decision["decision"],
        "diagnostic": decision,
        "target_free_contract_sha256": contract["target_free_contract_sha256"],
        "notes": "0-model train-side readout; no inference or submission generated.",
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup, contract preview, and execution


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
                "shift_bank_ft": get_nested(CONFIG, "likelihood.shift_bank_ft"),
                "block_rows": get_nested(CONFIG, "likelihood.block_rows"),
                "active_candidates": get_nested(CONFIG, "execution_contract.active_candidates"),
                "diagnostic_controls": get_nested(CONFIG, "execution_contract.diagnostic_controls"),
                "fold_strata": get_nested(CONFIG, "execution_contract.fold_strata"),
                "lightgbm_configs": get_nested(CONFIG, "execution_contract.lightgbm_configs"),
                "trained_folds": get_nested(CONFIG, "execution_contract.trained_folds"),
                "boosters": get_nested(CONFIG, "execution_contract.total_boosters"),
                "parent_regeneration": get_nested(
                    CONFIG, "execution_contract.parent_prediction_regeneration"
                ),
                "kaggle_push_approved": get_nested(
                    CONFIG, "execution_contract.kaggle_push_approved"
                ),
                "inference": get_nested(CONFIG, "execution_contract.inference"),
                "submission": get_nested(CONFIG, "execution_contract.submission"),
            },
            indent=2,
        )
    )


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP322_SUMMARY = run_full_experiment(CONFIG)
