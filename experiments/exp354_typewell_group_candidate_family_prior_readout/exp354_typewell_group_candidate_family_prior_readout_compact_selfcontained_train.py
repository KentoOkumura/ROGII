# %% [markdown]
# # exp354 Type-Well group candidate-family prior readout
#
# This Stage 0 notebook tests one fixed zero-model hypothesis: a soft
# outer-train Type-Well-group x physical-candidate-family error prior can rank
# candidate families on held-out wells.  It does not train a selector, alter a
# candidate value, regenerate PF/Beam candidates, run inference, or create a
# submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, and deterministic artifact helpers
# 3. Scientific contract and fixed candidate-family manifest
# 4. exp293 candidate-bank, fold, group, and hidden-like input checks
# 5. Post-freeze truth and well-family error summaries
# 6. Stable group-label control and outer-train prior generation
# 7. Held-out family-rank readout and Stage 0 gate
# 8. Setup, configuration, and input preview
# 9. Stage 0 orchestration, metrics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

try:
    from IPython.display import display
except ImportError:

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp354_typewell_group_candidate_family_prior_readout"
PACKAGE_DIR = Path.cwd()
EXPECTED_CANDIDATE_ORDER = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
EXPECTED_FAMILY_ORDER = (
    "geometry",
    "hmm_selfgr",
    "pf",
    "hmm",
    "beam",
    "virtual_combination",
)
EXPECTED_FAMILY_BY_CANDIDATE = {
    "exp226_k16": "geometry",
    "selfgr_hmm_a070": "hmm_selfgr",
    "likpf_mean": "pf",
    "exact_hmm": "hmm",
    "pf_ancc": "pf",
    "beam_mean": "beam",
    "exp226_k16__selfgr_hmm_a070": "virtual_combination",
    "exp226_k16__exact_hmm": "virtual_combination",
    "exp226_k16__likpf_mean": "virtual_combination",
    "selfgr_hmm_a070__likpf_mean": "virtual_combination",
    "likpf_mean__exact_hmm": "virtual_combination",
    "exp226_w500_50_50": "virtual_combination",
}
SURFACES = (
    "overall",
    "hidden_like_spatial",
    "hidden_like_typewell_purged",
)
CONTROLS = ("real_native_group", "stable_group_label_shuffle_within_fold")
FORBIDDEN_PRE_FREEZE_COLUMNS = {
    "tvt",
    "target",
    "true_tvt",
    "truth",
    "error",
    "abs_error",
    "squared_error",
    "oracle",
    "rank",
}


# %% [markdown]
# ## 2. Runtime, configuration, and deterministic artifact helpers

# %%
def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_config_path() -> Path:
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR
        / "experiments"
        / EXPERIMENT_NAME
        / "config.yaml",
        Path("/kaggle/working/config.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp354 config.yaml was not found")


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def sha256_path(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(
    path: Path, chunk_size: int = 4 * 1024 * 1024
) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return stable_json_sha256(schema)


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode()


def dataframe_content_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(dataframe_csv_bytes(frame)).hexdigest()


def write_json(path: Path, value: Any) -> dict[str, Any]:
    payload = stable_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    payload = dataframe_csv_bytes(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "bytes": len(payload),
        "schema_sha256": dataframe_schema_sha256(frame),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    payload = dataframe_csv_bytes(frame)
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return {
        "path": str(path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "bytes": len(compressed),
        "schema_sha256": dataframe_schema_sha256(frame),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_sha256": hashlib.sha256(compressed).hexdigest(),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
    }


def stable_digest(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def stable_int(*parts: Any, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return int(stable_digest(*parts)[:16], 16) % modulo


def resolve_existing(filename: str, candidates: Sequence[str]) -> Path:
    searched: list[str] = []
    for raw in candidates:
        candidate = Path(raw)
        variants = [candidate]
        if not candidate.is_absolute():
            variants.extend([PACKAGE_DIR / candidate, Path.cwd() / candidate])
        for variant in variants:
            searched.append(str(variant))
            if variant.is_file():
                return variant
            nested = variant / filename
            searched.append(str(nested))
            if nested.is_file():
                return nested
    raise FileNotFoundError(f"{filename} was not found; searched={searched}")


def resolve_existing_dir(candidates: Sequence[str], required_glob: str) -> Path:
    searched: list[str] = []
    for raw in candidates:
        candidate = Path(raw)
        variants = [candidate]
        if not candidate.is_absolute():
            variants.extend([PACKAGE_DIR / candidate, Path.cwd() / candidate])
        for variant in variants:
            searched.append(str(variant))
            if variant.is_dir() and next(variant.glob(required_glob), None) is not None:
                return variant
    raise FileNotFoundError(
        f"input directory with {required_glob!r} was not found; searched={searched}"
    )


def output_directory() -> Path:
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/artifacts")
    return PACKAGE_DIR / "artifacts"


# %% [markdown]
# ## 3. Scientific contract and fixed candidate-family manifest

# %%
def validate_scientific_contract(config: Mapping[str, Any]) -> dict[str, int]:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "ml_model",
        "implementation.enabled": True,
        "implementation.scope": "stage_0_compact_selfcontained_canonical",
        "implementation.canonical_notebook_adopted": True,
        "implementation.stage_1_implemented": False,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "validation.outer_folds": 5,
        "validation.fit_scope": "outer_train_wells_only",
        "validation.family_error_unit": (
            "equal_candidate_within_family_then_equal_well"
        ),
        "validation.rank_score": "shrunk_family_rmse_ascending",
        "validation.rank_aggregation": (
            "mean_within_well_spearman_then_equal_well"
        ),
        "model.candidate_bank.source": "exp293_deployable12",
        "model.candidate_bank.regenerate_candidates": False,
        "model.prior.group": "native_overlap_1",
        "model.prior.shrinkage_support_k_wells": 10,
        "model.prior.ranking_statistic": "rmse",
        "model.prior.family_candidate_reducer": "equal_candidate",
        "model.prior.well_reducer": "equal_well",
        "model.prior.best_family_tie_policy": "fixed_family_order",
        "model.stage_1.control_retrain": False,
        "execution_contract.stage_0.prior_variants": 1,
        "execution_contract.stage_0.negative_controls": 1,
        "execution_contract.stage_0.outer_folds": 5,
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.parent_control_retraining": False,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.num_workers": 1,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for key, expected_value in expected.items():
        observed = get_nested(config, key)
        if observed != expected_value:
            raise ValueError(
                f"exp354 scientific contract changed: {key}="
                f"{observed!r}, expected {expected_value!r}"
            )
    if tuple(get_nested(config, "model.candidate_bank.order") or ()) != (
        EXPECTED_CANDIDATE_ORDER
    ):
        raise ValueError("exp354 candidate order must match frozen exp293 deployable12")
    if tuple(get_nested(config, "model.candidate_bank.family_order") or ()) != (
        EXPECTED_FAMILY_ORDER
    ):
        raise ValueError("exp354 family order changed")
    if dict(get_nested(config, "model.candidate_bank.family_by_candidate") or {}) != (
        EXPECTED_FAMILY_BY_CANDIDATE
    ):
        raise ValueError("exp354 candidate-family mapping changed")
    if tuple(get_nested(config, "model.stage_0.controls") or ()) != CONTROLS:
        raise ValueError("exp354 fixes exactly one real prior and one stable shuffle")
    if list(get_nested(config, "model.prior.statistics") or []) != [
        "mae",
        "rmse",
        "best_family_rate",
    ]:
        raise ValueError("exp354 prior statistics changed")
    if list(get_nested(config, "model.prior.fallback_order") or []) != [
        "group_family",
        "global_family",
        "neutral",
    ]:
        raise ValueError("exp354 fallback order changed")
    forbidden = set(get_nested(config, "model.forbidden") or [])
    required_forbidden = {
        "exp311_or_exp312_group_statistics",
        "exp313_guard_output",
        "exp315_rank_features",
        "family_or_support_grid",
        "hard_family_router",
        "candidate_value_change",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp354 forbidden-action contract is incomplete")
    return {
        "prior_variants": 1,
        "negative_controls": 1,
        "reporting_folds": 5,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }


def validate_run_approval(config: Mapping[str, Any]) -> None:
    validate_scientific_contract(config)
    approvals = {
        "execution.implementation_approved": True,
        "execution.kaggle_push_approved": True,
        "execution.run_stage_0": True,
        "runtime.kaggle.train_run_on_push": True,
    }
    for key, expected in approvals.items():
        if get_nested(config, key) is not expected:
            raise RuntimeError(
                "exp354 Stage 0 package/push/run is not approved; "
                f"{key} must be true"
            )


def build_candidate_family_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    order = list(get_nested(config, "model.candidate_bank.order") or [])
    mapping = dict(get_nested(config, "model.candidate_bank.family_by_candidate") or {})
    family_order = list(get_nested(config, "model.candidate_bank.family_order") or [])
    family_position = {name: index for index, name in enumerate(family_order)}
    rows = [
        {
            "candidate_position": position,
            "candidate_id": candidate,
            "family": mapping[candidate],
            "family_position": family_position[mapping[candidate]],
            "candidate_role": (
                "primitive" if position < 6 else "fixed_virtual_combination"
            ),
        }
        for position, candidate in enumerate(order)
    ]
    manifest = pd.DataFrame(rows).sort_values(
        ["candidate_position"], kind="mergesort"
    )
    if tuple(manifest["candidate_id"]) != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("candidate-family manifest candidate identity mismatch")
    if tuple(dict.fromkeys(manifest["family"])) != EXPECTED_FAMILY_ORDER:
        raise ValueError("candidate-family manifest family identity mismatch")
    return manifest.reset_index(drop=True)


# %% [markdown]
# ## 4. exp293 candidate-bank, fold, group, and hidden-like input checks

# %%
@dataclass(frozen=True)
class Exp293Inputs:
    root: Path
    bank_manifest_path: Path
    block_assignment_path: Path
    candidate_bank_path: Path
    bank_manifest: dict[str, Any]
    block_assignment: pd.DataFrame
    candidate_values: np.memmap
    evidence: dict[str, Any]


@dataclass(frozen=True)
class FrozenInputs:
    exp293: Exp293Inputs
    family_manifest: pd.DataFrame
    group_membership: pd.DataFrame
    raw_train_dir: Path
    input_manifest: dict[str, Any]
    target_free_freeze_sha256: str
    truth_rows_before_freeze: int


def find_exp293_root(config: Mapping[str, Any]) -> Path:
    spec = get_nested(config, "data.exp293") or {}
    filenames = spec["filenames"]
    searched: list[str] = []
    for raw in spec["root_candidates"]:
        root = Path(raw)
        roots = [root]
        if not root.is_absolute():
            roots.extend([PACKAGE_DIR / root, Path.cwd() / root])
        for candidate in roots:
            searched.append(str(candidate))
            if all((candidate / filename).is_file() for filename in filenames.values()):
                return candidate
    raise FileNotFoundError(f"complete exp293 v2 output was not found: {searched}")


def load_exp293_inputs(config: Mapping[str, Any]) -> Exp293Inputs:
    spec = get_nested(config, "data.exp293") or {}
    root = find_exp293_root(config)
    filenames = spec["filenames"]
    manifest_path = root / filenames["bank_manifest"]
    block_path = root / filenames["block_assignment"]
    values_path = root / filenames["candidate_bank"]
    observed_sha = {
        "bank_manifest_raw_sha256": sha256_path(manifest_path),
        "block_assignment_raw_sha256": sha256_path(block_path),
        "block_assignment_decompressed_sha256": sha256_decompressed_gzip(block_path),
        "candidate_bank_raw_sha256": sha256_path(values_path),
    }
    expected_sha = {
        "bank_manifest_raw_sha256": spec["expected_bank_manifest_raw_sha256"],
        "block_assignment_raw_sha256": spec[
            "expected_block_assignment_raw_sha256"
        ],
        "block_assignment_decompressed_sha256": spec[
            "expected_block_assignment_decompressed_sha256"
        ],
        "candidate_bank_raw_sha256": spec["expected_candidate_bank_raw_sha256"],
    }
    if observed_sha != expected_sha:
        raise ValueError(
            f"exp293 SHA preflight failed: observed={observed_sha}, expected={expected_sha}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_rows = int(spec["expected_rows"])
    expected_candidates = int(spec["expected_candidate_count"])
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("exp293 bank manifest row count mismatch")
    if int(manifest.get("wells", -1)) != int(spec["expected_wells"]):
        raise ValueError("exp293 bank manifest well count mismatch")
    if tuple(manifest.get("candidate_ids", ())) != EXPECTED_CANDIDATE_ORDER:
        raise ValueError("exp293 bank manifest candidate order mismatch")
    if manifest.get("candidate_content_sha256") != spec[
        "expected_candidate_bank_content_sha256"
    ]:
        raise ValueError("exp293 candidate content SHA mismatch")
    if manifest.get("key_content_sha256") != spec["expected_key_content_sha256"]:
        raise ValueError("exp293 key content SHA mismatch")
    expected_bytes = expected_rows * expected_candidates * np.dtype("float32").itemsize
    if values_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"exp293 candidate bank bytes={values_path.stat().st_size}, "
            f"expected={expected_bytes}"
        )
    block = pd.read_csv(
        block_path,
        usecols=["id", "well", "well_row_idx", "outer_fold"],
        dtype={
            "id": str,
            "well": str,
            "well_row_idx": np.int32,
            "outer_fold": np.int8,
        },
    )
    if len(block) != expected_rows:
        raise ValueError("exp293 block assignment row count mismatch")
    if block["id"].duplicated().any():
        raise ValueError("exp293 block assignment contains duplicate ids")
    if block["well"].nunique() != int(spec["expected_wells"]):
        raise ValueError("exp293 block assignment well count mismatch")
    if set(map(int, block["outer_fold"].unique())) != set(range(5)):
        raise ValueError("exp293 outer-fold inventory mismatch")
    if block.groupby("well", sort=False)["outer_fold"].nunique().max() != 1:
        raise ValueError("at least one exp293 well spans multiple folds")
    values = np.memmap(
        values_path,
        mode="r",
        dtype="float32",
        shape=(expected_rows, expected_candidates),
    )
    if not np.isfinite(values[:: max(expected_rows // 1000, 1)]).all():
        raise ValueError("sampled exp293 candidate bank values are not finite")
    evidence = {
        "kernel_id": spec["expected_kernel_id"],
        "kernel_version": int(spec["expected_kernel_version"]),
        "root": str(root),
        "rows": expected_rows,
        "wells": int(spec["expected_wells"]),
        "candidate_count": expected_candidates,
        "candidate_ids": list(EXPECTED_CANDIDATE_ORDER),
        "candidate_content_sha256": manifest["candidate_content_sha256"],
        "key_content_sha256": manifest["key_content_sha256"],
        **observed_sha,
    }
    return Exp293Inputs(
        root=root,
        bank_manifest_path=manifest_path,
        block_assignment_path=block_path,
        candidate_bank_path=values_path,
        bank_manifest=manifest,
        block_assignment=block,
        candidate_values=values,
        evidence=evidence,
    )


def load_group_membership(
    config: Mapping[str, Any], block_assignment: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.typewell_group_assignments") or {}
    path = resolve_existing(str(spec["filename"]), list(spec["candidates"]))
    if sha256_path(path) != spec["expected_raw_sha256"]:
        raise ValueError("Type-Well group assignment SHA mismatch")
    source = pd.read_csv(path, dtype=str)
    required = {"method", "threshold", "cluster_id", "well_id", "cluster_size"}
    if not required.issubset(source.columns):
        raise ValueError(
            f"Type-Well group assignment columns missing: {required - set(source.columns)}"
        )
    selected = source[
        source["method"].eq(str(spec["method"]))
        & source["threshold"].eq(str(spec["threshold"]))
    ][["well_id", "cluster_id", "cluster_size"]].copy()
    selected = selected.rename(
        columns={"cluster_id": "real_group_id", "cluster_size": "group_size"}
    )
    selected["group_size"] = pd.to_numeric(
        selected["group_size"], errors="raise"
    ).astype(int)
    if selected.empty or selected["well_id"].duplicated().any():
        raise ValueError("native Type-Well group membership is empty or duplicated")
    well_fold = (
        block_assignment[["well", "outer_fold"]]
        .drop_duplicates()
        .rename(columns={"well": "well_id"})
    )
    if well_fold["well_id"].duplicated().any():
        raise ValueError("exp293 fold identity contains duplicate well mappings")
    membership = well_fold.merge(selected, on="well_id", how="left", validate="one_to_one")
    if membership["real_group_id"].isna().any():
        missing = membership.loc[
            membership["real_group_id"].isna(), "well_id"
        ].tolist()[:10]
        raise ValueError(f"Type-Well group membership missing exp293 wells: {missing}")
    evidence = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "method": str(spec["method"]),
        "threshold": str(spec["threshold"]),
        "selected_wells": len(membership),
        "selected_groups": int(membership["real_group_id"].nunique()),
    }
    return membership.sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    ), evidence


def attach_hidden_like_roles(
    membership: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like_assignments") or {}
    path = resolve_existing(str(spec["filename"]), list(spec["candidates"]))
    observed_sha = sha256_path(path)
    if observed_sha != spec["expected_raw_sha256"]:
        raise ValueError("hidden-like assignment SHA mismatch")
    source = pd.read_csv(path, dtype=str)
    spatial = str(spec["spatial_role_column"])
    purged = str(spec["typewell_purged_role_column"])
    required = {"well_id", spatial, purged}
    if not required.issubset(source.columns):
        raise ValueError(f"hidden-like role columns missing: {required - set(source.columns)}")
    roles = source[["well_id", spatial, purged]].copy()
    if roles["well_id"].duplicated().any():
        raise ValueError("hidden-like roles contain duplicate wells")
    out = membership.merge(roles, on="well_id", how="left", validate="one_to_one")
    if out[[spatial, purged]].isna().any().any():
        raise ValueError("hidden-like roles do not cover every exp293 well")
    out = out.rename(
        columns={
            spatial: "hidden_like_spatial_role",
            purged: "hidden_like_typewell_purged_role",
        }
    )
    evidence = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": observed_sha,
        "selected_role": str(spec["selected_role"]),
        "spatial_roles": out["hidden_like_spatial_role"].value_counts().to_dict(),
        "typewell_purged_roles": out[
            "hidden_like_typewell_purged_role"
        ].value_counts().to_dict(),
    }
    return out, evidence


def add_stable_group_label_shuffle(
    membership: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, str]:
    required = {"well_id", "outer_fold", "real_group_id"}
    if not required.issubset(membership.columns):
        raise ValueError(f"group shuffle input missing {required - set(membership.columns)}")
    base = membership.sort_values(
        ["outer_fold", "well_id"], kind="mergesort"
    ).reset_index(drop=True)
    group_content_sha = dataframe_content_sha256(
        base[["well_id", "outer_fold", "real_group_id"]]
    )
    shuffled_parts: list[pd.DataFrame] = []
    for fold, fold_frame in base.groupby("outer_fold", sort=True):
        part = fold_frame.copy().reset_index(drop=True)
        labels = part["real_group_id"].astype(str).to_numpy()
        if len(part) < 2:
            raise ValueError("each fold needs at least two wells for stable shuffle")
        offset = 1 + stable_int(
            seed,
            "exp354_group_label_shuffle",
            int(fold),
            group_content_sha,
            modulo=len(part) - 1,
        )
        part["shuffled_group_id"] = np.roll(labels, int(offset))
        part["shuffle_offset"] = int(offset)
        if sorted(part["shuffled_group_id"]) != sorted(part["real_group_id"]):
            raise AssertionError("stable shuffle did not preserve the group-label multiset")
        shuffled_parts.append(part)
    result = pd.concat(shuffled_parts, ignore_index=True).sort_values(
        ["outer_fold", "well_id"], kind="mergesort"
    )
    return result.reset_index(drop=True), group_content_sha


def freeze_target_free_inputs(config: Mapping[str, Any]) -> FrozenInputs:
    exp293 = load_exp293_inputs(config)
    family_manifest = build_candidate_family_manifest(config)
    membership, group_evidence = load_group_membership(
        config, exp293.block_assignment
    )
    membership, role_evidence = attach_hidden_like_roles(membership, config)
    membership, group_content_sha = add_stable_group_label_shuffle(
        membership, int(get_nested(config, "validation.seed"))
    )
    raw_train_dir = resolve_existing_dir(
        list(get_nested(config, "data.raw_train_dir_candidates") or []),
        str(get_nested(config, "data.raw_horizontal_glob")),
    )
    pre_freeze_columns = {
        str(column).lower()
        for frame in (family_manifest, membership, exp293.block_assignment)
        for column in frame.columns
    }
    leaked = pre_freeze_columns & FORBIDDEN_PRE_FREEZE_COLUMNS
    if leaked:
        raise ValueError(f"target-free input freeze exposes forbidden columns: {sorted(leaked)}")
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "status": "target_free_identity_frozen_before_truth",
        "truth_rows_before_freeze": 0,
        "outer_valid_truth_rows_before_prior_freeze": 0,
        "exp293": exp293.evidence,
        "candidate_family_manifest": {
            "rows": len(family_manifest),
            "families": list(EXPECTED_FAMILY_ORDER),
            "schema_sha256": dataframe_schema_sha256(family_manifest),
            "content_sha256": dataframe_content_sha256(family_manifest),
        },
        "group_membership": {
            **group_evidence,
            "content_sha256": dataframe_content_sha256(membership),
            "real_group_content_sha256": group_content_sha,
            "shuffle_content_sha256": dataframe_content_sha256(
                membership[
                    [
                        "well_id",
                        "outer_fold",
                        "real_group_id",
                        "shuffled_group_id",
                        "shuffle_offset",
                    ]
                ]
            ),
        },
        "hidden_like_assignments": role_evidence,
        "raw_train_dir": str(raw_train_dir),
        "raw_truth_loaded": False,
        "forbidden_scientific_inputs": [
            "exp311_or_exp312_group_statistics",
            "exp313_guard_output",
            "exp315_rank_features",
        ],
    }
    freeze_sha = stable_json_sha256(input_manifest)
    input_manifest["target_free_freeze_sha256"] = freeze_sha
    return FrozenInputs(
        exp293=exp293,
        family_manifest=family_manifest,
        group_membership=membership,
        raw_train_dir=raw_train_dir,
        input_manifest=input_manifest,
        target_free_freeze_sha256=freeze_sha,
        truth_rows_before_freeze=0,
    )


# %% [markdown]
# ## 5. Post-freeze truth and well-family error summaries

# %%
def _well_slices(block_assignment: pd.DataFrame) -> Iterable[tuple[str, slice]]:
    wells = block_assignment["well"].astype(str).to_numpy()
    if len(wells) == 0:
        return
    start = 0
    seen: set[str] = set()
    for stop in range(1, len(wells) + 1):
        if stop < len(wells) and wells[stop] == wells[start]:
            continue
        well = wells[start]
        if well in seen:
            raise ValueError(f"exp293 block rows are not contiguous for well {well}")
        seen.add(well)
        yield well, slice(start, stop)
        start = stop


def compute_well_family_errors(
    frozen: FrozenInputs, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frozen.truth_rows_before_freeze != 0:
        raise ValueError("truth was accessed before target-free input freeze")
    if stable_json_sha256(
        {
            key: value
            for key, value in frozen.input_manifest.items()
            if key != "target_free_freeze_sha256"
        }
    ) != frozen.target_free_freeze_sha256:
        raise ValueError("target-free input manifest changed before truth load")
    block = frozen.exp293.block_assignment
    values = frozen.exp293.candidate_values
    truth_column = str(get_nested(config, "data.raw_columns.truth"))
    visible_column = str(get_nested(config, "data.raw_columns.visible_input"))
    family_manifest = frozen.family_manifest
    family_indices = {
        family: family_manifest.loc[
            family_manifest["family"].eq(family), "candidate_position"
        ].to_numpy(dtype=np.int64)
        for family in EXPECTED_FAMILY_ORDER
    }
    rows: list[dict[str, Any]] = []
    truth_digest = hashlib.sha256()
    raw_file_evidence: list[dict[str, Any]] = []
    for well_id, row_slice in _well_slices(block):
        expected = block.iloc[row_slice]
        path = frozen.raw_train_dir / f"{well_id}__horizontal_well.csv"
        if not path.is_file():
            raise FileNotFoundError(f"raw horizontal well missing: {path}")
        raw = pd.read_csv(path, usecols=[truth_column, visible_column])
        visible = pd.to_numeric(raw[visible_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        suffix_row_idx = np.flatnonzero(~np.isfinite(visible)).astype(np.int32)
        expected_row_idx = expected["well_row_idx"].to_numpy(dtype=np.int32)
        if not np.array_equal(suffix_row_idx, expected_row_idx):
            raise ValueError(f"suffix row identity mismatch for {well_id}")
        truth = pd.to_numeric(
            raw.loc[suffix_row_idx, truth_column], errors="raise"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(truth).all():
            raise ValueError(f"suffix truth contains nonfinite values for {well_id}")
        expected_ids = np.array(
            [f"{well_id}_{int(row)}" for row in suffix_row_idx], dtype=object
        )
        if not np.array_equal(expected["id"].astype(str).to_numpy(), expected_ids):
            raise ValueError(f"suffix id identity mismatch for {well_id}")
        well_values = np.asarray(values[row_slice, :], dtype=np.float64)
        if well_values.shape != (len(truth), len(EXPECTED_CANDIDATE_ORDER)):
            raise ValueError(f"candidate matrix shape mismatch for {well_id}")
        if not np.isfinite(well_values).all():
            raise ValueError(f"candidate matrix contains nonfinite values for {well_id}")
        truth_digest.update(
            dataframe_csv_bytes(
                pd.DataFrame({"id": expected_ids, "true_tvt": truth})
            )
        )
        fold_values = expected["outer_fold"].unique()
        if len(fold_values) != 1:
            raise ValueError(f"well {well_id} spans multiple outer folds")
        fold = int(fold_values[0])
        family_stats: list[dict[str, Any]] = []
        error = well_values - truth[:, None]
        for family_position, family in enumerate(EXPECTED_FAMILY_ORDER):
            indices = family_indices[family]
            family_error = error[:, indices]
            mae = float(np.mean(np.abs(family_error), dtype=np.float64))
            mse = float(np.mean(np.square(family_error), dtype=np.float64))
            family_stats.append(
                {
                    "well_id": well_id,
                    "outer_fold": fold,
                    "family": family,
                    "family_position": family_position,
                    "candidate_count": len(indices),
                    "suffix_rows": len(truth),
                    "mae": mae,
                    "mse": mse,
                    "rmse": math.sqrt(mse),
                }
            )
        best_position = min(
            range(len(family_stats)),
            key=lambda position: (
                family_stats[position]["rmse"],
                family_stats[position]["family_position"],
            ),
        )
        for position, record in enumerate(family_stats):
            record["is_best_family"] = int(position == best_position)
            rows.append(record)
        raw_file_evidence.append(
            {
                "well_id": well_id,
                "rows": len(raw),
                "suffix_rows": len(truth),
                "raw_sha256": sha256_path(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["outer_fold", "well_id", "family_position"], kind="mergesort"
    )
    expected_rows = int(get_nested(config, "data.exp293.expected_wells")) * len(
        EXPECTED_FAMILY_ORDER
    )
    if len(frame) != expected_rows:
        raise ValueError(f"well-family error rows={len(frame)}, expected={expected_rows}")
    if frame[["well_id", "family"]].duplicated().any():
        raise ValueError("well-family error table contains duplicate keys")
    numeric = frame[["mae", "mse", "rmse"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("well-family error table contains nonfinite statistics")
    evidence = {
        "truth_attached_after_target_free_freeze": True,
        "target_free_freeze_sha256": frozen.target_free_freeze_sha256,
        "truth_rows": int(frame["suffix_rows"].groupby(frame["well_id"]).first().sum()),
        "truth_content_sha256": truth_digest.hexdigest(),
        "raw_horizontal_files": len(raw_file_evidence),
        "raw_horizontal_file_manifest_sha256": stable_json_sha256(raw_file_evidence),
        "well_family_error_schema_sha256": dataframe_schema_sha256(frame),
        "well_family_error_content_sha256": dataframe_content_sha256(frame),
    }
    return frame.reset_index(drop=True), evidence


# %% [markdown]
# ## 6. Stable group-label control and outer-train prior generation

# %%
def _aggregate_family_statistics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        raise ValueError("cannot aggregate an empty family table")
    return {
        "source_wells": int(frame["well_id"].nunique()),
        "mae": float(frame["mae"].mean()),
        "mse": float(frame["mse"].mean()),
        "rmse": math.sqrt(float(frame["mse"].mean())),
        "best_family_rate": float(frame["is_best_family"].mean()),
    }


def build_prior_schedule(
    well_family_error: pd.DataFrame,
    membership: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    expected_error_columns = {
        "well_id",
        "outer_fold",
        "family",
        "mae",
        "mse",
        "rmse",
        "is_best_family",
    }
    if not expected_error_columns.issubset(well_family_error.columns):
        raise ValueError("well-family error input is incomplete")
    k = float(get_nested(config, "model.prior.shrinkage_support_k_wells"))
    if k != 10.0:
        raise ValueError("exp354 fixes shrinkage support k=10 wells")
    membership_by_well = membership.set_index("well_id", drop=False)
    all_wells = set(well_family_error["well_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for control, group_column in (
        ("real_native_group", "real_group_id"),
        ("stable_group_label_shuffle_within_fold", "shuffled_group_id"),
    ):
        errors = well_family_error.merge(
            membership[["well_id", group_column]],
            on="well_id",
            how="left",
            validate="many_to_one",
        ).rename(columns={group_column: "group_id"})
        if errors["group_id"].isna().any():
            raise ValueError(f"{control} group mapping is incomplete")
        for fold in range(int(get_nested(config, "validation.outer_folds"))):
            train = errors.loc[errors["outer_fold"].ne(fold)].copy()
            valid_wells = sorted(
                errors.loc[errors["outer_fold"].eq(fold), "well_id"].astype(str).unique()
            )
            fit_wells = sorted(train["well_id"].astype(str).unique())
            if set(valid_wells) & set(fit_wells):
                raise ValueError(f"outer fold {fold} fit/valid well overlap is nonzero")
            if set(valid_wells) | set(fit_wells) != all_wells:
                raise ValueError(f"outer fold {fold} does not partition all wells")
            global_by_family = {
                family: _aggregate_family_statistics(
                    train.loc[train["family"].eq(family)]
                )
                for family in EXPECTED_FAMILY_ORDER
            }
            neutral = {
                "mae": float(
                    np.mean(
                        [global_by_family[family]["mae"] for family in EXPECTED_FAMILY_ORDER]
                    )
                ),
                "mse": float(
                    np.mean(
                        [global_by_family[family]["mse"] for family in EXPECTED_FAMILY_ORDER]
                    )
                ),
                "best_family_rate": 1.0 / len(EXPECTED_FAMILY_ORDER),
            }
            for well_id in valid_wells:
                valid_group = str(membership_by_well.loc[well_id, group_column])
                for family_position, family in enumerate(EXPECTED_FAMILY_ORDER):
                    global_stats = global_by_family.get(family)
                    group_frame = train.loc[
                        train["group_id"].eq(valid_group)
                        & train["family"].eq(family)
                    ]
                    if not group_frame.empty and global_stats is not None:
                        group_stats = _aggregate_family_statistics(group_frame)
                        source_wells = int(group_stats["source_wells"])
                        alpha = source_wells / (source_wells + k)
                        prior_mae = (
                            alpha * group_stats["mae"]
                            + (1.0 - alpha) * global_stats["mae"]
                        )
                        prior_mse = (
                            alpha * group_stats["mse"]
                            + (1.0 - alpha) * global_stats["mse"]
                        )
                        prior_best = (
                            alpha * group_stats["best_family_rate"]
                            + (1.0 - alpha) * global_stats["best_family_rate"]
                        )
                        selected_source = "group_family"
                        fallback_reason = ""
                        group_available = True
                    elif global_stats is not None:
                        source_wells = 0
                        alpha = 0.0
                        prior_mae = global_stats["mae"]
                        prior_mse = global_stats["mse"]
                        prior_best = global_stats["best_family_rate"]
                        selected_source = "global_family"
                        fallback_reason = "group_unseen_in_outer_train"
                        group_available = False
                    else:
                        source_wells = 0
                        alpha = 0.0
                        prior_mae = neutral["mae"]
                        prior_mse = neutral["mse"]
                        prior_best = neutral["best_family_rate"]
                        selected_source = "neutral"
                        fallback_reason = "family_has_no_outer_train_support"
                        group_available = False
                    rows.append(
                        {
                            "control": control,
                            "outer_fold": fold,
                            "well_id": well_id,
                            "group_id": valid_group,
                            "family": family,
                            "family_position": family_position,
                            "selected_source": selected_source,
                            "fallback_reason": fallback_reason,
                            "group_available": bool(group_available),
                            "source_wells": source_wells,
                            "global_source_wells": int(
                                global_stats["source_wells"]
                                if global_stats is not None
                                else 0
                            ),
                            "shrinkage_alpha": float(alpha),
                            "prior_mae": float(prior_mae),
                            "prior_mse": float(prior_mse),
                            "prior_rmse": math.sqrt(float(prior_mse)),
                            "prior_best_family_rate": float(prior_best),
                            "fit_well_count": len(fit_wells),
                            "fit_well_ids_sha256": stable_json_sha256(fit_wells),
                            "fit_valid_well_overlap": 0,
                            "outer_valid_truth_rows_before_prior_freeze": 0,
                        }
                    )
    schedule = pd.DataFrame(rows).sort_values(
        ["control", "outer_fold", "well_id", "family_position"],
        kind="mergesort",
    )
    expected_rows = (
        len(CONTROLS)
        * len(all_wells)
        * len(EXPECTED_FAMILY_ORDER)
    )
    if len(schedule) != expected_rows:
        raise ValueError(f"prior schedule rows={len(schedule)}, expected={expected_rows}")
    if schedule[
        ["control", "outer_fold", "well_id", "family"]
    ].duplicated().any():
        raise ValueError("prior schedule contains duplicate keys")
    finite_columns = [
        "shrinkage_alpha",
        "prior_mae",
        "prior_mse",
        "prior_rmse",
        "prior_best_family_rate",
    ]
    if not np.isfinite(schedule[finite_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("prior schedule contains nonfinite prior values")
    if int(schedule["fit_valid_well_overlap"].max()) != 0:
        raise ValueError("prior schedule leaks outer-valid wells into fit")
    freeze_sha = dataframe_content_sha256(schedule)
    schedule["prior_schedule_freeze_sha256"] = freeze_sha
    return schedule.reset_index(drop=True), freeze_sha


# %% [markdown]
# ## 7. Held-out family-rank readout and Stage 0 gate

# %%
def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if int(finite.sum()) < 2:
        return math.nan
    x_rank = pd.Series(x_values[finite]).rank(method="average").to_numpy(np.float64)
    y_rank = pd.Series(y_values[finite]).rank(method="average").to_numpy(np.float64)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def attach_heldout_errors_after_freeze(
    prior_schedule: pd.DataFrame,
    prior_schedule_freeze_sha256: str,
    well_family_error: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if len(prior_schedule_freeze_sha256) != 64:
        raise ValueError("prior schedule is not completely frozen")
    if not prior_schedule["prior_schedule_freeze_sha256"].eq(
        prior_schedule_freeze_sha256
    ).all():
        raise ValueError("prior schedule freeze SHA does not match every row")
    observed = well_family_error[
        [
            "well_id",
            "outer_fold",
            "family",
            "mae",
            "mse",
            "rmse",
            "is_best_family",
        ]
    ].rename(
        columns={
            "mae": "observed_mae",
            "mse": "observed_mse",
            "rmse": "observed_rmse",
            "is_best_family": "observed_is_best_family",
        }
    )
    readout = prior_schedule.merge(
        observed,
        on=["well_id", "outer_fold", "family"],
        how="left",
        validate="many_to_one",
    )
    readout = readout.merge(
        membership[
            [
                "well_id",
                "hidden_like_spatial_role",
                "hidden_like_typewell_purged_role",
            ]
        ],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if readout[
        [
            "observed_mae",
            "observed_mse",
            "observed_rmse",
            "observed_is_best_family",
        ]
    ].isna().any().any():
        raise ValueError("held-out family errors did not join after prior freeze")
    readout["prior_rmse_rank"] = readout.groupby(
        ["control", "outer_fold", "well_id"], sort=False
    )["prior_rmse"].rank(method="average", ascending=True)
    readout["observed_rmse_rank"] = readout.groupby(
        ["control", "outer_fold", "well_id"], sort=False
    )["observed_rmse"].rank(method="average", ascending=True)
    readout["prior_mae_rank"] = readout.groupby(
        ["control", "outer_fold", "well_id"], sort=False
    )["prior_mae"].rank(method="average", ascending=True)
    readout["observed_mae_rank"] = readout.groupby(
        ["control", "outer_fold", "well_id"], sort=False
    )["observed_mae"].rank(method="average", ascending=True)
    readout["prior_best_rate_rank"] = readout.groupby(
        ["control", "outer_fold", "well_id"], sort=False
    )["prior_best_family_rate"].rank(method="average", ascending=False)
    per_well: dict[tuple[str, int, str], tuple[float, float, float]] = {}
    for key, frame in readout.groupby(
        ["control", "outer_fold", "well_id"], sort=True
    ):
        per_well[key] = (
            spearman_rank_correlation(
                frame["prior_rmse_rank"], frame["observed_rmse_rank"]
            ),
            spearman_rank_correlation(
                frame["prior_mae_rank"], frame["observed_mae_rank"]
            ),
            spearman_rank_correlation(
                frame["prior_best_rate_rank"], frame["observed_rmse_rank"]
            ),
        )
    readout["well_spearman_rmse"] = [
        per_well[(str(control), int(fold), str(well))][0]
        for control, fold, well in zip(
            readout["control"],
            readout["outer_fold"],
            readout["well_id"],
            strict=False,
        )
    ]
    readout["well_spearman_mae"] = [
        per_well[(str(control), int(fold), str(well))][1]
        for control, fold, well in zip(
            readout["control"],
            readout["outer_fold"],
            readout["well_id"],
            strict=False,
        )
    ]
    readout["well_spearman_best_rate_vs_rmse"] = [
        per_well[(str(control), int(fold), str(well))][2]
        for control, fold, well in zip(
            readout["control"],
            readout["outer_fold"],
            readout["well_id"],
            strict=False,
        )
    ]
    return readout.sort_values(
        ["control", "outer_fold", "well_id", "family_position"],
        kind="mergesort",
    ).reset_index(drop=True)


def _unique_well_readout(readout: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "control",
        "outer_fold",
        "well_id",
        "group_available",
        "selected_source",
        "well_spearman_rmse",
        "well_spearman_mae",
        "well_spearman_best_rate_vs_rmse",
        "hidden_like_spatial_role",
        "hidden_like_typewell_purged_role",
    ]
    return readout[columns].drop_duplicates(
        ["control", "outer_fold", "well_id"]
    )


def build_readout_metrics(
    readout: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = _unique_well_readout(readout)
    fold_rows: list[dict[str, Any]] = []
    for (control, fold), frame in wells.groupby(
        ["control", "outer_fold"], sort=True
    ):
        finite = np.isfinite(frame["well_spearman_rmse"].to_numpy(np.float64))
        fold_rows.append(
            {
                "control": control,
                "outer_fold": int(fold),
                "wells": len(frame),
                "finite_spearman_wells": int(finite.sum()),
                "family_rank_spearman": float(
                    frame.loc[finite, "well_spearman_rmse"].mean()
                ),
                "family_rank_spearman_median": float(
                    frame.loc[finite, "well_spearman_rmse"].median()
                ),
                "mae_rank_spearman": float(
                    frame.loc[
                        np.isfinite(frame["well_spearman_mae"]),
                        "well_spearman_mae",
                    ].mean()
                ),
                "best_rate_vs_rmse_rank_spearman": float(
                    frame.loc[
                        np.isfinite(frame["well_spearman_best_rate_vs_rmse"]),
                        "well_spearman_best_rate_vs_rmse",
                    ].mean()
                ),
                "group_coverage": float(frame["group_available"].mean()),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["control", "outer_fold"], kind="mergesort"
    )
    selected_role = str(get_nested(config, "data.hidden_like_assignments.selected_role"))
    surface_rows: list[dict[str, Any]] = []
    for control, control_frame in wells.groupby("control", sort=True):
        masks = {
            "overall": np.ones(len(control_frame), dtype=bool),
            "hidden_like_spatial": control_frame[
                "hidden_like_spatial_role"
            ].eq(selected_role).to_numpy(),
            "hidden_like_typewell_purged": control_frame[
                "hidden_like_typewell_purged_role"
            ].eq(selected_role).to_numpy(),
        }
        for surface in SURFACES:
            frame = control_frame.loc[masks[surface]].copy()
            finite = np.isfinite(frame["well_spearman_rmse"].to_numpy(np.float64))
            surface_rows.append(
                {
                    "control": control,
                    "surface": surface,
                    "wells": len(frame),
                    "finite_spearman_wells": int(finite.sum()),
                    "family_rank_spearman": float(
                        frame.loc[finite, "well_spearman_rmse"].mean()
                    ),
                    "family_rank_spearman_median": float(
                        frame.loc[finite, "well_spearman_rmse"].median()
                    ),
                    "group_coverage": float(frame["group_available"].mean()),
                }
            )
    surface_metrics = pd.DataFrame(surface_rows).sort_values(
        ["control", "surface"], kind="mergesort"
    )
    return fold_metrics.reset_index(drop=True), surface_metrics.reset_index(drop=True)


def evaluate_stage_0_gate(
    frozen: FrozenInputs,
    prior_schedule: pd.DataFrame,
    readout: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    surface_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = get_nested(config, "model.stage_0.pass_requires_all") or {}
    real_fold = fold_metrics.loc[
        fold_metrics["control"].eq("real_native_group")
    ].copy()
    real_surface = surface_metrics.loc[
        surface_metrics["control"].eq("real_native_group")
    ].set_index("surface")
    shuffled_surface = surface_metrics.loc[
        surface_metrics["control"].eq(
            "stable_group_label_shuffle_within_fold"
        )
    ].set_index("surface")
    primary = float(real_surface.loc["overall", "family_rank_spearman"])
    shuffled = float(shuffled_surface.loc["overall", "family_rank_spearman"])
    delta = primary - shuffled
    positive_folds = int((real_fold["family_rank_spearman"] > 0.0).sum())
    coverage = float(real_surface.loc["overall", "group_coverage"])
    hidden_spatial = float(
        real_surface.loc["hidden_like_spatial", "family_rank_spearman"]
    )
    hidden_purged = float(
        real_surface.loc["hidden_like_typewell_purged", "family_rank_spearman"]
    )
    prior_columns = [
        "prior_mae",
        "prior_mse",
        "prior_rmse",
        "prior_best_family_rate",
    ]
    all_prior_finite = bool(
        np.isfinite(prior_schedule[prior_columns].to_numpy(np.float64)).all()
    )
    candidate_family_fold_identity_parity = bool(
        tuple(frozen.exp293.bank_manifest["candidate_ids"])
        == EXPECTED_CANDIDATE_ORDER
        and tuple(dict.fromkeys(frozen.family_manifest["family"]))
        == EXPECTED_FAMILY_ORDER
        and set(map(int, frozen.exp293.block_assignment["outer_fold"].unique()))
        == set(range(5))
        and len(readout)
        == (
            len(CONTROLS)
            * int(get_nested(config, "data.exp293.expected_wells"))
            * len(EXPECTED_FAMILY_ORDER)
        )
    )
    checks = {
        "candidate_family_fold_identity_parity": (
            candidate_family_fold_identity_parity
        ),
        "all_prior_values_finite": all_prior_finite,
        "minimum_heldout_group_coverage": coverage
        >= float(gates["minimum_heldout_group_coverage"]),
        "minimum_family_rank_spearman": primary
        >= float(gates["minimum_family_rank_spearman"]),
        "minimum_positive_folds": positive_folds
        >= int(gates["minimum_positive_folds"]),
        "minimum_real_minus_shuffle_spearman": delta
        >= float(gates["minimum_real_minus_shuffle_spearman"]),
        "hidden_like_spatial_spearman_nonnegative": hidden_spatial >= 0.0,
        "hidden_like_typewell_purged_spearman_nonnegative": hidden_purged >= 0.0,
        "outer_valid_truth_before_prior_freeze_zero": int(
            prior_schedule["outer_valid_truth_rows_before_prior_freeze"].max()
        )
        == 0,
        "fit_valid_well_overlap_zero": int(
            prior_schedule["fit_valid_well_overlap"].max()
        )
        == 0,
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "stage_1_eligible": passed,
        "decision": (
            "stage_0_passed_stage_1_requires_separate_user_approval"
            if passed
            else "stage_0_failed_close_without_rescue"
        ),
        "rescue_grid_allowed": False,
        "checks": checks,
        "heldout_group_coverage": coverage,
        "family_rank_spearman": primary,
        "positive_folds": positive_folds,
        "shuffle_family_rank_spearman": shuffled,
        "real_minus_shuffle_spearman": delta,
        "hidden_like_spatial_family_rank_spearman": hidden_spatial,
        "hidden_like_typewell_purged_family_rank_spearman": hidden_purged,
        "rank_score": "shrunk_family_rmse_ascending",
        "rank_aggregation": "mean_within_well_spearman_then_equal_well",
    }


# %% [markdown]
# ## 8. Setup, configuration, and input preview

# %%
CONFIG_PATH = resolve_config_path()
CONFIG = read_yaml(CONFIG_PATH)
EXECUTION_CONTRACT = validate_scientific_contract(CONFIG)

print("Experiment:", get_nested(CONFIG, "experiment.name"))
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Scientific parent:", get_nested(CONFIG, "lineage.parent"))
print("Downstream control:", get_nested(CONFIG, "lineage.downstream_control"))
print("Stage 0 execution contract:", EXECUTION_CONTRACT)
print("Candidate order:", list(EXPECTED_CANDIDATE_ORDER))
print("Family order:", list(EXPECTED_FAMILY_ORDER))
print(
    "Primary readout:",
    get_nested(CONFIG, "validation.rank_score"),
    "/",
    get_nested(CONFIG, "validation.rank_aggregation"),
)
print(
    "Stage 1 implemented / approved:",
    get_nested(CONFIG, "implementation.stage_1_implemented"),
    "/",
    get_nested(CONFIG, "model.stage_1.enabled_condition"),
)
display(build_candidate_family_manifest(CONFIG))


# %% [markdown]
# ## 9. Stage 0 orchestration, metrics, and generated artifacts

# %%
def build_scientific_contract_artifact(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "stage": "stage_0_zero_model_family_prior_readout",
        "scientific_parent": get_nested(config, "lineage.parent"),
        "downstream_control": get_nested(config, "lineage.downstream_control"),
        "candidate_order": list(EXPECTED_CANDIDATE_ORDER),
        "family_order": list(EXPECTED_FAMILY_ORDER),
        "family_by_candidate": EXPECTED_FAMILY_BY_CANDIDATE,
        "family_error_unit": get_nested(config, "validation.family_error_unit"),
        "rank_score": get_nested(config, "validation.rank_score"),
        "rank_aggregation": get_nested(config, "validation.rank_aggregation"),
        "statistics": list(get_nested(config, "model.prior.statistics") or []),
        "shrinkage_support_k_wells": get_nested(
            config, "model.prior.shrinkage_support_k_wells"
        ),
        "fallback_order": list(
            get_nested(config, "model.prior.fallback_order") or []
        ),
        "controls": list(CONTROLS),
        "surfaces": list(SURFACES),
        "execution_contract": validate_scientific_contract(config),
        "stage_1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }


def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_run_approval(config)
    started = time.perf_counter()
    out_dir = output_directory()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(get_nested(config, "artifacts.output_prefix"))
    paths = {
        "family_manifest": out_dir / f"{prefix}_candidate_family_manifest.csv",
        "group_membership": out_dir / f"{prefix}_group_membership.csv",
        "well_family_error": out_dir / f"{prefix}_well_family_error.csv.gz",
        "prior_schedule": out_dir / f"{prefix}_prior_schedule.csv.gz",
        "readout": out_dir / f"{prefix}_readout.csv.gz",
        "fold_metrics": out_dir / f"{prefix}_fold_metrics.csv",
        "surface_metrics": out_dir / f"{prefix}_surface_metrics.csv",
        "gate": out_dir / f"{prefix}_gate.json",
        "input_manifest": out_dir / f"{prefix}_input_manifest.json",
        "scientific_contract": out_dir / f"{prefix}_scientific_contract.json",
        "summary": out_dir / f"{prefix}_summary.json",
        "sha_manifest": out_dir / f"{prefix}_sha_manifest.csv",
    }
    frozen = freeze_target_free_inputs(config)
    print(
        "Target-free input freeze:",
        frozen.target_free_freeze_sha256,
        "| truth rows:",
        frozen.truth_rows_before_freeze,
    )
    scientific_contract = build_scientific_contract_artifact(config)
    output_evidence: dict[str, dict[str, Any]] = {
        "family_manifest": write_csv(paths["family_manifest"], frozen.family_manifest),
        "group_membership": write_csv(
            paths["group_membership"], frozen.group_membership
        ),
        "scientific_contract": write_json(
            paths["scientific_contract"], scientific_contract
        ),
    }
    well_family_error, truth_evidence = compute_well_family_errors(frozen, config)
    output_evidence["well_family_error"] = write_gzip_csv(
        paths["well_family_error"], well_family_error
    )
    prior_schedule, prior_freeze_sha = build_prior_schedule(
        well_family_error, frozen.group_membership, config
    )
    print("Prior schedule freeze:", prior_freeze_sha)
    output_evidence["prior_schedule"] = write_gzip_csv(
        paths["prior_schedule"], prior_schedule
    )
    readout = attach_heldout_errors_after_freeze(
        prior_schedule,
        prior_freeze_sha,
        well_family_error,
        frozen.group_membership,
    )
    fold_metrics, surface_metrics = build_readout_metrics(readout, config)
    gate = evaluate_stage_0_gate(
        frozen,
        prior_schedule,
        readout,
        fold_metrics,
        surface_metrics,
        config,
    )
    output_evidence["readout"] = write_gzip_csv(paths["readout"], readout)
    output_evidence["fold_metrics"] = write_csv(
        paths["fold_metrics"], fold_metrics
    )
    output_evidence["surface_metrics"] = write_csv(
        paths["surface_metrics"], surface_metrics
    )
    output_evidence["gate"] = write_json(paths["gate"], gate)
    completed_input_manifest = {
        **frozen.input_manifest,
        "raw_truth_loaded": True,
        "truth": truth_evidence,
        "prior_schedule_freeze_sha256": prior_freeze_sha,
        "prior_schedule_schema_sha256": dataframe_schema_sha256(prior_schedule),
        "prior_schedule_content_sha256": output_evidence["prior_schedule"][
            "content_sha256"
        ],
        "readout_schema_sha256": dataframe_schema_sha256(readout),
        "readout_content_sha256": output_evidence["readout"]["content_sha256"],
    }
    output_evidence["input_manifest"] = write_json(
        paths["input_manifest"], completed_input_manifest
    )
    runtime_seconds = time.perf_counter() - started
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "ml_model",
        "status": "stage_0_completed",
        "stage": "stage_0_zero_model_family_prior_readout",
        "runtime_seconds": runtime_seconds,
        "execution": {
            **EXECUTION_CONTRACT,
            "gpu": False,
            "internet": False,
            "parent_control_retraining": False,
            "candidate_regeneration": False,
            "stage_1_models": 0,
            "inference": False,
            "submission": False,
        },
        "input": completed_input_manifest,
        "gate": gate,
        "outputs": output_evidence,
    }
    output_evidence["summary"] = write_json(paths["summary"], summary)
    sha_rows = []
    for name, evidence in sorted(output_evidence.items()):
        path = Path(str(evidence["path"]))
        sha_rows.append(
            {
                "artifact": name,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "raw_sha256": sha256_path(path),
                "decompressed_content_sha256": (
                    sha256_decompressed_gzip(path)
                    if path.suffix == ".gz"
                    else ""
                ),
            }
        )
    sha_manifest = pd.DataFrame(sha_rows).sort_values(
        "artifact", kind="mergesort"
    )
    write_csv(paths["sha_manifest"], sha_manifest)
    expected_names = set(get_nested(config, "artifacts.expected_stage_0_artifacts") or [])
    actual_names = {path.name for path in paths.values()}
    if expected_names != actual_names:
        raise ValueError(
            f"Stage 0 artifact contract mismatch: expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )
    print("Stage 0 gate:", json.dumps(gate, indent=2, sort_keys=True))
    display(fold_metrics)
    display(surface_metrics)
    return summary


RUN_RESULT: dict[str, Any] | None = None
if os.environ.get("EXP354_IMPORT_ONLY") != "1":
    validate_run_approval(CONFIG)
    RUN_RESULT = run_stage_0_experiment(CONFIG)
