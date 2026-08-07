# %% [markdown]
# # exp283 self-GR top-K alternative mode branch future-evidence readout
#
# This zero-booster train-side diagnostic separates four target-free ambiguity
# event strata, causal self-GR top-3 proposals, and post-event typewell evidence.
# Event, proposal, and evidence tables are frozen and hashed before true TVT or
# hidden-like roles are read. The notebook never fits a model, changes a decoder,
# creates a corrected prediction, runs inference, or creates a submission.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Safe input readers and canonical target-free identity
# 4. Raw-GR preparation and target-free ambiguity events
# 5. Causal self-GR top-3 proposal generation
# 6. Future typewell evidence and geometry veto
# 7. Post-freeze truth attachment and scientific readouts
# 8. Guard evaluation and generated artifacts
# 9. Full Kaggle CPU orchestration
# 10. Setup, contract preview, and execution

# %% [markdown]
# ## 1. Imports and fixed experiment contract

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

EXPERIMENT_NAME = "exp283_self_gr_topk_alternative_mode_branch_future_evidence_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
TRUTH_TOKENS = ("truth", "true_tvt", "target", "error", "oracle")
EVENT_TRIGGER_COLUMNS = (
    "trigger_exp236_bimodal_segment_end",
    "trigger_exact_hmm_likpf_persistent",
    "trigger_exact_hmm_exp226_persistent",
    "trigger_low_shift_margin",
)
EVENT_CONTENT_COLUMNS = [
    "event_id",
    "well",
    "fold",
    "event_row_idx",
    "suffix_offset",
    "md_since",
    "proposal_start_row_idx",
    "future_start_row_idx",
    "future_end_row_idx_h256",
    *EVENT_TRIGGER_COLUMNS,
    "trigger_names",
    "shift_margin",
    "shift_margin_outer_train_q20",
    "truth_attached",
]
PROPOSAL_CONTENT_COLUMNS = [
    "event_id",
    "well",
    "fold",
    "event_row_idx",
    "branch_rank",
    "donor_source",
    "orientation",
    "donor_row_idx",
    "donor_anchor_tvt",
    "shuffled_donor_row_idx",
    "shuffled_anchor_tvt",
    "ncc17",
    "ncc31",
    "ncc51",
    "multiscale_agreement",
    "base_event_tvt",
    "geop_event_tvt",
    "anchor_shift_ft",
    "shuffled_anchor_shift_ft",
    "proposal_window_end_row_idx",
    "future_start_row_idx",
    "truth_attached",
]
EVIDENCE_CONTENT_COLUMNS = [
    "event_id",
    "well",
    "fold",
    "control",
    "branch_kind",
    "branch_rank",
    "donor_source",
    "orientation",
    "donor_row_idx",
    "anchor_tvt",
    "anchor_shift_ft",
    "horizon_rows",
    "row_count",
    "typewell_log_likelihood_mean",
    "native_typewell_coverage",
    "extended_typewell_coverage",
    "maximum_abs_step_ft",
    "maximum_abs_curvature_ft",
    "donor_receiver_rate_gap",
    "geometry_vetoed",
    "geometry_veto_reason",
    "selected_primary",
    "truth_attached",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP283_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
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
    path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")


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
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp283 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    output = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
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
    payload = json.dumps(to_jsonable(dict(value)), sort_keys=True, separators=(",", ":"))
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
    schema = [(column, str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()


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
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            checked.append(str(path))
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def parse_row_idx_from_id(values: pd.Series) -> np.ndarray:
    split = values.astype(str).str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("canonical id must use <well>_<row_idx>")
    return pd.to_numeric(split[1], errors="raise").to_numpy(np.int32)


def validate_target_free_columns(
    frame: pd.DataFrame,
    *,
    allowed: Iterable[str],
    forbidden: Iterable[str],
    label: str,
) -> None:
    columns = set(frame.columns)
    leaked = sorted(columns.intersection(set(forbidden)))
    unexpected = sorted(columns - set(allowed))
    missing = sorted(set(allowed) - columns)
    if leaked:
        raise ValueError(f"{label} contains forbidden columns: {leaked}")
    if unexpected:
        raise ValueError(f"{label} contains unexpected columns: {unexpected}")
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_freeze_hashes(event_sha: str, proposal_sha: str, evidence_sha: str) -> None:
    for label, value in (
        ("event", event_sha),
        ("proposal", proposal_sha),
        ("evidence", evidence_sha),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"post-freeze attachment requires a 64-character {label} content SHA")


def validate_scientific_contract(config: Mapping[str, Any]) -> None:
    fixed = {
        "experiment.route": "pf_beam",
        "lineage.parent": "exp282_longtail_prediction_zone_self_gr_loop_closure_readout",
        "validation.n_folds": 5,
        "event_detection.persistent_rows": 128,
        "event_detection.shift_margin_block_rows": 128,
        "event_detection.shift_margin_outer_train_quantile": 0.20,
        "event_detection.refractory_rows": 256,
        "proposal.max_alternative_branches": 3,
        "proposal.gr_rolling_mean_rows": 5,
        "proposal.primary_window_rows": 51,
        "proposal.donor_stride_rows": 3,
        "proposal.minimum_prediction_donor_gap_rows": 256,
        "evidence.primary_horizon_rows": 256,
        "model.active_variant_count": 1,
        "model.lightgbm_config_count": 0,
        "model.trained_fold_count": 0,
        "model.booster_count": 0,
        "model.hmm_regeneration_count": 0,
        "model.pf_regeneration_count": 0,
        "execution.total_boosters": 0,
    }
    for key, expected in fixed.items():
        actual = get_nested(config, key)
        if actual != expected:
            raise ValueError(f"fixed exp283 contract changed at {key}: {actual} != {expected}")
    if list(get_nested(config, "proposal.trailing_window_rows")) != [17, 31, 51]:
        raise ValueError("exp283 fixes trailing windows [17, 31, 51]")
    if list(get_nested(config, "proposal.donor_sources")) != [
        "known_prefix",
        "earlier_prediction_zone",
    ]:
        raise ValueError("exp283 fixes known-prefix and earlier-prediction donor order")
    if list(get_nested(config, "proposal.orientations")) != ["forward", "reverse"]:
        raise ValueError("exp283 fixes forward/reverse proposal orientation")
    if list(get_nested(config, "evidence.diagnostic_horizons_rows")) != [128, 512]:
        raise ValueError("exp283 fixes diagnostic horizons [128, 512]")
    if len(list(get_nested(config, "event_detection.shift_bank_ft"))) != 13:
        raise ValueError("exp283 fixes the exp280 13-shift bank")
    forbidden_flags = (
        "model.parent_control_retraining",
        "execution.control_or_parent_retraining",
        "execution.gpu",
        "execution.inference",
        "execution.submission",
        "execution.persist_full_pairwise_matrix",
        "inference.enabled",
        "inference.create_submission",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_flags):
        raise ValueError(
            "exp283 forbids retraining, GPU, inference, submission, and pair persistence"
        )


# %% [markdown]
# ## 3. Safe input readers and canonical target-free identity


# %%
def _check_source_hashes(path: Path, spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    raw_sha = sha256_path(path)
    decompressed_sha = sha256_gzip_decompressed(path) if path.suffix == ".gz" else raw_sha
    expected_raw = str(spec.get("expected_raw_sha256") or "")
    expected_decompressed = str(spec.get("expected_decompressed_sha256") or "")
    if expected_raw and raw_sha != expected_raw:
        raise ValueError(f"{label} raw SHA mismatch: {raw_sha} != {expected_raw}")
    if expected_decompressed and decompressed_sha != expected_decompressed:
        raise ValueError(
            f"{label} decompressed SHA mismatch: {decompressed_sha} != {expected_decompressed}"
        )
    return {
        "name": label,
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
    }


def load_exp226_safe(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(x) for x in spec.get("candidates", [])])
    manifest = _check_source_hashes(path, spec, "exp226_oof_target_free")
    columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    validate_target_free_columns(
        frame,
        allowed=columns,
        forbidden=[str(value) for value in spec.get("forbidden_columns", [])],
        label="exp226 target-free frame",
    )
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    frame["suffix_offset"] = pd.to_numeric(frame["suffix_offset"], errors="raise").astype(np.int32)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 target-free identity contains duplicate well/row keys")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    manifest.update(rows=len(frame), wells=int(frame["well_id"].nunique()))
    return frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    ), manifest


def load_exp236_safe(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp236_row_summary") or {}
    path = resolve_existing(str(spec["filename"]), [str(x) for x in spec.get("candidates", [])])
    manifest = _check_source_hashes(path, spec, "exp236_row_summary_target_free")
    columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=columns, dtype={"id": str, "well": str})
    validate_target_free_columns(
        frame,
        allowed=columns,
        forbidden=[str(value) for value in spec.get("forbidden_columns", [])],
        label="exp236 target-free frame",
    )
    frame["row_idx"] = parse_row_idx_from_id(frame["id"])
    if frame.duplicated(["well", "row_idx"]).any():
        raise ValueError("exp236 safe rows contain duplicate well/row keys")
    manifest.update(rows=len(frame), wells=int(frame["well"].nunique()))
    return frame, manifest


def load_exp209_safe(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.exp209_candidate_cache") or {}
    path = resolve_existing(str(spec["filename"]), [str(x) for x in spec.get("candidates", [])])
    manifest = _check_source_hashes(path, spec, "exp209_candidate_cache_target_free")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    forbidden = set(str(value) for value in spec.get("forbidden_columns", []))
    if forbidden.intersection(spec.get("safe_columns", [])):
        raise ValueError("exp209 safe-column configuration exposes forbidden columns")
    preferred = [str(value) for value in spec.get("safe_columns", [])]
    available = [column for column in preferred if column in header]
    required = {"id", "well", "hmm_mean_tvt"}
    if not required.issubset(available):
        raise ValueError(
            f"exp209 source is missing safe columns {sorted(required - set(available))}"
        )
    if "likpf_mean" not in available and "hmm_minus_likpf_mean" not in available:
        raise ValueError("exp209 source cannot derive likpf_mean")
    frame = pd.read_csv(path, usecols=available, dtype={"id": str, "well": str})
    validate_target_free_columns(
        frame,
        allowed=available,
        forbidden=forbidden,
        label="exp209 target-free frame",
    )
    frame["row_idx"] = parse_row_idx_from_id(frame["id"])
    exact = pd.to_numeric(frame["hmm_mean_tvt"], errors="raise").to_numpy(np.float64)
    if "likpf_mean" in frame:
        likpf = pd.to_numeric(frame["likpf_mean"], errors="raise").to_numpy(np.float64)
        derivation = "direct_likpf_mean"
    else:
        delta = pd.to_numeric(frame["hmm_minus_likpf_mean"], errors="raise").to_numpy(np.float64)
        likpf = exact - delta
        derivation = "hmm_mean_tvt_minus_hmm_minus_likpf_mean"
    output = frame[["well", "row_idx"]].copy()
    output["md_since"] = (
        pd.to_numeric(frame["md_since"], errors="raise").to_numpy(np.float64)
        if "md_since" in frame
        else np.nan
    )
    output["exact_hmm"] = exact
    output["likpf_mean"] = likpf
    if (
        output.duplicated(["well", "row_idx"]).any()
        or not np.isfinite(output[["exact_hmm", "likpf_mean"]].to_numpy(np.float64)).all()
    ):
        raise ValueError("exp209 safe candidates must be unique and finite")
    manifest.update(
        rows=len(output),
        wells=int(output["well"].nunique()),
        likpf_derivation=derivation,
    )
    return output, manifest


def resolve_exp263_cache_root(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp263_base") or {}
    path = resolve_existing(
        "cache_manifest.json", [str(x) for x in spec.get("manifest_candidates", [])]
    )
    actual_sha = sha256_path(path)
    expected_sha = str(spec.get("manifest_sha256") or "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"exp263 manifest SHA mismatch: {actual_sha} != {expected_sha}")
    manifest = json.loads(path.read_text())
    if int(manifest.get("rows", -1)) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp263 manifest canonical row count mismatch")
    if int(manifest.get("wells", -1)) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp263 manifest canonical well count mismatch")
    expected_id_sha = str(spec.get("canonical_id_sha256") or "")
    if expected_id_sha and str(manifest.get("canonical_id_sha256")) != expected_id_sha:
        raise ValueError("exp263 canonical ID SHA mismatch")
    return path.parent, {
        "name": "exp263_stage0_manifest_target_free",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": int(manifest["rows"]),
        "wells": int(manifest["wells"]),
        "canonical_id_sha256": str(manifest["canonical_id_sha256"]),
    }


def _exp263_partition_paths(cache_root: Path, candidate_id: str) -> list[Path]:
    paths = sorted((cache_root / "candidate_values" / candidate_id).glob("fold=*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no exp263 candidate partitions for {candidate_id}")
    return paths


def load_exp263_base(config: Mapping[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cache_root, root_manifest = resolve_exp263_cache_root(config)
    weights = {
        str(key): float(value)
        for key, value in dict(get_nested(config, "data.exp263_base.formula_weights") or {}).items()
    }
    if weights != {"exp226_k16": 0.5, "likpf_mean": 0.25, "exact_hmm": 0.25}:
        raise ValueError("exp263 fixed formula must remain 0.50/0.25/0.25")
    merged: pd.DataFrame | None = None
    manifests: list[dict[str, Any]] = [root_manifest]
    for candidate_id, weight in weights.items():
        paths = _exp263_partition_paths(cache_root, candidate_id)
        parts = [
            pd.read_parquet(path, columns=["well", "well_row_idx", "outer_fold", "candidate_tvt"])
            for path in paths
        ]
        candidate = pd.concat(parts, ignore_index=True)
        candidate["well"] = candidate["well"].astype(str)
        candidate["well_row_idx"] = pd.to_numeric(candidate["well_row_idx"], errors="raise").astype(
            np.int32
        )
        candidate["outer_fold"] = pd.to_numeric(candidate["outer_fold"], errors="raise").astype(
            np.int8
        )
        candidate["candidate_tvt"] = pd.to_numeric(candidate["candidate_tvt"], errors="raise")
        if candidate.duplicated(["well", "well_row_idx"]).any():
            raise ValueError(f"exp263 candidate {candidate_id} contains duplicate keys")
        candidate = candidate.rename(columns={"candidate_tvt": candidate_id})
        selected = candidate[["well", "well_row_idx", "outer_fold", candidate_id]]
        merged = (
            selected
            if merged is None
            else merged.merge(
                selected,
                on=["well", "well_row_idx", "outer_fold"],
                how="inner",
                validate="one_to_one",
            )
        )
        manifests.append(
            {
                "name": f"exp263_candidate_{candidate_id}_target_free",
                "path": str(cache_root / "candidate_values" / candidate_id),
                "files": len(paths),
                "rows": len(candidate),
                "formula_weight": weight,
                "raw_sha256": dataframe_content_sha(
                    pd.DataFrame(
                        {
                            "path": [str(path) for path in paths],
                            "bytes": [path.stat().st_size for path in paths],
                            "sha256": [sha256_path(path) for path in paths],
                        }
                    )
                ),
            }
        )
    assert merged is not None
    if len(merged) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp263 fixed formula coverage mismatch")
    values = np.zeros(len(merged), dtype=np.float64)
    for candidate_id, weight in weights.items():
        values += weight * merged[candidate_id].to_numpy(np.float64)
    output = merged[["well", "well_row_idx", "outer_fold"]].rename(
        columns={"well": "well_id", "well_row_idx": "row_idx", "outer_fold": "base_fold"}
    )
    output["base_tvt"] = values
    if not np.isfinite(output["base_tvt"]).all():
        raise ValueError("exp263 fixed base must be finite")
    return output, manifests


def build_target_free_identity(
    exp226: pd.DataFrame,
    exp209: pd.DataFrame,
    exp236: pd.DataFrame,
    exp263: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    identity = exp226.merge(
        exp209.rename(columns={"well": "well_id"}),
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    posterior_columns = [column for column in exp236.columns if column not in {"id", "well"}]
    identity = identity.merge(
        exp236.rename(columns={"well": "well_id"})[["well_id", *posterior_columns]],
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    identity = identity.merge(exp263, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    numeric = ["tvt_geop", "exact_hmm", "likpf_mean", "base_tvt", "base_fold"]
    if (
        identity[numeric].isna().any().any()
        or not np.isfinite(identity[numeric].to_numpy(np.float64)).all()
    ):
        raise ValueError("canonical target-free identity has missing or non-finite sources")
    # exp263's outer_fold is an independently rebuilt Stage 0 cache partition,
    # not the source-model OOF fold.  Its fixed candidate values are invariant to
    # that storage/evaluation partition, so row-wise equality with exp226.fold is
    # neither expected nor required.  Both labels must still be well-grouped.
    if int(identity.groupby("well_id", sort=False)["fold"].nunique().max()) != 1:
        raise ValueError("exp226 OOF fold must be constant within each well")
    if int(identity.groupby("well_id", sort=False)["base_fold"].nunique().max()) != 1:
        raise ValueError("exp263 Stage 0 partition must be constant within each well")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(identity) != expected_rows or identity["well_id"].nunique() != expected_wells:
        raise ValueError("canonical target-free row/well coverage mismatch")
    if sorted(identity["fold"].astype(int).unique()) != expected_folds:
        raise ValueError("canonical target-free fold coverage mismatch")
    if sorted(identity["base_fold"].astype(int).unique()) != expected_folds:
        raise ValueError("exp263 Stage 0 partition coverage mismatch")
    identity = identity.drop(columns="base_fold")
    return identity.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)


# %% [markdown]
# ## 4. Raw-GR preparation and target-free ambiguity events


# %%
def load_horizontal_score_safe(path: Path) -> pd.DataFrame:
    columns = ["MD", "GR", "TVT_input"]
    frame = pd.read_csv(path, usecols=columns)
    validate_target_free_columns(
        frame,
        allowed=columns,
        forbidden=["TVT", "target", "error", "oracle"],
        label="raw horizontal score-stage frame",
    )
    return frame


def prepare_well_target_free(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_target_free_columns(
        horizontal,
        allowed=["MD", "GR", "TVT_input"],
        forbidden=["TVT", "target", "error", "oracle"],
        label="well target-free input",
    )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    md = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(np.float64)
    known_positions = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_positions) < 4:
        raise ValueError("well needs at least four visible TVT_input rows")
    last_known = int(known_positions[-1])
    if not np.array_equal(known_positions, np.arange(last_known + 1)):
        raise ValueError("TVT_input must be one contiguous visible prefix")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw.to_numpy(np.float64)).all():
        raise ValueError("typewell requires at least two finite sorted TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    prefix_mean = float(raw_gr.iloc[: last_known + 1].mean())
    full_mean = float(raw_gr.mean())
    fallback = prefix_mean if np.isfinite(prefix_mean) else full_mean
    if not np.isfinite(fallback):
        fallback = float(np.mean(typewell_gr))
    gr = raw_gr.interpolate(limit_direction="both").fillna(fallback).to_numpy(np.float64)
    rolling_rows = int(get_nested(config, "proposal.gr_rolling_mean_rows"))
    gr_smooth = (
        pd.Series(gr).rolling(rolling_rows, center=True, min_periods=1).mean().to_numpy(np.float64)
    )
    known_gr = gr[known_positions]
    expected_known = np.interp(tvt_input[known_positions], typewell_tvt, typewell_gr)
    residual = known_gr - expected_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "evidence.emission.sigma_clip")
    ]
    gr_sigma = float(np.clip(np.std(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix GR residual sigma must be finite")
    return {
        "md": md,
        "tvt_input": tvt_input,
        "gr": gr,
        "gr_smooth": gr_smooth,
        "last_known_row": last_known,
        "prediction_start_row": last_known + 1,
        "last_known_md": float(md[last_known]),
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "gr_sigma": gr_sigma,
        "gr_missing_rows": int(raw_gr.isna().sum()),
    }


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    ranks = np.empty(len(values), dtype=np.int32)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int32)
    return ranks


def score_shift_margin_blocks(
    well_rows: pd.DataFrame,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows = well_rows.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    row_idx = rows["row_idx"].to_numpy(np.int64)
    suffix = rows["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix, np.arange(len(rows), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    shifts = np.asarray(get_nested(config, "event_detection.shift_bank_ft"), dtype=np.float64)
    geop = rows["tvt_geop"].to_numpy(np.float64)
    candidate_tvt = geop[:, None] + shifts[None, :]
    expected_gr = np.empty_like(candidate_tvt)
    for slot in range(len(shifts)):
        expected_gr[:, slot] = np.interp(
            candidate_tvt[:, slot], prepared["typewell_tvt"], prepared["typewell_gr"]
        )
    observed = np.asarray(prepared["gr"], dtype=np.float64)[row_idx]
    clip_value = float(get_nested(config, "event_detection.emission.log_likelihood_clip"))
    zscore = (observed[:, None] - expected_gr) / float(prepared["gr_sigma"])
    log_likelihood = -0.5 * np.minimum(zscore**2, clip_value)
    block_rows = int(get_nested(config, "event_detection.shift_margin_block_rows"))
    block_id = suffix // block_rows
    output: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        scores = log_likelihood[mask].mean(axis=0)
        ordered = np.sort(scores)[::-1]
        positions = np.flatnonzero(mask)
        output.append(
            {
                "well": str(rows["well_id"].iloc[0]),
                "fold": int(rows["fold"].iloc[0]),
                "block_id": int(block),
                "event_row_idx": int(row_idx[positions[-1]]),
                "suffix_offset": int(suffix[positions[-1]]),
                "shift_margin": float(ordered[0] - ordered[1]),
                "selected_shift_ft": float(shifts[int(np.argmax(scores))]),
                "score_finite": bool(np.isfinite(scores).all()),
            }
        )
    return pd.DataFrame(output)


def persistent_run_event_rows(
    mask: np.ndarray,
    row_idx: np.ndarray,
    *,
    persistent_rows: int,
) -> list[int]:
    flags = np.asarray(mask, dtype=bool)
    indices = np.asarray(row_idx, dtype=np.int64)
    if len(flags) != len(indices):
        raise ValueError("persistent mask and row identity length differ")
    events: list[int] = []
    start: int | None = None
    for position, active in enumerate(flags):
        contiguous = position == 0 or indices[position] == indices[position - 1] + 1
        if active and (start is None or contiguous):
            if start is None:
                start = position
            if position - start + 1 == persistent_rows:
                events.append(int(indices[position]))
        elif active:
            start = position
        else:
            start = None
    return events


def add_outer_train_margin_thresholds(
    margins: pd.DataFrame,
    *,
    quantile: float,
    expected_folds: Iterable[int],
) -> pd.DataFrame:
    output = margins.copy()
    output["shift_margin_outer_train_q20"] = np.nan
    for fold in expected_folds:
        train_values = output.loc[output["fold"] != int(fold), "shift_margin"]
        if train_values.empty or not np.isfinite(train_values).all():
            raise ValueError(f"fold {fold} has no finite outer-train shift margins")
        threshold = float(train_values.quantile(quantile, interpolation="linear"))
        output.loc[output["fold"] == int(fold), "shift_margin_outer_train_q20"] = threshold
    output["low_shift_margin"] = output["shift_margin"] <= output["shift_margin_outer_train_q20"]
    if output["shift_margin_outer_train_q20"].isna().any():
        raise ValueError("outer-train q20 was not assigned to every margin block")
    return output


def build_target_free_events(
    identity: pd.DataFrame,
    margins: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    persistent_rows = int(get_nested(config, "event_detection.persistent_rows"))
    threshold = float(get_nested(config, "event_detection.disagreement_threshold_ft"))
    refractory = int(get_nested(config, "event_detection.refractory_rows"))
    primary_window = int(get_nested(config, "proposal.primary_window_rows"))
    future_rows = int(get_nested(config, "event_detection.require_primary_future_rows"))
    event_map: dict[tuple[str, int], set[str]] = {}
    margin_lookup: dict[tuple[str, int], tuple[float, float]] = {}

    def add_event(well: str, row: int, trigger: str) -> None:
        event_map.setdefault((str(well), int(row)), set()).add(trigger)

    for well, part in identity.groupby("well_id", sort=True):
        local = part.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        row_idx = local["row_idx"].to_numpy(np.int64)
        bimodal = local.loc[local["bimodal_flag"].astype(bool) & (local["bimodal_segment_id"] >= 0)]
        for _, segment in bimodal.groupby("bimodal_segment_id", sort=True):
            add_event(str(well), int(segment["row_idx"].max()), EVENT_TRIGGER_COLUMNS[0])
        for row in persistent_run_event_rows(
            np.abs(local["exact_hmm"].to_numpy() - local["likpf_mean"].to_numpy()) >= threshold,
            row_idx,
            persistent_rows=persistent_rows,
        ):
            add_event(str(well), row, EVENT_TRIGGER_COLUMNS[1])
        for row in persistent_run_event_rows(
            np.abs(local["exact_hmm"].to_numpy() - local["tvt_geop"].to_numpy()) >= threshold,
            row_idx,
            persistent_rows=persistent_rows,
        ):
            add_event(str(well), row, EVENT_TRIGGER_COLUMNS[2])

    for row in margins.loc[margins["low_shift_margin"]].itertuples(index=False):
        add_event(str(row.well), int(row.event_row_idx), EVENT_TRIGGER_COLUMNS[3])
        margin_lookup[(str(row.well), int(row.event_row_idx))] = (
            float(row.shift_margin),
            float(row.shift_margin_outer_train_q20),
        )

    identity_lookup = identity.set_index(["well_id", "row_idx"], verify_integrity=True)
    rows: list[dict[str, Any]] = []
    raw_candidate_count = len(event_map)
    boundary_eligible_count = 0
    for well in sorted({key[0] for key in event_map}):
        candidates = sorted(row for item_well, row in event_map if item_well == well)
        kept: list[int] = []
        max_row = int(identity.loc[identity["well_id"] == well, "row_idx"].max())
        for event_row in candidates:
            if event_row - primary_window + 1 < 0 or event_row + future_rows > max_row:
                continue
            boundary_eligible_count += 1
            if kept and event_row - kept[-1] <= refractory:
                continue
            kept.append(event_row)
            item = identity_lookup.loc[(well, event_row)]
            triggers = event_map[(well, event_row)]
            margin, margin_q20 = margin_lookup.get((well, event_row), (np.nan, np.nan))
            rows.append(
                {
                    "event_id": f"{well}_{event_row}",
                    "well": well,
                    "fold": int(item["fold"]),
                    "event_row_idx": event_row,
                    "suffix_offset": int(item["suffix_offset"]),
                    "md_since": float(item["md_since"]),
                    "proposal_start_row_idx": event_row - primary_window + 1,
                    "future_start_row_idx": event_row + 1,
                    "future_end_row_idx_h256": event_row + future_rows,
                    **{column: column in triggers for column in EVENT_TRIGGER_COLUMNS},
                    "trigger_names": ",".join(
                        column.removeprefix("trigger_")
                        for column in EVENT_TRIGGER_COLUMNS
                        if column in triggers
                    ),
                    "shift_margin": margin,
                    "shift_margin_outer_train_q20": margin_q20,
                    "truth_attached": False,
                }
            )
    events = pd.DataFrame(rows, columns=EVENT_CONTENT_COLUMNS)
    if events.empty:
        raise ValueError("fixed event contract produced zero eligible events")
    if events.duplicated(["well", "event_row_idx"]).any() or events["event_id"].duplicated().any():
        raise ValueError("target-free event identity must be unique")
    if events[list(EVENT_TRIGGER_COLUMNS)].sum(axis=1).min() < 1:
        raise ValueError("each event must retain at least one target-free stratum")
    event_sha = assert_frozen_event_contract(events)
    return events.sort_values(["well", "event_row_idx"], kind="mergesort").reset_index(drop=True), {
        "raw_event_candidates": raw_candidate_count,
        "boundary_eligible_candidates": boundary_eligible_count,
        "events_after_refractory": len(events),
        "event_content_sha256": event_sha,
    }


def assert_frozen_event_contract(events: pd.DataFrame) -> str:
    missing = sorted(set(EVENT_CONTENT_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"event table is missing {missing}")
    leaked = sorted(
        column
        for column in events.columns
        if any(token in column.lower() for token in TRUTH_TOKENS) and column != "truth_attached"
    )
    if leaked:
        raise ValueError(f"target-free events contain forbidden truth columns: {leaked}")
    if bool(events["truth_attached"].astype(bool).any()):
        raise ValueError("target-free events cannot have truth_attached=true")
    return dataframe_content_sha(events, EVENT_CONTENT_COLUMNS)


# %% [markdown]
# ## 5. Causal self-GR top-3 proposal generation


# %%
def normalize_trailing_windows(
    signal: np.ndarray, ends: np.ndarray, window_rows: int
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    positions = np.asarray(ends, dtype=np.int64)
    offsets = np.arange(-window_rows + 1, 1, dtype=np.int64)
    if len(positions) == 0:
        return np.empty((0, window_rows), dtype=np.float64)
    if positions.min() - window_rows + 1 < 0 or positions.max() >= len(values):
        raise ValueError("trailing window endpoint is outside the signal")
    windows = values[positions[:, None] + offsets[None, :]]
    means = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    normalized = (windows - means) / (std + 1.0e-6)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized trailing GR windows must be finite")
    return normalized


def score_proposal_bank(
    signal: np.ndarray,
    event_row: int,
    donor_rows: np.ndarray,
    *,
    windows: Iterable[int],
) -> pd.DataFrame:
    donors = np.asarray(donor_rows, dtype=np.int64)
    if len(donors) == 0:
        return pd.DataFrame()
    output = pd.DataFrame(
        {
            "donor_row_idx": np.concatenate([donors, donors]),
            "orientation": ["forward"] * len(donors) + ["reverse"] * len(donors),
        }
    )
    for window in windows:
        donor_matrix = normalize_trailing_windows(signal, donors, int(window))
        receiver = normalize_trailing_windows(signal, np.asarray([event_row]), int(window))[0]
        forward = donor_matrix @ receiver / float(window)
        reverse = donor_matrix[:, ::-1] @ receiver / float(window)
        output[f"ncc{int(window)}"] = np.concatenate([forward, reverse])
    output["multiscale_agreement"] = output[["ncc17", "ncc31"]].mean(axis=1)
    return output


def _anchor_for_rows(
    source: str,
    donor_rows: np.ndarray,
    prepared: Mapping[str, Any],
    base_lookup: Mapping[int, float],
) -> np.ndarray:
    rows = np.asarray(donor_rows, dtype=np.int64)
    if source == "known_prefix":
        anchors = np.asarray(prepared["tvt_input"], dtype=np.float64)[rows]
    elif source == "earlier_prediction_zone":
        anchors = np.asarray([base_lookup[int(row)] for row in rows], dtype=np.float64)
    else:
        raise ValueError(f"unknown donor source {source}")
    if not np.isfinite(anchors).all():
        raise ValueError(f"{source} donor anchors must be finite")
    return anchors


def build_proposals_for_event(
    event: Mapping[str, Any],
    prepared: Mapping[str, Any],
    well_identity: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_row = int(event["event_row_idx"])
    primary = int(get_nested(config, "proposal.primary_window_rows"))
    stride = int(get_nested(config, "proposal.donor_stride_rows"))
    gap = int(get_nested(config, "proposal.minimum_prediction_donor_gap_rows"))
    windows = [int(value) for value in get_nested(config, "proposal.trailing_window_rows")]
    last_known = int(prepared["last_known_row"])
    prediction_start = int(prepared["prediction_start_row"])
    first_end = primary - 1
    known = np.arange(first_end, last_known + 1, stride, dtype=np.int64)
    prediction_first = max(prediction_start + primary - 1, first_end)
    prediction_last = event_row - gap
    prediction = (
        np.arange(prediction_first, prediction_last + 1, stride, dtype=np.int64)
        if prediction_last >= prediction_first
        else np.asarray([], dtype=np.int64)
    )
    base_lookup = dict(
        zip(
            well_identity["row_idx"].astype(int),
            well_identity["base_tvt"].astype(float),
            strict=True,
        )
    )
    event_item = well_identity.loc[well_identity["row_idx"] == event_row]
    if len(event_item) != 1:
        raise ValueError("event row is missing from canonical target-free identity")
    base_event = float(event_item["base_tvt"].iloc[0])
    geop_event = float(event_item["tvt_geop"].iloc[0])
    banks = {"known_prefix": known, "earlier_prediction_zone": prediction}
    candidate_parts: list[pd.DataFrame] = []
    for source in get_nested(config, "proposal.donor_sources"):
        donor_rows = banks[str(source)]
        scored = score_proposal_bank(
            np.asarray(prepared["gr_smooth"]), event_row, donor_rows, windows=windows
        )
        if scored.empty:
            continue
        scored["donor_source"] = str(source)
        scored["donor_anchor_tvt"] = _anchor_for_rows(
            str(source), scored["donor_row_idx"].to_numpy(np.int64), prepared, base_lookup
        )
        candidate_parts.append(scored)
    if not candidate_parts:
        return pd.DataFrame(columns=PROPOSAL_CONTENT_COLUMNS), {
            "event_id": str(event["event_id"]),
            "known_prefix_bank": len(known),
            "prediction_zone_bank": len(prediction),
            "ranked_candidates": 0,
            "selected_proposals": 0,
        }
    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates["source_priority"] = candidates["donor_source"].map(
        {"known_prefix": 0, "earlier_prediction_zone": 1}
    )
    candidates["orientation_priority"] = candidates["orientation"].map({"forward": 0, "reverse": 1})
    candidates = candidates.sort_values(
        [
            "ncc51",
            "multiscale_agreement",
            "source_priority",
            "orientation_priority",
            "donor_row_idx",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected_rows: list[pd.Series] = []
    dedup_rows = int(get_nested(config, "proposal.donor_center_dedup_rows"))
    dedup_anchor = float(get_nested(config, "proposal.anchor_dedup_ft"))
    maximum = int(get_nested(config, "proposal.max_alternative_branches"))
    for _, candidate in candidates.iterrows():
        if any(
            abs(int(candidate["donor_row_idx"]) - int(existing["donor_row_idx"])) <= dedup_rows
            or abs(float(candidate["donor_anchor_tvt"]) - float(existing["donor_anchor_tvt"]))
            <= dedup_anchor
            for existing in selected_rows
        ):
            continue
        selected_rows.append(candidate)
        if len(selected_rows) == maximum:
            break
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    if selected.empty:
        return pd.DataFrame(columns=PROPOSAL_CONTENT_COLUMNS), {
            "event_id": str(event["event_id"]),
            "known_prefix_bank": len(known),
            "prediction_zone_bank": len(prediction),
            "ranked_candidates": len(candidates),
            "selected_proposals": 0,
        }
    shuffled_rows: list[int] = []
    shuffled_anchors: list[float] = []
    seed = int(get_nested(config, "negative_control.seed"))
    for source, part in selected.groupby("donor_source", sort=True):
        bank = banks[str(source)]
        local_seed = stable_seed(
            EXPERIMENT_NAME, seed, event["well"], event_row, source, "donor_shuffle"
        )
        permutation = np.random.default_rng(local_seed).permutation(bank)
        assignment = dict(zip(bank.tolist(), permutation.tolist(), strict=True))
        for index, row in part.iterrows():
            shuffled_row = int(assignment[int(row["donor_row_idx"])])
            shuffled_rows.append((int(index), shuffled_row))
            anchor = float(
                _anchor_for_rows(str(source), np.asarray([shuffled_row]), prepared, base_lookup)[0]
            )
            shuffled_anchors.append((int(index), anchor))
    shuffled_row_map = dict(shuffled_rows)
    shuffled_anchor_map = dict(shuffled_anchors)
    output_rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        anchor = float(row["donor_anchor_tvt"])
        shuffled_anchor = float(shuffled_anchor_map[int(index)])
        output_rows.append(
            {
                "event_id": str(event["event_id"]),
                "well": str(event["well"]),
                "fold": int(event["fold"]),
                "event_row_idx": event_row,
                "branch_rank": int(index + 1),
                "donor_source": str(row["donor_source"]),
                "orientation": str(row["orientation"]),
                "donor_row_idx": int(row["donor_row_idx"]),
                "donor_anchor_tvt": anchor,
                "shuffled_donor_row_idx": int(shuffled_row_map[int(index)]),
                "shuffled_anchor_tvt": shuffled_anchor,
                "ncc17": float(row["ncc17"]),
                "ncc31": float(row["ncc31"]),
                "ncc51": float(row["ncc51"]),
                "multiscale_agreement": float(row["multiscale_agreement"]),
                "base_event_tvt": base_event,
                "geop_event_tvt": geop_event,
                "anchor_shift_ft": anchor - base_event,
                "shuffled_anchor_shift_ft": shuffled_anchor - base_event,
                "proposal_window_end_row_idx": event_row,
                "future_start_row_idx": event_row + 1,
                "truth_attached": False,
            }
        )
    output = pd.DataFrame(output_rows, columns=PROPOSAL_CONTENT_COLUMNS)
    return output, {
        "event_id": str(event["event_id"]),
        "known_prefix_bank": len(known),
        "prediction_zone_bank": len(prediction),
        "ranked_candidates": len(candidates),
        "selected_proposals": len(output),
    }


def assert_frozen_proposal_contract(proposals: pd.DataFrame) -> str:
    missing = sorted(set(PROPOSAL_CONTENT_COLUMNS) - set(proposals.columns))
    if missing:
        raise ValueError(f"proposal table is missing {missing}")
    if bool(proposals["truth_attached"].astype(bool).any()):
        raise ValueError("target-free proposals cannot have truth_attached=true")
    if proposals.duplicated(["event_id", "branch_rank"]).any():
        raise ValueError("proposal branch identity must be unique")
    numeric = [
        "donor_anchor_tvt",
        "shuffled_anchor_tvt",
        "ncc17",
        "ncc31",
        "ncc51",
        "multiscale_agreement",
        "base_event_tvt",
        "geop_event_tvt",
        "anchor_shift_ft",
        "shuffled_anchor_shift_ft",
    ]
    if not np.isfinite(proposals[numeric].to_numpy(np.float64)).all():
        raise ValueError("target-free proposals contain non-finite scores or anchors")
    return dataframe_content_sha(proposals, PROPOSAL_CONTENT_COLUMNS)


# %% [markdown]
# ## 6. Future typewell evidence and geometry veto


# %%
def _branch_path(
    branch_kind: str,
    anchor_tvt: float,
    future: pd.DataFrame,
    geop_event_tvt: float,
) -> np.ndarray:
    if branch_kind == "base":
        return future["base_tvt"].to_numpy(np.float64)
    return anchor_tvt + future["tvt_geop"].to_numpy(np.float64) - geop_event_tvt


def _mean_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.diff(array))) if len(array) >= 2 else 0.0


def build_future_evidence_for_event(
    event: Mapping[str, Any],
    event_proposals: pd.DataFrame,
    prepared: Mapping[str, Any],
    well_identity: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    event_row = int(event["event_row_idx"])
    identity = well_identity.set_index("row_idx")
    if not identity.index.is_unique:
        raise ValueError("well target-free row identity must be unique")
    horizons = [
        int(get_nested(config, "evidence.primary_horizon_rows")),
        *[int(value) for value in get_nested(config, "evidence.diagnostic_horizons_rows")],
    ]
    typewell_tvt = np.asarray(prepared["typewell_tvt"], dtype=np.float64)
    typewell_gr = np.asarray(prepared["typewell_gr"], dtype=np.float64)
    observed_gr = np.asarray(prepared["gr"], dtype=np.float64)
    extension = float(get_nested(config, "evidence.geometry_veto.typewell_extension_ft"))
    max_shift = float(get_nested(config, "evidence.geometry_veto.maximum_anchor_shift_ft"))
    clip_value = float(get_nested(config, "evidence.emission.log_likelihood_clip"))
    primary = int(get_nested(config, "evidence.primary_horizon_rows"))
    geop_event = float(identity.loc[event_row, "tvt_geop"])
    base_event = float(identity.loc[event_row, "base_tvt"])
    receiver_start = max(int(prepared["prediction_start_row"]), event_row - 50)
    receiver_rate = _mean_rate(
        identity.loc[np.arange(receiver_start, event_row + 1, dtype=np.int64), "tvt_geop"].to_numpy(
            np.float64
        )
    )
    rows: list[dict[str, Any]] = []
    branches: dict[str, list[dict[str, Any]]] = {"real": [], "shuffled": []}
    for control in branches:
        branches[control].append(
            {
                "branch_kind": "base",
                "branch_rank": 0,
                "donor_source": "base",
                "orientation": "base",
                "donor_row_idx": event_row,
                "anchor_tvt": base_event,
                "anchor_shift_ft": 0.0,
                "donor_rate": receiver_rate,
            }
        )
    for proposal in event_proposals.itertuples(index=False):
        for control, donor_column, anchor_column, shift_column in (
            ("real", "donor_row_idx", "donor_anchor_tvt", "anchor_shift_ft"),
            (
                "shuffled",
                "shuffled_donor_row_idx",
                "shuffled_anchor_tvt",
                "shuffled_anchor_shift_ft",
            ),
        ):
            donor_row = int(getattr(proposal, donor_column))
            source = str(proposal.donor_source)
            if source == "known_prefix":
                donor_series = np.asarray(prepared["tvt_input"], dtype=np.float64)
                start = max(0, donor_row - 50)
                donor_rate = _mean_rate(donor_series[start : donor_row + 1])
            else:
                donor_positions = np.arange(
                    max(int(prepared["prediction_start_row"]), donor_row - 50), donor_row + 1
                )
                donor_rate = _mean_rate(
                    identity.loc[donor_positions, "base_tvt"].to_numpy(np.float64)
                )
            branches[control].append(
                {
                    "branch_kind": "alternative",
                    "branch_rank": int(proposal.branch_rank),
                    "donor_source": source,
                    "orientation": str(proposal.orientation),
                    "donor_row_idx": donor_row,
                    "anchor_tvt": float(getattr(proposal, anchor_column)),
                    "anchor_shift_ft": float(getattr(proposal, shift_column)),
                    "donor_rate": donor_rate,
                }
            )
    maximum_available_row = int(identity.index.max())
    for control, branch_specs in branches.items():
        primary_scores: list[tuple[int, float, bool]] = []
        control_rows: list[dict[str, Any]] = []
        for branch in branch_specs:
            for horizon in horizons:
                end_row = event_row + horizon
                if end_row > maximum_available_row:
                    continue
                positions = np.arange(event_row + 1, end_row + 1, dtype=np.int64)
                future = identity.loc[positions].reset_index()
                path = _branch_path(
                    str(branch["branch_kind"]),
                    float(branch["anchor_tvt"]),
                    future,
                    geop_event,
                )
                expected_gr = np.interp(path, typewell_tvt, typewell_gr)
                zscore = (observed_gr[positions] - expected_gr) / float(prepared["gr_sigma"])
                likelihood = -0.5 * np.minimum(zscore**2, clip_value)
                steps = np.diff(np.r_[float(branch["anchor_tvt"]), path])
                curvature = np.diff(steps)
                native = (path >= typewell_tvt.min()) & (path <= typewell_tvt.max())
                extended = (path >= typewell_tvt.min() - extension) & (
                    path <= typewell_tvt.max() + extension
                )
                reasons: list[str] = []
                if not np.isfinite(path).all() or not np.isfinite(steps).all():
                    reasons.append("non_finite_path_or_step")
                if (
                    str(branch["branch_kind"]) == "alternative"
                    and abs(float(branch["anchor_shift_ft"])) > max_shift
                ):
                    reasons.append("anchor_shift_gt_80ft")
                if not bool(extended.all()):
                    reasons.append("outside_typewell_plus_40ft")
                vetoed = bool(reasons) and str(branch["branch_kind"]) != "base"
                row = {
                    "event_id": str(event["event_id"]),
                    "well": str(event["well"]),
                    "fold": int(event["fold"]),
                    "control": control,
                    "branch_kind": str(branch["branch_kind"]),
                    "branch_rank": int(branch["branch_rank"]),
                    "donor_source": str(branch["donor_source"]),
                    "orientation": str(branch["orientation"]),
                    "donor_row_idx": int(branch["donor_row_idx"]),
                    "anchor_tvt": float(branch["anchor_tvt"]),
                    "anchor_shift_ft": float(branch["anchor_shift_ft"]),
                    "horizon_rows": horizon,
                    "row_count": len(positions),
                    "typewell_log_likelihood_mean": float(np.mean(likelihood)),
                    "native_typewell_coverage": float(native.mean()),
                    "extended_typewell_coverage": float(extended.mean()),
                    "maximum_abs_step_ft": float(np.max(np.abs(steps))),
                    "maximum_abs_curvature_ft": float(np.max(np.abs(curvature)))
                    if len(curvature)
                    else 0.0,
                    "donor_receiver_rate_gap": float(branch["donor_rate"] - receiver_rate),
                    "geometry_vetoed": vetoed,
                    "geometry_veto_reason": ",".join(reasons),
                    "selected_primary": False,
                    "truth_attached": False,
                }
                control_rows.append(row)
                if horizon == primary:
                    primary_scores.append(
                        (
                            int(branch["branch_rank"]),
                            float(row["typewell_log_likelihood_mean"]),
                            vetoed,
                        )
                    )
        eligible = [item for item in primary_scores if not item[2]]
        selected_rank = max(eligible, key=lambda item: (item[1], -item[0]))[0] if eligible else 0
        for row in control_rows:
            row["selected_primary"] = int(row["branch_rank"]) == selected_rank
        rows.extend(control_rows)
    return pd.DataFrame(rows, columns=EVIDENCE_CONTENT_COLUMNS)


def assert_frozen_evidence_contract(evidence: pd.DataFrame) -> str:
    missing = sorted(set(EVIDENCE_CONTENT_COLUMNS) - set(evidence.columns))
    if missing:
        raise ValueError(f"future-evidence table is missing {missing}")
    if bool(evidence["truth_attached"].astype(bool).any()):
        raise ValueError("target-free future evidence cannot have truth_attached=true")
    identity = ["event_id", "control", "branch_rank", "horizon_rows"]
    if evidence.duplicated(identity).any():
        raise ValueError("future-evidence branch identity must be unique")
    finite = [
        "anchor_tvt",
        "anchor_shift_ft",
        "typewell_log_likelihood_mean",
        "native_typewell_coverage",
        "extended_typewell_coverage",
        "maximum_abs_step_ft",
        "maximum_abs_curvature_ft",
        "donor_receiver_rate_gap",
    ]
    if not np.isfinite(evidence[finite].to_numpy(np.float64)).all():
        raise ValueError("future-evidence table contains non-finite values")
    primary = evidence[evidence["horizon_rows"] == 256]
    selected_counts = primary.groupby(["event_id", "control"])["selected_primary"].sum()
    if not (selected_counts == 1).all():
        raise ValueError("each event/control must select exactly one H256 branch")
    return dataframe_content_sha(evidence, EVIDENCE_CONTENT_COLUMNS)


# %% [markdown]
# ## 7. Post-freeze truth attachment and scientific readouts


# %%
def load_hidden_like_postfreeze(
    config: Mapping[str, Any],
    *,
    event_sha: str,
    proposal_sha: str,
    evidence_sha: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_freeze_hashes(event_sha, proposal_sha, evidence_sha)
    spec = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing(str(spec["filename"]), [str(x) for x in spec.get("candidates", [])])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec.get("expected_sha256")):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str}).rename(columns={"well_id": "well"})
    roles = [str(value) for value in (spec.get("role_columns") or {}).values()]
    required = {"well", *roles}
    if not required.issubset(frame.columns) or frame["well"].duplicated().any():
        raise ValueError("hidden-like assignment schema/identity mismatch")
    return frame[["well", *roles]], {
        "name": "exp115_hidden_like_post_freeze",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
    }


def load_truth_for_event_wells(
    raw_dir: Path,
    wells: Iterable[str],
    *,
    event_sha: str,
    proposal_sha: str,
    evidence_sha: str,
) -> dict[str, np.ndarray]:
    require_freeze_hashes(event_sha, proposal_sha, evidence_sha)
    output: dict[str, np.ndarray] = {}
    for well in sorted(set(str(value) for value in wells)):
        path = raw_dir / f"{well}__horizontal_well.csv"
        truth = pd.read_csv(path, usecols=["TVT"])
        values = pd.to_numeric(truth["TVT"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"post-freeze truth contains non-finite values for {well}")
        output[well] = values
    return output


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    value = np.asarray(scores, dtype=np.float64)
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = pd.Series(value).rank(method="average").to_numpy(np.float64)
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _rmse_from_mse(values: pd.Series) -> float:
    return float(np.sqrt(pd.to_numeric(values, errors="raise").mean()))


def _set_unique_index(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    output = frame.set_index(column)
    if not output.index.is_unique:
        raise ValueError(f"{label} index must be unique")
    return output


def build_postfreeze_readouts(
    events: pd.DataFrame,
    proposals: pd.DataFrame,
    evidence: pd.DataFrame,
    identity: pd.DataFrame,
    truth_by_well: Mapping[str, np.ndarray],
    hidden: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    event_sha: str,
    proposal_sha: str,
    evidence_sha: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_freeze_hashes(event_sha, proposal_sha, evidence_sha)
    primary = int(get_nested(config, "evidence.primary_horizon_rows"))
    identity_by_well = {
        str(well): _set_unique_index(part, "row_idx", f"{well} target-free row")
        for well, part in identity.groupby("well_id", sort=True)
    }
    hidden_lookup = _set_unique_index(hidden, "well", "hidden-like well")
    event_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        well = str(event.well)
        event_id = str(event.event_id)
        event_row = int(event.event_row_idx)
        positions = np.arange(event_row + 1, event_row + primary + 1, dtype=np.int64)
        future = identity_by_well[well].loc[positions].reset_index()
        truth = np.asarray(truth_by_well[well], dtype=np.float64)
        future_truth = truth[positions]
        event_truth = float(truth[event_row])
        event_proposals = proposals.loc[proposals["event_id"] == event_id].sort_values(
            "branch_rank"
        )
        real_anchor_error = np.abs(
            event_proposals["donor_anchor_tvt"].to_numpy(np.float64) - event_truth
        )
        shuffled_anchor_error = np.abs(
            event_proposals["shuffled_anchor_tvt"].to_numpy(np.float64) - event_truth
        )
        real_h256 = evidence.loc[
            (evidence["event_id"] == event_id)
            & (evidence["control"] == "real")
            & (evidence["horizon_rows"] == primary)
        ].sort_values("branch_rank")
        shuffled_h256 = evidence.loc[
            (evidence["event_id"] == event_id)
            & (evidence["control"] == "shuffled")
            & (evidence["horizon_rows"] == primary)
        ].sort_values("branch_rank")
        branch_metrics: dict[int, dict[str, float]] = {}
        geop_event = float(identity_by_well[well].loc[event_row, "tvt_geop"])
        for row in real_h256.itertuples(index=False):
            path = _branch_path(str(row.branch_kind), float(row.anchor_tvt), future, geop_event)
            mse = float(np.mean((path - future_truth) ** 2))
            branch_metrics[int(row.branch_rank)] = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "score": float(row.typewell_log_likelihood_mean),
            }
        shuffled_metrics: dict[int, float] = {}
        for row in shuffled_h256.itertuples(index=False):
            path = _branch_path(str(row.branch_kind), float(row.anchor_tvt), future, geop_event)
            shuffled_metrics[int(row.branch_rank)] = float(np.mean((path - future_truth) ** 2))
        base_mse = branch_metrics[0]["mse"]
        selected_rank = int(real_h256.loc[real_h256["selected_primary"], "branch_rank"].iloc[0])
        shuffled_selected_rank = int(
            shuffled_h256.loc[shuffled_h256["selected_primary"], "branch_rank"].iloc[0]
        )
        selected_mse = branch_metrics[selected_rank]["mse"]
        shuffled_selected_mse = shuffled_metrics[shuffled_selected_rank]
        ordered_truth = sorted(branch_metrics, key=lambda rank: (branch_metrics[rank]["mse"], rank))
        unique_best = len(ordered_truth) == 1 or (
            branch_metrics[ordered_truth[1]]["mse"] - branch_metrics[ordered_truth[0]]["mse"]
            > 1.0e-12
        )
        base_unique_best = unique_best and ordered_truth[0] == 0
        score_order = (
            real_h256.sort_values(
                ["geometry_vetoed", "typewell_log_likelihood_mean", "branch_rank"],
                ascending=[True, False, True],
                kind="mergesort",
            )["branch_rank"]
            .astype(int)
            .tolist()
        )
        truth_best_rank = score_order.index(ordered_truth[0]) + 1
        for row in real_h256.loc[real_h256["branch_kind"] == "alternative"].itertuples(index=False):
            metrics = branch_metrics[int(row.branch_rank)]
            pair_rows.append(
                {
                    "event_id": event_id,
                    "well": well,
                    "fold": int(event.fold),
                    "branch_rank": int(row.branch_rank),
                    "donor_source": str(row.donor_source),
                    "orientation": str(row.orientation),
                    "alternative_better_than_base": metrics["mse"] < base_mse,
                    "score_margin_vs_base": float(
                        row.typewell_log_likelihood_mean - branch_metrics[0]["score"]
                    ),
                    "alternative_rmse": metrics["rmse"],
                    "base_rmse": branch_metrics[0]["rmse"],
                    "rmse_gain_vs_base": branch_metrics[0]["rmse"] - metrics["rmse"],
                }
            )
        hidden_row = (
            hidden_lookup.loc[well] if well in hidden_lookup.index else pd.Series(dtype=object)
        )
        horizon_metrics: dict[str, float] = {}
        for horizon in (128, 512):
            real_horizon = evidence.loc[
                (evidence["event_id"] == event_id)
                & (evidence["control"] == "real")
                & (evidence["horizon_rows"] == horizon)
            ].sort_values("branch_rank")
            shuffled_horizon = evidence.loc[
                (evidence["event_id"] == event_id)
                & (evidence["control"] == "shuffled")
                & (evidence["horizon_rows"] == horizon)
            ].sort_values("branch_rank")
            if real_horizon.empty or shuffled_horizon.empty:
                horizon_metrics.update(
                    {
                        f"base_mse_h{horizon}": np.nan,
                        f"selected_mse_h{horizon}": np.nan,
                        f"shuffled_selected_mse_h{horizon}": np.nan,
                    }
                )
                continue
            horizon_positions = np.arange(event_row + 1, event_row + horizon + 1, dtype=np.int64)
            horizon_future = identity_by_well[well].loc[horizon_positions].reset_index()
            horizon_truth = truth[horizon_positions]
            real_rows = _set_unique_index(real_horizon, "branch_rank", "real horizon branch")
            shuffled_rows = _set_unique_index(
                shuffled_horizon, "branch_rank", "shuffled horizon branch"
            )
            base_row = real_rows.loc[0]
            selected_row = real_rows.loc[selected_rank]
            shuffled_selected_row = shuffled_rows.loc[shuffled_selected_rank]
            base_path = _branch_path(
                str(base_row["branch_kind"]),
                float(base_row["anchor_tvt"]),
                horizon_future,
                geop_event,
            )
            selected_path = _branch_path(
                str(selected_row["branch_kind"]),
                float(selected_row["anchor_tvt"]),
                horizon_future,
                geop_event,
            )
            shuffled_selected_path = _branch_path(
                str(shuffled_selected_row["branch_kind"]),
                float(shuffled_selected_row["anchor_tvt"]),
                horizon_future,
                geop_event,
            )
            horizon_metrics.update(
                {
                    f"base_mse_h{horizon}": float(np.mean((base_path - horizon_truth) ** 2)),
                    f"selected_mse_h{horizon}": float(
                        np.mean((selected_path - horizon_truth) ** 2)
                    ),
                    f"shuffled_selected_mse_h{horizon}": float(
                        np.mean((shuffled_selected_path - horizon_truth) ** 2)
                    ),
                }
            )
        event_rows.append(
            {
                "event_id": event_id,
                "well": well,
                "fold": int(event.fold),
                "event_row_idx": event_row,
                "md_since": float(event.md_since),
                "event_true_tvt": event_truth,
                **{column: bool(getattr(event, column)) for column in EVENT_TRIGGER_COLUMNS},
                "proposal_count": len(event_proposals),
                "real_top1_within2": bool(len(real_anchor_error) and real_anchor_error[0] <= 2.0),
                "real_top3_within2": bool(np.any(real_anchor_error <= 2.0)),
                "real_top3_within5": bool(np.any(real_anchor_error <= 5.0)),
                "real_top3_within10": bool(np.any(real_anchor_error <= 10.0)),
                "shuffled_top3_within10": bool(np.any(shuffled_anchor_error <= 10.0)),
                "proposal_mrr_within10": float(
                    1.0 / (np.flatnonzero(real_anchor_error <= 10.0)[0] + 1)
                )
                if np.any(real_anchor_error <= 10.0)
                else 0.0,
                "shuffled_mrr_within10": float(
                    1.0 / (np.flatnonzero(shuffled_anchor_error <= 10.0)[0] + 1)
                )
                if np.any(shuffled_anchor_error <= 10.0)
                else 0.0,
                "base_mse_h256": base_mse,
                "selected_mse_h256": selected_mse,
                "shuffled_selected_mse_h256": shuffled_selected_mse,
                **horizon_metrics,
                "selected_branch_rank": selected_rank,
                "truth_best_branch_rank": ordered_truth[0],
                "truth_best_mrr": float(1.0 / truth_best_rank),
                "oracle_best_mse_h256": branch_metrics[ordered_truth[0]]["mse"],
                "base_unique_best": base_unique_best,
                "false_switch": base_unique_best and selected_rank != 0,
                "hidden_like_spatial": hidden_row.get("verification_like_spatial_role") == "valid",
                "hidden_like_typewell_purged": hidden_row.get(
                    "verification_like_typewell_purged_role"
                )
                == "valid",
                "longtail_1000_plus": float(event.md_since) >= 1000.0,
            }
        )
    return pd.DataFrame(event_rows), pd.DataFrame(pair_rows)


def proposal_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        return {"scope": scope, "events": 0}
    real = float(frame["real_top3_within10"].mean())
    shuffled = float(frame["shuffled_top3_within10"].mean())
    return {
        "scope": scope,
        "events": len(frame),
        "wells": int(frame["well"].nunique()),
        "top1_within2": float(frame["real_top1_within2"].mean()),
        "top3_within2": float(frame["real_top3_within2"].mean()),
        "top3_within5": float(frame["real_top3_within5"].mean()),
        "top3_within10": real,
        "shuffled_top3_within10": shuffled,
        "top3_within10_lift_vs_shuffled": real - shuffled,
        "proposal_mrr_within10": float(frame["proposal_mrr_within10"].mean()),
        "shuffled_mrr_within10": float(frame["shuffled_mrr_within10"].mean()),
    }


def verifier_metric_row(
    events: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    scope: str,
) -> dict[str, Any]:
    if events.empty:
        return {"scope": scope, "events": 0}
    event_ids = set(events["event_id"].astype(str))
    selected_pairs = pairs.loc[pairs["event_id"].astype(str).isin(event_ids)]
    auc = (
        binary_auc(
            selected_pairs["alternative_better_than_base"].astype(np.int8).to_numpy(),
            selected_pairs["score_margin_vs_base"].to_numpy(np.float64),
        )
        if len(selected_pairs)
        else float("nan")
    )
    base_rmse = _rmse_from_mse(events["base_mse_h256"])
    selected_rmse = _rmse_from_mse(events["selected_mse_h256"])
    unique = events.loc[events["base_unique_best"]]
    return {
        "scope": scope,
        "events": len(events),
        "pairs": len(selected_pairs),
        "branch_choice_auc": auc,
        "base_rmse_h256": base_rmse,
        "selected_rmse_h256": selected_rmse,
        "selected_rmse_gain_ft": base_rmse - selected_rmse,
        "shuffled_selected_rmse_h256": _rmse_from_mse(events["shuffled_selected_mse_h256"]),
        "truth_best_mrr": float(events["truth_best_mrr"].mean()),
        "oracle_best_rmse_h256": _rmse_from_mse(events["oracle_best_mse_h256"]),
        "base_unique_best_events": len(unique),
        "base_unique_best_false_switch_rate": float(unique["false_switch"].mean())
        if len(unique)
        else float("nan"),
    }


def build_metric_tables(
    event_readout: pd.DataFrame,
    pair_readout: pd.DataFrame,
    proposals: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    proposal_rows = [proposal_metric_row(event_readout, scope="overall")]
    verifier_rows = [verifier_metric_row(event_readout, pair_readout, scope="overall_h256")]
    for horizon in (128, 512):
        base_column = f"base_mse_h{horizon}"
        selected_column = f"selected_mse_h{horizon}"
        shuffled_column = f"shuffled_selected_mse_h{horizon}"
        available = event_readout.dropna(subset=[base_column, selected_column, shuffled_column])
        if available.empty:
            verifier_rows.append({"scope": f"overall_h{horizon}", "events": 0})
        else:
            base_rmse = _rmse_from_mse(available[base_column])
            selected_rmse = _rmse_from_mse(available[selected_column])
            verifier_rows.append(
                {
                    "scope": f"overall_h{horizon}",
                    "events": len(available),
                    "base_rmse_horizon": base_rmse,
                    "selected_rmse_horizon": selected_rmse,
                    "selected_rmse_gain_ft": base_rmse - selected_rmse,
                    "shuffled_selected_rmse_horizon": _rmse_from_mse(available[shuffled_column]),
                }
            )
    fold_rows: list[dict[str, Any]] = []
    for fold, part in event_readout.groupby("fold", sort=True):
        proposal = proposal_metric_row(part, scope=f"fold_{int(fold)}")
        verifier = verifier_metric_row(part, pair_readout, scope=f"fold_{int(fold)}")
        fold_rows.append(
            {
                "fold": int(fold),
                **proposal,
                **{f"verifier_{key}": value for key, value in verifier.items() if key != "scope"},
            }
        )
    scope_rows: list[dict[str, Any]] = []
    for column in (
        *EVENT_TRIGGER_COLUMNS,
        "longtail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ):
        part = event_readout.loc[event_readout[column].astype(bool)]
        proposal = proposal_metric_row(part, scope=column)
        verifier = verifier_metric_row(part, pair_readout, scope=column)
        scope_rows.append(
            {
                **proposal,
                **{f"verifier_{key}": value for key, value in verifier.items() if key != "scope"},
            }
        )
    source_rows: list[dict[str, Any]] = []
    proposal_truth = proposals.merge(
        event_readout[["event_id", "event_true_tvt", "truth_best_branch_rank"]],
        on="event_id",
        how="left",
    )
    proposal_truth["candidate_within10"] = (
        np.abs(proposal_truth["donor_anchor_tvt"] - proposal_truth["event_true_tvt"]) <= 10.0
    )
    proposal_truth["shuffled_candidate_within10"] = (
        np.abs(proposal_truth["shuffled_anchor_tvt"] - proposal_truth["event_true_tvt"]) <= 10.0
    )
    for (source, orientation), part in proposal_truth.groupby(
        ["donor_source", "orientation"], sort=True
    ):
        source_rows.append(
            {
                "scope": f"proposal_{source}_{orientation}",
                "events": int(part["event_id"].nunique()),
                "proposals": len(part),
                "mean_ncc51": float(part["ncc51"].mean()),
                "mean_multiscale_agreement": float(part["multiscale_agreement"].mean()),
                "candidate_within10": float(part["candidate_within10"].mean()),
                "shuffled_candidate_within10": float(part["shuffled_candidate_within10"].mean()),
                "candidate_within10_lift_vs_shuffled": float(
                    part["candidate_within10"].mean() - part["shuffled_candidate_within10"].mean()
                ),
            }
        )
    by_well_rows: list[dict[str, Any]] = []
    for well, part in event_readout.groupby("well", sort=True):
        proposal = proposal_metric_row(part, scope=str(well))
        verifier = verifier_metric_row(part, pair_readout, scope=str(well))
        by_well_rows.append(
            {
                "well": str(well),
                **proposal,
                **{f"verifier_{key}": value for key, value in verifier.items() if key != "scope"},
            }
        )
    return {
        "proposal_metrics": pd.DataFrame(proposal_rows),
        "verifier_metrics": pd.DataFrame(verifier_rows),
        "fold_metrics": pd.DataFrame(fold_rows),
        "scope_metrics": pd.DataFrame([*scope_rows, *source_rows]),
        "by_well_metrics": pd.DataFrame(by_well_rows),
    }


# %% [markdown]
# ## 8. Guard evaluation and generated artifacts


# %%
def evaluate_guards(
    identity: pd.DataFrame,
    events: pd.DataFrame,
    proposals: pd.DataFrame,
    evidence: pd.DataFrame,
    event_readout: pd.DataFrame,
    pair_readout: pd.DataFrame,
    metric_tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.guards") or {}
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    event_coverage = float(proposals["event_id"].nunique() / len(events)) if len(events) else 0.0
    proposal_finite = (
        float(
            np.isfinite(
                proposals[
                    ["ncc17", "ncc31", "ncc51", "donor_anchor_tvt", "shuffled_anchor_tvt"]
                ].to_numpy(np.float64)
            ).mean()
        )
        if len(proposals)
        else 0.0
    )
    evidence_finite = (
        float(np.isfinite(evidence["typewell_log_likelihood_mean"].to_numpy(np.float64)).mean())
        if len(evidence)
        else 0.0
    )
    proposal_overall = metric_tables["proposal_metrics"].iloc[0]
    verifier_overall = (
        metric_tables["verifier_metrics"]
        .loc[metric_tables["verifier_metrics"]["scope"] == "overall_h256"]
        .iloc[0]
    )
    fold_metrics = metric_tables["fold_metrics"]
    proposal_positive_folds = int((fold_metrics["top3_within10_lift_vs_shuffled"] > 0).sum())
    fold_aucs = {
        int(row.fold): float(row.verifier_branch_choice_auc)
        for row in fold_metrics.itertuples(index=False)
    }
    auc_each_fold = sorted(fold_aucs) == expected_folds and all(
        np.isfinite(value) and value >= float(guards["minimum_branch_choice_auc_each_fold"])
        for value in fold_aucs.values()
    )
    nonregressing_folds = int((fold_metrics["verifier_selected_rmse_gain_ft"] >= 0.0).sum())
    false_switch = float(verifier_overall["base_unique_best_false_switch_rate"])
    technical_checks = {
        "canonical_rows": len(identity) == int(get_nested(config, "validation.expected_rows")),
        "canonical_wells": identity["well_id"].nunique()
        == int(get_nested(config, "validation.expected_wells")),
        "canonical_identity_unique": not identity.duplicated(["well_id", "row_idx"]).any(),
        "event_identity_unique": not events["event_id"].duplicated().any(),
        "proposal_branch_identity_unique": not proposals.duplicated(
            ["event_id", "branch_rank"]
        ).any(),
        "evidence_branch_identity_unique": not evidence.duplicated(
            ["event_id", "control", "branch_rank", "horizon_rows"]
        ).any(),
        "event_coverage": event_coverage >= float(guards["required_event_coverage"]),
        "proposal_finite_coverage": proposal_finite
        >= float(guards["required_proposal_finite_coverage"]),
        "evidence_finite_coverage": evidence_finite
        >= float(guards["required_evidence_finite_coverage"]),
        "truth_before_freeze_zero": int(guards["required_truth_attachment_before_freeze"]) == 0
        and not events["truth_attached"].astype(bool).any()
        and not proposals["truth_attached"].astype(bool).any()
        and not evidence["truth_attached"].astype(bool).any(),
        "fold_coverage": sorted(event_readout["fold"].astype(int).unique()) == expected_folds,
    }
    scientific_checks = {
        "proposal_lift_pooled": float(proposal_overall["top3_within10_lift_vs_shuffled"])
        >= float(guards["minimum_top3_within10_lift_vs_shuffled"]),
        "proposal_lift_positive_5of5": proposal_positive_folds
        >= int(guards["minimum_positive_proposal_lift_folds"]),
        "branch_choice_auc_each_fold": auc_each_fold,
        "selected_h256_gain_pooled": float(verifier_overall["selected_rmse_gain_ft"])
        >= float(guards["minimum_selected_h256_rmse_gain_ft"]),
        "selected_nonregressing_5of5": nonregressing_folds
        >= int(guards["minimum_nonregressing_selected_folds"]),
        "base_unique_best_false_switch": np.isfinite(false_switch)
        and false_switch <= float(guards["maximum_base_unique_best_false_switch_rate"]),
    }
    return {
        "technical_passed": all(technical_checks.values()),
        "scientific_passed": all(scientific_checks.values()),
        "passed": all(technical_checks.values()) and all(scientific_checks.values()),
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "event_coverage": event_coverage,
        "proposal_finite_coverage": proposal_finite,
        "evidence_finite_coverage": evidence_finite,
        "proposal_positive_folds": proposal_positive_folds,
        "fold_aucs": fold_aucs,
        "nonregressing_folds": nonregressing_folds,
        "base_unique_best_false_switch_rate": false_switch,
    }


# %% [markdown]
# ## 9. Full Kaggle CPU orchestration


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp283 readout must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run."
        )
    if not bool(get_nested(config, "execution.kaggle_push_approved")):
        raise RuntimeError("exp283 Kaggle CPU execution is not approved")
    validate_scientific_contract(config)
    started = time.time()
    exp226, exp226_manifest = load_exp226_safe(config)
    exp209, exp209_manifest = load_exp209_safe(config)
    exp236, exp236_manifest = load_exp236_safe(config)
    exp263, exp263_manifests = load_exp263_base(config)
    identity = build_target_free_identity(exp226, exp209, exp236, exp263, config)
    raw_dir = train_data_dir(config)
    raw_wells = sorted(
        path.name.removesuffix("__horizontal_well.csv")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    if raw_wells != sorted(identity["well_id"].unique()):
        raise ValueError("raw train and canonical target-free well sets differ")

    margin_parts: list[pd.DataFrame] = []
    raw_manifest_rows: list[dict[str, Any]] = []
    for ordinal, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        horizontal = load_horizontal_score_safe(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        prepared = prepare_well_target_free(horizontal, typewell, config)
        margin_parts.append(
            score_shift_margin_blocks(identity.loc[identity["well_id"] == well], prepared, config)
        )
        raw_manifest_rows.append(
            {
                "well": well,
                "horizontal_rows": len(horizontal),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
                "gr_missing_rows": int(prepared["gr_missing_rows"]),
                "gr_sigma": float(prepared["gr_sigma"]),
            }
        )
        if ordinal % 25 == 0 or ordinal == len(raw_wells):
            print(f"shift-margin wells={ordinal}/{len(raw_wells)}")
    margins = add_outer_train_margin_thresholds(
        pd.concat(margin_parts, ignore_index=True),
        quantile=float(get_nested(config, "event_detection.shift_margin_outer_train_quantile")),
        expected_folds=get_nested(config, "validation.expected_folds"),
    )
    events, event_stats = build_target_free_events(identity, margins, config)
    event_sha = assert_frozen_event_contract(events)
    artifacts = artifact_dir()
    event_artifact = write_csv_gzip(
        events, artifacts / f"{OUTPUT_PREFIX}_target_free_events.csv.gz"
    )
    event_contract = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "event_detection": get_nested(config, "event_detection"),
        "proposal": get_nested(config, "proposal"),
        "evidence": get_nested(config, "evidence"),
        "event_stats": event_stats,
        "event_schema_sha256": dataframe_schema_sha(events),
        "event_content_sha256": event_sha,
    }
    event_contract["scientific_contract_sha256"] = mapping_sha256(event_contract)
    event_contract_path = artifacts / f"{OUTPUT_PREFIX}_event_contract.json"
    write_json(event_contract_path, event_contract)

    proposal_parts: list[pd.DataFrame] = []
    proposal_manifest_rows: list[dict[str, Any]] = []
    for well, well_events in events.groupby("well", sort=True):
        horizontal = load_horizontal_score_safe(raw_dir / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(raw_dir / f"{well}__typewell.csv")
        prepared = prepare_well_target_free(horizontal, typewell, config)
        well_identity = identity.loc[identity["well_id"] == well]
        for event in well_events.to_dict(orient="records"):
            proposals, manifest = build_proposals_for_event(event, prepared, well_identity, config)
            proposal_parts.append(proposals)
            proposal_manifest_rows.append(manifest)
    proposals = pd.concat(proposal_parts, ignore_index=True)
    proposal_sha = assert_frozen_proposal_contract(proposals)
    proposal_artifact = write_csv_gzip(
        proposals, artifacts / f"{OUTPUT_PREFIX}_target_free_proposals.csv.gz"
    )

    evidence_parts: list[pd.DataFrame] = []
    for well, well_events in events.groupby("well", sort=True):
        horizontal = load_horizontal_score_safe(raw_dir / f"{well}__horizontal_well.csv")
        typewell = pd.read_csv(raw_dir / f"{well}__typewell.csv")
        prepared = prepare_well_target_free(horizontal, typewell, config)
        well_identity = identity.loc[identity["well_id"] == well]
        for event in well_events.to_dict(orient="records"):
            event_proposals = proposals.loc[proposals["event_id"] == event["event_id"]]
            evidence_parts.append(
                build_future_evidence_for_event(
                    event, event_proposals, prepared, well_identity, config
                )
            )
    evidence = pd.concat(evidence_parts, ignore_index=True)
    evidence_sha = assert_frozen_evidence_contract(evidence)
    evidence_artifact = write_csv_gzip(
        evidence, artifacts / f"{OUTPUT_PREFIX}_target_free_future_evidence.csv.gz"
    )

    # True TVT and hidden-like roles are first read here, after all three target-free tables freeze.
    truth = load_truth_for_event_wells(
        raw_dir,
        events["well"],
        event_sha=event_sha,
        proposal_sha=proposal_sha,
        evidence_sha=evidence_sha,
    )
    hidden, hidden_manifest = load_hidden_like_postfreeze(
        config,
        event_sha=event_sha,
        proposal_sha=proposal_sha,
        evidence_sha=evidence_sha,
    )
    event_readout, pair_readout = build_postfreeze_readouts(
        events,
        proposals,
        evidence,
        identity,
        truth,
        hidden,
        config,
        event_sha=event_sha,
        proposal_sha=proposal_sha,
        evidence_sha=evidence_sha,
    )
    metric_tables = build_metric_tables(event_readout, pair_readout, proposals)
    guard = evaluate_guards(
        identity,
        events,
        proposals,
        evidence,
        event_readout,
        pair_readout,
        metric_tables,
        config,
    )

    output_paths: dict[str, Path] = {}
    for name, frame in metric_tables.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path
    event_readout_artifact = write_csv_gzip(
        event_readout, artifacts / f"{OUTPUT_PREFIX}_postfreeze_event_readout.csv.gz"
    )
    pair_readout_artifact = write_csv_gzip(
        pair_readout, artifacts / f"{OUTPUT_PREFIX}_postfreeze_branch_pair_readout.csv.gz"
    )
    raw_manifest = pd.DataFrame(raw_manifest_rows).sort_values("well", kind="mergesort")
    proposal_manifest = pd.DataFrame(proposal_manifest_rows).sort_values(
        "event_id", kind="mergesort"
    )
    raw_manifest_path = artifacts / f"{OUTPUT_PREFIX}_well_manifest.csv"
    proposal_manifest_path = artifacts / f"{OUTPUT_PREFIX}_proposal_manifest.csv"
    raw_manifest.to_csv(raw_manifest_path, index=False)
    proposal_manifest.to_csv(proposal_manifest_path, index=False)
    input_manifest = pd.DataFrame(
        [
            exp226_manifest,
            exp209_manifest,
            exp236_manifest,
            *exp263_manifests,
            hidden_manifest,
            {
                "name": "raw_train_well_files",
                "path": str(raw_dir),
                "rows": int(raw_manifest["horizontal_rows"].sum()),
                "wells": len(raw_manifest),
                "raw_sha256": dataframe_content_sha(
                    raw_manifest,
                    ["well", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
            },
        ]
    )
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)
    output_paths.update(
        input_manifest=input_manifest_path,
        well_manifest=raw_manifest_path,
        proposal_manifest=proposal_manifest_path,
    )
    overall_proposal = metric_tables["proposal_metrics"].iloc[0].to_dict()
    overall_verifier = (
        metric_tables["verifier_metrics"]
        .loc[metric_tables["verifier_metrics"]["scope"] == "overall_h256"]
        .iloc[0]
        .to_dict()
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "train_side_readout_completed_guard_passed"
        if guard["passed"]
        else "train_side_readout_completed_guard_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "canonical_rows": len(identity),
        "canonical_wells": int(identity["well_id"].nunique()),
        "events": len(events),
        "proposals": len(proposals),
        "evidence_rows": len(evidence),
        "active_audit_variants": 1,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "proposal": overall_proposal,
        "verifier": overall_verifier,
        "guard": guard,
        "freeze": {
            "truth_attachment_before_freeze": 0,
            "event_content_sha256": event_sha,
            "proposal_content_sha256": proposal_sha,
            "evidence_content_sha256": evidence_sha,
        },
        "artifacts": {
            "event_contract": str(event_contract_path),
            "events": event_artifact,
            "proposals": proposal_artifact,
            "future_evidence": evidence_artifact,
            "postfreeze_event_readout": event_readout_artifact,
            "postfreeze_branch_pair_readout": pair_readout_artifact,
            "file_sha256": {name: sha256_path(path) for name, path in output_paths.items()},
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": "allow_exp284_implementation_review"
        if guard["passed"]
        else "close_without_rescue_grid_or_decoder_connection",
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "route": "pf_beam",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "diagnostic": {
            "proposal": overall_proposal,
            "verifier": overall_verifier,
            "guard": guard,
            "freeze": summary["freeze"],
        },
        "notes": "No model, corrected prediction, inference, or submission is produced.",
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
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "stage": get_nested(CONFIG, "execution.stage"),
                "event_strata": list(EVENT_TRIGGER_COLUMNS),
                "max_alternative_branches": get_nested(CONFIG, "proposal.max_alternative_branches"),
                "primary_horizon_rows": get_nested(CONFIG, "evidence.primary_horizon_rows"),
                "active_audit_variants": get_nested(CONFIG, "execution.active_audit_variants"),
                "lightgbm_configs": get_nested(CONFIG, "execution.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "execution.trained_fold_count"),
                "boosters": get_nested(CONFIG, "execution.total_boosters"),
                "hmm_well_runs": get_nested(CONFIG, "execution.hmm_well_runs"),
                "pf_well_runs": get_nested(CONFIG, "execution.pf_well_runs"),
                "inference": get_nested(CONFIG, "execution.inference"),
                "submission": get_nested(CONFIG, "execution.submission"),
            },
            indent=2,
        )
    )

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP283_SUMMARY = run_full_experiment(CONFIG)
