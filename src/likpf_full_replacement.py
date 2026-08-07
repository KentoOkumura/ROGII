from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    FoldBundle,
    build_candidate_long_features,
    build_raw_context,
    candidate_ids,
    compact_feature_names,
    contract_by_id,
    deterministic_sample_indices,
    load_stage_d_compact_fold,
    logical_frame_sha256,
    read_yaml,
    sha256_file,
    sha256_json,
    validate_candidate_contract,
    verify_exp263_root,
    verify_stage_c_artifact_root,
    write_json,
)
from src.signed_residual_meta import (
    load_signed_compact_fold,
    signed_compact_feature_names,
    verify_signed_stage_s_root,
)

SEMANTIC_SLOT = "likpf_mean"
REPLACEMENT_VALUE_SOURCE = "likpf_scale_5_x1p0"
CHANGED_CANDIDATES = (
    "likpf_mean",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
UNCHANGED_CANDIDATES = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__exact_hmm",
)
MODEL_STAGE_NAMES = (
    "nested_selector_train",
    "signed_selector_train",
    "downstream_gpu_train",
)
FROZEN_PREDICTION_COLUMNS = (
    "likpf_scale_5_x1p0",
    "likpf_scale_5_x1p3",
    "likpf_mean_x1p0",
    "likpf_mean_x1p3",
)
FROZEN_PREDICTION_SCHEMA = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "last_known_tvt",
    "md_since",
    "raw_gr_observed",
    "well_missing_fraction",
    *FROZEN_PREDICTION_COLUMNS,
)


def _nested_get(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    chosen = list(frame.columns) if columns is None else [str(item) for item in columns]
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
    payload = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    return sha256_json(payload)


def inspect_gzip_csv(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(path, nrows=0, compression="gzip").columns.astype(str).tolist(),
    }


def resolve_by_patterns(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    marker_sha256: str = "",
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.exists():
            candidates.append(direct)
        if direct.is_absolute():
            continue
        for root in search_roots:
            if root.exists():
                candidates.extend(root.glob(raw))
    for candidate in dict.fromkeys(candidates):
        if not candidate.exists():
            continue
        if marker_sha256 and (
            not candidate.is_file() or sha256_file(candidate) != marker_sha256
        ):
            continue
        return candidate
    raise FileNotFoundError(f"no frozen input matches patterns={list(patterns)}")


def replacement_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    variants = [str(item) for item in model["active_variants"]]
    if variants != ["scale5_x1p0_full_replacement"]:
        raise ValueError("exp413 must contain exactly one scale5 x1.0 replacement variant")
    nested = dict(model["nested_selector"])
    signed = dict(model["signed_selector"])
    downstream = dict(model["downstream_tvt"])
    nested_count = (
        int(nested["outer_folds"])
        * int(nested["inner_folds"])
        * len(nested["objectives"])
    )
    signed_count = int(signed["outer_folds"]) * int(signed["inner_folds"])
    downstream_count = len(downstream["lightgbm_config_indices"]) * int(
        downstream["folds"]
    )
    expected = dict(model["execution_count"])
    observed = {
        "replacement_variants": len(variants),
        "cpu_selector_boosters": nested_count,
        "cpu_signed_selector_boosters": signed_count,
        "gpu_downstream_boosters": downstream_count,
        "total_boosters": nested_count + signed_count + downstream_count,
        "parent_control_retraining_boosters": 0,
        "train_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
    }
    for key, value in observed.items():
        if int(expected[key]) != value:
            raise ValueError(f"exp413 cost contract changed: {key}={expected[key]} != {value}")
    if nested_count != 40 or signed_count != 20 or downstream_count != 15:
        raise ValueError("exp413 must remain 40 CPU + 20 CPU + 15 GPU boosters")
    if [int(item) for item in downstream["lightgbm_config_indices"]] != [0, 1, 2]:
        raise ValueError("exp413 must keep downstream LightGBM configs 0, 1, and 2")
    return observed


def validate_replacement_contract(
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    validate_candidate_contract(contract)
    ids = candidate_ids(contract)
    replacement = dict(config["replacement"])
    bank = dict(replacement["candidate_bank"])
    changed = tuple(str(item) for item in bank["changed_slots"])
    unchanged = tuple(str(item) for item in bank["unchanged_slots"])
    if changed != CHANGED_CANDIDATES or unchanged != UNCHANGED_CANDIDATES:
        raise ValueError("exp413 changed/unchanged candidate inventory differs from steering")
    if len(ids) != 12 or set(ids) != set(changed).union(unchanged):
        raise ValueError("exp413 replacement inventory must cover exactly the fixed 12 candidates")
    if str(replacement["semantic_slot_id"]) != SEMANTIC_SLOT:
        raise ValueError("exp413 semantic slot must remain likpf_mean")
    if str(replacement["new_value_source"]) != REPLACEMENT_VALUE_SOURCE:
        raise ValueError("exp413 replacement source must remain likpf_scale_5_x1p0")
    if bool(replacement["old_value_allowed_in_candidate_or_model_input"]):
        raise ValueError("old arithmetic mean cannot be a candidate or model input")
    specs = contract_by_id(contract)
    expected_formulas = {
        str(name): [float(value) for value in weights]
        for name, weights in bank["formula_weights"].items()
    }
    for name, expected_weights in expected_formulas.items():
        observed_weights = [float(value) for value in specs[name]["weights"]]
        if observed_weights != expected_weights:
            raise ValueError(f"exp413 formula weights changed for {name}")
    cost = replacement_cost_contract(config)
    return {
        "candidate_order": ids,
        "candidate_count": len(ids),
        "changed_candidates": list(changed),
        "unchanged_candidates": list(unchanged),
        "semantic_slot": SEMANTIC_SLOT,
        "value_source": REPLACEMENT_VALUE_SOURCE,
        "cost_contract": cost,
    }


def require_stage_authorization(config: Mapping[str, Any], stage: str) -> None:
    if not bool(_nested_get(config, "authorization.implementation_approved", False)):
        raise RuntimeError("exp413 implementation approval is missing")
    if stage == "replacement_preflight":
        approved = bool(_nested_get(config, "authorization.stage_0_run_approved", False))
    elif stage == "nested_selector_train":
        approved = bool(_nested_get(config, "authorization.selector_train_approved", False))
    elif stage == "signed_selector_train":
        approved = bool(
            _nested_get(config, "authorization.signed_selector_train_approved", False)
        )
    elif stage == "downstream_gpu_train":
        approved = bool(
            _nested_get(config, "authorization.downstream_gpu_train_approved", False)
        )
    else:
        raise ValueError(f"unknown exp413 train stage: {stage}")
    run_flag = bool(_nested_get(config, f"execution.run_flags.{stage}", False))
    if not approved or not run_flag:
        raise RuntimeError(
            f"{stage} requires its authorization flag and execution.run_flags.{stage}=true"
        )


def load_frozen_scale5_predictions(
    path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(config["data"]["exp404_scale5_train_prediction"])
    report = inspect_gzip_csv(path)
    if report["raw_sha256"] != str(spec["expected_raw_sha256"]):
        raise ValueError("exp404 replacement raw gzip SHA mismatch")
    if report["decompressed_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp404 replacement decompressed SHA mismatch")
    if report["columns"] != list(FROZEN_PREDICTION_SCHEMA):
        raise ValueError("exp404 replacement column order differs from frozen schema")
    if int(report["data_rows"]) != int(spec["expected_rows"]):
        raise ValueError("exp404 replacement row count differs from frozen contract")

    frame = pd.read_csv(path, dtype={"id": str, "well_id": str}, compression="gzip")
    frame["id"] = frame["id"].astype(object)
    frame["well_id"] = frame["well_id"].astype(object)
    for column in ("row_idx", "suffix_offset"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["raw_gr_observed"] = frame["raw_gr_observed"].astype(bool)
    for column in (
        "last_known_tvt",
        "md_since",
        "well_missing_fraction",
        *FROZEN_PREDICTION_COLUMNS,
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
    if list(frame.columns) != list(FROZEN_PREDICTION_SCHEMA):
        raise ValueError("exp404 replacement parsed schema changed")
    if (
        len(frame) != int(spec["expected_rows"])
        or frame["well_id"].nunique() != int(spec["expected_wells"])
        or frame["id"].duplicated().any()
        or frame.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError("exp404 replacement identity or coverage mismatch")
    if not np.isfinite(frame[list(FROZEN_PREDICTION_COLUMNS)].to_numpy(np.float64)).all():
        raise ValueError("exp404 replacement predictions contain non-finite values")
    logical_columns = ["id", "well_id", "row_idx", *FROZEN_PREDICTION_COLUMNS]
    logical_sha = dataframe_content_sha(frame, logical_columns)
    if logical_sha != str(spec["expected_logical_sha256"]):
        raise ValueError("exp404 replacement logical SHA mismatch")
    schema_sha = dataframe_schema_sha(frame)
    if schema_sha != str(spec["expected_schema_sha256"]):
        raise ValueError("exp404 replacement schema SHA mismatch")
    evidence = {
        **report,
        "logical_columns": logical_columns,
        "logical_sha256": logical_sha,
        "schema_sha256": schema_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "frozen_before_truth_fold_or_hidden_like_attachment": True,
        "generated_pf_well_runs_in_exp413": 0,
        "source_kernel_id": spec["source_kernel_id"],
        "source_kernel_version": int(spec["source_kernel_version"]),
    }
    return frame, evidence


def _read_partition(root: Path, kind: str, candidate_id: str, fold: int) -> pd.DataFrame:
    paths = sorted((Path(root) / kind / candidate_id / f"fold={fold}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"missing {kind}/{candidate_id}/fold={fold} under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _sort_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["well", "well_row_idx"], kind="stable").reset_index(drop=True)


def _assert_replacement_alignment(
    parent: pd.DataFrame,
    replacement: pd.DataFrame,
) -> None:
    if len(parent) != len(replacement):
        raise ValueError("replacement and parent fold row count mismatch")
    if not parent["id"].astype(str).reset_index(drop=True).equals(
        replacement["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("replacement and parent id order mismatch")
    if not parent["well"].astype(str).reset_index(drop=True).equals(
        replacement["well_id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("replacement and parent well identity mismatch")
    if not np.array_equal(
        parent["well_row_idx"].to_numpy(np.int64),
        replacement["row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("replacement and parent row index mismatch")
    for column in ("last_known_tvt", "md_since"):
        parent_float32 = parent[column].to_numpy(np.float32)
        replacement_float32 = replacement[column].to_numpy(np.float32)
        if not np.array_equal(parent_float32, replacement_float32):
            delta = np.abs(
                parent_float32.astype(np.float64)
                - replacement_float32.astype(np.float64)
            )
            raise ValueError(
                f"replacement and parent {column} mismatch at float32 cache precision "
                f"(max_abs={float(delta.max(initial=0.0))})"
            )


def build_bank_from_primitives(
    primitive_values: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> np.ndarray:
    specs = contract_by_id(contract)
    ids = candidate_ids(contract)
    values: dict[str, np.ndarray] = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in primitive_values.items()
    }
    for name in ids:
        if name in values:
            continue
        spec = specs[name]
        parents = [str(item) for item in spec["parents"]]
        weights = np.asarray(spec["weights"], dtype=np.float32)
        combined = np.zeros_like(values[parents[0]], dtype=np.float32)
        for parent, weight in zip(parents, weights, strict=True):
            combined = (
                combined + np.float32(weight) * values[parent].astype(np.float32, copy=False)
            ).astype(np.float32)
        values[name] = combined
    return np.column_stack([values[name] for name in ids]).astype(np.float32)


def _formula_parity_max_abs(
    bank: np.ndarray,
    contract: Mapping[str, Any],
    formula_ids: Sequence[str],
) -> float:
    ids = candidate_ids(contract)
    id_to_pos = {name: position for position, name in enumerate(ids)}
    values = np.asarray(bank, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(ids):
        raise ValueError("candidate bank shape differs from frozen contract")
    maximum = 0.0
    specs = contract_by_id(contract)
    for name in formula_ids:
        spec = specs[name]
        parents = [str(item) for item in spec["parents"]]
        weights = np.asarray(spec["weights"], dtype=np.float32)
        reconstructed = np.zeros(values.shape[0], dtype=np.float32)
        for parent, weight in zip(parents, weights, strict=True):
            reconstructed = (
                reconstructed
                + np.float32(weight)
                * values[:, id_to_pos[parent]].astype(np.float32, copy=False)
            ).astype(np.float32)
        maximum = max(
            maximum,
            float(
                np.abs(
                    reconstructed.astype(np.float64)
                    - values[:, id_to_pos[name]].astype(np.float64)
                ).max(initial=0.0)
            ),
        )
    return maximum


class ReplacementCandidateCache:
    """Read the exp263 primitive cache while overlaying only the exp413 likPF slot."""

    def __init__(
        self,
        parent_root: Path,
        contract: Mapping[str, Any],
        replacement_root: Path,
    ):
        self.root = Path(parent_root)
        self.replacement_root = Path(replacement_root)
        self.contract = dict(contract)
        validate_candidate_contract(contract)
        self.ids = candidate_ids(contract)
        self.specs = contract_by_id(contract)
        self.primitive_ids = [
            name for name in self.ids if str(self.specs[name]["kind"]) == "primitive"
        ]

    def load_fold(self, fold: int) -> FoldBundle:
        primitive_frames: dict[str, pd.DataFrame] = {}
        confidence: dict[str, pd.DataFrame] = {}
        for name in self.primitive_ids:
            root = self.replacement_root if name == SEMANTIC_SLOT else self.root
            frame = _sort_candidate_frame(
                _read_partition(root, "candidate_values", name, fold)
            )
            conf = _sort_candidate_frame(
                _read_partition(root, "candidate_confidence", name, fold)
            )
            if len(frame) != len(conf):
                raise ValueError(f"candidate/confidence row mismatch for {name}")
            for column in KEY_COLUMNS:
                if column == "md_since":
                    equal = np.array_equal(
                        frame[column].to_numpy(),
                        conf[column].to_numpy(),
                        equal_nan=True,
                    )
                else:
                    equal = np.array_equal(
                        frame[column].to_numpy(), conf[column].to_numpy()
                    )
                if not equal:
                    raise ValueError(f"candidate/confidence key mismatch for {name}: {column}")
            primitive_frames[name] = frame
            confidence[name] = conf
        base_frame = primitive_frames[self.primitive_ids[0]]
        for name in self.primitive_ids[1:]:
            other = primitive_frames[name]
            if len(base_frame) != len(other):
                raise ValueError(f"primitive row mismatch for {name}")
            for column in KEY_COLUMNS:
                if column == "md_since":
                    equal = np.array_equal(
                        base_frame[column].to_numpy(),
                        other[column].to_numpy(),
                        equal_nan=True,
                    )
                else:
                    equal = np.array_equal(
                        base_frame[column].to_numpy(), other[column].to_numpy()
                    )
                if not equal:
                    raise ValueError(f"primitive key mismatch for {name}: {column}")
        base = base_frame[[*KEY_COLUMNS, "last_known_tvt"]].copy()
        primitive_values = {
            name: pd.to_numeric(frame["candidate_tvt"], errors="coerce").to_numpy(
                np.float32
            )
            for name, frame in primitive_frames.items()
        }
        values = build_bank_from_primitives(primitive_values, self.contract)
        primitive_available = {
            name: frame["candidate_available"].astype(bool).to_numpy()
            & np.isfinite(primitive_values[name])
            for name, frame in primitive_frames.items()
        }
        available_by_id: dict[str, np.ndarray] = dict(primitive_available)
        for name in self.ids:
            if name in available_by_id:
                continue
            parents = [str(item) for item in self.specs[name]["parents"]]
            available_by_id[name] = np.logical_and.reduce(
                [available_by_id[parent] for parent in parents]
            )
        available = np.column_stack(
            [available_by_id[name] for name in self.ids]
        ).astype(bool)
        if not available.all() or not np.isfinite(values).all():
            raise ValueError("replacement cache requires complete finite candidate coverage")
        return FoldBundle(
            base=base,
            values=values,
            available=available,
            confidence=confidence,
            candidate_ids=self.ids,
            specs=self.specs,
        )


def replacement_cache_factory(
    replacement_root: Path,
) -> Callable[[Path, Mapping[str, Any]], ReplacementCandidateCache]:
    def factory(
        parent_root: Path,
        contract: Mapping[str, Any],
    ) -> ReplacementCandidateCache:
        return ReplacementCandidateCache(parent_root, contract, replacement_root)

    return factory


def _write_replacement_fold(
    *,
    frozen: pd.DataFrame,
    parent_reference: pd.DataFrame,
    fold: int,
    replacement_root: Path,
) -> dict[str, Any]:
    reference = _sort_candidate_frame(parent_reference)
    source = frozen.set_index("id", drop=False)
    row_ids = reference["id"].astype(str)
    positions = source.index.get_indexer(row_ids)
    if np.any(positions < 0):
        raise ValueError(f"exp404 replacement misses parent fold {fold} ids")
    aligned = source.iloc[positions].reset_index(drop=True)
    _assert_replacement_alignment(reference, aligned)
    value = reference[[*KEY_COLUMNS, "last_known_tvt"]].copy()
    value["candidate_id"] = SEMANTIC_SLOT
    value["candidate_tvt"] = aligned[REPLACEMENT_VALUE_SOURCE].to_numpy(
        np.float32
    )
    value["candidate_available"] = np.isfinite(
        value["candidate_tvt"].to_numpy(np.float32)
    )
    confidence = reference[KEY_COLUMNS].copy()
    confidence["candidate_id"] = SEMANTIC_SLOT
    confidence["confidence_source"] = REPLACEMENT_VALUE_SOURCE
    confidence["confidence_valid"] = value["candidate_available"].to_numpy(bool)
    confidence["confidence_missing_fields"] = ""

    value_path = (
        replacement_root
        / "candidate_values"
        / SEMANTIC_SLOT
        / f"fold={fold}"
        / "part-000.parquet"
    )
    confidence_path = (
        replacement_root
        / "candidate_confidence"
        / SEMANTIC_SLOT
        / f"fold={fold}"
        / "part-000.parquet"
    )
    value_path.parent.mkdir(parents=True, exist_ok=True)
    confidence_path.parent.mkdir(parents=True, exist_ok=True)
    value.to_parquet(value_path, index=False)
    confidence.to_parquet(confidence_path, index=False)
    return {
        "fold": int(fold),
        "rows": len(value),
        "wells": int(value["well"].nunique()),
        "value_path": str(value_path.relative_to(replacement_root)),
        "value_file_sha256": sha256_file(value_path),
        "value_content_sha256": logical_frame_sha256(value),
        "confidence_path": str(confidence_path.relative_to(replacement_root)),
        "confidence_file_sha256": sha256_file(confidence_path),
        "confidence_content_sha256": logical_frame_sha256(confidence),
    }


def run_replacement_preflight(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    parent_cache_root: Path,
    frozen_prediction_path: Path,
    feature_schema_path: Path,
    feature_catalog_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the truth-free overlay cache and validate the 5/7 replacement graph."""

    contract_evidence = validate_replacement_contract(config, contract)
    parent_evidence = verify_exp263_root(parent_cache_root, parent_config)
    frozen, frozen_evidence = load_frozen_scale5_predictions(
        frozen_prediction_path, config
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replacement_root = output_dir / "replacement_candidate_cache"
    partition_rows: list[dict[str, Any]] = []
    frozen_index = pd.Index(frozen["id"].astype(str))
    expected_folds = [int(item) for item in config["validation"]["expected_folds"]]
    for fold in expected_folds:
        parent_reference = _read_partition(
            parent_cache_root, "candidate_values", "exp226_k16", fold
        )
        partition_rows.append(
            _write_replacement_fold(
                frozen=frozen,
                parent_reference=parent_reference,
                fold=fold,
                replacement_root=replacement_root,
            )
        )

    overlay = ReplacementCandidateCache(parent_cache_root, contract, replacement_root)
    ids = candidate_ids(contract)
    id_to_pos = {name: position for position, name in enumerate(ids)}
    changed_positions = [id_to_pos[name] for name in CHANGED_CANDIDATES]
    unchanged_positions = [id_to_pos[name] for name in UNCHANGED_CANDIDATES]
    changed_nonzero_rows = 0
    unchanged_max_abs = 0.0
    formula_max_abs = 0.0
    parent_old_mean_parity_max_abs = 0.0
    selector_probe_rows: list[dict[str, Any]] = []
    schema = json.loads(Path(feature_schema_path).read_text())
    expected_features = [str(item) for item in schema["features"]]
    if len(expected_features) != 88:
        raise ValueError("exp413 selector schema must remain exactly 88 features")
    selector_contract = dict(config["data"]["selector_contract"])
    if sha256_file(feature_schema_path) != str(
        selector_contract["feature_schema_file_sha256"]
    ):
        raise ValueError("selector feature schema file SHA mismatch")
    if str(schema["feature_schema_sha256"]) != str(
        selector_contract["feature_schema_logical_sha256"]
    ):
        raise ValueError("selector feature schema logical SHA mismatch")
    if sha256_file(feature_catalog_path) != str(
        selector_contract["feature_catalog_sha256"]
    ):
        raise ValueError("selector feature catalog SHA mismatch")
    feature_cfg = dict(parent_config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"][
        "candidates"
    ]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"][
        "candidates"
    ]
    for fold in expected_folds:
        bundle = overlay.load_fold(fold)
        old_likpf = _sort_candidate_frame(
            _read_partition(parent_cache_root, "candidate_values", SEMANTIC_SLOT, fold)
        )
        if not old_likpf["id"].astype(str).reset_index(drop=True).equals(
            bundle.base["id"].astype(str).reset_index(drop=True)
        ):
            raise ValueError("parity-only old mean is not aligned to replacement cache")
        frozen_positions = frozen_index.get_indexer(bundle.base["id"].astype(str))
        if np.any(frozen_positions < 0):
            raise ValueError("parity-only exp404 mean misses replacement cache rows")
        frozen_old_mean = frozen.iloc[frozen_positions][
            "likpf_mean_x1p0"
        ].to_numpy(np.float32)
        parent_old_mean = old_likpf["candidate_tvt"].to_numpy(np.float32)
        parent_old_mean_parity_max_abs = max(
            parent_old_mean_parity_max_abs,
            float(
                np.abs(
                    frozen_old_mean.astype(np.float64)
                    - parent_old_mean.astype(np.float64)
                ).max(initial=0.0)
            ),
        )
        primitives = {
            name: bundle.values[:, id_to_pos[name]]
            for name in (
                "exp226_k16",
                "selfgr_hmm_a070",
                "exact_hmm",
                "pf_ancc",
                "beam_mean",
            )
        }
        primitives[SEMANTIC_SLOT] = old_likpf["candidate_tvt"].to_numpy(np.float32)
        old_bank = build_bank_from_primitives(primitives, contract)
        delta = np.abs(bundle.values.astype(np.float64) - old_bank.astype(np.float64))
        changed_nonzero_rows += int(np.any(delta[:, changed_positions] > 0.0, axis=1).sum())
        unchanged_max_abs = max(
            unchanged_max_abs,
            float(delta[:, unchanged_positions].max(initial=0.0)),
        )
        formula_max_abs = max(
            formula_max_abs,
            _formula_parity_max_abs(bundle.values, contract, CHANGED_CANDIDATES[1:]),
        )
        context, _ = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=False
        )
        probe_indices = deterministic_sample_indices(
            bundle.base,
            min(1024, len(bundle.base)),
            "exp413",
            "replacement_preflight_selector88",
            fold,
        )
        probe, _ = build_candidate_long_features(
            bundle,
            context,
            probe_indices,
            feature_cfg,
            expected_features=expected_features,
        )
        if list(probe.columns) != expected_features:
            raise ValueError("replacement selector feature order differs from frozen 88 schema")
        selector_probe_rows.append(
            {
                "fold": fold,
                "base_rows": len(probe_indices),
                "candidate_long_rows": len(probe),
                "feature_count": len(probe.columns),
                "content_sha256": logical_frame_sha256(probe),
            }
        )
        del bundle, old_likpf, old_bank, primitives, context, probe
        gc.collect()
    if changed_nonzero_rows <= 0:
        raise ValueError("scale5 replacement did not change any candidate row")
    if unchanged_max_abs > 0.0:
        raise AssertionError("one of the seven frozen candidates changed")
    if formula_max_abs > 1.0e-6:
        raise AssertionError("replacement formula parity failed")
    if parent_old_mean_parity_max_abs > float(
        config["replacement"]["parent_old_mean_parity_max_abs_ft"]
    ):
        raise AssertionError("exp404 parity-only old mean differs from parent cache")
    if sum(int(item["rows"]) for item in partition_rows) != int(
        config["validation"]["expected_rows"]
    ):
        raise ValueError("replacement fold partitions do not cover all expected rows")
    if len(set(frozen["id"].astype(str))) != int(config["validation"]["expected_rows"]):
        raise ValueError("frozen replacement ids are not globally unique")

    shutil.copy2(feature_schema_path, output_dir / "feature_schema.json")
    shutil.copy2(feature_catalog_path, output_dir / "feature_catalog.csv")
    compact_schema = {
        "schema_version": "1.0.0",
        "features": compact_feature_names(contract),
    }
    compact_schema["compact_meta_schema_sha256"] = sha256_json(compact_schema)
    write_json(output_dir / "compact_meta_schema.json", compact_schema)
    semantic_manifest = {
        "schema_version": "1.0.0",
        "status": "replacement_preflight_complete",
        "semantic_slot": SEMANTIC_SLOT,
        "value_source": REPLACEMENT_VALUE_SOURCE,
        "old_mean_usage": "parity_audit_only",
        "old_mean_in_candidate_or_model_input": False,
        "candidate_order": ids,
        "changed_candidates": list(CHANGED_CANDIDATES),
        "unchanged_candidates": list(UNCHANGED_CANDIDATES),
        "partition_count": len(partition_rows),
        "partitions": partition_rows,
        "selector_feature_count": len(expected_features),
        "selector_probe": selector_probe_rows,
        "compact_feature_count": len(compact_schema["features"]),
        "feature_graph": config["replacement"]["feature_graph"],
    }
    semantic_manifest["manifest_logical_sha256"] = sha256_json(semantic_manifest)
    semantic_path = output_dir / "replacement_semantic_manifest.json"
    write_json(semantic_path, semantic_manifest)
    summary = {
        "status": "replacement_preflight_complete",
        "passed": True,
        "models_trained": 0,
        "pf_well_runs": 0,
        "contract": contract_evidence,
        "parent_cache": parent_evidence,
        "frozen_prediction": frozen_evidence,
        "changed_candidate_rows": changed_nonzero_rows,
        "unchanged_candidate_max_abs_error": unchanged_max_abs,
        "formula_parity_max_abs_error": formula_max_abs,
        "parent_old_mean_parity_max_abs_error": parent_old_mean_parity_max_abs,
        "replacement_semantic_manifest_sha256": sha256_file(semantic_path),
        "selector_feature_schema_file_sha256": sha256_file(
            output_dir / "feature_schema.json"
        ),
        "selector_feature_schema_logical_sha256": str(
            schema["feature_schema_sha256"]
        ),
        "compact_feature_schema_file_sha256": sha256_file(
            output_dir / "compact_meta_schema.json"
        ),
        "compact_feature_schema_logical_sha256": str(
            compact_schema["compact_meta_schema_sha256"]
        ),
        "clean273_contract": {
            "source_feature_count": 380,
            "feature_count": 273,
            "rebuild_policy": "reload_full_source_patch_primitive_then_rebuild_all_transforms",
            "named_likpf_columns_are_not_individually_patched": True,
            "content_sha256": None,
            "content_sha_deferred_to_downstream_surface_build": True,
        },
    }
    write_json(output_dir / "replacement_preflight.json", summary)
    return summary


def stage_c_runtime_config(
    config: Mapping[str, Any],
    parent_exp264_config: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(parent_exp264_config))
    runtime["experiment"] = copy.deepcopy(dict(config["experiment"]))
    runtime["execution"]["stage"] = "nested_compact_meta"
    runtime["execution"]["run_approved"] = True
    runtime["model"]["nested_downstream_stage"]["enabled"] = True
    runtime["model"]["nested_downstream_stage"]["planned_cpu_selector_boosters"] = 40
    runtime["model"]["nested_downstream_stage"]["parent_control_retraining"] = False
    return runtime


def _stage_c_sha_overrides(stage_c_root: Path) -> dict[str, Any]:
    schema = json.loads((stage_c_root / "compact_meta_schema.json").read_text())
    values = {
        "stage_c_nested_selector_metrics_sha256": sha256_file(
            stage_c_root / "nested_selector_metrics.json"
        ),
        "stage_c_nested_selector_model_manifest_sha256": sha256_file(
            stage_c_root / "nested_selector_model_manifest.json"
        ),
        "stage_c_nested_compact_manifest_sha256": sha256_file(
            stage_c_root / "nested_compact_manifest.json"
        ),
        "stage_c_compact_meta_schema_file_sha256": sha256_file(
            stage_c_root / "compact_meta_schema.json"
        ),
        "stage_c_compact_meta_schema_logical_sha256": str(
            schema["compact_meta_schema_sha256"]
        ),
    }
    values.update(
        {
            "stage_c_expected_nested_selector_metrics_sha256": values[
                "stage_c_nested_selector_metrics_sha256"
            ],
            "stage_c_expected_nested_selector_model_manifest_sha256": values[
                "stage_c_nested_selector_model_manifest_sha256"
            ],
            "stage_c_expected_nested_compact_manifest_sha256": values[
                "stage_c_nested_compact_manifest_sha256"
            ],
            "stage_c_expected_compact_meta_schema_file_sha256": values[
                "stage_c_compact_meta_schema_file_sha256"
            ],
            "stage_c_expected_compact_meta_schema_logical_sha256": values[
                "stage_c_compact_meta_schema_logical_sha256"
            ],
        }
    )
    return values


def stage_s_runtime_config(
    config: Mapping[str, Any],
    parent_exp335_config: Mapping[str, Any],
    stage_c_root: Path,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(parent_exp335_config))
    runtime["experiment"] = copy.deepcopy(dict(config["experiment"]))
    runtime["data"]["exp404_scale5_train_prediction"] = copy.deepcopy(
        dict(config["data"]["exp404_scale5_train_prediction"])
    )
    runtime["data"].update(_stage_c_sha_overrides(stage_c_root))
    runtime["execution"]["stage"] = "signed_selector_train"
    runtime["execution"]["implementation_complete"] = True
    runtime["execution"]["preflight_run_approved"] = True
    runtime["execution"]["selector_train_approved"] = True
    runtime["execution"]["run_selector_train"] = True
    runtime["execution"]["control_retraining"] = False
    return runtime


def _stage_s_sha_overrides(stage_s_root: Path) -> dict[str, Any]:
    schema = json.loads((stage_s_root / "signed_compact_schema.json").read_text())
    return {
        "stage_s_signed_selector_metrics_sha256": sha256_file(
            stage_s_root / "signed_selector_metrics.json"
        ),
        "stage_s_model_manifest_sha256": sha256_file(
            stage_s_root / "signed_selector_model_manifest.json"
        ),
        "stage_s_compact_manifest_sha256": sha256_file(
            stage_s_root / "signed_compact_manifest.json"
        ),
        "stage_s_compact_schema_file_sha256": sha256_file(
            stage_s_root / "signed_compact_schema.json"
        ),
        "stage_s_compact_schema_logical_sha256": str(
            schema["signed_compact_schema_sha256"]
        ),
        "stage_s_reproducibility_manifest_sha256": sha256_file(
            stage_s_root / "reproducibility_manifest.json"
        ),
    }


def downstream_runtime_config(
    config: Mapping[str, Any],
    parent_exp335_config: Mapping[str, Any],
    stage_c_root: Path,
    stage_s_root: Path,
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(parent_exp335_config))
    runtime["experiment"] = copy.deepcopy(dict(config["experiment"]))
    runtime["data"]["exp404_scale5_train_prediction"] = copy.deepcopy(
        dict(config["data"]["exp404_scale5_train_prediction"])
    )
    runtime["data"].update(_stage_c_sha_overrides(stage_c_root))
    runtime["data"].update(_stage_s_sha_overrides(stage_s_root))
    runtime["execution"]["stage"] = "downstream_tvt_train"
    runtime["execution"]["downstream_train_approved"] = True
    runtime["execution"]["run_downstream_train"] = True
    runtime["execution"]["control_retraining"] = False
    return runtime


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def _replacement_model_view(frozen: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "id",
        "well_id",
        "row_idx",
        "last_known_tvt",
        "md_since",
        REPLACEMENT_VALUE_SOURCE,
    ]
    view = frozen[columns].copy()
    view = view.rename(
        columns={
            "well_id": "well",
            "row_idx": "well_row_idx",
            REPLACEMENT_VALUE_SOURCE: "replacement_tvt",
        }
    )
    if any("likpf_mean_x1p0" in column for column in view.columns):
        raise AssertionError("parity-only old mean leaked into replacement model view")
    return view


def patch_base_replay_primitive(
    base_frame: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replacement = _replacement_model_view(frozen)
    base = base_frame.copy()
    index = pd.Index(replacement["id"].astype(str))
    if not index.is_unique:
        raise ValueError("replacement model view ids are duplicated")
    positions = index.get_indexer(base["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(base):
        raise ValueError("replacement primitive does not cover base replay rows one-to-one")
    aligned = replacement.iloc[positions].reset_index(drop=True)
    if not base["well"].astype(str).reset_index(drop=True).equals(
        aligned["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("replacement primitive/base replay well mismatch")
    base_row_idx = pd.to_numeric(
        base["id"].astype(str).str.extract(r"_(\d+)$", expand=False),
        errors="raise",
    ).to_numpy(np.int64)
    if not np.array_equal(
        base_row_idx,
        aligned["well_row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("replacement primitive/base replay row-index mismatch")
    anchor = base["last_known_tvt"].to_numpy(np.float32)
    aligned_anchor = aligned["last_known_tvt"].to_numpy(np.float32)
    if float(np.abs(anchor - aligned_anchor).max(initial=0.0)) > 1.0e-4:
        raise ValueError("replacement primitive/base replay anchor mismatch")
    base_md_since = base["md_since"].to_numpy(np.float32)
    aligned_md_since = aligned["md_since"].to_numpy(np.float32)
    if float(np.abs(base_md_since - aligned_md_since).max(initial=0.0)) > 1.0e-4:
        raise ValueError("replacement primitive/base replay md_since mismatch")
    old = base["likpf_mean_d"].to_numpy(np.float32).copy()
    base["likpf_mean_d"] = (
        aligned["replacement_tvt"].to_numpy(np.float32) - anchor
    ).astype(np.float32)
    evidence = {
        "rows": len(base),
        "wells": int(base["well"].nunique()),
        "semantic_slot": SEMANTIC_SLOT,
        "value_source": REPLACEMENT_VALUE_SOURCE,
        "old_mean_retained_in_output": False,
        "changed_rows": int((base["likpf_mean_d"].to_numpy(np.float32) != old).sum()),
        "replacement_delta_content_sha256": dataframe_content_sha(
            base[["id", "well", "likpf_mean_d"]],
            ["id", "well", "likpf_mean_d"],
        ),
    }
    if evidence["changed_rows"] <= 0:
        raise ValueError("base replay primitive replacement is a no-op")
    return base, evidence


def rebuild_learned_likelihood_source(
    *,
    config: Mapping[str, Any],
    exp099_source_path: Path,
    frozen: pd.DataFrame,
    raw_train_dir: Path,
    exp145_source_path: Path,
    exp145_config_path: Path,
    multiobs_source_path: Path,
    exp111_schema_path: Path,
    exp111_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute every exp145 transform affected by the replaced likPF candidate."""

    generator = _load_module(exp145_source_path, "exp413_exp145_generator")
    multiobs_module = _load_module(multiobs_source_path, "exp413_multiobs")
    source_spec = dict(config["data"]["exp099_train_feature_cache"])
    source_report = inspect_gzip_csv(exp099_source_path)
    if source_report["raw_sha256"] != str(source_spec["expected_raw_sha256"]):
        raise ValueError("exp099 source raw gzip SHA mismatch")
    if source_report["decompressed_sha256"] != str(
        source_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp099 source decompressed SHA mismatch")
    if int(source_report["data_rows"]) != int(source_spec["expected_rows"]):
        raise ValueError("exp099 source row count mismatch")
    generator_config = read_yaml(exp145_config_path)
    candidates = generator.candidate_specs_from_config(generator_config)
    required = generator.source_required_columns(generator_config, candidates)
    source = pd.read_csv(
        exp099_source_path,
        usecols=required,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    replacement = _replacement_model_view(frozen)
    replacement_index = pd.Index(replacement["id"].astype(str))
    positions = replacement_index.get_indexer(source["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(source):
        raise ValueError("replacement primitive does not cover exp099 rows one-to-one")
    aligned = replacement.iloc[positions].reset_index(drop=True)
    if not source["well"].astype(str).reset_index(drop=True).equals(
        aligned["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("replacement primitive/exp099 well mismatch")
    source["likpf_mean"] = aligned["replacement_tvt"].to_numpy(np.float32)
    source["likpf_mean_d"] = (
        source["likpf_mean"].to_numpy(np.float32)
        - source["last_known_tvt"].to_numpy(np.float32)
    ).astype(np.float32)
    candidate_names = [spec.name for spec in candidates]
    existing = source[["id", "well", *candidate_names]].copy()
    multiobs_config = _nested_get(
        generator_config, "generator.multi_observation_likelihood", {}
    )
    recomputed, well_summary = multiobs_module.build_multi_observation_candidate_frame(
        source,
        existing,
        train_dir=raw_train_dir,
        candidate_names=candidate_names,
        config=dict(multiobs_config or {}),
    )
    old_multiobs_columns = [
        column
        for column in source
        if column.startswith("multiobs_")
    ]
    source = source.drop(columns=old_multiobs_columns, errors="ignore").merge(
        recomputed,
        on=["id", "well"],
        how="left",
        validate="one_to_one",
    )
    row_features = generator.load_feature_schema(exp111_schema_path)
    model_features = generator.exp111_model_feature_columns(row_features)
    classifier, error_model, model_meta = generator.load_exp111_models(
        manifest_path=exp111_manifest_path
    )
    if model_meta["classifier_sha256_actual"] != str(
        model_meta["classifier"]["sha256"]
    ):
        raise ValueError("saved exp111 classifier SHA mismatch")
    if model_meta["expected_error_sha256_actual"] != str(
        model_meta["expected_error"]["sha256"]
    ):
        raise ValueError("saved exp111 expected-error model SHA mismatch")
    learned, _ = generator.generate_ml_features_from_frame(
        source,
        candidates=candidates,
        row_feature_columns=row_features,
        model_feature_columns=model_features,
        classifier=classifier,
        error_model=error_model,
        config=generator_config,
    )
    numeric = [column for column in learned if column not in {"id", "well"}]
    if not np.isfinite(learned[numeric].to_numpy(np.float32)).all():
        raise ValueError("rebuilt learned-likelihood source contains non-finite values")
    evidence = {
        "source": source_report,
        "rows": len(learned),
        "wells": int(learned["well"].nunique()),
        "features": len(numeric),
        "feature_schema_sha256": sha256_json(numeric),
        "feature_content_sha256": logical_frame_sha256(learned),
        "replacement_value_source": REPLACEMENT_VALUE_SOURCE,
        "multiobs_recomputed_for_all_candidates": True,
        "multiobs_well_summary_rows": len(well_summary),
        "saved_exp111_models_retrained": 0,
        "saved_exp111_model_meta": model_meta,
        "old_mean_in_model_input": False,
    }
    del source, existing, recomputed
    gc.collect()
    return learned, evidence


def build_replacement_clean273_surface(
    *,
    config: Mapping[str, Any],
    frozen_prediction_path: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    exp099_source_path: Path,
    exp145_source_path: Path,
    exp145_config_path: Path,
    multiobs_source_path: Path,
    exp111_schema_path: Path,
    exp111_manifest_path: Path,
    clean_allowlist_path: Path,
    raw_train_dir: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], Any, dict[str, Any]]:
    """Reload the full exp218 source and rebuild all 273 selected columns."""

    from src.candidate_selector_pipeline import apply_stage_d_base_feature_allowlist

    frozen, frozen_evidence = load_frozen_scale5_predictions(
        frozen_prediction_path, config
    )
    source_contracts = {
        exp218_source_path: str(config["data"]["exp218_source"]["script_sha256"]),
        exp218_config_path: str(config["data"]["exp218_source"]["config_sha256"]),
        exp145_source_path: str(config["data"]["exp145_source"]["script_sha256"]),
        exp145_config_path: str(config["data"]["exp145_source"]["config_sha256"]),
        multiobs_source_path: str(
            config["data"]["exp145_source"]["multiobs_script_sha256"]
        ),
        exp111_schema_path: str(config["data"]["exp111_saved_models"]["schema_sha256"]),
        exp111_manifest_path: str(
            config["data"]["exp111_saved_models"]["manifest_sha256"]
        ),
    }
    source_sha: dict[str, str] = {}
    for path, expected_sha in source_contracts.items():
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(f"replacement source SHA mismatch: {path}")
        source_sha[str(path)] = observed_sha
    exp218 = _load_module(exp218_source_path, "exp413_exp218")
    exp218_config = read_yaml(exp218_config_path)
    base, base_columns, base_meta = exp218.load_exp072_full_replay_cache_frame(
        _nested_get(exp218_config, "data.exp072_train_feature_cache_local"),
        max_rows=None,
    )
    base_spec = dict(config["data"]["exp072_train_feature_cache"])
    base_report = inspect_gzip_csv(Path(base_meta["source"]))
    if base_report["raw_sha256"] != str(base_spec["expected_raw_sha256"]):
        raise ValueError("exp072 base cache raw gzip SHA mismatch")
    if base_report["decompressed_sha256"] != str(
        base_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp072 base cache decompressed SHA mismatch")
    if (
        len(base) != int(base_spec["expected_rows"])
        or int(base["well"].nunique()) != int(base_spec["expected_wells"])
        or len(base_columns) != int(base_spec["expected_feature_count"])
    ):
        raise ValueError("exp072 base cache coverage or feature-count mismatch")
    if str(base_meta.get("schema_sha256") or "") != str(
        base_spec["expected_schema_sha256"]
    ):
        raise ValueError("exp072 base cache feature schema SHA mismatch")
    base_likpf_dependencies = [
        column for column in base_columns if "likpf" in column.lower()
    ]
    expected_base_likpf_dependencies = [
        str(item) for item in base_spec["expected_likpf_dependency_columns"]
    ]
    if base_likpf_dependencies != expected_base_likpf_dependencies:
        raise ValueError(
            "exp072 base likPF dependency inventory changed: "
            f"{base_likpf_dependencies}"
        )
    base, replacement_evidence = patch_base_replay_primitive(base, frozen)
    base, anchor_meta = exp218.add_anchor_columns(base, raw_train_dir)
    projection_cfg = dict(_nested_get(exp218_config, "model.u_projection", {}) or {})
    projection, projection_groups, projection_summary = exp218.build_u_projection_features(
        base,
        source_specs=dict(projection_cfg.get("sources") or {}),
        degree=int(projection_cfg.get("degree", 3)),
        robust_iters=int(projection_cfg.get("robust_iters", 3)),
        clip_sigma=float(projection_cfg.get("clip_sigma", 4.0)),
    )
    exp218._assign_aligned_float32_columns(
        base,
        projection,
        [column for column in projection if column not in {"id", "well"}],
    )
    learned_source, learned_evidence = rebuild_learned_likelihood_source(
        config=config,
        exp099_source_path=exp099_source_path,
        frozen=frozen,
        raw_train_dir=raw_train_dir,
        exp145_source_path=exp145_source_path,
        exp145_config_path=exp145_config_path,
        multiobs_source_path=multiobs_source_path,
        exp111_schema_path=exp111_schema_path,
        exp111_manifest_path=exp111_manifest_path,
    )
    learned_cfg = dict(
        _nested_get(exp218_config, "model.learned_likelihood_features", {}) or {}
    )
    learned, learned_groups, learned_summary = exp218.build_learned_likelihood_features(
        learned_source,
        base,
        learned_cfg,
    )
    exp218._assign_aligned_float32_columns(
        base,
        learned,
        [column for column in learned if column not in {"id", "well"}],
    )
    grwr_cfg = dict(
        _nested_get(
            exp218_config,
            "model.gr_wavelet_rotation_confidence_features",
            {},
        )
        or {}
    )
    grwr, grwr_groups, grwr_summary, grwr_meta = (
        exp218.build_gr_wavelet_rotation_confidence_features(
            base,
            train_dir=raw_train_dir,
            config=grwr_cfg,
        )
    )
    exp218._assign_aligned_float32_columns(
        base,
        grwr,
        [column for column in grwr if column not in {"id", "well"}],
    )
    groups = {**projection_groups, **learned_groups, **grwr_groups}
    active = list(
        _nested_get(
            exp218_config, "model.feature_ablation.active_variants", []
        )
        or []
    )
    parent_variant = next(
        item
        for item in active
        if str(item.get("name")) == "gr_wavelet_rotation_confidence_addonly"
    )
    source_features = exp218.feature_columns_for_variant(
        base_columns, groups, parent_variant
    )
    clean_spec = dict(config["data"]["clean_base_allowlist"])
    features, allowlist_evidence = apply_stage_d_base_feature_allowlist(
        source_features,
        allowlist_path=clean_allowlist_path,
        expected_source_count=int(clean_spec["expected_source_feature_count"]),
        expected_selected_count=int(clean_spec["expected_feature_count"]),
        expected_allowlist_sha256=str(clean_spec["sha256"]),
    )
    likpf_named_features = [
        feature for feature in features if "likpf" in feature.lower()
    ]
    expected_likpf_named_count = int(clean_spec["expected_likpf_named_feature_count"])
    if len(likpf_named_features) != expected_likpf_named_count:
        raise ValueError("clean273 likPF-named feature inventory changed")
    regenerated_columns = {
        *[column for column in projection if column not in {"id", "well"}],
        *[column for column in learned if column not in {"id", "well"}],
        *[column for column in grwr if column not in {"id", "well"}],
    }
    stale_named = sorted(
        set(likpf_named_features)
        - set(expected_base_likpf_dependencies)
        - regenerated_columns
    )
    if stale_named:
        raise ValueError(
            f"clean273 likPF dependencies were not regenerated: {stale_named}"
        )
    required = {"id", "well", "target", "last_known_tvt", "md_since", *features}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"replacement clean273 surface misses columns: {missing}")
    for start in range(0, len(features), 32):
        values = base[features[start : start + 32]].to_numpy(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("replacement clean273 surface contains non-finite values")
    content_columns = ["id", "well", *features]
    evidence = {
        "status": "replacement_clean273_rebuilt",
        "rows": len(base),
        "wells": int(base["well"].nunique()),
        "source_feature_count": len(source_features),
        "feature_count": len(features),
        "feature_schema_sha256": sha256_json(features),
        "feature_content_sha256": dataframe_content_sha(base, content_columns),
        "full_source_reloaded": True,
        "named_likpf_columns_patched_individually": False,
        "replacement_primitive": replacement_evidence,
        "frozen_prediction": frozen_evidence,
        "base_cache": {
            **base_meta,
            "gzip_contract": base_report,
            "likpf_dependency_columns": base_likpf_dependencies,
        },
        "source_sha256": source_sha,
        "anchor": anchor_meta,
        "allowlist": allowlist_evidence,
        "projection_summary_rows": len(projection_summary),
        "learned_summary_rows": len(learned_summary),
        "grwr_summary_rows": len(grwr_summary),
        "learned_likelihood": learned_evidence,
        "grwr": grwr_meta,
        "old_mean_in_model_input": False,
        "likpf_named_features": likpf_named_features,
        "likpf_named_features_all_rebuilt_from_replacement_source": True,
    }
    del frozen, projection, learned_source, learned, grwr
    gc.collect()
    return base, features, evidence, exp218, exp218_config


def _rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def load_saved_exp335_control(
    *,
    oof_path: Path,
    metrics_path: Path,
    model_manifest_path: Path,
    base_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(config["data"]["exp335_saved_control"])
    expected_sha = {
        "oof": str(spec["oof_sha256"]),
        "metrics": str(spec["metrics_sha256"]),
        "model_manifest": str(spec["model_manifest_sha256"]),
    }
    paths = {
        "oof": Path(oof_path),
        "metrics": Path(metrics_path),
        "model_manifest": Path(model_manifest_path),
    }
    actual_sha = {name: sha256_file(path) for name, path in paths.items()}
    for name in paths:
        if actual_sha[name] != expected_sha[name]:
            raise ValueError(f"saved exp335 {name} SHA mismatch")
    metrics = json.loads(paths["metrics"].read_text())
    manifest = json.loads(paths["model_manifest"].read_text())
    if int(metrics.get("model_count", -1)) != int(spec["expected_model_count"]):
        raise ValueError("saved exp335 metrics model count mismatch")
    if int(manifest.get("model_count", -1)) != int(spec["expected_model_count"]):
        raise ValueError("saved exp335 model manifest count mismatch")
    if int(manifest.get("feature_count", -1)) != int(spec["expected_feature_count"]):
        raise ValueError("saved exp335 feature count mismatch")

    parent_column = "signed_residual_meta_addonly__lgb_mean__pred_tvt"
    required = [
        "id",
        "well",
        "md_since",
        "last_known_tvt",
        "target",
        "outer_fold",
        "actual_tvt",
        parent_column,
    ]
    parent = pd.read_parquet(paths["oof"], columns=required)
    if (
        len(parent) != int(config["validation"]["expected_rows"])
        or parent["well"].nunique() != int(config["validation"]["expected_wells"])
        or parent["id"].astype(str).duplicated().any()
    ):
        raise ValueError("saved exp335 OOF identity or coverage mismatch")
    parent_index = pd.Index(parent["id"].astype(str))
    positions = parent_index.get_indexer(base_frame["id"].astype(str))
    if np.any(positions < 0) or len(np.unique(positions)) != len(base_frame):
        raise ValueError("saved exp335 OOF does not align one-to-one with clean273")
    parent = parent.iloc[positions].reset_index(drop=True)
    if not parent["well"].astype(str).reset_index(drop=True).equals(
        base_frame["well"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("saved exp335 OOF well alignment mismatch")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if (
        float(
            np.abs(parent["actual_tvt"].to_numpy(np.float32) - truth).max(initial=0.0)
        )
        > 1.0e-4
    ):
        raise ValueError("saved exp335 OOF truth differs from replacement clean273")
    observed_rmse = _rmse(truth, parent[parent_column].to_numpy(np.float32))
    expected_rmse = float(config["validation"]["primary_control"]["rmse"])
    if abs(observed_rmse - expected_rmse) > 1.0e-9:
        raise ValueError(f"saved exp335 OOF RMSE mismatch: {observed_rmse}")
    return parent, {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual_sha,
        "rows": len(parent),
        "wells": int(parent["well"].nunique()),
        "feature_count": int(manifest["feature_count"]),
        "model_count": int(manifest["model_count"]),
        "rmse": observed_rmse,
        "models_retrained": 0,
        "prediction_column": parent_column,
    }


def evaluate_replacement_gate(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    saved_parent: pd.DataFrame,
    oof_fold: np.ndarray,
    new_prediction: np.ndarray,
    hidden_like_assignment_path: Path,
    technical_checks: Mapping[str, bool],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    parent_column = "signed_residual_meta_addonly__lgb_mean__pred_tvt"
    parent = saved_parent[parent_column].to_numpy(np.float32)
    candidate = np.asarray(new_prediction, dtype=np.float32)
    if not np.isfinite(candidate).all():
        raise ValueError("replacement OOF prediction contains non-finite values")

    fold_rows: list[dict[str, Any]] = []
    for fold in [int(item) for item in config["validation"]["expected_folds"]]:
        mask = np.asarray(oof_fold) == fold
        if not np.any(mask):
            raise ValueError(f"replacement OOF fold {fold} has no rows")
        parent_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "saved_exp335_rmse": parent_rmse,
                "replacement_rmse": candidate_rmse,
                "delta_rmse_replacement_minus_exp335": candidate_rmse - parent_rmse,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base_frame["md_since"].to_numpy(np.float32)
    scope_masks = {
        "md_since_0_250": md_since <= 250.0,
        "md_since_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "md_since_1000_plus": md_since >= 1000.0,
    }
    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        parent_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "saved_exp335_rmse": parent_rmse,
                "replacement_rmse": candidate_rmse,
                "delta_rmse_replacement_minus_exp335": candidate_rmse - parent_rmse,
            }
        )
    assignment = pd.read_csv(
        hidden_like_assignment_path, dtype={"well_id": str}
    ).set_index("well_id")
    hidden_columns = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    hidden_rows: list[dict[str, Any]] = []
    for scope, column in hidden_columns.items():
        mask = (
            base_frame["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        )
        if not np.any(mask):
            raise ValueError(f"hidden-like assignment has no valid rows for {scope}")
        parent_rmse = _rmse(truth[mask], parent[mask])
        candidate_rmse = _rmse(truth[mask], candidate[mask])
        hidden_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "saved_exp335_rmse": parent_rmse,
                "replacement_rmse": candidate_rmse,
                "delta_rmse_replacement_minus_exp335": candidate_rmse - parent_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    by_well_source = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "saved_exp335": parent,
            "replacement": candidate,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in by_well_source.groupby("well", sort=True):
        parent_rmse = _rmse(group["actual_tvt"], group["saved_exp335"])
        replacement_rmse = _rmse(group["actual_tvt"], group["replacement"])
        well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "saved_exp335_rmse": parent_rmse,
                "replacement_rmse": replacement_rmse,
                "delta_rmse_replacement_minus_exp335": replacement_rmse - parent_rmse,
            }
        )
    by_well = pd.DataFrame(well_rows)
    pooled_parent = _rmse(truth, parent)
    pooled_candidate = _rmse(truth, candidate)
    improvement = pooled_parent - pooled_candidate
    nonworse_folds = int(
        (fold_metrics["delta_rmse_replacement_minus_exp335"] <= 0.0).sum()
    )
    scope_table = pd.concat([pd.DataFrame(scope_rows), hidden_metrics], ignore_index=True)
    promotion = dict(config["validation"]["promotion"])
    required_scopes = [str(item) for item in promotion["required_scopes"]]
    if set(scope_table["scope"]) != set(required_scopes):
        raise ValueError("replacement scope inventory differs from preregistered gate")
    maximum_scope_delta = float(
        scope_table["delta_rmse_replacement_minus_exp335"].max()
    )
    technical_passed = bool(technical_checks) and all(technical_checks.values())
    checks = {
        "minimum_pooled_rmse_gain": improvement
        >= float(promotion["minimum_pooled_rmse_gain_ft"]),
        "minimum_nonworse_folds": nonworse_folds
        >= int(promotion["minimum_nonworse_folds"]),
        "maximum_scope_delta": maximum_scope_delta
        <= float(promotion["maximum_scope_delta_rmse_ft"]),
        "all_technical_checks": technical_passed,
    }
    delta = by_well["delta_rmse_replacement_minus_exp335"]
    tail = {
        "by_well_delta_p95": float(delta.quantile(0.95)),
        "worst_well": str(by_well.loc[delta.idxmax(), "well"]),
        "worst_well_delta_rmse": float(delta.max()),
        "worsened_well_count_plus_1ft": int((delta > 1.0).sum()),
        "worsened_well_count_plus_3ft": int((delta > 3.0).sum()),
        "worsened_well_count_plus_5ft": int((delta > 5.0).sum()),
        "policy": "report_only_not_automatic_stop",
    }
    gate = {
        "saved_exp335_rmse": pooled_parent,
        "replacement_rmse": pooled_candidate,
        "gain_ft": improvement,
        "delta_rmse_replacement_minus_exp335": pooled_candidate - pooled_parent,
        "nonworse_folds": nonworse_folds,
        "maximum_scope_delta_rmse": maximum_scope_delta,
        "technical_checks": dict(technical_checks),
        "checks": checks,
        "tail_readout": tail,
        "passed": bool(all(checks.values())),
        "pass_action": promotion["pass_action"],
        "fail_action": promotion["fail_action"],
    }
    return gate, fold_metrics, pd.DataFrame(scope_rows), hidden_metrics, by_well


def _absolute_stage_c_evidence(
    root: Path,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(evidence)
    result["partitions"] = [
        {
            **dict(item),
            "path": str(Path(root) / str(item["path"]))
            if not Path(str(item["path"])).is_absolute()
            else str(item["path"]),
        }
        for item in evidence["partitions"]
    ]
    return result


def verify_replacement_stage_0_root(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the immutable replacement overlay and its frozen-source lineage."""

    root = Path(root)
    paths = {
        "preflight": root / "replacement_preflight.json",
        "semantic": root / "replacement_semantic_manifest.json",
        "feature_schema": root / "feature_schema.json",
        "feature_catalog": root / "feature_catalog.csv",
        "compact_schema": root / "compact_meta_schema.json",
    }
    for path in paths.values():
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"replacement Stage 0 contract file missing: {path}")
    preflight = json.loads(paths["preflight"].read_text())
    semantic = json.loads(paths["semantic"].read_text())
    feature_schema = json.loads(paths["feature_schema"].read_text())
    compact_schema = json.loads(paths["compact_schema"].read_text())
    if not bool(preflight.get("passed", False)):
        raise ValueError("replacement Stage 0 preflight did not pass")
    if str(preflight["replacement_semantic_manifest_sha256"]) != sha256_file(
        paths["semantic"]
    ):
        raise ValueError("replacement Stage 0 semantic manifest SHA mismatch")
    frozen = dict(preflight["frozen_prediction"])
    frozen_spec = dict(config["data"]["exp404_scale5_train_prediction"])
    frozen_pairs = {
        "raw_sha256": "expected_raw_sha256",
        "decompressed_sha256": "expected_decompressed_sha256",
        "logical_sha256": "expected_logical_sha256",
        "schema_sha256": "expected_schema_sha256",
    }
    for observed_key, expected_key in frozen_pairs.items():
        if str(frozen[observed_key]) != str(frozen_spec[expected_key]):
            raise ValueError(f"replacement Stage 0 frozen {observed_key} mismatch")
    if (
        str(semantic.get("semantic_slot")) != SEMANTIC_SLOT
        or str(semantic.get("value_source")) != REPLACEMENT_VALUE_SOURCE
        or bool(semantic.get("old_mean_in_candidate_or_model_input", True))
    ):
        raise ValueError("replacement Stage 0 semantic source contract mismatch")
    if tuple(semantic.get("changed_candidates", [])) != CHANGED_CANDIDATES:
        raise ValueError("replacement Stage 0 changed-candidate inventory mismatch")
    if tuple(semantic.get("unchanged_candidates", [])) != UNCHANGED_CANDIDATES:
        raise ValueError("replacement Stage 0 unchanged-candidate inventory mismatch")
    if int(semantic.get("partition_count", -1)) != 5:
        raise ValueError("replacement Stage 0 partition count mismatch")
    total_rows = 0
    for item in semantic.get("partitions", []):
        total_rows += int(item["rows"])
        for path_key, sha_key in (
            ("value_path", "value_file_sha256"),
            ("confidence_path", "confidence_file_sha256"),
        ):
            path = root / "replacement_candidate_cache" / str(item[path_key])
            if not path.exists() or sha256_file(path) != str(item[sha_key]):
                raise ValueError(f"replacement Stage 0 partition SHA mismatch: {path}")
    if total_rows != int(config["validation"]["expected_rows"]):
        raise ValueError("replacement Stage 0 partition row inventory mismatch")
    selector_spec = dict(config["data"]["selector_contract"])
    if sha256_file(paths["feature_schema"]) != str(
        selector_spec["feature_schema_file_sha256"]
    ):
        raise ValueError("replacement Stage 0 selector schema file SHA mismatch")
    if str(feature_schema["feature_schema_sha256"]) != str(
        selector_spec["feature_schema_logical_sha256"]
    ):
        raise ValueError("replacement Stage 0 selector schema logical SHA mismatch")
    if sha256_file(paths["feature_catalog"]) != str(
        selector_spec["feature_catalog_sha256"]
    ):
        raise ValueError("replacement Stage 0 selector catalog SHA mismatch")
    if (
        len(feature_schema.get("features", [])) != 88
        or len(compact_schema.get("features", [])) != 74
    ):
        raise ValueError("replacement Stage 0 selector/compact feature width mismatch")
    return {
        "root": str(root),
        "preflight_sha256": sha256_file(paths["preflight"]),
        "semantic_manifest_sha256": sha256_file(paths["semantic"]),
        "replacement_cache_root": str(root / "replacement_candidate_cache"),
        "rows": total_rows,
        "partition_count": 5,
        "selector_feature_count": 88,
        "compact_feature_count": 74,
        "frozen_prediction": {
            key: frozen[key]
            for key in (
                "raw_sha256",
                "decompressed_sha256",
                "logical_sha256",
                "schema_sha256",
            )
        },
    }


def run_replacement_stage_d(
    *,
    config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    contract: Mapping[str, Any],
    stage_c_root: Path,
    stage_s_root: Path,
    saved_parent_oof_path: Path,
    saved_parent_metrics_path: Path,
    saved_parent_model_manifest_path: Path,
    hidden_like_assignment_path: Path,
    frozen_prediction_path: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    exp099_source_path: Path,
    exp145_source_path: Path,
    exp145_config_path: Path,
    multiobs_source_path: Path,
    exp111_schema_path: Path,
    exp111_manifest_path: Path,
    clean_allowlist_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train exactly 15 replacement downstream boosters against saved exp335."""

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    validate_replacement_contract(config, contract)
    cost = replacement_cost_contract(config)
    if cost["gpu_downstream_boosters"] != 15:
        raise ValueError("replacement Stage D cost must remain exactly 15 boosters")
    stage_c_evidence = verify_replacement_stage_c_root(stage_c_root, runtime_config)
    stage_s_evidence = verify_replacement_stage_s_root(
        stage_s_root,
        runtime_config,
        stage_c_root=stage_c_root,
    )
    if stage_c_evidence["compact_features"] != compact_feature_names(contract):
        raise ValueError("replacement Stage C compact schema differs from candidate contract")
    if stage_s_evidence["features"] != signed_compact_feature_names(contract):
        raise ValueError("replacement Stage S signed schema differs from candidate contract")
    base, base_features, base_evidence, exp218, exp218_config = (
        build_replacement_clean273_surface(
            config=config,
            frozen_prediction_path=frozen_prediction_path,
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            exp099_source_path=exp099_source_path,
            exp145_source_path=exp145_source_path,
            exp145_config_path=exp145_config_path,
            multiobs_source_path=multiobs_source_path,
            exp111_schema_path=exp111_schema_path,
            exp111_manifest_path=exp111_manifest_path,
            clean_allowlist_path=clean_allowlist_path,
            raw_train_dir=raw_train_dir,
        )
    )
    required_base = list(
        dict.fromkeys(["id", "well", "target", "last_known_tvt", "md_since", *base_features])
    )
    base = base.loc[:, ~base.columns.duplicated()].loc[:, required_base].copy()
    parent, parent_evidence = load_saved_exp335_control(
        oof_path=saved_parent_oof_path,
        metrics_path=saved_parent_metrics_path,
        model_manifest_path=saved_parent_model_manifest_path,
        base_frame=base,
        config=config,
    )
    hidden_sha = sha256_file(hidden_like_assignment_path)
    expected_hidden_sha = str(config["data"]["hidden_like_assignment"]["sha256"])
    if hidden_sha != expected_hidden_sha:
        raise ValueError("hidden-like assignment SHA mismatch")
    parent_features = [str(item) for item in stage_c_evidence["compact_features"]]
    signed_features = [str(item) for item in stage_s_evidence["features"]]
    final_features = [*base_features, *parent_features, *signed_features]
    if len(base_features) != 273 or len(parent_features) != 74 or len(signed_features) != 23:
        raise ValueError("replacement Stage D component feature count changed")
    if len(final_features) != 370 or len(set(final_features)) != 370:
        raise ValueError("replacement Stage D final370 schema is not exact and unique")
    if any("likpf_mean_x1p0" in feature for feature in final_features):
        raise ValueError("parity-only old mean appears in final370 schema")

    stage_cfg = dict(config["model"]["downstream_tvt"])
    mode = dict(exp218_config["model"]["training"]["modes"][str(stage_cfg["mode"])])
    if not bool(mode.get("use_gpu", False)):
        raise ValueError("replacement Stage D must use the frozen GPU mode")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode
    )
    config_indices = [int(item) for item in stage_cfg["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    base_index = pd.Index(base["id"].astype(str))
    if not base_index.is_unique:
        raise ValueError("replacement clean273 ids are not unique")
    target = base["target"].to_numpy(np.float32)
    anchor = base["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    n_rows = len(base)
    oof_by_config = [
        np.full(n_rows, np.nan, dtype=np.float32) for _ in config_indices
    ]
    oof_fold = np.full(n_rows, -1, dtype=np.int8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "stage_d_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    stage_c_load_evidence = _absolute_stage_c_evidence(
        stage_c_root, stage_c_evidence
    )
    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_load_evidence,
            downstream_outer_fold=outer_fold,
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=stage_s_evidence,
            downstream_outer_fold=outer_fold,
        )
        for role, compact, signed in (
            ("train", compact_train, signed_train),
            ("valid", compact_valid, signed_valid),
        ):
            if not compact[KEY_COLUMNS].reset_index(drop=True).equals(
                signed[KEY_COLUMNS].reset_index(drop=True)
            ):
                raise ValueError(f"replacement compact/signed key mismatch: {role}")
        train_positions = base_index.get_indexer(compact_train["id"].astype(str))
        valid_positions = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_positions < 0) or np.any(valid_positions < 0):
            raise ValueError("replacement compact ids are absent from clean273")
        if len(np.unique(np.concatenate([train_positions, valid_positions]))) != n_rows:
            raise ValueError("replacement Stage D fold does not cover all rows once")
        if np.intersect1d(train_positions, valid_positions).size:
            raise ValueError("replacement Stage D train/valid overlap")
        if np.any(oof_fold[valid_positions] >= 0):
            raise ValueError("replacement Stage D OOF row assigned twice")
        oof_fold[valid_positions] = np.int8(outer_fold)
        x_train = np.empty((len(train_positions), 370), dtype=np.float32)
        x_valid = np.empty((len(valid_positions), 370), dtype=np.float32)
        chunk_columns = int(stage_cfg["matrix_copy_chunk_columns"])
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            source = base[columns]
            x_train[:, start:stop] = source.iloc[train_positions].to_numpy(
                np.float32, copy=True
            )
            x_valid[:, start:stop] = source.iloc[valid_positions].to_numpy(
                np.float32, copy=True
            )
        compact_start = len(base_features)
        signed_start = compact_start + len(parent_features)
        x_train[:, compact_start:signed_start] = compact_train[
            parent_features
        ].to_numpy(np.float32, copy=False)
        x_valid[:, compact_start:signed_start] = compact_valid[
            parent_features
        ].to_numpy(np.float32, copy=False)
        x_train[:, signed_start:] = signed_train[signed_features].to_numpy(
            np.float32, copy=False
        )
        x_valid[:, signed_start:] = signed_valid[signed_features].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(x_train).all() or not np.isfinite(x_valid).all():
            raise ValueError("replacement final370 matrix contains non-finite values")
        x_train_frame = pd.DataFrame(x_train, columns=final_features, copy=False)
        x_valid_frame = pd.DataFrame(x_valid, columns=final_features, copy=False)
        fold_predictions: list[np.ndarray] = []
        for family_position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train_frame,
                target[train_positions],
                eval_set=[(x_valid_frame, target[valid_positions])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(
                        int(stage_cfg["early_stopping_rounds"]), verbose=False
                    ),
                    log_evaluation(int(stage_cfg["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            residual = model.predict(
                x_valid_frame, num_iteration=best_iteration
            ).astype(np.float32)
            prediction = (anchor[valid_positions] + residual).astype(np.float32)
            oof_by_config[family_position][valid_positions] = residual
            fold_predictions.append(prediction)
            model_path = (
                model_dir / f"scale5_x1p0_full_replacement__lgb{config_index}__outer{outer_fold}.txt"
            )
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            model_rows.append(
                {
                    "variant": "scale5_x1p0_full_replacement",
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": outer_fold,
                    "feature_count": 370,
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                }
            )
            fold_model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_positions),
                    "rmse_tvt": _rmse(truth[valid_positions], prediction),
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                for feature, value in zip(final_features, importance, strict=True):
                    group = (
                        "signed_compact"
                        if feature in signed_features
                        else "nested_compact"
                        if feature in parent_features
                        else "clean_base"
                    )
                    importance_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "model": f"lgb{config_index}",
                            "importance_type": importance_type,
                            "feature": feature,
                            "feature_group": group,
                            "importance": float(value),
                        }
                    )
            print(
                json.dumps(
                    {
                        "stage": "D",
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": fold_model_rows[-1]["rmse_tvt"],
                        "completed_boosters": len(model_rows),
                        "planned_boosters": 15,
                        "saved_exp335_control_retraining": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, residual
            gc.collect()
        fold_mean = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_positions),
                "rmse_tvt": _rmse(truth[valid_positions], fold_mean),
                "best_iteration": None,
            }
        )
        del (
            compact_train,
            compact_valid,
            signed_train,
            signed_valid,
            x_train,
            x_valid,
            x_train_frame,
            x_valid_frame,
            fold_predictions,
        )
        gc.collect()
    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("replacement Stage D model/OOF contract is incomplete")
    if not np.array_equal(oof_fold, parent["outer_fold"].to_numpy(np.int8)):
        raise AssertionError("replacement Stage D fold assignment differs from exp335")
    if any(not np.isfinite(item).all() for item in oof_by_config):
        raise AssertionError("replacement Stage D OOF residual is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    technical_checks = {
        "input_and_replacement_sha": bool(base_evidence["frozen_prediction"]),
        "candidate_schema_12": len(candidate_ids(contract)) == 12,
        "clean273_schema": len(base_features) == 273,
        "nested74_schema": len(parent_features) == 74,
        "signed23_schema": len(signed_features) == 23,
        "final370_schema": len(final_features) == 370,
        "model_count_15": len(model_rows) == 15,
        "parent_control_retraining_zero": True,
        "old_mean_absent_from_model_input": bool(
            not base_evidence["old_mean_in_model_input"]
        ),
    }
    gate, fold_metrics, scope_metrics, hidden_metrics, by_well = (
        evaluate_replacement_gate(
            config=config,
            base_frame=base,
            saved_parent=parent,
            oof_fold=oof_fold,
            new_prediction=mean_prediction,
            hidden_like_assignment_path=hidden_like_assignment_path,
            technical_checks=technical_checks,
        )
    )
    prediction_frame = base[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    prediction_frame["saved_exp335__lgb_mean__pred_tvt"] = parent[
        "signed_residual_meta_addonly__lgb_mean__pred_tvt"
    ].to_numpy(np.float32)
    for config_index, residual in zip(config_indices, oof_by_config, strict=True):
        prediction_frame[
            f"scale5_x1p0_full_replacement__lgb{config_index}__pred_tvt"
        ] = (anchor + residual).astype(np.float32)
    prediction_frame[
        "scale5_x1p0_full_replacement__lgb_mean__pred_tvt"
    ] = mean_prediction
    paths = {
        "oof": output_dir / "stage_d_oof_predictions.parquet",
        "fold_metrics": output_dir / "stage_d_fold_metrics.csv",
        "scope_metrics": output_dir / "stage_d_scope_metrics.csv",
        "hidden_metrics": output_dir / "stage_d_hidden_like_metrics.csv",
        "by_well": output_dir / "stage_d_by_well.csv",
        "importance": output_dir / "stage_d_feature_importance.csv",
        "model_manifest": output_dir / "stage_d_model_manifest.json",
        "metrics": output_dir / "stage_d_metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(
        fold_metrics, on="outer_fold", how="left"
    ).to_csv(paths["fold_metrics"], index=False)
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    pd.DataFrame(importance_rows).to_csv(paths["importance"], index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "replacement_stage_d_complete",
        "semantic_slot": SEMANTIC_SLOT,
        "value_source": REPLACEMENT_VALUE_SOURCE,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "feature_groups": {
            "clean_base": base_features,
            "nested_compact": parent_features,
            "signed_compact": signed_features,
        },
        "saved_exp335_control_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": "stage_d_complete_gate_passed"
        if gate["passed"]
        else "stage_d_complete_gate_failed_closed",
        "rows": n_rows,
        "wells": int(base["well"].nunique()),
        "feature_counts": {
            "clean_base": 273,
            "nested_compact": 74,
            "signed_compact": 23,
            "final": 370,
        },
        "model_count": len(model_rows),
        "cost_contract": cost,
        "primary_gate": gate,
    }
    artifact_sha = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "metrics"
    }
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "deterministic_anchor": False,
        "gpu_bitwise_deterministic_claimed": False,
        "replacement_value_source": REPLACEMENT_VALUE_SOURCE,
        "clean_base": base_evidence,
        "stage_c": {
            key: value
            for key, value in stage_c_evidence.items()
            if key not in {"partitions", "compact_features"}
        },
        "stage_s": {
            key: value
            for key, value in stage_s_evidence.items()
            if key not in {"partitions", "features"}
        },
        "saved_exp335_control": parent_evidence,
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": hidden_sha,
        },
        "artifact_sha256": artifact_sha,
        "oof_prediction_sha256": artifact_sha["oof"],
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "submission_generated": False,
        "primary_gate": gate,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    write_json(paths["metrics"], metrics)
    return metrics


def verify_replacement_stage_c_root(
    root: Path,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = verify_stage_c_artifact_root(
        root,
        runtime_config,
        verify_partition_sha256=True,
        expected_compact_feature_count=74,
        require_score_guard=False,
    )
    root = Path(root)
    lineage_path = root / "replacement_stage_c_lineage.json"
    semantic_path = root / "replacement_semantic_manifest.json"
    if not lineage_path.exists() or not semantic_path.exists():
        raise FileNotFoundError("replacement Stage C lineage or semantic manifest is missing")
    lineage = json.loads(lineage_path.read_text())
    if (
        str(lineage.get("value_source")) != REPLACEMENT_VALUE_SOURCE
        or bool(lineage.get("old_mean_in_model_input", True))
        or int(lineage.get("models_trained", -1)) != 40
        or int(lineage.get("control_models_retrained", -1)) != 0
    ):
        raise ValueError("replacement Stage C lineage contract mismatch")
    if str(lineage.get("semantic_manifest_sha256")) != sha256_file(semantic_path):
        raise ValueError("replacement Stage C semantic manifest SHA mismatch")
    if str(lineage.get("nested_compact_manifest_sha256")) != sha256_file(
        root / "nested_compact_manifest.json"
    ):
        raise ValueError("replacement Stage C compact manifest lineage mismatch")
    frozen_spec = dict(runtime_config["data"]["exp404_scale5_train_prediction"])
    for lineage_key, config_key in (
        ("frozen_raw_sha256", "expected_raw_sha256"),
        ("frozen_decompressed_sha256", "expected_decompressed_sha256"),
        ("frozen_logical_sha256", "expected_logical_sha256"),
        ("frozen_schema_sha256", "expected_schema_sha256"),
    ):
        if str(lineage.get(lineage_key)) != str(frozen_spec[config_key]):
            raise ValueError(f"replacement Stage C {lineage_key} mismatch")
    evidence["replacement_lineage_sha256"] = sha256_file(lineage_path)
    evidence["replacement_semantic_manifest_sha256"] = sha256_file(semantic_path)
    evidence["stage_0_preflight_sha256"] = str(lineage["stage_0_preflight_sha256"])
    return evidence


def verify_replacement_stage_s_root(
    root: Path,
    runtime_config: Mapping[str, Any],
    *,
    stage_c_root: Path | None = None,
) -> dict[str, Any]:
    evidence = verify_signed_stage_s_root(
        root,
        runtime_config,
        verify_partition_sha=True,
        verify_model_sha=True,
        require_score_gate=False,
    )
    root = Path(root)
    lineage_path = root / "replacement_stage_s_lineage.json"
    semantic_path = root / "replacement_semantic_manifest.json"
    if not lineage_path.exists() or not semantic_path.exists():
        raise FileNotFoundError("replacement Stage S lineage or semantic manifest is missing")
    lineage = json.loads(lineage_path.read_text())
    if (
        str(lineage.get("value_source")) != REPLACEMENT_VALUE_SOURCE
        or bool(lineage.get("old_mean_in_model_input", True))
        or int(lineage.get("models_trained", -1)) != 20
        or int(lineage.get("control_models_retrained", -1)) != 0
    ):
        raise ValueError("replacement Stage S lineage contract mismatch")
    if str(lineage.get("semantic_manifest_sha256")) != sha256_file(semantic_path):
        raise ValueError("replacement Stage S semantic manifest SHA mismatch")
    if str(lineage.get("signed_compact_manifest_sha256")) != sha256_file(
        root / "signed_compact_manifest.json"
    ):
        raise ValueError("replacement Stage S compact manifest lineage mismatch")
    if stage_c_root is not None and str(lineage.get("stage_c_lineage_sha256")) != sha256_file(
        Path(stage_c_root) / "replacement_stage_c_lineage.json"
    ):
        raise ValueError("replacement Stage S parent Stage C lineage mismatch")
    if stage_c_root is not None and sha256_file(semantic_path) != sha256_file(
        Path(stage_c_root) / "replacement_semantic_manifest.json"
    ):
        raise ValueError("replacement Stage S and Stage C semantic manifests differ")
    evidence["replacement_lineage_sha256"] = sha256_file(lineage_path)
    evidence["replacement_semantic_manifest_sha256"] = sha256_file(semantic_path)
    return evidence


__all__ = [
    "CHANGED_CANDIDATES",
    "REPLACEMENT_VALUE_SOURCE",
    "ReplacementCandidateCache",
    "SEMANTIC_SLOT",
    "UNCHANGED_CANDIDATES",
    "build_bank_from_primitives",
    "build_replacement_clean273_surface",
    "dataframe_content_sha",
    "dataframe_schema_sha",
    "deep_merge",
    "downstream_runtime_config",
    "evaluate_replacement_gate",
    "inspect_gzip_csv",
    "load_saved_exp335_control",
    "load_frozen_scale5_predictions",
    "patch_base_replay_primitive",
    "replacement_cache_factory",
    "replacement_cost_contract",
    "require_stage_authorization",
    "resolve_by_patterns",
    "run_replacement_preflight",
    "run_replacement_stage_d",
    "stage_c_runtime_config",
    "stage_s_runtime_config",
    "validate_replacement_contract",
    "verify_replacement_stage_c_root",
    "verify_replacement_stage_0_root",
    "verify_replacement_stage_s_root",
]
