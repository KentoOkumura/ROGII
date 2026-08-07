from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    FoldBundle,
    _nested_get,
    _rmse_arrays,
    audit_raw_context_availability,
    build_stage_d_exp218_surface,
    candidate_ids,
    compact_feature_names,
    contract_by_id,
    load_stage_d_compact_fold,
    resolve_existing_path,
    run_stage_a,
    run_stage_c,
    sha256_file,
    sha256_json,
    verify_stage_c_artifact_root,
    write_json,
)

VARIANTS = (
    "mean4_only",
    "mean8_only",
    "mean4_mean8_disagreement",
)
BASE_CANDIDATE_COUNT = 12
REPLACED_CANDIDATE = "pf_ancc"
EXTERNAL_CANDIDATES = ("pf_ancc_seed_mean_4", "pf_ancc_seed_mean_8")
PF_SOURCE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "pf_ancc_seed0",
    "pf_ancc_seed_mean_4",
    "pf_ancc_seed_mean_8",
    "pf_ancc_seed_std_4",
    "pf_ancc_seed_std_8",
    "pf_ancc_particle_std_mean_4",
    "pf_ancc_particle_std_mean_8",
    "pf_ancc_mean8_minus_mean4",
)
PF_FLOAT_COLUMNS = PF_SOURCE_COLUMNS[3:]
DISAGREEMENT_FIELDS = (
    "pf_ancc_seed_std_4",
    "pf_ancc_seed_std_8",
    "pf_ancc_particle_std_mean_4",
    "pf_ancc_particle_std_mean_8",
    "pf_ancc_mean8_minus_mean4",
    "pf_ancc_mean8_minus_mean4_abs",
)


def schema_sha(frame: pd.DataFrame) -> str:
    payload = json.dumps(
        [(str(column), str(frame[column].dtype)) for column in frame.columns],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_gzip_content(path: Path, chunk_bytes: int = 2**20) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def build_variant_contract(
    full_contract: Mapping[str, Any], variant: str
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown PF ANCC selector variant: {variant}")
    contract = copy.deepcopy(dict(full_contract))
    declared = [str(item["id"]) for item in contract["score_candidates"]]
    if len(declared) != BASE_CANDIDATE_COUNT + len(EXTERNAL_CANDIDATES):
        raise ValueError(
            "full exp277 candidate contract must declare exp263 core12 and two PF means"
        )
    if tuple(declared[-2:]) != EXTERNAL_CANDIDATES:
        raise ValueError("exp277 external candidate order is not canonical")
    base_declared = declared[:BASE_CANDIDATE_COUNT]
    if base_declared.count(REPLACED_CANDIDATE) != 1:
        raise ValueError("exp263 core12 must contain exactly one pf_ancc replacement slot")
    selected_external = [
        str(item) for item in contract["variants"][variant]["external_candidates"]
    ]
    if not selected_external or any(
        name not in EXTERNAL_CANDIDATES for name in selected_external
    ):
        raise ValueError(f"{variant} has an invalid PF ANCC replacement")
    specs = {str(item["id"]): item for item in contract["score_candidates"]}
    active_ids: list[str] = []
    for name in base_declared:
        if name == REPLACED_CANDIDATE:
            active_ids.extend(selected_external)
        else:
            active_ids.append(name)
    contract["score_candidates"] = [specs[name] for name in active_ids]
    for domain_name, domain in contract["legal_domains"].items():
        domain_ids = [str(item) for item in domain["candidates"]]
        if domain_ids.count(REPLACED_CANDIDATE) != 1:
            raise ValueError(
                f"{domain_name} must contain exactly one pf_ancc replacement slot"
            )
        active_domain: list[str] = []
        for name in domain_ids:
            if name in EXTERNAL_CANDIDATES:
                continue
            if name == REPLACED_CANDIDATE:
                active_domain.extend(selected_external)
            else:
                active_domain.append(name)
        domain["candidates"] = active_domain
    contract["active_variant"] = variant
    contract["replaced_candidate"] = REPLACED_CANDIDATE
    contract["disagreement_enabled"] = bool(
        contract["variants"][variant]["disagreement_enabled"]
    )
    contract["candidate_id_model_encoding"]["width"] = len(
        contract["score_candidates"]
    )
    names = candidate_ids(contract)
    expected = 13 if variant == "mean4_mean8_disagreement" else 12
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(f"{variant} candidate count mismatch")
    if REPLACED_CANDIDATE in names:
        raise ValueError(f"{variant} retained the replaced pf_ancc candidate")
    if any(name not in names for name in selected_external):
        raise ValueError(f"{variant} lost a replacement candidate")
    for domain_name, domain in contract["legal_domains"].items():
        domain_ids = [str(item) for item in domain["candidates"]]
        if REPLACED_CANDIDATE in domain_ids:
            raise ValueError(f"{variant} retained pf_ancc in {domain_name}")
        if any(name not in domain_ids for name in selected_external):
            raise ValueError(
                f"{variant} replacement candidate is absent from {domain_name}"
            )
    return contract


def build_base_contract(full_contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(dict(full_contract))
    declared = candidate_ids(contract)
    if len(declared) != BASE_CANDIDATE_COUNT + len(EXTERNAL_CANDIDATES):
        raise ValueError("exp263 base contract requires the full exp277 declaration")
    base_ids = set(declared[:BASE_CANDIDATE_COUNT])
    if REPLACED_CANDIDATE not in base_ids:
        raise ValueError("exp263 base contract lost pf_ancc")
    contract["score_candidates"] = [
        item for item in contract["score_candidates"] if str(item["id"]) in base_ids
    ]
    for domain in contract["legal_domains"].values():
        domain["candidates"] = [
            item for item in domain["candidates"] if str(item) in base_ids
        ]
    contract["candidate_id_model_encoding"]["width"] = BASE_CANDIDATE_COUNT
    return contract


def raw_test_only_schema_guard(
    schema: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    features = [str(item) for item in schema.get("features", [])]
    raw_cfg = dict(config["features"]["raw_context"])
    forbidden_columns = [
        str(item).lower()
        for item in raw_cfg.get("forbidden_training_only_columns", [])
    ]
    forbidden_features = {
        feature
        for column in forbidden_columns
        for feature in (
            f"ctx__raw__{column}",
            f"ctx__raw_delta_last__{column}",
        )
    }
    hits = sorted(set(features).intersection(forbidden_features))
    if hits:
        raise ValueError(
            "selector schema retained training-only raw context features: "
            f"{hits}"
        )
    return {
        "passed": True,
        "feature_count": len(features),
        "horizontal_numeric_allowlist": [
            str(item) for item in raw_cfg["horizontal_numeric_allowlist"]
        ],
        "forbidden_training_only_columns": forbidden_columns,
        "forbidden_feature_hits": hits,
    }


def load_pf_candidate_source(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    data_cfg = dict(config["data"])
    raw_sha = sha256_file(path)
    expected_raw = str(data_cfg["exp271_expected_raw_sha256"])
    if raw_sha != expected_raw:
        raise ValueError(f"exp271 candidate raw SHA mismatch: {raw_sha}")
    decompressed_sha = sha256_gzip_content(path)
    expected_decompressed = str(data_cfg["exp271_expected_decompressed_sha256"])
    if decompressed_sha != expected_decompressed:
        raise ValueError(
            "exp271 candidate decompressed SHA mismatch: "
            f"{decompressed_sha} != {expected_decompressed}"
        )
    dtypes: dict[str, Any] = {
        "id": "string",
        "well": "string",
        "row_idx": "int64",
        **{column: "float32" for column in PF_FLOAT_COLUMNS},
    }
    frame = pd.read_csv(path, usecols=list(PF_SOURCE_COLUMNS), dtype=dtypes)
    # exp271 schema evidence was emitted with object strings.  Pandas 3 may
    # otherwise retain the new ``str`` extension dtype and change the schema SHA.
    frame["id"] = frame["id"].astype(str).astype(object)
    frame["well"] = frame["well"].astype(str).astype(object)
    expected_rows = int(config["validation"]["expected_rows"])
    expected_wells = int(config["validation"]["expected_wells"])
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError("exp271 candidate coverage mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("exp271 candidate ids are not unique")
    id_well = frame["id"].str.rsplit("_", n=1).str[0]
    if not id_well.equals(frame["well"]):
        raise ValueError("exp271 id/well identity mismatch")
    actual_schema_sha = schema_sha(frame)
    expected_schema_sha = str(data_cfg["exp271_expected_schema_sha256"])
    if actual_schema_sha != expected_schema_sha:
        raise ValueError(
            f"exp271 candidate schema SHA mismatch: {actual_schema_sha} != {expected_schema_sha}"
        )
    if not np.isfinite(frame[list(PF_FLOAT_COLUMNS)].to_numpy(np.float32)).all():
        raise ValueError("exp271 candidate path contains non-finite values")
    frame["pf_ancc_mean8_minus_mean4_abs"] = frame[
        "pf_ancc_mean8_minus_mean4"
    ].abs().astype(np.float32)
    evidence = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "schema_sha256": actual_schema_sha,
    }
    return frame.set_index("id", drop=False), evidence


def augment_fold_bundle(
    base_bundle: FoldBundle,
    pf_source: pd.DataFrame,
    variant_contract: Mapping[str, Any],
) -> FoldBundle:
    ids = candidate_ids(variant_contract)
    external = [name for name in ids if name in EXTERNAL_CANDIDATES]
    if not external:
        raise ValueError("variant contract has no PF ANCC replacement")
    base_ids = [str(item) for item in base_bundle.candidate_ids]
    if base_ids.count(REPLACED_CANDIDATE) != 1:
        raise ValueError("exp263 fold bundle lacks one pf_ancc replacement slot")
    replaced_position = base_ids.index(REPLACED_CANDIDATE)
    expected_ids = [
        *base_ids[:replaced_position],
        *external,
        *base_ids[replaced_position + 1 :],
    ]
    if ids != expected_ids:
        raise ValueError("variant candidate order does not replace the pf_ancc slot")
    base_ids = base_bundle.base["id"].astype(str)
    aligned = pf_source.reindex(base_ids.to_numpy())
    if aligned["id"].isna().any():
        raise ValueError("exp271 candidate source is missing exp263 ids")
    if not np.array_equal(
        aligned["well"].astype(str).to_numpy(),
        base_bundle.base["well"].astype(str).to_numpy(),
    ):
        raise ValueError("exp271/exp263 well alignment mismatch")
    if not np.array_equal(
        aligned["row_idx"].to_numpy(np.int64),
        base_bundle.base["well_row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("exp271/exp263 row alignment mismatch")
    external_values = aligned[external].to_numpy(np.float32)
    if not np.isfinite(external_values).all():
        raise ValueError("aligned exp271 candidate values are non-finite")
    values = np.column_stack(
        [
            base_bundle.values[:, :replaced_position],
            external_values,
            base_bundle.values[:, replaced_position + 1 :],
        ]
    ).astype(np.float32)
    available = np.column_stack(
        [
            base_bundle.available[:, :replaced_position],
            np.ones_like(external_values, dtype=bool),
            base_bundle.available[:, replaced_position + 1 :],
        ]
    )
    confidence = dict(base_bundle.confidence)
    confidence.pop(REPLACED_CANDIDATE, None)
    disagreement_enabled = bool(variant_contract.get("disagreement_enabled", False))
    for name in external:
        conf = base_bundle.base[KEY_COLUMNS].copy()
        conf["candidate_id"] = name
        conf["confidence_source"] = (
            "exp271_seed_particle_disagreement" if disagreement_enabled else "none"
        )
        conf["confidence_valid"] = disagreement_enabled
        conf["confidence_missing_fields"] = "" if disagreement_enabled else "not_enabled"
        if disagreement_enabled:
            for field in DISAGREEMENT_FIELDS:
                conf[field] = aligned[field].to_numpy(np.float32)
        confidence[name] = conf
    return FoldBundle(
        base=base_bundle.base,
        values=values,
        available=available,
        confidence=confidence,
        candidate_ids=ids,
        specs=contract_by_id(variant_contract),
    )


class PfAnccAugmentedCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        full_contract: Mapping[str, Any],
        pf_source: pd.DataFrame,
    ):
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = contract_by_id(contract)
        self.base = Exp263CandidateCache(root, build_base_contract(full_contract))
        self.pf_source = pf_source

    def load_fold(self, fold: int) -> FoldBundle:
        return augment_fold_bundle(
            self.base.load_fold(fold), self.pf_source, self.contract
        )


def resolve_pf_candidate_source(
    config: Mapping[str, Any], search_roots: Sequence[Path]
) -> Path:
    return resolve_existing_path(
        [str(item) for item in config["data"]["exp271_candidate_patterns"]],
        search_roots,
    )


def verify_exp263_core_partitions(
    root: Path,
    full_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    base_contract = build_base_contract(full_contract)
    primitive = [
        str(item["id"])
        for item in base_contract["score_candidates"]
        if str(item["kind"]) == "primitive"
    ]
    evidence: list[dict[str, Any]] = []
    for manifest_key, directory in (
        ("candidate_value_partitions", "candidate_values"),
        ("candidate_confidence_partitions", "candidate_confidence"),
    ):
        inventory = manifest[manifest_key]
        for candidate in primitive:
            items = inventory[candidate]
            if len(items) != int(config["validation"]["outer_folds"]):
                raise ValueError(f"exp263 {candidate}/{directory} fold inventory mismatch")
            for fold, item in enumerate(items):
                paths = sorted((root / directory / candidate / f"fold={fold}").glob("*.parquet"))
                if len(paths) != 1:
                    raise ValueError(f"exp263 {candidate}/{directory}/fold={fold} is incomplete")
                actual_sha = sha256_file(paths[0])
                expected_sha = str(item["file_sha256"])
                if actual_sha != expected_sha:
                    raise ValueError(f"exp263 partition SHA mismatch: {paths[0]}")
                evidence.append(
                    {
                        "kind": directory,
                        "candidate_id": candidate,
                        "outer_fold": fold,
                        "path": str(paths[0]),
                        "sha256": actual_sha,
                        "rows": int(item["rows"]),
                    }
                )
    expected_count = int(config["guards"]["technical"]["expected_exp263_partitions"])
    if len(evidence) != expected_count:
        raise ValueError(f"exp263 core partition count mismatch: {len(evidence)}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "partition_count": len(evidence),
        "partitions": evidence,
    }


def run_nested_selector_variant(
    *,
    config: Mapping[str, Any],
    full_contract: Mapping[str, Any],
    variant: str,
    cache_root: Path,
    pf_candidate_path: Path,
    raw_train_dir: Path,
    raw_test_dir: Path,
    output_dir: Path,
    parent_schema_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_variant_contract(full_contract, variant)
    pf_source, pf_evidence = load_pf_candidate_source(pf_candidate_path, config)
    exp263_partition_evidence = verify_exp263_core_partitions(
        cache_root, full_contract, config
    )
    cache = PfAnccAugmentedCache(cache_root, contract, full_contract, pf_source)

    def cache_factory(_root: Path, _contract: Mapping[str, Any]) -> PfAnccAugmentedCache:
        return cache

    raw_context_availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        config["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    raw_context_availability.to_csv(
        output_dir / "raw_context_availability_audit.csv", index=False
    )
    stage_a = run_stage_a(
        config=config,
        contract=contract,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        parent_schema_path=parent_schema_path,
        cache_factory=cache_factory,
    )
    feature_schema = json.loads((output_dir / "feature_schema.json").read_text())
    schema_guard = raw_test_only_schema_guard(feature_schema, config)
    stage_a["raw_context_availability_passed"] = bool(
        raw_context_availability["availability_pass"].all()
    )
    stage_a["raw_context_availability_rows"] = raw_context_availability.to_dict(
        orient="records"
    )
    stage_a["raw_test_only_schema_guard"] = schema_guard
    write_json(output_dir / "stage_a_summary.json", stage_a)
    nested = run_stage_c(
        config=config,
        contract=contract,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        cache_factory=cache_factory,
        hard_readout_enabled=False,
    )
    if nested["hard_readout_enabled"] is not False:
        raise AssertionError("hard top-1 readout must remain disabled")
    if nested["hard_primary_oof_rmse"] is not None:
        raise AssertionError("hard top-1 readout artifact was unexpectedly generated")
    expected_features = len(compact_feature_names(contract))
    if int(stage_a["compact_meta_feature_count"]) != expected_features:
        raise AssertionError("variant compact feature count mismatch")
    if int(nested["model_count"]) != 40:
        raise AssertionError("nested selector did not train exactly 40 models")
    summary = {
        "status": "nested_selector_variant_completed",
        "variant": variant,
        "candidate_count": len(candidate_ids(contract)),
        "disagreement_enabled": bool(contract["disagreement_enabled"]),
        "planned_cpu_boosters": 40,
        "actual_model_count": int(nested["model_count"]),
        "parent_control_retraining": False,
        "hard_top1_eligible": False,
        "feature_audit": stage_a,
        "raw_test_only_schema_guard": schema_guard,
        "nested_selector": nested,
        "pf_source": pf_evidence,
        "exp263_core_partitions": exp263_partition_evidence,
        "contract_sha256": sha256_json(contract),
    }
    summary_path = output_dir / "exp277_nested_summary.json"
    write_json(summary_path, summary)
    evidence_paths = [
        summary_path,
        output_dir / "feature_schema.json",
        output_dir / "raw_context_availability_audit.csv",
        output_dir / "compact_meta_schema.json",
        output_dir / "nested_selector_metrics.json",
        output_dir / "nested_selector_model_manifest.json",
        output_dir / "nested_compact_manifest.json",
    ]
    reproducibility = {
        "schema_version": "1.0.0",
        "status": "nested_selector_variant_completed",
        "variant": variant,
        "pf_source": pf_evidence,
        "exp263_core_partitions": exp263_partition_evidence,
        "contract_sha256": sha256_json(contract),
        "output_sha256": {path.name: sha256_file(path) for path in evidence_paths},
        "model_manifest_sha256": sha256_file(
            output_dir / "nested_selector_model_manifest.json"
        ),
        "feature_schema_sha256": sha256_file(output_dir / "feature_schema.json"),
        "raw_context_availability_audit_sha256": sha256_file(
            output_dir / "raw_context_availability_audit.csv"
        ),
        "raw_test_only_schema_guard": schema_guard,
        "submission_sha256": None,
    }
    reproducibility_path = output_dir / "exp277_nested_reproducibility_manifest.json"
    write_json(reproducibility_path, reproducibility)
    summary["reproducibility_manifest_sha256"] = sha256_file(reproducibility_path)
    return summary


def resolve_complete_nested_root(
    patterns: Sequence[str], search_roots: Sequence[Path], variant: str
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.exists() and direct.is_dir():
            candidates.append(direct)
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path for path in root.glob(raw) if path.is_dir())
        candidates.extend(path.parent for path in root.rglob("exp277_nested_summary.json"))
    for candidate in dict.fromkeys(candidates):
        summary_path = candidate / "exp277_nested_summary.json"
        required = [
            summary_path,
            candidate / "exp277_nested_reproducibility_manifest.json",
            candidate / "nested_compact_manifest.json",
            candidate / "nested_selector_metrics.json",
            candidate / "nested_selector_model_manifest.json",
            candidate / "compact_meta_schema.json",
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            continue
        summary = json.loads(summary_path.read_text())
        if str(summary.get("variant")) == variant:
            return candidate
    raise FileNotFoundError(f"complete exp277 nested root not found: {list(patterns)}")


def load_fixed_control_oof(
    path: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    actual_sha = sha256_file(path)
    expected_sha = str(config["data"]["exp264_fixed_control_oof_sha256"])
    if actual_sha != expected_sha:
        raise ValueError(f"exp264 fixed control OOF SHA mismatch: {actual_sha}")
    columns = [
        "id",
        "well",
        "md_since",
        "outer_fold",
        "actual_tvt",
        "matched_control__lgb_mean__pred_tvt",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    if len(frame) != int(config["validation"]["expected_rows"]):
        raise ValueError("fixed control row count mismatch")
    if frame["well"].nunique() != int(config["validation"]["expected_wells"]):
        raise ValueError("fixed control well count mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("fixed control ids are not unique")
    numeric = frame[
        ["md_since", "actual_tvt", "matched_control__lgb_mean__pred_tvt"]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("fixed control contains non-finite values")
    return frame, {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
    }


def downstream_guard(
    *,
    pooled_delta_rmse: float,
    fold_deltas: Sequence[float],
    distance_1000_plus_delta_rmse: float,
    hidden_like_deltas: Sequence[float],
    worst_well_delta_rmse: float,
    guard_config: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "overall_improved": pooled_delta_rmse < 0.0,
        "improved_at_least_3_of_5_folds": int(np.sum(np.asarray(fold_deltas) < 0.0))
        >= int(guard_config["min_improved_folds"]),
        "distance_1000_plus_nonworse": distance_1000_plus_delta_rmse
        <= float(guard_config["max_1000_plus_delta_rmse"]),
        "hidden_like_two_surfaces_nonworse": max(hidden_like_deltas)
        <= float(guard_config["max_hidden_like_delta_rmse"]),
        "worst_well_regression_bounded": worst_well_delta_rmse
        <= float(guard_config["max_worst_well_regression"]),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "pooled_delta_rmse": float(pooled_delta_rmse),
        "fold_deltas": [float(item) for item in fold_deltas],
        "improved_folds": int(np.sum(np.asarray(fold_deltas) < 0.0)),
        "distance_1000_plus_delta_rmse": float(distance_1000_plus_delta_rmse),
        "hidden_like_deltas": [float(item) for item in hidden_like_deltas],
        "worst_well_delta_rmse": float(worst_well_delta_rmse),
    }


def run_downstream_variant(
    *,
    config: Mapping[str, Any],
    full_contract: Mapping[str, Any],
    variant: str,
    nested_root: Path,
    fixed_control_oof_path: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    base_feature_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    contract = build_variant_contract(full_contract, variant)
    nested_summary = json.loads((Path(nested_root) / "exp277_nested_summary.json").read_text())
    if str(nested_summary.get("variant")) != variant:
        raise ValueError("nested selector variant mismatch")
    expected_compact_count = len(compact_feature_names(contract))
    nested_evidence = verify_stage_c_artifact_root(
        nested_root,
        config,
        expected_compact_feature_count=expected_compact_count,
    )
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=base_feature_allowlist_path,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    stage_cfg = dict(config["model"]["downstream_tvt_stage"])
    if len(base_features) != int(stage_cfg["expected_base_feature_count"]):
        raise ValueError("exp218 base feature count mismatch")
    compact_features = [str(item) for item in nested_evidence["compact_features"]]
    final_features = [*base_features, *compact_features]
    if len(compact_features) != int(stage_cfg["expected_compact_feature_count"]):
        raise ValueError("selector compact feature count mismatch")
    if len(final_features) != int(stage_cfg["expected_final_feature_count"]):
        raise ValueError("downstream final feature count mismatch")
    mode_name = str(stage_cfg["mode"])
    mode_config = dict(_nested_get(exp218_config, f"model.training.modes.{mode_name}", {}) or {})
    if not bool(mode_config.get("use_gpu", False)):
        raise ValueError("exp277 downstream must use the fixed exp218 GPU mode")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode_config
    )
    config_indices = [int(item) for item in stage_cfg["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    folds = int(stage_cfg["folds"])
    expected_models = len(config_indices) * folds
    if expected_models != int(stage_cfg["planned_gpu_boosters_per_variant"]):
        raise ValueError("downstream booster contract mismatch")

    fixed, fixed_evidence = load_fixed_control_oof(fixed_control_oof_path, config)
    fixed_by_id = fixed.set_index("id")
    base_ids = base_frame["id"].astype(str)
    aligned_fixed = fixed_by_id.reindex(base_ids.to_numpy())
    if aligned_fixed["well"].isna().any():
        raise ValueError("fixed control is missing exp218 base ids")
    if not np.array_equal(
        aligned_fixed["well"].astype(str).to_numpy(), base_frame["well"].astype(str).to_numpy()
    ):
        raise ValueError("fixed control/base well alignment mismatch")
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    fixed_truth = aligned_fixed["actual_tvt"].to_numpy(np.float32)
    if float(np.max(np.abs(truth - fixed_truth))) > 1.0e-4:
        raise ValueError("fixed control truth differs from exp218 base truth")
    control = aligned_fixed["matched_control__lgb_mean__pred_tvt"].to_numpy(np.float32)
    fixed_fold = aligned_fixed["outer_fold"].to_numpy(np.int8)

    n_rows = len(base_frame)
    base_index = pd.Index(base_ids, name="id")
    oof = [np.full(n_rows, np.nan, np.float32) for _ in config_indices]
    oof_fold = np.full(n_rows, -1, np.int8)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "downstream_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    chunk_columns = int(stage_cfg["matrix_copy_chunk_columns"])

    for outer_fold in range(folds):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=nested_root,
            stage_c_evidence=nested_evidence,
            downstream_outer_fold=outer_fold,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("nested compact ids are absent from exp218 base")
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != n_rows:
            raise ValueError("nested compact train/valid does not cover base once")
        if np.any(fixed_fold[valid_indices] != outer_fold):
            raise ValueError("nested/fixed-control outer fold mismatch")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("downstream valid rows assigned twice")
        oof_fold[valid_indices] = np.int8(outer_fold)
        x_train_values = np.empty((len(train_indices), len(final_features)), np.float32)
        x_valid_values = np.empty((len(valid_indices), len(final_features)), np.float32)
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            block = base_frame.loc[:, columns]
            x_train_values[:, start:stop] = block.iloc[train_indices].to_numpy(
                np.float32, copy=True
            )
            x_valid_values[:, start:stop] = block.iloc[valid_indices].to_numpy(
                np.float32, copy=True
            )
        offset = len(base_features)
        x_train_values[:, offset:] = compact_train[compact_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, offset:] = compact_valid[compact_features].to_numpy(
            np.float32, copy=False
        )
        x_train = pd.DataFrame(x_train_values, columns=final_features, copy=False)
        x_valid = pd.DataFrame(x_valid_values, columns=final_features, copy=False)
        fold_predictions: list[np.ndarray] = []
        for position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                target[train_indices],
                eval_set=[(x_valid, target[valid_indices])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(int(stage_cfg["early_stopping_rounds"]), verbose=False),
                    log_evaluation(int(stage_cfg["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            prediction = model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            oof[position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"{variant}__lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            model_rows.append(
                {
                    "variant": variant,
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": outer_fold,
                    "feature_count": len(final_features),
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(importance_type=importance_type)
                importance_rows.extend(
                    {
                        "variant": variant,
                        "model": f"lgb{config_index}",
                        "outer_fold": outer_fold,
                        "importance_type": importance_type,
                        "feature": feature,
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, importance, strict=True)
                )
            print(
                json.dumps(
                    {
                        "stage": "downstream",
                        "variant": variant,
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "completed_boosters": len(model_rows),
                        "planned_boosters": expected_models,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        mean_tvt = anchor[valid_indices] + np.mean(np.vstack(fold_predictions), axis=0)
        fold_rows.append(
            {
                "variant": variant,
                "outer_fold": outer_fold,
                "rows": len(valid_indices),
                "control_rmse": _rmse_arrays(truth[valid_indices], control[valid_indices]),
                "addonly_rmse": _rmse_arrays(truth[valid_indices], mean_tvt),
            }
        )
        fold_rows[-1]["delta_rmse"] = (
            fold_rows[-1]["addonly_rmse"] - fold_rows[-1]["control_rmse"]
        )
        del compact_train, compact_valid, x_train, x_valid, x_train_values, x_valid_values
        gc.collect()

    if len(model_rows) != expected_models or np.any(oof_fold < 0):
        raise AssertionError("downstream model/fold contract incomplete")
    if any(not np.isfinite(prediction).all() for prediction in oof):
        raise AssertionError("downstream OOF is incomplete")
    addonly_configs = [(anchor + prediction).astype(np.float32) for prediction in oof]
    addonly = np.mean(np.vstack(addonly_configs), axis=0).astype(np.float32)
    pooled_control = _rmse_arrays(truth, control)
    pooled_addonly = _rmse_arrays(truth, addonly)
    expected_control_rmse = float(config["data"]["exp264_fixed_control_expected_rmse"])
    if abs(pooled_control - expected_control_rmse) > 1.0e-6:
        raise ValueError(
            "fixed control RMSE parity failed: "
            f"{pooled_control} != {expected_control_rmse}"
        )

    by_well_source = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "truth": truth,
            "control": control,
            "addonly": addonly,
        }
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in by_well_source.groupby("well", sort=True):
        control_rmse = _rmse_arrays(group["truth"], group["control"])
        add_rmse = _rmse_arrays(group["truth"], group["addonly"])
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "control_rmse": control_rmse,
                "addonly_rmse": add_rmse,
                "delta_rmse": add_rmse - control_rmse,
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    md_since = base_frame["md_since"].to_numpy(np.float32)
    bucket_rows: list[dict[str, Any]] = []
    for bucket, mask in {
        "all": np.ones(n_rows, dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }.items():
        control_rmse = _rmse_arrays(truth[mask], control[mask])
        add_rmse = _rmse_arrays(truth[mask], addonly[mask])
        bucket_rows.append(
            {
                "bucket": bucket,
                "rows": int(mask.sum()),
                "control_rmse": control_rmse,
                "addonly_rmse": add_rmse,
                "delta_rmse": add_rmse - control_rmse,
            }
        )
    bucket_metrics = pd.DataFrame(bucket_rows)

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str}).set_index(
        "well_id"
    )
    hidden_rows: list[dict[str, Any]] = []
    for column in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        mask = base_frame["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        control_rmse = _rmse_arrays(truth[mask], control[mask])
        add_rmse = _rmse_arrays(truth[mask], addonly[mask])
        hidden_rows.append(
            {
                "assignment": column,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "control_rmse": control_rmse,
                "addonly_rmse": add_rmse,
                "delta_rmse": add_rmse - control_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    guard = downstream_guard(
        pooled_delta_rmse=pooled_addonly - pooled_control,
        fold_deltas=fold_metrics["delta_rmse"].tolist(),
        distance_1000_plus_delta_rmse=float(
            bucket_metrics.set_index("bucket").loc["1000_plus", "delta_rmse"]
        ),
        hidden_like_deltas=hidden_metrics["delta_rmse"].tolist(),
        worst_well_delta_rmse=float(by_well["delta_rmse"].max()),
        guard_config=config["guards"]["downstream_tvt_addonly"],
    )

    prediction_frame = base_frame[["id", "well", "md_since", "last_known_tvt"]].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    prediction_frame["fixed_control_pred_tvt"] = control
    for config_index, prediction in zip(config_indices, addonly_configs, strict=True):
        prediction_frame[f"{variant}__lgb{config_index}__pred_tvt"] = prediction
    prediction_frame[f"{variant}__lgb_mean__pred_tvt"] = addonly

    fold_path = output_dir / "downstream_fold_metrics.csv"
    oof_path = output_dir / "downstream_oof_predictions.parquet"
    importance_path = output_dir / "downstream_feature_importance.csv"
    by_well_path = output_dir / "downstream_by_well.csv"
    bucket_path = output_dir / "downstream_bucket_metrics.csv"
    hidden_path = output_dir / "downstream_hidden_like_metrics.csv"
    manifest_path = output_dir / "downstream_model_manifest.json"
    metrics_path = output_dir / "downstream_metrics.json"
    fold_metrics.to_csv(fold_path, index=False)
    prediction_frame.to_parquet(oof_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    hidden_metrics.to_csv(hidden_path, index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "downstream_variant_completed",
        "variant": variant,
        "model_count": len(model_rows),
        "control_retraining": False,
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "nested_input": nested_evidence,
        "fixed_control_input": fixed_evidence,
        "exp218_input": base_evidence,
    }
    write_json(manifest_path, model_manifest)
    metrics = {
        "status": "downstream_variant_completed",
        "variant": variant,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "model_count": len(model_rows),
        "control_retraining": False,
        "base_feature_count": len(base_features),
        "compact_feature_count": len(compact_features),
        "final_feature_count": len(final_features),
        "fixed_control_rmse": pooled_control,
        "addonly_rmse": pooled_addonly,
        "delta_rmse_addonly_minus_control": pooled_addonly - pooled_control,
        "guard": guard,
    }
    write_json(metrics_path, metrics)
    outputs = [
        metrics_path,
        fold_path,
        oof_path,
        manifest_path,
        importance_path,
        by_well_path,
        bucket_path,
        hidden_path,
    ]
    output_sha = {path.name: sha256_file(path) for path in outputs}
    reproducibility = {
        "schema_version": "1.0.0",
        "status": "downstream_variant_completed",
        "variant": variant,
        "nested_input": nested_evidence,
        "fixed_control_input": fixed_evidence,
        "exp218_input": base_evidence,
        "hidden_like_assignment_sha256": sha256_file(hidden_like_assignment_path),
        "model_manifest_sha256": output_sha[manifest_path.name],
        "prediction_sha256": output_sha[oof_path.name],
        "output_sha256": output_sha,
        "guard": guard,
    }
    write_json(output_dir / "downstream_reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = output_sha
    return metrics


def resolve_downstream_root(
    patterns: Sequence[str], search_roots: Sequence[Path], variant: str
) -> Path:
    candidates: list[Path] = []
    for raw in patterns:
        direct = Path(raw)
        if direct.exists() and direct.is_dir():
            candidates.append(direct)
    for root in search_roots:
        if not root.exists():
            continue
        for raw in patterns:
            if not Path(raw).is_absolute():
                candidates.extend(path for path in root.glob(raw) if path.is_dir())
        candidates.extend(path.parent for path in root.rglob("downstream_metrics.json"))
    for candidate in dict.fromkeys(candidates):
        metrics_path = candidate / "downstream_metrics.json"
        repro_path = candidate / "downstream_reproducibility_manifest.json"
        oof_path = candidate / "downstream_oof_predictions.parquet"
        required = [metrics_path, repro_path, oof_path]
        if not all(path.exists() and path.stat().st_size > 0 for path in required):
            continue
        metrics = json.loads(metrics_path.read_text())
        repro = json.loads(repro_path.read_text())
        if metrics.get("variant") != variant or repro.get("variant") != variant:
            continue
        expected_oof_sha = str(repro["output_sha256"][oof_path.name])
        if sha256_file(oof_path) != expected_oof_sha:
            raise ValueError(f"{variant} downstream OOF SHA mismatch")
        return candidate
    raise FileNotFoundError(f"complete downstream root not found for {variant}")


def aggregate_downstream_variants(
    *, variant_roots: Mapping[str, Path], output_dir: Path
) -> dict[str, Any]:
    if set(variant_roots) != set(VARIANTS):
        raise ValueError("aggregate requires all three canonical variants")
    comparison_rows: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None
    prediction_by_variant: dict[str, np.ndarray] = {}
    truth: np.ndarray | None = None
    control: np.ndarray | None = None
    input_evidence: dict[str, Any] = {}
    for variant in VARIANTS:
        root = Path(variant_roots[variant])
        metrics = json.loads((root / "downstream_metrics.json").read_text())
        repro = json.loads((root / "downstream_reproducibility_manifest.json").read_text())
        oof_path = root / "downstream_oof_predictions.parquet"
        if sha256_file(oof_path) != str(repro["output_sha256"][oof_path.name]):
            raise ValueError(f"aggregate input SHA mismatch for {variant}")
        pred_column = f"{variant}__lgb_mean__pred_tvt"
        frame = pd.read_parquet(
            oof_path,
            columns=[
                "id",
                "well",
                "md_since",
                "outer_fold",
                "actual_tvt",
                "fixed_control_pred_tvt",
                pred_column,
            ],
        )
        identity = frame[["id", "well", "md_since", "outer_fold"]].copy()
        if reference_identity is None:
            reference_identity = identity
            truth = frame["actual_tvt"].to_numpy(np.float32)
            control = frame["fixed_control_pred_tvt"].to_numpy(np.float32)
        else:
            if not identity.equals(reference_identity):
                raise ValueError(f"aggregate row identity mismatch for {variant}")
            if not np.array_equal(
                frame["actual_tvt"].to_numpy(np.float32), truth
            ) or not np.array_equal(
                frame["fixed_control_pred_tvt"].to_numpy(np.float32), control
            ):
                raise ValueError(f"aggregate fixed truth/control mismatch for {variant}")
        prediction_by_variant[variant] = frame[pred_column].to_numpy(np.float32)
        comparison_rows.append(
            {
                "variant": variant,
                "fixed_control_rmse": float(metrics["fixed_control_rmse"]),
                "addonly_rmse": float(metrics["addonly_rmse"]),
                "delta_rmse_addonly_minus_control": float(
                    metrics["delta_rmse_addonly_minus_control"]
                ),
                "guard_passed": bool(metrics["guard"]["passed"]),
                "improved_folds": int(metrics["guard"]["improved_folds"]),
                "distance_1000_plus_delta_rmse": float(
                    metrics["guard"]["distance_1000_plus_delta_rmse"]
                ),
                "hidden_like_max_delta_rmse": float(
                    max(metrics["guard"]["hidden_like_deltas"])
                ),
                "worst_well_delta_rmse": float(
                    metrics["guard"]["worst_well_delta_rmse"]
                ),
            }
        )
        input_evidence[variant] = {
            "root": str(root),
            "oof_sha256": sha256_file(oof_path),
            "metrics_sha256": sha256_file(root / "downstream_metrics.json"),
            "model_manifest_sha256": str(repro["model_manifest_sha256"]),
        }
    comparison = pd.DataFrame(comparison_rows)
    rmse = comparison.set_index("variant")["addonly_rmse"]
    tolerance = 1.0e-6
    mean8_dependency = bool(
        min(rmse["mean8_only"], rmse["mean4_mean8_disagreement"])
        < rmse["mean4_only"] - tolerance
    )
    four_seed_contract_supported = bool(
        comparison.set_index("variant").loc["mean4_only", "guard_passed"]
        and not mean8_dependency
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "aggregate_variant_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    summary = {
        "status": "aggregate_compare_completed",
        "variants": list(VARIANTS),
        "comparison": comparison.to_dict(orient="records"),
        "mean8_dependency": mean8_dependency,
        "four_seed_contract_supported": four_seed_contract_supported,
        "single_candidate_control": "mean4_only",
        "input_evidence": input_evidence,
    }
    summary_path = output_dir / "aggregate_summary.json"
    write_json(summary_path, summary)
    artifact_sha = {
        comparison_path.name: sha256_file(comparison_path),
        summary_path.name: sha256_file(summary_path),
    }
    reproducibility_path = output_dir / "aggregate_reproducibility_manifest.json"
    write_json(
        reproducibility_path,
        {
            "schema_version": "1.0.0",
            "status": "aggregate_compare_completed",
            "input_evidence": input_evidence,
            "output_sha256": artifact_sha,
            "model_manifest_sha256": None,
            "submission_sha256": None,
        },
    )
    artifact_sha[reproducibility_path.name] = sha256_file(reproducibility_path)
    summary["artifact_sha256"] = artifact_sha
    return summary


__all__ = [
    "BASE_CANDIDATE_COUNT",
    "DISAGREEMENT_FIELDS",
    "EXTERNAL_CANDIDATES",
    "PF_SOURCE_COLUMNS",
    "PfAnccAugmentedCache",
    "REPLACED_CANDIDATE",
    "VARIANTS",
    "aggregate_downstream_variants",
    "augment_fold_bundle",
    "build_base_contract",
    "build_variant_contract",
    "downstream_guard",
    "load_fixed_control_oof",
    "load_pf_candidate_source",
    "resolve_complete_nested_root",
    "resolve_downstream_root",
    "resolve_pf_candidate_source",
    "run_downstream_variant",
    "run_nested_selector_variant",
    "schema_sha",
    "sha256_gzip_content",
    "verify_exp263_core_partitions",
]
