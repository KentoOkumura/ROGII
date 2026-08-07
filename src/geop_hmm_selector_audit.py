from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    contract_by_id,
    load_stage_d_compact_fold,
    resolve_existing_path,
    run_stage_a,
    run_stage_b,
    run_stage_c,
    sha256_file,
    sha256_json,
    verify_stage_c_artifact_root,
    write_json,
)

BASE_CANDIDATE_COUNT = 12
ADDED_CANDIDATE = "geop_hmm"
GEOP_SOURCE_COLUMNS = (
    "id",
    "well",
    "row_idx",
    "fold",
    "geop_hmm",
    "geop_hmm_std",
    "geop_hmm_loglik",
)
GEOP_CONFIDENCE_FIELDS = (
    "sigma_tvt",
    "source_loglik",
    "loglik_per_row",
    "candidate_finite_source",
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


def build_base_contract(full_contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = copy.deepcopy(dict(full_contract))
    declared = candidate_ids(contract)
    if len(declared) != BASE_CANDIDATE_COUNT + 1 or declared[-1] != ADDED_CANDIDATE:
        raise ValueError("exp286 contract must append geop_hmm after the exp264 core12")
    base_ids = set(declared[:BASE_CANDIDATE_COUNT])
    contract["score_candidates"] = [
        item for item in contract["score_candidates"] if str(item["id"]) in base_ids
    ]
    for domain in contract["legal_domains"].values():
        domain["candidates"] = [
            item for item in domain["candidates"] if str(item) in base_ids
        ]
    contract["candidate_id_model_encoding"]["width"] = BASE_CANDIDATE_COUNT
    return contract


def validate_full_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    names = candidate_ids(contract)
    if len(names) != 13 or len(set(names)) != 13 or names[-1] != ADDED_CANDIDATE:
        raise ValueError("exp286 requires the exp264 core12 plus final geop_hmm")
    spec = contract_by_id(contract)[ADDED_CANDIDATE]
    if str(spec.get("kind")) != "primitive":
        raise ValueError("geop_hmm must use the same primitive kind as other raw paths")
    if str(spec.get("family")) != "geop_centered_exact_hmm":
        raise ValueError("geop_hmm family mismatch")
    encoding = contract["candidate_id_model_encoding"]
    if (
        str(encoding.get("type")) != "one_hot"
        or int(encoding.get("width", -1)) != 13
        or bool(encoding.get("ordinal_index_as_model_feature"))
        or not bool(encoding.get("keep_string_id_in_artifacts"))
    ):
        raise ValueError("geop_hmm candidate ID encoding contract is invalid")
    expected_domains = {"primitive_pair_bank": 12, "primitive_fixed_bank": 8}
    for domain_name, expected_count in expected_domains.items():
        domain = [str(item) for item in contract["legal_domains"][domain_name]["candidates"]]
        if len(domain) != expected_count or ADDED_CANDIDATE not in domain:
            raise ValueError(f"geop_hmm is not registered in {domain_name}")
    build_base_contract(contract)
    return {
        "candidate_count": len(names),
        "candidate_order": names,
        "geop_hmm_spec": spec,
        "primary_domain_count": expected_domains["primitive_pair_bank"],
        "fixed_domain_count": expected_domains["primitive_fixed_bank"],
        "candidate_contract_sha256": sha256_json(contract),
    }


@dataclass(frozen=True)
class GeopCandidateStore:
    fold_paths: dict[int, Path]
    fold_rows: dict[int, int]
    evidence: dict[str, Any]

    def load_fold(self, fold: int) -> pd.DataFrame:
        if fold not in self.fold_paths:
            raise ValueError(f"geop source fold is unavailable: {fold}")
        frame = pd.read_parquet(self.fold_paths[fold])
        frame["id"] = frame["id"].astype(str).astype(object)
        frame["well"] = frame["well"].astype(str).astype(object)
        if len(frame) != self.fold_rows[fold] or frame["id"].duplicated().any():
            raise ValueError(f"geop source fold {fold} ID inventory mismatch")
        if not frame["outer_fold"].eq(fold).all():
            raise ValueError(f"geop source fold {fold} contains cross-fold rows")
        frame["evaluation_rows_in_well"] = (
            frame.groupby("well", sort=False)["id"].transform("size").astype(np.int32)
        )
        frame["loglik_per_row"] = (
            frame["geop_hmm_loglik"] / frame["evaluation_rows_in_well"]
        ).astype(np.float32)
        return frame.set_index("id", drop=False)


def load_geop_candidate_source(
    path: Path,
    config: Mapping[str, Any],
    *,
    cache_root: Path | None = None,
) -> tuple[GeopCandidateStore, dict[str, Any]]:
    path = Path(path)
    source_cfg = dict(config["data"]["exp279_oof"])
    raw_sha = sha256_file(path)
    if raw_sha != str(source_cfg["expected_raw_gzip_sha256"]):
        raise ValueError(f"exp279 raw gzip SHA mismatch: {raw_sha}")
    decompressed_sha = sha256_gzip_content(path)
    if decompressed_sha != str(source_cfg["expected_decompressed_sha256"]):
        raise ValueError(f"exp279 decompressed SHA mismatch: {decompressed_sha}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected_rows = int(config["guards"]["technical"]["expected_rows"])
    expected_wells = int(config["guards"]["technical"]["expected_wells"])
    exp263_well_fold: dict[str, int] = {}
    exp263_fold_rows: dict[int, int] = {}
    if cache_root is not None:
        cache_root = Path(cache_root)
        for fold in range(5):
            paths = sorted(
                (cache_root / "candidate_values" / "exp226_k16" / f"fold={fold}").glob(
                    "*.parquet"
                )
            )
            if len(paths) != 1:
                raise ValueError(f"exp263 fold {fold} reference partition is incomplete")
            reference = pd.read_parquet(paths[0], columns=["well"])
            exp263_fold_rows[fold] = len(reference)
            for well in reference["well"].astype(str).unique():
                if well in exp263_well_fold:
                    raise ValueError("exp263 well appears in multiple outer folds")
                exp263_well_fold[well] = fold
    if cache_root is not None and len(exp263_well_fold) != expected_wells:
        raise ValueError("exp263 reference fold well inventory mismatch")
    temporary_root = Path(tempfile.mkdtemp(prefix="exp286_geop_source_"))
    fold_paths = {fold: temporary_root / f"fold={fold}.parquet" for fold in range(5)}
    writers: dict[int, Any] = {}
    fold_rows = {fold: 0 for fold in range(5)}
    total_rows = 0
    well_fold: dict[str, int] = {}
    well_loglik: dict[str, float] = {}
    source_outer_fold_match_rows = 0
    loaded_schema: list[tuple[str, str]] | None = None
    dtypes: dict[str, Any] = {
        "id": "string",
        "well": "string",
        "row_idx": "int64",
        "fold": "int8",
        "geop_hmm": "float32",
        "geop_hmm_std": "float32",
        "geop_hmm_loglik": "float32",
    }
    try:
        for chunk in pd.read_csv(
            path,
            usecols=list(GEOP_SOURCE_COLUMNS),
            dtype=dtypes,
            chunksize=200_000,
        ):
            chunk["id"] = chunk["id"].astype(str).astype(object)
            chunk["well"] = chunk["well"].astype(str).astype(object)
            if loaded_schema is None:
                loaded_schema = [
                    (str(column), str(chunk[column].dtype)) for column in chunk.columns
                ]
            id_well = chunk["id"].str.rsplit("_", n=1).str[0]
            if not id_well.equals(chunk["well"]):
                raise ValueError("exp279 id/well identity mismatch")
            numeric = chunk[
                ["geop_hmm", "geop_hmm_std", "geop_hmm_loglik"]
            ].to_numpy(np.float32)
            if not np.isfinite(numeric).all() or bool(
                (chunk["geop_hmm_std"] < 0).any()
            ):
                raise ValueError("exp279 geop path/confidence contains invalid values")
            if not chunk["fold"].isin(range(5)).all():
                raise ValueError("exp279 contains an invalid outer fold")
            if exp263_well_fold:
                assigned = chunk["well"].map(exp263_well_fold)
                if assigned.isna().any():
                    raise ValueError("exp279 contains a well absent from exp263 outer folds")
                chunk["outer_fold"] = assigned.astype(np.int8)
            else:
                chunk["outer_fold"] = chunk["fold"].astype(np.int8)
            source_outer_fold_match_rows += int(
                chunk["fold"].eq(chunk["outer_fold"]).sum()
            )
            for well, group in chunk.groupby("well", sort=False):
                fold_values = group["fold"].unique()
                loglik_values = group["geop_hmm_loglik"].unique()
                if len(fold_values) != 1 or len(loglik_values) != 1:
                    raise ValueError("exp279 fold/loglik is inconsistent within well")
                fold = int(group["outer_fold"].iloc[0])
                loglik = float(loglik_values[0])
                if str(well) in well_fold and well_fold[str(well)] != fold:
                    raise ValueError("exp279 well appears in multiple outer folds")
                if str(well) in well_loglik and well_loglik[str(well)] != loglik:
                    raise ValueError("exp279 source loglik changes within well")
                well_fold[str(well)] = fold
                well_loglik[str(well)] = loglik
            for fold in range(5):
                part = chunk.loc[chunk["outer_fold"].eq(fold)].reset_index(drop=True)
                if part.empty:
                    continue
                table = pa.Table.from_pandas(part, preserve_index=False)
                if fold not in writers:
                    writers[fold] = pq.ParquetWriter(
                        fold_paths[fold], table.schema, compression="zstd"
                    )
                writers[fold].write_table(table)
                fold_rows[fold] += len(part)
            total_rows += len(chunk)
    finally:
        for writer in writers.values():
            writer.close()
    if total_rows != expected_rows or len(well_fold) != expected_wells:
        raise ValueError("exp279 geop candidate coverage mismatch")
    if set(writers) != set(range(5)) or any(rows <= 0 for rows in fold_rows.values()):
        raise ValueError("exp279 geop fold materialization is incomplete")
    if exp263_fold_rows and fold_rows != exp263_fold_rows:
        raise ValueError(
            f"exp279/exp263 assigned fold row mismatch: {fold_rows} != {exp263_fold_rows}"
        )
    schema_payload = json.dumps(loaded_schema, ensure_ascii=False, separators=(",", ":"))
    evidence = {
        "source": "exp279_geop_centered_exact_hmm_oof",
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "rows": total_rows,
        "wells": len(well_fold),
        "fold_rows": fold_rows,
        "fold_assignment": (
            "exp263_reference_candidate_well_outer_fold"
            if exp263_well_fold
            else "exp279_source_fold"
        ),
        "exp279_source_fold_match_rate_to_exp263_outer_fold": (
            source_outer_fold_match_rows / total_rows
        ),
        "raw_gzip_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "prediction_content_sha256": str(source_cfg["expected_prediction_content_sha256"]),
        "loaded_schema_sha256": hashlib.sha256(schema_payload.encode("utf-8")).hexdigest(),
        "loaded_columns": list(GEOP_SOURCE_COLUMNS),
        "truth_columns_loaded": [],
        "materialization_policy": "chunked_gzip_to_temporary_fold_parquet",
        "temporary_fold_parquet": {
            str(fold): {
                "rows": fold_rows[fold],
                "sha256": sha256_file(fold_paths[fold]),
            }
            for fold in range(5)
        },
    }
    store = GeopCandidateStore(fold_paths=fold_paths, fold_rows=fold_rows, evidence=evidence)
    for fold in range(5):
        frame = store.load_fold(fold)
        if frame["well"].nunique() != sum(value == fold for value in well_fold.values()):
            raise ValueError(f"exp279 geop well inventory mismatch in fold {fold}")
    return store, evidence


def augment_fold_bundle(
    base_bundle: FoldBundle,
    geop_source: pd.DataFrame,
    full_contract: Mapping[str, Any],
) -> FoldBundle:
    ids = candidate_ids(full_contract)
    if ids != [*base_bundle.candidate_ids, ADDED_CANDIDATE]:
        raise ValueError("geop_hmm must be appended after the unchanged exp264 core12")
    base_ids = base_bundle.base["id"].astype(str)
    aligned = geop_source.reindex(base_ids.to_numpy())
    if aligned["id"].isna().any():
        raise ValueError("exp279 geop source is missing exp263 IDs")
    if not np.array_equal(
        aligned["well"].astype(str).to_numpy(),
        base_bundle.base["well"].astype(str).to_numpy(),
    ):
        raise ValueError("exp279/exp263 well alignment mismatch")
    if not np.array_equal(
        aligned["row_idx"].to_numpy(np.int64),
        base_bundle.base["well_row_idx"].to_numpy(np.int64),
    ):
        raise ValueError("exp279/exp263 row alignment mismatch")
    aligned_fold_column = "outer_fold" if "outer_fold" in aligned else "fold"
    if not np.array_equal(
        aligned[aligned_fold_column].to_numpy(np.int8),
        base_bundle.base["outer_fold"].to_numpy(np.int8),
    ):
        raise ValueError("exp279/exp263 outer-fold alignment mismatch")
    geop_values = aligned["geop_hmm"].to_numpy(np.float32)
    if not np.isfinite(geop_values).all():
        raise ValueError("aligned geop_hmm values are non-finite")
    values = np.column_stack([base_bundle.values, geop_values]).astype(np.float32)
    available = np.column_stack(
        [base_bundle.available, np.ones(len(geop_values), dtype=bool)]
    )
    confidence = dict(base_bundle.confidence)
    conf = base_bundle.base[KEY_COLUMNS].copy()
    conf["candidate_id"] = ADDED_CANDIDATE
    conf["confidence_source"] = "exp279_geop_centered_exact_hmm_posterior"
    valid = np.isfinite(
        aligned[["geop_hmm", "geop_hmm_std", "geop_hmm_loglik"]].to_numpy(np.float32)
    ).all(axis=1)
    conf["confidence_valid"] = valid
    conf["confidence_missing_fields"] = ""
    conf["sigma_tvt"] = aligned["geop_hmm_std"].to_numpy(np.float32)
    conf["source_loglik"] = aligned["geop_hmm_loglik"].to_numpy(np.float32)
    conf["loglik_per_row"] = aligned["loglik_per_row"].to_numpy(np.float32)
    conf["candidate_finite_source"] = np.isfinite(geop_values).astype(np.int8)
    confidence[ADDED_CANDIDATE] = conf
    return FoldBundle(
        base=base_bundle.base,
        values=values,
        available=available,
        confidence=confidence,
        candidate_ids=ids,
        specs=contract_by_id(full_contract),
    )


class GeopHmmAugmentedCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        geop_source: GeopCandidateStore,
    ):
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = contract_by_id(contract)
        self.base = Exp263CandidateCache(root, build_base_contract(contract))
        self.geop_source = geop_source

    def load_fold(self, fold: int) -> FoldBundle:
        return augment_fold_bundle(
            self.base.load_fold(fold), self.geop_source.load_fold(fold), self.contract
        )


def resolve_geop_candidate_source(
    config: Mapping[str, Any], search_roots: Sequence[Path]
) -> Path:
    return resolve_existing_path(
        [str(item) for item in config["data"]["exp279_oof"]["patterns"]], search_roots
    )


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
        for feature in (f"ctx__raw__{column}", f"ctx__raw_delta_last__{column}")
    }
    hits = sorted(set(features).intersection(forbidden_features))
    if hits:
        raise ValueError(f"selector schema retained training-only context: {hits}")
    return {
        "passed": True,
        "feature_count": len(features),
        "forbidden_training_only_columns": forbidden_columns,
        "forbidden_feature_hits": hits,
    }


def load_parent_baseline(
    metrics_path: Path,
    candidate_metrics_path: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    baseline_cfg = dict(config["data"]["parent_stage_b_v5"])
    actual = {
        "selector_metrics_sha256": sha256_file(metrics_path),
        "selector_candidate_metrics_sha256": sha256_file(candidate_metrics_path),
    }
    for key, value in actual.items():
        if value != str(baseline_cfg[f"expected_{key}"]):
            raise ValueError(f"parent Stage B v5 artifact SHA mismatch for {key}: {value}")
    metrics = json.loads(Path(metrics_path).read_text())
    candidates = pd.read_csv(candidate_metrics_path)
    if int(metrics.get("model_count", -1)) != 10 or len(candidates) != 60:
        raise ValueError("parent Stage B v5 baseline shape mismatch")
    return metrics, candidates, actual


def compare_with_parent_stage_b(
    *,
    output_dir: Path,
    new_metrics: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    parent_candidate_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    new_fold = pd.DataFrame(new_metrics["fold_metrics"]).sort_values("fold")
    parent_fold = pd.DataFrame(parent_metrics["fold_metrics"]).sort_values("fold")
    if not np.array_equal(new_fold["fold"].to_numpy(), parent_fold["fold"].to_numpy()):
        raise ValueError("new/parent selector fold mismatch")
    fold_comparison = parent_fold.add_prefix("parent__").join(
        new_fold.add_prefix("new__")
    )
    fold_comparison["fold"] = new_fold["fold"].to_numpy()
    for metric in (
        "hard_primary_rmse",
        "rank_regret_pred_abs_error",
        "rank_regret_p_within10",
        "top3_oracle_coverage_pred_abs_error",
        "top3_oracle_coverage_p_within10",
    ):
        fold_comparison[f"delta__{metric}__new_minus_parent"] = (
            new_fold[metric].to_numpy(np.float64)
            - parent_fold[metric].to_numpy(np.float64)
        )
    fold_path = output_dir / "selector_parent_fold_comparison.csv"
    fold_comparison.to_csv(fold_path, index=False)

    new_candidate = pd.read_csv(output_dir / "selector_candidate_metrics.csv")
    shared = new_candidate[new_candidate["candidate_id"].ne(ADDED_CANDIDATE)].merge(
        parent_candidate_metrics,
        on=["outer_fold", "candidate_id"],
        suffixes=("__new13", "__parent12"),
        validate="one_to_one",
    )
    if len(shared) != 60:
        raise ValueError("shared-12 candidate comparison is incomplete")
    for metric in ("expected_error_mae", "within10_logloss", "within10_brier"):
        shared[f"delta__{metric}__new13_minus_parent12"] = (
            shared[f"{metric}__new13"] - shared[f"{metric}__parent12"]
        )
    shared_path = output_dir / "selector_shared12_candidate_metric_comparison.csv"
    shared.to_csv(shared_path, index=False)

    selection = pd.read_csv(output_dir / "selector_selection_rate.csv")
    selected_by_objective = (
        selection.groupby("objective", as_index=True)["selected_rows"].sum().to_dict()
    )
    expected_rows = int(config["guards"]["technical"]["expected_rows"])
    if any(
        int(selected_by_objective.get(name, -1)) != expected_rows
        for name in ("pred_abs_error", "p_within10")
    ):
        raise ValueError("selector selection inventory does not cover every OOF row")
    geop_selection = {
        objective: int(
            selection.loc[
                selection["candidate_id"].eq(ADDED_CANDIDATE)
                & selection["objective"].eq(objective),
                "selected_rows",
            ].sum()
        )
        for objective in ("pred_abs_error", "p_within10")
    }
    geop_selection_share = {
        name: count / expected_rows for name, count in geop_selection.items()
    }

    feature_schema = json.loads((output_dir / "feature_schema.json").read_text())
    required_id_feature = f"id__candidate__{ADDED_CANDIDATE}"
    id_feature_present = required_id_feature in feature_schema["features"]
    coverage = pd.read_csv(output_dir / "confidence_coverage_by_candidate_fold.csv")
    required_coverage = coverage[
        coverage["candidate_id"].eq(ADDED_CANDIDATE)
        & coverage["field"].isin(["confidence_valid", *GEOP_CONFIDENCE_FIELDS])
    ]
    coverage_pairs = set(
        zip(required_coverage["outer_fold"], required_coverage["field"], strict=False)
    )
    expected_pairs = {
        (fold, field)
        for fold in range(5)
        for field in ("confidence_valid", *GEOP_CONFIDENCE_FIELDS)
    }
    confidence_coverage_passed = coverage_pairs == expected_pairs and bool(
        np.allclose(required_coverage["coverage"], 1.0)
    )

    new_hard = float(new_metrics["hard_primary_oof_rmse"])
    parent_hard = float(parent_metrics["hard_primary_oof_rmse"])
    improved_folds = int(
        (
            new_fold["hard_primary_rmse"].to_numpy()
            < parent_fold["hard_primary_rmse"].to_numpy()
        ).sum()
    )
    fixed_delta = float(
        new_metrics["hard_readout_guard"]["fixed_fallback_oof_rmse"]
        - parent_metrics["hard_readout_guard"]["fixed_fallback_oof_rmse"]
    )
    checks = {
        "hard_primary_rmse_improved_vs_parent12": new_hard < parent_hard,
        "hard_primary_improved_at_least_3_of_5_folds": improved_folds >= 3,
        "geop_selected_by_pred_abs_error": geop_selection["pred_abs_error"] > 0,
        "new13_score_guard_passed": bool(new_metrics["score_guard"]["passed"]),
        "fixed_fallback_parity": abs(fixed_delta) <= 1.0e-9,
        "geop_candidate_id_feature_present": id_feature_present,
        "geop_native_confidence_coverage_complete": confidence_coverage_passed,
    }
    comparison = {
        "status": "selector_parent_comparison_completed",
        "parent_candidate_count": 12,
        "new_candidate_count": 13,
        "parent_hard_primary_oof_rmse": parent_hard,
        "new_hard_primary_oof_rmse": new_hard,
        "delta_hard_primary_rmse_new13_minus_parent12": new_hard - parent_hard,
        "hard_primary_improved_folds": improved_folds,
        "fixed_fallback_oof_rmse_delta": fixed_delta,
        "geop_selection_rows": geop_selection,
        "geop_selection_share": geop_selection_share,
        "geop_candidate_id_feature": required_id_feature,
        "geop_candidate_id_feature_present": id_feature_present,
        "geop_confidence_fields": list(GEOP_CONFIDENCE_FIELDS),
        "geop_confidence_coverage_passed": confidence_coverage_passed,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "interpretation": (
            "hard path and shared-12 metrics are directly comparable; all-candidate pooled "
            "score metrics are auxiliary because candidate-long row count changed from 12 to 13"
        ),
        "artifacts": {
            "fold_comparison": fold_path.name,
            "shared12_candidate_comparison": shared_path.name,
        },
    }
    write_json(output_dir / "selector_parent_comparison.json", comparison)
    return comparison


def run_geop_hmm_selector_stage_b(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    geop_candidate_path: Path,
    raw_train_dir: Path,
    raw_test_dir: Path,
    output_dir: Path,
    parent_schema_path: Path,
    parent_metrics_path: Path,
    parent_candidate_metrics_path: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_evidence = validate_full_contract(contract)
    geop_source, geop_evidence = load_geop_candidate_source(
        geop_candidate_path, config, cache_root=cache_root
    )
    cache = GeopHmmAugmentedCache(cache_root, contract, geop_source)

    def cache_factory(_root: Path, _contract: Mapping[str, Any]) -> GeopHmmAugmentedCache:
        return cache

    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        config["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability_path = output_dir / "raw_context_availability_audit.csv"
    availability.to_csv(availability_path, index=False)
    if not bool(availability["availability_pass"].all()):
        raise ValueError("raw train/test context availability audit failed")
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
    stage_b = run_stage_b(
        config=config,
        contract=contract,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        cache_factory=cache_factory,
    )
    if int(stage_b["model_count"]) != 10:
        raise AssertionError("exp286 Stage B did not train exactly 10 selector models")
    parent_metrics, parent_candidates, parent_evidence = load_parent_baseline(
        parent_metrics_path, parent_candidate_metrics_path, config
    )
    comparison = compare_with_parent_stage_b(
        output_dir=output_dir,
        new_metrics=stage_b,
        parent_metrics=parent_metrics,
        parent_candidate_metrics=parent_candidates,
        config=config,
    )
    summary = {
        "status": "geop_hmm_full_all_well_selector_stage_b_completed",
        "planned_cpu_boosters": 10,
        "actual_model_count": int(stage_b["model_count"]),
        "parent_control_retraining": False,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "inference": False,
        "submission": False,
        "candidate_contract": contract_evidence,
        "geop_source": geop_evidence,
        "raw_test_only_schema_guard": schema_guard,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "parent_stage_b_v5": parent_evidence,
        "selector_addition_comparison": comparison,
    }
    summary_path = output_dir / "exp286_selector_stage_b_summary.json"
    write_json(summary_path, summary)
    evidence_paths = [
        summary_path,
        output_dir / "feature_schema.json",
        output_dir / "confidence_coverage_by_candidate_fold.csv",
        output_dir / "selector_metrics.json",
        output_dir / "selector_model_manifest.json",
        output_dir / "selector_parent_comparison.json",
        output_dir / "selector_parent_fold_comparison.csv",
        output_dir / "selector_shared12_candidate_metric_comparison.csv",
    ]
    reproducibility = {
        "schema_version": "1.0.0",
        "status": "geop_hmm_selector_stage_b_completed",
        "candidate_contract_sha256": contract_evidence["candidate_contract_sha256"],
        "geop_source": geop_evidence,
        "parent_stage_b_v5": parent_evidence,
        "feature_schema_logical_sha256": feature_schema["feature_schema_sha256"],
        "candidate_score_oof_sha256": stage_b["candidate_score_oof_sha256"],
        "compact_meta_oof_sha256": stage_b["compact_meta_oof_sha256"],
        "model_manifest_sha256": stage_b["model_manifest_sha256"],
        "output_file_sha256": {path.name: sha256_file(path) for path in evidence_paths},
        "submission_sha256": None,
    }
    reproducibility_path = output_dir / "exp286_selector_reproducibility_manifest.json"
    write_json(reproducibility_path, reproducibility)
    summary["reproducibility_manifest_sha256"] = sha256_file(reproducibility_path)
    return summary


def _normalize_stage_c_fold_metrics(
    frame: pd.DataFrame, *, label: str
) -> pd.DataFrame:
    """Normalize saved Stage C fold metrics to the exp286 outer-fold contract."""

    fold_columns = [column for column in ("outer_fold", "fold") if column in frame]
    if not fold_columns:
        raise ValueError(f"{label} is missing outer_fold/fold")
    normalized = frame.copy()
    if len(fold_columns) == 2 and not np.array_equal(
        normalized["outer_fold"].to_numpy(), normalized["fold"].to_numpy()
    ):
        raise ValueError(f"{label} outer_fold/fold columns disagree")
    normalized["outer_fold"] = normalized[fold_columns[0]].astype(int)
    if normalized["outer_fold"].duplicated().any():
        raise ValueError(f"{label} contains duplicate outer folds")
    if sorted(normalized["outer_fold"].tolist()) != list(range(5)):
        raise ValueError(f"{label} outer-fold inventory is not [0, 1, 2, 3, 4]")
    if "hard_primary_rmse" not in normalized:
        raise ValueError(f"{label} is missing hard_primary_rmse")
    return normalized.sort_values("outer_fold", kind="stable").reset_index(drop=True)


def run_geop_hmm_selector_stage_c(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    geop_candidate_path: Path,
    raw_train_dir: Path,
    raw_test_dir: Path,
    output_dir: Path,
    parent_schema_path: Path,
    parent_stage_c_metrics_path: Path,
    parent_stage_c_fold_metrics_path: Path,
) -> dict[str, Any]:
    """Build leakage-safe full13 nested compact features with exactly 40 CPU models."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_evidence = validate_full_contract(contract)
    geop_source, geop_evidence = load_geop_candidate_source(
        geop_candidate_path, config, cache_root=cache_root
    )
    cache = GeopHmmAugmentedCache(cache_root, contract, geop_source)

    def cache_factory(_root: Path, _contract: Mapping[str, Any]) -> GeopHmmAugmentedCache:
        return cache

    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        config["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability_path = output_dir / "raw_context_availability_audit.csv"
    availability.to_csv(availability_path, index=False)
    if not bool(availability["availability_pass"].all()):
        raise ValueError("raw train/test context availability audit failed")

    # Preflight all saved-parent comparison inputs before starting the costly
    # 40-model nested training run.  exp264 v6 stores its fold id as `fold`,
    # while newer outputs may use `outer_fold`.
    parent_expected = dict(config["data"]["parent_stage_c_v6"])
    parent_actual_sha = {
        "nested_selector_metrics_sha256": sha256_file(parent_stage_c_metrics_path),
        "nested_selector_fold_metrics_sha256": sha256_file(
            parent_stage_c_fold_metrics_path
        ),
    }
    for key, actual in parent_actual_sha.items():
        expected = str(parent_expected[f"expected_{key}"])
        if actual != expected:
            raise ValueError(f"parent Stage C v6 SHA mismatch for {key}: {actual}")
    parent_metrics = json.loads(parent_stage_c_metrics_path.read_text())
    parent_fold = _normalize_stage_c_fold_metrics(
        pd.read_csv(parent_stage_c_fold_metrics_path),
        label="parent Stage C v6 fold metrics",
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
    stage_c = run_stage_c(
        config=config,
        contract=contract,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=output_dir,
        cache_factory=cache_factory,
        hard_readout_enabled=True,
    )
    if int(stage_c["model_count"]) != 40:
        raise AssertionError("exp286 Stage C did not train exactly 40 selector models")
    if not bool(stage_c["leakage_audit"]["passed"]):
        raise AssertionError("exp286 Stage C nested leakage audit failed")
    if not bool(stage_c["score_guard"]["passed"]):
        raise AssertionError("exp286 Stage C selector score guard failed")

    new_fold = _normalize_stage_c_fold_metrics(
        pd.read_csv(output_dir / "nested_selector_metrics.csv"),
        label="exp286 Stage C fold metrics",
    )
    if parent_fold["outer_fold"].tolist() != new_fold["outer_fold"].tolist():
        raise ValueError("Stage C parent/new outer-fold inventory mismatch")
    fold_comparison = pd.DataFrame(
        {
            "outer_fold": new_fold["outer_fold"].astype(int),
            "parent12_hard_primary_rmse": parent_fold["hard_primary_rmse"].astype(float),
            "new13_hard_primary_rmse": new_fold["hard_primary_rmse"].astype(float),
        }
    )
    fold_comparison["delta_new13_minus_parent12"] = (
        fold_comparison["new13_hard_primary_rmse"]
        - fold_comparison["parent12_hard_primary_rmse"]
    )
    fold_comparison_path = output_dir / "stage_c_parent_fold_comparison.csv"
    fold_comparison.to_csv(fold_comparison_path, index=False)
    parent_hard = float(parent_metrics["hard_primary_oof_rmse"])
    new_hard = float(stage_c["hard_primary_oof_rmse"])
    comparison = {
        "status": "stage_c_parent_comparison_completed",
        "parent_candidate_count": 12,
        "new_candidate_count": 13,
        "parent12_hard_primary_oof_rmse": parent_hard,
        "new13_hard_primary_oof_rmse": new_hard,
        "delta_new13_minus_parent12": new_hard - parent_hard,
        "improved_folds": int(
            (fold_comparison["delta_new13_minus_parent12"] < 0).sum()
        ),
        "new13_score_guard_passed": bool(stage_c["score_guard"]["passed"]),
        "new13_leakage_audit_passed": bool(stage_c["leakage_audit"]["passed"]),
        "parent_stage_c_v6": parent_actual_sha,
        "fold_comparison": fold_comparison_path.name,
    }
    comparison_path = output_dir / "stage_c_parent_comparison.json"
    write_json(comparison_path, comparison)

    summary = {
        "status": "geop_hmm_full13_nested_compact_stage_c_completed",
        "planned_cpu_boosters": 40,
        "actual_model_count": int(stage_c["model_count"]),
        "parent_control_retraining": False,
        "hmm_well_runs": 0,
        "pf_well_runs": 0,
        "inference": False,
        "submission": False,
        "candidate_contract": contract_evidence,
        "geop_source": geop_evidence,
        "raw_test_only_schema_guard": schema_guard,
        "stage_a": stage_a,
        "stage_c": stage_c,
        "stage_c_parent_comparison": comparison,
    }
    summary_path = output_dir / "exp286_selector_stage_c_summary.json"
    write_json(summary_path, summary)
    evidence_paths = [
        summary_path,
        output_dir / "feature_schema.json",
        output_dir / "compact_meta_schema.json",
        output_dir / "nested_selector_metrics.json",
        output_dir / "nested_selector_model_manifest.json",
        output_dir / "nested_compact_manifest.json",
        comparison_path,
        fold_comparison_path,
    ]
    reproducibility = {
        "schema_version": "1.0.0",
        "status": "geop_hmm_selector_stage_c_completed",
        "candidate_contract_sha256": contract_evidence["candidate_contract_sha256"],
        "geop_source": geop_evidence,
        "parent_stage_c_v6": parent_actual_sha,
        "feature_schema_logical_sha256": feature_schema["feature_schema_sha256"],
        "nested_selector_model_manifest_sha256": stage_c[
            "nested_selector_model_manifest_sha256"
        ],
        "nested_compact_manifest_sha256": stage_c[
            "nested_compact_manifest_sha256"
        ],
        "nested_outer_valid_candidate_score_sha256": stage_c[
            "nested_outer_valid_candidate_score_sha256"
        ],
        "output_file_sha256": {path.name: sha256_file(path) for path in evidence_paths},
        "submission_sha256": None,
    }
    reproducibility_path = output_dir / "exp286_stage_c_reproducibility_manifest.json"
    write_json(reproducibility_path, reproducibility)
    summary["reproducibility_manifest_sha256"] = sha256_file(reproducibility_path)
    return summary


def stage_d_full13_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the approved add-only-only Stage D scope at exactly 15 GPU models."""

    experiment_code = str(config["experiment"]["name"]).split("_", maxsplit=1)[0]
    stage = dict(config["model"]["downstream_tvt_stage"])
    variants = [str(item) for item in stage["variants"]]
    config_indices = [int(item) for item in stage["lightgbm_config_indices"]]
    folds = int(stage["folds"])
    planned = int(stage["planned_gpu_boosters"])
    calculated = len(variants) * len(config_indices) * folds
    expected_scope = (
        "full13_compact350_addonly15_three_configs_five_folds_15_gpu_boosters"
    )
    if variants != ["selector_compact_addonly"]:
        raise ValueError(
            f"unexpected {experiment_code} Stage D variants: {variants}"
        )
    if config_indices != [0, 1, 2]:
        raise ValueError(
            f"unexpected {experiment_code} Stage D config indices: {config_indices}"
        )
    if folds != 5 or calculated != 15 or planned != 15:
        raise ValueError(
            f"{experiment_code} Stage D cost must be "
            "1 add-only variant x 3 configs x 5 folds = 15"
        )
    if bool(stage.get("control_retraining", True)):
        raise ValueError(
            f"{experiment_code} Stage D must not retrain the saved parent/control"
        )
    if not bool(stage.get("enabled", False)):
        raise ValueError(
            f"{experiment_code} Stage D downstream_tvt_stage.enabled must be true"
        )
    if not stage.get("approval_received_at"):
        raise ValueError(f"{experiment_code} Stage D user approval is missing")
    if str(stage.get("approval_scope", "")) != expected_scope:
        raise ValueError(f"{experiment_code} Stage D approval scope mismatch")
    expected_surface = {
        "feature_surface": "exp218_clean_273_drop_107_plus_full13_compact77",
        "expected_source_base_feature_count": 380,
        "expected_base_feature_count": 273,
        "expected_compact_feature_count": 77,
        "selector_compact_addonly_feature_count": 350,
    }
    for key, expected in expected_surface.items():
        if stage.get(key) != expected:
            raise ValueError(
                f"{experiment_code} Stage D feature surface mismatch for {key}: "
                f"{stage.get(key)} != {expected}"
            )
    return {
        "variants": variants,
        "lightgbm_config_indices": config_indices,
        "folds": folds,
        "boosters_per_variant": 15,
        "total_gpu_boosters": 15,
        "control_retraining": False,
        "approval_received_at": stage["approval_received_at"],
        "approval_scope": stage["approval_scope"],
    }


def load_parent_stage_d_reference(
    *, config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Load SHA-locked exp264 Stage D summaries without requiring its large OOF file."""

    parent_cfg = dict(config["data"]["parent_stage_d_v3"])
    expected_names = {
        "metrics": "expected_metrics_sha256",
        "fold_metrics": "expected_fold_metrics_sha256",
        "bucket_metrics": "expected_bucket_metrics_sha256",
        "hidden_like_metrics": "expected_hidden_like_metrics_sha256",
        "by_well": "expected_by_well_sha256",
    }
    sha: dict[str, str] = {}
    for name, expected_key in expected_names.items():
        path = Path(paths[name])
        if not path.exists():
            raise FileNotFoundError(f"parent Stage D reference missing: {path}")
        sha[name] = sha256_file(path)
        expected = str(parent_cfg[expected_key])
        if sha[name] != expected:
            raise ValueError(f"parent Stage D {name} SHA mismatch: {sha[name]}")
    metrics = json.loads(Path(paths["metrics"]).read_text())
    fold = pd.read_csv(paths["fold_metrics"])
    bucket = pd.read_csv(paths["bucket_metrics"])
    hidden = pd.read_csv(paths["hidden_like_metrics"])
    by_well = pd.read_csv(paths["by_well"], dtype={"well": str})
    parent_rmse = float(metrics["selector_compact_addonly_lgb_mean_rmse"])
    configured_rmse = float(parent_cfg["selector_compact_addonly_lgb_mean_rmse"])
    if not np.isclose(parent_rmse, configured_rmse, atol=1.0e-12, rtol=0.0):
        raise ValueError("parent Stage D RMSE differs from frozen config")
    return {
        "sha256": sha,
        "metrics": metrics,
        "fold_metrics": fold,
        "bucket_metrics": bucket,
        "hidden_like_metrics": hidden,
        "by_well": by_well,
    }


def run_fixed13_compact_stage_d_addonly(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    base_feature_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    parent_reference_paths: Mapping[str, Path],
    output_dir: Path,
    experiment_code: str,
) -> dict[str, Any]:
    """Train one 77-feature fixed13 compact add-only surface versus saved parent12."""

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    configured_code = str(config["experiment"]["name"]).split("_", maxsplit=1)[0]
    if str(experiment_code) != configured_code:
        raise ValueError(
            f"Stage D experiment code mismatch: {experiment_code} != {configured_code}"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cost = stage_d_full13_cost_contract(config)
    stage_c_evidence = verify_stage_c_artifact_root(
        stage_c_root,
        config,
        expected_compact_feature_count=77,
    )
    parent = load_parent_stage_d_reference(
        config=config, paths=parent_reference_paths
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
    compact_features = [str(item) for item in stage_c_evidence["compact_features"]]
    stage_cfg = dict(config["model"]["downstream_tvt_stage"])
    final_features = [*base_features, *compact_features]
    if len(base_features) != 273 or len(compact_features) != 77 or len(final_features) != 350:
        raise ValueError(
            f"{experiment_code} Stage D expected 273 base + 77 compact = 350 features"
        )
    mode_name = str(stage_cfg["mode"])
    mode_config = dict(
        _nested_get(exp218_config, f"model.training.modes.{mode_name}", {}) or {}
    )
    if not bool(mode_config.get("use_gpu", False)):
        raise ValueError(f"{experiment_code} Stage D approved mode must use GPU")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode_config
    )
    config_indices = [int(item) for item in cost["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]

    base_index = pd.Index(base_frame["id"].astype(str), name="id")
    if not base_index.is_unique:
        raise ValueError("Stage D base id index is not unique")
    n_rows = len(base_frame)
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof = [np.full(n_rows, np.nan, np.float32) for _ in params_family]
    oof_fold = np.full(n_rows, -1, np.int8)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    model_dir = output_dir / "stage_d_models" / "selector_compact_addonly"
    model_dir.mkdir(parents=True, exist_ok=True)
    chunk_columns = int(stage_cfg["matrix_copy_chunk_columns"])

    for outer_fold in range(5):
        compact_train, compact_valid = load_stage_d_compact_fold(
            stage_c_root=stage_c_root,
            stage_c_evidence=stage_c_evidence,
            downstream_outer_fold=outer_fold,
        )
        train_indices = base_index.get_indexer(compact_train["id"].astype(str))
        valid_indices = base_index.get_indexer(compact_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("Stage C compact ids are absent from the exp218 base surface")
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != n_rows:
            raise ValueError("Stage D train/valid compact rows do not cover base rows exactly once")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Stage D train/valid base indices overlap")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("Stage D OOF valid rows were assigned more than once")
        oof_fold[valid_indices] = np.int8(outer_fold)
        for frame, indices, role in (
            (compact_train, train_indices, "train"),
            (compact_valid, valid_indices, "valid"),
        ):
            if not np.array_equal(
                base_frame["well"].iloc[indices].astype(str).to_numpy(),
                frame["well"].astype(str).to_numpy(),
            ):
                raise ValueError(f"Stage D {role} well alignment mismatch")
            if float(
                np.max(
                    np.abs(
                        anchor[indices]
                        - frame["last_known_tvt"].to_numpy(np.float32)
                    )
                )
            ) > 1.0e-4:
                raise ValueError(f"Stage D {role} anchor alignment mismatch")
            if float(
                np.max(
                    np.abs(
                        base_frame["md_since"].iloc[indices].to_numpy(np.float32)
                        - frame["md_since"].to_numpy(np.float32)
                    )
                )
            ) > 1.0e-4:
                raise ValueError(f"Stage D {role} md_since alignment mismatch")

        x_train_values = np.empty((len(train_indices), 350), dtype=np.float32)
        x_valid_values = np.empty((len(valid_indices), 350), dtype=np.float32)
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            base_chunk = base_frame.loc[:, columns]
            x_train_values[:, start:stop] = base_chunk.iloc[train_indices].to_numpy(
                np.float32, copy=True
            )
            x_valid_values[:, start:stop] = base_chunk.iloc[valid_indices].to_numpy(
                np.float32, copy=True
            )
            del base_chunk
        x_train_values[:, len(base_features) :] = compact_train[compact_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, len(base_features) :] = compact_valid[compact_features].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(x_train_values).all() or not np.isfinite(x_valid_values).all():
            raise ValueError(
                f"{experiment_code} Stage D feature matrix contains nonfinite values"
            )
        x_train = pd.DataFrame(x_train_values, columns=final_features, copy=False)
        x_valid = pd.DataFrame(x_valid_values, columns=final_features, copy=False)
        fold_predictions: list[np.ndarray] = []
        for family_position, (config_index, params) in enumerate(
            zip(config_indices, params_family, strict=True)
        ):
            model = LGBMRegressor(**params)
            model.fit(
                x_train,
                target[train_indices],
                eval_set=[(x_valid, target[valid_indices])],
                eval_metric="rmse",
                callbacks=[
                    early_stopping(
                        int(stage_cfg["early_stopping_rounds"]), verbose=False
                    ),
                    log_evaluation(int(stage_cfg["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            prediction = model.predict(
                x_valid, num_iteration=best_iteration
            ).astype(np.float32)
            oof[family_position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            rmse_value = _rmse_arrays(
                truth[valid_indices], anchor[valid_indices] + prediction
            )
            fold_rows.append(
                {
                    "variant": "selector_compact_addonly",
                    "model": f"lgb{config_index}",
                    "outer_fold": outer_fold,
                    "rows": len(valid_indices),
                    "train_rows": len(train_indices),
                    "features": 350,
                    "best_iteration": best_iteration,
                    "rmse_tvt": rmse_value,
                }
            )
            model_rows.append(
                {
                    "variant": "selector_compact_addonly",
                    "model": f"lgb{config_index}",
                    "config_index": config_index,
                    "outer_fold": outer_fold,
                    "feature_count": 350,
                    "best_iteration": best_iteration,
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": sha256_file(model_path),
                    "params": params,
                }
            )
            for importance_type in ("gain", "split"):
                values = model.booster_.feature_importance(importance_type=importance_type)
                importance_rows.extend(
                    {
                        "variant": "selector_compact_addonly",
                        "model": f"lgb{config_index}",
                        "outer_fold": outer_fold,
                        "importance_type": importance_type,
                        "feature": feature,
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, values, strict=True)
                )
            print(
                json.dumps(
                    {
                        "stage": "D",
                        "variant": "selector_compact_addonly_full13",
                        "model": f"lgb{config_index}",
                        "outer_fold": outer_fold,
                        "rmse_tvt": rmse_value,
                        "best_iteration": best_iteration,
                        "completed_boosters": len(model_rows),
                        "planned_boosters": 15,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, prediction
            gc.collect()
        fold_mean = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_rows.append(
            {
                "variant": "selector_compact_addonly",
                "model": "lgb_mean",
                "outer_fold": outer_fold,
                "rows": len(valid_indices),
                "train_rows": len(train_indices),
                "features": 350,
                "best_iteration": None,
                "rmse_tvt": _rmse_arrays(
                    truth[valid_indices], anchor[valid_indices] + fold_mean
                ),
            }
        )
        del (
            compact_train,
            compact_valid,
            train_indices,
            valid_indices,
            x_train,
            x_valid,
            x_train_values,
            x_valid_values,
            fold_predictions,
            fold_mean,
        )
        gc.collect()

    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError(
            f"{experiment_code} Stage D did not complete exactly 15 GPU models"
        )
    if not all(np.isfinite(prediction).all() for prediction in oof):
        raise AssertionError(f"{experiment_code} Stage D OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof), axis=0).astype(np.float32)
    mean_tvt = (anchor + mean_residual).astype(np.float32)
    new_rmse = _rmse_arrays(truth, mean_tvt)

    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    for config_index, prediction in zip(config_indices, oof, strict=True):
        prediction_frame[f"selector_compact_addonly__lgb{config_index}__pred_tvt"] = (
            anchor + prediction
        ).astype(np.float32)
    prediction_frame["selector_compact_addonly__lgb_mean__pred_tvt"] = mean_tvt

    fold_metrics = pd.DataFrame(fold_rows)
    new_fold = fold_metrics[fold_metrics["model"].eq("lgb_mean")].sort_values(
        "outer_fold", kind="stable"
    )
    parent_fold = parent["fold_metrics"]
    parent_fold = parent_fold[
        parent_fold["variant"].eq("selector_compact_addonly")
        & parent_fold["model"].eq("lgb_mean")
    ].sort_values("outer_fold", kind="stable")
    fold_comparison = pd.DataFrame(
        {
            "outer_fold": new_fold["outer_fold"].astype(int).to_numpy(),
            "parent12_rmse": parent_fold["rmse_tvt"].astype(float).to_numpy(),
            "new13_rmse": new_fold["rmse_tvt"].astype(float).to_numpy(),
        }
    )
    fold_comparison["delta_new13_minus_parent12"] = (
        fold_comparison["new13_rmse"] - fold_comparison["parent12_rmse"]
    )

    by_well_rows: list[dict[str, Any]] = []
    for well, group in prediction_frame.groupby("well", sort=True):
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "new13_rmse": _rmse_arrays(
                    group["actual_tvt"],
                    group["selector_compact_addonly__lgb_mean__pred_tvt"],
                ),
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    parent_by_well = parent["by_well"][
        ["well", "selector_compact_addonly_rmse"]
    ].rename(columns={"selector_compact_addonly_rmse": "parent12_rmse"})
    by_well = by_well.merge(parent_by_well, on="well", how="inner", validate="one_to_one")
    if len(by_well) != int(config["guards"]["technical"]["expected_wells"]):
        raise ValueError("Stage D parent/new by-well inventory mismatch")
    by_well["delta_new13_minus_parent12"] = (
        by_well["new13_rmse"] - by_well["parent12_rmse"]
    )

    md_since = base_frame["md_since"].to_numpy(np.float32)
    masks = {
        "all": np.ones(n_rows, dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    parent_bucket = parent["bucket_metrics"].set_index("bucket")
    bucket_rows: list[dict[str, Any]] = []
    for bucket, mask in masks.items():
        parent_rmse = float(
            parent_bucket.loc[bucket, "selector_compact_addonly_rmse"]
        )
        current_rmse = _rmse_arrays(truth[mask], mean_tvt[mask])
        bucket_rows.append(
            {
                "bucket": bucket,
                "rows": int(mask.sum()),
                "parent12_rmse": parent_rmse,
                "new13_rmse": current_rmse,
                "delta_new13_minus_parent12": current_rmse - parent_rmse,
            }
        )
    bucket_metrics = pd.DataFrame(bucket_rows)

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    assignment_by_well = assignment.set_index("well_id")
    parent_hidden = parent["hidden_like_metrics"].set_index("assignment")
    hidden_rows: list[dict[str, Any]] = []
    for column in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        role = base_frame["well"].astype(str).map(assignment_by_well[column])
        mask = role.eq("valid").to_numpy()
        parent_rmse = float(
            parent_hidden.loc[column, "selector_compact_addonly_rmse"]
        )
        current_rmse = _rmse_arrays(truth[mask], mean_tvt[mask])
        hidden_rows.append(
            {
                "assignment": column,
                "role": "valid",
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "parent12_rmse": parent_rmse,
                "new13_rmse": current_rmse,
                "delta_new13_minus_parent12": current_rmse - parent_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)

    parent_rmse = float(
        parent["metrics"]["selector_compact_addonly_lgb_mean_rmse"]
    )
    delta = new_rmse - parent_rmse
    guard_cfg = dict(config["guards"]["stage_d_vs_parent12"])
    bucket_lookup = bucket_metrics.set_index("bucket")
    by_well_p95 = float(
        by_well["delta_new13_minus_parent12"].quantile(0.95)
    )
    checks = {
        "pooled_rmse_improved": delta < 0.0,
        "improved_folds": int(
            (fold_comparison["delta_new13_minus_parent12"] < 0).sum()
        )
        >= int(guard_cfg["min_improved_folds"]),
        "near_non_regression": float(
            bucket_lookup.loc["near_0_250", "delta_new13_minus_parent12"]
        )
        <= float(guard_cfg["max_near_delta_rmse"]),
        "distance_1000_plus_non_regression": float(
            bucket_lookup.loc["1000_plus", "delta_new13_minus_parent12"]
        )
        <= float(guard_cfg["max_1000_plus_delta_rmse"]),
        "hidden_like_non_regression": float(
            hidden_metrics["delta_new13_minus_parent12"].max()
        )
        <= float(guard_cfg["max_hidden_like_delta_rmse"]),
        "worst_well_non_regression": float(
            by_well["delta_new13_minus_parent12"].max()
        )
        <= float(guard_cfg["max_worst_well_regression"]),
    }
    if "max_by_well_p95_delta_rmse" in guard_cfg:
        checks["by_well_p95_non_regression"] = by_well_p95 <= float(
            guard_cfg["max_by_well_p95_delta_rmse"]
        )
    comparison = {
        "parent12_selector_compact_addonly_rmse": parent_rmse,
        "new13_selector_compact_addonly_rmse": new_rmse,
        "delta_new13_minus_parent12": delta,
        "improved_folds": int(
            (fold_comparison["delta_new13_minus_parent12"] < 0).sum()
        ),
        "improved_wells": int((by_well["delta_new13_minus_parent12"] < 0).sum()),
        "worsened_wells": int((by_well["delta_new13_minus_parent12"] > 0).sum()),
        "median_well_delta": float(by_well["delta_new13_minus_parent12"].median()),
        "by_well_p95_delta": by_well_p95,
        "worst_well_delta": float(by_well["delta_new13_minus_parent12"].max()),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }

    fold_path = output_dir / "stage_d_fold_metrics.csv"
    fold_compare_path = output_dir / "stage_d_parent_fold_comparison.csv"
    oof_path = output_dir / "stage_d_oof_predictions.parquet"
    importance_path = output_dir / "stage_d_feature_importance.csv"
    by_well_path = output_dir / "stage_d_by_well_comparison.csv"
    bucket_path = output_dir / "stage_d_bucket_comparison.csv"
    hidden_path = output_dir / "stage_d_hidden_like_comparison.csv"
    manifest_path = output_dir / "stage_d_model_manifest.json"
    metrics_path = output_dir / "stage_d_metrics.json"
    fold_metrics.to_csv(fold_path, index=False)
    fold_comparison.to_csv(fold_compare_path, index=False)
    prediction_frame.to_parquet(oof_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    bucket_metrics.to_csv(bucket_path, index=False)
    hidden_metrics.to_csv(hidden_path, index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": f"{experiment_code}_stage_d_full13_addonly_15_gpu_models_completed",
        "cost_contract": cost,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_surface": final_features,
        "feature_schema_sha256": sha256_json(final_features),
        "stage_c_input": stage_c_evidence,
        "exp218_input": base_evidence,
        "parent_stage_d_reference_sha256": parent["sha256"],
    }
    write_json(manifest_path, model_manifest)
    metrics = {
        "status": f"{experiment_code}_stage_d_full13_addonly_15_gpu_models_completed",
        "cost_contract": cost,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": {"base": 273, "compact": 77, "final": 350},
        "comparison_vs_parent12": comparison,
        "saved_control_lgb_mean_rmse": float(
            parent["metrics"]["matched_control_lgb_mean_rmse"]
        ),
        "model_count": len(model_rows),
        "stage_c_input_sha256": {
            "nested_selector_model_manifest": stage_c_evidence["sha256"][
                "nested_selector_model_manifest"
            ],
            "nested_compact_manifest": stage_c_evidence["sha256"][
                "nested_compact_manifest"
            ],
        },
    }
    write_json(metrics_path, metrics)
    artifact_paths = [
        metrics_path,
        fold_path,
        fold_compare_path,
        oof_path,
        manifest_path,
        importance_path,
        by_well_path,
        bucket_path,
        hidden_path,
    ]
    reproducibility = {
        "schema_version": "1.0.0",
        "status": f"{experiment_code}_stage_d_full13_addonly_completed",
        "cost_contract": cost,
        "stage_c_input": stage_c_evidence,
        "parent_stage_d_reference_sha256": parent["sha256"],
        "model_manifest_sha256": sha256_file(manifest_path),
        "oof_prediction_sha256": sha256_file(oof_path),
        "output_file_sha256": {
            path.name: sha256_file(path) for path in artifact_paths
        },
        "submission_sha256": None,
    }
    write_json(
        output_dir / f"{experiment_code}_stage_d_reproducibility_manifest.json",
        reproducibility,
    )
    return metrics


def run_geop_hmm_stage_d_addonly(
    *,
    config: Mapping[str, Any],
    stage_c_root: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    base_feature_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    parent_reference_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Run the exp286 geop-HMM fixed13 compact add-only Stage D."""

    return run_fixed13_compact_stage_d_addonly(
        config=config,
        stage_c_root=stage_c_root,
        exp218_source_path=exp218_source_path,
        exp218_config_path=exp218_config_path,
        base_feature_allowlist_path=base_feature_allowlist_path,
        hidden_like_assignment_path=hidden_like_assignment_path,
        raw_train_dir=raw_train_dir,
        parent_reference_paths=parent_reference_paths,
        output_dir=output_dir,
        experiment_code="exp286",
    )


__all__ = [
    "ADDED_CANDIDATE",
    "GEOP_CONFIDENCE_FIELDS",
    "GeopCandidateStore",
    "GeopHmmAugmentedCache",
    "augment_fold_bundle",
    "build_base_contract",
    "compare_with_parent_stage_b",
    "load_geop_candidate_source",
    "raw_test_only_schema_guard",
    "resolve_geop_candidate_source",
    "run_geop_hmm_selector_stage_b",
    "run_geop_hmm_selector_stage_c",
    "run_geop_hmm_stage_d_addonly",
    "run_fixed13_compact_stage_d_addonly",
    "schema_sha",
    "sha256_gzip_content",
    "stage_d_full13_cost_contract",
    "validate_full_contract",
]
