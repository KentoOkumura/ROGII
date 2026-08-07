from __future__ import annotations

import gc
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    IncrementalParquetWriter,
    ShapeState,
    build_candidate_long_features,
    build_nested_inner_fold_maps,
    build_raw_context,
    build_stage_d_exp218_surface,
    candidate_contract_sha,
    candidate_ids,
    deterministic_sample_indices,
    load_feature_schema,
    load_stage_d_compact_fold,
    resolve_exp263_cache_root,
    sha256_file,
    sha256_json,
    verify_exp263_root,
    write_json,
)

SIGNED_TARGET_COLUMN = "candidate_signed_residual"
SIGNED_PREDICTION_COLUMN = "pred_signed_residual"
PARENT_OBJECTIVES = ("pred_abs_error", "p_within10")


def signed_compact_feature_names(contract: Mapping[str, Any]) -> list[str]:
    names = [
        f"selector__pred_signed_residual__{candidate_id}"
        for candidate_id in candidate_ids(contract)
    ]
    for domain in ("primitive_pair_bank", "primitive_fixed_bank"):
        for objective in PARENT_OBJECTIVES:
            prefix = f"selector__{domain}__{objective}"
            names.extend(
                [
                    f"{prefix}__signed_residual_at_top1",
                    f"{prefix}__signed_corrected_top1_minus_anchor",
                ]
            )
    names.extend(
        [
            "selector__pred_signed_residual_mean",
            "selector__pred_signed_residual_std",
            "selector__pred_signed_residual_range",
        ]
    )
    if len(names) != 23 or len(set(names)) != 23:
        raise AssertionError("signed compact schema must contain exactly 23 unique features")
    return names


def signed_compact_schema(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "feature_count": 23,
        "features": signed_compact_feature_names(contract),
        "candidate_order": candidate_ids(contract),
        "top1_source": "saved_exp264_pred_abs_error_and_p_within10",
        "target_formula": "true_tvt-candidate_tvt",
    }
    payload["signed_compact_schema_sha256"] = sha256_json(payload)
    return payload


def add_signed_residual_labels(
    metadata: pd.DataFrame,
    truth: np.ndarray,
    n_candidates: int,
    *,
    formula_atol: float = 1.0e-3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"candidate_tvt", "candidate_available"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"signed residual metadata columns missing: {missing}")
    if len(metadata) != len(truth) * int(n_candidates):
        raise ValueError("signed residual label row count mismatch")

    candidate_tvt = pd.to_numeric(metadata["candidate_tvt"], errors="coerce").to_numpy(
        np.float32
    )
    available = metadata["candidate_available"].astype(bool).to_numpy()
    true_tvt = np.repeat(np.asarray(truth, dtype=np.float32), int(n_candidates))
    if not np.isfinite(true_tvt).all():
        raise ValueError("signed residual truth contains non-finite values")
    if not available.all() or not np.isfinite(candidate_tvt).all():
        raise ValueError(
            "signed residual label requires complete finite candidate coverage; "
            "unavailable candidates are never assigned an implicit zero label"
        )

    signed = (true_tvt - candidate_tvt).astype(np.float32)
    reconstructed = (candidate_tvt + signed).astype(np.float32)
    parity_error = np.abs(reconstructed.astype(np.float64) - true_tvt.astype(np.float64))
    max_abs_error = float(parity_error.max(initial=0.0))
    if max_abs_error > float(formula_atol):
        raise AssertionError(
            "signed residual label formula parity failed: "
            f"max_abs_error={max_abs_error} atol={formula_atol}"
        )

    labels = metadata.copy()
    labels["true_tvt"] = true_tvt
    labels[SIGNED_TARGET_COLUMN] = signed
    evidence = {
        "formula": "true_tvt-candidate_tvt",
        "rows": len(labels),
        "available_rows": int(available.sum()),
        "finite_labels": int(np.isfinite(signed).sum()),
        "formula_parity_max_abs_error": max_abs_error,
        "formula_parity_atol": float(formula_atol),
        "formula_parity_passed": max_abs_error <= float(formula_atol),
    }
    return labels, evidence


def _assert_base_parent_alignment(base: pd.DataFrame, parent: pd.DataFrame) -> None:
    if len(base) != len(parent):
        raise ValueError(f"saved exp264 compact row mismatch: {len(base)} != {len(parent)}")
    for column in KEY_COLUMNS:
        if column not in parent:
            raise ValueError(f"saved exp264 compact key missing: {column}")
        left = base[column].to_numpy()
        right = parent[column].to_numpy()
        if column == "md_since":
            equal = np.array_equal(left, right, equal_nan=True)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise ValueError(f"saved exp264 compact key mismatch: {column}")


def parent_compact_columns(contract: Mapping[str, Any]) -> list[str]:
    columns = list(KEY_COLUMNS) + ["last_known_tvt"]
    ids = candidate_ids(contract)
    for objective in PARENT_OBJECTIVES:
        columns.extend(f"selector__{objective}__{candidate_id}" for candidate_id in ids)
    for domain in ("primitive_pair_bank", "primitive_fixed_bank"):
        for objective in PARENT_OBJECTIVES:
            columns.append(f"selector__{domain}__{objective}__top1_value")
    return columns


def build_signed_compact_meta(
    base: pd.DataFrame,
    candidate_values: np.ndarray,
    pred_signed_residual: np.ndarray,
    saved_exp264_compact: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    top1_value_atol: float = 1.0e-5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ids = candidate_ids(contract)
    n_rows = len(base)
    expected_shape = (n_rows, len(ids))
    values = np.asarray(candidate_values, dtype=np.float32)
    signed = np.asarray(pred_signed_residual, dtype=np.float32)
    if values.shape != expected_shape or signed.shape != expected_shape:
        raise ValueError(
            "signed compact matrix shape mismatch: "
            f"values={values.shape} signed={signed.shape} expected={expected_shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(signed).all():
        raise ValueError("signed compact requires finite candidate values and predictions")
    _assert_base_parent_alignment(base, saved_exp264_compact)

    output = base[list(KEY_COLUMNS) + ["last_known_tvt"]].copy()
    for position, candidate_id in enumerate(ids):
        output[f"selector__pred_signed_residual__{candidate_id}"] = signed[:, position]

    id_to_position = {candidate_id: position for position, candidate_id in enumerate(ids)}
    rows = np.arange(n_rows, dtype=np.int64)
    anchor = pd.to_numeric(base["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    top1_parity: dict[str, float] = {}
    for domain in ("primitive_pair_bank", "primitive_fixed_bank"):
        domain_ids = [str(item) for item in contract["legal_domains"][domain]["candidates"]]
        domain_positions = np.asarray(
            [id_to_position[candidate_id] for candidate_id in domain_ids], dtype=np.int64
        )
        for objective in PARENT_OBJECTIVES:
            score_columns = [f"selector__{objective}__{candidate_id}" for candidate_id in ids]
            missing = [column for column in score_columns if column not in saved_exp264_compact]
            if missing:
                raise ValueError(f"saved exp264 candidate score columns missing: {missing}")
            scores = saved_exp264_compact[score_columns].to_numpy(np.float32, copy=False)
            if not np.isfinite(scores).all():
                raise ValueError("saved exp264 compact contains non-finite candidate scores")
            domain_scores = scores[:, domain_positions]
            maximize = objective == "p_within10"
            order = np.argsort(
                -domain_scores if maximize else domain_scores,
                axis=1,
                kind="stable",
            )
            selected = domain_positions[order[:, 0]]
            selected_value = values[rows, selected]
            parent_top1_column = f"selector__{domain}__{objective}__top1_value"
            if parent_top1_column not in saved_exp264_compact:
                raise ValueError(f"saved exp264 top1 column missing: {parent_top1_column}")
            parent_top1 = pd.to_numeric(
                saved_exp264_compact[parent_top1_column], errors="coerce"
            ).to_numpy(np.float32)
            parity = np.abs(selected_value.astype(np.float64) - parent_top1.astype(np.float64))
            max_abs_error = float(parity.max(initial=0.0))
            top1_parity[f"{domain}__{objective}"] = max_abs_error
            if max_abs_error > float(top1_value_atol):
                raise AssertionError(
                    "saved exp264 top1 identity parity failed for "
                    f"{domain}/{objective}: {max_abs_error} > {top1_value_atol}"
                )
            selected_signed = signed[rows, selected]
            prefix = f"selector__{domain}__{objective}"
            output[f"{prefix}__signed_residual_at_top1"] = selected_signed
            output[f"{prefix}__signed_corrected_top1_minus_anchor"] = (
                selected_value + selected_signed - anchor
            ).astype(np.float32)

    output["selector__pred_signed_residual_mean"] = np.mean(signed, axis=1)
    output["selector__pred_signed_residual_std"] = np.std(signed, axis=1)
    output["selector__pred_signed_residual_range"] = np.ptp(signed, axis=1)
    actual_features = [column for column in output if column.startswith("selector__")]
    expected_features = signed_compact_feature_names(contract)
    if actual_features != expected_features:
        raise ValueError(
            "signed compact schema/order mismatch: "
            f"actual={actual_features} expected={expected_features}"
        )
    matrix = output[expected_features].to_numpy(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("signed compact contains non-finite values")
    evidence = {
        "rows": n_rows,
        "feature_count": len(expected_features),
        "top1_value_parity": top1_parity,
        "top1_value_parity_max_abs_error": max(top1_parity.values(), default=0.0),
        "top1_value_parity_atol": float(top1_value_atol),
        "passed": max(top1_parity.values(), default=0.0) <= float(top1_value_atol),
    }
    return output, evidence


def evaluate_signed_residual_gate(
    metrics: pd.DataFrame,
    *,
    minimum_improved_outer_folds: int = 4,
) -> dict[str, Any]:
    required = {"fold", "long_rows", "signed_sse", "prior_signed_sse"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"signed selector metric columns missing: {missing}")
    if metrics.empty or metrics["fold"].duplicated().any():
        raise ValueError("signed selector metrics require one row per outer fold")
    rows = pd.to_numeric(metrics["long_rows"], errors="raise").to_numpy(np.int64)
    if np.any(rows <= 0):
        raise ValueError("signed selector fold metrics contain non-positive row counts")
    signed_sse = pd.to_numeric(metrics["signed_sse"], errors="raise").to_numpy(np.float64)
    prior_sse = pd.to_numeric(metrics["prior_signed_sse"], errors="raise").to_numpy(
        np.float64
    )
    if not np.isfinite(signed_sse).all() or not np.isfinite(prior_sse).all():
        raise ValueError("signed selector SSE metrics must be finite")
    pooled_rmse = float(np.sqrt(signed_sse.sum() / rows.sum()))
    pooled_prior_rmse = float(np.sqrt(prior_sse.sum() / rows.sum()))
    improved_folds = int((signed_sse / rows < prior_sse / rows).sum())
    passed = bool(
        pooled_rmse < pooled_prior_rmse
        and improved_folds >= int(minimum_improved_outer_folds)
    )
    return {
        "pooled_signed_residual_rmse": pooled_rmse,
        "pooled_candidate_outer_train_mean_prior_rmse": pooled_prior_rmse,
        "pooled_improvement": pooled_prior_rmse - pooled_rmse,
        "improved_outer_folds": improved_folds,
        "required_improved_outer_folds": int(minimum_improved_outer_folds),
        "pooled_improved": pooled_rmse < pooled_prior_rmse,
        "passed": passed,
    }


class ParquetBatchCursor:
    def __init__(self, path: Path, columns: Sequence[str], batch_size: int):
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.columns = [str(column) for column in columns]
        self._iterator = iter(
            pq.ParquetFile(self.path).iter_batches(
                batch_size=int(batch_size), columns=self.columns
            )
        )
        self._buffer = pd.DataFrame(columns=self.columns)
        self.rows_read = 0

    def take(self, rows: int) -> pd.DataFrame:
        needed = int(rows)
        if needed <= 0:
            raise ValueError("ParquetBatchCursor.take requires rows > 0")
        parts: list[pd.DataFrame] = []
        while needed:
            if self._buffer.empty:
                try:
                    self._buffer = next(self._iterator).to_pandas()
                except StopIteration as exc:
                    raise ValueError(
                        f"saved exp264 compact ended before requested rows: {self.path}"
                    ) from exc
            count = min(needed, len(self._buffer))
            parts.append(self._buffer.iloc[:count].copy())
            self._buffer = self._buffer.iloc[count:].reset_index(drop=True)
            needed -= count
            self.rows_read += count
        return pd.concat(parts, ignore_index=True)

    def finish(self) -> None:
        if not self._buffer.empty:
            raise ValueError(f"saved exp264 compact has unread rows: {self.path}")
        try:
            next(self._iterator)
        except StopIteration:
            return
        raise ValueError(f"saved exp264 compact has extra row groups: {self.path}")


def resolve_saved_exp264_stage_c_root(
    config: Mapping[str, Any], search_roots: Sequence[Path]
) -> Path:
    patterns = [
        str(item) for item in config.get("data", {}).get("exp264_stage_c_root_patterns", [])
    ]
    candidates: list[Path] = []
    for raw in patterns:
        path = Path(raw)
        if (path / "nested_compact_manifest.json").exists():
            candidates.append(path)
    for search_root in search_roots:
        if not Path(search_root).exists():
            continue
        for pattern in patterns:
            if Path(pattern).is_absolute():
                continue
            for path in Path(search_root).glob(pattern):
                candidate = path if path.is_dir() else path.parent
                if (candidate / "nested_compact_manifest.json").exists():
                    candidates.append(candidate)
        candidates.extend(
            path.parent
            for path in Path(search_root).rglob("nested_compact_manifest.json")
            if path.is_file()
        )
    unique = sorted(set(candidates))
    if not unique:
        raise FileNotFoundError("corrected exp264 Stage C artifact root not found")
    expected = str(
        config.get("data", {}).get("stage_c_nested_compact_manifest_sha256", "")
    )
    for candidate in unique:
        manifest = candidate / "nested_compact_manifest.json"
        if not expected or sha256_file(manifest) == expected:
            return candidate
    raise ValueError("no corrected exp264 Stage C root matches the frozen manifest SHA")


def verify_saved_exp264_stage_c_root(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_partition_sha: bool,
    require_score_guard: bool = True,
) -> dict[str, Any]:
    root = Path(root)
    data_cfg = dict(config.get("data", {}))
    files = {
        "nested_selector_metrics": root / "nested_selector_metrics.json",
        "nested_selector_model_manifest": root / "nested_selector_model_manifest.json",
        "nested_compact_manifest": root / "nested_compact_manifest.json",
        "compact_meta_schema": root / "compact_meta_schema.json",
    }
    expected = {
        "nested_selector_metrics": data_cfg.get("stage_c_nested_selector_metrics_sha256"),
        "nested_selector_model_manifest": data_cfg.get(
            "stage_c_nested_selector_model_manifest_sha256"
        ),
        "nested_compact_manifest": data_cfg.get("stage_c_nested_compact_manifest_sha256"),
        "compact_meta_schema": data_cfg.get("stage_c_compact_meta_schema_file_sha256"),
    }
    file_sha: dict[str, str] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        frozen = str(expected.get(name) or "")
        if frozen and observed != frozen:
            raise ValueError(f"saved exp264 Stage C {name} SHA mismatch: {observed}")
        file_sha[name] = observed

    compact_schema = json.loads(files["compact_meta_schema"].read_text())
    logical_sha = str(compact_schema.get("compact_meta_schema_sha256", ""))
    expected_logical = str(data_cfg.get("stage_c_compact_meta_schema_logical_sha256", ""))
    if logical_sha != expected_logical:
        raise ValueError("saved exp264 compact logical schema SHA mismatch")
    if len(compact_schema.get("features", [])) != 74:
        raise ValueError("saved exp264 compact schema must contain 74 features")

    selector_metrics = json.loads(files["nested_selector_metrics"].read_text())
    selector_models = json.loads(files["nested_selector_model_manifest"].read_text())
    if require_score_guard and not bool(
        selector_metrics.get("score_guard", {}).get("passed", False)
    ):
        raise ValueError("saved exp264 Stage C score guard did not pass")
    if not bool(selector_metrics.get("leakage_audit", {}).get("passed", False)):
        raise ValueError("saved exp264 Stage C leakage audit did not pass")
    if int(selector_metrics.get("model_count", -1)) != 40 or int(
        selector_models.get("model_count", -1)
    ) != 40:
        raise ValueError("saved exp264 Stage C must contain 40 selector models")

    manifest = json.loads(files["nested_compact_manifest"].read_text())
    partitions = list(manifest.get("partitions", []))
    expected_partitions = int(
        config.get("validation", {}).get("expected_compact_partitions", 25)
    )
    expected_rows = int(config.get("validation", {}).get("expected_rows", -1)) * int(
        config.get("validation", {}).get("n_outer_folds", 5)
    )
    if len(partitions) != expected_partitions:
        raise ValueError("saved exp264 compact partition count mismatch")
    if int(manifest.get("rows", -1)) != expected_rows:
        raise ValueError("saved exp264 compact total row count mismatch")
    seen: set[tuple[int, str, int]] = set()
    verified_partition_count = 0
    for item in partitions:
        key = (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        )
        if key in seen:
            raise ValueError(f"duplicate saved exp264 compact partition: {key}")
        seen.add(key)
        path = root / str(item["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        if verify_partition_sha:
            observed = sha256_file(path)
            if observed != str(item["sha256"]):
                raise ValueError(f"saved exp264 compact partition SHA mismatch: {path}")
            verified_partition_count += 1
    return {
        "root": str(root),
        "file_sha256": file_sha,
        "compact_meta_schema_logical_sha256": logical_sha,
        "compact_feature_count": len(compact_schema["features"]),
        "compact_features": [str(item) for item in compact_schema["features"]],
        "partition_count": len(partitions),
        "rows": int(manifest["rows"]),
        "partition_sha_verified": bool(verify_partition_sha),
        "verified_partition_count": verified_partition_count,
        "partitions": partitions,
    }


def verify_stage_a_feature_contract(
    feature_schema_path: Path,
    feature_catalog_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    schema_file_sha = sha256_file(feature_schema_path)
    catalog_file_sha = sha256_file(feature_catalog_path)
    if schema_file_sha != str(data_cfg["corrected_stage_a_feature_schema_file_sha256"]):
        raise ValueError("corrected exp264 Stage A feature schema file SHA mismatch")
    if catalog_file_sha != str(data_cfg["corrected_stage_a_feature_catalog_file_sha256"]):
        raise ValueError("corrected exp264 Stage A feature catalog file SHA mismatch")
    schema = load_feature_schema(feature_schema_path)
    if schema["feature_schema_sha256"] != str(
        data_cfg["corrected_stage_a_feature_schema_logical_sha256"]
    ):
        raise ValueError("corrected exp264 Stage A logical feature schema SHA mismatch")
    features = [str(item) for item in schema["features"]]
    if len(features) != int(data_cfg["corrected_stage_a_selector_feature_count"]):
        raise ValueError("corrected exp264 Stage A selector feature count mismatch")
    catalog = pd.read_csv(feature_catalog_path)
    selected_mask = catalog["selected"].astype(str).str.lower().eq("true")
    catalog_features = catalog.loc[selected_mask, "feature"].astype(str).tolist()
    if catalog_features != features:
        raise ValueError("corrected exp264 Stage A catalog/schema order mismatch")
    return {
        "feature_schema_path": str(feature_schema_path),
        "feature_catalog_path": str(feature_catalog_path),
        "feature_schema_file_sha256": schema_file_sha,
        "feature_catalog_file_sha256": catalog_file_sha,
        "feature_schema_logical_sha256": schema["feature_schema_sha256"],
        "feature_count": len(features),
        "features": features,
    }


def stage_s_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    selector = dict(config["model"]["selector"])
    execution = dict(selector["execution_count"])
    variants = [str(item) for item in selector["active_variants"]]
    objectives = [str(item) for item in selector["objectives"]]
    outer_folds = int(execution["outer_folds"])
    inner_folds = int(execution["inner_folds"])
    expected_boosters = outer_folds * inner_folds * len(objectives)
    if variants != ["signed_residual_l2"] or objectives != ["signed_residual"]:
        raise ValueError("Stage S must contain one signed-residual L2 variant/objective")
    if int(execution["planned_cpu_boosters"]) != expected_boosters:
        raise ValueError("Stage S planned CPU booster count mismatch")
    if int(execution["existing_selector_retraining_boosters"]) != 0:
        raise ValueError("Stage S must not retrain the saved exp264 selector")
    if bool(config["execution"]["control_retraining"]):
        raise ValueError("Stage S control retraining must remain disabled")
    return {
        "active_variants": len(variants),
        "objectives": len(objectives),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "planned_cpu_boosters": expected_boosters,
        "existing_selector_retraining_boosters": 0,
        "downstream_gpu_boosters": 0,
        "runtime": "kaggle_cpu",
    }


def _partition_index(parent_evidence: Mapping[str, Any]) -> dict[tuple[int, str, int], dict]:
    return {
        (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        ): dict(item)
        for item in parent_evidence["partitions"]
    }


def run_stage_s_preflight(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    parent_stage_c_root: Path,
    feature_schema_path: Path,
    feature_catalog_path: Path,
    output_dir: Path,
    verify_parent_partition_sha: bool = True,
    require_parent_score_guard: bool = True,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    cost = stage_s_cost_contract(config)
    cache_evidence = verify_exp263_root(cache_root, config)
    if cache_evidence["rows"] != int(config["validation"]["expected_rows"]):
        raise ValueError("exp263 cache row count mismatch")
    if cache_evidence["wells"] != int(config["validation"]["expected_wells"]):
        raise ValueError("exp263 cache well count mismatch")
    feature_evidence = verify_stage_a_feature_contract(
        feature_schema_path, feature_catalog_path, config
    )
    parent_evidence = verify_saved_exp264_stage_c_root(
        parent_stage_c_root,
        config,
        verify_partition_sha=verify_parent_partition_sha,
        require_score_guard=require_parent_score_guard,
    )
    ids = candidate_ids(contract)
    if len(ids) != int(config["model"]["selector"]["candidate_count"]):
        raise ValueError("candidate contract count differs from exp335 config")
    expected_candidate_order = [
        str(item) for item in config["model"]["selector"].get("candidate_order", ids)
    ]
    if ids != expected_candidate_order:
        raise ValueError("candidate contract order differs from frozen exp335 order")
    schema = signed_compact_schema(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "signed_compact_schema.json"
    write_json(schema_path, schema)
    parent_partitions = _partition_index(parent_evidence)
    probe_fold = 0
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    probe_bundle = cache.load_fold(probe_fold)
    probe_rows = min(1024, len(probe_bundle.base))
    if probe_rows <= 0:
        raise ValueError("Stage S preflight candidate cache probe is empty")
    probe_item = parent_partitions[(probe_fold, "valid", probe_fold)]
    probe_cursor = ParquetBatchCursor(
        parent_stage_c_root / str(probe_item["path"]),
        parent_compact_columns(contract),
        batch_size=probe_rows,
    )
    probe_parent = probe_cursor.take(probe_rows)
    _, probe_evidence = build_signed_compact_meta(
        probe_bundle.base.iloc[:probe_rows].reset_index(drop=True),
        probe_bundle.values[:probe_rows],
        np.zeros((probe_rows, len(ids)), dtype=np.float32),
        probe_parent,
        contract,
    )
    del probe_bundle, probe_cursor, probe_parent
    gc.collect()
    summary = {
        "status": "stage_s_preflight_complete",
        "cost_contract": cost,
        "exp263_cache": cache_evidence,
        "feature_contract": {
            key: value for key, value in feature_evidence.items() if key != "features"
        },
        "saved_exp264_stage_c": {
            key: value for key, value in parent_evidence.items() if key != "partitions"
        },
        "candidate_count": len(ids),
        "candidate_order": ids,
        "candidate_contract_sha256": candidate_contract_sha(contract),
        "signed_compact_schema_sha256": schema["signed_compact_schema_sha256"],
        "signed_compact_schema_file_sha256": sha256_file(schema_path),
        "saved_parent_alignment_probe": {
            "fold": probe_fold,
            "rows": probe_rows,
            "parent_partition_sha256": str(probe_item["sha256"]),
            "top1_value_parity_max_abs_error": probe_evidence[
                "top1_value_parity_max_abs_error"
            ],
            "passed": probe_evidence["passed"],
        },
        "models_trained": 0,
        "control_models_trained": 0,
        "downstream_models_trained": 0,
        "passed": True,
    }
    write_json(output_dir / "stage_s_preflight.json", summary)
    return summary


def run_stage_s(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache_root: Path,
    parent_stage_c_root: Path,
    feature_schema_path: Path,
    feature_catalog_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
    require_parent_score_guard: bool = True,
    cache_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    preflight = run_stage_s_preflight(
        config=config,
        contract=contract,
        cache_root=cache_root,
        parent_stage_c_root=parent_stage_c_root,
        feature_schema_path=feature_schema_path,
        feature_catalog_path=feature_catalog_path,
        output_dir=output_dir,
        verify_parent_partition_sha=True,
        require_parent_score_guard=require_parent_score_guard,
        cache_factory=cache_factory,
    )
    parent_evidence = verify_saved_exp264_stage_c_root(
        parent_stage_c_root,
        config,
        verify_partition_sha=False,
        require_score_guard=require_parent_score_guard,
    )
    parent_partitions = _partition_index(parent_evidence)
    schema = load_feature_schema(feature_schema_path)
    features = [str(item) for item in schema["features"]]
    cache = (
        Exp263CandidateCache(cache_root, contract)
        if cache_factory is None
        else cache_factory(cache_root, contract)
    )
    ids = cache.ids
    n_candidates = len(ids)
    n_outer_folds = int(config["validation"]["n_outer_folds"])
    n_inner_folds = int(config["validation"]["n_inner_folds"])
    feature_cfg = dict(config["features"])
    feature_cfg["primary_domain"] = contract["legal_domains"]["primitive_pair_bank"][
        "candidates"
    ]
    feature_cfg["fixed_domain"] = contract["legal_domains"]["primitive_fixed_bank"][
        "candidates"
    ]
    train_cfg = dict(config["model"]["selector"]["training"])
    max_train_base_rows = int(train_cfg["max_train_base_rows_per_outer_fold"])
    max_valid_base_rows = int(train_cfg["max_valid_base_rows_for_early_stopping"])
    sample_base_rows_per_source = max(
        max_valid_base_rows,
        int(math.ceil(max_train_base_rows / max(n_inner_folds - 1, 1))),
    )

    sampled: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sampled_base_wells: dict[int, np.ndarray] = {}
    fold_label_summary: dict[int, dict[str, np.ndarray]] = {}
    fold_well_counts: dict[int, pd.DataFrame] = {}
    formula_parity_max = 0.0
    for source_fold in range(n_outer_folds):
        bundle = cache.load_fold(source_fold)
        if not bundle.available.all() or not np.isfinite(bundle.values).all():
            raise ValueError("Stage S requires complete finite exp263 candidate coverage")
        context, truth = build_raw_context(
            bundle.base, raw_train_dir, feature_cfg, require_truth=True
        )
        assert truth is not None
        sample_indices = deterministic_sample_indices(
            bundle.base,
            sample_base_rows_per_source,
            "exp264",
            "stage_c_sample",
            source_fold,
        )
        long, metadata = build_candidate_long_features(
            bundle,
            context,
            sample_indices,
            feature_cfg,
            expected_features=features,
        )
        labels, formula_evidence = add_signed_residual_labels(
            metadata, truth[sample_indices], n_candidates
        )
        formula_parity_max = max(
            formula_parity_max, float(formula_evidence["formula_parity_max_abs_error"])
        )
        sampled[source_fold] = (long, labels)
        sampled_base_wells[source_fold] = (
            bundle.base.iloc[sample_indices]["well"].astype(str).to_numpy()
        )
        full_signed = (truth[:, None] - bundle.values).astype(np.float32)
        fold_label_summary[source_fold] = {
            "signed_sum": full_signed.sum(axis=0, dtype=np.float64),
            "count": np.full(n_candidates, len(bundle.base), dtype=np.float64),
        }
        fold_well_counts[source_fold] = (
            bundle.base.groupby("well", sort=True).size().rename("rows").reset_index()
        )
        del bundle, context, truth, long, metadata, labels, full_signed
        gc.collect()

    inner_maps, fold_manifest = build_nested_inner_fold_maps(
        fold_well_counts,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
    )
    fold_manifest_path = output_dir / "signed_nested_fold_manifest.csv"
    fold_manifest.to_csv(fold_manifest_path, index=False)

    def sampled_descriptor(downstream_outer_fold: int) -> pd.DataFrame:
        assignment = inner_maps[downstream_outer_fold]
        parts: list[pd.DataFrame] = []
        for source_fold in range(n_outer_folds):
            if source_fold == downstream_outer_fold:
                continue
            wells = sampled_base_wells[source_fold]
            inner = np.asarray([assignment.get(str(well), -1) for well in wells], dtype=np.int8)
            if np.any(inner < 0):
                raise AssertionError("sampled outer-train well lacks inner-fold assignment")
            parts.append(
                pd.DataFrame(
                    {
                        "source_fold": np.int8(source_fold),
                        "base_position": np.arange(len(wells), dtype=np.int32),
                        "well": wells,
                        "inner_fold": inner,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    def bounded_descriptor(
        descriptor: pd.DataFrame, limit: int, *seed_parts: Any
    ) -> pd.DataFrame:
        selected = deterministic_sample_indices(
            descriptor, min(int(limit), len(descriptor)), *seed_parts
        )
        return descriptor.iloc[selected].sort_values(
            ["source_fold", "base_position"], kind="stable"
        )

    def gather_sampled_long(
        descriptor: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_parts: list[pd.DataFrame] = []
        label_parts: list[pd.DataFrame] = []
        for source_fold, group in descriptor.groupby("source_fold", sort=True):
            base_positions = group["base_position"].to_numpy(np.int64)
            long_positions = (
                base_positions[:, None] * n_candidates
                + np.arange(n_candidates, dtype=np.int64)[None, :]
            ).reshape(-1)
            source_features, source_labels = sampled[int(source_fold)]
            feature_parts.append(source_features.iloc[long_positions])
            label_parts.append(source_labels.iloc[long_positions])
        return (
            pd.concat(feature_parts, ignore_index=True).astype(np.float32),
            pd.concat(label_parts, ignore_index=True),
        )

    common = dict(config["model"]["selector"]["lightgbm_common"])
    seed = int(config["validation"]["seed"])
    num_round = int(train_cfg["num_boost_round"])

    def model_callbacks() -> list[Any]:
        return [
            early_stopping(int(train_cfg["early_stopping_rounds"]), verbose=False),
            log_evaluation(int(train_cfg["log_evaluation_period"])),
        ]

    model_dir = output_dir / "signed_nested_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    compact_root = output_dir / "signed_nested_compact_meta"
    compact_root.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "signed_outer_valid_candidate_score.parquet"
    score_writer = IncrementalParquetWriter(score_path)
    model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    compact_partition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    candidate_metric_rows: list[dict[str, Any]] = []
    top1_parity_max = 0.0

    for downstream_outer_fold in range(n_outer_folds):
        assignment = inner_maps[downstream_outer_fold]
        descriptor = sampled_descriptor(downstream_outer_fold)
        outer_valid_wells = set(
            fold_well_counts[downstream_outer_fold]["well"].astype(str)
        )
        models_by_inner: dict[int, Any] = {}
        model_sha_by_inner: dict[int, str] = {}
        for inner_fold in range(n_inner_folds):
            train_pool = descriptor[descriptor["inner_fold"].ne(inner_fold)]
            valid_pool = descriptor[descriptor["inner_fold"].eq(inner_fold)]
            fit_train = bounded_descriptor(
                train_pool,
                max_train_base_rows,
                "exp264",
                "stage_c_train",
                downstream_outer_fold,
                inner_fold,
            )
            fit_valid = bounded_descriptor(
                valid_pool,
                max_valid_base_rows,
                "exp264",
                "stage_c_valid",
                downstream_outer_fold,
                inner_fold,
            )
            train_wells = set(fit_train["well"].astype(str))
            valid_wells = set(fit_valid["well"].astype(str))
            if train_wells.intersection(valid_wells):
                raise AssertionError("Stage S inner train/valid well overlap")
            if train_wells.intersection(outer_valid_wells):
                raise AssertionError("Stage S outer-valid well leaked into selector fit")
            x_train, y_train = gather_sampled_long(fit_train)
            x_valid, y_valid = gather_sampled_long(fit_valid)
            model = LGBMRegressor(
                objective="regression_l2",
                n_estimators=num_round,
                random_state=seed + 20_000 * downstream_outer_fold + inner_fold,
                **common,
            )
            model.fit(
                x_train,
                y_train[SIGNED_TARGET_COLUMN],
                eval_set=[(x_valid, y_valid[SIGNED_TARGET_COLUMN])],
                eval_metric="rmse",
                callbacks=model_callbacks(),
            )
            model_path = model_dir / (
                f"selector_signed_residual_outer{downstream_outer_fold}_inner{inner_fold}.txt"
            )
            model.booster_.save_model(str(model_path))
            model_sha = sha256_file(model_path)
            models_by_inner[inner_fold] = model
            model_sha_by_inner[inner_fold] = model_sha
            model_rows.append(
                {
                    "downstream_outer_fold": downstream_outer_fold,
                    "inner_fold": inner_fold,
                    "objective": "signed_residual",
                    "path": str(model_path.relative_to(output_dir)),
                    "sha256": model_sha,
                    "best_iteration": int(model.best_iteration_),
                    "fit_train_base_rows": len(fit_train),
                    "fit_valid_base_rows": len(fit_valid),
                    "fit_train_long_rows": len(x_train),
                    "fit_valid_long_rows": len(x_valid),
                    "fit_train_wells": len(train_wells),
                    "fit_valid_wells": len(valid_wells),
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(importance_type=importance_type)
                for feature, value in zip(features, importance, strict=True):
                    importance_rows.append(
                        {
                            "downstream_outer_fold": downstream_outer_fold,
                            "inner_fold": inner_fold,
                            "objective": "signed_residual",
                            "feature": feature,
                            "importance_type": importance_type,
                            "importance": float(value),
                        }
                    )
            del x_train, x_valid, y_train, y_valid
            gc.collect()

        outer_model_set_sha = sha256_json(
            [
                {"inner_fold": inner_fold, "signed_residual": model_sha_by_inner[inner_fold]}
                for inner_fold in range(n_inner_folds)
            ]
        )
        prior_sum = sum(
            fold_label_summary[source_fold]["signed_sum"]
            for source_fold in range(n_outer_folds)
            if source_fold != downstream_outer_fold
        )
        prior_count = sum(
            fold_label_summary[source_fold]["count"]
            for source_fold in range(n_outer_folds)
            if source_fold != downstream_outer_fold
        )
        signed_prior = prior_sum / prior_count
        fold_actual: list[np.ndarray] = []
        fold_prediction: list[np.ndarray] = []
        chunk_size = int(train_cfg["predict_base_row_chunk_size"])
        for source_fold in range(n_outer_folds):
            role = "valid" if source_fold == downstream_outer_fold else "train"
            bundle = cache.load_fold(source_fold)
            if not bundle.available.all() or not np.isfinite(bundle.values).all():
                raise ValueError("Stage S prediction requires complete finite candidate coverage")
            context, truth = build_raw_context(
                bundle.base, raw_train_dir, feature_cfg, require_truth=True
            )
            assert truth is not None
            shape_state = ShapeState.from_bundle(bundle.base, bundle.values)
            parent_item = parent_partitions[(downstream_outer_fold, role, source_fold)]
            parent_path = parent_stage_c_root / str(parent_item["path"])
            parent_cursor = ParquetBatchCursor(
                parent_path, parent_compact_columns(contract), chunk_size
            )
            partition_path = (
                compact_root
                / f"downstream_outer_fold={downstream_outer_fold}"
                / f"role={role}"
                / f"source_outer_fold={source_fold}"
                / "part-00000.parquet"
            )
            compact_writer = IncrementalParquetWriter(partition_path)
            source_inner = None
            if role == "train":
                source_inner = np.asarray(
                    [assignment.get(str(well), -1) for well in bundle.base["well"]],
                    dtype=np.int8,
                )
                if np.any(source_inner < 0):
                    raise AssertionError("outer-train row lacks nested inner-fold assignment")
            for start in range(0, len(bundle.base), chunk_size):
                stop = min(start + chunk_size, len(bundle.base))
                indices = np.arange(start, stop, dtype=np.int64)
                long, metadata = build_candidate_long_features(
                    bundle,
                    context,
                    indices,
                    feature_cfg,
                    shape_state=shape_state,
                    expected_features=features,
                )
                x = long.astype(np.float32)
                signed_matrix = np.zeros((len(indices), n_candidates), dtype=np.float32)
                if role == "valid":
                    for inner_fold in range(n_inner_folds):
                        model = models_by_inner[inner_fold]
                        prediction = model.predict(x, num_iteration=model.best_iteration_)
                        signed_matrix += prediction.reshape(
                            len(indices), n_candidates
                        ).astype(np.float32)
                    signed_matrix /= np.float32(n_inner_folds)
                    selector_model_count = n_inner_folds
                else:
                    assert source_inner is not None
                    chunk_inner = source_inner[indices]
                    for inner_fold in np.unique(chunk_inner):
                        base_positions = np.flatnonzero(chunk_inner == inner_fold)
                        long_positions = (
                            base_positions[:, None] * n_candidates
                            + np.arange(n_candidates, dtype=np.int64)[None, :]
                        ).reshape(-1)
                        model = models_by_inner[int(inner_fold)]
                        prediction = model.predict(
                            x.iloc[long_positions], num_iteration=model.best_iteration_
                        )
                        signed_matrix[base_positions] = prediction.reshape(
                            len(base_positions), n_candidates
                        ).astype(np.float32)
                    selector_model_count = 1
                if not np.isfinite(signed_matrix).all():
                    raise ValueError("Stage S selector produced non-finite prediction")

                parent_chunk = parent_cursor.take(len(indices))
                compact, compact_evidence = build_signed_compact_meta(
                    bundle.base.iloc[indices].reset_index(drop=True),
                    bundle.values[indices],
                    signed_matrix,
                    parent_chunk,
                    contract,
                )
                top1_parity_max = max(
                    top1_parity_max,
                    float(compact_evidence["top1_value_parity_max_abs_error"]),
                )
                compact["downstream_outer_fold"] = np.int8(downstream_outer_fold)
                compact["nested_role"] = role
                compact["signed_selector_model_count"] = np.int8(selector_model_count)
                compact_writer.write(compact)

                if role == "valid":
                    labels, formula_evidence = add_signed_residual_labels(
                        metadata, truth[indices], n_candidates
                    )
                    formula_parity_max = max(
                        formula_parity_max,
                        float(formula_evidence["formula_parity_max_abs_error"]),
                    )
                    actual = labels[SIGNED_TARGET_COLUMN].to_numpy(np.float32).reshape(
                        len(indices), n_candidates
                    )
                    score = metadata.copy()
                    score["actual_signed_residual"] = actual.reshape(-1)
                    score[SIGNED_PREDICTION_COLUMN] = signed_matrix.reshape(-1)
                    score["candidate_outer_train_mean_prior"] = np.tile(
                        signed_prior.astype(np.float32), len(indices)
                    )
                    score["downstream_outer_fold"] = np.int8(downstream_outer_fold)
                    score["nested_model_count"] = np.int8(n_inner_folds)
                    score["nested_model_set_sha"] = outer_model_set_sha
                    score["feature_schema_sha"] = schema["feature_schema_sha256"]
                    score["candidate_contract_sha"] = candidate_contract_sha(contract)
                    score_writer.write(score)
                    fold_actual.append(actual)
                    fold_prediction.append(signed_matrix.copy())
                del long, metadata, x, signed_matrix, parent_chunk, compact
            parent_cursor.finish()
            compact_writer.close()
            compact_partition_rows.append(
                {
                    "downstream_outer_fold": downstream_outer_fold,
                    "role": role,
                    "source_outer_fold": source_fold,
                    "rows": compact_writer.rows,
                    "wells": int(bundle.base["well"].nunique()),
                    "selector_model_count": n_inner_folds if role == "valid" else 1,
                    "path": str(partition_path.relative_to(output_dir)),
                    "sha256": sha256_file(partition_path),
                    "model_set_sha256": outer_model_set_sha,
                    "saved_exp264_partition_sha256": str(parent_item["sha256"]),
                }
            )
            del bundle, context, truth, shape_state, parent_cursor, compact_writer
            gc.collect()

        actual_matrix = np.concatenate(fold_actual, axis=0)
        prediction_matrix = np.concatenate(fold_prediction, axis=0)
        prior_matrix = np.broadcast_to(signed_prior, actual_matrix.shape)
        residual = prediction_matrix.astype(np.float64) - actual_matrix.astype(np.float64)
        prior_residual = prior_matrix.astype(np.float64) - actual_matrix.astype(np.float64)
        long_rows = actual_matrix.size
        metric_rows.append(
            {
                "scope": "outer_valid_inner_ensemble",
                "fold": downstream_outer_fold,
                "base_rows": len(actual_matrix),
                "long_rows": long_rows,
                "signed_residual_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                "prior_signed_residual_rmse": float(
                    np.sqrt(np.mean(np.square(prior_residual)))
                ),
                "signed_sse": float(np.square(residual).sum()),
                "prior_signed_sse": float(np.square(prior_residual).sum()),
                "sign_accuracy": float(
                    np.mean(np.sign(prediction_matrix) == np.sign(actual_matrix))
                ),
            }
        )
        for position, candidate_id in enumerate(ids):
            candidate_residual = residual[:, position]
            candidate_prior_residual = prior_residual[:, position]
            candidate_metric_rows.append(
                {
                    "fold": downstream_outer_fold,
                    "candidate_id": candidate_id,
                    "rows": len(actual_matrix),
                    "signed_residual_rmse": float(
                        np.sqrt(np.mean(np.square(candidate_residual)))
                    ),
                    "prior_signed_residual_rmse": float(
                        np.sqrt(np.mean(np.square(candidate_prior_residual)))
                    ),
                    "sign_accuracy": float(
                        np.mean(
                            np.sign(prediction_matrix[:, position])
                            == np.sign(actual_matrix[:, position])
                        )
                    ),
                }
            )
        del models_by_inner, actual_matrix, prediction_matrix, prior_matrix
        gc.collect()

    score_writer.close()
    model_count = len(model_rows)
    expected_model_count = n_outer_folds * n_inner_folds
    if model_count != expected_model_count:
        raise AssertionError("Stage S model count mismatch")
    partition_manifest = pd.DataFrame(compact_partition_rows)
    partition_manifest_path = output_dir / "signed_compact_partition_manifest.csv"
    partition_manifest.to_csv(partition_manifest_path, index=False)
    expected_partition_count = n_outer_folds * n_outer_folds
    expected_compact_rows = int(config["validation"]["expected_rows"]) * n_outer_folds
    compact_rows = int(partition_manifest["rows"].sum())
    if len(partition_manifest) != expected_partition_count:
        raise AssertionError("Stage S signed compact partition count mismatch")
    if compact_rows != expected_compact_rows:
        raise AssertionError("Stage S signed compact row coverage mismatch")
    expected_score_rows = int(config["validation"]["expected_rows"]) * n_candidates
    if score_writer.rows != expected_score_rows:
        raise AssertionError("Stage S outer-valid candidate score coverage mismatch")

    importance = pd.DataFrame(importance_rows)
    importance_path = output_dir / "signed_feature_importance_by_outer_inner.csv"
    importance.to_csv(importance_path, index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "signed_selector_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    candidate_metrics_path = output_dir / "signed_selector_candidate_metrics.csv"
    pd.DataFrame(candidate_metric_rows).to_csv(candidate_metrics_path, index=False)
    gate = evaluate_signed_residual_gate(
        metrics,
        minimum_improved_outer_folds=int(
            config["guards"]["stage_s"]["minimum_improved_outer_folds_vs_prior"]
        ),
    )
    technical_gate = {
        "model_count": model_count,
        "expected_model_count": expected_model_count,
        "outer_valid_excluded_from_inner_assignments": True,
        "inner_train_valid_well_disjoint": True,
        "outer_train_compact_source": "inner_oof",
        "outer_valid_compact_source": "four_inner_model_ensemble",
        "candidate_order_match": ids
        == [str(item) for item in config["model"]["selector"]["candidate_order"]],
        "feature_count": len(features),
        "expected_feature_count": int(
            config["model"]["selector"]["input_feature_count"]
        ),
        "compact_partition_count": len(partition_manifest),
        "compact_rows": compact_rows,
        "outer_valid_score_long_rows": score_writer.rows,
        "formula_parity_max_abs_error": formula_parity_max,
        "formula_parity_atol": float(config["guards"]["stage_s"]["formula_parity_atol"]),
        "saved_exp264_top1_value_parity_max_abs_error": top1_parity_max,
        "saved_exp264_top1_value_parity_atol": float(
            config["guards"]["stage_s"]["saved_top1_value_parity_atol"]
        ),
    }
    technical_gate["passed"] = bool(
        technical_gate["model_count"] == technical_gate["expected_model_count"]
        and technical_gate["candidate_order_match"]
        and technical_gate["feature_count"] == technical_gate["expected_feature_count"]
        and technical_gate["compact_partition_count"] == expected_partition_count
        and technical_gate["compact_rows"] == expected_compact_rows
        and technical_gate["outer_valid_score_long_rows"] == expected_score_rows
        and formula_parity_max <= technical_gate["formula_parity_atol"]
        and top1_parity_max <= technical_gate["saved_exp264_top1_value_parity_atol"]
    )

    model_manifest = {
        "schema_version": "1.0.0",
        "status": "signed_nested_selector_completed",
        "objective": "regression_l2",
        "target": "true_tvt-candidate_tvt",
        "candidate_order": ids,
        "feature_schema_sha256": schema["feature_schema_sha256"],
        "signed_compact_schema_sha256": preflight["signed_compact_schema_sha256"],
        "models": model_rows,
        "model_count": model_count,
        "fold_manifest_sha256": sha256_file(fold_manifest_path),
        "technical_gate": technical_gate,
    }
    model_manifest_path = output_dir / "signed_selector_model_manifest.json"
    write_json(model_manifest_path, model_manifest)
    compact_manifest = {
        "schema_version": "1.0.0",
        "status": "signed_nested_compact_completed",
        "layout": "downstream_outer_fold/role/source_outer_fold",
        "signed_compact_schema_sha256": preflight["signed_compact_schema_sha256"],
        "saved_exp264_compact_manifest_sha256": preflight["saved_exp264_stage_c"][
            "file_sha256"
        ]["nested_compact_manifest"],
        "partition_count": len(partition_manifest),
        "rows": compact_rows,
        "expected_rows": expected_compact_rows,
        "partitions": compact_partition_rows,
    }
    compact_manifest_path = output_dir / "signed_compact_manifest.json"
    write_json(compact_manifest_path, compact_manifest)
    summary = {
        "status": "stage_s_complete",
        "model_count": model_count,
        "compact_partition_count": len(partition_manifest),
        "compact_rows": compact_rows,
        "outer_valid_score_long_rows": score_writer.rows,
        "score_gate": gate,
        "technical_gate": technical_gate,
        "stage_s_gate_passed": bool(gate["passed"] and technical_gate["passed"]),
        "signed_selector_model_manifest_sha256": sha256_file(model_manifest_path),
        "signed_compact_manifest_sha256": sha256_file(compact_manifest_path),
        "signed_outer_valid_candidate_score_sha256": sha256_file(score_path),
        "signed_selector_metrics_sha256": sha256_file(metrics_path),
        "signed_feature_importance_sha256": sha256_file(importance_path),
    }
    write_json(output_dir / "signed_selector_metrics.json", summary)
    reproducibility = {
        "status": "stage_s_complete",
        "seed": seed,
        "sampling_namespace": "exp264_stage_c",
        "exp263_cache": preflight["exp263_cache"],
        "feature_contract": preflight["feature_contract"],
        "saved_exp264_stage_c": preflight["saved_exp264_stage_c"],
        "candidate_contract_sha256": preflight["candidate_contract_sha256"],
        "signed_compact_schema_sha256": preflight["signed_compact_schema_sha256"],
        "signed_selector_model_manifest_sha256": summary[
            "signed_selector_model_manifest_sha256"
        ],
        "signed_compact_manifest_sha256": summary["signed_compact_manifest_sha256"],
        "signed_outer_valid_candidate_score_sha256": summary[
            "signed_outer_valid_candidate_score_sha256"
        ],
        "stage_s_gate_passed": summary["stage_s_gate_passed"],
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    return summary


def stage_d_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the approved exp335 downstream scope at exactly 15 GPU boosters."""

    stage = dict(config["model"]["downstream_tvt"])
    execution = dict(stage["execution_count"])
    variants = [str(item) for item in stage["active_variants"]]
    config_indices = [int(item) for item in stage["lightgbm_config_indices"]]
    folds = int(execution["folds"])
    calculated = len(variants) * len(config_indices) * folds
    if variants != ["signed_residual_meta_addonly"]:
        raise ValueError("exp335 Stage D must contain only signed_residual_meta_addonly")
    if config_indices != [0, 1, 2]:
        raise ValueError("exp335 Stage D must reuse LightGBM configs 0, 1, and 2")
    if int(execution["variants"]) != 1 or int(execution["lightgbm_configs"]) != 3:
        raise ValueError("exp335 Stage D variant/config count changed")
    if folds != 5 or calculated != 15 or int(execution["planned_gpu_boosters"]) != 15:
        raise ValueError("exp335 Stage D must be 1 variant x 3 configs x 5 folds")
    if int(execution["control_retraining_boosters"]) != 0:
        raise ValueError("saved exp264 control retraining must remain zero")
    if bool(config["execution"].get("control_retraining", False)):
        raise ValueError("execution.control_retraining must remain false")
    expected_counts = {
        "expected_source_base_feature_count": 380,
        "expected_base_feature_count": 273,
        "saved_compact_feature_count": 74,
        "signed_compact_feature_count": 23,
        "final_feature_count": 370,
    }
    for key, expected in expected_counts.items():
        if int(stage[key]) != expected:
            raise ValueError(f"exp335 Stage D {key} changed: {stage[key]} != {expected}")
    return {
        "active_variants": variants,
        "lightgbm_config_indices": config_indices,
        "folds": folds,
        "planned_gpu_boosters": calculated,
        "saved_control_retraining_boosters": 0,
        "feature_counts": {
            "clean_base": 273,
            "saved_exp264_compact": 74,
            "signed_residual_compact": 23,
            "final": 370,
        },
        "runtime": "kaggle_t4",
    }


def verify_signed_stage_s_root(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_partition_sha: bool,
    verify_model_sha: bool = True,
    require_score_gate: bool = True,
) -> dict[str, Any]:
    """Verify the completed Stage S gate, models, schema, and 25 compact partitions."""

    root = Path(root)
    data = dict(config["data"])
    files = {
        "metrics": root / "signed_selector_metrics.json",
        "model_manifest": root / "signed_selector_model_manifest.json",
        "compact_manifest": root / "signed_compact_manifest.json",
        "compact_schema": root / "signed_compact_schema.json",
        "reproducibility": root / "reproducibility_manifest.json",
    }
    expected_sha = {
        "metrics": data["stage_s_signed_selector_metrics_sha256"],
        "model_manifest": data["stage_s_model_manifest_sha256"],
        "compact_manifest": data["stage_s_compact_manifest_sha256"],
        "compact_schema": data["stage_s_compact_schema_file_sha256"],
        "reproducibility": data["stage_s_reproducibility_manifest_sha256"],
    }
    file_sha: dict[str, str] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Stage S contract file missing: {path}")
        observed = sha256_file(path)
        expected = str(expected_sha[name])
        if observed != expected:
            raise ValueError(f"Stage S {name} SHA mismatch: {observed} != {expected}")
        file_sha[name] = observed

    metrics = json.loads(files["metrics"].read_text())
    models = json.loads(files["model_manifest"].read_text())
    manifest = json.loads(files["compact_manifest"].read_text())
    schema = json.loads(files["compact_schema"].read_text())
    if require_score_gate and not bool(metrics.get("stage_s_gate_passed", False)):
        raise ValueError("Stage S combined gate did not pass")
    if require_score_gate and not bool(metrics.get("score_gate", {}).get("passed", False)):
        raise ValueError("Stage S score gate did not pass")
    if not bool(metrics.get("technical_gate", {}).get("passed", False)):
        raise ValueError("Stage S technical gate did not pass")
    if int(models.get("model_count", -1)) != 20 or int(metrics.get("model_count", -1)) != 20:
        raise ValueError("Stage S must contain exactly 20 selector models")
    features = [str(item) for item in schema.get("features", [])]
    if len(features) != 23 or len(set(features)) != 23:
        raise ValueError("Stage S compact schema must contain 23 unique features")
    logical_sha = str(schema.get("signed_compact_schema_sha256", ""))
    if logical_sha != str(data["stage_s_compact_schema_logical_sha256"]):
        raise ValueError("Stage S compact logical schema SHA mismatch")
    if logical_sha != str(manifest.get("signed_compact_schema_sha256", "")):
        raise ValueError("Stage S schema and compact manifest disagree")

    model_evidence: list[dict[str, Any]] = []
    for item in models.get("models", []):
        path = root / str(item["path"])
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage S model missing: {path}")
        observed = sha256_file(path) if verify_model_sha else str(item["sha256"])
        if observed != str(item["sha256"]):
            raise ValueError(f"Stage S model SHA mismatch: {path}")
        model_evidence.append({"path": str(path), "sha256": observed})
    if len(model_evidence) != 20:
        raise ValueError("Stage S model inventory is incomplete")

    partitions: list[dict[str, Any]] = []
    total_rows = 0
    for item in manifest.get("partitions", []):
        path = root / str(item["path"])
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage S compact partition missing: {path}")
        observed = sha256_file(path) if verify_partition_sha else str(item["sha256"])
        if observed != str(item["sha256"]):
            raise ValueError(f"Stage S compact partition SHA mismatch: {path}")
        evidence = dict(item)
        evidence["path"] = str(path)
        evidence["sha256"] = observed
        partitions.append(evidence)
        total_rows += int(item["rows"])
    if len(partitions) != 25 or total_rows != 18_919_945:
        raise ValueError("Stage S compact partition inventory mismatch")
    return {
        "root": str(root),
        "file_sha256": file_sha,
        "model_count": len(model_evidence),
        "model_sha_verified": bool(verify_model_sha),
        "partition_count": len(partitions),
        "partition_sha_verified": bool(verify_partition_sha),
        "rows": total_rows,
        "features": features,
        "feature_count": len(features),
        "signed_compact_schema_sha256": logical_sha,
        "partitions": partitions,
        "score_gate": metrics["score_gate"],
        "technical_gate": metrics["technical_gate"],
    }


def _load_parent_compact_fold(
    root: Path,
    evidence: Mapping[str, Any],
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    load_evidence = dict(evidence)
    load_evidence["partitions"] = [
        {
            **dict(item),
            "path": str(Path(root) / str(item["path"]))
            if not Path(str(item["path"])).is_absolute()
            else str(item["path"]),
        }
        for item in evidence["partitions"]
    ]
    return load_stage_d_compact_fold(
        stage_c_root=Path(root),
        stage_c_evidence=load_evidence,
        downstream_outer_fold=int(outer_fold),
    )


def load_signed_compact_fold(
    *,
    stage_s_evidence: Mapping[str, Any],
    downstream_outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [str(item) for item in stage_s_evidence["features"]]
    columns = [
        *KEY_COLUMNS,
        "last_known_tvt",
        "downstream_outer_fold",
        "nested_role",
        "signed_selector_model_count",
        *features,
    ]
    by_role: dict[str, list[pd.DataFrame]] = {"train": [], "valid": []}
    selected = [
        item
        for item in stage_s_evidence["partitions"]
        if int(item["downstream_outer_fold"]) == int(downstream_outer_fold)
    ]
    for item in sorted(selected, key=lambda value: int(value["source_outer_fold"])):
        role = str(item["role"])
        if role not in by_role:
            raise ValueError(f"unexpected Stage S compact role: {role}")
        frame = pd.read_parquet(Path(item["path"]), columns=columns)
        if len(frame) != int(item["rows"]):
            raise ValueError(f"Stage S compact partition row mismatch: {item['path']}")
        if not frame["outer_fold"].eq(int(item["source_outer_fold"])).all():
            raise ValueError(f"Stage S source outer fold mismatch: {item['path']}")
        expected_models = 4 if role == "valid" else 1
        if not frame["signed_selector_model_count"].eq(expected_models).all():
            raise ValueError(f"Stage S selector model-count mismatch: {item['path']}")
        by_role[role].append(frame)
    if len(by_role["train"]) != 4 or len(by_role["valid"]) != 1:
        raise ValueError("Stage S fold must contain four train and one valid partitions")
    train = pd.concat(by_role["train"], ignore_index=True)
    valid = pd.concat(by_role["valid"], ignore_index=True)
    for role, frame in (("train", train), ("valid", valid)):
        if not frame["nested_role"].eq(role).all():
            raise ValueError(f"Stage S nested role mismatch: {role}")
        if not frame["downstream_outer_fold"].eq(int(downstream_outer_fold)).all():
            raise ValueError(f"Stage S downstream fold mismatch: {role}")
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"Stage S compact ids are duplicated in {role}")
        if not np.isfinite(frame[["last_known_tvt", *features]].to_numpy(np.float32)).all():
            raise ValueError(f"Stage S compact has non-finite values in {role}")
    if set(train["well"].astype(str)).intersection(set(valid["well"].astype(str))):
        raise ValueError("Stage S compact train/valid wells overlap")
    return train, valid


def load_saved_exp264_stage_d(
    *,
    oof_path: Path,
    metrics_path: Path,
    base_frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the frozen exp264 347-feature OOF and clean273 control without fitting."""

    oof_path = Path(oof_path)
    metrics_path = Path(metrics_path)
    data = dict(config["data"])
    oof_sha = sha256_file(oof_path)
    metrics_sha = sha256_file(metrics_path)
    if oof_sha != str(data["saved_exp264_stage_d_oof_sha256"]):
        raise ValueError("saved exp264 Stage D OOF SHA mismatch")
    if metrics_sha != str(data["saved_exp264_stage_d_metrics_sha256"]):
        raise ValueError("saved exp264 Stage D metrics SHA mismatch")
    metrics = json.loads(metrics_path.read_text())
    expected_rmse = float(config["validation"]["saved_control"]["rmse"])
    if int(metrics.get("model_count", -1)) != 30 or abs(
        float(metrics["selector_compact_addonly_lgb_mean_rmse"]) - expected_rmse
    ) > 1.0e-9:
        raise ValueError("saved exp264 Stage D metrics contract changed")
    columns = [
        "id",
        "well",
        "outer_fold",
        "actual_tvt",
        "matched_control__lgb_mean__pred_tvt",
        "selector_compact_addonly__lgb_mean__pred_tvt",
    ]
    frame = pd.read_parquet(oof_path, columns=columns)
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("saved exp264 OOF ids are duplicated")
    indexed = frame.set_index(frame["id"].astype(str), drop=False)
    base_ids = base_frame["id"].astype(str)
    if set(indexed.index) != set(base_ids):
        raise ValueError("saved exp264 OOF ids differ from the clean273 base")
    frame = indexed.loc[base_ids].reset_index(drop=True)
    if not frame["well"].astype(str).equals(base_frame["well"].astype(str).reset_index(drop=True)):
        raise ValueError("saved exp264 OOF well alignment mismatch")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    if float(np.max(np.abs(frame["actual_tvt"].to_numpy(np.float32) - truth))) > 1.0e-4:
        raise ValueError("saved exp264 OOF truth differs from the clean273 base")
    parent = frame["selector_compact_addonly__lgb_mean__pred_tvt"].to_numpy(np.float32)
    observed_rmse = _rmse(truth, parent)
    if abs(observed_rmse - expected_rmse) > 1.0e-9:
        raise ValueError(f"saved exp264 OOF RMSE mismatch: {observed_rmse}")
    return frame, {
        "oof_path": str(oof_path),
        "oof_sha256": oof_sha,
        "metrics_path": str(metrics_path),
        "metrics_sha256": metrics_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "saved_parent_rmse": observed_rmse,
        "saved_model_count": int(metrics["model_count"]),
        "models_retrained": 0,
    }


def stage_d_retained_base_columns(base_features: Sequence[str]) -> list[str]:
    """Return the Stage D base surface schema without duplicating model features."""

    required = ["id", "well", "target", "last_known_tvt", "md_since"]
    return list(dict.fromkeys([*required, *(str(item) for item in base_features)]))


def _rmse(actual: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def evaluate_stage_d_guards(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    saved_parent: pd.DataFrame,
    oof_fold: np.ndarray,
    new_prediction: np.ndarray,
    hidden_like_assignment_path: Path,
    signed_gain_total: float,
    signed_gain_max_feature_share: float,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    parent = saved_parent["selector_compact_addonly__lgb_mean__pred_tvt"].to_numpy(np.float32)
    clean = saved_parent["matched_control__lgb_mean__pred_tvt"].to_numpy(np.float32)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mask = np.asarray(oof_fold) == fold
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        fold_rows.append(
            {
                "outer_fold": fold,
                "rows": int(mask.sum()),
                "saved_exp264_rmse": parent_rmse,
                "signed_residual_meta_rmse": new_rmse,
                "delta_rmse_new_minus_exp264": new_rmse - parent_rmse,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    md_since = base_frame["md_since"].to_numpy(np.float32)
    masks = {
        "all": np.ones(len(base_frame), dtype=bool),
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        bucket_rows.append(
            {
                "bucket": name,
                "rows": int(mask.sum()),
                "saved_exp264_rmse": parent_rmse,
                "signed_residual_meta_rmse": new_rmse,
                "delta_rmse_new_minus_exp264": new_rmse - parent_rmse,
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
        if not np.any(mask):
            raise ValueError(f"hidden-like assignment has no valid rows for {column}")
        parent_rmse = _rmse(truth[mask], parent[mask])
        new_rmse = _rmse(truth[mask], new_prediction[mask])
        hidden_rows.append(
            {
                "assignment": column,
                "rows": int(mask.sum()),
                "wells": int(base_frame.loc[mask, "well"].nunique()),
                "saved_exp264_rmse": parent_rmse,
                "signed_residual_meta_rmse": new_rmse,
                "delta_rmse_new_minus_exp264": new_rmse - parent_rmse,
            }
        )
    hidden_metrics = pd.DataFrame(hidden_rows)
    well_source = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "clean273": clean,
            "saved_exp264_347": parent,
            "signed_residual_370": new_prediction,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in well_source.groupby("well", sort=True):
        clean_rmse = _rmse(group["actual_tvt"], group["clean273"])
        parent_rmse = _rmse(group["actual_tvt"], group["saved_exp264_347"])
        new_rmse = _rmse(group["actual_tvt"], group["signed_residual_370"])
        well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                "clean273_rmse": clean_rmse,
                "saved_exp264_347_rmse": parent_rmse,
                "signed_residual_370_rmse": new_rmse,
                "saved_exp264_minus_clean_delta": parent_rmse - clean_rmse,
                "new_minus_clean_delta": new_rmse - clean_rmse,
                "new_minus_exp264_delta": new_rmse - parent_rmse,
            }
        )
    by_well = pd.DataFrame(well_rows)
    guard_cfg = dict(config["guards"]["downstream_scientific_support"])
    pooled_parent = _rmse(truth, parent)
    pooled_new = _rmse(truth, new_prediction)
    improvement = pooled_parent - pooled_new
    nonworse_folds = int((fold_metrics["delta_rmse_new_minus_exp264"] <= 0.0).sum())
    scope_deltas = [
        *bucket_metrics.loc[
            bucket_metrics["bucket"].ne("all"), "delta_rmse_new_minus_exp264"
        ].astype(float),
        *hidden_metrics["delta_rmse_new_minus_exp264"].astype(float),
    ]
    by_well_p95 = float(by_well["new_minus_exp264_delta"].quantile(0.95))
    worst_well_delta = float(by_well["new_minus_exp264_delta"].max())
    scientific_checks = {
        "minimum_pooled_improvement": improvement
        >= float(guard_cfg["minimum_pooled_rmse_improvement_vs_exp264"]),
        "minimum_nonworse_folds": nonworse_folds
        >= int(guard_cfg["minimum_nonworse_folds_vs_exp264"]),
        "all_scopes_nonworse": max(scope_deltas)
        <= float(guard_cfg["maximum_scope_delta_rmse_vs_exp264"]),
        "by_well_delta_p95_nonpositive": by_well_p95
        <= float(guard_cfg["maximum_by_well_delta_p95_vs_exp264"]),
        "worst_well_delta_within_limit": worst_well_delta
        <= float(guard_cfg["maximum_worst_well_delta_rmse_vs_exp264"]),
        "signed_features_have_nonzero_gain": float(signed_gain_total) > 0.0,
    }
    scientific = {
        "saved_exp264_rmse": pooled_parent,
        "signed_residual_meta_rmse": pooled_new,
        "improvement_ft": improvement,
        "delta_rmse_new_minus_exp264": pooled_new - pooled_parent,
        "nonworse_folds": nonworse_folds,
        "maximum_scope_delta_rmse": max(scope_deltas),
        "by_well_delta_p95": by_well_p95,
        "worst_well": str(by_well.loc[by_well["new_minus_exp264_delta"].idxmax(), "well"]),
        "worst_well_delta_rmse": worst_well_delta,
        "signed_feature_gain_total": float(signed_gain_total),
        "signed_gain_max_feature_share": float(signed_gain_max_feature_share),
        "checks": scientific_checks,
        "passed": bool(all(scientific_checks.values())),
    }

    threshold_counts: dict[str, dict[str, int | bool]] = {}
    threshold_checks: list[bool] = []
    for threshold in (1.0, 3.0, 5.0):
        parent_count = int((by_well["saved_exp264_minus_clean_delta"] > threshold).sum())
        new_count = int((by_well["new_minus_clean_delta"] > threshold).sum())
        passed = new_count <= parent_count
        threshold_counts[f"plus_{threshold:g}ft"] = {
            "saved_exp264_vs_clean_count": parent_count,
            "new_vs_clean_count": new_count,
            "nonincrease": passed,
        }
        threshold_checks.append(passed)
    parent_worst_vs_clean = float(by_well["saved_exp264_minus_clean_delta"].max())
    new_worst_vs_clean = float(by_well["new_minus_clean_delta"].max())
    promotion_checks = {
        "scientific_support": scientific["passed"],
        "worst_well_vs_clean_nonincrease": new_worst_vs_clean <= parent_worst_vs_clean,
        "plus_1_3_5ft_counts_nonincrease": all(threshold_checks),
        "no_guard_relaxation": bool(config["guards"]["promotion"]["require_no_guard_relaxation"]),
    }
    promotion = {
        "saved_exp264_worst_well_delta_vs_clean": parent_worst_vs_clean,
        "new_worst_well_delta_vs_clean": new_worst_vs_clean,
        "worsened_well_threshold_counts": threshold_counts,
        "checks": promotion_checks,
        "passed": bool(all(promotion_checks.values())),
    }
    return scientific, promotion, fold_metrics, bucket_metrics, hidden_metrics, by_well


def run_stage_d(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    parent_stage_c_root: Path,
    stage_s_root: Path,
    saved_parent_oof_path: Path,
    saved_parent_metrics_path: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    clean_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Train the approved 370-feature add-only surface and compare to saved exp264."""

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not bool(config["execution"].get("downstream_train_approved", False)) or not bool(
        config["execution"].get("run_downstream_train", False)
    ):
        raise RuntimeError("Stage D requires downstream approval and run flag")
    cost = stage_d_cost_contract(config)
    parent_evidence = verify_saved_exp264_stage_c_root(
        parent_stage_c_root, config, verify_partition_sha=True
    )
    stage_s_evidence = verify_signed_stage_s_root(
        stage_s_root, config, verify_partition_sha=True, verify_model_sha=True
    )
    expected_signed = signed_compact_feature_names(contract)
    if stage_s_evidence["features"] != expected_signed:
        raise ValueError("Stage S signed compact schema/order differs from candidate contract")
    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=clean_allowlist_path,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    retained = stage_d_retained_base_columns(base_features)
    base_frame = base_frame.loc[:, ~base_frame.columns.duplicated()].loc[:, retained].copy()
    if not base_frame.columns.is_unique:
        raise ValueError("Stage D retained base surface columns are not unique")
    saved_parent, saved_parent_evidence = load_saved_exp264_stage_d(
        oof_path=saved_parent_oof_path,
        metrics_path=saved_parent_metrics_path,
        base_frame=base_frame,
        config=config,
    )
    hidden_sha = sha256_file(hidden_like_assignment_path)
    if hidden_sha != str(config["data"]["hidden_like_assignment_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")

    parent_features = [str(item) for item in parent_evidence["compact_features"]]
    signed_features = [str(item) for item in stage_s_evidence["features"]]
    final_features = [*base_features, *parent_features, *signed_features]
    if len(final_features) != int(cost["feature_counts"]["final"]) or len(
        set(final_features)
    ) != len(final_features):
        raise ValueError("Stage D final 370-feature schema is not unique and exact")
    stage_cfg = dict(config["model"]["downstream_tvt"])
    mode = dict(exp218_config["model"]["training"]["modes"][str(stage_cfg["mode"])])
    if not bool(mode.get("use_gpu", False)):
        raise ValueError("Stage D exp218 mode must use GPU")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode
    )
    config_indices = [int(item) for item in cost["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    base_index = pd.Index(base_frame["id"].astype(str))
    if not base_index.is_unique:
        raise ValueError("clean273 base ids are not unique")
    n_rows = len(base_frame)
    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    oof_by_config = [np.full(n_rows, np.nan, np.float32) for _ in params_family]
    oof_fold = np.full(n_rows, -1, np.int8)
    model_dir = output_dir / "stage_d_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    chunk_columns = int(stage_cfg["matrix_copy_chunk_columns"])
    parent_partition_index = {
        (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        ): str(item["sha256"])
        for item in parent_evidence["partitions"]
    }
    for item in stage_s_evidence["partitions"]:
        key = (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        )
        if str(item["saved_exp264_partition_sha256"]) != parent_partition_index[key]:
            raise ValueError(f"Stage S parent partition lineage SHA mismatch: {key}")

    for outer_fold in range(int(cost["folds"])):
        parent_train, parent_valid = _load_parent_compact_fold(
            parent_stage_c_root, parent_evidence, outer_fold
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=stage_s_evidence,
            downstream_outer_fold=outer_fold,
        )
        for role, parent, signed in (
            ("train", parent_train, signed_train),
            ("valid", parent_valid, signed_valid),
        ):
            if not parent[list(KEY_COLUMNS)].reset_index(drop=True).equals(
                signed[list(KEY_COLUMNS)].reset_index(drop=True)
            ):
                raise ValueError(f"saved74 and signed23 key alignment mismatch: {role}")
            if float(
                np.max(
                    np.abs(
                        parent["last_known_tvt"].to_numpy(np.float32)
                        - signed["last_known_tvt"].to_numpy(np.float32)
                    )
                )
            ) > 1.0e-4:
                raise ValueError(f"saved74 and signed23 anchor alignment mismatch: {role}")
        train_indices = base_index.get_indexer(parent_train["id"].astype(str))
        valid_indices = base_index.get_indexer(parent_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("compact ids are absent from clean273 base")
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != n_rows:
            raise ValueError("Stage D fold does not cover clean273 rows exactly once")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("Stage D train/valid indices overlap")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("Stage D OOF valid rows were assigned twice")
        oof_fold[valid_indices] = np.int8(outer_fold)
        x_train_values = np.empty((len(train_indices), len(final_features)), np.float32)
        x_valid_values = np.empty((len(valid_indices), len(final_features)), np.float32)
        for start in range(0, len(base_features), chunk_columns):
            stop = min(start + chunk_columns, len(base_features))
            columns = base_features[start:stop]
            source = base_frame[columns]
            x_train_values[:, start:stop] = source.iloc[train_indices].to_numpy(
                np.float32, copy=True
            )
            x_valid_values[:, start:stop] = source.iloc[valid_indices].to_numpy(
                np.float32, copy=True
            )
        parent_start = len(base_features)
        signed_start = parent_start + len(parent_features)
        x_train_values[:, parent_start:signed_start] = parent_train[parent_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, parent_start:signed_start] = parent_valid[parent_features].to_numpy(
            np.float32, copy=False
        )
        x_train_values[:, signed_start:] = signed_train[signed_features].to_numpy(
            np.float32, copy=False
        )
        x_valid_values[:, signed_start:] = signed_valid[signed_features].to_numpy(
            np.float32, copy=False
        )
        if not np.isfinite(x_train_values).all() or not np.isfinite(x_valid_values).all():
            raise ValueError("Stage D 370-feature matrix contains non-finite values before fit")
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
                    early_stopping(int(stage_cfg["early_stopping_rounds"]), verbose=False),
                    log_evaluation(int(stage_cfg["log_evaluation_period"])),
                ],
            )
            best_iteration = int(model.best_iteration_ or params["n_estimators"])
            prediction = model.predict(x_valid, num_iteration=best_iteration).astype(np.float32)
            oof_by_config[family_position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            rmse_value = _rmse(truth[valid_indices], anchor[valid_indices] + prediction)
            model_rows.append(
                {
                    "variant": "signed_residual_meta_addonly",
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
            fold_model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "model": f"lgb{config_index}",
                    "rows": len(valid_indices),
                    "rmse_tvt": rmse_value,
                    "best_iteration": best_iteration,
                }
            )
            for importance_type in ("gain", "split"):
                importance = model.booster_.feature_importance(importance_type=importance_type)
                importance_rows.extend(
                    {
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "importance_type": importance_type,
                        "feature": feature,
                        "feature_group": (
                            "signed_residual_compact"
                            if feature in signed_features
                            else "saved_exp264_compact"
                            if feature in parent_features
                            else "clean_base"
                        ),
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, importance, strict=True)
                )
            print(
                json.dumps(
                    {
                        "stage": "D",
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": rmse_value,
                        "best_iteration": best_iteration,
                        "completed_boosters": len(model_rows),
                        "planned_boosters": cost["planned_gpu_boosters"],
                        "saved_control_retraining": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del model, prediction
            gc.collect()
        fold_mean = np.mean(np.vstack(fold_predictions), axis=0).astype(np.float32)
        fold_model_rows.append(
            {
                "outer_fold": outer_fold,
                "model": "lgb_mean",
                "rows": len(valid_indices),
                "rmse_tvt": _rmse(truth[valid_indices], anchor[valid_indices] + fold_mean),
                "best_iteration": None,
            }
        )
        del (
            parent_train,
            parent_valid,
            signed_train,
            signed_valid,
            x_train,
            x_valid,
            x_train_values,
            x_valid_values,
            fold_predictions,
            fold_mean,
        )
        gc.collect()

    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("Stage D 15-model OOF contract is incomplete")
    if not np.array_equal(
        oof_fold, saved_parent["outer_fold"].to_numpy(np.int8)
    ):
        raise AssertionError("Stage D fold assignment differs from saved exp264")
    for prediction in oof_by_config:
        if not np.isfinite(prediction).all():
            raise AssertionError("Stage D OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    importance = pd.DataFrame(importance_rows)
    signed_gain = (
        importance[
            importance["importance_type"].eq("gain")
            & importance["feature_group"].eq("signed_residual_compact")
        ]
        .groupby("feature", as_index=False)["importance"]
        .sum()
    )
    signed_gain_total = float(signed_gain["importance"].sum())
    signed_gain_max_share = (
        float(signed_gain["importance"].max() / signed_gain_total)
        if signed_gain_total > 0.0
        else 1.0
    )
    scientific, promotion, fold_metrics, bucket_metrics, hidden_metrics, by_well = (
        evaluate_stage_d_guards(
            config=config,
            base_frame=base_frame,
            saved_parent=saved_parent,
            oof_fold=oof_fold,
            new_prediction=mean_prediction,
            hidden_like_assignment_path=hidden_like_assignment_path,
            signed_gain_total=signed_gain_total,
            signed_gain_max_feature_share=signed_gain_max_share,
        )
    )
    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    for config_index, residual in zip(config_indices, oof_by_config, strict=True):
        prediction_frame[f"signed_residual_meta_addonly__lgb{config_index}__pred_tvt"] = (
            anchor + residual
        ).astype(np.float32)
    prediction_frame["signed_residual_meta_addonly__lgb_mean__pred_tvt"] = mean_prediction
    paths = {
        "oof": output_dir / "stage_d_oof_predictions.parquet",
        "fold_metrics": output_dir / "stage_d_fold_metrics.csv",
        "bucket_metrics": output_dir / "stage_d_bucket_metrics.csv",
        "hidden_metrics": output_dir / "stage_d_hidden_like_metrics.csv",
        "by_well": output_dir / "stage_d_by_well.csv",
        "importance": output_dir / "stage_d_feature_importance.csv",
        "model_manifest": output_dir / "stage_d_model_manifest.json",
        "metrics": output_dir / "stage_d_metrics.json",
    }
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(fold_metrics, on="outer_fold", how="left").to_csv(
        paths["fold_metrics"], index=False
    )
    bucket_metrics.to_csv(paths["bucket_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    importance.to_csv(paths["importance"], index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "stage_d_15_gpu_boosters_completed",
        "cost_contract": cost,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": sha256_json(final_features),
        "feature_groups": {
            "clean_base": base_features,
            "saved_exp264_compact": parent_features,
            "signed_residual_compact": signed_features,
        },
        "saved_control_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": "stage_d_complete_promotion_passed"
        if promotion["passed"]
        else "stage_d_complete_guard_failed",
        "cost_contract": cost,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "feature_counts": cost["feature_counts"],
        "scientific_support_gate": scientific,
        "train_side_promotion_gate": promotion,
        "model_count": len(model_rows),
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "cost_contract": cost,
        "saved_exp264_stage_c": {
            key: value for key, value in parent_evidence.items() if key != "partitions"
        },
        "stage_s_input": {
            key: value
            for key, value in stage_s_evidence.items()
            if key not in {"partitions", "features"}
        },
        "clean_base_input": base_evidence,
        "saved_exp264_stage_d": saved_parent_evidence,
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": hidden_sha,
        },
        "artifact_sha256": artifact_sha,
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "oof_prediction_sha256": artifact_sha["oof"],
        "gpu_bitwise_deterministic_claimed": False,
        "submission_generated": False,
        "scientific_support_gate": scientific,
        "train_side_promotion_gate": promotion,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    return metrics


__all__ = [
    "ParquetBatchCursor",
    "SIGNED_PREDICTION_COLUMN",
    "SIGNED_TARGET_COLUMN",
    "add_signed_residual_labels",
    "build_signed_compact_meta",
    "evaluate_signed_residual_gate",
    "parent_compact_columns",
    "resolve_exp263_cache_root",
    "resolve_saved_exp264_stage_c_root",
    "run_stage_s",
    "run_stage_s_preflight",
    "run_stage_d",
    "signed_compact_feature_names",
    "signed_compact_schema",
    "stage_s_cost_contract",
    "stage_d_cost_contract",
    "stage_d_retained_base_columns",
    "evaluate_stage_d_guards",
    "load_saved_exp264_stage_d",
    "load_signed_compact_fold",
    "verify_signed_stage_s_root",
    "verify_saved_exp264_stage_c_root",
    "verify_stage_a_feature_contract",
]
