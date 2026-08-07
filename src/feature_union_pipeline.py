from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    build_stage_d_exp218_surface,
    load_stage_d_compact_fold,
    sha256_file,
    sha256_json,
    write_json,
)
from src.fold_safe_formation_pipeline import (
    canonical_formation_feature_names,
    logical_feature_content_sha256,
)
from src.signed_residual_meta import load_signed_compact_fold

VARIANT = "formation74_signed23_union_addonly"
CONTROL_NAMES = ("exp264", "exp287", "exp335")
ADDED_GROUPS = ("fold_safe_formation", "signed_residual_compact")


def _rmse(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> float:
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def union_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze exp372 at one union variant and exactly 15 new GPU boosters."""

    if str(config["experiment"]["route"]) != "ml_model":
        raise ValueError("exp372 must remain on the ml_model route")
    stage = dict(config["model"]["downstream_tvt"])
    execution = dict(stage["execution_count"])
    variants = [str(value) for value in stage["active_variants"]]
    config_indices = [int(value) for value in stage["lightgbm_config_indices"]]
    folds = int(execution["folds"])
    calculated = len(variants) * len(config_indices) * folds
    if variants != [VARIANT]:
        raise ValueError(f"exp372 must contain only {VARIANT}")
    if config_indices != [0, 1, 2]:
        raise ValueError("exp372 must reuse LightGBM configs 0, 1, and 2")
    if (
        int(execution["variants"]) != 1
        or int(execution["lightgbm_configs"]) != 3
        or folds != 5
        or calculated != 15
        or int(execution["planned_gpu_boosters"]) != 15
    ):
        raise ValueError("exp372 must be 1 variant x 3 configs x 5 folds = 15 boosters")
    zero_fields = (
        "parent_control_retraining_boosters",
        "exp287_standalone_retraining_boosters",
        "exp335_standalone_retraining_boosters",
        "selector_retraining_boosters",
        "formation_generation_runs",
        "signed_generation_runs",
    )
    for field in zero_fields:
        if int(execution[field]) != 0:
            raise ValueError(f"exp372 forbidden execution count is nonzero: {field}")
    expected_counts = {
        "expected_source_base_feature_count": 380,
        "expected_base_feature_count": 273,
        "saved_compact_feature_count": 74,
        "formation_feature_count": 74,
        "signed_feature_count": 23,
        "final_feature_count": 444,
    }
    for field, expected in expected_counts.items():
        if int(stage[field]) != expected:
            raise ValueError(f"exp372 {field} changed: {stage[field]} != {expected}")
    return {
        "active_variants": variants,
        "lightgbm_config_indices": config_indices,
        "folds": folds,
        "planned_gpu_boosters": calculated,
        "parent_control_retraining_boosters": 0,
        "standalone_parent_retraining_boosters": 0,
        "selector_retraining_boosters": 0,
        "feature_generation_runs": 0,
        "feature_counts": {
            "clean_base": 273,
            "saved_exp264_compact": 74,
            "fold_safe_formation": 74,
            "signed_residual_compact": 23,
            "final": 444,
        },
        "runtime": "kaggle_t4",
    }


def load_clean_feature_contract(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int = 273,
) -> tuple[list[str], dict[str, Any]]:
    """Read the frozen clean-feature allowlist without opening target/error data."""

    path = Path(path)
    observed = sha256_file(path)
    if observed != str(expected_sha256):
        raise ValueError(f"clean feature allowlist SHA mismatch: {observed}")
    frame = pd.read_csv(path)
    if "feature" not in frame:
        raise ValueError("clean feature allowlist has no feature column")
    features = frame["feature"].astype(str).tolist()
    if len(features) != int(expected_count) or len(set(features)) != len(features):
        raise ValueError("clean feature allowlist must contain 273 ordered unique features")
    return features, {
        "path": str(path),
        "sha256": observed,
        "feature_count": len(features),
        "feature_schema_sha256": sha256_json(features),
    }


def freeze_union_feature_schema(
    *,
    clean_features: Sequence[str],
    parent_features: Sequence[str],
    formation_features: Sequence[str],
    signed_features: Sequence[str],
    forbidden_columns: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Freeze the 444-column order before any target or error is opened."""

    groups = {
        "clean_base": [str(value) for value in clean_features],
        "saved_exp264_compact": [str(value) for value in parent_features],
        "fold_safe_formation": [str(value) for value in formation_features],
        "signed_residual_compact": [str(value) for value in signed_features],
    }
    expected = {
        "clean_base": 273,
        "saved_exp264_compact": 74,
        "fold_safe_formation": 74,
        "signed_residual_compact": 23,
    }
    for name, count in expected.items():
        if len(groups[name]) != count or len(set(groups[name])) != count:
            raise ValueError(f"exp372 {name} schema must contain {count} unique features")
    if groups["fold_safe_formation"] != canonical_formation_feature_names():
        raise ValueError("exp372 formation schema differs from the canonical exp287 order")
    final = [
        *groups["clean_base"],
        *groups["saved_exp264_compact"],
        *groups["fold_safe_formation"],
        *groups["signed_residual_compact"],
    ]
    if len(final) != 444 or len(set(final)) != 444:
        collisions = sorted(
            {feature for feature in final if final.count(feature) > 1}
        )
        raise ValueError(f"exp372 final schema is not 444-column unique: {collisions[:30]}")
    forbidden = set(str(value) for value in forbidden_columns)
    leaked = sorted(forbidden.intersection(final))
    if leaked:
        raise ValueError(f"exp372 final schema contains forbidden columns: {leaked}")
    contract = {
        "schema_version": "1.0.0",
        "variant": VARIANT,
        "frozen_order": list(groups),
        "feature_counts": {name: len(values) for name, values in groups.items()},
        "feature_groups": groups,
        "features": final,
        "feature_schema_sha256": sha256_json(final),
        "truth_or_error_loaded_before_schema_freeze": 0,
    }
    return final, contract


def _validate_partition_layout(
    partitions: Sequence[Mapping[str, Any]],
    *,
    expected_partitions: int,
    expected_rows: int,
    split_by_source_fold: bool,
) -> None:
    if len(partitions) != int(expected_partitions):
        raise ValueError("saved feature partition count mismatch")
    if sum(int(item["rows"]) for item in partitions) != int(expected_rows):
        raise ValueError("saved feature partition row inventory mismatch")
    seen: set[tuple[Any, ...]] = set()
    for item in partitions:
        fold = int(item["downstream_outer_fold"])
        role = str(item["role"])
        if fold not in range(5) or role not in {"train", "valid"}:
            raise ValueError(f"invalid saved feature fold/role: {(fold, role)}")
        if split_by_source_fold:
            source = int(item["source_outer_fold"])
            key = (fold, role, source)
            if (role == "valid") != (source == fold):
                raise ValueError(f"invalid nested source fold role: {key}")
        else:
            key = (fold, role)
        if key in seen:
            raise ValueError(f"duplicate saved feature partition: {key}")
        seen.add(key)
    for fold in range(5):
        if split_by_source_fold:
            train = [key for key in seen if key[0] == fold and key[1] == "train"]
            valid = [key for key in seen if key[0] == fold and key[1] == "valid"]
            if len(train) != 4 or len(valid) != 1:
                raise ValueError(f"nested fold inventory mismatch for fold={fold}")
        elif (fold, "train") not in seen or (fold, "valid") not in seen:
            raise ValueError(f"formation fold inventory mismatch for fold={fold}")


def verify_parent_compact_root(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_partition_sha: bool,
) -> dict[str, Any]:
    """Verify corrected exp264 Stage C metadata and all 25 saved74 partitions."""

    root = Path(root)
    data = dict(config["data"])
    files = {
        "metrics": root / "nested_selector_metrics.json",
        "model_manifest": root / "nested_selector_model_manifest.json",
        "manifest": root / "nested_compact_manifest.json",
        "schema": root / "compact_meta_schema.json",
    }
    expected = {
        "metrics": data["exp264_nested_selector_metrics_sha256"],
        "model_manifest": data["exp264_nested_selector_model_manifest_sha256"],
        "manifest": data["exp264_nested_compact_manifest_sha256"],
        "schema": data["exp264_compact_schema_file_sha256"],
    }
    observed_files: dict[str, str] = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != str(expected[name]):
            raise ValueError(f"exp264 {name} SHA mismatch: {observed}")
        observed_files[name] = observed
    metrics = json.loads(files["metrics"].read_text())
    model_manifest = json.loads(files["model_manifest"].read_text())
    if not bool(metrics.get("score_guard", {}).get("passed", False)):
        raise ValueError("saved exp264 score guard did not pass")
    if not bool(metrics.get("leakage_audit", {}).get("passed", False)):
        raise ValueError("saved exp264 leakage audit did not pass")
    if int(metrics.get("model_count", -1)) != 40 or int(
        model_manifest.get("model_count", -1)
    ) != 40:
        raise ValueError("saved exp264 selector model inventory changed")
    schema = json.loads(files["schema"].read_text())
    features = [str(value) for value in schema.get("features", [])]
    logical_schema_sha = str(schema.get("compact_meta_schema_sha256", ""))
    if (
        len(features) != 74
        or len(set(features)) != 74
        or logical_schema_sha != str(data["exp264_compact_schema_logical_sha256"])
    ):
        raise ValueError("saved exp264 compact schema contract changed")
    manifest = json.loads(files["manifest"].read_text())
    if str(manifest.get("compact_meta_schema_sha256", "")) != logical_schema_sha:
        raise ValueError("saved exp264 manifest/schema disagree")
    partitions = [dict(item) for item in manifest.get("partitions", [])]
    _validate_partition_layout(
        partitions,
        expected_partitions=int(data["expected_parent_compact_partitions"]),
        expected_rows=int(data["expected_rows_across_each_five_fold_surface"]),
        split_by_source_fold=True,
    )
    resolved: list[dict[str, Any]] = []
    for item in partitions:
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        observed = sha256_file(path) if verify_partition_sha else str(item["sha256"])
        if observed != str(item["sha256"]):
            raise ValueError(f"saved exp264 partition SHA mismatch: {path}")
        resolved.append({**item, "path": str(path), "sha256": observed})
    return {
        "root": str(root),
        "file_sha256": observed_files,
        "features": features,
        "feature_count": len(features),
        "feature_schema_sha256": logical_schema_sha,
        "partitions": resolved,
        "partition_count": len(resolved),
        "partition_sha_verified": bool(verify_partition_sha),
    }


def verify_signed_compact_root(
    root: Path,
    schema_path: Path,
    config: Mapping[str, Any],
    *,
    parent_evidence: Mapping[str, Any],
    verify_partition_sha: bool,
) -> dict[str, Any]:
    """Verify exp335 compact metadata and all strict-nested signed23 partitions."""

    root = Path(root)
    schema_path = Path(schema_path)
    data = dict(config["data"])
    manifest_path = root / str(data["exp335_signed_compact_manifest"])
    reproducibility_path = root / "reproducibility_manifest.json"
    files = {
        "manifest": (
            manifest_path,
            str(data["exp335_signed_compact_manifest_sha256"]),
        ),
        "schema": (
            schema_path,
            str(data["exp335_signed_compact_schema_file_sha256"]),
        ),
        "reproducibility": (
            reproducibility_path,
            str(data["exp335_stage_s_reproducibility_manifest_sha256"]),
        ),
    }
    observed_files: dict[str, str] = {}
    for name, (path, expected) in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"exp335 {name} SHA mismatch: {observed}")
        observed_files[name] = observed
    schema = json.loads(schema_path.read_text())
    features = [str(value) for value in schema.get("features", [])]
    logical_schema_sha = str(schema.get("signed_compact_schema_sha256", ""))
    if (
        len(features) != 23
        or len(set(features)) != 23
        or logical_schema_sha != str(data["exp335_signed_compact_schema_logical_sha256"])
    ):
        raise ValueError("saved exp335 signed schema contract changed")
    manifest = json.loads(manifest_path.read_text())
    if str(manifest.get("signed_compact_schema_sha256", "")) != logical_schema_sha:
        raise ValueError("saved exp335 manifest/schema disagree")
    if str(manifest.get("saved_exp264_compact_manifest_sha256", "")) != str(
        data["exp264_nested_compact_manifest_sha256"]
    ):
        raise ValueError("saved exp335 parent manifest lineage changed")
    partitions = [dict(item) for item in manifest.get("partitions", [])]
    _validate_partition_layout(
        partitions,
        expected_partitions=int(data["expected_signed_partitions"]),
        expected_rows=int(data["expected_rows_across_each_five_fold_surface"]),
        split_by_source_fold=True,
    )
    parent_index = {
        (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        ): str(item["sha256"])
        for item in parent_evidence["partitions"]
    }
    resolved: list[dict[str, Any]] = []
    for item in partitions:
        key = (
            int(item["downstream_outer_fold"]),
            str(item["role"]),
            int(item["source_outer_fold"]),
        )
        if str(item["saved_exp264_partition_sha256"]) != parent_index.get(key):
            raise ValueError(f"exp335 parent partition lineage mismatch: {key}")
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        observed = sha256_file(path) if verify_partition_sha else str(item["sha256"])
        if observed != str(item["sha256"]):
            raise ValueError(f"saved exp335 partition SHA mismatch: {path}")
        resolved.append({**item, "path": str(path), "sha256": observed})
    return {
        "root": str(root),
        "file_sha256": observed_files,
        "features": features,
        "feature_count": len(features),
        "feature_schema_sha256": logical_schema_sha,
        "partitions": resolved,
        "partition_count": len(resolved),
        "partition_sha_verified": bool(verify_partition_sha),
    }


def verify_formation_root(
    root: Path,
    config: Mapping[str, Any],
    *,
    verify_partition_sha: bool,
    verify_logical_content_sha: bool,
) -> dict[str, Any]:
    """Verify exp287 manifest plus all 10 saved formation74 partitions."""

    root = Path(root)
    data = dict(config["data"])
    manifest_path = root / str(data["exp287_formation_manifest"])
    relationship_path = root / "formation_feature_relationship_audit.csv"
    if not manifest_path.is_file() or not relationship_path.is_file():
        raise FileNotFoundError(f"exp287 formation contract missing under {root}")
    manifest_sha = sha256_file(manifest_path)
    relationship_sha = sha256_file(relationship_path)
    if manifest_sha != str(data["exp287_formation_manifest_sha256"]):
        raise ValueError(f"exp287 formation manifest SHA mismatch: {manifest_sha}")
    if relationship_sha != str(data["exp287_formation_relationship_audit_sha256"]):
        raise ValueError(f"exp287 relationship audit SHA mismatch: {relationship_sha}")
    manifest = json.loads(manifest_path.read_text())
    features = canonical_formation_feature_names()
    feature_schema_sha = sha256_json(features)
    if (
        int(manifest.get("feature_count", -1)) != 74
        or str(manifest.get("feature_schema_sha256", "")) != feature_schema_sha
        or str(manifest.get("relationship_audit_sha256", "")) != relationship_sha
    ):
        raise ValueError("saved exp287 formation manifest contract changed")
    partitions = [dict(item) for item in manifest.get("partitions", [])]
    _validate_partition_layout(
        partitions,
        expected_partitions=int(data["expected_formation_partitions"]),
        expected_rows=int(data["expected_rows_across_each_five_fold_surface"]),
        split_by_source_fold=False,
    )
    resolved: list[dict[str, Any]] = []
    for item in partitions:
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        observed_file = (
            sha256_file(path) if verify_partition_sha else str(item["file_sha256"])
        )
        if observed_file != str(item["file_sha256"]):
            raise ValueError(f"saved exp287 formation partition SHA mismatch: {path}")
        observed_logical = str(item["logical_content_sha256"])
        if verify_logical_content_sha:
            frame = pd.read_parquet(path, columns=["id", "well", *features])
            if len(frame) != int(item["rows"]):
                raise ValueError(f"saved exp287 formation row mismatch: {path}")
            observed_logical = logical_feature_content_sha256(frame, features)
            if observed_logical != str(item["logical_content_sha256"]):
                raise ValueError(f"saved exp287 logical content SHA mismatch: {path}")
        resolved.append(
            {
                **item,
                "path": str(path),
                "file_sha256": observed_file,
                "logical_content_sha256": observed_logical,
            }
        )
    return {
        "root": str(root),
        "manifest_sha256": manifest_sha,
        "relationship_audit_sha256": relationship_sha,
        "features": features,
        "feature_count": len(features),
        "feature_schema_sha256": feature_schema_sha,
        "partitions": resolved,
        "partition_count": len(resolved),
        "partition_sha_verified": bool(verify_partition_sha),
        "logical_content_sha_verified": bool(verify_logical_content_sha),
    }


def load_parent_compact_fold(
    evidence: Mapping[str, Any],
    *,
    downstream_outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader_evidence = dict(evidence)
    compact_features = [str(value) for value in evidence.get("features", [])]
    if len(compact_features) != 74 or len(set(compact_features)) != 74:
        raise ValueError("verified exp264 evidence must contain 74 unique features")
    loader_evidence["compact_features"] = compact_features
    return load_stage_d_compact_fold(
        stage_c_root=Path(evidence["root"]),
        stage_c_evidence=loader_evidence,
        downstream_outer_fold=int(downstream_outer_fold),
    )


def load_formation_fold(
    evidence: Mapping[str, Any],
    *,
    downstream_outer_fold: int,
    parent_train: pd.DataFrame,
    parent_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [str(value) for value in evidence["features"]]
    by_role: dict[str, pd.DataFrame] = {}
    selected = [
        item
        for item in evidence["partitions"]
        if int(item["downstream_outer_fold"]) == int(downstream_outer_fold)
    ]
    for item in selected:
        role = str(item["role"])
        frame = pd.read_parquet(Path(item["path"]), columns=["id", "well", *features])
        if len(frame) != int(item["rows"]):
            raise ValueError(f"formation partition row mismatch: {item['path']}")
        by_role[role] = frame
    if set(by_role) != {"train", "valid"}:
        raise ValueError("formation fold must contain exactly train and valid roles")
    for role, parent in (("train", parent_train), ("valid", parent_valid)):
        formation = by_role[role]
        if not formation[["id", "well"]].reset_index(drop=True).equals(
            parent[["id", "well"]].reset_index(drop=True)
        ):
            raise ValueError(f"saved74 and formation74 key alignment mismatch: {role}")
        if not np.isfinite(formation[features].to_numpy(np.float32, copy=False)).all():
            raise ValueError(f"formation74 contains non-finite values: {role}")
    return by_role["train"], by_role["valid"]


def validate_role_alignment(
    *,
    role: str,
    parent: pd.DataFrame,
    formation: pd.DataFrame,
    signed: pd.DataFrame,
) -> None:
    if not parent[list(KEY_COLUMNS)].reset_index(drop=True).equals(
        signed[list(KEY_COLUMNS)].reset_index(drop=True)
    ):
        raise ValueError(f"saved74 and signed23 key alignment mismatch: {role}")
    if not parent[["id", "well"]].reset_index(drop=True).equals(
        formation[["id", "well"]].reset_index(drop=True)
    ):
        raise ValueError(f"saved74 and formation74 key alignment mismatch: {role}")
    anchor_delta = np.abs(
        parent["last_known_tvt"].to_numpy(np.float32)
        - signed["last_known_tvt"].to_numpy(np.float32)
    )
    if float(anchor_delta.max(initial=0.0)) > 1.0e-4:
        raise ValueError(f"saved74 and signed23 anchor alignment mismatch: {role}")


def assemble_union_matrix(
    *,
    base_frame: pd.DataFrame,
    base_index: pd.Index,
    base_features: Sequence[str],
    parent: pd.DataFrame,
    parent_features: Sequence[str],
    formation: pd.DataFrame,
    formation_features: Sequence[str],
    signed: pd.DataFrame,
    signed_features: Sequence[str],
    chunk_columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one role in frozen group order while bounding temporary copies."""

    indices = base_index.get_indexer(parent["id"].astype(str))
    if np.any(indices < 0):
        raise ValueError("saved feature ids are absent from clean273 base")
    groups = [
        list(map(str, base_features)),
        list(map(str, parent_features)),
        list(map(str, formation_features)),
        list(map(str, signed_features)),
    ]
    feature_count = sum(len(group) for group in groups)
    if feature_count != 444:
        raise ValueError("union matrix feature count changed")
    values = np.empty((len(indices), feature_count), dtype=np.float32)
    for start in range(0, len(groups[0]), int(chunk_columns)):
        stop = min(start + int(chunk_columns), len(groups[0]))
        columns = groups[0][start:stop]
        values[:, start:stop] = base_frame[columns].iloc[indices].to_numpy(
            np.float32, copy=True
        )
    parent_start = len(groups[0])
    formation_start = parent_start + len(groups[1])
    signed_start = formation_start + len(groups[2])
    values[:, parent_start:formation_start] = parent[groups[1]].to_numpy(
        np.float32, copy=False
    )
    values[:, formation_start:signed_start] = formation[groups[2]].to_numpy(
        np.float32, copy=False
    )
    values[:, signed_start:] = signed[groups[3]].to_numpy(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("exp372 444-feature matrix contains non-finite values before fit")
    return indices, values


def audit_added_feature_relationships(
    *,
    train_values: np.ndarray,
    valid_values: np.ndarray,
    feature_contract: Mapping[str, Any],
    sample_rows: int,
) -> pd.DataFrame:
    """Report added-family duplicates/correlation without pruning or selection."""

    features = [str(value) for value in feature_contract["features"]]
    groups = {
        str(group): [str(value) for value in values]
        for group, values in feature_contract["feature_groups"].items()
    }
    group_by_feature = {
        feature: group for group, values in groups.items() for feature in values
    }
    added = [*groups["fold_safe_formation"], *groups["signed_residual_compact"]]
    total_rows = len(train_values) + len(valid_values)
    chosen = np.linspace(
        0,
        max(total_rows - 1, 0),
        num=min(int(sample_rows), total_rows),
        dtype=np.int64,
    )
    train_mask = chosen < len(train_values)
    sample = np.empty((len(chosen), len(features)), dtype=np.float32)
    sample[train_mask] = train_values[chosen[train_mask]]
    sample[~train_mask] = valid_values[chosen[~train_mask] - len(train_values)]
    means = sample.mean(axis=0, dtype=np.float64)
    centered = sample.astype(np.float64) - means
    scales = np.sqrt(np.mean(centered * centered, axis=0))
    nonconstant = scales > 0.0
    normalized = np.zeros_like(sample, dtype=np.float32)
    normalized[:, nonconstant] = (
        centered[:, nonconstant] / scales[nonconstant]
    ).astype(np.float32)
    index_by_feature = {feature: index for index, feature in enumerate(features)}
    added_indices = np.asarray([index_by_feature[feature] for feature in added], dtype=int)
    correlations = (
        normalized[:, added_indices].T @ normalized / max(len(sample), 1)
    ).astype(np.float32)
    sample_hashes = [
        hashlib.sha256(
            np.ascontiguousarray(sample[:, index], dtype="<f4").tobytes()
        ).hexdigest()
        for index in range(len(features))
    ]
    rows: list[dict[str, Any]] = []
    for added_position, feature in enumerate(added):
        feature_index = index_by_feature[feature]
        correlation = np.abs(correlations[added_position].astype(np.float64))
        correlation[feature_index] = -1.0
        related_index = int(np.argmax(correlation))
        duplicate_candidates = [
            index
            for index, digest in enumerate(sample_hashes)
            if index != feature_index and digest == sample_hashes[feature_index]
        ]
        exact_matches: list[str] = []
        for index in duplicate_candidates:
            if np.array_equal(
                train_values[:, feature_index], train_values[:, index]
            ) and np.array_equal(valid_values[:, feature_index], valid_values[:, index]):
                exact_matches.append(features[index])
        rows.append(
            {
                "feature": feature,
                "feature_group": group_by_feature[feature],
                "sample_rows": len(sample),
                "constant_on_sample": not bool(nonconstant[feature_index]),
                "exact_duplicate_count": len(exact_matches),
                "exact_duplicate_features": json.dumps(exact_matches, sort_keys=True),
                "max_abs_pearson": float(max(correlation[related_index], 0.0)),
                "max_abs_pearson_feature": features[related_index],
                "max_abs_pearson_feature_group": group_by_feature[features[related_index]],
                "policy": "report_only_no_pruning",
            }
        )
    return pd.DataFrame(rows)


def load_saved_controls(
    *,
    base_frame: pd.DataFrame,
    control_paths: Mapping[str, Path],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and align the three frozen OOF controls without retraining."""

    controls_cfg = dict(config["validation"]["saved_controls"])
    base_ids = base_frame["id"].astype(str)
    if base_ids.duplicated().any():
        raise ValueError("clean273 base ids are duplicated")
    truth = (
        base_frame["last_known_tvt"].to_numpy(np.float32)
        + base_frame["target"].to_numpy(np.float32)
    ).astype(np.float32)
    merged = pd.DataFrame(
        {
            "id": base_ids,
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
        }
    )
    evidence: dict[str, Any] = {}
    reference_fold: np.ndarray | None = None
    for name in CONTROL_NAMES:
        path = Path(control_paths[name])
        control = dict(controls_cfg[name])
        observed_sha = sha256_file(path)
        if observed_sha != str(control["oof_sha256"]):
            raise ValueError(f"{name} OOF SHA mismatch: {observed_sha}")
        columns = [
            "id",
            "well",
            "outer_fold",
            "actual_tvt",
            str(control["prediction_column"]),
        ]
        if name == "exp264":
            columns.append(str(control["clean_prediction_column"]))
        frame = pd.read_parquet(path, columns=columns)
        if frame["id"].astype(str).duplicated().any():
            raise ValueError(f"{name} OOF ids are duplicated")
        indexed = frame.set_index(frame["id"].astype(str), drop=False)
        if set(indexed.index) != set(base_ids):
            raise ValueError(f"{name} OOF ids differ from clean273 base")
        aligned = indexed.loc[base_ids].reset_index(drop=True)
        if not aligned["well"].astype(str).equals(
            base_frame["well"].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"{name} OOF well alignment mismatch")
        if float(
            np.max(
                np.abs(
                    aligned["actual_tvt"].to_numpy(np.float32)
                    - truth
                )
            )
        ) > 1.0e-4:
            raise ValueError(f"{name} OOF truth differs from clean273 base")
        fold = aligned["outer_fold"].to_numpy(np.int8)
        if reference_fold is None:
            reference_fold = fold
            merged["outer_fold"] = fold
        elif not np.array_equal(reference_fold, fold):
            raise ValueError(f"{name} OOF fold assignment differs from exp264")
        prediction = aligned[str(control["prediction_column"])].to_numpy(np.float32)
        observed_rmse = _rmse(truth, prediction)
        if abs(observed_rmse - float(control["rmse"])) > 1.0e-7:
            raise ValueError(
                f"{name} frozen OOF RMSE mismatch: {observed_rmse} != {control['rmse']}"
            )
        merged[name] = prediction
        if name == "exp264":
            merged["clean273"] = aligned[
                str(control["clean_prediction_column"])
            ].to_numpy(np.float32)
        evidence[name] = {
            "path": str(path),
            "sha256": observed_sha,
            "rows": len(aligned),
            "wells": int(aligned["well"].nunique()),
            "rmse": observed_rmse,
            "models_retrained": 0,
        }
    if len(merged) != int(config["validation"]["expected_rows"]) or int(
        merged["well"].nunique()
    ) != int(config["validation"]["expected_wells"]):
        raise ValueError("saved control row/well inventory mismatch")
    return merged, evidence


def evaluate_union_guards(
    *,
    config: Mapping[str, Any],
    base_frame: pd.DataFrame,
    controls: pd.DataFrame,
    oof_fold: np.ndarray,
    new_prediction: np.ndarray,
    hidden_like_assignment_path: Path,
    importance: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Apply the frozen incremental-utility and tail-promotion AND gates."""

    truth = controls["actual_tvt"].to_numpy(np.float32)
    union = np.asarray(new_prediction, dtype=np.float32)
    if not np.array_equal(oof_fold, controls["outer_fold"].to_numpy(np.int8)):
        raise ValueError("union OOF fold assignment differs from frozen controls")
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mask = np.asarray(oof_fold) == fold
        row = {
            "outer_fold": fold,
            "rows": int(mask.sum()),
            "union_rmse": _rmse(truth[mask], union[mask]),
        }
        for name in CONTROL_NAMES:
            row[f"{name}_rmse"] = _rmse(truth[mask], controls.loc[mask, name])
        row["better_standalone_rmse"] = min(row["exp287_rmse"], row["exp335_rmse"])
        row["better_standalone"] = (
            "exp287" if row["exp287_rmse"] <= row["exp335_rmse"] else "exp335"
        )
        row["delta_union_minus_better_standalone"] = (
            row["union_rmse"] - row["better_standalone_rmse"]
        )
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    md_since = base_frame["md_since"].to_numpy(np.float32)
    bucket_masks = {
        "near_0_250": md_since <= 250.0,
        "mid_250_1000": (md_since > 250.0) & (md_since < 1000.0),
        "1000_plus": md_since >= 1000.0,
    }
    bucket_rows: list[dict[str, Any]] = []
    for scope, mask in bucket_masks.items():
        row = {
            "scope": scope,
            "rows": int(mask.sum()),
            "union_rmse": _rmse(truth[mask], union[mask]),
            "exp287_rmse": _rmse(truth[mask], controls.loc[mask, "exp287"]),
            "exp335_rmse": _rmse(truth[mask], controls.loc[mask, "exp335"]),
        }
        row["better_standalone_rmse"] = min(row["exp287_rmse"], row["exp335_rmse"])
        row["better_standalone"] = (
            "exp287" if row["exp287_rmse"] <= row["exp335_rmse"] else "exp335"
        )
        row["delta_union_minus_better_standalone"] = (
            row["union_rmse"] - row["better_standalone_rmse"]
        )
        bucket_rows.append(row)
    bucket_metrics = pd.DataFrame(bucket_rows)

    hidden_path = Path(hidden_like_assignment_path)
    hidden_sha = sha256_file(hidden_path)
    if hidden_sha != str(config["data"]["hidden_like_assignment_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    assignment = pd.read_csv(hidden_path, dtype={"well_id": str}).set_index("well_id")
    hidden_rows: list[dict[str, Any]] = []
    hidden_scopes = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    for scope, column in hidden_scopes.items():
        mask = base_frame["well"].astype(str).map(assignment[column]).eq("valid").to_numpy()
        if not np.any(mask):
            raise ValueError(f"hidden-like assignment has no valid rows for {column}")
        row = {
            "scope": scope,
            "assignment": column,
            "rows": int(mask.sum()),
            "wells": int(base_frame.loc[mask, "well"].nunique()),
            "union_rmse": _rmse(truth[mask], union[mask]),
            "exp287_rmse": _rmse(truth[mask], controls.loc[mask, "exp287"]),
            "exp335_rmse": _rmse(truth[mask], controls.loc[mask, "exp335"]),
        }
        row["better_standalone_rmse"] = min(row["exp287_rmse"], row["exp335_rmse"])
        row["better_standalone"] = (
            "exp287" if row["exp287_rmse"] <= row["exp335_rmse"] else "exp335"
        )
        row["delta_union_minus_better_standalone"] = (
            row["union_rmse"] - row["better_standalone_rmse"]
        )
        hidden_rows.append(row)
    hidden_metrics = pd.DataFrame(hidden_rows)

    by_well_source = pd.DataFrame(
        {
            "well": base_frame["well"].astype(str),
            "actual_tvt": truth,
            "clean273": controls["clean273"].to_numpy(np.float32),
            "exp264": controls["exp264"].to_numpy(np.float32),
            "exp287": controls["exp287"].to_numpy(np.float32),
            "exp335": controls["exp335"].to_numpy(np.float32),
            "union": union,
        }
    )
    well_rows: list[dict[str, Any]] = []
    for well, group in by_well_source.groupby("well", sort=True):
        metrics = {
            name: _rmse(group["actual_tvt"], group[name])
            for name in ("clean273", *CONTROL_NAMES, "union")
        }
        well_rows.append(
            {
                "well": str(well),
                "rows": len(group),
                **{f"{name}_rmse": value for name, value in metrics.items()},
                "union_minus_exp264_delta": metrics["union"] - metrics["exp264"],
                "union_minus_clean273_delta": metrics["union"] - metrics["clean273"],
            }
        )
    by_well = pd.DataFrame(well_rows)

    gain = importance[importance["importance_type"].eq("gain")].copy()
    family_gain: dict[str, dict[str, Any]] = {}
    for group in ADDED_GROUPS:
        selected = gain[gain["feature_group"].eq(group)]
        gain_by_fold = selected.groupby("outer_fold")["importance"].sum().reindex(
            range(5), fill_value=0.0
        )
        family_gain[group] = {
            "total_gain": float(selected["importance"].sum()),
            "positive_gain_folds": int((gain_by_fold > 0.0).sum()),
            "gain_by_fold": {
                str(int(fold)): float(value) for fold, value in gain_by_fold.items()
            },
        }

    incremental_cfg = dict(config["guards"]["incremental_utility"])
    pooled_union = _rmse(truth, union)
    qualifying_folds = int(
        (
            fold_metrics["delta_union_minus_better_standalone"]
            <= float(incremental_cfg["maximum_fold_delta_vs_better_standalone_rmse"])
        ).sum()
    )
    scope_deltas = pd.concat(
        [
            bucket_metrics["delta_union_minus_better_standalone"],
            hidden_metrics["delta_union_minus_better_standalone"],
        ],
        ignore_index=True,
    )
    incremental_checks = {
        "pooled_union_at_or_below_maximum": pooled_union
        <= float(incremental_cfg["maximum_union_rmse"]),
        "minimum_qualifying_folds": qualifying_folds
        >= int(
            incremental_cfg[
                "minimum_folds_with_delta_le_0p02_vs_better_standalone"
            ]
        ),
        "all_fixed_scopes_nonworse": float(scope_deltas.max())
        <= float(incremental_cfg["maximum_scope_delta_vs_better_standalone_rmse"]),
        "formation_total_gain_positive": family_gain["fold_safe_formation"]["total_gain"]
        > 0.0,
        "signed_total_gain_positive": family_gain["signed_residual_compact"]["total_gain"]
        > 0.0,
        "formation_positive_gain_folds": family_gain["fold_safe_formation"][
            "positive_gain_folds"
        ]
        >= int(incremental_cfg["minimum_positive_gain_folds_per_added_family"]),
        "signed_positive_gain_folds": family_gain["signed_residual_compact"][
            "positive_gain_folds"
        ]
        >= int(incremental_cfg["minimum_positive_gain_folds_per_added_family"]),
    }
    incremental = {
        "checks": incremental_checks,
        "passed": bool(all(incremental_checks.values())),
        "pooled_union_rmse": pooled_union,
        "best_standalone_rmse": min(
            _rmse(truth, controls["exp287"]),
            _rmse(truth, controls["exp335"]),
        ),
        "qualifying_folds": qualifying_folds,
        "maximum_scope_delta": float(scope_deltas.max()),
        "family_gain": family_gain,
    }

    tail_cfg = dict(config["guards"]["tail_promotion"])
    p95_delta = float(by_well["union_minus_exp264_delta"].quantile(0.95))
    worst_delta = float(by_well["union_minus_exp264_delta"].max())
    thresholds = [float(value) for value in tail_cfg["clean273_worsened_well_thresholds_ft"]]
    maxima = dict(tail_cfg["maximum_worsened_well_counts_vs_clean273"])
    worsened_counts = {
        "plus_1_ft": int((by_well["union_minus_clean273_delta"] > thresholds[0]).sum()),
        "plus_3_ft": int((by_well["union_minus_clean273_delta"] > thresholds[1]).sum()),
        "plus_5_ft": int((by_well["union_minus_clean273_delta"] > thresholds[2]).sum()),
    }
    tail_checks = {
        "by_well_delta_p95_nonpositive": p95_delta
        <= float(tail_cfg["maximum_by_well_delta_p95_vs_exp264_rmse"]),
        "worst_well_delta_bounded": worst_delta
        <= float(tail_cfg["maximum_worst_well_delta_vs_exp264_rmse"]),
        "clean273_plus_1_count_bounded": worsened_counts["plus_1_ft"]
        <= int(maxima["plus_1_ft"]),
        "clean273_plus_3_count_bounded": worsened_counts["plus_3_ft"]
        <= int(maxima["plus_3_ft"]),
        "clean273_plus_5_count_bounded": worsened_counts["plus_5_ft"]
        <= int(maxima["plus_5_ft"]),
    }
    tail = {
        "checks": tail_checks,
        "passed": bool(all(tail_checks.values())),
        "by_well_delta_p95": p95_delta,
        "worst_well_delta": worst_delta,
        "worst_well": str(
            by_well.loc[by_well["union_minus_exp264_delta"].idxmax(), "well"]
        ),
        "worsened_well_counts_vs_clean273": worsened_counts,
    }
    return (
        incremental,
        tail,
        fold_metrics,
        bucket_metrics,
        hidden_metrics,
        by_well,
    )


def run_feature_union_train(
    *,
    config: Mapping[str, Any],
    parent_root: Path,
    formation_root: Path,
    signed_root: Path,
    signed_schema_path: Path,
    exp218_source_path: Path,
    exp218_config_path: Path,
    clean_allowlist_path: Path,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    control_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Train the approved 444-feature union without regenerating any parent feature."""

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    if not bool(config["execution"].get("implementation_approved", False)):
        raise RuntimeError("exp372 implementation approval is missing")
    if not bool(config["execution"].get("train_run_approved", False)):
        raise RuntimeError("exp372 train run approval is missing")
    if not bool(config["execution"].get("run_train", False)):
        raise RuntimeError("exp372 run_train is disabled")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cost = union_cost_contract(config)

    # Target/error remains unopened until all four feature schemas are frozen.
    parent_evidence = verify_parent_compact_root(
        parent_root, config, verify_partition_sha=True
    )
    signed_evidence = verify_signed_compact_root(
        signed_root,
        signed_schema_path,
        config,
        parent_evidence=parent_evidence,
        verify_partition_sha=True,
    )
    formation_evidence = verify_formation_root(
        formation_root,
        config,
        verify_partition_sha=True,
        verify_logical_content_sha=True,
    )
    clean_features, clean_evidence = load_clean_feature_contract(
        clean_allowlist_path,
        expected_sha256=str(config["data"]["clean_273_allowlist_sha256"]),
    )
    final_features, feature_contract = freeze_union_feature_schema(
        clean_features=clean_features,
        parent_features=parent_evidence["features"],
        formation_features=formation_evidence["features"],
        signed_features=signed_evidence["features"],
        forbidden_columns=config["features"]["forbidden_columns"],
    )
    write_json(output_dir / "feature_contract.json", feature_contract)

    base_frame, base_features, base_evidence, exp218, exp218_config = (
        build_stage_d_exp218_surface(
            exp218_source_path=exp218_source_path,
            exp218_config_path=exp218_config_path,
            base_feature_allowlist_path=clean_allowlist_path,
            raw_train_dir=raw_train_dir,
            config=config,
        )
    )
    if base_features != clean_features:
        raise ValueError("runtime clean273 order differs from the pre-fit frozen allowlist")
    retained = list(
        dict.fromkeys(["id", "well", "target", "last_known_tvt", "md_since", *base_features])
    )
    base_frame = base_frame.loc[:, ~base_frame.columns.duplicated()].loc[:, retained].copy()
    if (
        len(base_frame) != int(config["validation"]["expected_rows"])
        or int(base_frame["well"].nunique())
        != int(config["validation"]["expected_wells"])
    ):
        raise ValueError("clean273 base row/well inventory mismatch")
    controls, control_evidence = load_saved_controls(
        base_frame=base_frame,
        control_paths=control_paths,
        config=config,
    )
    hidden_sha = sha256_file(hidden_like_assignment_path)
    if hidden_sha != str(config["data"]["hidden_like_assignment_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")

    stage_cfg = dict(config["model"]["downstream_tvt"])
    mode = dict(exp218_config["model"]["training"]["modes"][str(stage_cfg["mode"])])
    if not bool(mode.get("use_gpu", False)):
        raise ValueError("exp372 must reuse the exp218 GPU mode")
    overrides = dict(mode.get("common_overrides", {}))
    expected_overrides = {
        "gpu_use_dp": True,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 8,
        "num_threads": 8,
    }
    if any(overrides.get(key) != value for key, value in expected_overrides.items()):
        raise ValueError("exp372 exp218 GPU reproducibility overrides changed")
    params_family = exp218.apply_mode_overrides(
        exp218.exp063_lgb_config_family(fast=False), mode
    )
    config_indices = [int(value) for value in cost["lightgbm_config_indices"]]
    params_family = [params_family[index] for index in config_indices]
    base_index = pd.Index(base_frame["id"].astype(str))
    if not base_index.is_unique:
        raise ValueError("clean273 base ids are not unique")
    prefit_fold = np.full(len(base_frame), -1, np.int8)
    for outer_fold in range(int(cost["folds"])):
        parent_train, parent_valid = load_parent_compact_fold(
            parent_evidence, downstream_outer_fold=outer_fold
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=signed_evidence,
            downstream_outer_fold=outer_fold,
        )
        formation_train, formation_valid = load_formation_fold(
            formation_evidence,
            downstream_outer_fold=outer_fold,
            parent_train=parent_train,
            parent_valid=parent_valid,
        )
        validate_role_alignment(
            role="train",
            parent=parent_train,
            formation=formation_train,
            signed=signed_train,
        )
        validate_role_alignment(
            role="valid",
            parent=parent_valid,
            formation=formation_valid,
            signed=signed_valid,
        )
        train_indices = base_index.get_indexer(parent_train["id"].astype(str))
        valid_indices = base_index.get_indexer(parent_valid["id"].astype(str))
        if np.any(train_indices < 0) or np.any(valid_indices < 0):
            raise ValueError("saved feature ids are absent from clean273 base")
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != len(
            base_frame
        ):
            raise ValueError("prefit fold does not cover clean273 rows exactly once")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("prefit train/valid indices overlap")
        if np.any(prefit_fold[valid_indices] >= 0):
            raise ValueError("prefit OOF valid rows were assigned twice")
        prefit_fold[valid_indices] = np.int8(outer_fold)
        del (
            parent_train,
            parent_valid,
            signed_train,
            signed_valid,
            formation_train,
            formation_valid,
        )
        gc.collect()
    if np.any(prefit_fold < 0) or not np.array_equal(
        prefit_fold, controls["outer_fold"].to_numpy(np.int8)
    ):
        raise ValueError("prefit saved feature folds differ from frozen OOF controls")

    target = base_frame["target"].to_numpy(np.float32)
    anchor = base_frame["last_known_tvt"].to_numpy(np.float32)
    truth = (anchor + target).astype(np.float32)
    n_rows = len(base_frame)
    oof_by_config = [np.full(n_rows, np.nan, np.float32) for _ in params_family]
    oof_fold = np.full(n_rows, -1, np.int8)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    fold_model_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    relationship_audit: pd.DataFrame | None = None
    group_by_feature = {
        feature: group
        for group, features in feature_contract["feature_groups"].items()
        for feature in features
    }

    for outer_fold in range(int(cost["folds"])):
        parent_train, parent_valid = load_parent_compact_fold(
            parent_evidence, downstream_outer_fold=outer_fold
        )
        signed_train, signed_valid = load_signed_compact_fold(
            stage_s_evidence=signed_evidence,
            downstream_outer_fold=outer_fold,
        )
        formation_train, formation_valid = load_formation_fold(
            formation_evidence,
            downstream_outer_fold=outer_fold,
            parent_train=parent_train,
            parent_valid=parent_valid,
        )
        validate_role_alignment(
            role="train",
            parent=parent_train,
            formation=formation_train,
            signed=signed_train,
        )
        validate_role_alignment(
            role="valid",
            parent=parent_valid,
            formation=formation_valid,
            signed=signed_valid,
        )
        train_indices, train_values = assemble_union_matrix(
            base_frame=base_frame,
            base_index=base_index,
            base_features=base_features,
            parent=parent_train,
            parent_features=parent_evidence["features"],
            formation=formation_train,
            formation_features=formation_evidence["features"],
            signed=signed_train,
            signed_features=signed_evidence["features"],
            chunk_columns=int(stage_cfg["matrix_copy_chunk_columns"]),
        )
        valid_indices, valid_values = assemble_union_matrix(
            base_frame=base_frame,
            base_index=base_index,
            base_features=base_features,
            parent=parent_valid,
            parent_features=parent_evidence["features"],
            formation=formation_valid,
            formation_features=formation_evidence["features"],
            signed=signed_valid,
            signed_features=signed_evidence["features"],
            chunk_columns=int(stage_cfg["matrix_copy_chunk_columns"]),
        )
        if len(np.unique(np.concatenate([train_indices, valid_indices]))) != n_rows:
            raise ValueError("exp372 fold does not cover clean273 rows exactly once")
        if np.intersect1d(train_indices, valid_indices).size:
            raise ValueError("exp372 train/valid indices overlap")
        if np.any(oof_fold[valid_indices] >= 0):
            raise ValueError("exp372 OOF valid rows were assigned twice")
        oof_fold[valid_indices] = np.int8(outer_fold)
        if outer_fold == 0:
            relationship_audit = audit_added_feature_relationships(
                train_values=train_values,
                valid_values=valid_values,
                feature_contract=feature_contract,
                sample_rows=int(stage_cfg["relationship_sample_rows"]),
            )
            relationship_audit.to_csv(
                output_dir / "feature_relationship_audit.csv", index=False
            )
        x_train = pd.DataFrame(train_values, columns=final_features, copy=False)
        x_valid = pd.DataFrame(valid_values, columns=final_features, copy=False)
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
            oof_by_config[family_position][valid_indices] = prediction
            fold_predictions.append(prediction)
            model_path = model_dir / f"lgb{config_index}__outer{outer_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iteration)
            rmse_value = _rmse(
                truth[valid_indices], anchor[valid_indices] + prediction
            )
            model_rows.append(
                {
                    "variant": VARIANT,
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
                values = model.booster_.feature_importance(
                    importance_type=importance_type
                )
                importance_rows.extend(
                    {
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "importance_type": importance_type,
                        "feature": feature,
                        "feature_group": group_by_feature[feature],
                        "importance": float(value),
                    }
                    for feature, value in zip(final_features, values, strict=True)
                )
            print(
                json.dumps(
                    {
                        "experiment": "exp372",
                        "variant": VARIANT,
                        "outer_fold": outer_fold,
                        "model": f"lgb{config_index}",
                        "rmse_tvt": rmse_value,
                        "best_iteration": best_iteration,
                        "completed_boosters": len(model_rows),
                        "planned_boosters": cost["planned_gpu_boosters"],
                        "control_retraining": 0,
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
                "rmse_tvt": _rmse(
                    truth[valid_indices], anchor[valid_indices] + fold_mean
                ),
                "best_iteration": None,
            }
        )
        del (
            parent_train,
            parent_valid,
            signed_train,
            signed_valid,
            formation_train,
            formation_valid,
            x_train,
            x_valid,
            train_values,
            valid_values,
            fold_predictions,
            fold_mean,
        )
        gc.collect()

    if len(model_rows) != 15 or np.any(oof_fold < 0):
        raise AssertionError("exp372 15-model OOF contract is incomplete")
    for prediction in oof_by_config:
        if not np.isfinite(prediction).all():
            raise AssertionError("exp372 OOF prediction is incomplete")
    mean_residual = np.mean(np.vstack(oof_by_config), axis=0).astype(np.float32)
    mean_prediction = (anchor + mean_residual).astype(np.float32)
    importance = pd.DataFrame(importance_rows)
    (
        incremental,
        tail,
        fold_metrics,
        bucket_metrics,
        hidden_metrics,
        by_well,
    ) = evaluate_union_guards(
        config=config,
        base_frame=base_frame,
        controls=controls,
        oof_fold=oof_fold,
        new_prediction=mean_prediction,
        hidden_like_assignment_path=hidden_like_assignment_path,
        importance=importance,
    )
    technical_checks = {
        "three_input_manifests_and_partition_sha_verified": True,
        "formation_logical_content_sha_verified": True,
        "expected_rows": len(base_frame) == int(config["validation"]["expected_rows"]),
        "expected_wells": int(base_frame["well"].nunique())
        == int(config["validation"]["expected_wells"]),
        "id_well_fold_role_alignment": True,
        "outer_train_valid_overlap_zero": True,
        "truth_or_error_loaded_before_schema_freeze_zero": feature_contract[
            "truth_or_error_loaded_before_schema_freeze"
        ]
        == 0,
        "feature_count_444": len(final_features) == 444,
        "feature_names_unique": len(set(final_features)) == 444,
        "model_matrix_finite": True,
        "fifteen_unique_model_slots": len(
            {(row["outer_fold"], row["config_index"]) for row in model_rows}
        )
        == 15,
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
    }
    promotion = {
        "technical_passed": technical["passed"],
        "incremental_utility_passed": incremental["passed"],
        "tail_promotion_passed": tail["passed"],
    }
    promotion["passed"] = bool(all(promotion.values()))

    prediction_frame = base_frame[
        ["id", "well", "md_since", "last_known_tvt", "target"]
    ].copy()
    prediction_frame["outer_fold"] = oof_fold
    prediction_frame["actual_tvt"] = truth
    for config_index, residual in zip(config_indices, oof_by_config, strict=True):
        prediction_frame[f"{VARIANT}__lgb{config_index}__pred_tvt"] = (
            anchor + residual
        ).astype(np.float32)
    prediction_frame[f"{VARIANT}__lgb_mean__pred_tvt"] = mean_prediction
    paths = {
        "feature_contract": output_dir / "feature_contract.json",
        "relationship_audit": output_dir / "feature_relationship_audit.csv",
        "oof": output_dir / "oof_predictions.parquet",
        "fold_metrics": output_dir / "fold_metrics.csv",
        "bucket_metrics": output_dir / "bucket_metrics.csv",
        "hidden_metrics": output_dir / "hidden_like_metrics.csv",
        "by_well": output_dir / "by_well_metrics.csv",
        "importance": output_dir / "feature_importance.csv",
        "model_manifest": output_dir / "model_manifest.json",
        "metrics": output_dir / "metrics.json",
    }
    if relationship_audit is None:
        raise AssertionError("exp372 relationship audit was not generated")
    prediction_frame.to_parquet(paths["oof"], index=False)
    pd.DataFrame(fold_model_rows).merge(
        fold_metrics, on="outer_fold", how="left"
    ).to_csv(paths["fold_metrics"], index=False)
    bucket_metrics.to_csv(paths["bucket_metrics"], index=False)
    hidden_metrics.to_csv(paths["hidden_metrics"], index=False)
    by_well.to_csv(paths["by_well"], index=False)
    importance.to_csv(paths["importance"], index=False)
    model_manifest = {
        "schema_version": "1.0.0",
        "status": "completed_15_gpu_boosters",
        "variant": VARIANT,
        "cost_contract": cost,
        "model_count": len(model_rows),
        "models": model_rows,
        "feature_count": len(final_features),
        "feature_schema_sha256": feature_contract["feature_schema_sha256"],
        "feature_groups": feature_contract["feature_groups"],
        "control_retraining_boosters": 0,
    }
    write_json(paths["model_manifest"], model_manifest)
    metrics = {
        "schema_version": "1.0.0",
        "status": (
            "train_complete_promoted"
            if promotion["passed"]
            else "train_complete_guard_failed_closed"
        ),
        "variant": VARIANT,
        "cost_contract": cost,
        "rows": n_rows,
        "wells": int(base_frame["well"].nunique()),
        "technical_gate": technical,
        "incremental_utility_gate": incremental,
        "tail_promotion_gate": tail,
        "promotion_gate": promotion,
        "model_count": len(model_rows),
    }
    write_json(paths["metrics"], metrics)
    artifact_sha = {name: sha256_file(path) for name, path in paths.items()}
    reproducibility = {
        "schema_version": "1.0.0",
        "status": metrics["status"],
        "cost_contract": cost,
        "feature_contract": feature_contract,
        "clean_base_input": base_evidence,
        "clean_allowlist_input": clean_evidence,
        "saved_exp264_compact": {
            key: value
            for key, value in parent_evidence.items()
            if key != "partitions"
        },
        "saved_exp287_formation": {
            key: value
            for key, value in formation_evidence.items()
            if key not in {"partitions", "features"}
        },
        "saved_exp335_signed": {
            key: value
            for key, value in signed_evidence.items()
            if key not in {"partitions", "features"}
        },
        "saved_controls": control_evidence,
        "hidden_like_assignment": {
            "path": str(hidden_like_assignment_path),
            "sha256": hidden_sha,
        },
        "artifact_sha256": artifact_sha,
        "model_manifest_sha256": artifact_sha["model_manifest"],
        "oof_prediction_sha256": artifact_sha["oof"],
        "gpu_bitwise_deterministic_claimed": False,
        "submission_generated": False,
        "promotion_gate": promotion,
    }
    write_json(output_dir / "reproducibility_manifest.json", reproducibility)
    metrics["artifact_sha256"] = artifact_sha
    metrics["reproducibility_manifest_sha256"] = sha256_file(
        output_dir / "reproducibility_manifest.json"
    )
    return metrics


__all__ = [
    "ADDED_GROUPS",
    "VARIANT",
    "assemble_union_matrix",
    "audit_added_feature_relationships",
    "evaluate_union_guards",
    "freeze_union_feature_schema",
    "load_clean_feature_contract",
    "load_formation_fold",
    "load_parent_compact_fold",
    "load_saved_controls",
    "run_feature_union_train",
    "union_cost_contract",
    "validate_role_alignment",
    "verify_formation_root",
    "verify_parent_compact_root",
    "verify_signed_compact_root",
]
