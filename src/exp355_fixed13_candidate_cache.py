from __future__ import annotations

import copy
import gzip
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    FoldBundle,
    candidate_ids,
    sha256_file,
    write_json,
)

BASE_CANDIDATE_IDS = (
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
BASE_PRIMARY_IDS = BASE_CANDIDATE_IDS[:11]
BASE_FIXED_IDS = (*BASE_CANDIDATE_IDS[:6], BASE_CANDIDATE_IDS[11])
ADDED_CANDIDATE_ID = "exp355_dip_rate_hmm"
EXP355_ALLOWLIST = ("well_id", "row_idx", "fold", "candidate_tvt")


def sha256_decompressed_gzip(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _expand_paths(patterns: Sequence[str], search_roots: Sequence[Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in patterns:
        path = Path(raw)
        if path.exists():
            found.add(path)
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            if Path(pattern).is_absolute():
                continue
            found.update(path for path in root.glob(pattern) if path.exists())
    return sorted(found)


def resolve_file_by_sha(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    expected_file_sha256: str,
    expected_decompressed_sha256: str | None = None,
    label: str,
) -> Path:
    candidates = [path for path in _expand_paths(patterns, search_roots) if path.is_file()]
    file_matches = [
        path for path in candidates if sha256_file(path) == str(expected_file_sha256)
    ]
    if expected_decompressed_sha256:
        file_matches = [
            path
            for path in file_matches
            if sha256_decompressed_gzip(path) == str(expected_decompressed_sha256)
        ]
    if len(file_matches) != 1:
        evidence = {
            str(path): {
                "file_sha256": sha256_file(path),
                "decompressed_sha256": (
                    sha256_decompressed_gzip(path)
                    if expected_decompressed_sha256 and path.suffix == ".gz"
                    else None
                ),
            }
            for path in candidates
        }
        raise FileNotFoundError(
            f"{label} did not resolve to exactly one SHA-matched file: {evidence}"
        )
    return file_matches[0]


def resolve_parent_score_file(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    expected_file_sha256: str,
) -> Path:
    return resolve_file_by_sha(
        patterns,
        search_roots,
        expected_file_sha256=expected_file_sha256,
        label="corrected exp264 Stage C outer-valid candidate score",
    )


def validate_fixed13_contract(contract: Mapping[str, Any]) -> None:
    ids = tuple(candidate_ids(contract))
    if ids != (*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID):
        raise ValueError(f"fixed13 candidate order mismatch: {ids}")
    if len(set(ids)) != 13:
        raise ValueError("fixed13 candidate IDs must be unique")
    specs = {str(item["id"]): item for item in contract["score_candidates"]}
    added = specs[ADDED_CANDIDATE_ID]
    if str(added.get("kind")) != "dip_rate_prior_exact_hmm":
        raise ValueError("exp355 candidate kind must remain dip_rate_prior_exact_hmm")
    if added.get("parents"):
        raise ValueError("exp355 candidate must not be reconstructed as a formula")
    domains = contract["legal_domains"]
    primary = tuple(domains["primitive_pair_bank"]["candidates"])
    fixed = tuple(domains["primitive_fixed_bank"]["candidates"])
    if primary != (*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID):
        raise ValueError("fixed13 primary domain differs from add-one contract")
    if fixed != BASE_FIXED_IDS:
        raise ValueError("fixed fallback domain changed from exp264")


def base_exp264_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_fixed13_contract(contract)
    base = copy.deepcopy(dict(contract))
    base["score_candidates"] = [
        item
        for item in base["score_candidates"]
        if str(item["id"]) != ADDED_CANDIDATE_ID
    ]
    base["legal_domains"]["primitive_pair_bank"]["candidates"] = list(BASE_PRIMARY_IDS)
    base["legal_domains"]["primitive_fixed_bank"]["candidates"] = list(BASE_FIXED_IDS)
    if "candidate_id_model_encoding" in base:
        base["candidate_id_model_encoding"]["width"] = 12
    return base


def exp355_prediction_content_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["fold", "well_id", "row_idx"], kind="stable"
    ).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("|".join(EXP355_ALLOWLIST).encode())
    for well, part in ordered.groupby("well_id", sort=True):
        digest.update(str(well).encode())
        digest.update(b"\0")
        digest.update(part["row_idx"].to_numpy(np.int64).astype("<i8", copy=False).tobytes())
        digest.update(
            part["fold"].to_numpy(np.int8).astype("<i1", copy=False).tobytes()
        )
        digest.update(
            part["candidate_tvt"]
            .to_numpy(np.float64)
            .astype("<f8", copy=False)
            .tobytes()
        )
    return digest.hexdigest()


def load_exp355_oof(
    path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_file_sha256: str,
    expected_decompressed_sha256: str,
    expected_prediction_logical_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    file_sha = sha256_file(path)
    decompressed_sha = sha256_decompressed_gzip(path)
    if file_sha != str(expected_file_sha256):
        raise ValueError(f"exp355 OOF file SHA mismatch: {file_sha}")
    if decompressed_sha != str(expected_decompressed_sha256):
        raise ValueError(f"exp355 OOF decompressed SHA mismatch: {decompressed_sha}")
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    missing = set(EXP355_ALLOWLIST) - set(header)
    if missing:
        raise ValueError(f"exp355 OOF allowlist columns are missing: {sorted(missing)}")
    frame = pd.read_csv(
        path,
        usecols=list(EXP355_ALLOWLIST),
        dtype={
            "well_id": str,
            "row_idx": np.int32,
            "fold": np.int8,
            "candidate_tvt": np.float64,
        },
    )
    if list(frame.columns) != list(EXP355_ALLOWLIST):
        frame = frame.loc[:, list(EXP355_ALLOWLIST)]
    if len(frame) != int(expected_rows):
        raise ValueError(f"exp355 OOF row mismatch: {len(frame)}")
    if frame["well_id"].nunique() != int(expected_wells):
        raise ValueError("exp355 OOF well count mismatch")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp355 OOF contains duplicate well/row keys")
    if set(frame["fold"].astype(int)) != set(range(5)):
        raise ValueError("exp355 OOF source-fold coverage mismatch")
    well_fold_counts = frame.groupby("well_id", sort=False)["fold"].nunique()
    if not well_fold_counts.eq(1).all():
        raise ValueError("each exp355 well must have exactly one source outer fold")
    prediction = frame["candidate_tvt"].to_numpy(np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("exp355 OOF contains non-finite predictions")
    content_sha = exp355_prediction_content_sha256(frame)
    source_fold_rows = (
        frame.groupby("fold", sort=True).size().astype(int).to_dict()
    )
    source_fold_wells = (
        frame[["well_id", "fold"]]
        .drop_duplicates()
        .groupby("fold", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "path": str(path),
        "header_columns": header,
        "loaded_columns": list(frame.columns),
        "truth_or_error_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "source_fold_role": "saved_exp226_oof_provenance_not_selector_feature",
        "source_fold_column": "fold",
        "source_fold_rows": {
            str(fold): int(source_fold_rows[fold]) for fold in range(5)
        },
        "source_fold_wells": {
            str(fold): int(source_fold_wells[fold]) for fold in range(5)
        },
        "each_well_has_one_source_fold": True,
        "file_sha256": file_sha,
        "decompressed_sha256": decompressed_sha,
        "upstream_prediction_logical_sha256": expected_prediction_logical_sha256,
        "post_read_prediction_content_sha256": content_sha,
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }
    return frame, manifest


class Exp355Fixed13CandidateCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        exp355_oof: pd.DataFrame,
        exp355_manifest: Mapping[str, Any],
    ):
        validate_fixed13_contract(contract)
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = {
            str(item["id"]): dict(item) for item in contract["score_candidates"]
        }
        self.base_cache = Exp263CandidateCache(root, base_exp264_contract(contract))
        self.exp355_manifest = dict(exp355_manifest)
        self.exp355_by_key = exp355_oof.sort_values(
            ["well_id", "row_idx"], kind="stable"
        ).set_index(["well_id", "row_idx"])[["fold", "candidate_tvt"]]
        if not self.exp355_by_key.index.is_unique:
            raise ValueError("exp355 global key index is not unique")
        self._selector_fold_audits: dict[int, dict[str, Any]] = {}

    def load_fold(self, fold: int) -> FoldBundle:
        base = self.base_cache.load_fold(int(fold))
        expected = base.base.sort_values(
            ["well", "well_row_idx"], kind="stable"
        ).reset_index()
        if not np.all(expected["outer_fold"].to_numpy(np.int8) == np.int8(fold)):
            raise ValueError(f"exp263 base fold identity mismatch in fold {fold}")
        selector_keys = pd.MultiIndex.from_arrays(
            [
                expected["well"].astype(str).to_numpy(),
                expected["well_row_idx"].to_numpy(np.int64),
            ],
            names=["well_id", "row_idx"],
        )
        added = self.exp355_by_key.reindex(selector_keys)
        missing = added["candidate_tvt"].isna() | added["fold"].isna()
        if missing.any():
            missing_keys = [
                (str(well), int(row))
                for well, row in selector_keys[missing.to_numpy()][:5]
            ]
            raise ValueError(
                f"exp355 global key join is missing exp263 fold {fold} rows: "
                f"{missing_keys}"
            )
        source_fold_sorted = added["fold"].to_numpy(np.int8)
        if not np.isin(source_fold_sorted, np.arange(5, dtype=np.int8)).all():
            raise ValueError(f"exp355 fold provenance is outside 0..4 in selector fold {fold}")
        source_well_frame = pd.DataFrame(
            {
                "well": expected["well"].astype(str).to_numpy(),
                "source_outer_fold": source_fold_sorted,
            }
        ).drop_duplicates()
        if source_well_frame.groupby("well")["source_outer_fold"].nunique().max() != 1:
            raise ValueError("exp355 source fold changed within a well after key join")
        source_row_counts = np.bincount(source_fold_sorted, minlength=5)
        source_well_counts = (
            source_well_frame.groupby("source_outer_fold", sort=True)
            .size()
            .reindex(range(5), fill_value=0)
            .to_numpy(np.int64)
        )
        self._selector_fold_audits[int(fold)] = {
            "selector_outer_fold": int(fold),
            "rows": len(expected),
            "wells": int(expected["well"].nunique()),
            "missing_key_rows": 0,
            "source_fold_row_counts": {
                str(source_fold): int(source_row_counts[source_fold])
                for source_fold in range(5)
            },
            "source_fold_well_counts": {
                str(source_fold): int(source_well_counts[source_fold])
                for source_fold in range(5)
            },
            "same_source_and_selector_fold_rows": int(
                np.sum(source_fold_sorted == np.int8(fold))
            ),
            "same_source_and_selector_fold_fraction": float(
                np.mean(source_fold_sorted == np.int8(fold))
            ),
        }
        inverse = np.empty(len(expected), dtype=np.int64)
        inverse[expected["index"].to_numpy(np.int64)] = np.arange(len(expected))
        prediction_sorted = added["candidate_tvt"].to_numpy(np.float32)
        prediction = prediction_sorted[inverse]
        if not np.isfinite(prediction).all():
            raise ValueError(f"exp355 fold {fold} contains a non-finite prediction")
        values = np.column_stack([base.values, prediction]).astype(np.float32)
        available = np.column_stack(
            [base.available, np.ones(len(base.base), dtype=bool)]
        ).astype(bool)
        confidence = dict(base.confidence)
        confidence[ADDED_CANDIDATE_ID] = base.base[KEY_COLUMNS].assign(
            confidence_valid=False,
            confidence_source="exp355_no_source_native_confidence",
        )
        return FoldBundle(
            base=base.base,
            values=values,
            available=available,
            confidence=confidence,
            candidate_ids=list(self.ids),
            specs=dict(self.specs),
        )

    def selector_repartition_manifest(self, *, expected_rows: int) -> dict[str, Any]:
        if set(self._selector_fold_audits) != set(range(5)):
            raise ValueError("all five exp263 selector folds must be audited before fit")
        selector_fold_rows = {
            str(fold): int(self._selector_fold_audits[fold]["rows"])
            for fold in range(5)
        }
        source_fold_rows = {
            str(source_fold): int(
                sum(
                    self._selector_fold_audits[selector_fold][
                        "source_fold_row_counts"
                    ][str(source_fold)]
                    for selector_fold in range(5)
                )
            )
            for source_fold in range(5)
        }
        expected_source_fold_rows = {
            str(fold): int(value)
            for fold, value in self.exp355_manifest["source_fold_rows"].items()
        }
        total_rows = int(sum(selector_fold_rows.values()))
        checks = {
            "all_selector_folds_audited": True,
            "global_key_join_rows_match": total_rows == int(expected_rows),
            "missing_key_rows_zero": all(
                int(self._selector_fold_audits[fold]["missing_key_rows"]) == 0
                for fold in range(5)
            ),
            "source_fold_row_counts_preserved": (
                source_fold_rows == expected_source_fold_rows
            ),
            "source_fold_not_used_as_model_feature": True,
        }
        return {
            "policy": "global_key_join_then_exp263_selector_fold_repartition",
            "selector_fold_source": "exp263_row_count_balanced_outer_fold",
            "candidate_source_fold": "exp355_saved_exp226_oof_fold",
            "candidate_is_oof_with_respect_to_its_own_well": True,
            "source_fold_equals_selector_fold_required": False,
            "source_fold_used_as_model_feature": False,
            "rows": total_rows,
            "selector_fold_rows": selector_fold_rows,
            "source_fold_rows": source_fold_rows,
            "overlap_by_selector_fold": [
                self._selector_fold_audits[fold] for fold in range(5)
            ],
            "checks": checks,
            "passed": bool(all(checks.values())),
        }


def _score_row_group_summary(
    frame: pd.DataFrame,
    *,
    all_candidate_ids: Sequence[str],
    primary_ids: Sequence[str],
    fixed_candidate: str,
) -> pd.DataFrame:
    expected_count = len(all_candidate_ids)
    counts = frame.groupby("id", sort=False)["candidate_id"].size()
    if not bool(counts.eq(expected_count).all()):
        raise ValueError("candidate score row group contains an incomplete candidate set")
    if not set(frame["candidate_id"].astype(str).unique()).issubset(set(all_candidate_ids)):
        raise ValueError("candidate score row group contains an unknown candidate")
    primary = frame[frame["candidate_id"].isin(primary_ids)]
    selected_idx = primary.groupby("id", sort=False)["pred_abs_error"].idxmin()
    selected = primary.loc[
        selected_idx,
        [
            "id",
            "well",
            "outer_fold",
            "md_since",
            "candidate_id",
            "actual_abs_error",
        ],
    ].copy()
    selected = selected.rename(
        columns={
            "candidate_id": "selected_candidate",
            "actual_abs_error": "selected_abs_error",
        }
    )
    fixed = frame.loc[
        frame["candidate_id"].eq(fixed_candidate), ["id", "actual_abs_error"]
    ].rename(columns={"actual_abs_error": "fixed_abs_error"})
    if len(fixed) != len(selected):
        raise ValueError("fixed fallback coverage differs from selected rows")
    output = selected.merge(fixed, on="id", how="left", validate="one_to_one")
    if output[["selected_abs_error", "fixed_abs_error"]].isna().any().any():
        raise ValueError("selector score summary contains missing errors")
    return output


def summarize_selector_score_parquet(
    path: Path,
    *,
    all_candidate_ids: Sequence[str],
    primary_ids: Sequence[str],
    fixed_candidate: str = "exp226_w500_50_50",
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = [
        "id",
        "well",
        "outer_fold",
        "md_since",
        "candidate_id",
        "actual_abs_error",
        "pred_abs_error",
    ]
    parts: list[pd.DataFrame] = []
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
        parts.append(
            _score_row_group_summary(
                frame,
                all_candidate_ids=all_candidate_ids,
                primary_ids=primary_ids,
                fixed_candidate=fixed_candidate,
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output["id"].duplicated().any():
        raise ValueError("selector score summary contains duplicate row IDs")
    return output


def _rmse_from_abs_error(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def build_fixed13_integration_readout(
    *,
    new_score_path: Path,
    parent_score_path: Path,
    hidden_like_assignment_path: Path,
    contract: Mapping[str, Any],
    score_summary: Mapping[str, Any],
    guard_config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validate_fixed13_contract(contract)
    new = summarize_selector_score_parquet(
        new_score_path,
        all_candidate_ids=(*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID),
        primary_ids=(*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID),
    ).rename(
        columns={
            "selected_candidate": "new_selected_candidate",
            "selected_abs_error": "new_selected_abs_error",
            "fixed_abs_error": "new_fixed_abs_error",
        }
    )
    parent = summarize_selector_score_parquet(
        parent_score_path,
        all_candidate_ids=BASE_CANDIDATE_IDS,
        primary_ids=BASE_PRIMARY_IDS,
    ).rename(
        columns={
            "selected_candidate": "parent_selected_candidate",
            "selected_abs_error": "parent_selected_abs_error",
            "fixed_abs_error": "parent_fixed_abs_error",
        }
    )
    joined = new.merge(
        parent[
            [
                "id",
                "well",
                "outer_fold",
                "md_since",
                "parent_selected_candidate",
                "parent_selected_abs_error",
                "parent_fixed_abs_error",
            ]
        ],
        on="id",
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_parent"),
    )
    if len(joined) != len(new) or len(joined) != len(parent):
        raise ValueError("fixed13 and fixed12 selector score identities differ")
    for column in ("well", "outer_fold", "md_since"):
        left = joined[f"{column}_new"].to_numpy()
        right = joined[f"{column}_parent"].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"fixed13/fixed12 identity mismatch: {column}")
        joined[column] = left
    fixed_parity_max_abs = float(
        np.max(
            np.abs(
                joined["new_fixed_abs_error"].to_numpy(np.float64)
                - joined["parent_fixed_abs_error"].to_numpy(np.float64)
            )
        )
    )
    if fixed_parity_max_abs > 1.0e-6:
        raise ValueError(
            f"fixed fallback errors changed after exp355 addition: {fixed_parity_max_abs}"
        )

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    assignment = assignment.set_index("well_id")
    scope_masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(joined), dtype=bool),
        "near_0_250": joined["md_since"].to_numpy(np.float64) <= 250.0,
        "distance_1000_plus": joined["md_since"].to_numpy(np.float64) >= 1000.0,
    }
    for fold in range(5):
        scope_masks[f"fold_{fold}"] = joined["outer_fold"].to_numpy(np.int8) == fold
    for role_column in (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    ):
        role = joined["well"].astype(str).map(assignment[role_column])
        scope_masks[role_column] = role.eq("valid").to_numpy()

    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        if not np.any(mask):
            raise ValueError(f"empty selector audit scope: {scope}")
        new_rmse = _rmse_from_abs_error(joined.loc[mask, "new_selected_abs_error"])
        parent_rmse = _rmse_from_abs_error(
            joined.loc[mask, "parent_selected_abs_error"]
        )
        fixed_rmse = _rmse_from_abs_error(joined.loc[mask, "new_fixed_abs_error"])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "fixed13_hard_rmse": new_rmse,
                "parent_fixed12_hard_rmse": parent_rmse,
                "fixed_fallback_rmse": fixed_rmse,
                "delta_fixed13_minus_parent": new_rmse - parent_rmse,
                "delta_fixed13_minus_fixed_fallback": new_rmse - fixed_rmse,
            }
        )
    scope_metrics = pd.DataFrame(scope_rows)

    usage_rows = []
    for scope in ("pooled", *(f"fold_{fold}" for fold in range(5))):
        mask = scope_masks[scope]
        selected = joined.loc[mask, "new_selected_candidate"].eq(
            ADDED_CANDIDATE_ID
        )
        usage_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "exp355_top1_rows": int(selected.sum()),
                "exp355_top1_fraction": float(selected.mean()),
            }
        )
    usage = pd.DataFrame(usage_rows)

    by_well_rows = []
    for well, part in joined.groupby("well", sort=True):
        new_rmse = _rmse_from_abs_error(part["new_selected_abs_error"])
        parent_rmse = _rmse_from_abs_error(part["parent_selected_abs_error"])
        fixed_rmse = _rmse_from_abs_error(part["new_fixed_abs_error"])
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "fixed13_hard_rmse": new_rmse,
                "parent_fixed12_hard_rmse": parent_rmse,
                "fixed_fallback_rmse": fixed_rmse,
                "delta_fixed13_minus_parent": new_rmse - parent_rmse,
                "exp355_top1_fraction": float(
                    part["new_selected_candidate"].eq(ADDED_CANDIDATE_ID).mean()
                ),
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    scope_lookup = scope_metrics.set_index("scope")
    usage_lookup = usage.set_index("scope")
    fold_improvements = int(
        sum(
            scope_lookup.loc[f"fold_{fold}", "delta_fixed13_minus_parent"] < 0.0
            for fold in range(5)
        )
    )
    positive_usage_folds = int(
        sum(
            usage_lookup.loc[f"fold_{fold}", "exp355_top1_fraction"] > 0.0
            for fold in range(5)
        )
    )
    hidden_deltas = [
        float(scope_lookup.loc[column, "delta_fixed13_minus_parent"])
        for column in (
            "verification_like_spatial_role",
            "verification_like_typewell_purged_role",
        )
    ]
    by_well_p95 = float(by_well["delta_fixed13_minus_parent"].quantile(0.95))
    worst = by_well.loc[by_well["delta_fixed13_minus_parent"].idxmax()]
    checks = {
        "selector_score_guard": bool(score_summary["score_guard"]["passed"]),
        "exp355_usage_pooled": float(
            usage_lookup.loc["pooled", "exp355_top1_fraction"]
        )
        >= float(guard_config["minimum_added_candidate_primary_top1_fraction"]),
        "exp355_usage_folds": positive_usage_folds
        >= int(guard_config["minimum_positive_usage_folds"]),
        "pooled_nonworse_than_parent": float(
            scope_lookup.loc["pooled", "delta_fixed13_minus_parent"]
        )
        <= float(guard_config["maximum_pooled_delta_rmse_vs_parent_fixed12_selector"]),
        "improved_parent_folds": fold_improvements
        >= int(guard_config["minimum_improved_folds_vs_parent_fixed12_selector"]),
        "near_nonworse": float(
            scope_lookup.loc["near_0_250", "delta_fixed13_minus_parent"]
        )
        <= float(guard_config["maximum_near_0_250_delta_rmse"]),
        "distance_1000_plus_nonworse": float(
            scope_lookup.loc["distance_1000_plus", "delta_fixed13_minus_parent"]
        )
        <= float(guard_config["maximum_1000_plus_delta_rmse"]),
        "hidden_like_nonworse": max(hidden_deltas)
        <= float(guard_config["maximum_hidden_like_delta_rmse"]),
        "by_well_p95_nonworse": by_well_p95
        <= float(guard_config["maximum_by_well_p95_delta_rmse"]),
        "worst_well_nonworse": float(worst["delta_fixed13_minus_parent"])
        <= float(guard_config["maximum_worst_well_delta_rmse"]),
    }
    gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "fixed_fallback_error_parity_max_abs_ft": fixed_parity_max_abs,
        "fold_improvements_vs_parent": fold_improvements,
        "positive_exp355_usage_folds": positive_usage_folds,
        "exp355_usage_pooled": float(
            usage_lookup.loc["pooled", "exp355_top1_fraction"]
        ),
        "by_well_p95_delta_rmse": by_well_p95,
        "worst_well": str(worst["well"]),
        "worst_well_delta_rmse": float(worst["delta_fixed13_minus_parent"]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    scope_path = output_dir / "exp373_fixed13_vs_fixed12_scope_metrics.csv"
    usage_path = output_dir / "exp373_fixed13_candidate_usage.csv"
    by_well_path = output_dir / "exp373_fixed13_vs_fixed12_by_well.csv"
    gate_path = output_dir / "exp373_scientific_gate.json"
    scope_metrics.to_csv(scope_path, index=False)
    usage.to_csv(usage_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    write_json(gate_path, gate)
    gate["artifact_sha256"] = {
        scope_path.name: sha256_file(scope_path),
        usage_path.name: sha256_file(usage_path),
        by_well_path.name: sha256_file(by_well_path),
        gate_path.name: sha256_file(gate_path),
    }
    return gate


def write_exp373_input_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    exp355_manifest: Mapping[str, Any],
    parent_score_path: Path,
) -> None:
    payload = {
        "experiment": config["experiment"]["name"],
        "candidate_order": candidate_ids(contract),
        "primary_domain": contract["legal_domains"]["primitive_pair_bank"]["candidates"],
        "fixed_domain": contract["legal_domains"]["primitive_fixed_bank"]["candidates"],
        "execution": config["execution"],
        "exp355_oof": dict(exp355_manifest),
        "parent_exp264_score": {
            "path": str(parent_score_path),
            "file_sha256": sha256_file(parent_score_path),
        },
    }
    write_json(path, payload)


__all__ = [
    "ADDED_CANDIDATE_ID",
    "BASE_CANDIDATE_IDS",
    "BASE_FIXED_IDS",
    "BASE_PRIMARY_IDS",
    "Exp355Fixed13CandidateCache",
    "base_exp264_contract",
    "build_fixed13_integration_readout",
    "exp355_prediction_content_sha256",
    "load_exp355_oof",
    "resolve_file_by_sha",
    "resolve_parent_score_file",
    "sha256_decompressed_gzip",
    "validate_fixed13_contract",
    "write_exp373_input_contract",
]
