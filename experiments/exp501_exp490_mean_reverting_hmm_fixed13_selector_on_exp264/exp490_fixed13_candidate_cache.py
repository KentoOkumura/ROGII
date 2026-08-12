from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    FoldBundle,
    candidate_ids,
    logical_frame_sha256,
    sha256_file,
    write_json,
)
from src.exp486_fixed13_candidate_cache import (
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    resolve_csv_by_payload_sha,
    resolve_file_by_sha,
    sha256_csv_payload,
    summarize_selector_score_parquet,
)
from src.exp486_fixed13_candidate_cache import (
    build_fixed13_integration_readout as _build_fixed13_integration_readout,
)
from src.exp486_fixed13_candidate_cache import (
    build_incumbent_reranking_diagnostic as _build_incumbent_reranking_diagnostic,
)
from src.exp486_fixed13_candidate_cache import (
    build_postfreeze_addone_novelty_readout as _build_postfreeze_addone_novelty_readout,
)

ADDED_CANDIDATE_ID = "exp490_geometry_mean_reverting_hmm"
EXP490_PREDICTION_COLUMN = "geometry_mean_reverting_hmm"
EXP490_NATIVE_FIELDS = (
    "geometry_mean_reverting_delta_mean",
    "geometry_mean_reverting_hmm_std",
)
EXP490_INPUT_ALLOWLIST = (
    "well",
    "row_idx",
    "suffix_offset",
    EXP490_PREDICTION_COLUMN,
    *EXP490_NATIVE_FIELDS,
)
EXP490_PREPARED_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "candidate_tvt",
    *EXP490_NATIVE_FIELDS,
)


def validate_fixed13_contract(contract: Mapping[str, Any]) -> None:
    ids = tuple(candidate_ids(contract))
    expected_ids = (*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID)
    if ids != expected_ids:
        raise ValueError(f"fixed13 candidate order mismatch: {ids}")
    if len(set(ids)) != 13:
        raise ValueError("fixed13 candidate IDs must be unique")
    specs = {str(item["id"]): item for item in contract["score_candidates"]}
    added = specs[ADDED_CANDIDATE_ID]
    if str(added.get("kind")) != "geometry_centered_mean_reverting_offset_hmm":
        raise ValueError("exp490 candidate kind changed")
    if str(added.get("prediction_column")) != EXP490_PREDICTION_COLUMN:
        raise ValueError("exp490 prediction column changed")
    if added.get("parents"):
        raise ValueError("exp490 candidate must not be reconstructed as a formula")
    native = tuple(str(item) for item in added.get("native_confidence", {}).keys())
    if native != EXP490_NATIVE_FIELDS:
        raise ValueError(f"exp490 native-confidence contract changed: {native}")
    domains = contract["legal_domains"]
    primary = tuple(domains["primitive_pair_bank"]["candidates"])
    fixed = tuple(domains["primitive_fixed_bank"]["candidates"])
    if primary != (*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID):
        raise ValueError("fixed13 primary domain differs from add-one contract")
    if fixed != BASE_FIXED_IDS:
        raise ValueError("fixed fallback domain changed from exp264")
    added_contract = contract.get("added_candidate_contract", {})
    if tuple(added_contract.get("included", [])) != (ADDED_CANDIDATE_ID,):
        raise ValueError("exp501 must include exactly one exp490 candidate")
    if added_contract.get("new_pair_or_blend_candidates"):
        raise ValueError("exp501 must not add pair or blend candidates")
    if bool(added_contract.get("fixed_fallback_changed")):
        raise ValueError("exp501 fixed fallback must remain unchanged")


def base_exp264_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_fixed13_contract(contract)
    base = copy.deepcopy(dict(contract))
    base["score_candidates"] = [
        item for item in base["score_candidates"] if str(item["id"]) != ADDED_CANDIDATE_ID
    ]
    base["legal_domains"]["primitive_pair_bank"]["candidates"] = list(BASE_PRIMARY_IDS)
    base["legal_domains"]["primitive_fixed_bank"]["candidates"] = list(BASE_FIXED_IDS)
    base["candidate_id_model_encoding"]["width"] = 12
    return base


def exp490_content_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["well_id", "row_idx"], kind="stable").reset_index(drop=True)
    return logical_frame_sha256(ordered.loc[:, list(EXP490_PREPARED_COLUMNS)])


def load_exp490_target_free_inputs(
    prediction_path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_prediction_gzip_raw_sha256: str,
    expected_prediction_payload_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_file_sha = sha256_file(prediction_path)
    prediction_payload_sha = sha256_csv_payload(prediction_path)
    if prediction_payload_sha != str(expected_prediction_payload_sha256):
        raise ValueError("exp490 prediction decompressed payload SHA mismatch")
    if prediction_path.suffix == ".gz" and prediction_file_sha != str(
        expected_prediction_gzip_raw_sha256
    ):
        raise ValueError("exp490 prediction gzip raw SHA mismatch")

    header = pd.read_csv(prediction_path, nrows=0).columns.astype(str).tolist()
    missing = set(EXP490_INPUT_ALLOWLIST) - set(header)
    if missing:
        raise ValueError(f"exp490 prediction allowlist columns missing: {sorted(missing)}")
    frame = pd.read_csv(
        prediction_path,
        usecols=list(EXP490_INPUT_ALLOWLIST),
        dtype={
            "well": str,
            "row_idx": np.int32,
            "suffix_offset": np.int32,
            EXP490_PREDICTION_COLUMN: np.float64,
            **{field: np.float64 for field in EXP490_NATIVE_FIELDS},
        },
    ).loc[:, list(EXP490_INPUT_ALLOWLIST)]
    frame = frame.rename(
        columns={"well": "well_id", EXP490_PREDICTION_COLUMN: "candidate_tvt"}
    )
    frame.insert(
        0,
        "id",
        frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str),
    )
    key = ["id", "well_id", "row_idx", "suffix_offset"]
    if frame.duplicated(key).any():
        raise ValueError("exp490 target-free input contains duplicate global keys")
    if len(frame) != int(expected_rows):
        raise ValueError(f"exp490 target-free row mismatch: {len(frame)}")
    if frame["well_id"].nunique() != int(expected_wells):
        raise ValueError("exp490 target-free well count mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("exp490 target-free input contains duplicate IDs")
    expected_id = frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str)
    id_key_mismatches = int((frame["id"].astype(str) != expected_id).sum())
    if id_key_mismatches:
        raise ValueError(f"exp490 id/key mismatch rows: {id_key_mismatches}")
    source_index = pd.MultiIndex.from_frame(frame[["well_id", "row_idx"]])
    source_order_monotonic = bool(source_index.is_monotonic_increasing)
    ordered = (
        frame
        if source_order_monotonic
        else frame.sort_values(["well_id", "row_idx"], kind="stable")
    )
    expected_suffix = ordered.groupby("well_id", sort=False).cumcount().to_numpy(np.int64)
    suffix_mismatches = int(
        np.sum(ordered["suffix_offset"].to_numpy(np.int64) != expected_suffix)
    )
    if suffix_mismatches:
        raise ValueError(f"exp490 suffix-offset sequence mismatch rows: {suffix_mismatches}")
    native = frame[["candidate_tvt", *EXP490_NATIVE_FIELDS]].to_numpy(np.float64)
    if not np.isfinite(native).all():
        raise ValueError("exp490 candidate or native confidence contains non-finite values")
    if bool((frame["geometry_mean_reverting_hmm_std"] < 0.0).any()):
        raise ValueError("exp490 geometry_mean_reverting_hmm_std must be non-negative")
    frame = frame.loc[:, list(EXP490_PREPARED_COLUMNS)]
    manifest = {
        "prediction_path": str(prediction_path),
        "prediction_header_columns": header,
        "prediction_loaded_columns": list(EXP490_INPUT_ALLOWLIST),
        "forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "id_key_mismatches": id_key_mismatches,
        "id_source": "constructed_from_allowlisted_well_and_row_idx",
        "suffix_sequence_mismatches": suffix_mismatches,
        "source_order_monotonic_well_row_idx": source_order_monotonic,
        "prediction_file_sha256": prediction_file_sha,
        "prediction_payload_sha256": prediction_payload_sha,
        "prediction_expected_gzip_raw_sha256": expected_prediction_gzip_raw_sha256,
        "upstream_source_fold_column_loaded": False,
        "upstream_source_fold_used_as_model_feature": False,
        "upstream_exp490_group_safe_oof_contract": True,
        "native_confidence_fields": list(EXP490_NATIVE_FIELDS),
        "candidate_and_native_confidence_finite_fraction": float(
            np.isfinite(native).all(axis=1).mean()
        ),
        "post_read_content_sha256": logical_frame_sha256(
            ordered.loc[:, list(EXP490_PREPARED_COLUMNS)]
        ),
    }
    return frame, manifest


class Exp490Fixed13CandidateCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        exp490_inputs: pd.DataFrame,
        exp490_manifest: Mapping[str, Any],
    ):
        validate_fixed13_contract(contract)
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = {str(item["id"]): dict(item) for item in contract["score_candidates"]}
        self.base_cache = Exp263CandidateCache(root, base_exp264_contract(contract))
        self.exp490_manifest = dict(exp490_manifest)
        self.exp490_by_key = exp490_inputs.sort_values(
            ["well_id", "row_idx"], kind="stable"
        ).set_index(["well_id", "row_idx"])[
            ["candidate_tvt", "suffix_offset", *EXP490_NATIVE_FIELDS]
        ]
        if not self.exp490_by_key.index.is_unique:
            raise ValueError("exp490 global key index is not unique")
        self._selector_fold_audits: dict[int, dict[str, Any]] = {}

    def load_fold(self, fold: int) -> FoldBundle:
        base = self.base_cache.load_fold(int(fold))
        expected = base.base.sort_values(["well", "well_row_idx"], kind="stable").reset_index()
        if not np.all(expected["outer_fold"].to_numpy(np.int8) == np.int8(fold)):
            raise ValueError(f"exp263 base fold identity mismatch in fold {fold}")
        selector_keys = pd.MultiIndex.from_arrays(
            [
                expected["well"].astype(str).to_numpy(),
                expected["well_row_idx"].to_numpy(np.int64),
            ],
            names=["well_id", "row_idx"],
        )
        added = self.exp490_by_key.reindex(selector_keys)
        required = ["candidate_tvt", "suffix_offset", *EXP490_NATIVE_FIELDS]
        missing = added[required].isna().any(axis=1)
        if missing.any():
            missing_keys = [
                (str(well), int(row)) for well, row in selector_keys[missing.to_numpy()][:5]
            ]
            raise ValueError(
                f"exp490 global key join is missing exp263 fold {fold} rows: {missing_keys}"
            )
        expected_suffix = expected.groupby("well", sort=False).cumcount().to_numpy(np.int64)
        suffix_mismatches = int(
            np.sum(added["suffix_offset"].to_numpy(np.int64) != expected_suffix)
        )
        if suffix_mismatches:
            raise ValueError(
                f"exp490/exp263 suffix-offset parity failed in fold {fold}: "
                f"{suffix_mismatches} rows"
            )
        self._selector_fold_audits[int(fold)] = {
            "selector_outer_fold": int(fold),
            "rows": len(expected),
            "wells": int(expected["well"].nunique()),
            "missing_key_rows": 0,
            "suffix_offset_mismatch_rows": 0,
            "candidate_generation": "saved_exp490_group_safe_full_oof_mean_reverting_hmm",
            "upstream_source_fold_loaded": False,
            "upstream_source_fold_used_as_model_feature": False,
        }
        inverse = np.empty(len(expected), dtype=np.int64)
        inverse[expected["index"].to_numpy(np.int64)] = np.arange(len(expected))
        prediction = added["candidate_tvt"].to_numpy(np.float32)[inverse]
        native = added[list(EXP490_NATIVE_FIELDS)].to_numpy(np.float32)[inverse]
        valid = np.isfinite(np.column_stack([prediction, native])).all(axis=1)
        valid &= native[:, 1] >= 0.0
        if not bool(valid.all()):
            raise ValueError(f"exp490 fold {fold} candidate/native confidence is invalid")
        values = np.column_stack([base.values, prediction]).astype(np.float32)
        available = np.column_stack(
            [base.available, np.ones(len(base.base), dtype=bool)]
        ).astype(bool)
        confidence = dict(base.confidence)
        conf = base.base[KEY_COLUMNS].copy()
        conf["candidate_id"] = ADDED_CANDIDATE_ID
        conf["confidence_source"] = "exp490_mean_reverting_hmm_target_free_state"
        conf["confidence_valid"] = valid
        conf["confidence_missing_fields"] = ""
        for position, field in enumerate(EXP490_NATIVE_FIELDS):
            conf[field] = native[:, position]
        confidence[ADDED_CANDIDATE_ID] = conf
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
            str(fold): int(self._selector_fold_audits[fold]["rows"]) for fold in range(5)
        }
        total_rows = int(sum(selector_fold_rows.values()))
        checks = {
            "all_selector_folds_audited": True,
            "global_key_join_rows_match": total_rows == int(expected_rows),
            "missing_key_rows_zero": all(
                int(self._selector_fold_audits[fold]["missing_key_rows"]) == 0
                for fold in range(5)
            ),
            "suffix_offset_parity": all(
                int(self._selector_fold_audits[fold]["suffix_offset_mismatch_rows"]) == 0
                for fold in range(5)
            ),
            "upstream_exp490_group_safe_oof_contract": bool(
                self.exp490_manifest["upstream_exp490_group_safe_oof_contract"]
            ),
            "upstream_source_fold_not_loaded": not bool(
                self.exp490_manifest["upstream_source_fold_column_loaded"]
            ),
            "upstream_source_fold_not_used_as_model_feature": not bool(
                self.exp490_manifest["upstream_source_fold_used_as_model_feature"]
            ),
        }
        return {
            "policy": "global_key_join_then_exp263_selector_fold_repartition",
            "selector_fold_source": "exp263_row_count_balanced_outer_fold",
            "candidate_generation": "saved_exp490_group_safe_full_oof_mean_reverting_hmm",
            "upstream_source_fold_role": "safety_audit_only",
            "upstream_source_fold_loaded": False,
            "upstream_source_fold_used_as_model_feature": False,
            "rows": total_rows,
            "selector_fold_rows": selector_fold_rows,
            "overlap_by_selector_fold": [
                self._selector_fold_audits[fold] for fold in range(5)
            ],
            "checks": checks,
            "passed": bool(all(checks.values())),
        }


def pair_selector_scores(
    *, new_score_path: Path, parent_score_path: Path, contract: Mapping[str, Any]
) -> pd.DataFrame:
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
                "well_row_idx",
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
    identity_columns = ("well", "well_row_idx", "outer_fold", "md_since")
    for column in identity_columns:
        left = joined[f"{column}_new"].to_numpy()
        right = joined[f"{column}_parent"].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"fixed13/fixed12 identity mismatch: {column}")
        joined[column] = left
    return joined.drop(
        columns=[
            *(f"{column}_new" for column in identity_columns),
            *(f"{column}_parent" for column in identity_columns),
        ]
    )


def build_fixed13_integration_readout(**kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    return _build_fixed13_integration_readout(
        **kwargs,
        added_candidate_id=ADDED_CANDIDATE_ID,
        added_label="exp490",
        output_prefix="exp501",
        success_decision="PASS_EXP490_MEAN_REVERTING_HMM_FIXED13_SELECTOR",
    )


def build_postfreeze_addone_novelty_readout(**kwargs: Any) -> dict[str, Any]:
    return _build_postfreeze_addone_novelty_readout(
        **kwargs,
        added_candidate_id=ADDED_CANDIDATE_ID,
        base_primary_ids=BASE_PRIMARY_IDS,
        added_label="exp490",
        output_prefix="exp501",
    )


def build_incumbent_reranking_diagnostic(**kwargs: Any) -> dict[str, Any]:
    contract = kwargs.get("contract")
    if not isinstance(contract, Mapping):
        raise TypeError("reranking diagnostic requires a candidate contract mapping")
    validate_fixed13_contract(contract)
    return _build_incumbent_reranking_diagnostic(
        **kwargs,
        added_candidate_id=ADDED_CANDIDATE_ID,
        base_candidate_ids=BASE_CANDIDATE_IDS,
        base_primary_ids=BASE_PRIMARY_IDS,
        added_label="exp490",
        output_prefix="exp501",
        validate_contract=False,
    )


def write_exp501_input_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    exp490_manifest: Mapping[str, Any],
    parent_score_path: Path,
) -> None:
    payload = {
        "experiment": config["experiment"]["name"],
        "candidate_order": candidate_ids(contract),
        "primary_domain": contract["legal_domains"]["primitive_pair_bank"]["candidates"],
        "fixed_domain": contract["legal_domains"]["primitive_fixed_bank"]["candidates"],
        "execution": config["execution"],
        "exp490_target_free_inputs": dict(exp490_manifest),
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
    "EXP490_INPUT_ALLOWLIST",
    "EXP490_NATIVE_FIELDS",
    "EXP490_PREDICTION_COLUMN",
    "Exp490Fixed13CandidateCache",
    "base_exp264_contract",
    "build_fixed13_integration_readout",
    "build_incumbent_reranking_diagnostic",
    "build_postfreeze_addone_novelty_readout",
    "exp490_content_sha256",
    "load_exp490_target_free_inputs",
    "pair_selector_scores",
    "resolve_csv_by_payload_sha",
    "resolve_file_by_sha",
    "sha256_csv_payload",
    "validate_fixed13_contract",
    "write_exp501_input_contract",
]
