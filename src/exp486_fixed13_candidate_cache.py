from __future__ import annotations

import copy
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
ADDED_CANDIDATE_ID = "exp486_absolute_geometry_likpf"
EXP486_PREDICTION_COLUMN = "likpf_scale5_absolute_geometry_unary"
EXP486_PREDICTION_ALLOWLIST = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    EXP486_PREDICTION_COLUMN,
)
EXP486_LEDGER_FIELDS = (
    "geometry_residual_mean",
    "geometry_residual_std",
    "geometry_log_factor_mean",
    "effective_sample_size",
    "resampled_seed_fraction",
)
EXP486_LEDGER_ALLOWLIST = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    *EXP486_LEDGER_FIELDS,
)
EXP486_PREPARED_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "candidate_tvt",
    *EXP486_LEDGER_FIELDS,
)


def sha256_csv_payload(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash decompressed gzip bytes or plain CSV bytes."""

    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as stream:
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


def resolve_csv_by_payload_sha(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    expected_payload_sha256: str,
    expected_gzip_raw_sha256: str | None,
    label: str,
) -> Path:
    candidates = [path for path in _expand_paths(patterns, search_roots) if path.is_file()]
    matches: list[Path] = []
    evidence: dict[str, dict[str, Any]] = {}
    for path in candidates:
        file_sha = sha256_file(path)
        payload_sha = sha256_csv_payload(path)
        gzip_raw_pass = (
            path.suffix != ".gz"
            or not expected_gzip_raw_sha256
            or file_sha == str(expected_gzip_raw_sha256)
        )
        evidence[str(path)] = {
            "file_sha256": file_sha,
            "payload_sha256": payload_sha,
            "is_gzip": path.suffix == ".gz",
            "gzip_raw_pass": gzip_raw_pass,
        }
        if payload_sha == str(expected_payload_sha256) and gzip_raw_pass:
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{label} did not resolve to exactly one SHA-matched file: {evidence}"
        )
    return matches[0]


def resolve_file_by_sha(
    patterns: Sequence[str],
    search_roots: Sequence[Path],
    *,
    expected_file_sha256: str,
    label: str,
) -> Path:
    candidates = [path for path in _expand_paths(patterns, search_roots) if path.is_file()]
    matches = [path for path in candidates if sha256_file(path) == str(expected_file_sha256)]
    if len(matches) != 1:
        evidence = {str(path): sha256_file(path) for path in candidates}
        raise FileNotFoundError(
            f"{label} did not resolve to exactly one SHA-matched file: {evidence}"
        )
    return matches[0]


def validate_fixed13_contract(contract: Mapping[str, Any]) -> None:
    ids = tuple(candidate_ids(contract))
    if ids != (*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID):
        raise ValueError(f"fixed13 candidate order mismatch: {ids}")
    if len(set(ids)) != 13:
        raise ValueError("fixed13 candidate IDs must be unique")
    specs = {str(item["id"]): item for item in contract["score_candidates"]}
    added = specs[ADDED_CANDIDATE_ID]
    if str(added.get("kind")) != "absolute_geometry_likelihood_pf":
        raise ValueError("exp486 candidate kind changed")
    if str(added.get("prediction_column")) != EXP486_PREDICTION_COLUMN:
        raise ValueError("exp486 prediction column changed")
    if added.get("parents"):
        raise ValueError("exp486 candidate must not be reconstructed as a formula")
    native = tuple(str(item) for item in added.get("native_confidence", {}).keys())
    if native != EXP486_LEDGER_FIELDS:
        raise ValueError(f"exp486 native-confidence contract changed: {native}")
    domains = contract["legal_domains"]
    primary = tuple(domains["primitive_pair_bank"]["candidates"])
    fixed = tuple(domains["primitive_fixed_bank"]["candidates"])
    if primary != (*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID):
        raise ValueError("fixed13 primary domain differs from add-one contract")
    if fixed != BASE_FIXED_IDS:
        raise ValueError("fixed fallback domain changed from exp264")
    added_contract = contract.get("added_candidate_contract", {})
    if tuple(added_contract.get("included", [])) != (ADDED_CANDIDATE_ID,):
        raise ValueError("exp496 must include exactly one exp486 candidate")
    if added_contract.get("new_pair_or_blend_candidates"):
        raise ValueError("exp496 must not add pair or blend candidates")
    if bool(added_contract.get("fixed_fallback_changed")):
        raise ValueError("exp496 fixed fallback must remain unchanged")


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


def exp486_content_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["well_id", "row_idx"], kind="stable").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("|".join(EXP486_PREPARED_COLUMNS).encode())
    digest.update(b"\n")
    for column in EXP486_PREPARED_COLUMNS:
        values = ordered[column]
        if column in {"id", "well_id"}:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\0")
        elif column in {"row_idx", "suffix_offset"}:
            digest.update(values.to_numpy(np.int64).astype("<i8", copy=False).tobytes())
        else:
            digest.update(values.to_numpy(np.float64).astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def load_exp486_target_free_inputs(
    prediction_path: Path,
    absolute_ledger_path: Path,
    freeze_manifest_path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_prediction_gzip_raw_sha256: str,
    expected_prediction_payload_sha256: str,
    expected_prediction_upstream_logical_sha256: str,
    expected_ledger_gzip_raw_sha256: str,
    expected_ledger_payload_sha256: str,
    expected_freeze_manifest_sha256: str,
    expected_scientific_contract_sha256: str,
    expected_exp226_geometry_decompressed_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_file_sha = sha256_file(prediction_path)
    prediction_payload_sha = sha256_csv_payload(prediction_path)
    ledger_file_sha = sha256_file(absolute_ledger_path)
    ledger_payload_sha = sha256_csv_payload(absolute_ledger_path)
    freeze_file_sha = sha256_file(freeze_manifest_path)
    if prediction_payload_sha != str(expected_prediction_payload_sha256):
        raise ValueError("exp486 prediction decompressed payload SHA mismatch")
    if prediction_path.suffix == ".gz" and prediction_file_sha != str(
        expected_prediction_gzip_raw_sha256
    ):
        raise ValueError("exp486 prediction gzip raw SHA mismatch")
    if ledger_payload_sha != str(expected_ledger_payload_sha256):
        raise ValueError("exp486 absolute ledger decompressed payload SHA mismatch")
    if absolute_ledger_path.suffix == ".gz" and ledger_file_sha != str(
        expected_ledger_gzip_raw_sha256
    ):
        raise ValueError("exp486 absolute ledger gzip raw SHA mismatch")
    if freeze_file_sha != str(expected_freeze_manifest_sha256):
        raise ValueError("exp486 freeze manifest SHA mismatch")

    freeze = json.loads(freeze_manifest_path.read_text())
    freeze_checks = {
        "frozen_before_truth_attachment": bool(freeze.get("frozen_before_truth_attachment")),
        "rows": int(freeze.get("rows", -1)) == int(expected_rows),
        "wells": int(freeze.get("wells", -1)) == int(expected_wells),
        "scientific_contract": str(freeze.get("scientific_contract_sha256"))
        == str(expected_scientific_contract_sha256),
        "prediction_logical": str(freeze.get("prediction_logical_sha256"))
        == str(expected_prediction_upstream_logical_sha256),
        "absolute_ledger_logical": str(freeze.get("absolute_ledger_logical_sha256"))
        == str(expected_ledger_payload_sha256),
        "sha_readback": bool(freeze.get("sha_readback", {}).get("pass")),
    }
    if not all(freeze_checks.values()):
        raise ValueError(f"exp486 freeze manifest contract mismatch: {freeze_checks}")

    prediction_header = pd.read_csv(prediction_path, nrows=0).columns.astype(str).tolist()
    ledger_header = pd.read_csv(absolute_ledger_path, nrows=0).columns.astype(str).tolist()
    missing_prediction = set(EXP486_PREDICTION_ALLOWLIST) - set(prediction_header)
    missing_ledger = set(EXP486_LEDGER_ALLOWLIST) - set(ledger_header)
    if missing_prediction:
        raise ValueError(
            f"exp486 prediction allowlist columns missing: {sorted(missing_prediction)}"
        )
    if missing_ledger:
        raise ValueError(
            f"exp486 absolute-ledger allowlist columns missing: {sorted(missing_ledger)}"
        )
    prediction = pd.read_csv(
        prediction_path,
        usecols=list(EXP486_PREDICTION_ALLOWLIST),
        dtype={
            "id": str,
            "well_id": str,
            "row_idx": np.int32,
            "suffix_offset": np.int32,
            EXP486_PREDICTION_COLUMN: np.float64,
        },
    ).loc[:, list(EXP486_PREDICTION_ALLOWLIST)]
    ledger = pd.read_csv(
        absolute_ledger_path,
        usecols=list(EXP486_LEDGER_ALLOWLIST),
        dtype={
            "id": str,
            "well_id": str,
            "row_idx": np.int32,
            "suffix_offset": np.int32,
            **{field: np.float64 for field in EXP486_LEDGER_FIELDS},
        },
    ).loc[:, list(EXP486_LEDGER_ALLOWLIST)]
    prediction = prediction.rename(columns={EXP486_PREDICTION_COLUMN: "candidate_tvt"})
    key = ["id", "well_id", "row_idx", "suffix_offset"]
    if prediction.duplicated(key).any() or ledger.duplicated(key).any():
        raise ValueError("exp486 target-free input contains duplicate global keys")
    frame = prediction.merge(ledger, on=key, how="inner", validate="one_to_one")
    if len(frame) != len(prediction) or len(frame) != len(ledger):
        raise ValueError("exp486 prediction and absolute ledger key coverage differ")
    if len(frame) != int(expected_rows):
        raise ValueError(f"exp486 target-free row mismatch: {len(frame)}")
    if frame["well_id"].nunique() != int(expected_wells):
        raise ValueError("exp486 target-free well count mismatch")
    if frame["id"].duplicated().any():
        raise ValueError("exp486 target-free input contains duplicate IDs")
    expected_id = frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str)
    id_key_mismatches = int((frame["id"].astype(str) != expected_id).sum())
    if id_key_mismatches:
        raise ValueError(f"exp486 id/key mismatch rows: {id_key_mismatches}")
    native = frame[["candidate_tvt", *EXP486_LEDGER_FIELDS]].to_numpy(np.float64)
    if not np.isfinite(native).all():
        raise ValueError("exp486 candidate or native confidence contains non-finite values")
    if bool((frame["geometry_residual_std"] < 0.0).any()):
        raise ValueError("exp486 geometry_residual_std must be non-negative")
    if bool((frame["effective_sample_size"] <= 0.0).any()):
        raise ValueError("exp486 effective_sample_size must be positive")
    resampled = frame["resampled_seed_fraction"].to_numpy(np.float64)
    if bool(((resampled < 0.0) | (resampled > 1.0)).any()):
        raise ValueError("exp486 resampled_seed_fraction must be in [0, 1]")
    frame = frame.loc[:, list(EXP486_PREPARED_COLUMNS)]
    manifest = {
        "prediction_path": str(prediction_path),
        "absolute_ledger_path": str(absolute_ledger_path),
        "freeze_manifest_path": str(freeze_manifest_path),
        "prediction_header_columns": prediction_header,
        "absolute_ledger_header_columns": ledger_header,
        "prediction_loaded_columns": list(EXP486_PREDICTION_ALLOWLIST),
        "absolute_ledger_loaded_columns": list(EXP486_LEDGER_ALLOWLIST),
        "forbidden_truth_error_control_role_fold_scope_gate_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "id_key_mismatches": id_key_mismatches,
        "prediction_file_sha256": prediction_file_sha,
        "prediction_payload_sha256": prediction_payload_sha,
        "prediction_expected_gzip_raw_sha256": expected_prediction_gzip_raw_sha256,
        "prediction_upstream_logical_sha256": expected_prediction_upstream_logical_sha256,
        "prediction_upstream_logical_verified_by_exact_payload_and_freeze_manifest": True,
        "absolute_ledger_file_sha256": ledger_file_sha,
        "absolute_ledger_payload_sha256": ledger_payload_sha,
        "absolute_ledger_expected_gzip_raw_sha256": expected_ledger_gzip_raw_sha256,
        "freeze_manifest_sha256": freeze_file_sha,
        "freeze_manifest_checks": freeze_checks,
        "scientific_contract_sha256": expected_scientific_contract_sha256,
        "exp226_geometry_decompressed_sha256": expected_exp226_geometry_decompressed_sha256,
        "upstream_exp226_group_safe_oof_contract": True,
        "upstream_fold_column_loaded": False,
        "upstream_fold_used_as_model_feature": False,
        "native_confidence_fields": list(EXP486_LEDGER_FIELDS),
        "candidate_and_native_confidence_finite_fraction": float(
            np.isfinite(native).all(axis=1).mean()
        ),
        "post_read_content_sha256": exp486_content_sha256(frame),
    }
    return frame, manifest


class Exp486Fixed13CandidateCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        exp486_inputs: pd.DataFrame,
        exp486_manifest: Mapping[str, Any],
    ):
        validate_fixed13_contract(contract)
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = {str(item["id"]): dict(item) for item in contract["score_candidates"]}
        self.base_cache = Exp263CandidateCache(root, base_exp264_contract(contract))
        self.exp486_manifest = dict(exp486_manifest)
        self.exp486_by_key = exp486_inputs.sort_values(
            ["well_id", "row_idx"], kind="stable"
        ).set_index(["well_id", "row_idx"])[
            ["candidate_tvt", "suffix_offset", *EXP486_LEDGER_FIELDS]
        ]
        if not self.exp486_by_key.index.is_unique:
            raise ValueError("exp486 global key index is not unique")
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
        added = self.exp486_by_key.reindex(selector_keys)
        required = ["candidate_tvt", "suffix_offset", *EXP486_LEDGER_FIELDS]
        missing = added[required].isna().any(axis=1)
        if missing.any():
            missing_keys = [
                (str(well), int(row)) for well, row in selector_keys[missing.to_numpy()][:5]
            ]
            raise ValueError(
                f"exp486 global key join is missing exp263 fold {fold} rows: {missing_keys}"
            )
        self._selector_fold_audits[int(fold)] = {
            "selector_outer_fold": int(fold),
            "rows": len(expected),
            "wells": int(expected["well"].nunique()),
            "missing_key_rows": 0,
            "candidate_generation": ("fold_safe_exp226_oof_geometry_plus_target_free_per_well_pf"),
            "upstream_source_fold_loaded": False,
            "upstream_source_fold_used_as_model_feature": False,
        }
        inverse = np.empty(len(expected), dtype=np.int64)
        inverse[expected["index"].to_numpy(np.int64)] = np.arange(len(expected))
        prediction = added["candidate_tvt"].to_numpy(np.float32)[inverse]
        if not np.isfinite(prediction).all():
            raise ValueError(f"exp486 fold {fold} contains a non-finite prediction")
        values = np.column_stack([base.values, prediction]).astype(np.float32)
        available = np.column_stack([base.available, np.ones(len(base.base), dtype=bool)]).astype(
            bool
        )
        native = added[list(EXP486_LEDGER_FIELDS)].to_numpy(np.float32)[inverse]
        valid = np.isfinite(np.column_stack([prediction, native])).all(axis=1)
        valid &= native[:, 1] >= 0.0
        valid &= native[:, 3] > 0.0
        valid &= (native[:, 4] >= 0.0) & (native[:, 4] <= 1.0)
        if not bool(valid.all()):
            raise ValueError(f"exp486 fold {fold} native confidence is invalid")
        confidence = dict(base.confidence)
        conf = base.base[KEY_COLUMNS].copy()
        conf["candidate_id"] = ADDED_CANDIDATE_ID
        conf["confidence_source"] = "exp486_absolute_geometry_mechanism_ledger"
        conf["confidence_valid"] = valid
        conf["confidence_missing_fields"] = ""
        for position, field in enumerate(EXP486_LEDGER_FIELDS):
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
                int(self._selector_fold_audits[fold]["missing_key_rows"]) == 0 for fold in range(5)
            ),
            "upstream_exp226_group_safe_oof_contract": bool(
                self.exp486_manifest["upstream_exp226_group_safe_oof_contract"]
            ),
            "upstream_source_fold_not_loaded": not bool(
                self.exp486_manifest["upstream_fold_column_loaded"]
            ),
            "upstream_source_fold_not_used_as_model_feature": not bool(
                self.exp486_manifest["upstream_fold_used_as_model_feature"]
            ),
        }
        return {
            "policy": "global_key_join_then_exp263_selector_fold_repartition",
            "selector_fold_source": "exp263_row_count_balanced_outer_fold",
            "candidate_generation": ("fold_safe_exp226_oof_geometry_plus_target_free_per_well_pf"),
            "upstream_source_fold_role": "safety_audit_only",
            "upstream_source_fold_loaded": False,
            "upstream_source_fold_used_as_model_feature": False,
            "rows": total_rows,
            "selector_fold_rows": selector_fold_rows,
            "overlap_by_selector_fold": [self._selector_fold_audits[fold] for fold in range(5)],
            "checks": checks,
            "passed": bool(all(checks.values())),
        }


def _score_row_group_summary(
    frame: pd.DataFrame,
    *,
    all_candidate_ids: Sequence[str],
    primary_ids: Sequence[str],
    fixed_candidate: str = "exp226_w500_50_50",
) -> pd.DataFrame:
    expected_count = len(all_candidate_ids)
    counts = frame.groupby("id", sort=False)["candidate_id"].size()
    if not bool(counts.eq(expected_count).all()):
        raise ValueError("candidate score row group contains an incomplete candidate set")
    observed = set(frame["candidate_id"].astype(str).unique())
    if not observed.issubset(set(all_candidate_ids)):
        raise ValueError("candidate score row group contains an unknown candidate")
    primary = frame.loc[frame["candidate_id"].isin(primary_ids)]
    selected_idx = primary.groupby("id", sort=False)["pred_abs_error"].idxmin()
    selected = primary.loc[
        selected_idx,
        [
            "id",
            "well",
            "well_row_idx",
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
    fixed = frame.loc[frame["candidate_id"].eq(fixed_candidate), ["id", "actual_abs_error"]].rename(
        columns={"actual_abs_error": "fixed_abs_error"}
    )
    output = selected.merge(fixed, on="id", how="left", validate="one_to_one")
    if output[["selected_abs_error", "fixed_abs_error"]].isna().any().any():
        raise ValueError("selector score summary contains missing errors")
    return output


def summarize_selector_score_parquet(
    path: Path,
    *,
    all_candidate_ids: Sequence[str],
    primary_ids: Sequence[str],
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = [
        "id",
        "well",
        "well_row_idx",
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
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output["id"].duplicated().any():
        raise ValueError("selector score summary contains duplicate row IDs")
    return output


def pair_selector_scores(
    *,
    new_score_path: Path,
    parent_score_path: Path,
    contract: Mapping[str, Any],
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
    for column in ("well", "well_row_idx", "outer_fold", "md_since"):
        left = joined[f"{column}_new"].to_numpy()
        right = joined[f"{column}_parent"].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"fixed13/fixed12 identity mismatch: {column}")
        joined[column] = left
    return joined.drop(
        columns=[
            *(f"{column}_new" for column in ("well", "well_row_idx", "outer_fold", "md_since")),
            *(f"{column}_parent" for column in ("well", "well_row_idx", "outer_fold", "md_since")),
        ]
    )


def _rmse_from_abs_error(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _attach_raw_scope_context(
    paired: pd.DataFrame,
    *,
    raw_train_dir: Path,
    high_missing_fraction_threshold: float,
) -> pd.DataFrame:
    output = paired.copy()
    output["raw_gr_observed"] = False
    output["well_missing_fraction"] = np.nan
    for well, positions in output.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        path = Path(raw_train_dir) / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw_gr = pd.read_csv(path, usecols=["GR"])["GR"]
        row_idx = output.iloc[pos]["well_row_idx"].to_numpy(np.int64)
        if row_idx.min(initial=0) < 0 or row_idx.max(initial=-1) >= len(raw_gr):
            raise ValueError(f"raw scope row index out of bounds for well={well}")
        observed = np.isfinite(
            pd.to_numeric(raw_gr.iloc[row_idx], errors="coerce").to_numpy(np.float64)
        )
        output.loc[pos, "raw_gr_observed"] = observed
        output.loc[pos, "well_missing_fraction"] = float((~observed).mean())
    if output["well_missing_fraction"].isna().any():
        raise ValueError("raw scope context coverage is incomplete")
    output["missing_fraction_high"] = output["well_missing_fraction"].ge(
        float(high_missing_fraction_threshold)
    )
    return output


def build_fixed13_integration_readout(
    *,
    paired: pd.DataFrame,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    score_summary: Mapping[str, Any],
    guard_config: Mapping[str, Any],
    output_dir: Path,
    added_candidate_id: str = ADDED_CANDIDATE_ID,
    added_label: str = "exp486",
    output_prefix: str = "exp496",
    success_decision: str = "PASS_EXP486_ABSOLUTE_FIXED13_SELECTOR",
) -> tuple[dict[str, Any], pd.DataFrame]:
    usage_rows_column = f"{added_label}_top1_rows"
    usage_fraction_column = f"{added_label}_top1_fraction"
    incumbent_rows_column = (
        f"incumbent_choice_change_rows_when_{added_label}_not_top1"
    )
    incumbent_fraction_column = (
        f"incumbent_choice_change_fraction_when_{added_label}_not_top1"
    )
    joined = _attach_raw_scope_context(
        paired,
        raw_train_dir=raw_train_dir,
        high_missing_fraction_threshold=float(guard_config["high_missing_fraction_threshold"]),
    )
    fixed_parity_max_abs = float(
        np.max(
            np.abs(
                joined["new_fixed_abs_error"].to_numpy(np.float64)
                - joined["parent_fixed_abs_error"].to_numpy(np.float64)
            )
        )
    )
    if fixed_parity_max_abs != 0.0:
        raise ValueError(
            f"fixed fallback errors changed after {added_label} addition: "
            f"{fixed_parity_max_abs}"
        )

    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str})
    assignment = assignment.set_index("well_id")
    scope_masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(joined), dtype=bool),
        "raw_gr_observed": joined["raw_gr_observed"].to_numpy(bool),
        "raw_gr_missing": ~joined["raw_gr_observed"].to_numpy(bool),
        "missing_fraction_high": joined["missing_fraction_high"].to_numpy(bool),
        "distance_0_250": joined["md_since"].to_numpy(np.float64) <= 250.0,
        "distance_1000_plus": joined["md_since"].to_numpy(np.float64) >= 1000.0,
    }
    for fold in range(5):
        scope_masks[f"fold_{fold}"] = joined["outer_fold"].to_numpy(np.int8) == fold
    role_mapping = {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }
    for scope, role_column in role_mapping.items():
        role = joined["well"].astype(str).map(assignment[role_column])
        scope_masks[scope] = role.eq("valid").to_numpy()

    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        if not np.any(mask):
            raise ValueError(f"empty selector audit scope: {scope}")
        new_rmse = _rmse_from_abs_error(joined.loc[mask, "new_selected_abs_error"])
        parent_rmse = _rmse_from_abs_error(joined.loc[mask, "parent_selected_abs_error"])
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

    usage_rows: list[dict[str, Any]] = []
    for scope in ("pooled", *(f"fold_{fold}" for fold in range(5))):
        mask = scope_masks[scope]
        selected = joined.loc[mask, "new_selected_candidate"].eq(added_candidate_id)
        usage_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                usage_rows_column: int(selected.sum()),
                usage_fraction_column: float(selected.mean()),
            }
        )
    usage = pd.DataFrame(usage_rows)

    by_well_rows: list[dict[str, Any]] = []
    for well, part in joined.groupby("well", sort=True):
        new_rmse = _rmse_from_abs_error(part["new_selected_abs_error"])
        parent_rmse = _rmse_from_abs_error(part["parent_selected_abs_error"])
        fixed_rmse = _rmse_from_abs_error(part["new_fixed_abs_error"])
        added_selected = part["new_selected_candidate"].eq(added_candidate_id)
        incumbent_changed = ~added_selected & part["new_selected_candidate"].ne(
            part["parent_selected_candidate"]
        )
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "fixed13_hard_rmse": new_rmse,
                "parent_fixed12_hard_rmse": parent_rmse,
                "fixed_fallback_rmse": fixed_rmse,
                "delta_fixed13_minus_parent": new_rmse - parent_rmse,
                usage_fraction_column: float(added_selected.mean()),
                incumbent_rows_column: int(incumbent_changed.sum()),
                incumbent_fraction_column: float(
                    incumbent_changed.sum() / max((~added_selected).sum(), 1)
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
            usage_lookup.loc[f"fold_{fold}", usage_fraction_column] > 0.0
            for fold in range(5)
        )
    )
    required_scopes = [str(item) for item in guard_config["required_scopes"]]
    missing_required = set(required_scopes) - set(scope_lookup.index)
    if missing_required:
        raise ValueError(f"integration guard scopes missing: {sorted(missing_required)}")
    scope_limit = float(guard_config["maximum_scope_delta_rmse_ft"])
    by_well_p95 = float(by_well["delta_fixed13_minus_parent"].quantile(0.95))
    worst = by_well.loc[by_well["delta_fixed13_minus_parent"].idxmax()]
    checks: dict[str, bool] = {
        "selector_score_guard": bool(score_summary["score_guard"]["passed"]),
        f"{added_label}_usage_pooled": float(
            usage_lookup.loc["pooled", usage_fraction_column]
        )
        >= float(guard_config["minimum_added_candidate_primary_top1_fraction"]),
        f"{added_label}_usage_folds": positive_usage_folds
        >= int(guard_config["minimum_positive_usage_folds"]),
        "pooled_nonworse_than_parent": float(
            scope_lookup.loc["pooled", "delta_fixed13_minus_parent"]
        )
        <= float(guard_config["maximum_pooled_delta_rmse_vs_parent_fixed12_selector"]),
        "improved_parent_folds": fold_improvements
        >= int(guard_config["minimum_improved_folds_vs_parent_fixed12_selector"]),
        "by_well_p95_nonworse": by_well_p95
        <= float(guard_config["maximum_by_well_p95_delta_rmse_ft"]),
        "worst_well_nonworse": float(worst["delta_fixed13_minus_parent"])
        <= float(guard_config["maximum_worst_well_delta_rmse_ft"]),
    }
    checks.update(
        {
            f"scope_{scope}_nonworse": float(scope_lookup.loc[scope, "delta_fixed13_minus_parent"])
            <= scope_limit
            for scope in required_scopes
        }
    )
    gate = {
        "passed": bool(all(checks.values())),
        "decision": (
            success_decision
            if all(checks.values())
            else str(guard_config["failure_decision"])
        ),
        "checks": checks,
        "fixed_fallback_error_parity_max_abs_ft": fixed_parity_max_abs,
        "fold_improvements_vs_parent": fold_improvements,
        f"positive_{added_label}_usage_folds": positive_usage_folds,
        f"{added_label}_usage_pooled": float(
            usage_lookup.loc["pooled", usage_fraction_column]
        ),
        "by_well_p95_delta_rmse": by_well_p95,
        "worst_well": str(worst["well"]),
        "worst_well_delta_rmse": float(worst["delta_fixed13_minus_parent"]),
        "same_oof_rescue_allowed": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    scope_path = output_dir / f"{output_prefix}_fixed13_vs_fixed12_scope_metrics.csv"
    usage_path = output_dir / f"{output_prefix}_fixed13_candidate_usage.csv"
    by_well_path = output_dir / f"{output_prefix}_fixed13_vs_fixed12_by_well.csv"
    gate_path = output_dir / f"{output_prefix}_scientific_gate.json"
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
    return gate, by_well


def _novelty_metric_rows(
    grouped: pd.DataFrame,
    *,
    granularity: str,
    tie_atol_squared_ft: float,
    added_candidate_id: str = ADDED_CANDIDATE_ID,
    base_primary_ids: Sequence[str] = BASE_PRIMARY_IDS,
    added_label: str = "exp486",
) -> list[dict[str, Any]]:
    index_columns = ["well", "outer_fold", "group_id"]
    sse = grouped.pivot(index=index_columns, columns="candidate_id", values="sse")
    rows = grouped.loc[
        grouped["candidate_id"].eq(added_candidate_id),
        [*index_columns, "rows"],
    ].set_index(index_columns)["rows"]
    if set(sse.columns).intersection(base_primary_ids) != set(base_primary_ids):
        raise ValueError(f"{granularity} novelty lost a fixed12 primary candidate")
    if added_candidate_id not in sse:
        raise ValueError(f"{granularity} novelty lost the {added_label} candidate")
    if sse.isna().any().any() or rows.isna().any():
        raise ValueError(f"{granularity} novelty has incomplete candidate groups")
    base_best = sse.loc[:, list(base_primary_ids)].min(axis=1).to_numpy(np.float64)
    added = sse[added_candidate_id].to_numpy(np.float64)
    group_rows = rows.reindex(sse.index).to_numpy(np.int64)
    folds = sse.index.get_level_values("outer_fold").to_numpy(np.int8)
    records: list[dict[str, Any]] = []
    for scope, selected in (
        ("pooled", np.ones(len(sse), dtype=bool)),
        *((f"fold_{fold}", folds == fold) for fold in range(5)),
    ):
        if not bool(selected.any()):
            raise ValueError(f"empty novelty scope: {granularity}/{scope}")
        selected_rows = int(group_rows[selected].sum())
        base = base_best[selected]
        candidate = added[selected]
        strict = candidate + float(tie_atol_squared_ft) < base
        oracle = np.minimum(base, candidate)
        base_rmse = float(np.sqrt(base.sum() / selected_rows))
        add_one_rmse = float(np.sqrt(oracle.sum() / selected_rows))
        records.append(
            {
                "granularity": granularity,
                "scope": scope,
                "rows": selected_rows,
                "groups": int(selected.sum()),
                "fixed12_primary_oracle_rmse": base_rmse,
                "fixed13_add_one_oracle_rmse": add_one_rmse,
                "oracle_improvement_ft": base_rmse - add_one_rmse,
                "strict_unique_best_groups": int(strict.sum()),
                "strict_unique_best_fraction": float(strict.mean()),
                "strict_unique_best_rows": int(group_rows[selected][strict].sum()),
            }
        )
    return records


def build_postfreeze_addone_novelty_readout(
    *,
    new_score_path: Path,
    output_dir: Path,
    tie_atol_squared_ft: float = 1.0e-12,
    added_candidate_id: str = ADDED_CANDIDATE_ID,
    base_primary_ids: Sequence[str] = BASE_PRIMARY_IDS,
    added_label: str = "exp486",
    output_prefix: str = "exp496",
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(new_score_path)
    columns = [
        "well",
        "well_row_idx",
        "outer_fold",
        "candidate_id",
        "actual_abs_error",
    ]
    minimum_row_by_well: dict[str, int] = {}
    maximum_row_by_well: dict[str, int] = {}
    row_count_by_well: dict[str, int] = {}
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns[:4]).to_pandas()
        added = frame.loc[
            frame["candidate_id"].eq(added_candidate_id),
            ["well", "well_row_idx"],
        ]
        local = added.groupby("well", sort=False)["well_row_idx"].agg(["min", "max", "size"])
        for well, row in local.iterrows():
            key = str(well)
            minimum = int(row["min"])
            maximum = int(row["max"])
            minimum_row_by_well[key] = min(minimum_row_by_well.get(key, minimum), minimum)
            maximum_row_by_well[key] = max(maximum_row_by_well.get(key, maximum), maximum)
            row_count_by_well[key] = row_count_by_well.get(key, 0) + int(row["size"])
    if not minimum_row_by_well:
        raise ValueError(
            f"{added_label} candidate is absent from the frozen selector score"
        )
    noncontiguous = [
        well
        for well, minimum in minimum_row_by_well.items()
        if maximum_row_by_well[well] - minimum + 1 != row_count_by_well[well]
    ]
    if noncontiguous:
        raise ValueError(
            "H512 novelty requires contiguous evaluation rows within each well: "
            f"{noncontiguous[:5]}"
        )

    parts: list[pd.DataFrame] = []
    primary = set((*base_primary_ids, added_candidate_id))
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
        frame = frame.loc[frame["candidate_id"].isin(primary)].copy()
        if frame.empty:
            continue
        errors = pd.to_numeric(frame["actual_abs_error"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(errors).all():
            raise ValueError("post-freeze novelty contains a non-finite error")
        minimum = frame["well"].astype(str).map(minimum_row_by_well)
        within_well = frame["well_row_idx"].to_numpy(np.int64) - minimum.to_numpy(np.int64)
        if bool((within_well < 0).any()):
            raise ValueError("post-freeze novelty block offset is negative")
        frame["group_id"] = (within_well // 512).astype(np.int32)
        frame["sse"] = np.square(errors)
        parts.append(
            frame.groupby(
                ["well", "outer_fold", "group_id", "candidate_id"],
                sort=False,
                observed=True,
            )
            .agg(sse=("sse", "sum"), rows=("sse", "size"))
            .reset_index()
        )
    h512 = (
        pd.concat(parts, ignore_index=True)
        .groupby(
            ["well", "outer_fold", "group_id", "candidate_id"],
            sort=True,
            observed=True,
        )
        .agg(sse=("sse", "sum"), rows=("rows", "sum"))
        .reset_index()
    )
    whole = (
        h512.groupby(
            ["well", "outer_fold", "candidate_id"],
            sort=True,
            observed=True,
        )
        .agg(sse=("sse", "sum"), rows=("rows", "sum"))
        .reset_index()
    )
    whole["group_id"] = 0
    metrics = pd.DataFrame(
        [
            *_novelty_metric_rows(
                h512,
                granularity="h512",
                tie_atol_squared_ft=tie_atol_squared_ft,
                added_candidate_id=added_candidate_id,
                base_primary_ids=base_primary_ids,
                added_label=added_label,
            ),
            *_novelty_metric_rows(
                whole,
                granularity="whole_well",
                tie_atol_squared_ft=tie_atol_squared_ft,
                added_candidate_id=added_candidate_id,
                base_primary_ids=base_primary_ids,
                added_label=added_label,
            ),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{output_prefix}_postfreeze_addone_novelty.csv"
    summary_path = output_dir / f"{output_prefix}_postfreeze_addone_novelty.json"
    metrics.to_csv(metrics_path, index=False)
    pooled = metrics.loc[metrics["scope"].eq("pooled")].set_index("granularity")
    summary = {
        "status": "diagnostic_only_after_selector_prediction_freeze",
        "candidate": added_candidate_id,
        "affects_training_or_scientific_gate": False,
        "tie_atol_squared_ft": float(tie_atol_squared_ft),
        "h512": pooled.loc["h512"].to_dict(),
        "whole_well": pooled.loc["whole_well"].to_dict(),
        "metrics_sha256": sha256_file(metrics_path),
    }
    write_json(summary_path, summary)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def _selector_uncertainty_rows(
    new_score_path: Path,
    *,
    primary_ids: Sequence[str],
    all_candidate_ids: Sequence[str],
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(new_score_path)
    columns = ["id", "candidate_id", "pred_abs_error", "p_within10"]
    parts: list[pd.DataFrame] = []
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(row_group, columns=columns).to_pandas()
        counts = frame.groupby("id", sort=False)["candidate_id"].size()
        if not bool(counts.eq(len(all_candidate_ids)).all()):
            raise ValueError("reranking score row group has incomplete candidates")
        error = frame.pivot(index="id", columns="candidate_id", values="pred_abs_error")
        probability = frame.pivot(index="id", columns="candidate_id", values="p_within10")
        error_values = error.loc[:, list(primary_ids)].to_numpy(np.float64)
        top_two = np.partition(error_values, kth=1, axis=1)[:, :2]
        top_two.sort(axis=1)
        probability_values = probability.loc[:, list(all_candidate_ids)].to_numpy(np.float64)
        normalized = probability_values / np.maximum(
            probability_values.sum(axis=1, keepdims=True), 1.0e-12
        )
        entropy = -np.sum(normalized * np.log(np.clip(normalized, 1.0e-12, 1.0)), axis=1)
        parts.append(
            pd.DataFrame(
                {
                    "id": error.index.astype(str),
                    "pred_abs_error_margin": top_two[:, 1] - top_two[:, 0],
                    "p_within10_candidate_entropy": entropy,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output["id"].duplicated().any():
        raise ValueError("reranking score diagnostic contains duplicate row IDs")
    return output


def _quantile_codes(values: np.ndarray, bins: int) -> tuple[np.ndarray, list[float]]:
    edges = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0.0, 1.0, bins + 1))
    unique = np.unique(edges)
    if len(unique) < 2:
        return np.zeros(len(values), dtype=np.int8), [float(unique[0]), float(unique[0])]
    codes = np.searchsorted(unique[1:-1], values, side="right").astype(np.int8)
    return codes, [float(item) for item in unique]


def build_incumbent_reranking_diagnostic(
    *,
    new_score_path: Path,
    paired: pd.DataFrame,
    by_well: pd.DataFrame,
    contract: Mapping[str, Any],
    output_dir: Path,
    quantile_bins: int,
    added_candidate_id: str = ADDED_CANDIDATE_ID,
    base_candidate_ids: Sequence[str] = BASE_CANDIDATE_IDS,
    base_primary_ids: Sequence[str] = BASE_PRIMARY_IDS,
    added_label: str = "exp486",
    output_prefix: str = "exp496",
    validate_contract: bool = True,
) -> dict[str, Any]:
    if validate_contract:
        validate_fixed13_contract(contract)
    expected_ids = (*base_candidate_ids, added_candidate_id)
    if tuple(candidate_ids(contract)) != expected_ids:
        raise ValueError("reranking contract candidate order changed")
    usage_fraction_column = f"{added_label}_top1_fraction"
    incumbent_rows_column = (
        f"incumbent_change_rows_when_{added_label}_not_top1"
    )
    incumbent_fraction_column = (
        f"incumbent_change_fraction_when_{added_label}_not_top1"
    )
    score = _selector_uncertainty_rows(
        new_score_path,
        primary_ids=(*base_primary_ids, added_candidate_id),
        all_candidate_ids=expected_ids,
    )
    frame = paired.merge(score, on="id", how="inner", validate="one_to_one")
    if len(frame) != len(paired):
        raise ValueError("reranking score diagnostic identity coverage mismatch")
    added_selected = frame["new_selected_candidate"].eq(added_candidate_id)
    frame["incumbent_choice_changed"] = ~added_selected & frame["new_selected_candidate"].ne(
        frame["parent_selected_candidate"]
    )
    margin_codes, margin_edges = _quantile_codes(
        frame["pred_abs_error_margin"].to_numpy(np.float64), int(quantile_bins)
    )
    entropy_codes, entropy_edges = _quantile_codes(
        frame["p_within10_candidate_entropy"].to_numpy(np.float64),
        int(quantile_bins),
    )
    frame["margin_quantile"] = margin_codes
    frame["entropy_quantile"] = entropy_codes

    rows: list[dict[str, Any]] = []
    dimensions = {
        "pooled": np.zeros(len(frame), dtype=np.int8),
        "margin_quantile": margin_codes,
        "entropy_quantile": entropy_codes,
    }
    for dimension, codes in dimensions.items():
        for code in sorted(np.unique(codes).tolist()):
            selected = codes == code
            part = frame.loc[selected]
            part_added = part["new_selected_candidate"].eq(added_candidate_id)
            eligible = ~part_added
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": int(code),
                    "rows": len(part),
                    usage_fraction_column: float(part_added.mean()),
                    incumbent_rows_column: int(
                        part.loc[eligible, "incumbent_choice_changed"].sum()
                    ),
                    incumbent_fraction_column: float(
                        part.loc[eligible, "incumbent_choice_changed"].mean()
                    )
                    if bool(eligible.any())
                    else 0.0,
                    "fixed13_hard_rmse": _rmse_from_abs_error(part["new_selected_abs_error"]),
                    "parent_fixed12_hard_rmse": _rmse_from_abs_error(
                        part["parent_selected_abs_error"]
                    ),
                    "delta_fixed13_minus_parent": _rmse_from_abs_error(
                        part["new_selected_abs_error"]
                    )
                    - _rmse_from_abs_error(part["parent_selected_abs_error"]),
                }
            )
    diagnostic = pd.DataFrame(rows)
    zero_usage = by_well[usage_fraction_column].eq(0.0)
    usage = by_well[usage_fraction_column].astype(float)
    delta = by_well["delta_fixed13_minus_parent"].astype(float)

    def safe_correlation(method: str) -> float | None:
        if usage.nunique(dropna=True) <= 1 or delta.nunique(dropna=True) <= 1:
            return None
        value = float(usage.corr(delta, method=method))
        return value if np.isfinite(value) else None

    nonadded = ~added_selected
    summary = {
        "status": "diagnostic_only_after_selector_and_gate_freeze",
        "affects_training_or_scientific_gate": False,
        "score_margin_definition": "primary_pred_abs_error_top2_minus_top1",
        "entropy_definition": "all_candidate_p_within10_normalized_shannon_entropy",
        "quantile_bins_requested": int(quantile_bins),
        "margin_edges": margin_edges,
        "entropy_edges": entropy_edges,
        f"pooled_incumbent_change_fraction_when_{added_label}_not_top1": (
            float(frame.loc[nonadded, "incumbent_choice_changed"].mean())
            if bool(nonadded.any())
            else None
        ),
        "usage_delta_pearson": safe_correlation("pearson"),
        "usage_delta_spearman": safe_correlation("spearman"),
        "zero_usage_wells": int(zero_usage.sum()),
        "zero_usage_improved_wells": int(
            (zero_usage & by_well["delta_fixed13_minus_parent"].lt(0.0)).sum()
        ),
        "zero_usage_regressed_wells": int(
            (zero_usage & by_well["delta_fixed13_minus_parent"].gt(0.0)).sum()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / f"{output_prefix}_incumbent_reranking_diagnostic.csv"
    summary_path = output_dir / f"{output_prefix}_incumbent_reranking_summary.json"
    diagnostic.to_csv(diagnostic_path, index=False)
    summary["diagnostic_sha256"] = sha256_file(diagnostic_path)
    write_json(summary_path, summary)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def write_exp496_input_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    exp486_manifest: Mapping[str, Any],
    parent_score_path: Path,
) -> None:
    payload = {
        "experiment": config["experiment"]["name"],
        "candidate_order": candidate_ids(contract),
        "primary_domain": contract["legal_domains"]["primitive_pair_bank"]["candidates"],
        "fixed_domain": contract["legal_domains"]["primitive_fixed_bank"]["candidates"],
        "execution": config["execution"],
        "exp486_target_free_inputs": dict(exp486_manifest),
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
    "EXP486_LEDGER_ALLOWLIST",
    "EXP486_LEDGER_FIELDS",
    "EXP486_PREDICTION_ALLOWLIST",
    "Exp486Fixed13CandidateCache",
    "base_exp264_contract",
    "build_fixed13_integration_readout",
    "build_incumbent_reranking_diagnostic",
    "build_postfreeze_addone_novelty_readout",
    "exp486_content_sha256",
    "load_exp486_target_free_inputs",
    "pair_selector_scores",
    "resolve_csv_by_payload_sha",
    "resolve_file_by_sha",
    "sha256_csv_payload",
    "validate_fixed13_contract",
    "write_exp496_input_contract",
]
