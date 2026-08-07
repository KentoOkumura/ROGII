# %% [markdown]
# # exp505 exp490 tau500 fade fixed13 on exp413 — Stage C train candidate
#
# Replace only the raw exp490 slot in exp501's fixed13 selector with the frozen
# alpha=1, tau=500 fade toward exp357. This notebook implements Stage C only:
# deterministic candidate construction, truth-late feature freeze, and the
# unchanged outer-5 / inner-4 dual selector. Stage D, inference, and submission
# remain disabled until the preregistered Stage C gate passes and the user gives
# a separate approval.

# %% [markdown]
# ## Contents
# 1. Imports and immutable boundary
# 2. Notebook-safe runtime and path helpers
# 3. Frozen candidate, feature, and compute contracts
# 4. Tau-500 fade loader and exp263 fold adapter
# 5. Frozen input checks and truth-free Stage A
# 6. Stage C strict-nested dual selector
# 7. Truth-late direct, scope, tail, and reranking readouts
# 8. Feature importance and generated artifacts
# 9. Reproducibility summary and fixed stop

# %% [markdown]
# ## 1. Imports and immutable boundary

# %%
from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    FoldBundle,
    audit_raw_context_availability,
    candidate_ids,
    compact_feature_names,
    logical_frame_sha256,
    read_yaml,
    resolve_existing_path,
    resolve_exp263_cache_root,
    run_stage_a,
    run_stage_c,
    sha256_file,
    write_json,
)
from src.exp486_fixed13_candidate_cache import (
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    build_incumbent_reranking_diagnostic,
    build_postfreeze_addone_novelty_readout,
    resolve_csv_by_payload_sha,
    resolve_file_by_sha,
    sha256_csv_payload,
    summarize_selector_score_parquet,
)

EXPERIMENT_NAME = "exp505_exp490_tau500_fade_fixed13_on_exp413"
ADDED_CANDIDATE_ID = "exp490_tau500_fade_mean_reverting_hmm"
RAW_EXP490_CANDIDATE_ID = "exp490_geometry_mean_reverting_hmm"
EXP490_PREDICTION_COLUMN = "geometry_mean_reverting_hmm"
PARENT_PREDICTION_COLUMN = "exp357_parent_prediction"
DISTANCE_COLUMN = "md_since"
EXP490_NATIVE_FIELDS = (
    "geometry_mean_reverting_delta_mean",
    "geometry_mean_reverting_hmm_std",
)
EXP490_INPUT_ALLOWLIST = (
    "well",
    "row_idx",
    "suffix_offset",
    DISTANCE_COLUMN,
    EXP490_PREDICTION_COLUMN,
    PARENT_PREDICTION_COLUMN,
    *EXP490_NATIVE_FIELDS,
)
EXP490_PREPARED_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    DISTANCE_COLUMN,
    "candidate_tvt",
    "raw_exp490_tvt",
    "parent_tvt",
    *EXP490_NATIVE_FIELDS,
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP505_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()

# %% [markdown]
# ## 2. Notebook-safe runtime and path helpers
#
# Runtime helpers resolve support files from the working tree and config name,
# so the generated notebook does not depend on a source-file location.

# %%
def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def experiment_dir() -> Path:
    candidate = project_root() / "experiments" / EXPERIMENT_NAME
    return candidate if candidate.exists() else Path.cwd()


def find_support_file(filename: str) -> Path:
    for path in (Path.cwd() / filename, experiment_dir() / filename):
        if path.exists():
            return path
    matches = sorted(Path.cwd().rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"{filename} did not resolve uniquely: {matches}")


def runtime_output_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if in_notebook_runtime() and KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_raw_split(split: str) -> Path:
    candidates = [
        project_root() / "data" / "raw" / split,
        Path.cwd() / "data" / "raw" / split,
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / split,
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / split,
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob("*__horizontal_well.csv")):
            return path
    matches = [
        path
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{split}")
        if path.is_dir() and any(path.glob("*__horizontal_well.csv"))
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"raw {split} directory did not resolve: {matches}")


def search_roots() -> list[Path]:
    return [Path.cwd(), project_root(), KAGGLE_INPUT_ROOT, Path("/tmp/kaggle-output")]


CONFIG = read_yaml(find_support_file("config.yaml"))
CONTRACT = read_yaml(find_support_file("candidate_contract.yaml"))
FEATURE_CONTRACT = read_yaml(find_support_file("feature_contract.yaml"))
OUTPUT_DIR = runtime_output_dir()

# %% [markdown]
# ## 3. Frozen candidate, feature, and compute contracts
#
# Implementation approval does not approve a Kaggle run. The execution guard
# requires a later explicit approval for exactly 1 variant, 2 objectives,
# outer 5 x inner 4 = 40 CPU boosters. Parent/control retraining, HMM/PF/Beam
# regeneration, Stage D GPU training, inference, and submission stay at zero.

# %%
def validate_exp505_contract(contract: Mapping[str, Any]) -> None:
    ids = tuple(candidate_ids(contract))
    expected = (*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID)
    if ids != expected or len(set(ids)) != 13:
        raise ValueError(f"exp505 fixed13 candidate order mismatch: {ids}")
    specs = {str(item["id"]): item for item in contract["score_candidates"]}
    added = specs[ADDED_CANDIDATE_ID]
    if str(added.get("kind")) != "tau500_fade_geometry_centered_mean_reverting_offset_hmm":
        raise ValueError("exp505 added-candidate kind changed")
    if float(added.get("alpha")) != 1.0 or float(added.get("tau_ft")) != 500.0:
        raise ValueError("exp505 fade alpha/tau changed")
    if added.get("parents"):
        raise ValueError("fade candidate must not become a score-bank pair formula")
    native = tuple(str(item) for item in added.get("native_confidence", {}).keys())
    if native != EXP490_NATIVE_FIELDS:
        raise ValueError(f"exp505 native confidence changed: {native}")
    domains = contract["legal_domains"]
    if tuple(domains["primitive_pair_bank"]["candidates"]) != (
        *BASE_PRIMARY_IDS,
        ADDED_CANDIDATE_ID,
    ):
        raise ValueError("exp505 primary domain changed")
    if tuple(domains["primitive_fixed_bank"]["candidates"]) != BASE_FIXED_IDS:
        raise ValueError("exp505 fixed fallback changed")
    added_contract = contract["added_candidate_contract"]
    if tuple(added_contract["included"]) != (ADDED_CANDIDATE_ID,):
        raise ValueError("exp505 must contain exactly one fade candidate")
    if bool(added_contract["raw_exp490_retained"]):
        raise ValueError("raw exp490 must not remain as a fourteenth candidate")


def base_exp264_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_exp505_contract(contract)
    base = copy.deepcopy(dict(contract))
    base["score_candidates"] = [
        item for item in base["score_candidates"] if str(item["id"]) != ADDED_CANDIDATE_ID
    ]
    base["legal_domains"]["primitive_pair_bank"]["candidates"] = list(BASE_PRIMARY_IDS)
    base["legal_domains"]["primitive_fixed_bank"]["candidates"] = list(BASE_FIXED_IDS)
    base["candidate_id_model_encoding"]["width"] = 12
    return base


def fade_prediction(
    md_since: np.ndarray,
    parent_prediction: np.ndarray,
    exp490_prediction: np.ndarray,
    *,
    alpha: float = 1.0,
    tau_ft: float = 500.0,
) -> np.ndarray:
    md = np.asarray(md_since, dtype=np.float64)
    parent = np.asarray(parent_prediction, dtype=np.float64)
    candidate = np.asarray(exp490_prediction, dtype=np.float64)
    if float(alpha) != 1.0 or float(tau_ft) != 500.0:
        raise ValueError("exp505 permits only alpha=1 and tau=500")
    if not (np.isfinite(md).all() and np.isfinite(parent).all() and np.isfinite(candidate).all()):
        raise ValueError("fade inputs must be finite")
    if bool((md < 0.0).any()):
        raise ValueError("md_since must be non-negative; clipping is forbidden")
    weight = 1.0 - np.exp(-md / float(tau_ft))
    return parent + float(alpha) * weight * (candidate - parent)


validate_exp505_contract(CONTRACT)
candidate_order = candidate_ids(CONTRACT)
compact_names = compact_feature_names(CONTRACT)
stage_c_execution = CONFIG["execution_contract"]["stage_c"]
cost_contract = {
    "variants": int(stage_c_execution["variants"]),
    "objectives": int(stage_c_execution["objectives"]),
    "outer_folds": int(stage_c_execution["outer_folds"]),
    "inner_folds": int(stage_c_execution["inner_folds"]),
    "planned_cpu_selector_boosters": int(stage_c_execution["planned_cpu_selector_boosters"]),
    "parent_control_retraining_boosters": int(
        CONFIG["execution_contract"]["parent_control_retraining_boosters"]
    ),
    "new_hmm_well_runs": int(CONFIG["execution_contract"]["new_hmm_well_runs"]),
    "new_pf_well_runs": int(CONFIG["execution_contract"]["new_pf_well_runs"]),
    "new_beam_well_runs": int(CONFIG["execution_contract"]["new_beam_well_runs"]),
    "stage_d_enabled": bool(CONFIG["execution_contract"]["stage_d"]["enabled"]),
    "inference_runs": int(CONFIG["execution_contract"]["inference_runs"]),
    "submission_runs": int(CONFIG["execution_contract"]["submission_runs"]),
}
assert cost_contract == {
    "variants": 1,
    "objectives": 2,
    "outer_folds": 5,
    "inner_folds": 4,
    "planned_cpu_selector_boosters": 40,
    "parent_control_retraining_boosters": 0,
    "new_hmm_well_runs": 0,
    "new_pf_well_runs": 0,
    "new_beam_well_runs": 0,
    "stage_d_enabled": False,
    "inference_runs": 0,
    "submission_runs": 0,
}
assert tuple(FEATURE_CONTRACT["pre_feature_source_allowlist"]) == EXP490_INPUT_ALLOWLIST
assert FEATURE_CONTRACT["fold_contract"]["exp490_source_fold_feature_allowed"] is False
assert FEATURE_CONTRACT["fold_contract"]["exp490_source_fold_split_allowed"] is False
assert len(compact_names) == int(CONFIG["features"]["expected_compact_feature_count"])
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": CONFIG["experiment"]["route"],
            "selector_parent": CONFIG["lineage"]["parent"],
            "candidate_order": candidate_order,
            "compact_feature_count": len(compact_names),
            "execution": cost_contract,
            "stage_c_run_approved": bool(stage_c_execution["run_approved"]),
            "stage_d_implemented": False,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 4. Tau-500 fade loader and exp263 fold adapter
#
# The CSV parser reads exactly eight target-free columns. The fade is computed
# before any truth-bearing input is opened, then globally joined by
# `(well,row_idx)`. Both `suffix_offset` and the exp263 `md_since` are checked
# in every selector fold. Native exp490 state diagnostics are retained unchanged;
# candidate shape and bank features are recomputed from the faded prediction.

# %%
def load_exp490_fade_inputs(
    prediction_path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_prediction_gzip_raw_sha256: str,
    expected_prediction_payload_sha256: str,
    alpha: float,
    tau_ft: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    file_sha = sha256_file(prediction_path)
    payload_sha = sha256_csv_payload(prediction_path)
    if payload_sha != str(expected_prediction_payload_sha256):
        raise ValueError("exp490 decompressed payload SHA mismatch")
    if prediction_path.suffix == ".gz" and file_sha != str(expected_prediction_gzip_raw_sha256):
        raise ValueError("exp490 raw gzip SHA mismatch")
    header = pd.read_csv(prediction_path, nrows=0).columns.astype(str).tolist()
    missing = set(EXP490_INPUT_ALLOWLIST) - set(header)
    if missing:
        raise ValueError(f"exp490 fade allowlist columns missing: {sorted(missing)}")
    frame = pd.read_csv(
        prediction_path,
        usecols=list(EXP490_INPUT_ALLOWLIST),
        dtype={
            "well": str,
            "row_idx": np.int32,
            "suffix_offset": np.int32,
            **{column: np.float64 for column in EXP490_INPUT_ALLOWLIST[3:]},
        },
    ).loc[:, list(EXP490_INPUT_ALLOWLIST)]
    faded = fade_prediction(
        frame[DISTANCE_COLUMN].to_numpy(np.float64),
        frame[PARENT_PREDICTION_COLUMN].to_numpy(np.float64),
        frame[EXP490_PREDICTION_COLUMN].to_numpy(np.float64),
        alpha=float(alpha),
        tau_ft=float(tau_ft),
    )
    frame = frame.rename(
        columns={
            "well": "well_id",
            EXP490_PREDICTION_COLUMN: "raw_exp490_tvt",
            PARENT_PREDICTION_COLUMN: "parent_tvt",
        }
    )
    frame["candidate_tvt"] = faded
    frame.insert(0, "id", frame["well_id"].astype(str) + "_" + frame["row_idx"].astype(str))
    key = ["id", "well_id", "row_idx", "suffix_offset"]
    if frame.duplicated(key).any() or frame["id"].duplicated().any():
        raise ValueError("exp490 fade input contains duplicate keys")
    if len(frame) != int(expected_rows) or frame["well_id"].nunique() != int(expected_wells):
        raise ValueError("exp490 fade row/well count mismatch")
    ordered = frame.sort_values(["well_id", "row_idx"], kind="stable").reset_index(drop=True)
    expected_suffix = ordered.groupby("well_id", sort=False).cumcount().to_numpy(np.int64)
    suffix_mismatches = int(
        np.sum(ordered["suffix_offset"].to_numpy(np.int64) != expected_suffix)
    )
    if suffix_mismatches:
        raise ValueError(f"exp490 suffix sequence mismatch rows: {suffix_mismatches}")
    numeric = frame[
        [DISTANCE_COLUMN, "candidate_tvt", "raw_exp490_tvt", "parent_tvt", *EXP490_NATIVE_FIELDS]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("exp490 fade candidate or confidence contains non-finite values")
    if bool((frame["geometry_mean_reverting_hmm_std"] < 0.0).any()):
        raise ValueError("geometry_mean_reverting_hmm_std must be non-negative")
    recomputed = fade_prediction(
        frame[DISTANCE_COLUMN].to_numpy(np.float64),
        frame["parent_tvt"].to_numpy(np.float64),
        frame["raw_exp490_tvt"].to_numpy(np.float64),
        alpha=float(alpha),
        tau_ft=float(tau_ft),
    )
    formula_max_abs = float(
        np.max(np.abs(recomputed - frame["candidate_tvt"].to_numpy(np.float64)))
    )
    frame = frame.loc[:, list(EXP490_PREPARED_COLUMNS)]
    manifest = {
        "prediction_path": str(prediction_path),
        "prediction_header_columns": header,
        "prediction_loaded_columns": list(EXP490_INPUT_ALLOWLIST),
        "loaded_column_count": len(EXP490_INPUT_ALLOWLIST),
        "forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded": 0,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "suffix_sequence_mismatches": suffix_mismatches,
        "prediction_file_sha256": file_sha,
        "prediction_payload_sha256": payload_sha,
        "upstream_source_fold_column_loaded": False,
        "upstream_source_fold_used_as_model_feature": False,
        "fade_alpha": float(alpha),
        "fade_tau_ft": float(tau_ft),
        "fade_formula_recompute_max_abs_ft": formula_max_abs,
        "candidate_and_native_confidence_finite_fraction": float(
            np.isfinite(numeric).all(axis=1).mean()
        ),
        "post_read_content_sha256": logical_frame_sha256(
            frame.sort_values(["well_id", "row_idx"], kind="stable").reset_index(drop=True)
        ),
    }
    return frame, manifest


class Exp505FadeFixed13CandidateCache:
    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        fade_inputs: pd.DataFrame,
        fade_manifest: Mapping[str, Any],
        md_since_atol_ft: float,
    ):
        validate_exp505_contract(contract)
        self.ids = candidate_ids(contract)
        self.specs = {str(item["id"]): dict(item) for item in contract["score_candidates"]}
        self.base_cache = Exp263CandidateCache(root, base_exp264_contract(contract))
        self.fade_manifest = dict(fade_manifest)
        self.md_since_atol_ft = float(md_since_atol_ft)
        self.fade_by_key = fade_inputs.sort_values(
            ["well_id", "row_idx"], kind="stable"
        ).set_index(["well_id", "row_idx"])[
            ["candidate_tvt", "suffix_offset", DISTANCE_COLUMN, *EXP490_NATIVE_FIELDS]
        ]
        if not self.fade_by_key.index.is_unique:
            raise ValueError("exp505 fade global-key index is not unique")
        self._selector_fold_audits: dict[int, dict[str, Any]] = {}

    def load_fold(self, fold: int) -> FoldBundle:
        base = self.base_cache.load_fold(int(fold))
        expected = base.base.sort_values(["well", "well_row_idx"], kind="stable").reset_index()
        if not np.all(expected["outer_fold"].to_numpy(np.int8) == np.int8(fold)):
            raise ValueError(f"exp263 fold identity mismatch in fold {fold}")
        selector_keys = pd.MultiIndex.from_arrays(
            [
                expected["well"].astype(str).to_numpy(),
                expected["well_row_idx"].to_numpy(np.int64),
            ],
            names=["well_id", "row_idx"],
        )
        added = self.fade_by_key.reindex(selector_keys)
        required = ["candidate_tvt", "suffix_offset", DISTANCE_COLUMN, *EXP490_NATIVE_FIELDS]
        missing = added[required].isna().any(axis=1)
        if missing.any():
            raise ValueError(f"exp505 fade global-key join misses fold {fold} rows")
        expected_suffix = expected.groupby("well", sort=False).cumcount().to_numpy(np.int64)
        suffix_mismatches = int(
            np.sum(added["suffix_offset"].to_numpy(np.int64) != expected_suffix)
        )
        md_delta = np.abs(
            added[DISTANCE_COLUMN].to_numpy(np.float64)
            - expected["md_since"].to_numpy(np.float64)
        )
        md_max_abs = float(md_delta.max(initial=0.0))
        md_mismatches = int(np.sum(md_delta > self.md_since_atol_ft))
        if suffix_mismatches or md_mismatches:
            raise ValueError(
                f"exp490/exp263 parity failed in fold {fold}: "
                f"suffix={suffix_mismatches}, md_since={md_mismatches}, max={md_max_abs}"
            )
        inverse = np.empty(len(expected), dtype=np.int64)
        inverse[expected["index"].to_numpy(np.int64)] = np.arange(len(expected))
        prediction = added["candidate_tvt"].to_numpy(np.float32)[inverse]
        native = added[list(EXP490_NATIVE_FIELDS)].to_numpy(np.float32)[inverse]
        valid = np.isfinite(np.column_stack([prediction, native])).all(axis=1)
        valid &= native[:, 1] >= 0.0
        if not bool(valid.all()):
            raise ValueError(f"exp505 fold {fold} candidate/confidence is invalid")
        values = np.column_stack([base.values, prediction]).astype(np.float32)
        available = np.column_stack(
            [base.available, np.ones(len(base.base), dtype=bool)]
        ).astype(bool)
        confidence = dict(base.confidence)
        conf = base.base[KEY_COLUMNS].copy()
        conf["candidate_id"] = ADDED_CANDIDATE_ID
        conf["confidence_source"] = "exp490_target_free_state_with_tau500_fade_prediction"
        conf["confidence_valid"] = valid
        conf["confidence_missing_fields"] = ""
        for position, field in enumerate(EXP490_NATIVE_FIELDS):
            conf[field] = native[:, position]
        confidence[ADDED_CANDIDATE_ID] = conf
        self._selector_fold_audits[int(fold)] = {
            "selector_outer_fold": int(fold),
            "rows": len(expected),
            "wells": int(expected["well"].nunique()),
            "missing_key_rows": 0,
            "suffix_offset_mismatch_rows": suffix_mismatches,
            "md_since_mismatch_rows": md_mismatches,
            "md_since_max_abs_delta_ft": md_max_abs,
            "upstream_source_fold_loaded": False,
            "upstream_source_fold_used_as_model_feature": False,
        }
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
        rows = sum(int(self._selector_fold_audits[fold]["rows"]) for fold in range(5))
        checks = {
            "all_selector_folds_audited": True,
            "global_key_join_rows_match": rows == int(expected_rows),
            "missing_key_rows_zero": all(
                int(self._selector_fold_audits[fold]["missing_key_rows"]) == 0
                for fold in range(5)
            ),
            "suffix_offset_parity": all(
                int(self._selector_fold_audits[fold]["suffix_offset_mismatch_rows"]) == 0
                for fold in range(5)
            ),
            "md_since_parity": all(
                int(self._selector_fold_audits[fold]["md_since_mismatch_rows"]) == 0
                for fold in range(5)
            ),
            "source_fold_not_loaded": not bool(
                self.fade_manifest["upstream_source_fold_column_loaded"]
            ),
            "source_fold_not_used": not bool(
                self.fade_manifest["upstream_source_fold_used_as_model_feature"]
            ),
        }
        return {
            "policy": "global_key_join_then_exp263_selector_fold_repartition",
            "selector_fold_source": "exp263_row_count_balanced_outer_fold",
            "candidate_generation": "saved_exp490_minus_exp357_alpha1_tau500_fade",
            "rows": rows,
            "selector_fold_rows": {
                str(fold): int(self._selector_fold_audits[fold]["rows"])
                for fold in range(5)
            },
            "overlap_by_selector_fold": [
                self._selector_fold_audits[fold] for fold in range(5)
            ],
            "checks": checks,
            "passed": bool(all(checks.values())),
        }


def candidate_bank_novelty_readout(
    cache: Exp505FadeFixed13CandidateCache,
) -> dict[str, Any]:
    def correlation(
        left_all: np.ndarray, right_all: np.ndarray, mask: np.ndarray
    ) -> float | None:
        left = left_all[mask]
        right = right_all[mask]
        if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    rows: list[dict[str, Any]] = []
    for fold in range(5):
        bundle = cache.load_fold(fold)
        fade = bundle.values[:, -1].astype(np.float64)
        near_max_ft = float(CONFIG["features"]["diagnostics"]["near_prefix_max_ft"])
        near = bundle.base["md_since"].to_numpy(np.float64) <= near_max_ft
        for position, candidate_id in enumerate(BASE_CANDIDATE_IDS):
            base = bundle.values[:, position].astype(np.float64)

            rows.append(
                {
                    "outer_fold": fold,
                    "base_candidate": candidate_id,
                    "rows": len(fade),
                    "near_prefix_rows": int(near.sum()),
                    "exact_equal_rows": int(np.equal(fade, base).sum()),
                    "near_prefix_exact_equal_rows": int(np.equal(fade[near], base[near]).sum()),
                    "full_vector_exact_duplicate": bool(np.array_equal(fade, base)),
                    "pearson": correlation(
                        fade, base, np.ones(len(fade), dtype=bool)
                    ),
                    "near_prefix_pearson": correlation(fade, base, near),
                }
            )
    frame = pd.DataFrame(rows)
    path = OUTPUT_DIR / "exp505_fade_candidate_bank_novelty.csv"
    frame.to_csv(path, index=False)
    summary = {
        "status": "truth_free_pre_feature_freeze_diagnostic",
        "near_prefix_definition": (
            f"md_since_le_{CONFIG['features']['diagnostics']['near_prefix_max_ft']:g}_ft"
        ),
        "full_vector_exact_duplicate_pairs": int(frame["full_vector_exact_duplicate"].sum()),
        "near_prefix_exact_equal_rows_max": int(frame["near_prefix_exact_equal_rows"].max()),
        "maximum_full_pearson": float(frame["pearson"].dropna().max()),
        "maximum_near_prefix_pearson": float(frame["near_prefix_pearson"].dropna().max()),
        "table_sha256": sha256_file(path),
    }
    write_json(OUTPUT_DIR / "exp505_fade_candidate_bank_novelty.json", summary)
    return summary

# %% [markdown]
# ## 5. Frozen input checks and truth-free Stage A
#
# The exp490 source is opened through the eight-column allowlist, SHA checked,
# faded, and repartitioned before raw TVT or saved selector scores are read.
# Stage A permits only mechanical all-missing, constant, and exact-duplicate
# feature drops.

# %%
if EXECUTE_NOTEBOOK:
    if not bool(stage_c_execution["enabled"]) or not bool(stage_c_execution["run_approved"]):
        raise RuntimeError(
            "exp505 Stage C Kaggle run is not authorized. Reconfirm exactly "
            "1 variant / 2 objectives / outer5 x inner4 / 40 CPU boosters / "
            "control retraining 0 / HMM-PF-Beam rerun 0 / Stage D GPU 0."
        )
    approved_scope = (
        "stage_c_tau500_fade_fixed13_1_variant_2_objectives_"
        "5_outer_4_inner_40_cpu_boosters_no_control_retraining"
    )
    if stage_c_execution.get("approved_scope") != approved_scope:
        raise RuntimeError("exp505 Stage C approval scope does not match the frozen contract")

    started = time.perf_counter()
    roots = search_roots()
    raw_train_dir = find_raw_split("train")
    raw_test_dir = find_raw_split("test")
    cache_root = resolve_exp263_cache_root(CONFIG, roots)
    source_cfg = CONFIG["data"]["exp490_source"]
    prediction_cfg = source_cfg["prediction"]
    prediction_path = resolve_csv_by_payload_sha(
        prediction_cfg["patterns"],
        roots,
        expected_payload_sha256=prediction_cfg["expected_decompressed_sha256"],
        expected_gzip_raw_sha256=prediction_cfg["expected_raw_gzip_sha256"],
        label="saved exp490 Stage 1 full OOF predictions",
    )
    fade_inputs, fade_manifest = load_exp490_fade_inputs(
        prediction_path,
        expected_rows=int(source_cfg["expected_rows"]),
        expected_wells=int(source_cfg["expected_wells"]),
        expected_prediction_gzip_raw_sha256=prediction_cfg["expected_raw_gzip_sha256"],
        expected_prediction_payload_sha256=prediction_cfg["expected_decompressed_sha256"],
        alpha=float(CONFIG["fade_candidate"]["alpha"]),
        tau_ft=float(CONFIG["fade_candidate"]["tau_ft"]),
    )
    fade_cache = Exp505FadeFixed13CandidateCache(
        cache_root,
        CONTRACT,
        fade_inputs=fade_inputs,
        fade_manifest=fade_manifest,
        md_since_atol_ft=float(
            CONFIG["guards"]["stage_c_technical_requires_all"][
                "md_since_parity_absolute_tolerance_ft"
            ]
        ),
    )

    def fade_cache_factory(
        _root: Path, _contract: Mapping[str, Any]
    ) -> Exp505FadeFixed13CandidateCache:
        if Path(_root) != Path(cache_root):
            raise ValueError("exp505 cache root changed after input freeze")
        if candidate_ids(_contract) != candidate_order:
            raise ValueError("exp505 candidate order changed after input freeze")
        return fade_cache

    for fold in range(5):
        preview = fade_cache.load_fold(fold)
        assert preview.values.shape[1] == 13
        assert preview.available.all()
        assert preview.candidate_ids == candidate_order
        assert set(EXP490_NATIVE_FIELDS).issubset(
            preview.confidence[ADDED_CANDIDATE_ID].columns
        )
    repartition_manifest = fade_cache.selector_repartition_manifest(
        expected_rows=int(CONFIG["validation"]["expected_rows"])
    )
    if not bool(repartition_manifest["passed"]):
        raise RuntimeError(f"exp505 selector repartition failed: {repartition_manifest}")
    novelty_prefreeze = candidate_bank_novelty_readout(fade_cache)
    fade_manifest["selector_fold_repartition"] = repartition_manifest
    write_json(OUTPUT_DIR / "fade_candidate_manifest.json", fade_manifest)
    write_json(OUTPUT_DIR / "exp505_selector_fold_repartition.json", repartition_manifest)

    parent_schema_path = resolve_existing_path(
        CONFIG["data"]["exp251_selected_feature_schema_patterns"], roots
    )
    availability = audit_raw_context_availability(
        raw_train_dir,
        raw_test_dir,
        CONFIG["features"]["raw_context"]["horizontal_numeric_allowlist"],
    )
    availability.to_csv(OUTPUT_DIR / "raw_context_availability_audit.csv", index=False)
    stage_a = run_stage_a(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        parent_schema_path=parent_schema_path,
        cache_factory=fade_cache_factory,
    )
    compact_schema = json.loads((OUTPUT_DIR / "compact_meta_schema.json").read_text())
    if compact_schema["features"] != compact_names:
        raise ValueError("exp505 compact schema differs from the frozen 77-column contract")
    print(json.dumps({"fade_manifest": fade_manifest, "stage_a": stage_a}, indent=2))

# %% [markdown]
# ## 6. Stage C strict-nested dual selector
#
# The unchanged exp501 protocol trains four inner models per objective for each
# outer fold. Outer-train compact rows use inner OOF scores; outer-valid compact
# rows use the four-model ensemble. Exactly 40 CPU boosters and 25 compact
# partitions are allowed.

# %%
if EXECUTE_NOTEBOOK:
    stage_c = run_stage_c(
        config=CONFIG,
        contract=CONTRACT,
        cache_root=cache_root,
        raw_train_dir=raw_train_dir,
        output_dir=OUTPUT_DIR,
        cache_factory=fade_cache_factory,
        hard_readout_enabled=True,
    )
    nested_cfg = CONFIG["model"]["nested_downstream_stage"]
    technical_checks = {
        "model_count": int(stage_c["model_count"]) == 40,
        "compact_partition_count": int(stage_c["compact_partition_count"])
        == int(nested_cfg["expected_compact_partitions"]),
        "compact_rows": int(stage_c["compact_rows"])
        == int(nested_cfg["expected_compact_rows"]),
        "outer_valid_score_long_rows": int(stage_c["outer_valid_score_long_rows"])
        == int(nested_cfg["expected_outer_valid_score_long_rows"]),
        "leakage_audit": bool(stage_c["leakage_audit"]["passed"]),
        "pre_feature_allowlist_exactly_eight": fade_manifest["prediction_loaded_columns"]
        == list(EXP490_INPUT_ALLOWLIST),
        "forbidden_columns_loaded_before_freeze": int(
            fade_manifest["forbidden_truth_error_role_episode_fold_scope_gate_columns_loaded"]
        )
        == 0,
        "fade_formula_parity": float(fade_manifest["fade_formula_recompute_max_abs_ft"])
        <= float(
            CONFIG["guards"]["stage_c_technical_requires_all"][
                "require_fade_formula_parity_max_abs_ft"
            ]
        ),
        "global_key_suffix_md_since_parity": bool(repartition_manifest["passed"]),
        "source_fold_not_loaded_or_used": (
            not bool(fade_manifest["upstream_source_fold_column_loaded"])
            and not bool(fade_manifest["upstream_source_fold_used_as_model_feature"])
        ),
        "raw_and_decompressed_sha": (
            fade_manifest["prediction_file_sha256"]
            == prediction_cfg["expected_raw_gzip_sha256"]
            and fade_manifest["prediction_payload_sha256"]
            == prediction_cfg["expected_decompressed_sha256"]
        ),
    }
    if not all(technical_checks.values()):
        raise RuntimeError(f"exp505 pre-readout technical checks failed: {technical_checks}")
    print(json.dumps({"stage_c": stage_c, "technical_checks": technical_checks}, indent=2))

# %% [markdown]
# ## 7. Truth-late direct, scope, tail, and reranking readouts
#
# Only after the faded score, hard choice, compact77, and their SHA values are
# frozen do we open raw TVT, saved raw-exp501 scores, fixed12 scores, and
# hidden-like assignments. The all-AND progression gate is exactly the frozen
# pooled/fold/scope/usage/tail contract; no tau, threshold, feature, or model
# rescue is attempted after a failure.

# %%
def _rmse(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _rename_summary(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(
        columns={
            "selected_candidate": f"{prefix}_selected_candidate",
            "selected_abs_error": f"{prefix}_selected_abs_error",
            "fixed_abs_error": f"{prefix}_fixed_abs_error",
        }
    )


def pair_stage_c_scores(
    *,
    new_score_path: Path,
    raw_exp501_score_path: Path,
    parent_exp264_score_path: Path,
) -> pd.DataFrame:
    new = _rename_summary(
        summarize_selector_score_parquet(
            new_score_path,
            all_candidate_ids=(*BASE_CANDIDATE_IDS, ADDED_CANDIDATE_ID),
            primary_ids=(*BASE_PRIMARY_IDS, ADDED_CANDIDATE_ID),
        ),
        "new",
    )
    raw = _rename_summary(
        summarize_selector_score_parquet(
            raw_exp501_score_path,
            all_candidate_ids=(*BASE_CANDIDATE_IDS, RAW_EXP490_CANDIDATE_ID),
            primary_ids=(*BASE_PRIMARY_IDS, RAW_EXP490_CANDIDATE_ID),
        ),
        "raw",
    )
    fixed12 = _rename_summary(
        summarize_selector_score_parquet(
            parent_exp264_score_path,
            all_candidate_ids=BASE_CANDIDATE_IDS,
            primary_ids=BASE_PRIMARY_IDS,
        ),
        "fixed12",
    )
    identity = ["id", "well", "well_row_idx", "outer_fold", "md_since"]
    joined = new
    for frame, prefix in (
        (raw, "raw"),
        (fixed12, "fixed12"),
    ):
        joined = joined.merge(
            frame,
            on="id",
            how="inner",
            validate="one_to_one",
            suffixes=("", f"_{prefix}"),
        )
        for column in identity[1:]:
            right = f"{column}_{prefix}"
            if not np.array_equal(joined[column].to_numpy(), joined[right].to_numpy()):
                raise ValueError(f"selector score identity mismatch: {prefix}/{column}")
            joined = joined.drop(columns=right)
    if len(joined) != int(CONFIG["validation"]["expected_rows"]):
        raise ValueError("selector score identity coverage mismatch")
    return joined


def attach_raw_scope_context(frame: pd.DataFrame, raw_train_dir: Path) -> pd.DataFrame:
    output = frame.copy()
    output["raw_gr_observed"] = False
    output["well_missing_fraction"] = np.nan
    for well, positions in output.groupby("well", sort=False).indices.items():
        pos = np.asarray(positions, dtype=np.int64)
        raw = pd.read_csv(Path(raw_train_dir) / f"{well}__horizontal_well.csv", usecols=["GR"])
        row_idx = output.iloc[pos]["well_row_idx"].to_numpy(np.int64)
        observed = np.isfinite(
            pd.to_numeric(raw.iloc[row_idx]["GR"], errors="coerce").to_numpy(np.float64)
        )
        output.loc[pos, "raw_gr_observed"] = observed
        output.loc[pos, "well_missing_fraction"] = float((~observed).mean())
    if output["well_missing_fraction"].isna().any():
        raise ValueError("raw scope context coverage is incomplete")
    output["missing_fraction_high"] = output["well_missing_fraction"].ge(
        float(CONFIG["guards"]["scope_thresholds"]["high_missing_fraction"])
    )
    return output


def direct_fade_readout(fade_inputs: pd.DataFrame, raw_train_dir: Path) -> dict[str, Any]:
    sse = 0.0
    rows = 0
    by_well: list[dict[str, Any]] = []
    for well, part in fade_inputs.groupby("well_id", sort=True):
        raw = pd.read_csv(
            Path(raw_train_dir) / f"{well}__horizontal_well.csv", usecols=["TVT"]
        )
        row_idx = part["row_idx"].to_numpy(np.int64)
        truth = pd.to_numeric(raw.iloc[row_idx]["TVT"], errors="raise").to_numpy(np.float64)
        prediction = part["candidate_tvt"].to_numpy(np.float64)
        error = prediction - truth
        well_sse = float(np.square(error).sum())
        sse += well_sse
        rows += len(part)
        by_well.append(
            {
                "well": str(well),
                "rows": len(part),
                "rmse": np.sqrt(well_sse / len(part)),
            }
        )
    value = float(np.sqrt(sse / rows))
    expected = float(
        CONFIG["guards"]["stage_c_progression_requires_all"]["fade_direct_rmse_expected"]
    )
    result = {
        "rows": rows,
        "wells": len(by_well),
        "rmse": value,
        "expected_exp503_rmse": expected,
        "absolute_delta_vs_expected_ft": abs(value - expected),
        "parity_passed": abs(value - expected)
        <= float(
            CONFIG["guards"]["stage_c_progression_requires_all"][
                "fade_direct_rmse_absolute_tolerance_ft"
            ]
        ),
        "truth_read_stage": "after_selector_score_choice_compact_and_sha_freeze",
    }
    write_json(OUTPUT_DIR / "exp505_fade_direct_prediction_readout.json", result)
    return result


def build_stage_c_gate(
    paired: pd.DataFrame,
    *,
    hidden_like_assignment_path: Path,
    raw_train_dir: Path,
    direct: Mapping[str, Any],
    technical: dict[str, bool],
) -> tuple[dict[str, Any], pd.DataFrame]:
    joined = attach_raw_scope_context(paired, raw_train_dir)
    assignment = pd.read_csv(hidden_like_assignment_path, dtype={"well_id": str}).set_index(
        "well_id"
    )
    scope_thresholds = CONFIG["guards"]["scope_thresholds"]
    masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(joined), dtype=bool),
        "raw_gr_observed": joined["raw_gr_observed"].to_numpy(bool),
        "raw_gr_missing": ~joined["raw_gr_observed"].to_numpy(bool),
        "missing_fraction_high": joined["missing_fraction_high"].to_numpy(bool),
        "distance_0_250": joined["md_since"].to_numpy(np.float64)
        <= float(scope_thresholds["near_max_ft"]),
        "distance_1000_plus": joined["md_since"].to_numpy(np.float64)
        >= float(scope_thresholds["long_min_ft"]),
    }
    for fold in range(5):
        masks[f"fold_{fold}"] = joined["outer_fold"].to_numpy(np.int8) == fold
    for scope, role_column in {
        "hidden_like_spatial": "verification_like_spatial_role",
        "hidden_like_typewell_purged": "verification_like_typewell_purged_role",
    }.items():
        masks[scope] = (
            joined["well"].astype(str).map(assignment[role_column]).eq("valid").to_numpy()
        )

    scope_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    for scope, mask in masks.items():
        if not bool(mask.any()):
            raise ValueError(f"empty exp505 audit scope: {scope}")
        new_rmse = _rmse(joined.loc[mask, "new_selected_abs_error"])
        raw_rmse = _rmse(joined.loc[mask, "raw_selected_abs_error"])
        fixed12_rmse = _rmse(joined.loc[mask, "fixed12_selected_abs_error"])
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "exp505_fade_fixed13_rmse": new_rmse,
                "raw_exp501_fixed13_rmse": raw_rmse,
                "fixed12_rmse": fixed12_rmse,
                "fixed_fallback_rmse": _rmse(joined.loc[mask, "new_fixed_abs_error"]),
                "delta_exp505_minus_raw_exp501": new_rmse - raw_rmse,
                "delta_exp505_minus_fixed12": new_rmse - fixed12_rmse,
                "delta_raw_exp501_minus_fixed12": raw_rmse - fixed12_rmse,
            }
        )
        if scope == "pooled" or scope.startswith("fold_"):
            selected = joined.loc[mask, "new_selected_candidate"].eq(ADDED_CANDIDATE_ID)
            usage_rows.append(
                {
                    "scope": scope,
                    "rows": int(mask.sum()),
                    "fade_top1_rows": int(selected.sum()),
                    "fade_top1_fraction": float(selected.mean()),
                }
            )
    scope_metrics = pd.DataFrame(scope_rows)
    usage = pd.DataFrame(usage_rows)
    by_well_rows: list[dict[str, Any]] = []
    for well, part in joined.groupby("well", sort=True):
        new_rmse = _rmse(part["new_selected_abs_error"])
        raw_rmse = _rmse(part["raw_selected_abs_error"])
        fixed12_rmse = _rmse(part["fixed12_selected_abs_error"])
        fade_selected = part["new_selected_candidate"].eq(ADDED_CANDIDATE_ID)
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "exp505_fade_fixed13_rmse": new_rmse,
                "raw_exp501_fixed13_rmse": raw_rmse,
                "fixed12_rmse": fixed12_rmse,
                "delta_exp505_minus_raw_exp501": new_rmse - raw_rmse,
                "delta_exp505_minus_fixed12": new_rmse - fixed12_rmse,
                "delta_raw_exp501_minus_fixed12": raw_rmse - fixed12_rmse,
                "fade_top1_fraction": float(fade_selected.mean()),
            }
        )
    by_well = pd.DataFrame(by_well_rows)
    scope_lookup = scope_metrics.set_index("scope")
    usage_lookup = usage.set_index("scope")
    guard = CONFIG["guards"]["stage_c_progression_requires_all"]
    fixed_fallback_max_abs = float(
        max(
            np.max(
                np.abs(
                    joined["new_fixed_abs_error"].to_numpy(np.float64)
                    - joined["raw_fixed_abs_error"].to_numpy(np.float64)
                )
            ),
            np.max(
                np.abs(
                    joined["new_fixed_abs_error"].to_numpy(np.float64)
                    - joined["fixed12_fixed_abs_error"].to_numpy(np.float64)
                )
            ),
        )
    )
    technical.update(
        {
            "fixed_fallback_error_parity": fixed_fallback_max_abs == 0.0,
        }
    )
    raw_tail = by_well["delta_raw_exp501_minus_fixed12"]
    new_tail = by_well["delta_exp505_minus_fixed12"]
    raw_p95 = float(raw_tail.quantile(0.95))
    new_p95 = float(new_tail.quantile(0.95))
    raw_worst = float(raw_tail.max())
    new_worst = float(new_tail.max())
    nonworse_folds = int(
        sum(
            float(scope_lookup.loc[f"fold_{fold}", "delta_exp505_minus_raw_exp501"])
            <= float(guard["fold_nonworse_tolerance_ft"])
            for fold in range(5)
        )
    )
    positive_usage_folds = int(
        sum(
            float(usage_lookup.loc[f"fold_{fold}", "fade_top1_fraction"]) > 0.0
            for fold in range(5)
        )
    )
    checks: dict[str, bool] = {
        "technical_all": bool(all(technical.values())),
        "fade_direct_exp503_parity": bool(direct["parity_passed"]),
        "pooled_nonworse_than_raw_exp501": float(
            scope_lookup.loc["pooled", "delta_exp505_minus_raw_exp501"]
        )
        <= -float(guard["minimum_gain_vs_raw_exp501_hard_oof_ft"]),
        "nonworse_folds_vs_raw_exp501": nonworse_folds
        >= int(guard["minimum_nonworse_folds_vs_raw_exp501"]),
        "fade_usage_pooled": float(usage_lookup.loc["pooled", "fade_top1_fraction"])
        >= float(guard["minimum_fade_candidate_top1_fraction"]),
        "fade_usage_folds": positive_usage_folds >= int(guard["minimum_positive_usage_folds"]),
        "by_well_p95_material_reduction": raw_p95 - new_p95
        >= float(guard["minimum_by_well_p95_reduction_vs_raw_exp501_ft"]),
        "worst_well_material_reduction": raw_worst - new_worst
        >= float(guard["minimum_worst_well_reduction_vs_raw_exp501_ft"]),
    }
    scope_limit = float(guard["maximum_scope_delta_rmse_vs_raw_exp501_ft"])
    checks.update(
        {
            f"scope_{scope}_nonworse": float(
                scope_lookup.loc[scope, "delta_exp505_minus_raw_exp501"]
            )
            <= scope_limit
            for scope in guard["required_scopes"]
        }
    )
    passed = bool(all(checks.values()))
    gate = {
        "passed": passed,
        "decision": (
            "PASS_PERMIT_SEPARATE_STAGE_D_IMPLEMENTATION_APPROVAL"
            if passed
            else "FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE"
        ),
        "checks": checks,
        "technical_checks": technical,
        "fixed_fallback_error_parity_max_abs_ft": fixed_fallback_max_abs,
        "fade_direct_prediction": dict(direct),
        "nonworse_folds_vs_raw_exp501": nonworse_folds,
        "positive_fade_usage_folds": positive_usage_folds,
        "fade_usage_pooled": float(usage_lookup.loc["pooled", "fade_top1_fraction"]),
        "raw_exp501_by_well_p95_delta_vs_fixed12_ft": raw_p95,
        "exp505_by_well_p95_delta_vs_fixed12_ft": new_p95,
        "by_well_p95_reduction_ft": raw_p95 - new_p95,
        "raw_exp501_worst_well_delta_vs_fixed12_ft": raw_worst,
        "exp505_worst_well_delta_vs_fixed12_ft": new_worst,
        "worst_well_reduction_ft": raw_worst - new_worst,
        "same_oof_rescue_allowed": False,
        "stage_d_implemented": False,
        "stage_d_run_approved": False,
    }
    scope_path = OUTPUT_DIR / "exp505_fixed13_vs_raw_exp501_scope_metrics.csv"
    usage_path = OUTPUT_DIR / "exp505_fade_candidate_usage.csv"
    by_well_path = OUTPUT_DIR / "exp505_fixed13_vs_raw_exp501_by_well.csv"
    gate_path = OUTPUT_DIR / "stage_c_scientific_gate.json"
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


if EXECUTE_NOTEBOOK:
    new_score_path = OUTPUT_DIR / "nested_outer_valid_candidate_score.parquet"
    raw_control_cfg = CONFIG["data"]["exp501_saved_control"]
    raw_score_path = resolve_file_by_sha(
        raw_control_cfg["score_patterns"],
        roots,
        expected_file_sha256=raw_control_cfg["outer_valid_candidate_score_sha256"],
        label="saved raw-exp501 outer-valid candidate score",
    )
    parent264_cfg = CONFIG["data"]["parent_exp264_stage_c"]
    parent264_score_path = resolve_file_by_sha(
        parent264_cfg["score_patterns"],
        roots,
        expected_file_sha256=parent264_cfg["expected_score_file_sha256"],
        label="saved exp264 outer-valid candidate score",
    )
    hidden_like_path = resolve_existing_path(
        CONFIG["data"]["hidden_like_assignment_patterns"], roots
    )
    if sha256_file(hidden_like_path) != CONFIG["data"]["hidden_like_assignment_expected_sha256"]:
        raise ValueError("hidden-like assignment SHA mismatch")
    paired = pair_stage_c_scores(
        new_score_path=new_score_path,
        raw_exp501_score_path=raw_score_path,
        parent_exp264_score_path=parent264_score_path,
    )
    direct = direct_fade_readout(fade_inputs, raw_train_dir)
    gate, by_well = build_stage_c_gate(
        paired,
        hidden_like_assignment_path=hidden_like_path,
        raw_train_dir=raw_train_dir,
        direct=direct,
        technical=technical_checks,
    )
    novelty = build_postfreeze_addone_novelty_readout(
        new_score_path=new_score_path,
        output_dir=OUTPUT_DIR,
        added_candidate_id=ADDED_CANDIDATE_ID,
        base_primary_ids=BASE_PRIMARY_IDS,
        added_label="fade",
        output_prefix="exp505",
    )
    reranking_paired = paired.assign(
        parent_selected_candidate=paired["raw_selected_candidate"],
        parent_selected_abs_error=paired["raw_selected_abs_error"],
    )
    reranking_by_well = by_well.assign(
        delta_fixed13_minus_parent=by_well["delta_exp505_minus_raw_exp501"]
    )
    reranking = build_incumbent_reranking_diagnostic(
        new_score_path=new_score_path,
        paired=reranking_paired,
        by_well=reranking_by_well,
        contract=CONTRACT,
        output_dir=OUTPUT_DIR,
        quantile_bins=int(CONFIG["features"]["diagnostics"]["reranking_quantile_bins"]),
        added_candidate_id=ADDED_CANDIDATE_ID,
        base_candidate_ids=BASE_CANDIDATE_IDS,
        base_primary_ids=BASE_PRIMARY_IDS,
        added_label="fade",
        output_prefix="exp505",
        validate_contract=False,
    )
    print(
        json.dumps(
            {
                "stage_c_scientific_gate": gate,
                "postfreeze_novelty": novelty,
                "incumbent_reranking": reranking,
            },
            indent=2,
        )
    )

# %% [markdown]
# ## 8. Feature importance and generated artifacts
#
# Mean gain importance is saved and plotted for both selector objectives. It is
# report-only and cannot trigger a post-hoc feature subset.

# %%
if EXECUTE_NOTEBOOK:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for feature-importance plots")
    importance = pd.read_csv(
        OUTPUT_DIR / "nested_feature_importance_by_objective_outer_inner.csv"
    )
    importance_summary = (
        importance.groupby(["objective", "feature", "importance_type"], sort=True)[
            "importance"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "importance_mean",
                "std": "importance_std",
                "count": "model_count",
            }
        )
    )
    importance_summary_path = OUTPUT_DIR / "exp505_feature_importance_summary.csv"
    importance_summary.to_csv(importance_summary_path, index=False)
    gain = importance_summary.loc[importance_summary["importance_type"].eq("gain")].copy()
    gain["rank"] = gain.groupby("objective")["importance_mean"].rank(
        method="first", ascending=False
    )
    top_gain = gain.loc[gain["rank"].le(20)].sort_values(
        ["objective", "importance_mean"], ascending=[True, True]
    )
    objectives = sorted(top_gain["objective"].unique().tolist())
    fig, axes = plt.subplots(1, len(objectives), figsize=(8 * len(objectives), 8))
    if len(objectives) == 1:
        axes = [axes]
    for axis, objective in zip(axes, objectives, strict=True):
        selected = top_gain.loc[top_gain["objective"].eq(objective)]
        axis.barh(selected["feature"], selected["importance_mean"])
        axis.set_title(f"{objective}: mean gain")
        axis.set_xlabel("mean gain across outer/inner models")
    fig.tight_layout()
    importance_plot_path = OUTPUT_DIR / "exp505_feature_importance_top20.png"
    fig.savefig(importance_plot_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(top_gain[["objective", "rank", "feature", "importance_mean"]])

# %% [markdown]
# ## 9. Reproducibility summary and fixed stop
#
# A Stage C fail closes the branch. A pass only permits a separate conversation
# about Stage D implementation and its 15 GPU boosters; this notebook contains
# no Stage D, current-test, inference, or submission path.

# %%
if EXECUTE_NOTEBOOK:
    summary = {
        "status": "kaggle_cpu_stage_c_completed",
        "decision": gate["decision"],
        "stage_c_scientific_gate_passed": bool(gate["passed"]),
        "execution": cost_contract,
        "rows": int(CONFIG["validation"]["expected_rows"]),
        "wells": int(CONFIG["validation"]["expected_wells"]),
        "candidate_count": len(candidate_order),
        "compact_feature_count": len(compact_names),
        "stage_a": stage_a,
        "stage_c": stage_c,
        "technical_checks": technical_checks,
        "scientific_gate": gate,
        "direct_fade_readout": direct,
        "prefreeze_candidate_bank_novelty": novelty_prefreeze,
        "postfreeze_addone_novelty": novelty,
        "incumbent_reranking": reranking,
        "fade_input_manifest": fade_manifest,
        "selector_fold_repartition": repartition_manifest,
        "raw_exp501_score_sha256": sha256_file(raw_score_path),
        "parent_exp264_score_sha256": sha256_file(parent264_score_path),
        "feature_importance_summary_sha256": sha256_file(importance_summary_path),
        "feature_importance_plot_sha256": sha256_file(importance_plot_path),
        "elapsed_seconds": time.perf_counter() - started,
        "clean_independent_validation": False,
        "selection_bias_note": CONFIG["validation"]["selection_bias_note"],
        "deterministic_submission_anchor": False,
        "stage_d_implemented": False,
        "stage_d_gpu_boosters": 0,
        "inference": False,
        "submission": False,
    }
    summary_path = OUTPUT_DIR / "exp505_summary.json"
    write_json(summary_path, summary)
    reproducibility_path = OUTPUT_DIR / "reproducibility_manifest.json"
    reproducibility = json.loads(reproducibility_path.read_text())
    reproducibility.update(
        {
            "exp505_status": summary["status"],
            "decision": summary["decision"],
            "deterministic_submission_anchor": False,
            "fade_input_manifest": fade_manifest,
            "selector_fold_repartition": repartition_manifest,
            "scientific_gate": gate,
            "exp505_summary_sha256": sha256_file(summary_path),
        }
    )
    write_json(reproducibility_path, reproducibility)
    if KAGGLE_WORKING_ROOT.exists():
        write_json(KAGGLE_WORKING_ROOT / "metrics.json", summary)
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
