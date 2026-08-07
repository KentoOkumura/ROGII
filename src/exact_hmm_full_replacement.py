from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.candidate_selector_pipeline import (
    KEY_COLUMNS,
    Exp263CandidateCache,
    FoldBundle,
    build_candidate_long_features,
    build_raw_context,
    candidate_ids,
    compact_feature_names,
    contract_by_id,
    deterministic_sample_indices,
    logical_frame_sha256,
    sha256_file,
    sha256_json,
    validate_candidate_contract,
    write_json,
)
from src.exp374_fixed13_candidate_cache import load_exp374_predictions
from src.exp389_fixed13_candidate_cache import (
    BASE_CANDIDATE_IDS,
    BASE_FIXED_IDS,
    BASE_PRIMARY_IDS,
    load_exp389_predictions,
    resolve_file_by_sha,
    resolve_parent_score_file,
    summarize_selector_score_parquet,
)

SEMANTIC_SLOT = "exact_hmm"
HUBER_REPLACEMENT_VALUE_SOURCE = "exp389_huber_delta1p345_exact_hmm"
STUDENT_T_REPLACEMENT_VALUE_SOURCE = "exp374_student_t_df4_exact_hmm"
SUPPORTED_REPLACEMENT_VALUE_SOURCES = (
    HUBER_REPLACEMENT_VALUE_SOURCE,
    STUDENT_T_REPLACEMENT_VALUE_SOURCE,
)
# Backward-compatible alias retained for exp492 callers.
REPLACEMENT_VALUE_SOURCE = HUBER_REPLACEMENT_VALUE_SOURCE
CHANGED_CANDIDATES = (
    "exact_hmm",
    "exp226_k16__exact_hmm",
    "likpf_mean__exact_hmm",
    "exp226_w500_50_50",
)
UNCHANGED_CANDIDATES = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "pf_ancc",
    "beam_mean",
    "exp226_k16__selfgr_hmm_a070",
    "exp226_k16__likpf_mean",
    "selfgr_hmm_a070__likpf_mean",
)


def validate_fixed12_replacement_contract(
    contract: Mapping[str, Any],
    *,
    expected_replacement_value_source: str | None = None,
) -> dict[str, Any]:
    """Validate a frozen exp264 fixed12 exact-HMM replacement contract."""

    validate_candidate_contract(contract)
    ids = tuple(candidate_ids(contract))
    if ids != BASE_CANDIDATE_IDS:
        raise ValueError(f"fixed12 candidate order mismatch: {ids}")
    if len(set(ids)) != 12:
        raise ValueError("fixed12 candidate IDs must be unique")

    domains = contract["legal_domains"]
    primary = tuple(domains["primitive_pair_bank"]["candidates"])
    fixed = tuple(domains["primitive_fixed_bank"]["candidates"])
    if primary != BASE_PRIMARY_IDS:
        raise ValueError("fixed12 primary domain differs from exp264")
    if fixed != BASE_FIXED_IDS:
        raise ValueError("fixed12 fallback domain differs from exp264")

    replacement = dict(contract["replacement"])
    replacement_value_source = str(replacement.get("new_value_source"))
    if replacement_value_source not in SUPPORTED_REPLACEMENT_VALUE_SOURCES:
        raise ValueError(
            "unsupported fixed12 exact-HMM replacement value source: "
            f"{replacement_value_source!r}"
        )
    if (
        expected_replacement_value_source is not None
        and replacement_value_source
        != str(expected_replacement_value_source)
    ):
        raise ValueError(
            "fixed12 exact-HMM replacement source mismatch: "
            f"{replacement_value_source!r} != "
            f"{expected_replacement_value_source!r}"
        )

    expected_replacement = {
        "semantic_slot_id": SEMANTIC_SLOT,
        "old_value_source": "exp209_gaussian_exact_hmm",
        "new_value_source": replacement_value_source,
        "candidate_id_changes": 0,
        "candidate_order_changes": 0,
        "added_candidates": 0,
        "removed_candidates": 0,
        "changed_value_candidates": 4,
        "unchanged_value_candidates": 8,
    }
    for key, expected in expected_replacement.items():
        if replacement.get(key) != expected:
            raise ValueError(
                f"fixed12 replacement contract changed: {key}="
                f"{replacement.get(key)!r} != {expected!r}"
            )

    specs = contract_by_id(contract)
    changed = tuple(
        name
        for name in ids
        if str(specs[name].get("value_status")) == "changed"
    )
    unchanged = tuple(
        name
        for name in ids
        if str(specs[name].get("value_status")) == "unchanged"
    )
    if changed != CHANGED_CANDIDATES:
        raise ValueError(f"changed candidate inventory mismatch: {changed}")
    if unchanged != UNCHANGED_CANDIDATES:
        raise ValueError(f"unchanged candidate inventory mismatch: {unchanged}")
    if (
        str(specs[SEMANTIC_SLOT].get("semantic_value_source"))
        != replacement_value_source
    ):
        raise ValueError(
            "exact_hmm semantic value source differs from the replacement "
            "contract"
        )

    expected_formulas = {
        "exp226_k16__exact_hmm": (
            ("exp226_k16", "exact_hmm"),
            (0.5, 0.5),
        ),
        "likpf_mean__exact_hmm": (
            ("likpf_mean", "exact_hmm"),
            (0.5, 0.5),
        ),
        "exp226_w500_50_50": (
            ("exp226_k16", "likpf_mean", "exact_hmm"),
            (0.5, 0.25, 0.25),
        ),
    }
    for name, (parents, weights) in expected_formulas.items():
        observed_parents = tuple(str(item) for item in specs[name]["parents"])
        observed_weights = tuple(float(item) for item in specs[name]["weights"])
        if observed_parents != parents or observed_weights != weights:
            raise ValueError(f"replacement formula changed for {name}")

    expected_native_confidence = {
        HUBER_REPLACEMENT_VALUE_SOURCE: {
            "sigma_tvt": (
                "huber_delta1p345_on_exp209_absolute_tvt_hmm_std"
            ),
            "source_loglik": (
                "huber_delta1p345_on_exp209_absolute_tvt_hmm_loglik"
            ),
        },
        STUDENT_T_REPLACEMENT_VALUE_SOURCE: {
            "sigma_tvt": (
                "student_t_df4_on_exp209_absolute_tvt_hmm_std"
            ),
            "source_loglik": (
                "student_t_df4_on_exp209_absolute_tvt_hmm_loglik"
            ),
        },
    }[replacement_value_source]
    confidence = dict(contract["confidence_contract"][SEMANTIC_SLOT])
    for key, expected in expected_native_confidence.items():
        if str(confidence.get(key)) != expected:
            raise ValueError(
                f"exact_hmm native confidence mapping changed: {key}"
            )

    encoding = dict(contract["candidate_id_model_encoding"])
    if int(encoding["width"]) != 12:
        raise ValueError("fixed12 candidate one-hot width must remain 12")
    return {
        "candidate_order": list(ids),
        "primary_domain": list(primary),
        "fixed_domain": list(fixed),
        "changed_candidates": list(changed),
        "unchanged_candidates": list(unchanged),
        "semantic_slot": SEMANTIC_SLOT,
        "replacement_value_source": replacement_value_source,
    }


def replacement_cost_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    execution = dict(config["execution"])
    observed = {
        "active_variants": int(execution["active_variants"]),
        "objectives": int(execution["lightgbm_objectives"]),
        "outer_folds": int(execution["outer_folds"]),
        "inner_folds": int(execution["inner_folds"]),
        "planned_cpu_selector_boosters": int(
            execution["planned_cpu_selector_boosters"]
        ),
        "parent_control_retraining": bool(
            execution["parent_control_retraining"]
        ),
        "gpu_boosters": int(execution["gpu_boosters"]),
        "downstream_tvt_training": bool(execution["downstream_tvt_training"]),
        "inference": bool(execution["inference"]),
        "submission": bool(execution["submission"]),
    }
    expected = {
        "active_variants": 1,
        "objectives": 2,
        "outer_folds": 5,
        "inner_folds": 4,
        "planned_cpu_selector_boosters": 40,
        "parent_control_retraining": False,
        "gpu_boosters": 0,
        "downstream_tvt_training": False,
        "inference": False,
        "submission": False,
    }
    if observed != expected:
        raise ValueError(
            f"{config['experiment']['name']} execution contract changed: "
            f"{observed}"
        )
    trained_boosters = int(execution["trained_cpu_boosters"])
    if trained_boosters not in {
        0,
        expected["planned_cpu_selector_boosters"],
    }:
        raise ValueError(
            f"{config['experiment']['name']} must record either zero "
            "pre-run boosters or the complete frozen 40-booster run, got "
            f"{trained_boosters}"
        )
    return observed


def load_huber_replacement_predictions(
    path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_file_sha256: str | None,
    expected_decompressed_sha256: str,
    expected_post_read_prediction_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, manifest = load_exp389_predictions(
        path,
        expected_rows=expected_rows,
        expected_wells=expected_wells,
        expected_file_sha256=expected_file_sha256,
        expected_decompressed_sha256=expected_decompressed_sha256,
        expected_prediction_logical_sha256=expected_decompressed_sha256,
    )
    observed = str(manifest["post_read_prediction_content_sha256"])
    if observed != str(expected_post_read_prediction_sha256):
        raise ValueError(
            "exp389 post-read prediction content SHA mismatch: "
            f"{observed} != {expected_post_read_prediction_sha256}"
        )
    manifest["post_read_prediction_content_sha_verified"] = True
    return frame, manifest


def load_student_t_replacement_predictions(
    path: Path,
    *,
    expected_rows: int,
    expected_wells: int,
    expected_file_sha256: str | None,
    expected_decompressed_sha256: str,
    expected_post_read_prediction_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the target-free exp374 Student-t prediction allowlist."""

    frame, manifest = load_exp374_predictions(
        path,
        expected_rows=expected_rows,
        expected_wells=expected_wells,
        expected_file_sha256=expected_file_sha256,
        expected_decompressed_sha256=expected_decompressed_sha256,
        expected_prediction_logical_sha256=expected_decompressed_sha256,
    )
    observed = str(manifest["post_read_prediction_content_sha256"])
    if observed != str(expected_post_read_prediction_sha256):
        raise ValueError(
            "exp374 post-read prediction content SHA mismatch: "
            f"{observed} != {expected_post_read_prediction_sha256}"
        )
    manifest["post_read_prediction_content_sha_verified"] = True
    return frame, manifest


def build_candidate_bank_from_primitives(
    primitive_values: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> np.ndarray:
    specs = contract_by_id(contract)
    ids = candidate_ids(contract)
    values = {
        str(name): np.asarray(value, dtype=np.float32)
        for name, value in primitive_values.items()
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
                combined
                + np.float32(weight)
                * values[parent].astype(np.float32, copy=False)
            ).astype(np.float32)
        values[name] = combined
    return np.column_stack([values[name] for name in ids]).astype(np.float32)


def _build_candidate_availability(
    primitive_availability: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> np.ndarray:
    specs = contract_by_id(contract)
    ids = candidate_ids(contract)
    available = {
        str(name): np.asarray(value, dtype=bool)
        for name, value in primitive_availability.items()
    }
    for name in ids:
        if name in available:
            continue
        parents = [str(item) for item in specs[name]["parents"]]
        combined = np.ones_like(available[parents[0]], dtype=bool)
        for parent in parents:
            combined &= available[parent]
        available[name] = combined
    return np.column_stack([available[name] for name in ids]).astype(bool)


def _formula_parity_max_abs(
    bank: np.ndarray,
    contract: Mapping[str, Any],
    formula_ids: Sequence[str],
) -> float:
    ids = candidate_ids(contract)
    positions = {name: index for index, name in enumerate(ids)}
    specs = contract_by_id(contract)
    maximum = 0.0
    for name in formula_ids:
        spec = specs[name]
        reconstructed = np.zeros(bank.shape[0], dtype=np.float32)
        for parent, weight in zip(
            spec["parents"], spec["weights"], strict=True
        ):
            reconstructed = (
                reconstructed
                + np.float32(weight)
                * bank[:, positions[str(parent)]].astype(np.float32, copy=False)
            ).astype(np.float32)
        maximum = max(
            maximum,
            float(
                np.abs(
                    reconstructed.astype(np.float64)
                    - bank[:, positions[name]].astype(np.float64)
                ).max(initial=0.0)
            ),
        )
    return maximum


class Exp389Fixed12ReplacementCache:
    """Overlay one frozen exact-HMM source on exp264's semantic slot."""

    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        exp389_predictions: pd.DataFrame,
        exp389_manifest: Mapping[str, Any],
        base_cache: Any | None = None,
    ):
        contract_evidence = validate_fixed12_replacement_contract(contract)
        self.contract = dict(contract)
        self.ids = candidate_ids(contract)
        self.specs = contract_by_id(contract)
        self.replacement_value_source = str(
            contract_evidence["replacement_value_source"]
        )
        if (
            self.replacement_value_source
            == STUDENT_T_REPLACEMENT_VALUE_SOURCE
        ):
            self.replacement_label = "exp374"
            self.confidence_source = (
                "exp374_student_t_df4_exact_hmm_posterior"
            )
        else:
            self.replacement_label = "exp389"
            self.confidence_source = (
                "exp389_huber_delta1p345_exact_hmm_posterior"
            )
        self.base_cache = (
            Exp263CandidateCache(root, contract)
            if base_cache is None
            else base_cache
        )
        self.replacement_source_manifest = dict(exp389_manifest)
        self.replacement_by_key = exp389_predictions.sort_values(
            ["well_id", "row_idx"], kind="stable"
        ).set_index(["well_id", "row_idx"])[
            [
                "candidate_tvt",
                "candidate_std",
                "hmm_loglik",
                "evaluation_rows_in_well",
                "loglik_per_row",
            ]
        ]
        if not self.replacement_by_key.index.is_unique:
            raise ValueError(
                f"{self.replacement_label} global key index is not unique"
            )
        # Backward-compatible attributes retained for the exp492 implementation.
        self.exp389_manifest = self.replacement_source_manifest
        self.exp389_by_key = self.replacement_by_key
        self._fold_audits: dict[int, dict[str, Any]] = {}

    def load_fold(self, fold: int) -> FoldBundle:
        base = self.base_cache.load_fold(int(fold))
        if base.candidate_ids != self.ids:
            raise ValueError("parent cache candidate order differs from fixed12 contract")
        if not np.all(
            base.base["outer_fold"].to_numpy(np.int8) == np.int8(fold)
        ):
            raise ValueError(f"parent cache outer-fold identity mismatch: {fold}")

        selector_keys = pd.MultiIndex.from_arrays(
            [
                base.base["well"].astype(str).to_numpy(),
                base.base["well_row_idx"].to_numpy(np.int64),
            ],
            names=["well_id", "row_idx"],
        )
        replacement = self.replacement_by_key.reindex(selector_keys)
        required = [
            "candidate_tvt",
            "candidate_std",
            "hmm_loglik",
            "evaluation_rows_in_well",
            "loglik_per_row",
        ]
        missing = replacement[required].isna().any(axis=1)
        if bool(missing.any()):
            examples = [
                (str(well), int(row))
                for well, row in selector_keys[missing.to_numpy()][:5]
            ]
            raise ValueError(
                f"{self.replacement_label} global key join misses exp263 "
                f"fold {fold} rows: {examples}"
            )

        ids = self.ids
        positions = {name: index for index, name in enumerate(ids)}
        primitive_ids = [
            name for name in ids if str(self.specs[name]["kind"]) == "primitive"
        ]
        primitive_values = {
            name: base.values[:, positions[name]].astype(np.float32, copy=False)
            for name in primitive_ids
        }
        replacement_tvt = replacement["candidate_tvt"].to_numpy(np.float32)
        if not np.isfinite(replacement_tvt).all():
            raise ValueError(
                f"{self.replacement_label} fold {fold} replacement is "
                "non-finite"
            )
        primitive_values[SEMANTIC_SLOT] = replacement_tvt
        values = build_candidate_bank_from_primitives(
            primitive_values, self.contract
        )

        primitive_available = {
            name: base.available[:, positions[name]].astype(bool, copy=False)
            for name in primitive_ids
        }
        primitive_available[SEMANTIC_SLOT] = np.ones(len(base.base), dtype=bool)
        available = _build_candidate_availability(
            primitive_available, self.contract
        )

        confidence = dict(base.confidence)
        native = replacement[
            ["candidate_std", "hmm_loglik", "loglik_per_row"]
        ].to_numpy(np.float32)
        valid = (
            np.isfinite(np.column_stack([replacement_tvt, native])).all(axis=1)
            & (native[:, 0] >= 0.0)
        )
        if not bool(valid.all()):
            raise ValueError(
                f"{self.replacement_label} fold {fold} native confidence "
                "is invalid"
            )
        conf = base.base[KEY_COLUMNS].copy()
        conf["candidate_id"] = SEMANTIC_SLOT
        conf["confidence_source"] = self.confidence_source
        conf["confidence_valid"] = valid
        conf["confidence_missing_fields"] = ""
        conf["sigma_tvt"] = native[:, 0]
        conf["source_loglik"] = native[:, 1]
        conf["loglik_per_row"] = native[:, 2]
        confidence[SEMANTIC_SLOT] = conf

        delta = np.abs(
            values.astype(np.float64) - base.values.astype(np.float64)
        )
        unchanged_max = float(
            delta[:, [positions[name] for name in UNCHANGED_CANDIDATES]].max(
                initial=0.0
            )
        )
        changed_max = {
            name: float(delta[:, positions[name]].max(initial=0.0))
            for name in CHANGED_CANDIDATES
        }
        formula_max = _formula_parity_max_abs(
            values, self.contract, CHANGED_CANDIDATES[1:]
        )
        unchanged_available_equal = bool(
            np.array_equal(
                available[
                    :, [positions[name] for name in UNCHANGED_CANDIDATES]
                ],
                base.available[
                    :, [positions[name] for name in UNCHANGED_CANDIDATES]
                ],
            )
        )
        if unchanged_max != 0.0 or not unchanged_available_equal:
            raise AssertionError(
                f"unchanged fixed12 candidate parity failed in fold {fold}"
            )
        if formula_max > 1.0e-6:
            raise AssertionError(
                f"replacement formula parity failed in fold {fold}: {formula_max}"
            )

        self._fold_audits[int(fold)] = {
            "selector_outer_fold": int(fold),
            "rows": len(base.base),
            "wells": int(base.base["well"].nunique()),
            "missing_key_rows": 0,
            "changed_candidate_max_abs_ft": changed_max,
            "unchanged_candidate_max_abs_ft": unchanged_max,
            "unchanged_candidate_availability_equal": unchanged_available_equal,
            "formula_parity_max_abs_ft": formula_max,
            "truth_or_error_columns_loaded": int(
                self.replacement_source_manifest[
                    "truth_or_error_columns_loaded"
                ]
            ),
            "candidate_source_fold": None,
            "source_fold_used_as_model_feature": False,
        }
        return FoldBundle(
            base=base.base,
            values=values,
            available=available,
            confidence=confidence,
            candidate_ids=list(ids),
            specs=dict(self.specs),
        )

    def replacement_manifest(self, *, expected_rows: int) -> dict[str, Any]:
        if set(self._fold_audits) != set(range(5)):
            raise ValueError("all five selector folds must be audited before fit")
        rows = int(sum(item["rows"] for item in self._fold_audits.values()))
        changed_max = {
            name: max(
                float(self._fold_audits[fold]["changed_candidate_max_abs_ft"][name])
                for fold in range(5)
            )
            for name in CHANGED_CANDIDATES
        }
        unchanged_max = max(
            float(self._fold_audits[fold]["unchanged_candidate_max_abs_ft"])
            for fold in range(5)
        )
        formula_max = max(
            float(self._fold_audits[fold]["formula_parity_max_abs_ft"])
            for fold in range(5)
        )
        checks = {
            "all_selector_folds_audited": True,
            "global_key_join_rows_match": rows == int(expected_rows),
            "missing_key_rows_zero": all(
                int(self._fold_audits[fold]["missing_key_rows"]) == 0
                for fold in range(5)
            ),
            "changed_candidates_exactly_four": set(changed_max)
            == set(CHANGED_CANDIDATES),
            "each_changed_candidate_differs_from_parent": all(
                value > 0.0 for value in changed_max.values()
            ),
            "unchanged_candidate_value_parity": unchanged_max == 0.0,
            "unchanged_candidate_availability_parity": all(
                bool(
                    self._fold_audits[fold][
                        "unchanged_candidate_availability_equal"
                    ]
                )
                for fold in range(5)
            ),
            "replacement_formula_parity": formula_max <= 1.0e-6,
            "truth_or_error_columns_loaded_before_feature_freeze": all(
                int(
                    self._fold_audits[fold][
                        "truth_or_error_columns_loaded"
                    ]
                )
                == 0
                for fold in range(5)
            ),
            "target_free_candidate_requires_no_source_fold": not bool(
                self.replacement_source_manifest[
                    "candidate_requires_oof_fold"
                ]
            ),
            "source_fold_not_used_as_model_feature": True,
        }
        return {
            "policy": "global_key_join_then_exp263_selector_fold_repartition",
            "semantic_slot": SEMANTIC_SLOT,
            "replacement_value_source": self.replacement_value_source,
            "candidate_order": list(self.ids),
            "changed_candidates": list(CHANGED_CANDIDATES),
            "unchanged_candidates": list(UNCHANGED_CANDIDATES),
            "rows": rows,
            "selector_fold_rows": {
                str(fold): int(self._fold_audits[fold]["rows"])
                for fold in range(5)
            },
            "changed_candidate_max_abs_ft": changed_max,
            "unchanged_candidate_max_abs_ft": unchanged_max,
            "formula_parity_max_abs_ft": formula_max,
            "fold_audits": [
                self._fold_audits[fold] for fold in range(5)
            ],
            "checks": checks,
            "passed": bool(all(checks.values())),
        }


class Exp374Fixed12ReplacementCache(Exp389Fixed12ReplacementCache):
    """Student-t named adapter for the generic fixed12 replacement cache."""

    def __init__(
        self,
        root: Path,
        contract: Mapping[str, Any],
        *,
        exp374_predictions: pd.DataFrame,
        exp374_manifest: Mapping[str, Any],
        base_cache: Any | None = None,
    ):
        validate_fixed12_replacement_contract(
            contract,
            expected_replacement_value_source=(
                STUDENT_T_REPLACEMENT_VALUE_SOURCE
            ),
        )
        super().__init__(
            root,
            contract,
            exp389_predictions=exp374_predictions,
            exp389_manifest=exp374_manifest,
            base_cache=base_cache,
        )
        self.exp374_manifest = self.replacement_source_manifest
        self.exp374_by_key = self.replacement_by_key


def run_fixed12_stage_a_rebuild(
    *,
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    contract: Mapping[str, Any],
    cache: Exp389Fixed12ReplacementCache,
    raw_train_dir: Path,
    parent_feature_schema_path: Path,
    parent_feature_catalog_path: Path,
    parent_compact_schema_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify the overlay against the frozen 88/74 schemas without refreezing."""

    contract_evidence = validate_fixed12_replacement_contract(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_spec = config["data"]["parent_exp264"]
    expected_files = {
        parent_feature_schema_path: str(
            parent_spec["stage_a_feature_schema_file_sha256"]
        ),
        parent_feature_catalog_path: str(
            parent_spec["stage_a_feature_catalog_file_sha256"]
        ),
        parent_compact_schema_path: str(
            parent_spec["stage_c_compact_schema_file_sha256"]
        ),
    }
    for path, expected_sha in expected_files.items():
        observed = sha256_file(path)
        if observed != expected_sha:
            raise ValueError(
                f"parent frozen schema input SHA mismatch: {path} {observed}"
            )

    feature_schema = json.loads(parent_feature_schema_path.read_text())
    expected_features = [str(item) for item in feature_schema["features"]]
    if len(expected_features) != int(
        config["validation"]["expected_candidate_long_features"]
    ):
        raise ValueError("parent selector schema must contain exactly 88 features")
    if str(feature_schema["feature_schema_sha256"]) != str(
        parent_spec["stage_a_feature_schema_logical_sha256"]
    ):
        raise ValueError("parent selector logical feature SHA mismatch")

    compact_schema = json.loads(parent_compact_schema_path.read_text())
    expected_compact = [str(item) for item in compact_schema["features"]]
    generated_compact = compact_feature_names(contract)
    if expected_compact != generated_compact:
        raise ValueError("replacement compact74 names/order differ from exp264")
    if len(generated_compact) != int(
        config["validation"]["expected_compact_features"]
    ):
        raise ValueError("replacement compact schema must contain 74 features")

    feature_cfg = copy.deepcopy(dict(parent_config["features"]))
    feature_cfg["primary_domain"] = contract["legal_domains"][
        "primitive_pair_bank"
    ]["candidates"]
    feature_cfg["fixed_domain"] = contract["legal_domains"][
        "primitive_fixed_bank"
    ]["candidates"]
    probe_rows: list[dict[str, Any]] = []
    for fold in range(int(config["validation"]["outer_folds"])):
        bundle = cache.load_fold(fold)
        context, truth = build_raw_context(
            bundle.base,
            raw_train_dir,
            feature_cfg,
            require_truth=False,
        )
        if truth is not None:
            raise AssertionError("Stage A replacement probe loaded truth")
        indices = deterministic_sample_indices(
            bundle.base,
            min(1024, len(bundle.base)),
            str(config["experiment"]["name"]),
            "fixed12_replacement_stage_a",
            fold,
        )
        generated, _ = build_candidate_long_features(
            bundle,
            context,
            indices,
            feature_cfg,
        )
        missing_features = [
            feature
            for feature in expected_features
            if feature not in generated.columns
        ]
        if missing_features:
            raise ValueError(
                "replacement does not naturally generate frozen selector features: "
                f"{missing_features}"
            )
        probe = generated.loc[:, expected_features].copy()
        all_missing = [
            feature for feature in expected_features if probe[feature].isna().all()
        ]
        if all_missing:
            raise ValueError(
                "replacement makes selected parent features all-missing: "
                f"{all_missing}"
            )
        probe_rows.append(
            {
                "fold": fold,
                "base_rows": len(indices),
                "candidate_long_rows": len(probe),
                "feature_count": len(probe.columns),
                "naturally_generated_feature_count": len(generated.columns),
                "selected_all_missing_feature_count": 0,
                "content_sha256": logical_frame_sha256(probe),
            }
        )

    overlay = cache.replacement_manifest(
        expected_rows=int(config["validation"]["expected_rows"])
    )
    if not bool(overlay["passed"]):
        raise RuntimeError(f"fixed12 replacement Stage A failed: {overlay}")

    shutil.copy2(parent_feature_schema_path, output_dir / "feature_schema.json")
    shutil.copy2(parent_feature_catalog_path, output_dir / "feature_catalog.csv")
    shutil.copy2(
        parent_compact_schema_path, output_dir / "compact_meta_schema.json"
    )
    semantic_manifest = {
        "schema_version": "1.0.0",
        "status": "fixed12_replacement_stage_a_complete",
        "semantic_slot": SEMANTIC_SLOT,
        "replacement_value_source": contract_evidence[
            "replacement_value_source"
        ],
        "candidate_order": candidate_ids(contract),
        "changed_candidates": list(CHANGED_CANDIDATES),
        "unchanged_candidates": list(UNCHANGED_CANDIDATES),
        "feature_schema_mode": "exact_parent_88_names_and_order_no_refreeze",
        "compact_schema_mode": "exact_parent_74_names_and_order",
        "selector_probe": probe_rows,
        "overlay": overlay,
    }
    semantic_manifest["manifest_logical_sha256"] = sha256_json(
        semantic_manifest
    )
    write_json(
        output_dir / "replacement_semantic_manifest.json", semantic_manifest
    )
    summary = {
        "status": "fixed12_replacement_stage_a_complete",
        "passed": True,
        "models_trained": 0,
        "truth_rows_loaded_before_feature_freeze": 0,
        "candidate_count": 12,
        "changed_candidate_count": 4,
        "unchanged_candidate_count": 8,
        "feature_count": len(expected_features),
        "compact_feature_count": len(expected_compact),
        "selector_probe": probe_rows,
        "overlay": overlay,
        "feature_schema_file_sha256": sha256_file(
            output_dir / "feature_schema.json"
        ),
        "feature_schema_logical_sha256": str(
            feature_schema["feature_schema_sha256"]
        ),
        "feature_catalog_file_sha256": sha256_file(
            output_dir / "feature_catalog.csv"
        ),
        "compact_schema_file_sha256": sha256_file(
            output_dir / "compact_meta_schema.json"
        ),
        "replacement_semantic_manifest_sha256": sha256_file(
            output_dir / "replacement_semantic_manifest.json"
        ),
    }
    write_json(output_dir / "stage_a_summary.json", summary)
    write_json(
        output_dir / "reproducibility_manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "stage_a_inputs_and_features_frozen",
            "experiment": config["experiment"]["name"],
            "seed": config["reproducibility"]["seed"],
            "deterministic_anchor": False,
            "replacement_prediction": dict(
                cache.replacement_source_manifest
            ),
            "candidate_contract": contract_evidence,
            "replacement_semantic_manifest_sha256": summary[
                "replacement_semantic_manifest_sha256"
            ],
            "feature_schema_file_sha256": summary[
                "feature_schema_file_sha256"
            ],
            "feature_schema_logical_sha256": summary[
                "feature_schema_logical_sha256"
            ],
            "feature_catalog_file_sha256": summary[
                "feature_catalog_file_sha256"
            ],
            "compact_schema_file_sha256": summary[
                "compact_schema_file_sha256"
            ],
        },
    )
    return summary


def stage_c_runtime_config(
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exp264 Stage C runtime without changing its model contract."""

    runtime = copy.deepcopy(dict(parent_config))
    runtime["experiment"] = copy.deepcopy(dict(config["experiment"]))
    runtime["validation"]["seed"] = int(config["validation"]["seed"])
    runtime["validation"]["outer_folds"] = int(
        config["validation"]["outer_folds"]
    )
    runtime["validation"]["inner_folds"] = int(
        config["validation"]["inner_folds"]
    )
    runtime["model"]["training"] = copy.deepcopy(
        dict(config["model"]["training"])
    )
    runtime["model"]["lightgbm_common"] = copy.deepcopy(
        dict(config["model"]["lightgbm_common"])
    )
    stage = runtime["model"]["nested_downstream_stage"]
    stage["enabled"] = True
    stage["planned_cpu_selector_boosters"] = 40
    stage["parent_control_retraining"] = False
    runtime["execution"]["stage"] = "nested_compact_meta"
    runtime["execution"]["run_approved"] = True
    return runtime


def _rmse_from_abs_error(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def build_fixed12_replacement_readout(
    *,
    new_score_path: Path,
    parent_score_path: Path,
    hidden_like_assignment_path: Path,
    contract: Mapping[str, Any],
    score_summary: Mapping[str, Any],
    technical_checks: Mapping[str, bool],
    saved_control: Mapping[str, Any],
    guard_config: Mapping[str, Any],
    output_dir: Path,
    artifact_prefix: str = "exp492",
) -> dict[str, Any]:
    """Compare replacement and parent hard selectors after prediction freeze."""

    validate_fixed12_replacement_contract(contract)
    new = summarize_selector_score_parquet(
        new_score_path,
        all_candidate_ids=BASE_CANDIDATE_IDS,
        primary_ids=BASE_PRIMARY_IDS,
    ).rename(
        columns={
            "selected_candidate": "replacement_selected_candidate",
            "selected_abs_error": "replacement_selected_abs_error",
            "fixed_abs_error": "replacement_fixed_abs_error",
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
        parent,
        on="id",
        how="inner",
        validate="one_to_one",
        suffixes=("_replacement", "_parent"),
    )
    if len(joined) != len(new) or len(joined) != len(parent):
        raise ValueError("replacement and parent score identities differ")
    for column in ("well", "outer_fold", "md_since"):
        left = joined[f"{column}_replacement"].to_numpy()
        right = joined[f"{column}_parent"].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"replacement/parent score identity mismatch: {column}"
            )
        joined[column] = left

    assignment = pd.read_csv(
        hidden_like_assignment_path, dtype={"well_id": str}
    ).set_index("well_id")
    scope_masks: dict[str, np.ndarray] = {
        "pooled": np.ones(len(joined), dtype=bool),
        "near_0_250": joined["md_since"].to_numpy(np.float64) <= 250.0,
        "distance_1000_plus": (
            joined["md_since"].to_numpy(np.float64) >= 1000.0
        ),
    }
    for fold in range(5):
        scope_masks[f"fold_{fold}"] = (
            joined["outer_fold"].to_numpy(np.int8) == fold
        )
    hidden_scopes = (
        "verification_like_spatial_role",
        "verification_like_typewell_purged_role",
    )
    for role_column in hidden_scopes:
        role = joined["well"].astype(str).map(assignment[role_column])
        scope_masks[role_column] = role.eq("valid").to_numpy()

    scope_rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks.items():
        if not bool(mask.any()):
            raise ValueError(f"empty replacement audit scope: {scope}")
        replacement_rmse = _rmse_from_abs_error(
            joined.loc[mask, "replacement_selected_abs_error"]
        )
        parent_rmse = _rmse_from_abs_error(
            joined.loc[mask, "parent_selected_abs_error"]
        )
        replacement_fixed_rmse = _rmse_from_abs_error(
            joined.loc[mask, "replacement_fixed_abs_error"]
        )
        parent_fixed_rmse = _rmse_from_abs_error(
            joined.loc[mask, "parent_fixed_abs_error"]
        )
        scope_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "replacement_hard_rmse": replacement_rmse,
                "parent_hard_rmse": parent_rmse,
                "delta_replacement_minus_parent": (
                    replacement_rmse - parent_rmse
                ),
                "replacement_fixed_fallback_rmse": replacement_fixed_rmse,
                "parent_fixed_fallback_rmse": parent_fixed_rmse,
                "delta_fixed_fallback_replacement_minus_parent": (
                    replacement_fixed_rmse - parent_fixed_rmse
                ),
            }
        )
    scope_metrics = pd.DataFrame(scope_rows)

    usage_rows: list[dict[str, Any]] = []
    changed_set = set(CHANGED_CANDIDATES)
    for scope in ("pooled", *(f"fold_{fold}" for fold in range(5))):
        mask = scope_masks[scope]
        replacement_selected = joined.loc[
            mask, "replacement_selected_candidate"
        ].astype(str)
        parent_selected = joined.loc[
            mask, "parent_selected_candidate"
        ].astype(str)
        usage_rows.append(
            {
                "scope": scope,
                "rows": int(mask.sum()),
                "changed_family_top1_rows": int(
                    replacement_selected.isin(changed_set).sum()
                ),
                "changed_family_top1_fraction": float(
                    replacement_selected.isin(changed_set).mean()
                ),
                "selected_candidate_changed_rows": int(
                    (replacement_selected.to_numpy() != parent_selected.to_numpy()).sum()
                ),
                "selected_candidate_changed_fraction": float(
                    (
                        replacement_selected.to_numpy()
                        != parent_selected.to_numpy()
                    ).mean()
                ),
            }
        )
    usage = pd.DataFrame(usage_rows)

    by_well_rows: list[dict[str, Any]] = []
    for well, part in joined.groupby("well", sort=True):
        replacement_rmse = _rmse_from_abs_error(
            part["replacement_selected_abs_error"]
        )
        parent_rmse = _rmse_from_abs_error(
            part["parent_selected_abs_error"]
        )
        by_well_rows.append(
            {
                "well": str(well),
                "rows": len(part),
                "replacement_hard_rmse": replacement_rmse,
                "parent_hard_rmse": parent_rmse,
                "delta_replacement_minus_parent": (
                    replacement_rmse - parent_rmse
                ),
                "changed_family_top1_fraction": float(
                    part["replacement_selected_candidate"]
                    .astype(str)
                    .isin(changed_set)
                    .mean()
                ),
            }
        )
    by_well = pd.DataFrame(by_well_rows)

    lookup = scope_metrics.set_index("scope")
    parent_pooled = float(lookup.loc["pooled", "parent_hard_rmse"])
    parent_fixed = float(
        lookup.loc["pooled", "parent_fixed_fallback_rmse"]
    )
    if (
        abs(parent_pooled - float(saved_control["hard_primary_oof_rmse"]))
        > 1.0e-6
    ):
        raise ValueError("saved exp264 parent hard RMSE parity failed")
    if (
        abs(parent_fixed - float(saved_control["fixed_fallback_oof_rmse"]))
        > 1.0e-6
    ):
        raise ValueError("saved exp264 parent fixed fallback RMSE parity failed")
    parent_fold_parity_max_abs = 0.0
    expected_parent_folds = saved_control.get("hard_primary_fold_rmse")
    if expected_parent_folds is not None:
        if len(expected_parent_folds) != 5:
            raise ValueError("saved exp264 parent fold RMSE inventory changed")
        parent_fold_parity_max_abs = max(
            abs(
                float(lookup.loc[f"fold_{fold}", "parent_hard_rmse"])
                - float(expected_parent_folds[fold])
            )
            for fold in range(5)
        )
        if parent_fold_parity_max_abs > 1.0e-6:
            raise ValueError("saved exp264 parent fold RMSE parity failed")

    fold_improvements = int(
        sum(
            float(
                lookup.loc[
                    f"fold_{fold}", "delta_replacement_minus_parent"
                ]
            )
            < 0.0
            for fold in range(5)
        )
    )
    hidden_max = max(
        float(lookup.loc[scope, "delta_replacement_minus_parent"])
        for scope in hidden_scopes
    )
    by_well_delta = by_well["delta_replacement_minus_parent"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst = by_well.loc[by_well_delta.idxmax()]
    checks = {
        "all_technical_checks": bool(technical_checks)
        and all(bool(value) for value in technical_checks.values()),
        "selector_score_guard": bool(score_summary["score_guard"]["passed"]),
        "pooled_nonworse_than_parent": float(
            lookup.loc["pooled", "delta_replacement_minus_parent"]
        )
        <= float(
            guard_config[
                "maximum_pooled_delta_rmse_vs_parent_fixed12_selector"
            ]
        ),
        "improved_parent_folds": fold_improvements
        >= int(
            guard_config[
                "minimum_improved_folds_vs_parent_fixed12_selector"
            ]
        ),
        "near_nonworse": float(
            lookup.loc["near_0_250", "delta_replacement_minus_parent"]
        )
        <= float(guard_config["maximum_near_0_250_delta_rmse_ft"]),
        "distance_1000_plus_nonworse": float(
            lookup.loc[
                "distance_1000_plus", "delta_replacement_minus_parent"
            ]
        )
        <= float(guard_config["maximum_1000_plus_delta_rmse_ft"]),
        "hidden_like_nonworse": hidden_max
        <= float(guard_config["maximum_hidden_like_delta_rmse_ft"]),
        "by_well_p95_nonworse": by_well_p95
        <= float(guard_config["maximum_by_well_p95_delta_rmse_ft"]),
        "worst_well_nonworse": float(
            worst["delta_replacement_minus_parent"]
        )
        <= float(guard_config["maximum_worst_well_delta_rmse_ft"]),
    }
    pooled_usage = usage.set_index("scope").loc["pooled"]
    gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "parent_hard_rmse": parent_pooled,
        "parent_fold_rmse_parity_max_abs_ft": parent_fold_parity_max_abs,
        "replacement_hard_rmse": float(
            lookup.loc["pooled", "replacement_hard_rmse"]
        ),
        "delta_replacement_minus_parent": float(
            lookup.loc["pooled", "delta_replacement_minus_parent"]
        ),
        "fold_improvements_vs_parent": fold_improvements,
        "maximum_hidden_like_delta_rmse": hidden_max,
        "by_well_p95_delta_rmse": by_well_p95,
        "worst_well": str(worst["well"]),
        "worst_well_delta_rmse": float(
            worst["delta_replacement_minus_parent"]
        ),
        "changed_family_usage": {
            "policy": "report_only",
            "top1_rows": int(pooled_usage["changed_family_top1_rows"]),
            "top1_fraction": float(
                pooled_usage["changed_family_top1_fraction"]
            ),
        },
        "fixed_fallback": {
            "policy": "report_only_because_exact_hmm_value_changes",
            "parent_rmse": parent_fixed,
            "replacement_rmse": float(
                lookup.loc["pooled", "replacement_fixed_fallback_rmse"]
            ),
            "delta_replacement_minus_parent": float(
                lookup.loc[
                    "pooled",
                    "delta_fixed_fallback_replacement_minus_parent",
                ]
            ),
        },
        "pass_action": guard_config["pass_action"],
        "fail_action": guard_config["fail_action"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    scope_path = (
        output_dir
        / f"{artifact_prefix}_fixed12_replacement_scope_metrics.csv"
    )
    usage_path = (
        output_dir / f"{artifact_prefix}_fixed12_replacement_usage.csv"
    )
    by_well_path = (
        output_dir / f"{artifact_prefix}_fixed12_replacement_by_well.csv"
    )
    gate_path = output_dir / f"{artifact_prefix}_scientific_gate.json"
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


def write_fixed12_input_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    replacement_manifest: Mapping[str, Any],
    parent_score_path: Path,
) -> None:
    write_json(
        path,
        {
            "experiment": config["experiment"]["name"],
            "candidate_order": candidate_ids(contract),
            "primary_domain": contract["legal_domains"][
                "primitive_pair_bank"
            ]["candidates"],
            "fixed_domain": contract["legal_domains"][
                "primitive_fixed_bank"
            ]["candidates"],
            "execution": replacement_cost_contract(config),
            "replacement_predictions": dict(replacement_manifest),
            "parent_exp264_score": {
                "path": str(parent_score_path),
                "file_sha256": sha256_file(parent_score_path),
            },
        },
    )


def write_exp492_input_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    replacement_manifest: Mapping[str, Any],
    parent_score_path: Path,
) -> None:
    """Backward-compatible exp492 wrapper."""

    write_fixed12_input_contract(
        path,
        config=config,
        contract=contract,
        replacement_manifest=replacement_manifest,
        parent_score_path=parent_score_path,
    )


__all__ = [
    "CHANGED_CANDIDATES",
    "Exp374Fixed12ReplacementCache",
    "Exp389Fixed12ReplacementCache",
    "HUBER_REPLACEMENT_VALUE_SOURCE",
    "REPLACEMENT_VALUE_SOURCE",
    "SEMANTIC_SLOT",
    "STUDENT_T_REPLACEMENT_VALUE_SOURCE",
    "SUPPORTED_REPLACEMENT_VALUE_SOURCES",
    "UNCHANGED_CANDIDATES",
    "build_candidate_bank_from_primitives",
    "build_fixed12_replacement_readout",
    "load_huber_replacement_predictions",
    "load_student_t_replacement_predictions",
    "replacement_cost_contract",
    "resolve_file_by_sha",
    "resolve_parent_score_file",
    "run_fixed12_stage_a_rebuild",
    "stage_c_runtime_config",
    "validate_fixed12_replacement_contract",
    "write_exp492_input_contract",
    "write_fixed12_input_contract",
]
