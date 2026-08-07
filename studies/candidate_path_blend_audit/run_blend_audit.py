from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "studies/candidate_path_blend_audit/outputs"
DEFAULT_WORK = Path("/tmp/candidate_path_blend_audit_work")
N_EXPECTED = 3_783_989
N_FOLDS = 5
CHUNK_ROWS = 100_000

STACKING_LIKE_PATHS = {
    "hmm_lgb_exp148",
    "hmm_exp218_shrink_a050",
    "hmm_exp218_residual_scale",
}
RAWTEST_PATH_STATUSES = {
    "rawtest_inference_exists",
    "rawtest_regeneration_exists",
    "available_in_existing_rawtest_cache",
}

BASE_CACHE = Path(
    "/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v2_stream/"
    "artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str
    layer: str
    family: str
    rawtest_status: str
    note: str = ""


@dataclass(frozen=True)
class ExternalCandidate:
    candidate: Candidate
    column: str
    transform: str = "direct"
    filters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ExternalSource:
    name: str
    paths: tuple[Path, ...]
    candidates: tuple[ExternalCandidate, ...]
    well_column: str
    row_idx_column: str | None = None
    id_column: str | None = None
    anchor_column: str | None = None


def c(
    name: str,
    source: str,
    family: str,
    rawtest_status: str,
    note: str = "",
    layer: str = "candidate_path",
) -> Candidate:
    return Candidate(name, source, layer, family, rawtest_status, note)


BASE_DIRECT = {
    "last_anchor": "last_known_tvt",
    "pf_ancc": "pf_ancc",
    "pf_z": "pf_z",
}
BASE_DELTA = {
    "beam_cons": "beam_cons_d",
    "beam_loose": "beam_loose_d",
    "beam_vcons": "beam_vcons_d",
    "beam_sm5": "beam_sm5_d",
    "beam_vloose": "beam_vloose_d",
    "beam_mid": "beam_mid_d",
    "beam_stiff": "beam_stiff_d",
    "beam_mean": "beam_mean_d",
    "beam_median": "beam_med_d",
    "sc8": "sc8_d",
    "sc15": "sc15_d",
    "sc25": "sc25_d",
    "sc_cons": "sc_cons_d",
    "sc_ens": "sc_ens_d",
    "hyb": "hyb_d",
    "signal_mean": "sig_mean_d",
    "formation_mean": "form_mean_d",
    "tvt_dense": "tvt_dense_d",
    "tvt_densew": "tvt_densew_d",
    "tvt_dense50": "tvt_dense50_d",
    "slope_all": "slp_b_d_all",
    "slope_last50": "slp_b_d_50",
    "likpf_mean": "likpf_mean_d",
}
for _formation in ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]:
    BASE_DIRECT[f"formation_{_formation.lower()}_full"] = f"tvtF_{_formation}"
    BASE_DIRECT[f"formation_{_formation.lower()}_weighted"] = f"tvtFw_{_formation}"
    BASE_DIRECT[f"formation_{_formation.lower()}_last50"] = f"tvtF50_{_formation}"


def base_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    current = {
        "pf_ancc",
        "beam_mean",
        "likpf_mean",
        "sc_ens",
        "hyb",
        "tvt_dense",
        "tvt_densew",
        "tvt_dense50",
    }
    for name in [*BASE_DIRECT, *BASE_DELTA]:
        if name.startswith("beam_"):
            family = "beam"
        elif name.startswith("sc"):
            family = "ncc"
        elif name.startswith("formation_"):
            family = "formation"
        elif name.startswith("tvt_dense"):
            family = "dense"
        elif name.startswith("slope"):
            family = "slope"
        elif name in {"pf_ancc", "pf_z", "likpf_mean"}:
            family = "pf"
        else:
            family = "aggregate_or_anchor"
        note = "exp237 base8" if name in current else "exp072 cache alternative"
        out.append(c(name, "exp072", family, "available_in_existing_rawtest_cache", note))
    return out


def external_sources() -> list[ExternalSource]:
    e209 = Path(
        "/tmp/exp209_blend_audit/artifacts/"
        "exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz"
    )
    e223 = Path(
        "/tmp/kaggle-output/exp223-selfgr-hmm-train-v1/artifacts/"
        "exp223_joint_typewell_self_gr_hmm_likelihood_probe_"
        "joint_typewell_self_gr_hmm_likelihood_probe_train_features.csv.gz"
    )
    e226 = Path(
        "/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/"
        "train_v1/artifacts/"
        "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz"
    )
    e221 = Path(
        "/tmp/kaggle-output/exp221_train_v3/artifacts/"
        "exp221_lgb_oof_gaussian_emission_hmm_on_exp148_"
        "lgb_oof_gaussian_emission_hmm_train_features.csv.gz"
    )
    e225 = Path(
        "/tmp/kaggle-output/exp225-state-known-tvt-self-gr-hmm-emission-train-v1/artifacts/"
        "exp225_state_known_tvt_self_gr_hmm_emission_"
        "state_known_tvt_self_gr_hmm_emission_train_features.csv.gz"
    )
    e231 = Path(
        "/tmp/exp231_v3_output/artifacts/"
        "exp231_same_typewell_horizontal_gr_atlas_gated_hmm_emission_"
        "same_typewell_horizontal_gr_atlas_gated_hmm_emission_train_features.csv.gz"
    )
    e234 = Path(
        "/tmp/kaggle-output/exp234_crossfitted_residual_scale_emission_hmm_on_exp218/train_v1/"
        "artifacts/exp234_crossfitted_residual_scale_emission_hmm_on_exp218_"
        "crossfitted_residual_scale_emission_hmm_train_features.csv.gz"
    )
    e192 = Path(
        "/tmp/kaggle-output/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/"
        "train_v1_stream/artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_"
        "full_replay_cache_pixiux_likpf_hard_window_pct50_public_replay_train_features.csv.gz"
    )
    e103 = Path(
        "/tmp/candidate_blend_source/exp103/artifacts/"
        "exp103_pf_z_xy_likpf_ensemble_parity_candidate_wide.csv.gz"
    )
    e106 = Path(
        "/tmp/candidate_blend_source/exp106/artifacts/"
        "exp106_strict_exp072_pf_z_multiseed_scale_cache_candidate_wide.csv.gz"
    )
    e232_t2 = Path(
        "/tmp/kaggle-output/exp232-temp-t2-v2/artifacts/runs/temp_t2/"
        "exp232_adaptive_robust_likelihood_pf_row_candidates.csv.gz"
    )
    e232_t4 = Path(
        "/tmp/kaggle-output/exp232-temp-t4-v2/artifacts/runs/temp_t4/"
        "exp232_adaptive_robust_likelihood_pf_row_candidates.csv.gz"
    )
    e233_02 = Path(
        "/tmp/kaggle-output/exp233-mix-e02-v3/artifacts/"
        "exp233_adaptive_outlier_mixture_likelihood_pf_row_candidates.csv.gz"
    )
    e233_05 = Path(
        "/tmp/kaggle-output/exp233-mix-e05-v4/artifacts/"
        "exp233_adaptive_outlier_mixture_likelihood_pf_row_candidates.csv.gz"
    )
    medoid_paths = tuple(
        Path(f"/tmp/exp243_outputs/shard{i}/artifacts/exp243_pf_seed_medoids_row_candidates.csv.gz")
        for i in range(4)
    )
    exp148 = Path(
        "/tmp/kaggle-output/exp148_train_v1/artifacts/"
        "exp148_learned_likelihood_fulltrain_addonly_on_exp092_predictions.csv.gz"
    )
    exp193 = ROOT / (
        "experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/"
        "kaggle/output/train_v1/artifacts/"
        "exp193_typewell_late_interval_context_features_addonly_on_exp148_predictions.csv.gz"
    )
    exp218 = Path(
        "/tmp/kaggle-output/exp218_gr_wavelet_rotation_confidence_features_on_exp148/"
        "train_v1/artifacts/exp218_gr_wavelet_rotation_confidence_features_on_exp148_predictions.csv.gz"
    )
    exp237 = Path(
        "/tmp/kaggle-output/exp237_hmm_exp226_candidate_selector_on_exp183/train_v1/artifacts/"
        "exp237_hmm_exp226_candidate_selector_on_exp183_oof_predictions.csv.gz"
    )
    exp251 = Path(
        "/tmp/exp251_v4_output/artifacts/"
        "exp251_raw_test_safe_dual_objective_candidate_ranker_oof_predictions.csv.gz"
    )
    exp255 = Path(
        "/tmp/kaggle-output/exp255_nested_selector_gated_bounded_direct_readout_on_exp238/"
        "train_v1/artifacts/exp255_nested_selector_gated_bounded_direct_readout_on_exp238_selected_oof.csv.gz"
    )
    exp259 = Path(
        "/tmp/exp259_v1_output/artifacts/"
        "exp259_coordinate_equivariance_path_warp_augmentation_training_oof_predictions.csv.gz"
    )
    exp240 = Path(
        "/tmp/candidate_blend_source/exp240/artifacts/"
        "exp240_shrinkage_residual_scale_emission_hmm_on_exp218_"
        "shrinkage_residual_scale_emission_hmm_train_features.csv.gz"
    )

    path_sources = [
        ExternalSource(
            "exp209",
            (e209,),
            (
                ExternalCandidate(
                    c("exact_hmm", "exp209", "hmm", "rawtest_regeneration_exists"), "hmm_mean_tvt"
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp223",
            (e223,),
            (
                ExternalCandidate(
                    c("selfgr_hmm_a070", "exp223", "hmm_selfgr", "rawtest_regeneration_exists"),
                    "hmm_selfgr_boost_only_a070_c100_mean_tvt",
                ),
                ExternalCandidate(
                    c("selfgr_hmm_a150", "exp223", "hmm_selfgr", "diagnostic_only"),
                    "hmm_selfgr_boost_only_a150_c100_mean_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp226",
            (e226,),
            (
                ExternalCandidate(
                    c("exp226_k16", "exp226", "geometry", "rawtest_inference_exists"), "tvt_pred"
                ),
            ),
            "well_id",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp221",
            (e221,),
            (
                ExternalCandidate(
                    c("hmm_lgb_exp148", "exp221", "hmm_ml", "rawtest_inference_exists"),
                    "hmm_lgb_exp148_lgb_mean_s2000_l0500_mean_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp225",
            (e225,),
            (
                ExternalCandidate(
                    c("hmm_state_selfgr", "exp225", "hmm_selfgr", "rejected_no_rawtest"),
                    "hmm_selfgr_state_known_tvt_curve_boost_only_a070_c100_mean_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp231",
            (e231,),
            (
                ExternalCandidate(
                    c("hmm_peer_atlas", "exp231", "hmm_atlas", "rejected_no_rawtest"),
                    "hmm_peer_atlas_a025_mean_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp234",
            (e234,),
            (
                ExternalCandidate(
                    c("hmm_exp218_residual_scale", "exp234", "hmm_ml", "rejected_no_rawtest"),
                    "hmm_lgb_exp218_lgb_mean_band_sf0250_sc4000_l0500_mean_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp192",
            (e192,),
            (
                ExternalCandidate(
                    c("exp192_pf_ancc", "exp192", "pf_hard_window", "train_cache_only"), "pf_ancc"
                ),
                ExternalCandidate(
                    c("exp192_pf_z", "exp192", "pf_hard_window", "train_cache_only"), "pf_z"
                ),
                ExternalCandidate(
                    c("exp192_beam_mean", "exp192", "pf_hard_window", "train_cache_only"),
                    "beam_mean_d",
                    "anchor_plus",
                ),
                ExternalCandidate(
                    c("exp192_beam_sm5", "exp192", "pf_hard_window", "train_cache_only"),
                    "beam_sm5_d",
                    "anchor_plus",
                ),
                ExternalCandidate(
                    c("exp192_likpf", "exp192", "pf_hard_window", "train_cache_only"),
                    "likpf_mean_d",
                    "anchor_plus",
                ),
            ),
            "well",
            id_column="id",
            anchor_column="last_known_tvt",
        ),
        ExternalSource(
            "exp103",
            (e103,),
            tuple(
                ExternalCandidate(
                    c(f"exp103_{col}", "exp103", "xy_likpf", "train_candidate_only"), col
                )
                for col in [
                    "xy_likpf_mean",
                    "xy_likpf_scale_3",
                    "xy_likpf_scale_5",
                    "xy_likpf_scale_8",
                    "xy_likpf_scale_12",
                ]
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp106",
            (e106,),
            tuple(
                ExternalCandidate(
                    c(f"exp106_{col}", "exp106", "pf_z_multiseed", "train_candidate_only"), col
                )
                for col in [
                    "pf_z_ms_mean",
                    "pf_z_ms_best_lik_seed",
                    "pf_z_ms_scale_3",
                    "pf_z_ms_scale_5",
                    "pf_z_ms_scale_8",
                    "pf_z_ms_scale_12",
                ]
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp232_t2",
            (e232_t2,),
            (
                ExternalCandidate(
                    c("pf_temp_t2", "exp232", "robust_pf", "rejected_no_rawtest"), "pf_temp_t2_mean"
                ),
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp232_t4",
            (e232_t4,),
            (
                ExternalCandidate(
                    c("pf_temp_t4", "exp232", "robust_pf", "rejected_no_rawtest"), "pf_temp_t4_mean"
                ),
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp233_e02",
            (e233_02,),
            (
                ExternalCandidate(
                    c("pf_mix_e02", "exp233", "mixture_pf", "rejected_no_rawtest"),
                    "pf_mix_eps_0p02_mean",
                ),
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp233_e05",
            (e233_05,),
            (
                ExternalCandidate(
                    c("pf_mix_e05", "exp233", "mixture_pf", "rejected_no_rawtest"),
                    "pf_mix_eps_0p05_mean",
                ),
            ),
            "well",
            row_idx_column="row_idx",
        ),
        ExternalSource(
            "exp243",
            medoid_paths,
            tuple(
                ExternalCandidate(
                    c(f"pf_medoid_k8_m{i}", "exp243", "pf_medoid", "candidate_only_no_rawtest"),
                    f"pf_seed_medoid_k8_m{i}",
                )
                for i in range(8)
            ),
            "well",
            row_idx_column="row_idx",
        ),
    ]

    output_sources = [
        ExternalSource(
            "exp148_output",
            (exp148,),
            (
                ExternalCandidate(
                    c(
                        "exp148_lgb_mean",
                        "exp148",
                        "ml_output",
                        "rawtest_inference_exists",
                        layer="model_output",
                    ),
                    "pred_tvt",
                    filters=(("model", "lgb_mean"),),
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp193_output",
            (exp193,),
            (
                ExternalCandidate(
                    c(
                        "exp193_lgb_mean",
                        "exp193",
                        "ml_output",
                        "rawtest_inference_exists",
                        layer="model_output",
                    ),
                    "pred_tvt",
                    filters=(("model", "lgb_mean"),),
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp218_output",
            (exp218,),
            (
                ExternalCandidate(
                    c(
                        "exp218_lgb_mean",
                        "exp218",
                        "ml_output",
                        "rawtest_inference_exists",
                        layer="model_output",
                    ),
                    "pred_tvt",
                    filters=(("model", "lgb_mean"),),
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp237_output",
            (exp237,),
            (
                ExternalCandidate(
                    c(
                        "exp237_row_selector",
                        "exp237",
                        "selector_output",
                        "rawtest_parity_failed",
                        layer="model_output",
                    ),
                    "selected_tvt",
                    filters=(("variant", "lgb_candidate_error_ranker"), ("mode", "oof")),
                ),
                ExternalCandidate(
                    c(
                        "exp237_viterbi",
                        "exp237",
                        "selector_output",
                        "rawtest_parity_failed",
                        layer="model_output",
                    ),
                    "selected_tvt",
                    filters=(
                        (
                            "variant",
                            "viterbi_sw200_bias000_jw100_jf025_d0075_std999999_md0000_seg001",
                        ),
                        ("mode", "viterbi"),
                    ),
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp251_output",
            (exp251,),
            (
                ExternalCandidate(
                    c(
                        "exp251_probability_row",
                        "exp251",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "raw_test_regenerated_copcf_probability_rowwise_tvt",
                ),
                ExternalCandidate(
                    c(
                        "exp251_error_row",
                        "exp251",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "raw_test_regenerated_copcf_expected_error_rowwise_tvt",
                ),
                ExternalCandidate(
                    c(
                        "exp251_error_viterbi",
                        "exp251",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "raw_test_regenerated_copcf_expected_error_fixed_viterbi_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp255_output",
            (exp255,),
            (
                ExternalCandidate(
                    c(
                        "exp238_addonly",
                        "exp238",
                        "ml_output",
                        "rawtest_inference_exists_with_parity_caveat",
                        layer="model_output",
                    ),
                    "base_pred_tvt",
                ),
                ExternalCandidate(
                    c(
                        "exp255_assertive",
                        "exp255",
                        "selector_correction_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "prediction_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
        ExternalSource(
            "exp259_output",
            (exp259,),
            (
                ExternalCandidate(
                    c(
                        "exp259_probability_row",
                        "exp259",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "exact_tvt_datum_shift_probability_rowwise_tvt",
                ),
                ExternalCandidate(
                    c(
                        "exp259_error_row",
                        "exp259",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "exact_tvt_datum_shift_expected_error_rowwise_tvt",
                ),
                ExternalCandidate(
                    c(
                        "exp259_error_viterbi",
                        "exp259",
                        "selector_output",
                        "guard_failed_no_inference",
                        layer="model_output",
                    ),
                    "exact_tvt_datum_shift_expected_error_fixed_viterbi_tvt",
                ),
            ),
            "well",
            id_column="id",
        ),
    ]
    exp240_source = ExternalSource(
        "exp240",
        (exp240,),
        (
            ExternalCandidate(
                c(
                    "hmm_exp218_shrink_a050",
                    "exp240",
                    "hmm_ml",
                    "closed_no_rawtest_port",
                ),
                "hmm_lgb_exp218_selected_shrinkage_band_sf0250_sc4000_l0500_mean_tvt",
            ),
        ),
        "well",
        id_column="id",
    )
    # Keep newly recovered sources at the end so an existing memmap can be
    # extended without changing the established candidate column order.
    return path_sources + output_sources + [exp240_source]


def candidate_catalog(sources: list[ExternalSource]) -> list[Candidate]:
    candidates = base_candidates()
    candidates.extend(item.candidate for source in sources for item in source.candidates)
    names = [item.name for item in candidates]
    if len(names) != len(set(names)):
        raise ValueError("duplicate candidate names")
    return candidates


def assign_group_folds(well_names: list[str], counts_by_code: np.ndarray) -> np.ndarray:
    order_by_name = np.argsort(np.asarray(well_names, dtype=object))
    group_counts = counts_by_code[order_by_name]
    size_order = np.argsort(group_counts, kind="stable")[::-1]
    fold_sizes = np.zeros(N_FOLDS, dtype=np.int64)
    fold_by_sorted_group = np.empty(len(well_names), dtype=np.uint8)
    for group_pos in size_order:
        fold = int(np.argmin(fold_sizes))
        fold_sizes[fold] += int(group_counts[group_pos])
        fold_by_sorted_group[group_pos] = fold
    fold_by_code = np.empty(len(well_names), dtype=np.uint8)
    fold_by_code[order_by_name] = fold_by_sorted_group
    return fold_by_code


def prepare_base(
    work: Path, candidates: list[Candidate]
) -> tuple[
    np.memmap, np.memmap, np.memmap, np.memmap, list[str], dict[str, tuple[int, int, int, int]]
]:
    if not BASE_CACHE.exists():
        raise FileNotFoundError(BASE_CACHE)
    k = len(candidates)
    pred = np.memmap(work / "predictions.f32", mode="w+", dtype=np.float32, shape=(N_EXPECTED, k))
    pred[:] = np.nan
    truth = np.memmap(work / "truth.f32", mode="w+", dtype=np.float32, shape=N_EXPECTED)
    md_since = np.memmap(work / "md_since.f32", mode="w+", dtype=np.float32, shape=N_EXPECTED)
    well_code = np.memmap(work / "well_code.u16", mode="w+", dtype=np.uint16, shape=N_EXPECTED)
    row_idx_store = np.memmap(work / "row_idx.i32", mode="w+", dtype=np.int32, shape=N_EXPECTED)
    name_to_col = {item.name: idx for idx, item in enumerate(candidates)}
    usecols = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        *BASE_DIRECT.values(),
        *BASE_DELTA.values(),
    ]
    usecols = list(dict.fromkeys(usecols))
    well_names: list[str] = []
    code_for_well: dict[str, int] = {}
    meta: dict[str, list[int]] = {}
    offset = 0
    for chunk_no, frame in enumerate(
        pd.read_csv(
            BASE_CACHE,
            usecols=usecols,
            chunksize=CHUNK_ROWS,
            dtype={"id": str, "well": str},
        ),
        start=1,
    ):
        n = len(frame)
        end = offset + n
        ids = frame["id"].astype(str)
        rows = ids.str.rsplit("_", n=1).str[-1].astype(np.int32).to_numpy()
        wells = frame["well"].astype(str).to_numpy()
        anchor = frame["last_known_tvt"].to_numpy(np.float32)
        truth[offset:end] = anchor + frame["target"].to_numpy(np.float32)
        md_since[offset:end] = frame["md_since"].to_numpy(np.float32)
        row_idx_store[offset:end] = rows
        chunk_codes = np.empty(n, dtype=np.uint16)
        for well in pd.unique(wells):
            mask = wells == well
            positions = np.flatnonzero(mask)
            if well not in code_for_well:
                code_for_well[well] = len(well_names)
                well_names.append(well)
                meta[well] = [
                    offset + int(positions[0]),
                    int(rows[positions[0]]),
                    0,
                    code_for_well[well],
                ]
            code = code_for_well[well]
            chunk_codes[positions] = code
            meta[well][2] += int(len(positions))
        well_code[offset:end] = chunk_codes
        for name, column in BASE_DIRECT.items():
            pred[offset:end, name_to_col[name]] = frame[column].to_numpy(np.float32)
        for name, column in BASE_DELTA.items():
            pred[offset:end, name_to_col[name]] = anchor + frame[column].to_numpy(np.float32)
        offset = end
        if chunk_no % 10 == 0:
            print(f"[base] rows={offset:,}", flush=True)
    if offset != N_EXPECTED:
        raise ValueError(f"base row count {offset} != {N_EXPECTED}")
    counts = np.bincount(np.asarray(well_code), minlength=len(well_names))
    for well, values in meta.items():
        start, first_row, count, code = values
        if count != int(counts[code]):
            raise ValueError(f"non-contiguous or count mismatch: {well}")
        expected = np.arange(first_row, first_row + count, dtype=np.int32)
        actual = np.asarray(row_idx_store[start : start + count])
        if not np.array_equal(expected, actual):
            raise ValueError(f"row_idx is not contiguous for {well}")
    pred.flush()
    truth.flush()
    md_since.flush()
    well_code.flush()
    row_idx_store.flush()
    return (
        pred,
        truth,
        md_since,
        well_code,
        well_names,
        {key: tuple(value) for key, value in meta.items()},
    )


def global_indices(
    frame: pd.DataFrame, source: ExternalSource, meta: dict[str, tuple[int, int, int, int]]
) -> np.ndarray:
    wells = frame[source.well_column].astype(str).to_numpy()
    if source.row_idx_column is not None:
        rows = pd.to_numeric(frame[source.row_idx_column], errors="raise").to_numpy(np.int32)
    elif source.id_column is not None:
        rows = (
            frame[source.id_column]
            .astype(str)
            .str.rsplit("_", n=1)
            .str[-1]
            .astype(np.int32)
            .to_numpy()
        )
    else:
        raise ValueError(source.name)
    starts = np.fromiter((meta[well][0] for well in wells), dtype=np.int64, count=len(wells))
    firsts = np.fromiter((meta[well][1] for well in wells), dtype=np.int32, count=len(wells))
    counts = np.fromiter((meta[well][2] for well in wells), dtype=np.int32, count=len(wells))
    local = rows - firsts
    if np.any(local < 0) or np.any(local >= counts):
        raise ValueError(f"row outside base span in {source.name}")
    return starts + local.astype(np.int64)


def load_external(
    source: ExternalSource,
    pred: np.memmap,
    candidates: list[Candidate],
    meta: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    for path in source.paths:
        if not path.exists():
            raise FileNotFoundError(path)
    name_to_col = {item.name: idx for idx, item in enumerate(candidates)}
    usecols = {source.well_column}
    if source.row_idx_column:
        usecols.add(source.row_idx_column)
    if source.id_column:
        usecols.add(source.id_column)
    if source.anchor_column:
        usecols.add(source.anchor_column)
    for item in source.candidates:
        usecols.add(item.column)
        usecols.update(key for key, _ in item.filters)
    seen = {item.candidate.name: np.zeros(N_EXPECTED, dtype=bool) for item in source.candidates}
    read_rows = 0
    for path in source.paths:
        text_columns = {source.well_column: str}
        if source.id_column:
            text_columns[source.id_column] = str
        text_columns.update({key: str for item in source.candidates for key, _ in item.filters})
        for frame in pd.read_csv(
            path,
            usecols=list(usecols),
            chunksize=CHUNK_ROWS,
            dtype=text_columns,
        ):
            read_rows += len(frame)
            idx_all = global_indices(frame, source, meta)
            for item in source.candidates:
                mask = np.ones(len(frame), dtype=bool)
                for column, value in item.filters:
                    mask &= frame[column].astype(str).to_numpy() == value
                if not mask.any():
                    continue
                idx = idx_all[mask]
                tracker = seen[item.candidate.name]
                if tracker[idx].any():
                    raise ValueError(f"duplicate rows for {source.name}/{item.candidate.name}")
                values = pd.to_numeric(frame.loc[mask, item.column], errors="coerce").to_numpy(
                    np.float32
                )
                if item.transform == "anchor_plus":
                    if source.anchor_column is None:
                        raise ValueError(source.name)
                    values = values + pd.to_numeric(
                        frame.loc[mask, source.anchor_column], errors="coerce"
                    ).to_numpy(np.float32)
                pred[idx, name_to_col[item.candidate.name]] = values
                tracker[idx] = True
    coverage = {name: int(mask.sum()) for name, mask in seen.items()}
    pred.flush()
    print(f"[external] {source.name}: read={read_rows:,}, coverage={coverage}", flush=True)
    return {
        "source": source.name,
        "paths": [str(path) for path in source.paths],
        "sizes": [path.stat().st_size for path in source.paths],
        "rows_read": read_rows,
        "coverage": coverage,
    }


def calculate_grams(
    pred: np.memmap,
    truth: np.memmap,
    well_code: np.memmap,
    fold_by_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = pred.shape[1]
    fold_grams = np.zeros((N_FOLDS, k, k), dtype=np.float64)
    fold_counts = np.zeros(N_FOLDS, dtype=np.int64)
    finite_counts = np.zeros(k, dtype=np.int64)
    for start in range(0, len(truth), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(truth))
        block = np.asarray(pred[start:end])
        finite_counts += np.isfinite(block).sum(axis=0)
        residual = (
            block.astype(np.float64) - np.asarray(truth[start:end], dtype=np.float64)[:, None]
        )
        row_folds = fold_by_code[np.asarray(well_code[start:end], dtype=np.int64)]
        for fold in range(N_FOLDS):
            mask = row_folds == fold
            fold_counts[fold] += int(mask.sum())
            r = residual[mask]
            fold_grams[fold] += r.T @ r
        if start // CHUNK_ROWS % 10 == 0:
            print(f"[gram] rows={end:,}", flush=True)
    total = fold_grams.sum(axis=0)
    return total, fold_grams, fold_counts, finite_counts


def pair_alpha(g: np.ndarray, i: int, j: int) -> float:
    denom = g[i, i] + g[j, j] - 2.0 * g[i, j]
    if denom <= 1e-12:
        return 0.5
    alpha = (g[i, i] - g[i, j]) / denom
    return float(np.clip(alpha, 0.0, 1.0))


def quadratic_sse(g: np.ndarray, indices: list[int], weights: np.ndarray) -> float:
    sub = g[np.ix_(indices, indices)]
    return float(weights @ sub @ weights)


def pair_table(
    candidates: list[Candidate], total: np.ndarray, folds: np.ndarray, fold_counts: np.ndarray
) -> pd.DataFrame:
    n = int(fold_counts.sum())
    single_rmse = np.sqrt(np.diag(total) / n)
    rows: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(candidates)), 2):
        equal_sse = 0.25 * (total[i, i] + total[j, j] + 2.0 * total[i, j])
        equal_rmse = math.sqrt(equal_sse / n)
        best_single = min(single_rmse[i], single_rmse[j])
        alpha_full = pair_alpha(total, i, j)
        full_rmse = math.sqrt(
            quadratic_sse(total, [i, j], np.array([1.0 - alpha_full, alpha_full])) / n
        )
        grid = np.linspace(0.0, 1.0, 21)
        grid_rmses = [
            math.sqrt(quadratic_sse(total, [i, j], np.array([1.0 - a, a])) / n) for a in grid
        ]
        best_grid_pos = int(np.argmin(grid_rmses))
        fold_weights: list[float] = []
        cross_sse = 0.0
        fold_wins = 0
        for fold in range(N_FOLDS):
            train_g = total - folds[fold]
            alpha = pair_alpha(train_g, i, j)
            fold_weights.append(alpha)
            hold_sse = quadratic_sse(folds[fold], [i, j], np.array([1.0 - alpha, alpha]))
            cross_sse += hold_sse
            parent_sse = min(folds[fold, i, i], folds[fold, j, j])
            fold_wins += int(hold_sse < parent_sse)
        cross_rmse = math.sqrt(cross_sse / n)
        denom_corr = math.sqrt(total[i, i] * total[j, j])
        rows.append(
            {
                "candidate_a": candidates[i].name,
                "candidate_b": candidates[j].name,
                "layer_a": candidates[i].layer,
                "layer_b": candidates[j].layer,
                "family_a": candidates[i].family,
                "family_b": candidates[j].family,
                "rmse_a": single_rmse[i],
                "rmse_b": single_rmse[j],
                "better_single_rmse": best_single,
                "residual_cosine": total[i, j] / denom_corr if denom_corr else np.nan,
                "equal_50_rmse": equal_rmse,
                "equal_50_delta_vs_better": equal_rmse - best_single,
                "full_oof_opt_weight_b": alpha_full,
                "full_oof_opt_rmse_diagnostic": full_rmse,
                "grid05_weight_b_diagnostic": grid[best_grid_pos],
                "grid05_rmse_diagnostic": grid_rmses[best_grid_pos],
                "crossfit_weight_b_mean": np.mean(fold_weights),
                "crossfit_weight_b_std": np.std(fold_weights),
                "crossfit_weight_b_min": np.min(fold_weights),
                "crossfit_weight_b_max": np.max(fold_weights),
                "crossfit_rmse": cross_rmse,
                "crossfit_delta_vs_better": cross_rmse - best_single,
                "crossfit_folds_beating_better_parent": fold_wins,
                "crossfit_fold_weights_b": json.dumps(fold_weights),
            }
        )
    return pd.DataFrame(rows).sort_values(["equal_50_delta_vs_better", "equal_50_rmse"])


def simplex_weights(g: np.ndarray) -> np.ndarray:
    k = len(g)
    scale = max(float(np.max(np.abs(g))), 1.0)
    scaled = g / scale
    starts = [np.full(k, 1.0 / k)]
    for index in np.argsort(np.diag(scaled))[: min(3, k)]:
        start = np.zeros(k)
        start[index] = 1.0
        starts.append(start)
    best: tuple[float, np.ndarray] | None = None
    for x0 in starts:
        result = minimize(
            lambda w: float(w @ scaled @ w),
            x0,
            jac=lambda w: 2.0 * scaled @ w,
            bounds=[(0.0, 1.0)] * k,
            constraints={
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        weights = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            continue
        weights /= weights.sum()
        objective = float(weights @ scaled @ weights)
        if best is None or objective < best[0]:
            best = (objective, weights)
    if best is not None:
        return best[1]

    # Singular near-duplicate candidates can defeat line search. Iteratively remove
    # negative unconstrained weights; this is a deterministic long-only fallback.
    active = list(range(k))
    while active:
        sub = scaled[np.ix_(active, active)]
        direction = np.linalg.pinv(sub, rcond=1e-12) @ np.ones(len(active))
        denom = float(direction.sum())
        if abs(denom) <= 1e-15:
            active = active[:-1]
            continue
        active_weights = direction / denom
        if np.min(active_weights) >= -1e-10:
            weights = np.zeros(k)
            weights[active] = np.maximum(active_weights, 0.0)
            return weights / weights.sum()
        del active[int(np.argmin(active_weights))]
    weights = np.zeros(k)
    weights[int(np.argmin(np.diag(scaled)))] = 1.0
    return weights


def multiway_tables(
    candidates: list[Candidate],
    total: np.ndarray,
    folds: np.ndarray,
    fold_counts: np.ndarray,
    pair_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    n = int(fold_counts.sum())
    single = np.sqrt(np.diag(total) / n)
    path_idx = [i for i, item in enumerate(candidates) if item.layer == "candidate_path"]
    top_single = sorted(path_idx, key=lambda i: single[i])[:18]
    improving_pairs = pair_df[
        (pair_df["layer_a"] == "candidate_path")
        & (pair_df["layer_b"] == "candidate_path")
        & (pair_df["equal_50_delta_vs_better"] < 0)
    ].head(30)
    name_to_idx = {item.name: i for i, item in enumerate(candidates)}
    extra = [
        name_to_idx[name]
        for row in improving_pairs.itertuples()
        for name in [row.candidate_a, row.candidate_b]
    ]
    shortlist = list(dict.fromkeys([*top_single, *extra]))[:24]
    equal_rows: list[dict[str, Any]] = []
    for size in range(2, min(6, len(shortlist)) + 1):
        for combo in itertools.combinations(shortlist, size):
            weights = np.full(size, 1.0 / size)
            rmse = math.sqrt(quadratic_sse(total, list(combo), weights) / n)
            equal_rows.append(
                {
                    "size": size,
                    "members": "|".join(candidates[i].name for i in combo),
                    "rmse": rmse,
                    "delta_vs_best_member": rmse - min(single[list(combo)]),
                }
            )
    equal_df = pd.DataFrame(equal_rows).sort_values(["rmse", "size"])

    triple_rows: list[dict[str, Any]] = []
    triple_shortlist = shortlist[:20]
    for combo in itertools.combinations(triple_shortlist, 3):
        full_w = simplex_weights(total[np.ix_(combo, combo)])
        cross_sse = 0.0
        fold_weights: list[list[float]] = []
        fold_wins = 0
        for fold in range(N_FOLDS):
            train = (total - folds[fold])[np.ix_(combo, combo)]
            weights = simplex_weights(train)
            fold_weights.append(weights.tolist())
            hold = quadratic_sse(folds[fold], list(combo), weights)
            cross_sse += hold
            fold_wins += int(hold < min(folds[fold, i, i] for i in combo))
        cross_rmse = math.sqrt(cross_sse / n)
        triple_rows.append(
            {
                "members": "|".join(candidates[i].name for i in combo),
                "best_member_rmse": min(single[list(combo)]),
                "full_oof_weights_diagnostic": json.dumps(full_w.tolist()),
                "full_oof_rmse_diagnostic": math.sqrt(
                    quadratic_sse(total, list(combo), full_w) / n
                ),
                "crossfit_rmse": cross_rmse,
                "crossfit_delta_vs_best_member": cross_rmse - min(single[list(combo)]),
                "crossfit_folds_beating_best_parent": fold_wins,
                "crossfit_fold_weights": json.dumps(fold_weights),
            }
        )
    triple_df = pd.DataFrame(triple_rows).sort_values(
        ["crossfit_rmse", "crossfit_delta_vs_best_member"]
    )
    return equal_df, triple_df, shortlist


def portfolio_table(
    candidates: list[Candidate],
    total: np.ndarray,
    folds: np.ndarray,
    fold_counts: np.ndarray,
    shortlist: list[int],
) -> pd.DataFrame:
    n = int(fold_counts.sum())
    rows: list[dict[str, Any]] = []
    for size in [6, 10, min(15, len(shortlist))]:
        combo = shortlist[:size]
        if len(combo) != size:
            continue
        full_w = simplex_weights(total[np.ix_(combo, combo)])
        cross_sse = 0.0
        fold_weights = []
        for fold in range(N_FOLDS):
            w = simplex_weights((total - folds[fold])[np.ix_(combo, combo)])
            fold_weights.append(w.tolist())
            cross_sse += quadratic_sse(folds[fold], combo, w)
        rows.append(
            {
                "size": size,
                "members": "|".join(candidates[i].name for i in combo),
                "full_oof_weights_diagnostic": json.dumps(full_w.tolist()),
                "full_oof_rmse_diagnostic": math.sqrt(quadratic_sse(total, combo, full_w) / n),
                "crossfit_rmse": math.sqrt(cross_sse / n),
                "crossfit_fold_weights": json.dumps(fold_weights),
            }
        )
    return pd.DataFrame(rows).sort_values("crossfit_rmse")


def candidate_metrics(
    candidates: list[Candidate], total: np.ndarray, n: int, finite_counts: np.ndarray
) -> pd.DataFrame:
    rmse = np.sqrt(np.diag(total) / n)
    rows = []
    for i, item in enumerate(candidates):
        rows.append(
            {
                **item.__dict__,
                "rows_finite": int(finite_counts[i]),
                "coverage": finite_counts[i] / n,
                "rmse": rmse[i],
            }
        )
    return pd.DataFrame(rows).sort_values(["layer", "rmse"])


def duplicate_table(candidates: list[Candidate], total: np.ndarray, n: int) -> pd.DataFrame:
    rows = []
    for i, j in itertools.combinations(range(len(candidates)), 2):
        diff_sse = total[i, i] + total[j, j] - 2.0 * total[i, j]
        diff_rmse = math.sqrt(max(diff_sse, 0.0) / n)
        if diff_rmse <= 0.25:
            rows.append(
                {
                    "candidate_a": candidates[i].name,
                    "candidate_b": candidates[j].name,
                    "prediction_difference_rmse": diff_rmse,
                }
            )
    columns = ["candidate_a", "candidate_b", "prediction_difference_rmse"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("prediction_difference_rmse")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--reuse-matrix", action="store_true")
    parser.add_argument("--extend-matrix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    sources = external_sources()
    candidates = candidate_catalog(sources)
    catalog_path = args.work / "candidate_catalog.json"
    if args.extend_matrix and catalog_path.exists():
        old_catalog = json.loads(catalog_path.read_text())
        new_catalog = [item.__dict__ for item in candidates]
        if new_catalog[: len(old_catalog)] != old_catalog or len(new_catalog) <= len(old_catalog):
            raise ValueError("new catalog is not a strict append of the existing catalog")
        old_pred = np.memmap(
            args.work / "predictions.f32",
            mode="r",
            dtype=np.float32,
            shape=(N_EXPECTED, len(old_catalog)),
        )
        extended_path = args.work / "predictions.extended.f32"
        extended = np.memmap(
            extended_path,
            mode="w+",
            dtype=np.float32,
            shape=(N_EXPECTED, len(candidates)),
        )
        for start in range(0, N_EXPECTED, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, N_EXPECTED)
            extended[start:end, : len(old_catalog)] = old_pred[start:end]
            extended[start:end, len(old_catalog) :] = np.nan
        extended.flush()
        del extended, old_pred
        extended_path.replace(args.work / "predictions.f32")
        pred = np.memmap(
            args.work / "predictions.f32",
            mode="r+",
            dtype=np.float32,
            shape=(N_EXPECTED, len(candidates)),
        )
        truth = np.memmap(args.work / "truth.f32", mode="r", dtype=np.float32, shape=N_EXPECTED)
        md_since = np.memmap(
            args.work / "md_since.f32", mode="r", dtype=np.float32, shape=N_EXPECTED
        )
        well_code = np.memmap(
            args.work / "well_code.u16", mode="r", dtype=np.uint16, shape=N_EXPECTED
        )
        meta_json = json.loads((args.work / "base_meta.json").read_text())
        well_names = meta_json["well_names"]
        meta = {key: tuple(value) for key, value in meta_json["meta"].items()}
        old_names = {item["name"] for item in old_catalog}
        added_sources = [
            source
            for source in sources
            if any(item.candidate.name not in old_names for item in source.candidates)
        ]
        manifests = json.loads((args.work / "source_manifest.json").read_text())
        manifests.extend(load_external(source, pred, candidates, meta) for source in added_sources)
        catalog_path.write_text(json.dumps(new_catalog, indent=2, ensure_ascii=False) + "\n")
        (args.work / "source_manifest.json").write_text(
            json.dumps(manifests, indent=2, ensure_ascii=False) + "\n"
        )
    elif args.reuse_matrix and catalog_path.exists():
        old = json.loads(catalog_path.read_text())
        if old != [item.__dict__ for item in candidates]:
            raise ValueError("candidate catalog changed; rerun without --reuse-matrix")
        pred = np.memmap(
            args.work / "predictions.f32",
            mode="r+",
            dtype=np.float32,
            shape=(N_EXPECTED, len(candidates)),
        )
        truth = np.memmap(args.work / "truth.f32", mode="r", dtype=np.float32, shape=N_EXPECTED)
        md_since = np.memmap(
            args.work / "md_since.f32", mode="r", dtype=np.float32, shape=N_EXPECTED
        )
        well_code = np.memmap(
            args.work / "well_code.u16", mode="r", dtype=np.uint16, shape=N_EXPECTED
        )
        meta_json = json.loads((args.work / "base_meta.json").read_text())
        well_names = meta_json["well_names"]
        meta = {key: tuple(value) for key, value in meta_json["meta"].items()}
        manifests = json.loads((args.work / "source_manifest.json").read_text())
    else:
        pred, truth, md_since, well_code, well_names, meta = prepare_base(args.work, candidates)
        manifests = [load_external(source, pred, candidates, meta) for source in sources]
        catalog_path.write_text(
            json.dumps([item.__dict__ for item in candidates], indent=2, ensure_ascii=False) + "\n"
        )
        (args.work / "base_meta.json").write_text(
            json.dumps({"well_names": well_names, "meta": meta}, ensure_ascii=False) + "\n"
        )
        (args.work / "source_manifest.json").write_text(
            json.dumps(manifests, indent=2, ensure_ascii=False) + "\n"
        )

    counts_by_code = np.bincount(np.asarray(well_code), minlength=len(well_names))
    fold_by_code = assign_group_folds(well_names, counts_by_code)
    total, folds, fold_counts, finite_counts = calculate_grams(pred, truth, well_code, fold_by_code)
    valid = finite_counts == N_EXPECTED
    if not valid.all():
        missing = {candidates[i].name: int(finite_counts[i]) for i in np.flatnonzero(~valid)}
        raise ValueError(f"incomplete candidate coverage: {missing}")

    metrics = candidate_metrics(candidates, total, N_EXPECTED, finite_counts)
    pairs = pair_table(candidates, total, folds, fold_counts)
    equal_multi, triples, shortlist = multiway_tables(candidates, total, folds, fold_counts, pairs)
    portfolios = portfolio_table(candidates, total, folds, fold_counts, shortlist)
    duplicates = duplicate_table(candidates, total, N_EXPECTED)

    metrics.to_csv(args.output / "candidate_metrics.csv", index=False)
    pairs.to_csv(args.output / "pair_blends.csv", index=False)
    equal_multi.to_csv(args.output / "equal_multiway_blends.csv", index=False)
    triples.to_csv(args.output / "crossfit_triple_blends.csv", index=False)
    portfolios.to_csv(args.output / "crossfit_portfolios.csv", index=False)
    duplicates.to_csv(args.output / "near_duplicate_predictions.csv", index=False)
    pd.DataFrame({"well": well_names, "rows": counts_by_code, "fold": fold_by_code}).to_csv(
        args.output / "well_folds.csv", index=False
    )
    metrics_by_rmse = metrics.sort_values("rmse")
    path_metrics = metrics[metrics.layer == "candidate_path"].sort_values("rmse")
    output_metrics = metrics[metrics.layer == "model_output"].sort_values("rmse")
    useful_path_pairs = pairs[
        (pairs.layer_a == "candidate_path")
        & (pairs.layer_b == "candidate_path")
        & (pairs.equal_50_delta_vs_better < 0)
    ].sort_values("equal_50_rmse")
    useful_output_pairs = pairs[
        (pairs.layer_a == "model_output")
        & (pairs.layer_b == "model_output")
        & (pairs.equal_50_delta_vs_better < 0)
    ].sort_values("equal_50_rmse")
    summary = {
        "rows": N_EXPECTED,
        "wells": len(well_names),
        "candidates": len(candidates),
        "candidate_paths": sum(item.layer == "candidate_path" for item in candidates),
        "model_outputs": sum(item.layer == "model_output" for item in candidates),
        "fold_rows": fold_counts.tolist(),
        "best_candidate_overall": metrics_by_rmse.iloc[0].to_dict(),
        "best_candidate_path": path_metrics.iloc[0].to_dict(),
        "best_model_output": output_metrics.iloc[0].to_dict(),
        "best_useful_equal_path_pair": useful_path_pairs.iloc[0].to_dict(),
        "best_useful_equal_model_output_pair": useful_output_pairs.iloc[0].to_dict(),
        "best_equal_multiway": equal_multi.iloc[0].to_dict(),
        "best_crossfit_triple": triples.iloc[0].to_dict(),
        "best_crossfit_portfolio": portfolios.iloc[0].to_dict(),
        "source_manifest": manifests,
        "method_note": (
            "fixed equal blends are target-free; full_oof_opt is diagnostic; "
            "crossfit weights are fit on other wells only"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n"
    )

    eligible_names = {
        item.name
        for item in candidates
        if item.layer == "candidate_path" and item.name not in STACKING_LIKE_PATHS
    }
    deployable_names = {
        item.name
        for item in candidates
        if item.name in eligible_names and item.rawtest_status in RAWTEST_PATH_STATUSES
    }

    def members_allowed(value: str, allowed: set[str]) -> bool:
        return set(value.split("|")).issubset(allowed)

    non_stacking_metrics = path_metrics[path_metrics.name.isin(eligible_names)]
    non_stacking_pairs = pairs[
        pairs.candidate_a.isin(eligible_names) & pairs.candidate_b.isin(eligible_names)
    ]
    non_stacking_equal = equal_multi[
        equal_multi.members.map(lambda value: members_allowed(value, eligible_names))
    ].sort_values("rmse")
    non_stacking_triples = triples[
        triples.members.map(lambda value: members_allowed(value, eligible_names))
    ].sort_values("crossfit_rmse")
    deployable_triples = non_stacking_triples[
        non_stacking_triples.members.map(lambda value: members_allowed(value, deployable_names))
    ]
    fixed_members = ("exp226_k16", "likpf_mean", "exact_hmm")
    fixed_weights = np.asarray((0.5, 0.25, 0.25), dtype=np.float64)
    fixed_indices = [
        next(i for i, item in enumerate(candidates) if item.name == name) for name in fixed_members
    ]
    fixed_sse = float(fixed_weights @ total[np.ix_(fixed_indices, fixed_indices)] @ fixed_weights)
    fixed_fold_rmse = []
    exp226_index = fixed_indices[0]
    exp226_fold_rmse = []
    for fold in range(N_FOLDS):
        fixed_fold_sse = float(
            fixed_weights @ folds[fold][np.ix_(fixed_indices, fixed_indices)] @ fixed_weights
        )
        fixed_fold_rmse.append(math.sqrt(fixed_fold_sse / fold_counts[fold]))
        exp226_fold_rmse.append(
            math.sqrt(float(folds[fold][exp226_index, exp226_index]) / fold_counts[fold])
        )

    non_stacking_summary = {
        "scope": "candidate paths excluding HMM centered on or emitted from LGB predictions",
        "excluded_paths": sorted(STACKING_LIKE_PATHS),
        "eligible_candidate_paths": len(eligible_names),
        "best_single_path": non_stacking_metrics.iloc[0].to_dict(),
        "best_fixed_equal_pair": non_stacking_pairs.sort_values("equal_50_rmse").iloc[0].to_dict(),
        "best_crossfit_pair": non_stacking_pairs.sort_values("crossfit_rmse").iloc[0].to_dict(),
        "best_equal_multiway": non_stacking_equal.iloc[0].to_dict(),
        "best_crossfit_triple": non_stacking_triples.iloc[0].to_dict(),
        "best_rawtest_compatible_crossfit_triple": deployable_triples.iloc[0].to_dict(),
        "fixed_exp226_plus_w500": {
            "members": list(fixed_members),
            "weights": fixed_weights.tolist(),
            "interpretation": "50/50 average of exp226 and blend_likpf_hmm_w500",
            "rmse": math.sqrt(fixed_sse / N_EXPECTED),
            "fold_rmse": fixed_fold_rmse,
            "folds_beating_exp226": int(
                sum(a < b for a, b in zip(fixed_fold_rmse, exp226_fold_rmse, strict=True))
            ),
        },
        "method_note": (
            "fixed equal blends are target-free; full_oof_opt is diagnostic; "
            "crossfit weights are fit on other wells only"
        ),
    }
    (args.output / "non_stacking_scope_summary.json").write_text(
        json.dumps(non_stacking_summary, indent=2, ensure_ascii=False, default=float) + "\n"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "rows",
                    "wells",
                    "candidates",
                    "best_candidate_overall",
                    "best_candidate_path",
                    "best_model_output",
                    "best_useful_equal_path_pair",
                    "best_useful_equal_model_output_pair",
                    "best_equal_multiway",
                    "best_crossfit_triple",
                    "best_crossfit_portfolio",
                ]
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
