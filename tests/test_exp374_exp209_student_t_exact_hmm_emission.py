from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp374_exp209_student_t_exact_hmm_emission"
TRAIN_SOURCE = EXP_DIR / (
    "exp374_exp209_student_t_exact_hmm_emission_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = EXP_DIR / (
    "exp374_exp209_student_t_exact_hmm_emission_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
    / "exact_hmm_smoother.py"
)


def load_module(path: Path, name: str):
    previous = os.environ.get("EXP374_IMPORT_ONLY")
    os.environ["EXP374_IMPORT_ONLY"] = "1"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("EXP374_IMPORT_ONLY", None)
        else:
            os.environ["EXP374_IMPORT_ONLY"] = previous


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp374_train_test")


@pytest.fixture(scope="module")
def config(train):
    return train.read_yaml(EXP_DIR / "config.yaml")


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 38
    known_rows = 30
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(rows, dtype=float) * 10.0,
            "Z": np.arange(rows, dtype=float) * 0.2,
            "GR": np.linspace(40.0, 64.0, rows),
            "TVT_input": np.r_[
                100.0 + np.arange(known_rows, dtype=float),
                np.full(rows - known_rows, np.nan),
            ],
        }
    )
    horizontal.loc[[4, 33, 35], "GR"] = np.nan
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(80.0, 180.0, 401),
            "GR": np.linspace(35.0, 70.0, 401),
        }
    )
    return horizontal, typewell


def test_contract_records_one_candidate_and_explicit_run_approval(train, config) -> None:
    train.validate_scientific_contract(config)
    assert config["experiment"]["route"] == "pf_beam"
    assert config["implementation"]["enabled"] is True
    assert config["implementation"]["canonical_notebook_adopted"] is True
    assert config["model"]["active_variants"] == [
        "student_t_df4_on_exp209_absolute_tvt"
    ]
    assert config["execution_contract"] == {
        "scientific_variants": 1,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
        "parent_control_retraining": False,
    }
    assert config["execution"]["implementation_approved"] is True
    assert config["execution"]["kaggle_push_approved"] is True
    assert config["execution"]["train_run_approved"] is True
    assert config["execution"]["run_hmm"] is True
    train.validate_scientific_contract(config, require_run_approval=True)

    blocked = deepcopy(config)
    blocked["execution"]["run_hmm"] = False
    with pytest.raises(RuntimeError, match="package/push/run is not approved"):
        train.validate_scientific_contract(blocked, require_run_approval=True)


def test_student_t_formula_is_fixed_df4_without_clip_or_temperature(train) -> None:
    zscore = np.array([0.0, 1.0, 3.0, 10.0, 100.0])
    observed = train.student_t_log_likelihood(zscore, 4.0)
    expected = -2.5 * np.log1p(np.square(zscore) / 4.0)
    np.testing.assert_allclose(observed, expected)
    assert observed[0] == 0.0
    assert observed[-1] > -0.5 * np.square(zscore[-1])
    with pytest.raises(ValueError, match="positive and finite"):
        train.student_t_log_likelihood(zscore, 0.0)


def test_exp209_sigma_and_state_grammar_are_unchanged(train, config) -> None:
    horizontal, typewell = synthetic_well()
    prepared = train.prepare_hmm_inputs(horizontal, typewell, config)
    known = horizontal["TVT_input"].notna().to_numpy()
    typewell_at_known = np.interp(
        horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    expected_residual = (
        horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
        - typewell_at_known
    )
    expected_sigma = float(np.clip(np.std(expected_residual, ddof=0), 10.0, 60.0))
    assert prepared["sigma_gr"] == pytest.approx(expected_sigma)
    assert prepared["observation_audit"]["missing_gr_fill_value"] == 0.0
    assert prepared["rates"].shape == (41,)
    assert prepared["rates"][0] == pytest.approx(-prepared["rates"][-1])
    assert prepared["raw_gr_observed"].tolist() == [
        True,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
    ]


def test_candidate_matches_parent_exp209_student_t_mode(train, config) -> None:
    parent = load_module(PARENT_SOURCE, "exp209_parent_for_exp374")
    horizontal, typewell = synthetic_well()
    prepared = train.prepare_hmm_inputs(horizontal, typewell, config)
    candidate = train.run_exact_hmm_variant(
        prepared,
        float(prepared["sigma_gr"]),
        config,
    )
    expected = parent.run_hmm2(
        horizontal,
        typewell,
        step=0.35,
        n_rates=41,
        rate_span=0.10,
        sig_r=0.002,
        sig_p=0.02,
        df=4.0,
        emission="t",
        lam=1.0,
        sigma_mode="std",
        start_sig=0.75,
        r0_sig=0.01,
        band_pad=100.0,
        mom=0.998,
        rate_center="zero",
        return_post=True,
    )
    np.testing.assert_allclose(
        candidate["mean"], expected["mean_eval"], rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        candidate["std"], expected["std_eval"], rtol=0.0, atol=1.0e-10
    )
    assert candidate["loglik"] == pytest.approx(expected["loglik"], abs=1.0e-10)
    assert candidate["posterior_row_sum_max_abs_error"] < 1.0e-6
    assert candidate["emission_finite_coverage"] == 1.0
    assert candidate["degrees_of_freedom"] == 4.0


def test_exact_kernel_matches_parent_for_identical_emission(train) -> None:
    parent = load_module(PARENT_SOURCE, "exp209_kernel_for_exp374")
    emission = np.array(
        [[-0.1, -0.2, -0.4], [-0.3, -0.1, -0.2], [-0.2, -0.3, -0.1]],
        dtype=np.float32,
    )
    dm = np.ones(3, dtype=np.float64)
    dz = np.zeros(3, dtype=np.float64)
    rates = np.array([-0.01, 0.0, 0.01], dtype=np.float64)
    args = (
        emission,
        dm,
        dz,
        0.35,
        rates,
        0.002,
        0.02,
        1.0,
        0.75,
        0.0,
        0.01,
        1.0,
        0.998,
    )
    observed_post, observed_loglik = train._hmm2_fb(*args)
    expected_post, expected_loglik = parent._hmm2_fb(*args)
    assert np.array_equal(observed_post, expected_post)
    assert observed_loglik == expected_loglik


def test_horizontal_loader_and_late_truth_boundary_are_fail_closed(
    train, tmp_path: Path
) -> None:
    pd.DataFrame(
        {
            "MD": [0.0, 1.0],
            "Z": [0.0, 0.1],
            "GR": [50.0, np.nan],
            "TVT_input": [100.0, np.nan],
            "TVT": [100.0, 101.0],
            "error": [0.0, 99.0],
        }
    ).to_csv(tmp_path / "a__horizontal_well.csv", index=False)
    frame = train.load_horizontal_without_truth("a", tmp_path)
    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    with pytest.raises(RuntimeError, match="frozen prediction"):
        train._require_frozen_prediction({})
    train._require_frozen_prediction(
        {
            "frozen_before_truth_attachment": True,
            "decompressed_sha256": "a" * 64,
            "content_sha256": "a" * 64,
        }
    )


def test_control_preflight_uses_sha_and_identity_only_before_truth(
    train, config, tmp_path: Path
) -> None:
    local = deepcopy(config)
    wells = ["a", "b"]
    identities = [(well, row_idx) for well in wells for row_idx in range(5)]
    hmm_path = tmp_path / "hmm.csv.gz"
    exp072_path = tmp_path / "exp072.csv.gz"
    fold_path = tmp_path / "fold.csv.gz"
    hidden_path = tmp_path / "hidden.csv"
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "hmm_mean_tvt": np.ones(10),
        }
    ).to_csv(hmm_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "id": [f"{well}_{row}" for well, row in identities],
            "well": [well for well, _ in identities],
            "md_since": np.arange(10, dtype=float),
            "last_known_tvt": np.ones(10),
            "likpf_mean_d": np.zeros(10),
        }
    ).to_csv(exp072_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": [well for well, _ in identities],
            "row_idx": [row for _, row in identities],
            "suffix_offset": [row for _, row in identities],
            "fold": np.arange(10) % 5,
            "tvt_geop": np.zeros(10),
            "tvt_true": np.zeros(10),
        }
    ).to_csv(fold_path, index=False, compression="gzip")
    pd.DataFrame(
        {
            "well_id": wells,
            "verification_like_spatial_role": ["valid", "train"],
            "verification_like_typewell_purged_role": ["train", "valid"],
        }
    ).to_csv(hidden_path, index=False)
    local["validation"].update({"expected_rows": 10, "expected_wells": 2})
    local["data"]["saved_controls"].update(
        {
            "hmm_cache_filename": hmm_path.name,
            "exp072_cache_filename": exp072_path.name,
            "candidates": [str(tmp_path)],
            "expected_hmm_prediction_decompressed_sha256": train.inspect_gzip_csv(
                hmm_path
            )["decompressed_sha256"],
            "expected_exp072_cache_decompressed_sha256": train.inspect_gzip_csv(
                exp072_path
            )["decompressed_sha256"],
        }
    )
    local["data"]["fold_assignment"].update(
        {
            "filename": fold_path.name,
            "candidates": [str(tmp_path)],
            "expected_decompressed_sha256": train.inspect_gzip_csv(fold_path)[
                "decompressed_sha256"
            ],
        }
    )
    local["data"]["hidden_like_assignment"].update(
        {
            "filename": hidden_path.name,
            "candidates": [str(tmp_path)],
            "expected_sha256": train.sha256_path(hidden_path),
            "expected_role_counts": {
                "hidden_like_spatial": {"train": 1, "valid": 1},
                "hidden_like_typewell_purged": {"train": 1, "valid": 1},
            },
        }
    )
    report = train.preflight_controls_and_assignments(local)
    assert report["fold_assignment"]["well_ids"] == wells
    assert report["fold_assignment"]["data_rows"] == 10
    assert "tvt_geop" not in local["data"]["fold_assignment"]["safe_columns"]
    broken = deepcopy(local)
    broken["data"]["saved_controls"][
        "expected_hmm_prediction_decompressed_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="saved exp209 HMM"):
        train.preflight_controls_and_assignments(broken)


def test_promotion_gate_requires_all_preregistered_scopes(train, config) -> None:
    local = deepcopy(config)
    wells = ["a", "b", "c", "d", "e"]
    raw_patterns = {
        "a": [True] * 5,
        "b": [True] * 5,
        "c": [False, True, True, True, True],
        "d": [False, False, True, True, True],
        "e": [False] * 5,
    }
    rows: list[dict[str, object]] = []
    for well in wells:
        pattern = raw_patterns[well]
        missing_fraction = 1.0 - float(np.mean(pattern))
        for fold, observed in enumerate(pattern):
            candidate = 0.8 if observed else 1.0
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "true_tvt": 0.0,
                    "md_since": [100.0, 500.0, 1200.0, 1400.0, 1600.0][fold],
                    "raw_hmm_tvt": 1.0,
                    "likpf_mean": 1.0,
                    "raw_hmm_likpf_50_50": 1.0,
                    "raw_gr_observed": observed,
                    "evaluation_missing_fraction": missing_fraction,
                    "hidden_like_spatial": True,
                    "hidden_like_typewell_purged": True,
                    f"{train.VARIANT_ORDER[0]}_hmm_tvt": candidate,
                    f"{train.VARIANT_ORDER[0]}_likpf_50_50": 0.5 * candidate + 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    local["validation"].update({"expected_rows": len(frame), "expected_wells": len(wells)})
    local["execution_contract"]["hmm_well_runs"] = len(wells)
    local["references"].update(
        {
            "exp209_raw_hmm_rmse": 1.0,
            "exp209_likpf_rmse": 1.0,
            "exp209_hmm_likpf_50_50_rmse": 1.0,
        }
    )
    paired, by_well = train.build_paired_metrics(frame, local)
    runtime = pd.DataFrame(
        {
            "well_id": wells,
            "variant": [train.VARIANT_ORDER[0]] * len(wells),
            "posterior_row_sum_max_abs_error": np.zeros(len(wells)),
            "student_t_degrees_of_freedom": np.full(len(wells), 4.0),
            "emission_finite_coverage": np.ones(len(wells)),
            "raw_observed_rows": [sum(raw_patterns[well]) for well in wells],
            "raw_missing_rows": [5 - sum(raw_patterns[well]) for well in wells],
        }
    )
    observation_audit = pd.DataFrame(
        {
            "well_id": wells,
            "exp209_zero_fill_sigma": np.full(len(wells), 30.0),
        }
    )
    preflight = {"raw_train": {"content_sha256": "a" * 64}}
    gate = train.evaluate_promotion_gate(
        paired,
        by_well,
        frame,
        runtime,
        observation_audit,
        preflight,
        1.0,
        local,
    )
    assert gate["passed"] is True
    assert gate["primary_direct_gate"]["folds_improved"] == 5
    assert gate["primary_direct_gate"]["raw_observed_improvement_ft"] >= 0.05
    assert all(gate["primary_direct_gate"]["required_non_regression_scopes"].values())
    assert gate["technical_gate"]["student_t_df_contract_passed"] is True

    broken_frame = frame.copy()
    missing = ~broken_frame["raw_gr_observed"]
    broken_frame.loc[missing, f"{train.VARIANT_ORDER[0]}_hmm_tvt"] = 1.1
    broken_frame.loc[missing, f"{train.VARIANT_ORDER[0]}_likpf_50_50"] = 1.05
    broken_paired, broken_by_well = train.build_paired_metrics(broken_frame, local)
    broken = train.evaluate_promotion_gate(
        broken_paired,
        broken_by_well,
        broken_frame,
        runtime,
        observation_audit,
        preflight,
        1.0,
        local,
    )
    assert broken["passed"] is False
    assert broken["decision"] == "student_t_exp209_failed_close_without_rescue"
    assert (
        broken["primary_direct_gate"]["required_non_regression_scopes"]["raw_gr_missing"]
        is False
    )


def test_inference_and_sources_are_fail_closed_and_self_contained(config) -> None:
    inference = load_module(INFERENCE_SOURCE, "exp374_inference_test")
    contract = inference.validate_disabled_inference(config)
    assert contract["inference_enabled"] is False
    assert contract["run_inference"] is False
    with pytest.raises(RuntimeError, match="inference and submission are disabled"):
        inference.stop_disabled_inference(config)
    train_text = TRAIN_SOURCE.read_text()
    inference_text = INFERENCE_SOURCE.read_text()
    assert "def student_t_log_likelihood" in train_text
    assert "def compute_exp209_zero_fill_sigma_audit" in train_text
    assert "def _hmm2_fb" in train_text
    assert "def generate_and_freeze_predictions" in train_text
    assert "def evaluate_promotion_gate" in train_text
    assert "from settings import" not in train_text
    assert "from exact_hmm_smoother import" not in train_text
    assert "Path(__file__)" not in train_text
    assert "Path(__file__)" not in inference_text
    assert "submission.csv" not in train_text
