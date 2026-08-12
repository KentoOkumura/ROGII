from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = ROOT / "experiments" / "exp404_scale5_sigma_gr_likelihood_pf_ablation"
TRAIN_SOURCE = (
    EXPERIMENT_DIR / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_train.py"
)
INFERENCE_SOURCE = (
    EXPERIMENT_DIR
    / "exp404_scale5_sigma_gr_likelihood_pf_ablation_compact_selfcontained_inference.py"
)
PARENT_SOURCE = (
    ROOT
    / "experiments"
    / "exp072_exp063_full_replay_feature_cache"
    / "public_notebook_replay_audit.py"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train():
    return load_module(TRAIN_SOURCE, "exp404_train_contract")


@pytest.fixture(scope="module")
def inference():
    return load_module(INFERENCE_SOURCE, "exp404_inference_contract")


@pytest.fixture(scope="module")
def config(train):
    return train.load_experiment_config(EXPERIMENT_DIR)


def synthetic_well() -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal = pd.DataFrame(
        {
            "MD": np.arange(12, dtype=float) + 1.0,
            "Z": np.linspace(0.0, 1.1, 12),
            "GR": [
                50.0,
                np.nan,
                70.0,
                80.0,
                65.0,
                60.0,
                np.nan,
                55.0,
                58.0,
                np.nan,
                62.0,
                64.0,
            ],
            "TVT_input": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    typewell = pd.DataFrame(
        {
            "TVT": np.linspace(90.0, 130.0, 81),
            "GR": np.linspace(45.0, 85.0, 81),
        }
    )
    return horizontal, typewell


def synthetic_metric_frame() -> pd.DataFrame:
    rows = 12
    truth = np.linspace(100.0, 111.0, rows)
    return pd.DataFrame(
        {
            "id": [f"w{i // 6}_{i}" for i in range(rows)],
            "well_id": ["w0"] * 6 + ["w1"] * 6,
            "true_tvt": truth,
            "fold": [0] * 6 + [1] * 6,
            "raw_gr_observed": [True, False] * 6,
            "well_missing_fraction": [0.4] * 6 + [0.1] * 6,
            "md_since": [1100.0, 1200.0] + [100.0] * 10,
            "hidden_like_spatial": [True] + [False] * 11,
            "hidden_like_typewell_purged": [False, True] + [False] * 10,
            "saved_exp072_likpf_mean": truth + 1.0,
            "likpf_mean_x1p0": truth + 1.0,
            "likpf_mean_x1p3": truth + 0.4,
            "likpf_scale_5_x1p0": truth + 1.0,
            "likpf_scale_5_x1p3": truth + 0.2,
        }
    )


def test_frozen_scientific_contract_and_execution_boundary(train, config) -> None:
    contract = train.validate_scientific_contract(config)

    assert contract["primary_control"] == "likpf_scale_5_x1p0"
    assert contract["primary_candidate"] == "likpf_scale_5_x1p3"
    assert contract["parity_columns"] == ["likpf_mean_x1p0", "likpf_mean_x1p3"]
    assert contract["execution_counts"]["scientific_variants"] == 2
    assert contract["execution_counts"]["candidate_pf_well_runs"] == 1_546
    assert contract["execution_counts"]["seed_well_trajectories"] == 197_888
    assert contract["execution_counts"]["particle_starts"] == 98_944_000
    assert len(contract["scientific_contract_sha256"]) == 64
    run_contract = train.validate_scientific_contract(config, require_run_approval=True)
    assert run_contract["scientific_contract_sha256"] == contract["scientific_contract_sha256"]
    missing_roles = copy.deepcopy(config)
    missing_roles["data"]["hidden_like_assignment"].pop("expected_role_counts")
    with pytest.raises(ValueError, match="expected_role_counts"):
        train.validate_scientific_contract(missing_roles)


def test_frozen_version1_predictions_can_resume_without_pf_regeneration(
    train, config, tmp_path: Path
) -> None:
    candidate = pd.DataFrame(
        {
            "id": ["w0_0", "w0_1"],
            "well_id": ["w0", "w0"],
            "row_idx": [0, 1],
            "suffix_offset": [0, 1],
            "last_known_tvt": [100.0, 100.0],
            "md_since": [1.0, 2.0],
            "raw_gr_observed": [True, False],
            "well_missing_fraction": [0.5, 0.5],
            "likpf_scale_5_x1p0": [101.0, 102.0],
            "likpf_scale_5_x1p3": [100.8, 101.8],
            "likpf_mean_x1p0": [101.2, 102.2],
            "likpf_mean_x1p3": [101.1, 102.1],
        }
    )
    candidate["id"] = candidate["id"].astype(object)
    candidate["well_id"] = candidate["well_id"].astype(object)
    audit = pd.DataFrame(
        {
            "well_id": ["w0"],
            "status": ["ok"],
            "gs_base": [20.0],
            "gs_x1p0": [20.0],
            "gs_x1p3": [26.0],
            "post_multiplier_clip_count_x1p0": [0],
            "post_multiplier_clip_count_x1p3": [0],
            "seed_base": [train.stable_seed("likpf", "train", "w0")],
            "seed_base_x1p0": [train.stable_seed("likpf", "train", "w0")],
            "seed_base_x1p3": [train.stable_seed("likpf", "train", "w0")],
            "pf_well_runs": [2],
            "seeds": [128],
            "seed_well_trajectories": [256],
            "particles": [500],
            "particle_starts": [128_000],
        }
    )
    prediction_path = tmp_path / "predictions.csv.gz.bin"
    audit_path = tmp_path / "audit.csv"
    contract_path = tmp_path / "contract.json"
    train.write_deterministic_gzip_csv(candidate, prediction_path)
    audit.to_csv(audit_path, index=False)
    local_config = copy.deepcopy(config)
    contract = train.build_scientific_contract(local_config)
    train.write_json(contract_path, contract)
    report = train.inspect_gzip_csv(prediction_path)
    resume = local_config["data"]["frozen_prediction_resume"]
    resume.update(
        {
            "prediction_filename": prediction_path.name,
            "audit_filename": audit_path.name,
            "contract_filename": contract_path.name,
            "candidates": [str(tmp_path)],
            "expected_prediction_raw_sha256": report["raw_sha256"],
            "expected_prediction_decompressed_sha256": report["decompressed_sha256"],
            "expected_prediction_logical_sha256": train.dataframe_content_sha(
                candidate,
                ["id", "well_id", "row_idx", *train.PREDICTION_COLUMNS],
            ),
            "expected_prediction_schema_sha256": train.dataframe_schema_sha(candidate),
            "expected_audit_raw_sha256": train.sha256_path(audit_path),
            "expected_scientific_contract_sha256": contract["scientific_contract_sha256"],
            "expected_rows": 2,
            "expected_wells": 1,
            "reused_pf_well_runs": 2,
            "reused_seed_well_trajectories": 256,
            "reused_particle_starts": 128_000,
        }
    )
    ledger = train.TruthAccessLedger()
    loaded, loaded_audit, frozen, frozen_paths = train.load_and_freeze_saved_predictions(
        tmp_path / "working",
        local_config,
        ledger,
    )

    pd.testing.assert_frame_equal(loaded, candidate)
    pd.testing.assert_frame_equal(loaded_audit, audit)
    assert frozen["generated_pf_well_runs_this_kernel"] == 0
    assert frozen["reused_pf_well_runs"] == 2
    assert ledger.prediction_frozen is True
    assert all(value == 0 for value in ledger.report()["before_freeze"].values())
    assert train.sha256_path(frozen_paths["paired_predictions"]) == report["raw_sha256"]


def test_inference_is_fail_closed(inference) -> None:
    config = inference.load_config(EXPERIMENT_DIR)
    status = inference.validate_inference_is_disabled(config)

    assert status["implementation_scope"] == "train_side_paired_scale5_pf_audit_only"
    assert status["inference_enabled"] is False
    assert status["run_inference"] is False
    assert status["create_submission"] is False
    assert status["submit_to_kaggle"] is False


def test_stable_seed_is_common_and_excludes_multiplier(train) -> None:
    key = "likpf::train::well-a"
    expected = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % 2_147_483_647 + 1

    seed_x1p0 = train.stable_seed("likpf", "train", "well-a")
    seed_x1p3 = train.stable_seed("likpf", "train", "well-a")
    assert seed_x1p0 == expected == seed_x1p3
    assert train.stable_seed("likpf", "train", "well-b") != expected
    assert "multiplier" not in key


def test_gr_scale_pair_is_clipped_then_multiplied_once_without_reclip(train) -> None:
    horizontal, typewell = synthetic_well()
    prepared_x1p0 = train.prepare_likelihood_pf_inputs(
        horizontal, typewell, multiplier=1.0, grid_step=0.2
    )
    prepared_x1p3 = train.prepare_likelihood_pf_inputs(
        horizontal, typewell, multiplier=1.3, grid_step=0.2
    )
    known = horizontal["TVT_input"].notna().to_numpy()
    typewell_at_known = np.interp(
        horizontal.loc[known, "TVT_input"].to_numpy(np.float64),
        typewell["TVT"].to_numpy(np.float64),
        typewell["GR"].to_numpy(np.float64),
    )
    residual = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64) - typewell_at_known
    expected_base = float(np.clip(np.nanstd(residual), 10.0, 60.0))

    assert prepared_x1p0["scale_audit"]["base_scale"] == pytest.approx(expected_base)
    assert prepared_x1p3["scale_audit"]["base_scale"] == pytest.approx(expected_base)
    assert prepared_x1p0["scale_audit"]["candidate_scale"] == pytest.approx(expected_base)
    assert prepared_x1p3["scale_audit"]["candidate_scale"] == pytest.approx(expected_base * 1.3)
    assert prepared_x1p0["scale_audit"]["post_multiplier_clip_count"] == 0
    assert prepared_x1p3["scale_audit"]["post_multiplier_clip_count"] == 0
    assert prepared_x1p0["scale_audit"]["known_gr_missing_rows"] == 1
    assert np.isfinite(prepared_x1p0["eval_gr"]).all()


def test_pf_kernel_is_exact_exp072_fixture_parity(train) -> None:
    if "numba" not in sys.modules:
        numba_stub = types.ModuleType("numba")

        def identity_njit(*args, **kwargs):
            del kwargs
            if args and callable(args[0]):
                return args[0]

            def decorator(function):
                return function

            return decorator

        numba_stub.njit = identity_njit
        sys.modules["numba"] = numba_stub
    parent = load_module(PARENT_SOURCE, "exp072_parent_for_exp404")
    md = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    z = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    gr = np.array([50.0, 52.0, 54.0, 53.0, 51.0], dtype=np.float64)
    grid_gr = np.linspace(40.0, 70.0, 151, dtype=np.float64)
    args = (
        md,
        z,
        gr,
        grid_gr,
        90.0,
        0.2,
        20.0,
        100.0,
        0.01,
        24,
        4,
        12345,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )

    observed = train._pf_lik_allseeds(*args)
    expected_predictions, expected_likelihoods = parent._pf_lik_allseeds(*args)

    assert np.array_equal(observed[0], expected_predictions)
    assert np.array_equal(observed[1], expected_likelihoods)
    assert observed[2].shape == (4,)
    assert observed[3].shape == (4,)
    assert observed[4].shape == (4,)


def test_seed_aggregation_is_fixed_to_scale5_with_mean_parity_only(train) -> None:
    predictions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    likelihoods = np.array([-3.0, -2.0, -1.0], dtype=np.float64)

    outputs = train.aggregate_seed_predictions(predictions, likelihoods, [5.0])
    centered = likelihoods - likelihoods.max()
    weights = np.exp(centered / 5.0)
    weights /= weights.sum()

    np.testing.assert_allclose(outputs["pf_scale_5"], (weights[:, None] * predictions).sum(0))
    np.testing.assert_allclose(outputs["pf_mean"], [3.0, 4.0])
    assert set(outputs) == {"pf_scale_5", "pf_mean"}


def test_horizontal_loader_excludes_truth_and_ledger_requires_freeze(train, tmp_path: Path) -> None:
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
    ledger = train.TruthAccessLedger()

    assert list(frame.columns) == ["MD", "Z", "GR", "TVT_input"]
    with pytest.raises(RuntimeError, match="requires a frozen prediction"):
        ledger.require_frozen()
    with pytest.raises(RuntimeError, match="frozen prediction"):
        train._require_frozen_prediction({})
    ledger.mark_frozen()
    ledger.require_frozen()


def test_metric_outputs_compare_only_paired_scale5_variants(train) -> None:
    frame = synthetic_metric_frame()
    primary, by_well = train.build_metric_outputs(frame)

    overall = primary.loc[primary["scope"].eq("overall")].iloc[0]
    assert overall["candidate"] == "likpf_scale_5_x1p3"
    assert overall["control"] == "likpf_scale_5_x1p0"
    assert overall["comparison"] == "paired_scale5_gs_x1p3_vs_x1p0"
    assert overall["improvement_ft"] == pytest.approx(0.8)
    assert len(by_well) == 2
    assert set(primary["scope"]) == {
        "overall",
        "fold_0",
        "fold_1",
        "raw_gr_observed",
        "raw_gr_missing",
        "missing_fraction_high",
        "md_since_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    }


def test_parity_metrics_are_nonpromotional_and_gate_all_three_contracts(train, config) -> None:
    frame = synthetic_metric_frame()
    local_config = copy.deepcopy(config)
    truth = frame["true_tvt"].to_numpy(np.float64)
    local_config["data"]["exp072_reference"]["expected_saved_mean_rmse_ft"] = train.rmse(
        truth, frame["saved_exp072_likpf_mean"].to_numpy(np.float64)
    )
    local_config["data"]["exp400_reference"]["mean_x1p3_rmse_ft"] = train.rmse(
        truth, frame["likpf_mean_x1p3"].to_numpy(np.float64)
    )
    local_config["data"]["exp400_reference"]["scale_5_x1p3_rmse_ft"] = train.rmse(
        truth, frame["likpf_scale_5_x1p3"].to_numpy(np.float64)
    )

    parity = train.build_parity_metrics(frame, local_config)

    assert parity["passed"] is True
    assert parity["policy"] == "technical_parity_only_not_scientific_candidate_selection"
    assert set(parity["checks"]) == {
        "saved_exp072_mean_reference",
        "x1p0_mean_vs_saved_exp072",
        "x1p3_mean_vs_exp400",
        "x1p3_scale5_vs_exp400",
    }


def test_promotion_gate_enforces_paired_counts_seeds_scopes_and_tail_guards(train, config) -> None:
    frame = synthetic_metric_frame()
    primary, by_well = train.build_metric_outputs(frame)
    local_config = copy.deepcopy(config)
    local_config["validation"]["expected_rows"] = len(frame)
    local_config["validation"]["expected_wells"] = 2
    local_config["validation"]["expected_folds"] = [0, 1]
    local_config["model"]["execution_count"].update(
        {
            "candidate_pf_well_runs": 4,
            "seed_well_trajectories": 512,
            "particle_starts": 256_000,
            "reporting_folds": 2,
        }
    )
    local_config["guards"]["scientific"]["minimum_improved_folds"] = 2
    audit = pd.DataFrame(
        {
            "well_id": ["w0", "w1"],
            "status": ["ok", "ok"],
            "gs_base": [10.0, 20.0],
            "gs_x1p0": [10.0, 20.0],
            "gs_x1p3": [13.0, 26.0],
            "post_multiplier_clip_count_x1p0": [0, 0],
            "post_multiplier_clip_count_x1p3": [0, 0],
            "seed_base": [
                train.stable_seed("likpf", "train", "w0"),
                train.stable_seed("likpf", "train", "w1"),
            ],
            "seed_base_x1p0": [
                train.stable_seed("likpf", "train", "w0"),
                train.stable_seed("likpf", "train", "w1"),
            ],
            "seed_base_x1p3": [
                train.stable_seed("likpf", "train", "w0"),
                train.stable_seed("likpf", "train", "w1"),
            ],
            "pf_well_runs": [2, 2],
            "seeds": [128, 128],
            "seed_well_trajectories": [256, 256],
            "particles": [500, 500],
            "particle_starts": [128_000, 128_000],
        }
    )
    ledger = train.TruthAccessLedger()
    ledger.mark_frozen()
    parity = {"passed": True, "checks": {}}
    frozen = {"logical_content_sha256": "a" * 64}
    gate = train.evaluate_promotion_gate(
        frame,
        primary,
        by_well,
        parity,
        audit,
        {"content_sha256": "b" * 64},
        frozen,
        ledger,
        1.0,
        local_config,
    )

    assert gate["technical_gate"]["passed"] is True
    assert gate["primary_scientific_gate"]["passed"] is True
    assert gate["passed"] is True
    assert gate["technical_gate"]["execution_counts"]["candidate_pf_well_runs"] == 4
    assert gate["technical_gate"]["common_seed_labels_across_variants"] is True


def test_notebook_sources_are_not_file_relative_or_submission_creating() -> None:
    train_source = TRAIN_SOURCE.read_text()
    inference_source = INFERENCE_SOURCE.read_text()

    assert "__file__" not in train_source
    assert "__file__" not in inference_source
    assert "shutil.copyfile" not in inference_source
    assert "sample_submission" not in inference_source
    assert "run_full_experiment(CONFIG)" in train_source
    assert "scale_3" not in train_source
    assert "scale_8" not in train_source
    assert "scale_12" not in train_source
